"""Focused iteration isolation, concurrency, evidence projection and consume-once behavior."""

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from jit_mas.pipeline import write_json
from jit_mas.schemas import ExperienceSnapshot, digest
from jit_mas.test_release import TestRelease
from scripts import run_ours_iteration as runner


def test_selection_requires_original_partition_and_unique_explicit_tasks():
    manifest = {"memberships": {"researchrubrics": {"evolution": ["rr-one", "rr-two"],
                                                     "validation": ["rr-val"]},
                               "deepsearchqa": {"evolution": ["dsqa-one"]}}}
    selected = runner.select_tasks(manifest, ["researchrubrics", "deepsearchqa"],
                                   ["researchrubrics=rr-two", "deepsearchqa=dsqa-one"], "evolution", 2)
    assert selected == {"researchrubrics": ["rr-two"], "deepsearchqa": ["dsqa-one"]}
    with pytest.raises(ValueError, match="partition"):
        runner.select_tasks(manifest, ["researchrubrics"], ["rr-val"], "evolution", 2)
    with pytest.raises(ValueError, match="only once"):
        runner.select_tasks(manifest, ["researchrubrics"], ["rr-one", "rr-one"], "evolution", 2)


def test_evidence_projection_preserves_bytes_and_subset_identity(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "selected"
    source.mkdir()
    manifest = {"version": "public-shared-evidence-v1", "count": 2,
                "renderer_validation": "prevalidated-structure-v1", "tasks": {}}
    for task_id in ("selected", "unused"):
        path = runner.evidence_pack_path(source, task_id)
        path.write_bytes(json.dumps({"task_id": task_id}).encode("utf-8"))
        manifest["tasks"][task_id] = {"file": path.name, "file_sha256": runner.file_hash(path)}
    manifest["manifest_sha256"] = digest(manifest)
    write_json(source / "manifest.json", manifest)
    original_bytes = (source / "manifest.json").read_bytes()

    def load(tasks, directory, expected_count):
        projected = runner.read_json(directory / "manifest.json")
        assert expected_count == projected["count"] == 1
        assert set(projected["tasks"]) == set(tasks) == {"selected"}
        return tasks

    monkeypatch.setattr(runner, "load_evidence_tasks", load)
    tasks, files = runner.project_evidence({"selected": "public"}, source, destination)
    assert tasks == {"selected": "public"}
    assert runner.evidence_pack_path(source, "selected").read_bytes() == runner.evidence_pack_path(destination, "selected").read_bytes()
    assert (source / "manifest.json").read_bytes() == original_bytes
    assert not runner.evidence_pack_path(destination, "unused").exists()
    assert str(source / "manifest.json") in files


def iteration_fixture(tmp_path):
    iteration = object.__new__(runner.OursIteration)
    iteration.output = tmp_path
    iteration.snapshot = ExperienceSnapshot()
    iteration.workers = iteration.judge_workers = 3
    iteration.config = SimpleNamespace(check_native=lambda: None)
    slots = [{"slot_id": f"slot-{index}", "task_id": f"task-{index}", "benchmark": "deepsearchqa",
              "method": "ours", "experience_hash": digest(iteration.snapshot), "bounds": [0, 1]}
             for index in range(3)]
    iteration.release = TestRelease(tmp_path / "release", slots)
    iteration.assert_frozen = lambda: None
    iteration.summary = lambda: {"complete": True}
    return iteration


def test_tasks_run_concurrently_and_judgments_wait_for_seal(tmp_path, monkeypatch):
    iteration = iteration_fixture(tmp_path)
    barrier = threading.Barrier(3)
    guard = threading.Lock()
    events = []

    def pipeline(slot):
        def run_task(task_id, snapshot, **options):
            assert options["defer_evaluation"] is True and options["attribution"] is False
            assert options["resume"] is False
            barrier.wait(timeout=3)
            with guard:
                events.append("actor")
            directory = tmp_path / "runs" / slot["slot_id"] / "attempt"
            answer = task_id + " answer"
            write_json(directory / "submission.json", {"answer": answer, "answer_hash": digest(answer)})
            return {"task_id": task_id, "status": "submitted_unscored", "evaluation": None,
                    "proposals": [], "run_dir": str(directory), "answer_hash": digest(answer),
                    "experience_hash": digest(snapshot), "budget": {"tokens": 10}}

        return SimpleNamespace(run_task=run_task), SimpleNamespace(close=lambda: None)

    def score(pipeline, record):
        assert (iteration.release.directory / "seal.json").exists()
        assert len(list((iteration.release.directory / "submissions").glob("*.json"))) == 3
        with guard:
            events.append("judge")
        return {"slot": record["slot"], "complete": True, "official_score": 1}

    iteration.pipeline = pipeline
    monkeypatch.setattr(runner, "score_submission", score)
    iteration.run()
    assert events[:3] == ["actor"] * 3 and events[3:] == ["judge"] * 3
    first_events = list(events)
    iteration.run()
    assert events == first_events


def test_interrupted_generation_and_judgment_are_consumed_without_model_calls(tmp_path):
    iteration = iteration_fixture(tmp_path)
    iteration.pipeline = lambda slot: pytest.fail("Consumed slots must never construct models")
    for slot_id, slot in iteration.release.slots.items():
        write_json(iteration.release._path(slot_id, "submission_started"), {"slot_hash": digest(slot)})
        iteration.submit(slot)
    iteration.release.seal()
    for slot in iteration.release.slots.values():
        iteration.score(slot)
        result = runner.read_json(iteration.release._path(slot["slot_id"], "evaluations"))
        assert result["status"] == "submission_failed"
        assert result["official_score"] is None
    slot = next(iter(iteration.release.slots.values()))
    iteration.release._path(slot["slot_id"], "evaluations").unlink()
    write_json(iteration.release._path(slot["slot_id"], "evaluation_started"), {})
    iteration.score(slot)
    assert runner.read_json(iteration.release._path(slot["slot_id"], "evaluations"))["status"] == "interrupted_evaluation"


def test_writing_summary_uses_native_bounds_and_failure_scores_remain_null(tmp_path):
    iteration = iteration_fixture(tmp_path)
    iteration.registration = {"registration_hash": "registered", "partition": "evolution", "configuration": {}}
    iteration.references = []
    slot = next(iter(iteration.release.slots.values()))
    slot.update(benchmark="writingbench", bounds=[1, 10])
    write_json(iteration.release._path(slot["slot_id"]), {"status": "submitted", "slot": slot,
               "submission": {"answer": "answer", "answer_hash": digest("answer")},
               "outcome": {"budget": {"tokens": 20}}})
    write_json(iteration.release._path(slot["slot_id"], "evaluations"),
               {"slot": slot, "complete": True, "official_score": 7 / 9, "evaluation": {"raw": {"native_mean": 8}}})
    report = runner.OursIteration.summary(iteration)
    assert report["rows"][0]["native_score"] == 8
    assert report["rows"][0]["normalized_score"] == pytest.approx(7 / 9)
    assert report["rows"][1]["native_score"] is None
    assert report["by_benchmark"]["deepsearchqa"]["normalized_mean_failure_zero"] is None


def test_interrupted_generation_retains_settled_budget(tmp_path):
    iteration = iteration_fixture(tmp_path)
    iteration.pipeline = lambda slot: pytest.fail("Interrupted generation cannot resample")
    slot = next(iter(iteration.release.slots.values()))
    budget = {"model_calls": 3, "tokens": 123, "reserved_tokens": 0, "tool_calls": 0}
    write_json(iteration.release._path(slot["slot_id"], "submission_started"), {})
    write_json(tmp_path / "runs" / slot["slot_id"] / "attempt" / "budget.json", budget)
    iteration.submit(slot)
    assert runner.read_json(iteration.release._path(slot["slot_id"]))["budget"] == budget


def test_summary_rejects_tampered_scored_result(tmp_path):
    iteration = iteration_fixture(tmp_path)
    iteration.registration = {"registration_hash": "registered", "partition": "evolution", "configuration": {}}
    iteration.references = []
    slot = next(iter(iteration.release.slots.values()))
    write_json(iteration.release._path(slot["slot_id"]), {"status": "submitted", "slot": slot,
               "submission": {"answer": "answer", "answer_hash": digest("answer")}, "outcome": {"budget": {}}})
    evaluation = {"slot": slot, "complete": True, "official_score": 0.5}
    write_json(iteration.release._path(slot["slot_id"], "evaluations"), evaluation)
    report = runner.OursIteration.summary(iteration)
    write_json(tmp_path / "summary.json", report)
    evaluation["official_score"] = 1.0
    write_json(iteration.release._path(slot["slot_id"], "evaluations"), evaluation)
    with pytest.raises(runner.CheckpointIntegrityError, match="summary"):
        runner.OursIteration.summary(iteration)


def test_deferred_scoring_enforces_remaining_original_active_time():
    config = SimpleNamespace(max_model_calls=None, max_total_tokens=100, max_tool_calls=None, task_timeout=10)
    ledgers = []

    def create(role, agent, ledger, stage):
        assert role == "judge"
        ledgers.append(ledger)
        return object()

    pipeline = SimpleNamespace(config=config, models=SimpleNamespace(create=create), private_records={"task": {}})
    pipeline.evaluator_factory = lambda judge: SimpleNamespace(evaluate=lambda *args, **kwargs: {
        "task_id": "task", "evaluator_version": "fixture", "score": 1, "complete": True, "feedback": []})
    record = {"slot": {"task_id": "task"}, "submission": {"answer": "answer", "answer_hash": digest("answer")},
              "outcome": {"budget": {"model_calls": 1, "tokens": 10, "tool_calls": 0,
                                       "reserved_tokens": 0, "wall_seconds": 9, "request_queue_idle_seconds": 2}}}
    result = runner.score_submission(pipeline, record)
    assert result["official_score"] == 1
    assert ledgers[0].timeout_seconds == 3
    record["outcome"]["budget"]["wall_seconds"] = 12
    with pytest.raises(TimeoutError, match="Original task") as raised:
        runner.score_submission(pipeline, record)
    assert raised.value.evaluation_budget["model_calls"] == 0
    assert len(ledgers) == 1


def test_iteration_can_raise_rubric_parallelism_without_changing_team_limits():
    config = runner.MASConfig(judge_parallel=64, max_inflight_requests=64)
    assert config.judge_parallel == 64
    assert config.max_inflight_requests == 64
    with pytest.raises(ValueError):
        runner.MASConfig(judge_parallel=65)
