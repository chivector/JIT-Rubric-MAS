"""Offline contract tests for the two-arm ResearchRubrics pilot launcher."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

import scripts.run_rr_two_arm_pilot as pilot
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.schemas import PublicTask
from scripts.models.base import ChatMessage


JOINT_MANIFEST = Path(__file__).resolve().parents[2] / "paper" / "experiments" / "joint_task_splits_v5.json"


def test_pipeline_binds_parallel_judges_to_one_metered_credential_safe_task_ledger(monkeypatch, tmp_path):
    from benchmark.adapter.researchrubrics import split_item

    generator_secret = "offline-execution-key-canary"
    judge_secret = "offline-judge-key-canary"
    monkeypatch.setenv("RR_EXEC_API_KEY", generator_secret)
    monkeypatch.setenv("RR_JUDGE_API_KEY", judge_secret)
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("Network access is forbidden"))
    args = SimpleNamespace(
        data="offline.jsonl", split_path=str(tmp_path / "split.json"), judge_parallel=2,
        exec_model="offline-generator", exec_endpoint="https://example.org/v1",
        judge_model="offline-judge", judge_endpoint="https://example.com/v1", timeout=10,
    )
    config = pilot._config(args)
    manifest = pilot.SplitManifest(test=["synthetic"])
    barrier = Barrier(2)
    created = []

    class Transport:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.calls = []

        def __call__(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            assert self.agent_id != "judge"
            rubric_index = int(self.agent_id.rsplit("_", 1)[1])
            if rubric_index < 2:
                barrier.wait(timeout=5)
            else:
                raise RuntimeError(f"Provider reflected {judge_secret}")
            score = 1.0 if rubric_index == 0 else 0.0
            response = {"verdict": "Satisfied" if score else "Not Satisfied", "score": score,
                        "confidence": 0.7, "reasoning": "Offline scripted judgment.",
                        "evidence_quotes": [], "missing_elements": []}
            return ChatMessage(role="assistant", content="```json\n" + json.dumps(response) + "\n```")

        def get_token_counts(self):
            rubric_index = int(self.agent_id.rsplit("_", 1)[1])
            return {"input_token_count": 11 + 20 * rubric_index, "output_token_count": 3 + 2 * rubric_index}

    class Provider:
        def create(self, role, agent_id, ledger, stage):
            assert role == "judge"
            model = MeteredModel(Transport(agent_id), ledger, stage, agent_id, config.models[role].max_tokens)
            created.append(model)
            return model

    def original_evaluator_factory(_judge):
        pytest.fail("Parallel evaluation used the original sequential factory")

    def make_pipeline(*_args, **_kwargs):
        return SimpleNamespace(models=Provider(), tools=[], evaluator_factory=original_evaluator_factory)

    monkeypatch.setattr(pilot, "make_pipeline", make_pipeline)
    pipeline = pilot._pipeline(config, object(), tmp_path / "parallel", args, manifest, None)
    assert isinstance(pipeline.models, pilot.SafeModels)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000, max_tool_calls=None)
    judge = pipeline.models.create("judge", "judge", ledger, "evaluation")
    identity_only = pipeline.evaluator_factory(None)
    evaluator = pipeline.evaluator_factory(judge)
    assert identity_only.evaluator_version == evaluator.evaluator_version
    _, private = split_item({"sample_id": "synthetic", "prompt": "Explain the public box.", "rubrics": [
        {"criterion": f"Offline criterion {index}", "weight": weight}
        for index, weight in enumerate((5, -3, 2))]})
    with pytest.raises(ValueError, match="task ledger"):
        identity_only.evaluate("answer", private_record=private)
    assert len(created) == 1
    result = evaluator.evaluate("answer", private_record=private)

    assert [model.agent_id for model in created] == ["judge", "judge_rubric_0", "judge_rubric_1", "judge_rubric_2"]
    assert len({id(model.model) for model in created}) == 4
    assert all(model.ledger is ledger for model in created)
    assert not created[0].model.calls
    assert [len(model.model.calls) for model in created[1:]] == [1, 1, 1]
    assert result["input_token_count"] == 42
    assert result["output_token_count"] == 8
    assert result["score"] == 5 / 7
    assert result["failed_count"] == 1
    assert result["complete"] is False
    assert "[REDACTED]" in result["feedback"][2]["error"]
    budget = ledger.snapshot()
    assert budget["model_calls"] == 3
    assert budget["reserved_tokens"] == 0
    assert {record["agent_id"] for record in budget["records"]} == {f"judge_rubric_{index}" for index in range(3)}
    assert all(record["stage"] == "evaluation" for record in budget["records"])
    known_records = [record for record in budget["records"] if not record["estimated"]]
    assert sum(record["input_tokens"] for record in known_records) == 42
    assert sum(record["output_tokens"] for record in known_records) == 8
    artifacts = json.dumps({"evaluation": result, "budget": budget, "config": config.model_dump(mode="json")})
    assert generator_secret not in artifacts
    assert judge_secret not in artifacts

    args.judge_parallel = 1
    sequential = pilot._pipeline(config, object(), tmp_path / "sequential", args, manifest, None)
    assert sequential.evaluator_factory is original_evaluator_factory


def test_manifest_uses_the_frozen_researchrubrics_20_10_33_membership():
    manifest = pilot._manifest(JOINT_MANIFEST)

    assert len(manifest.evolution) == 20
    assert len(manifest.validation) == 10
    assert len(manifest.test) == 33
    assert not set(manifest.evolution) & set(manifest.validation)
    assert not set(manifest.evolution) & set(manifest.test)
    assert not set(manifest.validation) & set(manifest.test)


def test_incomplete_and_failed_tasks_keep_the_test_denominator():
    report = pilot._score_rows([
        {"task_id": "complete", "evaluation": {"score": 0.5, "complete": True}},
        {"task_id": "incomplete", "evaluation": {"score": 1.0, "complete": False}},
    ], [{"task_id": "failed", "error_type": "SyntheticError"}],
        ["complete", "incomplete", "failed"])

    assert report["denominator"] == 3
    assert report["completed"] == report["scored_denominator"] == 1
    assert [row["score"] for row in report["tasks"]] == [0.5, None, None]


def test_test_release_seals_all_66_slots_including_generation_failures(tmp_path):
    manifest = pilot._manifest(JOINT_MANIFEST)
    reports = {}
    for arm in ("baseline", "ours"):
        run_dir = tmp_path / arm / "submission"
        run_dir.mkdir(parents=True)
        answer = f"Synthetic {arm} answer."
        submission = {"answer": answer, "answer_hash": pilot.digest(answer)}
        (run_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
        outcome = {"task_id": manifest.test[0], "status": "submitted_unscored", "evaluation": None,
                   "run_dir": str(run_dir), "answer_hash": submission["answer_hash"], "proposals": []}
        reports[arm] = {"submitted_outcomes": [outcome]}

    release = pilot._test_release(tmp_path, manifest, reports)

    assert len(release.slots) == 66
    assert (tmp_path / "test_release" / "seal.json").is_file()
    failed = release.evaluate(f"baseline:{manifest.test[1]}",
                              lambda _record: pytest.fail("Generation failure must not call a judge"))
    assert failed["status"] == "submission_failed"
    assert failed["official_score"] is None
    tampered = tmp_path / "baseline" / "submission" / "submission.json"
    tampered.write_text(json.dumps({"answer": "Changed", "answer_hash": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        pilot._test_release(tmp_path / "another", manifest, reports)


def _synthetic_pinned_inputs(tmp_path):
    joint = json.loads(JOINT_MANIFEST.read_text(encoding="utf-8"))
    membership = joint["memberships"]["researchrubrics"]
    selected = membership["evolution"] + membership["validation"] + membership["test"]
    rows = [{"sample_id": task_id, "prompt": f"Explain synthetic box {task_id}.",
             "rubrics": [{"criterion": "Explain the box accurately.", "weight": 1, "axis": "accuracy"}]}
            for task_id in selected]
    data = tmp_path / "processed_data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    expected_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    joint["benchmarks"]["researchrubrics"]["dataset_sha256"] = expected_hash
    joint_path = tmp_path / "joint.json"
    joint_path.write_text(json.dumps(joint), encoding="utf-8")
    return data, joint_path, expected_hash


def test_baseline_only_release_registers_33_slots_and_retains_failed_task(tmp_path):
    manifest = pilot._manifest(JOINT_MANIFEST)
    reports = {"baseline": {"submitted_outcomes": [], "failures": [
        {"task_id": manifest.test[-1], "error_type": "ConnectionFailure",
         "budget": {"usage_unknown": True}}]}}
    release = pilot._test_release(tmp_path, manifest, reports)

    assert list(release.slots) == [f"baseline:{task_id}" for task_id in manifest.test]
    assert len(release.slots) == 33
    assert len(pilot._read(tmp_path / "test_release" / "seal.json")["submission_hashes"]) == 33
    failed = release.evaluate(f"baseline:{manifest.test[-1]}",
                              lambda _record: pytest.fail("Failed generation cannot call a judge"))
    assert failed["status"] == "submission_failed"
    assert failed["official_score"] is None
    assert pilot._paired_comparison(reports, manifest)["available"] is False


def test_main_baseline_only_calls_baseline_and_seals_only_33_test_slots(monkeypatch, tmp_path):
    data, joint, _expected_hash = _synthetic_pinned_inputs(tmp_path)
    manifest = pilot._manifest(joint)
    calls = []
    monkeypatch.setenv("RR_EXEC_API_KEY", "offline-generator-key")
    monkeypatch.setenv("RR_JUDGE_API_KEY", "offline-judge-key")
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("Network access is forbidden"))

    def fake_baseline(args, config, received_manifest, evidence_dir, output):
        calls.append("baseline")
        assert received_manifest.test == manifest.test
        assert args.arm == "baseline"
        output.mkdir()
        outcomes = []
        for task_id in manifest.test:
            task_dir = output / "tasks" / task_id
            task_dir.mkdir(parents=True)
            answer = f"Synthetic answer for {task_id}."
            submission = {"answer": answer, "answer_hash": pilot.digest(answer)}
            pilot.write_json(task_dir / "submission.json", submission)
            outcomes.append({"task_id": task_id, "status": "submitted_unscored", "evaluation": None,
                             "run_dir": str(task_dir), "answer_hash": submission["answer_hash"], "proposals": []})
        return {"arm": "single_agent_direct", "status": "submitted", "submitted_outcomes": outcomes,
                "failures": [], "results": {"mean_score": None, "completed": 0}}

    def fake_scoring(args, config, received_manifest, evidence_dir, output, reports):
        calls.append("score")
        assert set(reports) == {"baseline"}
        release = pilot._test_release(output, received_manifest, reports)
        assert len(release.slots) == 33
        assert all(slot["method"] == "baseline" for slot in release.slots.values())
        reports["baseline"].update(status="completed", results={"completed": 33, "denominator": 33,
                                                               "scored_denominator": 33, "mean_score": 0.5})

    monkeypatch.setattr(pilot, "run_baseline", fake_baseline)
    monkeypatch.setattr(pilot, "run_ours", lambda *_args, **_kwargs: pytest.fail("Baseline-only launched ours"))
    monkeypatch.setattr(pilot, "_score_deferred", fake_scoring)
    output = tmp_path / "baseline-only"

    assert pilot.main(["--data", str(data), "--joint-manifest", str(joint), "--output", str(output),
                       "--closed-book", "--arm", "baseline"]) == 0

    assert calls == ["baseline", "score"]
    comparison = pilot._read(output / "comparison.json")
    assert set(comparison["reports"]) == {"baseline"}
    assert comparison["paired"]["available"] is False
    assert comparison["metadata"]["selected_arm"] == "baseline"
    assert comparison["metadata"]["parallel_arms"] is False
    assert len(pilot._read(output / "test_release" / "inventory.json")["slots"]) == 33
    assert not (output / "ours").exists()


def _saved_baseline_generation(tmp_path):
    data, joint, expected_hash = _synthetic_pinned_inputs(tmp_path)
    manifest = pilot._manifest(joint)
    args = SimpleNamespace(data=str(data), joint_manifest=str(joint), exec_model="offline-generator",
                           exec_endpoint="https://example.org/v1", judge_model="offline-judge",
                           judge_endpoint="https://example.com/v1", timeout=10, arm="baseline",
                           closed_book=True, reuse_baseline_from=tmp_path / "source")
    config = pilot._config(args)
    source = args.reuse_baseline_from
    task_id = manifest.test[0]
    run_dir = source / "baseline" / "tasks" / task_id
    run_dir.mkdir(parents=True)
    answer = f"Saved synthetic answer for {task_id}."
    submission = {"answer": answer, "answer_hash": pilot.digest(answer), "submitted_at": "offline-time"}
    budget = {"model_calls": 1, "tokens": 8, "reserved_tokens": 0, "tool_calls": 0, "records": [
        {"call_id": "saved-generation", "kind": "model", "stage": "inference", "agent_id": "direct",
         "input_tokens": 5, "output_tokens": 3, "estimated": False}]}
    outcome = {"task_id": task_id, "method": "direct_single", "mode": "evaluate",
               "status": "submitted_unscored", "evaluation": None, "run_dir": str(run_dir.resolve()),
               "answer_hash": submission["answer_hash"], "submitted_at": submission["submitted_at"],
               "budget": budget, "proposals": [], "experience_updates": []}
    metadata = {"manifest": manifest.model_dump(mode="json"), "execution_model": args.exec_model,
                "data_sha256": expected_hash, "knowledge_policy": "model_general_knowledge_allowed",
                "config": {"models": {"exec": config.models["exec"].model_dump(mode="json")}}}
    pilot.write_json(source / "pilot_metadata.json", metadata)
    pilot.write_json(source / "generation_reports.json", {"baseline": {"submitted_outcomes": [outcome]}})
    pilot.write_json(run_dir / "submission.json", submission)
    pilot.write_json(run_dir / "complete.json", outcome)
    pilot.write_json(run_dir / "budget.json", budget)
    return args, config, manifest, outcome


def test_baseline_reuse_preserves_frozen_answer_and_budget_with_import_provenance(tmp_path):
    args, config, manifest, saved = _saved_baseline_generation(tmp_path)
    source = args.reuse_baseline_from
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}

    reusable = pilot._baseline_reuse(args, manifest, config)
    imported = pilot._import_baseline_outcome(reusable[saved["task_id"]], tmp_path / "destination")

    assert list(reusable) == [manifest.test[0]]
    assert imported["answer_hash"] == saved["answer_hash"]
    assert imported["budget"] == saved["budget"]
    assert imported["generation_reused_from"] == saved["run_dir"]
    assert "_reuse_source_dir" not in imported
    destination = Path(imported["run_dir"])
    assert destination == (tmp_path / "destination" / "tasks" / saved["task_id"]).resolve()
    assert pilot._read(destination / "complete.json") == imported
    assert (destination / "submission.json").read_bytes() == (Path(saved["run_dir"]) / "submission.json").read_bytes()
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("change", ["answer", "outcome_hash", "model", "split_order", "dataset_hash",
                                    "unknown_budget", "reserved_budget", "duplicate_id", "outside_test",
                                    "already_scored", "execution_config", "knowledge_policy"])
def test_baseline_reuse_rejects_changed_or_unsettled_generation(tmp_path, change):
    args, config, manifest, saved = _saved_baseline_generation(tmp_path)
    source = args.reuse_baseline_from
    metadata = pilot._read(source / "pilot_metadata.json")
    reports = pilot._read(source / "generation_reports.json")
    outcome = reports["baseline"]["submitted_outcomes"][0]
    if change == "answer":
        submission_path = Path(saved["run_dir"]) / "submission.json"
        submission = pilot._read(submission_path)
        submission["answer"] = "Changed answer."
        pilot.write_json(submission_path, submission)
    elif change == "outcome_hash":
        outcome["answer_hash"] = "invalid"
    elif change == "model":
        metadata["execution_model"] = "another-generator"
    elif change == "split_order":
        metadata["manifest"]["test"].reverse()
    elif change == "dataset_hash":
        metadata["data_sha256"] = "invalid"
    elif change == "unknown_budget":
        outcome["budget"]["usage_unknown"] = True
    elif change == "reserved_budget":
        outcome["budget"]["reserved_tokens"] = 1
    elif change == "duplicate_id":
        reports["baseline"]["submitted_outcomes"].append(dict(outcome))
    elif change == "outside_test":
        outcome["task_id"] = manifest.evolution[0]
    elif change == "already_scored":
        outcome["evaluation"] = {"complete": True, "score": 0.5}
    elif change == "execution_config":
        metadata["config"]["models"]["exec"]["max_tokens"] += 1
    elif change == "knowledge_policy":
        metadata["knowledge_policy"] = "fixed_shared_evidence_only"
    pilot.write_json(source / "pilot_metadata.json", metadata)
    pilot.write_json(source / "generation_reports.json", reports)

    with pytest.raises((ValueError, pilot.CheckpointIntegrityError)):
        pilot._baseline_reuse(args, manifest, config)


def _sealed_baseline_with_cached_evaluation(tmp_path):
    args, config, manifest, outcome = _saved_baseline_generation(tmp_path)
    release = pilot._test_release(tmp_path / "destination", manifest,
                                  {"baseline": {"submitted_outcomes": [outcome], "failures": []}})
    row = {"slot": release.slots[f"baseline:{outcome['task_id']}"], "complete": True,
           "official_score": 0.2, "answer_hash": outcome["answer_hash"],
           "evaluation": {"evaluator_version": "offline-evaluator", "complete": True},
           "evaluation_budget": {"reserved_tokens": 0, "model_calls": 1, "tokens": 8, "records": []}}
    first = tmp_path / "first-score"
    second = tmp_path / "second-score"
    filename = f"{pilot.digest(row['slot']['slot_id'])}.json"
    pilot.write_json(first / "test_release" / "evaluations" / filename, row)
    later = {**row, "official_score": 0.9}
    pilot.write_json(second / "test_release" / "evaluations" / filename, later)
    args.reuse_evaluations_from = [first, second]
    return args, release, row, first, second, filename


def test_cached_evaluation_reuse_preserves_first_complete_judgment_and_source(tmp_path):
    args, release, row, first, second, filename = _sealed_baseline_with_cached_evaluation(tmp_path)
    first_path = first / "test_release" / "evaluations" / filename
    second_path = second / "test_release" / "evaluations" / filename
    before = first_path.read_bytes(), second_path.read_bytes()

    cached = pilot._cached_evaluations(args, release, "baseline", "offline-evaluator")

    assert list(cached) == [row["slot"]["task_id"]]
    assert cached[row["slot"]["task_id"]] == {**row, "evaluation_reused_from": str(first_path.resolve())}
    assert (first_path.read_bytes(), second_path.read_bytes()) == before


@pytest.mark.parametrize("skip", ["partial", "evaluator_version"])
def test_cached_evaluation_reuse_skips_incomplete_or_different_judge(tmp_path, skip):
    args, release, row, first, second, filename = _sealed_baseline_with_cached_evaluation(tmp_path)
    if skip == "partial":
        row["complete"] = False
    else:
        row["evaluation"]["evaluator_version"] = "different-evaluator"
    pilot.write_json(first / "test_release" / "evaluations" / filename, row)

    cached = pilot._cached_evaluations(args, release, "baseline", "offline-evaluator")

    assert cached[row["slot"]["task_id"]]["evaluation_reused_from"] == str(
        (second / "test_release" / "evaluations" / filename).resolve())


@pytest.mark.parametrize("invalid", ["answer_hash", "unsettled_budget"])
def test_cached_evaluation_reuse_rejects_changed_answer_or_unsettled_calls(tmp_path, invalid):
    args, release, row, first, _second, filename = _sealed_baseline_with_cached_evaluation(tmp_path)
    if invalid == "answer_hash":
        row["answer_hash"] = "invalid"
    else:
        row["evaluation_budget"]["reserved_tokens"] = 1
    pilot.write_json(first / "test_release" / "evaluations" / filename, row)

    with pytest.raises(ValueError, match="(?i)(answer|unsettled)"):
        pilot._cached_evaluations(args, release, "baseline", "offline-evaluator")


def test_preflight_verifies_actual_bytes_without_credentials_or_network(monkeypatch, tmp_path):
    data, joint, expected_hash = _synthetic_pinned_inputs(tmp_path)
    args = SimpleNamespace(data=str(data), joint_manifest=str(joint), exec_model="synthetic-generator",
                           exec_endpoint="https://example.com/v1", judge_model="synthetic-judge",
                           judge_endpoint="https://example.org/v1", timeout=10)
    for name in ("RR_EXEC_API_KEY", "RR_JUDGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = pilot._config(args)
    manifest = pilot._manifest(joint)

    report = pilot._preflight(args, manifest, config, None, require_credentials=False)

    assert report["data_sha256"] == report["dataset_sha256"] == expected_hash
    assert report["selected_task_count"] == report["task_count"] == 63
    data.write_text(data.read_text(encoding="utf-8").replace("Explain the box accurately.", "Changed rubric."),
                    encoding="utf-8")
    with pytest.raises((ValueError, pilot.CheckpointIntegrityError), match="(?i)(hash|bytes|pinned|dataset)"):
        pilot._preflight(args, manifest, config, None, require_credentials=False)


def test_check_only_never_launches_arms_or_requires_credentials(monkeypatch, tmp_path):
    data, joint, expected_hash = _synthetic_pinned_inputs(tmp_path)
    for name in ("RR_EXEC_API_KEY", "RR_JUDGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pilot, "run_baseline", lambda *_args, **_kwargs: pytest.fail("Check-only launched baseline"))
    monkeypatch.setattr(pilot, "run_ours", lambda *_args, **_kwargs: pytest.fail("Check-only launched ours"))
    output = tmp_path / "checked"

    assert pilot.main(["--data", str(data), "--joint-manifest", str(joint), "--output", str(output),
                       "--check-only", "--closed-book"]) == 0

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("*.json")]
    assert any(report.get("data_sha256") == expected_hash for report in reports)
    assert not (output / "baseline").exists()
    assert not (output / "ours").exists()


@pytest.mark.parametrize("reused_count", [0, 32])
def test_baseline_submits_exactly_one_output_per_test_task_without_network(monkeypatch, tmp_path, reused_count):
    manifest = pilot._manifest(JOINT_MANIFEST)
    args = SimpleNamespace(data="fixture.jsonl", timeout=10)
    config = SimpleNamespace(max_model_calls=10_000, max_total_tokens=1_000_000,
                             models={"exec": SimpleNamespace(max_tokens=8192)})
    calls = []
    args.baseline_reuse = {}
    for task_id in manifest.test[:reused_count]:
        source = tmp_path / "saved" / task_id
        source.mkdir(parents=True)
        answer = f"Saved answer for {task_id}."
        submission = {"answer": answer, "answer_hash": pilot.digest(answer)}
        pilot.write_json(source / "submission.json", submission)
        args.baseline_reuse[task_id] = {"task_id": task_id, "status": "submitted_unscored",
            "evaluation": None, "run_dir": str(source), "_reuse_source_dir": str(source),
            "answer_hash": submission["answer_hash"], "proposals": [],
            "budget": {"model_calls": 1, "tokens": 8, "reserved_tokens": 0, "tool_calls": 0, "records": []}}

    class Model:
        def __call__(self, messages, **_kwargs):
            task_id = json.loads(messages[-1]["content"])["task_id"]
            calls.append(task_id)
            return ChatMessage(role="assistant", content=f"Synthetic answer for {task_id}.")

        def get_token_counts(self):
            return {"input_token_count": 5, "output_token_count": 3}

    class Provider:
        def create(self, role, agent_id, ledger, stage):
            assert role == "exec"
            return MeteredModel(Model(), ledger, stage, agent_id, 8192)

    class FakePipeline:
        tasks = {task_id: PublicTask(task_id=task_id, question="Explain the synthetic box.")
                 for task_id in manifest.test}
        private_records = {task_id: {"sample_id": task_id, "rubrics": []} for task_id in manifest.test}
        models = Provider()
        evaluator_factory = object()

    def fake_pipeline(*_args, **_kwargs):
        return FakePipeline()

    monkeypatch.setattr(pilot, "_pipeline", fake_pipeline)

    report = pilot.run_baseline(args, config, manifest, None, tmp_path / "baseline")

    assert report["arm"] == "single_agent_direct"
    assert report["results"]["denominator"] == len(manifest.test) == 33
    assert report["results"]["completed"] == 0
    assert report["failures"] == []
    assert calls == manifest.test[reused_count:]
    assert len(list((tmp_path / "baseline" / "tasks").iterdir())) == 33
    assert len(list((tmp_path / "baseline").rglob("submission.json"))) == 33
    assert not list(tmp_path.rglob("evaluation.json"))
    assert all(outcome["budget"]["model_calls"] == 1 for outcome in report["submitted_outcomes"])
    assert sum("generation_reused_from" in outcome for outcome in report["submitted_outcomes"]) == reused_count


def test_main_starts_both_arms_before_waiting_for_either(monkeypatch, tmp_path):
    data = tmp_path / "data.jsonl"
    joint = tmp_path / "joint.json"
    data.write_text("{}\n", encoding="utf-8")
    joint.write_text("{}\n", encoding="utf-8")
    manifest = pilot.SplitManifest(seed=1, evolution=["evo"], validation=["val"], test=["test"])
    barrier = Barrier(2)
    starts = []

    monkeypatch.setattr(pilot, "_manifest", lambda _path: manifest)
    config = pilot.MASConfig(backend="scripted")
    monkeypatch.setattr(pilot, "_config", lambda _args: config)
    monkeypatch.setattr(pilot, "_preflight", lambda *_args, **_kwargs: {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "joint_manifest_sha256": hashlib.sha256(joint.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "task_count": 3, "selected_task_count": 3, "evidence_mode": "closed_book_public_tasks"})

    def fake_arm(*_args, **_kwargs):
        starts.append(True)
        barrier.wait(timeout=3)
        return {"results": {"mean_score": 0.5, "completed": 1},
                "test": {"mean_score": 0.5, "completed": 1},
                "submitted_outcomes": [], "failures": [], "test_failures": [], "status": "completed"}

    monkeypatch.setattr(pilot, "run_baseline", fake_arm)
    monkeypatch.setattr(pilot, "run_ours", fake_arm)
    monkeypatch.setattr(pilot, "_score_deferred", lambda *_args, **_kwargs: None, raising=False)

    exit_code = pilot.main(["--data", str(data), "--joint-manifest", str(joint),
                            "--output", str(tmp_path / "pilot"), "--closed-book"])

    assert exit_code == 0
    assert len(starts) == 2
    comparison = json.loads((tmp_path / "pilot" / "comparison.json").read_text(encoding="utf-8"))
    assert set(comparison["reports"]) == {"baseline", "ours"}


def test_fatal_arm_error_preserves_durable_submission_and_checkpoint_provenance(monkeypatch, tmp_path):
    output = tmp_path / "ours"
    output.mkdir()
    durable = {"arm": "ours_evolve_then_test", "status": "submitting", "started_at": "synthetic-start",
               "submitted_outcomes": [{"task_id": "saved", "evaluation": None,
                                        "status": "submitted_unscored"}],
               "test_failures": [{"task_id": "failed", "budget": {"model_calls": 1}}],
               "selected_checkpoint": "C5", "selected_snapshot_hash": "frozen-state",
               "checkpoint_selection": {"selected": {"position": 5}}}
    (output / "report.json").write_text(json.dumps(durable), encoding="utf-8")
    monkeypatch.setenv("RR_EXEC_API_KEY", "secret-canary")

    def failed_arm(*_args, **_kwargs):
        raise SystemExit("provider reflected secret-canary")

    report = pilot._run_arm(failed_arm, None, None, None, None, output)

    assert report["status"] == "failed"
    assert report["error_type"] == "SystemExit"
    assert "secret-canary" not in report["error"]
    for key in ("submitted_outcomes", "test_failures", "started_at", "selected_checkpoint",
                "selected_snapshot_hash", "checkpoint_selection"):
        assert report[key] == durable[key]
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_cost_summary_counts_durable_generation_and_scoring_once(tmp_path):
    generation = {"call_id": "generation", "kind": "model", "stage": "inference", "agent_id": "direct",
                  "input_tokens": 5, "output_tokens": 3, "estimated": False, "wall_seconds": 0.5}
    evaluation = {"call_id": "evaluation", "kind": "model", "stage": "evaluation", "agent_id": "judge",
                  "input_tokens": 10, "output_tokens": 2, "estimated": True, "wall_seconds": 0.25}
    for label in ("original", "reused"):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "budget.json").write_text(json.dumps({"records": [generation]}), encoding="utf-8")
    (tmp_path / "scoring_progress.json").write_text(
        json.dumps({"results": [{"evaluation_budget": {"records": [evaluation]}}]}), encoding="utf-8")

    report = pilot._cost_summary(tmp_path)

    assert report["model_calls"] == 2
    assert report["tokens"] == 20
    assert report["by_stage"]["inference"]["model_calls"] == 1
    assert report["by_stage"]["evaluation"]["estimated_calls"] == 1
    assert report["monetary_cost"] is None


@pytest.mark.parametrize("ours_status", ["submitted", "inconclusive", "failed"])
def test_deferred_scoring_retains_incomplete_and_inconclusive_status_with_real_seal(
        monkeypatch, tmp_path, ours_status):
    manifest = pilot.SplitManifest(test=["complete", "incomplete"])
    reports = {}
    for arm in ("baseline", "ours"):
        store = pilot.ExperienceStore(tmp_path / arm / "experience.sqlite")
        store.close()
        outcomes = []
        for task_id in manifest.test:
            run_dir = tmp_path / arm / task_id
            run_dir.mkdir()
            answer = f"Synthetic {arm} {task_id}."
            submission = {"answer": answer, "answer_hash": pilot.digest(answer)}
            (run_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
            outcomes.append({"task_id": task_id, "status": "submitted_unscored", "evaluation": None,
                             "answer_hash": submission["answer_hash"], "run_dir": str(run_dir), "proposals": []})
        reports[arm] = {"status": "submitted" if arm == "baseline" else ours_status,
                        "submitted_outcomes": outcomes, "test_failures": [], "failures": []}

    monkeypatch.setattr(pilot, "_pipeline", lambda *_args, **_kwargs: object())
    scored = []

    def scorer(_pipeline, record):
        assert (tmp_path / "test_release" / "seal.json").is_file()
        assert len(pilot._read(tmp_path / "test_release" / "seal.json")["submission_hashes"]) == 4
        scored.append(record["slot"]["slot_id"])
        complete = record["slot"]["task_id"] == "complete"
        return {"slot": record["slot"], "complete": complete,
                "official_score": 0.5 if complete else None,
                "evaluation_budget": {"model_calls": 1, "tokens": 10, "records": []}}

    monkeypatch.setattr(pilot, "score_with_pipeline", scorer)

    pilot._score_deferred(SimpleNamespace(), None, manifest, None, tmp_path, reports)

    assert len(scored) == 4
    assert reports["baseline"]["status"] == "incomplete"
    assert reports["ours"]["status"] == ("incomplete" if ours_status == "submitted" else ours_status)
    for arm in ("baseline", "ours"):
        summary = reports[arm]["results" if arm == "baseline" else "test"]
        assert summary["denominator"] == 2
        assert summary["completed"] == summary["scored_denominator"] == 1
        assert summary["all_task_mean"] is None
        assert reports[arm]["experience_store_unchanged_by_test"]


@pytest.mark.parametrize("c0_score,c20_score,selected_label,selected_version", [
    (0.2, 0.8, "C20", 20),
    (0.8, 0.2, "C0", 0),
])
def test_ours_selects_validation_checkpoint_before_any_test_output(
        monkeypatch, tmp_path, c0_score, c20_score, selected_label, selected_version):
    manifest = pilot._manifest(JOINT_MANIFEST)
    args = SimpleNamespace(data="fixture.jsonl", timeout=10)
    config = pilot.MASConfig(backend="scripted")
    events = []
    test_snapshots = []

    class FakePipeline:
        def __init__(self, store, output):
            self.store = store
            self.output = Path(output)
            self.benchmark_dataset = SimpleNamespace(
                lower_bounds={task_id: 0.0 for task_id in manifest.validation},
                upper_bounds={task_id: 1.0 for task_id in manifest.validation},
                dataset_sha256="synthetic-dataset")

        def run(self, mode, task_ids, resume=False):
            assert mode == "evolve"
            assert len(task_ids) == 1
            events.append("evolution")
            snapshot = pilot.ExperienceSnapshot(version=self.store.snapshot().version + 1)
            with self.store.db:
                self.store.db.execute("INSERT INTO snapshots VALUES(?,?)",
                                      (snapshot.version, snapshot.model_dump_json()))
                self.store.db.execute("UPDATE state SET value=? WHERE key='current'",
                                      (str(snapshot.version),))
            return [{"task_id": task_ids[0], "evaluation": {"score": 1.0, "complete": True},
                     "budget": {"model_calls": 1, "tokens": 10}}]

        def run_task(self, task_id, snapshot, **kwargs):
            assert self.store.read_only
            if kwargs["mode"] == "validate":
                events.append(f"C{snapshot.version}")
                score = c0_score if snapshot.version == 0 else c20_score if snapshot.version == 20 else 0.1
                return {"task_id": task_id, "evaluation": {"score": score, "complete": True},
                        "budget": {"model_calls": 1, "tokens": 10}}
            assert kwargs["mode"] == "evaluate"
            assert kwargs.get("defer_evaluation") is True
            events.append("test")
            test_snapshots.append(snapshot)
            return {"task_id": task_id, "evaluation": None, "status": "submitted_unscored",
                    "budget": {"model_calls": 1, "tokens": 10}}

    def fake_pipeline(_config, store, output, *_args, **_kwargs):
        return FakePipeline(store, output)

    monkeypatch.setattr(pilot, "_pipeline", fake_pipeline)

    report = pilot.run_ours(args, config, manifest, None, tmp_path / "ours")

    assert report["selected_checkpoint"] == selected_label
    assert set(report["checkpoints"]) == {"C0", "C5", "C10", "C15", "C20"}
    assert all(events.count(label) == 10 for label in report["checkpoints"])
    assert events.index("C20") < events.index("test")
    assert events.count("evolution") == 20
    assert report["test_failures"] == []
    assert len(test_snapshots) == len(manifest.test) == 33
    assert all(snapshot.version == selected_version for snapshot in test_snapshots)
