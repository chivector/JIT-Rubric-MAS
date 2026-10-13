import json
from types import SimpleNamespace

import pytest

from jit_mas.agent_pool import effective_pool
from jit_mas.schemas import AgentPoolSnapshot, ExperienceSnapshot, SplitManifest, digest
from scripts.mas_baseline_methods import JudgeEnvelopeModel
from scripts.summarize_high_score_experiment import summarize
import scripts.run_high_score_native_experiment as experiment


class Store:
    def __init__(self):
        self.state = ExperienceSnapshot()

    def snapshot(self):
        return self.state.model_copy(deep=True)

    def task_run(self, mode, task_id):
        return None

    def close(self):
        pass


def outcome(task_id, directory):
    return {"task_id": task_id, "run_dir": str(directory), "evaluation": {"score": 0.8, "complete": True},
            "budget": {"model_calls": 2, "tokens": 100}, "pool_operations": [{"kind": "add"}]}


def test_evolution_commits_and_uncommitted_operations_are_not_applied(tmp_path):
    store = Store()
    calls = []

    def evolve(mode, task_ids):
        calls.append((mode, task_ids))
        store.state.version += 1
        store.state.agent_pool = effective_pool(store.state.agent_pool)
        store.state.agent_pool.profiles[0].prompt += " Updated prompt."
        return [outcome(task_ids[0], tmp_path / "run")]

    pipeline = SimpleNamespace(run=evolve)
    record = experiment.execute_slot(pipeline, store, [], tmp_path, "writingbench", "evolution", 1, "task:1")
    assert calls == [("evolve", ["task:1"])]
    assert record["experience_version_after"] == 1
    assert record["harness_modification_count"] == 1
    assert record["operations_requested"] == [{"kind": "add"}]
    assert record["operations_applied"] == []
    assert record["agent_pool_size_before"] == len(effective_pool(AgentPoolSnapshot()).profiles)


def test_version_only_update_is_not_harness_modification(tmp_path):
    before = effective_pool(AgentPoolSnapshot()).model_dump(mode="json")
    after = json.loads(json.dumps(before))
    after["profiles"][0]["version"] += 1
    changes = experiment._changes(before, after)
    assert changes["harness_modification_count"] == 0
    record = {"task_id": "writingbench:1", "benchmark": "writingbench", "phase": "evolution",
              "round_index": 0, "status": "complete", "pool_before": before, "pool_after": after,
              "budget": {"model_calls": 0, "tokens": 0}, **changes}
    experiment._append(tmp_path / "records.jsonl", record)
    result = summarize(tmp_path)
    curve = result["curves"]["writingbench"]["root:native"]["jsonl"]
    exported = json.loads(open(curve, encoding="utf-8").read())
    assert exported["harness_modification_count"] == 0
    assert exported["harness_changed_agent_count"] == 0
    assert exported["cumulative_harness_changed_agent_count"] == 0


def test_client_unwrapping_and_judge_envelope():
    client = SimpleNamespace(close=lambda: None)
    wrapped = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(client=client)))
    provider = SimpleNamespace(create=lambda *args, **kwargs: wrapped)
    clients = experiment._track_clients(provider)
    assert provider.create("exec", "agent", None, "inference") is wrapped
    assert isinstance(provider.create("judge", "judge", None, "evaluation"), JudgeEnvelopeModel)
    assert clients == [client, client]


def test_failure_budget_is_retained_and_next_task_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "synthetic-secret")
    store = Store()
    calls = []

    def execute(task_id, snapshot, **kwargs):
        calls.append(task_id)
        if task_id == "task:failed":
            error = RuntimeError("failed synthetic-secret")
            error.jit_mas_run_failure = {"budget": {"model_calls": 1, "tokens": 50}}
            raise error
        return outcome(task_id, tmp_path / "run")

    pipeline = SimpleNamespace(run_task=execute)
    failure = experiment.execute_slot(pipeline, store, [], tmp_path, "writingbench", "test", 1, "task:failed")
    complete = experiment.execute_slot(pipeline, store, [], tmp_path, "writingbench", "test", 2, "task:ok")
    assert calls == ["task:failed", "task:ok"]
    assert failure["status"] == "failed"
    assert failure["budget"]["tokens"] == 50
    assert complete["status"] == "complete"
    assert "synthetic-secret" not in (tmp_path / "records.jsonl").read_text(encoding="utf-8")


def test_resume_skips_success_and_failure_receipts(tmp_path, monkeypatch):
    store = Store()
    calls = []
    pipeline = SimpleNamespace(tasks={}, models=SimpleNamespace(create=lambda *args: None))

    def execute(task_id, snapshot, **kwargs):
        calls.append(task_id)
        if task_id == "task:failed":
            raise RuntimeError("synthetic failure")
        return outcome(task_id, tmp_path / "run")

    pipeline.run_task = execute
    monkeypatch.setattr(experiment, "ExperienceStore", lambda path: store)
    monkeypatch.setattr(experiment, "make_pipeline", lambda *args, **kwargs: pipeline)
    config = SimpleNamespace(check_native=lambda: None, model_dump=lambda **kwargs: {})
    materials = [{"name": "writingbench", "data": tmp_path / "data", "attached": {}, "checker": None,
                  "split": SplitManifest(test=["task:failed", "task:ok"])}]
    directory = tmp_path / "output"
    experiment.run(config, materials, {"identity": "same"}, directory)
    report = experiment.run(config, materials, {"identity": "same"}, directory, resume=True)
    assert calls == ["task:failed", "task:ok"]
    assert report["record_count"] == 2


def test_resume_recovers_partial_jsonl_from_slot_without_rerunning(tmp_path, monkeypatch):
    store = Store()
    calls = []
    pipeline = SimpleNamespace(tasks={}, models=SimpleNamespace(create=lambda *args: None))

    def execute(task_id, snapshot, **kwargs):
        calls.append(task_id)
        return outcome(task_id, tmp_path / "run")

    pipeline.run_task = execute
    monkeypatch.setattr(experiment, "ExperienceStore", lambda path: store)
    monkeypatch.setattr(experiment, "make_pipeline", lambda *args, **kwargs: pipeline)
    config = SimpleNamespace(check_native=lambda: None, model_dump=lambda **kwargs: {})
    materials = [{"name": "writingbench", "data": tmp_path / "data", "attached": {}, "checker": None,
                  "split": SplitManifest(test=["task:1", "task:2"])}]
    directory = tmp_path / "output"
    experiment.run(config, materials, {"identity": "same"}, directory)
    record_path = directory / "records.jsonl"
    original = record_path.read_bytes().splitlines(keepends=True)
    partial = original[1][:30]
    record_path.write_bytes(original[0] + partial)
    report = experiment.run(config, materials, {"identity": "same"}, directory, resume=True)
    assert calls == ["task:1", "task:2"]
    assert report["record_count"] == 2
    assert len(experiment._records(record_path)) == 2
    backups = list(directory.glob("records.jsonl.partial-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == partial


def test_record_recovery_preserves_complete_final_line_and_rejects_other_corruption(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"slot_id":"first"}')
    assert experiment._records(path, recover=True) == [{"slot_id": "first"}]
    experiment._append(path, {"slot_id": "second"})
    assert len(experiment._records(path)) == 2
    for content in (b'{"slot_id":\n', b'{"slot_id":\n{"slot_id":"second"}\n'):
        path.write_bytes(content)
        with pytest.raises(json.JSONDecodeError):
            experiment._records(path, recover=True)
        assert path.read_bytes() == content


def test_interrupted_resume_recovers_budget_without_model_calls(tmp_path):
    store = Store()
    run = tmp_path / "runs" / "saved"
    run.mkdir(parents=True)
    baseline = store.snapshot()
    experiment.write_json(run / "run_manifest.json", {"comparison": {"task": {"task_id": "task:1"}},
                          "mode": "evaluate", "experience_hash": digest(baseline)})
    experiment.write_json(run / "budget.json", {"model_calls": 3, "tokens": 125})
    experiment.write_json(run / "harness.json", {"repair_count": 2, "meta_trajectory": [{"stage": "repair"}]})
    started = {"phase": "test", "task_id": "task:1", "pool_before": experiment._pool(baseline),
               "snapshot_before_hash": digest(baseline)}
    record = experiment._interrupted_record(started, store, tmp_path / "runs")
    assert record["status"] == "interrupted"
    assert record["budget"]["tokens"] == 125
    assert record["harness_repair_count"] == 2


def test_evidence_loader_receives_full_manifest_for_selected_subset(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": "native_jit", "unsafe_local": True}), encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text("data", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    experiment.write_json(evidence / "manifest.json", {"tasks": {
        task_id: {"file": task_id + ".json", "file_sha256": "sha"} for task_id in ("task:1", "task:2")}})
    tasks = {"task:1": object(), "task:2": object()}
    calls = []
    monkeypatch.setattr(experiment, "load_benchmark", lambda *args, **kwargs:
                        SimpleNamespace(tasks=tasks, dataset_sha256="data-sha"))

    def load(selected, location):
        calls.append(set(selected))
        return selected

    monkeypatch.setattr(experiment, "load_evidence_tasks", load)
    plan = tmp_path / "plan.json"
    experiment.write_json(plan, {"benchmarks": [{"name": "researchrubrics", "data": str(data),
                          "evidence_dir": str(evidence), "test_ids": ["task:1"]}]})
    _, materials, _ = experiment.load_inputs(config, plan)
    assert calls == [{"task:1", "task:2"}]
    assert materials[0]["split"].test == ["task:1"]


def test_initial_snapshot_is_seeded_before_test(tmp_path, monkeypatch):
    observed = []

    def make(config, store, directory, **kwargs):
        observed.append(store.snapshot().version)
        return SimpleNamespace(tasks={}, models=SimpleNamespace(create=lambda *args: None),
                               run_task=lambda task_id, snapshot, **kwargs: outcome(task_id, directory))

    monkeypatch.setattr(experiment, "make_pipeline", make)
    config = SimpleNamespace(check_native=lambda: None, model_dump=lambda **kwargs: {})
    initial = ExperienceSnapshot(version=4, agent_pool=effective_pool(AgentPoolSnapshot()))
    materials = [{"name": "writingbench", "data": tmp_path / "data", "attached": {}, "checker": None,
                  "split": SplitManifest(test=["task:1"]), "initial_snapshot": initial}]
    report = experiment.run(config, materials, {"identity": "same"}, tmp_path / "output")
    records = experiment._records(tmp_path / "output" / "records.jsonl")
    assert observed == [4]
    assert report["record_count"] == 1
    assert records[0]["snapshot_before_hash"] == digest(initial)
