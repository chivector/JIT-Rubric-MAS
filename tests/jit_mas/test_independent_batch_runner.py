"""Runner tests use recorded artifacts and never call a model API."""

from types import SimpleNamespace

import pytest

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.independent_batch_campaign import state_sources
from jit_mas.pipeline import write_json
from jit_mas.schemas import AgentPoolOperation, ExperienceSnapshot, digest
from scripts.run_independent_batch_experiment import IndependentBatchEnvironment, IndependentEnvironment


class RecordedCampaign:
    def __init__(self, snapshot, tasks):
        self.registration_hash = "registration"
        self.snapshot = snapshot
        self.trajectories = {("researchrubrics", 0): {"batches": [{"evolution_task_ids": tasks}]}}
        self.candidates, self.finished, self.claims = {}, [], []

    def checkpoint(self, source, run_id, position):
        return {"snapshot": self.snapshot}

    def candidate(self, source, run_id, batch_index, candidate_index):
        return self.candidates.get(candidate_index)

    def freeze_candidates(self, claim, candidates):
        self.candidates = {row["candidate_index"]: row for row in candidates}

    def finish(self, claim, result, *, status):
        self.finished.append((claim, result, status))

    def interrupted_claims(self):
        return self.claims

    def advance(self):
        return None


def environment(tmp_path):
    instance = IndependentBatchEnvironment.__new__(IndependentBatchEnvironment)
    instance.output = tmp_path
    snapshot = ExperienceSnapshot()
    tasks = [f"task{index}" for index in range(5)]
    instance.campaign = RecordedCampaign(snapshot, tasks)
    instance.assert_frozen = lambda: None
    instance._frozen_store = lambda checkpoint: SimpleNamespace(snapshot=lambda: snapshot, close=lambda: None)
    claim = {"slot_id": "update:researchrubrics:run0:b0", "kind": "batch_evolution", "task_id": "batch:rr0",
             "source": "researchrubrics", "run_id": 0, "batch_index": 0, "state_hash": digest(snapshot)}
    return instance, claim, tasks


def recorded_budget(call_id, tokens):
    return {"model_calls": 1, "tokens": tokens, "reserved_tokens": 0,
            "records": [{"call_id": call_id, "kind": "model", "stage": "update",
                         "input_tokens": tokens, "output_tokens": 0, "estimated": False}]}


def candidate_receipt(claim, tasks, index):
    snapshot = ExperienceSnapshot(version=index + 1)
    return {"candidate_index": index, "status": "complete", "snapshot": snapshot.model_dump(mode="json"),
            "state_hash": digest(snapshot), "base_state_hash": claim["state_hash"], "source_task_ids": tasks,
            "budget": recorded_budget(f"candidate{index}", 10 + index)}


def test_evo_runner_executes_all_five_tasks_from_frozen_state_without_attribution(tmp_path):
    instance, claim, tasks = environment(tmp_path)
    calls = []

    def run_task(task_id, snapshot, **kwargs):
        calls.append((task_id, digest(snapshot), kwargs))
        return {"task_id": task_id, "evaluation": {"complete": True, "score": 0.5},
                "experience_updates": [], "budget": recorded_budget(task_id, 2)}

    instance._pipeline = lambda *args: SimpleNamespace(run_task=run_task)
    for task_id in tasks:
        instance.execute({**claim, "kind": "evolution", "slot_id": task_id, "task_id": task_id})
    assert [row[0] for row in calls] == tasks
    assert {row[1] for row in calls} == {claim["state_hash"]}
    assert all(row[2]["mode"] == "evolve" and row[2]["attribution"] is False for row in calls)
    assert all(result["after_snapshot_hash"] == claim["state_hash"] for _, result, _ in instance.campaign.finished)


@pytest.mark.parametrize("aggregate", [True, False])
def test_recovery_preserves_completed_candidate_files_and_failed_reflection_costs(tmp_path, aggregate):
    instance, claim, tasks = environment(tmp_path)
    instance.campaign.claims = [claim]
    instance._pipeline = lambda *args: pytest.fail("Recovery must not make model calls")
    root = instance.slot_directory(claim)
    candidates = [candidate_receipt(claim, tasks, index) for index in range(3 if aggregate else 2)]
    reflections = [{"task_id": tasks[0], "status": "failed", "budget": recorded_budget("reflection", 7)}]
    if aggregate:
        write_json(root / "batch_evolution.json", {"batch_id": claim["task_id"], "source_task_ids": tasks,
            "base_state_hash": claim["state_hash"], "no_intra_batch_evolution": True,
            "reflections": reflections, "candidates": candidates})
    else:
        write_json(root / "reflections" / "first" / "reflection.json", reflections[0])
        for candidate in candidates:
            write_json(root / f"candidate{candidate['candidate_index']}.json", candidate)
    instance.recover()
    _, result, status = instance.campaign.finished[0]
    assert status == "complete"
    assert result["budget"]["tokens"] == (40 if aggregate else 28)
    assert result["budget"]["model_calls"] == (4 if aggregate else 3)
    assert result["budget"]["usage_unknown"] is (not aggregate)
    assert instance.campaign.candidates[0]["snapshot"] == candidates[0]["snapshot"]
    assert (instance.campaign.candidates[2]["snapshot"] is not None) is aggregate


def test_durable_candidate_source_or_base_change_is_rejected(tmp_path):
    instance, claim, tasks = environment(tmp_path)
    candidate = candidate_receipt(claim, tasks, 0)
    candidate["source_task_ids"] = ["outside:evo"]
    write_json(instance.slot_directory(claim) / "candidate0.json", candidate)
    with pytest.raises(CheckpointIntegrityError, match="source or input state"):
        instance._batch_artifacts(claim)


def test_state_sources_includes_every_operation_source_task():
    snapshot = ExperienceSnapshot()
    snapshot.agent_pool.structural_history = [AgentPoolOperation(
        operation_id="operation", kind="reorganize", source_task_id="first", source_task_ids=["first", "second"],
        base_pool_version=0, evidence=["feedback:first"], rationale="Observed change", expected_benefit="Coverage",
        token_cost_tradeoff="Recorded cost")]
    assert state_sources(snapshot) == {"first", "second"}


@pytest.mark.parametrize("name, value", [
    ("JIT_MAS_MODEL_ATTEMPTS", "2"),
    ("JIT_MAS_RETRY_BASE_SECONDS", "2"),
    ("JIT_MAS_RETRY_MAX_SECONDS", "20"),
    ("JIT_MAS_RETRY_JITTER_SECONDS", "0"),
    ("MODULAR_AGENT_API_FAILURE_ACTION", "raise"),
])
def test_frozen_transport_identity_rejects_changed_retry_settings(monkeypatch, name, value):
    monkeypatch.setenv("JIT_MAS_MODEL_ATTEMPTS", "1")
    monkeypatch.setenv("JIT_MAS_RETRY_BASE_SECONDS", "5")
    monkeypatch.setenv("JIT_MAS_RETRY_MAX_SECONDS", "30")
    monkeypatch.setenv("JIT_MAS_RETRY_JITTER_SECONDS", "1")
    monkeypatch.setenv("MODULAR_AGENT_API_FAILURE_ACTION", "exit")
    monkeypatch.setattr(IndependentEnvironment, "assert_frozen", lambda _self: None)
    instance = IndependentBatchEnvironment.__new__(IndependentBatchEnvironment)
    instance.identity = {"transport": instance._transport_identity()}
    assert instance.identity["transport"]["connection_retry_version"] == "bounded-exponential-jitter-v1"
    instance.assert_frozen()
    monkeypatch.setenv(name, value)
    with pytest.raises(CheckpointIntegrityError, match="transport retry"):
        instance.assert_frozen()
