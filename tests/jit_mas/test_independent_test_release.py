from pathlib import Path
import json
import threading
from types import SimpleNamespace

import pytest

from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest
from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from scripts.run_independent_test_release import IndependentTestReleaseRunner
from scripts.run_jit_mas import make_pipeline


class FakeCampaign:
    protocol = {"target_test_tasks": {"deepresearch_bench_ii": ["drb:1"]}}
    trajectories = {
        ("researchrubrics", 0): {"test_task_ids": ["rr:1", "rr:2"],
                                "evolution_task_ids": ["evo:1"]},
    }

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def selection(self, source, run_id):
        if source == "researchrubrics" and run_id == 0:
            return {"selected": {"position": 5, "state_hash": digest(self.snapshot)}}
        return None

    def checkpoint(self, source, run_id, position):
        return {"state_hash": digest(self.snapshot), "snapshot": self.snapshot}


def runner(tmp_path, snapshot=None):
    instance = object.__new__(IndependentTestReleaseRunner)
    instance.campaign = FakeCampaign(snapshot or ExperienceSnapshot())
    instance.output = Path(tmp_path)
    instance._split_lock = threading.Lock()
    return instance


def test_test_split_is_target_only_and_immutable(tmp_path):
    instance = runner(tmp_path)
    path = instance._split_path("deepresearch_bench_ii")
    assert path.exists()
    assert instance._test_tasks("deepresearch_bench_ii") == ["drb:1"]
    assert instance._test_tasks("researchrubrics") == ["rr:1", "rr:2"]
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest = SplitManifest.model_validate(document["runtime_split_manifest"])
    assert manifest.evolution == []
    assert manifest.validation == []
    assert manifest.test == ["drb:1"]
    assert manifest.stream == []
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        instance._split_path("deepresearch_bench_ii")


def test_static_and_selected_states_are_source_bound(tmp_path):
    snapshot = ExperienceSnapshot()
    instance = runner(tmp_path, snapshot)
    static = {"source": "static", "run_id": None}
    assert instance._ready_state(static) == snapshot
    selected = {"source": "researchrubrics", "run_id": 0,
                "state_hash": digest(snapshot)}
    assert instance._ready_state(selected) == snapshot


def test_adapter_identity_sidecar_is_write_once(tmp_path):
    instance = object.__new__(IndependentTestReleaseRunner)
    instance.output = Path(tmp_path)
    instance._identity_path = instance.output / "test_adapter_identity.json"
    instance.campaign = SimpleNamespace(protocol={"protocol_sha256": "protocol"})
    instance.environment = SimpleNamespace(identity={
        "launch_file_sha256": "launch", "configuration_file_sha256": "config",
        "joint_manifest_sha256": "manifest", "runner_sha256": "runner"})
    instance._assert_adapter_identity(create=True)
    body = json.loads(instance._identity_path.read_text(encoding="utf-8"))
    body["protocol_sha256"] = "changed"
    instance._identity_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        instance._assert_adapter_identity()


def test_interrupted_test_retains_original_budget_without_resampling(tmp_path):
    instance = runner(tmp_path)
    claim = {"kind": "test", "task_id": "drb:1", "slot_id": "one",
             "state_hash": digest(ExperienceSnapshot())}
    budget_path = tmp_path / "test_runs" / digest("one") / "attempt" / "budget.json"
    budget_path.parent.mkdir(parents=True)
    budget_path.write_text(json.dumps({"model_calls": 2, "tokens": 123}), encoding="utf-8")
    recorded = []
    instance.campaign = SimpleNamespace(
        interrupted_claims=lambda: [claim],
        finish=lambda actual, result, status: recorded.append((actual, result, status)))
    instance._recover_interrupted_tests()
    actual, result, status = recorded[0]
    assert actual == claim and status == "failed"
    assert result["budget"] == {"model_calls": 2, "tokens": 123}
    assert result["recovered"] is True


def test_interrupted_test_recovers_durable_unscored_artifact(tmp_path):
    instance = runner(tmp_path)
    claim = {"kind": "test", "task_id": "drb:1", "slot_id": "one",
             "state_hash": digest(ExperienceSnapshot())}
    run_dir = tmp_path / "test_runs" / digest("one") / "attempt"
    run_dir.mkdir(parents=True)
    outcome = {"task_id": "drb:1", "experience_hash": claim["state_hash"],
               "status": "submitted_unscored", "answer_hash": digest("answer"),
               "budget": {"model_calls": 2, "tokens": 123}}
    (run_dir / "complete.json").write_text(json.dumps(outcome), encoding="utf-8")
    recorded = []
    instance.campaign = SimpleNamespace(
        interrupted_claims=lambda: [claim],
        submit_test=lambda *args, **kwargs: recorded.append((args, kwargs)))
    instance._recover_interrupted_tests()
    assert recorded[0][0][1] == run_dir / "submission.json"
    assert recorded[0][1]["budget"] == outcome["budget"]


def test_submit_claims_only_when_a_worker_is_available(tmp_path):
    instance = runner(tmp_path)
    events = []
    pending = [{"slot_id": str(index)} for index in range(3)]

    def claim(*args, **kwargs):
        if not pending:
            return None
        item = pending.pop(0)
        events.append(("claim", item["slot_id"]))
        return item

    def submit(item):
        events.append(("submit", item["slot_id"]))
        return "submitted"

    instance.campaign = SimpleNamespace(claim=claim, records=lambda **kwargs: [])
    instance._assert_adapter_identity = lambda: None
    instance._recover_interrupted_tests = lambda: None
    instance._select_ready_trajectories = lambda: None
    instance._submit_claim = submit
    report = instance.submit(workers=1)
    assert report["submitted"] == 3
    assert events == [(phase, str(index)) for index in range(3) for phase in ("claim", "submit")]


def test_interrupted_judgment_never_calls_the_pipeline_again(tmp_path):
    instance = runner(tmp_path)
    row = {"slot_id": "one", "result_hash": "hash"}
    started = tmp_path / "test_evaluation_started" / f"{digest('one')}.json"
    started.parent.mkdir()
    started.write_text("{}", encoding="utf-8")
    instance._pipeline = lambda *args: pytest.fail("Interrupted scoring was resampled")
    with pytest.raises(RuntimeError, match="silently resampled"):
        instance._score_one(row)


def test_evaluation_receipt_hash_detects_changed_score(tmp_path):
    instance = runner(tmp_path)
    row = {"slot_id": "one", "result": {"submission_hash": "submission"}}
    result = {"slot": row, "submission_sha256": "submission", "official_score": 0.5}
    result["receipt_sha256"] = digest(result)
    path = instance._evaluation_path("one")
    path.parent.mkdir()
    path.write_text(json.dumps(result), encoding="utf-8")
    assert instance._score_one(row) == result
    result["official_score"] = 1.0
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="receipt changed"):
        instance._score_one(row)


def test_score_saved_string_submission_path_without_actor_replay(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs")
        task_id = pipeline.manifest.test[0]
        submission = {"answer": "Boundary conditions: use the registered workload.",
                      "answer_hash": digest("Boundary conditions: use the registered workload.")}
        path = tmp_path / "submission.json"
        path.write_text(json.dumps(submission), encoding="utf-8")
        row = {"slot_id": "saved-submission", "task_id": task_id,
               "source": "researchrubrics", "target": "researchrubrics", "run_id": 0,
               "result_hash": "registered-result",
               "result": {"submission_path": str(path), "submission_hash": digest(submission),
                          "budget": {"model_calls": 2, "tokens": 100, "tool_calls": 0,
                                     "reserved_tokens": 0}}}
        instance = runner(tmp_path)
        instance._ready_state = lambda actual: store.snapshot()
        instance._pipeline = lambda actual, snapshot: (pipeline, None, tmp_path / "runs")

        result = instance._score_one(row)

        assert result["complete"] is True
        assert result["official_score"] == 1.0
        assert result["answer_hash"] == submission["answer_hash"]
        assert result["submission_sha256"] == row["result"]["submission_hash"]
        assert result["generation_budget"] == row["result"]["budget"]
        assert result["evaluation_budget"]["model_calls"] == 1
        assert all(call["role"] == "judge" for call in pipeline.models.calls)
        assert len(pipeline.models.calls) == 1
        assert instance._score_one(row) == result
        assert len(pipeline.models.calls) == 1
    finally:
        store.close()
