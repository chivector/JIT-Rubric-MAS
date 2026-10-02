"""Offline safety checks for the isolated RR continuation helper."""

import json
from pathlib import Path

import pytest

from scripts.recover_rr_two_arm_ours import (
    _assert_processes_gone, _test_started, _verify_runtime, _recover_validation_result,
)
from jit_mas.schemas import ExperienceSnapshot, digest


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_recovery_refuses_any_durable_test_activity(tmp_path):
    ours = tmp_path / "ours"
    output = tmp_path / "output"
    dump(ours / "report.json", {"submitted_outcomes": [], "test_failures": []})
    assert not _test_started(ours, output)
    dump(ours / "test" / "slots" / "task" / "started.json", {"task_id": "task"})
    assert _test_started(ours, output)


def test_recovery_verifies_every_frozen_runtime_file(tmp_path):
    runtime = tmp_path / "runtime"
    source = runtime / "jit_mas" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(runtime / "source_manifest.json", {"jit_mas/module.py": digest})
    _verify_runtime(runtime)
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        _verify_runtime(runtime)


def test_recovery_rejects_unlisted_importable_runtime_source(tmp_path):
    runtime = tmp_path / "runtime"
    source = runtime / "jit_mas" / "module.py"
    extra = runtime / "jit_mas" / "extra.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    extra.write_text("value = 2\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(runtime / "source_manifest.json", {"jit_mas/module.py": digest})
    with pytest.raises(RuntimeError, match="Unlisted"):
        _verify_runtime(runtime)


def test_recovery_refuses_live_launcher_pid(monkeypatch):
    monkeypatch.setattr("scripts.recover_rr_two_arm_ours._process_exists", lambda pid: pid == 7)
    with pytest.raises(RuntimeError, match="child is live"):
        _assert_processes_gone({"method_pid": 7, "preflight_pid": 0})


def test_validation_recovery_binds_c5_reused_cache_to_c0_durable_result(tmp_path):
    ours = tmp_path / "ours"
    snapshot = ExperienceSnapshot()
    state_hash = digest(snapshot)
    answer = "durable answer"
    answer_hash = digest(answer)
    evaluation = {
        "task_id": "val-1", "evaluator_version": "judge-v1", "score": 0.7,
        "complete": True, "aggregation": "sum", "zero_denominator": False,
        "source": "benchmark", "raw": {"submission_answer_hash": answer_hash}, "rubrics": [],
    }
    comparison = {"task": {"task_id": "val-1"}, "repeat": 0}
    budget = {"model_calls": 1, "tokens": 2}
    run_dir = ours / "checkpoints" / "C0" / "runs" / "run-key"
    result = {
        "run_key": "run-key", "task_id": "val-1", "mode": "validate", "repeat": 0,
        "status": "submitted", "experience_version": snapshot.version,
        "experience_hash": state_hash, "comparison_fingerprint": digest(comparison),
        "answer_hash": answer_hash, "evaluation": evaluation, "budget": budget,
        "run_dir": str(run_dir), "experience_updates": [], "agent_pool_updates": [],
    }
    dump(ours / "checkpoints" / "C0" / "slots" / "val-1" / "result.json", result)
    dump(run_dir / "complete.json", result)
    dump(run_dir / "run_manifest.json", {"experience_hash": state_hash,
         "experience_version": snapshot.version, "mode": "validate", "run_key": "run-key",
         "comparison": comparison})
    dump(run_dir / "submission.json", {"answer": answer, "answer_hash": answer_hash})
    dump(run_dir / "evaluation.json", evaluation)
    dump(run_dir / "budget.json", budget)
    cache = {"first_position": 0, "state_hash": state_hash,
             "snapshot": snapshot.model_dump(mode="json")}
    # This cache represents C5 reusing C0's immutable state; no C5 result is
    # accepted by the recovery reader.
    assert _recover_validation_result(ours, cache, "val-1", snapshot, 0)["run_key"] == "run-key"
