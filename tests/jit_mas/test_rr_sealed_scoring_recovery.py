"""Offline integrity checks for the RR sealed scoring recovery command."""

import json
import hashlib
from pathlib import Path

import pytest

from jit_mas.schemas import digest
from scripts.recover_rr_sealed_scoring import _copy_release, _prior_budget, audit_source


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    source = tmp_path / "source"
    release = source / "test_release"
    slots, hashes = [], {}
    for index in range(33):
        slot_id = f"ours:task-{index}"
        task_id = f"task-{index}"
        slots.append({"slot_id": slot_id, "task_id": task_id, "method": "ours", "repeat": 0})
        record = {"slot": slots[-1], "status": "failed", "error_type": "generation"}
        if index < 27:
            answer = {"answer": f"answer-{index}", "answer_hash": digest(f"answer-{index}")}
            outcome = {"task_id": task_id, "answer_hash": answer["answer_hash"],
                       "status": "submitted_unscored", "run_dir": str(source / "run"),
                       "budget": {"model_calls": 1, "tokens": 10, "tool_calls": 0,
                                  "reserved_tokens": 0}}
            record = {"slot": slots[-1], "status": "submitted", "submission": answer,
                      "outcome": outcome}
        submission_path = release / "submissions" / f"{digest(slot_id)}.json"
        _write(submission_path, record)
        hashes[slot_id] = digest(record)
        if index < 27:
            score = {"slot": slots[-1], "official_score": 0.5, "complete": index < 22}
            if index >= 22:
                score.update(official_score=None, complete=False, evaluation_budget={
                    "model_calls": 2, "tokens": 20, "tool_calls": 0, "reserved_tokens": 0})
            _write(release / "evaluations" / f"{digest(slot_id)}.json", score)
        else:
            _write(release / "evaluations" / f"{digest(slot_id)}.json",
                   {"slot": slots[-1], "official_score": None, "complete": False,
                    "status": "submission_failed"})
    inventory = {"slots": slots, "inventory_hash": digest(slots)}
    _write(release / "inventory.json", inventory)
    _write(release / "seal.json", {"inventory_hash": inventory["inventory_hash"],
                                     "submission_hashes": hashes})
    _write(source / "pilot_metadata.json", {})
    runtime = tmp_path / "runtime"
    runtime_file = runtime / "jit_mas" / "module.py"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("value = 1\n", encoding="utf-8")
    _write(runtime / "source_manifest.json", {"jit_mas/module.py": hashlib.sha256(
        runtime_file.read_bytes()).hexdigest()})
    return source, runtime


def test_audit_and_copy_preserve_22_5_6_inventory(tmp_path):
    source, runtime = _fixture(tmp_path)
    audit = audit_source(source, runtime)
    assert (len(audit["complete"]), len(audit["incomplete"]), len(audit["generation_failed"])) == (22, 5, 6)
    output = tmp_path / "recovery"
    _copy_release(audit, output)
    assert len(list((output / "test_release" / "evaluations").glob("*.json"))) == 28
    assert len(list((output / "archive" / "v35" / "evaluations").glob("*.json"))) == 33
    assert (output / "test_release" / "seal.json").read_text() == (source / "test_release" / "seal.json").read_text()


def test_budget_recovery_rejects_unsettled_history():
    with pytest.raises(ValueError, match="unsettled or unknown"):
        _prior_budget({"evaluation_budget": {"model_calls": 1, "tokens": 2,
                                              "tool_calls": 0, "reserved_tokens": 1}})
    with pytest.raises(ValueError, match="unsettled or unknown"):
        _prior_budget({"evaluation_budget": {"usage_unknown": True, "model_calls": 1,
                                              "tokens": 2, "tool_calls": 0}})
