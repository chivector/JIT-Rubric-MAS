"""Metadata-only inventory tests.

The body files in these fixtures are deliberate canaries. A passing test must
not parse them, because submissions and evaluations are outside this reader's
metadata boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_paper_experiment_inventory import (
    _audit_release,
    build_inventory,
    canonical_digest,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _release(root: Path, *, sealed: bool = True) -> Path:
    release = root / "release"
    slots = [{"slot_id": "ours:t1", "task_id": "t1", "benchmark": "writingbench", "method": "ours", "repeat": 0}]
    _write(release / "inventory.json", {"slots": slots, "inventory_hash": canonical_digest(slots)})
    if sealed:
        _write(release / "seal.json", {"inventory_hash": canonical_digest(slots), "submission_hashes": {slots[0]["slot_id"]: "a" * 64}})
    body_name = f"{canonical_digest(slots[0]['slot_id'])}.json"
    _write(release / "submissions" / body_name, "BODY_CANARY_SUBMISSION_DO_NOT_PARSE")
    _write(release / "evaluations" / body_name, "BODY_CANARY_EVALUATION_DO_NOT_PARSE")
    return release


def _summary(iteration: str, *, sealed: bool = True) -> dict:
    return {
        "version": "fixture", "iteration": iteration, "scope": "source_evo", "partition": "source_evo",
        "sealed": sealed, "comparison_complete": False, "formal_protocol_result": False,
        "judge_policy": "fixture", "registration_hash": "b" * 64,
        "rows": [{"generation_status": "submitted", "evaluation_status": "completed", "native_score": 0.8}],
        "arms": {"ours": {"benchmarks": {"writingbench": {"slots": 1, "completed": 1, "native_mean_all_complete": 0.8, "normalized_mean_failure_zero": 0.8}}}},
        "complete_paired_task_ids": ["writingbench:t1"], "costs": {"ours": {"tokens": 12, "secret": "BODY_CANARY_COST"}},
    }


def test_release_audit_only_lists_body_filenames(tmp_path):
    release = _release(tmp_path)
    audit = _audit_release(tmp_path, release, "release")
    assert audit["sealed"] is True
    assert audit["metadata_integrity_ok"] is True
    assert audit["submission_count"] == audit["evaluation_count"] == 1
    assert audit["submission_file_set_complete"] is True
    assert audit["evaluation_file_set_complete"] is True
    assert audit["body_hashes_verified"] is False
    # Invalid body canaries prove that no submission/evaluation JSON was parsed.
    assert "BODY_CANARY" not in json.dumps(audit)


def test_unsealed_summary_does_not_project_scores(tmp_path):
    outputs = tmp_path / "outputs"
    campaign = outputs / "development_pilot_20261004_v19"
    _write(campaign / "registration.json", {"iteration": "v19", "formal_protocol_result": False, "configuration": {"seed": 1}, "code": {"runtime": "a" * 64, "runner": "b" * 64}})
    _write(campaign / "summary.json", _summary("v19", sealed=False))
    document = build_inventory(tmp_path, outputs)
    row = document["campaigns"][0]
    assert row["sealed_summary"] is False
    assert row["scores"] == {}
    assert row["budgets"]["ours"]["tokens"] == 12
    assert "BODY_CANARY_COST" not in json.dumps(document)


def test_sealed_summary_projects_only_aggregate_scores_and_explicit_recommendation(tmp_path):
    outputs = tmp_path / "outputs"
    campaign = outputs / "development_pilot_20261004_v19"
    _write(campaign / "registration.json", {"iteration": "v19", "partition": "exposed_test", "development_exposure": True, "formal_protocol_result": False, "registration_hash": "b" * 64, "configuration": {"seed": 1}, "code": {"runtime": "a" * 64, "runner": "b" * 64}})
    _write(campaign / "summary.json", _summary("v19"))
    _release(campaign)
    document = build_inventory(tmp_path, outputs, recommended_version="v19")
    row = document["campaigns"][0]
    assert row["sealed_summary"] is True
    assert row["scores"]["writingbench:ours"]["native_mean_all_complete"] == pytest.approx(0.8)
    assert row["scores"]["writingbench:ours"]["native_mean_complete"] is True
    assert document["recommendation"]["version"] == "v19"
    assert document["recommendation"]["formal_winner"] is False
    assert document["recommendation"]["automatic_score_selection"] is False


def test_no_implicit_winner_and_dynamic_v5_gap(tmp_path):
    (tmp_path / "paper/experiments").mkdir(parents=True)
    _write(tmp_path / "paper/experiments/joint_protocol_v5.json", {
        "version": "fixture-v5", "status": "SUBSET_PROTOCOL_FROZEN_NOT_RUN", "protocol_sha256": "c" * 64,
        "workload": {"evolution_slots": 1, "validation_slots": 1, "test_slots": 2, "total_task_slots": 4},
        "results": [],
    })
    document = build_inventory(tmp_path, tmp_path / "outputs")
    assert document["recommendation"] is None
    assert document["formal_v5"]["nominal_total_slots"] == 4
    assert document["formal_v5"]["results_record_count"] == 0
    assert document["formal_v5"]["missing_completed_slots"] == 4


def test_recommendation_requires_exposed_test_campaign(tmp_path):
    outputs = tmp_path / "outputs"
    campaign = outputs / "development_pilot_20261004_v23"
    _write(campaign / "registration.json", {"iteration": "v23", "partition": "source_evo", "development_exposure": True})
    _write(campaign / "summary.json", {"iteration": "v23", "sealed": True, "rows": []})
    with pytest.raises(ValueError, match="exposed TEST"):
        build_inventory(tmp_path, outputs, recommended_version="v23")


def test_reader_rejects_body_path(tmp_path):
    with pytest.raises(ValueError):
        # The function is intentionally private but protects accidental use.
        from scripts.build_paper_experiment_inventory import _read_json
        _read_json(tmp_path / "submissions" / "answer.json")
