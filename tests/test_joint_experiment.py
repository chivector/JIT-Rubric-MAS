"""Offline protocol-controller tests for the formal joint v5 runner."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.run_joint_experiment import (
    EXPECTED,
    JointCampaign,
    build_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/experiments/joint_protocol_v5.json"
MANIFEST = ROOT / "paper/experiments/joint_task_splits_v5.json"


def _campaign(tmp_path: Path) -> JointCampaign:
    campaign = JointCampaign(tmp_path / "joint", PROTOCOL, MANIFEST)
    campaign.register()
    return campaign


def test_frozen_membership_builds_exact_formal_workload_without_api():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    slots = build_inventory(manifest)
    assert len(slots) == EXPECTED["total"] == 2541
    assert sum(row["kind"] == "evolution" for row in slots) == 180
    assert sum(row["kind"] == "validation" for row in slots) == 450
    assert sum(row["kind"] == "test" for row in slots) == 1911
    assert len({row["slot_id"] for row in slots}) == 2541
    assert sum(row.get("method") == "ours_selected" for row in slots) == 819
    assert sum(row.get("method") == "direct" for row in slots) == 273


def test_register_and_status_are_idempotent_and_preserve_identity(tmp_path):
    campaign = _campaign(tmp_path)
    before = campaign.status()
    again = campaign.register()
    assert before == again
    assert before["slot_count"] == 2541
    assert before["counts"]["test:pending"] == 1911
    registration = json.loads(campaign.registration_path.read_text(encoding="utf-8"))
    assert registration["test_resampling"] is False
    assert registration["selection_uses_test"] is False
    assert registration["test_feedback_updates_experience"] is False


def test_started_slot_is_not_resampled_and_failure_is_terminal(tmp_path):
    campaign = _campaign(tmp_path)
    slot_id = next(row["slot_id"] for row in campaign.slots if row["kind"] == "evolution")
    claim = campaign.claim(slot_id, "offline-test")
    with pytest.raises(ValueError, match="resampled"):
        campaign.claim(slot_id, "retry")
    campaign.finish(claim, {"slot_id": slot_id, "error_type": "RuntimeError", "after_state_sha256": "f" * 64}, status="failed")
    with pytest.raises(ValueError, match="resampled"):
        campaign.claim(slot_id, "retry-again")
    assert campaign.status()["counts"]["evolution:failed"] == 1


def test_test_seal_waits_for_all_slots_and_never_releases_feedback(tmp_path):
    campaign = _campaign(tmp_path)
    with pytest.raises(ValueError, match="1,911"):
        campaign.seal_test()
    test_id = next(row["slot_id"] for row in campaign.slots if row["kind"] == "test")
    claim = campaign.claim(test_id)
    campaign.finish(claim, {"slot_id": test_id, "submission_hash": "a" * 64, "score": None}, status="submitted")
    assert not (campaign.output / "test_seal.json").exists()
    assert campaign.status()["feedback_released"] is False


def test_registered_manifest_change_fails_closed(tmp_path):
    protocol = tmp_path / "joint_protocol_v5.json"
    manifest = tmp_path / "joint_task_splits_v5.json"
    shutil.copy2(PROTOCOL, protocol)
    shutil.copy2(MANIFEST, manifest)
    campaign = JointCampaign(tmp_path / "joint", protocol, manifest)
    campaign.register()
    original = campaign.manifest_path.read_text(encoding="utf-8")
    campaign.manifest_path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        campaign.status()


def test_selection_requires_nine_complete_per_source_and_does_not_use_test(tmp_path):
    campaign = _campaign(tmp_path)
    state = campaign._state()
    # Fill only C0 VAL: eight complete RR slots makes the candidate ineligible.
    val = [row for row in campaign.slots if row["kind"] == "validation" and row["run_id"] == 0 and row["checkpoint"] == 0]
    for row in val:
        state[row["slot_id"]] = {"status": "failed", "attempts": 1, "result": {"slot_id": row["slot_id"], "complete": False}}
    rr = [row for row in val if row["benchmark"] == "researchrubrics"][:8]
    for row in rr:
        state[row["slot_id"]] = {"status": "complete", "attempts": 1, "result": {"slot_id": row["slot_id"], "complete": True, "score": 0.5}}
    campaign.state_path.write_text(json.dumps(state), encoding="utf-8")
    bounds = {row["task_id"]: [0.0, 1.0] for row in val}
    selected = campaign.validation_selection(0, bounds)
    assert selected["selected"] is None
    assert selected["selection_uses_test"] is False
