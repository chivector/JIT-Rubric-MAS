"""Verify independent registration without model calls or benchmark answers."""

from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from jit_mas.independent_protocol import (
    CHECKPOINT_POSITIONS, RUN_IDS, SOURCES, TARGETS, build_protocol,
    select_checkpoint, slot_registry, source_order, validate_protocol,
    validation_summary,
)
from jit_mas.schemas import digest
from scripts.prepare_independent_protocol import main


DIRECTORY = Path(__file__).resolve().parents[2] / "paper" / "experiments"


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DIRECTORY / "joint_task_splits_v5.json").read_text(encoding="utf-8"))


def test_independent_registration_preserves_source_memberships_and_orders(manifest):
    protocol = build_protocol(manifest)
    assert validate_protocol(protocol, manifest) == protocol
    assert protocol["status"] == "PROTOCOL_FROZEN_NOT_RUN"
    assert protocol["results"] == []
    assert len(protocol["trajectories"]) == 9
    assert "source benchmark's 10" in protocol["normalization"]["selection"]
    for trajectory in protocol["trajectories"]:
        source, run_id = trajectory["source"], trajectory["run_id"]
        expected = set(manifest["memberships"][source]["evolution"])
        assert trajectory["evolution_task_ids"] == [
            task_id for task_id in manifest["joint_evolution_schedule"][run_id]["task_ids"]
            if task_id in expected
        ]
        assert trajectory["validation_task_ids"] == manifest["memberships"][source]["validation"]
        assert trajectory["test_task_ids"] == manifest["memberships"][source]["test"]
        assert trajectory["checkpoint_positions"] == list(CHECKPOINT_POSITIONS)


def test_registry_has_unique_complete_inventory_and_all_source_target_bindings(manifest):
    registry = slot_registry(build_protocol(manifest))
    artifacts = registry["artifacts"]
    assert len(artifacts) == len({row["slot_id"] for row in artifacts}) == 3381
    assert Counter(row["kind"] for row in artifacts) == {
        "evolution": 180, "validation": 450, "test": 2751,
    }
    assert registry["test_bindings"] == {
        "selected_source_test": 399, "migration_test": 1260,
        "static_test": 1092, "total": 2751,
    }
    bindings = {(row["source"], row["run_id"], row["target"])
                for row in artifacts if row["kind"] == "test" and row["source"] != "static"}
    assert bindings == {(source, run_id, target)
                        for source in SOURCES for run_id in RUN_IDS
                        for target in (source,) + TARGETS}
    assert {row["status"] for row in artifacts} == {"pending"}
    assert registry["seal_barrier"] == {"required_test_slots": 2751, "status": "pending"}


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_parent_membership_is_rejected_even_if_rehashed(manifest, rehash):
    changed = copy.deepcopy(manifest)
    changed["memberships"]["researchrubrics"]["validation"][0] = "forged_task"
    if rehash:
        changed.pop("manifest_sha256")
        changed["manifest_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="pinned"):
        build_protocol(changed)


def test_rehashed_independent_protocol_must_still_match_registered_parent(manifest):
    protocol = build_protocol(manifest)
    protocol["trajectories"][0]["validation_task_ids"][0] = "forged_task"
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = digest(protocol)
    with pytest.raises(ValueError, match="differs from the frozen"):
        validate_protocol(protocol, manifest)


def test_source_selector_missing_scores_remain_null_and_do_not_borrow_other_sources():
    task_ids = [f"researchrubrics:val{index}" for index in range(10)]
    bounds = dict.fromkeys(task_ids, (-1, 1))
    records = [{"task_id": task_id, "complete": True, "score": 0} for task_id in task_ids[:9]]
    summary = validation_summary(records, task_ids, bounds)
    assert summary["eligible"]
    assert summary["selection_utility"] == pytest.approx(0.45)
    assert summary["official_scores"][task_ids[-1]] is None
    assert not validation_summary(records[:8], task_ids, bounds)["eligible"]
    with pytest.raises(ValueError, match="exactly once"):
        validation_summary(records + [{"task_id": "deepsearchqa:val0"}], task_ids, bounds)


def test_checkpoint_selection_uses_completion_then_position_then_hash():
    checkpoints = [
        {"eligible": True, "selection_utility": 0.5, "complete_evaluations": 9,
         "position": 0, "state_hash": "a"},
        {"eligible": True, "selection_utility": 0.5, "complete_evaluations": 10,
         "position": 5, "state_hash": "z"},
        {"eligible": True, "selection_utility": 0.5, "complete_evaluations": 10,
         "position": 10, "state_hash": "a"},
    ]
    assert select_checkpoint(checkpoints) == checkpoints[1]
    assert select_checkpoint([]) is None


def test_cli_generates_and_checks_frozen_registration(tmp_path, capsys):
    args = ["--joint-manifest", str(DIRECTORY / "joint_task_splits_v5.json"),
            "--output-dir", str(tmp_path)]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_calls"] == 0
    assert output["slots"] == {"evolution": 180, "validation": 450, "test": 2751, "total": 3381}
    assert main(args + ["--check"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    path = tmp_path / "independent_slot_registry_v5.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the frozen"):
        main(args + ["--check"])
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        main(args)
