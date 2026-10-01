"""No network, model calls or private benchmark answers needed for v5 checks."""

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import pytest

from jit_mas.joint_protocol import BENCHMARKS, SOURCES
from jit_mas.joint_subset import (
    ALL_PARTITIONS, CHECKPOINT_POSITIONS, COUNTS, DRBII_STRATA_SHA256,
    PARENT_SHA256, build_subset, protocol_for, render_assignments, select_members,
    subset_validation_summary, validate_subset, workload,
)
from jit_mas.schemas import digest
from scripts.prepare_joint_subset import main


DIRECTORY = Path(__file__).resolve().parents[2] / "paper" / "experiments"


@pytest.fixture(scope="module")
def parent():
    return json.loads((DIRECTORY / "joint_task_splits_v4.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen():
    return json.loads((DIRECTORY / "joint_task_splits_v5.json").read_text(encoding="utf-8"))


def rehash(document):
    document.pop("manifest_sha256", None)
    document["manifest_sha256"] = digest(document)


def test_frozen_parent_anchors_and_deterministic_reconstruction(parent, frozen):
    assert parent["manifest_sha256"] == PARENT_SHA256
    assert frozen == build_subset(parent, list(reversed(frozen["drbii_public_strata"])))
    assert validate_subset(frozen, parent) == frozen
    assert digest(frozen["drbii_public_strata"]) == DRBII_STRATA_SHA256


def test_counts_complete_partition_membership_and_inheritance(parent, frozen):
    assert frozen["totals"] == {"evolution": 60, "validation": 30, "test": 273, "unused": 2611}
    parent_index = {row["task_id"]: row for row in parent["task_index"]}
    seen = set()
    grouped = defaultdict(set)
    for name in BENCHMARKS:
        allocation = frozen["memberships"][name]
        assert set(allocation) == set(ALL_PARTITIONS)
        for part, count in zip(ALL_PARTITIONS, (*COUNTS[name], frozen["counts"][name]["unused"])):
            ids = allocation[part]
            assert len(ids) == count
            assert not seen.intersection(ids)
            seen.update(ids)
            if part != "unused":
                assert set(ids) <= set(parent["memberships"][name][part])
    assert seen == set(parent_index)
    for row in frozen["task_index"]:
        original = parent_index[row["task_id"]]
        assert row["parent_partition"] == original["partition"]
        for field in ("source_id", "source_row", "group_id", "question_sha256", "former_development", "test_slice"):
            assert row[field] == original[field]
        grouped[(row["benchmark"], row["group_id"])].add(row["partition"])
    assert all(len(parts) == 1 for parts in grouped.values())


def test_exposures_never_enter_selected_test(frozen):
    rows = frozen["task_index"]
    exposed_rr = [row for row in rows if row["benchmark"] == "researchrubrics" and row["former_development"]]
    assert len(exposed_rr) == 18
    assert {row["partition"] for row in exposed_rr} == {"unused"}
    for row in rows:
        if row["benchmark"] == "writingbench" and row["source_row"] == 1:
            assert row["partition"] in ("evolution", "unused")
    rr_test = [row for row in rows if row["benchmark"] == "researchrubrics" and row["partition"] == "test"]
    assert len(rr_test) == 33
    assert all(row["test_slice"] == "clean_unseen" for row in rr_test)


def test_drbii_balanced_languages_and_themes(frozen):
    metadata = {row["task_id"]: row for row in frozen["drbii_public_strata"]}
    selected = frozen["memberships"]["deepresearch_bench_ii"]["test"]
    assert Counter(metadata[key]["language"] for key in selected) == {"en": 20, "zh": 20}
    for row in frozen["task_index"]:
        if row["benchmark"] == "deepresearch_bench_ii":
            assert row["selection_stratum"] == metadata[row["task_id"]]["theme"]


def test_schedules_preserve_parent_relative_order_and_balance(parent, frozen):
    index = {row["task_id"]: row for row in frozen["task_index"]}
    assert frozen["checkpoint_positions"] == list(CHECKPOINT_POSITIONS)
    for run, old in zip(frozen["joint_evolution_schedule"], parent["joint_evolution_schedule"]):
        assert run["order_seed"] == old["order_seed"]
        assert len(run["task_ids"]) == len(set(run["task_ids"])) == 60
        assert [key for stage in run["stages"] for key in stage] == run["task_ids"]
        for stage in run["stages"]:
            assert [index[key]["benchmark"] for key in stage] == list(SOURCES) * 5
        for name in SOURCES:
            members = set(frozen["memberships"][name]["evolution"])
            assert [key for key in run["task_ids"] if key in members] == [key for key in old["task_ids"] if key in members]


@pytest.mark.parametrize("field,value", [("source_id", "forged"), ("source_row", 9999),
                                          ("question_sha256", "0" * 64), ("selection_stratum", "forged")])
def test_self_rehashed_row_tampering_is_rejected(parent, frozen, field, value):
    changed = copy.deepcopy(frozen)
    changed["task_index"][0][field] = value
    rehash(changed)
    with pytest.raises(ValueError, match="deterministic pinned"):
        validate_subset(changed, parent)


def test_self_rehashed_public_source_labels_are_rejected(parent, frozen):
    changed = copy.deepcopy(frozen)
    changed["drbii_public_strata"][0]["theme"] = "forged"
    changed["drbii_public_strata_sha256"] = digest(changed["drbii_public_strata"])
    rehash(changed)
    with pytest.raises(ValueError, match="independently pinned release"):
        validate_subset(changed, parent)


def test_parent_cannot_be_swapped_even_with_recomputed_hash(parent, frozen):
    changed = copy.deepcopy(parent)
    changed["membership_seed"] = "another-valid-looking-seed"
    rehash(changed)
    with pytest.raises(ValueError, match="independently pinned v4"):
        validate_subset(frozen, changed)


def test_cross_partition_task_swap_is_rejected(parent, frozen):
    changed = copy.deepcopy(frozen)
    source = changed["memberships"]["deepsearchqa"]
    a, b = source["evolution"][0], source["test"][0]
    source["evolution"][0], source["test"][0] = b, a
    for row in changed["task_index"]:
        if row["task_id"] in (a, b):
            row["partition"] = "test" if row["task_id"] == a else "evolution"
    rehash(changed)
    with pytest.raises(ValueError, match="deterministic pinned"):
        validate_subset(changed, parent)


def test_private_fields_and_schedule_tampering_are_rejected(parent, frozen):
    for location in ("task_index", "joint_evolution_schedule"):
        changed = copy.deepcopy(frozen)
        changed[location][0]["unexpected_private_payload"] = "not permitted"
        rehash(changed)
        with pytest.raises(ValueError):
            validate_subset(changed, parent)


def test_duplicate_groups_are_indivisible_and_impossible_quota_fails():
    rows = [{"task_id": key, "group_id": "same", "selection_stratum": "all"} for key in ("a", "b")]
    assert set(select_members(rows, {"a", "b"}, 2, "seed")) == {"a", "b"}
    with pytest.raises(ValueError, match="Exact quotas"):
        select_members(rows, {"a", "b"}, 1, "seed")
    with pytest.raises(ValueError, match="split a duplicate group"):
        select_members(rows, {"a"}, 1, "seed")


def test_public_proportional_strata_and_input_order_invariance():
    rows = [{"task_id": str(i), "group_id": str(i), "selection_stratum": "major" if i < 8 else "minor"}
            for i in range(10)]
    selection = select_members(rows, {row["task_id"] for row in rows}, 5, "seed")
    assert len(set(selection) & {"8", "9"}) == 1
    assert selection == select_members(list(reversed(rows)), {row["task_id"] for row in rows}, 5, "seed")


def test_workload_static_reuse_and_frozen_protocol(frozen):
    value = workload()
    assert value["evolution_slots"] == 180
    assert value["validation_slots"] == 450
    assert value["selected_test_slots"] == 819
    assert value["static_test_slots"] == 1092
    assert value["test_slots"] == 1911
    assert value["total_task_slots"] == 2541
    assert value["v4_same_five_methods_task_slots"] == 61134
    assert value["task_slot_reduction_percent"] == pytest.approx(95.8435567769)
    protocol = json.loads((DIRECTORY / "joint_protocol_v5.json").read_text(encoding="utf-8"))
    assert protocol == protocol_for(frozen)
    assert protocol["status"] == "SUBSET_PROTOCOL_FROZEN_NOT_RUN"
    assert protocol["test"]["static_baseline_artifacts_per_task_total"] == 1
    assert protocol["comparison"]["core_methods"] == ["ours_initial", "ours_selected", "direct", "jit_matched", "rubric_fixed"]
    assert protocol["universal_generator_release"]["default_run_id"] == 0


def test_subset_validation_single_repeat_thresholds_and_native_scales(frozen):
    ids = {name: frozen["memberships"][name]["validation"] for name in SOURCES}
    bounds = {key: (1, 10) if name == "writingbench" else (0, 1)
              for name in SOURCES for key in ids[name]}
    records = [{"benchmark": name, "task_id": key, "repeat": 0, "complete": True,
                "score": bounds[key][1]} for name in SOURCES for key in ids[name]]
    summary = subset_validation_summary(records, ids, bounds)
    assert summary["eligible"]
    assert summary["complete_evaluations"] == 30
    assert summary["selection_utility"] == 1
    assert all(row["minimum_complete"] == 9 for row in summary["per_benchmark"].values())
    assert subset_validation_summary(records[1:], ids, bounds)["eligible"]
    assert not subset_validation_summary(records[2:], ids, bounds)["eligible"]


def test_readable_assignments_and_metadata_only_cli(frozen, capsys):
    assert (DIRECTORY / "task_assignments_v5.md").read_text(encoding="utf-8") == render_assignments(frozen)
    assert main(["--parent", str(DIRECTORY / "joint_task_splits_v4.json"),
                 "--check", str(DIRECTORY / "joint_task_splits_v5.json"),
                 "--drbii-data", "does-not-exist.jsonl"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] and output["model_calls"] == 0


def test_cli_refuses_overwriting_different_frozen_artifacts(tmp_path, monkeypatch, frozen):
    from scripts import prepare_joint_subset
    monkeypatch.setattr(prepare_joint_subset, "public_drbii_strata", lambda *_: frozen["drbii_public_strata"])
    args = ["--parent", str(DIRECTORY / "joint_task_splits_v4.json"), "--output-dir", str(tmp_path)]
    assert main(args) == 0
    assert main(args) == 0
    path = tmp_path / "joint_protocol_v5.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        main(args)
