import copy
import json
from pathlib import Path

import pytest

from jit_mas.joint_protocol import (BENCHMARKS, COUNTS, SOURCES, build_joint_split,
                                    compact_rows, validate_joint_split, validation_summary)
from jit_mas.schemas import digest
from scripts.prepare_joint_protocol import (inherited_splits, inventories_from_assignment,
                                             main, protocol_for)


DIRECTORY = Path(__file__).resolve().parents[2] / "paper/experiments"


def frozen():
    return json.loads((DIRECTORY / "joint_task_splits_v4.json").read_text(encoding="utf-8"))


def rehash(document):
    document["manifest_sha256"] = digest({key: value for key, value in document.items()
                                          if key != "manifest_sha256"})


def test_real_membership_is_complete_public_and_preserves_old_source_val():
    doc = validate_joint_split(frozen())
    assert len(doc["task_index"]) == 2974
    assert sum(count["test"] for count in doc["counts"].values()) == 2424
    assert len({row["task_id"] for row in doc["task_index"]}) == 2974
    for name, old in inherited_splits(DIRECTORY).items():
        for part in ("evolution", "validation"):
            assert doc["memberships"][name][part] == old["runtime_split_manifest"][part]
        assert set(doc["memberships"][name]["test"]) == set(old["runtime_split_manifest"]["test"] + old["development"])
    assert all("development" not in members for members in doc["memberships"].values())
    assert sum(row["test_slice"] == "historically_exposed_or_reserved" for row in doc["task_index"]) == 18
    assert sum(row["test_slice"] == "clean_unseen" for row in doc["task_index"]) == 2406
    assert "writingbench:1" in doc["memberships"]["writingbench"]["evolution"]
    assert all(not {"question", "answer", "rubrics", "checklist", "kwargs"}.intersection(row)
               for row in doc["task_index"])


def test_exact_ids_rebuild_without_private_data_or_network(capsys):
    doc = frozen()
    rebuilt = build_joint_split(inventories_from_assignment(doc), inherited_splits(DIRECTORY))
    assert rebuilt == doc
    assert main(["--directory", str(DIRECTORY), "--check", str(DIRECTORY / "joint_task_splits_v4.json")]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_three_mixed_orders_are_full_permutations_and_source_order_is_preserved():
    doc, inherited = frozen(), inherited_splits(DIRECTORY)
    lookup = {row["task_id"]: row["benchmark"] for row in doc["task_index"]}
    orders = doc["joint_evolution_schedule"]
    assert len({tuple(run["task_ids"]) for run in orders}) == 3
    for run in orders:
        assert len(run["task_ids"]) == len(set(run["task_ids"])) == 330
        for name in ("researchrubrics", "deepsearchqa"):
            assert [key for key in run["task_ids"] if lookup[key] == name] == inherited[name]["evolution_schedule"][run["run_id"]]["task_ids"]
    assert doc["checkpoint_positions"] == [0, 55, 110, 165, 220, 275, 330]


@pytest.mark.parametrize("mutation", ["overlap", "private", "group", "cross_benchmark", "exposure", "schedule", "row_alias", "source_id", "swapped_rows"])
def test_semantically_invalid_rehashed_manifest_fails_closed(mutation):
    doc = frozen()
    if mutation == "overlap":
        members = doc["memberships"]["researchrubrics"]
        members["test"][0] = members["evolution"][0]
    elif mutation == "private":
        doc["task_index"][0]["answer"] = "PRIVATE_CANARY"
    elif mutation == "group":
        rr = [row for row in doc["task_index"] if row["benchmark"] == "researchrubrics"]
        source = next(row for row in rr if row["partition"] == "evolution")
        target = next(row for row in rr if row["partition"] == "test")
        source["group_id"] = target["group_id"]
    elif mutation == "cross_benchmark":
        source = doc["task_index"][0]
        target = next(row for row in doc["task_index"] if row["benchmark"] == "ifeval")
        source["question_sha256"] = target["question_sha256"]
    elif mutation == "exposure":
        row = next(row for row in doc["task_index"] if row["test_slice"] == "historically_exposed_or_reserved")
        row["test_slice"] = "clean_unseen"
    elif mutation == "schedule":
        doc["joint_evolution_schedule"][0]["task_ids"].reverse()
    elif mutation == "row_alias":
        doc["task_index"][0]["source_row"] = 999999
    elif mutation == "source_id":
        row = next(row for row in doc["task_index"] if row["task_id"] == "writingbench:1")
        row["source_id"] = 999999
    else:
        doc["task_index"][0]["source_row"], doc["task_index"][1]["source_row"] = (
            doc["task_index"][1]["source_row"], doc["task_index"][0]["source_row"])
    rehash(doc)
    with pytest.raises(ValueError):
        validate_joint_split(doc)


def validation_fixture():
    ids = {name: [f"{name}:v{i}" for i in range(COUNTS[name][1])] for name in SOURCES}
    bounds = {key: {"researchrubrics": (-1, 1), "deepsearchqa": (0, 1), "writingbench": (1, 10)}[name]
              for name in SOURCES for key in ids[name]}
    records = [{"benchmark": name, "task_id": key, "repeat": repeat, "complete": True,
                "score": {"researchrubrics": 1, "deepsearchqa": 0, "writingbench": 1}[name]}
               for name in SOURCES for key in ids[name] for repeat in range(2)]
    return ids, bounds, records


def test_selector_is_benchmark_equal_weight_not_20_100_100_task_weighted():
    ids, bounds, records = validation_fixture()
    result = validation_summary(records, ids, bounds)
    assert result["eligible"]
    assert result["selection_utility"] == pytest.approx(1 / 3)
    assert result["complete_evaluations"] == 440
    assert {name: row["minimum_complete"] for name, row in result["per_benchmark"].items()} == {
        "researchrubrics": 36, "deepsearchqa": 180, "writingbench": 180}


def test_per_benchmark_completion_not_masked_by_large_benchmark_and_missing_scores_remain_null():
    ids, bounds, records = validation_fixture()
    for row in records[:5]:
        row.update(complete=False, score=None)
    before = copy.deepcopy(records)
    result = validation_summary(records, ids, bounds)
    assert result["complete_evaluations"] == 435
    assert not result["eligible"]
    assert not result["per_benchmark"]["researchrubrics"]["eligible"]
    assert records == before
    assert validation_summary(records[5:], ids, bounds) == result


def test_rr_penalties_and_writing_scale_are_normalized_from_fixed_bounds():
    ids, bounds, records = validation_fixture()
    for row in records:
        row["score"] = {"researchrubrics": 0, "deepsearchqa": 0.5, "writingbench": 5.5}[row["benchmark"]]
    assert validation_summary(records, ids, bounds)["selection_utility"] == pytest.approx(0.5)
    key = ids["researchrubrics"][0]
    bounds[key] = (0, 0)
    assert validation_summary(records, ids, bounds)["per_benchmark"]["researchrubrics"]["selection_utility"] == pytest.approx(0.475)


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "outside", "reversed_bounds"])
def test_selector_rejects_invalid_observations(mutation):
    ids, bounds, records = validation_fixture()
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "nan":
        records[0]["score"] = float("nan")
    elif mutation == "outside":
        records[0]["score"] = 100
    else:
        bounds[records[0]["task_id"]] = (1, -1)
    with pytest.raises(ValueError):
        validation_summary(records, ids, bounds)


def test_protocol_has_one_shared_deployment_version_and_no_empirical_results():
    protocol = protocol_for(frozen())
    assert protocol["universal_generator_release"]["default_run_id"] == 0
    assert protocol["validation"]["slots_per_checkpoint"] == 440
    assert protocol["results"] == []
    assert "per benchmark" in protocol["scope"]
    assert set(protocol["counts"]) == set(BENCHMARKS)
    assert compact_rows([4, 2, 1, 2, 8]) == "1-2, 4, 8"


@pytest.mark.parametrize("benchmark,bad_bounds", [("writingbench", (0, 10)), ("deepsearchqa", (0, 100))])
def test_native_validation_scale_cannot_be_overridden(benchmark, bad_bounds):
    ids, bounds, records = validation_fixture()
    bounds[ids[benchmark][0]] = bad_bounds
    with pytest.raises(ValueError, match="native scale"):
        validation_summary(records, ids, bounds)


def test_selector_accepts_frozen_small_counts_and_single_artifact():
    ids, bounds, records = validation_fixture()
    ids = {name: keys[:10] for name, keys in ids.items()}
    records = [row for row in records if row["repeat"] == 0 and row["task_id"] in ids[row["benchmark"]]]
    counts = dict.fromkeys(SOURCES, 10)
    result = validation_summary(records, ids, bounds, counts=counts, repeats=1)
    assert result["complete_evaluations"] == 30
    assert result["selection_utility"] == pytest.approx(1 / 3)
    assert all(row["minimum_complete"] == 9 for row in result["per_benchmark"].values())
    assert not validation_summary(records[2:], ids, bounds, counts=counts, repeats=1)["eligible"]
    with pytest.raises(ValueError, match="count"):
        validation_summary(records, ids, bounds)


@pytest.mark.parametrize("counts,repeats", [(None, 0), (None, True), (None, 1.0),
                                          ({}, 1), (dict.fromkeys(SOURCES, 0), 1),
                                          (dict.fromkeys(SOURCES, True), 1)])
def test_selector_rejects_invalid_registered_layout(counts, repeats):
    ids, bounds, records = validation_fixture()
    with pytest.raises(ValueError, match="positive integers"):
        validation_summary(records, ids, bounds, counts=counts, repeats=repeats)
