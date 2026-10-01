import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.experiment_splits import build_split, cross_benchmark_overlaps, validate_split
from jit_mas.schemas import PublicTask, digest


def dataset(n=40):
    tasks = {f"task-{i}": PublicTask(task_id=f"task-{i}", question=f"Public query {i}") for i in range(n)}
    metadata = {key: {"group_id": digest(task.question), "question_sha256": digest(task.question),
                      "problem_category": "rare" if i == 0 else f"field-{i % 3}"}
                for i, (key, task) in enumerate(tasks.items())}
    return SimpleNamespace(name="synthetic", tasks=tasks, public_metadata=metadata, dataset_sha256="synthetic-data")


def spec(n=40):
    return {"counts": {"development": 4, "evolution": 10, "validation": 6, "test": n - 20},
            "expected_tasks": n, "split_seed": "fixed", "batch_size": 5,
            "official_split_policy": "synthetic custom partition"}


def test_partition_is_exact_deterministic_stratified_and_private_free():
    data = dataset()
    a = build_split(data, spec())
    data.tasks = dict(reversed(list(data.tasks.items())))
    b = build_split(data, spec())
    assert a == b
    manifest = validate_split(a, data, spec())
    assert (len(a["development"]), len(manifest.evolution), len(manifest.validation), len(manifest.test)) == (4, 10, 6, 20)
    assert a["checkpoint_positions"] == [0, 5, 10]
    assert len(a["evolution_schedule"]) == 3
    assert all(set(row["task_ids"]) == set(manifest.evolution) for row in a["evolution_schedule"])
    assert len({tuple(row["task_ids"]) for row in a["evolution_schedule"]}) == 3
    assert "Public query" not in json.dumps(a)


def test_exposure_quarantines_entire_duplicate_group():
    data = dataset()
    data.public_metadata["task-1"]["group_id"] = data.public_metadata["task-0"]["group_id"]
    doc = build_split(data, spec(), exposed_ids=["task-0"])
    assert {"task-0", "task-1"} <= set(doc["development"])
    assert doc["exposed_task_ids"] == ["task-0"]


def test_duplicate_group_never_crosses_partitions():
    data = dataset()
    for i in range(8):
        data.public_metadata[f"task-{i}"]["group_id"] = f"duplicate-{i // 2}"
    doc = build_split(data, spec())
    members = {row["task_id"]: row["partition"] for row in doc["public_index"]}
    for i in range(0, 8, 2):
        assert members[f"task-{i}"] == members[f"task-{i+1}"]


def test_infeasible_group_quota_and_tampered_manifest_fail_closed():
    data = dataset()
    for meta in data.public_metadata.values():
        meta["group_id"] = "same-family"
    with pytest.raises(ValueError, match="Exact quotas"):
        build_split(data, spec())
    data = dataset()
    doc = build_split(data, spec())
    doc["runtime_split_manifest"]["test"][0] = "changed"
    with pytest.raises(ValueError, match="integrity"):
        validate_split(doc)


def test_target_only_transfer_and_cross_benchmark_duplicate_audit():
    data = dataset()
    target = {**spec(), "counts": {"development": 0, "evolution": 0, "validation": 0, "test": 40}, "batch_size": 0}
    doc = build_split(data, target)
    assert doc["evolution_schedule"] == doc["checkpoint_positions"] == []
    assert len(doc["runtime_split_manifest"]["test"]) == 40
    other = copy.deepcopy(doc)
    other["benchmark"] = "other"
    assert len(cross_benchmark_overlaps([doc, other])) == 40


def test_rehashed_malformed_schedule_or_private_index_is_rejected():
    original = build_split(dataset(), spec())
    for change in ("order", "private", "group"):
        doc = copy.deepcopy(original)
        if change == "order":
            doc["evolution_schedule"][0]["task_ids"] = list(reversed(doc["evolution_schedule"][0]["task_ids"]))
        elif change == "private":
            doc["public_index"][0]["answer"] = "PRIVATE_CANARY"
        else:
            for row in doc["public_index"]:
                row["group_id"] = "all-the-same"
        doc["manifest_sha256"] = digest({key: value for key, value in doc.items() if key != "manifest_sha256"})
        with pytest.raises(ValueError):
            validate_split(doc)


def test_suite_workload_and_fixed_validation_sizes():
    root = Path(__file__).resolve().parents[2]
    suite = json.loads((root / "paper/experiments/benchmark_suite_v3.json").read_text())
    rr, ds, target = (suite["benchmarks"][name] for name in ("researchrubrics", "deepsearchqa", "deepresearch_bench_ii"))
    assert rr["counts"] == dict(development=18, evolution=30, validation=20, test=33)
    assert ds["counts"] == dict(development=50, evolution=150, validation=100, test=600)
    assert target["counts"] == dict(development=0, evolution=0, validation=0, test=132)
    cost = suite["cost"]
    assert cost["deepsearchqa_evolution"] == 3 * 150
    assert cost["deepsearchqa_validation"] == 3 * 7 * 100 * 2
    assert cost["deepsearchqa_core_test"] == (3 + 3 + 1 + 1 + 1 + 1) * 600 * 3
    assert cost["deepsearchqa_total"] == sum(cost[key] for key in ("deepsearchqa_evolution", "deepsearchqa_validation", "deepsearchqa_core_test"))
    assert cost["deepresearch_bench_ii_total"] == (3 + 1 + 1 + 1 + 1) * 132 * 3
    assert cost["mandatory_total"] == cost["researchrubrics_all"] + cost["deepsearchqa_total"] + cost["deepresearch_bench_ii_total"]
    assert suite["results"] == []
