"""Machine-readable registration for the independent-evolution v5 track."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from .schemas import digest


SOURCES = ("researchrubrics", "deepsearchqa", "writingbench")
TARGETS = ("deepresearch_bench_ii", "ifeval", "ifbench")
BENCHMARKS = SOURCES + TARGETS
RUN_IDS = (0, 1, 2)
CHECKPOINT_POSITIONS = (0, 5, 10, 15, 20)
SOURCE_COUNTS = {
    "researchrubrics": {"evolution": 20, "validation": 10, "test": 33},
    "deepsearchqa": {"evolution": 20, "validation": 10, "test": 50},
    "writingbench": {"evolution": 20, "validation": 10, "test": 50},
}
TARGET_COUNTS = {
    "deepresearch_bench_ii": {"test": 40},
    "ifeval": {"test": 50},
    "ifbench": {"test": 50},
}
TEST_METHODS = ("ours_initial", "direct", "jit_matched", "rubric_fixed")
PROTOCOL_VERSION = "jit-compose-independent-protocol-v5"
JOINT_MANIFEST_SHA256 = "e4c2c6735b3625b01335f3942bc1ac256b34ea564eb777950157a347dc4261cd"


def _manifest_without_hash(document: Mapping) -> dict:
    return {key: value for key, value in document.items() if key not in {"manifest_sha256", "protocol_sha256"}}


def _require_joint_manifest(document: Mapping) -> None:
    if document.get("version") != "jit-compose-joint-subset-v5":
        raise ValueError("Independent registration requires joint subset v5 task manifest")
    if (document.get("manifest_sha256") != JOINT_MANIFEST_SHA256
            or digest(_manifest_without_hash(document)) != JOINT_MANIFEST_SHA256):
        raise ValueError("Independent registration requires the pinned joint subset v5 manifest")
    if set(document.get("benchmarks", {})) != set(BENCHMARKS):
        raise ValueError("Joint manifest must register all six benchmarks")
    if len(document.get("joint_evolution_schedule", [])) != 3:
        raise ValueError("Joint manifest must contain all three frozen run orders")


def _memberships(document: Mapping, name: str) -> dict[str, list[str]]:
    memberships = document.get("memberships", {}).get(name)
    if not isinstance(memberships, dict):
        raise ValueError(f"Missing membership for {name}")
    return {key: list(value) for key, value in memberships.items()}


def source_order(document: Mapping, source: str, run_id: int) -> list[str]:
    """Return one source's frozen relative order for a registered run."""
    _require_joint_manifest(document)
    if source not in SOURCES or run_id not in RUN_IDS:
        raise ValueError("Independent trajectories require a registered source and run")
    memberships = _memberships(document, source)
    expected = set(memberships.get("evolution", []))
    schedule = document["joint_evolution_schedule"][run_id]
    if schedule.get("run_id") != run_id:
        raise ValueError("Run identity differs from the frozen schedule")
    ordered = [task_id for task_id in schedule["task_ids"] if task_id in expected]
    if len(ordered) != 20 or set(ordered) != expected:
        raise ValueError("Frozen source schedule does not contain exactly 20 EVO tasks")
    return ordered


def independent_split(document: Mapping, source: str) -> dict:
    """Build a per-source runtime split while preserving registered memberships."""
    _require_joint_manifest(document)
    if source not in SOURCES:
        raise ValueError("Only source benchmarks have independent EVO/VAL trajectories")
    memberships = _memberships(document, source)
    expected = SOURCE_COUNTS[source]
    for partition, count in expected.items():
        values = memberships.get(partition, [])
        if len(values) != count or len(set(values)) != count:
            raise ValueError(f"Frozen {source} {partition} membership differs from registration")
    return {
        "version": "jit-compose-independent-runtime-split-v5",
        "benchmark": source,
        "evolution": list(memberships["evolution"]),
        "validation": list(memberships["validation"]),
        "test": list(memberships["test"]),
        "stream": [],
        "source_membership_hash": digest(memberships),
    }


def build_protocol(joint_manifest: Mapping) -> dict:
    """Create the immutable independent registration from the pinned v5 manifest."""
    _require_joint_manifest(joint_manifest)
    trajectories = []
    for source in SOURCES:
        split = independent_split(joint_manifest, source)
        for run_id in RUN_IDS:
            order = source_order(joint_manifest, source, run_id)
            trajectories.append({
                "trajectory_id": f"{source}:run{run_id}",
                "source": source,
                "run_id": run_id,
                "order_seed": joint_manifest["joint_evolution_schedule"][run_id]["order_seed"],
                "evolution_task_ids": order,
                "validation_task_ids": list(split["validation"]),
                "test_task_ids": list(split["test"]),
                "checkpoint_positions": list(CHECKPOINT_POSITIONS),
            })
    counts = {name: dict(SOURCE_COUNTS[name]) for name in SOURCES}
    counts.update({name: dict(TARGET_COUNTS[name]) for name in TARGETS})
    target_test_tasks = {
        name: list(_memberships(joint_manifest, name).get("test", [])) for name in TARGETS
    }
    protocol = {
        "version": PROTOCOL_VERSION,
        "status": "PROTOCOL_FROZEN_NOT_RUN",
        "parent_manifest_sha256": joint_manifest.get("manifest_sha256", ""),
        "benchmarks": list(BENCHMARKS),
        "sources": list(SOURCES),
        "targets": list(TARGETS),
        "counts": counts,
        "target_test_tasks": target_test_tasks,
        "trajectories": trajectories,
        "evolution": {
            "trajectories": 9,
            "runs": 3,
            "tasks_per_trajectory": 20,
            "checkpoint_positions": list(CHECKPOINT_POSITIONS),
            "validation_artifacts_per_task": 1,
            "validation_repeats": 1,
            "minimum_complete_fraction": 0.9,
            "source_order": "Filter each frozen joint run order by one source; preserve relative order",
            "stores": "One writable SQLite ExperienceStore per source and run",
            "validation_isolation": "Immutable read-only snapshots; no validation feedback or writes to EVO",
            "failure_policy": "Consume each task position once; preserve null/incomplete outcomes; no resampling",
        },
        "test": {
            "selected_source_states": "Each source run's own VAL-selected state on its source TEST",
            "migration": "Every source and run is evaluated on every target benchmark",
            "static_methods": list(TEST_METHODS),
            "seal_barrier": "All 2,751 TEST artifact slots are sealed before scoring or feedback release",
            "registry_scope": {"selected_source_test": 399, "migration_test": 1260, "static_test": 1092,
                               "total": 2751},
        },
        "workload": {
            "evolution_slots": 180,
            "validation_slots": 450,
            "test_slots": 2751,
            "total_slots": 3381,
            "first_stage_slots": 75,
            "first_stage_definition": "All sources, run 0, C0/C5 validation and first five EVO tasks",
        },
        "normalization": {
            "researchrubrics": "Per-task theoretical [L_t,U_t] bounds fixed before judging",
            "deepsearchqa": "Native F1 [0,1]",
            "writingbench": "(native checklist mean - 1)/9",
            "selection": "Mean normalized utility over the source benchmark's 10 fixed VAL tasks",
            "missing": "Official score remains null; selection substitutes normalized zero",
        },
        "results": [],
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def validate_protocol(protocol: Mapping, joint_manifest: Mapping | None = None) -> dict:
    """Validate the protocol and, when supplied, reconstruct its pinned identity."""
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected independent protocol version")
    content = _manifest_without_hash(protocol)
    if digest(content) != protocol.get("protocol_sha256"):
        raise ValueError("Independent protocol hash is invalid")
    if joint_manifest is not None and protocol != build_protocol(joint_manifest):
        raise ValueError("Independent protocol differs from the frozen task manifest")
    trajectories = protocol.get("trajectories", [])
    expected = {(source, run_id) for source in SOURCES for run_id in RUN_IDS}
    actual = {(row.get("source"), row.get("run_id")) for row in trajectories}
    if actual != expected or len(trajectories) != len(expected):
        raise ValueError("Independent protocol must contain exactly nine trajectories")
    for row in trajectories:
        if len(row.get("evolution_task_ids", [])) != 20 or len(set(row["evolution_task_ids"])) != 20:
            raise ValueError("Every trajectory must contain exactly 20 unique EVO tasks")
        if len(row.get("validation_task_ids", [])) != 10 or len(set(row["validation_task_ids"])) != 10:
            raise ValueError("Every trajectory must contain exactly 10 unique VAL tasks")
        if row.get("checkpoint_positions") != list(CHECKPOINT_POSITIONS):
            raise ValueError("Independent checkpoints must be C0/C5/C10/C15/C20")
    for target in TARGETS:
        if len(protocol.get("target_test_tasks", {}).get(target, [])) != TARGET_COUNTS[target]["test"]:
            raise ValueError("Target TEST membership differs from registration")
    if protocol["workload"] != {"evolution_slots": 180, "validation_slots": 450,
                                 "test_slots": 2751, "total_slots": 3381,
                                 "first_stage_slots": 75,
                                 "first_stage_definition": "All sources, run 0, C0/C5 validation and first five EVO tasks"}:
        raise ValueError("Independent workload registry changed")
    return dict(protocol)


def normalize_score(score, bounds: Iterable[float]) -> float:
    """Normalize one finite native score using frozen theoretical task bounds."""
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise ValueError("Score bounds must be finite and ordered")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("Complete scores must be finite numbers")
    value = float(score)
    if value < lower - 1e-12 or value > upper + 1e-12:
        raise ValueError("Native score is outside its registered theoretical bounds")
    return 0.0 if upper == lower else max(0.0, min(1.0, (value - lower) / (upper - lower)))


def validation_summary(records: Iterable[Mapping], task_ids: Iterable[str], bounds: Mapping[str, Iterable[float]]) -> dict:
    """Compute one checkpoint's normalized selector utility and eligibility."""
    task_ids = list(task_ids)
    if len(task_ids) != 10 or len(set(task_ids)) != 10:
        raise ValueError("A source checkpoint requires exactly 10 fixed validation tasks")
    by_task = {}
    for row in records:
        task_id = row.get("task_id")
        if task_id not in task_ids or task_id in by_task:
            raise ValueError("Validation records must contain each task exactly once")
        by_task[task_id] = row
    normalized, complete = [], 0
    for task_id in task_ids:
        row = by_task.get(task_id, {})
        if row.get("complete") is True:
            normalized.append(normalize_score(row.get("score"), bounds[task_id]))
            complete += 1
        else:
            normalized.append(0.0)
    return {"slots": len(task_ids), "complete_evaluations": complete,
            "minimum_complete": 9, "eligible": complete >= 9,
            "selection_utility": sum(normalized) / len(normalized),
            "official_scores": {task_id: (by_task[task_id].get("score") if by_task.get(task_id, {}).get("complete") is True else None)
                                for task_id in task_ids}}


def select_checkpoint(checkpoints: Iterable[Mapping], tolerance: float = 1e-12):
    """Apply the frozen max utility, completion, early position, hash tie rule."""
    rows = [row for row in checkpoints if row.get("eligible")]
    if not rows:
        return None
    maximum = max(float(row["selection_utility"]) for row in rows)
    tied = [row for row in rows if float(row["selection_utility"]) >= maximum - tolerance]
    return min(tied, key=lambda row: (-int(row["complete_evaluations"]), int(row["position"]), row["state_hash"]))


def slot_registry(protocol: Mapping) -> dict:
    """Return the complete logical inventory without claiming artifacts exist."""
    validate_protocol(protocol)
    target_tasks = protocol["target_test_tasks"]
    source_test = sum(len(row["test_task_ids"]) for row in protocol["trajectories"])
    migration = sum(len(target_tasks[target]) for target in TARGETS) * len(protocol["trajectories"])
    static = sum(protocol["counts"][name]["test"] for name in BENCHMARKS) * 4
    entries = []
    for source in SOURCES:
        for run_id in RUN_IDS:
            row = next(item for item in protocol["trajectories"] if item["source"] == source and item["run_id"] == run_id)
            for task_id in row["evolution_task_ids"]:
                entries.append({"slot_id": f"evo:{source}:run{run_id}:{task_id}", "kind": "evolution",
                                "source": source, "run_id": run_id, "task_id": task_id, "status": "pending"})
            for position in CHECKPOINT_POSITIONS:
                for task_id in row["validation_task_ids"]:
                    entries.append({"slot_id": f"val:{source}:run{run_id}:c{position}:{task_id}", "kind": "validation",
                                    "source": source, "run_id": run_id, "position": position,
                                    "task_id": task_id, "status": "pending"})
            for task_id in row["test_task_ids"]:
                entries.append({"slot_id": f"test:{source}:run{run_id}:{source}:{task_id}", "kind": "test",
                                "method": "ours_selected", "source": source, "run_id": run_id,
                                "target": source, "task_id": task_id, "status": "pending"})
            for target in TARGETS:
                for target_task_id in target_tasks[target]:
                    entries.append({"slot_id": f"test:{source}:run{run_id}:{target}:{target_task_id}", "kind": "test",
                                    "method": "ours_selected", "source": source, "run_id": run_id,
                                    "target": target, "task_id": target_task_id, "status": "pending"})
    for target in BENCHMARKS:
        target_tasks_for_static = (target_tasks[target] if target in TARGETS
                                   else sorted({task_id for row in protocol["trajectories"]
                                                if row["source"] == target for task_id in row["test_task_ids"]}))
        for method in TEST_METHODS:
            for task_id in target_tasks_for_static:
                entries.append({"slot_id": f"test:static:{method}:{target}:{task_id}", "kind": "test",
                                "method": method, "source": "static", "run_id": None,
                                "target": target, "task_id": task_id, "status": "pending"})
    registry = {"version": "jit-compose-independent-slot-registry-v5", "protocol_sha256": protocol["protocol_sha256"],
                "counts": {"evolution": 180, "validation": 450, "test": 2751, "total": 3381},
                "test_bindings": {"selected_source_test": source_test, "migration_test": migration,
                                  "static_test": static, "total": source_test + migration + static},
                "artifacts": entries,
                "seal_barrier": {"required_test_slots": 2751, "status": "pending"}}
    registry["registry_sha256"] = digest(registry)
    return registry
