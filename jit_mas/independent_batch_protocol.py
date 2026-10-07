"""Pure helpers for the corrected batched independent-evolution protocol.

This module is intentionally separate from the frozen v5 runner.  It describes
the next protocol revision in which five EVO tasks share one input state, three
candidate states are produced after the batch, and VAL selects one candidate
before the next batch starts.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .schemas import digest


BATCH_SIZE = 5
BATCHES_PER_TRAJECTORY = 8
CANDIDATES_PER_BATCH = 3
VALIDATION_TASKS_PER_CANDIDATE = 10
BATCH_VALIDATION_SLOTS = CANDIDATES_PER_BATCH * VALIDATION_TASKS_PER_CANDIDATE
EVOLUTION_TASKS_PER_TRAJECTORY = BATCH_SIZE * BATCHES_PER_TRAJECTORY
CANDIDATES_PER_TRAJECTORY = CANDIDATES_PER_BATCH * BATCHES_PER_TRAJECTORY
VALIDATION_SLOTS_PER_TRAJECTORY = (
    CANDIDATES_PER_TRAJECTORY * VALIDATION_TASKS_PER_CANDIDATE
)
TRAJECTORIES = 9
SOURCES = ("researchrubrics", "deepsearchqa", "writingbench")
TARGETS = ("deepresearch_bench_ii", "ifeval", "ifbench")
RUN_IDS = (0, 1, 2)
TEST_SLOTS = 2751
PROTOCOL_VERSION = "jit-compose-independent-batch-protocol-v6"
# v7 is intentionally additive: the v6 constants above remain available for
# validating frozen campaigns, while a v7 manifest is detected by its fourth
# source and receives the one-run/Ours-only inventory below.
V7_SOURCES = ("researchrubrics", "deepsearchqa", "writingbench", "deepresearch_bench_ii")
V7_TARGETS = ("ifeval", "ifbench")
V7_RUN_IDS = (0,)
V7_TEST_SLOTS = 573
V7_PROTOCOL_VERSION = "jit-compose-independent-batch-protocol-v7"
TERMINAL_TASK_STATUSES = frozenset({"complete", "incomplete", "failed", "missing"})


def _unique_tasks(tasks: Iterable[str], expected: int, label: str) -> list[str]:
    values = list(tasks)
    if any(not isinstance(task, str) or not task.strip() for task in values):
        raise ValueError(f"{label} must contain nonempty task IDs")
    if len(values) != expected or len(set(values)) != expected:
        raise ValueError(f"{label} must contain exactly {expected} unique tasks")
    return values


def split_batches(evolution_task_ids: Iterable[str]) -> list[list[str]]:
    """Split one trajectory's 40 EVO tasks into eight fixed five-task batches."""
    tasks = _unique_tasks(evolution_task_ids, EVOLUTION_TASKS_PER_TRAJECTORY, "EVO tasks")
    return [tasks[start:start + BATCH_SIZE]
            for start in range(0, EVOLUTION_TASKS_PER_TRAJECTORY, BATCH_SIZE)]


def batch_schedule(evolution_task_ids: Iterable[str], validation_task_ids: Iterable[str]) -> dict:
    """Return a metadata-only schedule with three candidates per EVO batch.

    Candidate VAL is deliberately attached to the batch's candidate state, not
    to individual EVO tasks.  No C0 candidate is included: the first three
    candidates are produced after the first five-task batch.
    """
    batches = split_batches(evolution_task_ids)
    validation = _unique_tasks(validation_task_ids, VALIDATION_TASKS_PER_CANDIDATE,
                               "VAL tasks")
    result = []
    for batch_index, tasks in enumerate(batches):
        candidates = []
        for candidate_index in range(CANDIDATES_PER_BATCH):
            candidates.append({
                "candidate_id": f"b{batch_index}c{candidate_index}",
                "batch_index": batch_index,
                "candidate_index": candidate_index,
                "evolution_task_ids": list(tasks),
                "validation_task_ids": list(validation),
                "base_position": batch_index * BATCH_SIZE,
            })
        result.append({
            "batch_index": batch_index,
            "evolution_task_ids": list(tasks),
            "candidates": candidates,
        })
    return {
        "batches": result,
        "counts": {
            "evolution_tasks": EVOLUTION_TASKS_PER_TRAJECTORY,
            "batches": BATCHES_PER_TRAJECTORY,
            "candidates": CANDIDATES_PER_TRAJECTORY,
            "validation_slots": VALIDATION_SLOTS_PER_TRAJECTORY,
        },
        "candidate_policy": "three candidates after each five-task batch; no C0 validation",
        "intra_batch_evolution": False,
    }


def build_protocol(manifest: Mapping) -> dict:
    """Build a fresh registration from extended memberships and frozen orders.

    The manifest supplies memberships and three joint_evolution_schedule rows.
    This helper accepts the extended 40-task sets without importing the old v5
    source-order validator or modifying any existing frozen registration.
    """
    memberships = manifest.get("memberships", {})
    v7 = all(source in memberships and memberships[source].get("evolution")
             for source in V7_SOURCES)
    sources = V7_SOURCES if v7 else SOURCES
    targets = V7_TARGETS if v7 else TARGETS
    run_ids = V7_RUN_IDS if v7 else RUN_IDS
    schedules = manifest.get("joint_evolution_schedule", [])
    if len(schedules) != len(run_ids) or {item.get("run_id") for item in schedules} != set(run_ids):
        raise ValueError("The manifest requires one registered v7 run order" if v7
                         else "The manifest requires three registered run orders")
    trajectories = []
    for source in sources:
        partitions = memberships.get(source, {})
        evolution = _unique_tasks(partitions.get("evolution", []),
                                  EVOLUTION_TASKS_PER_TRAJECTORY, f"{source} EVO tasks")
        validation = _unique_tasks(partitions.get("validation", []),
                                   VALIDATION_TASKS_PER_CANDIDATE, f"{source} VAL tasks")
        test_count = (33 if source == "researchrubrics" else
                      40 if source == "deepresearch_bench_ii" else 50)
        tests = _unique_tasks(partitions.get("test", []), test_count, f"{source} TEST tasks")
        if len(set(evolution + validation + tests)) != len(evolution + validation + tests):
            raise ValueError("EVO, VAL and TEST memberships must be disjoint")
        for run_id in run_ids:
            schedule = next(item for item in schedules if item["run_id"] == run_id)
            ordered = [task_id for task_id in schedule.get("task_ids", [])
                       if task_id in set(evolution)]
            if len(ordered) != len(evolution) or set(ordered) != set(evolution):
                raise ValueError("Every run order must cover its source EVO tasks exactly once")
            trajectories.append({
                "trajectory_id": f"{source}:run{run_id}",
                "source": source,
                "run_id": run_id,
                "order_seed": schedule.get("order_seed"),
                "evolution_task_ids": ordered,
                "validation_task_ids": list(validation),
                "test_task_ids": list(tests),
                **batch_schedule(ordered, validation),
            })
    target_tasks = {
        target: _unique_tasks(memberships.get(target, {}).get("test", []),
                              40 if target == "deepresearch_bench_ii" else 50,
                              f"{target} TEST tasks")
        for target in targets
    }
    trajectory_count = len(trajectories)
    test_slots = (sum(len(row["test_task_ids"]) for row in trajectories)
                  + trajectory_count * sum(len(tasks) for tasks in target_tasks.values())) if v7 else TEST_SLOTS
    protocol = {
        "version": V7_PROTOCOL_VERSION if v7 else PROTOCOL_VERSION,
        "status": "SPECIFIED_NOT_RUN",
        "parent_manifest_sha256": digest(manifest),
        "sources": list(sources),
        "targets": list(targets),
        "run_ids": list(run_ids),
        "test_methods": ["ours_selected"] if v7 else None,
        "static_methods": [] if v7 else None,
        "trajectories": trajectories,
        "target_test_tasks": target_tasks,
        "evolution": {
            "tasks_per_trajectory": EVOLUTION_TASKS_PER_TRAJECTORY,
            "batches_per_trajectory": BATCHES_PER_TRAJECTORY,
            "batch_size": BATCH_SIZE,
            "candidates_per_batch": CANDIDATES_PER_BATCH,
            "candidate_creation": "After all five batch tasks terminate",
            "intra_batch_evolution": False,
            "continue_from": "The batch VAL winner before the next batch starts",
            "initial_state_validation": False,
        },
        "workload": protocol_workload(trajectories=trajectory_count,
                                       test_slots=test_slots),
        "results": [],
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def validate_protocol(protocol: Mapping, manifest: Mapping | None = None) -> dict:
    """Require an unchanged registration bound to the extended task manifest."""
    version = protocol.get("version")
    is_v7 = version == V7_PROTOCOL_VERSION
    if version not in {PROTOCOL_VERSION, V7_PROTOCOL_VERSION}:
        raise ValueError("Unexpected independent batch protocol version")
    content = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if digest(content) != protocol.get("protocol_sha256"):
        raise ValueError("Independent batch protocol hash is invalid")
    if manifest is not None and protocol != build_protocol(manifest):
        raise ValueError("Independent batch protocol differs from its extended task manifest")
    trajectories = protocol.get("trajectories", [])
    sources = tuple(protocol.get("sources", V7_SOURCES if is_v7 else SOURCES))
    run_ids = tuple(protocol.get("run_ids", V7_RUN_IDS if is_v7 else RUN_IDS))
    expected_pairs = {(source, run_id) for source in sources for run_id in run_ids}
    if (len(trajectories) != len(expected_pairs)
            or {(row.get("source"), row.get("run_id")) for row in trajectories} != expected_pairs):
        raise ValueError("Independent batch protocol has an invalid source trajectory inventory")
    for row in trajectories:
        expected = batch_schedule(row["evolution_task_ids"], row["validation_task_ids"])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Independent batch schedule differs from its fixed task order")
    expected_test_slots = protocol.get("workload", {}).get("test_slots", TEST_SLOTS)
    if protocol.get("workload") != protocol_workload(trajectories=len(trajectories),
                                                      test_slots=expected_test_slots):
        raise ValueError("Independent batch workload changed")
    if is_v7 and tuple(sources) != V7_SOURCES or is_v7 and tuple(run_ids) != V7_RUN_IDS:
        raise ValueError("v7 requires four sources and one run")
    if is_v7 and protocol.get("test_methods") != ["ours_selected"]:
        raise ValueError("v7 is Ours-only and cannot register static methods")
    return dict(protocol)


def protocol_workload(*, trajectories: int = TRAJECTORIES, test_slots: int = TEST_SLOTS) -> dict:
    """Return arithmetic for the corrected source trajectories."""
    if type(trajectories) is not int or trajectories < 1:
        raise ValueError("trajectories must be a positive integer")
    if type(test_slots) is not int or test_slots < 0:
        raise ValueError("test_slots must be a nonnegative integer")
    return {
        "trajectories": trajectories,
        "evolution_tasks": trajectories * EVOLUTION_TASKS_PER_TRAJECTORY,
        "candidates": trajectories * CANDIDATES_PER_TRAJECTORY,
        "validation_slots": trajectories * VALIDATION_SLOTS_PER_TRAJECTORY,
        "candidates_per_batch": CANDIDATES_PER_BATCH,
        "validation_slots_per_batch": BATCH_VALIDATION_SLOTS,
        "test_slots": test_slots,
        "total_task_slots": (trajectories * (EVOLUTION_TASKS_PER_TRAJECTORY
                                              + VALIDATION_SLOTS_PER_TRAJECTORY)
                             + test_slots),
    }


def validate_batch_isolation(batch: Mapping, *, base_state_hash: str,
                             task_runs: Sequence[Mapping]) -> None:
    """Ensure all five EVO tasks read the same state and do not mutate it.

    Candidate creation is a post-batch operation.  A task run may record a
    receipt or feedback, but its committed experience state must remain the
    batch base until the three candidate states are created.
    """
    tasks = list(batch.get("evolution_task_ids", []))
    if len(tasks) != BATCH_SIZE or len(set(tasks)) != BATCH_SIZE:
        raise ValueError("A batch must contain exactly five unique EVO tasks")
    if not isinstance(base_state_hash, str) or not base_state_hash:
        raise ValueError("A batch requires a nonempty base state hash")
    if len(task_runs) != BATCH_SIZE:
        raise ValueError("A batch must have one receipt for each EVO task")
    seen = []
    for run in task_runs:
        if run.get("task_id") not in tasks or run["task_id"] in seen:
            raise ValueError("Batch receipts must cover each EVO task exactly once")
        seen.append(run["task_id"])
        if run.get("status") not in TERMINAL_TASK_STATUSES:
            raise ValueError("Candidate generation waits for all five EVO tasks to terminate")
        if run.get("input_state_hash") != base_state_hash:
            raise ValueError("All EVO tasks in a batch must read its base state")
        if run.get("after_state_hash") != base_state_hash:
            raise ValueError("EVO tasks cannot commit updates inside a batch")


def validate_candidate_states(candidates: Sequence[Mapping], *, base_state_hash: str,
                              batch_index: int) -> None:
    """Validate the three post-batch candidates before VAL is released."""
    if len(candidates) != CANDIDATES_PER_BATCH:
        raise ValueError("Each batch must produce exactly three candidates")
    if type(batch_index) is not int or batch_index not in range(BATCHES_PER_TRAJECTORY):
        raise ValueError("Candidate states require a registered batch index")
    if any(item.get("batch_index") != batch_index for item in candidates):
        raise ValueError("Candidate states cannot mix different EVO batches")
    if {item.get("candidate_index") for item in candidates} != set(range(CANDIDATES_PER_BATCH)):
        raise ValueError("Candidate states require indices zero, one and two")
    ids = [item.get("candidate_id") for item in candidates]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidates require unique nonempty candidate IDs")
    if any(item.get("base_state_hash") != base_state_hash for item in candidates):
        raise ValueError("All candidates must derive from the same batch base state")
    if any(not isinstance(item.get("state_hash"), str) or not item["state_hash"]
           for item in candidates):
        raise ValueError("Every candidate must bind a nonempty state hash")


def _normalize(score, bounds: Iterable[float]) -> float:
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise ValueError("Score bounds must be finite and ordered")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("Complete scores must be finite numbers")
    value = float(score)
    if value < lower - 1e-12 or value > upper + 1e-12:
        raise ValueError("Native score is outside its registered theoretical bounds")
    return 0.0 if upper == lower else max(0.0, min(1.0, (value - lower) / (upper - lower)))


def validation_summary(records: Iterable[Mapping], task_ids: Iterable[str],
                      bounds: Mapping[str, Iterable[float]]) -> dict:
    """Summarize one candidate's fixed ten-task VAL without inventing scores."""
    task_ids = _unique_tasks(task_ids, VALIDATION_TASKS_PER_CANDIDATE, "VAL tasks")
    by_task = {}
    for row in records:
        task_id = row.get("task_id")
        if task_id not in task_ids or task_id in by_task:
            raise ValueError("Validation records must contain each task exactly once")
        if row.get("status") not in TERMINAL_TASK_STATUSES:
            raise ValueError("Selection waits for every candidate VAL slot to terminate")
        by_task[task_id] = row
    if set(by_task) != set(task_ids):
        raise ValueError("Selection waits for all ten candidate VAL slots to terminate")
    normalized, complete = [], 0
    official = {}
    for task_id in task_ids:
        row = by_task.get(task_id, {})
        if row.get("complete") is True:
            normalized.append(_normalize(row.get("score"), bounds[task_id]))
            complete += 1
            official[task_id] = row.get("score")
        else:
            normalized.append(0.0)
            official[task_id] = None
    return {
        "slots": len(task_ids),
        "complete_evaluations": complete,
        "minimum_complete": 9,
        "eligible": complete >= 9,
        "selection_utility": sum(normalized) / len(normalized),
        "official_scores": official,
    }


def select_batch_candidate(candidates: Iterable[Mapping], tolerance: float = 1e-12):
    """Select one eligible candidate using fixed utility and deterministic ties."""
    candidates = list(candidates)
    if (len(candidates) != CANDIDATES_PER_BATCH
            or {row.get("candidate_index") for row in candidates} != set(range(CANDIDATES_PER_BATCH))):
        raise ValueError("Selection requires exactly three registered candidates")
    batch_index = candidates[0].get("batch_index")
    if type(batch_index) is not int or batch_index not in range(BATCHES_PER_TRAJECTORY):
        raise ValueError("Selection requires a registered batch identity")
    if any(not isinstance(row.get("base_state_hash"), str) or not row["base_state_hash"]
           or not isinstance(row.get("state_hash"), str) or not row["state_hash"]
           for row in candidates):
        raise ValueError("Selection requires bound candidate and base state hashes")
    if any(row.get("batch_index") != candidates[0].get("batch_index")
           or row.get("base_state_hash") != candidates[0].get("base_state_hash")
           for row in candidates):
        raise ValueError("Selection cannot mix candidate batches or base states")
    if any(row.get("slots") != VALIDATION_TASKS_PER_CANDIDATE for row in candidates):
        raise ValueError("Selection waits for all 30 batch VAL slots to terminate")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Selection tolerance must be finite and nonnegative")
    for row in candidates:
        complete = row.get("complete_evaluations")
        utility = row.get("selection_utility")
        if (type(complete) is not int or complete not in range(VALIDATION_TASKS_PER_CANDIDATE + 1)
                or row.get("eligible") is not (complete >= 9)):
            raise ValueError("Candidate eligibility differs from its completed VAL slots")
        if (isinstance(utility, bool) or not isinstance(utility, (int, float))
                or not math.isfinite(utility) or not 0 <= utility <= 1):
            raise ValueError("Candidate utility must be a finite normalized value")
    rows = [dict(row) for row in candidates if row.get("eligible")]
    if not rows:
        return None
    maximum = max(float(row["selection_utility"]) for row in rows)
    tied = [row for row in rows
            if float(row["selection_utility"]) >= maximum - tolerance]
    return min(tied, key=lambda row: (-int(row["complete_evaluations"]),
                                      int(row["candidate_index"]), row["state_hash"]))
