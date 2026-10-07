"""Synthetic checks for the corrected batched independent protocol."""

import pytest

from jit_mas.independent_batch_protocol import (
    BATCH_SIZE,
    CANDIDATES_PER_BATCH,
    CANDIDATES_PER_TRAJECTORY,
    VALIDATION_SLOTS_PER_TRAJECTORY,
    batch_schedule,
    build_protocol,
    protocol_workload,
    select_batch_candidate,
    split_batches,
    validate_batch_isolation,
    validate_candidate_states,
    validate_protocol,
    validation_summary,
)


def _tasks(prefix, count):
    return [f"{prefix}{index}" for index in range(count)]


def test_schedule_has_eight_batches_three_candidates_and_no_c0():
    schedule = batch_schedule(_tasks("e", 40), _tasks("v", 10))
    assert len(schedule["batches"]) == 8
    assert all(len(batch["evolution_task_ids"]) == BATCH_SIZE
               for batch in schedule["batches"])
    assert all(len(batch["candidates"]) == CANDIDATES_PER_BATCH
               for batch in schedule["batches"])
    assert [item["candidate_id"] for item in schedule["batches"][0]["candidates"]] == [
        "b0c0", "b0c1", "b0c2"]
    assert schedule["counts"]["candidates"] == CANDIDATES_PER_TRAJECTORY
    assert schedule["counts"]["validation_slots"] == VALIDATION_SLOTS_PER_TRAJECTORY
    assert schedule["intra_batch_evolution"] is False


def test_workload_matches_nine_trajectory_campaign():
    assert protocol_workload() == {
        "trajectories": 9,
        "evolution_tasks": 360,
        "candidates": 216,
        "validation_slots": 2160,
        "candidates_per_batch": 3,
        "validation_slots_per_batch": 30,
        "test_slots": 2751,
        "total_task_slots": 5271,
    }


def test_batch_receipts_must_share_state_and_cannot_update_inside_batch():
    tasks = _tasks("e", 5)
    batch = {"evolution_task_ids": tasks}
    receipts = [{"task_id": task, "status": "complete", "input_state_hash": "base",
                 "after_state_hash": "base"} for task in tasks]
    validate_batch_isolation(batch, base_state_hash="base", task_runs=receipts)
    receipts[2]["input_state_hash"] = "changed"
    with pytest.raises(ValueError, match="base state"):
        validate_batch_isolation(batch, base_state_hash="base", task_runs=receipts)
    receipts[2]["input_state_hash"] = "base"
    receipts[2]["after_state_hash"] = "updated"
    with pytest.raises(ValueError, match="inside a batch"):
        validate_batch_isolation(batch, base_state_hash="base", task_runs=receipts)
    receipts[2]["after_state_hash"] = "base"
    receipts[2]["status"] = "started"
    with pytest.raises(ValueError, match="all five EVO tasks"):
        validate_batch_isolation(batch, base_state_hash="base", task_runs=receipts)


def test_candidate_states_are_three_independent_post_batch_derivations():
    candidates = [{"candidate_id": f"c{index}", "base_state_hash": "base", "batch_index": 0,
                   "candidate_index": index,
                   "state_hash": f"state{index}"} for index in range(3)]
    validate_candidate_states(candidates, base_state_hash="base", batch_index=0)
    with pytest.raises(ValueError, match="exactly three"):
        validate_candidate_states(candidates[:2], base_state_hash="base", batch_index=0)
    candidates[1]["base_state_hash"] = "other"
    with pytest.raises(ValueError, match="same batch base"):
        validate_candidate_states(candidates, base_state_hash="base", batch_index=0)
    candidates[1]["base_state_hash"] = "base"
    candidates[1]["batch_index"] = 1
    with pytest.raises(ValueError, match="different EVO batches"):
        validate_candidate_states(candidates, base_state_hash="base", batch_index=0)


def test_selection_requires_all_three_candidates_and_thirty_terminal_val_slots():
    candidates = [{"candidate_index": index, "state_hash": f"state{index}",
                   "batch_index": 0, "base_state_hash": "base",
                   "eligible": True, "complete_evaluations": 10, "slots": 10,
                   "selection_utility": index / 10} for index in range(3)]
    assert select_batch_candidate(candidates)["candidate_index"] == 2
    with pytest.raises(ValueError, match="exactly three"):
        select_batch_candidate(candidates[:2])
    candidates[1]["slots"] = 9
    with pytest.raises(ValueError, match="all 30"):
        select_batch_candidate(candidates)
    candidates[1]["slots"] = 10
    candidates[1]["batch_index"] = 1
    with pytest.raises(ValueError, match="mix candidate batches"):
        select_batch_candidate(candidates)


def test_validation_missing_scores_stay_null_and_selection_uses_candidate_index():
    task_ids = _tasks("v", 10)
    bounds = {task_id: (0, 1) for task_id in task_ids}
    records = [{"task_id": task_id, "status": "complete", "complete": True, "score": 0.5}
               for task_id in task_ids]
    records[-1] = {"task_id": task_ids[-1], "status": "failed", "complete": False, "score": None}
    summary = validation_summary(records, task_ids, bounds)
    assert summary["eligible"]
    assert summary["official_scores"][task_ids[-1]] is None
    candidates = [
        {"candidate_index": 2, "batch_index": 0, "base_state_hash": "base",
         "state_hash": "z", **summary},
        {"candidate_index": 0, "state_hash": "a", "eligible": True, "slots": 10,
         "batch_index": 0, "base_state_hash": "base",
         "complete_evaluations": 10, "selection_utility": 0.5},
        {"candidate_index": 1, "state_hash": "b", "eligible": True, "slots": 10,
         "batch_index": 0, "base_state_hash": "base",
         "complete_evaluations": 10, "selection_utility": 0.5},
    ]
    assert select_batch_candidate(candidates)["candidate_index"] == 0
    assert select_batch_candidate([{**item, "eligible": False, "complete_evaluations": 8}
                                   for item in candidates]) is None
    with pytest.raises(ValueError, match="all ten"):
        validation_summary(records[:-1], task_ids, bounds)
    records[-1]["status"] = "started"
    with pytest.raises(ValueError, match="every candidate VAL slot"):
        validation_summary(records, task_ids, bounds)


def test_split_rejects_wrong_task_count_or_duplicates():
    with pytest.raises(ValueError, match="40 unique"):
        split_batches(_tasks("e", 39))
    tasks = _tasks("e", 40)
    tasks[-1] = tasks[0]
    with pytest.raises(ValueError, match="40 unique"):
        split_batches(tasks)


def test_extended_manifest_builds_new_registration_without_old_source_order():
    sources = ("researchrubrics", "deepsearchqa", "writingbench")
    memberships = {source: {"evolution": _tasks(source + ":e", 40),
                            "validation": _tasks(source + ":v", 10),
                            "test": _tasks(source + ":t", 33 if source == "researchrubrics" else 50)}
                   for source in sources}
    memberships.update({target: {"test": _tasks(target + ":t", count)}
                        for target, count in (("deepresearch_bench_ii", 40),
                                              ("ifeval", 50), ("ifbench", 50))})
    order = [task for source in sources for task in memberships[source]["evolution"]]
    manifest = {"memberships": memberships,
                "joint_evolution_schedule": [{"run_id": run_id, "order_seed": 10 + run_id,
                                               "task_ids": order}
                                              for run_id in range(3)]}
    protocol = build_protocol(manifest)
    assert validate_protocol(protocol, manifest) == protocol
    assert len(protocol["trajectories"]) == 9
    assert protocol["evolution"]["initial_state_validation"] is False
    assert protocol["workload"]["total_task_slots"] == 5271
    assert all(len(item["batches"]) == 8 for item in protocol["trajectories"])
    assert all(len(item["evolution_task_ids"]) == 40 for item in protocol["trajectories"])
    protocol["evolution"]["intra_batch_evolution"] = True
    with pytest.raises(ValueError, match="hash is invalid"):
        validate_protocol(protocol, manifest)
    manifest["memberships"]["researchrubrics"]["validation"][0] = memberships["researchrubrics"]["evolution"][0]
    with pytest.raises(ValueError, match="disjoint"):
        build_protocol(manifest)
