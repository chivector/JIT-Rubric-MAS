"""Check the registered schedule using manifest metadata only, never task data."""

import hashlib
import json
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[2] / "paper" / "experiments"


def read_split():
    return json.loads((EXPERIMENTS / "splits_v2.json").read_text(encoding="utf-8"))


def read_protocol():
    return json.loads((EXPERIMENTS / "protocol_v2.json").read_text(encoding="utf-8"))


def rank(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_checkpoint_split_preserves_pinned_v1_membership_and_provenance():
    document = read_split()
    assert document["source_manifest"] == "splits_v1.json"
    source_bytes = (EXPERIMENTS / document["source_manifest"]).read_bytes()
    source = json.loads(source_bytes)
    assert document["source_manifest_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    for key in ("counts", "development_quarantine", "runtime_split_manifest", "provenance"):
        assert document[key] == source[key]
    assert document["version"] == "jit-compose-paper-splits-v2"
    assert document["status"] == "SPECIFIED_NOT_RUN"
    assert document["launch_allowed"] is False
    assert "not official JIT/RR" in document["adoption_status"]


def test_checkpoint_split_counts_are_whole_disjoint_tasks():
    document = read_split()
    assert document["counts"] == {
        "total": 101, "development_quarantine": 18,
        "evolution": 30, "validation": 20, "test": 33,
    }
    manifest = document["runtime_split_manifest"]
    assert not manifest["stream"]
    groups = {"development_quarantine": document["development_quarantine"],
              **{name: manifest[name] for name in ("evolution", "validation", "test")}}
    all_ids = [task_id for task_ids in groups.values() for task_id in task_ids]
    assert len(all_ids) == len(set(all_ids)) == document["counts"]["total"]
    assert all(len(task_ids) == document["counts"][name] for name, task_ids in groups.items())


def test_three_evolution_orders_and_six_batches_follow_exact_hash_rule():
    document = read_split()
    schedule = document["evolution_schedule"]
    assert schedule["order_seed_prefix"] == "jit-compose-evolution-v2\n"
    assert schedule["order_seeds"] == [20261001, 20261002, 20261003]
    assert schedule["batch_size"] == 5
    assert schedule["checkpoint_positions"] == [0, 5, 10, 15, 20, 25, 30]
    assert "not successful updates" in schedule["position_unit"]
    assert len(schedule["runs"]) == 3
    for run_id, run in enumerate(schedule["runs"]):
        assert run["run_id"] == run_id
        assert run["order_seed"] == schedule["order_seeds"][run_id]
        prefix = f'{schedule["order_seed_prefix"]}{run["order_seed"]}\n'
        expected = sorted(document["runtime_split_manifest"]["evolution"],
                          key=lambda task_id: rank(prefix + task_id))
        assert run["ordered_task_ids"] == expected
        assert run["batches"] == [expected[start:start + 5] for start in range(0, 30, 5)]
        assert len(run["batches"]) == 6
        assert all(len(batch) == 5 for batch in run["batches"])
    assert len({tuple(run["ordered_task_ids"]) for run in schedule["runs"]}) == 3


def test_validation_and_test_orders_use_the_complete_inherited_sets():
    document = read_split()
    assert document["validation_order"] == document["runtime_split_manifest"]["validation"]
    assert document["test_order"] == document["runtime_split_manifest"]["test"]
    assert len(document["validation_order"]) == 20
    assert len(document["test_order"]) == 33


def test_human_audit_uses_ten_score_independent_test_ids():
    document = read_split()
    audit = document["human_audit"]
    assert audit["rank_prefix"] == "jit-compose-human-audit-v2\n"
    assert audit["count"] == 10
    expected = sorted(document["test_order"],
                      key=lambda task_id: rank(audit["rank_prefix"] + task_id))[:10]
    assert audit["task_ids"] == expected
    assert len(set(audit["task_ids"])) == 10
    assert set(audit["task_ids"]) <= set(document["test_order"])


def test_split_is_metadata_only_and_semantic_audit_still_blocks_launch():
    document = read_split()
    forbidden = {"task_index", "public_prompt_excerpt", "prompt", "question", "rubrics",
                 "private_record", "reference_answer", "scores", "answer", "api_key"}

    def check_keys(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(document)
    audit = document["public_overlap_audit"]
    assert audit["exact_normalized_question_cross_partition"] == []
    assert audit["exact_url_cross_partition"] == []
    assert audit["topic_and_semantic_source_review"] == "REQUIRED_BEFORE_FORMAL_FREEZE"
    assert audit["topic_disjoint_claim"] is False
    blockers = " ".join(document["launch_blockers"])
    assert "before any formal API run" in blockers
    assert "versioned preregistration amendment" in blockers
    assert "never silently reshuffle" in blockers
    assert document["launch_allowed"] is False


def test_checkpoint_protocol_matches_split_and_all_three_registered_schedules():
    split, protocol = read_split(), read_protocol()
    assert protocol["schema_version"] == "jit-compose-checkpoint-protocol-v2"
    assert protocol["split_file"] == "splits_v2.json"
    assert protocol["counts"] == {key: value for key, value in split["counts"].items()
                                   if key != "total"}
    evolution = protocol["evolution"]
    schedule = split["evolution_schedule"]
    assert evolution["independent_runs"] == len(schedule["runs"]) == 3
    assert evolution["run_ids"] == [run["run_id"] for run in schedule["runs"]]
    assert evolution["order_seeds"] == schedule["order_seeds"]
    assert evolution["checkpoint_positions"] == schedule["checkpoint_positions"]
    assert evolution["batch_size"] == schedule["batch_size"] == 5
    assert evolution["batches_per_run"] == 6
    assert evolution["passes_over_evolution_set"] == 1
    assert evolution["source_task_attempts_per_run"] == protocol["counts"]["evolution"] == 30
    assert evolution["batch_size"] * evolution["batches_per_run"] == 30


def test_full_validation_schedule_and_completeness_threshold_are_specified():
    protocol = read_protocol()
    evolution, validation = protocol["evolution"], protocol["validation"]
    assert validation["task_scope"] == "The same entire fixed 20-task validation set at every checkpoint"
    assert "All seven checkpoints" in validation["candidate_scope"]
    assert "including C0" in validation["candidate_scope"]
    assert validation["iterations_per_run"] == len(evolution["checkpoint_positions"]) == 7
    assert validation["artifact_repeats_per_task"] == len(validation["repeat_ids"]) == 2
    assert validation["repeat_ids"] == [0, 1]
    slots_per_checkpoint = protocol["counts"]["validation"] * validation["artifact_repeats_per_task"]
    assert slots_per_checkpoint == validation["minimum_complete_denominator"] == 40
    assert validation["minimum_complete_evaluations"] == 36
    assert validation["minimum_complete_evaluations"] / slots_per_checkpoint == 0.9
    total = slots_per_checkpoint * validation["iterations_per_run"] * evolution["independent_runs"]
    assert total == validation["nominal_task_executions_all_runs"] == 840
    assert validation["attribution"] is False
    assert "read-only" in validation["store_access"]
    assert "snapshot version and content hash" in validation["store_access"]


def test_selection_never_restores_per_update_quality_gates_or_best_run_tuning():
    protocol = read_protocol()
    update, evolution = protocol["update_policy"], protocol["evolution"]
    assert update["rule"] == "direct_after_attribution"
    for key in ("per_update_quality_gate", "parameter_training", "evaluator_evolution",
                "val_content_to_memory", "test_content_to_memory"):
        assert update[key] is False
    assert "at most one write per source task" in update["proposal_selection"]
    assert evolution["continue_from"] == "Latest trajectory state, never the validation winner"
    assert evolution["early_stopping"] is False
    assert evolution["repeat_source_answers"] is False
    assert evolution["select_best_run"] is False
    assert "including execution failure" in evolution["position_unit"]
    assert protocol["failures"]["whole_task_retries"] == 0
    assert protocol["failures"]["low_score_retries"] == 0


def test_static_controls_are_shared_and_evolved_states_use_all_test_repeats():
    protocol = read_protocol()
    test = protocol["test"]
    assert test["artifact_repeats_per_state_per_task"] == len(test["repeat_ids"]) == 3
    assert test["repeat_ids"] == [0, 1, 2]
    assert "Each of the3 run-specific validation winners" in test["evolved_states"]
    assert "not rerun9 times" in test["static_control_reuse"]
    assert "reused_from" in test["duplicate_state_reuse"]
    assert "never best-of3" in test["test_result_policy"]
    for key in ("attribution", "writes", "adapt_after_test"):
        assert test[key] is False
    methods = {method["id"]: method for method in protocol["methods"]}
    static_slots = protocol["counts"]["test"] * test["artifact_repeats_per_state_per_task"]
    assert static_slots == 99
    for method in ("ours_initial", "direct", "jit_matched", "rubric_fixed"):
        assert methods[method]["evolves"] is False
        assert methods[method]["nominal_test_slots"] == static_slots
    evolved_slots = static_slots * protocol["evolution"]["independent_runs"]
    assert evolved_slots == 297
    for method in ("ours_selected", "ours_terminal", "no_explicit_rubrics",
                   "global_only_planning", "global_only_attribution"):
        assert methods[method]["nominal_test_slots"] == evolved_slots


def test_nominal_workload_arithmetic_and_unexecuted_status():
    protocol = read_protocol()
    workload = protocol["workload"]
    runs = protocol["evolution"]["independent_runs"]
    source = runs * protocol["evolution"]["source_task_attempts_per_run"]
    assert workload["full_evolution"] == source == 90
    validation = (runs * len(protocol["evolution"]["checkpoint_positions"])
                  * protocol["counts"]["validation"]
                  * protocol["validation"]["artifact_repeats_per_task"])
    assert workload["full_validation"] == validation == 840
    methods = {method["id"]: method for method in protocol["methods"]}
    core_ids = ("ours_initial", "ours_selected", "ours_terminal", "direct", "jit_matched", "rubric_fixed")
    core_test = sum(methods[method]["nominal_test_slots"] for method in core_ids)
    assert workload["core_test"] == core_test == 990
    assert workload["core_total"] == source + validation + core_test == 1920
    ablations = ("no_explicit_rubrics", "global_only_planning", "global_only_attribution")
    for method in ablations:
        assert methods[method]["evolves"] is True
        assert workload["per_evolving_ablation"] == source + validation + methods[method]["nominal_test_slots"] == 1227
    assert workload["three_evolving_ablations"] == len(ablations) * workload["per_evolving_ablation"] == 3681
    diagnostic = protocol["mechanism_diagnostic"]
    shared_rubrics = runs * protocol["counts"]["test"] * diagnostic["artifact_repeats"]
    assert diagnostic["mandatory"] is True
    assert shared_rubrics == diagnostic["additional_shared_rubric_generations"] == 297
    diagnostic_slots = shared_rubrics * len(diagnostic["conditions"])
    assert workload["guidance_diagnostic"] == diagnostic["nominal_task_slots"] == diagnostic_slots == 594
    assert workload["mandatory_total"] == (workload["core_total"]
        + workload["three_evolving_ablations"] + workload["guidance_diagnostic"]) == 6195
    assert protocol["results"] == []
    assert protocol["status"] == "SPECIFIED_NOT_RUN"
    assert protocol["launch_allowed"] is False
    assert "NOT API requests" in workload["unit"]
    assert "pending" in methods["ours_selected"]["implementation"]
    assert any("checkpoint scheduler" in requirement for requirement in protocol["required_before_launch"])
