import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_paper_splits import build_partition, public_projection, validate_partition


def fixture():
    rows = [{"sample_id": f"task-{i}", "prompt": f"Explain topic {i}",
             "rubrics": [{"criterion": "PRIVATE_CRITERION_CANARY", "weight": 987654}],
             "reference_answer": "PRIVATE_REFERENCE_CANARY"} for i in range(101)]
    exposure = {"excluded_task_ids": [f"task-{i}" for i in range(14)]}
    pilot = {"tasks": [{"public_task": {"task_id": f"task-{i}"}} for i in range(14, 18)]}
    return rows, exposure, pilot


def test_public_projection_quarantines_all_pilots_without_private_fields():
    rows, exposure, pilot = fixture()
    tasks = [public_projection(row) for row in rows]
    result = build_partition(tasks, exposure, pilot)
    assert result["counts"] == {"total": 101, "development_quarantine": 18,
                                "evolution": 30, "validation": 20, "test": 33}
    assert "PRIVATE_" not in json.dumps(result) and "987654" not in json.dumps(result)
    assert result == build_partition(list(reversed(tasks)), exposure, pilot)
    manifest = validate_partition(result)
    assert not set(result["development_quarantine"]) & set(manifest.test)
    assert not result["launch_allowed"] and result["status"] == "DRAFT_NOT_RUN"


def test_cross_partition_public_overlap_is_reported_not_hidden():
    rows, exposure, pilot = fixture()
    rows[20]["prompt"] = rows[30]["prompt"] = rows[0]["prompt"] = "A shared task https://example.org/source"
    result = build_partition([public_projection(row) for row in rows], exposure, pilot)
    audit = result["public_overlap_audit"]
    assert audit["exact_normalized_question_cross_partition"]
    assert audit["exact_url_cross_partition"]
    assert audit["topic_and_semantic_source_review"] == "REQUIRED_BEFORE_FORMAL_FREEZE"


@pytest.mark.parametrize("kind", ["overlap", "count", "index", "order"])
def test_partition_checker_rejects_mutation(kind):
    rows, exposure, pilot = fixture()
    result = copy.deepcopy(build_partition([public_projection(row) for row in rows], exposure, pilot))
    if kind == "overlap":
        result["development_quarantine"].append(result["runtime_split_manifest"]["test"][0])
    elif kind == "count":
        result["counts"]["test"] += 1
    elif kind == "index":
        result["task_index"][0]["partition"] = "test"
    else:
        result["runtime_split_manifest"]["test"].reverse()
    with pytest.raises(ValueError):
        validate_partition(result)


def test_checked_in_paper_protocol_is_obsolete_archive_and_not_filled_with_pilot_scores():
    root = Path(__file__).resolve().parents[2] / "paper"
    split = json.loads((root / "experiments/splits_v1.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "experiments/protocol_v1.json").read_text(encoding="utf-8"))
    manifest = validate_partition(split)
    assert protocol["planned_counts"] == {key: value for key, value in split["counts"].items() if key != "total"}
    assert not protocol["launch_allowed"] and protocol["status"] == "DRAFT_NOT_RUN"
    assert protocol["adoption_status"] == split["adoption_status"] == "OBSOLETE_ON_HOLD_METHOD_CHANGE"
    assert not protocol["results"] and not protocol["proposed_model_configuration"]["evaluator_updates"]
    assert len(manifest.validation) == 4 * protocol["validation_policy"]["min_tasks"]
    template = json.loads((root / "author_notes/results_template.json").read_text(encoding="utf-8"))
    assert not template["results"] and not template["formal_protocol"]["pilot_scores_allowed_in_main_table"]
    assert template["formal_protocol"]["specification"] == "experiments/joint_protocol_v5.json"
    assert template["formal_protocol"]["status"] == "SUBSET_PROTOCOL_FROZEN_NOT_RUN"
    assert "validation_policy" not in template
    assert "direct_after_attribution" in json.dumps(template["experience_update_policy"])


def test_draft_split_checker_cannot_authorize_benchmark_launch():
    rows, exposure, pilot = fixture()
    result = build_partition([public_projection(row) for row in rows], exposure, pilot)
    result["launch_allowed"] = True
    with pytest.raises(ValueError, match="draft"):
        validate_partition(result)


def test_subset_template_counts_shared_controls_once_and_keeps_results_empty():
    root = Path(__file__).resolve().parents[2] / "paper"
    template = json.loads((root / "author_notes/results_template.json").read_text(encoding="utf-8"))
    registration = template["formal_protocol"]
    counts = registration["planned_counts"]
    assert tuple(sum(row[key] for row in counts.values()) for key in ("evolution", "validation", "test")) == (60, 30, 273)
    assert registration["selected_unique_tasks"] == 363
    assert registration["selected_unique_tasks"] + registration["unused_tasks"] == 2974
    slots = registration["nominal_core_slots"]
    assert slots == {"evolution": 3 * 60, "validation": 3 * 5 * 30,
                     "selected_test": 3 * 273, "four_static_controls_test": 4 * 273}
    assert sum(slots.values()) == registration["nominal_core_task_attempts_before_state_reuse"] == 2541
    assert "ours_terminal" not in registration["core_methods"]
    assert template["checkpoint_selection_plan"]["validation_artifact_repeats"] == 1
    assert template["checkpoint_selection_plan"]["test_repeats_per_evolved_state"] == 1
    assert not template["results"] and not registration["formal_results_executed"]
