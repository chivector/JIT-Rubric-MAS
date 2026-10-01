import pytest

from jit_mas.experiment_analysis import aggregate_slots, contrast, holm, paired_statistics


def rows():
    slots, results = [], {}
    for task in ("a", "b", "c"):
        for method, runs in (("ours_selected", range(3)), ("ours_initial", [None])):
            for run in runs:
                for repeat in range(3):
                    slot = {"slot_id": f"{task}:{method}:{run}:{repeat}", "benchmark": "fixture",
                            "task_id": task, "method": method, "run_id": run, "repeat": repeat}
                    slots.append(slot)
                    results[slot["slot_id"]] = {"slot": slot, "complete": True,
                        "official_score": 0.8 if method == "ours_selected" else 0.3}
    return slots, results, {"fixture": {key: -0.25 for key in ("a", "b", "c")}}


def test_task_unit_and_static_control_not_duplicated_into_nine_repeats():
    slots, results, bounds = rows()
    summary, trajectories = aggregate_slots(slots, results, bounds)
    assert summary[("fixture", "ours_initial", "a")]["slots"] == 3
    assert summary[("fixture", "ours_selected", "a")]["slots"] == 9
    effect = contrast(summary, "fixture", "ours_selected", "ours_initial", bootstrap=100, permutations=200)
    assert effect["task_count"] == 3
    assert effect["mean_difference"] == pytest.approx(0.5)
    assert len(trajectories) == 4


def test_missing_score_retains_denominator_and_suppresses_significance():
    slots, results, bounds = rows()
    results.pop(slots[0]["slot_id"])
    summary, _ = aggregate_slots(slots, results, bounds)
    value = summary[("fixture", "ours_selected", "a")]
    assert value["official_mean"] is None and value["complete_slots"] == 8
    assert value["mean_bounds"] == pytest.approx([(8 * 0.8 - 0.25) / 9, (8 * 0.8 + 1) / 9])
    effect = contrast(summary, "fixture", "ours_selected", "ours_initial")
    assert effect["complete"] is False and effect["p_value"] is None


def test_holm_preserves_unavailable_registered_comparisons():
    assert holm({"a": 0.01, "b": 0.03, "missing": None}) == {"a": 0.03, "b": 0.06, "missing": None}
    with pytest.raises(ValueError):
        holm({"bad": float("nan")})


def test_statistics_are_seeded_and_reject_incomplete_or_unpaired_input():
    assert paired_statistics([0, 0], bootstrap=10, permutations=20)["p_value"] == 1
    assert paired_statistics([1, -1, 0.2], bootstrap=50, permutations=100) == paired_statistics([1, -1, 0.2], bootstrap=50, permutations=100)
    with pytest.raises(ValueError):
        paired_statistics([float("nan")])
    slots, results, bounds = rows()
    summary, _ = aggregate_slots(slots, results, bounds)
    del summary[("fixture", "ours_initial", "a")]
    with pytest.raises(ValueError, match="identical"):
        contrast(summary, "fixture", "ours_selected", "ours_initial")


def test_zero_denominator_missing_task_keeps_zero_feasible_interval():
    slot = {"slot_id": "zero", "benchmark": "researchrubrics", "method": "ours_initial", "task_id": "zero"}
    summary, _ = aggregate_slots([slot], {}, {"researchrubrics": {"zero": 0}}, {"researchrubrics": {"zero": 0}})
    value = summary[("researchrubrics", "ours_initial", "zero")]
    assert value["official_mean"] is None and value["mean_bounds"] == [0, 0]
