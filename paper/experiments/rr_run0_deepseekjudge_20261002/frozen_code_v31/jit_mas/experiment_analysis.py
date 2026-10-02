"""Paired whole-task statistics, preserving static-control reuse and missingness."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def holm(p_values):
    ordered = sorted((float(value), key) for key, value in p_values.items() if value is not None)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value, _ in ordered):
        raise ValueError("Invalid p-value")
    # Unavailable registered comparisons still count in the family, never shrink it.
    count, maximum = len(p_values), 0.0
    result = {key: None for key in p_values}
    for index, (value, key) in enumerate(ordered):
        maximum = max(maximum, min(1.0, (count - index) * value))
        result[key] = maximum
    return result


def paired_statistics(differences, *, bootstrap=10000, permutations=100000, seed=20261001):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Require finite, nonempty whole-task paired differences")
    if type(bootstrap) is not int or bootstrap < 1 or type(permutations) is not int or permutations < 1:
        raise ValueError("Positive resampling counts required")
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, bootstrap, 1000):
        indices = rng.integers(0, len(values), size=(min(1000, bootstrap - start), len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    low, high = np.quantile(draws, [0.025, 0.975])
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, 1000):
        signs = rng.integers(0, 2, size=(min(1000, permutations - start), len(values))) * 2 - 1
        statistics = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-15))
    return {"task_count": len(values), "mean_difference": observed,
            "confidence_interval_95": [float(low), float(high)],
            "p_value": (1 + extreme) / (permutations + 1),
            "bootstrap_replicates": bootstrap, "sign_flip_draws": permutations, "seed": seed,
            "uncertainty_scope": "Paired task sampling, conditional on registered evolved states"}


def aggregate_slots(slots, results, lower_bounds, upper_bounds=None):
    """Each slot has one result or a missing value; never average successful rows only."""
    groups = defaultdict(list)
    runs = defaultdict(lambda: defaultdict(list))
    if len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("Duplicate registered slots")
    for slot in slots:
        key = (slot["benchmark"], slot["method"], slot["task_id"])
        lower = float(lower_bounds[slot["benchmark"]][slot["task_id"]])
        upper = float(upper_bounds[slot["benchmark"]][slot["task_id"]]) if upper_bounds else 1.0
        result = results.get(slot["slot_id"], {})
        if result.get("slot") is not None and result["slot"] != slot:
            raise ValueError("Evaluation does not match its registered slot")
        value = result.get("official_score")
        complete = result.get("complete") is True and isinstance(value, (int, float)) and not isinstance(value, bool)
        if complete and (not math.isfinite(value) or value < lower - 1e-12 or value > upper + 1e-12):
            raise ValueError("Observed score lies outside benchmark feasible bounds")
        row = {"complete": complete, "lower": float(value) if complete else lower,
               "upper": float(value) if complete else upper,
               "score": float(value) if complete else None}
        groups[key].append(row)
        runs[(slot["benchmark"], slot["method"], slot.get("run_id"))][slot["task_id"]].append(row)

    def summarize(rows):
        completed = sum(row["complete"] for row in rows)
        return {"slots": len(rows), "complete_slots": completed, "complete": completed == len(rows),
                "official_mean": sum(row["score"] for row in rows) / len(rows) if completed == len(rows) else None,
                "mean_bounds": [sum(row["lower"] for row in rows) / len(rows),
                                sum(row["upper"] for row in rows) / len(rows)]}

    summaries = {key: summarize(rows) for key, rows in groups.items()}
    trajectories = []
    for (benchmark, method, run_id), tasks in sorted(runs.items(), key=lambda item: str(item[0])):
        values = [summarize(rows) for rows in tasks.values()]
        trajectories.append({"benchmark": benchmark, "method": method, "run_id": run_id,
            "task_count": len(values), "complete": all(value["complete"] for value in values),
            "official_mean": sum(value["official_mean"] for value in values) / len(values)
                             if all(value["complete"] for value in values) else None,
            "mean_bounds": [sum(value["mean_bounds"][index] for value in values) / len(values) for index in (0, 1)]})
    return summaries, trajectories


def contrast(summaries, benchmark, treatment, control, **resampling):
    left = {key[2]: value for key, value in summaries.items() if key[:2] == (benchmark, treatment)}
    right = {key[2]: value for key, value in summaries.items() if key[:2] == (benchmark, control)}
    if not left or left.keys() != right.keys():
        raise ValueError("Paired methods require the identical complete registered task inventory")
    task_ids = sorted(left)
    complete = all(left[key]["complete"] and right[key]["complete"] for key in task_ids)
    result = {"benchmark": benchmark, "treatment": treatment, "control": control,
              "task_count": len(task_ids), "complete": complete,
              "mean_difference_bounds": [sum(left[key]["mean_bounds"][0] - right[key]["mean_bounds"][1] for key in task_ids) / len(task_ids),
                                         sum(left[key]["mean_bounds"][1] - right[key]["mean_bounds"][0] for key in task_ids) / len(task_ids)],
              "p_value": None, "superiority_claim_allowed": False}
    if complete:
        result.update(paired_statistics([left[key]["official_mean"] - right[key]["official_mean"] for key in task_ids], **resampling))
    return result
