"""Summarize a sealed campaign without generating answers or changing checkpoints."""

import argparse
import json
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.experiment_analysis import aggregate_slots, contrast, holm
from jit_mas.pipeline import write_json
from jit_mas.test_release import TestRelease
from scripts.run_benchmark_test import status


def summarize(campaign):
    campaign = Path(campaign)
    release = TestRelease(campaign)
    if not (campaign / "seal.json").is_file():
        raise ValueError("Cannot analyze an unsealed test campaign")
    release.seal()
    registration = json.loads((campaign / "conditions.json").read_text(encoding="utf-8"))
    slots = list(release.slots.values())
    results = {}
    for slot in slots:
        path = release._path(slot["slot_id"], "evaluations")
        if path.is_file():
            results[slot["slot_id"]] = json.loads(path.read_text(encoding="utf-8"))
    bounds, upper_bounds = {}, {}
    if registration["software_test_only"]:
        for slot in slots:
            bounds.setdefault(slot["benchmark"], {})[slot["task_id"]] = 0.0
            upper_bounds.setdefault(slot["benchmark"], {})[slot["task_id"]] = 1.0
    else:
        for condition in registration["conditions"].values():
            descriptor = condition["descriptor"]
            if descriptor["benchmark"] not in bounds:
                dataset = load_benchmark(descriptor["benchmark"], descriptor["data"])
                if dataset.dataset_sha256 != condition["identity"]["data"]["dataset_sha256"]:
                    raise ValueError("Analysis dataset differs from registered release")
                bounds[dataset.name] = dataset.lower_bounds
                upper_bounds[dataset.name] = dataset.upper_bounds
    summaries, trajectories = aggregate_slots(slots, results, bounds, upper_bounds)
    comparisons, families = {}, {"primary": {}, "secondary": {}, "mechanism": {}}

    def add(benchmark, treatment, control, family):
        available = {key[1] for key in summaries if key[0] == benchmark}
        name = f"{benchmark}:{treatment}-minus-{control}"
        if treatment in available and control in available:
            comparisons[name] = contrast(summaries, benchmark, treatment, control)
            families[family][name] = comparisons[name]["p_value"]
        else:
            families[family][name] = None

    for benchmark in ("researchrubrics", "deepsearchqa", "deepresearch_bench_ii"):
        add(benchmark, "ours_selected", "ours_initial", "primary")
        for baseline in ("jit_matched", "rubric_fixed"):
            add(benchmark, "ours_selected", baseline, "secondary")
    for baseline in ("no_explicit_rubrics", "global_only_planning", "global_only_attribution"):
        add("researchrubrics", "ours_selected", baseline, "mechanism")
    add("researchrubrics", "GO", "G", "mechanism")
    for family, p_values in families.items():
        for name, adjusted in holm(p_values).items():
            if name in comparisons:
                row = comparisons[name]
                row["holm_family"] = family
                row["holm_p_value"] = adjusted
                row["superiority_claim_allowed"] = (row["complete"] and adjusted is not None and adjusted < 0.05
                                                     and row["confidence_interval_95"][0] > 0)
    return {"version": "jit-compose-analysis-v3", "software_test_only": registration["software_test_only"],
        "registered_task_slots": len(slots), "evaluation_records": len(results),
        "task_summaries": [{"benchmark": key[0], "method": key[1], "task_id": key[2], **value}
                           for key, value in sorted(summaries.items())],
        "trajectories": trajectories, "comparisons": comparisons, "accounting": status(campaign),
        "interpretation": "Separate benchmark metrics; whole-task paired inference conditional on evolved states. Missing evaluations retain feasible intervals and do not support superiority. Software fixtures are not empirical results."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise ValueError("Do not overwrite an existing statistical report")
    report = summarize(args.campaign)
    write_json(args.output, report)
    print(json.dumps({"registered_task_slots": report["registered_task_slots"],
                      "evaluation_records": report["evaluation_records"],
                      "software_test_only": report["software_test_only"], "model_calls": 0}))


if __name__ == "__main__":
    main()
