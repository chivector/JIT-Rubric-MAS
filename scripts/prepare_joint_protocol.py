"""Freeze or independently rebuild the six-benchmark joint task assignment.

Only public metadata is written. This command makes no model calls and does
not start evolution, validation, test submission, or benchmark judging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.joint_protocol import (BENCHMARKS, COUNTS, PARTITIONS, ROW_FIELDS,
                                    SOURCES, build_joint_split, render_assignments,
                                    validate_joint_split)
from jit_mas.schemas import digest


DEFAULT_DATA = {
    "researchrubrics": "outputs/researchrubrics_live_20260930/processed_data.jsonl",
    "deepsearchqa": "dataset/deepsearchqa/DSQA-pinned-v3.csv",
    "deepresearch_bench_ii": "dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inherited_splits(directory):
    return {name: read_json(directory / f"{name}_splits_v3.json") for name in DEFAULT_DATA}


def existing_inventories(data_paths, directory):
    suite = read_json(directory / "benchmark_suite_v3.json")
    output = {}
    for name, path in data_paths.items():
        data = load_benchmark(name, path)
        if data.dataset_sha256 != suite["release_hashes"][name]:
            raise ValueError("Existing dataset hash differs from pinned release")
        rows = []
        for position, task_id in enumerate(data.tasks, 1):
            meta = data.public_metadata[task_id]
            rows.append({"task_id": task_id, "source_id": meta["source_id"], "source_row": position,
                         "question_sha256": meta["question_sha256"], "group_id": meta["group_id"],
                         "stratum": meta.get("problem_category", "all")})
        spec = suite["benchmarks"][name]
        output[name] = {"benchmark": name, "dataset_revision": spec["dataset_revision"],
                        "dataset_sha256": data.dataset_sha256, "data_url": spec["data_url"], "rows": rows}
    return output


def inventories_from_assignment(document):
    return {name: {**document["benchmarks"][name], "rows": [
                {field: row[field] for field in ROW_FIELDS}
                for row in document["task_index"] if row["benchmark"] == name]}
            for name in BENCHMARKS}


def protocol_for(document):
    counts = {name: dict(zip(PARTITIONS, COUNTS[name])) for name in BENCHMARKS}
    protocol = {
        "version": "jit-compose-joint-protocol-v4", "status": "PROTOCOL_FROZEN_NOT_RUN",
        "supersedes": "benchmark_suite_v3.json (retained as immutable historical registration)",
        "scope": "One task-conditioned MAS generator with shared cross-benchmark experience, not a separate selected generator per benchmark",
        "split_manifest": "joint_task_splits_v4.json", "split_manifest_sha256": document["manifest_sha256"],
        "readable_task_numbers": "task_assignments_v4.md", "counts": counts,
        "totals": {"evolution": 330, "validation": 220, "test_inventory": 2424,
                   "primary_unexposed_test_inventory": 2406, "supplementary_exposed_or_reserved_test": 18},
        "development_partition": "Removed; existing EVO/VAL preserved; former RR and DSQA Dev join TEST inventory",
        "exposures": {
            "researchrubrics": "All 18 formerly quarantined rows remain labelled in TEST. Only the original 33 clean rows support primary independent-test claims",
            "writingbench": "Author index 1 was partially exposed by a parser error while preparing metadata; its entire duplicate group is assigned to EVO before stratification",
            "meaning": "Clean means no recorded project development exposure, not a claim about foundation-model pretraining contamination",
        },
        "evolution": {
            "source_benchmarks": list(SOURCES), "stores": "One shared mutable store per joint replicate",
            "order_seeds": [20261001, 20261002, 20261003], "passes": 1,
            "stages": 6, "tasks_per_stage": {"researchrubrics": 5, "deepsearchqa": 25, "writingbench": 25},
            "order": "Within each stage, repeat RR, (DSQA, WritingBench) x5 five times; preserve frozen within-source run order",
            "checkpoint_positions": list(range(0, 331, 55)),
            "updates": "Direct per-source writes after existing structural/evidence checks; no accept/hold/reject quality gate",
            "failures": "Consume source positions once; no replacement or repeated source sampling",
            "continuation": "Always latest trajectory state; no rollback or early stopping based on VAL",
            "retrieval": "Benchmark-agnostic applicability retrieval; all experience source IDs must belong to global EVO; exclude global VAL and TEST",
        },
        "validation": {
            "scope": "The complete fixed 220-task union at every checkpoint, not the recent batch or an expanding window",
            "artifacts_per_task": 2, "slots_per_checkpoint": 440,
            "minimum_complete_by_benchmark": {"researchrubrics": 36, "deepsearchqa": 180, "writingbench": 180},
            "native_score_normalization": {"researchrubrics": "(score-L_t)/(U_t-L_t); native task-theoretical bounds fixed before responses; degenerate range maps to 0",
                                           "deepsearchqa": "Native F1 in [0,1]", "writingbench": "(native mean checklist score - 1)/9"},
            "aggregation": "Mean normalized task-repeat utility within benchmark, then three benchmarks equally weighted 1/3 each",
            "missing": "Observed native score stays null; normalized selection surrogate substitutes 0 for every missing slot",
            "eligibility": "All three benchmark-specific 90% thresholds must pass",
            "selection": "All seven whole-state checkpoints including C0; global max utility, tolerance 1e-12, then more complete slots, earlier source position, lexical state hash",
            "isolation": "Read-only snapshots, no attribution/write, no VAL feedback supplied to source actors or human mid-run tuning",
            "cache": "Identical state and execution identity reuse prior VAL observations; no repeated sampling for a higher score",
        },
        "universal_generator_release": {
            "default_run_id": 0, "default_order_seed": 20261001,
            "checkpoint": "The run-0 joint-VAL winner; no eligible winner means no release, not fallback to a favorable replicate",
            "replicates": "Runs 1 and 2 are robustness replicates; all three are reported, no TEST-based run choice",
            "required_components": ["generator/runtime code and prompt hashes", "configuration and retrieval/tool policy",
                                    "model and serving identity", "selected joint ExperienceSnapshot",
                                    "JIT archive and dependency identity", "dataset, split, protocol and checkpoint journal hashes"],
            "not_claimed": ["Newly trained model weights", "One fixed MAS team used for all tasks", "A deployed final version before evolution has run"],
            "test_binding": "For each run, all six benchmarks must use the identical selected state and generator/config identity; benchmark adapters differ only for task input and scoring",
        },
        "test": {
            "read_only": True, "attribution": False, "artifacts_per_task_per_state": 3,
            "static_baseline_artifacts_per_task_total": 3,
            "source_benchmark_tests": ["researchrubrics", "deepsearchqa", "writingbench"],
            "unseen_benchmark_tests": ["deepresearch_bench_ii", "ifeval", "ifbench"],
            "release_boundary": "Seal all registered six-benchmark submissions before any TEST judging/feedback release",
            "adaptation": "No target tuning, experience updates, checkpoint/source switching, or extra inter-agent discussion",
            "instruction_track": "Single-turn IFEval and IFBench with no checker-guided response repair",
        },
        "comparison": {
            "core_methods": ["ours_initial", "ours_selected", "ours_terminal", "direct", "jit_matched", "rubric_fixed"],
            "matching": "Same backbone, input evidence, and common resource ceilings per task; actual cost reported",
            "ablations": "No explicit rubrics / global-only planning / global-only attribution use the same joint source/VAL opportunities; RR clean slice is the mechanism endpoint",
            "guidance": "RR G/GO uses the same joint-selected state and precommitted quality guidance",
        },
        "evaluation": {
            "researchrubrics": "Pinned signed-weight native rubric score; primary clean 33, exposed/reserved 18 separate",
            "deepsearchqa": "Native task F1, precision/recall and fully-correct rate separate",
            "writingbench": "Mean author checklist score 1-10 per task, then task macro mean; hidden checklist stays evaluator-only",
            "deepresearch_bench_ii": "Fraction of native rubric labels equal to 1; blocked-source rate separate",
            "ifeval": "Author prompt-level strict accuracy; loose and instruction-level scores auxiliary",
            "ifbench": "Author prompt-level loose accuracy; strict and instruction-level scores auxiliary",
            "aggregate_policy": "Native TEST scores reported separately, not the normalized VAL utility as a leaderboard score",
            "families": "Four primary research/writing benchmarks; two instruction-following diagnostics reported as a separate family",
            "uncertainty": "Whole-task paired bootstrap 10000, seed20261001; sign-flip100000 and Holm within preregistered families; between-trajectory variation reported separately",
            "failure_policy": "No whole-task quality resampling; missing observed scores, completion and feasible bounds retained",
        },
        "execution_profile": "Retain configs/benchmark_suite_v3.yaml model/resource limits; benchmark-specific scorer identities must be pinned in the run bundle",
        "results": [],
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--existing-inventory", type=Path, help="Optional metadata-only inventory; otherwise project pinned local releases")
    parser.add_argument("--data", type=Path, help="JSON mapping of the three existing benchmark names to local raw files")
    parser.add_argument("--check", type=Path, help="Validate and deterministically reconstruct a frozen joint manifest")
    args = parser.parse_args(argv)
    inherited = inherited_splits(args.directory)
    if args.check:
        document = validate_joint_split(read_json(args.check))
        rebuilt = build_joint_split(inventories_from_assignment(document), inherited)
        if rebuilt != document:
            raise ValueError("Rebuilt memberships/orders differ from frozen public inventory")
        protocol = read_json(args.check.parent / "joint_protocol_v4.json")
        if protocol != protocol_for(document):
            raise ValueError("Protocol differs from registered joint rules")
        if (args.check.parent / "task_assignments_v4.md").read_text(encoding="utf-8") != render_assignments(document):
            raise ValueError("Readable task numbers differ from the JSON manifest")
        print(json.dumps({"valid": True, "tasks": len(document["task_index"]),
                          "joint_evolution_tasks": 330, "validation_tasks": 220,
                          "test_tasks": 2424, "model_calls": 0}))
        return 0
    inventories = (read_json(args.existing_inventory)["benchmarks"] if args.existing_inventory
                   else existing_inventories(read_json(args.data) if args.data else DEFAULT_DATA, args.directory))
    for name in ("writingbench", "ifeval", "ifbench"):
        inventories[name] = read_json(args.directory / f"{name}_inventory_v4.json")
    document = build_joint_split(inventories, inherited)
    output = args.output_dir or args.directory
    output.mkdir(parents=True, exist_ok=True)
    files = {"joint_task_splits_v4.json": document, "joint_protocol_v4.json": protocol_for(document)}
    for name, value in files.items():
        text = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError("Refusing to overwrite different frozen protocol artifacts; version the amendment")
        if not path.exists():
            path.write_text(text, encoding="utf-8", newline="\n")
    rendered = render_assignments(document)
    path = output / "task_assignments_v4.md"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError("Refusing to overwrite a different frozen readable task list")
    if not path.exists():
        path.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({"manifest_sha256": document["manifest_sha256"],
                      "counts": document["counts"], "model_calls": 0}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
