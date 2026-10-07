"""Freeze the metadata-only 40-question independent batch amendment."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from jit_mas.experiment_splits import ORDER_SEEDS, rank
from jit_mas.joint_protocol import BENCHMARKS, SOURCES, compact_rows
from jit_mas.joint_subset import ALL_PARTITIONS, select_members, validate_subset
from jit_mas.schemas import digest


VERSION = "jit-compose-independent-batch-subset-v6"
SEED = "jit-compose-independent-batch-subset-v6"
EVOLUTION_COUNT = 40
BATCH_SIZE = 5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_batch_subset(parent, previous):
    validate_subset(previous, parent)
    memberships = copy.deepcopy(previous["memberships"])
    additions = {}
    for source in SOURCES:
        public = [row for row in previous["task_index"] if row["benchmark"] == source]
        existing = set(memberships[source]["evolution"])
        eligible = set(parent["memberships"][source]["evolution"]) - existing
        if source == "researchrubrics":
            eligible |= (set(parent["memberships"][source]["validation"])
                         - set(memberships[source]["validation"]))
        additions[source] = select_members(
            public, eligible, EVOLUTION_COUNT - len(existing),
            f"{SEED}:{source}:evolution-additions")
        memberships[source]["evolution"].extend(additions[source])
        chosen = set(memberships[source]["evolution"])
        memberships[source]["unused"] = [
            task_id for task_id in memberships[source]["unused"] if task_id not in chosen]
    rows = copy.deepcopy(previous["task_index"])
    assignments = {task_id: part for partitions in memberships.values()
                   for part, tasks in partitions.items() for task_id in tasks}
    groups = defaultdict(set)
    for row in rows:
        row["previous_partition"] = row["partition"]
        row["partition"] = assignments[row["task_id"]]
        groups[(row["benchmark"], row["group_id"])].add(row["partition"])
    if any(len(parts) > 1 for parts in groups.values()):
        raise ValueError("A duplicate-question group crosses amended partitions")
    schedules = []
    for run_id, order_seed in enumerate(ORDER_SEEDS):
        ordered = {
            source: sorted(memberships[source]["evolution"],
                           key=lambda task_id: rank(f"{SEED}:order:{order_seed}:{source}", task_id))
            for source in SOURCES}
        schedules.append({
            "run_id": run_id,
            "order_seed": order_seed,
            "task_ids": [ordered[source][position]
                         for position in range(EVOLUTION_COUNT) for source in SOURCES],
        })
    document = {
        "version": VERSION,
        "status": "BATCH_SUBSET_FROZEN_NOT_RUN",
        "membership_seed": SEED,
        "parent_manifest": "joint_task_splits_v4.json",
        "parent_manifest_sha256": parent["manifest_sha256"],
        "previous_manifest": "joint_task_splits_v5.json",
        "previous_manifest_sha256": previous["manifest_sha256"],
        "benchmarks": copy.deepcopy(previous["benchmarks"]),
        "counts": {name: {part: len(memberships[name][part]) for part in ALL_PARTITIONS}
                   for name in BENCHMARKS},
        "totals": {part: sum(len(memberships[name][part]) for name in BENCHMARKS)
                   for part in ALL_PARTITIONS},
        "memberships": memberships,
        "task_index": rows,
        "drbii_public_strata": copy.deepcopy(previous["drbii_public_strata"]),
        "drbii_public_strata_sha256": previous["drbii_public_strata_sha256"],
        "joint_evolution_schedule": schedules,
        "batch_positions": list(range(BATCH_SIZE, EVOLUTION_COUNT + 1, BATCH_SIZE)),
        "batch_size": BATCH_SIZE,
        "candidates_per_batch": 3,
        "allocation": "Retain every v5 EVO/VAL/TEST member; add 20 EVO tasks per source using public strata and deterministic SHA256 ranks",
        "membership_amendment": {
            "researchrubrics": "User-authorized: all 30 parent EVO tasks plus the 10 parent VAL tasks not retained in current VAL; current VAL and clean TEST unchanged",
            "other_sources": "Add 20 tasks from each original parent EVO partition; no VAL or TEST transfer",
            "added_evolution_task_ids": additions,
        },
        "order_policy": "Three fixed SHA256 source orders; filter each schedule by source for nine independent trajectories, each starting from zero",
        "batch_policy": "All five tasks read one frozen state; generate three candidate states after the batch and select one using fixed VAL before starting the next batch",
        "unused_policy": previous["unused_policy"],
        "exposure_policy": "All 18 former RR exposed/reserved TEST rows remain unused; recorded WritingBench exposure writingbench:1 stays in its original parent EVO partition and is added to v6 EVO; no known exposure enters VAL or TEST",
        "row_alias_warning": previous["row_alias_warning"],
        "overlap_scope": previous["overlap_scope"],
        "results": [],
    }
    document["manifest_sha256"] = digest(document)
    return document


def validate_batch_subset(document, parent, previous):
    content = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("version") != VERSION or digest(content) != document.get("manifest_sha256"):
        raise ValueError("Batch subset manifest integrity check failed")
    if document != build_batch_subset(parent, previous):
        raise ValueError("Batch subset differs from the fixed membership amendment or order")
    return document


def render_assignments(document):
    lines = ["# Independent v6 Exact Batch Task Assignments", "",
             "Numbers are 1-based data-record positions in the pinned source file; CSV header excluded.",
             "Do not renumber after sorting or filtering. JSON maps each row to source_id and stable task_id.",
             "All v5 VAL/TEST memberships and original 20 EVO tasks per source are retained.",
             "ResearchRubrics additionally uses the 10 parent EVO and 10 unused parent VAL tasks, as authorized.",
             "DeepSearchQA and WritingBench each add 20 tasks only from their original parent EVO partitions.",
             "Known exposed WritingBench row 1 is included in EVO only, matching its original parent allocation; no exposure enters VAL/TEST.",
             "Unused tasks are excluded, not a development or replacement pool.",
             f"Manifest SHA256: `{document['manifest_sha256']}`", ""]
    index = {row["task_id"]: row for row in document["task_index"]}
    for name in BENCHMARKS:
        source = document["benchmarks"][name]
        rows = [row for row in document["task_index"] if row["benchmark"] == name]
        lines += [f"## {name}", "", f"Revision: `{source['dataset_revision']}`",
                  f"Data SHA256: `{source['dataset_sha256']}`", f"Source: {source['data_url']}", ""]
        for part in ALL_PARTITIONS:
            numbers = [row["source_row"] for row in rows if row["partition"] == part]
            lines += [f"### {part.upper()} ({len(numbers)})", "", compact_rows(numbers), ""]
        if name in SOURCES:
            added = document["membership_amendment"]["added_evolution_task_ids"][name]
            lines += ["### ADDED EVO (20)", "",
                      compact_rows([index[task_id]["source_row"] for task_id in added]), ""]
        if name == "deepresearch_bench_ii":
            strata = {row["task_id"]: row for row in document["drbii_public_strata"]}
            for language in ("en", "zh"):
                numbers = [row["source_row"] for row in rows if row["partition"] == "test"
                           and strata[row["task_id"]]["language"] == language]
                lines += [f"### TEST Language {language} ({len(numbers)})", "",
                          compact_rows(numbers), ""]
    lines += ["## Independent Evolution Batches", "",
              "Each source/run starts from zero. Each batch uses one unchanged state for five questions.",
              "Generate three candidates after the batch, evaluate on the same 10 VAL tasks and select one.",
              "The next batch starts from that winner. The schedule combines IDs only for compatibility; no cross-benchmark training.", ""]
    for run in document["joint_evolution_schedule"]:
        lines += [f"### Run {run['run_id']} (seed {run['order_seed']})", ""]
        for source in SOURCES:
            tasks = [task_id for task_id in run["task_ids"]
                     if index[task_id]["benchmark"] == source]
            lines += [f"#### {source}", ""]
            for batch_index, start in enumerate(range(0, len(tasks), BATCH_SIZE), 1):
                numbers = [str(index[task_id]["source_row"])
                           for task_id in tasks[start:start + BATCH_SIZE]]
                lines += [f"Batch {batch_index}: {', '.join(numbers)}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    parent = read_json(args.directory / "joint_task_splits_v4.json")
    previous = read_json(args.directory / "joint_task_splits_v5.json")
    output = args.output_dir or args.directory
    document = build_batch_subset(parent, previous)
    validate_batch_subset(document, parent, previous)
    files = {
        "joint_task_splits_v6.json": json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        "task_assignments_v6.md": render_assignments(document),
    }
    for name, content in files.items():
        path = output / name
        if args.check or path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Frozen batch artifact differs from reconstruction: {name}")
    if not args.check:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = output / name
            if not path.exists():
                path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"valid": True, "manifest_sha256": document["manifest_sha256"],
                      "totals": document["totals"], "model_calls": 0}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
