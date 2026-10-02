"""Pinned, public-only subset of the immutable six-benchmark v4 inventory."""

from __future__ import annotations

from collections import defaultdict
import copy
from types import SimpleNamespace

from .experiment_splits import ORDER_SEEDS, _groups, _select, rank
from .joint_protocol import (BENCHMARKS, PARTITIONS, SOURCES, compact_rows,
                             validate_joint_split, validation_summary)
from .schemas import digest


VERSION = "jit-compose-joint-subset-v5"
SEED = "jit-compose-joint-subset-v5"
PARENT_SHA256 = "f6a07c37a2cd5505d97461cd923c820767479a94072b4db73c046c0a5e0a0aea"
# Independent anchor obtained from public language/theme fields of the pinned
# raw release, not a self-hash supplied by a proposed subset manifest.
DRBII_STRATA_SHA256 = "775c89e74aead815689c729a389c0e2a0deb482db1cc15b1a9ea5fdf675af778"
COUNTS = {
    "researchrubrics": (20, 10, 33), "deepsearchqa": (20, 10, 50),
    "writingbench": (20, 10, 50), "deepresearch_bench_ii": (0, 0, 40),
    "ifeval": (0, 0, 50), "ifbench": (0, 0, 50),
}
ALL_PARTITIONS = PARTITIONS + ("unused",)
CHECKPOINT_POSITIONS = (0, 15, 30, 45, 60)


def _trusted_inputs(parent, drbii_strata):
    validate_joint_split(parent)
    if parent["manifest_sha256"] != PARENT_SHA256:
        raise ValueError("Subset parent is not the independently pinned v4 manifest")
    fields = {"task_id", "language", "theme"}
    if (not isinstance(drbii_strata, list)
            or any(not isinstance(row, dict) or set(row) != fields for row in drbii_strata)):
        raise ValueError("DRBII strata must contain only public task IDs, languages and themes")
    strata = sorted(drbii_strata, key=lambda row: row["task_id"])
    if digest(strata) != DRBII_STRATA_SHA256:
        raise ValueError("DRBII public strata differ from the independently pinned release")
    ids = {row["task_id"] for row in parent["task_index"]
           if row["benchmark"] == "deepresearch_bench_ii"}
    if {row["task_id"] for row in strata} != ids:
        raise ValueError("DRBII public strata identity differs from the parent inventory")
    return strata


def select_members(rows, eligible, count, seed):
    """Public proportional selection with exact quotas and intact groups."""
    index = {row["task_id"]: row for row in rows}
    eligible = set(eligible)
    if not eligible <= index.keys():
        raise ValueError("Subset selection contains an unknown public task ID")
    selected_groups = {index[key]["group_id"] for key in eligible}
    if any(row["group_id"] in selected_groups and key not in eligible
           for key, row in index.items()):
        raise ValueError("Eligible pool would split a duplicate group")
    metadata = {key: {"group_id": row["group_id"],
                      "problem_category": row["selection_stratum"]}
                for key, row in index.items()}
    return _select(_groups(SimpleNamespace(public_metadata=metadata), eligible), count, seed)


def _build_subset(parent, drbii_strata):
    strata = _trusted_inputs(parent, drbii_strata)
    drbii_index = {row["task_id"]: row for row in strata}
    rows = []
    for original in parent["task_index"]:
        row = copy.deepcopy(original)
        name, key = row["benchmark"], row["task_id"]
        row["parent_partition"] = row["partition"]
        row["partition"] = "unused"
        row["selection_stratum"] = (
            drbii_index[key]["theme"] if name == "deepresearch_bench_ii"
            else row["stratum"] if name in ("deepsearchqa", "writingbench") else "all")
        rows.append(row)
    memberships = {}
    for name in BENCHMARKS:
        public = [row for row in rows if row["benchmark"] == name]
        allocations = {}
        for part, count in zip(PARTITIONS, COUNTS[name]):
            pool = [row["task_id"] for row in public if row["parent_partition"] == part]
            if name == "researchrubrics" and part == "test":
                pool = [row["task_id"] for row in public if row["test_slice"] == "clean_unseen"]
            if name == "deepresearch_bench_ii" and part == "test":
                chosen = []
                for language in ("en", "zh"):
                    ids = [key for key in pool if drbii_index[key]["language"] == language]
                    chosen.extend(select_members(public, ids, 20, f"{SEED}:{name}:{part}:{language}"))
                allocations[part] = sorted(chosen, key=lambda key: rank(f"{SEED}:{name}:{part}", key))
            else:
                allocations[part] = select_members(public, pool, count, f"{SEED}:{name}:{part}")
        assignments = {key: part for part, keys in allocations.items() for key in keys}
        allocations["unused"] = sorted(
            (row["task_id"] for row in public if row["task_id"] not in assignments),
            key=lambda key: rank(f"{SEED}:{name}:unused", key))
        for row in public:
            row["partition"] = assignments.get(row["task_id"], "unused")
        memberships[name] = allocations
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row["benchmark"], row["group_id"])].add(row["partition"])
    if any(len(parts) > 1 for parts in grouped.values()):
        raise ValueError("Duplicate group crosses selected and unused partitions")
    schedules = []
    for run_id, seed in enumerate(ORDER_SEEDS):
        parent_order = parent["joint_evolution_schedule"][run_id]["task_ids"]
        ordered = {name: [key for key in parent_order if key in set(memberships[name]["evolution"])]
                   for name in SOURCES}
        stages = []
        for stage in range(4):
            stages.append([ordered[name][stage * 5 + offset]
                           for offset in range(5) for name in SOURCES])
        schedules.append({"run_id": run_id, "order_seed": seed, "stages": stages,
                          "task_ids": [key for stage in stages for key in stage]})
    document = {
        "version": VERSION, "status": "SUBSET_PROTOCOL_FROZEN_NOT_RUN",
        "membership_seed": SEED, "parent_manifest": "joint_task_splits_v4.json",
        "parent_manifest_sha256": PARENT_SHA256,
        "benchmarks": copy.deepcopy(parent["benchmarks"]),
        "counts": {name: {part: len(memberships[name][part]) for part in ALL_PARTITIONS}
                   for name in BENCHMARKS},
        "totals": {part: sum(len(memberships[name][part]) for name in BENCHMARKS)
                   for part in ALL_PARTITIONS},
        "memberships": memberships, "task_index": rows,
        "drbii_public_strata": strata, "drbii_public_strata_sha256": DRBII_STRATA_SHA256,
        "joint_evolution_schedule": schedules, "checkpoint_positions": list(CHECKPOINT_POSITIONS),
        "allocation": "Within each frozen parent partition only; exact quotas, intact normalized-question groups, proportional public strata, deterministic SHA256 ties",
        "unused_policy": "Excluded from evolution, VAL, mandatory TEST and release selection; no Dev or replacement pool",
        "exposure_policy": "All 18 former RR exposed/reserved TEST rows are unused; prior WritingBench exposure can only remain EVO or become unused",
        "row_alias_warning": parent["row_alias_warning"],
        "overlap_scope": parent["overlap_scope"], "results": [],
    }
    document["manifest_sha256"] = digest(document)
    return document


def build_subset(parent, drbii_strata):
    return _build_subset(parent, drbii_strata)


def validate_subset(document, parent):
    """Reconstruct from independent parent/strata anchors, not submitted hashes."""
    content = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("version") != VERSION or digest(content) != document.get("manifest_sha256"):
        raise ValueError("Subset manifest integrity check failed")
    expected = _build_subset(parent, document.get("drbii_public_strata"))
    if document != expected:
        raise ValueError("Subset differs from deterministic pinned membership, mapping or schedule")
    return document


def subset_validation_summary(records, validation_ids, task_bounds):
    return validation_summary(records, validation_ids, task_bounds,
                              counts={name: 10 for name in SOURCES}, repeats=1)


def workload():
    runs, sources, checkpoints, validation, test = 3, 60, 5, 30, 273
    evolution_slots = runs * sources
    validation_slots = runs * checkpoints * validation
    selected_test_slots, static_test_slots = runs * test, 4 * test
    total = evolution_slots + validation_slots + selected_test_slots + static_test_slots
    parent_same_core = 3 * 330 + 3 * 7 * 220 * 2 + (3 + 4) * 2424 * 3
    return {"unique_selected_tasks": 363, "unused_tasks": 2611,
            "evolution_slots": evolution_slots, "validation_slots": validation_slots,
            "selected_test_slots": selected_test_slots, "static_test_slots": static_test_slots,
            "test_slots": selected_test_slots + static_test_slots, "total_task_slots": total,
            "v4_same_five_methods_task_slots": parent_same_core,
            "task_slot_reduction_percent": 100 * (1 - total / parent_same_core),
            "unit": "Nominal task-artifact slots before identical-state reuse, NOT model calls or token/cost ceilings",
            "excluded": "Shared evidence preparation, scorer setup, attribution details and optional extra studies are accounted separately"}


def protocol_for(document):
    protocol = {
        "version": "jit-compose-joint-protocol-v5", "status": "SUBSET_PROTOCOL_FROZEN_NOT_RUN",
        "parent_protocol": "joint_protocol_v4.json (immutable; v5 only subsamples within parent partitions)",
        "split_manifest": "joint_task_splits_v5.json", "split_manifest_sha256": document["manifest_sha256"],
        "readable_task_numbers": "task_assignments_v5.md", "counts": document["counts"],
        "totals": document["totals"], "workload": workload(),
        "scope": "One benchmark-agnostic, task-conditioned MAS generator per joint replicate; same frozen selected version on all six benchmarks",
        "development_partition": "None; unused tasks are not a development or replacement set",
        "evolution": {
            "source_benchmarks": list(SOURCES), "stores": "One shared mutable experience store per replicate",
            "order_seeds": list(ORDER_SEEDS), "passes": 1, "stages": 4,
            "tasks_per_stage": {name: 5 for name in SOURCES},
            "order": "Filter each frozen parent run order to selected sources; interleave RR, DSQA, WritingBench one each, five times per stage",
            "checkpoint_positions": list(CHECKPOINT_POSITIONS),
            "updates": "Direct per-source writes after structural/evidence checks; no accept/hold/reject gate",
            "failure_policy": "Consume each source position once; no replacement or quality resampling",
            "continuation": "Latest trajectory, no VAL rollback or early stopping",
            "retrieval": "Only the 60 global EVO task IDs may source shared experience; exclude VAL, TEST and unused",
        },
        "validation": {
            "scope": "All fixed 30 VAL tasks at each of all five checkpoints, never a recent batch",
            "artifacts_per_task": 1, "slots_per_checkpoint": 30,
            "minimum_complete_by_benchmark": {name: 9 for name in SOURCES},
            "native_score_normalization": {
                "researchrubrics": "(score-L_t)/(U_t-L_t), theoretical per-task bounds fixed before responses; degenerate range maps to zero",
                "deepsearchqa": "Native F1 [0,1]", "writingbench": "(native checklist mean - 1)/9"},
            "aggregation": "Within-benchmark normalized task mean, then equal 1/3 benchmark weights",
            "missing": "Observed score remains null; conservative normalized selector substitutes zero",
            "selection": "All five checkpoints including C0; max utility (1e-12 tolerance), more completed slots, earlier position, lexical state hash",
            "isolation": "Read-only snapshots, no experience writes or feedback to source actors or mid-run human tuning",
            "cache": "Reuse identical state and execution identity observations, never resample for better VAL",
        },
        "universal_generator_release": {
            "default_run_id": 0, "default_order_seed": ORDER_SEEDS[0],
            "checkpoint": "Run 0 joint-VAL winner; no eligible state means no release",
            "replicates": "Report all three selected states; never select run or state using TEST",
            "components": ["code and prompt hashes", "configuration and retrieval/tool policy", "model and serving identity",
                           "joint ExperienceSnapshot", "JIT archive/dependencies", "data/split/protocol/journal hashes"],
            "test_binding": "Same selected state and generator/config hash across all six benchmarks within each run; only input/scoring adapters differ",
            "not_claimed": "Not new model weights, not a fixed team, not a completed evolved deployment",
        },
        "test": {
            "read_only": True, "attribution": False, "artifacts_per_task_per_selected_state": 1,
            "static_baseline_artifacts_per_task_total": 1,
            "static_sharing": "Each static baseline is generated once per task and shared across the three paired trajectory comparisons, not three independent observations",
            "release_boundary": "Seal all mandatory submissions across six benchmarks before any TEST scoring/feedback release",
            "adaptation": "No updates, target tuning, source switching, checker-guided repair or extra negotiation",
            "instruction_track": "Single-turn IFEval and IFBench",
            "unused": "Never score or use for extra checkpoint selection without a versioned amendment",
        },
        "comparison": {
            "core_methods": ["ours_initial", "ours_selected", "direct", "jit_matched", "rubric_fixed"],
            "static_methods": ["ours_initial", "direct", "jit_matched", "rubric_fixed"],
            "optional_not_in_default_workload": ["ours_terminal", "mechanism ablations", "G/GO guidance"],
            "matching": "Same backbone, task evidence and resource ceilings; actual cost reported separately",
        },
        "evaluation": {
            "researchrubrics": "Pinned signed-weight native score on all 33 inherited clean TEST tasks",
            "deepsearchqa": "Native task F1; precision, recall and fully-correct rate auxiliary",
            "writingbench": "Native mean checklist score 1-10 per task, then task macro mean",
            "deepresearch_bench_ii": "Native fraction of labels equal to 1; blocked-source rate separate; 20 en and 20 zh",
            "ifeval": "Author prompt-level strict accuracy; loose/instruction-level auxiliary",
            "ifbench": "Author prompt-level loose accuracy; strict/instruction-level auxiliary",
            "aggregate_policy": "Report native benchmark metrics separately, never VAL utility as a leaderboard score",
            "uncertainty": "Whole-task paired bootstrap 10000 and sign-flip 100000, seed 20261001; Holm within research/writing and instruction diagnostic families; report trajectory variation separately",
            "scope_limit": "Small fixed subset study, not a full-benchmark or SOTA claim; no outcome-driven task replacement",
        },
        "runtime_status": "Metadata preparation and selector are executable; legacy v3 per-benchmark runner is not a v5 joint runner",
        "execution_profile": "Retain configs/benchmark_suite_v3.yaml resource limits; scorer/model identities must be fixed in any real run bundle",
        "results": [],
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def render_assignments(document):
    lines = ["# Joint v5 Exact Subset Task Assignments", "",
             "Numbers are 1-based data-record positions in the pinned source file; CSV header excluded.",
             "Do not renumber after sorting or filtering. JSON maps each row to source_id and stable task_id.",
             "Only selected EVO/VAL/TEST tasks are run; unused is not Dev or a replacement pool.",
             "All v4 source/VAL/TEST boundaries and recorded exposures are preserved.",
             f"Manifest SHA256: `{document['manifest_sha256']}`", ""]
    for name in BENCHMARKS:
        source = document["benchmarks"][name]
        rows = [row for row in document["task_index"] if row["benchmark"] == name]
        lines += [f"## {name}", "", f"Revision: `{source['dataset_revision']}`",
                  f"Data SHA256: `{source['dataset_sha256']}`", f"Source: {source['data_url']}", ""]
        for part in ALL_PARTITIONS:
            numbers = [row["source_row"] for row in rows if row["partition"] == part]
            lines += [f"### {part.upper()} ({len(numbers)})", "", compact_rows(numbers), ""]
        if name == "deepresearch_bench_ii":
            strata = {row["task_id"]: row for row in document["drbii_public_strata"]}
            for language in ("en", "zh"):
                numbers = [row["source_row"] for row in rows if row["partition"] == "test"
                           and strata[row["task_id"]]["language"] == language]
                lines += [f"### TEST Language {language} ({len(numbers)})", "", compact_rows(numbers), ""]
    index = {row["task_id"]: row for row in document["task_index"]}
    lines += ["## Joint Evolution Stages", "", "Each entry is benchmark:source_row.", ""]
    for run in document["joint_evolution_schedule"]:
        lines += [f"### Run {run['run_id']} (seed {run['order_seed']})", ""]
        for number, stage in enumerate(run["stages"], 1):
            aliases = [f"{index[key]['benchmark']}:{index[key]['source_row']}" for key in stage]
            lines += [f"Stage {number}, checkpoint C{15 * number}:", "", ", ".join(aliases), ""]
    return "\n".join(lines)
