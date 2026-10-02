"""Public-only membership and selection rules for the joint v4 experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from types import SimpleNamespace

from .experiment_splits import ORDER_SEEDS, _groups, _select, rank, validate_split
from .schemas import digest


SOURCES = ("researchrubrics", "deepsearchqa", "writingbench")
TARGETS = ("deepresearch_bench_ii", "ifeval", "ifbench")
BENCHMARKS = SOURCES + TARGETS
COUNTS = {
    "researchrubrics": (30, 20, 51), "deepsearchqa": (150, 100, 650),
    "writingbench": (150, 100, 750), "deepresearch_bench_ii": (0, 0, 132),
    "ifeval": (0, 0, 541), "ifbench": (0, 0, 300),
}
PARTITIONS = ("evolution", "validation", "test")
ROW_FIELDS = {"task_id", "source_id", "source_row", "question_sha256", "group_id", "stratum"}
VERSION = "jit-compose-joint-split-v4"
SEED = "jit-compose-joint-v4"
# Computed from all six pinned raw releases after public-only projection.
# These anchors are independent of the self-hash in a submitted split document.
PUBLIC_INVENTORY_HASHES = {
    "researchrubrics": "86b683b7719e147bf7e35a69fbcb34470a2d75bb1c8801086706427369b251d8",
    "deepsearchqa": "248644c6bcf3c0b7863bae7456a953dc80d475f5ad0534c6afc6e54789e24527",
    "deepresearch_bench_ii": "9e2d34abbcf9869a11b62b8eb497b7d1f05ec200f8a3996a975eceba2489748b",
    "writingbench": "2e0bae8f0e166a65220a43cc7a06934f4b3f5a0c113dbe3844a3f6b5562a270f",
    "ifeval": "0c9d67280f4df0b199d793e53678dee53763d8e61bacb9c20bb9402990c3c1e5",
    "ifbench": "818f13a741b5ca5b68937f4cee50553a67599e100456e7038fca0a855f227b11",
}


def inventory_fingerprint(inventory):
    """Bind original row aliases to independently pinned public projections."""
    content = {key: inventory[key] for key in
               ("benchmark", "dataset_revision", "dataset_sha256", "data_url")}
    content["rows"] = sorted(inventory["rows"], key=lambda row: row["source_row"])
    content["known_exposures"] = inventory.get("known_exposures", [])
    return digest(content)


def _writing_membership(inventory):
    rows = {row["task_id"]: row for row in inventory["rows"]}
    exposed = {entry["task_id"] for entry in inventory.get("known_exposures", [])}
    if not exposed <= rows.keys():
        raise ValueError("Exposed writing task is absent from inventory")
    exposed_groups = {rows[key]["group_id"] for key in exposed}
    forced = {key for key, row in rows.items() if row["group_id"] in exposed_groups}
    if len(forced) > 150:
        raise ValueError("Exposed writing groups exceed evolution quota")
    data = SimpleNamespace(public_metadata={key: {**row, "problem_category": row["stratum"]}
                                           for key, row in rows.items()})
    remaining = set(rows) - forced
    picked = _select(_groups(data, remaining), 150 - len(forced), f"{SEED}:writing:evolution")
    evolution = sorted(forced | set(picked), key=lambda key: rank(SEED, key))
    remaining.difference_update(evolution)
    validation = _select(_groups(data, remaining), 100, f"{SEED}:writing:validation")
    remaining.difference_update(validation)
    return {"evolution": evolution, "validation": validation,
            "test": sorted(remaining, key=lambda key: rank(SEED, key))}


def _validate_inventory(name, inventory):
    rows = inventory["rows"]
    if inventory["benchmark"] != name or len(rows) != sum(COUNTS[name]):
        raise ValueError("Inventory name/count differs from protocol")
    if any(set(row) != ROW_FIELDS for row in rows):
        raise ValueError("Only public identity and stratification fields are permitted")
    if (len({row["task_id"] for row in rows}) != len(rows)
            or {row["source_row"] for row in rows} != set(range(1, len(rows) + 1))):
        raise ValueError("Inventory task IDs or original row aliases are not unique/complete")
    if any(not isinstance(row["task_id"], str) or not row["task_id"] for row in rows):
        raise ValueError("Invalid public task identity")
    if inventory_fingerprint(inventory) != PUBLIC_INVENTORY_HASHES[name]:
        raise ValueError("Public row aliases or release identity differ from independently pinned metadata")
    checksum = inventory.get("inventory_sha256")
    if checksum and digest({key: value for key, value in inventory.items()
                            if key != "inventory_sha256"}) != checksum:
        raise ValueError("Public inventory integrity check failed")


def build_joint_split(inventories, inherited):
    """Preserve existing source/VAL IDs; remove Dev without erasing exposure."""
    if set(inventories) != set(BENCHMARKS):
        raise ValueError("The joint suite requires exactly six public inventories")
    allocations, rows, sources = {}, [], {}
    for name in BENCHMARKS:
        inventory = inventories[name]
        _validate_inventory(name, inventory)
        index = {row["task_id"]: row for row in inventory["rows"]}
        old_development = set()
        if name in inherited:
            old = inherited[name]
            manifest = validate_split(old)
            if old["benchmark"] != name or old["dataset_sha256"] != inventory["dataset_sha256"]:
                raise ValueError("Inherited split and public inventory releases differ")
            if set(index) != {row["task_id"] for row in old["public_index"]}:
                raise ValueError("Inherited task inventory changed")
            for row in old["public_index"]:
                if any(row[field] != index[row["task_id"]][field]
                       for field in ("question_sha256", "group_id", "stratum")):
                    raise ValueError("Inherited public metadata changed")
            old_development = set(old["development"])
            allocation = {part: list(getattr(manifest, part)) for part in PARTITIONS}
            allocation["test"] += list(old["development"])
        elif name == "writingbench":
            allocation = _writing_membership(inventory)
        else:
            allocation = {"evolution": [], "validation": [], "test": list(index)}
        allocations[name] = allocation
        sources[name] = {field: inventory[field] for field in
                         ("benchmark", "dataset_revision", "dataset_sha256", "data_url")}
        sources[name]["known_exposures"] = copy.deepcopy(inventory.get("known_exposures", []))
        sources[name]["source_row_convention"] = (
            "1-based data-record order in the pinned file, before sorting; CSV header excluded")
        assignments = {key: part for part, members in allocation.items() for key in members}
        for row in sorted(index.values(), key=lambda row: row["source_row"]):
            key, part = row["task_id"], assignments[row["task_id"]]
            prior = key in old_development
            exposed_test = name == "researchrubrics" and prior
            rows.append({**row, "benchmark": name, "partition": part,
                         "former_development": prior,
                         "test_slice": ("historically_exposed_or_reserved" if exposed_test
                                        else "clean_unseen") if part == "test" else None})
    schedules = []
    for run_id, seed in enumerate(ORDER_SEEDS):
        ordered = {
            name: (inherited[name]["evolution_schedule"][run_id]["task_ids"]
                   if name in inherited else sorted(allocations[name]["evolution"],
                       key=lambda key: rank(f"{SEED}:order:{seed}", key)))
            for name in SOURCES}
        stages = []
        for stage in range(6):
            # One RR task and five tasks from each larger source per mini-block.
            block = []
            for offset in range(5):
                block.append(ordered["researchrubrics"][stage * 5 + offset])
                for index in range(5):
                    position = stage * 25 + offset * 5 + index
                    block.extend([ordered["deepsearchqa"][position], ordered["writingbench"][position]])
            stages.append(block)
        schedules.append({"run_id": run_id, "order_seed": seed,
                          "stages": stages, "task_ids": [key for stage in stages for key in stage]})
    document = {
        "version": VERSION, "membership_seed": SEED, "benchmarks": sources,
        "counts": {name: dict(zip(PARTITIONS, COUNTS[name])) for name in BENCHMARKS},
        "memberships": allocations, "task_index": rows, "joint_evolution_schedule": schedules,
        "checkpoint_positions": list(range(0, 331, 55)),
        "inherited_manifest_hashes": {name: doc["manifest_sha256"] for name, doc in inherited.items()},
        "row_alias_warning": "Readable row numbers refer to the pinned raw data order, not sorted task IDs; use task_id and data hash as identity",
        "exposure_policy": "Former RR quarantine enters test inventory but never clean primary estimates; prior-exposed WritingBench groups enter evolution",
        "overlap_scope": "Normalized exact-question audit only; not a semantic/source-disjoint guarantee",
        "results": [],
    }
    document["manifest_sha256"] = digest(document)
    validate_joint_split(document)
    return document


def validate_joint_split(document):
    content = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("version") != VERSION or digest(content) != document.get("manifest_sha256"):
        raise ValueError("Joint split integrity check failed")
    if set(document["memberships"]) != set(BENCHMARKS) or set(document["counts"]) != set(BENCHMARKS):
        raise ValueError("Joint benchmark inventory differs from registration")
    rows = document["task_index"]
    allowed = ROW_FIELDS | {"benchmark", "partition", "former_development", "test_slice"}
    if any(set(row) != allowed for row in rows):
        raise ValueError("Joint task index contains non-public or unexpected fields")
    index = {row["task_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Joint task namespace collision")
    grouped, cross = defaultdict(set), defaultdict(set)
    seen = set()
    for name in BENCHMARKS:
        _validate_inventory(name, {**document["benchmarks"][name], "rows": [
            {field: row[field] for field in ROW_FIELDS} for row in rows if row["benchmark"] == name]})
        members = document["memberships"][name]
        if set(members) != set(PARTITIONS) or document["counts"][name] != dict(zip(PARTITIONS, COUNTS[name])):
            raise ValueError("Unexpected partition/count; Dev is not a v4 partition")
        expected_rows = set(range(1, sum(COUNTS[name]) + 1))
        if {row["source_row"] for row in rows if row["benchmark"] == name} != expected_rows:
            raise ValueError("Original row aliases are incomplete or inconsistent")
        for part, count in zip(PARTITIONS, COUNTS[name]):
            keys = members[part]
            if len(keys) != count or len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError("Partition overlap or count mismatch")
            seen.update(keys)
            for key in keys:
                row = index[key]
                if row["benchmark"] != name or row["partition"] != part:
                    raise ValueError("Membership and public row index differ")
                expected_slice = ("historically_exposed_or_reserved"
                                  if name == "researchrubrics" and row["former_development"]
                                  else "clean_unseen") if part == "test" else None
                if row["test_slice"] != expected_slice:
                    raise ValueError("Historical exposure label was changed")
                grouped[(name, row["group_id"])].add(part)
                cross[row["question_sha256"]].add(name)
        for exposure in document["benchmarks"][name]["known_exposures"]:
            if name == "writingbench" and exposure["task_id"] not in members["evolution"]:
                raise ValueError("Exposed writing tasks must enter evolution, not VAL/TEST")
    if seen != set(index) or any(len(parts) > 1 for parts in grouped.values()):
        raise ValueError("Incomplete inventory or duplicate group crosses partitions")
    if any(len(names) > 1 for names in cross.values()):
        raise ValueError("Cross-benchmark exact duplicates require an explicit protocol amendment")
    if sum(row["test_slice"] == "historically_exposed_or_reserved" for row in rows) != 18:
        raise ValueError("Inherited RR exposure/reservation inventory must remain explicit")
    expected_sources = {key for name in SOURCES for key in document["memberships"][name]["evolution"]}
    if document["checkpoint_positions"] != list(range(0, 331, 55)):
        raise ValueError("Joint checkpoints must follow six complete mixed stages")
    if len(document["joint_evolution_schedule"]) != len(ORDER_SEEDS):
        raise ValueError("Missing joint source-order replicate")
    for run_id, run in enumerate(document["joint_evolution_schedule"]):
        if run["run_id"] != run_id or run["order_seed"] != ORDER_SEEDS[run_id] or len(run["stages"]) != 6:
            raise ValueError("Invalid joint source-order identity")
        ordered = [key for stage in run["stages"] for key in stage]
        if ordered != run["task_ids"] or len(ordered) != 330 or set(ordered) != expected_sources:
            raise ValueError("Each joint source must appear exactly once")
        for stage in run["stages"]:
            if Counter(index[key]["benchmark"] for key in stage) != dict(zip(SOURCES, (5, 25, 25))):
                raise ValueError("Every stage must contain 5 RR, 25 DSQA and 25 WritingBench tasks")
    return document


def validation_summary(records, validation_ids, task_bounds, *, counts=None, repeats=2):
    """Executable reference for equal-benchmark VAL utility, never a test metric.

    Records use native scores. Incomplete/missing slots retain no observed score;
    only the conservative normalized selector substitutes zero for those slots.
    Counts/repeats must come from the frozen protocol; defaults retain v4.
    """
    if set(validation_ids) != set(SOURCES):
        raise ValueError("Require all three fixed validation sets")
    counts = {name: COUNTS[name][1] for name in SOURCES} if counts is None else dict(counts)
    if (set(counts) != set(SOURCES) or any(type(value) is not int or value < 1 for value in counts.values())
            or type(repeats) is not int or repeats < 1):
        raise ValueError("Registered validation counts and repeats must be positive integers")
    all_ids = [key for name in SOURCES for key in validation_ids[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Validation task IDs must be globally unique")
    expected = {(name, key, repeat) for name in SOURCES for key in validation_ids[name] for repeat in range(repeats)}
    observations = {}
    for row in records:
        slot = (row["benchmark"], row["task_id"], row["repeat"])
        if slot not in expected or slot in observations:
            raise ValueError("Unexpected or duplicate validation slot")
        observations[slot] = row
    summary = {}
    for name in SOURCES:
        ids = list(validation_ids[name])
        if len(ids) != counts[name] or len(ids) != len(set(ids)):
            raise ValueError("Validation set differs from fixed benchmark count")
        complete, values = 0, []
        for key in ids:
            lower, upper = map(float, task_bounds[key])
            if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
                raise ValueError("Task bounds must be finite, ordered and fixed before judging")
            fixed = {"deepsearchqa": (0.0, 1.0), "writingbench": (1.0, 10.0)}
            if name in fixed and (lower, upper) != fixed[name]:
                raise ValueError("Benchmark score range differs from the registered native scale")
            if name == "researchrubrics" and not ((lower <= 0 and upper == 1) or (lower, upper) == (0, 0)):
                raise ValueError("Invalid ResearchRubrics theoretical bounds")
            for repeat in range(repeats):
                row = observations.get((name, key, repeat), {})
                value = 0.0
                if row.get("complete") is True:
                    score = row.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("Complete validation needs a finite observed native score")
                    if not lower - 1e-12 <= score <= upper + 1e-12:
                        raise ValueError("Observed score falls outside registered theoretical bounds")
                    complete += 1
                    value = min(1.0, max(0.0, (score - lower) / (upper - lower))) if upper > lower else 0.0
                values.append(value)
        total = len(values)
        threshold = math.ceil(total * 0.9 - 1e-12)
        summary[name] = {"slots": total, "complete": complete, "minimum_complete": threshold,
                         "eligible": complete >= threshold, "selection_utility": sum(values) / total}
    return {"per_benchmark": summary, "eligible": all(row["eligible"] for row in summary.values()),
            "complete_evaluations": sum(row["complete"] for row in summary.values()),
            "selection_utility": sum(row["selection_utility"] for row in summary.values()) / 3}


def compact_rows(numbers):
    numbers = sorted(set(numbers))
    if not numbers:
        return "none"
    spans, start, end = [], numbers[0], numbers[0]
    for number in numbers[1:]:
        if number == end + 1:
            end = number
        else:
            spans.append(str(start) if start == end else f"{start}-{end}")
            start = end = number
    spans.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(spans)


def render_assignments(document):
    validate_joint_split(document)
    lines = ["# Joint v4 Exact Task Assignments", "",
             "All numbers below are 1-based data-record positions in the pinned source file (CSV header excluded).",
             "Do not renumber after sorting/filtering. The JSON task_index maps every number to its original source_id and stable task_id.",
             "Membership lists are sorted for reading; execution order is the separately frozen joint_evolution_schedule.",
             "No Dev partition. Former RR quarantine is supplementary TEST, not clean held-out evidence.",
             f"Manifest SHA256: `{document['manifest_sha256']}`", ""]
    for name in BENCHMARKS:
        source = document["benchmarks"][name]
        selected = [row for row in document["task_index"] if row["benchmark"] == name]
        lines += [f"## {name}", "", f"Revision: `{source['dataset_revision']}`",
                  f"Data SHA256: `{source['dataset_sha256']}`", f"Source: {source['data_url']}", ""]
        for part in PARTITIONS:
            numbers = [row["source_row"] for row in selected if row["partition"] == part]
            lines += [f"### {part.upper()} ({len(numbers)})", "", compact_rows(numbers), ""]
        exposed = [row["source_row"] for row in selected if row["test_slice"] == "historically_exposed_or_reserved"]
        if exposed:
            clean = [row["source_row"] for row in selected if row["test_slice"] == "clean_unseen"]
            lines += [f"### Primary Clean TEST ({len(clean)})", "", compact_rows(clean), "",
                      f"### Supplementary Exposed/Reserved TEST ({len(exposed)})", "", compact_rows(exposed), ""]
    index = {row["task_id"]: row for row in document["task_index"]}
    lines += ["## Joint Evolution Stages", "",
              "Each entry is benchmark:source_row. All 3 orders share the same EVO/VAL/TEST membership.", ""]
    for run in document["joint_evolution_schedule"]:
        lines += [f"### Run {run['run_id']} (seed {run['order_seed']})", ""]
        for stage, tasks in enumerate(run["stages"], 1):
            aliases = [f"{index[key]['benchmark']}:{index[key]['source_row']}" for key in tasks]
            lines += [f"Stage {stage}, checkpoint C{55 * stage}:", "", ", ".join(aliases), ""]
    return "\n".join(lines)
