"""Public-metadata-only, deterministic benchmark partitions and source orders."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib

from .schemas import SplitManifest, digest


PARTITIONS = ("development", "evolution", "validation", "test")
ORDER_SEEDS = (20261001, 20261002, 20261003)


def rank(seed, value):
    return hashlib.sha256(f"{seed}\n{value}".encode("utf-8")).hexdigest()


def _group_subsets(groups, limit):
    reachable = {0: ()}
    for group in groups:
        size = len(group)
        for count in sorted(tuple(reachable), reverse=True):
            if count + size <= limit and count + size not in reachable:
                reachable[count + size] = reachable[count] + (group,)
    return reachable


def _select(groups_by_stratum, count, seed):
    """Minimize proportional allocation error subject to intact groups/exact N."""
    total = sum(len(group) for groups in groups_by_stratum.values() for group in groups)
    if not 0 <= count <= total:
        raise ValueError("Partition quota exceeds remaining public tasks")
    allocations = {0: (Fraction(0), ())}
    choices = {}
    for stratum in sorted(groups_by_stratum):
        groups = sorted(groups_by_stratum[stratum], key=lambda group: rank(seed, "|".join(group)))
        choices[stratum] = _group_subsets(groups, count)
        population = sum(map(len, groups))
        ideal = Fraction(count * population, total) if total else Fraction(0)
        next_allocations = {}
        for allocated, (cost, path) in allocations.items():
            for take in choices[stratum]:
                if allocated + take > count:
                    continue
                candidate = (cost + abs(take - ideal), path + ((stratum, take),))
                if allocated + take not in next_allocations or candidate < next_allocations[allocated + take]:
                    next_allocations[allocated + take] = candidate
        allocations = next_allocations
    if count not in allocations:
        raise ValueError("Exact quotas cannot preserve duplicate groups; amend counts before running")
    selected = []
    for stratum, take in allocations[count][1]:
        for group in choices[stratum][take]:
            selected.extend(group)
    return sorted(selected, key=lambda key: rank(seed, key))


def _groups(dataset, remaining):
    groups = defaultdict(list)
    for key in sorted(remaining):
        groups[dataset.public_metadata[key]["group_id"]].append(key)
    stratified = defaultdict(list)
    for ids in groups.values():
        strata = sorted({dataset.public_metadata[key].get("problem_category", "all") for key in ids})
        stratified["|".join(strata)].append(tuple(ids))
    return stratified


def build_split(dataset, specification, *, inherited=None, exposed_ids=()):
    counts = specification["counts"]
    if set(counts) != set(PARTITIONS) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Require four nonnegative integer partition counts")
    ids = set(dataset.tasks)
    if sum(counts.values()) != len(ids) or len(ids) != specification["expected_tasks"]:
        raise ValueError("Dataset inventory differs from the registered benchmark release")
    exposed = set(exposed_ids)
    if not exposed <= ids or len(exposed) > counts["development"]:
        raise ValueError("Exposures must fit entirely inside the registered development quota")
    seed = specification["split_seed"]
    if inherited is not None:
        inherited_groups = inherited["runtime_split_manifest"]
        partitions = {"development": inherited["development_quarantine"],
                      **{key: list(inherited_groups[key]) for key in PARTITIONS[1:]}}
        if not exposed <= set(partitions["development"]):
            raise ValueError("New exposure overlaps a frozen held-out partition; versioned amendment required")
        if inherited["provenance"]["dataset_sha256"] != dataset.dataset_sha256:
            raise ValueError("Inherited split describes a different dataset release")
    else:
        # Quarantine the entire duplicate group when any member has prior exposure.
        exposed_groups = {dataset.public_metadata[key]["group_id"] for key in exposed}
        quarantine = {key for key in ids if dataset.public_metadata[key]["group_id"] in exposed_groups}
        if len(quarantine) > counts["development"]:
            raise ValueError("Exposed duplicate groups exceed development quota")
        partitions, remaining = {}, ids - quarantine
        for name in PARTITIONS[:-1]:
            already = quarantine if name == "development" else set()
            picked = _select(_groups(dataset, remaining), counts[name] - len(already), f"{seed}:{name}")
            partitions[name] = sorted(already, key=lambda key: rank(seed, key)) + picked
            remaining.difference_update(picked)
        partitions["test"] = sorted(remaining, key=lambda key: rank(f"{seed}:test", key))
    assignments = {key: name for name, members in partitions.items() for key in members}
    if set(assignments) != ids or sum(map(len, partitions.values())) != len(ids):
        raise ValueError("Partition inventory is incomplete or overlapping")
    for name in PARTITIONS:
        if len(partitions[name]) != counts[name]:
            raise ValueError("Partition count does not match registration")
    group_partitions = defaultdict(set)
    for key, name in assignments.items():
        group_partitions[dataset.public_metadata[key]["group_id"]].add(name)
    if any(len(names) > 1 for names in group_partitions.values()):
        raise ValueError("Normalized duplicate group crosses registered partitions")
    batch = specification["batch_size"]
    schedules = []
    if counts["evolution"]:
        if type(batch) is not int or batch < 1 or counts["evolution"] % batch:
            raise ValueError("Evolution count must comprise complete registered batches")
        for run_id, order_seed in enumerate(ORDER_SEEDS):
            if inherited and "evolution_schedule" in inherited:
                order = inherited["evolution_schedule"]["runs"][run_id]["ordered_task_ids"]
            else:
                order = sorted(partitions["evolution"], key=lambda key: rank(f"{seed}:order:{order_seed}", key))
            if len(order) != counts["evolution"] or set(order) != set(partitions["evolution"]):
                raise ValueError("Invalid inherited evolution order")
            schedules.append({"run_id": run_id, "order_seed": order_seed, "task_ids": order,
                              "batches": [order[start:start + batch] for start in range(0, len(order), batch)]})
    manifest = SplitManifest(seed=ORDER_SEEDS[0], **{name: partitions[name] for name in PARTITIONS[1:]})
    index = [{"task_id": key, "partition": assignments[key],
              "question_sha256": dataset.public_metadata[key]["question_sha256"],
              "group_id": dataset.public_metadata[key]["group_id"],
              "stratum": dataset.public_metadata[key].get("problem_category", "all")}
             for key in sorted(ids)]
    document = {"version": "jit-compose-benchmark-split-v3", "benchmark": dataset.name,
        "dataset_sha256": dataset.dataset_sha256, "specification_sha256": digest(specification),
        "official_split_policy": specification["official_split_policy"], "counts": counts,
        "development": partitions["development"], "runtime_split_manifest": manifest.model_dump(mode="json"),
        "evolution_schedule": schedules,
        "checkpoint_positions": list(range(0, counts["evolution"] + 1, batch)) if batch else [],
        "exposed_task_ids": sorted(exposed), "public_index": index,
        "allocation": "Minimum absolute proportional quota deviation over public category strata, intact exact-question groups, deterministic SHA256 ties; rare strata need not occur in every subset",
        "scope": "Task-level split; exact-question grouping is not a semantic or topic-disjoint guarantee",
        "results": []}
    document["manifest_sha256"] = digest(document)
    return document


def validate_split(document, dataset=None, specification=None):
    content = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("version") != "jit-compose-benchmark-split-v3" or digest(content) != document.get("manifest_sha256"):
        raise ValueError("Registered split integrity check failed")
    manifest = SplitManifest.model_validate(document["runtime_split_manifest"])
    groups = {"development": document["development"],
              **{name: getattr(manifest, name) for name in PARTITIONS[1:]}}
    flattened = [key for keys in groups.values() for key in keys]
    if manifest.stream or len(flattened) != len(set(flattened)) or any(len(groups[k]) != document["counts"][k] for k in PARTITIONS):
        raise ValueError("Registered split overlap or count mismatch")
    index = document["public_index"]
    expected = {key: name for name, members in groups.items() for key in members}
    public_fields = {"task_id", "partition", "question_sha256", "group_id", "stratum"}
    if (len(index) != len(flattened) or any(set(row) != public_fields for row in index)
            or {row["task_id"]: row["partition"] for row in index} != expected):
        raise ValueError("Public metadata index differs from partition membership")
    grouped = defaultdict(set)
    for row in index:
        grouped[row["group_id"]].add(row["partition"])
    if any(len(partitions) > 1 for partitions in grouped.values()):
        raise ValueError("Duplicate group crosses partition boundary")
    schedules = document["evolution_schedule"]
    if manifest.evolution:
        if len(schedules) != 3:
            raise ValueError("Require all three registered evolution orders")
        for run_id, schedule in enumerate(schedules):
            ordered = schedule["task_ids"]
            if (schedule["run_id"] != run_id or schedule["order_seed"] != ORDER_SEEDS[run_id]
                    or len(ordered) != len(manifest.evolution) or set(ordered) != set(manifest.evolution)
                    or [key for batch in schedule["batches"] for key in batch] != ordered):
                raise ValueError("Evolution schedule is not a complete registered source permutation")
            positions = [0]
            for batch in schedule["batches"]:
                if not batch:
                    raise ValueError("Source batches cannot be empty")
                positions.append(positions[-1] + len(batch))
            if positions != document["checkpoint_positions"]:
                raise ValueError("Checkpoint positions differ from source batches")
    elif schedules or document["checkpoint_positions"]:
        raise ValueError("A frozen-transfer benchmark cannot have evolution checkpoints")
    if dataset is not None and (dataset.name != document["benchmark"] or dataset.dataset_sha256 != document["dataset_sha256"]
                                or set(flattened) != set(dataset.tasks)):
        raise ValueError("Dataset no longer matches registered split")
    if specification is not None and digest(specification) != document["specification_sha256"]:
        raise ValueError("Benchmark specification differs from frozen split")
    if dataset is not None:
        for row in index:
            meta = dataset.public_metadata[row["task_id"]]
            if any(row[key] != meta[key] for key in ("question_sha256", "group_id")):
                raise ValueError("Public metadata no longer matches frozen dataset")
    return manifest


def cross_benchmark_overlaps(documents):
    groups = defaultdict(list)
    for doc in documents:
        for row in doc["public_index"]:
            groups[row["question_sha256"]].append({"benchmark": doc["benchmark"],
                "task_id": row["task_id"], "partition": row["partition"]})
    return [members for members in groups.values() if len({row["benchmark"] for row in members}) > 1]
