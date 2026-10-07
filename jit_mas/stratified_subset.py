"""Public-only domain-balanced amendment of the independent batch inventory."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy

from .experiment_splits import ORDER_SEEDS, _group_subsets, rank
from .joint_protocol import BENCHMARKS, PARTITIONS, SOURCES, compact_rows
from .joint_subset import ALL_PARTITIONS, validate_subset
from .schemas import digest


VERSION = "jit-compose-independent-batch-stratified-subset-v6"
SEED = "jit-compose-independent-batch-domain-balanced-v6-20261006"
RR_DOMAINS_SHA256 = "7f569cc85d8cdcc74f6db6ed7da689115b9c3953f102fb03b8e06b6b4ff759c7"
EXPOSURE_TASKS_SHA256 = "6ed3ab4fb91be41b76a5985414060c6bb455815d9911f69fec9c05f6d3f8032c"
COUNTS = {
    "researchrubrics": (40, 10, 33), "deepsearchqa": (40, 10, 50),
    "writingbench": (40, 10, 50), "deepresearch_bench_ii": (0, 0, 40),
    "ifeval": (0, 0, 50), "ifbench": (0, 0, 50),
}
BATCH_SIZE = 5


def _equal_targets(capacities, count, seed):
    targets = dict.fromkeys(capacities, 0)
    for position in range(count):
        active = [label for label in capacities if targets[label] < capacities[label]]
        if not active:
            raise ValueError("Stratified selection quota exceeds available capacity")
        label = min(active, key=lambda value: (targets[value], rank(seed, value)))
        targets[label] += 1
    return targets


def select_equal_strata(rows, eligible, count, seed, *,
                        stratum_field="selection_stratum", fixed_ids=()):
    """Select equal public-stratum quotas, retaining complete duplicate groups.

    Rare strata are exhausted before their quota is redistributed.  Mandatory
    fixed IDs count toward the full quota.  If exact equal quotas cannot retain
    duplicate groups, minimize total absolute quota deviation at exact size.
    """
    rows = list(rows)
    index = {row["task_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Public task IDs must be unique")
    eligible, fixed = set(eligible), set(fixed_ids)
    if not eligible <= index.keys() or not fixed <= eligible:
        raise ValueError("Stratified selection contains an unknown or ineligible task ID")
    if type(count) is not int or not len(fixed) <= count <= len(eligible):
        raise ValueError("Stratified selection quota is invalid")
    selected_groups = {index[key].get("group_id", key) for key in eligible}
    if any(row.get("group_id", key) in selected_groups and key not in eligible
           for key, row in index.items()):
        raise ValueError("Eligible pool would split a duplicate group")
    grouped = defaultdict(list)
    for task_id in sorted(eligible):
        grouped[index[task_id].get("group_id", task_id)].append(task_id)
    by_stratum = defaultdict(list)
    fixed_counts = Counter()
    for members in grouped.values():
        labels = sorted({str(index[key].get(stratum_field, "all") or "all")
                         for key in members})
        label = "|".join(labels)
        if fixed.intersection(members):
            fixed.update(members)
            fixed_counts[label] += len(members)
        else:
            by_stratum[label].append(tuple(members))
    if len(fixed) > count:
        raise ValueError("Fixed duplicate groups exceed the selection quota")
    capacities = {label: sum(map(len, groups)) + fixed_counts[label]
                  for label, groups in by_stratum.items()}
    capacities.update({label: capacities.get(label, 0) or size
                       for label, size in fixed_counts.items()})
    targets = _equal_targets(capacities, count, seed)
    remaining = count - len(fixed)
    allocations = {0: (0, 0, ())}
    choices = {}
    for label in sorted(capacities, key=lambda value: rank(seed, value)):
        groups = sorted(by_stratum[label], key=lambda group: rank(seed, "|".join(group)))
        choices[label] = _group_subsets(groups, remaining)
        next_allocations = {}
        for allocated, (absolute_cost, squared_cost, path) in allocations.items():
            for take in choices[label]:
                total = allocated + take
                if total > remaining:
                    continue
                deviation = take + fixed_counts[label] - targets[label]
                candidate = (absolute_cost + abs(deviation), squared_cost + deviation ** 2,
                             path + ((label, take),))
                if total not in next_allocations or candidate < next_allocations[total]:
                    next_allocations[total] = candidate
        allocations = next_allocations
    if remaining not in allocations:
        raise ValueError("Exact quotas cannot preserve duplicate groups")
    selected = list(fixed)
    for label, take in allocations[remaining][2]:
        selected.extend(key for group in choices[label][take] for key in group)
    return sorted(selected, key=lambda task_id: rank(seed, task_id))


def _writing_members(rows, eligible, count, seed, fixed_ids=()):
    eligible, fixed = set(eligible), set(fixed_ids)
    index = {row["task_id"]: row for row in rows}
    provisional = select_equal_strata(rows, eligible, count, seed, fixed_ids=fixed)
    quotas = Counter(index[key]["domain"] for key in provisional)
    options = {}
    for domain, quota in quotas.items():
        domain_pool = {key for key in eligible if index[key]["domain"] == domain}
        english = {key for key in domain_pool if index[key]["language"] == "en"}
        chinese = {key for key in domain_pool if index[key]["language"] == "zh"}
        if english | chinese != domain_pool:
            raise ValueError("WritingBench requires pinned en/zh public language labels")
        options[domain] = [take for take in range(quota + 1)
                           if len(fixed & english) <= take <= len(english)
                           and len(fixed & chinese) <= quota - take <= len(chinese)]
    allocations = {0: (0, ())}
    for domain in sorted(quotas, key=lambda value: rank(seed, value)):
        next_allocations = {}
        for total, (cost, path) in allocations.items():
            for english_count in options[domain]:
                candidate = (cost + abs(2 * english_count - quotas[domain]),
                             path + ((domain, english_count),))
                target = total + english_count
                if target not in next_allocations or candidate < next_allocations[target]:
                    next_allocations[target] = candidate
        allocations = next_allocations
    if not allocations:
        raise ValueError("WritingBench language quotas cannot satisfy domain counts")
    english_total = min(allocations, key=lambda total: (
        abs(2 * total - count), allocations[total]))
    selected = []
    for domain, english_count in allocations[english_total][1]:
        for language, quota in (("en", english_count), ("zh", quotas[domain] - english_count)):
            pool = {key for key in eligible if index[key]["domain"] == domain
                    and index[key]["language"] == language}
            selected.extend(select_equal_strata(
                rows, pool, quota, f"{seed}:{domain}:{language}",
                fixed_ids=fixed & pool))
    return sorted(selected, key=lambda task_id: rank(seed, task_id))


def _trusted_metadata(parent, previous, rr_domains, exposure_register):
    validate_subset(previous, parent)
    domains = sorted(rr_domains, key=lambda row: row["task_id"])
    if (any(set(row) != {"task_id", "domain"} for row in domains)
            or digest(domains) != RR_DOMAINS_SHA256):
        raise ValueError("ResearchRubrics public domains differ from the pinned raw release")
    exposures = sorted(
        ({key: row[key] for key in ("benchmark", "task_id", "question_sha256")}
         for row in exposure_register["tasks"]),
        key=lambda row: (row["benchmark"], row["task_id"]))
    if digest(exposures) != EXPOSURE_TASKS_SHA256:
        raise ValueError("Known TEST exposures differ from the pinned public register")
    index = {row["task_id"]: row for row in parent["task_index"]}
    if any(index[row["task_id"]]["question_sha256"] != row["question_sha256"]
           or index[row["task_id"]]["benchmark"] != row["benchmark"] for row in exposures):
        raise ValueError("Exposure identities differ from the parent inventory")
    return domains, exposures


def _audit(rows, pools, policies):
    audit = {}
    for name in BENCHMARKS:
        public = [row for row in rows if row["benchmark"] == name]
        all_domains = {row["domain"] for row in public}
        audit[name] = {}
        for part in PARTITIONS:
            chosen = [row for row in public if row["partition"] == part]
            available = [row for row in public if row["task_id"] in pools[name][part]]
            domain_counts = Counter(row["domain"] for row in chosen)
            available_counts = Counter(row["domain"] for row in available)
            targets = _equal_targets(available_counts, len(chosen), f"{SEED}:{name}:{part}")
            domain_targets = {domain: targets.get(domain, 0) for domain in sorted(all_domains)}
            deviations = {domain: domain_counts[domain] - domain_targets[domain]
                          for domain in sorted(all_domains)}
            uniform_ideal = len(chosen) / len(all_domains) if all_domains else 0.0
            audit[name][part] = {
                "count": len(chosen),
                "domain_counts": dict(sorted(domain_counts.items())),
                "language_counts": dict(sorted(Counter(row["language"] for row in chosen
                                                      if row["language"]).items())),
                "domain_language_counts": dict(sorted(Counter(
                    f"{row['language']}|{row['domain']}" for row in chosen
                    if row["language"]).items())),
                "available_domain_counts": dict(sorted(available_counts.items())),
                "equal_target_domain_counts": domain_targets,
                "domain_quota_deviations": deviations,
                "quota_absolute_deviation": sum(abs(value) for value in deviations.values()),
                "uniform_domain_ideal_count": uniform_ideal,
                "uniform_domain_absolute_deviation": sum(abs(domain_counts[domain] - uniform_ideal)
                                                         for domain in sorted(all_domains)),
                "registered_exposed_task_ids": sorted(row["task_id"] for row in chosen
                                                      if row["known_registered_exposure"]),
                "supplemental_exposed_task_ids": sorted(row["task_id"] for row in chosen
                                                        if row["known_supplemental_exposure"]),
                "unrepresented_available_domains": sorted(set(available_counts) - domain_counts.keys()),
                "unavailable_domains": sorted(all_domains - available_counts.keys()),
                "selection_policy": policies[name][part],
            }
    return audit


def build_stratified_subset(parent, previous, rr_domains, exposure_register):
    """Rebuild domain-balanced memberships using only independently pinned metadata."""
    domains, exposures = _trusted_metadata(parent, previous, rr_domains, exposure_register)
    rr_index = {row["task_id"]: row["domain"] for row in domains}
    drbii_index = {row["task_id"]: row for row in previous["drbii_public_strata"]}
    known_exposures = {row["task_id"] for row in exposures}
    supplemental = {task_id for event in exposure_register.get("supplemental_exposure_events", [])
                    for task_id in event.get("known_additional_legacy_task_ids", [])}
    excluded = known_exposures
    rows = copy.deepcopy(previous["task_index"])
    for row in rows:
        name, task_id = row["benchmark"], row["task_id"]
        row["previous_partition"] = row["partition"]
        row["partition"] = "unused"
        row["language"] = ""
        row["known_registered_exposure"] = task_id in known_exposures
        row["known_supplemental_exposure"] = task_id in supplemental
        if name == "researchrubrics":
            row["domain"] = rr_index[task_id]
        elif name == "writingbench":
            row["language"], row["domain"] = row["stratum"].split("|", 1)
        elif name == "deepresearch_bench_ii":
            row["language"] = drbii_index[task_id]["language"]
            row["domain"] = drbii_index[task_id]["theme"]
        else:
            row["domain"] = row["stratum"]
        row["selection_stratum"] = row["domain"]
    memberships, pools, policies = {}, {}, {}
    for name in BENCHMARKS:
        public = [row for row in rows if row["benchmark"] == name]
        pools[name] = {part: set(parent["memberships"][name][part]) for part in PARTITIONS}
        policies[name] = {part: "equal domain quotas within the original parent partition"
                          for part in PARTITIONS}
        allocations = {}
        for part, count in zip(PARTITIONS, COUNTS[name]):
            pool = pools[name][part]
            if part != "evolution":
                pool.difference_update(excluded)
            seed = f"{SEED}:{name}:{part}"
            if name == "researchrubrics":
                if part == "validation":
                    allocations[part] = select_equal_strata(public, pool, count, seed)
                elif part == "evolution":
                    continue
                else:
                    pool.intersection_update(previous["memberships"][name]["test"])
                    allocations[part] = sorted(pool, key=lambda task_id: rank(seed, task_id))
                    policies[name][part] = "retain all 33 historically designated clean-unseen parent TEST tasks"
            elif name == "writingbench":
                fixed = {entry["task_id"] for entry in parent["benchmarks"][name]["known_exposures"]}
                allocations[part] = _writing_members(public, pool, count, seed,
                                                     fixed if part == "evolution" else ())
                policies[name][part] = "equal primary-domain quotas; balance language within each domain and over the full subset"
            elif name == "deepresearch_bench_ii" and part == "test":
                allocations[part] = []
                for language in ("en", "zh"):
                    language_pool = {row["task_id"] for row in public
                                     if row["task_id"] in pool and row["language"] == language}
                    allocations[part].extend(select_equal_strata(
                        public, language_pool, 20, f"{seed}:{language}"))
                allocations[part].sort(key=lambda task_id: rank(seed, task_id))
                policies[name][part] = "20 en and 20 zh; equal theme quotas within each language after known-exposure exclusion"
            else:
                allocations[part] = select_equal_strata(public, pool, count, seed)
                if name in ("ifeval", "ifbench"):
                    policies[name][part] = "no public domain labels; deterministic sampling after known-exposure exclusion"
        if name == "researchrubrics":
            transferred = pools[name]["validation"] - set(allocations["validation"])
            pool = pools[name]["evolution"] | transferred
            pools[name]["evolution"] = pool
            allocations["evolution"] = sorted(pool, key=lambda task_id: rank(
                f"{SEED}:{name}:evolution", task_id))
            policies[name]["evolution"] = "retain 30 original parent EVO tasks and transfer the 10 parent VAL tasks not selected for VAL; no EVO-to-VAL transfer"
        assigned = {key: part for part, keys in allocations.items() for key in keys}
        allocations["unused"] = sorted(
            (row["task_id"] for row in public if row["task_id"] not in assigned),
            key=lambda task_id: rank(f"{SEED}:{name}:unused", task_id))
        for row in public:
            row["partition"] = assigned.get(row["task_id"], "unused")
        if any(len(allocations[part]) != count for part, count in zip(PARTITIONS, COUNTS[name])):
            actual = {part: len(allocations[part]) for part in PARTITIONS}
            raise ValueError(f"Balanced amendment cannot satisfy {name} partition counts: {actual}")
        memberships[name] = allocations
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row["benchmark"], row["group_id"])].add(row["partition"])
    if any(len(parts) > 1 for parts in grouped.values()):
        raise ValueError("A duplicate-question group crosses amended partitions")
    if excluded.intersection(key for name in BENCHMARKS for part in ("validation", "test")
                             for key in memberships[name][part]):
        raise ValueError("A known exposed task enters VAL or TEST")
    schedules = []
    for run_id, order_seed in enumerate(ORDER_SEEDS):
        ordered = {name: sorted(memberships[name]["evolution"], key=lambda task_id: rank(
            f"{SEED}:order:{order_seed}:{name}", task_id)) for name in SOURCES}
        schedules.append({"run_id": run_id, "order_seed": order_seed,
                          "task_ids": [ordered[name][position]
                                       for position in range(40) for name in SOURCES]})
    document = {
        "version": VERSION, "status": "STRATIFIED_BATCH_SUBSET_FROZEN_NOT_RUN",
        "membership_seed": SEED, "parent_manifest": "joint_task_splits_v4.json",
        "parent_manifest_sha256": parent["manifest_sha256"],
        "previous_manifest": "joint_task_splits_v5.json",
        "previous_manifest_sha256": previous["manifest_sha256"],
        "historical_batch_manifest": "joint_task_splits_v6.json (unchanged historical registration)",
        "benchmarks": copy.deepcopy(previous["benchmarks"]),
        "counts": {name: {part: len(memberships[name][part]) for part in ALL_PARTITIONS}
                   for name in BENCHMARKS},
        "totals": {part: sum(len(memberships[name][part]) for name in BENCHMARKS)
                   for part in ALL_PARTITIONS},
        "memberships": memberships, "task_index": rows,
        "researchrubrics_public_domains": domains,
        "researchrubrics_public_domains_sha256": RR_DOMAINS_SHA256,
        "drbii_public_strata": copy.deepcopy(previous["drbii_public_strata"]),
        "drbii_public_strata_sha256": previous["drbii_public_strata_sha256"],
        "known_test_exposures": exposures, "known_test_exposures_sha256": EXPOSURE_TASKS_SHA256,
        "exposure_register": "test_exposure_20261004.json",
        "exposure_register_sha256": digest(exposure_register),
        "supplemental_known_exposed_task_ids": sorted(supplemental),
        "stratification_audit": _audit(rows, pools, policies),
        "joint_evolution_schedule": schedules, "batch_positions": list(range(5, 41, 5)),
        "batch_size": BATCH_SIZE, "candidates_per_batch": 3,
        "allocation": "Equal public primary-domain quotas with rare-stratum capacity redistribution; intact exact-question groups and deterministic SHA256 ties. RR retains historical EVO/TEST boundaries; sources never transfer old EVO into VAL or TEST.",
        "order_policy": "Three fixed SHA256 source orders; each source/run starts from zero and splits its 40 tasks into eight five-task batches",
        "batch_policy": "All five tasks read one frozen state; three post-batch candidates; fixed ten-task VAL selects the next state",
        "unused_policy": previous["unused_policy"],
        "exposure_policy": "Exclude the five explicitly registered target exposures from VAL/TEST; all 18 former RR exposed/reserved TEST rows remain unused; WritingBench row 1 stays EVO. Retain the fixed 33 historically designated clean RR TEST tasks, including the explicitly marked supplemental exposure. Historical incidental exposure range may be unknown, so this is not a claim that all retained historical TEST tasks are unexposed.",
        "row_alias_warning": previous["row_alias_warning"],
        "overlap_scope": previous["overlap_scope"], "results": [],
    }
    document["manifest_sha256"] = digest(document)
    return document


def validate_stratified_subset(document, parent, previous, exposure_register):
    content = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("version") != VERSION or digest(content) != document.get("manifest_sha256"):
        raise ValueError("Stratified batch subset manifest integrity check failed")
    expected = build_stratified_subset(parent, previous,
                                       document["researchrubrics_public_domains"], exposure_register)
    if document != expected:
        raise ValueError("Stratified subset differs from pinned membership, audit or order")
    return document


def render_assignments(document):
    lines = ["# Independent v6 Domain-Balanced Task Assignments", "",
             "This is a new registration; the historical v5/v6 manifests and campaigns remain unchanged.",
             "Numbers are 1-based data-record positions in the pinned source file; CSV header excluded.",
             "Each subset balances available primary domains. Rare or unavailable domains are reported below.",
             "ResearchRubrics retains 30 parent EVO and 33 historical clean-designated TEST tasks; it balances ten VAL choices within the 20 parent VAL tasks.",
             "WritingBench balances six primary domains, then en/zh inside each domain and over the subset.",
             "DeepResearch Bench II balances themes separately within 20 en and 20 zh tasks.",
             "Five registered target exposures are excluded from VAL/TEST. Fixed historical RR TEST retains a marked supplemental exposure; unknown historical exposure remains an audit limitation.",
             f"Manifest SHA256: `{document['manifest_sha256']}`", ""]
    for name in BENCHMARKS:
        source = document["benchmarks"][name]
        rows = [row for row in document["task_index"] if row["benchmark"] == name]
        lines += [f"## {name}", "", f"Revision: `{source['dataset_revision']}`",
                  f"Data SHA256: `{source['dataset_sha256']}`", f"Source: {source['data_url']}", ""]
        domains = sorted({row["domain"] for row in rows})
        lines += ["### Domain Distribution", "", "| Domain | EVO | VAL | TEST |",
                  "| --- | ---: | ---: | ---: |"]
        for domain in domains:
            counts = [document["stratification_audit"][name][part]["domain_counts"].get(domain, 0)
                      for part in PARTITIONS]
            lines.append(f"| {domain} | {counts[0]} | {counts[1]} | {counts[2]} |")
        lines.append("")
        for part in ALL_PARTITIONS:
            selected = [row for row in rows if row["partition"] == part]
            lines += [f"### {part.upper()} ({len(selected)})", "",
                      compact_rows([row["source_row"] for row in selected]), ""]
            if part in PARTITIONS:
                audit = document["stratification_audit"][name][part]
                lines += [f"Policy: {audit['selection_policy']}", ""]
                if audit["language_counts"]:
                    lines += ["Languages: " + ", ".join(f"{language}={count}"
                              for language, count in audit["language_counts"].items()), ""]
                lines += [f"Absolute deviation from capacity-aware available-domain quotas: {audit['quota_absolute_deviation']}", "",
                          f"Unconstrained ideal over all release domains: {audit['uniform_domain_ideal_count']:.3f} tasks/domain; absolute deviation: {audit['uniform_domain_absolute_deviation']:.3f}", ""]
                if audit["supplemental_exposed_task_ids"]:
                    lines += ["Retained historical supplemental exposure IDs: " + ", ".join(
                        audit["supplemental_exposed_task_ids"]), ""]
                if audit["unrepresented_available_domains"]:
                    lines += ["Available but unrepresented: " + "; ".join(
                        audit["unrepresented_available_domains"]), ""]
                if audit["unavailable_domains"] and selected:
                    lines += ["Absent from eligible parent pool: " + "; ".join(
                        audit["unavailable_domains"]), ""]
    index = {row["task_id"]: row for row in document["task_index"]}
    lines += ["## Independent Evolution Batches", "",
              "Each source/run starts from zero; five tasks share one state before three candidates are produced.", ""]
    for run in document["joint_evolution_schedule"]:
        lines += [f"### Run {run['run_id']} (seed {run['order_seed']})", ""]
        for name in SOURCES:
            tasks = [key for key in run["task_ids"] if index[key]["benchmark"] == name]
            lines += [f"#### {name}", ""]
            for batch_index, start in enumerate(range(0, len(tasks), BATCH_SIZE), 1):
                numbers = ", ".join(str(index[key]["source_row"])
                                    for key in tasks[start:start + BATCH_SIZE])
                lines += [f"Batch {batch_index}: {numbers}", ""]
    return "\n".join(lines)
