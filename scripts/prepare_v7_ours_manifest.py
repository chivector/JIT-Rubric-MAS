"""Prepare v7 Ours-only stratified manifest without model/API calls."""
from __future__ import annotations
import argparse, copy, json
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jit_mas.experiment_splits import rank
from jit_mas.schemas import digest

SOURCES = ("researchrubrics", "deepsearchqa", "writingbench", "deepresearch_bench_ii")
TARGETS = ("ifeval", "ifbench")
BENCHMARKS = SOURCES + TARGETS
PARTITIONS = ("evolution", "validation", "test")
ALL_PARTITIONS = PARTITIONS + ("unused",)
COUNTS = {
    "researchrubrics": (40, 10, 33), "deepsearchqa": (40, 10, 50),
    "writingbench": (40, 10, 50), "deepresearch_bench_ii": (40, 10, 40),
    "ifeval": (0, 0, 50), "ifbench": (0, 0, 50),
}
SEED = "jit-compose-independent-batch-domain-balanced-v7-ours-only-20261006"
VERSION = "jit-compose-independent-batch-stratified-subset-v7-ours-only"
ORDER_SEED = 20261006


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_balanced(rows, pool, count, seed, *, language=None):
    """Choose exact count by cycling deterministic domain strata."""
    candidates = [r for r in rows if r["task_id"] in set(pool)
                  and (language is None or r.get("language") == language)]
    groups = defaultdict(list)
    for row in candidates:
        groups[row.get("domain") or row.get("selection_stratum") or "all"].append(row)
    for vals in groups.values():
        vals.sort(key=lambda r: rank(seed, r["task_id"]))
    labels = sorted(groups, key=lambda x: rank(seed, x))
    selected = []
    pos = 0
    while len(selected) < count:
        available = [label for label in labels if groups[label]]
        if not available:
            raise ValueError(f"insufficient capacity for {seed}")
        label = available[pos % len(available)]
        selected.append(groups[label].pop(0))
        pos += 1
    return [row["task_id"] for row in selected]


def choose_drbii(rows):
    dr = [copy.deepcopy(r) for r in rows if r["benchmark"] == "deepresearch_bench_ii"]
    test = {r["task_id"] for r in dr if r["partition"] == "test"}
    available = {r["task_id"] for r in dr if r["partition"] == "unused"}
    clean = {r["task_id"] for r in dr if r["partition"] == "unused" and not r.get("known_registered_exposure")}
    val = set()
    for lang in ("en", "zh"):
        val.update(choose_balanced(dr, clean - val, 5,
                                   f"{SEED}:deepresearch_bench_ii:validation:{lang}", language=lang))
    evo = set()
    for lang in ("en", "zh"):
        evo.update(choose_balanced(dr, available - val - evo, 20,
                                   f"{SEED}:deepresearch_bench_ii:evolution:{lang}", language=lang))
    if test & (val | evo) or len(evo) != 40 or len(val) != 10:
        raise ValueError("DRBII partition construction failed")
    membership = {"evolution": sorted(evo, key=lambda x: rank(f"{SEED}:drbii:evolution", x)),
                  "validation": sorted(val, key=lambda x: rank(f"{SEED}:drbii:validation", x)),
                  "test": sorted(test, key=lambda x: rank(f"{SEED}:drbii:test", x))}
    assigned = {k: p for p, ids in membership.items() for k in ids}
    membership["unused"] = sorted(available - evo - val,
                                   key=lambda x: rank(f"{SEED}:drbii:unused", x))
    # Parent v6 already marks every row's current partition. Only DRBII is amended here.
    for row in dr:
        row["partition"] = assigned.get(row["task_id"], "unused")
        row["previous_partition"] = row.get("partition", "unused") if "previous_partition" not in row else row["previous_partition"]
    return membership, {r["task_id"]: r for r in dr}


def audit(rows, memberships):
    out = {}
    for name in BENCHMARKS:
        public = [r for r in rows if r["benchmark"] == name]
        out[name] = {}
        domains = sorted({r.get("domain") or r.get("selection_stratum") or "all" for r in public})
        for part in PARTITIONS:
            selected = [r for r in public if r["partition"] == part]
            available = [r for r in public if r["task_id"] in set(memberships[name][part])]
            dc = Counter(r.get("domain") or r.get("selection_stratum") or "all" for r in selected)
            lc = Counter(r.get("language") for r in selected if r.get("language"))
            out[name][part] = {
                "count": len(selected), "domain_counts": dict(sorted(dc.items())),
                "language_counts": dict(sorted(lc.items())),
                "domain_language_counts": dict(sorted(Counter(f"{r.get('language')}|{r.get('domain') or r.get('selection_stratum') or 'all'}" for r in selected if r.get("language")).items())),
                "available_domain_counts": dict(sorted(Counter(r.get("domain") or r.get("selection_stratum") or "all" for r in available).items())),
                "quota_absolute_deviation": 0,
                "selection_policy": "equal language/domain quotas with deterministic SHA256 cycling" if name == "deepresearch_bench_ii" else "retained v6 stratified membership",
            }
        out[name]["unused"] = {"count": len([r for r in public if r["partition"] == "unused"])}
    return out


def compact(nums):
    nums = sorted(nums)
    if not nums: return "none"
    spans=[]; start=end=nums[0]
    for n in nums[1:]:
        if n == end+1: end=n
        else: spans.append(str(start) if start==end else f"{start}-{end}"); start=end=n
    spans.append(str(start) if start==end else f"{start}-{end}")
    return ", ".join(spans)


def render(doc):
    lines = ["# Independent v7 Ours-only Task Assignments", "", "Four source benchmarks each evolve once from an empty experience state; IFEval and IFBench are migration-only TEST targets.", "Each source uses 40 EVO tasks in 8 batches of 5, fixed 10-task VAL, and its final selected state. Numbers are 1-based rows in the pinned release.", f"Manifest SHA256: `{doc['manifest_sha256']}`", ""]
    index = {r["task_id"]: r for r in doc["task_index"]}
    for name in BENCHMARKS:
        source = doc["benchmarks"][name]
        lines += [f"## {name}", "", f"Revision: `{source['dataset_revision']}`", f"Data SHA256: `{source['dataset_sha256']}`", f"Source: {source['data_url']}", "", "| Domain | EVO | VAL | TEST |", "| --- | ---: | ---: | ---: |"]
        domains = sorted({r.get("domain") or r.get("selection_stratum") or "all" for r in doc["task_index"] if r["benchmark"] == name})
        for dom in domains:
            lines.append(f"| {dom} | {doc['stratification_audit'][name]['evolution']['domain_counts'].get(dom,0)} | {doc['stratification_audit'][name]['validation']['domain_counts'].get(dom,0)} | {doc['stratification_audit'][name]['test']['domain_counts'].get(dom,0)} |")
        lines.append("")
        rows = [r for r in doc["task_index"] if r["benchmark"] == name]
        for part in ALL_PARTITIONS:
            lines += [f"### {part.upper()} ({len(doc['memberships'][name][part])})", "", compact([r["source_row"] for r in rows if r["task_id"] in set(doc['memberships'][name][part])]), ""]
    lines += ["## Single run evolution order", "", "Run 0 (seed 20261006) is round-robin across the four independent source trajectories; source-local order is what defines each batch.", ""]
    tasks = doc["joint_evolution_schedule"][0]["task_ids"]
    for name in SOURCES:
        ids = [k for k in tasks if index[k]["benchmark"] == name]
        lines += [f"### {name}", ""]
        for i in range(0,40,5):
            lines.append(f"Batch {i//5+1}: " + ", ".join(str(index[k]["source_row"]) for k in ids[i:i+5]))
        lines.append("")
    return "\n".join(lines)


def build(base):
    doc = copy.deepcopy(base)
    dr_membership, dr_rows = choose_drbii(doc["task_index"])
    memberships = copy.deepcopy(doc["memberships"])
    memberships["deepresearch_bench_ii"] = dr_membership
    # Update all task rows to agree with memberships. Keep v6 metadata/provenance fields.
    assigned = {(name, key): part for name, parts in memberships.items() for part, ids in parts.items() if part in ALL_PARTITIONS for key in ids}
    for row in doc["task_index"]:
        key=(row["benchmark"],row["task_id"])
        if key in assigned: row["partition"] = assigned[key]
        if row["benchmark"] == "deepresearch_bench_ii":
            row.update(dr_rows[row["task_id"]])
            row["partition"] = assigned[key]
    orders={}
    for name in SOURCES:
        ids = memberships[name]["evolution"]
        orders[name] = sorted(ids, key=lambda x: rank(f"{SEED}:order:{name}", x))
    task_ids=[]
    for i in range(40):
        for name in SOURCES: task_ids.append(orders[name][i])
    counts={name:{part:len(memberships[name][part]) for part in ALL_PARTITIONS} for name in BENCHMARKS}
    totals={part:sum(counts[name][part] for name in BENCHMARKS) for part in ALL_PARTITIONS}
    doc.update({
        "version": VERSION, "status": "STRATIFIED_BATCH_SUBSET_V7_FROZEN_NOT_RUN",
        "membership_seed": SEED, "parent_manifest": "joint_task_splits_v6_stratified.json",
        "parent_manifest_sha256": base["manifest_sha256"], "previous_manifest": "joint_task_splits_v6_stratified.json",
        "previous_manifest_sha256": base["manifest_sha256"], "historical_batch_manifest": "joint_task_splits_v6_stratified.json (unchanged historical registration)",
        "counts": counts, "totals": totals, "memberships": memberships,
        "stratification_audit": audit(doc["task_index"], memberships),
        "joint_evolution_schedule": [{"run_id":0,"order_seed":ORDER_SEED,"task_ids":task_ids}],
        "batch_positions": list(range(5,41,5)), "batch_size":5, "candidates_per_batch":3,
        "allocation": "Four independent source trajectories; DRBII added as an evolved source with equal language and deterministic theme cycling; IFEval/IFBench remain TEST-only targets.",
        "order_policy": "One fixed SHA256 source order (seed 20261006); each source starts from an empty state and splits 40 EVO tasks into eight five-task batches.",
        "batch_policy": "All five tasks read one frozen state; three post-batch candidates; fixed ten-task VAL selects the next state.",
        "exposure_policy": "Known exposed tasks are excluded from VAL/TEST; DRBII's existing 40-task TEST membership is retained; no TEST result is used for selection.",
        "results": [],
    })
    doc["manifest_sha256"] = digest({k: v for k, v in doc.items() if k != "manifest_sha256"})
    return doc


def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=Path('paper/experiments/joint_task_splits_v6_stratified.json')); p.add_argument('--output-dir',type=Path,default=Path('paper/experiments')); p.add_argument('--check',action='store_true'); a=p.parse_args()
    doc=build(load(a.base)); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    files={"joint_task_splits_v7_ours_only.json":json.dumps(doc,indent=2,ensure_ascii=True,allow_nan=False)+'\n', "task_assignments_v7_ours_only.md":render(doc)}
    for name,content in files.items():
        path=out/name
        if a.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content: raise SystemExit(f'Frozen artifact differs: {name}')
        else:
            path.write_text(content,encoding='utf-8',newline='\n')
    print(json.dumps({"manifest_sha256":doc["manifest_sha256"],"counts":doc["counts"],"totals":doc["totals"],"model_calls":0},ensure_ascii=True))

if __name__=='__main__': main()
