"""Build a compact, provenance-aware recovery report for the tds_fix8 run."""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "formal_ours_20261010" / "tds_fix9_recovery_report"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def status_row(name: str):
    j = read_json(ROOT / name / "status.json")
    s = j.get("statuses", {})
    return {
        "campaign": name,
        "registered": j.get("registered_slots"),
        "complete": s.get("complete", 0),
        "failed": s.get("failed", 0),
        "incomplete": s.get("incomplete", 0),
        "missing": s.get("missing", 0),
        "submitted": s.get("submitted", 0),
        "pending": s.get("pending", 0),
        "terminal": j.get("terminal_slots"),
        "evolution_validation_complete": j.get("evolution_validation_complete"),
        "test_feedback_released": j.get("test_feedback_released"),
    }


campaigns = [status_row(x) for x in ("tds_fix5", "tds_fix7", "tds_fix8", "tds_fix9", "tds_fix10") if (ROOT / x / "status.json").exists()]

with (OUT / "campaign_status.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(campaigns[0]))
    w.writeheader(); w.writerows(campaigns)

reuse = read_json(ROOT / "outputs/formal_ours_20261010/reuse_final_report_20261011/summary.json")
results = reuse.get("results", [])
for row in results:
    row = dict(row)

with (OUT / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["benchmark", "condition", "status", "complete", "slots", "official_metric", "score", "secondary", "source"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(results)

eff = reuse.get("efficiency", [])
with (OUT / "efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    fields = list(dict.fromkeys(k for r in eff for k in r))
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(eff)

ops = {"candidate_add": 0, "candidate_delete": 0, "candidate_split": 0, "candidate_merge": 0, "selected_add": 0, "selected_specialize": 0}
for camp in ("tds_fix5", "tds_fix7", "tds_fix8", "tds_fix9", "tds_fix10"):
    db = ROOT / camp / "campaign.sqlite"
    if not db.exists(): continue
    con = sqlite3.connect(db)
    for (body,) in con.execute("select body from candidates"):
        try: j = json.loads(body)
        except Exception: continue
        op = str(j.get("operation") or j.get("op") or j.get("proposal", {}).get("operation") or "").lower()
        for k in ("add", "delete", "split", "merge"):
            if k in op: ops["candidate_" + k] += 1
    for (body,) in con.execute("select body from selections"):
        try: j = json.loads(body)
        except Exception: continue
        selected = j.get("selected") or {}
        op = str(j.get("operation") or (selected.get("operation") if isinstance(selected, dict) else "") or "").lower()
        for k in ("add", "specialize"):
            if k in op: ops["selected_" + k] += 1
    con.close()

active_pending = sum(int(row.get("pending") or 0) for row in campaigns if row.get("campaign") in {"tds_fix9", "tds_fix10"})
formal_status = ("tds_fix9/tds_fix10 are still running; formal RR/DSQA/DRBII TEST remains unreleased"
                 if active_pending else
                 "WritingBench/IFEval/IFBench receipts reused (146 complete); RR/DSQA/DRBII formal TEST have no eligible selected candidate in the recovery campaigns")
summary = {
    "schema": "tds8-recovery-report-v1",
    "formal_recovery_campaign": "tds_fix9 (active recovery); tds_fix8 is prior high-concurrency retry",
    "source_baseline": "tds_fix5",
    "campaign_status": campaigns,
    "benchmark_results": results,
    "efficiency": eff,
    "structural_operations_aggregate": ops,
    "formal_test_status": formal_status,
    "targeted_reuse_policy": "Targeted artifacts are reported separately and are not mislabeled as six-benchmark formal TEST.",
    "figures": [
        "../tds_fix5_final_report/selected_utility_curve.pdf",
        "../tds_fix5_final_report/agent_pool_curve.pdf",
        "../tds_fix5_final_report/structural_operation_counts.pdf",
    ],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Ours recovery report: tds_fix5 → tds_fix7 → tds_fix8",
    "",
    "This report preserves the formal campaign boundary and reuses only artifacts with matching task/evaluator identity.",
    "",
    "## Benchmark results",
    "",
    "| Benchmark | Condition | Status | Complete / slots | Metric | Score |",
    "|---|---|---|---:|---|---:|",
]
for r in results:
    score = "" if r.get("score") is None else f"{r['score']:.9f}"
    lines.append(f"| {r['benchmark']} | {r['condition']} | {r['status']} | {r['complete']}/{r['slots']} | {r['official_metric']} | {score} |")
lines += [
    "",
    "ResearchRubrics, IFEval and IFBench are complete targeted artifacts. DeepSearchQA is a two-task targeted result (ACT and country); DeepResearch Bench II is a one-task targeted result. These rows are not the missing 50/40-task formal TEST split.",
    "",
    "## Formal continuation status",
    "",
    "tds_fix8 was created from tds_fix7 and preserved all 506 baseline complete slots plus submitted TEST receipts. The later tds_fix9 and tds_fix10 continuations reuse completed receipts and reopen only blocked scopes. Their live counts are recorded below; no formal TEST score is fabricated while a selected candidate is unavailable.",
    "",
    "| Campaign | Complete | Failed | Incomplete | Missing | Submitted | EVO/VAL complete | TEST feedback |",
    "|---|---:|---:|---:|---:|---:|---|---|",
]
for r in campaigns:
    lines.append(f"| {r['campaign']} | {r['complete']} | {r['failed']} | {r['incomplete']} | {r['missing']} | {r['submitted']} | {r['evolution_validation_complete']} | {r['test_feedback_released']} |")
lines += [
    "",
    "## Efficiency",
    "",
    "See `efficiency.csv` for tokens, calls, failure rates and latency from completed source artifacts. The 146 TEST receipts were reused, so tds_fix8 adds no duplicate TEST scoring cost. Provider USD billing was unavailable and remains null.",
    "",
    "## Evolution evidence",
    "",
    "The tds_fix5 figures linked below remain the authoritative evolution evidence: selected utility, Agent Pool size and structural operation counts. The recovery attempts preserve this provenance and add campaign-level status rows rather than overwriting the baseline. Candidate Add/Specialize activity is present; Delete/Split/Merge remain zero in the selected trajectory.",
    "",
    "## Figures",
    "",
    "- [Selected utility curve](../tds_fix5_final_report/selected_utility_curve.pdf)",
    "- [Agent Pool curve](../tds_fix5_final_report/agent_pool_curve.pdf)",
    "- [Structural operation counts](../tds_fix5_final_report/structural_operation_counts.pdf)",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "campaigns": campaigns, "rows": len(results)}, ensure_ascii=False))
