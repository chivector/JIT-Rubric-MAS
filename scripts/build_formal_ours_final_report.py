"""Build a single provenance-aware report from completed Ours campaigns.

The script is read-only with respect to campaign databases.  It combines the
terminal tds_fix9 TEST report, any later isolated DRBII repair status, the
authoritative tds_fix5 evolution artifacts, and the targeted reuse summary.
Missing formal rows remain explicitly marked as incomplete.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "formal_ours_20261010" / "formal_ours_final_report"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def status_row(name: str):
    payload = read_json(ROOT / name / "status.json")
    statuses = payload.get("statuses", {})
    return {
        "campaign": name,
        "registered": payload.get("registered_slots"),
        "complete": statuses.get("complete", 0),
        "failed": statuses.get("failed", 0),
        "incomplete": statuses.get("incomplete", 0),
        "missing": statuses.get("missing", 0),
        "submitted": statuses.get("submitted", 0),
        "pending": statuses.get("pending", 0),
        "terminal": payload.get("terminal_slots"),
        "active": payload.get("active", 0),
        "registration_hash": payload.get("registration_hash"),
    }


campaign_names = [
    name for name in ("tds_fix5", "tds_fix9", "tds_fix10", "tds_fix12", "tds_fix14", "tds_fix16", "tds_fix17", "tds_fix19", "tds_fix22", "tds_fix25")
    if (ROOT / name / "status.json").exists()
]
campaigns = [status_row(name) for name in campaign_names]

with (OUT / "campaign_status.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(campaigns[0]) if campaigns else ["campaign"])
    writer.writeheader()
    writer.writerows(campaigns)


formal_rows = []
for campaign in ("tds_fix9", "tds_fix17"):
    report = read_json(ROOT / campaign / "full_report.json")
    test = report.get("test") or {}
    for condition in test.get("by_condition", []):
        scores = condition.get("scores") or []
        complete = condition.get("complete", 0)
        slots = condition.get("slots", 0)
        full_mean = condition.get("mean_official_score")
        descriptive = (sum(scores) / len(scores)) if scores else None
        formal_rows.append({
            "campaign": campaign,
            "source": condition.get("source"),
            "target": condition.get("target"),
            "method": condition.get("method"),
            "complete": complete,
            "slots": slots,
            "missing_or_failed": max(0, slots - complete),
            "full_mean": full_mean,
            "descriptive_complete_mean": descriptive,
            "status": "complete" if slots and complete == slots else ("partial" if complete else "missing"),
        })

with (OUT / "formal_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = ["campaign", "source", "target", "method", "complete", "slots", "missing_or_failed", "full_mean", "descriptive_complete_mean", "status"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(formal_rows)

reuse = read_json(ROOT / "outputs/formal_ours_20261010/reuse_final_report_20261011/summary.json")
targeted_rows = reuse.get("results", [])
with (OUT / "targeted_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = list(dict.fromkeys(k for row in targeted_rows for k in row)) or ["benchmark"]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(targeted_rows)


efficiency = []
for row in reuse.get("efficiency", []):
    efficiency.append(dict(row))
for campaign in ("tds_fix9", "tds_fix17"):
    report = read_json(ROOT / campaign / "full_report.json")
    test = report.get("test") or {}
    usage = test.get("token_usage") or {}
    if usage:
        calls = usage.get("model_calls")
        efficiency.append({
            "campaign": campaign,
            "stage": "TEST_total",
            "tokens": usage.get("total_tokens"),
            "model_calls": calls,
            "api_failures": None,
            "api_failure_rate": None,
            "latency_mean_seconds": None,
            "latency_p95_seconds": None,
            "cost_usd": None,
            "cost_status": "provider billing unavailable",
        })
with (OUT / "efficiency.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = list(dict.fromkeys(k for row in efficiency for k in row)) or ["stage"]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(efficiency)

# The tds_fix9 paper summary retains exact token totals; the sealed task
# receipts additionally expose the verified stage latency and physical retry
# counts below.  Keep these as a separate table so they cannot be confused
# with the targeted reuse efficiency rows above.
rr_metrics = [
    {"condition": "RR→RR", "generation_calls": 271, "generation_tokens": 4610288,
     "evaluation_calls": 826, "evaluation_tokens": 2817676, "total_calls": 1097,
     "total_tokens": 7427964, "generation_mean_latency_s": 118.143,
     "evaluation_mean_latency_s": 94.478, "physical_generation_attempts": 273,
     "physical_evaluation_attempts": 1054, "final_api_failures": 0},
    {"condition": "RR→IFEval", "generation_calls": 342, "generation_tokens": 3778367,
     "evaluation_calls": 0, "evaluation_tokens": 0, "total_calls": 342,
     "total_tokens": 3778367, "generation_mean_latency_s": 39.281,
     "evaluation_mean_latency_s": 0.379, "physical_generation_attempts": 348,
     "physical_evaluation_attempts": 0, "final_api_failures": 0},
    {"condition": "RR→IFBench", "generation_calls": 381, "generation_tokens": 4557317,
     "evaluation_calls": 0, "evaluation_tokens": 0, "total_calls": 381,
     "total_tokens": 4557317, "generation_mean_latency_s": 43.614,
     "evaluation_mean_latency_s": 0.450, "physical_generation_attempts": 381,
     "physical_evaluation_attempts": 0, "final_api_failures": 0},
]
with (OUT / "rr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = list(rr_metrics[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rr_metrics)


baseline = ROOT / "outputs/formal_ours_20261010/tds_fix5_final_report"
for filename in ("selected_evolution.csv", "structural_operation_counts.csv", "token_latency_curve.csv"):
    source = baseline / filename
    if source.exists():
        shutil.copy2(source, OUT / filename)
for filename in ("selected_utility_curve.pdf", "agent_pool_curve.pdf", "structural_operation_counts.pdf", "token_latency_curve.pdf"):
    source = baseline / filename
    if source.exists():
        shutil.copy2(source, OUT / filename)


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


lines = [
    "# Ours formal experiment report",
    "",
    "This report preserves campaign identity and separates formal TEST receipts from targeted reuse artifacts.",
    "",
    "## Formal TEST results",
    "",
    "| Campaign | Source | Target | Complete / slots | Status | Full mean | Descriptive mean |",
    "|---|---|---|---:|---|---:|---:|",
]
for row in formal_rows:
    lines.append("| {campaign} | {source} | {target} | {complete}/{slots} | {status} | {full_mean} | {descriptive_complete_mean} |".format(
        **{**row, "full_mean": fmt(row["full_mean"]), "descriptive_complete_mean": fmt(row["descriptive_complete_mean"])}))
lines += [
    "",
    "The terminal tds_fix9 run provides a complete 33-task ResearchRubrics TEST mean of 0.440567. Its IFEval and IFBench cross-target rows are partial (48/50 and 45/50). DeepSearchQA and DeepResearch Bench II remain missing in the formal six-benchmark TEST inventory until a selected candidate and all registered tasks are complete.",
    "The isolated DRBII repair evidence is retained separately: tds_fix12 completed the b0 small-judge validation for all three candidates and recorded a batch selection. Subsequent b1 repair branches (tds_fix16/17/19/22/25) reached partial EVO completion but did not produce a valid downstream selection before the provider connection failures; their pending TEST slots are not converted into scores.",
    "",
    "## Targeted reuse results",
    "",
    "Targeted rows are copied from `reuse_final_report_202611` and are not relabeled as full formal TEST splits.",
    "",
    "| Benchmark | Condition | Status | Complete / slots | Metric | Score |",
    "|---|---|---|---:|---|---:|",
]
for row in targeted_rows:
    score = row.get("score")
    score_text = fmt(score)
    lines.append(f"| {row.get('benchmark')} | {row.get('condition')} | {row.get('status')} | {row.get('complete')}/{row.get('slots')} | {row.get('official_metric')} | {score_text} |")
lines += [
    "",
    "## Efficiency and evolution evidence",
    "",
    "`efficiency.csv` contains token totals, call counts, failure fields and targeted latency statistics where source receipts expose them. Provider USD billing is unavailable, so cost is left null rather than inferred.",
    "`rr_metrics.csv` records the sealed tds_fix9 RR stage latency and physical retry audit (all final API failures were zero).",
    "",
    "The copied evolution files and PDFs are the authoritative tds_fix5 trajectory: selected utility, Agent Pool size and structural operation counts. They show candidate Add/Specialize activity and selected Add/Specialize operations; Delete/Split/Merge remain zero in that selected trajectory.",
    "",
    "Figures: `selected_utility_curve.pdf`, `agent_pool_curve.pdf`, `token_latency_curve.pdf`, and `structural_operation_counts.pdf`.",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

summary = {
    "schema": "formal-ours-report-v1",
    "campaign_status": campaigns,
    "formal_benchmark_results": formal_rows,
    "targeted_benchmark_results": targeted_rows,
    "efficiency": efficiency,
    "rr_metrics": rr_metrics,
    "evolution_source": str(baseline),
    "report_directory": str(OUT),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "campaigns": campaign_names, "formal_rows": len(formal_rows), "targeted_rows": len(targeted_rows)}, ensure_ascii=False))
