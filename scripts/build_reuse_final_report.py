"""Build a provenance-aware benchmark report from already completed artifacts.

This report deliberately keeps targeted and formal-continuation results in separate
rows. It never treats a one-task or targeted result as a full independent v7 run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "formal_ours_20261010" / "reuse_final_report_20261011"
OUT.mkdir(parents=True, exist_ok=True)


RESULTS = [
    {"benchmark": "ResearchRubrics", "condition": "TEST33", "status": "reused_targeted_full_split", "complete": 33, "slots": 33, "official_metric": "compliance", "score": 0.8946257078812102, "source": "outputs/targeted_optimization_20261010/researchrubrics_test33_recovered"},
    {"benchmark": "IFEval", "condition": "TEST40", "status": "reused_targeted_full_split", "complete": 40, "slots": 40, "official_metric": "prompt_strict_accuracy", "score": 0.975, "secondary": "instruction_strict_accuracy=0.9841269841; prompt_loose_accuracy=0.975; instruction_loose_accuracy=0.9841269841", "source": "outputs/targeted_optimization_20261010/ifeval_v11"},
    {"benchmark": "IFBench", "condition": "TEST40", "status": "reused_targeted_full_split", "complete": 40, "slots": 40, "official_metric": "prompt/instruction strict+loose", "score": 1.0, "secondary": "all four reported accuracies=1.0", "source": "outputs/targeted_optimization_20261010/ifbench_v11_final"},
    {"benchmark": "DeepSearchQA", "condition": "targeted ACT + country", "status": "reused_targeted_two_tasks", "complete": 2, "slots": 2, "official_metric": "Precision/Recall/F1", "score": 1.0, "secondary": "ACT=1.0/1.0/1.0; country=1.0/1.0/1.0; not the 50-task v7 source TEST", "source": "outputs/targeted_optimization_20261010/research_v5_enhanced + research_v6_focused"},
    {"benchmark": "DeepResearch Bench II", "condition": "targeted one task", "status": "reused_targeted_one_task", "complete": 1, "slots": 1, "official_metric": "Overall", "score": 0.8620689655172413, "secondary": "InformationRecall=0.7307692308; Analysis=1.0; Presentation=0.8333333333; not the 40-task v7 source TEST", "source": "outputs/targeted_optimization_20261010/research_v5_enhanced"},
    {"benchmark": "WritingBench", "condition": "tds_fix5 source TEST", "status": "formal_partial_continuation", "complete": 49, "slots": 50, "official_metric": "normalized score", "score": 0.8040816326530612, "secondary": "native author-checklist mean normalized per task; 1 failed", "source": "tds_fix5/test_report.json"},
    {"benchmark": "WritingBench", "condition": "tds_fix5 -> IFEval", "status": "formal_partial_continuation", "complete": 50, "slots": 50, "official_metric": "IFEval accuracy", "score": 0.9, "secondary": "source=writingbench", "source": "tds_fix5/test_report.json"},
    {"benchmark": "WritingBench", "condition": "tds_fix5 -> IFBench", "status": "formal_partial_continuation", "complete": 47, "slots": 50, "official_metric": "IFBench accuracy", "score": 0.48936170212765956, "secondary": "3 failed/incomplete", "source": "tds_fix5/test_report.json"},
]

EFFICIENCY = [
    {"benchmark": "ResearchRubrics", "condition": "TEST33", "tokens": 5039652, "model_calls": 866, "api_attempts": 866, "failed_api_attempts": 7, "api_failure_rate": 0.008083140877598153, "latency_mean_s": 70.0748181818159, "latency_p95_s": 131.843, "cost_usd": None, "source": "outputs/benchmark_dashboard_20261010/efficiency.csv"},
    {"benchmark": "IFEval", "condition": "TEST40", "tokens": 18427, "model_calls": 41, "api_attempts": 41, "failed_api_attempts": 0, "api_failure_rate": 0.0, "latency_mean_s": 2.11335, "latency_p95_s": 4.687, "cost_usd": None, "source": "outputs/benchmark_dashboard_20261010/efficiency.csv"},
    {"benchmark": "IFBench", "condition": "TEST40", "tokens": 34348, "model_calls": 42, "api_attempts": 42, "failed_api_attempts": 0, "api_failure_rate": 0.0, "latency_mean_s": 2.51645, "latency_p95_s": 7.453, "cost_usd": None, "source": "outputs/benchmark_dashboard_20261010/efficiency.csv"},
    {"benchmark": "DeepSearchQA", "condition": "targeted ACT + country", "tokens": 15469, "model_calls": 3, "api_attempts": 3, "failed_api_attempts": 0, "api_failure_rate": 0.0, "latency_mean_s": None, "latency_p95_s": None, "cost_usd": None, "source": "outputs/benchmark_dashboard_20261010/efficiency.csv"},
    {"benchmark": "DeepResearch Bench II", "condition": "targeted one task", "tokens": 30398, "model_calls": 3, "api_attempts": 3, "failed_api_attempts": 0, "api_failure_rate": 0.0, "latency_mean_s": 63.656, "latency_p95_s": 63.656, "cost_usd": None, "source": "outputs/benchmark_dashboard_20261010/efficiency.csv"},
    {"benchmark": "WritingBench", "condition": "tds_fix5 TEST all attempted", "tokens": 15051823, "model_calls": 1137, "api_attempts": 1137, "failed_api_attempts": None, "api_failure_rate": None, "latency_mean_s": None, "latency_p95_s": None, "cost_usd": None, "source": "tds_fix5/test_report.json"},
]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


write_csv(OUT / "benchmark_results.csv", RESULTS)
write_csv(OUT / "efficiency.csv", EFFICIENCY)

summary = {
    "schema": "reuse-final-report-v1",
    "generated_from_existing_artifacts": True,
    "full_v7_status": "not_complete",
    "policy": "Reuse only when task set, evaluator, and artifact identity are explicit; targeted and formal-continuation rows are not pooled.",
    "results": RESULTS,
    "efficiency": EFFICIENCY,
    "formal_partial_source": "tds_fix5",
    "formal_partial_status": {"evo_complete": 76, "val_complete": 413, "test_submitted": 146, "test_registered": 573},
    "structural_operations": {"candidate_add": 3, "selected_add": 1, "candidate_delete": 0, "candidate_split": 0, "candidate_merge": 0},
    "cost_policy": "Provider billing was not returned; USD cost remains null.",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Ours benchmark results assembled from reusable artifacts",
    "",
    "This is a provenance-aware consolidation of completed artifacts. It is not a claim that the independent v7 six-benchmark campaign is complete.",
    "",
    "## Results",
    "",
    "| Benchmark | Condition | Status | Complete / slots | Metric | Score |",
    "|---|---|---|---:|---|---:|",
]
for row in RESULTS:
    score = "" if row["score"] is None else f'{row["score"]:.9f}'
    lines.append(f'| {row["benchmark"]} | {row["condition"]} | {row["status"]} | {row["complete"]}/{row["slots"]} | {row["official_metric"]} | {score} |')
lines += [
    "",
    "ResearchRubrics, IFEval, and IFBench rows come from completed targeted artifacts. DeepSearchQA and DeepResearch Bench II rows are one- or two-task targeted results. WritingBench rows include the formal tds_fix5 continuation, whose full source TEST is 49/50 and whose migrated IFEval/IFBench conditions are 50/50 and 47/50.",
    "",
    "## Formal continuation coverage",
    "",
    "tds_fix5 registered 1725 slots: EVO complete 76, VAL complete 413, and TEST submitted 146/573. It is therefore a partial continuation, not a full six-benchmark result. Its selected structural operation totals are Add=1 and Specialize=3; Delete/Split/Merge were zero in the selected trajectory.",
    "",
    "## Efficiency and cost",
    "",
    "See `efficiency.csv` for tokens, calls, API attempts, failure rates, and latency where the source artifact recorded them. Provider billing was not returned, so USD cost is null. Existing targeted artifacts are not pooled with tds_fix5 token totals.",
    "",
    "## Provenance",
    "",
    "- `tds_fix5_final_report/`: formal partial continuation and evolution evidence.",
    "- `targeted_optimization_20261010/`: completed targeted benchmark artifacts reused above.",
    "- `benchmark_dashboard_20261010/`: efficiency and official metric aggregation for targeted artifacts.",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "rows": len(RESULTS), "efficiency_rows": len(EFFICIENCY)}, ensure_ascii=False))
