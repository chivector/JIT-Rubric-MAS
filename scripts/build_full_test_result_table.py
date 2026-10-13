"""Build a two-column status table from durable full-test receipts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "full_test_results_20261012"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def summarize(manifest_path: Path, records_path: Path, expected: int) -> dict:
    manifest = read_json(manifest_path)
    rows = read_records(records_path)
    complete_rows = [
        row for row in rows
        if row.get("status") == "complete"
        and isinstance((row.get("score_result") or {}).get("score"), (int, float))
    ]
    scores = [row["score_result"]["score"] for row in complete_rows]
    counts = {"failed": 0, "incomplete": 0}
    for row in rows:
        if row.get("status") in counts:
            counts[row["status"]] += 1
    return {
        "expected": expected,
        "records": len(rows),
        "complete_scored": len(scores),
        "failed": counts["failed"],
        "incomplete": counts["incomplete"],
        "mean_score_complete": sum(scores) / len(scores) if scores else None,
        "fresh_complete": manifest.get("complete_tasks", 0),
        "fresh_total": manifest.get("total_tasks", 0),
        "fresh_failed_shards": manifest.get("failed_shards", 0),
    }


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def fresh_status(summary: dict) -> str:
    if summary["fresh_complete"] >= summary["expected"]:
        return f"Fresh full TEST complete: {summary['fresh_complete']}/{summary['expected']} tasks."
    if summary["fresh_total"]:
        return f"Fresh full TEST partial: {summary['fresh_complete']}/{summary['expected']} tasks."
    preflight = OUT / "endpoint_preflight_20261012.json"
    if preflight.exists():
        probe = read_json(preflight)
        if probe.get("status") == "offline":
            return f"Fresh full TEST not started: gateway preflight returned {probe.get('error', 'offline')} (endpoint offline)."
    if summary["fresh_failed_shards"]:
        return "Fresh full TEST not scored: all shards failed before producing receipts."
    return "Fresh full TEST not started."


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dsqa = summarize(
        ROOT / "outputs/full_test_runs_20261012/dsqa_rerun/manifest.json",
        ROOT / "tdsqa_full_20261012/records.jsonl",
        600,
    )
    drbii = summarize(
        ROOT / "tdrbii_full_test_20261012_rerun/manifest.json",
        ROOT / "tdrbii_full_20261012/records.jsonl",
        132,
    )
    rows = [
        {
            "benchmark": "DeepSearchQA",
            "result": (
                f"{fresh_status(dsqa)} "
                f"Reusable receipts: {dsqa['complete_scored']}/{dsqa['records']} tasks scored, "
                f"mean F1 {fmt(dsqa['mean_score_complete'])}; "
                f"{dsqa['expected'] - dsqa['complete_scored']} tasks still lack final scores."
            ),
        },
        {
            "benchmark": "DeepResearch Bench II",
            "result": (
                f"{fresh_status(drbii)} "
                f"Reusable receipts: {drbii['complete_scored']}/{drbii['records']} tasks scored, "
                f"mean Overall {fmt(drbii['mean_score_complete'])}; "
                f"{drbii['expected'] - drbii['complete_scored']} tasks still lack final scores."
            ),
        },
    ]
    (OUT / "benchmark_results.md").write_text(
        "| benchmark | result |\n|---|---|\n"
        + "\n".join(f"| {row['benchmark']} | {row['result']} |" for row in rows)
        + "\n",
        encoding="utf-8",
    )
    with (OUT / "benchmark_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "result"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "benchmark_results.json").write_text(
        json.dumps({"rows": rows, "sources": {
            "deepsearchqa_fresh": "outputs/full_test_runs_20261012/dsqa_rerun/manifest.json",
            "deepsearchqa_receipts": "tdsqa_full_20261012/records.jsonl",
            "drbii_fresh": "tdrbii_full_test_20261012_rerun/manifest.json",
            "drbii_receipts": "tdrbii_full_20261012/records.jsonl",
        }}, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "audit.json").write_text(
        json.dumps({"deepsearchqa": dsqa, "deepresearch_bench_ii": drbii}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
