import json

from scripts.build_formal_ours_report import (
    discover_new_results,
    discover_batch_reports,
    fmt,
    latex_table,
    new_curve_rows,
    new_summary,
)


def _record(task_id, *, phase="test", tokens=10, complete=True):
    return {
        "task_id": task_id,
        "benchmark": "writingbench",
        "phase": phase,
        "run_id": "native",
        "round_index": int(task_id.rsplit(":", 1)[1]),
        "status": "complete" if complete else "incomplete",
        "latency_seconds": 1.5,
        "budget": {"tokens": tokens, "model_calls": 1, "tool_calls": 0,
                   "cost": None, "records": []},
        "outcome": {"evaluation": {"complete": complete,
                                     "raw": {"native_mean": 8.0} if complete else {}}},
    }


def test_discover_prefers_records_and_does_not_count_report_twice(tmp_path):
    root = tmp_path / "formal"
    run = root / "native_pilot"
    run.mkdir(parents=True)
    records = [_record("writingbench:1"), _record("writingbench:2", complete=False)]
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    # The runner report repeats those records and must not add two more rows.
    (run / "report.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    rows, files = discover_new_results(root, root / "report")
    assert len(rows) == 2
    assert sum(path.name == "records.jsonl" for path in files) == 1
    assert sum(path.name == "report.json" for path in files) == 1


def test_new_summary_uses_nested_budget_and_separate_missing_score_denominator(tmp_path):
    rows = [_record("writingbench:1", tokens=11), _record("writingbench:2", tokens=13, complete=False)]
    for row in rows:
        row["budget"] = None
        row["outcome"]["budget"] = {"tokens": 11 if row["task_id"].endswith(":1") else 13,
                                      "model_calls": 1, "tool_calls": 0, "cost": None, "records": []}
    summaries = new_summary(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["sample_count"] == 2
    assert summary["scored_count"] == 1
    assert summary["efficiency"]["tokens"] == 24
    assert summary["metric_details"]["nativeMean"]["mean_scored_all"] == 8.0
    assert summary["metric_details"]["nativeMean"]["mean_all_tasks_scored_missing_as_zero"] == 4.0
    assert summary["efficiency"]["actual_cost"] is None


def test_new_curve_keeps_runs_and_phases_separate():
    rows = [_record("writingbench:1", phase="evolution"), _record("writingbench:2", phase="test")]
    curves = new_curve_rows(rows)
    assert {(row["phase"], row["run_id"]) for row in curves} == {("evolution", "native"), ("test", "native")}
    assert all(row["curve_kind"] == "new native task trajectory" for row in curves)


def test_missing_values_are_latex_safe():
    assert fmt(None) == "N/A"
    assert "��" not in latex_table([{"value": None}], [("value", "Value")])


def test_batch_aggregate_is_metadata_and_not_task_count(tmp_path):
    root = tmp_path / "formal"
    campaign = root / "targeted_batch"
    campaign.mkdir(parents=True)
    (campaign / "full_report.json").write_text(json.dumps({"registered_slots": 10,
                                                             "statuses": {"complete": 2}}), encoding="utf-8")
    reports = discover_batch_reports(root, root / "report")
    assert len(reports) == 1
    assert reports[0]["campaign"] == "targeted_batch"
    assert reports[0]["data_status"] == "new/formal_batch"

    live = root / "live_batch"
    live.mkdir()
    (live / "status.json").write_text(json.dumps({"registered_slots": 12,
                                                     "statuses": {"started": 2}}), encoding="utf-8")
    reports = discover_batch_reports(root, root / "report")
    assert {row["campaign"] for row in reports} == {"targeted_batch", "live_batch"}
    assert next(row for row in reports if row["campaign"] == "live_batch")["aggregate_kind"] == "live_status"
