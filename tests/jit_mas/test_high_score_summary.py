import csv
import json

import pytest

from scripts.summarize_high_score_experiment import (
    aggregate_metrics, combined_budget, estimated_cost, load_records, summarize, summarize_efficiency,
)


def budget(tokens=100, calls=1):
    return {"tokens": tokens, "model_calls": calls, "tool_calls": 0, "cost": None,
            "wall_seconds": 2, "by_stage": {"inference": {"tokens": tokens, "model_calls": calls,
                                                           "tool_calls": 0, "cost": None}},
            "records": [{"kind": "model", "stage": "inference", "input_tokens": 60,
                         "output_tokens": 40, "error": "", "request": {
                             "response_model": "model", "transport_retry": {
                                 "attempts": [{"status": "success"}]}}}]}


def test_instruction_metrics_use_instruction_weighting_and_preserve_missing_loose():
    rows = [{"benchmark": "ifeval", "task_id": "ifeval:1", "score_result": {
        "complete": True, "score": 1, "pass_count": 1, "check_count": 1,
        "instruction_accuracy": 1}}, {"benchmark": "ifeval", "task_id": "ifeval:2", "score_result": {
        "complete": True, "score": 0, "pass_count": 1, "check_count": 3,
        "instruction_accuracy": 1 / 3}}]
    metrics = aggregate_metrics(rows)["ifeval"]["root:native:test"]["metrics"]
    assert metrics["prompt_strict_accuracy"]["mean"] == 0.5
    assert metrics["instruction_strict_accuracy"]["mean"] == 0.5
    assert metrics["instruction_strict_accuracy"]["aggregation"] == "instruction-weighted"
    assert metrics["prompt_loose_accuracy"]["mean"] is None


def test_summary_deduplicates_compact_receipts_and_exports_native_curves(tmp_path):
    root = tmp_path / "experiment"
    compact = root / "compact"
    (compact / "records").mkdir(parents=True)
    row = {"task_id": "ifbench:1", "benchmark": "ifbench", "budget": budget(),
           "complete": True, "score_result": {"complete": True, "score": 1,
                                              "pass_count": 2, "check_count": 2,
                                              "instruction_accuracy": 1}}
    (compact / "records" / "task.json").write_text(json.dumps(row), encoding="utf-8")
    (compact / "report.json").write_text(json.dumps({"records": [row]}), encoding="utf-8")
    native = root / "native"
    native.mkdir()
    native_row = {"task_id": "deepresearch_bench_ii:1", "benchmark": "drbii", "phase": "evolve",
                  "round_index": 0, "budget": budget(), "outcome": {"evaluation": {
                      "complete": True, "score": 0.8, "official_metrics": {
                          "Overall": 0.8, "InformationRecall": 0.7, "Analysis": 0.9,
                          "Presentation": 1}}}, "pool_before": {"profiles": []},
                  "pool_after": {"profiles": []}, "operations_requested": [], "operations_applied": []}
    (native / "records.jsonl").write_text(json.dumps(native_row) + "\n", encoding="utf-8")
    slots = native / "slots" / "task"
    slots.mkdir(parents=True)
    (slots / "record.json").write_text(json.dumps(native_row), encoding="utf-8")
    (slots / "started.json").write_text(json.dumps(native_row), encoding="utf-8")
    runs = native / "runs" / "task"
    runs.mkdir(parents=True)
    (runs / "complete.json").write_text(json.dumps(native_row), encoding="utf-8")
    rates = {"default": {"input_per_million": 2, "output_per_million": 4}}
    result = summarize(root, rates=rates)
    assert result["records"] == 2
    assert result["efficiency_summary"]["total"]["tokens"] == 200
    assert result["efficiency_summary"]["total"]["api_attempts"] == 2
    assert result["efficiency_summary"]["total"]["actual_cost"] is None
    assert result["efficiency_summary"]["total"]["estimated_cost"] == 0.00056
    curve = result["curves"]["deepresearch_bench_ii"]["native:native"]["jsonl"]
    exported = json.loads(open(curve, encoding="utf-8").read())
    assert exported["performance_Overall"] == 0.8
    assert exported["round_index"] == 0
    with open(result["curves"]["ifbench"]["compact:compact"]["csv"], encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["round_index"] == ""
    assert (root / "summary" / "run_summary.md").exists()
    assert len(load_records(root)) == 2


def test_unavailable_latency_and_unsettled_cost_stay_null():
    row = {"task_id": "researchrubrics:1", "benchmark": "researchrubrics", "budget": {
        "model_calls": 2, "tokens": 3, "records": []}}
    combined = combined_budget(row)
    assert combined["wall_seconds"] is None
    assert estimated_cost(combined, {"default": {"input_per_million": 1, "output_per_million": 1}}) is None


def test_inherited_generation_does_not_replace_candidate_evaluation_budget():
    generation = budget()
    evaluation = budget()
    evaluation["records"][0]["stage"] = "evaluation"
    evaluation["by_stage"] = {"evaluation": evaluation["by_stage"]["inference"]}
    row = {"budget": evaluation, "generation_budget": generation, "latency_seconds": 4}
    combined = combined_budget(row)
    assert combined["tokens"] == 200
    assert combined["model_calls"] == 2
    assert len(combined["records"]) == 2
    assert combined["by_stage"]["inference"]["tokens"] == 100
    assert combined["by_stage"]["evaluation"]["tokens"] == 100
    assert combined["wall_seconds"] == 4


def test_separate_runs_keep_same_task_as_separate_conditions(tmp_path):
    row = {"task_id": "ifbench:1", "benchmark": "ifbench", "budget": budget(),
           "score_result": {"complete": True, "score": 1}}
    for name in ("pilot", "compact"):
        run = tmp_path / name
        run.mkdir()
        (run / "report.json").write_text(json.dumps({"records": [row]}), encoding="utf-8")
    records = load_records(tmp_path)
    assert len(records) == 2
    metrics = aggregate_metrics(records)["ifbench"]
    assert set(metrics) == {"pilot:compact:test", "compact:compact:test"}
    assert all(condition["task_count"] == 1 for condition in metrics.values())


def test_nested_instruction_raw_and_fixed_prompt_denominator():
    rows = [{"benchmark": "ifbench", "task_id": "ifbench:1", "score_result": {
        "complete": True, "score": 1, "evaluation": {"raw": {
            "pass_count": 2, "check_count": 2, "instruction_accuracy": 1}}}},
            {"benchmark": "ifbench", "task_id": "ifbench:2", "score_result": {
                "complete": False, "score": None}}]
    metrics = aggregate_metrics(rows)["ifbench"]["root:native:test"]["metrics"]
    assert metrics["prompt_loose_accuracy"]["mean_all_tasks"] == 0.5
    assert metrics["prompt_loose_accuracy"]["mean_observed"] == 1
    assert metrics["instruction_loose_accuracy"]["mean_observed"] == 1


def test_official_supplement_changes_only_metrics_without_duplication(tmp_path):
    run = tmp_path / "compact"
    run.mkdir()
    row = {"task_id": "ifbench:1", "benchmark": "ifbench", "budget": budget(),
           "score_result": {"complete": True, "score": 1}}
    report = run / "report.json"
    report.write_text(json.dumps({"records": [row]}), encoding="utf-8")
    (run / "official_metrics.json").write_text(json.dumps({
        "schema": "compact-official-metrics-v1", "source_report": str(report), "records": [{
            "task_id": "ifbench:1", "strict_result": {"complete": True,
                "follow_all_instructions": False, "pass_count": 1, "check_count": 2},
            "loose_result": {"complete": True, "follow_all_instructions": True,
                             "pass_count": 2, "check_count": 2}}]}), encoding="utf-8")
    records = load_records(tmp_path)
    assert len(records) == 1 and records[0]["budget"]["tokens"] == 100
    metrics = aggregate_metrics(records)["ifbench"]["compact:compact:test"]["metrics"]
    assert metrics["prompt_strict_accuracy"]["mean"] == 0
    assert metrics["instruction_strict_accuracy"]["mean"] == 0.5
    assert metrics["prompt_loose_accuracy"]["mean"] == 1


@pytest.mark.parametrize("with_supplement", [False, True])
def test_missing_compact_task_retains_full_denominator_and_unknown_usage(tmp_path, with_supplement):
    run = tmp_path / "compact"
    run.mkdir()
    row = {"task_id": "ifbench:1", "benchmark": "ifbench", "budget": budget(),
           "complete": True, "score_result": {"complete": True, "score": 1,
                                              "pass_count": 2, "check_count": 2}}
    report = run / "report.json"
    source = json.dumps({"benchmark": "ifbench", "task_ids": ["ifbench:1", "ifbench:2"],
                         "records": [row]})
    report.write_text(source, encoding="utf-8")
    if with_supplement:
        (run / "official_metrics.json").write_text(json.dumps({
            "schema": "compact-official-metrics-v1", "source_report": str(report), "records": [
                {"task_id": "ifbench:1", "instruction_count": 2,
                 "strict_result": {"complete": True, "follow_all_instructions": False,
                                   "pass_count": 1, "check_count": 2},
                 "loose_result": {"complete": True, "follow_all_instructions": True,
                                  "pass_count": 2, "check_count": 2}},
                {"task_id": "ifbench:2", "status": "not_attempted", "instruction_count": 3,
                 "strict_result": None, "loose_result": None}]}), encoding="utf-8")
    records = load_records(tmp_path)
    assert len(records) == 2
    missing = next(record for record in records if record["task_id"] == "ifbench:2")
    assert missing["status"] == "not_attempted" and missing["budget"] is None
    result = summarize(tmp_path)
    condition = result["benchmark_metrics"]["ifbench"]["compact:compact:test"]
    assert condition["task_count"] == 2 and condition["scored_tasks"] == 1
    assert condition["failed_tasks"] == 1 and condition["complete_tasks"] == 1
    metrics = condition["metrics"]
    assert metrics["prompt_loose_accuracy"]["mean"] == 0.5
    assert metrics["prompt_loose_accuracy"]["mean_scored_all"] == 1
    if with_supplement:
        assert metrics["instruction_strict_accuracy"]["mean"] == 0.2
        assert metrics["instruction_loose_accuracy"]["mean"] == 0.4
        assert metrics["instruction_loose_accuracy"]["total_instructions"] == 5
    else:
        assert metrics["instruction_loose_accuracy"]["mean"] is None
    efficiency = result["efficiency_summary"]["total"]
    assert efficiency["task_count"] == 2 and efficiency["task_failure_rate"] == 0.5
    assert efficiency["tokens"] is None and efficiency["tokens_known_rows"] == 1
    assert efficiency["model_calls"] is None and efficiency["api_attempts"] is None
    assert efficiency["latency_seconds_sum"] is None
    assert report.read_text(encoding="utf-8") == source


def test_failed_evolution_retains_scored_artifact_and_failure_state(tmp_path):
    run = tmp_path / "native"
    artifacts = run / "runs" / "artifact"
    artifacts.mkdir(parents=True)
    (artifacts / "evaluation.json").write_text(json.dumps({"complete": True, "score": 0.75,
                                                           "raw": {"native_mean": 7.75}}), encoding="utf-8")
    (artifacts / "budget.json").write_text(json.dumps(budget()), encoding="utf-8")
    row = {"task_id": "writingbench:1", "benchmark": "writingbench", "phase": "evolution",
           "round_index": 1, "status": "failed", "outcome": None,
           "failure": {"run_failure": {"run_dir": str(artifacts)}},
           "harness_repair_count": 1, "harness_meta_trajectory": [{"candidate": "a"}],
           "harness_failed_attempts": [{"error": "repair"}]}
    (run / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = summarize(tmp_path)
    efficiency = result["efficiency_summary"]["total"]
    assert efficiency["failed_tasks"] == 1 and efficiency["task_failure_rate"] == 1
    assert efficiency["tokens"] == 100
    metrics = result["benchmark_metrics"]["writingbench"]["native:native:evolution"]
    assert metrics["complete_tasks"] == 0 and metrics["scored_tasks"] == 1
    assert metrics["metrics"]["nativeMean"]["mean"] == 7.75
    assert metrics["evaluation_complete_tasks"] == 1
    assert metrics["metrics"]["nativeMean"]["mean_scored_all"] == 7.75
    curve = result["curves"]["writingbench"]["native:native"]["jsonl"]
    exported = json.loads(open(curve, encoding="utf-8").read())
    assert exported["failure"] is True and exported["evaluation_complete"] is True
    assert exported["score_per_round"] == 0.75 and exported["harness_repair_count"] == 1
    assert exported["harness_meta_trajectory"] == [{"candidate": "a"}]


def test_instruction_failure_uses_full_instruction_denominator():
    rows = [{"benchmark": "ifbench", "task_id": "ifbench:1", "instruction_count": 3,
             "score_result": {"complete": True, "score": 0},
             "loose_result": {"complete": True, "follow_all_instructions": False,
                              "pass_count": 2, "check_count": 3}},
            {"benchmark": "ifbench", "task_id": "ifbench:2", "instruction_count": 1,
             "score_result": {"complete": False, "score": None}, "loose_result": None}]
    metrics = aggregate_metrics(rows)["ifbench"]["root:native:test"]["metrics"]
    instruction = metrics["instruction_loose_accuracy"]
    assert instruction["mean"] == 0.5
    assert instruction["mean_observed"] == 2 / 3
    assert instruction["total_instructions"] == 4 and instruction["observed_instructions"] == 3


def mixed_budget():
    return {"tokens": 150, "model_calls": 2, "tool_calls": 0, "cost": None,
            "wall_seconds": 6, "by_stage": {
                "inference": {"tokens": 100, "model_calls": 1, "tool_calls": 0, "cost": None},
                "evaluation": {"tokens": 50, "model_calls": 1, "tool_calls": 0, "cost": None}},
            "records": [{"kind": "model", "stage": "inference", "input_tokens": 60,
                         "output_tokens": 40, "estimated": False, "wall_seconds": 2, "error": "",
                         "request": {"queue_seconds": 0.25, "transport_retry": {
                             "attempts": [{"status": "success"}]}}},
                        {"kind": "model", "stage": "evaluation", "input_tokens": 20,
                         "output_tokens": 30, "estimated": True, "wall_seconds": 5, "error": "Timeout",
                         "request": {"queue_seconds": 0.5, "transport_retry": {
                             "attempts": [{"status": "retry"}, {"status": "failed"}]}}}]}


def test_efficiency_keeps_provider_estimates_and_stage_call_receipts():
    row = {"task_id": "writingbench:1", "benchmark": "writingbench", "budget": mixed_budget(),
           "score_result": {"complete": False, "score": None}}
    summary = summarize_efficiency([row])
    total = summary["total"]
    assert total["input_tokens"] == 80 and total["output_tokens"] == 70
    assert total["actual_input_tokens"] == 60 and total["actual_output_tokens"] == 40
    assert total["estimated_input_tokens"] == 20 and total["estimated_output_tokens"] == 30
    assert total["actual_tokens"] == 100 and total["estimated_tokens"] == 50
    assert total["actual_model_calls"] == total["estimated_model_calls"] == 1
    assert total["token_usage_source"] == "mixed" and total["token_usage_complete"] is True
    assert total["api_calls"] == total["api_attempts"] == 3
    assert total["failed_api_attempts"] == 2 and total["api_failure_rate"] == 2 / 3
    assert total["retry_count"] == total["failed_calls"] == 1
    assert total["model_call_failure_rate"] == 0.5
    assert total["latency_seconds_sum"] == 6 and total["call_wall_seconds_sum"] == 7
    assert total["request_queue_seconds_sum"] == 0.75
    assert total["actual_cost"] is None and total["estimated_cost"] is None
    assert total["cost_evidence"]["rate_card_supplied"] is False
    assert total["cost_evidence"]["actual_cost_missing_reason"] == "provider billing not recorded"
    stage = total["by_stage"]["evaluation"]
    assert stage["input_tokens"] == 20 and stage["output_tokens"] == 30
    assert stage["actual_tokens"] == 0 and stage["estimated_tokens"] == 50
    assert stage["api_attempts"] == 2 and stage["failed_api_attempts"] == 2
    assert stage["call_wall_seconds_sum"] == 5 and stage["retry_count"] == 1


def test_unsettled_usage_preserves_observations_without_complete_totals():
    value = mixed_budget()
    value["model_calls"] = 3
    value["tokens"] = 175
    row = {"task_id": "writingbench:1", "benchmark": "writingbench", "budget": value}
    total = summarize_efficiency([row])["total"]
    assert total["tokens"] == 175
    assert total["input_tokens"] is None and total["output_tokens"] is None
    assert total["observed_input_tokens"] == 80 and total["observed_output_tokens"] == 70
    assert total["unattributed_tokens"] == 25 and total["unattributed_model_calls"] == 1
    assert total["token_usage_source"] == "unknown" and total["token_usage_unknown_rows"] == 1
    assert total["api_attempts"] is None and total["model_call_failure_rate"] is None
    assert total["call_wall_seconds_sum"] is None
    assert total["by_stage"]["inference"]["input_tokens"] is None
    assert total["by_stage"]["inference"]["actual_input_tokens"] == 60


def test_absent_budget_does_not_turn_observed_usage_into_exact_totals():
    rows = [{"task_id": "writingbench:1", "benchmark": "writingbench", "budget": mixed_budget()},
            {"task_id": "writingbench:2", "benchmark": "writingbench"}]
    total = summarize_efficiency(rows)["total"]
    assert total["input_tokens"] is None and total["tokens"] is None
    assert total["actual_tokens"] == 100 and total["estimated_tokens"] == 50
    assert total["token_usage_unknown_rows"] == 1
    assert total["by_stage"]["evaluation"]["api_attempts"] is None


def test_scored_mean_includes_truncated_execution_and_ignores_incomplete_evaluation(tmp_path):
    run = tmp_path / "compact_writingbench"
    run.mkdir()
    rows = []
    for task_id, native, status in (("writingbench:18", 8.2, "complete"),
                                    ("writingbench:104", 6.8, "incomplete"),
                                    ("writingbench:203", 7.2, "complete")):
        rows.append({"task_id": task_id, "benchmark": "writingbench", "phase": "test",
                     "status": status, "budget": budget(), "evaluation": {
                         "complete": True, "score": (native - 1) / 9, "raw": {"native_mean": native}}})
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = summarize(tmp_path)
    condition = result["benchmark_metrics"]["writingbench"]["compact_writingbench:native:test"]
    assert condition["execution_complete_tasks"] == condition["complete_tasks"] == 2
    assert condition["scored_tasks"] == condition["evaluation_complete_tasks"] == 3
    assert condition["metrics"]["nativeMean"]["mean_scored_all"] == pytest.approx(7.4)
    normalized = condition["metrics"]["normalized_score"]
    assert normalized["mean_scored_all"] == pytest.approx(0.7111111111111111)
    assert normalized["mean_all_tasks_scored_missing_as_zero"] == pytest.approx(0.7111111111111111)
    efficiency = result["efficiency_summary"]["total"]
    assert efficiency["complete_tasks"] == 2 and efficiency["evaluation_complete_tasks"] == 3
    summary_text = (tmp_path / "summary/run_summary.md").read_text(encoding="utf-8")
    assert "| Execution complete | Scored |" in summary_text
    assert "| Scored mean | Full-task denominator (missing=0) |" in summary_text
    assert "| 2/3 | 3/3 | nativeMean=7.400000, normalized_score=0.711111 |" in summary_text
    failed_evaluation = {"task_id": "writingbench:4", "benchmark": "writingbench", "status": "failed",
                         "evaluation": {"complete": False, "score": 1, "raw": {"native_mean": 10}}}
    condition = aggregate_metrics(rows + [failed_evaluation])["writingbench"]["root:native:test"]
    assert condition["metrics"]["nativeMean"]["mean_scored_all"] == pytest.approx(7.4)
    assert condition["metrics"]["nativeMean"]["mean_all_tasks_scored_missing_as_zero"] == pytest.approx(5.55)


def test_invalid_operation_attempts_are_distinct_from_validated_and_applied(tmp_path):
    run = tmp_path / "native"
    artifacts = run / "runs" / "artifact"
    artifacts.mkdir(parents=True)
    operation = {"operation_id": "writer-specialize", "kind": "specialize",
                 "source_task_id": "writingbench:5", "target_agent_ids": ["writer"],
                 "profiles": [{"pool_agent_id": "writer", "prompt": "candidate"}]}
    errors = [{"type": "contract", "message": "conflicting local update"}]
    meta = {"complete": False, "decision": None, "calls": [
        {"phase": "evolution_integrate", "attempt": attempt,
         "response": json.dumps({"pool_operations": [operation]}), "validation_errors": errors}
        for attempt in (0, 1)]}
    (artifacts / "meta_evolution.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [{"task_id": "writingbench:5", "benchmark": "writingbench", "phase": "evolution",
             "round_index": 1, "status": "failed", "outcome": None, "budget": budget(),
             "failure": {"run_failure": {"run_dir": str(artifacts)}},
             "operations_requested": [], "operations_applied": []},
            {"task_id": "writingbench:18", "benchmark": "writingbench", "phase": "test",
             "round_index": 1, "status": "complete", "budget": budget(tokens=200),
             "evaluation": {"complete": True, "score": 0.5},
             "operations_requested": [], "operations_applied": []}]
    raw = "".join(json.dumps(row) + "\n" for row in rows)
    (run / "records.jsonl").write_text(raw, encoding="utf-8")
    result = summarize(tmp_path)
    curve_paths = result["curves"]["writingbench"]["native:native"]
    combined = [json.loads(line) for line in open(curve_paths["jsonl"], encoding="utf-8")]
    first = combined[0]
    assert first["operation_counts_proposed"]["specialize"] == 2
    assert first["operation_counts_proposed_unique"]["specialize"] == 1
    assert first["operation_counts_validated"]["specialize"] == 0
    assert first["operation_counts_applied"]["specialize"] == 0
    assert first["cumulative_operation_proposed_unique_specialize"] == 1
    assert [attempt["valid"] for attempt in first["evolution_operation_attempts"]] == [False, False]
    assert first["evolution_operation_attempts"][0]["pool_operations"][0]["target_agent_ids"] == ["writer"]
    assert len(first["evolution_operation_attempts"][0]["pool_operations"][0]["operation_sha256"]) == 64
    test_rows = [json.loads(line) for line in open(curve_paths["phases"]["test"]["jsonl"], encoding="utf-8")]
    assert test_rows[0]["cumulative_tokens"] == 200
    assert test_rows[0]["cumulative_operation_proposed_unique_specialize"] == 0
    operation_summary = result["evolution_operations"]
    evolution = next(row for row in operation_summary if row["phase"] == "evolution")
    assert evolution["operation_counts_proposed"]["specialize"] == 2
    assert evolution["operation_counts_proposed_unique"]["specialize"] == 1
    assert (run / "records.jsonl").read_text(encoding="utf-8") == raw
