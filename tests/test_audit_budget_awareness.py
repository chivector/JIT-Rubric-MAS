import json

from scripts.audit_budget_awareness import audit, main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def call(phase, remaining, *, task_id="task-1", plan_tokens=None, remaining_calls=None,
         expected_calls=1, attempt=0, invalid=False, seconds=100):
    entry = {
        "phase": phase, "attempt": attempt,
        "messages": [{"role": "system", "content": "test"}, {"role": "user", "content": json.dumps({
            "task": {"task_id": task_id}, "limits": {"resource_budget": {
                "remaining_tokens": remaining, "remaining_model_calls": remaining_calls,
                "remaining_seconds": seconds}}})}],
        "response": json.dumps({"team": {"budget_plan": {"agents": [{
            "expected_input_tokens": plan_tokens, "expected_output_tokens": 0,
            "expected_model_calls": expected_calls}]}}}) if plan_tokens is not None else "{}",
    }
    if invalid:
        entry["validation_errors"] = [{"message": "Invalid assignment"}]
    return entry


def test_reconcile_forecast_uses_same_request_budget_and_nested_manifest_identity(tmp_path):
    run = tmp_path / "runs" / "run-1"
    write_json(run / "planning_calls.json", [call("predict", 1000),
        call("reconcile", 400, plan_tokens=500, remaining_calls=1, expected_calls=2)])
    write_json(run / "run_manifest.json", {"comparison": {"task": {"task_id": "task-1"}}})
    result = audit(tmp_path)
    row = result["rows"][0]
    assert row["task_id"] == "task-1"
    assert row["task_id_source"] == "run_manifest.comparison.task.task_id"
    assert row["reconcile_remaining_tokens"] == 400
    assert row["plan_fits_remaining_tokens"] is False
    assert row["plan_fits_remaining_model_calls"] is False
    assert result["plans_exceeding_remaining_tokens"] == 1


def test_corrected_reconcile_uses_corrected_budget_and_excludes_rejected_response(tmp_path):
    run = tmp_path / "run"
    write_json(run / "planning_calls.json", [call("predict", 1000),
        call("reconcile", 600, plan_tokens=100, invalid=True),
        call("reconcile", 300, plan_tokens=350, attempt=1)])
    row = audit(tmp_path)["rows"][0]
    assert row["reconcile_attempt"] == 1
    assert row["reconcile_attempts"] == 2
    assert row["planned_input_tokens"] == 350
    assert row["plan_fits_remaining_tokens"] is False


def test_deadline_countdown_alone_does_not_count_as_live_resource_change(tmp_path):
    write_json(tmp_path / "run" / "planning_calls.json", [call("predict", 1000, seconds=100),
        call("reconcile", 1000, plan_tokens=100, seconds=90)])
    result = audit(tmp_path)
    assert result["tasks_with_live_budget_change"] == 0
    assert result["rows"][0]["deadline_changed_between_plans"] is True
    assert result["savings_identified"] is False
    assert result["adaptive_effect_identified"] is False


def test_payload_fallback_unbounded_allowance_and_conflicting_identity(tmp_path):
    write_json(tmp_path / "valid" / "planning_calls.json", [call("predict", 1000),
        call("reconcile", 900, plan_tokens=100)])
    write_json(tmp_path / "conflict" / "run_manifest.json", {"task_id": "different-task"})
    write_json(tmp_path / "conflict" / "planning_calls.json", [call("predict", 1000)])
    result = audit(tmp_path)
    valid = next(row for row in result["rows"] if row["task_id"] == "task-1")
    assert valid["task_id_source"] == "planning_payload.task.task_id"
    assert valid["plan_fits_remaining_model_calls"] is None
    assert result["plans_with_unbounded_model_calls"] == 1
    assert result["tasks_missing_identity"] == result["tasks_with_conflicting_identity"] == 1
    assert result["plans_with_unknown_token_fit"] == 1


def test_cli_writes_all_audit_formats_and_keeps_rows_in_json(tmp_path):
    write_json(tmp_path / "source" / "run" / "planning_calls.json", [call("predict", 1000),
        call("reconcile", 900, plan_tokens=100)])
    output = tmp_path / "report" / "budget_audit.json"
    main([str(tmp_path / "source"), "--output", str(output)])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["rows"][0]["task_id"] == "task-1"
    assert output.with_suffix(".csv").is_file()
    assert "do not establish token, latency or cost savings" in output.with_suffix(".md").read_text(encoding="utf-8")
