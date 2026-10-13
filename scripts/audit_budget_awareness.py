"""Audit the budget information visible in persisted native planning receipts.

The native runner writes a planning_calls.json for every task.  This read-only
audit records budget refreshes and compares a reconciled team's explicit plan
with the resource_budget in that same request. These receipts establish budget
visibility and forecast feasibility, not savings or a causal adaptive response.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def payload(entry):
    if not isinstance(entry, dict):
        return {}
    messages = entry.get("messages") or []
    if len(messages) < 2 or not isinstance(messages[1], dict):
        return {}
    try:
        value = json.loads(messages[1].get("content", "{}"))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def response(entry):
    if not isinstance(entry, dict):
        return {}
    try:
        value = json.loads(entry.get("response", "{}"))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def budget_limits(entry):
    limits = payload(entry).get("limits")
    value = limits.get("resource_budget") if isinstance(limits, dict) else None
    return value if isinstance(value, dict) else {}


def task_identity(run_dir, calls):
    """Resolve current and historical manifest layouts, then verify call IDs."""
    manifest = read_json(run_dir / "run_manifest.json", {}) or {}
    candidates = []
    if isinstance(manifest, dict):
        candidates.append(("run_manifest.task_id", manifest.get("task_id")))
        for label, value in (("run_manifest.task", manifest.get("task")),
                             ("run_manifest.comparison.task", (manifest.get("comparison") or {}).get("task"))):
            if isinstance(value, dict):
                candidates.append((label + ".task_id", value.get("task_id")))
    for entry in calls:
        value = payload(entry).get("task")
        if isinstance(value, dict):
            candidates.append(("planning_payload.task.task_id", value.get("task_id")))
    candidates = [(source, value) for source, value in candidates if isinstance(value, str) and value]
    identities = sorted({value for _, value in candidates})
    if len(identities) > 1:
        return None, None, identities
    return (candidates[0][1], candidates[0][0], []) if candidates else (None, None, [])


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def difference(before, after):
    return before - after if numeric(before) and numeric(after) else None


def resource_change(before, after):
    # A countdown alone is not evidence that consumed tokens/calls were refreshed.
    fields = ("max_tokens", "used_tokens", "reserved_tokens", "remaining_tokens",
              "max_model_calls", "used_model_calls", "remaining_model_calls",
              "max_tool_calls", "used_tool_calls", "remaining_tool_calls")
    return any(before.get(key) != after.get(key) for key in fields)


def planned_totals(team):
    plan = team.get("budget_plan") if isinstance(team, dict) else None
    if not isinstance(plan, dict):
        return None
    agents = plan.get("agents") or []
    keys = ("expected_input_tokens", "expected_output_tokens", "expected_model_calls", "expected_tool_calls")
    totals = {key: sum(int(row.get(key) or 0) for row in agents if isinstance(row, dict)) for key in keys}
    totals["reserved_future_tokens"] = int(plan.get("reserved_future_tokens") or 0)
    totals["reserved_future_model_calls"] = int(plan.get("reserved_future_model_calls") or 0)
    totals["agent_count"] = len(agents)
    return totals


def audit(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob("**/planning_calls.json")):
        if "summary" in path.parts:
            continue
        calls = read_json(path, [])
        if not isinstance(calls, list):
            continue
        calls = [item for item in calls if isinstance(item, dict)]
        limits = [budget_limits(item) for item in calls if budget_limits(item)]
        if not limits:
            continue
        first, last = limits[0], limits[-1]
        run_dir = path.parent
        ledger = read_json(run_dir / "budget.json", {}) or {}
        reconcile_attempts = [item for item in calls if item.get("phase") == "reconcile"]
        reconcile = next((item for item in reversed(reconcile_attempts)
                          if not item.get("validation_errors") and isinstance(response(item).get("team"), dict)), None)
        team = response(reconcile).get("team") if reconcile else None
        planned = planned_totals(team)
        reconcile_limits = budget_limits(reconcile)
        # A response is forecast against the live budget presented to that exact
        # reconcile attempt, including a corrected attempt, never the initial cap.
        remaining = reconcile_limits.get("remaining_tokens")
        calls_remaining = reconcile_limits.get("remaining_model_calls")
        tools_remaining = reconcile_limits.get("remaining_tool_calls")
        fits_tokens = None
        fits_calls = None
        fits_tools = None
        if planned is not None and numeric(remaining):
            fits_tokens = (planned["expected_input_tokens"] + planned["expected_output_tokens"]
                           + planned["reserved_future_tokens"] <= remaining)
        if planned is not None and numeric(calls_remaining):
            fits_calls = planned["expected_model_calls"] + planned["reserved_future_model_calls"] <= calls_remaining
        if planned is not None and numeric(tools_remaining):
            fits_tools = planned["expected_tool_calls"] <= tools_remaining
        task_id, task_id_source, conflicts = task_identity(run_dir, calls)
        rows.append({
            "run_dir": str(run_dir),
            "task_id": task_id,
            "task_id_source": task_id_source,
            "task_id_conflicts": conflicts,
            "planning_calls": len(calls),
            "reconcile_attempts": len(reconcile_attempts),
            "reconcile_attempt": reconcile.get("attempt") if reconcile else None,
            "reconcile_response_accepted": reconcile is not None,
            "first_remaining_tokens": first.get("remaining_tokens"),
            "last_remaining_tokens": last.get("remaining_tokens"),
            "tokens_consumed_before_last_plan": difference(first.get("remaining_tokens"), last.get("remaining_tokens")),
            "reconcile_remaining_tokens": remaining,
            "reconcile_remaining_model_calls": calls_remaining,
            "reconcile_remaining_tool_calls": tools_remaining,
            "plan_fit_budget_timepoint": "selected_reconcile_request" if reconcile else None,
            "first_used_model_calls": first.get("used_model_calls"),
            "last_used_model_calls": last.get("used_model_calls"),
            "ledger_tokens": ledger.get("tokens"),
            "ledger_model_calls": ledger.get("model_calls"),
            "planned_input_tokens": planned.get("expected_input_tokens") if planned else None,
            "planned_output_tokens": planned.get("expected_output_tokens") if planned else None,
            "planned_model_calls": planned.get("expected_model_calls") if planned else None,
            "planned_tool_calls": planned.get("expected_tool_calls") if planned else None,
            "planned_reserved_future_tokens": planned.get("reserved_future_tokens") if planned else None,
            "planned_reserved_future_model_calls": planned.get("reserved_future_model_calls") if planned else None,
            "planned_agent_count": planned.get("agent_count") if planned else None,
            "plan_fits_remaining_tokens": fits_tokens,
            "plan_fits_remaining_model_calls": fits_calls,
            "plan_fits_remaining_tool_calls": fits_tools,
            "budget_changed_between_plans": resource_change(first, last),
            "deadline_changed_between_plans": first.get("remaining_seconds") != last.get("remaining_seconds"),
        })
    changed = [row for row in rows if row["budget_changed_between_plans"]]
    token_deltas = [row["tokens_consumed_before_last_plan"] for row in rows
                    if isinstance(row["tokens_consumed_before_last_plan"], (int, float))]
    summary = {
        "schema": "budget-awareness-audit-v2",
        "source_root": str(root),
        "task_count": len(rows),
        "unique_task_count": len({row["task_id"] for row in rows if row["task_id"]}),
        "tasks_missing_identity": sum(row["task_id"] is None for row in rows),
        "tasks_with_conflicting_identity": sum(bool(row["task_id_conflicts"]) for row in rows),
        "tasks_with_live_budget_change": len(changed),
        "live_budget_change_rate": len(changed) / len(rows) if rows else None,
        "mean_tokens_consumed_before_last_plan": mean(token_deltas) if token_deltas else None,
        "plans_with_explicit_budget": sum(row["planned_agent_count"] is not None for row in rows),
        "plans_fitting_remaining_tokens": sum(row["plan_fits_remaining_tokens"] is True for row in rows),
        "plans_exceeding_remaining_tokens": sum(row["plan_fits_remaining_tokens"] is False for row in rows),
        "plans_with_unknown_token_fit": sum(row["plan_fits_remaining_tokens"] is None for row in rows),
        "plans_fitting_remaining_model_calls": sum(row["plan_fits_remaining_model_calls"] is True for row in rows),
        "plans_exceeding_remaining_model_calls": sum(row["plan_fits_remaining_model_calls"] is False for row in rows),
        "plans_with_unbounded_model_calls": sum(row["reconcile_response_accepted"] and row["reconcile_remaining_model_calls"] is None for row in rows),
        "plan_fit_budget_timepoint": "same accepted reconcile request (before the reconcile response cost)",
        "supported_conclusions": ["Persisted planning requests received the live shared resource ledger.",
                                  "Accepted reconciliation forecasts can be compared with their own request-time remaining resources."],
        "unsupported_conclusions": ["Budget information caused roster/call/token allocation to adapt.",
                                    "Budget-aware planning reduced tokens, latency or monetary cost.",
                                    "A fitting forecast implies all subsequent stages stayed within that forecast."],
        "adaptive_effect_identified": False,
        "savings_identified": False,
        "ledger_tokens_total": sum(row["ledger_tokens"] or 0 for row in rows),
        "ledger_model_calls_total": sum(row["ledger_model_calls"] or 0 for row in rows),
        "rows": rows,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="native run root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    rows = result["rows"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    md_path = args.output.with_suffix(".md")
    md_path.write_text("\n".join([
        "# Live budget planning audit", "",
        f"Task runs with planning receipts: {result['task_count']} ({result['unique_task_count']} unique task IDs)",
        f"Missing / conflicting task identities: {result['tasks_missing_identity']} / {result['tasks_with_conflicting_identity']}",
        f"Tasks whose resource budget changed between planning calls: {result['tasks_with_live_budget_change']} "
        f"({result['live_budget_change_rate']:.1%})" if result["live_budget_change_rate"] is not None else "Tasks whose resource budget changed between planning calls: —",
        f"Plans with explicit TeamBudgetPlan: {result['plans_with_explicit_budget']}",
        f"Plans fitting their reconcile-request remaining token budget: {result['plans_fitting_remaining_tokens']}",
        f"Plans exceeding their reconcile-request remaining token budget: {result['plans_exceeding_remaining_tokens']}",
        f"Plans with unknown token fit: {result['plans_with_unknown_token_fit']}",
        f"Accepted plans with unbounded model-call allowance: {result['plans_with_unbounded_model_calls']}",
        "",
        "Feasibility is measured against the same accepted reconcile request, before charging that response; "
        "it is a forecast check, not proof of execution adherence. Resource changes exclude the deadline countdown alone.",
        "",
        "These receipts show live budget visibility and refresh. They do not identify a causal change in roster, "
        "calls or token allocation, and they do not establish token, latency or cost savings. "
        "A controlled budget perturbation or matched ablation is needed to measure that adaptive effect.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tasks": result["task_count"],
                      "changed": result["tasks_with_live_budget_change"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
