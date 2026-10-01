"""Explicit iterative ceilings remain binding during team organization."""

import json

import pytest

from jit_mas.budget import BudgetLedger
from jit_mas.config import MASConfig
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph, TeamBudgetPlan, TeamSpec


def test_iterative_planner_cannot_remove_configured_team_ceiling():
    task = PublicTask(task_id="budgeted", question="Explain the result")
    candidate = AgentSpec(agent_id="writer", role="Writer", capability="writing", max_calls=None)
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate])
    response = {"graph": {"rubrics": []}, "team": {
        "agents": [candidate.model_dump(mode="json")], "synthesizer_id": "writer",
        "total_max_calls": None, "execution_mode": "iterative_shared_ledger"}}
    payloads = []

    def model(messages):
        payloads.append(json.loads(messages[1]["content"]))
        return json.dumps(response)

    analyzer = GlobalAnalyzer(model, execution_mode="iterative_shared_ledger", total_max_calls=2)
    with pytest.raises(ValueError, match="resource limits"):
        analyzer.reconcile(task, prediction, [])
    assert payloads[0]["limits"]["total_max_calls"] == 2
    uncapped = GlobalAnalyzer(model, execution_mode="iterative_shared_ledger", total_max_calls=None)
    assert uncapped.reconcile(task, prediction, []).team.total_max_calls is None


def test_fixed_team_cannot_remove_configured_iterative_ceiling():
    team = TeamSpec(agents=[AgentSpec(agent_id="writer", role="Writer", capability="writing", max_calls=None)],
                    synthesizer_id="writer", total_max_calls=None, execution_mode="iterative_shared_ledger")
    with pytest.raises(ValueError, match="configured.*limits"):
        MASConfig(execution_mode="iterative_shared_ledger", fixed_team=team, team_max_calls=2)
    assert MASConfig(execution_mode="iterative_shared_ledger", fixed_team=team,
                     team_max_calls=None).fixed_team == team


def budget_plan(**changes):
    return {
        "agents": [{"agent_id": "writer", "expected_model_calls": 1,
                    "expected_input_tokens": 300, "expected_output_tokens": 100,
                    "expected_tool_calls": 0, "expected_communication_bytes": 80,
                    "rationale": "One retained writer covers the simple task without a separate reviewer."}],
        "reserved_future_tokens": 0, "reserved_future_model_calls": 0,
        "quality_cost_tradeoff": "Use one writer and self-check because the task has no independent evidence need.",
        "stopping_policy": "Submit the complete deliverable after its required checks are satisfied.",
        **changes,
    }


def budget_response(plan=None, **agent_changes):
    agent = AgentSpec(agent_id="writer", role="Writer", capability="writing", max_tokens=128,
                      **agent_changes)
    team = {"agents": [agent.model_dump(mode="json")], "synthesizer_id": "writer",
            "total_max_calls": 1}
    if plan is not None:
        team["budget_plan"] = plan
    return {"graph": {"rubrics": []}, "team": team}


def ledger_context(ledger):
    return {**ledger.snapshot(), "max_total_tokens": ledger.max_tokens,
            "max_model_calls": ledger.max_calls, "max_tool_calls": ledger.max_tool_calls}


def test_live_usage_and_pending_reservations_refresh_on_budget_correction():
    ledger = BudgetLedger(max_tokens=1000, max_calls=10)
    pending = ledger.reserve("other", "concurrent", 100, 100)
    payloads = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        payloads.append(payload)
        charge = ledger.reserve("planning", "global", 90, 10)
        ledger.settle(charge, 90, 10)
        estimate = budget_plan()
        estimate["agents"][0]["expected_input_tokens"] = 650 if len(payloads) == 1 else 500
        return json.dumps(budget_response(estimate))

    analyzer = GlobalAnalyzer(model, budget_context=lambda: ledger_context(ledger))
    task = PublicTask(task_id="budgeted", question="Explain the result")
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=budget_response()["team"]["agents"])
    result = analyzer.reconcile(task, prediction, [])
    first = payloads[0]["limits"]["resource_budget"]
    corrected = payloads[1]["limits"]["resource_budget"]
    assert first["used_tokens"] == 0 and first["reserved_tokens"] == 200
    assert first["remaining_tokens"] == 800
    assert corrected["used_tokens"] == 100 and corrected["remaining_tokens"] == 700
    assert corrected["reserved_tokens"] == 200
    assert "records" not in first
    assert "Planned tokens=750 exceeds remaining shared tokens=700" in str(payloads[1]["response_correction"])
    assert result.team.budget_plan.totals()["input_tokens"] == 500
    assert result.team.budget_plan.totals()["output_tokens"] == 100
    ledger.settle(pending, 0, 0)


@pytest.mark.parametrize("changes,error", [
    ({"reserved_future_tokens": 1001}, "Planned tokens"),
    ({"reserved_future_model_calls": 3}, "Planned model_calls"),
])
def test_future_harness_evaluation_and_learning_reserves_must_fit(changes, error):
    ledger = BudgetLedger(max_tokens=1000, max_calls=3)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(budget_response(budget_plan(**changes))),
                              budget_context=lambda: ledger_context(ledger), max_corrections=0)
    task = PublicTask(task_id="budgeted", question="Explain the result")
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=budget_response()["team"]["agents"])
    with pytest.raises(ValueError, match=error):
        analyzer.reconcile(task, prediction, [])


def test_budget_context_requires_auditable_plan_but_legacy_without_context_is_readable():
    response = budget_response()
    task = PublicTask(task_id="budgeted", question="Explain the result")
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=response["team"]["agents"])
    model = lambda _: json.dumps(response)
    analyzer = GlobalAnalyzer(model, budget_context=lambda: ledger_context(BudgetLedger()), max_corrections=0)
    with pytest.raises(ValueError, match="explicit team.budget_plan"):
        analyzer.reconcile(task, prediction, [])
    assert GlobalAnalyzer(model).reconcile(task, prediction, []).team.budget_plan is None


def test_uncapped_iterative_call_estimates_do_not_create_a_round_limit():
    response = budget_response(budget_plan(), max_calls=None)
    response["team"].update(execution_mode="iterative_shared_ledger", total_max_calls=None)
    response["team"]["budget_plan"]["agents"][0].update(
        expected_model_calls=3, expected_input_tokens=600, expected_output_tokens=300)
    ledger = BudgetLedger(max_calls=None, max_tool_calls=None, max_tokens=1000)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response), execution_mode="iterative_shared_ledger",
                              total_max_calls=None, budget_context=lambda: ledger_context(ledger))
    task = PublicTask(task_id="budgeted", question="Explain the result")
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=response["team"]["agents"])
    result = analyzer.reconcile(task, prediction, [])
    assert result.team.total_max_calls is None
    assert result.team.agents[0].max_calls is None
    assert result.team.budget_plan.agents[0].expected_model_calls == 3


def test_tool_estimates_cannot_exceed_remaining_external_tool_budget():
    response = budget_response(budget_plan(), tools=["lookup"])
    response["team"]["budget_plan"]["agents"][0]["expected_tool_calls"] = 2
    ledger = BudgetLedger(max_tool_calls=1)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response), budget_context=lambda: ledger_context(ledger),
                              max_corrections=0)
    task = PublicTask(task_id="budgeted", question="Explain the result", tools=["lookup"])
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=response["team"]["agents"])
    analyzer.execution_mode = "iterative_shared_ledger"
    response["team"]["execution_mode"] = "iterative_shared_ledger"
    with pytest.raises(ValueError, match="Planned tool_calls=2"):
        analyzer.reconcile(task, prediction, [])


def test_prediction_cannot_reserve_output_above_remaining_tokens():
    ledger = BudgetLedger(max_tokens=100)
    response = {"graph": {"rubrics": []}, "candidates": budget_response()["team"]["agents"]}
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response), budget_context=lambda: ledger_context(ledger),
                              max_corrections=0)
    with pytest.raises(ValueError, match="remaining shared token budget"):
        analyzer.predict(PublicTask(task_id="budgeted", question="Explain the result"))


@pytest.mark.parametrize("change,error", [
    ({"agent_id": "unknown"}, "every selected agent"),
    ({"expected_model_calls": 2}, "single-pass budget estimates"),
    ({"expected_output_tokens": 129}, "response ceilings"),
    ({"expected_tool_calls": 1}, "allowed agent tools"),
])
def test_typed_budget_estimates_match_roster_and_enforced_role_limits(change, error):
    estimate = budget_plan()
    estimate["agents"][0].update(change)
    with pytest.raises(ValueError, match=error):
        TeamSpec.model_validate(budget_response(estimate)["team"])


def test_duplicate_budget_estimates_cannot_charge_one_agent_twice():
    estimate = budget_plan()
    estimate["agents"].append(dict(estimate["agents"][0]))
    with pytest.raises(ValueError, match="every selected agent exactly once"):
        TeamSpec.model_validate(budget_response(estimate)["team"])
    assert TeamBudgetPlan.model_validate(budget_plan()).totals()["communication_bytes"] == 80
