"""Planning guidance reaches live requests without relaxing structural guards."""

import copy
import json

import pytest

from jit_mas.budget import BudgetLedger
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph


def prepared_team(mode):
    max_calls = None if mode == "iterative_shared_ledger" else 1
    agents = [
        AgentSpec(agent_id="analyst", role="Analyst", capability="analysis",
                  rubric_ids=["accuracy"], max_tokens=128, max_calls=max_calls),
        AgentSpec(agent_id="writer", role="Writer", capability="synthesis",
                  rubric_ids=["accuracy"], depends_on=["analyst"], max_tokens=256,
                  max_calls=max_calls),
    ]
    graph = RubricGraph(rubrics=[{"rubric_id": "accuracy", "requirement": "Explain accurately"}])
    prediction = Prediction(graph=graph, candidates=agents)
    team = {
        "execution_mode": mode, "agents": [agent.model_dump(mode="json") for agent in agents],
        "synthesizer_id": "writer", "coverage": {"accuracy": ["analyst", "writer"]},
        "primary": {"accuracy": "analyst"}, "reviewers": {"accuracy": ["writer"]},
        "total_max_calls": None if max_calls is None else 2,
        "budget_plan": {
            "agents": [{"agent_id": agent.agent_id, "expected_model_calls": 1,
                        "expected_input_tokens": 50, "expected_output_tokens": agent.max_tokens,
                        "rationale": "Compact contribution or complete synthesis."} for agent in agents],
            "quality_cost_tradeoff": "Preserve substantive material without duplicating full drafts.",
            "stopping_policy": "Complete the requested artifact while respecting shared budgets.",
        },
    }
    return prediction, {"graph": graph.model_dump(mode="json"), "team": team}


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
def test_role_quality_guidance_is_delivered_to_every_planning_phase(mode):
    prediction, response = prepared_team(mode)
    prompts = {}

    def model(messages):
        payload = json.loads(messages[1]["content"])
        phase = payload["phase"]
        prompts[phase] = " ".join(messages[0]["content"].split())
        if phase == "predict":
            return prediction.model_dump_json()
        if phase == "local_plan":
            candidate = payload["candidate"]
            return json.dumps({key: candidate[key] for key in
                               ("agent_id", "capability", "rubric_ids", "depends_on", "max_calls")})
        return json.dumps(response)

    analyzer = GlobalAnalyzer(model, execution_mode=mode, total_max_calls=response["team"]["total_max_calls"])
    result = analyzer.build(PublicTask(task_id="quality", question="Explain the conclusion and its evidence."))
    for phase in ("predict", "local_plan", "reconcile"):
        assert "competing full deliverable" in prompts[phase]
        assert "3000-5000-word" in prompts[phase]
        assert "consequential defects" in prompts[phase]
        assert "supported correction" in prompts[phase]
        assert "never guess" in prompts[phase] or "must not be invented" in prompts[phase]
    assert "mechanically derive every agent.rubric_ids" in prompts["reconcile"]
    assert "expected_output_tokens <= agent.max_tokens * expected_model_calls" in prompts["reconcile"]
    if mode == "iterative_shared_ledger":
        assert "does not force a one-call estimate" in prompts["reconcile"]
        assert "Do not introduce a fixed round count" in prompts["reconcile"]
        assert all(agent.max_calls is None for agent in result.team.agents)
    assert result.team.agents[-1].max_tokens == 256


def test_iterative_arithmetic_correction_preserves_valid_multiple_call_estimate():
    prediction, response = prepared_team("iterative_shared_ledger")
    bad = copy.deepcopy(response)
    bad["team"]["budget_plan"]["agents"][0].update(expected_model_calls=3,
                                                        expected_output_tokens=385)
    corrected = copy.deepcopy(bad)
    corrected["team"]["budget_plan"]["agents"][0]["expected_output_tokens"] = 384
    observed = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        observed.append(payload)
        if "response_correction" not in payload:
            return json.dumps(bad)
        errors = payload["response_correction"]["validation_errors"]
        assert any("Expected output tokens exceed the role response ceilings" in error["message"]
                   for error in errors)
        assert "expected_output_tokens <= agent.max_tokens *" in messages[0]["content"]
        return json.dumps(corrected)

    ledger = BudgetLedger(max_calls=None, max_tokens=2000)
    analyzer = GlobalAnalyzer(model, execution_mode="iterative_shared_ledger", total_max_calls=None,
                              budget_context=ledger.resource_context)
    result = analyzer.reconcile(PublicTask(task_id="budget", question="Explain accurately"), prediction, [])
    assert len(observed) == 2
    assert result.team.agents[0].max_calls is None
    estimate = result.team.budget_plan.agents[0]
    assert estimate.expected_model_calls == 3
    assert estimate.expected_output_tokens == 384
    assert bad["team"]["budget_plan"]["agents"][0]["expected_output_tokens"] == 385


def test_coverage_correction_requires_actual_assignment_change():
    prediction, response = prepared_team("single_pass")
    bad = copy.deepcopy(response)
    bad["team"]["agents"][1]["rubric_ids"] = []
    responses = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        if "response_correction" not in payload:
            value = bad
        else:
            assert "writer.rubric_ids must be ['accuracy']" in str(payload["response_correction"])
            assert "mechanically derive every agent.rubric_ids" in messages[0]["content"]
            value = response
        responses.append(copy.deepcopy(value))
        return json.dumps(value)

    result = GlobalAnalyzer(model, total_max_calls=2).reconcile(
        PublicTask(task_id="coverage", question="Explain accurately"), prediction, [])
    assert len(responses) == 2
    assert result.team.agents[1].rubric_ids == ["accuracy"]
    assert responses[0]["team"]["agents"][1]["rubric_ids"] == []
