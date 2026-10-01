"""Explicit iterative ceilings remain binding during team organization."""

import json

import pytest

from jit_mas.config import MASConfig
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph, TeamSpec


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
