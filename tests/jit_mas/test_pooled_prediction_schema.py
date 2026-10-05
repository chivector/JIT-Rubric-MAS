"""Pooled prediction transport binds persistent identities before reconciliation."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError

from jit_mas.agent_pool import seed_pool
from jit_mas.planning import GlobalAnalyzer, _pooled_prediction_schema
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph


def _candidate(agent_id: str, pool_agent_id: str, *, version: int = 1) -> AgentSpec:
    return AgentSpec(agent_id=agent_id, role="Task role", capability="writing",
                     pool_agent_id=pool_agent_id, pool_agent_version=version)


def test_pooled_prediction_schema_binds_catalogue_identity_but_not_task_id():
    catalogue = [
        {"pool_agent_id": "writer", "version": 3},
        {"pool_agent_id": "analyst", "version": 2},
    ]
    schema = _pooled_prediction_schema(catalogue, max_agents=2)
    Draft202012Validator.check_schema(schema)

    assert schema["properties"]["candidates"]["maxItems"] == 2
    branches = schema["$defs"]["AgentSpec"]["anyOf"]
    assert {(branch["properties"]["pool_agent_id"]["const"],
             branch["properties"]["pool_agent_version"]["const"])
            for branch in branches if "const" in branch["properties"]["pool_agent_id"]} == {
                ("writer", 3), ("analyst", 2)}
    assert all("pool_agent_id" in branch["required"]
               and "pool_agent_version" in branch["required"] for branch in branches)
    assert all("const" not in branch["properties"]["agent_id"] for branch in branches)

    valid = Prediction(graph=RubricGraph(rubrics=[]), candidates=[
        _candidate("task_specific_writer", "writer", version=3),
        _candidate("another_task_name", "analyst", version=2),
    ]).model_dump(mode="json")
    validator = Draft202012Validator(schema)
    validator.validate(valid)

    invalid_version = copy.deepcopy(valid)
    invalid_version["candidates"][0]["pool_agent_version"] = 4
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid_version)

    invalid_member = copy.deepcopy(valid)
    invalid_member["candidates"][0]["pool_agent_id"] = "critic"
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid_member)


def test_predict_uses_pool_schema_and_keeps_one_correction_for_duplicate_pool_members():
    pool = seed_pool()
    responses = [
        Prediction(graph=RubricGraph(rubrics=[]), candidates=[
            _candidate("first_writer", "writer"),
            _candidate("second_writer", "writer"),
        ]).model_dump(mode="json"),
        Prediction(graph=RubricGraph(rubrics=[]), candidates=[
            _candidate("first_writer", "writer"),
            _candidate("analyst_role", "analyst"),
        ]).model_dump(mode="json"),
    ]
    requests = []

    def model(messages, **kwargs):
        requests.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(responses[len(requests) - 1])

    analyzer = GlobalAnalyzer(model, max_agents=3, agent_pool=pool)
    analyzer.planning_response_format = "json_schema"
    result = analyzer.predict(PublicTask(
        task_id="schema",
        question=("Research and compare evidence across alternatives, countries and periods; "
                   "produce a reproducible table and cite sources for every claim. " * 12)))

    assert len(requests) == len(analyzer.call_records) == 2
    schema = requests[0]["kwargs"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["candidates"]["maxItems"] == 3
    assert {(branch["properties"]["pool_agent_id"]["const"],
             branch["properties"]["pool_agent_version"]["const"])
            for branch in schema["$defs"]["AgentSpec"]["anyOf"]
            if "const" in branch["properties"]["pool_agent_id"]} == {
                (profile.pool_agent_id, profile.version) for profile in pool.profiles}
    assert "response_correction" in json.loads(requests[1]["messages"][1]["content"])
    assert [(agent.agent_id, agent.pool_agent_id) for agent in result.candidates] == [
        ("first_writer", "writer"), ("analyst_role", "analyst")]
