"""Temporary full harnesses execute without modifying the retained Agent Pool."""

import copy
import json
import shutil

import pytest
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError

from jit.harness_ops import WORKSPACE_DIR
from jit_mas.agent_pool import catalogue, seed_pool
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.execution import TeamExecutor
from jit_mas.planning import GlobalAnalyzer, _pooled_prediction_schema, _pooled_reconciliation_schema
from jit_mas.schemas import (
    AgentHarnessPolicy, AgentProfile, AgentSpec, LocalPlan, Prediction,
    PublicTask, RubricGraph, TeamSpec, digest,
)
from scripts.models.base import ChatMessage


def temporary_agent(agent_id="specialist", *, parent="analyst", max_calls=1):
    profile = AgentProfile(
        pool_agent_id="statistical_specialist", role="Statistical Specialist",
        capabilities=["statistical verification"], parent_agent_id=parent,
        prompt="Independently derive the statistical result and report assumptions.",
        skills={"counterexample": "Check the result with a limiting case."},
        reasoning_strategy="Test assumptions before accepting the derivation.",
        planning_strategy="Derive, compare and publish the supported result.",
        communication="Publish the derivation and unresolved assumptions.",
        harness=AgentHarnessPolicy(memory_policy="recent", memory_window=3),
    )
    return AgentSpec(
        agent_id=agent_id, role=profile.role, capability=profile.capabilities[0],
        pool_agent_id=profile.pool_agent_id, pool_agent_version=profile.version,
        temporary_profile=profile, max_calls=max_calls,
        creation_rationale="The task needs a statistical harness absent from the general analyst; "
                           "one concise independent derivation provides a useful quality/cost tradeoff.",
    )


def writer_agent(*, max_calls=1):
    return AgentSpec(agent_id="writer", role="Writer", capability="writing",
                     pool_agent_id="writer", pool_agent_version=1,
                     depends_on=["specialist"], max_calls=max_calls)


def test_prediction_schema_supports_new_full_harnesses_alongside_exact_pool_bindings():
    pool = seed_pool()
    schema = _pooled_prediction_schema(catalogue(pool), max_agents=3)
    temporary_branch = schema["$defs"]["AgentSpec"]["anyOf"][-1]
    assert "(?<!" not in json.dumps(temporary_branch)
    assert "(?!" not in json.dumps(temporary_branch)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    record = Prediction(graph=RubricGraph(rubrics=[]),
                        candidates=[temporary_agent(), writer_agent()]).model_dump(mode="json")
    validator.validate(record)

    invalid = copy.deepcopy(record)
    invalid["candidates"][0]["temporary_profile"]["source_task_ids"] = ["invented-prior-task"]
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid)
    invalid = copy.deepcopy(record)
    invalid["candidates"][0]["pool_agent_id"] = "analyst"
    invalid["candidates"][0]["temporary_profile"]["pool_agent_id"] = "analyst"
    validator.validate(invalid)
    analyzer = GlobalAnalyzer(lambda _messages: "{}", agent_pool=seed_pool())
    with pytest.raises(ValueError, match="conflicts with an Agent Pool member"):
        analyzer._validate_prediction(PublicTask(task_id="collision", question="Check the binding."),
                                      Prediction.model_validate(invalid))
    invalid = copy.deepcopy(record)
    del invalid["candidates"][0]["temporary_profile"]["skills"]
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid)
    invalid = copy.deepcopy(record)
    invalid["candidates"][0]["creation_rationale"] = ""
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid)


def test_prediction_schema_can_create_a_harness_when_the_catalogue_is_empty():
    schema = _pooled_prediction_schema([], max_agents=1)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate(Prediction(graph=RubricGraph(rubrics=[]),
                                  candidates=[temporary_agent(parent=None)]).model_dump(mode="json"))
    assert len(schema["$defs"]["AgentSpec"]["anyOf"]) == 1


@pytest.mark.parametrize("change", ["profile", "rationale"])
def test_reconciliation_preserves_complete_temporary_creation_record(change):
    candidate = temporary_agent()
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate])
    response = {"graph": {"rubrics": []}, "team": {
        "agents": [candidate.model_dump(mode="json")],
        "synthesizer_id": candidate.agent_id, "total_max_calls": 1}}
    validator = Draft202012Validator(_pooled_reconciliation_schema(prediction, max_agents=1))
    validator.check_schema(validator.schema)
    validator.validate(response)
    invalid = copy.deepcopy(response)
    if change == "profile":
        invalid["team"]["agents"][0]["temporary_profile"]["prompt"] = "Replace the recorded harness."
    else:
        invalid["team"]["agents"][0]["creation_rationale"] = "An unrelated new creation claim."
    with pytest.raises(SchemaValidationError):
        validator.validate(invalid)
    analyzer = GlobalAnalyzer(lambda _messages: json.dumps(invalid),
                              agent_pool=seed_pool(), max_corrections=0)
    with pytest.raises(ValueError, match="recorded temporary_profile/creation_rationale"):
        analyzer.reconcile(PublicTask(task_id="freeze", question="Explain the derivation."),
                           prediction, [])


def test_planning_adapts_temporary_harness_and_records_unselected_creation_without_pool_writes():
    pool = seed_pool()
    original_hash = digest(pool)
    candidates = [temporary_agent(), writer_agent()]
    unused = temporary_agent("unused")
    unused.temporary_profile.pool_agent_id = "unused_specialist"
    unused.pool_agent_id = "unused_specialist"
    candidates.append(unused)
    requests = []

    def model(messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        requests.append(payload)
        if payload["phase"] == "predict":
            return Prediction(graph=RubricGraph(rubrics=[]), candidates=candidates).model_dump_json()
        if payload["phase"] == "local_plan":
            candidate = payload["candidate"]
            return LocalPlan(agent_id=candidate["agent_id"], capability=candidate["capability"],
                             depends_on=candidate["depends_on"], selected_skills=None).model_dump_json()
        selected = [item for item in payload["prediction"]["candidates"] if item["agent_id"] != "unused"]
        return json.dumps({"graph": {"rubrics": []}, "team": {
            "agents": selected, "synthesizer_id": "writer", "total_max_calls": 2}})

    analyzer = GlobalAnalyzer(model, agent_pool=pool)
    planned = analyzer.build(PublicTask(task_id="build", question="Explain this statistical result."))
    specialist = planned.team.agents[0]
    assert specialist.selected_skills == ["counterexample"]
    assert specialist.harness == candidates[0].temporary_profile.harness
    assert specialist.reasoning_strategy == candidates[0].temporary_profile.reasoning_strategy
    assert specialist.temporary_profile == candidates[0].temporary_profile
    assert digest(pool) == original_hash
    assert {item.agent_id for item in planned.team.agents} == {"specialist", "writer"}
    assert {item.agent_id for item in analyzer.last_prediction.candidates} == {"specialist", "writer", "unused"}
    local_request = next(payload for payload in requests
                         if payload["phase"] == "local_plan" and payload["candidate"]["agent_id"] == "specialist")
    assert local_request["agent_profile"] == candidates[0].temporary_profile.model_dump(mode="json")
    assert local_request["agent_profile"]["memory"] == []


@pytest.mark.parametrize("invalid_profile", ["identity_collision", "unknown_parent", "unavailable_tool"])
def test_prediction_rejects_invalid_temporary_harness_bindings(invalid_profile):
    candidate = temporary_agent()
    if invalid_profile == "identity_collision":
        candidate.temporary_profile.pool_agent_id = "analyst"
        candidate.pool_agent_id = "analyst"
    elif invalid_profile == "unknown_parent":
        candidate.temporary_profile.parent_agent_id = "missing-parent"
    else:
        candidate.temporary_profile.preferred_tools = ["unavailable_search"]
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate])
    analyzer = GlobalAnalyzer(lambda _messages: prediction.model_dump_json(),
                              agent_pool=seed_pool(), max_corrections=0)
    with pytest.raises(ValueError):
        analyzer.predict(PublicTask(task_id="invalid", question="Explain the result."))


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
def test_mixed_persistent_and_temporary_harnesses_execute_from_frozen_team(mode):
    pool = seed_pool()
    pool_hash = digest(pool)
    max_calls = 1 if mode == "single_pass" else None
    specialist = temporary_agent(max_calls=max_calls)
    specialist.selected_skills = ["counterexample"]
    team = TeamSpec(execution_mode=mode, agents=[specialist, writer_agent(max_calls=max_calls)],
                    synthesizer_id="writer", total_max_calls=2 if max_calls else None)
    task = PublicTask(task_id="execute", question="Explain the statistical result and its assumptions.")
    synth = JITHarnessSynthesizer()
    artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team, agent_pool=pool)
    requests = {}

    class Model:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def __call__(self, messages, **kwargs):
            requests[self.agent_id] = copy.deepcopy(messages)
            answer = "The assumptions and independently derived result."
            response = {"answer": answer, "evidence_ids": [], "checkpoints": {}}
            if self.agent_id != "writer":
                response["ledger"] = {"requirements": [], "outline": [answer],
                                      "evidence_spans": [], "source_references": []}
            return ChatMessage(role="assistant", content=json.dumps(response))

        def get_token_counts(self):
            return {"input_token_count": 2, "output_token_count": 2}

    try:
        result = TeamExecutor(Model).execute(task, team, artifact)
        assert result.terminated_reason == "final_answer"
        assert digest(pool) == pool_hash
        assert artifact.meta_trajectory == []
        assert artifact.selection["temporary_agent_ids"] == ["specialist"]
        assert result.metadata["temporary_agent_ids"] == ["specialist"]
        temporary_context = json.loads(requests["specialist"][1]["content"])["persistent_agent"]
        assert temporary_context["temporary"] is True
        assert temporary_context["pool_agent_id"] == specialist.pool_agent_id
        assert temporary_context["prompt"] == specialist.temporary_profile.prompt
        assert temporary_context["skills"] == specialist.temporary_profile.skills
        assert temporary_context["harness"] == specialist.temporary_profile.harness.model_dump(mode="json")
        assert temporary_context["memory"] == []
        assert temporary_context["creation_rationale"] == specialist.creation_rationale
        writer_context = json.loads(requests["writer"][1]["content"])["persistent_agent"]
        assert writer_context["temporary"] is False
    finally:
        for generated in synth._agents.values():
            path = generated.workspace_dir.resolve()
            assert path.parent == WORKSPACE_DIR.resolve().parent and path.name.startswith("mas_")
            if path.is_dir():
                shutil.rmtree(path)
