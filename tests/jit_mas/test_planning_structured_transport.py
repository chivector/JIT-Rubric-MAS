"""Actual opt-in planning requests and validation; synthetic public data only."""
import copy
import hashlib
import json
import shutil

import pytest

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.execution import TeamExecutor
from jit_mas.planning import GlobalAnalyzer, PLANNING_SCHEMA_NAMES
from jit_mas.schemas import (
    AgentPoolSnapshot, AgentProfile, AgentSpec, LocalPlan, Prediction, PublicTask,
    RubricGraph,
)
from scripts.models.base import ChatMessage


def prepared(mode="iterative_shared_ledger", *, max_calls=None):
    graph = RubricGraph(rubrics=[{"rubric_id": "coverage", "requirement": "Explain each requested factor."}])
    agent = AgentSpec(agent_id="public_writer", role="Author", capability="composition",
                      rubric_ids=["coverage"], max_tokens=128, max_calls=max_calls)
    prediction = Prediction(graph=graph, candidates=[agent])
    local = LocalPlan(agent_id=agent.agent_id, capability=agent.capability,
                      rubric_ids=agent.rubric_ids, max_calls=max_calls)
    response = {"graph": graph.model_dump(mode="json"), "team": {
        "execution_mode": mode, "agents": [agent.model_dump(mode="json")],
        "synthesizer_id": agent.agent_id, "coverage": {"coverage": [agent.agent_id]},
        "primary": {"coverage": agent.agent_id}, "reviewers": {}, "total_max_calls": max_calls,
        "budget_plan": {"agents": [{"agent_id": agent.agent_id, "expected_model_calls": 1,
                                   "expected_input_tokens": 20, "expected_output_tokens": 128,
                                   "rationale": "Complete the public artifact."}],
                        "quality_cost_tradeoff": "Keep relevant content within the ledger budget.",
                        "stopping_policy": "Stop when the complete artifact is valid."}}}
    return prediction, local, response


class Provider:
    def __init__(self, replies=None, prepared_records=None, finish_reasons=None):
        self.replies = iter(replies) if replies is not None else None
        self.prepared = prepared_records or prepared()
        self.finish_reasons = iter(finish_reasons or [])
        self.requests = []
        self.last_request_metadata = {}

    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        self.requests.append(copy.deepcopy({"messages": messages, "payload": payload, "kwargs": kwargs}))
        self.last_request_metadata = {"finish_reason": next(self.finish_reasons, "stop"),
                                      "response_model": "synthetic-model"}
        if self.replies is not None:
            value = next(self.replies)
        else:
            value = {"predict": self.prepared[0].model_dump(mode="json"),
                     "local_plan": self.prepared[1].model_dump(mode="json"),
                     "reconcile": self.prepared[2]}[payload["phase"]]
        return ChatMessage(role="assistant", content=value if isinstance(value, str) else json.dumps(value))

    def get_token_counts(self):
        return {"input_token_count": 7, "output_token_count": 3}


def bound(provider, *, mode="iterative_shared_ledger", ceiling=None, schema=True,
          max_tokens=512, ledger=None, agent_pool=None):
    ledger = ledger or BudgetLedger(max_calls=None, max_tokens=2_000_000)
    model = MeteredModel(provider, ledger, "inference", "planning", max_tokens=max_tokens)
    analyzer = GlobalAnalyzer(model, execution_mode=mode, total_max_calls=ceiling,
                              budget_context=ledger.resource_context, agent_pool=agent_pool)
    analyzer.planning_response_format = "json_schema" if schema else "json_object"
    return analyzer, ledger


def invoke(analyzer, phase, task, records):
    prediction, local, _ = records
    if phase == "predict":
        return analyzer.predict(task)
    if phase == "local_plan":
        return analyzer.local_plan(task, prediction, prediction.candidates[0])
    return analyzer.reconcile(task, prediction, [local])


def test_all_three_actual_requests_have_strict_schema_and_uncapped_plan_and_audit():
    task = PublicTask(task_id="synthetic-transport", question="Explain the listed factors: " + "factor " * 900,
                      constraints=["Retain every supplied factor without inventing exclusions."])
    before = task.model_dump(mode="json")
    records = prepared()
    provider = Provider(prepared_records=records)
    analyzer, ledger = bound(provider)
    result = analyzer.build(task)
    assert len(provider.requests) == 3
    for request, audit in zip(provider.requests, analyzer.call_records):
        phase = request["payload"]["phase"]
        actual = request["kwargs"]["response_format"]
        assert actual["type"] == "json_schema"
        assert actual["json_schema"]["name"] == PLANNING_SCHEMA_NAMES[phase]
        assert actual["json_schema"]["strict"] is True
        assert actual == audit["response_format"]
        assert audit["response_format_hash"] == hashlib.sha256(json.dumps(
            actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert request["payload"]["planning_output_budget"] == {"max_tokens_per_response": 512}
        assert request["kwargs"]["max_tokens"] == 512
        assert request["payload"]["task"] == before
        assert audit["finish_metadata"]["finish_reason"] == "stop"
    assert task.model_dump(mode="json") == before
    assert result.team.total_max_calls is None
    assert result.team.agents[0].max_calls is None
    assert result.local_plans[0].max_calls is None
    assert ledger.snapshot()["model_calls"] == 3
    assert ledger.snapshot()["tokens"] == 30
    assert ledger.snapshot()["reserved_tokens"] == 0


@pytest.mark.parametrize("phase", ["predict", "local_plan", "reconcile"])
def test_provider_cannot_ignore_null_call_contract_and_get_finite_caps_accepted(phase):
    records = prepared(max_calls=1)
    provider = Provider(prepared_records=records)
    analyzer, ledger = bound(provider)
    with pytest.raises(ValueError, match="must be null"):
        invoke(analyzer, phase, PublicTask(task_id="synthetic-null-contract", question="Explain factors."), records)
    assert len(provider.requests) == len(analyzer.call_records) == 2
    for request in provider.requests:
        schema = request["kwargs"]["response_format"]["json_schema"]["schema"]
        containers = [schema, *schema.get("$defs", {}).values()]
        capped = [node for node in containers if "max_calls" in node.get("properties", {})]
        assert capped
        assert all(node["properties"]["max_calls"] == {"type": "null", "const": None} for node in capped)
        assert all("max_calls" in node["required"] for node in capped)
    assert records[0].candidates[0].max_calls == 1
    assert records[1].max_calls == 1
    assert records[2]["team"]["total_max_calls"] == 1
    assert ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 20


@pytest.mark.parametrize("mode,ceiling", [("single_pass", 1), ("iterative_shared_ledger", 1)])
@pytest.mark.parametrize("phase", ["predict", "local_plan", "reconcile"])
def test_single_pass_and_explicit_finite_ceiling_are_retained(mode, ceiling, phase):
    records = prepared(mode, max_calls=1)
    provider = Provider(prepared_records=records)
    analyzer, _ = bound(provider, mode=mode, ceiling=ceiling)
    result = invoke(analyzer, phase, PublicTask(task_id="synthetic-finite", question="Explain factors."), records)
    if phase == "predict":
        assert result.candidates[0].max_calls == 1
    elif phase == "local_plan":
        assert result.max_calls == 1
    else:
        assert result.team.total_max_calls == result.team.agents[0].max_calls == 1
    assert len(provider.requests) == 1


@pytest.mark.parametrize("field,limit,phase", [("task_prompt", 4096, "predict"),
    ("rationale", 1024, "reconcile"), ("selection_rationale", 1024, "reconcile")])
def test_bounded_output_annotations_are_validated_without_truncating_inputs_or_records(field, limit, phase):
    records = prepared()
    prediction = records[0].model_dump(mode="json")
    response = copy.deepcopy(records[2])
    if field == "task_prompt":
        prediction["candidates"][0][field] = "x" * (limit + 1)
        value = prediction
    elif field == "rationale":
        response["team"]["budget_plan"]["agents"][0][field] = "x" * (limit + 1)
        value = response
    else:
        response["team"][field] = "x" * (limit + 1)
        value = response
    task = PublicTask(task_id="synthetic-text-bound", question="Required evidence: " + "fact " * 4000)
    provider = Provider(replies=[value, value])
    analyzer, ledger = bound(provider)
    with pytest.raises(ValueError, match="planning annotation limit"):
        invoke(analyzer, phase, task, records)
    assert len(provider.requests) == 2
    assert all(r["payload"]["task"]["question"] == task.question for r in provider.requests)
    assert all(json.loads(row["response"]) == value for row in analyzer.call_records)
    assert ledger.snapshot()["tokens"] == 20


@pytest.mark.parametrize("phase", ["predict", "local_plan", "reconcile"])
@pytest.mark.parametrize("pooled", [False, True])
def test_communication_bound_is_enforced_and_corrected_without_changing_public_inputs(phase, pooled):
    records = prepared()
    agent_pool = None
    if pooled:
        profiles = [AgentProfile(pool_agent_id=identity, role="Author", capabilities=["composition"],
                                 prompt="Compose the complete public artifact.")
                    for identity in ("public_composer", "public_analyst")]
        agent_pool = AgentPoolSnapshot(profiles=profiles)
        records[0].candidates[0] = AgentSpec.model_validate({
            **records[0].candidates[0].model_dump(mode="json"),
            "pool_agent_id": profiles[0].pool_agent_id, "pool_agent_version": profiles[0].version,
        })
        records[2]["team"]["agents"][0] = records[0].candidates[0].model_dump(mode="json")
    before_records = [records[0].model_dump(mode="json"), records[1].model_dump(mode="json"),
                      copy.deepcopy(records[2])]
    valid = copy.deepcopy({"predict": before_records[0], "local_plan": before_records[1],
                           "reconcile": before_records[2]}[phase])
    holder = (valid["candidates"][0] if phase == "predict" else
              valid["team"]["agents"][0] if phase == "reconcile" else valid)
    holder["communication"] = "Publish the substantive ledger once.".ljust(1024)
    invalid = copy.deepcopy(valid)
    invalid_holder = (invalid["candidates"][0] if phase == "predict" else
                      invalid["team"]["agents"][0] if phase == "reconcile" else invalid)
    invalid_holder["communication"] += "x"
    task = PublicTask(task_id="synthetic-communication-bound",
                      question="Preserve all supplied facts: " + "public fact " * 2000,
                      attachments=["Public evidence: " + "source fact " * 2000])
    before_task = task.model_dump(mode="json")
    provider = Provider(replies=[invalid, valid])
    analyzer, ledger = bound(provider, agent_pool=agent_pool)
    invoke(analyzer, phase, task, records)
    assert len(provider.requests) == len(analyzer.call_records) == 2
    for request in provider.requests:
        schema = request["kwargs"]["response_format"]["json_schema"]["schema"]
        if phase == "local_plan":
            assert schema["properties"]["communication"]["maxLength"] == 1024
        else:
            agent_schema = schema["$defs"]["AgentSpec"]
            branches = agent_schema.get("anyOf", [agent_schema])
            assert all(branch["properties"]["communication"]["maxLength"] == 1024
                       for branch in branches)
            if pooled and phase == "predict":
                assert len(branches) == 3
        assert request["payload"]["task"] == before_task
        assert "state the handoff protocol once" in request["messages"][0]["content"]
    correction = provider.requests[1]["payload"]["response_correction"]
    assert "communication" in correction["validation_errors"][0]["message"]
    assert json.loads(correction["previous_response"]) == invalid
    assert [json.loads(row["response"]) for row in analyzer.call_records] == [invalid, valid]
    assert task.model_dump(mode="json") == before_task
    assert [records[0].model_dump(mode="json"), records[1].model_dump(mode="json"),
            records[2]] == before_records
    assert ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 20


def test_truncated_correction_keeps_full_audit_and_original_inputs_but_does_not_replay_tail():
    records = prepared()
    broken = '{"graph":{"rubrics":[{"requirement":"' + "BROKEN_PUBLIC_TAIL " * 4000
    provider = Provider(replies=[broken, records[0].model_dump(mode="json")],
                        finish_reasons=["length", "stop"])
    analyzer, ledger = bound(provider)
    task = PublicTask(task_id="synthetic-truncation", question="Preserve the original complete public request.")
    prediction = analyzer.predict(task)
    correction = provider.requests[1]["payload"]["response_correction"]
    assert "previous_response" not in correction
    assert correction["previous_response_sha256"] == hashlib.sha256(broken.encode()).hexdigest()
    assert correction["previous_response_characters"] == len(broken)
    assert correction["previous_response_finish_metadata"]["finish_reason"] == "length"
    assert "BROKEN_PUBLIC_TAIL" not in provider.requests[1]["messages"][1]["content"]
    assert provider.requests[1]["payload"]["task"] == task.model_dump(mode="json")
    assert analyzer.call_records[0]["response"] == broken
    assert prediction.candidates[0].max_calls is None
    assert ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 20


def test_nontruncated_invalid_correction_still_contains_the_original_response():
    records = prepared()
    bad = records[0].model_dump(mode="json")
    bad["candidates"][0]["max_calls"] = 1
    provider = Provider(replies=[bad, records[0].model_dump(mode="json")])
    analyzer, _ = bound(provider)
    analyzer.predict(PublicTask(task_id="synthetic-retained-response", question="Explain factors."))
    correction = provider.requests[1]["payload"]["response_correction"]
    assert json.loads(correction["previous_response"]) == bad
    assert "previous_response_sha256" not in correction


def test_large_json_decode_correction_is_digest_only_and_compact():
    records = prepared()
    broken = "{" + "\"graph\": [" + ("BROKEN " * 5000)
    provider = Provider(replies=[broken, records[0].model_dump(mode="json")])
    analyzer, _ = bound(provider)
    analyzer.predict(PublicTask(task_id="synthetic-large-json-error", question="Explain factors."))
    correction = provider.requests[1]["payload"]["response_correction"]
    assert correction["previous_response_characters"] == len(broken)
    assert correction["previous_response_sha256"] == hashlib.sha256(broken.encode()).hexdigest()
    assert "previous_response" not in correction
    assert "minified JSON object" in correction["instruction"]
    assert "BROKEN" not in provider.requests[1]["messages"][1]["content"]


def test_schema_mode_has_no_fallback_or_extra_attempt_when_provider_output_is_invalid():
    provider = Provider(replies=['{"graph":', '{"graph":', prepared()[0].model_dump(mode="json")],
                        finish_reasons=["length", "length", "stop"])
    analyzer, ledger = bound(provider)
    with pytest.raises(json.JSONDecodeError):
        analyzer.predict(PublicTask(task_id="synthetic-no-fallback", question="Explain factors."))
    assert len(provider.requests) == 2
    assert all(row["kwargs"]["response_format"]["type"] == "json_schema" for row in provider.requests)
    assert ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 20


def test_budget_failure_does_not_create_a_contract_retry_or_schema_fallback():
    provider = Provider()
    ledger = BudgetLedger(max_calls=None, max_tokens=1)
    analyzer, _ = bound(provider, ledger=ledger)
    with pytest.raises(BudgetExceeded):
        analyzer.predict(PublicTask(task_id="synthetic-hard-budget", question="Explain factors."))
    assert provider.requests == []
    assert ledger.snapshot()["model_calls"] == 0


def test_default_mode_leaves_requests_and_finite_method_caps_unchanged():
    records = prepared(max_calls=1)
    provider = Provider(prepared_records=records)
    analyzer, ledger = bound(provider, schema=False)
    result = analyzer.build(PublicTask(task_id="synthetic-default-planning", question="Explain factors."))
    assert result.team.total_max_calls == result.team.agents[0].max_calls == 1
    for row in provider.requests:
        assert "response_format" not in row["kwargs"]
        assert "planning_output_budget" not in row["payload"]
    assert all("response_format" not in row for row in analyzer.call_records)
    assert ledger.snapshot()["model_calls"] == 3


def test_pooled_identity_binding_remains_in_the_actual_strict_reconcile_schema():
    records = prepared()
    profile = AgentProfile(pool_agent_id="public_composer", role="Author", capabilities=["composition"],
                           prompt="Compose the complete public artifact.")
    candidate = AgentSpec.model_validate({
        **records[0].candidates[0].model_dump(mode="json"),
        "pool_agent_id": profile.pool_agent_id, "pool_agent_version": profile.version,
    })
    records[0].candidates[0] = candidate
    records[2]["team"]["agents"][0] = candidate.model_dump(mode="json")
    provider = Provider(prepared_records=records)
    analyzer, ledger = bound(provider, agent_pool=AgentPoolSnapshot(profiles=[profile]))
    result = analyzer.reconcile(PublicTask(task_id="synthetic-pool-schema", question="Explain factors."),
                                records[0], [])
    schema = provider.requests[0]["kwargs"]["response_format"]["json_schema"]["schema"]
    agent_schema = schema["$defs"]["AgentSpec"]["anyOf"][0]
    assert agent_schema["properties"]["agent_id"]["const"] == candidate.agent_id
    assert agent_schema["properties"]["pool_agent_id"]["const"] == profile.pool_agent_id
    assert agent_schema["properties"]["pool_agent_version"]["const"] == profile.version
    assert agent_schema["properties"]["max_calls"] == {"type": "null", "const": None}
    assert result.team.agents[0].pool_agent_id == profile.pool_agent_id
    assert ledger.snapshot()["model_calls"] == 1


def test_uncapped_structured_team_leaves_existing_execution_protocol_correction_room():
    records = prepared()
    provider = Provider(prepared_records=records)
    analyzer, ledger = bound(provider)
    task = PublicTask(task_id="synthetic-execution-room", question="Write a complete technical guide.")
    planned = analyzer.build(task)
    assert planned.team.agents[0].max_calls is planned.team.total_max_calls is None
    synth = JITHarnessSynthesizer()
    calls = []

    class ExecutionProvider:
        def __call__(self, messages, **kwargs):
            calls.append(copy.deepcopy({"messages": messages, "kwargs": kwargs}))
            reply = ('{"answer":"unfinished' if len(calls) == 1 else json.dumps({
                "answer": "A complete useful public artifact.", "evidence_ids": [], "checkpoints": {},
            }))
            return ChatMessage(role="assistant", content=reply)

        def get_token_counts(self):
            return {"input_token_count": 7, "output_token_count": 3}

    try:
        artifact = synth.synthesize(task, planned.graph, planned.team)
        result = TeamExecutor(lambda aid: MeteredModel(
            ExecutionProvider(), ledger, "execution", aid, 128), ledger=ledger).execute(
                task, planned.team, artifact)
        assert result.answer == "A complete useful public artifact."
        assert len(calls) == 2
        assert all(call["kwargs"]["max_tokens"] == 128 for call in calls)
        assert ledger.snapshot()["model_calls"] == 5
        assert ledger.snapshot()["tokens"] == 50
        assert planned.team.agents[0].max_calls is planned.team.total_max_calls is None
        assert "unfinished" not in calls[1]["messages"][-1]["content"]
    finally:
        for agent in synth._agents.values():
            path = agent.workspace_dir.resolve()
            assert path.parent.name == "workspaces" and path.name.startswith("mas_")
            if path.is_dir():
                shutil.rmtree(path)
