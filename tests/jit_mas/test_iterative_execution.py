"""Iterative execution publishes completed answers and cites delivered observations."""

import copy
import json
import time
from types import SimpleNamespace

import pytest

from jit_mas.agent_pool import seed_pool
from jit_mas.budget import BudgetLedger
from jit_mas.execution import (
    TeamMemory,
    TeamPlanning,
    TeamServices,
    _SinglePassModel,
    _run_agent_iterative,
    run_team,
)
from jit_mas.schemas import AgentHarnessPolicy, AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.kernel.types import TaskInput, ToolSelection
from scripts.models.base import ChatMessage


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        reply = next(self.replies)
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return ChatMessage(role="assistant", content=reply if isinstance(reply, str) else json.dumps(reply))


def make_team(*, tools=(), contributors=False, checkpoints=()):
    agents = []
    if contributors:
        agents.append(AgentSpec(agent_id="searcher", role="Searcher", capability="research",
                                max_calls=None))
    agents.append(AgentSpec(agent_id="writer", role="Writer", capability="writing",
                            tools=list(tools), max_calls=None, checkpoints=list(checkpoints),
                            depends_on=["searcher"] if contributors else []))
    return TeamSpec(execution_mode="iterative_shared_ledger", agents=agents,
                    synthesizer_id="writer", total_max_calls=None)


def make_services(team, models, *, tools=None, pool=None):
    installed = tools or {}
    services = TeamServices(lambda agent_id: models[agent_id],
                            PublicTask(task_id="iterative-review", question="Write a researched guide.",
                                       tools=list(installed)), RubricGraph(rubrics=[]),
                            execution_mode="iterative_shared_ledger", agent_pool=pool)
    memory = TeamMemory()
    memory.initialize("coordinator", TaskInput(task="Write a guide."))
    context = SimpleNamespace(
        tool_policy=SimpleNamespace(select_tools=lambda *args: ToolSelection(tools=installed)),
        memory=memory,
        planning=TeamPlanning(),
        model=None,
        prompt_templates={},
        get_tool_schemas=lambda selected: json.dumps([
            {"name": name, "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}} for name in selected]),
        execute_tool=lambda name, arguments: installed[name](**arguments),
    )
    return services, context


def run_writer(team, services, context):
    return _run_agent_iterative(team.agents[-1].model_dump(mode="json"),
                                team.model_dump(mode="json"), context, services)


def test_writer_can_cite_delivered_upstream_artifact():
    team = make_team(contributors=True)
    observed = {}

    def complete(messages):
        payload = json.loads(messages[1]["content"])
        upstream = payload["shared_ledger"]["contributions"][0]
        assert upstream["answer"] == "Supported findings"
        assert payload["scheduling"]["reactivate_completed_agents"] is False
        return {"answer": "Final guide", "evidence_ids": [upstream["event_id"]]}

    model = ScriptedModel([complete])
    services, context = make_services(team, {"writer": model})
    observed["artifact"] = services.event("searcher", "artifact_published", "Supported findings")
    services.artifacts["searcher"] = {"answer": "Supported findings", "event_id": observed["artifact"],
                                      "version": 1, "complete": True}
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer" and result.answer == "Final guide"
    final_event = next(event for event in services.events if event["kind"] == "final_answer")
    assert final_event["parent_event_ids"] == [observed["artifact"]]


@pytest.mark.parametrize("failure", ["truncated JSON", RuntimeError("provider failure")])
def test_failed_writer_draft_never_becomes_final_submission(failure):
    team = make_team()
    model = ScriptedModel([{"answer": "Unfinished draft", "continue": True}, failure])
    services, context = make_services(team, {"writer": model})
    result = run_team("Write a guide.", context, team, services)
    assert result.answer is None and result.terminated_reason == "error"
    assert result.sub_runs[0].answer is None
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_timed_out_writer_draft_never_becomes_final_submission():
    team = make_team()

    def draft(messages):
        services.started_at = time.monotonic() - services.timeout_seconds - 1
        return {"answer": "Unfinished draft", "continue": True}

    model = ScriptedModel([draft])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "timeout"
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_failed_contributor_draft_blocks_downstream_writer():
    team = make_team(contributors=True)
    producer = ScriptedModel([{"answer": "Unfinished contribution", "continue": True}, "bad JSON"])
    writer = ScriptedModel([{"answer": "This must not run"}])
    services, context = make_services(team, {"searcher": producer, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.answer is None and not writer.calls
    assert result.sub_runs[-1].terminated_reason == "dependency_failed"
    assert "searcher" not in services.artifacts


def test_new_peer_message_is_delivered_and_citable_on_next_turn():
    team = make_team(contributors=True)
    published = {}

    def draft(messages):
        published["message"] = services.event("searcher", "peer_message", {"content": "Check uncertainty"},
                                              recipient="writer")
        services.event("searcher", "peer_message", {"content": "Another recipient"}, recipient="other")
        return {"answer": "Initial draft", "continue": True}

    def complete(messages):
        update = next(message["content"] for message in messages
                      if message["content"].startswith("Updated shared ledger: "))
        ledger = json.loads(update.removeprefix("Updated shared ledger: "))
        assert [message["event_id"] for message in ledger["communications"]] == [published["message"]]
        return {"answer": "Revised guide", "evidence_ids": [published["message"]]}

    model = ScriptedModel([draft, complete])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Revised guide" and result.terminated_reason == "final_answer"
    assert published["message"] in result.metadata["observed_evidence_ids"]


def test_peer_observation_published_during_call_cannot_be_cited_before_delivery():
    team = make_team(contributors=True)

    def unsupported_completion(messages):
        unseen = services.event("searcher", "retrieved", {"output": "Evidence published after call started"})
        return {"answer": "Unsupported guide", "evidence_ids": [unseen]}

    model = ScriptedModel([unsupported_completion])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert "not observed" in str(result.trajectory[-1].error)


def test_unknown_message_recipient_fails_before_external_tool_dispatch():
    team = make_team(tools=["search"])
    dispatched = []
    model = ScriptedModel([{"tools": [
        {"name": "search", "arguments": {"query": "sources"}},
        {"name": "send_message", "arguments": {"recipient": "invented", "content": "Please help"}},
    ]}])
    services, context = make_services(team, {"writer": model},
                                      tools={"search": lambda query: dispatched.append(query)})
    result = run_writer(team, services, context)
    assert result.answer is None and not dispatched
    assert "exact peer agent_id" in str(result.trajectory[-1].error)


def test_tool_results_are_citable_after_delivery_and_checks_wait_for_completion():
    team = make_team(tools=["search"], checkpoints=["source_check"])

    def request(messages):
        payload = json.loads(messages[1]["content"])
        assert json.loads(payload["allowed_tool_schemas"])[0]["name"] == "search"
        assert payload["communication_tools"][0]["name"] == "send_message"
        return {"tools": [{"name": "search", "arguments": {"query": "direct sources"}}]}

    def complete(messages):
        observation = json.loads(messages[-1]["content"].removeprefix("Public observations: "))[0]
        assert observation["output"] == "Retrieved evidence"
        evidence_id = observation["event_id"]
        return {"answer": "Supported guide", "evidence_ids": [evidence_id], "checkpoints": {
            "source_check": {"status": "passed", "reason": "Read the retrieved source",
                             "evidence_ids": [evidence_id]}}}

    model = ScriptedModel([request, complete])
    services, context = make_services(team, {"writer": model},
                                      tools={"search": lambda query: "Retrieved evidence"})
    result = run_writer(team, services, context)
    assert result.answer == "Supported guide" and len(model.calls) == 2
    assert result.trajectory[0].tool_calls[0].name == "search"


def test_terminal_turn_cannot_reuse_a_previous_draft_without_answer():
    team = make_team()
    model = ScriptedModel([{"answer": "Initial draft", "continue": True}, {"continue": False}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert "complete answer" in str(result.trajectory[-1].error)


def test_recent_memory_receives_tool_result_and_peer_update_together():
    team = make_team(tools=["search"], contributors=True)
    pool = seed_pool()
    team.agents[-1] = AgentSpec(**{**team.agents[-1].model_dump(mode="json"),
        "pool_agent_id": "writer", "pool_agent_version": 1,
        "harness": AgentHarnessPolicy(memory_policy="recent", memory_window=1)})
    published = {}

    def search(query):
        published["message"] = services.event("searcher", "peer_message", {"content": "Verify this source"},
                                              recipient="writer")
        return "Retrieved evidence"

    def complete(messages):
        assert len(messages) == 3
        observations, refreshed = messages[-1]["content"].split("\nUpdated shared ledger: ")
        observation = json.loads(observations.removeprefix("Public observations: "))[0]
        ledger = json.loads(refreshed)
        assert ledger["communications"][0]["event_id"] == published["message"]
        return {"answer": "Supported guide", "evidence_ids": [observation["event_id"], published["message"]]}

    model = ScriptedModel([{"tools": [{"name": "search", "arguments": {"query": "sources"}}]}, complete])
    services, context = make_services(team, {"writer": model}, tools={"search": search}, pool=pool)
    result = run_writer(team, services, context)
    assert result.answer == "Supported guide" and result.terminated_reason == "final_answer"


@pytest.mark.parametrize("role_limit", [1, None])
def test_iterative_role_call_limit_discards_an_unfinished_draft(role_limit):
    team = make_team()
    team.agents[-1].max_calls = role_limit
    model = ScriptedModel([{"answer": "Unfinished draft", "continue": True},
                           {"answer": "Completed guide", "continue": False}])
    services, context = make_services(team, {"writer": model})
    services.model_factory = lambda agent_id: _SinglePassModel(
        model, services, team.agents[-1].model_dump(mode="json"), team.model_dump(mode="json"))
    result = run_writer(team, services, context)
    if role_limit is None:
        assert result.answer == "Completed guide" and result.terminated_reason == "final_answer"
        assert len(model.calls) == services.calls == 2
    else:
        assert result.answer is None and result.terminated_reason == "error"
        assert len(model.calls) == services.calls == 1
        assert "AgentSpec.max_calls exhausted" in str(result.trajectory[-1].error)
        assert "writer" not in services.artifacts
        assert not any(event["kind"] == "final_answer" for event in services.events)


def test_single_pass_team_supports_an_unlimited_tool_budget():
    team = TeamSpec(agents=[
        AgentSpec(agent_id="searcher", role="Searcher", capability="research", tools=["search"]),
        AgentSpec(agent_id="writer", role="Writer", capability="writing", depends_on=["searcher"]),
    ], synthesizer_id="writer", total_max_calls=2)
    searcher = ScriptedModel([{"tools": [{"name": "search", "arguments": {"query": "source"}}]}])
    writer = ScriptedModel([{"answer": "Supported guide"}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer},
                                      tools={"search": lambda query: "Retrieved source"})
    services.execution_mode = "single_pass"
    services.ledger = BudgetLedger(max_tool_calls=None)
    result = run_team("Write a guide.", context, team, services)
    assert result.answer == "Supported guide" and result.terminated_reason == "final_answer"
    assert services.ledger.snapshot()["tool_calls"] == 1
    assert len(searcher.calls) == len(writer.calls) == 1
    assert services.shared_ledger["tool_evidence"][0]["content"]["output"] == "Retrieved source"
