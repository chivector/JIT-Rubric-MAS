"""Peer requests resume role histories and final submission waits for collaboration."""

import copy
import json
import time
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import TeamMemory, TeamPlanning, TeamServices, _SinglePassModel, run_team
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.kernel.types import TaskInput, ToolSelection
from scripts.models.base import ChatMessage


class RoleModel:
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
        return ChatMessage(role="assistant", content=json.dumps(reply))


def setup_roles(searcher_replies, writer_replies, *, max_calls=None, ledger=None):
    team = TeamSpec(execution_mode="iterative_shared_ledger", agents=[
        AgentSpec(agent_id="searcher", role="Searcher", capability="research", max_calls=max_calls),
        AgentSpec(agent_id="writer", role="Writer", capability="writing", depends_on=["searcher"],
                  max_calls=None),
    ], synthesizer_id="writer", total_max_calls=None, max_parallel=1)
    models = {"searcher": RoleModel(searcher_replies), "writer": RoleModel(writer_replies)}
    task = PublicTask(task_id="cooperative", question="Produce a supported guide.")
    services = TeamServices(lambda agent_id: models[agent_id], task, RubricGraph(rubrics=[]),
                            execution_mode="iterative_shared_ledger", ledger=ledger)
    memory = TeamMemory()
    memory.initialize("coordinator", TaskInput(task=task.question))
    context = SimpleNamespace(memory=memory, planning=TeamPlanning(), model=None,
        prompt_templates={}, get_tool_schemas=lambda tools: "[]",
        tool_policy=SimpleNamespace(select_tools=lambda *args: ToolSelection(tools={})))
    if max_calls is not None or ledger is not None:
        agents = {agent.agent_id: agent.model_dump(mode="json") for agent in team.agents}

        def model_factory(agent_id):
            model = models[agent_id]
            if ledger is not None:
                model = MeteredModel(model, ledger, "execution", agent_id, max_tokens=128)
            return _SinglePassModel(model, services, agents[agent_id], team.model_dump(mode="json"))

        services.model_factory = model_factory
    return team, models, services, context


def ask(recipient, content, answer=None):
    reply = {"continue": True, "tools": [
        {"name": "send_message", "arguments": {"recipient": recipient, "content": content}}]}
    if answer is not None:
        reply["answer"] = answer
    return reply


def latest_ledger(messages):
    for message in reversed(messages):
        if "Updated shared ledger: " in message["content"]:
            return json.loads(message["content"].split("Updated shared ledger: ")[-1])
    return json.loads(messages[1]["content"])["shared_ledger"]


def test_writer_reactivates_completed_searcher_with_private_history_preserved():
    private_marker = "Only the Searcher sees this private reasoning marker"

    def revise(messages):
        assert private_marker in json.dumps(messages)
        ledger = latest_ledger(messages)
        request = ledger["communications"][-1]
        assert request["agent_id"] == "writer" and "date" in request["content"]["content"]
        return {"answer": "Verified publication date", "evidence_ids": [request["event_id"]]}

    def submit(messages):
        assert private_marker not in json.dumps(messages)
        contribution = latest_ledger(messages)["contributions"][0]
        assert contribution["answer"] == "Verified publication date"
        return {"answer": "Final verified guide", "evidence_ids": [contribution["event_id"]]}

    team, models, services, context = setup_roles(
        [{"answer": "Initial search findings", "reasoning": private_marker}, revise],
        [ask("searcher", "Please verify the publication date", "Initial draft"), submit])
    result = run_team("Produce a guide", context, team, services)
    assert result.answer == "Final verified guide"
    assert [len(model.calls) for model in models.values()] == [2, 2]
    assert [len(run.trajectory) for run in result.sub_runs] == [2, 2]
    assert [step.step_number for step in result.sub_runs[0].trajectory] == [1, 2]
    assert services.artifacts["searcher"]["version"] == 2
    resumed = [event for event in services.events if event["kind"] == "agent_resumed"]
    assert [event["agent_id"] for event in resumed] == ["searcher"]
    final_events = [event for event in services.events if event["kind"] == "final_answer"]
    assert len(final_events) == 1
    assert final_events[0]["parent_event_ids"] == [services.artifacts["searcher"]["event_id"]]
    assert result.metadata["scheduling"]["reactivate_completed_agents"] is True


@pytest.mark.parametrize("requests", [1, 5, 12])
def test_peer_clarification_has_no_fixed_round_limit(requests):
    searcher = [{"answer": f"Findings version {version}"} for version in range(requests + 1)]
    writer = [ask("searcher", f"Clarify item {number}") for number in range(requests)]
    writer.append({"answer": "Complete guide"})
    team, models, services, context = setup_roles(searcher, writer)
    result = run_team("Produce a guide", context, team, services)
    assert result.answer == "Complete guide"
    assert all(len(model.calls) == requests + 1 for model in models.values())
    assert len([event for event in services.events if event["kind"] == "agent_resumed"]) == requests


def test_upstream_can_request_writer_feedback_before_initial_dependency_completes():
    def inspect_feedback(messages):
        ledger = latest_ledger(messages)
        assert any("Include uncertainty" in message["content"]["content"]
                   for message in ledger["communications"])
        return {"answer": "Evidence with uncertainty explained"}

    team, models, services, context = setup_roles(
        [ask("writer", "Which evidence is missing?", "Initial evidence draft"), inspect_feedback],
        [ask("searcher", "Include uncertainty in the source summary", "Writer feedback"),
         {"answer": "Final guide with uncertainty"}])
    result = run_team("Produce a guide", context, team, services)
    assert result.answer == "Final guide with uncertainty"
    assert all(len(model.calls) == 2 for model in models.values())
    assert all(run.terminated_reason in {"subtask_complete", "final_answer"}
               for run in result.sub_runs)


def test_resumed_peer_failure_invalidates_prior_draft_and_blocks_submission():
    team, models, services, context = setup_roles(
        [{"answer": "Initial findings"}, RuntimeError("Search provider failed")],
        [ask("searcher", "Check this conflicting claim", "Unfinished writer draft"),
         {"answer": "This must never submit"}])
    result = run_team("Produce a guide", context, team, services)
    assert result.answer is None
    assert result.sub_runs[-1].terminated_reason == "dependency_failed"
    assert len(models["writer"].calls) == 1
    assert not services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_writer_completed_draft_resumes_when_peer_artifact_changes_afterward():
    def final_revision(messages):
        ledger = latest_ledger(messages)
        assert ledger["contributions"][0]["answer"] == "Revised evidence after writer feedback"
        return {"answer": "Final guide using the revised evidence"}

    team, models, services, context = setup_roles(
        [{"answer": "Initial findings"}, ask("writer", "Please review this revised evidence", "Evidence draft"),
         {"answer": "Revised evidence after writer feedback"}],
        [ask("searcher", "Check the evidence"), {"answer": "Writer draft and feedback"}, final_revision])
    result = run_team("Produce a guide", context, team, services)
    assert result.answer == "Final guide using the revised evidence"
    assert all(len(model.calls) == 3 for model in models.values())
    assert len([event for event in services.events if event["kind"] == "final_answer"]) == 1
    resumed = [event for event in services.events if event["kind"] == "agent_resumed"]
    assert any(event["agent_id"] == "writer" and event["content"]["reason"] == "updated_team_artifacts"
               for event in resumed)
    final_event = next(event for event in services.events if event["kind"] == "final_answer")
    assert final_event["event_id"] in result.sub_runs[-1].metadata["event_ids"]


def test_reactivated_role_retains_its_configured_call_budget():
    team, models, services, context = setup_roles(
        [{"answer": "Initial findings"}, {"answer": "Unused response"}],
        [ask("searcher", "Check the dates", "Writer draft")], max_calls=1)
    result = run_team("Produce a guide", context, team, services)
    assert result.answer is None
    assert len(models["searcher"].calls) == 1
    assert services.call_counts == {"searcher": 1, "writer": 1}
    assert "max_calls exhausted" in str(result.sub_runs[0].trajectory[-1].error)
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_reactivated_role_timeout_cannot_leave_a_final_submission():
    def expire(messages):
        services.started_at = time.monotonic() - services.timeout_seconds - 1
        return {"answer": "Unfinished revision", "continue": True}

    team, models, services, context = setup_roles(
        [{"answer": "Initial findings"}, expire],
        [ask("searcher", "Verify this evidence", "Writer draft"), {"answer": "Unused answer"}])
    result = run_team("Produce a guide", context, team, services)
    assert result.answer is None and services.cancelled.is_set()
    assert len(models["writer"].calls) == 1
    assert not services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_cooperative_role_traces_match_metered_calls_and_communication():
    ledger = BudgetLedger(max_calls=None, max_tokens=200_000, max_tool_calls=None)
    team, models, services, context = setup_roles(
        [{"answer": "Initial findings"}, {"answer": "Verified findings"}],
        [ask("searcher", "Verify dates"), {"answer": "Final guide"}], ledger=ledger)
    result = run_team("Produce a guide", context, team, services)
    assert result.answer == "Final guide"
    assert services.calls == sum(len(run.trajectory) for run in result.sub_runs) == 4
    assert ledger.snapshot()["model_calls"] == 4
    assert ledger.snapshot()["by_stage"]["execution"]["communication_bytes"] > 0
    assert all(step.model_input_messages is not None and step.model_output_messages is not None
               for run in result.sub_runs for step in run.trajectory)
    for run in result.sub_runs:
        assert run.metadata["model_calls"] == len(run.trajectory)
