"""Truncated model output is never a submission or an unbounded retry."""

import json
import shutil

import pytest

from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import ResponseProtocolError, TeamExecutor, _parse_response
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


@pytest.mark.parametrize("content", [
    '{"answer":"unfinished', '"plain answer"', "plain answer", "[]", "",
    '{"answer":42}', '{"answer":""}', '{"checkpoints":[]}',
    '{"evidence_ids":"event-1"}', '{"tools":[{}]}',
    '{"tools":[{"name":"final_answer","arguments":{"answer":123}}]}',
    '{"tools":[{"name":"complete","arguments":{"answer":""}}]}',
    '{"tools":[{"name":"final_answer","arguments":"{\\"checkpoints\\":[]}"}]}',
    '{"tools":[{"name":"lookup","arguments":[]}]}',
])
def test_incomplete_or_malformed_protocol_is_not_coerced_into_an_answer(content):
    with pytest.raises(ResponseProtocolError):
        _parse_response(ChatMessage(role="assistant", content=content))


def test_complete_json_and_fenced_json_remain_supported():
    response = {"answer": "Complete.", "checkpoints": {"Checked": True}, "evidence_ids": []}
    assert _parse_response(response) == response
    assert _parse_response("```json\n" + json.dumps(response) + "\n```") == response


@pytest.mark.parametrize("payload", [
    {"answer": "The narrative ends with ", "a quoted phrase": "the rest of the narrative."},
    {"tools": [{"name": "final_answer", "arguments": {
        "answer": "The narrative ends with ", "a quoted phrase": "the rest of the narrative."}}]},
    {"tools": [{"name": "complete", "arguments": {
        "answer": "Contribution summary.", "unexpected_body": "Further useful facts."}}]},
])
def test_unknown_fields_cannot_silently_drop_substantive_completion_content(payload):
    with pytest.raises(ResponseProtocolError, match="Unknown execution response fields"):
        _parse_response(ChatMessage(role="assistant", content=json.dumps(payload)))


def test_documented_and_legacy_private_execution_fields_remain_supported():
    response = {"answer": "Complete.", "checkpoints": {}, "evidence_ids": [],
                "ledger": None, "continue": False, "tools": [],
                "think": {"assumption": "private"}, "reasoning": "Retained private context."}
    assert _parse_response(response) == response
    completion = {key: value for key, value in response.items() if key != "tools"}
    tool_response = {"tools": [{"name": "final_answer", "arguments": completion}]}
    assert _parse_response(tool_response) == tool_response


@pytest.fixture
def execution_case():
    task = PublicTask(task_id="output-budget", question="Write a self-contained technical guide.")
    synth = JITHarnessSynthesizer()
    ledger = BudgetLedger()

    def execute(replies, max_calls=1):
        team = TeamSpec(agents=[AgentSpec(agent_id="final", role="Editor", capability="writing",
            max_calls=max_calls, max_tokens=4096)], synthesizer_id="final", total_max_calls=max_calls)
        artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team)
        calls = []

        class Model:
            def __call__(self, messages, **kwargs):
                calls.append({"messages": messages, "kwargs": kwargs})
                reply = replies[len(calls) - 1]
                if isinstance(reply, Exception):
                    raise reply
                return ChatMessage(role="assistant", content=reply)

            def get_token_counts(self):
                return {"input_token_count": 5, "output_token_count": 3}

        result = TeamExecutor(lambda aid: MeteredModel(Model(), ledger, "execution", aid, 128),
                              ledger=ledger).execute(task, team, artifact)
        return result, calls, ledger.snapshot()

    try:
        yield execute
    finally:
        for agent in synth._agents.values():
            path = agent.workspace_dir.resolve()
            assert path.parent.name == "workspaces" and path.name.startswith("mas_")
            if path.is_dir():
                shutil.rmtree(path)


def test_actual_output_cap_and_single_call_are_visible_and_metered(execution_case):
    truncated = '{"answer":"unfinished'
    result, calls, budget = execution_case([truncated, '{"answer":"A shorter complete guide."}'])
    assert result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["output_budget"] == {"max_tokens_per_response": 128, "max_model_calls": 1}
    assert all(call["kwargs"]["max_tokens"] == 128 for call in calls)
    assert "hard ceiling of 128" in calls[0]["messages"][0]["content"]
    assert "final deliverable itself" in calls[0]["messages"][0]["content"]
    assert "only model call" in calls[0]["messages"][-1]["content"]
    assert "ResponseProtocolError" in result.sub_runs[0].trajectory[0].observations
    assert "no role recall" in result.sub_runs[0].trajectory[0].observations


def test_exhausted_protocol_budget_never_publishes_truncated_answer(execution_case):
    result, calls, budget = execution_case(['{"answer":"unfinished'], max_calls=1)
    assert result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    assert result.sub_runs[0].terminated_reason == "error"
    assert not any(event["kind"] == "final_answer" for event in result.metadata["events"])


def test_transport_failure_does_not_consume_a_protocol_retry(execution_case):
    result, calls, budget = execution_case([RuntimeError("provider unavailable")])
    assert result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    assert result.sub_runs[0].terminated_reason == "error"


def test_tool_style_terminal_completion_cannot_bypass_answer_validation(execution_case):
    result, calls, budget = execution_case([
        '{"tools":[{"name":"final_answer","arguments":{"answer":123}}]}',
        '{"tools":[{"name":"final_answer","arguments":{"answer":"Valid final answer."}}]}'])
    assert result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    assert "ResponseProtocolError" in result.sub_runs[0].trajectory[0].observations


def test_invalid_terminal_is_checked_before_any_prior_tool_side_effect(execution_case):
    result, calls, budget = execution_case([json.dumps({"tools": [
        {"name": "raise_issue", "arguments": {"content": "Must not be published"}},
        {"name": "final_answer", "arguments": {"answer": 123}}]}),
        '{"answer":"A valid correction."}'])
    assert result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    assert budget["tool_calls"] == 0
    assert not any(event["kind"] == "issue" for event in result.metadata["events"])
