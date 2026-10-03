"""JSON generation constraints preserve strict parsing and natural-text call paths."""

import copy
import json
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig, NativeModels
from jit_mas.execution import TeamMemory, TeamPlanning, TeamServices, _run_agent
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.kernel.types import TaskInput, ToolSelection
from scripts.models.base import ChatMessage


def test_native_planning_json_default_preserves_other_roles_and_schema_override(monkeypatch):
    import openai

    requests = []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        text = '{"ok":true}'
        message = SimpleNamespace(content=text, tool_calls=None,
                                  model_dump=lambda **unused: {"role": "assistant", "content": text})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
                               model="offline", id="offline-request", system_fingerprint="offline")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OFFLINE_JSON_FORMAT_KEY", "offline-credential")
    monkeypatch.setenv("JIT_MAS_MODEL_ATTEMPTS", "1")
    roles = ("global", "local", "meta", "exec", "judge")
    config = MASConfig(backend="native_jit", unsafe_local=True, models={
        role: ModelConfig(model="offline", endpoint="https://example.invalid/v1",
                          key_env="OFFLINE_JSON_FORMAT_KEY", max_tokens=64)
        for role in roles})
    provider = NativeModels(config)
    ledger = BudgetLedger()
    schema_format = {"type": "json_schema", "json_schema": {
        "name": "OfflineRecord", "strict": True, "schema": {"type": "object"}}}
    for role in roles:
        model = provider.create(role, role, ledger, "format-check")
        model([{"role": "user", "content": "Return a JSON object."}])
        if role in {"global", "local"}:
            assert requests[-1]["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in requests[-1]
        model([{"role": "user", "content": "Return a JSON object."}],
              response_format=schema_format)
        assert requests[-1]["response_format"] == schema_format
    assert len(requests) == ledger.snapshot()["model_calls"] == 10
    assert ledger.snapshot()["tokens"] == 140


def execute_protocol(mode, replies, *, max_calls=1):
    calls = []
    values = iter(replies)

    class Model:
        def __call__(self, messages, **kwargs):
            calls.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
            return ChatMessage(role="assistant", content=next(values))

        def get_token_counts(self):
            return {"input_token_count": 10, "output_token_count": 4}

    ledger = BudgetLedger(max_calls=max_calls, max_tokens=100_000)
    model = MeteredModel(Model(), ledger, "execution", "writer", 2048)
    task = PublicTask(task_id="public-copy", question="Write an English script in a conversational voice.")
    agent = AgentSpec(agent_id="writer", role="Writer", capability="writing",
                      max_calls=max_calls, max_tokens=2048)
    team = TeamSpec(agents=[agent], synthesizer_id="writer", execution_mode=mode,
                    total_max_calls=max_calls)
    services = TeamServices(lambda aid: model, task, RubricGraph(rubrics=[]),
                            ledger=ledger, execution_mode=mode)
    memory = TeamMemory()
    memory.initialize("coordinator", TaskInput(task=task.question))
    context = SimpleNamespace(
        memory=memory, planning=TeamPlanning(), model=None, prompt_templates={},
        tool_policy=SimpleNamespace(select_tools=lambda *args: ToolSelection(tools={})),
        get_tool_schemas=lambda selected: "[]")
    result = _run_agent(agent.model_dump(mode="json"), team.model_dump(mode="json"),
                        context, services)
    return result, calls, ledger.snapshot(), services


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
def test_execution_json_generation_preserves_multiline_artifact(mode):
    answer = 'A first-person scene with a "quoted phrase".\n\nA complete second paragraph.'
    reply = json.dumps({"answer": answer, "checkpoints": {}, "evidence_ids": [], "continue": False})
    result, calls, budget, services = execute_protocol(mode, [reply])
    assert result.terminated_reason == "final_answer" and result.answer == answer
    assert len(calls) == budget["model_calls"] == 1
    assert calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
    assert calls[0]["kwargs"]["max_tokens"] == 2048
    assert json.loads(result.trajectory[0].model_output_messages.content)["answer"] == answer
    assert [event for event in services.events if event["kind"] == "final_answer"]


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
@pytest.mark.parametrize("reply", ['{"answer":"A paragraph.\nAnother paragraph."}',
                                  '{"answer":"A complete script."}]}'])
def test_json_mode_does_not_coerce_malformed_text_or_expand_call_budget(mode, reply):
    result, calls, budget, services = execute_protocol(mode, [reply])
    assert result.terminated_reason == "error" and result.answer is None
    assert len(calls) == budget["model_calls"] == 1
    assert calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
    assert result.trajectory[0].model_output_messages.content == reply
    assert result.trajectory[0].error
    assert not [event for event in services.events if event["kind"] == "final_answer"]


def test_iterative_json_correction_preserves_failed_response_and_meters_both_calls():
    malformed = '{"answer":"A paragraph.\nAnother paragraph."}'
    answer = "A paragraph.\nAnother paragraph."
    valid = json.dumps({"answer": answer, "checkpoints": {}, "evidence_ids": [], "continue": False})
    result, calls, budget, services = execute_protocol(
        "iterative_shared_ledger", [malformed, valid], max_calls=2)
    assert result.terminated_reason == "final_answer" and result.answer == answer
    assert len(calls) == budget["model_calls"] == 2
    assert budget["tokens"] == 28
    assert result.trajectory[0].error
    assert result.trajectory[0].model_output_messages.content == malformed
    assert all(call["kwargs"]["response_format"] == {"type": "json_object"} for call in calls)
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
