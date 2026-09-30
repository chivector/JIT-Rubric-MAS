"""MAS message histories must traverse the real JIT client request preparation."""

import copy
import json
import shutil
from types import SimpleNamespace

import pytest

from scripts.models.base import get_clean_message_list


def test_adjacent_text_and_converted_roles_preserve_boundaries_without_mutation():
    messages = [{"role": "system", "content": "policy"},
                {"role": "user", "content": "task"},
                {"role": "tool-response", "content": "observation"},
                {"role": "user", "content": "Allowed tool schemas: []"}]
    before = copy.deepcopy(messages)
    clean = get_clean_message_list(messages, {"tool-response": "user"})
    assert clean == [{"role": "system", "content": "policy"},
                     {"role": "user", "content": "task\n\nobservation\n\nAllowed tool schemas: []"}]
    assert messages == before


@pytest.mark.parametrize("text_first", [True, False])
def test_mixed_content_preserves_images_and_text(text_first):
    parts = [{"type": "text", "text": "caption"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    contents = ["instruction", parts] if text_first else [parts, "instruction"]
    messages = [{"role": "user", "content": content} for content in contents]
    before = copy.deepcopy(messages)
    clean = get_clean_message_list(messages)
    expected = ([{"type": "text", "text": "instruction"}] + parts if text_first
                else parts + [{"type": "text", "text": "instruction"}])
    assert clean[0]["content"] == expected
    assert messages == before


def test_flatten_preserves_all_text_parts_and_strings():
    clean = get_clean_message_list([
        {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
        {"role": "user", "content": "three"}], flatten_messages_as_text=True)
    assert clean == [{"role": "user", "content": "one\n\ntwo\n\nthree"}]
    with pytest.raises(ValueError, match="non-text"):
        get_clean_message_list([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "example"}}]}], flatten_messages_as_text=True)


def test_team_executes_through_real_client_preflight_without_network(monkeypatch):
    from openai.types.chat import ChatCompletion
    from jit_mas.bridge import JITHarnessSynthesizer
    from jit_mas.budget import BudgetLedger, MeteredModel
    from jit_mas.execution import TeamExecutor
    from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
    from scripts.models.openai_server import OpenAIServerModel

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        assert "Allowed tool schemas: []" in kwargs["messages"][-1]["content"]
        assert len(kwargs["messages"]) == 2
        return ChatCompletion(id="offline-response", created=0, model="offline", object="chat.completion",
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                      "content": json.dumps({"answer": "Checked response", "checkpoints": {"Done": True}})}}],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    task = PublicTask(task_id="client-preflight", question="Compare two technical choices.")
    team = TeamSpec(agents=[AgentSpec(agent_id="worker", role="Writer", capability="writing", max_calls=1,
            checkpoints=["Done"]), AgentSpec(agent_id="final", role="Synthesizer", capability="synthesis",
            depends_on=["worker"], max_calls=1, checkpoints=["Done"])],
        synthesizer_id="final", total_max_calls=2)
    synth = JITHarnessSynthesizer()
    ledger = BudgetLedger()

    def factory(aid):
        model = OpenAIServerModel(model_id="offline", api_base="https://example.invalid/v1",
                                  api_key="offline", max_attempts=1)
        return MeteredModel(model, ledger, "execution", aid)

    try:
        artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team)
        result = TeamExecutor(factory, ledger=ledger).execute(task, team, artifact)
        assert result.terminated_reason == "final_answer"
        assert len(result.sub_runs) == len(requests) == 2
        assert ledger.snapshot()["tokens"] == 16
        assert all(not record["estimated"] for record in ledger.snapshot()["records"] if record["kind"] == "model")
        assert "Checked response" not in requests[0]["messages"][1]["content"]
        assert "Checked response" in requests[1]["messages"][1]["content"]
    finally:
        for agent in synth._agents.values():
            path = agent.workspace_dir.resolve()
            assert path.parent.name == "workspaces" and path.name.startswith("mas_")
            if path.is_dir():
                shutil.rmtree(path)
