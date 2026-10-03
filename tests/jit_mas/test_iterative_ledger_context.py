"""Lossless current ledger delivery without replaying superseded snapshots."""

import copy
import json
from types import SimpleNamespace

import tiktoken

from jit_mas.agent_pool import seed_pool
from jit_mas.execution import (
    TeamMemory, TeamPlanning, TeamServices, _iterative_public_ledger,
    _replace_iterative_ledger_snapshot, _run_agent_iterative, content_hash,
)
from jit_mas.schemas import AgentHarnessPolicy, AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.kernel.types import TaskInput, ToolSelection
from scripts.models.base import ChatMessage


def _setup(respond, *, recent=False):
    calls, outputs = [], []

    class Model:
        def __call__(self, messages, **_kwargs):
            calls.append(copy.deepcopy(messages))
            reply = respond(messages, len(calls))
            output = json.dumps(reply)
            outputs.append(output)
            return ChatMessage(role="assistant", content=output)

    writer = AgentSpec(agent_id="writer", role="Writer", capability="writing",
                       max_calls=None, tools=["search"], depends_on=["searcher"])
    pool = None
    if recent:
        pool = seed_pool()
        writer = AgentSpec(**{**writer.model_dump(mode="json"),
            "pool_agent_id": "writer", "pool_agent_version": 1,
            "harness": AgentHarnessPolicy(memory_policy="recent", memory_window=1)})
    team = TeamSpec(execution_mode="iterative_shared_ledger", agents=[
        AgentSpec(agent_id="searcher", role="Searcher", capability="research", max_calls=None),
        writer,
    ], synthesizer_id="writer", total_max_calls=None)
    services = TeamServices(lambda _aid: Model(), PublicTask(
        task_id="ledger-context", question="Write a guide comparing the qualifying candidates.",
        tools=["search"]), RubricGraph(rubrics=[]), execution_mode="iterative_shared_ledger",
        agent_pool=pool)
    memory = TeamMemory()
    memory.initialize("coordinator", TaskInput(task="Write a guide."))
    tools = {"search": lambda query: "Observed tool evidence: the quoted unit is kilograms."}
    context = SimpleNamespace(
        tool_policy=SimpleNamespace(select_tools=lambda *_args: ToolSelection(tools=tools)),
        memory=memory, planning=TeamPlanning(), model=None, prompt_templates={},
        get_tool_schemas=lambda _selected: json.dumps([{"name": "search", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]),
        execute_tool=lambda name, arguments: tools[name](**arguments),
    )
    return team, services, context, calls, outputs


def _publish(services, revision):
    outline = [f"Candidate {index}: qualifies in 2025; quantity {index + 1} kg; "
               "the supplied primary source supports its eligibility and date."
               for index in range(30)]
    outline.append("Gamma qualifies" if revision == 0 else
                   "Gamma is withdrawn because it does not meet the public inclusion condition.")
    outline.append(f"Revised calculation {revision}: 3 kg plus 4 kg equals 7 kg.")
    ledger = {"requirements": ["Include all supported qualifying candidates"],
              "outline": outline,
              "evidence_spans": [{"text": "The source records 3 kg and 4 kg.", "source_ref": "web-0"}],
              "source_references": [{"source_id": "web-0", "locator": "https://example.org/measurements"}]}
    event = services.event("searcher", "artifact_published", {"ledger": ledger, "revision": revision})
    services.artifacts["searcher"] = {"agent_id": "searcher", "answer": "Verified candidate findings.",
                                     "event_id": event, "version": revision + 1, "complete": True,
                                     "checkpoint_reports": {}, "ledger": ledger}
    return ledger


def _snapshot(messages):
    content = next(message["content"] for message in reversed(messages)
                   if message["role"] == "user" and "Updated shared ledger: " in message["content"])
    return json.loads(content.rsplit("Updated shared ledger: ", 1)[1])


def test_revisions_keep_current_full_ledger_observations_and_immutable_audit():
    expected = {}

    def respond(messages, turn):
        if turn > 1:
            shared = _snapshot(messages)
            assert shared["contributions"][0]["ledger"] == expected["ledger"]
            assert len(shared["outline"]) == 32
            assert "Gamma qualifies" not in json.dumps(messages, ensure_ascii=False)
            assert shared["source_references"][0]["locator"] == "https://example.org/measurements"
            assert shared["evidence_spans"][0]["text"] == "The source records 3 kg and 4 kg."
            assert "Observed tool evidence: the quoted unit is kilograms." in json.dumps(messages)
            assert all(event["event_id"] in json.dumps(messages) for event in services.events
                       if event["kind"] == "peer_message")
            payload = json.loads(messages[1]["content"])
            assert payload["shared_ledger"]["snapshot_sha256"] == content_hash(shared)
            assert "complete current shared_ledger JSON" in payload["shared_ledger"]["snapshot_location"]
            assert payload["public_task"] == services.public_task.model_dump(mode="json")
            assert sum(message["content"].startswith("Updated shared ledger: ") or
                       "\nUpdated shared ledger: " in message["content"]
                       for message in messages if message["role"] == "user") == 1
        if turn == 5:
            return {"answer": "The 30 supported candidates qualify; Gamma is excluded. "
                              "The corrected measured total is 7 kilograms."}
        expected["ledger"] = _publish(services, turn)
        services.event("searcher", "peer_message", {"content": f"Check corrected revision {turn}."},
                       recipient="writer")
        if turn == 1:
            return {"tools": [{"name": "search", "arguments": {"query": "units"}}]}
        return {"answer": f"Draft revision {turn}: preserve every supported candidate and correct the units.",
                "continue": True}

    team, services, context, calls, outputs = _setup(respond)
    _publish(services, 0)
    initial = copy.deepcopy(_iterative_public_ledger(services, "writer"))
    result = _run_agent_iterative(team.agents[-1].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "final_answer" and len(calls) == 5, [
        str(step.error) for step in result.trajectory]
    assert "Gamma qualifies" in json.dumps(result.trajectory[0].model_input_messages)
    assert "Gamma qualifies" in json.dumps(services.events[0])
    assert result.trajectory[1].model_input_messages == calls[1]
    assert _snapshot(result.trajectory[1].model_input_messages)["contributions"][0]["version"] == 2
    assert _snapshot(result.trajectory[-1].model_input_messages)["contributions"][0]["version"] == 5

    # Reconstruct the previous append-only context from the same actual outputs,
    # observations and complete snapshots; no content or model turn is removed.
    legacy = copy.deepcopy(calls[0])
    legacy_calls = [copy.deepcopy(legacy)]
    for index, actual in enumerate(calls[1:], 1):
        legacy.append({"role": "assistant", "content": outputs[index - 1]})
        if index == 1:
            observation = next(message["content"].split("\nUpdated shared ledger: ", 1)[0]
                               for message in actual if message["content"].startswith("Public observations: "))
            legacy.append({"role": "user", "content": observation})
        update = "Updated shared ledger: " + json.dumps(_snapshot(actual), ensure_ascii=False)
        if len(legacy) > 2 and legacy[-1]["role"] == "user":
            legacy[-1]["content"] += "\n" + update
        else:
            legacy.append({"role": "user", "content": update})
        payload = json.loads(actual[1]["content"])
        payload["shared_ledger"] = copy.deepcopy(initial)
        legacy[1]["content"] = json.dumps(payload, ensure_ascii=False)
        legacy_calls.append(copy.deepcopy(legacy))
    encoding = tiktoken.get_encoding("cl100k_base")
    count = lambda history: sum(len(encoding.encode(message["content"], disallowed_special=()))
                                for message in history)
    compact_tokens = sum(map(count, calls))
    legacy_tokens = sum(map(count, legacy_calls))
    assert compact_tokens < legacy_tokens * 0.8
    print(f"ledger input tokens: append-only={legacy_tokens}, latest={compact_tokens}, "
          f"saved={legacy_tokens - compact_tokens} ({1 - compact_tokens / legacy_tokens:.1%})")


def test_recent_memory_replays_current_snapshot_when_no_peer_revision_occurs():
    def respond(messages, turn):
        if turn == 1:
            _publish(services, 1)
            return {"answer": "Initial guide draft", "continue": True}
        shared = _snapshot(messages)
        assert shared["contributions"][0]["version"] == 2
        assert len(shared["outline"]) == 32
        if turn == 2:
            return {"answer": "Refined guide draft with the corrected units", "continue": True}
        assert messages[-1]["content"].startswith("Continue the assigned work")
        assert messages[-2]["role"] == "assistant"
        return {"answer": "The supported candidates qualify; Gamma is excluded for the stated condition."}

    team, services, context, calls, _outputs = _setup(respond, recent=True)
    _publish(services, 0)
    result = _run_agent_iterative(team.agents[-1].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "final_answer" and len(calls) == 3
    assert _snapshot(calls[1]) == _snapshot(calls[2])


def test_snapshot_replacement_preserves_exact_nonledger_user_text():
    observation = 'Public observations: [{"output":"literal Updated shared ledger: token"}]'
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "original task"},
                {"role": "user", "content": observation}]
    state = {}
    _replace_iterative_ledger_snapshot(messages, {"outline": ["old claim"]}, state)
    messages.append({"role": "assistant", "content": "private reasoning retained"})
    correction = "Protocol correction: preserve all supplied measurements."
    messages.append({"role": "user", "content": correction})
    _replace_iterative_ledger_snapshot(messages, {"outline": ["corrected claim"]}, state)
    assert messages[2] == {"role": "user", "content": observation}
    assert messages[3] == {"role": "assistant", "content": "private reasoning retained"}
    assert messages[4]["content"].startswith(correction + "\nUpdated shared ledger: ")
    assert "old claim" not in json.dumps(messages)
