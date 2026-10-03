"""Synthetic prompt delivery and artifact preservation; no model-quality claims."""

import copy
import json
from types import SimpleNamespace

import pytest

from jit_mas.execution import TeamMemory, TeamPlanning, TeamServices, run_team
from jit_mas.output_contract import PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.kernel.types import TaskInput, ToolSelection
from scripts.models.base import ChatMessage


class SyntheticModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        reply = next(self.replies)
        if callable(reply):
            reply = reply(messages)
        return ChatMessage(role="assistant", content=json.dumps(reply, ensure_ascii=False))


def execute(question, replies, mode, checkpoints=()):
    task = PublicTask(task_id="synthetic-public-constraints", question=question)
    agent = AgentSpec(agent_id="writer", role="Writer", capability="constrained writing",
                      checkpoints=list(checkpoints),
                      max_calls=1 if mode == "single_pass" else None)
    team = TeamSpec(execution_mode=mode, agents=[agent], synthesizer_id="writer",
                    total_max_calls=1 if mode == "single_pass" else None)
    model = SyntheticModel(replies)
    services = TeamServices(lambda _: model, task, RubricGraph(rubrics=[]), execution_mode=mode)
    memory = TeamMemory()
    memory.initialize("coordinator", TaskInput(task=question))
    context = SimpleNamespace(
        memory=memory, planning=TeamPlanning(), model=None, prompt_templates={},
        get_tool_schemas=lambda _: "[]",
        tool_policy=SimpleNamespace(select_tools=lambda *args: ToolSelection(tools={})),
    )
    result = run_team(question, context, team, services)
    return result, model, services


MODES = ["single_pass", "iterative_shared_ledger"]
JSON_ARTIFACT = json.dumps(
    {"quote": 'He said "ready".', "city": "北京", "lines": "a\nb"},
    ensure_ascii=False, separators=(",", ":"),
)


@pytest.mark.parametrize("mode", MODES)
def test_planning_models_receive_construction_advice_and_the_unchanged_public_task(mode):
    task = PublicTask(task_id="synthetic-plan", question="Return exactly three lowercase words.")
    candidate = AgentSpec(agent_id="writer", role="Writer", capability="constrained writing",
                          max_calls=1 if mode == "single_pass" else None)
    draft = {"graph": {"rubrics": []}, "candidates": [candidate.model_dump(mode="json")]}
    local = {"agent_id": "writer", "capability": candidate.capability,
             "expected_outputs": ["The requested three-word artifact"], "tools": [],
             "max_calls": candidate.max_calls}
    team = TeamSpec(execution_mode=mode, agents=[candidate], synthesizer_id="writer",
                    total_max_calls=1 if mode == "single_pass" else None)
    responses = iter([draft, local, {"graph": {"rubrics": []}, "team": team.model_dump(mode="json")}])
    received = []

    def model(messages):
        assert PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload["task"] == task.model_dump(mode="json")
        received.append(copy.deepcopy(payload))
        return json.dumps(next(responses))

    analyzer = GlobalAnalyzer(model, execution_mode=mode, total_max_calls=team.total_max_calls)
    prediction = analyzer.predict(task)
    local_plan = analyzer.local_plan(task, prediction, prediction.candidates[0])
    planned = analyzer.reconcile(task, prediction, [local_plan])
    assert len(received) == len(analyzer.call_records) == 3
    assert planned.team.agents[0].tools == []
    assert planned.team.agents[0].max_calls == candidate.max_calls
    assert task.question == "Return exactly three lowercase words."


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind,question,answer", [
    pytest.param(
        "json", "Return only this JSON object, preserving its string values: " + JSON_ARTIFACT,
        JSON_ARTIFACT, id="json-escaping",
    ),
    pytest.param(
        "paragraph",
        "Return only two lowercase paragraphs separated by one blank line. Each paragraph "
        "must have exactly four words, counting whitespace-separated words. The word clock "
        "must appear exactly once in each paragraph. Start with soft and end with blue.",
        "soft clock moves slowly\n\na clock shines blue",
        id="paragraph-count-keyword-case",
    ),
    pytest.param(
        "sentence",
        "Return only one uppercase sentence with exactly six whitespace-separated words. "
        'Include CLOCK exactly once, start with READY and end with the literal text "NOW.".',
        "READY THE CLOCK IS WORKING NOW.",
        id="sentence-count-keyword-case",
    ),
])
def test_constructed_artifact_is_delivered_verbatim_in_both_execution_modes(mode, kind, question, answer):
    def submit(messages):
        system = messages[0]["content"]
        public = json.loads(messages[1]["content"])["public_task"]
        assert PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT in system
        assert public["question"] == question and public["tools"] == []
        return {"answer": answer, "continue": False}

    result, model, services = execute(question, [submit], mode)
    assert result.answer == answer
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 1
    final_events = [event for event in services.events if event["kind"] == "final_answer"]
    assert len(final_events) == 1 and final_events[0]["content"] == answer
    assert not [event for event in services.events if event["kind"] == "tool_result"]
    # These assertions use the synthetic prompt's own explicit conventions. They
    # validate preservation of our fixture, not the ability of a real model to count.
    if kind == "json":
        assert json.loads(result.answer) == {"quote": 'He said "ready".', "city": "北京", "lines": "a\nb"}
    elif kind == "paragraph":
        paragraphs = result.answer.split("\n\n")
        assert [len(paragraph.split()) for paragraph in paragraphs] == [4, 4]
        assert [paragraph.split().count("clock") for paragraph in paragraphs] == [1, 1]
        assert result.answer.islower() and result.answer.startswith("soft") and result.answer.endswith("blue")
    else:
        assert len(result.answer.split()) == 6 and result.answer.split().count("CLOCK") == 1
        assert result.answer.isupper() and result.answer.startswith("READY") and result.answer.endswith("NOW.")


def test_later_revision_keeps_prompt_and_submits_only_the_repaired_decoded_answer():
    question = (
        "Return only two lowercase paragraphs separated by one blank line. Each paragraph "
        "must have exactly four words, counting whitespace-separated words. Include clock "
        "exactly once per paragraph, start with soft and end with blue."
    )
    draft = "Soft CLOCK moves very slowly\n\na clock shines blue"
    repaired = "soft clock moves slowly\n\na clock shines blue"

    def revise(messages):
        assert PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT in messages[0]["content"]
        prior = [json.loads(message["content"]) for message in messages
                 if message["role"] == "assistant"]
        assert prior[-1]["answer"] == draft
        return {"answer": repaired, "continue": False}

    result, model, services = execute(question, [
        {"answer": draft, "continue": True}, revise,
    ], "iterative_shared_ledger")
    assert len(model.calls) == 2 and result.answer == repaired
    assert result.terminated_reason == "final_answer"
    assert result.sub_runs[0].trajectory[0].error is None
    final_events = [event for event in services.events if event["kind"] == "final_answer"]
    assert len(final_events) == 1 and final_events[0]["content"] == repaired
    assert result.sub_runs[0].trajectory[0].model_output_messages.content == json.dumps(
        {"answer": draft, "continue": True}, ensure_ascii=False)


@pytest.mark.parametrize("mode", MODES)
def test_incompatible_public_constraints_are_not_silently_rewritten_or_marked_verified(mode):
    question = "Return only A. The same output must be exactly B. No precedence is given."
    explanation = "The two exact literal requirements conflict; no public precedence is supplied."

    def submit(messages):
        assert PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT in messages[0]["content"]
        return {"answer": "A", "continue": False, "checkpoints": {
            "Output feasibility": {"status": "unverified", "reason": explanation, "evidence_ids": []},
        }}

    result, model, _ = execute(question, [submit], mode, checkpoints=["Output feasibility"])
    assert result.answer == "A" and len(model.calls) == 1
    checkpoint = result.sub_runs[0].metadata["checkpoint_reports"]["Output feasibility"]
    assert checkpoint["status"] == "unverified" and checkpoint["reason"] == explanation
    assert checkpoint["independently_verified"] is False
