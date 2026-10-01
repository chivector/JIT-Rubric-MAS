"""Observed handoffs and honest checks, through the installed team executor."""

import json
import shutil
import threading
from pathlib import Path

import pytest

from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import (
    ResponseProtocolError, TeamExecutor, TeamServices, _SinglePassModel, _parse_response, content_hash,
)
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


@pytest.fixture
def execute_team():
    synth = JITHarnessSynthesizer()

    def execute(team, respond, *, tool_calls=0, auto_ledger=True, tools=None):
        task = PublicTask(task_id="quality-handoff", question="Explain the argument and its assumptions.",
                          tools=list(tools or {}))
        ledger = BudgetLedger(max_calls=20, max_tokens=200_000, max_tool_calls=tool_calls)
        inputs = {}

        class Model:
            def __init__(self, aid):
                self.aid = aid

            def __call__(self, messages, **kwargs):
                history = inputs.setdefault(self.aid, [])
                history.append(messages)
                payload = json.loads(messages[1]["content"])
                reply = respond(self.aid, payload, len(history), messages)
                if auto_ledger and self.aid != team.synthesizer_id and "answer" in reply:
                    reply.setdefault("ledger", {"requirements": ["Address the public task"],
                                                "outline": ["Present the contribution"],
                                                "evidence_spans": [], "source_references": []})
                return ChatMessage(role="assistant", content=json.dumps(reply))

            def get_token_counts(self):
                return {"input_token_count": 20, "output_token_count": 10}

        artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team)
        result = TeamExecutor(lambda aid: MeteredModel(Model(aid), ledger, "execution", aid),
                              ledger=ledger, tools=tools).execute(task, team, artifact)
        return result, inputs, ledger.snapshot()

    yield execute
    root = Path(__file__).resolve().parents[2] / "scripts" / "workspaces"
    for agent in synth._agents.values():
        path = agent.workspace_dir.resolve()
        assert path.parent == root.resolve() and path.name.startswith("mas_")
        if path.is_dir():
            shutil.rmtree(path)


def chain():
    return TeamSpec(agents=[
        AgentSpec(agent_id="writer", role="Writer", capability="argument exposition", max_calls=1),
        AgentSpec(agent_id="reviewer", role="Reviewer", capability="argument verification",
                  depends_on=["writer"], max_calls=1),
        AgentSpec(agent_id="editor", role="Editor", capability="argument synthesis",
                  depends_on=["reviewer"], max_calls=1),
    ], synthesizer_id="editor", total_max_calls=3, max_parallel=1)


def test_transitive_dependency_delivers_original_artifact_not_foreign_history(execute_team):
    def respond(aid, payload, _call, messages):
        assert not any(row["role"] == "assistant" for row in messages)
        sources = {row["agent_id"]: row for row in payload["shared_ledger"]["contributions"]}
        if aid == "writer":
            assert not sources
            return {"answer": "Original argument with assumptions."}
        if aid == "reviewer":
            assert list(sources) == ["writer"]
            return {"answer": "A short review, not a copy of the argument."}
        assert set(sources) == {"writer", "reviewer"}
        assert sources["writer"]["answer"] == "Original argument with assumptions."
        assert sources["writer"]["dependency_kind"] == "transitive"
        assert sources["reviewer"]["dependency_kind"] == "direct"
        assert all(set(row) == {"agent_id", "answer", "event_id", "content_hash", "version",
                               "checkpoint_reports", "dependency_kind"}
                   for row in sources.values())
        return {"answer": "Corrected full argument.",
                "evidence_ids": [row["event_id"] for row in sources.values()]}

    result, inputs, budget = execute_team(chain(), respond)
    assert result.answer == "Corrected full argument."
    assert len(inputs) == budget["model_calls"] == 3
    received = [e for e in result.metadata["events"]
                if e["kind"] == "shared_ledger_read" and e["agent_id"] == "editor"]
    assert len(received) == 1
    assert {e["agent_id"] for e in result.metadata["shared_ledger"]["contributions"]} == {"writer", "reviewer"}


def test_diamond_handoff_deduplicates_shared_ancestor(execute_team):
    team = chain()
    team.agents[-1].depends_on = ["writer", "reviewer"]

    def respond(aid, payload, *_):
        if aid == "editor":
            assert [r["agent_id"] for r in payload["shared_ledger"]["contributions"]] == ["writer", "reviewer"]
        return {"answer": aid}

    result, _, _ = execute_team(team, respond)
    assert result.answer == "editor"


def test_each_role_receives_its_complete_output_shape_without_a_recall(execute_team):
    team = chain()
    team.agents[0].checkpoints = ["Assumptions checked"]

    def respond(aid, payload, call, messages):
        assert call == 1
        example = payload["completion_example"]
        assert ("ledger" in example) == (aid != team.synthesizer_id)
        assert set(example["checkpoints"]) == set(payload["agent"]["checkpoints"])
        assert json.dumps(example, ensure_ascii=False) in messages[-1]["content"]
        assert "No external tool requests are available" in messages[-1]["content"]
        assert 'Complete with {"answer"' not in messages[0]["content"]
        if aid != team.synthesizer_id:
            assert "ledger is a sibling of answer" in messages[0]["content"]
            assert set(example["ledger"]) == {"requirements", "outline", "evidence_spans", "source_references"}
            example["ledger"]["outline"] = ["State the assumptions before the conclusion."]
        example["answer"] = aid + " artifact"
        for check in example["checkpoints"].values():
            check["reason"] = "The supplied task does not establish the missing assumption."
        return example

    result, inputs, budget = execute_team(team, respond, auto_ledger=False)
    assert result.answer == "editor artifact"
    assert budget["model_calls"] == 3 and budget["tool_calls"] == 0
    assert all(len(calls) == 1 for calls in inputs.values())


def test_explained_failed_check_is_reported_not_falsely_verified(execute_team):
    team = chain()
    team.agents[0].checkpoints = ["Proof checked"]

    def respond(aid, payload, *_):
        if aid == "writer":
            return {"answer": "The proof has an unsupported step.", "checkpoints": {
                "Proof checked": {"status": "failed", "reason": "An assumption is missing."}}}
        report = payload["shared_ledger"]["contributions"][0]["checkpoint_reports"]["Proof checked"]
        assert report["status"] == "failed"
        assert report["basis"] == "self_reported"
        assert report["independently_verified"] is False
        return {"answer": "The missing assumption must be stated."}

    result, _, budget = execute_team(team, respond)
    assert result.answer == "The missing assumption must be stated."
    assert budget["model_calls"] == 3
    assert result.sub_runs[0].metadata["checkpoint_reports"]["Proof checked"]["status"] == "failed"


def test_boolean_true_is_kept_as_unverified_self_report(execute_team):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    checkpoints=["Checked"], max_calls=1)], synthesizer_id="solo")
    result, _, _ = execute_team(team, lambda *_: {"answer": "Done.", "checkpoints": {"Checked": True}})
    assert result.answer == "Done."
    assert result.sub_runs[0].metadata["checkpoint_reports"]["Checked"] == {
        "status": "passed", "reason": "", "evidence_ids": [],
        "basis": "self_reported", "independently_verified": False}


def test_completed_check_is_preserved_without_upgrading_to_passed(execute_team):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    checkpoints=["Checked"], max_calls=1)], synthesizer_id="solo")
    result, _, budget = execute_team(team, lambda *_: {
        "answer": "The claim needs an additional assumption.", "checkpoints": {"Checked": {
            "status": "completed", "reason": "Reviewed the claim and found an unstated assumption.",
            "evidence_ids": []}}})
    assert result.answer == "The claim needs an additional assumption."
    report = result.sub_runs[0].metadata["checkpoint_reports"]["Checked"]
    assert report["status"] == "completed"
    assert report["basis"] == "self_reported" and report["independently_verified"] is False
    assert budget["model_calls"] == 1


@pytest.mark.parametrize("check", [
    {"status": "completed", "reason": ""},
    {"status": "completed", "reason": "Checked", "evidence_ids": [False]},
])
def test_completed_check_keeps_reason_and_evidence_shape_guards(check):
    with pytest.raises(ResponseProtocolError):
        _parse_response({"answer": "Result", "checkpoints": {"Checked": check}})


def test_completed_check_cannot_cite_unobserved_evidence(execute_team):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    checkpoints=["Checked"], max_calls=1)], synthesizer_id="solo")
    result, _, _ = execute_team(team, lambda *_: {"answer": "Result", "checkpoints": {"Checked": {
        "status": "completed", "reason": "Checked", "evidence_ids": ["unobserved"]}}})
    assert result.answer is None
    assert "not observed" in str(result.sub_runs[0].trajectory[0].error)


@pytest.mark.parametrize("checkpoint", [False, None])
def test_missing_or_unexplained_false_check_still_requires_completion(execute_team, checkpoint):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    checkpoints=["Checked"], max_calls=1)], synthesizer_id="solo")
    checks = {} if checkpoint is None else {"Checked": checkpoint}
    result, _, _ = execute_team(team, lambda *_: {"answer": "Done.", "checkpoints": checks})
    assert result.answer is None


@pytest.mark.parametrize("check", ["passed", {"status": "verified", "reason": "yes"},
                                    {"status": [], "reason": "yes"},
                                    {"status": "failed", "reason": ""},
                                    {"status": "unverified", "reason": "unknown", "evidence_ids": [3]}])
def test_structured_check_report_must_be_well_formed(check):
    with pytest.raises(ResponseProtocolError):
        _parse_response({"answer": "Result", "checkpoints": {"Check": check}})


def test_known_private_event_cannot_be_cited_as_observed_evidence(execute_team):
    team = chain()

    def respond(aid, payload, *_):
        if aid == "editor":
            first_artifact = payload["shared_ledger"]["contributions"][0]["event_id"]
            run_id = first_artifact.rsplit(":e", 1)[0]
            return {"answer": "A guessed citation.", "evidence_ids": [run_id + ":e2"]}
        return {"answer": aid}

    result, _, _ = execute_team(team, respond)
    assert result.answer is None
    assert "not observed" in str(result.sub_runs[-1].trajectory[-1].error)


def test_no_tool_budget_is_explicit_and_quality_prompt_is_domain_general(execute_team):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    max_calls=1)], synthesizer_id="solo")

    def respond(_aid, payload, _call, messages):
        assert payload["coordination"] == "single_pass_shared_ledger"
        assert "collaboration_tools" not in payload
        assert "Remaining shared tool calls: 0" in messages[-1]["content"]
        system = messages[0]["content"]
        assert "fallible planning hypotheses" in system
        assert "unsupported step as a simplification" in system
        assert "nontechnical tasks" in system
        assert "internal rubric IDs" in system
        return {"answer": "A complete explanation."}

    result, _, budget = execute_team(team, respond)
    assert result.answer == "A complete explanation."
    assert budget["tool_calls"] == 0


@pytest.mark.parametrize("completion", [
    {"answer": "Result", "evidence_ids": ["unobserved"]},
    {"answer": "Result", "checkpoints": {"Checked": {
        "status": "passed", "reason": "Checked it", "evidence_ids": ["unobserved"]}}},
])
def test_invalid_completion_is_rejected_before_tool_side_effects(execute_team, completion):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    checkpoints=["Checked"], max_calls=1)], synthesizer_id="solo")
    response = {"tools": [{"name": "raise_issue", "arguments": {"content": "must not publish"}},
                          {"name": "complete", "arguments": completion}]}
    result, _, budget = execute_team(team, lambda *_: response, tool_calls=1)
    assert result.answer is None
    assert budget["tool_calls"] == 0
    assert not any(event["kind"] == "issue" for event in result.metadata["events"])
    assert "not observed" in str(result.sub_runs[0].trajectory[0].error)


def test_split_completion_fields_cannot_bypass_evidence_preflight(execute_team):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    max_calls=1)], synthesizer_id="solo")
    response = {"answer": "Result", "tools": [
        {"name": "raise_issue", "arguments": {"content": "must not publish"}},
        {"name": "complete", "arguments": {"evidence_ids": ["unobserved"]}}]}
    result, _, budget = execute_team(team, lambda *_: response, tool_calls=1)
    assert result.answer is None
    assert budget["tool_calls"] == 0
    assert "not observed" in str(result.sub_runs[0].trajectory[0].error)


@pytest.mark.parametrize("tool", ["send_message", "read_evidence", "raise_issue"])
def test_inter_agent_tools_fail_once_without_waiting_or_tool_charge(execute_team, tool):
    team = TeamSpec(agents=[AgentSpec(agent_id="solo", role="Writer", capability="writing",
                                    max_calls=2)], synthesizer_id="solo", total_max_calls=2)

    def respond(_aid, _payload, call, _messages):
        assert call == 1
        return {"tools": [{"name": tool, "arguments": {"content": "Missing premise"}}]}

    result, _, budget = execute_team(team, respond, tool_calls=1)
    assert result.answer is None
    assert "communication tools are unavailable" in str(result.sub_runs[0].trajectory[0].error)
    assert budget["tool_calls"] == 0
    assert budget["model_calls"] == 1
    assert not {"message_sent", "message_consumed", "issue", "evidence_read"}.intersection(
        event["kind"] for event in result.metadata["events"])


def test_worker_cannot_publish_tool_side_effect_before_illegal_final_answer(execute_team):
    def respond(aid, *_):
        if aid == "writer":
            return {"tools": [
                {"name": "raise_issue", "arguments": {"content": "must not publish"}},
                {"name": "final_answer", "arguments": {"answer": "Unauthorized final"}}]}
        return {"answer": aid}

    result, _, budget = execute_team(chain(), respond, tool_calls=1)
    assert result.answer is None
    assert budget["tool_calls"] == 0
    assert not any(event["kind"] == "issue" for event in result.metadata["events"])
    assert "only the synthesizer" in str(result.sub_runs[0].trajectory[0].error)


def test_generated_harness_contract_preserves_honest_quality_reporting():
    from jit_mas.bridge import _mas_contract

    contract = _mas_contract()
    assert "self-reports, never independent verification" in contract
    assert "Do not require unconditional true" in contract
    assert "shared ledger" in contract.lower()
    assert "predicted rubrics are fallible" in contract


def test_parallel_contributors_publish_one_frozen_ledger_and_writer_reads_once(execute_team):
    team = TeamSpec(agents=[
        AgentSpec(agent_id="analyst", role="Analyst", capability="requirements"),
        AgentSpec(agent_id="evidence", role="Evidence", capability="source selection"),
        AgentSpec(agent_id="writer", role="Writer", capability="synthesis",
                  depends_on=["analyst", "evidence"], max_calls=5),
    ], synthesizer_id="writer", total_max_calls=10, max_parallel=2)
    barrier = threading.Barrier(2, timeout=3)

    def respond(aid, payload, call, messages):
        assert call == 1
        assert not any(row["role"] == "assistant" for row in messages)
        shared = payload["shared_ledger"]
        if aid != "writer":
            assert shared["contributions"] == []
            barrier.wait()
        if aid == "analyst":
            return {"answer": "Separate claims and assumptions.", "ledger": {
                "requirements": ["Explain assumptions"], "outline": ["Claim", "Assumption"],
                "evidence_spans": [], "source_references": []}}
        if aid == "evidence":
            return {"answer": "The only supplied source is the public task.", "ledger": {
                "requirements": [], "outline": [],
                "evidence_spans": [{"text": "Explain the argument and its assumptions.", "source_ref": "task"}],
                "source_references": [{"source_id": "task", "locator": "public_task.question"}]}}
        assert shared["requirements"] == [{"agent_id": "analyst", "text": "Explain assumptions"}]
        assert shared["source_references"][0]["agent_id"] == "evidence"
        assert [item["agent_id"] for item in shared["contributions"]] == ["analyst", "evidence"]
        assert "Read the structured shared ledger once" in messages[0]["content"]
        shared["requirements"].clear()  # The model-side copy must not mutate the frozen snapshot.
        return {"answer": "The argument follows only under its stated assumptions."}

    result, inputs, budget = execute_team(team, respond, auto_ledger=False)
    assert result.answer == "The argument follows only under its stated assumptions."
    assert {aid: len(rows) for aid, rows in inputs.items()} == {"analyst": 1, "evidence": 1, "writer": 1}
    assert budget["model_calls"] == 3 and budget["tokens"] == 90 and budget["reserved_tokens"] == 0
    shared = result.metadata["shared_ledger"]
    assert shared["requirements"]
    assert result.metadata["shared_ledger_hash"] == content_hash(shared)
    expected_bytes = len(json.dumps(shared, ensure_ascii=False).encode("utf-8"))
    assert budget["by_stage"]["execution"]["communication_bytes"] == expected_bytes
    assert budget["tool_calls"] == 0
    events = result.metadata["events"]
    assert len([e for e in events if e["kind"] == "agent_call"]) == 3
    assert len([e for e in events if e["kind"] == "shared_ledger_read"]) == 1
    assert len([e for e in events if e["kind"] == "shared_ledger_ready"]) == 1
    assert not {"message_sent", "message_consumed", "issue", "evidence_read"}.intersection(e["kind"] for e in events)


@pytest.mark.parametrize("ledger", [None, {},
    {"requirements": [], "outline": [], "evidence_spans": [{"text": "Claim", "source_ref": "unknown"}],
     "source_references": []},
    {"requirements": [3], "outline": [], "evidence_spans": [], "source_references": []},
    {"requirements": [], "outline": [], "evidence_spans": [],
     "source_references": [{"source_id": "one", "locator": "task"}, {"source_id": "one", "locator": "task"}]},
])
def test_invalid_contribution_fails_without_waiting_for_or_recalling_writer(execute_team, ledger):
    team = chain()
    result, inputs, budget = execute_team(team, lambda *_: {"answer": "Unsupported", "ledger": ledger},
                                         auto_ledger=False)
    assert result.answer is None and list(inputs) == ["writer"]
    assert budget["model_calls"] == 1
    assert result.sub_runs[-1].terminated_reason == "dependency_failed"


@pytest.mark.parametrize("writer", [False, True])
def test_disallowed_batch_is_rejected_before_any_external_side_effect(execute_team, writer):
    calls = []

    class Lookup:
        name, description, inputs = "lookup", "Fixture lookup", {}

        def __call__(self):
            calls.append("called")
            return "Source data"

    team = chain()
    selected = team.agents[-1] if writer else team.agents[0]
    selected.tools = ["lookup"]

    def respond(aid, *_):
        if aid != selected.agent_id:
            return {"answer": aid}
        requests = [{"name": "lookup", "arguments": {}}]
        if not writer:
            requests.append({"name": "send_message", "arguments": {"recipient": "editor", "content": "Reply"}})
        return {"answer": "Should fail", "tools": requests}

    result, _, budget = execute_team(team, respond, tool_calls=5, tools={"lookup": Lookup()})
    assert result.answer is None and calls == [] and budget["tool_calls"] == 0


def test_terminal_completion_cannot_smuggle_nested_tool_requests():
    with pytest.raises(ResponseProtocolError, match="Nested tool requests"):
        _parse_response({"tools": [{"name": "complete", "arguments": {
            "answer": "Answer", "tools": [{"name": "send_message", "arguments": {}}]}}]})


def test_cancellation_stops_remaining_external_batch_before_new_side_effects(execute_team, monkeypatch):
    calls = []

    class Lookup:
        name, description, inputs = "lookup", "Fixture lookup", {}

        def __call__(self):
            calls.append("called")
            return "Source data"

    original_event = TeamServices.event

    def cancel_after_first_result(services, aid, kind, content, **kwargs):
        event_id = original_event(services, aid, kind, content, **kwargs)
        if kind == "retrieved":
            services.cancelled.set()
        return event_id

    monkeypatch.setattr(TeamServices, "event", cancel_after_first_result)
    team = chain()
    team.agents[0].tools = ["lookup"]
    response = {"tools": [{"name": "lookup", "arguments": {}}, {"name": "lookup", "arguments": {}}]}
    result, inputs, budget = execute_team(team, lambda *_: response, tool_calls=2, tools={"lookup": Lookup()})
    assert result.answer is None and list(inputs) == ["writer"]
    assert calls == ["called"] and budget["tool_calls"] == 1
    assert result.sub_runs[-1].terminated_reason == "cancelled"


def test_role_call_guard_rejects_reentry_before_model_or_token_charge():
    ledger = BudgetLedger()

    class Model:
        def __call__(self, messages, **kwargs):
            return ChatMessage(role="assistant", content='{"answer":"Once"}')

        def get_token_counts(self):
            return {"input_token_count": 5, "output_token_count": 3}

    services = TeamServices(lambda aid: None, PublicTask(task_id="one", question="Respond once"))
    agent = {"agent_id": "writer", "role": "Writer"}
    team = {"synthesizer_id": "writer", "total_max_calls": 10}
    first = _SinglePassModel(MeteredModel(Model(), ledger, "execution", "writer"), services, agent, team)
    first([])
    second = _SinglePassModel(MeteredModel(Model(), ledger, "execution", "writer"), services, agent, team)
    with pytest.raises(RuntimeError, match="only one model call"):
        second([])
    assert services.calls == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["tokens"] == 8
