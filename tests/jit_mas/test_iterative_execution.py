"""Iterative execution publishes completed answers and cites delivered observations."""

import copy
import json
import time
from types import SimpleNamespace

import pytest

from jit_mas.agent_pool import seed_pool
from jit_mas.budget import BudgetLedger
from jit_mas.execution import (
    CONTRIBUTOR_COMPACTNESS_POLICY_VERSION,
    TeamMemory,
    TeamPlanning,
    TeamServices,
    _SinglePassModel,
    _run_agent_iterative,
    _completion_quality_error,
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


def make_services(team, models, *, tools=None, pool=None, knowledge_policy=None):
    installed = tools or {}
    services = TeamServices(lambda agent_id: models[agent_id],
                            PublicTask(task_id="iterative-review", question="Write a researched guide.",
                                       tools=list(installed)), RubricGraph(rubrics=[]),
                            execution_mode="iterative_shared_ledger", agent_pool=pool,
                            knowledge_policy=knowledge_policy)
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


def test_title_only_public_deliverable_gets_one_quality_correction():
    team = make_team()
    title = {"answer": "# A researched guide", "continue": False}
    model = ScriptedModel([title, {"answer": "A substantive guide with the requested explanation.",
                                   "continue": False}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "A substantive guide with the requested explanation."
    assert len(model.calls) == 2
    assert any(event["kind"] == "quality_warning" for event in services.events)
    assert result.trajectory[0].error is not None


def test_short_format_public_task_is_not_forced_to_have_a_long_answer():
    task = PublicTask(task_id="short", question="Return only the name of the capital city.")
    assert _completion_quality_error("Paris", task) is None


def test_completion_quality_rejects_status_disclaimer_without_report_body():
    task = PublicTask(task_id="report", question="Write a comprehensive report on the policy.")
    assert _completion_quality_error("No external sources were observed in this run.", task)


@pytest.mark.parametrize("answer", [
    "# Uranium market report [Remembered and unverified citations; no external sources observed.]",
    "# Market report\nNo external sources were observed in this run.",
    "The essay is the primary deliverable.",
])
def test_title_with_source_status_and_pure_meta_completion_is_rejected(answer):
    task = PublicTask(task_id="report", question="Write a comprehensive report on the policy.")
    assert _completion_quality_error(answer, task)


@pytest.mark.parametrize("answer", [
    "This report is about uranium prices. Supply disruptions raise prices while inventories buffer shocks.",
    "No external sources were retrieved in this run. Supply disruptions raise prices while inventories buffer shocks.",
    "# Market report\nSupply disruptions raise prices while inventories buffer shocks.",
    "# Market report  Supply disruptions raise prices while inventories buffer shocks.",
    "# Market report. Supply disruptions raise prices while inventories buffer shocks.",
])
def test_short_substantive_body_is_kept_even_with_title_or_status_disclaimer(answer):
    task = PublicTask(task_id="report", question="Write a brief report on uranium prices.")
    assert _completion_quality_error(answer, task) is None


def test_nonterminal_title_draft_does_not_consume_quality_correction():
    team = make_team()
    model = ScriptedModel([{"answer": "# A researched guide", "continue": True},
                           {"answer": "# A researched guide", "continue": False},
                           {"answer": "The guide explains the relevant assumptions and conclusions."}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 3
    assert result.trajectory[0].error is None
    assert len([event for event in services.events if event["kind"] == "quality_warning"]) == 1


def test_contributor_short_summary_with_substantive_ledger_remains_valid():
    team = make_team(contributors=True)
    contributor = {"answer": "# Research complete", "ledger": {
        "requirements": ["Explain source assumptions"], "outline": ["Discuss the central claim"],
        "evidence_spans": [], "source_references": []}}
    searcher = ScriptedModel([contributor])
    writer = ScriptedModel([{"answer": "The guide states the central claim and its assumptions."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "final_answer"
    assert len(searcher.calls) == len(writer.calls) == 1
    assert not [event for event in services.events if event["kind"] == "quality_warning"]


def test_iterative_protocol_correction_recovers_invalid_contributor_ledger_shape():
    team = make_team(contributors=True)
    malformed = {"answer": "Research contribution", "continue": False,
                 "ledger": {"requirements": ["Explain assumptions"], "outline": [3],
                             "evidence_spans": [], "source_references": []}}
    corrected = {"answer": "Research contribution with the requested assumptions.", "continue": False,
                 "ledger": {"requirements": ["Explain assumptions"],
                            "outline": ["Discuss the central claim"],
                            "evidence_spans": [], "source_references": []}}
    def fixed(messages):
        assert json.loads(messages[-2]["content"]) == malformed
        assert messages[-1]["content"].startswith("Protocol correction:")
        assert "ledger.outline must be" in messages[-1]["content"]
        return corrected

    searcher = ScriptedModel([malformed, fixed])
    writer = ScriptedModel([{"answer": "The guide explains the central claim and assumptions."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "final_answer"
    assert len(searcher.calls) == 2
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    searcher_output = [event for event in services.events if event["kind"] == "model_output"
                       and event["agent_id"] == "searcher"]
    assert json.loads(searcher_output[0]["content"]["content"]) == malformed
    assert services.artifacts["searcher"]["ledger"] == corrected["ledger"]


def test_repeated_invalid_contributor_ledger_fails_after_one_correction():
    team = make_team(contributors=True)
    malformed = {"answer": "Research contribution", "continue": False,
                 "ledger": {"requirements": ["Explain assumptions"], "outline": [3],
                             "evidence_spans": [], "source_references": []}}
    searcher = ScriptedModel([malformed, malformed])
    writer = ScriptedModel([{"answer": "Must never run"}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "error"
    assert len(searcher.calls) == 2
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert any("ledger.outline must be" in event["content"] for event in services.events
               if event["kind"] == "execution_error")
    assert "searcher" not in services.artifacts


@pytest.mark.parametrize("bad_ledger,error", [
    ({"requirements": [], "outline": [], "evidence_spans": [],
      "source_references": [{"source_id": "remembered"}]}, "source_id and locator"),
    ({"requirements": [], "outline": [], "evidence_spans": [{"text": "Claim"}],
      "source_references": []}, "text and a declared source_ref"),
])
def test_contributor_repairs_source_shape_only_by_returning_new_ledger(bad_ledger, error):
    team = make_team(contributors=True)
    invalid = {"answer": "Remembered policy reasoning.", "ledger": bad_ledger}
    repaired = {"answer": "Policy reasoning with uncertainty stated.", "ledger": {
        "requirements": [], "outline": [], "evidence_spans": [], "source_references": []}}
    model = ScriptedModel([invalid, repaired])
    services, context = make_services(team, {"searcher": model},
                                      knowledge_policy="model_general_knowledge_allowed")
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "subtask_complete"
    assert error in str(result.trajectory[0].error)
    assert json.loads(result.trajectory[0].model_output_messages.content) == invalid
    assert result.trajectory[1].error is None
    assert services.artifacts["searcher"]["ledger"] == repaired["ledger"]
    assert "Contributors publish a compact substantive summary in answer" in model.calls[0][0]["content"]
    assert "the synthesizer publishes the complete requested deliverable in answer" in model.calls[0][0]["content"]
    assert "publish evidence_spans=[] and source_references=[]" in model.calls[0][0]["content"]
    assert "FINAL ROLE OVERRIDE" in model.calls[0][0]["content"]
    assert "answer <=1200" in model.calls[0][0]["content"]
    assert "at most 12 items" in model.calls[0][0]["content"]
    assert "<=512 characters per item" in model.calls[0][0]["content"]
    assert "soft target" in model.calls[0][0]["content"]
    assert "terminate with continue=false" not in model.calls[0][0]["content"]


def test_ledger_protocol_correction_retains_both_metered_calls_and_raw_error():
    from jit_mas.budget import MeteredModel

    team = make_team(contributors=True)
    invalid = {"answer": "Contribution", "ledger": {"requirements": [], "outline": [2],
        "evidence_spans": [], "source_references": []}}
    valid = {"answer": "Reasoned contribution.", "ledger": {"requirements": [], "outline": [],
        "evidence_spans": [], "source_references": []}}
    raw = ScriptedModel([invalid, valid])
    # The contributor compactness guidance is part of the metered prompt; leave
    # enough shared budget for both the malformed response and its correction.
    ledger = BudgetLedger(max_calls=2, max_tokens=30000, max_tool_calls=0)
    model = MeteredModel(raw, ledger, "execution", "searcher", 4096)
    services, context = make_services(team, {"searcher": model})
    services.ledger = ledger
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "subtask_complete"
    assert len(result.trajectory) == len(raw.calls) == ledger.snapshot()["model_calls"] == 2
    assert len([record for record in ledger.snapshot()["records"] if record["kind"] == "model"]) == 2
    assert "ledger.outline must be" in str(result.trajectory[0].error)
    assert json.loads(result.trajectory[0].model_output_messages.content) == invalid


def test_json_object_contributor_budget_overruns_preserve_the_complete_contribution():
    team = make_team(contributors=True)
    contribution = {"answer": "x" * 1211, "ledger": {
        "requirements": [f"Requirement {i}" for i in range(19)],
        "outline": ["f" * 513],
        "evidence_spans": [{"text": "e" * 513, "source_ref": "task"}],
        "source_references": [{"source_id": "task", "locator": "l" * 2049}]}}
    untouched = copy.deepcopy(contribution)
    model = ScriptedModel([contribution])
    services, context = make_services(team, {"searcher": model})
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "subtask_complete" and len(model.calls) == 1
    assert result.trajectory[0].error is None
    assert services.artifacts["searcher"]["answer"] == untouched["answer"]
    assert services.artifacts["searcher"]["ledger"] == untouched["ledger"]
    assert json.loads(result.trajectory[0].model_output_messages.content) == untouched
    warnings = [event for event in services.events if event["kind"] == "compactness_warning"]
    assert len(warnings) == 1
    assert warnings[0]["content"] == {
        "policy": CONTRIBUTOR_COMPACTNESS_POLICY_VERSION,
        "content_preserved": True,
        "measurements": [
            {"field": "answer.characters", "actual": 1211, "suggested_max": 1200},
            {"field": "ledger.requirements.items", "actual": 19, "suggested_max": 12},
            {"field": "ledger.outline[0].characters", "actual": 513, "suggested_max": 512},
            {"field": "ledger.source_references[0].locator.characters", "actual": 2049,
             "suggested_max": 2048},
            {"field": "ledger.evidence_spans[0].text.characters", "actual": 513,
             "suggested_max": 512},
        ]}
    assert not [event for event in services.events if event["kind"] == "protocol_warning"]


def test_oversized_contributor_still_requires_a_declared_evidence_source():
    team = make_team(contributors=True)
    invalid = {"answer": "x" * 1211, "ledger": {
        "requirements": [f"Requirement {i}" for i in range(19)], "outline": [],
        "evidence_spans": [{"text": "e" * 513, "source_ref": "invented"}],
        "source_references": []}}
    model = ScriptedModel([invalid, invalid])
    services, context = make_services(team, {"searcher": model})
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "error" and len(model.calls) == 2
    assert "declared source_ref" in str(result.trajectory[-1].error)
    assert "searcher" not in services.artifacts
    assert not [event for event in services.events if event["kind"] == "compactness_warning"]
    for step in result.trajectory:
        assert json.loads(step.model_output_messages.content) == invalid


def test_contributor_shape_correction_preserves_fact_rich_handoff_guidance():
    team = make_team(contributors=True)
    bad = {"answer": "x" * 1201, "ledger": {
        "requirements": [], "outline": [3], "evidence_spans": [], "source_references": []}}
    valid = {"answer": "Compact contribution summary.", "ledger": {
        "requirements": ["Explain the requested result."],
        "outline": ["The result follows after checking the stated assumptions and units."],
        "evidence_spans": [], "source_references": []}}
    model = ScriptedModel([bad, valid])
    services, context = make_services(team, {"searcher": model})
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "subtask_complete" and len(model.calls) == 2
    correction = model.calls[1][-1]["content"]
    assert "Keep answer <=1200 characters as a compact summary" in correction
    assert "at most 12 items" in correction and "<=512 characters" in correction
    assert "merge overlapping requirements while preserving public-task coverage" in correction
    assert "Put the substantive contribution body in answer" not in correction
    for prompt in (model.calls[0][0]["content"], correction):
        assert "not a section heading alone" in prompt
        assert "3000-5000-word" in prompt
        assert "checked intermediate results" in prompt
        assert "legal or policy obligations" in prompt


def test_truncated_json_correction_does_not_replay_failed_assistant_text():
    team = make_team(checkpoints=["accuracy"])
    bad = "{" + "x" * 3000
    valid = {"answer": "Recovered.", "checkpoints": {"accuracy": {
        "status": "completed", "reason": "Checked.", "evidence_ids": []}}, "continue": False}
    model = ScriptedModel([bad, valid])
    services, context = make_services(team, {"writer": model})
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                  team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "final_answer"
    assert result.trajectory[0].model_output_messages.content == bad
    assert all("x" * 3000 not in json.dumps(call[0]) for call in model.calls[1:])


def test_malformed_checkpoint_cannot_hide_an_illegal_tool_for_correction():
    team = make_team(checkpoints=["accuracy"])
    invalid = {"answer": "Draft", "checkpoints": {"accuracy": "wrong shape"},
               "tools": [{"name": "illegal", "arguments": {}}]}
    model = ScriptedModel([invalid, {"answer": "Must never run"}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "error" and result.answer is None
    assert len(model.calls) == 1
    assert not [event for event in services.events if event["kind"] == "protocol_warning"]


def test_malformed_checkpoint_cannot_hide_a_fabricated_event_for_correction():
    team = make_team(checkpoints=["accuracy"])
    invalid = {"answer": "Draft", "evidence_ids": ["invented-event"],
               "checkpoints": {"accuracy": "wrong shape"}}
    model = ScriptedModel([invalid, {"answer": "Must never run"}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "error" and result.answer is None
    assert len(model.calls) == 1
    assert not [event for event in services.events if event["kind"] == "protocol_warning"]


def test_truncated_execution_json_gets_one_model_authored_shape_correction():
    team = make_team()
    malformed = '{"answer": "unfinished'
    model = ScriptedModel([malformed, {"answer": "The guide presents complete reasoning."}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert result.trajectory[0].model_output_messages.content == malformed
    assert "Expected a complete JSON object" in str(result.trajectory[0].error)
    system = model.calls[0][0]["content"]
    correction = model.calls[1][-1]["content"]
    assert "80-85%" in system and "soft planning target" in system
    assert "Never invent a long directory, catalog, numbered sequence" in system
    assert "rewrite one complete JSON response from the original task" in correction
    assert "Preserve the key arguments, calculations, required items" in correction
    assert "delete duplicate passages, guessed directories" in correction


def test_fabricated_evidence_id_does_not_receive_ledger_shape_correction():
    team = make_team(contributors=True)
    malformed = {"answer": "Research contribution", "continue": False,
                 "evidence_ids": ["invented-event"],
                 "ledger": {"requirements": ["Explain assumptions"], "outline": [3],
                             "evidence_spans": [], "source_references": []}}
    searcher = ScriptedModel([malformed])
    writer = ScriptedModel([{"answer": "Must never run"}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "error"
    assert len(searcher.calls) == 1
    assert not [event for event in services.events if event["kind"] == "protocol_warning"]


def test_final_tool_payload_is_checked_after_merging_completion_fields():
    team = make_team()
    model = ScriptedModel([
        {"answer": "The complete guide explains the requested topic.", "continue": True,
         "tools": [{"name": "final_answer", "arguments": {"answer": "# A researched guide"}}]},
        {"tools": [{"name": "final_answer", "arguments": {
            "answer": "The guide states the central claim and its assumptions."}}]},
    ])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert "substantive deliverable" in str(result.trajectory[0].error)


def test_repeated_title_after_quality_correction_fails_without_final_artifact():
    team = make_team()
    title = {"answer": "# A researched guide", "continue": False}
    model = ScriptedModel([title, title, {"answer": "Must never run"}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == len(result.trajectory) == 2
    assert "writer" not in services.artifacts
    assert len([event for event in services.events if event["kind"] == "quality_warning"]) == 1
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_quality_correction_does_not_expand_role_call_budget():
    team = make_team()
    team.agents[-1].max_calls = 1
    model = ScriptedModel([{"answer": "# A researched guide"}, {"answer": "Must never run"}])
    services, context = make_services(team, {"writer": model})
    services.model_factory = lambda agent_id: _SinglePassModel(
        model, services, team.agents[-1].model_dump(mode="json"), team.model_dump(mode="json"))
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == services.calls == 1
    assert "AgentSpec.max_calls exhausted" in str(result.trajectory[-1].error)


def test_general_knowledge_researcher_completes_without_external_observations():
    pool = seed_pool()
    team = make_team(checkpoints=["Address peer requests if any"])
    team.agents[-1] = AgentSpec(**{**team.agents[-1].model_dump(mode="json"),
        "pool_agent_id": "searcher", "pool_agent_version": 1})

    def complete(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["knowledge_policy"] == "model_general_knowledge_allowed"
        assert payload["allowed_tool_schemas"] == "[]"
        assert payload["persistent_agent"]["role"] == "Knowledge Researcher"
        assert payload["persistent_agent"]["prompt"].startswith("Organize relevant model general knowledge")
        assert "never claim external retrieval" in messages[0]["content"]
        assert "do not continue merely to wait for nonexistent tools" in messages[0]["content"]
        assert "status=not_applicable" in messages[0]["content"]
        return {"answer": "A guide based on general knowledge with explicit uncertainty.",
                "continue": False, "evidence_ids": [], "checkpoints": {
                    "Address peer requests if any": {"status": "not_applicable",
                        "reason": "No peer request was received.", "evidence_ids": []}}}

    model = ScriptedModel([complete])
    services, context = make_services(team, {"writer": model}, pool=pool,
                                      knowledge_policy="model_general_knowledge_allowed")
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 1
    assert not any(event["kind"] == "retrieved" for event in services.events)
    assert next(profile for profile in pool.profiles if profile.pool_agent_id == "searcher").role == "Searcher"


def test_writer_can_cite_delivered_upstream_artifact():
    team = make_team(contributors=True)
    observed = {}

    def complete(messages):
        payload = json.loads(messages[1]["content"])
        upstream = payload["shared_ledger"]["contributions"][0]
        assert upstream["answer"] == "Supported findings"
        assert payload["scheduling"]["reactivate_completed_agents"] is True
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


def test_repeated_nonterminal_response_gets_correction_then_fails_without_progress():
    repeated = {"answer": "A title only", "continue": True,
                "evidence_ids": [], "checkpoints": {}}
    reordered = json.dumps(dict(reversed(list(repeated.items()))), indent=4)
    model = ScriptedModel([repeated, reordered, repeated])
    team = make_team()
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == 3
    assert "Continue the assigned work with substantive facts, reasoning, results" in model.calls[1][-1]["content"]
    assert "Continuation correction" in model.calls[2][-1]["content"]
    warnings = [event for event in services.events if event["kind"] == "continuation_warning"]
    assert len(warnings) == 1
    assert "Repeated unchanged nonterminal response" in str(result.trajectory[-1].error)
    assert "writer" not in services.artifacts
    assert len([event for event in services.events if event["kind"] == "artifact_published"]) == 2


def test_nonterminal_progress_resets_no_progress_guard():
    replies = [
        {"answer": "Initial draft", "continue": True, "evidence_ids": [], "checkpoints": {}},
        {"answer": "Initial draft", "continue": True, "evidence_ids": [], "checkpoints": {}},
        {"answer": "Expanded draft with the requested analysis", "continue": True,
         "evidence_ids": [], "checkpoints": {}},
        {"answer": "Expanded draft with the requested analysis", "continue": True,
         "evidence_ids": [], "checkpoints": {}},
        {"answer": "Complete guide", "continue": False, "evidence_ids": [], "checkpoints": {}},
    ]
    model = ScriptedModel(replies)
    team = make_team()
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Complete guide" and result.terminated_reason == "final_answer"
    assert len(model.calls) == 5
    assert len([event for event in services.events if event["kind"] == "continuation_warning"]) == 2


def test_repeated_nonterminal_response_can_recover_after_correction():
    repeated = {"answer": "A title only", "continue": True}

    def corrected(messages):
        assert "Continuation correction" in messages[-1]["content"]
        return {"answer": "Completed substantive guide", "continue": False}

    model = ScriptedModel([repeated, repeated, corrected])
    team = make_team()
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Completed substantive guide" and result.terminated_reason == "final_answer"
    assert len(model.calls) == 3


def test_identical_responses_with_new_peer_input_do_not_trigger_no_progress_guard():
    repeated = {"answer": "Still assessing the peer findings", "continue": True}
    team = make_team(contributors=True)
    counter = iter(range(3))

    def receive_peer(messages):
        services.event("searcher", "peer_message", {"content": f"New finding {next(counter)}"},
                       recipient="writer")
        return repeated

    model = ScriptedModel([receive_peer, receive_peer, receive_peer,
                           {"answer": "Guide with the received findings", "continue": False}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Guide with the received findings"
    assert len(model.calls) == 4
    assert not [event for event in services.events if event["kind"] == "continuation_warning"]


def test_identical_tool_requests_keep_receiving_new_observations():
    team = make_team(tools=["search"])
    request = {"continue": True, "tools": [{"name": "search", "arguments": {"query": "sources"}}]}
    observed = []

    def search(query):
        observed.append(query)
        return f"New source batch {len(observed)}"

    model = ScriptedModel([request, request, request, {"answer": "Guide from the observations"}])
    services, context = make_services(team, {"writer": model}, tools={"search": search})
    result = run_writer(team, services, context)
    assert result.answer == "Guide from the observations" and observed == ["sources"] * 3
    assert len(model.calls) == 4
    assert all(call[-1]["content"].startswith("Public observations: ") for call in model.calls[1:])
    assert not [event for event in services.events if event["kind"] == "continuation_warning"]


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


def test_iterative_protocol_correction_recovers_invalid_checkpoint_shape():
    team = make_team(checkpoints=["accuracy"])
    model = ScriptedModel([
        {"answer": "Draft", "continue": True, "checkpoints": {"accuracy": "status=complete"}},
        {"answer": "Complete guide", "continue": False,
         "checkpoints": {"accuracy": {"status": "passed", "reason": "Checked.", "evidence_ids": []}}},
    ])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Complete guide" and result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert len(result.trajectory) == 2
    assert str(result.trajectory[0].error).startswith("A checkpoint must be")
    assert "ResponseProtocolError" in result.trajectory[0].observations
    assert result.trajectory[0].model_output_messages.content == json.dumps({
        "answer": "Draft", "continue": True, "checkpoints": {"accuracy": "status=complete"}})
    assert result.trajectory[1].error is None
    assert result.metadata["checkpoint_reports"]["accuracy"]["status"] == "passed"
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert not [event for event in services.events if event["kind"] == "execution_error"]
    assert len([event for event in services.events if event["kind"] == "artifact_published"]) == 1


def test_repeated_invalid_checkpoint_fails_after_one_correction_without_tool_dispatch():
    team = make_team(tools=["search"], checkpoints=["accuracy"])
    invalid = {"answer": "Unvalidated draft", "checkpoints": {"accuracy": "status=complete"},
               "tools": [{"name": "search", "arguments": {"query": "must not dispatch"}}]}
    model = ScriptedModel([invalid, invalid, {
        "answer": "This response must never run", "checkpoints": {"accuracy": True}}])
    dispatched = []
    services, context = make_services(team, {"writer": model},
                                      tools={"search": lambda query: dispatched.append(query)})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == len(result.trajectory) == 2
    assert all(str(step.error).startswith("A checkpoint must be") for step in result.trajectory)
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert len([event for event in services.events if event["kind"] == "execution_error"]) == 1
    assert not dispatched and "writer" not in services.artifacts
    assert not any(event["kind"] in {"artifact_published", "final_answer"} for event in services.events)


def test_checkpoint_correction_still_requires_every_exact_assigned_checkpoint():
    team = make_team(checkpoints=["Verify source assumptions", "Address uncertainty"])
    model = ScriptedModel([
        {"answer": "Draft", "checkpoints": {"Verify source assumptions": "not checked"}},
        {"answer": "Nominal final answer", "continue": False, "checkpoints": {
            "Verify source assumptions": True,
            "Address uncertainties": {"status": "unverified", "reason": "No source available."}}},
    ])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == 2
    assert str(result.trajectory[0].error).startswith("A checkpoint must be")
    assert "Unconfirmed checkpoints" in str(result.trajectory[1].error)
    assert "Address uncertainty" in str(result.trajectory[1].error)
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


@pytest.mark.parametrize("invalid, error", [
    ({"answer": "Unsupported guide", "evidence_ids": ["unseen"]}, "Completion cites evidence not observed"),
    ({"answer": "Unsupported guide", "checkpoints": {"accuracy": {
        "status": "passed", "reason": "Cited a source", "evidence_ids": ["unseen"]}}},
     "Checkpoint cites evidence not observed"),
    ({"tools": [{"name": "send_message", "arguments": {
        "recipient": "invented", "content": "Please help"}}]}, "exact peer agent_id"),
])
@pytest.mark.parametrize("after_checkpoint_correction", [False, True])
def test_unrelated_protocol_errors_fail_without_a_checkpoint_correction(
        invalid, error, after_checkpoint_correction):
    team = make_team(contributors=True, checkpoints=["accuracy"])
    replies = [invalid, {"answer": "This must never run", "checkpoints": {"accuracy": True}}]
    if after_checkpoint_correction:
        replies.insert(0, {"answer": "Draft", "checkpoints": {"accuracy": "wrong shape"}})
    model = ScriptedModel(replies)
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == len(result.trajectory) == 1 + after_checkpoint_correction
    assert error in str(result.trajectory[-1].error)
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == after_checkpoint_correction
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_checkpoint_correction_does_not_reset_repeated_no_progress_protection():
    team = make_team(checkpoints=["accuracy"])
    repeated = {"answer": "A title only", "continue": True, "checkpoints": {"accuracy": True}}
    model = ScriptedModel([
        repeated, repeated,
        {"answer": "A title only", "continue": True, "checkpoints": {"accuracy": "wrong shape"}},
        repeated,
        {"answer": "This must never run", "checkpoints": {"accuracy": True}},
    ])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == 4
    assert "Repeated unchanged nonterminal response" in str(result.trajectory[-1].error)
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert len([event for event in services.events if event["kind"] == "continuation_warning"]) == 1
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_recent_memory_retains_invalid_draft_with_its_protocol_correction():
    team = make_team(checkpoints=["accuracy"])
    pool = seed_pool()
    team.agents[-1] = AgentSpec(**{**team.agents[-1].model_dump(mode="json"),
        "pool_agent_id": "writer", "pool_agent_version": 1,
        "harness": AgentHarnessPolicy(memory_policy="recent", memory_window=1)})
    invalid = {"answer": "Draft whose substance must be preserved", "continue": True,
               "checkpoints": {"accuracy": "checked carefully"}}

    def corrected(messages):
        assert len(messages) == 4
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert json.loads(messages[-2]["content"]) == invalid
        assert messages[-1]["content"].startswith("Protocol correction:")
        assert "Every assigned checkpoint value" in messages[-1]["content"]
        assert json.loads(messages[1]["content"])["agent"]["checkpoints"] == ["accuracy"]
        return {"answer": "Substantive revised guide", "continue": False,
                "checkpoints": {"accuracy": {"status": "unverified", "reason": "No external source available.",
                                             "evidence_ids": []}}}

    model = ScriptedModel([invalid, corrected])
    services, context = make_services(team, {"writer": model}, pool=pool)
    result = run_writer(team, services, context)
    assert result.answer == "Substantive revised guide" and result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert result.trajectory[0].error is not None and result.trajectory[1].error is None


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


def test_iterative_synthesizer_receives_final_artifact_contract_even_with_critic_role():
    team = make_team()
    team.agents[-1].role = "Critic"
    model = ScriptedModel([{"answer": "Complete requested guide", "continue": False}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Complete requested guide"
    system = model.calls[0][0]["content"]
    assert "final synthesizer even if your persistent role is a reviewer or critic" in system
    assert "cannot replace the requested deliverable" in system


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
