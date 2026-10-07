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
    FINAL_ARTIFACT_CONTRACT, FINAL_QUALITY_SPINE, FINAL_SUBMISSION_GATE,
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


@pytest.mark.parametrize("question,answer", [
    ("Write a headline for a newspaper article about migratory birds.", "Title: Birds Return Home"),
    ("Write a three-line poem using Markdown heading lines.", "# Birds rise\n# Wings cross the moon\n# Dawn finds home"),
    ("Provide a title for a report on materials.", "# Materials for Tomorrow"),
    ("为关于候鸟迁徙的新闻报道拟一个标题。", "# 候鸟重返故乡"),
    ("Return only Paris.", "Paris"),
    ("Name the capital city discussed in the report.", "Paris"),
    ("只输出北京。", "北京"),
])
def test_requested_heading_genre_and_exact_short_answers_remain_valid(question, answer):
    task = PublicTask(task_id="heading-genre", question=question)
    assert _completion_quality_error(answer, task) is None


@pytest.mark.parametrize("question,answer", [
    ("Write a report discussing newspaper headlines.", "# Birds Return Home"),
    ("Write a report and name the assumptions behind its comparison.", "# Materials report"),
    ("Write a headline and then write a complete report on migratory birds.", "Title: Birds Return Home"),
    ("Write a headline and a complete report on migratory birds.", "Title: Birds Return Home"),
    ("撰写一份关于两种材料的研究报告。", "# 材料研究报告"),
    ("Return only the complete report on materials.", "# Materials report"),
    ("只输出完整研究报告正文。", "# 材料研究报告"),
    ("拟一个标题并撰写完整研究报告。", "# 材料研究报告"),
])
def test_title_only_response_cannot_replace_a_required_complete_report(question, answer):
    task = PublicTask(task_id="complete-report", question=question)
    assert _completion_quality_error(answer, task)


@pytest.mark.parametrize("question,source,answer,expected_error", [
    ("Write a complete materials report.", "The source says to name the material.",
     "# Materials report", True),
    ("Return only Paris.", "Write a complete report about the source material.", "Paris", False),
])
def test_shared_evidence_source_text_does_not_change_the_requested_output_guard(
        question, source, answer, expected_error):
    from jit_mas.evidence import EVIDENCE_QUESTION_HEADER, EVIDENCE_TASK_CONSTRAINT, EVIDENCE_SCOPE, _json

    task_id = "output-evidence-boundary"
    body = {"task_id": task_id, "scope": EVIDENCE_SCOPE,
            "sources": [{"source_id": "source", "text": source}]}
    public = PublicTask(task_id=task_id, question=question + EVIDENCE_QUESTION_HEADER + _json(body),
                        constraints=[EVIDENCE_TASK_CONSTRAINT])
    assert bool(_completion_quality_error(answer, public)) is expected_error


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
    assert "answer <=2400" in model.calls[0][0]["content"]
    assert "24 items per list" in model.calls[0][0]["content"]
    assert "advisory compactness targets, not completeness limits" in model.calls[0][0]["content"]
    assert "cumulative snapshot" in model.calls[0][0]["content"]
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
    # This fixture has no provider usage, so UTF-8 bytes conservatively bound the
    # expanded public-quality prompt on both the malformed and correction calls.
    # This test exercises accounting rather than a small token-limit boundary;
    # use the common task envelope so prompt growth cannot block its correction.
    ledger = BudgetLedger(max_calls=2, max_tokens=2_000_000, max_tool_calls=0)
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


def test_resumed_recent_memory_receives_current_snapshot_and_retracts_invalid_items():
    team = make_team(contributors=True)
    pool = seed_pool()
    team.agents[0] = AgentSpec(**{**team.agents[0].model_dump(mode="json"),
        "pool_agent_id": "searcher", "pool_agent_version": 1,
        "harness": AgentHarnessPolicy(memory_policy="recent", memory_window=1)})
    team.agents[-1] = AgentSpec(**{**team.agents[-1].model_dump(mode="json"),
        "pool_agent_id": "writer", "pool_agent_version": 1})
    first_ledger = {"requirements": ["List every qualifying item"],
                    "outline": ["Alpha qualifies", "Beta qualifies", "Gamma qualifies"],
                    "evidence_spans": [], "source_references": []}
    revised_ledger = {"requirements": first_ledger["requirements"],
                      "outline": ["Alpha qualifies", "Beta qualifies", "Delta qualifies",
                                  "Gamma is withdrawn because it does not meet the inclusion condition."],
                      "evidence_spans": [], "source_references": []}

    def initial(messages):
        assert json.loads(messages[1]["content"])["own_contribution"] is None
        return {"answer": "Initial qualification results.", "ledger": first_ledger}

    def revise(messages):
        payload = json.loads(messages[1]["content"])
        own = payload["own_contribution"]
        assert own["agent_id"] == "searcher" and own["ledger"] == first_ledger
        assert own["complete"] is False
        assert not any(message["role"] == "assistant" for message in messages)
        assert "cumulative snapshot" in messages[0]["content"]
        assert "partial delta" in messages[0]["content"]
        assert "Verify Gamma and include Delta" in messages[-1]["content"]
        return {"answer": "Corrected qualification results; Gamma is withdrawn.",
                "ledger": revised_ledger}

    def finalize(messages):
        refreshed = next(message["content"].rsplit("Updated shared ledger: ", 1)[1]
                         for message in reversed(messages)
                         if "Updated shared ledger: " in message.get("content", ""))
        shared = json.loads(refreshed)
        contribution = next(item for item in shared["contributions"] if item["agent_id"] == "searcher")
        assert contribution["ledger"] == revised_ledger
        assert "Gamma qualifies" not in [item["text"] for item in shared["outline"]]
        assert shared["tool_evidence"] == []
        return {"answer": "Alpha, Beta, Delta."}

    searcher = ScriptedModel([initial, revise])
    writer = ScriptedModel([{"tools": [{"name": "send_message", "arguments": {
        "recipient": "searcher", "content": "Verify Gamma and include Delta"}}]}, finalize])
    services, context = make_services(team, {"searcher": searcher, "writer": writer}, pool=pool)
    result = run_team("List every qualifying item.", context, team, services)
    assert result.answer == "Alpha, Beta, Delta.", [str(step.error) for sub_run in result.sub_runs
                                                  for step in sub_run.trajectory]
    assert len(searcher.calls) == len(writer.calls) == 2
    assert services.artifacts["searcher"]["ledger"] == revised_ledger
    publications = [event for event in services.events
                    if event["kind"] == "artifact_published" and event["agent_id"] == "searcher"]
    assert len(publications) == 2
    assert any(event["kind"] == "agent_resumed" and event["agent_id"] == "searcher"
               for event in services.events)


def test_iterative_exact_short_answer_is_submitted_without_padding():
    team = make_team()

    def complete(messages):
        assert "not a target answer length" in messages[0]["content"]
        assert "short answers need no padding" in messages[0]["content"]
        assert json.loads(messages[1]["content"])["public_task"]["question"] == "Return only Paris."
        return {"answer": "Paris"}

    model = ScriptedModel([complete])
    services, context = make_services(team, {"writer": model})
    services.public_task = PublicTask(task_id="short-output", question="Return only Paris.")
    result = run_writer(team, services, context)
    assert result.answer == "Paris" and result.terminated_reason == "final_answer"
    assert len(model.calls) == 1


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
    assert "original system's advisory compactness and handoff guidance" in correction
    assert "cumulative current-valid ledger" in correction
    assert "supported sets and source references" in correction
    assert "Put the substantive contribution body in answer" not in correction
    system = model.calls[0][0]["content"]
    assert "not a section heading alone" in system
    assert "3000-5000-word" in system
    assert "legal or policy obligations" in system
    assert all("checked intermediate results" in prompt for prompt in (system, correction))


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
    assert "80-85%" in system and "not a target answer length" in system
    assert "Never invent a long directory, catalog, numbered sequence" in system
    assert "rewrite one complete JSON response from the original task" in correction
    assert "Preserve the key arguments, calculations, required items" in correction
    assert "delete duplicate passages, guessed directories" in correction


@pytest.mark.parametrize("tool_name", [None, "final_answer", "complete"])
def test_unknown_prose_fields_require_one_model_authored_complete_answer(tool_name):
    team = make_team(checkpoints=["accuracy"])
    broken = {"answer": "The report explains ", "a quoted phrase": "and its implications.",
              "checkpoints": {"accuracy": True}}
    malformed = ({"tools": [{"name": tool_name, "arguments": broken}]}
                 if tool_name else broken)
    complete = {"answer": 'The report explains "a quoted phrase" and its implications.',
                "checkpoints": {"accuracy": True}, "continue": False}

    def corrected(messages):
        assert json.loads(messages[-2]["content"]) == malformed
        assert "Unknown execution response fields" in messages[-1]["content"]
        assert "never split its prose across invented JSON keys" in messages[-1]["content"]
        assert "Escape quotes inside the answer string" in messages[-1]["content"]
        assert "writer" not in services.artifacts
        return complete

    model = ScriptedModel([malformed, corrected])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer" and result.answer == complete["answer"]
    assert len(model.calls) == len(result.trajectory) == 2
    assert json.loads(result.trajectory[0].model_output_messages.content) == malformed
    assert "Unknown execution response fields" in str(result.trajectory[0].error)
    assert result.trajectory[1].error is None
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert len([event for event in services.events if event["kind"] == "artifact_published"]) == 1


def test_repeated_unknown_prose_fields_fail_without_submission_or_tool_dispatch():
    team = make_team(tools=["search"])
    malformed = {"answer": "The report explains ", "a quoted phrase": "and its implications.",
                 "tools": [{"name": "search", "arguments": {"query": "must not dispatch"}}]}
    model = ScriptedModel([malformed, malformed, {"answer": "Must never run."}])
    dispatched = []
    services, context = make_services(team, {"writer": model},
                                      tools={"search": lambda query: dispatched.append(query)})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "error" and result.answer is None
    assert len(model.calls) == len(result.trajectory) == 2
    assert not dispatched and "writer" not in services.artifacts
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1
    assert not any(event["kind"] in {"artifact_published", "final_answer"} for event in services.events)


def test_unknown_contributor_fields_must_be_repaired_into_the_supported_ledger():
    team = make_team(contributors=True)
    ledger = {"requirements": [], "outline": ["Useful facts and checked assumptions."],
              "evidence_spans": [], "source_references": []}
    malformed = {"answer": "Contribution summary.", "ledger": ledger,
                 "unexpected_body": "Further useful facts."}
    corrected = {"answer": "Contribution summary.", "ledger": {
        **ledger, "outline": [*ledger["outline"], "Further useful facts."]}}
    searcher = ScriptedModel([malformed, corrected])
    writer = ScriptedModel([{"answer": "The complete guide incorporates the contributed facts."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "final_answer"
    assert len(searcher.calls) == 2 and len(writer.calls) == 1
    assert services.artifacts["searcher"]["ledger"] == corrected["ledger"]
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1


@pytest.mark.parametrize("extra_field", ["entities", "items"])
def test_unknown_ledger_field_is_repaired_without_silent_content_loss(extra_field):
    team = make_team(contributors=True)
    initial = {"requirements": [], "outline": ["A valid supported claim."],
               "evidence_spans": [], "source_references": []}
    malformed = {"answer": "Contribution summary.", "ledger": {
        **initial, extra_field: ["Entity Alpha meets every public qualification condition."]}}
    corrected = {"answer": "Contribution summary.", "ledger": {
        **initial, "outline": [*initial["outline"], *malformed["ledger"][extra_field]]}}

    def repair(messages):
        correction = messages[-1]["content"]
        assert "Unknown contributor ledger fields: " in correction and extra_field in correction
        assert "Put substantive content from unsupported ledger keys into outline" in correction
        assert "Entity Alpha" not in correction
        assert json.loads(messages[-2]["content"]) == malformed
        return corrected

    searcher = ScriptedModel([malformed, repair])
    writer = ScriptedModel([{"answer": "The guide includes the qualification result for Entity Alpha."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "final_answer"
    assert len(searcher.calls) == 2 and len(writer.calls) == 1
    assert services.artifacts["searcher"]["ledger"] == corrected["ledger"]
    warnings = [event for event in services.events if event["kind"] == "protocol_warning"]
    assert len(warnings) == 1 and "Entity Alpha" not in warnings[0]["content"]


def test_repeated_unknown_ledger_fields_fail_after_the_existing_single_correction():
    team = make_team(contributors=True)
    malformed = {"answer": "Contribution summary.", "ledger": {
        "requirements": [], "outline": [], "evidence_spans": [], "source_references": [],
        "items": ["Content that must not be silently lost."]}}
    searcher = ScriptedModel([malformed, malformed, {"answer": "Must never run."}])
    writer = ScriptedModel([{"answer": "Must never run."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "error" and result.answer is None
    assert len(searcher.calls) == 2 and len(writer.calls) == 0
    assert "searcher" not in services.artifacts
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1


def test_unknown_prose_fields_cannot_hide_fabricated_evidence_from_correction():
    team = make_team()
    malformed = {"answer": "The report explains ", "a quoted phrase": "and its implications.",
                 "evidence_ids": ["invented-event"]}
    model = ScriptedModel([malformed, {"answer": "Must never run."}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "error" and result.answer is None
    assert len(model.calls) == 1
    assert not [event for event in services.events if event["kind"] == "protocol_warning"]


def test_fabricated_evidence_id_gets_one_ledger_shape_correction():
    team = make_team(contributors=True)
    malformed = {"answer": "Research contribution", "continue": False,
                 "evidence_ids": ["invented-event"],
                 "ledger": {"requirements": ["Explain assumptions"], "outline": [3],
                             "evidence_spans": [], "source_references": []}}
    corrected = {"answer": "Research contribution with its limitation.", "continue": False,
                 "evidence_ids": [],
                 "ledger": {"requirements": ["Explain assumptions"],
                            "outline": ["State the central claim and its limitation."],
                            "evidence_spans": [], "source_references": []}}
    searcher = ScriptedModel([malformed, corrected])
    writer = ScriptedModel([{"answer": "The guide states the claim and its limitation."}])
    services, context = make_services(team, {"searcher": searcher, "writer": writer})
    result = run_team("Write a guide.", context, team, services)
    assert result.terminated_reason == "final_answer"
    assert len(searcher.calls) == 2
    warnings = [event for event in services.events if event["kind"] == "protocol_warning"]
    assert len(warnings) == 1
    assert "unobserved evidence citation is correctable once" in searcher.calls[1][-1]["content"]


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


def test_peer_observation_published_during_call_gets_correction_before_delivery():
    team = make_team(contributors=True)

    def unsupported_completion(messages):
        unseen = services.event("searcher", "retrieved", {"output": "Evidence published after call started"})
        return {"answer": "Unsupported guide", "evidence_ids": [unseen]}

    model = ScriptedModel([unsupported_completion,
                           {"answer": "Guide with an explicit evidence limitation.",
                            "evidence_ids": []}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Guide with an explicit evidence limitation."
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1


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


@pytest.mark.parametrize("initial_checks", [{}, {"accuracy": False}, {"Accuracy": True}])
def test_missing_checkpoint_gets_one_honest_model_authored_correction(initial_checks):
    team = make_team(checkpoints=["accuracy"])
    invalid = {"answer": "A guide with a claim that requires external verification.",
               "checkpoints": initial_checks}

    def corrected(messages):
        assert 'Exact assigned checkpoint keys: ["accuracy"]' in messages[-1]["content"]
        assert "Do not invent a passed finding" in messages[-1]["content"]
        return {"answer": "A guide identifying the unsupported claim and its limitation.",
                "checkpoints": {"accuracy": {"status": "unverified",
                    "reason": "The input does not provide evidence for the disputed claim.",
                    "evidence_ids": []}}}

    model = ScriptedModel([invalid, corrected])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == len(result.trajectory) == 2
    assert str(result.trajectory[0].error).startswith("Unconfirmed checkpoints")
    assert result.trajectory[0].model_output_messages.content == json.dumps(invalid)
    assert result.metadata["checkpoint_reports"]["accuracy"]["status"] == "unverified"
    assert result.metadata["checkpoint_reports"]["accuracy"]["independently_verified"] is False


def test_repeated_missing_checkpoint_fails_after_one_correction():
    team = make_team(checkpoints=["accuracy"])
    invalid = {"answer": "A draft without its assigned check."}
    model = ScriptedModel([invalid, invalid])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == 2
    assert all(str(step.error).startswith("Unconfirmed checkpoints") for step in result.trajectory)
    assert "writer" not in services.artifacts
    assert not any(event["kind"] == "final_answer" for event in services.events)


def test_missing_checkpoint_correction_drops_fabricated_evidence():
    team = make_team(checkpoints=["accuracy"])
    model = ScriptedModel([
        {"answer": "An unsupported answer.", "evidence_ids": ["invented"]},
        {"answer": "An answer with an explicit limitation.", "evidence_ids": [],
         "checkpoints": {"accuracy": {"status": "unverified",
                                         "reason": "No observed evidence was available.",
                                         "evidence_ids": []}}},
    ])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "An answer with an explicit limitation."
    assert result.terminated_reason == "final_answer"
    assert len(model.calls) == 2
    assert result.metadata["checkpoint_reports"]["accuracy"]["status"] == "unverified"
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == 1


def test_missing_checkpoint_correction_keeps_configured_role_call_ceiling():
    team = make_team(checkpoints=["accuracy"])
    team.agents[-1].max_calls = 1
    model = ScriptedModel([{"answer": "A draft without its assigned check."}])
    services, context = make_services(team, {"writer": model})
    services.model_factory = lambda agent_id: _SinglePassModel(
        model, services, team.agents[-1].model_dump(mode="json"), team.model_dump(mode="json"))
    result = run_writer(team, services, context)
    assert result.answer is None and result.terminated_reason == "error"
    assert len(model.calls) == services.calls == 1
    assert "AgentSpec.max_calls exhausted" in str(result.trajectory[-1].error)


def test_closed_book_ledger_correction_preserves_knowledge_in_outline():
    team = make_team(contributors=True)
    invalid = {"answer": "Remembered mechanisms for the guide.", "ledger": {
        "requirements": ["Explain the mechanism"], "outline": [],
        "source_references": ["Remembered report, not retrieved"],
        "evidence_spans": ["A remembered mechanism and its limitation"]}}

    def corrected(messages):
        assert "do not convert remembered citations into source objects" in messages[-1]["content"]
        assert "move useful remembered claims" in messages[-1]["content"]
        return {"answer": "Remembered mechanisms for the guide.", "ledger": {
            "requirements": ["Explain the mechanism"],
            "outline": ["A remembered mechanism and its limitation; report not retrieved."],
            "source_references": [], "evidence_spans": []}}

    model = ScriptedModel([invalid, corrected])
    services, context = make_services(team, {"searcher": model},
                                      knowledge_policy="model_general_knowledge_allowed")
    result = _run_agent_iterative(team.agents[0].model_dump(mode="json"),
                                 team.model_dump(mode="json"), context, services)
    assert result.terminated_reason == "subtask_complete"
    assert len(model.calls) == 2
    handoff = services.artifacts["searcher"]["ledger"]
    assert handoff["outline"] == ["A remembered mechanism and its limitation; report not retrieved."]
    assert handoff["source_references"] == handoff["evidence_spans"] == []


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


@pytest.mark.parametrize("invalid, error, correctable", [
    ({"answer": "Unsupported guide", "evidence_ids": ["unseen"]},
     "Completion cites evidence not observed", True),
    ({"answer": "Unsupported guide", "checkpoints": {"accuracy": {
        "status": "passed", "reason": "Cited a source", "evidence_ids": ["unseen"]}}},
     "Checkpoint cites evidence not observed", True),
    ({"tools": [{"name": "send_message", "arguments": {
        "recipient": "invented", "content": "Please help"}}]},
     "exact peer agent_id", False),
])
@pytest.mark.parametrize("after_checkpoint_correction", [False, True])
def test_protocol_errors_allow_one_evidence_correction_but_reject_authority_errors(
        invalid, error, correctable, after_checkpoint_correction):
    team = make_team(contributors=True, checkpoints=["accuracy"])
    replies = [invalid, {"answer": "This must never run", "checkpoints": {"accuracy": True}}]
    if after_checkpoint_correction:
        replies.insert(0, {"answer": "Draft", "checkpoints": {"accuracy": "wrong shape"}})
    model = ScriptedModel(replies)
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    should_recover = correctable and not after_checkpoint_correction
    if should_recover:
        assert result.answer == "This must never run"
        assert result.terminated_reason == "final_answer"
    else:
        assert result.answer is None and result.terminated_reason == "error"
    expected_calls = 1 + after_checkpoint_correction + should_recover
    assert len(model.calls) == len(result.trajectory) == expected_calls
    error_step = result.trajectory[0] if should_recover else result.trajectory[-1]
    assert error in str(error_step.error)
    assert len([event for event in services.events if event["kind"] == "protocol_warning"]) == (
        after_checkpoint_correction or should_recover)
    if should_recover:
        assert any(event["kind"] == "final_answer" for event in services.events)
    else:
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
    team.agents[-1].task_prompt = "Review the draft. Do not rewrite the entire answer."
    original_prompt = team.agents[-1].task_prompt
    model = ScriptedModel([{"answer": "Complete requested guide", "continue": False}])
    services, context = make_services(team, {"writer": model})
    result = run_writer(team, services, context)
    assert result.answer == "Complete requested guide"
    system = model.calls[0][0]["content"]
    assert "final synthesizer even if your persistent role is a reviewer or critic" in system
    assert "cannot replace the requested deliverable" in system
    assert FINAL_ARTIFACT_CONTRACT in system
    assert FINAL_SUBMISSION_GATE in system
    assert FINAL_QUALITY_SPINE in system
    payload = json.loads(model.calls[0][1]["content"])
    assert payload["agent"]["execution_role"] == "final_writer"
    assert payload["agent"]["task_prompt"].endswith(FINAL_ARTIFACT_CONTRACT)
    assert original_prompt in payload["agent"]["task_prompt"]
    assert payload["terminal_assignment"] == FINAL_ARTIFACT_CONTRACT
    assert team.agents[-1].task_prompt == original_prompt


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
