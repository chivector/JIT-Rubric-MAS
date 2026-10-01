"""Scripted software checks; these do not measure real-model task quality."""

import copy
import json
from collections import Counter

import pytest

from jit_mas.attribution import RubricAttributor, feedback_view
from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.planning import GlobalAnalyzer, JsonModelCalls
from jit_mas.schemas import (
    AttributionFinding, EvaluationFeedback, LocalPlan, PlannedTeam, Prediction,
    PublicTask, RubricAlignment, RubricGraph, TeamSpec,
)


def rubric(rid="r1", requirement="Compare alternatives with reproducible evidence"):
    return {"rubric_id": rid, "requirement": requirement}


def agent(aid="a1", rubrics=None, **kwargs):
    return {"agent_id": aid, "role": "Analyst", "capability": "evidence comparison",
            "rubric_ids": ["r1"] if rubrics is None else rubrics,
            "responsibilities": ["Check the source evidence"], "max_calls": 1, **kwargs}


def team(agents=None, coverage=None):
    agents = agents or [agent()]
    coverage = {"r1": ["a1"]} if coverage is None else coverage
    return {"agents": agents, "synthesizer_id": agents[-1]["agent_id"],
            "coverage": coverage, "primary": {key: val[0] for key, val in coverage.items()}}


class Scripted:
    def __init__(self, handler):
        self.handler = handler
        self.inputs = []

    def __call__(self, messages):
        assert len(messages) == 2
        data = json.loads(messages[1]["content"])
        self.inputs.append(copy.deepcopy(data))
        return json.dumps(self.handler(data))


def test_local_challenge_changes_graph_and_team_and_preserves_initial():
    initial = {"graph": {"rubrics": [rubric()]}, "candidates": [agent()]}
    revised = {"graph": {"rubrics": [rubric(), rubric("r2", "Check conflicting evidence")]},
               "team": team([agent("a1", ["r1"]),
                             agent("a2", ["r1", "r2"], depends_on=["a1"])],
                            {"r1": ["a1", "a2"], "r2": ["a2"]})}
    global_model = Scripted(lambda data: initial if data["phase"] == "predict" else revised)
    local_model = Scripted(lambda data: {
        "agent_id": "a1", "capability": "evidence comparison", "rubric_ids": ["r1"],
        "additions": [rubric("r2", "Check conflicting evidence")],
        "challenge": "A separate conflicting-evidence check is needed",
        "required_inputs": ["primary source claims"], "expected_outputs": ["evidence table"]})
    analyzer = GlobalAnalyzer(global_model, lambda aid: local_model)
    public = PublicTask(task_id="task", question="Compare two evidence-based proposals",
                        constraints=["Discuss limitations"])
    result = analyzer.build(public)
    assert len(analyzer.last_prediction.graph.rubrics) == 1
    assert len(result.graph.rubrics) == 2
    assert len(result.team.agents) == 2
    assert result.team.agents[1].depends_on == ["a1"]
    assert result.local_plans[0].challenge
    assert global_model.inputs[1]["local_plans"][0]["additions"]
    assert all("feedback" not in item for item in global_model.inputs + local_model.inputs)
    assert local_model.inputs[0]["task"]["constraints"] == ["Discuss limitations"]


@pytest.mark.parametrize("legacy_echo", [False, True])
def test_reconcile_omits_plan_echo_schema_but_preserves_independent_local_plans(legacy_echo):
    prediction = Prediction(graph=RubricGraph(rubrics=[rubric()]), candidates=[agent()])
    plans = [LocalPlan(agent_id="a1", capability="evidence comparison", rubric_ids=["r1"],
                       additions=[rubric("r2", "Check source uncertainty")],
                       required_inputs=["primary sources"], expected_outputs=["evidence table"],
                       uncovered=["source uncertainty"], risks=["unavailable sources"],
                       challenge="A reviewer must check conflicting evidence")]
    originals = copy.deepcopy(plans)
    response = {"graph": prediction.graph.model_dump(), "team": team()}
    if legacy_echo:
        response["local_plans"] = [dict(plans[0].model_dump(), challenge="Altered by reconciler")]

    def model(messages):
        schema = json.loads(messages[0]["content"].split("conforming to this JSON Schema:\n")[1])
        assert set(schema["properties"]) == {"schema_version", "graph", "team"}
        assert "local_plans" not in json.dumps(schema)
        assert schema["additionalProperties"] is False
        payload = json.loads(messages[1]["content"])
        assert payload["local_plans"] == [plan.model_dump(mode="json") for plan in originals]
        return json.dumps(response)

    result = GlobalAnalyzer(model).reconcile(PublicTask(task_id="t", question="Compare evidence"),
                                             prediction, plans)
    assert result.local_plans == originals
    assert result.local_plans is not plans
    assert result.local_plans[0] is not plans[0]
    assert result.local_plans[0].additions[0] is not plans[0].additions[0]
    result.local_plans[0].additions[0].requirement = "Changed stored plan"
    result.local_plans[0].required_inputs.append("Changed stored inputs")
    assert plans == originals
    plans[0].risks.append("Changed source risks")
    assert result.local_plans[0].risks == originals[0].risks


@pytest.mark.parametrize("extra", [{"unexpected": True}, {"local_plans": [{"unknown": True}]}])
def test_reconcile_response_retains_strict_validation(extra):
    prediction = Prediction(graph=RubricGraph(rubrics=[rubric()]), candidates=[agent()])
    response = {"graph": prediction.graph.model_dump(), "team": team(), **extra}
    with pytest.raises(ValueError):
        GlobalAnalyzer(Scripted(lambda data: response)).reconcile(
            PublicTask(task_id="t", question="Compare evidence"), prediction, [])


def reviewed_team_response():
    agents = [agent("author"), agent("reviewer", depends_on=["author"]),
              agent("final", depends_on=["reviewer"])]
    final_team = team(agents, {"r1": ["author", "reviewer", "final"]})
    final_team.update(reviewers={"r1": ["reviewer"]}, total_max_calls=6)
    return {"graph": {"rubrics": [rubric()]}, "team": final_team}


@pytest.mark.parametrize("reviewers", [["reviewer"], ["final"], []])
def test_reconcile_prompt_explains_cross_field_checks_and_allows_optional_reviews(reviewers):
    response = reviewed_team_response()
    response["team"]["reviewers"] = {"r1": reviewers} if reviewers else {}

    def model(messages):
        prompt = " ".join(messages[0]["content"].split("conforming to this JSON Schema:")[0].split())
        for invariant in (
            "1 <= len(team.agents) <= limits.max_agents",
            "Agent IDs are unique",
            "synthesizer_id references an existing selected agent",
            "no self-dependency or cycle",
            "synthesizer depends transitively on every other selected agent",
            "every graph edge endpoint exists in graph.rubrics",
            "Every rubric has nonempty coverage and a primary owner in coverage[rubric_id]",
            "Each agent.rubric_ids is exactly the set",
            "reviewer != primary[rubric_id]",
            "primary owner must be an ancestor of the reviewer through depends_on",
            "reviewers is optional",
            "self-checks in checkpoints",
            "sum(agent.max_calls) <= team.total_max_calls <= limits.total_max_calls",
            "team.max_parallel <= limits.max_parallel",
            "All agent tools must be in task.tools",
        ):
            assert invariant in prompt
        return json.dumps(response)

    analyzer = GlobalAnalyzer(model, max_agents=3, total_max_calls=6)
    prediction = Prediction(graph=response["graph"], candidates=[agent("author")])
    result = analyzer.reconcile(PublicTask(task_id="t", question="Compare evidence"), prediction, [])
    assert result.team.reviewers == response["team"]["reviewers"]
    assert [a.agent_id for a in result.team.agents] == ["author", "reviewer", "final"]


@pytest.mark.parametrize("change,error", [
    (lambda t: t.update(reviewers={"r1": ["author"]}), "Independent reviewer"),
    (lambda t: t.update(primary={"r1": "final"}), "Independent reviewer"),
    (lambda t: t["agents"][1].update(depends_on=[]), "Independent reviewer"),
    (lambda t: t.update(agents=[dict(a, max_calls=1) for a in t["agents"]] + [agent("extra")]),
     "configured resource limits"),
    (lambda t: t["agents"].append(agent("author")), "Duplicate agent ID"),
    (lambda t: t.update(synthesizer_id="absent"), "unknown synthesizer"),
    (lambda t: t["agents"][0].update(depends_on=["absent"]), "Unknown execution dependency"),
    (lambda t: t["agents"][0].update(depends_on=["final"]), "contain a cycle"),
    (lambda t: t.update(reviewers={"r1": ["absent"]}), "Unknown rubric owner/reviewer"),
    (lambda t: t.update(primary={"r1": "absent"}), "Unknown primary owner"),
    (lambda t: t["coverage"].update(r1=["reviewer", "final"]), "Primary owner must be in coverage"),
    (lambda t: t.update(primary={}, reviewers={}), "requires coverage and a primary owner"),
    (lambda t: t["coverage"].update(unknown=["author"]), "unknown rubrics"),
    (lambda t: t["agents"][0].update(rubric_ids=[]), "assignments disagree"),
    (lambda t: t["agents"][0].update(tools=["unavailable"]), "unavailable tools"),
    (lambda t: t.update(total_max_calls=2), "allocations exceed team call budget"),
    (lambda t: t.update(total_max_calls=7), "configured resource limits"),
    (lambda t: t.update(max_parallel=3), "configured resource limits"),
    (lambda t: t["agents"][2].update(depends_on=[]), "synthesizer must depend on every"),
])
def test_reconcile_still_rejects_cross_field_violations_without_rewriting(change, error):
    response = reviewed_team_response()
    change(response["team"])
    original = copy.deepcopy(response)
    analyzer = GlobalAnalyzer(Scripted(lambda data: response), max_agents=3, total_max_calls=6)
    prediction = Prediction(graph=response["graph"], candidates=[agent("author")])
    with pytest.raises(ValueError, match=error):
        analyzer.reconcile(PublicTask(task_id="t", question="Compare evidence"), prediction, [])
    assert response == original
    assert json.loads(analyzer.call_records[0]["response"]) == original
    assert len(analyzer.call_records) == 2
    assert json.loads(analyzer.call_records[1]["response"]) == original
    assert all(row["validation_errors"] for row in analyzer.call_records)


@pytest.mark.parametrize("final_tokens", [1024, 8192])
def test_planning_prompts_budget_complete_final_output_in_one_call(final_tokens):
    agents = [dict(agent("research"), max_calls=1, max_tokens=4096),
              dict(agent("final", depends_on=["research"]), max_calls=1, max_tokens=final_tokens)]
    graph = {"rubrics": [rubric()]}
    final_team = team(agents, {"r1": ["research", "final"]})
    final_team.update(total_max_calls=6, max_parallel=1)
    phases = []

    def model(messages):
        prompt = " ".join(messages[0]["content"].split("conforming to this JSON Schema:")[0].split())
        payload = json.loads(messages[1]["content"])
        phases.append(payload["phase"])
        assert payload["limits"]["total_max_calls"] == 6
        if payload["phase"] == "predict":
            assert "complete expected output, including JSON encoding and checkpoint overhead" in prompt
            assert "Set every AgentSpec.max_calls=1 explicitly" in prompt
            assert "no execution-time JSON correction" in prompt
            return json.dumps({"graph": graph, "candidates": agents})
        if payload["phase"] == "local_plan":
            assert "In expected_outputs and risks" in prompt
            assert "entire requested deliverable, not merely editorial feedback" in prompt
            assert "Set LocalPlan.max_calls=1" in prompt
            assert "without a return call" in prompt
            candidate = payload["candidate"]
            return json.dumps({"agent_id": candidate["agent_id"], "capability": candidate["capability"],
                "rubric_ids": ["r1"], "max_calls": 1,
                "expected_outputs": ["Complete final artifact within the response budget"],
                "risks": ["JSON closure and checkpoints need output space"]})
        assert "independently emit the complete final deliverable" in prompt
        assert "JSON escaping, evidence IDs, checkpoints and a margin for valid closure" in prompt
        assert "max_tokens is a per-response output ceiling" in prompt
        assert "genuinely short requested summary may need fewer tokens than its sources" in prompt
        assert "exactly one model call per selected role" in prompt
        assert "synthesizer_id identifies that final Writer" in prompt
        assert "Unused team.total_max_calls is a ceiling" in prompt
        assert "without duplicating drafts, review narration or preambles" in prompt
        return json.dumps({"graph": graph, "team": final_team})

    question = "Give a short synthesis of the research" if final_tokens < 4096 else "Write a complete article"
    analyzer = GlobalAnalyzer(model, lambda aid: model, max_agents=2, max_parallel=1, total_max_calls=6)
    result = analyzer.build(PublicTask(task_id="budget", question=question))
    assert phases == ["predict", "local_plan", "local_plan", "reconcile"]
    assert result.team.agents[-1].max_tokens == final_tokens
    assert result.team.agents[-1].max_calls == 1
    assert sum(a.max_calls for a in result.team.agents) == 2 <= result.team.total_max_calls == 6
    assert all(plan.max_calls == 1 and plan.risks for plan in result.local_plans)


def test_planning_does_not_impose_extra_calls_when_single_call_limit_is_explicit():
    only = dict(agent(), max_calls=1, max_tokens=256)
    prediction = {"graph": {"rubrics": [rubric()]}, "candidates": [only]}
    final_team = team([only])
    final_team.update(total_max_calls=1, max_parallel=1)
    model = Scripted(lambda data: prediction if data["phase"] == "predict" else
                     {"graph": prediction["graph"], "team": final_team})
    result = GlobalAnalyzer(model, max_agents=1, max_parallel=1, total_max_calls=1).build(
        PublicTask(task_id="brief", question="Give one concise conclusion"), local_planning=False)
    assert result.team.agents[0].max_calls == result.team.total_max_calls == 1
    assert result.team.agents[0].max_tokens == 256


def star_team_response():
    agents = [agent("writer", max_tokens=12000),
              agent("reviewer", depends_on=["writer"], max_tokens=4000),
              agent("editor", depends_on=["writer"], max_tokens=3000)]
    value = team(agents, {"r1": ["writer", "reviewer", "editor"]})
    value.update(synthesizer_id="writer", reviewers={"r1": ["reviewer"]}, total_max_calls=6)
    return {"graph": {"rubrics": [rubric()]}, "team": value}


def test_star_dag_and_unsupported_output_budget_receive_one_precise_correction():
    bad = star_team_response()
    original = copy.deepcopy(bad)
    corrected = copy.deepcopy(bad)
    corrected["team"]["synthesizer_id"] = "editor"
    corrected["team"]["agents"][0]["max_tokens"] = 8192
    corrected["team"]["agents"][2].update(depends_on=["writer", "reviewer"], max_tokens=8192)
    corrected["team"]["agents"][2]["responsibilities"] = ["Produce the complete final article"]

    def respond(payload):
        assert payload["limits"]["execution_max_tokens"] == 8192
        if "response_correction" not in payload:
            return bad
        error = payload["response_correction"]["validation_errors"][0]["message"]
        for detail in ("writer.max_tokens=12000", "limits.execution_max_tokens=8192",
                       "synthesizer_id='writer'", "ancestors=[]",
                       "missing_contributor_ids=['editor', 'reviewer']",
                       "('writer', 'editor')", "('writer', 'reviewer')",
                       "terminal_candidates=['editor', 'reviewer']", "Each agent executes once",
                       "Changing only the synthesizer_id may be insufficient", "never add backward edges"):
            assert detail in error
        assert json.loads(payload["response_correction"]["previous_response"]) == original
        return corrected

    analyzer = GlobalAnalyzer(Scripted(respond), max_agents=3, total_max_calls=6,
                              execution_max_tokens=8192)
    prediction = Prediction(graph=bad["graph"], candidates=[agent("writer")])
    result = analyzer.reconcile(PublicTask(task_id="t", question="Explain the topic"), prediction, [])
    assert result.team.synthesizer_id == "editor"
    assert result.team.agents[2].depends_on == ["writer", "reviewer"]
    assert result.team.agents[2].max_tokens == result.team.agents[0].max_tokens == 8192
    assert bad == original
    assert len(analyzer.call_records) == 2


@pytest.mark.parametrize("cycle", [False, True])
def test_star_dag_cannot_be_repaired_only_by_changing_synthesizer_or_adding_back_edges(cycle):
    response = star_team_response()
    if cycle:
        response["team"]["agents"][0]["depends_on"] = ["reviewer", "editor"]
    else:
        response["team"]["synthesizer_id"] = "editor"
    analyzer = GlobalAnalyzer(Scripted(lambda data: response), max_agents=3, total_max_calls=6,
                              max_corrections=0)
    prediction = Prediction(graph=response["graph"], candidates=[agent("writer")])
    with pytest.raises(ValueError, match="contain a cycle" if cycle else "missing_contributor_ids=\\['reviewer'\\]"):
        analyzer.reconcile(PublicTask(task_id="t", question="Explain"), prediction, [])


@pytest.mark.parametrize("execution_max_tokens", [None, 8192])
def test_execution_output_limit_reaches_all_planning_phases_and_bounds_prediction(execution_max_tokens):
    phases = []

    def respond(data):
        phases.append(data["phase"])
        if execution_max_tokens is None:
            assert "execution_max_tokens" not in data["limits"]
        else:
            assert data["limits"]["execution_max_tokens"] == execution_max_tokens
        tokens = 8192 if "response_correction" in data or data["phase"] == "reconcile" else 12000
        current = agent(max_tokens=tokens)
        if data["phase"] == "predict":
            if "response_correction" in data:
                message = data["response_correction"]["validation_errors"][0]["message"]
                assert "a1.max_tokens=12000" in message and "execution_max_tokens=8192" in message
            return {"graph": {"rubrics": [rubric()]}, "candidates": [current]}
        if data["phase"] == "local_plan":
            return {"agent_id": "a1", "capability": "evidence comparison", "rubric_ids": ["r1"]}
        return {"graph": {"rubrics": [rubric()]}, "team": team([current])}

    model = Scripted(respond)
    analyzer = GlobalAnalyzer(model, lambda aid: model, execution_max_tokens=execution_max_tokens)
    result = analyzer.build(PublicTask(task_id="t", question="Explain"))
    assert result.team.agents[0].max_tokens == 8192
    assert phases == (["predict", "predict", "local_plan", "reconcile"] if execution_max_tokens
                      else ["predict", "local_plan", "reconcile"])
    assert analyzer.last_prediction.candidates[0].max_tokens == (8192 if execution_max_tokens else 12000)


@pytest.mark.parametrize("invalid", [0, -1, True, 8192.5])
def test_execution_output_limit_requires_positive_integer(invalid):
    with pytest.raises(ValueError, match="execution_max_tokens"):
        GlobalAnalyzer(lambda messages: None, execution_max_tokens=invalid)


def test_pipeline_propagates_configured_execution_output_limit(tmp_path, monkeypatch):
    import jit_mas.bridge  # Initialize the native JIT interface before model fixtures.
    from jit_mas.config import MASConfig, ModelConfig
    from jit_mas.experience import ExperienceStore
    from jit_mas.offline import FixtureModels
    from scripts.models.openai_server import OpenAIServerModel
    from scripts.run_jit_mas import make_pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning regression must not issue real model requests")

    monkeypatch.setattr(OpenAIServerModel, "__call__", forbidden)
    config = MASConfig(backend="scripted", models={"exec": ModelConfig(max_tokens=8192)})
    provider = FixtureModels()
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(config, store, tmp_path / "runs", fixture_models=provider)
        pipeline.run_task(pipeline.manifest.evolution[0], store.snapshot(), mode="evolve", resume=False)
    finally:
        store.close()
    payloads = [json.loads(call["messages"][-1]["content"]) for call in provider.calls
                if call["role"] in {"global", "local"}]
    planning = [item for item in payloads if item["phase"] in {"predict", "local_plan", "reconcile"}]
    assert {item["phase"] for item in planning} == {"predict", "local_plan", "reconcile"}
    assert all(item["limits"]["execution_max_tokens"] == 8192 for item in planning)


@pytest.mark.parametrize("bad", ['{"graph":', '{"graph":{"rubrics":[]}}', None])
def test_structured_correction_is_observable_fresh_and_metered(bad):
    valid = {"graph": {"rubrics": [rubric()]}, "candidates": [agent()]}
    calls = []

    class Model:
        def __call__(self, messages, **kwargs):
            calls.append(copy.deepcopy(messages))
            assert len(messages) == 2
            return bad if len(calls) == 1 else json.dumps(valid)

        def get_token_counts(self):
            return {"input_token_count": 3, "output_token_count": 2}

    ledger = BudgetLedger(max_calls=2, max_tokens=100_000)
    analyzer = GlobalAnalyzer(MeteredModel(Model(), ledger, "planning", "global", 4096))
    result = analyzer.predict(PublicTask(task_id="t", question="Compare evidence"))
    assert result.graph.rubrics[0].rubric_id == "r1"
    first, corrected = [json.loads(messages[1]["content"]) for messages in calls]
    correction = corrected.pop("response_correction")
    assert corrected == first
    assert correction["previous_response"] == bad
    assert correction["validation_errors"]
    assert all("input" not in error for error in correction["validation_errors"])
    assert [row["attempt"] for row in analyzer.call_records] == [0, 1]
    assert analyzer.call_records[0]["response"] == bad
    assert "validation_errors" not in analyzer.call_records[1]
    assert ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 10


@pytest.mark.parametrize("semantic", [False, True])
def test_reconcile_corrects_schema_and_semantic_contracts_without_changing_local_testimony(semantic):
    valid = reviewed_team_response()
    bad = copy.deepcopy(valid)
    if semantic:
        bad["team"]["max_parallel"] = 3
    else:
        bad["team"]["reviewers"] = {"r1": ["author"]}
    calls = []

    def model(messages):
        calls.append(copy.deepcopy(messages))
        return json.dumps(bad if len(calls) == 1 else valid)

    analyzer = GlobalAnalyzer(model, max_agents=3, max_parallel=2, total_max_calls=6)
    prediction = Prediction(graph=valid["graph"], candidates=[agent("author")])
    plans = [LocalPlan(agent_id="author", capability="evidence comparison", challenge="Original challenge")]
    result = analyzer.reconcile(PublicTask(task_id="t", question="Compare evidence"), prediction, plans)
    assert result.team.reviewers == {"r1": ["reviewer"]}
    assert result.local_plans == plans and result.local_plans[0] is not plans[0]
    correction = json.loads(calls[1][1]["content"])["response_correction"]
    expected = "configured resource limits" if semantic else "Independent reviewer"
    assert expected in correction["validation_errors"][0]["message"]


def test_local_corrections_never_mix_independent_agent_histories():
    records = {}
    candidates = [agent("a1"), agent("a2")]
    prediction = Prediction(graph=RubricGraph(rubrics=[rubric()]), candidates=candidates)

    def factory(aid):
        records[aid] = []

        def model(messages):
            records[aid].append(copy.deepcopy(messages))
            return json.dumps({"agent_id": aid, "capability": "wrong" if len(records[aid]) == 1
                               else "evidence comparison", "challenge": "PRIVATE_" + aid})

        return model

    analyzer = GlobalAnalyzer(lambda messages: None, factory)
    for candidate in prediction.candidates:
        result = analyzer.local_plan(PublicTask(task_id="t", question="Compare"), prediction, candidate)
        assert result.challenge == "PRIVATE_" + candidate.agent_id
    for aid, messages in records.items():
        assert len(messages) == 2
        assert all(len(call) == 2 for call in messages)
        assert "PRIVATE_" + ("a2" if aid == "a1" else "a1") not in json.dumps(messages)
        corrected = json.loads(messages[1][1]["content"])
        assert corrected["agent_id"] == aid
        assert "PRIVATE_" + aid in corrected["response_correction"]["previous_response"]


@pytest.mark.parametrize("error", [ValueError("transport"), PermissionError("authentication"),
                                   TimeoutError("timeout"), BudgetExceeded("budget")])
def test_structured_corrections_do_not_retry_model_exceptions(error):
    calls = []

    def model(messages):
        calls.append(messages)
        raise error

    analyzer = GlobalAnalyzer(model)
    with pytest.raises(type(error), match=str(error)):
        analyzer.predict(PublicTask(task_id="t", question="Compare"))
    assert len(calls) == 1


@pytest.mark.parametrize("max_corrections", [0, 1])
def test_structured_correction_budget_is_bounded(max_corrections):
    calls = []

    def model(messages):
        calls.append(messages)
        return '{"graph":'

    analyzer = GlobalAnalyzer(model, max_corrections=max_corrections)
    with pytest.raises(json.JSONDecodeError):
        analyzer.predict(PublicTask(task_id="t", question="Compare"))
    assert len(calls) == len(analyzer.call_records) == 1 + max_corrections
    assert all(row["validation_errors"][0]["type"] == "json_decode" for row in analyzer.call_records)
    with pytest.raises(ValueError, match="At most one"):
        JsonModelCalls(max_corrections=2)


def test_distinct_scripted_tasks_change_structure_not_only_role_names():
    simple = {"graph": {"rubrics": [rubric("r1", "Maintain narrative continuity")]},
              "candidates": [agent(max_tokens=1024)]}
    complex_agents = [agent("a1"), agent("a2", depends_on=["a1"], max_tokens=8192)]
    complex_prediction = {"graph": {"rubrics": [rubric()]}, "candidates": complex_agents}
    teams = []
    for question, prediction, final_team in [
            ("Write a short story", simple, team(simple["candidates"])),
            ("Compare competing experimental results", complex_prediction,
             team(complex_agents, {"r1": ["a1", "a2"]}))]:
        model = Scripted(lambda data, p=prediction, t=final_team:
                         p if data["phase"] == "predict" else {"graph": p["graph"], "team": t})
        teams.append(GlobalAnalyzer(model).build(PublicTask(task_id=question, question=question),
                                                local_planning=False).team)
    assert len(teams[0].agents) != len(teams[1].agents)
    assert teams[0].agents[-1].max_tokens != teams[1].agents[-1].max_tokens
    assert teams[0].agents[-1].depends_on != teams[1].agents[-1].depends_on


def test_execution_experience_is_filtered_by_capability_and_contexts_are_fresh():
    experiences = [
        {"experience_id": "R", "bank": "rubric", "instruction": "Check implicit conditions"},
        {"experience_id": "O", "bank": "organization", "instruction": "Assign a source reviewer"},
        {"experience_id": "E", "bank": "execution", "capability": "evidence comparison",
         "instruction": "Cross-check contradicting claims"},
        {"experience_id": "X", "bank": "execution", "capability": "verse composition",
         "instruction": "Check rhyme"}]
    initial = Prediction(graph=RubricGraph(rubrics=[rubric()]), candidates=[agent()])
    model = Scripted(lambda data: {"agent_id": data["candidate"]["agent_id"],
                                  "capability": data["candidate"]["capability"]})
    analyzer = GlobalAnalyzer(model)
    analyzer.local_plan(PublicTask(task_id="t", question="Compare"), initial,
                        initial.candidates[0], experiences)
    assert [item["experience_id"] for item in model.inputs[0]["experiences"]] == ["R", "O", "E"]
    other = initial.candidates[0].model_copy(update={"agent_id": "other"})
    analyzer.local_plan(PublicTask(task_id="t", question="Compare"), initial, other, [])
    assert model.inputs[1]["experiences"] == []
    assert model.inputs[1]["agent_id"] == "other"


def test_predict_rejects_unavailable_tools_and_no_rubrics_mode_is_enforced():
    model = Scripted(lambda data: {"graph": {"rubrics": [rubric()]},
                                  "candidates": [agent(tools=["filesystem"])]})
    with pytest.raises(ValueError, match="unavailable tools"):
        GlobalAnalyzer(model).predict(PublicTask(task_id="t", question="Analyze"))
    with pytest.raises(ValueError, match="disabled"):
        GlobalAnalyzer(model, explicit_rubrics=False).predict(PublicTask(task_id="t", question="Analyze"))
    empty = Scripted(lambda data: {"graph": {"rubrics": []}, "candidates": [agent(rubrics=[])]}
                     if data["phase"] == "predict" else
                     {"graph": {"rubrics": []}, "team": team([agent(rubrics=[])], {})})
    result = GlobalAnalyzer(empty, explicit_rubrics=False).build(
        PublicTask(task_id="t", question="Analyze"), local_planning=False)
    assert not result.graph.rubrics and result.team.agents[0].responsibilities


def feedback():
    return EvaluationFeedback(task_id="t", evaluator_version="scripted-test", score=0.4,
                              complete=True, rubrics=[
        {"rubric_id": "e1", "criterion": "Cite the original results", "weight": 1, "score": 0},
        {"rubric_id": "e2", "criterion": "Compare uncertainties", "weight": 1, "score": 1},
        {"rubric_id": "e3", "criterion": "Mention implementation costs", "weight": 1, "score": 0}])


def test_alignment_correction_uses_original_graph_and_evaluation_only():
    graph = RubricGraph(rubrics=[rubric()])
    expected = feedback()

    def respond(data):
        assert data["prediction"] == graph.model_dump(mode="json")
        assert data["feedback"] == feedback_view(expected)
        return {"matches": [{"predicted_ids": ["r1"],
            "evaluated_ids": ["e1" if "response_correction" in data else "invented"],
            "relation": "equivalent", "confidence": 0.7, "rationale": "Evidence requirement"}]}

    analyzer = RubricAttributor(Scripted(respond))
    aligned = analyzer.align(graph, expected)
    assert aligned.matches[0].evaluated_ids == ["e1"]
    assert aligned.missed_evaluated_ids == ["e2", "e3"]
    assert len(analyzer.call_records) == 2
    assert "unknown rubric references" in analyzer.call_records[0]["validation_errors"][0]["message"]


def detailed_feedback():
    result = feedback()
    result.complete = False
    result.raw = {"judge_inputs": "AUDIT_ONLY_CANARY" * 2000,
                  "raw_responses": ["DUPLICATE_JUDGE_RESPONSE"] * 100}
    for index, rubric_feedback in enumerate(result.rubrics):
        rubric_feedback.rubric_id = f"official-task-id:r{index}:long-exact-suffix"
        rubric_feedback.criterion += " C" * 6000
        rubric_feedback.reason = "Reason must remain complete: " + "R" * 14000
        rubric_feedback.evidence = ["Exact quote " + "E" * 12000, "Paraphrased ... quote"]
        rubric_feedback.raw = {"raw_response": "AUDIT_ONLY_CANARY", "judge_input": "AUDIT_ONLY_CANARY",
            "axis": "Synthesis", "confidence": 0.73, "missing_elements": ["A missing condition"],
            "evidence_locations": [
                {"quote": rubric_feedback.evidence[0], "verified": True, "start": 5,
                 "end": 5 + len(rubric_feedback.evidence[0])},
                {"quote": rubric_feedback.evidence[1], "verified": False, "start": None, "end": None}],
            "error": None}
    result.rubrics[1].weight = -2
    result.rubrics[2].score = None
    result.rubrics[2].status = "error"
    result.rubrics[2].verdict = "Error"
    result.rubrics[2].raw["error"] = "Judge did not return valid JSON"
    return result


def test_feedback_projection_is_lossless_for_semantics_and_immutable_for_saved_audit():
    official = detailed_feedback()
    before = official.model_dump(mode="json")
    projected = feedback_view(official)
    assert {key: value for key, value in projected.items() if key != "rubrics"} == {
        key: value for key, value in before.items() if key not in {"rubrics", "raw"}}
    assert len(projected["rubrics"]) == len(official.rubrics)
    for original, row in zip(official.rubrics, projected["rubrics"]):
        for field, value in original.model_dump(mode="json", exclude={"raw"}).items():
            assert row[field] == value
        for field in ("axis", "confidence", "evidence_locations", "missing_elements", "error"):
            assert row[field] == original.raw[field]
        assert "raw" not in row and "raw_response" not in row and "judge_input" not in row
    assert "AUDIT_ONLY_CANARY" not in json.dumps(projected)
    assert projected["rubrics"][1]["weight"] == -2
    assert projected["rubrics"][2]["status"] == "error"
    assert projected["rubrics"][2]["score"] is None
    projected["rubrics"][0]["evidence_locations"][1]["verified"] = True
    projected["rubrics"][0]["evidence"].append("changed view")
    assert official.model_dump(mode="json") == before


@pytest.mark.parametrize("weight,score,status,contribution,desired,assessment", [
    (-2, 0, "ok", 0, 0, "penalty_avoided"),
    (-2, 1, "ok", -2, 0, "penalty_applied"),
    (3, 1, "ok", 3, 1, "positive_criterion_met"),
    (3, 0, "ok", 0, 1, "positive_criterion_missed"),
    (3, 0.5, "ok", 1.5, 1, "positive_criterion_partial"),
    (0, 1, "ok", 0, None, "zero_weight"),
    (-2, 0, "error", 0, 0, "evaluation_unavailable"),
    (3, None, "error", None, 1, "evaluation_unavailable"),
])
def test_feedback_projection_interprets_signed_scores_without_altering_official_fields(
        weight, score, status, contribution, desired, assessment):
    official = feedback()
    official.rubrics[0].weight = weight
    official.rubrics[0].score = score
    official.rubrics[0].status = status
    original = official.model_dump(mode="json")
    row = feedback_view(official)["rubrics"][0]
    assert row["signed_contribution"] == contribution
    assert row["desired_score"] == desired
    assert row["assessment"] == assessment
    for key, value in original["rubrics"][0].items():
        if key != "raw":
            assert row[key] == value
    assert official.model_dump(mode="json") == original


def test_all_attribution_phases_use_compact_feedback_but_keep_full_local_execution():
    official = detailed_feedback()
    before = official.model_dump(mode="json")
    expected = feedback_view(official)
    graph = RubricGraph(rubrics=[rubric()])
    spec = TeamSpec.model_validate(team())
    local_run = {"metadata": {"agent_id": "a1", "raw": {"observed": "KEEP_LOCAL_RAW"}},
                 "trajectory": [{"model_input_messages": [{"content": "LOCAL_INPUT" + "x" * 18000}],
                                 "model_output_messages": {"content": "LOCAL_OUTPUT" + "y" * 18000}}]}
    result = {"answer": "Submitted", "metadata": {"events": [
        {"event_id": "observed", "agent_id": "a1", "kind": "artifact_published", "content": "Evidence"}]},
        "sub_runs": [local_run]}
    phases = Counter()

    def respond(data):
        phase = data["phase"]
        phases[phase] += 1
        assert "AUDIT_ONLY_CANARY" not in json.dumps(data)
        if phase.startswith("attribute_local"):
            assert data["feedback"] == expected["rubrics"]
            assert data["local_execution"] == local_run
            assert data["local_execution"]["metadata"]["raw"]["observed"] == "KEEP_LOCAL_RAW"
        else:
            assert data["feedback"] == expected
            assert "LOCAL_INPUT" not in json.dumps(data)
        if phase == "align":
            assert data["valid_predicted_ids"] == ["r1"]
            assert data["valid_evaluated_ids"] == [row.rubric_id for row in official.rubrics]
            return {"matches": [{"predicted_ids": ["r1"], "evaluated_ids": data["valid_evaluated_ids"],
                "relation": "split", "confidence": 0.7, "rationale": "Related evidence requirements"}]}
        if phase == "attribute_global":
            return {"questions": {"a1": ["Explain the evidence"]}}
        if phase == "attribute_local":
            return {"evidence_requests": ["observed"]}
        return {"findings": []}

    analyzer = RubricAttributor(Scripted(respond), max_parallel=1)
    assert analyzer.attribute(PublicTask(task_id="t", question="Compare"), graph, graph, spec,
                              result, official) == []
    assert phases == {"align": 2, "attribute_global": 1, "attribute_local": 1,
                      "attribute_local_followup": 1, "attribute_integrate": 1}
    assert official.model_dump(mode="json") == before
    for call in analyzer.call_records:
        system = call["messages"][0]["content"]
        if call["phase"] == "align":
            assert "preserving every character including long prefixes and suffixes" in system
            assert "nonempty predicted_ids AND evaluated_ids" in system
        else:
            assert "do not treat unverified quotes as exact observations" in system
        assert "Not Satisfied (score=0) avoids that penalty" in system

    observed_scoring = []

    def propose(data):
        observed_scoring.append(data["scoring_context"])
        return {"proposals": []}

    analyzer.global_model = Scripted(propose)
    supported = [AttributionFinding(finding_id="f", rubric_ids=["r1"], categories=["organization"],
                                   hypothesis="Check evidence", supporting_evidence=["observed"])]
    analyzer.propose(PublicTask(task_id="t", question="Compare"), supported, 0)
    analyzer.propose(PublicTask(task_id="other", question="Different task"), supported, 0)
    assert observed_scoring[0]["task_id"] == "t"
    assert len(observed_scoring[0]["criteria"]) == len(official.rubrics)
    assert observed_scoring[0]["criteria"][1]["weight"] == -2
    assert observed_scoring[0]["criteria"][1]["signed_contribution"] == -2
    assert observed_scoring[1] is None


def test_attribution_corrections_preserve_evidence_exchange_limits():
    graph = RubricGraph(rubrics=[rubric()])
    spec = TeamSpec.model_validate(team())
    alignment = RubricAlignment(matches=[{"predicted_ids": ["r1"], "evaluated_ids": ["e1"],
        "relation": "equivalent", "confidence": 0.7, "rationale": "Evidence requirement"}])
    result = {"answer": "Submitted", "metadata": {"events": [
        {"event_id": "observed", "agent_id": "a1", "kind": "artifact_published", "content": "Evidence"}]},
        "sub_runs": [{"metadata": {"agent_id": "a1"}, "trajectory": [
            {"model_input_messages": [{"content": "PRIVATE_a1"}],
             "model_output_messages": {"content": "Observed reply"}}]}]}
    counts = Counter()

    def respond(data):
        phase = data["phase"]
        counts[phase] += 1
        correcting = "response_correction" in data
        if phase == "attribute_global":
            return {"questions": {"a1" if correcting else "unknown": ["Was evidence checked?"]}}
        if phase == "attribute_local":
            return {"evidence_requests": ["observed" if correcting else "invented"]}
        if phase == "attribute_local_followup":
            assert data["remaining_exchanges"] == 0
            assert [item["event_id"] for item in data["requested_evidence"]] == ["observed"]
            return {"evidence_requests": [] if correcting else ["observed"]}
        return {"findings": [{"finding_id": "f", "rubric_ids": ["r1"], "agent_ids": ["a1"],
            "categories": ["execution"], "hypothesis": "Evidence supports the answer", "success": True,
            "supporting_evidence": ["observed" if correcting else "invented"]}]}

    analyzer = RubricAttributor(Scripted(respond), max_parallel=1)
    findings = analyzer.attribute(PublicTask(task_id="t", question="Compare"), graph, graph, spec,
        result, feedback(), global_alignment=alignment, planned_alignment=alignment)
    assert findings[0].supporting_evidence == ["observed"]
    assert counts == {"attribute_global": 2, "attribute_local": 2,
                      "attribute_local_followup": 2, "attribute_integrate": 2}
    assert all("PRIVATE_a1" not in json.dumps(row["messages"]) for row in analyzer.call_records
               if row["phase"] in {"attribute_global", "attribute_integrate"})
    assert sum("validation_errors" in row for row in analyzer.call_records) == 4


def test_semantic_alignment_paraphrase_partial_many_to_many_and_omissions():
    graph = RubricGraph(rubrics=[rubric(), rubric("r2", "Discuss uncertainty"),
                               rubric("r3", "Readable prose"), rubric("r4", "Trace source support")])
    before = graph.model_dump()
    model = Scripted(lambda data: {"matches": [
        {"predicted_ids": ["r1", "r2"], "evaluated_ids": ["e1", "e2"],
         "relation": "split", "confidence": 0.7, "rationale": "Combined evidence and uncertainty"},
        {"predicted_ids": ["r4"], "evaluated_ids": ["e1"], "relation": "equivalent",
         "confidence": 0.9, "rationale": "Paraphrased primary-source traceability"},
        {"predicted_ids": ["r2"], "evaluated_ids": ["e2"], "relation": "uncertain",
         "confidence": 0.4, "rationale": "Unclear quantitative extent"}]})
    alignment = RubricAttributor(model).align(graph, feedback())
    assert alignment.missed_evaluated_ids == ["e3"]
    assert alignment.unmatched_predicted_ids == ["r3"]
    assert len(alignment.matches[0].evaluated_ids) == 2
    assert alignment.matches[-1].confidence == 0.4
    assert graph.model_dump() == before


def test_collaborative_attribution_preserves_full_local_context_and_abstains():
    graph = RubricGraph(rubrics=[rubric()])
    spec = TeamSpec.model_validate(team([agent(), agent("a2", depends_on=["a1"])],
                                       {"r1": ["a1", "a2"]}))
    findings = [
        {"finding_id": "omission", "rubric_ids": ["e3"], "categories": ["execution"],
         "agent_ids": ["a1"], "hypothesis": "Cost requirement was omitted",
         "supporting_evidence": ["planning:global", "feedback:e3"]},
        {"finding_id": "handoff", "rubric_ids": ["e1"], "categories": ["organization"],
         "agent_ids": ["a1", "a2"], "hypothesis": "Source was sent without its locator",
         "supporting_evidence": ["sent"], "opposing_evidence": ["read"]},
        {"finding_id": "execution", "rubric_ids": ["e1"], "categories": ["execution"],
         "agent_ids": ["a2"], "hypothesis": "Received evidence was not checked",
         "supporting_evidence": ["read"]},
        {"finding_id": "unknown", "rubric_ids": ["e2"], "categories": ["execution"],
         "agent_ids": ["a1"], "hypothesis": "Uncertain", "supporting_evidence": []},
        {"finding_id": "success", "rubric_ids": ["e2"], "categories": ["execution"],
         "agent_ids": ["a2"], "hypothesis": "Uncertainty was checked", "success": True,
         "supporting_evidence": ["read", "feedback:e2"]}]
    result = {"answer": "Submitted", "metadata": {"events": [
        {"event_id": "sent", "agent_id": "a1", "kind": "artifact_published", "content": "source"},
        {"event_id": "read", "agent_id": "a2", "kind": "artifact_consumed", "content": "source"}]},
        "sub_runs": [{"metadata": {"agent_id": aid}, "trajectory": [
            {"model_input_messages": [{"content": aid + "-PRIVATE-" + "x" * 12000}],
             "model_output_messages": {"content": "Observed reply"}}]} for aid in ["a1", "a2"]]}

    def respond(data):
        if data["phase"] == "attribute_global":
            return {"questions": {"a1": ["Was evidence sent?"], "a2": ["Was it consumed?"]}}
        if data["phase"].startswith("attribute_local"):
            encoded = json.dumps(data)
            aid = data["agent_id"]
            assert aid + "-PRIVATE-" in encoded and "x" * 12000 in encoded
            assert ("a2" if aid == "a1" else "a1") + "-PRIVATE-" not in encoded
            return {"findings": []}
        return {"findings": findings}

    model = Scripted(respond)
    attribution = RubricAttributor(model)
    alignment = RubricAlignment(matches=[{"predicted_ids": ["r1"], "evaluated_ids": ["e1", "e2"],
        "relation": "split", "confidence": 0.8, "rationale": "evidence and uncertainty"}],
        missed_evaluated_ids=["e3"])
    output = attribution.attribute(PublicTask(task_id="t", question="Compare"), graph, graph,
                                   spec, result, feedback(), global_alignment=alignment,
                                   planned_alignment=alignment)
    assert output[0].categories == ["prediction"] and output[0].agent_ids == []
    assert output[1].categories == ["organization"]
    assert output[2].categories == ["execution"]
    assert output[3].categories == ["external_or_uncertain"]
    assert output[4].success
    global_calls = [data for data in model.inputs if data["agent_id"] == "global"]
    assert all("-PRIVATE-" not in json.dumps(data) for data in global_calls)
    assert global_calls[0]["shared_artifacts"][0]["event_id"] == "sent"


def test_success_update_requires_the_observed_evidence_chain_without_promotion_metadata():
    finding = AttributionFinding(finding_id="s", rubric_ids=["r1"], categories=["execution"],
                                 success=True, hypothesis="Cross-check supported the answer",
                                 supporting_evidence=["checked"], opposing_evidence=["cost"])
    proposal = {"proposal_id": "p", "source_task_id": "t", "base_version": 0,
                "experience": {"experience_id": "x", "bank": "execution",
                    "instruction": "Cross-check source disagreements", "applicability": "Conflicting evidence",
                    "capability": "evidence comparison", "source_task_ids": ["t"],
                    "evidence": ["checked"], "counterevidence": ["cost"]},
                "diff": "+ cross-check before synthesis", "rationale": "Observed source consistency",
                "evidence": ["checked"], "expected_benefit": "Fewer unsupported comparisons"}
    model = Scripted(lambda data: {"proposals": [proposal]})
    analyzer = RubricAttributor(model)
    task = PublicTask(task_id="t", question="Compare")
    proposed = analyzer.propose(task, [finding], 0)[0]
    assert "validation_status" not in proposed.experience.model_dump()
    assert "validation_plan" not in proposed.model_dump()
    proposal["evidence"] = ["invented"]
    with pytest.raises(ValueError, match="not supported"):
        analyzer.propose(task, [finding], 0)
    assert analyzer.propose(task, [], 0) == []

    valid = copy.deepcopy(proposal)
    valid["evidence"] = ["checked"]
    invalid = copy.deepcopy(valid)
    invalid["experience"]["quality_decision"] = "approved"
    correcting = RubricAttributor(Scripted(lambda data: {
        "proposals": [valid if "response_correction" in data else invalid]}))
    repaired = correcting.propose(task, [finding], 0)
    assert "quality_decision" not in repaired[0].experience.model_dump()
    assert "validation_result" not in repaired[0].model_dump()
    assert len(correcting.call_records) == 2
    assert "Extra inputs" in correcting.call_records[0]["validation_errors"][0]["message"]

    for invalid_field in ("evidence", "counterevidence"):
        invalid_reference = copy.deepcopy(valid)
        if invalid_field == "evidence":
            invalid_reference["evidence"] = [finding.finding_id]
            invalid_reference["experience"]["evidence"] = [finding.finding_id]
        else:
            invalid_reference["experience"]["counterevidence"] = ["invented-counter"]

        def respond(data):
            assert data["valid_supporting_evidence_ids"] == ["checked"]
            assert data["valid_counterevidence_ids"] == ["checked", "cost"]
            if "response_correction" in data:
                error = data["response_correction"]["validation_errors"][0]["message"]
                assert "checked" in error
                if invalid_field == "evidence":
                    assert "invalid_proposal_evidence_ids=['s']" in error
                    assert "invalid_experience_evidence_ids=['s']" in error
                    assert "finding_id is for rationale" in error
                else:
                    assert "invalid_counterevidence_ids=['invented-counter']" in error
                    assert "valid_counterevidence_ids=['checked', 'cost']" in error
                return {"proposals": [valid]}
            return {"proposals": [invalid_reference]}

        checked = RubricAttributor(Scripted(respond))
        result = checked.propose(task, [finding], 0)
        assert result[0].evidence == ["checked"]
        assert result[0].experience.evidence == ["checked"]
        assert "validation_status" not in result[0].experience.model_dump()
        assert len(checked.call_records) == 2
        instructions = checked.call_records[0]["messages"][0]["content"]
        assert "refer to finding IDs only in rationale" in instructions
        assert "numerical threshold or target" in instructions
        assert "public task features" in instructions
