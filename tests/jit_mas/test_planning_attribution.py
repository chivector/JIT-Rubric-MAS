"""Scripted software checks; these do not measure real-model task quality."""

import copy
import json

import pytest

from jit_mas.attribution import RubricAttributor
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import (
    AttributionFinding, EvaluationFeedback, LocalPlan, PlannedTeam, Prediction,
    PublicTask, RubricAlignment, RubricGraph, TeamSpec,
)


def rubric(rid="r1", requirement="Compare alternatives with reproducible evidence"):
    return {"rubric_id": rid, "requirement": requirement}


def agent(aid="a1", rubrics=None, **kwargs):
    return {"agent_id": aid, "role": "Analyst", "capability": "evidence comparison",
            "rubric_ids": ["r1"] if rubrics is None else rubrics,
            "responsibilities": ["Check the source evidence"], "max_calls": 2, **kwargs}


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


def test_success_proposal_is_staged_and_requires_the_observed_evidence_chain():
    finding = AttributionFinding(finding_id="s", rubric_ids=["r1"], categories=["execution"],
                                 success=True, hypothesis="Cross-check supported the answer",
                                 supporting_evidence=["checked"], opposing_evidence=["cost"])
    proposal = {"proposal_id": "p", "source_task_id": "t", "base_version": 0,
                "experience": {"experience_id": "x", "bank": "execution",
                    "instruction": "Cross-check source disagreements", "applicability": "Conflicting evidence",
                    "capability": "evidence comparison", "source_task_ids": ["t"],
                    "evidence": ["checked"], "counterevidence": ["cost"]},
                "diff": "+ cross-check before synthesis", "rationale": "Observed source consistency",
                "evidence": ["checked"], "expected_benefit": "Fewer unsupported comparisons",
                "validation_plan": "Rebuild old and candidate teams on independent validation tasks"}
    model = Scripted(lambda data: {"proposals": [proposal]})
    analyzer = RubricAttributor(model)
    task = PublicTask(task_id="t", question="Compare")
    assert analyzer.propose(task, [finding], 0)[0].experience.validation_status == "staged"
    proposal["evidence"] = ["invented"]
    with pytest.raises(ValueError, match="not supported"):
        analyzer.propose(task, [finding], 0)
    assert analyzer.propose(task, [], 0) == []
