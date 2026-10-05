"""Formal responsibility routing for post-evaluation bi-level evolution."""

import copy
import json

import pytest

from jit_mas.attribution import RubricAttributor, build_credit_assignments
from jit_mas.schemas import (
    AttributionFinding,
    EvaluationFeedback,
    PublicTask,
    RubricAlignment,
    RubricGraph,
    TeamSpec,
)


def routing_inputs():
    graph = RubricGraph(rubrics=[
        {"rubric_id": rubric_id, "requirement": "Check " + rubric_id}
        for rubric_id in ("evidence", "accuracy", "unowned", "unmatched")
    ], edges=[
        {"source": "evidence", "target": "accuracy", "relation": "support",
         "rationale": "Evidence supports accurate synthesis"},
        {"source": "unowned", "target": "unmatched", "relation": "tradeoff",
         "rationale": "Unassigned requirement relationship"},
    ])
    team = TeamSpec(agents=[
        {"agent_id": "searcher", "role": "Searcher", "capability": "research",
         "rubric_ids": ["evidence"]},
        {"agent_id": "writer", "role": "Writer", "capability": "writing"},
        {"agent_id": "critic", "role": "Critic", "capability": "verification",
         "depends_on": ["searcher"]},
    ], synthesizer_id="writer", coverage={"evidence": ["searcher"], "accuracy": ["writer"]},
        primary={"evidence": "searcher", "accuracy": "writer"},
        reviewers={"evidence": ["critic"]})
    alignment = RubricAlignment(matches=[
        {"predicted_ids": ["evidence"], "evaluated_ids": ["judge-source", "judge-coverage"],
         "relation": "split", "confidence": 0.9, "rationale": "Evidence splits into two checks"},
        {"predicted_ids": ["evidence", "accuracy"], "evaluated_ids": ["judge-joint"],
         "relation": "merge", "confidence": 0.8, "rationale": "Joint evidence and accuracy check"},
        {"predicted_ids": ["unowned"], "evaluated_ids": ["judge-unowned"],
         "relation": "equivalent", "confidence": 0.8, "rationale": "Predicted but unassigned"},
    ], missed_evaluated_ids=["judge-missed"], unmatched_predicted_ids=["unmatched"])
    feedback = EvaluationFeedback(task_id="task", evaluator_version="fixture", complete=True,
        score=0.4, rubrics=[
            {"rubric_id": rubric_id, "criterion": "Assess " + rubric_id, "weight": 1, "score": 0}
            for rubric_id in ("judge-source", "judge-coverage", "judge-joint", "judge-unowned", "judge-missed")
        ])
    return graph, team, alignment, feedback


def test_credit_assignment_routes_splits_merges_and_independent_reviewers():
    graph, team, alignment, feedback = routing_inputs()
    before = copy.deepcopy((graph, team, alignment, feedback))

    assignments = build_credit_assignments(graph, team, alignment, feedback)

    assert assignments["searcher"].predicted_rubric_ids == ["evidence"]
    assert assignments["searcher"].evaluated_rubric_ids == ["judge-coverage", "judge-joint", "judge-source"]
    assert assignments["searcher"].assignment_basis == {"evidence": ["agent_spec", "coverage", "primary"]}
    assert assignments["writer"].predicted_rubric_ids == ["accuracy"]
    assert assignments["writer"].evaluated_rubric_ids == ["judge-joint"]
    assert assignments["writer"].assignment_basis == {"accuracy": ["coverage", "primary"]}
    assert assignments["critic"].assignment_basis == {"evidence": ["reviewer"]}
    assert assignments["critic"].evaluated_rubric_ids == assignments["searcher"].evaluated_rubric_ids
    assert all(not assignment.causal_identification for assignment in assignments.values())
    assert (graph, team, alignment, feedback) == before


def test_graph_edges_provide_context_without_propagating_ownership():
    graph, team, alignment, feedback = routing_inputs()
    assignments = build_credit_assignments(graph, team, alignment, feedback)

    for assignment in assignments.values():
        assert assignment.graph_edges == [graph.edges[0]]
        assert not {"judge-unowned", "judge-missed"}.intersection(assignment.evaluated_rubric_ids)
    assert "accuracy" not in assignments["searcher"].predicted_rubric_ids
    assignments["searcher"].graph_edges[0].rationale = "Local interpretation"
    assert graph.edges[0].rationale == "Evidence supports accurate synthesis"


def test_meta_assignment_routes_additional_reflection_without_inventing_frozen_ownership():
    graph, team, alignment, feedback = routing_inputs()
    assignments = build_credit_assignments(graph, team, alignment, feedback,
        meta_assignments={"writer": ["judge-missed", "judge-joint"]},
        meta_assignment_rationales={"writer": "Inspect whether the final synthesis omitted a requested facet."})

    writer = assignments["writer"]
    assert writer.predicted_rubric_ids == ["accuracy"]
    assert writer.evaluated_rubric_ids == ["judge-joint", "judge-missed"]
    assert writer.meta_assigned_evaluated_rubric_ids == ["judge-missed"]
    assert writer.assignment_rationale == "Inspect whether the final synthesis omitted a requested facet."
    assert "judge-missed" not in writer.assignment_basis
    assert not writer.causal_identification


@pytest.mark.parametrize("assignments,rationales,error", [
    ({"unknown": ["judge-missed"]}, {"unknown": "Review context"}, "unknown agent"),
    ({"writer": ["invented"]}, {"writer": "Review context"}, "unknown evaluated rubric"),
    ({"writer": ["judge-missed"]}, {}, "needs a rationale"),
    ({"writer": ["judge-missed"]}, {"writer": " "}, "cannot be empty"),
])
def test_meta_assignment_validates_exact_references_and_nonempty_rationale(assignments, rationales, error):
    graph, team, alignment, feedback = routing_inputs()
    with pytest.raises(ValueError, match=error):
        build_credit_assignments(graph, team, alignment, feedback,
                                 meta_assignments=assignments, meta_assignment_rationales=rationales)


@pytest.mark.parametrize("reference", ["predicted", "evaluated", "responsibility"])
def test_credit_assignment_rejects_invented_rubric_references(reference):
    graph, team, alignment, feedback = routing_inputs()
    if reference == "responsibility":
        team.agents[0].rubric_ids = ["invented"]
    else:
        field = "predicted_ids" if reference == "predicted" else "evaluated_ids"
        setattr(alignment.matches[0], field, ["invented"])

    with pytest.raises(ValueError, match="unknown"):
        build_credit_assignments(graph, team, alignment, feedback)


def test_attributor_uses_formal_local_feedback_and_keeps_unowned_failures_global():
    graph, team, alignment, feedback = routing_inputs()
    payloads = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        payloads.append(payload)
        if payload["phase"] == "attribute_global":
            return json.dumps({"rubric_assignments": {"writer": ["judge-missed"]},
                               "rubric_assignment_rationale": {
                                   "writer": "Inspect the final synthesis for an omitted requested facet."}})
        if payload["phase"] == "attribute_integrate":
            return json.dumps({"findings": [
                {"finding_id": rubric_id, "rubric_ids": [rubric_id], "agent_ids": ["writer"],
                 "categories": ["execution"], "hypothesis": "A requirement was missed",
                 "supporting_evidence": ["feedback:" + rubric_id]}
                for rubric_id in ("judge-unowned", "judge-missed")
            ]})
        return json.dumps({"findings": []})

    result = {"answer": "Submitted answer", "metadata": {"events": []}, "sub_runs": [
        {"metadata": {"agent_id": agent.agent_id}, "trajectory": []} for agent in team.agents
    ]}
    attributor = RubricAttributor(model, max_parallel=1)
    findings = attributor.attribute(PublicTask(task_id="task", question="Write a comparison"),
        graph, graph, team, result, feedback, global_alignment=alignment, planned_alignment=alignment)

    global_payload = next(payload for payload in payloads if payload["phase"] == "attribute_global")
    assert len(global_payload["feedback"]["rubrics"]) == len(feedback.rubrics)
    assert set(global_payload["credit_assignments"]) == {"searcher", "writer", "critic"}
    assert global_payload["credit_assignments"]["writer"]["meta_assigned_evaluated_rubric_ids"] == []
    assert attributor.last_credit_assignments["writer"].meta_assigned_evaluated_rubric_ids == ["judge-missed"]
    for payload in payloads:
        if payload["phase"] != "attribute_local":
            continue
        assignment = attributor.last_credit_assignments[payload["agent_id"]]
        assert payload["credit_assignment"] == assignment.model_dump(mode="json")
        assert {rubric["rubric_id"] for rubric in payload["feedback"]} == set(assignment.evaluated_rubric_ids)
        assert [rubric["rubric_id"] for rubric in payload["assigned_predicted_rubrics"]] == assignment.predicted_rubric_ids
    assert findings[0].agent_ids == [] and findings[0].categories == ["organization"]
    assert findings[1].agent_ids == [] and findings[1].categories == ["prediction"]


def test_incomplete_feedback_cannot_start_credit_assignment_or_attribution():
    graph, team, alignment, feedback = routing_inputs()
    feedback.complete = False

    with pytest.raises(ValueError, match="complete evaluation feedback"):
        build_credit_assignments(graph, team, alignment, feedback)
    attributor = RubricAttributor(lambda _messages: pytest.fail("No attribution model call is allowed"))
    with pytest.raises(ValueError, match="complete evaluation feedback"):
        attributor.attribute(PublicTask(task_id="task", question="Write a comparison"),
            graph, graph, team, {}, feedback)
    assert attributor.last_credit_assignments == {} and attributor.call_records == []


@pytest.mark.parametrize("meta_assignment", [False, True])
def test_unaligned_judge_id_collision_never_infers_execution_blame(meta_assignment):
    graph = RubricGraph(rubrics=[{"rubric_id": "shared-id", "requirement": "Provide accurate evidence"}])
    team = TeamSpec(agents=[{"agent_id": "writer", "role": "Writer", "capability": "writing",
                            "rubric_ids": ["shared-id"]}], synthesizer_id="writer",
                    coverage={"shared-id": ["writer"]}, primary={"shared-id": "writer"})
    feedback = EvaluationFeedback(task_id="task", evaluator_version="fixture", complete=True, score=0,
        rubrics=[{"rubric_id": "shared-id", "criterion": "Discuss a different unpredicted facet",
                  "weight": 1, "score": 0}])
    alignment = RubricAlignment(missed_evaluated_ids=["shared-id"], unmatched_predicted_ids=["shared-id"])
    payloads = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        payloads.append(payload)
        if payload["phase"] == "attribute_global" and meta_assignment:
            return json.dumps({"rubric_assignments": {"writer": ["shared-id"]},
                               "rubric_assignment_rationale": {
                                   "writer": "Inspect whether final synthesis omitted the judge's facet."}})
        if payload["phase"] == "attribute_integrate":
            return json.dumps({"findings": [{"finding_id": "missed", "rubric_ids": ["shared-id"],
                "agent_ids": ["writer"], "categories": ["execution"],
                "hypothesis": "The unpredicted requirement was missed",
                "supporting_evidence": ["feedback:shared-id"]}]})
        return json.dumps({"findings": []})

    attributor = RubricAttributor(model, max_parallel=1)
    result = {"answer": "Submitted", "metadata": {"events": []}, "sub_runs": [
        {"metadata": {"agent_id": "writer"}, "trajectory": []}]}
    findings = attributor.attribute(PublicTask(task_id="task", question="Write an explanation"),
        graph, graph, team, result, feedback, global_alignment=alignment, planned_alignment=alignment)

    assert findings[0].categories == ["prediction"] and findings[0].agent_ids == []
    assignment = attributor.last_credit_assignments["writer"]
    assert assignment.predicted_rubric_ids == ["shared-id"]
    assert assignment.evaluated_rubric_ids == (["shared-id"] if meta_assignment else [])
    local = next(payload for payload in payloads if payload["phase"] == "attribute_local")
    assert [rubric["rubric_id"] for rubric in local["feedback"]] == assignment.evaluated_rubric_ids


def proposal(bank):
    return {"proposal_id": "proposal", "source_task_id": "task", "base_version": 0,
            "experience": {"experience_id": "experience", "bank": bank,
                           "instruction": "Check conflicting evidence before synthesis",
                           "applicability": "Comparisons", "capability": "verification",
                           "source_task_ids": ["task"], "evidence": ["observed"]},
            "diff": "+ conditional check", "rationale": "Supported process lesson",
            "evidence": ["observed"], "expected_benefit": "Fewer unsupported claims"}


@pytest.mark.parametrize("reflections", [[], [{"agent_id": "critic", "evidence": ["observed"]}]])
def test_meta_proposal_integrates_reflections_and_restricts_experience_banks(reflections):
    payloads = []

    def model(messages):
        payloads.append(json.loads(messages[1]["content"]))
        return json.dumps({"proposals": [proposal("organization")]})

    finding = AttributionFinding(finding_id="finding", rubric_ids=[], categories=["organization"],
                                 hypothesis="Review improves handoff", supporting_evidence=["observed"])
    attributor = RubricAttributor(model)
    results = attributor.propose(PublicTask(task_id="task", question="Compare alternatives"),
                                [finding], 0, agent_reflections=reflections)
    assert results[0].experience.bank == "organization"
    assert payloads[0]["agent_reflections"] == reflections
    assert payloads[0]["allowed_experience_banks"] == ["rubric", "organization"]


def test_full_harness_proposals_reject_execution_bank_while_legacy_keeps_it():
    finding = AttributionFinding(finding_id="finding", rubric_ids=[], categories=["execution"],
                                 hypothesis="An observed check helped", supporting_evidence=["observed"])
    task = PublicTask(task_id="task", question="Compare alternatives")
    attributor = RubricAttributor(lambda _messages: json.dumps({"proposals": [proposal("execution")]}),
                                 max_corrections=0)

    with pytest.raises(ValueError, match="allowed experience banks"):
        attributor.propose(task, [finding], 0, agent_reflections=[])
    assert attributor.propose(task, [finding], 0)[0].experience.bank == "execution"


@pytest.mark.parametrize("proposal_ids", [["", "valid"], [" ", "valid"], ["duplicate", "duplicate"]])
def test_proposal_candidates_require_distinct_nonempty_ids(proposal_ids):
    finding = AttributionFinding(finding_id="finding", rubric_ids=[], categories=["organization"],
                                 hypothesis="Observed role handoff", supporting_evidence=["observed"])
    candidates = [dict(proposal("organization"), proposal_id=proposal_id) for proposal_id in proposal_ids]
    attributor = RubricAttributor(lambda _messages: json.dumps({"proposals": candidates}), max_corrections=0)

    with pytest.raises(ValueError, match="nonempty and unique"):
        attributor.propose(PublicTask(task_id="task", question="Compare alternatives"), [finding], 0,
                           agent_reflections=[])
