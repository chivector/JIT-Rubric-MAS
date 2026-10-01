"""Task-scoped transfer and evidence quality must not reward source-rubric overfitting."""

import copy
import json

import pytest

from jit_mas.attribution import (
    ALIGN_PROMPT, FEEDBACK_IDS_PROMPT, PROPOSE_PROMPT, RubricAttributor, feedback_view,
)
from jit_mas.experience import capability_matches, experience_applicability, retrieve
from jit_mas.schemas import (
    AttributionFinding, ChangeProposal, EvaluationFeedback, Experience, ExperienceSnapshot,
    PublicTask, RubricAlignment, RubricGraph, TeamSpec,
)


def comparison_advice(**changes):
    values = dict(experience_id="conditional-comparison", bank="execution",
        instruction="Compare shared properties and differences, verifying each claim.",
        applicability="Comparative articles", capability="comparison_article_writing",
        task_signals=["comparison", "article writing"], source_task_ids=["earlier"],
        evidence=["submission:earlier"])
    values.update(changes)
    return Experience(**values)


@pytest.mark.parametrize("question", [
    "Write an article explaining a theorem for beginners.",
    "Write a technical article explaining neural networks.",
    "Write an article about algorithms without comparison.",
])
def test_comparison_advice_does_not_leak_through_generic_article_words(question):
    task = PublicTask(task_id="later", question=question)
    entry = comparison_advice()
    diagnostic = experience_applicability(entry, task, capability="technical article writer")
    assert not diagnostic["matched"]
    assert diagnostic["reason"] == "public_task_scope_mismatch"
    assert retrieve(ExperienceSnapshot(experiences=[entry]), task) == []


def test_new_comparison_subject_and_generic_writer_receive_grounded_advice():
    task = PublicTask(task_id="later", question="Write an article comparing caching and indexing.")
    entry = comparison_advice()
    diagnostic = experience_applicability(entry, task, capability="technical writing")
    assert diagnostic["matched"] and diagnostic["task_grounded"]
    assert diagnostic["reason"] == "public_task_and_role_function_match"
    assert diagnostic["signal_matches"][0]["matched_terms"] == ["compare"]
    assert len(retrieve(ExperienceSnapshot(experiences=[entry]), task,
                        capability="technical writing")) == 1
    assert not experience_applicability(entry, task, capability="numerical verification")["matched"]
    assert not capability_matches(entry.capability, "technical writing")


@pytest.mark.parametrize("capability", ["technical verification", "article fact-checking", "quality analysis"])
def test_grounded_task_does_not_promote_style_or_format_words_to_role_functions(capability):
    task = PublicTask(task_id="later", question="Write an article comparing caching and indexing.")
    entry = comparison_advice(capability="technical article quality comparison writing")
    assert not experience_applicability(entry, task, capability=capability)["matched"]
    assert not capability_matches("technical article quality", capability)


def test_role_matching_requires_operation_not_just_shared_domain():
    assert not capability_matches("machine learning comparison", "machine learning explanation")
    assert capability_matches("mathematical verification", "Verify statistical assumptions")
    assert capability_matches("comparison writing", "Compare distributed storage systems")


def test_signals_are_necessary_conditions_and_legacy_scope_is_not_task_evidence():
    task = PublicTask(task_id="later", question="Write an article comparing caching and indexing.")
    scoped = comparison_advice(task_signals=["comparison", "security"])
    assert not experience_applicability(scoped, task)["matched"]
    legacy = comparison_advice(task_signals=[], applicability="")
    diagnostic = experience_applicability(legacy, task)
    assert diagnostic["matched"] and not diagnostic["task_grounded"]
    assert diagnostic["reason"] == "legacy_task_scope_unknown"
    assert not experience_applicability(legacy, task, capability="technical writing")["matched"]
    assert experience_applicability(legacy, task, capability="system comparison")["matched"]


def test_explicit_applicability_boundary_applies_without_task_signals():
    entry = comparison_advice(task_signals=[])
    other = PublicTask(task_id="later", question="Write an article explaining a theorem.")
    assert not experience_applicability(entry, other)["matched"]
    assert retrieve(ExperienceSnapshot(experiences=[entry]), other) == []
    comparative = PublicTask(task_id="later", question="Write an article comparing caching and indexing.")
    diagnostic = experience_applicability(entry, comparative, capability="technical writer")
    assert diagnostic["matched"] and diagnostic["task_grounded"]


def test_organization_advice_uses_task_scope_but_not_execution_capability():
    entry = comparison_advice(bank="organization")
    task = PublicTask(task_id="later", question="Write an article comparing caching and indexing.")
    assert experience_applicability(entry, task, capability="numerical verification")["matched"]
    assert not experience_applicability(entry, PublicTask(task_id="other", question="Write a poem"))["matched"]


def test_applicability_operation_boundary_cannot_be_replaced_by_format_signals():
    entry = comparison_advice(task_signals=["article writing"])
    diagnostic = experience_applicability(entry,
        PublicTask(task_id="later", question="Write a technical article explaining a theorem."))
    assert not diagnostic["matched"]
    assert diagnostic["applicability_operations"] == ["compare"]


def finding():
    return AttributionFinding(finding_id="observed-gap", rubric_ids=["private-count"],
        categories=["prediction"], hypothesis="Shared properties were omitted.",
        supporting_evidence=["submission:earlier", "feedback:private-count"])


def proposal(instruction):
    return ChangeProposal(proposal_id="conditional-improvement", source_task_id="earlier",
        base_version=0, experience=comparison_advice(instruction=instruction,
            evidence=finding().supporting_evidence),
        diff="Add scoped comparison practice", rationale="Observed coverage gap",
        evidence=finding().supporting_evidence, expected_benefit="More balanced coverage")


def scoring_context():
    return {"task_id": "earlier", "criteria": [{"rubric_id": "private-count",
        "criterion": "The response contains at least four explicit sentences or bullets about shared properties."}]}


def validate(instruction, question="Write a comparison article.", *, context=None):
    task = PublicTask(task_id="earlier", question=question)
    item = proposal(instruction)
    RubricAttributor._validate_proposals([item], task, [finding()], 0, [],
        scoring_context=scoring_context() if context is None else context)
    return item


@pytest.mark.parametrize("instruction", [
    "Explicitly enumerate at least four shared characteristics in a dedicated section.",
    "Provide exactly 4 examples of similarities.",
    "List four points of commonality before differences.",
])
def test_private_content_threshold_is_rejected_across_wording_and_number_notation(instruction):
    with pytest.raises(ValueError, match="potential source-specific content-count threshold"):
        validate(instruction)


@pytest.mark.parametrize("instruction,question", [
    ("Compare shared properties and differences; use the number of examples requested by the task.",
     "Write a comparison article."),
    ("Provide four examples, checking each against the public request.",
     "Write a comparison article with four examples."),
    ("Check a formula such as n >= 4 / epsilon^2 against its assumptions and derivation.",
     "Write a comparison article."),
    ("For a 4-dimensional vector, verify units and shape before calculating its norm.",
     "Write a comparison article."),
    ("Include four verification steps to check the claimed result.",
     "Write a comparison article."),
    ("Provide four shared characteristics as requested by the task.",
     "Write a comparison article with four examples of similarities."),
])
def test_public_content_counts_and_mathematical_parameters_are_not_private_leakage(instruction, question):
    assert validate(instruction, question).experience.instruction == instruction


def test_foreign_scoring_context_cannot_constrain_proposals():
    context = scoring_context()
    context["task_id"] = "unrelated"
    assert validate("Provide four examples.", context=context)


def test_public_number_in_different_content_category_does_not_exempt_private_threshold():
    context = {"task_id": "earlier", "criteria": [{"criterion": "Include at least two differences."}]}
    with pytest.raises(ValueError, match="potential source-specific content-count threshold"):
        validate("List two differences.", "Write a comparison article with two examples.", context=context)


def test_threshold_error_gets_one_audited_model_correction_without_output_rewriting():
    bad = proposal("Enumerate at least four shared characteristics.").model_dump(mode="json")
    good = proposal("Compare shared properties and differences; check each against evidence.").model_dump(mode="json")

    def model(messages):
        payload = json.loads(messages[1]["content"])
        if "response_correction" in payload:
            correction = payload["response_correction"]
            assert "potential source-specific content-count threshold" in correction["validation_errors"][0]["message"]
            assert json.loads(correction["previous_response"])["proposals"] == [bad]
            return json.dumps({"proposals": [good]})
        return json.dumps({"proposals": [bad]})

    attributor = RubricAttributor(model)
    attributor._scoring_context = scoring_context()
    result = attributor.propose(PublicTask(task_id="earlier", question="Write a comparison article."),
                                [finding()], 0)
    assert result[0].experience.instruction == good["experience"]["instruction"]
    assert len(attributor.call_records) == 2
    assert json.loads(attributor.call_records[0]["response"])["proposals"] == [bad]


def test_risk_only_judge_audit_survives_compaction_without_score_or_evidence_mutation():
    audit = {"mode": "risk_only", "semantic_consistency": "unassessed",
             "risk_flags": ["negative_weight_polarity_requires_review"], "score_modified": False}
    feedback = EvaluationFeedback(task_id="t", evaluator_version="official", score=-0.3,
        complete=True, raw={"quality_audit": copy.deepcopy(audit), "large_audit": "not needed"},
        rubrics=[{"rubric_id": "r", "criterion": "Contains irrelevant content", "weight": -3,
            "score": 1, "verdict": "Satisfied", "reason": "The answer stays on topic.",
            "raw": {"quality_audit": copy.deepcopy(audit)}}])
    original = feedback.model_dump(mode="json")
    view = feedback_view(feedback)
    assert view["score"] == -0.3 and view["complete"]
    assert view["rubrics"][0]["score"] == 1
    assert view["rubrics"][0]["signed_contribution"] == -3
    assert view["rubrics"][0]["quality_audit"] == audit
    assert view["quality_audit"] == audit
    view["rubrics"][0]["quality_audit"]["risk_flags"].clear()
    assert feedback.model_dump(mode="json") == original


def test_proposal_context_preserves_judge_reason_and_risk_without_inventing_conflict():
    audit = {"mode": "risk_only", "semantic_consistency": "unassessed",
             "risk_flags": ["negative_weight_polarity_requires_review"]}
    feedback = EvaluationFeedback(task_id="earlier", evaluator_version="official", score=-0.3,
        complete=True, raw={"quality_audit": audit}, rubrics=[{
            "rubric_id": "private-count", "criterion": "Contains irrelevant content",
            "weight": -3, "score": 1, "verdict": "Satisfied", "reason": "The answer stays on topic.",
            "raw": {"quality_audit": audit}}])
    requests = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        requests.append(payload)
        return json.dumps({"proposals": []} if payload["phase"] == "propose" else {"findings": []})

    task = PublicTask(task_id="earlier", question="Write a comparison article.")
    attributor = RubricAttributor(model, local_attribution=False)
    graph = RubricGraph(rubrics=[])
    team = TeamSpec(agents=[{"agent_id": "author", "role": "Writer", "capability": "writing"}],
                    synthesizer_id="author")
    attributor.attribute(task, graph, graph, team, {"answer": "Submitted answer"}, feedback,
        global_alignment=RubricAlignment(), planned_alignment=RubricAlignment())
    attributor.propose(task, [finding()], 0)
    scoring = requests[-1]["scoring_context"]
    assert scoring["quality_audit"] == audit
    assert scoring["criteria"][0]["quality_audit"] == audit
    assert scoring["criteria"][0]["reason"] == "The answer stays on topic."
    assert scoring["criteria"][0]["signed_contribution"] == -3


@pytest.mark.parametrize("prompt", [ALIGN_PROMPT, FEEDBACK_IDS_PROMPT, PROPOSE_PROMPT])
def test_attribution_instructions_distinguish_evidence_and_unassessed_judge_risk(prompt):
    assert "semantic_consistency=unassessed is not proof" in prompt
    assert "checkpoint or agent statement" in prompt
    assert "observable derivation" in prompt
    assert "do not silently flip scores" in prompt
