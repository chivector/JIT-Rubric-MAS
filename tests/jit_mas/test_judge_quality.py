"""Opt-in offline risk screening never repairs or replaces official judgments."""

import copy
import hashlib

import pytest

from benchmark.adapter.researchrubrics import (
    QUALITY_AUDIT_VERSION, ResearchRubricsAdapter, split_item,
)
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.pipeline import convert_feedback
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import ChangeProposal, Experience


def private_record():
    return split_item({"sample_id": "quality-task", "prompt": "Explain the supplied evidence.",
        "rubrics": [{"criterion": "Includes the requested explanation", "weight": 5},
                    {"criterion": "Contains irrelevant material", "weight": -3}]})[1]


def judgment(score=1, quotes=None, reasoning="The document has no irrelevant material."):
    return {"verdict": "Satisfied" if score else "Not Satisfied", "score": score,
            "confidence": 0.8, "reasoning": reasoning, "evidence_quotes": quotes or [],
            "missing_elements": []}


class Judge:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result

    def get_token_counts(self):
        return {"input_token_count": 20, "output_token_count": 10}


def evaluate(mode="off", replies=None, answer="real evidence here"):
    judge = Judge(replies or [judgment(quotes=["evidence"]), judgment(quotes=["not a direct quote"])])
    ledger = BudgetLedger()
    model = MeteredModel(judge, ledger, "evaluation", "judge", 128)
    adapter = ResearchRubricsAdapter(judge=model, judge_id="offline-quality-test", max_attempts=1,
                                    quality_audit_mode=mode)
    result = adapter.evaluate(answer, private_record=private_record())
    return result, judge, ledger.snapshot(), adapter


def without_durations(value):
    if isinstance(value, dict):
        return {key: without_durations(item) for key, item in value.items()
                if key not in {"duration", "quality_audit"}}
    if isinstance(value, list):
        return [without_durations(item) for item in value]
    return value


def test_opt_in_sidecar_preserves_official_protocol_scores_and_metered_usage():
    original, old_judge, old_budget, old_adapter = evaluate()
    screened, new_judge, new_budget, new_adapter = evaluate("risk_only")
    assert "quality_audit" not in original
    assert all("quality_audit" not in row for row in original["feedback"])
    assert without_durations(screened) == without_durations(original)
    assert old_judge.calls == new_judge.calls
    assert old_adapter.evaluator_version == new_adapter.evaluator_version
    assert len(old_judge.calls) == len(new_judge.calls) == 2
    assert old_budget["model_calls"] == new_budget["model_calls"] == 2
    assert old_budget["tokens"] == new_budget["tokens"] == 60
    assert screened["quality_audit"]["model_calls"] == 0
    assert screened["score"] == 2 / 5 and screened["complete"]


def test_negative_weight_with_apparently_conflicting_reasoning_is_not_flipped():
    result, _, _, _ = evaluate("risk_only")
    penalty = result["feedback"][1]
    audit = penalty["quality_audit"]
    assert penalty["score"] == 1 and penalty["verdict"] == "Satisfied"
    assert penalty["reasoning"] == "The document has no irrelevant material."
    assert penalty["weight"] == -3
    assert "negative_weight_polarity_requires_review" in audit["risk_flags"]
    assert audit["semantic_consistency"] == "unassessed"
    assert audit["score_modified"] is False
    assert penalty["status"] == "ok" and penalty["success"]


def test_quote_risks_are_exact_screening_counts_not_fabrication_verdicts():
    result, _, _, _ = evaluate("risk_only", [judgment(quotes=["evidence", "Evidence"]), judgment()])
    positive, negative = [row["quality_audit"] for row in result["feedback"]]
    assert positive["risk_flags"] == ["unverified_evidence_quote"]
    assert positive["quote_count"] == 2
    assert positive["verified_quote_count"] == positive["unverified_quote_count"] == 1
    assert "no_verified_evidence_quote" in negative["risk_flags"]
    summary = result["quality_audit"]
    assert summary["version"] == QUALITY_AUDIT_VERSION
    assert summary["screened_rows"] == summary["flagged_rows"] == 2
    assert summary["negative_weight_rows"] == 1
    assert summary["quote_count"] == 2 and summary["unverified_quote_count"] == 1
    assert summary["risk_counts"] == {"negative_weight_polarity_requires_review": 1,
        "unverified_evidence_quote": 1, "no_verified_evidence_quote": 1}
    assert summary["semantic_consistency"] == "unassessed"
    assert "not proven contradictions" in summary["note"]


def test_unflagged_row_is_still_semantically_unassessed_and_hash_bound():
    result, _, _, _ = evaluate("risk_only")
    audit = result["feedback"][0]["quality_audit"]
    assert audit["risk_flags"] == []
    assert audit["semantic_consistency"] == "unassessed"
    assert audit["answer_sha256"] == hashlib.sha256(b"real evidence here").hexdigest()
    assert audit["answer_sha256"] == result["quality_audit"]["answer_sha256"]
    changed, _, _, _ = evaluate("risk_only", [judgment(quotes=["evidence"], reasoning="Different reasoning"), judgment()])
    assert changed["feedback"][0]["quality_audit"]["judgment_sha256"] != audit["judgment_sha256"]


def test_judge_failure_is_not_retried_by_risk_screening():
    result, judge, budget, _ = evaluate("risk_only", [RuntimeError("offline judge failure"), judgment(0)])
    assert not result["complete"] and result["feedback"][0]["status"] == "error"
    assert len(judge.calls) == result["model_calls"] == budget["model_calls"] == 2
    assert result["quality_audit"]["model_calls"] == 0
    assert all(row["quality_audit"]["semantic_consistency"] == "unassessed" for row in result["feedback"])


def test_quality_mode_rejects_implicit_semantic_model_calls():
    with pytest.raises(ValueError, match="off or risk_only"):
        ResearchRubricsAdapter(quality_audit_mode="semantic")


@pytest.mark.parametrize("score", [0, 1])
def test_risk_flags_do_not_create_an_experience_promotion_decision(tmp_path, score):
    result, _, _, _ = evaluate("risk_only", [judgment(score), judgment(0)])
    feedback = convert_feedback(result)
    assert feedback.raw["quality_audit"]["flagged_rows"] == 2
    proposal = ChangeProposal(proposal_id="p", source_task_id="source", base_version=0,
        experience=Experience(experience_id="e", bank="execution", instruction="Check relevant evidence.",
            applicability="Explanatory tasks", source_task_ids=["source"], evidence=["observed"]),
        diff="Add an evidence check", rationale="Preserve evidence", evidence=["observed"],
        expected_benefit="Clearer explanations")
    store = ExperienceStore(tmp_path / "experience.sqlite")
    try:
        assert store.commit(proposal).applied_proposals == ["p"]
        assert "validation_status" not in store.snapshot().experiences[0].model_dump()
    finally:
        store.close()
