"""Simultaneous edits and fixed-call ordinary refinement preserve public artifacts."""
import copy
import json

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import refine_public_answer
from jit_mas.public_text_patches import PublicPatchRevision, apply_public_text_patches
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage

REVIEW = {"issues": [{"defect": "Fix a public fact", "public_basis": "The supplied passage", "repair": "Correct the fact"}]}

def patches(*pairs):
    return PublicPatchRevision(edits=[{"issue_index": 0, "old_text": a, "new_text": b} for a, b in pairs])

def test_simultaneous_replacements_preserve_untargeted_unicode_layout_and_sources():
    draft = "甲。\r\n\r\nScore 2.7 meets 2.8.\n\nSources: [A]."
    answer, receipt = apply_public_text_patches(draft, patches(("meets", "does not meet"), ("[A]", "[A], [B]")), REVIEW)
    assert answer == "甲。\r\n\r\nScore 2.7 does not meet 2.8.\n\nSources: [A], [B]."
    assert receipt["edit_count"] == 2 and receipt["unchanged_outside_source_spans"]
    assert receipt["draft_hash"] == digest(draft) and receipt["answer_hash"] == digest(answer)

@pytest.mark.parametrize("draft,edits", [
    ("aaa", (("aa", "X"),)),  # Overlapping occurrences cannot pass str.count.
    ("A A", (("A", "B"),)),
    ("ABC", (("D", "E"),)),
    ("ABC", (("AB", "X"), ("BC", "Y"))),
    ("ABC", (("A", "X"), ("X", "Y"))),  # Edits do not target prior edits.
    ("ABC", (("ABC", ""),)),
])
def test_ambiguous_missing_overlapping_chained_and_empty_final_edits_fail(draft, edits):
    with pytest.raises(ValueError):
        apply_public_text_patches(draft, patches(*edits), REVIEW)

def test_empty_review_only_allows_no_op_and_missed_defects_are_not_certified():
    draft = "A wrong initial fact.\n\nA second paragraph."
    answer, receipt = apply_public_text_patches(draft, PublicPatchRevision(edits=[]), {"issues": []})
    assert answer == draft and receipt["edit_count"] == 0
    assert "not semantic correctness" in receipt["limitations"]
    with pytest.raises(ValueError, match="issue index"):
        apply_public_text_patches(draft, patches(("wrong", "correct")), {"issues": []})

def test_render_revalidates_model_construct_and_mutable_edit_lists():
    bad = PublicPatchRevision.model_construct(edits=[{"issue_index": False, "old_text": "A", "new_text": "B"}])
    with pytest.raises(ValidationError):
        apply_public_text_patches("A", bad, REVIEW)
    with pytest.raises(ValidationError):
        PublicPatchRevision(edits=[{"issue_index": 0, "old_text": "", "new_text": "B"}])

class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
    def create(self, role, agent_id, ledger, stage):
        return MeteredModel(self, ledger, stage, agent_id, 8192)
    def __call__(self, messages, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "kwargs": kwargs}))
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return ChatMessage(role="assistant", content=json.dumps(reply))
    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}

def refine(review, revision, *, guarded=True, positional=False):
    question = ("Write a scene. Include word oak as the 2nd word in the 1st sentence."
                if positional else "Write a two-paragraph scene.")
    task = PublicTask(task_id="OPAQUE_PRIVATE_ROUTE", question=question)
    draft = "The boat arrived.\n\nBirds sang."
    result = RunResult(answer=draft, terminated_reason="final_answer", metadata={"private_feedback": "DO_NOT_SEND"})
    config = MASConfig(backend="scripted", public_refinement=True, public_refinement_guard=guarded,
        public_revision_mode="patch", public_refinement_response_format="json_schema",
        public_positional_construction=positional,
        models={"exec": ModelConfig(max_tokens=8192), "global": ModelConfig(max_tokens=16000)})
    models = Models([review, revision])
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    snapshots = []
    refine_public_answer(task, result, models, ledger, config, audit_writer=lambda x: snapshots.append(copy.deepcopy(x)))
    return result, models, ledger, snapshots

def test_empty_review_full_artifact_is_copied_locally_with_two_calls_and_hash_receipts():
    result, models, ledger, snapshots = refine({"issues": []}, {"edits": []})
    assert result.answer == "The boat arrived.\n\nBirds sang."
    audit = result.metadata["public_refinement"]
    assert audit["public_patch_revision"]["edit_count"] == 0
    assert audit["revision"] == {"edits": []} and audit["revision_answer_hash"] == digest(result.answer)
    assert audit == snapshots[-1] and audit["audit_hash"] == digest({k:v for k,v in audit.items() if k != "audit_hash"})
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100
    assert "DO_NOT_SEND" not in json.dumps(models.requests)
    assert "OPAQUE_PRIVATE_ROUTE" not in json.dumps(models.requests)
    assert "PublicPatchRevision" == models.requests[-1]["kwargs"]["response_format"]["json_schema"]["name"]

def test_bad_patch_application_keeps_cost_and_raw_validated_revision_under_guard():
    result, models, ledger, _ = refine(REVIEW, {"edits": [{"issue_index": 0, "old_text": "Missing text", "new_text": "B"}]})
    audit = result.metadata["public_refinement"]
    assert audit["selection_reason"] == "invalid_public_patch_application"
    assert audit["status"] == "completed_with_component_failure"
    assert audit["public_patch_revision"]["status"] == "failed"
    assert audit["revision"]["edits"] and audit["calls"][-1]["status"] == "validated"
    assert audit["component_failures"][0]["failure_phase"] == "patch_application"
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2

def test_bad_patch_without_guard_fails_and_typed_construction_keeps_its_schema():
    with pytest.raises(ValueError, match="exactly once"):
        refine(REVIEW, {"edits": [{"issue_index": 0, "old_text": "Missing", "new_text": "B"}]}, guarded=False)
    result, models, _, _ = refine({"issues": []}, {"preceding_sentences": [], "prefix_words": ["An"], "keyword": "oak",
        "suffix_words": ["grew"], "following_sentences": []}, positional=True)
    assert result.answer == "An oak grew."
    assert "public_patch_revision" not in result.metadata["public_refinement"]
    assert models.requests[-1]["kwargs"]["response_format"]["json_schema"]["name"] == "PublicWordSlotsResponse"

def test_default_config_keeps_full_revision_and_patch_requires_refinement():
    assert MASConfig(backend="scripted").public_revision_mode == "full"
    assert not MASConfig(backend="scripted").public_membership_observations
    with pytest.raises(ValidationError, match="requires public_refinement"):
        MASConfig(backend="scripted", public_revision_mode="patch")
