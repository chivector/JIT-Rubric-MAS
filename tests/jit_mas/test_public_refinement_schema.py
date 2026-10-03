"""Generic schema transport and one-attempt failure tests; synthetic only."""
import copy
import json

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import PublicReview, PublicRevision, refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.created, self.requests = [], []

    def create(self, role, agent_id, ledger, stage):
        self.created.append((role, agent_id, ledger, stage))
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return ChatMessage(role="assistant", content=response if isinstance(response, str) else json.dumps(response))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def inputs(format="json_schema", cap=8192):
    task = PublicTask(task_id="synthetic-private-route", question="Write a short fictional scene.")
    result = RunResult(answer="Original draft.", terminated_reason="final_answer", metadata={
        "private_feedback": "PRIVATE_CANARY", "call_counts": {"writer": 1}})
    config = MASConfig(backend="scripted", public_refinement=True,
                       public_refinement_response_format=format,
                       models={"global": ModelConfig(max_tokens=16000), "exec": ModelConfig(max_tokens=cap)})
    ledger = BudgetLedger(None, 2_000_000, None, timeout_seconds=900)
    return task, result, config, ledger


def test_response_format_defaults_to_object_and_public_refinement_stays_off():
    config = MASConfig()
    assert config.public_refinement_response_format == "json_object" and not config.public_refinement
    assert config.public_positional_construction is False
    with pytest.raises(ValidationError):
        MASConfig(public_refinement_response_format="automatic_fallback")


@pytest.mark.parametrize("options", [
    {"public_refinement": False, "public_refinement_response_format": "json_schema"},
    {"public_refinement": True, "public_refinement_response_format": "json_object"},
])
def test_positional_construction_requires_active_schema_refinement(options):
    with pytest.raises(ValidationError, match="requires public_refinement and json_schema"):
        MASConfig(public_positional_construction=True, **options)


def positional_inputs():
    task, result, config, ledger = inputs()
    task.question = ("Write a fictional scene. Include keyword brightly in the second sentence, "
                     "as the third word of that sentence.")
    config.public_positional_construction = True
    return task, result, config, ledger


def test_positional_revision_uses_two_calls_and_submits_rendered_complete_scene():
    task, result, config, ledger = positional_inputs()
    slots = {"preceding_sentences": ["The curtain rose."], "prefix_words": ["She", "smiled"],
             "keyword": "brightly", "suffix_words": ["as", "the", "crowd", "cheered"],
             "following_sentences": ["The lights dimmed."]}
    models = Models([{"issues": []}, slots])
    refine_public_answer(task, result, models, ledger, config)
    assert result.answer == "The curtain rose. She smiled brightly as the crowd cheered. The lights dimmed."
    assert result.answer.split(". ")[1].split()[2] == "brightly"
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    audit = result.metadata["public_refinement"]
    assert audit["revision"] == slots and audit["answer_hash"] == digest(result.answer)
    assert audit["public_positional_construction"]["matched_public_spans"]
    schema = models.requests[1]["kwargs"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["prefix_words"]["minItems"] == 2
    assert "answer" not in schema["properties"]
    assert "PRIVATE_CANARY" not in json.dumps(models.requests)
    assert "synthetic-private-route" not in json.dumps(models.requests)


def test_invalid_word_slots_keep_original_draft_and_charged_failure_without_retry():
    task, result, config, ledger = positional_inputs()
    models = Models([{"issues": []}, {"preceding_sentences": ["The curtain rose."],
        "prefix_words": ["She smiled"], "keyword": "brightly", "suffix_words": [],
        "following_sentences": []}])
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, config)
    assert result.answer == "Original draft."
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert result.metadata["public_refinement"]["revision"] is None


def test_conflicting_public_positions_use_regular_revision_without_guessing_a_rule():
    task, result, config, ledger = positional_inputs()
    task.constraints = ["Place keyword softly as the fourth word in the third sentence."]
    models = Models([{"issues": []}, {"answer": "A complete fictional scene with an acknowledged conflict."}])
    refine_public_answer(task, result, models, ledger, config)
    assert result.metadata["public_refinement"]["public_positional_construction"]["active"] is False
    assert "answer" in models.requests[1]["kwargs"]["response_format"]["json_schema"]["schema"]["properties"]
    assert ledger.snapshot()["model_calls"] == 2


@pytest.mark.parametrize("cap,expected_cap", [(4096, 4096), (16000, 8192)])
def test_both_global_stages_send_actual_distinct_strict_schema_and_audit_it(cap, expected_cap):
    task, result, config, ledger = inputs(cap=cap)
    models = Models([{"issues": []}, {"answer": "Complete revised scene."}])
    snapshots = []
    refine_public_answer(task, result, models, ledger, config,
                         knowledge_policy="model_general_knowledge_allowed",
                         audit_writer=lambda audit: snapshots.append(copy.deepcopy(audit)))
    assert result.answer == "Complete revised scene." and result.metadata["call_counts"] == {"writer": 1}
    assert [(r, a, s) for r, a, _, s in models.created] == [
        ("global", "public-review", "inference"), ("global", "public-revision", "inference")]
    assert all(shared is ledger for _, _, shared, _ in models.created)
    audit = result.metadata["public_refinement"]
    for request, call, schema in zip(models.requests, audit["calls"], (PublicReview, PublicRevision)):
        expected = {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}}
        assert request["kwargs"]["response_format"] == expected
        assert request["kwargs"]["max_tokens"] == expected_cap
        assert 0 < request["kwargs"]["timeout"] <= 900
        assert call["response_format"] == expected and call["response_format_hash"] == digest(expected)
        assert "PRIVATE_CANARY" not in json.dumps(request["messages"])
        assert "synthetic-private-route" not in json.dumps(request["messages"])
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit["draft"] == "Original draft." and audit["status"] == "completed"
    assert audit == snapshots[-1]
    assert audit["audit_hash"] == digest({key: value for key, value in audit.items() if key != "audit_hash"})


def test_legacy_object_format_is_explicitly_sent_and_audited():
    task, result, config, ledger = inputs(format="json_object")
    models = Models([{"issues": []}, {"answer": "Complete revised scene."}])
    refine_public_answer(task, result, models, ledger, config)
    for request, call in zip(models.requests, result.metadata["public_refinement"]["calls"]):
        assert request["kwargs"]["response_format"] == call["response_format"] == {"type": "json_object"}
        assert call["response_format_hash"] == digest({"type": "json_object"})


@pytest.mark.parametrize("stage,bad", [
    ("review", '{"issues": [], "issues": []}'),
    ("review", {"issues": [], "score": 1}),
    ("review", {"issues": [{"defect": "x", "public_basis": "public request", "repair": "x"}] * 9}),
    ("revision", '{"answer": "first", "answer": "second"}'),
    ("revision", {"answer": "Valid-looking scene.", "extra": "forbidden"}),
    ("revision", {"answer": "   "}),
    ("revision", {"answer": True}),
])
def test_requested_strict_schema_does_not_relax_local_validator_or_select_fallback(stage, bad):
    task, result, config, ledger = inputs()
    models = Models(([{"issues": []}] if stage == "revision" else []) + [bad, {"answer": "UNAUTHORIZED_RETRY"}])
    with pytest.raises((ValueError, ValidationError)):
        refine_public_answer(task, result, models, ledger, config)
    expected_calls = 2 if stage == "revision" else 1
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == expected_calls
    assert ledger.snapshot()["tokens"] == 50 * expected_calls and ledger.snapshot()["reserved_tokens"] == 0
    assert result.answer == audit["draft"] == "Original draft." and audit["status"] == "failed"
    assert audit["calls"][-1]["status"] == "failed" and "response" in audit["calls"][-1]
    assert all(request["kwargs"]["response_format"]["type"] == "json_schema" for request in models.requests)


def test_schema_api_rejection_is_one_charged_failure_without_object_fallback_or_retry():
    task, result, config, ledger = inputs()
    models = Models([RuntimeError("synthetic schema rejection"), {"issues": []}, {"answer": "UNAUTHORIZED_RETRY"}])
    with pytest.raises(RuntimeError, match="synthetic schema rejection"):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert models.requests[0]["kwargs"]["response_format"]["type"] == "json_schema"
    assert ledger.snapshot()["tokens"] > 0 and ledger.snapshot()["records"][0]["estimated"]
    assert ledger.snapshot()["reserved_tokens"] == 0
    audit = result.metadata["public_refinement"]
    assert audit["status"] == "failed" and audit["review"] is None and audit["revision"] is None
    assert result.answer == "Original draft." and len(audit["calls"]) == 1
    assert audit["calls"][0]["response_format_hash"] == digest(models.requests[0]["kwargs"]["response_format"])
