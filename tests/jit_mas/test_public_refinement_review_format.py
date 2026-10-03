"""Mixed review/prose schema transport preserves local validation and charging."""
import copy
import json

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import PublicReview, refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        assert role == "global" and stage == "inference"
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        response = next(self.replies)
        return ChatMessage(role="assistant", content=response if isinstance(response, str) else json.dumps(response))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def inputs(positional=False):
    task = PublicTask(task_id="synthetic-only", question="Write a complete report from the supplied public notes.")
    if positional:
        task.question = "Write a scene. Include keyword brightly in the second sentence, as the third word of that sentence."
    result = RunResult(answer="Original draft.", terminated_reason="final_answer", metadata={"private_feedback": "PRIVATE_CANARY"})
    config = MASConfig(backend="scripted", public_refinement=True,
        public_refinement_response_format="json_schema_review", public_positional_construction=positional,
        models={"global": ModelConfig(max_tokens=16000), "exec": ModelConfig(max_tokens=8192)})
    ledger = BudgetLedger(None, 2_000_000, None, timeout_seconds=900)
    return task, result, config, ledger


def test_review_schema_and_ordinary_prose_object_transport_preserve_complete_artifact_and_audit():
    task, result, config, ledger = inputs()
    artifact = "A complete report.\n\n" + "Supported public details remain in the final artifact. " * 100
    models = Models([{"issues": []}, {"answer": artifact}])
    snapshots = []
    refine_public_answer(task, result, models, ledger, config,
                        audit_writer=lambda audit: snapshots.append(copy.deepcopy(audit)))
    expected_review = {"type": "json_schema", "json_schema": {
        "name": "PublicReview", "strict": True, "schema": PublicReview.model_json_schema()}}
    assert models.requests[0]["kwargs"]["response_format"] == expected_review
    assert models.requests[1]["kwargs"]["response_format"] == {"type": "json_object"}
    assert result.answer == artifact
    audit = result.metadata["public_refinement"]
    assert audit == snapshots[-1] and audit["draft"] == "Original draft."
    for call, request in zip(audit["calls"], models.requests):
        assert call["response_format"] == request["kwargs"]["response_format"]
        assert call["response_format_hash"] == digest(call["response_format"])
        assert request["kwargs"]["max_tokens"] == 8192
    assert "PRIVATE_CANARY" not in json.dumps(models.requests)
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["tokens"] == 100
    assert ledger.snapshot()["reserved_tokens"] == 0 and audit["status"] == "completed"


def test_supported_positional_revision_retains_strict_dynamic_schema_and_rendered_answer():
    task, result, config, ledger = inputs(positional=True)
    slots = {"preceding_sentences": ["The curtain rose."], "prefix_words": ["She", "smiled"],
        "keyword": "brightly", "suffix_words": ["as", "the", "crowd", "cheered"], "following_sentences": []}
    models = Models([{"issues": []}, slots])
    refine_public_answer(task, result, models, ledger, config)
    formats = [request["kwargs"]["response_format"] for request in models.requests]
    assert [item["type"] for item in formats] == ["json_schema", "json_schema"]
    assert formats[0]["json_schema"]["name"] == "PublicReview"
    schema = formats[1]["json_schema"]
    assert schema["strict"] is True and "answer" not in schema["schema"]["properties"]
    assert schema["schema"]["properties"]["prefix_words"]["minItems"] == 2
    assert result.answer == "The curtain rose. She smiled brightly as the crowd cheered."
    assert result.metadata["public_refinement"]["revision"] == slots
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["tokens"] == 100


def test_unsupported_positional_rule_uses_ordinary_object_revision_without_guessing():
    task, result, config, ledger = inputs(positional=True)
    task.constraints = ["Place keyword softly as the fourth word in the third sentence."]
    models = Models([{"issues": []}, {"answer": "Complete supported artifact."}])
    refine_public_answer(task, result, models, ledger, config)
    assert result.metadata["public_refinement"]["public_positional_construction"]["active"] is False
    assert models.requests[1]["kwargs"]["response_format"] == {"type": "json_object"}
    assert result.answer == "Complete supported artifact."


@pytest.mark.parametrize("stage,bad", [
    ("review", '{"issues":[],"issues":[]}'),
    ("review", '{"issues":['),
    ("revision", '{"answer":"first","answer":"second"}'),
    ("revision", '{"answer":"unfinished"'),
    ("revision", {"answer": "   "}),
    ("revision", {"answer": "valid-looking", "score": 1}),
])
def test_mixed_format_still_rejects_invalid_output_without_retry_or_fallback(stage, bad):
    task, result, config, ledger = inputs()
    replies = ([{"issues": []}] if stage == "revision" else []) + [bad, {"answer": "UNAUTHORIZED_RETRY"}]
    models = Models(replies)
    with pytest.raises((ValueError, ValidationError)):
        refine_public_answer(task, result, models, ledger, config)
    count = 2 if stage == "revision" else 1
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == count
    assert ledger.snapshot()["tokens"] == 50 * count and ledger.snapshot()["reserved_tokens"] == 0
    assert result.answer == audit["draft"] == "Original draft." and audit["status"] == "failed"
    assert audit["calls"][-1]["status"] == "failed" and audit["calls"][-1]["response"]


@pytest.mark.parametrize("mode", ["json_schema", "json_schema_review"])
def test_positional_config_accepts_both_supported_schema_modes(mode):
    config = MASConfig(public_refinement=True, public_refinement_response_format=mode,
                       public_positional_construction=True)
    assert config.public_positional_construction
