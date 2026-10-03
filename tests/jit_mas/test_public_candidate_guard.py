"""Public candidate selection never uses an evaluator or expands task budgets."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels
from jit_mas.public_refinement import PUBLIC_REFINEMENT_IDS, refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.created = []

    def create(self, role, agent_id, ledger, stage):
        self.created.append((role, agent_id, stage))
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return ChatMessage(role="assistant", content=reply if isinstance(reply, str) else json.dumps(reply))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def inputs(*, guarded=True, answer=None, question="Write a complete report from the supplied public notes.",
           max_calls=None):
    draft = answer if answer is not None else (
        "The first section explains the public facts and their relevance in detail. " * 12
        + "\n\nThe second section compares the requested cases with concrete examples. " * 12
        + "\n\nThe final section explains the practical implications and limitations. " * 12
    )
    task = PublicTask(task_id="OPAQUE_PRIVATE_ROUTE", question=question)
    result = RunResult(answer=draft, terminated_reason="final_answer", metadata={
        "private_feedback": "PRIVATE_CANARY", "raw_judge": {"score": "PRIVATE_CANARY"},
        "call_counts": {"writer": 1}, "model_calls_used": 1,
        "terminal_receipt": {"event_id": "public-final-event", "answer_hash": digest(draft)},
        "artifacts": {"writer": {"answer": draft, "event_id": "public-final-event", "ledger": {}}},
    })
    config = MASConfig(backend="scripted", public_refinement=True, public_refinement_guard=guarded,
                       public_refinement_response_format="json_schema",
                       models={"global": ModelConfig(max_tokens=16000), "exec": ModelConfig(max_tokens=8192)})
    ledger = BudgetLedger(max_calls, 2_000_000, 0, timeout_seconds=900)
    return task, result, config, ledger


def run(inputs_tuple, revision, review=None):
    task, result, config, ledger = inputs_tuple
    models = Models([{"issues": []} if review is None else review, revision])
    snapshots = []
    refine_public_answer(task, result, models, ledger, config, synthesizer_id="writer",
                         audit_writer=lambda value: snapshots.append(copy.deepcopy(value)))
    return result.metadata["public_refinement"], models, snapshots


def test_title_only_revision_is_rejected_without_losing_valid_execution_receipts_or_costs():
    fixture = inputs()
    task, result, config, ledger = fixture
    original_answer, original_metadata = result.answer, copy.deepcopy(result.metadata)
    audit, models, snapshots = run(fixture, {"answer": "Public facts: a report"})
    assert result.answer == original_answer
    assert audit["status"] == "completed" and audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "catastrophic_revision_body_loss"
    assert audit["revision"] == {"answer": "Public facts: a report"}
    assert audit["revision_answer_hash"] == digest("Public facts: a report")
    assert audit["revision_answer_hash"] != audit["draft_hash"] == audit["answer_hash"]
    assert audit["calls"][-1]["status"] == "validated"
    assert audit["answer_hash"] == digest(original_answer)
    assert audit["revision_public_diagnostics"]["characters"] < audit["selected_public_diagnostics"]["characters"]
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is True
    assert {key: value for key, value in result.metadata.items() if key != "public_refinement"} == original_metadata
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit == snapshots[-1]
    assert audit["audit_hash"] == digest({key: value for key, value in audit.items() if key != "audit_hash"})
    serialized = json.dumps(models.requests)
    assert "PRIVATE_CANARY" not in serialized and "OPAQUE_PRIVATE_ROUTE" not in serialized
    assert "transparent public structural guard" in models.requests[-1]["messages"][0]["content"]
    assert "candidate comparison, fallback" not in models.requests[-1]["messages"][0]["content"]


@pytest.mark.parametrize("revision", [
    "A complete concise report.\n\nIt describes the cases and their practical implications.",
    "The concise report preserves the relevant details and comparison. " * 8,
])
def test_guard_does_not_reject_normal_shortening_or_multiple_body_chunks(revision):
    fixture = inputs()
    audit, _, _ = run(fixture, {"answer": revision})
    assert fixture[1].answer == revision
    assert audit["selected_candidate"] == "revision" and audit["status"] == "completed"


@pytest.mark.parametrize("draft", ["Brief initial answer.", "a" * 1500])
def test_short_draft_or_single_chunk_cannot_trigger_body_loss_heuristic(draft):
    fixture = inputs(answer=draft)
    audit, _, _ = run(fixture, {"answer": "Corrected concise answer."})
    assert fixture[1].answer == "Corrected concise answer."
    assert audit["selected_candidate"] == "revision"


@pytest.mark.parametrize("invalid_revision", [
    '{"answer":"unfinished', '{"answer":"one","answer":"two"}',
    {"answer": "complete-looking", "additionalProperties": False},
    {"answer": " "}, {"answer": 7},
])
def test_invalid_revision_selects_draft_with_explicit_component_failure_and_raw_response(invalid_revision):
    fixture = inputs()
    draft = fixture[1].answer
    audit, models, _ = run(fixture, invalid_revision)
    assert fixture[1].answer == draft
    assert audit["status"] == "completed_with_component_failure"
    assert audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "invalid_local_revision_response"
    assert audit["revision"] is None and audit["revision_hash"] is None
    assert audit["calls"][-1]["status"] == "failed"
    assert audit["calls"][-1]["failure_phase"] == "response_validation"
    assert audit["calls"][-1]["response"]
    assert audit["component_failures"][0]["response_hash"] == audit["calls"][-1]["response_hash"]
    assert audit["component_failures"][0]["error_type"] == audit["calls"][-1]["error_type"]
    assert len(audit["budget_records"]) == len(models.requests) == 2
    assert fixture[3].snapshot()["tokens"] == 100


def test_invalid_review_still_fails_without_attempting_revision():
    task, result, config, ledger = inputs()
    models = Models(['{"issues":[],"score":1}', {"answer": "UNAUTHORIZED_SECOND_CALL"}])
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert audit["status"] == "failed" and audit["review"] is None
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert audit["calls"][-1]["failure_phase"] == "response_validation"
    assert "selected_candidate" not in audit


@pytest.mark.parametrize("error", [ValueError("synthetic model identity mismatch"),
                                  RuntimeError("synthetic transport failure"),
                                  TimeoutError("synthetic provider timeout")])
def test_provider_errors_are_never_converted_to_successful_draft_selection(error):
    task, result, config, ledger = inputs()
    models = Models([{"issues": []}, error])
    with pytest.raises(type(error), match="synthetic"):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert audit["status"] == "failed" and "selected_candidate" not in audit
    assert audit["calls"][-1]["failure_phase"] == "provider"
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit["budget_records"][-1]["estimated"] is True


def test_call_budget_remains_binding_with_guard_enabled():
    task, result, config, ledger = inputs(max_calls=1)
    models = Models([{"issues": []}])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, config)
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 1


def test_local_parse_failure_at_expired_deadline_does_not_select_initial(monkeypatch):
    task, result, config, ledger = inputs()

    class ExpiringModels(Models):
        def __call__(self, messages, **kwargs):
            response = super().__call__(messages, **kwargs)
            if len(self.requests) == 2:
                monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0.0)
            return response

    models = ExpiringModels([{"issues": []}, '{"answer":"unfinished'])
    with pytest.raises(TimeoutError, match="budget exhausted"):
        refine_public_answer(task, result, models, ledger, config)
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert "selected_candidate" not in result.metadata["public_refinement"]
    assert ledger.snapshot()["model_calls"] == 2


@pytest.mark.parametrize("construction", ["position", "numeric", "projection"])
def test_initial_draft_cannot_replace_unvalidated_final_public_construction(construction):
    question = "Write a complete scene."
    if construction == "position":
        question += " Include keyword brightly in the second sentence, as the third word of that sentence."
    elif construction == "numeric":
        question += " Include exactly 2 numbers in the response."
    task, result, config, ledger = inputs(question=question)
    if construction == "position":
        config.public_positional_construction = True
    elif construction == "numeric":
        config.public_numeric_construction = True
    else:
        result.metadata["public_positional_draft_projection"] = {"active": True}
    models = Models([{"issues": []}, '{"answer":"unfinished'])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is False
    assert audit["status"] == "failed" and "selected_candidate" not in audit
    assert ledger.snapshot()["model_calls"] == 2


@pytest.mark.parametrize("guarded", [False, True])
def test_missing_execution_terminal_artifact_is_never_rescued(guarded):
    task, result, config, ledger = inputs(guarded=guarded)
    result.terminated_reason = "budget_exceeded"
    models = Models([])
    with pytest.raises(ValueError, match="valid final-answer draft"):
        refine_public_answer(task, result, models, ledger, config)
    assert not models.requests and "public_refinement" not in result.metadata


def test_guard_default_off_preserves_strict_revision_failure_and_revision_only_selection():
    assert MASConfig().public_refinement_guard is False
    task, result, config, ledger = inputs(guarded=False)
    models = Models([{"issues": []}, {"answer": "looks valid", "extra": 1}])
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, config)
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert "public_candidate_guard" not in result.metadata["public_refinement"]
    fixture = inputs(guarded=False)
    audit, _, _ = run(fixture, {"answer": "Only a title"})
    assert fixture[1].answer == "Only a title"
    assert audit["selection_policy"] == "submit_validated_revision_only_without_score_selection"


def test_guard_requires_public_refinement():
    with pytest.raises(ValidationError, match="requires public_refinement"):
        MASConfig(public_refinement_guard=True)


class GuardPipelineModels(FixtureModels):
    def __init__(self):
        super().__init__()
        self.refinement = Models([{"issues": []}, '{"answer":"unfinished'])

    def create(self, role, agent_id, ledger, stage):
        if agent_id in PUBLIC_REFINEMENT_IDS:
            return self.refinement.create(role, agent_id, ledger, stage)
        model = super().create(role, agent_id, ledger, stage)
        if role == "judge":
            class NeverInvokedJudge:
                def __call__(self, *args, **kwargs):
                    raise AssertionError("Private evaluation must not be invoked before deferred submission")

                def __getattr__(self, name):
                    return getattr(model, name)

            return NeverInvokedJudge()
        return model


def test_pipeline_seals_guard_selected_draft_with_costs_before_any_evaluation(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    provider = GuardPipelineModels()
    pipeline = make_pipeline(MASConfig(backend="scripted", public_refinement=True,
        public_refinement_guard=True, max_agents=1, team_max_calls=1, max_model_calls=None),
        store, tmp_path / "runs", fixture_models=provider)
    try:
        outcome = pipeline.run_task("test-poem", store.snapshot(), mode="evaluate", defer_evaluation=True)
        directory = Path(outcome["run_dir"])
        draft = json.loads((directory / "execution_draft.json").read_text())
        execution = json.loads((directory / "execution.json").read_text())
        audit = json.loads((directory / "public_refinement.json").read_text())
        submission = json.loads((directory / "submission.json").read_text())
        trace = json.loads((directory / "call_trace.json").read_text())
        assert audit["status"] == "completed_with_component_failure"
        assert submission["answer"] == execution["answer"] == draft["answer"] == audit["draft"]
        assert submission["answer_hash"] == outcome["answer_hash"] == audit["answer_hash"] == digest(draft["answer"])
        assert len(trace["public_refinement_calls"]) == len(audit["budget_records"]) == 2
        assert not (directory / "evaluation.json").exists()
        assert outcome["status"] == "submitted_unscored"
        assert "PRIVATE_CANARY" not in json.dumps(provider.refinement.requests)
    finally:
        store.close()
