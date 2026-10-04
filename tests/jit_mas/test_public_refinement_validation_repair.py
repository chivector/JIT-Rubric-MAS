"""Bounded public typed-response repairs with a real shared meter; synthetic only."""
import copy
import json

import pytest
from pydantic import ValidationError

from jit_mas import public_refinement
from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_numeric_template import template_plan, render as render_template
from jit_mas.public_refinement import refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Models:
    def __init__(self, replies, hook=None):
        self.replies = iter(replies)
        self.hook = hook
        self.created, self.requests = [], []

    def create(self, role, agent_id, ledger, stage):
        self.created.append((role, agent_id, ledger, stage))
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        reply = next(self.replies)
        if self.hook:
            self.hook(len(self.requests))
        if isinstance(reply, BaseException):
            raise reply
        return ChatMessage(role="assistant", content=reply if isinstance(reply, str) else json.dumps(reply))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


@pytest.mark.parametrize('provider_failure', [False, True])
def test_compact_refinement_retains_rich_basis_before_requests_including_failures(provider_failure):
    task, result, config, ledger, invalid, valid = scenario('template')
    config.public_membership_observations = True
    config.public_membership_input_format = 'compact'
    published = []

    def check_persisted_basis(_):
        assert published
        audit = published[-1]['public_membership_input_audit']
        projected = published[-1]['public_input']['public_membership_observations']
        assert audit['full_observation_hash'] == digest(audit['observations'])
        assert projected['full_observation_hash'] == audit['full_observation_hash']
        assert audit['model_input_hash'] == digest(projected)

    replies = [RuntimeError('synthetic review provider failure')] if provider_failure else [
        {'issues': []}, invalid, valid]
    models = Models(replies, hook=check_persisted_basis)
    if provider_failure:
        with pytest.raises(RuntimeError, match='synthetic review provider failure'):
            refine_public_answer(task, result, models, ledger, config, audit_writer=published.append)
    else:
        refine_public_answer(task, result, models, ledger, config, audit_writer=published.append)
    assert 'public_membership_input_audit' in published[-1]
    for request in models.requests:
        payload = json.loads(request['messages'][-1]['content'])
        assert 'public_membership_input_audit' not in payload
        assert 'observations' not in payload['public_membership_observations']
        assert 'PUBLIC CONDITION MATRIX' in request['messages'][0]['content']
    assert len(models.requests) == ledger.snapshot()['model_calls'] == (1 if provider_failure else 3)


def test_full_membership_refinement_keeps_legacy_input_and_no_compact_audit():
    from jit_mas.public_membership import build_public_membership_observations

    task, result, config, ledger, _, valid = scenario('template')
    config.public_membership_observations = True
    expected = build_public_membership_observations(task, {'contributions': [], 'tool_observations': []}, result.answer)
    models = Models([{'issues': []}, valid])
    refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata['public_refinement']
    assert audit['public_input']['public_membership_observations'] == expected
    assert 'public_membership_input_audit' not in audit
    assert len(models.requests) == ledger.snapshot()['model_calls'] == 2
    assert all('PUBLIC CONDITION MATRIX' not in request['messages'][0]['content'] for request in models.requests)


def scenario(kind="template", *, retries=1, guarded=True, transport="json_schema"):
    positional = kind == "position"
    question = ("Write a compact scene. Include keyword Amber in the second sentence, "
                "as the third word of that sentence." if positional else
                "Write a compact notice. Include exactly 2 numbers. "
                "Use at least 2 different coordinating conjunctions.")
    task = PublicTask(task_id="PRIVATE_ROUTE_CANARY", question=question)
    config = MASConfig(backend="scripted", public_refinement=True,
        public_refinement_guard=guarded, public_refinement_response_format="json_schema_review",
        public_construction_validation_retries=retries,
        public_positional_construction=positional, public_numeric_construction=not positional,
        public_numeric_construction_layout="named_objects" if kind == "named" else
            "template" if kind == "template" else "array",
        public_construction_response_format=transport,
        models={"global": ModelConfig(max_tokens=16000, timeout=37),
                "exec": ModelConfig(max_tokens=4096)})
    result = RunResult(answer="The original compact draft remains intact.", terminated_reason="final_answer",
        metadata={"private_score": "PRIVATE_METADATA_CANARY", "call_counts": {"writer": 1}})
    if positional:
        valid = {"preceding_sentences": ["The doors opened."], "prefix_words": ["They", "saw"],
                 "keyword": "Amber", "suffix_words": ["nearby"], "following_sentences": ["They smiled."]}
        invalid = copy.deepcopy(valid)
        invalid["prefix_words"] = ["They saw"]
    elif kind == "template":
        valid = {"answer_template": "There were <NUM_A> crates and <NUM_B> letters but no reply.",
                 "numeric_values": {"NUM_A": "7", "NUM_B": "12"}}
        invalid = copy.deepcopy(valid)
        invalid["answer_template"] = "There were <NUM_A> crates and letters but no reply."
    else:
        numbers = [{"before": "They saw", "number": "7", "after": "crates."},
                   {"before": "They counted", "number": "12", "after": "letters."}]
        clauses = [{"before": "The doors opened", "after": "the people waited."},
                   {"before": "The light dimmed", "after": "the people stayed."}]
        valid = {"opening": "The hall was quiet.", "number_slots": numbers,
                 "conjunction_clauses": clauses, "closing": "They returned home."}
        if kind == "named":
            valid["number_slots"] = {"slot_2": numbers[1], "slot_1": numbers[0]}
            valid["conjunction_clauses"] = {"but": clauses[1], "and": clauses[0]}
        invalid = copy.deepcopy(valid)
        invalid["number_slots"] = [] if kind == "array" else {}
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    return task, result, config, ledger, invalid, valid


@pytest.mark.parametrize("kind", ["position", "template", "array", "named"])
@pytest.mark.parametrize("transport", ["json_schema", "json_object"])
@pytest.mark.parametrize("guarded", [False, True])
def test_one_typed_repair_uses_original_schema_payload_same_ledger_and_preserves_failure(kind, transport, guarded):
    task, result, config, ledger, invalid, valid = scenario(kind, transport=transport, guarded=guarded)
    old_task, old_metadata = task.model_dump(), copy.deepcopy(result.metadata)
    models = Models([{"issues": []}, invalid, valid])
    published = []
    refine_public_answer(task, result, models, ledger, config, audit_writer=published.append)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 3
    assert ledger.snapshot()["tokens"] == 150 and ledger.snapshot()["reserved_tokens"] == 0
    assert [(role, agent_id, stage) for role, agent_id, _, stage in models.created] == [
        ("global", "public-review", "inference"), ("global", "public-revision", "inference"),
        ("global", "public-revision", "inference")]
    assert all(shared is ledger for _, _, shared, _ in models.created)
    assert [call["attempt"] for call in audit["calls"]] == [1, 1, 2]
    assert [call["status"] for call in audit["calls"]] == ["validated", "failed", "validated"]
    assert audit["status"] == "completed_with_component_failure" and audit["revision"] == valid
    assert len(audit["component_failures"]) == 1 and audit["component_failures"][0]["attempt"] == 1
    failed = audit["calls"][1]
    assert failed["failure_phase"] == "response_validation" and failed["response_hash"] == digest(json.dumps(invalid))
    original_payload = json.loads(models.requests[1]["messages"][-1]["content"])
    repair_payload = json.loads(models.requests[2]["messages"][-1]["content"])
    repair = repair_payload.pop("public_construction_validation_repair")
    assert original_payload == repair_payload
    assert repair["previous_response"] == failed["response"]
    assert repair["previous_response_hash"] == failed["response_hash"]
    assert repair["validation_errors"] == json.loads(json.dumps(failed["validation_errors"]))
    assert all(not any(key in error for key in ["input", "ctx", "url", "context"]) for error in repair["validation_errors"])
    assert "PRIVATE_METADATA_CANARY" not in json.dumps(models.requests)
    assert "PRIVATE_ROUTE_CANARY" not in json.dumps(models.requests)
    assert task.model_dump() == old_task and result.metadata["call_counts"] == old_metadata["call_counts"]
    assert models.requests[1]["kwargs"]["response_format"] == models.requests[2]["kwargs"]["response_format"]
    assert all(request["kwargs"]["max_tokens"] == 4096 and 0 < request["kwargs"]["timeout"] <= 37 for request in models.requests)
    assert audit["audit_hash"] == digest({k: v for k, v in audit.items() if k != "audit_hash"})
    assert published[-1] == audit
    if kind == "template":
        plan = template_plan(task)
        assert result.answer == render_template(plan.model.model_validate(valid))
        assert repair["marker_counts"]["required_marker_counts"] == [
            {"marker": "<NUM_A>", "count": 1}, {"marker": "<NUM_B>", "count": 0}]
        assert repair["marker_counts"]["missing_markers"] == ["<NUM_B>"]
        assert repair["marker_counts"]["duplicate_markers"] == []
        assert json.loads(repair["previous_response"])["numeric_values"] == valid["numeric_values"]
    else:
        assert "marker_counts" not in repair


@pytest.mark.parametrize("bad", ['{', '[]', '{"answer_template": "a", "answer_template": "b"}',
    '{"answer_template": "<NUM_A> and <NUM_B> but calm.", "numeric_values": {"NUM_A": "7", "NUM_B": "12"}, "extra": true}'])
def test_json_and_schema_failures_receive_only_one_same_schema_repair(bad):
    task, result, config, ledger, _, valid = scenario()
    models = Models([{"issues": []}, bad, valid])
    refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == 3 and audit["status"] == "completed_with_component_failure"
    assert audit["calls"][1]["response"] == bad and audit["calls"][1]["response_hash"] == digest(bad)
    assert audit["component_failures"][0]["validation_errors"]
    assert ledger.snapshot()["tokens"] == 150 and ledger.snapshot()["reserved_tokens"] == 0


def test_default_zero_retains_original_two_call_failure_contract():
    assert MASConfig().public_construction_validation_retries == 0
    task, result, config, ledger, invalid, valid = scenario(retries=0)
    models = Models([{"issues": []}, invalid, valid])
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["status"] == "failed" and audit["revision"] is None
    assert "public_construction_validation_repair" not in audit and "component_failures" not in audit
    assert result.answer == audit["draft"] and ledger.snapshot()["tokens"] == 100


def test_valid_typed_revision_never_spends_optional_repair_or_selects_another_candidate():
    task, result, config, ledger, _, valid = scenario()
    models = Models([{"issues": []}, valid, {"answer": "UNAUTHORIZED_CANDIDATE"}])
    refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["status"] == "completed" and "component_failures" not in audit
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0
    assert audit["selected_candidate"] == "revision" and audit["revision"] == valid


def test_revision_request_parameters_stay_identical_across_repair_except_remaining_timeout():
    task, result, config, ledger, invalid, valid = scenario()
    config.public_revision_frequency_penalty = 0.4
    models = Models([{"issues": []}, invalid, valid])
    refine_public_answer(task, result, models, ledger, config)
    first, second = [dict(request["kwargs"]) for request in models.requests[1:]]
    assert first.pop("timeout") > 0 and second.pop("timeout") > 0
    assert first == second and first["frequency_penalty"] == 0.4
    assert "frequency_penalty" not in models.requests[0]["kwargs"]
    assert all(call["model_call_options_hash"] == digest(call["model_call_options"])
               for call in result.metadata["public_refinement"]["calls"])


def test_second_invalid_typed_response_fails_without_third_revision_or_initial_fallback():
    task, result, config, ledger, invalid, valid = scenario()
    models = Models([{"issues": []}, invalid, invalid, valid])
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 3
    assert [row["attempt"] for row in audit["component_failures"]] == [1, 2]
    assert audit["status"] == "failed" and audit["revision"] is None
    assert result.answer == audit["draft"] and "selected_candidate" not in audit
    assert ledger.snapshot()["tokens"] == 150 and ledger.snapshot()["reserved_tokens"] == 0


@pytest.mark.parametrize("stage,exception", [("review", ValueError("bad transport")),
    ("review", TimeoutError("transport timeout")), ("revision", ValueError("bad transport")),
    ("revision", TimeoutError("transport timeout"))])
def test_provider_errors_are_not_local_validation_repairs(stage, exception):
    task, result, config, ledger, _, valid = scenario()
    models = Models(([{"issues": []}] if stage == "revision" else []) + [exception, valid])
    with pytest.raises(type(exception)):
        refine_public_answer(task, result, models, ledger, config)
    expected_calls = 2 if stage == "revision" else 1
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == expected_calls
    assert audit["calls"][-1]["failure_phase"] == "provider"
    assert audit["status"] == "failed" and audit["public_construction_validation_repair"]["repair_attempts"] == 0
    assert ledger.snapshot()["tokens"] > 0 and ledger.snapshot()["reserved_tokens"] == 0


def test_invalid_review_is_charged_once_and_never_repaired():
    task, result, config, ledger, _, valid = scenario()
    models = Models(['{"issues": [], "issues": []}', {"issues": []}, valid])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert audit["calls"][-1]["failure_phase"] == "response_validation"
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0 and audit["review"] is None


@pytest.mark.parametrize("kind", ["position", "template"])
def test_renderer_error_after_validation_is_not_repaired(kind, monkeypatch):
    task, result, config, ledger, _, valid = scenario(kind)
    module = __import__("jit_mas.public_word_slots" if kind == "position" else
                        "jit_mas.public_numeric_template", fromlist=["render"])
    def fail_render(_):
        raise ValueError("synthetic renderer failure")
    monkeypatch.setattr(module, "render", fail_render)
    models = Models([{"issues": []}, valid, valid])
    with pytest.raises(ValueError, match="renderer failure"):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["calls"][-1]["status"] == "validated" and audit["status"] == "failed"
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0


def test_shared_call_budget_prevents_repair_without_bypassing_caps():
    task, result, config, _, invalid, valid = scenario()
    ledger = BudgetLedger(2, 2_000_000, 0, timeout_seconds=900)
    models = Models([{"issues": []}, invalid, valid])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit["calls"][-1]["attempt"] == 2 and audit["calls"][-1]["failure_phase"] == "provider"
    assert audit["status"] == "failed" and result.answer == audit["draft"]


def test_expired_shared_deadline_stops_before_repair(monkeypatch):
    task, result, config, ledger, invalid, valid = scenario()
    models = Models([{"issues": []}, invalid, valid], hook=lambda count:
        monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0) if count == 2 else None)
    with pytest.raises(TimeoutError, match="before construction repair"):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0


def test_deadline_expiring_after_valid_response_does_not_trigger_validation_repair(monkeypatch):
    task, result, config, ledger, _, valid = scenario()
    models = Models([{"issues": []}, valid, valid], hook=lambda count:
        monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0) if count == 2 else None)
    with pytest.raises(TimeoutError, match="during public refinement"):
        refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["calls"][-1]["failure_phase"] == "budget_deadline"
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0


@pytest.mark.parametrize("mode", ["ordinary", "patch", "unsupported", "mixed"])
def test_non_typed_revisions_do_not_receive_construction_repair(mode):
    task, result, config, ledger, _, valid = scenario()
    if mode in ["ordinary", "patch"]:
        config.public_numeric_construction = False
        task.question = "Write a compact notice."
        if mode == "patch":
            config.public_revision_mode = "patch"
    elif mode == "unsupported":
        task.question = "Write a compact notice. Use approximately two numeric values."
    else:
        config.public_positional_construction = True
        task.question += " Include keyword Amber in the second sentence, as the third word of that sentence."
    models = Models([{"issues": []}, "{", valid])
    refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["public_construction_validation_repair"]["active"] is False
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0
    assert audit["selected_candidate"] == "initial_draft" and audit["selection_reason"] == "invalid_local_revision_response"


def test_initial_generation_failure_does_not_enter_review_or_repair():
    task, result, config, ledger, _, valid = scenario()
    result.answer, result.terminated_reason = None, "model_error"
    models = Models([{"issues": []}, valid])
    with pytest.raises(ValueError, match="valid final-answer draft"):
        refine_public_answer(task, result, models, ledger, config)
    assert not models.requests and ledger.snapshot()["model_calls"] == 0


def test_patch_application_failure_keeps_existing_guard_without_construction_repair(monkeypatch):
    task, result, config, ledger, _, _ = scenario()
    task.question = "Write a compact notice."
    config.public_numeric_construction = False
    config.public_revision_mode = "patch"
    from jit_mas import public_text_patches
    def fail_patch(*_):
        raise ValueError("synthetic patch application failure")
    monkeypatch.setattr(public_text_patches, "apply_public_text_patches", fail_patch)
    models = Models([{"issues": []}, {"edits": []}, {"edits": []}])
    refine_public_answer(task, result, models, ledger, config)
    audit = result.metadata["public_refinement"]
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit["selection_reason"] == "invalid_public_patch_application"
    assert audit["public_construction_validation_repair"]["repair_attempts"] == 0


def test_audit_writer_failure_is_not_misclassified_as_local_response_validation():
    task, result, config, ledger, invalid, valid = scenario()
    models = Models([{"issues": []}, invalid, valid])
    def broken_writer(audit):
        if audit["calls"] and audit["calls"][-1]["status"] == "failed":
            raise ValueError("synthetic artifact write failure")
    with pytest.raises(ValueError, match="artifact write failure"):
        refine_public_answer(task, result, models, ledger, config, audit_writer=broken_writer)
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert result.metadata["public_refinement"]["public_construction_validation_repair"]["repair_attempts"] == 0


@pytest.mark.parametrize("template,missing,duplicates", [
    ("<NUM_A> <NUM_A> and calm but waiting.", ["<NUM_B>"], ["<NUM_A>"]),
    ("<NUM_A> and <NUM_B> but <NUM_Q>.", [], []),
])
def test_template_cross_field_diagnostics_do_not_rewrite_raw_response(template, missing, duplicates):
    task, result, config, ledger, invalid, valid = scenario()
    invalid["answer_template"] = template
    raw = json.dumps(invalid)
    models = Models([{"issues": []}, raw, valid])
    refine_public_answer(task, result, models, ledger, config)
    payload = json.loads(models.requests[-1]["messages"][-1]["content"])["public_construction_validation_repair"]
    assert payload["previous_response"] == raw
    assert payload["marker_counts"]["missing_markers"] == missing
    assert payload["marker_counts"]["duplicate_markers"] == duplicates
    assert result.metadata["public_refinement"]["calls"][1]["response"] == raw
