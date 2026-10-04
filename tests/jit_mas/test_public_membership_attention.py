"""Synthetic public contradiction attention; no private answer-set oracle."""

import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.evidence import EVIDENCE_QUESTION_HEADER, EVIDENCE_SCOPE, EVIDENCE_TASK_CONSTRAINT
from jit_mas.public_membership import (
    MAX_ATTENTION_CHECKS, build_public_membership_observations,
    public_membership_attention_queue, public_membership_model_input,
)
from jit_mas.public_refinement import (
    PublicReview, _membership_attention_review_schema, _membership_attention_review_links,
    _strict_json, refine_public_answer,
)
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


def task_fixture(*, question="For 2023, list entities whose score is at least 2.8 points.",
                 text="score uses points.\nentity,score\nBirch,2.7\nCedar,2.8\nFir,unknown",
                 date="2023 cohort", constraints=None):
    task = PublicTask(task_id="PRIVATE_ROUTE_CANARY", question=question,
                      constraints=list(constraints or []))
    if text is None:
        return task
    sha = hashlib.sha256(text.encode()).hexdigest()
    source = {"source_id": "web0", "source_sha256": sha, "date": date, "status": "ok",
              "kind": "provided_table", "title": "Supplied cohort table",
              "locator": "https://example.org/table", "retrieved_at": "2099-12-01",
              "truncated": False,
              "segments": [{"text": text, "source_sha256": sha,
                            "span_start": 0, "span_end": len(text)}]}
    pack = {"task_id": task.task_id, "scope": EVIDENCE_SCOPE, "sources": [source]}
    task.question += EVIDENCE_QUESTION_HEADER + json.dumps(pack, sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    task.constraints.append(EVIDENCE_TASK_CONSTRAINT)
    return task


def observations(task=None, draft="Birch and Cedar qualify. Fir remains uncertain."):
    task = task or task_fixture()
    return build_public_membership_observations(task, {
        "contributions": [], "private_evaluation": "PRIVATE_SCORE_CANARY"}, draft)


def queue_fixture(draft="Birch and Cedar qualify.", **kwargs):
    task = task_fixture(**kwargs)
    rich = observations(task, draft)
    return task, rich, public_membership_attention_queue(rich, draft)


def reviewed(queue, draft, *, scope="matches", use="affirmative_inclusion"):
    checks, issues = [], []
    for item in queue["items"]:
        task_quote = item["original_condition_span"]["text"].strip()
        source_quote = item["observation"]["cell_span"]["text"]
        unknown = scope == "unknown" or use == "unknown"
        issue = scope == "matches" and use == "affirmative_inclusion"
        index = len(issues) if issue else None
        if issue:
            issues.append({"defect": "A claimed member fails a supplied numeric condition.",
                "public_basis": f"{item['attention_id']} {item['source']['source_id']} "
                    f"[TASK_QUOTE]{task_quote}[/TASK_QUOTE] cell={source_quote}; draft={draft}",
                "repair": "Recheck original scope and repair the unsupported inclusion."})
        checks.append({"attention_id": item["attention_id"], "original_scope": scope,
            "draft_use": use, "disposition": "issue" if issue else "unknown" if unknown else "no_issue",
            "task_quote": task_quote, "source_quote": source_quote,
            "draft_quote": item["draft_literal_span"]["text"],
            "issue_index": index,
            "reason": "Judgment based on the original request and actual draft context."})
    return {"issues": issues, "attention_checks": checks}


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return ChatMessage(role="assistant", content=response if isinstance(response, str) else json.dumps(response))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def config(**kwargs):
    return MASConfig(backend="scripted", public_refinement=True,
        public_membership_observations=True, public_membership_input_format="compact",
        public_membership_attention_checks=True, public_refinement_response_format="json_schema_review",
        models={"exec": ModelConfig(max_tokens=8192), "global": ModelConfig(max_tokens=16000)}, **kwargs)


def result_fixture(draft):
    return RunResult(answer=draft, terminated_reason="final_answer", metadata={
        "private_feedback": "PRIVATE_SCORE_CANARY", "call_counts": {"writer": 1}})


def test_default_off_and_required_public_dependencies():
    assert MASConfig().public_membership_attention_checks is False
    for options in ({}, {"public_refinement": True}, {"public_membership_observations": True}):
        with pytest.raises(ValidationError, match="attention checks require"):
            MASConfig(public_membership_attention_checks=True, **options)


def test_queue_preserves_source_condition_cell_and_draft_anchors_without_membership_upgrade():
    draft = "Birch and Cedar qualify. Fir remains uncertain."
    task, rich, queue = queue_fixture(draft)
    frozen = copy.deepcopy(rich)
    item, = queue["items"]
    assert item["attention_id"] == "T001.R001." + rich["numeric_thresholds"][0]["threshold_id"]
    assert item["observation"]["arithmetic_truth"] is False
    assert item["observation"]["observed_value"] == "2.7"
    assert item["binding"]["operator"] == ">=" and item["binding"]["bound"] == "2.8"
    assert item["binding"]["unit"] == "points"
    assert item["source_scope"]["declared_source_scope"] == "2023 cohort"
    assert item["original_condition_span"]["text_hash"] == digest(item["original_condition_span"]["text"])
    assert item["numeric_bound_span"]["text"] == "at least 2.8 points"
    assert item["numeric_bound_span"]["text"] in item["original_condition_span"]["text"]
    assert item["draft_literal_span"]["text"] == "Birch"
    context = item["draft_context_span"]
    assert draft[context["start"]:context["end"]] == context["text"]
    assert context["text_hash"] == digest(context["text"])
    assert item["membership_status"] == "unknown"
    assert queue["full_observation_hash"] == digest(rich) and queue["draft_hash"] == digest(draft)
    assert queue["coverage"] == {"total_eligible_checks": 1, "queued_checks": 1,
        "maximum_queued_checks": MAX_ATTENTION_CHECKS, "truncated": False,
        "unqueued_attention_ids": [], "full_matrix_retained": True, "complete_membership_verified": False}
    assert rich == frozen
    assert "PRIVATE_ROUTE_CANARY" not in json.dumps(queue)
    assert "PRIVATE_SCORE_CANARY" not in json.dumps(queue)
    # Row TRUE and UNKNOWN remain in the full matrix, never enter the FALSE queue.
    assert [r[4] for r in public_membership_model_input(rich, "compact")["table_observations"][0]["rows"]] == [[False], [True], [None]]


@pytest.mark.parametrize("draft", ["Birch qualifies.", 'The note says "Birch qualifies".',
                                   "Birch is excluded.", "BIRCH is mentioned without a decision."])
def test_mentions_queue_even_quotation_or_exclusion_without_interpreting_them(draft):
    _, _, queue = queue_fixture(draft)
    assert len(queue["items"]) == 1
    assert queue["items"][0]["membership_status"] == "unknown"


@pytest.mark.parametrize("draft,kwargs", [
    ("Birchwood and Cedar qualify.", {}), ("Cedar qualifies.", {}),
    ("Birch qualifies.", {"date": "2024 cohort"}),
    ("Birch qualifies.", {"text": "entity,score\nBirch,2.7"}),
    ("Birch qualifies.", {"text": None}),
    ("Birch qualifies.", {"question": "For 2023, list entities whose score is at least 2.8 km."}),
])
def test_unknown_unbound_missing_or_nonliteral_does_not_create_attention(draft, kwargs):
    assert not queue_fixture(draft, **kwargs)[2]["items"]


def test_queue_is_capped_in_original_source_order_with_all_omitted_ids_recorded():
    names = [f"Entity{i:03}" for i in range(MAX_ATTENTION_CHECKS + 3)]
    text = "score uses points.\nentity,score\n" + "\n".join(f"{name},2.7" for name in names)
    _, rich, queue = queue_fixture(", ".join(reversed(names)), text=text)
    assert [item["entity"] for item in queue["items"]] == names[:MAX_ATTENTION_CHECKS]
    assert queue["coverage"]["total_eligible_checks"] == len(names)
    assert queue["coverage"]["truncated"] is True
    assert len(queue["coverage"]["unqueued_attention_ids"]) == 3
    assert len(public_membership_model_input(rich, "compact")["table_observations"][0]["rows"]) == len(names)


def test_stale_draft_cannot_reuse_the_observation_identity():
    _, rich, _ = queue_fixture()
    with pytest.raises(ValueError, match="original observed draft"):
        public_membership_attention_queue(rich, "Birch is excluded.")


def test_protocol_accepts_valid_issue_and_multiple_checks_can_share_a_source_backed_issue():
    draft = "Birch qualifies."
    _, _, queue = queue_fixture(draft,
        question="For 2023, list entities satisfying all conditions:\n1. Score is at least 2.8 points.\n2. Income is at least 4 USD.",
        text="score uses points. income uses USD.\nentity,score,income\nBirch,2.7,3")
    assert len(queue["items"]) == 2
    review = reviewed(queue, draft)
    review["issues"][0]["public_basis"] += " " + review["issues"][1]["public_basis"]
    review["issues"] = review["issues"][:1]
    for check in review["attention_checks"]:
        check["issue_index"] = 0
    parsed = _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))
    assert len(parsed.issues) == 1 and len(parsed.attention_checks) == 2


@pytest.mark.parametrize("scope,use,disposition", [
    ("matches", "quotation", "no_issue"), ("matches", "exclusion", "no_issue"),
    ("does_not_match", "affirmative_inclusion", "no_issue"),
    ("unknown", "affirmative_inclusion", "unknown"), ("matches", "unknown", "unknown"),
    ("does_not_match", "unknown", "unknown"),
])
def test_unknown_scope_and_nonaffirmative_use_do_not_force_an_issue(scope, use, disposition):
    draft = "Birch is mentioned."
    _, _, queue = queue_fixture(draft)
    review = reviewed(queue, draft, scope=scope, use=use)
    parsed = _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))
    assert not parsed.issues and parsed.attention_checks[0].disposition == disposition


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "invented_id", "extra_field", "blank_reason", "false_task_quote",
    "false_source_quote", "false_draft_quote", "missing_entity_quote", "missing_issue", "bad_index",
    "missing_task_basis", "missing_draft_basis", "misclassified_disposition",
])
def test_material_protocol_failures_are_rejected(change):
    draft = "Birch qualifies."
    _, _, queue = queue_fixture(draft)
    review = reviewed(queue, draft)
    check = review["attention_checks"][0]
    if change == "missing":
        review["attention_checks"] = []
    elif change == "duplicate":
        review["attention_checks"].append(copy.deepcopy(check))
    elif change == "invented_id":
        check["attention_id"] = "fabricated"
    elif change == "extra_field":
        check["private_score"] = 1
    elif change == "blank_reason":
        check["reason"] = " "
    elif change == "false_task_quote":
        check["task_quote"] = "Invented nationality condition"
    elif change == "false_source_quote":
        check["source_quote"] = "9.9"
    elif change == "false_draft_quote":
        check["draft_quote"] = "Birch is absent."
    elif change == "missing_entity_quote":
        check["draft_quote"] = "qualifies"
    elif change == "missing_issue":
        review["issues"] = []
    elif change == "bad_index":
        check["issue_index"] = 7
    elif change == "missing_task_basis":
        review["issues"][0]["public_basis"] = review["issues"][0]["public_basis"].replace(check["task_quote"], "other")
    elif change == "missing_draft_basis":
        review["issues"][0]["public_basis"] = review["issues"][0]["public_basis"].replace(check["draft_quote"], "other")
    else:
        check["disposition"] = "no_issue"
    with pytest.raises(ValidationError):
        _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))


def test_dynamic_schema_rejects_same_length_duplicate_coverage():
    draft = "Birch qualifies."
    _, _, queue = queue_fixture(draft,
        question="For 2023, list entities satisfying all conditions:\n1. Score is at least 2.8 points.\n2. Income is at least 4 USD.",
        text="score uses points. income uses USD.\nentity,score,income\nBirch,2.7,3")
    review = reviewed(queue, draft)
    review["attention_checks"][1] = copy.deepcopy(review["attention_checks"][0])
    with pytest.raises(ValidationError, match="exactly once"):
        _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))


def test_quote_presence_is_only_protocol_validation_not_semantic_scope_certification():
    draft = "Birch is excluded."
    _, _, queue = queue_fixture(draft)
    # The local validator cannot certify a model's mistaken affirmative judgment.
    parsed = _strict_json(json.dumps(reviewed(queue, draft)), _membership_attention_review_schema(queue, draft))
    assert parsed.attention_checks[0].draft_use == "affirmative_inclusion"
    assert queue["items"][0]["membership_status"] == "unknown"
    assert "quote presence does not prove" in queue["limitations"]


def test_active_attention_precedes_matrix_and_preserves_two_call_cost_raw_audit_and_rich_hash():
    draft = "Birch and Cedar qualify."
    task, rich, queue = queue_fixture(draft)
    review = reviewed(queue, draft)
    models = Models([review, {"answer": "Cedar qualifies."}])
    result = result_fixture(draft)
    ledger = BudgetLedger(None, 2_000_000, 0)
    snapshots = []
    refine_public_answer(task, result, models, ledger, config(), audit_writer=snapshots.append)
    assert result.answer == "Cedar qualifies."
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    payload = json.loads(models.requests[0]["messages"][1]["content"])
    assert list(payload)[:2] == ["phase", "public_membership_attention"]
    assert payload["public_membership_attention"]["full_observation_hash"] == digest(rich)
    assert len(payload["public_membership_observations"]["table_observations"][0]["rows"]) == 3
    schema = models.requests[0]["kwargs"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["attention_checks"]["minItems"] == schema["properties"]["attention_checks"]["maxItems"] == 1
    assert models.requests[1]["kwargs"]["response_format"] == {"type": "json_object"}
    serialized = json.dumps(models.requests)
    assert "PRIVATE_SCORE_CANARY" not in serialized
    assert "task_id" not in payload["public_task"]
    # The original canonical public question includes its supplied pack's ID;
    # derived attention and observations must not expose a routing field.
    assert "PRIVATE_ROUTE_CANARY" not in json.dumps(payload["public_membership_attention"])
    assert "PRIVATE_ROUTE_CANARY" not in json.dumps(payload["public_membership_observations"])
    audit = result.metadata["public_refinement"]
    assert audit["public_membership_input_audit"]["observations"] == rich
    assert audit["public_membership_attention_checks"]["queue_hash"] == digest(queue)
    assert audit["review"] == review and audit["calls"][0]["parsed"] == review
    links = audit["public_membership_attention_checks"]["review_links"]
    assert links["quote_link_presence_validated"] is True
    assert links["semantic_scope_verified"] is False
    assert links["links"][0]["attention_id"] == queue["items"][0]["attention_id"]
    assert links["links"][0]["issue_index"] == 0
    assert links["links"][0]["source"]["source_id"] == "web0"
    assert audit["calls"][0]["response"] == json.dumps(review)
    assert audit == snapshots[-1] and audit["status"] == "completed"


@pytest.mark.parametrize("enabled,table,draft", [
    (False, True, "Birch qualifies."), (True, False, "An ordinary scene."),
    (True, True, "Cedar qualifies."),
])
def test_disabled_non_table_or_empty_queue_keeps_ordinary_review_schema(enabled, table, draft):
    task = task_fixture(text=None if not table else "score uses points.\nentity,score\nBirch,2.7\nCedar,2.8")
    cfg = config()
    cfg.public_membership_attention_checks = enabled
    models = Models([{"issues": []}, {"answer": draft}])
    result = result_fixture(draft)
    ledger = BudgetLedger(None, 2_000_000, 0)
    refine_public_answer(task, result, models, ledger, cfg)
    assert ledger.snapshot()["model_calls"] == 2 and result.answer == draft
    assert "public_membership_attention" not in json.loads(models.requests[0]["messages"][1]["content"])
    assert "PUBLIC ATTENTION CHECKS FIRST" not in models.requests[0]["messages"][0]["content"]
    assert models.requests[0]["kwargs"]["response_format"]["json_schema"]["schema"] == PublicReview.model_json_schema()


@pytest.mark.parametrize("scope,use,draft", [
    ("matches", "quotation", 'The supplied comment says "Birch qualifies".'),
    ("matches", "exclusion", "Birch is excluded. Cedar qualifies."),
    ("unknown", "affirmative_inclusion", "Birch qualifies subject to an uncertain source scope."),
])
def test_attention_does_not_automatically_delete_a_quotation_exclusion_or_unknown(scope, use, draft):
    task, _, queue = queue_fixture(draft)
    models = Models([reviewed(queue, draft, scope=scope, use=use), {"answer": draft}])
    result = result_fixture(draft)
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config())
    assert result.answer == draft and len(models.requests) == 2


@pytest.mark.parametrize("typed", [False, True])
def test_failed_attention_review_is_charged_not_retried_or_guarded_even_with_typed_repair(typed):
    task = task_fixture(constraints=["Include keyword softly as the third word in the second sentence."] if typed else [])
    draft = "Birch qualifies. We speak softly today."
    cfg = config(public_refinement_guard=True, public_positional_construction=typed,
                 public_construction_validation_retries=1)
    models = Models([{"issues": []}, {"answer": "UNAUTHORIZED_RETRY"}])
    result = result_fixture(draft)
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(ValidationError):
        refine_public_answer(task, result, models, ledger, cfg)
    audit = result.metadata["public_refinement"]
    assert audit["status"] == "failed" and audit["review"] is None and audit["revision"] is None
    assert audit["calls"][0]["failure_phase"] == "response_validation"
    assert audit["calls"][0]["response"] == '{"issues": []}'
    assert result.answer == draft and len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["tokens"] == 50


def test_provider_failure_keeps_identity_and_one_failed_metered_attempt():
    task = task_fixture()
    result = result_fixture("Birch qualifies.")
    failure = RuntimeError("Synthetic provider identity mismatch")
    models = Models([failure, {"issues": []}])
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(RuntimeError) as caught:
        refine_public_answer(task, result, models, ledger, config(public_refinement_guard=True))
    assert caught.value is failure and len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert result.metadata["public_refinement"]["calls"][0]["failure_phase"] == "provider"


def test_attention_does_not_bypass_the_shared_call_budget():
    draft = "Birch qualifies."
    task, _, queue = queue_fixture(draft)
    result = result_fixture(draft)
    models = Models([reviewed(queue, draft), {"answer": "A forbidden extra call."}])
    ledger = BudgetLedger(1, 2_000_000, 0)
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, config())
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 1
    assert result.metadata["public_refinement"]["status"] == "failed"
    assert result.metadata["public_refinement"]["review"] is not None


@pytest.mark.parametrize("input_format", ["full", "compact"])
def test_attention_transport_uses_rich_basis_in_both_existing_input_formats(input_format):
    draft = "Birch qualifies."
    task, rich, queue = queue_fixture(draft)
    result = result_fixture(draft)
    cfg = config()
    cfg.public_membership_input_format = input_format
    cfg.public_refinement_response_format = "json_object"
    models = Models([reviewed(queue, draft), {"answer": "No unsupported inclusion."}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), cfg)
    audit = result.metadata["public_refinement"]
    assert audit["public_membership_attention_checks"]["queue"] == queue
    assert audit["public_membership_attention_checks"]["queue"]["full_observation_hash"] == digest(rich)
    assert models.requests[0]["kwargs"]["response_format"] == {"type": "json_object"}
    assert "attention_checks" in models.requests[0]["messages"][0]["content"]
    assert audit["public_input_hash"] == digest(audit["public_input"])
    if input_format == "full":
        assert audit["public_input"]["public_membership_observations"] == rich
    else:
        assert audit["public_membership_input_audit"]["observations"] == rich


def test_ordinary_invalid_revision_keeps_existing_guard_and_all_attention_call_costs():
    draft = "Birch qualifies."
    task, _, queue = queue_fixture(draft)
    result = result_fixture(draft)
    models = Models([reviewed(queue, draft), "not valid JSON"])
    ledger = BudgetLedger(None, 2_000_000, 0)
    refine_public_answer(task, result, models, ledger, config(public_refinement_guard=True))
    audit = result.metadata["public_refinement"]
    assert result.answer == draft and audit["status"] == "completed_with_component_failure"
    assert audit["selection_reason"] == "invalid_local_revision_response"
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is True
    assert audit["review"]["attention_checks"][0]["disposition"] == "issue"
    assert audit["calls"][1]["response"] == "not valid JSON"
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["tokens"] == 100
    # Retained initial proves engineering termination only, not membership.
    assert audit["public_membership_attention_checks"]["queue"]["items"][0]["membership_status"] == "unknown"


@pytest.mark.parametrize("use,reversed_rows,task_quote", [
    ("affirmative_inclusion", False, "score is at least 2.8 points"),
    ("exclusion", False, "at least 2.8 points"),
    ("affirmative_inclusion", True, "list entities whose score is at least 2.8 points"),
])
def test_three_anonymous_live_response_shapes_accept_complete_condition_and_row_quotes(
        use, reversed_rows, task_quote):
    # Generic reworded fixtures reproduce only the failed response shapes:
    # a full-condition quotation, a bare-bound quotation, and reordered rows.
    text = "score uses points.\nentity,score\n" + (
        "Cedar,2.8\nBirch,2.7" if reversed_rows else "Birch,2.7\nCedar,2.8")
    draft = "Birch qualifies." if use == "affirmative_inclusion" else "Birch is excluded. Cedar qualifies."
    _, _, queue = queue_fixture(draft, text=text)
    review = reviewed(queue, draft, use=use)
    check = review["attention_checks"][0]
    check["task_quote"] = task_quote
    check["source_quote"] = "Birch,2.7"
    if review["issues"]:
        review["issues"][0]["public_basis"] = (
            f"[TASK_QUOTE]{task_quote}[/TASK_QUOTE]; supplied row Birch,2.7; draft {draft}")
    parsed = _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))
    links = _membership_attention_review_links(queue, parsed.model_dump(mode="json"), draft)
    assert links["quote_link_presence_validated"] is True
    link, = links["links"]
    cell = queue["items"][0]["observation"]["cell_span"]
    assert link["source_quote_span"]["start"] <= cell["start"]
    assert link["source_quote_span"]["end"] >= cell["end"]
    assert link["source"]["source_id"] == "web0"
    assert link["issue_index"] == (0 if use == "affirmative_inclusion" else None)
    assert link["semantic_membership_verified"] is False


@pytest.mark.parametrize("source_quote", [
    "Cedar,2.7,9.9",  # same observed number from another entity's actual row
    "9.9",          # exact text from the right row but the wrong numeric cell
    "Birch has 2.7", # contains the number but is not exact source-row text
])
def test_source_quote_must_cover_the_correct_cell_in_its_exact_row(source_quote):
    draft = "Birch qualifies."
    _, _, queue = queue_fixture(draft,
        text="score uses points. income uses USD.\nentity,score,income\nBirch,2.7,9.9\nCedar,2.7,9.9")
    review = reviewed(queue, draft)
    review["attention_checks"][0]["source_quote"] = source_quote
    review["issues"][0]["public_basis"] += " " + source_quote
    with pytest.raises(ValidationError, match="observed cell span"):
        _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))


def test_quote_from_another_public_condition_is_not_this_queue_items_basis():
    draft = "Birch qualifies."
    _, _, queue = queue_fixture(draft, constraints=["Use plain language."])
    review = reviewed(queue, draft)
    review["attention_checks"][0]["task_quote"] = "Use plain language."
    review["issues"][0]["public_basis"] += " Use plain language."
    with pytest.raises(ValidationError, match="original public condition"):
        _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))


def test_markdown_exclusion_shape_copies_the_exact_literal_and_keeps_reading_context():
    draft = "**Birch**: excluded because the supplied score fails.\n**Cedar** qualifies."
    task, _, queue = queue_fixture(draft)
    item, = queue["items"]
    assert item["draft_literal_span"]["text"] == "Birch"
    assert item["draft_context_span"]["text"] == draft.splitlines()[0]
    review = reviewed(queue, draft, use="exclusion")
    assert review["attention_checks"][0]["draft_quote"] == "Birch"
    models = Models([review, {"answer": draft}])
    result = result_fixture(draft)
    ledger = BudgetLedger(None, 2_000_000, 0)
    refine_public_answer(task, result, models, ledger, config())
    assert result.answer == draft and not result.metadata["public_refinement"]["review"]["issues"]
    assert ledger.snapshot()["model_calls"] == 2
    schema = models.requests[0]["kwargs"]["response_format"]["json_schema"]["schema"]
    quote_schema = schema["$defs"]["PublicMembershipAttentionItem"]["properties"]["draft_quote"]
    assert quote_schema.get("const") == "Birch"
    assert "COPY" in models.requests[0]["messages"][0]["content"]


@pytest.mark.parametrize("bad_quote", ["birch", "Birc", "**Birch**", "Birch: excluded"])
def test_case_deletion_markdown_or_paraphrase_cannot_replace_the_exact_located_literal(bad_quote):
    draft = "**Birch**: excluded."
    _, _, queue = queue_fixture(draft)
    review = reviewed(queue, draft, use="exclusion")
    review["attention_checks"][0]["draft_quote"] = bad_quote
    with pytest.raises(ValidationError):
        _strict_json(json.dumps(review), _membership_attention_review_schema(queue, draft))


def test_another_queued_items_valid_literal_cannot_be_bound_to_the_wrong_id():
    draft = "Birch and Cedar are excluded."
    _, _, queue = queue_fixture(draft,
        text="score uses points.\nentity,score\nBirch,2.7\nCedar,2.6")
    review = reviewed(queue, draft, use="exclusion")
    review["attention_checks"][0]["draft_quote"] = "Cedar"
    # Both literals are provider-schema options; the local item binding rejects
    # using a valid literal from a different queued row.
    schema = _membership_attention_review_schema(queue, draft)
    assert schema.model_json_schema()["$defs"]["PublicMembershipAttentionItem"]["properties"]["draft_quote"]["enum"] == ["Birch", "Cedar"]
    with pytest.raises(ValidationError, match="exact located literal"):
        _strict_json(json.dumps(review), schema)


def test_context_window_is_exact_bounded_and_does_not_certify_cross_line_scope():
    draft = "This is a quotation, not my selected set.\n" + "x" * 600 + " Birch " + "y" * 600
    _, _, queue = queue_fixture(draft)
    item, = queue["items"]
    context = item["draft_context_span"]
    assert context["text"] == draft[context["start"]:context["end"]]
    assert item["draft_context_truncated"] is True
    assert len(context["text"]) == 512 + len("Birch")
    assert item["membership_status"] == "unknown"
    assert "cross-line qualifications" in queue["draft_context_convention"]
