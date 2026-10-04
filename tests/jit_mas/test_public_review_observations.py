"""Public diagnostics and narrowly scoped preservation across real refinement."""
import copy
import json

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import refine_public_answer
from jit_mas.public_review_observations import (
    artifact_observations, empty_review_line_break_regression,
    numeric_relation_observations, review_basis_observations,
)
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


def task(question="Write a short scene.", constraints=None):
    return PublicTask(task_id="OPAQUE_PRIVATE_CANARY", question=question, constraints=constraints or [])


def test_complete_artifact_han_and_layout_counts_do_not_invent_words_or_body_scope():
    answer = "甲乙 A。\n\n丙B\n"
    observed = artifact_observations(answer)
    assert observed["han_ideograph_characters"] == 3
    assert observed["nonempty_lines"] == 2
    assert observed["paragraphs_separated_by_blank_lines"] == 2
    assert observed["line_break_events"] == 3
    assert observed["artifact_hash"] == digest(answer)
    assert "words or tokens" in observed["counting_basis"]
    assert "body-only" in observed["counting_basis"]
    assert artifact_observations("한かなABC12")['han_ideograph_characters'] == 0
    assert artifact_observations(42)["status"] == "unknown"


@pytest.mark.parametrize("expression,truth", [
    ("2.7 >= 2.8", False), ("2.7 < 2.8", True), ("2.80 = 2.8", True),
    ("-3 > -4", True), ("+3 <= 3.000", True), ("4 ≥ 5", False),
    ("4 ≤ 4", True), ("4 != 5", True), ("4 ≠ 4", False),
    ("0.10000000000000000000000000000001 > 0.1", True),
    ("12345678901234567890123456789012 > 12345678901234567890123456789011", True),
    ("2 < 3.", True),
    ("2.7 >= 2.8, so the condition is false.", False),
    ("2 < 3,so retain this result.", True),
    ("2 < 3; another observation follows.", True),
    ("2 < 3)", True),
])
def test_printed_decimal_relations_are_exact_and_scoped_to_the_printed_span(expression, truth):
    text = "Observed comparison: " + expression
    observations = numeric_relation_observations({"public_basis": text})
    row, = observations["comparisons"]
    assert row['arithmetic_truth'] is truth
    assert text[row["start"]:row["end"]] == row["expression"]
    assert row["path"] == ["public_basis"] and row["text_hash"] == digest(text)
    assert observations["false_arithmetic_relations"] == (0 if truth else 1)
    assert not observations["truncated"]


@pytest.mark.parametrize("text", [
    "1,234 > 2", "2 > 1,234", "2e3 > 1", "2 > 1e3", "x2 > 1",
    "2 > 1.2.3", ".5 > .1", "−2 > −3", "½ > 0", "true > false",
    "9" * 33 + " > 1", "2 > " + "1" * 33,
    "1/2 > 1/3", "1 / 2 > 1 / 3", "3*2 > 1+4", "3 * 2 > 1 + 4",
    "3^2 > 2^2", "3 ^ 2 > 2 ^ 2", "40% > 20%", "40 > 20 %",
    "1:2 > 1:3", "1 : 2 > 1 : 3",
    "3 +2 > 4", "3 -2 < 0", "(3) +2 > 4",
    "1, 234 > 2", "2 > 1, 234",
])
def test_unsupported_numeric_syntax_cannot_silently_pass_from_an_ascii_suffix(text):
    assert numeric_relation_observations(text)["comparisons"] == []


def test_false_arithmetic_can_correctly_describe_failure_and_is_not_a_defect_certificate():
    observed = numeric_relation_observations("The condition 2.7 >= 2.8 is FALSE, so reject this row.")
    assert observed["false_arithmetic_relations"] == 1
    assert "not automatically a defect" in observed["limitations"]
    assert "membership are not verified" in observed["limitations"]


def test_overlapping_chain_and_bounded_observations_are_explicit():
    chain = numeric_relation_observations("1 < 2 < 3")
    assert [row["expression"] for row in chain["comparisons"]] == ["1 < 2", "2 < 3"]
    limited = numeric_relation_observations(["2 > 1"] * 130)
    assert len(limited["comparisons"]) == 128 and limited["truncated"]
    assert limited["comparisons"][-1]["path"] == [127]


def test_public_basis_quote_presence_is_separate_from_arithmetic_and_entailment():
    public = task("Include each supplier whose score is at least 2.8.")
    review = {"issues": [
        {"defect": "Wrong inclusion label", "public_basis": "[TASK_QUOTE]score is at least 2.8[/TASK_QUOTE]; 2.7 >= 2.8 is FALSE", "repair": "Recheck the supplier"},
        {"public_basis": "[TASK_QUOTE]Nationality must be verified[/TASK_QUOTE]"},
        {"public_basis": "Source-backed factual correction, without a task quote"},
    ]}
    observations = review_basis_observations(public, review)
    assert [row["quote_state"] for row in observations["issues"]] == [
        "exact_public_substring", "unverified", "unknown"]
    match = observations["issues"][0]["task_quote_observations"][0]["match"]
    assert public.question[match["start"]:match["end"]] == "score is at least 2.8"
    assert observations["numeric_relations"]["false_arithmetic_relations"] == 1
    assert "OPAQUE_PRIVATE_CANARY" not in json.dumps(observations)
    assert "does not prove" in observations["limitations"]


@pytest.mark.parametrize("question", [
    "Write in one paragraph.", "Return a single line.", "Write in 1 paragraph.",
    "Remove newlines.", "Return CSV.", "Copy verbatim.", "Reformat the text.",
    "写成一个自然段。", "不要换行。", "压缩成1段。", "写成一个段落。", "按以下格式输出。",
])
def test_layout_conversions_and_unknown_scopes_suppress_narrow_preservation(question):
    observations = empty_review_line_break_regression(task(question), "甲。\n\n乙。", "甲。乙。", {"issues": []})
    assert not observations["regression"] and observations["layout_scope"] == "unknown"


class Models:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "kwargs": kwargs}))
        return ChatMessage(role="assistant", content=json.dumps(next(self.responses)))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def refine(draft, revision, issues=None, *, guarded=True, question="Write a short scene."):
    result = RunResult(answer=draft, terminated_reason="final_answer", metadata={"private_feedback": "DO_NOT_EMIT_PRIVATE"})
    config = MASConfig(backend="scripted", public_refinement=True, public_refinement_guard=guarded,
                       public_refinement_response_format="json_schema",
                       models={"exec": ModelConfig(max_tokens=8192), "global": ModelConfig(max_tokens=16000)})
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    models = Models([{"issues": issues or []}, {"answer": revision}])
    snapshots = []
    refine_public_answer(task(question), result, models, ledger,
                         config, audit_writer=lambda value: snapshots.append(copy.deepcopy(value)))
    return result, result.metadata["public_refinement"], models, ledger, snapshots


def test_empty_review_pure_line_break_deletion_preserves_initial_with_all_calls_and_hashes():
    draft, revision = "天亮了。\r\n\r\n舟靠岸。", "天亮了。舟靠岸。"
    result, audit, models, ledger, snapshots = refine(draft, revision)
    assert result.answer == draft
    assert audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "empty_review_line_break_only_regression"
    assert audit["revision"]["answer"] == revision and audit["revision_answer_hash"] == digest(revision)
    assert audit["answer_hash"] == digest(draft)
    assert audit["public_candidate_guard"]["empty_review_line_break_guard"]["regression"]
    assert audit["audit_hash"] == digest({k: v for k, v in audit.items() if k != "audit_hash"})
    assert audit == snapshots[-1] and ledger.snapshot()["model_calls"] == len(models.requests) == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    for request in models.requests:
        assert "DO_NOT_EMIT_PRIVATE" not in json.dumps(request)
        assert "OPAQUE_PRIVATE_CANARY" not in json.dumps(request)


@pytest.mark.parametrize("draft,revision", [
    ("A.\n\nB.", "A. B."), ("A.\n\nB.", "A.B!"),
    ("A.\nB.", "A.B."), ("甲。\n\n乙。", "甲。\n乙。"),
])
def test_other_spacing_content_and_noncollapsed_layout_changes_are_not_selected(draft, revision):
    result, audit, _, _, _ = refine(draft, revision)
    assert result.answer == revision and audit["selected_candidate"] == "revision"
    assert not audit["public_candidate_guard"]["empty_review_line_break_guard"]["regression"]


def test_nonempty_validated_review_does_not_trigger_no_issue_layout_policy():
    issues = [{"defect": "Repair the ending", "public_basis": "Original scene purpose", "repair": "Use a clearer ending"}]
    result, audit, _, _, _ = refine("甲。\n\n乙。", "甲。乙。", issues)
    assert result.answer == "甲。乙。" and audit["selected_candidate"] == "revision"


def test_large_multparagraph_flattening_is_retained_as_initial_layout_guard():
    issues = [{"defect": "Repair a supported detail", "public_basis": "Original request", "repair": "Keep the complete artifact"}]
    draft = ("第一段保留完整事实和上下文。" * 80 + "\n\n"
             + "第二段保留比较、例子和限制。" * 80 + "\n\n"
             + "第三段保留行动建议和结论。" * 80)
    result, audit, _, _, _ = refine(draft, "改写后的单行答案。" * 150, issues)
    assert result.answer == draft
    assert audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "severe_public_layout_regression"
    guard = audit["public_candidate_guard"]["empty_review_line_break_guard"]
    assert guard["severe_regression"] is True


def test_default_off_keeps_legacy_payload_and_candidate_behavior():
    result, audit, models, _, _ = refine("甲。\n\n乙。", "甲。乙。", guarded=False)
    assert result.answer == "甲。乙。" and "public_candidate_guard" not in audit
    assert "public_review_basis_observations" not in audit
    for request in models.requests:
        payload = json.loads(request["messages"][1]["content"])
        assert "public_review_observations" not in payload
        assert "public_review_basis_observations" not in payload
        assert "Additional public observation protocol" not in request["messages"][0]["content"]


def test_exact_review_arithmetic_and_anchor_observations_reach_revision_without_schema_changes():
    issues = [{"defect": "Wrong comparison label", "public_basis": "[TASK_QUOTE]score is at least 2.8[/TASK_QUOTE]; 2.7 >= 2.8 is TRUE", "repair": "Recompute eligibility"}]
    _, audit, models, _, _ = refine("Score: 2.7.", "This supplier fails the threshold.", issues,
                                   question="Include a supplier whose score is at least 2.8.")
    payload = json.loads(models.requests[1]["messages"][1]["content"])
    observations = payload["public_review_basis_observations"]
    assert observations == audit["public_review_basis_observations"]
    assert observations["issues"][0]["quote_state"] == "exact_public_substring"
    assert observations["numeric_relations"]["comparisons"][0]["arithmetic_truth"] is False
    assert set(audit["review"]) == {"issues"} and set(audit["revision"]) == {"answer"}
