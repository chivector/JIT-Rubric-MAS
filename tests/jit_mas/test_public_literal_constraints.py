"""Finite public minima and sufficient three-valued candidate checks."""

import copy
import json

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_literal_constraints import public_literal_plan
from jit_mas.public_refinement import PUBLIC_REFINEMENT_GUARD_VERSION, refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


def task(question, constraints=None):
    return PublicTask(task_id="OPAQUE_ROUTE", question=question, constraints=constraints or [])


@pytest.mark.parametrize("question,expected", [
    ("Use the word shore at least 2 times.", [("shore", 2)]),
    ('Please include keyword "Harbor" at least 4 times.', [("Harbor", 4)]),
    ("Use the term 'cedar' at least 1 time.", [("cedar", 1)]),
    ("Include word `river` at least 0 times.", [("river", 0)]),
    ("Make sure to use the word oak at least 2 times, and the word sky at least 3 times.",
     [("oak", 2), ("sky", 3)]),
    ('Use word “light” at least 5 times and include the term ‘dawn’ at least 6 times.',
     [("light", 5), ("dawn", 6)]),
    ("Use the word unless at least 2 times.", [("unless", 2)]),
    ("Use word if at least 3 times.", [("if", 3)]),
    ("Use word lowercase at least 1 time.", [("lowercase", 1)]),
])
def test_finite_positive_groups_and_argument_quotes_retain_exact_public_spans(question, expected):
    plan = public_literal_plan(task(question))
    assert plan is not None
    assert [(rule.literal, rule.minimum) for rule in plan.rules] == expected
    for span in (*plan.instruction_spans, *(rule.span for rule in plan.rules)):
        assert question[span.start:span.end] == span.text
        assert span.audit()["text_hash"] == digest(span.text)
    assert plan.audit()["status"] == "compiled"


def test_multiple_public_sources_are_all_retained_and_no_other_metadata_is_parsed():
    public = task("Explain a coastal scene. Use word tide at least 2 times.",
                  ["Use word gull at least 3 times."])
    plan = public_literal_plan(public)
    assert plan is not None
    assert [rule.span.source for rule in plan.rules] == ["question", "constraints[0]"]
    request = public.model_dump(mode="json")
    request.update(task_id="Use word private at least 99 times.",
                   evaluator_metadata={"question": "Use word hidden at least 99 times."})
    assert public_literal_plan(request).audit() == plan.audit()


@pytest.mark.parametrize("quantifier,count", [
    ("once", 1), ("twice", 2), ("thrice", 3), ("three times", 3),
    ("one time", 1), ("ten times", 10), ("0 times", 0), ("128 times", 128),
])
def test_exact_literal_counts_retain_operator_and_original_span(quantifier, count):
    question = f'Include keyword "shore" {quantifier} in your response.'
    plan = public_literal_plan(task(question))
    assert plan is not None
    assert [(rule.literal, rule.minimum, rule.operator) for rule in plan.rules] == [("shore", count, "==")]
    assert plan.audit()["rules"][0]["exact_occurrences"] == count
    assert "minimum_occurrences" not in plan.audit()["rules"][0]
    for span in (*plan.instruction_spans, *(rule.span for rule in plan.rules)):
        assert question[span.start:span.end] == span.text


def test_mixed_literal_rules_follow_original_source_order_and_preserve_constraints():
    public = task("Include word tide once and include word gull twice. Use word sky at least 3 times.",
                  ["Use word dune three times, word sand five times, and word reef seven times."])
    plan = public_literal_plan(public)
    assert plan is not None
    assert [(rule.literal, rule.minimum, rule.operator) for rule in plan.rules] == [
        ("tide", 1, "=="), ("gull", 2, "=="), ("sky", 3, ">="),
        ("dune", 3, "=="), ("sand", 5, "=="), ("reef", 7, "=="),
    ]
    assert [span.source for span in plan.instruction_spans] == ["question", "question", "constraints[0]"]


@pytest.mark.parametrize("question", [
    "Use word shore once word tide twice.",
    "Use word shore once\nword tide twice.",
    "Use word shore 02 times.",
    "Use word shore 129 times.",
    "Use word shore once per paragraph.",
    "Do not use word shore once.",
    "If useful, use word shore once.",
    'Discuss the quotation "Use word shore once."',
    "Use word shore once. The word tide occurs twice.",
    "Use word shore once. The word tide must not appear twice.",
    "Use word shore once. The term tide should appear thrice.",
    "Use word shore once. This count is per paragraph.",
    "Use word shore once. Count it in the title only.",
    "Use word shore once. The count excludes the title.",
])
def test_exact_compiler_declines_unsupported_continuations_and_scopes(question):
    assert public_literal_plan(task(question)) is None


def test_an_unsupported_exact_constraint_invalidates_rules_from_other_public_sources():
    assert public_literal_plan(task("Use word shore once.", ["The word tide occurs twice."])) is None


@pytest.mark.parametrize("answer,status", [
    ("shore", "pass"), ("(shore).", "pass"), ("shore shore", "fail"),
    ("nothing", "fail"), ("SHORE", "unknown"), ("shore Shore", "unknown"),
    ("shore shores", "unknown"), ("shore-shore", "unknown"),
    ("shore\u0301", "unknown"), ("xshore", "unknown"),
])
def test_exact_literal_diagnostics_require_sufficient_lower_and_upper_evidence(answer, status):
    plan = public_literal_plan(task("Use word shore once."))
    assert plan.diagnose(answer)["status"] == status


@pytest.mark.parametrize("question", [
    "Do not use word shore at least 2 times.",
    "Never include word shore at least 2 times.",
    "If useful, use word shore at least 2 times.",
    "If useful. Use word shore at least 2 times.",
    "Use word shore at least 2 times only if the mood fits.",
    "Use word shore at least 2 times per paragraph.",
    "Use word shore at least 2 times in the title.",
    'Repeat this quotation: "Use word shore at least 2 times."',
    '"Use word shore at least 2 times."',
    'Copy the following text verbatim. Use word shore at least 2 times.',
    "Example:\nUse word shore at least 2 times.",
    "Examples\nUse word shore at least 2 times.",
    "Here is an example. Use word shore at least 2 times.",
    "> Use word shore at least 2 times.",
    "```text\nUse word shore at least 2 times.\n```",
    "Use the word sea breeze at least 2 times.",
    'Use word "sea breeze" at least 2 times.',
    "Use word shore at least two times.",
    "Use word shore exactly 2 times.",
    "Use word shore at most 2 times.",
    "Repeat word shore at least 2 times.",
    "Use word café at least 2 times.",
    "Use word shore at least 129 times.",
    "Use word shore at least 99999999999999999999999999999 times.",
    "Use word shore at least 02 times.",
    "Use word shore at least -1 times.",
    "Use word shore at least 2 times. Use word tide exactly 3 times.",
    "Use word shore at least 2 times. Include the word sky somewhere.",
    "Use word shore at least 2 times. Return it in lowercase.",
    "Use word shore at least 2 times. Use case-sensitive matching.",
    "Use word shore at least 2 times. Repeat cedar at least 3 times.",
    "Use word shore at least 2 times. Mention cedar exactly 3 times.",
    "Use word shore at least 2 times. Repeat cedar twice.",
    "Use word shore at least 2 times. Write the entire response in capital letters.",
    "Use word shore at least 2 times. This requirement applies only to the closing paragraph.",
])
def test_unsupported_or_scoped_frequency_families_make_the_whole_plan_unknown(question):
    assert public_literal_plan(task(question)) is None


@pytest.mark.parametrize("constraint", ["Only if you use English.",
                                        "Return the entire answer in lowercase.",
                                        "Matching is case-insensitive."])
def test_global_scope_in_another_public_source_cannot_be_silently_ignored(constraint):
    assert public_literal_plan(task("Use word shore at least 2 times.", [constraint])) is None


@pytest.mark.parametrize("literal", ["r\u0130ver", "\u0131", "\u017fhore", "\u212aabc"])
@pytest.mark.parametrize("quotes", ["", '"', "'"])
def test_unicode_ignorecase_exceptions_cannot_be_ascii_literals_or_quote_delimiters(literal, quotes):
    question = f"Use word {quotes}{literal}{quotes} at least 2 times."
    assert public_literal_plan(task(question)) is None
    assert public_literal_plan(task("Use word shore at least 2 times. " + question)) is None


def test_huge_numeric_or_literal_arguments_decline_without_integer_conversion_or_partial_rules():
    assert public_literal_plan(task("Use word shore at least " + "9" * 5000 + " times.")) is None
    assert public_literal_plan(task("Use word " + "a" * 65 + " at least 2 times.")) is None
    assert public_literal_plan(task("Use word shore at least 128 times.")) is not None


@pytest.mark.parametrize("answer,exact,folded,broad,status", [
    ("river river", 2, 2, 2, "pass"),
    ("[river], (river).", 2, 2, 2, "pass"),
    ("RIVER RIVER", 0, 2, 2, "unknown"),
    ("river RIVER", 1, 2, 2, "unknown"),
    ("rivers rivers", 0, 0, 2, "unknown"),
    ("river-river river's", 0, 0, 3, "unknown"),
    ("river’s river's", 0, 0, 2, "unknown"),
    ("xriver river2 中文river", 0, 0, 3, "unknown"),
    ("riverriver", 0, 0, 2, "unknown"),
    ("river-river2 river-river2", 0, 0, 4, "unknown"),
    ("river's2 river's2", 0, 0, 2, "unknown"),
    ("river-é river-é", 0, 0, 2, "unknown"),
    ("river’s2 river’s2", 0, 0, 2, "unknown"),
    ("river\u2010river river\u2011river", 0, 0, 4, "unknown"),
    ("river\u2013river river\u2014river", 0, 0, 4, "unknown"),
    ("river\u2018s river\u2018s", 0, 0, 2, "unknown"),
    ("river\u0301 river\u0301", 0, 0, 2, "unknown"),
    ("river riverine", 1, 1, 2, "unknown"),
    ("river alone", 1, 1, 1, "fail"),
    ("", 0, 0, 0, "fail"),
])
def test_three_valued_checks_use_sufficient_strict_pass_and_wider_failure(answer, exact, folded, broad, status):
    plan = public_literal_plan(task("Use word river at least 2 times."))
    diagnostics = plan.diagnose(answer)
    row = diagnostics["rules"][0]
    assert row["exact_strict_count"] == exact
    assert row["casefold_strict_count"] == folded
    assert row["casefold_substring_count"] == broad
    assert diagnostics["status"] == row["status"] == status
    assert diagnostics["artifact_hash"] == digest(answer)
    assert plan.diagnose({"answer": answer})["status"] == "unknown"


def test_duplicate_minima_and_original_literal_case_are_not_dropped_or_normalized_away():
    plan = public_literal_plan(task("Use word Oak at least 2 times. Use word oak at least 3 times."))
    assert [(rule.literal, rule.minimum) for rule in plan.rules] == [("Oak", 2), ("oak", 3)]
    result = plan.diagnose("Oak Oak oak oak oak")
    assert result["status"] == "pass"
    assert [row["exact_strict_count"] for row in result["rules"]] == [2, 3]
    assert [row["casefold_strict_count"] for row in result["rules"]] == [5, 5]


def test_overlapping_broad_occurrences_remain_unknown_without_independent_strict_words():
    plan = public_literal_plan(task("Use word aa at least 3 times."))
    row = plan.diagnose("aaaa")["rules"][0]
    assert row["exact_strict_count"] == 0 and row["casefold_substring_count"] == 3
    assert row["status"] == "unknown"


def test_complete_artifact_counts_are_independent_of_bounded_diagnostic_frequency_tables():
    def alphabetic(index):
        letters = []
        while True:
            letters.append(chr(ord("a") + index % 26))
            index //= 26
            if not index:
                return "term" + "".join(reversed(letters))

    artifact = " ".join(alphabetic(index) for index in range(1500)) + " zebra zebra"
    plan = public_literal_plan(task("Use word zebra at least 2 times."))
    assert plan.diagnose(artifact)["rules"][0]["exact_strict_count"] == 2
    assert plan.diagnose(artifact)["status"] == "pass"


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "kwargs": kwargs}))
        return ChatMessage(role="assistant", content=json.dumps(next(self.replies)))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


def refine(question, initial, revision, *, guarded=True, projection=False, numeric=False):
    public = task(question)
    result = RunResult(answer=initial, terminated_reason="final_answer", metadata={
        "private_feedback": "PRIVATE_CANARY", "model_calls_used": 1,
        "terminal_receipt": {"answer_hash": digest(initial), "event_id": "public-event"}})
    if projection:
        result.metadata["public_positional_draft_projection"] = {"active": True}
    config = MASConfig(backend="scripted", public_refinement=True, public_refinement_guard=guarded,
                       public_refinement_response_format="json_schema", public_numeric_construction=numeric,
                       models={"exec": ModelConfig(max_tokens=8192), "global": ModelConfig(max_tokens=16000)})
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    models = Models([{"issues": []}, {"answer": revision} if not numeric else revision])
    snapshots = []
    refine_public_answer(public, result, models, ledger, config,
                         audit_writer=lambda value: snapshots.append(copy.deepcopy(value)))
    return result, result.metadata["public_refinement"], models, ledger, snapshots


def test_valid_initial_is_selected_only_after_strict_pass_and_at_least_one_sufficient_revision_failure():
    question = "Make sure to use word oak at least 2 times, and word sky at least 3 times."
    initial, revision = "oak oak sky sky sky.", "oaks oaks sky sky."
    result, audit, models, ledger, snapshots = refine(question, initial, revision)
    assert result.answer == initial
    assert audit["version"] == PUBLIC_REFINEMENT_GUARD_VERSION == "public-artifact-regression-guard-v5"
    assert audit["status"] == "completed" and audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "explicit_public_literal_minimum_regression"
    checks = audit["public_candidate_guard"]["literal_candidate_checks"]
    assert checks["initial"]["status"] == "pass" and checks["revision"]["status"] == "fail"
    assert [row["status"] for row in checks["revision"]["rules"]] == ["unknown", "fail"]
    assert audit["revision"] == {"answer": revision} and audit["calls"][-1]["status"] == "validated"
    assert audit["answer_hash"] == audit["draft_hash"] == checks["initial"]["artifact_hash"] == digest(initial)
    assert audit["revision_answer_hash"] == checks["revision"]["artifact_hash"] == digest(revision)
    assert result.metadata["terminal_receipt"]["answer_hash"] == digest(initial)
    for request in models.requests:
        payload = json.loads(request["messages"][1]["content"])
        assert payload["public_literal_constraints"] == audit["public_candidate_guard"]["public_literal_constraints"]
        assert payload["draft_literal_constraint_diagnostics"]["status"] == "pass"
        assert "PRIVATE_CANARY" not in json.dumps(request) and "OPAQUE_ROUTE" not in json.dumps(request)
    assert len(models.requests) == ledger.snapshot()["model_calls"] == len(audit["budget_records"]) == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit == snapshots[-1]
    assert audit["audit_hash"] == digest({key: value for key, value in audit.items() if key != "audit_hash"})


@pytest.mark.parametrize("initial,revision,initial_status,revision_status", [
    ("oak oak", "oaks oaks", "pass", "unknown"),
    ("OAK OAK", "Nothing remains.", "unknown", "fail"),
    ("oak", "Nothing remains.", "fail", "fail"),
    ("oak", "oak oak", "fail", "pass"),
    ("oak oak", "oak oak oak", "pass", "pass"),
])
def test_unknown_initial_or_revision_and_actual_improvement_do_not_select_initial(initial, revision,
                                                                                initial_status, revision_status):
    result, audit, _, _, _ = refine("Use word oak at least 2 times.", initial, revision)
    assert result.answer == revision and audit["selected_candidate"] == "revision"
    checks = audit["public_candidate_guard"]["literal_candidate_checks"]
    assert checks["initial"]["status"] == initial_status
    assert checks["revision"]["status"] == revision_status


def test_exact_literal_guard_preserves_valid_frequency_when_a_rewrite_repeats_it():
    result, audit, models, ledger, snapshots = refine("Use word shore once.", "shore", "shore shore")
    assert result.answer == "shore"
    assert audit["selection_reason"] == "explicit_public_literal_count_regression"
    checks = audit["public_candidate_guard"]["literal_candidate_checks"]
    assert checks["initial"]["status"] == "pass" and checks["revision"]["status"] == "fail"
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert audit == snapshots[-1]


@pytest.mark.parametrize("question", ["Write a short scene.",
    "If useful, use word oak at least 2 times.", "Use word oak exactly 2 times."])
def test_unknown_or_absent_compiler_cannot_select_by_frequency_or_hide_behind_body_guard(question):
    result, audit, models, ledger, _ = refine(question, "oak oak", "Nothing remains.")
    assert result.answer == "Nothing remains." and audit["selected_candidate"] == "revision"
    assert audit["public_candidate_guard"]["public_literal_constraints"]["status"] == "unknown"
    assert "public_literal_constraints" not in json.loads(models.requests[0]["messages"][1]["content"])
    assert ledger.snapshot()["model_calls"] == 2


def test_projected_initial_cannot_be_selected_even_when_compiled_minima_pass():
    result, audit, _, _, _ = refine("Use word oak at least 2 times.", "oak oak", "Nothing remains.",
                                   projection=True)
    assert result.answer == "Nothing remains."
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is False
    assert audit["selected_candidate"] == "revision"


def test_numeric_construction_receipt_still_controls_submission_even_with_literal_regression():
    question = "Use word oak at least 2 times. Include exactly 2 numbers in the response."
    revision = {"opening": "The guide begins.", "number_slots": [
        {"before": "It presents", "number": "3", "after": "practical examples."},
        {"before": "It explains", "number": "5", "after": "useful methods."}], "closing": ""}
    result, audit, _, _, _ = refine(question, "oak oak", revision, numeric=True)
    assert result.answer == "The guide begins. It presents 3 practical examples. It explains 5 useful methods."
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is False
    assert audit["public_candidate_guard"]["literal_candidate_checks"]["initial"]["status"] == "pass"
    assert audit["public_candidate_guard"]["literal_candidate_checks"]["revision"]["status"] == "fail"
    assert audit["selected_candidate"] == "revision"


def test_default_off_never_exposes_compiler_or_changes_revision_only_selection():
    result, audit, models, ledger, _ = refine("Use word oak at least 2 times.", "oak oak", "Nothing remains.",
                                            guarded=False)
    assert result.answer == "Nothing remains."
    assert audit["version"] == "public-draft-review-revision-v3" and "public_candidate_guard" not in audit
    assert all("public_literal_constraints" not in json.loads(request["messages"][1]["content"])
               for request in models.requests)
    assert ledger.snapshot()["model_calls"] == 2


def large_draft():
    return "\n\n".join(["Supported public observations connect causes and effects. " * 30] * 3)


def long_heading():
    return "# " + "A careful overview of seasonal changes and practical responses " * 4


def test_long_single_markdown_heading_cannot_replace_a_large_body_and_all_candidate_costs_are_retained():
    initial, revision = large_draft(), long_heading()
    assert len(revision) > 200
    result, audit, models, ledger, snapshots = refine("Write a complete analysis of a seasonal process.",
                                                    initial, revision)
    assert result.answer == initial and audit["selected_candidate"] == "initial_draft"
    assert audit["selection_reason"] == "markdown_heading_only_large_draft_body_loss"
    structure = audit["public_candidate_guard"]["markdown_heading_only_guard"]
    assert structure["public_title_scope"]["status"] == "ordinary"
    assert structure["revision_diagnostics"] == {"single_strict_markdown_atx_heading": True,
                                                "nonempty_lines": 1,
                                                "body_non_whitespace_characters": 0}
    assert audit["revision"] == {"answer": revision} and audit["calls"][-1]["status"] == "validated"
    assert audit["answer_hash"] == audit["draft_hash"] == digest(initial)
    assert audit["revision_answer_hash"] == digest(revision)
    assert len(models.requests) == ledger.snapshot()["model_calls"] == len(audit["budget_records"]) == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
    assert audit == snapshots[-1]


@pytest.mark.parametrize("revision", [
    long_heading() + "\n\nThe body explains the requested observations and implications.",
    "A concise answer preserves the essential observations and implications. " * 3,
    "####### " + long_heading()[2:],
    "#" + long_heading()[2:],
    "    " + long_heading(),
])
def test_new_branch_does_not_raise_the_old_short_body_threshold_or_accept_nonheading_markers(revision):
    assert len(revision.strip()) > 100
    result, audit, _, _, _ = refine("Write a complete analysis of a seasonal process.", large_draft(), revision)
    assert result.answer == revision and audit["selected_candidate"] == "revision"
    assert audit["public_candidate_guard"]["markdown_heading_only_guard"]["revision_diagnostics"][
        "single_strict_markdown_atx_heading"] is False


@pytest.mark.parametrize("question", ["Return only a title.", "Write just a headline for a seasonal report.",
                                      "Provide a title only.", "Only return a title.",
                                      "Just give a headline.", "Output a title and nothing else."])
@pytest.mark.parametrize("revision", [long_heading(), "# A seasonal portrait"])
def test_explicit_positive_title_only_requests_suppress_heading_and_original_short_collapse_selection(question, revision):
    result, audit, _, _, _ = refine(question, large_draft(), revision)
    assert result.answer == revision and audit["selected_candidate"] == "revision"
    scope = audit["public_candidate_guard"]["markdown_heading_only_guard"]["public_title_scope"]
    assert scope["status"] == "title_only" and scope["public_spans"]


@pytest.mark.parametrize("question", ["If useful, return only a title.",
                                      "Return only a concise label without extra commentary.",
                                      "Write a headline. Include no other text.",
                                      "Write a title.", "Give a headline for a seasonal report.",
                                      "Generate the title and a report."])
def test_complex_only_output_scope_remains_unknown_and_does_not_activate_new_heading_selection(question):
    result, audit, _, _, _ = refine(question, large_draft(), long_heading())
    assert result.answer == long_heading() and audit["selected_candidate"] == "revision"
    assert audit["public_candidate_guard"]["markdown_heading_only_guard"]["public_title_scope"]["status"] == "unknown"


def test_quoted_title_only_instruction_is_data_and_cannot_cancel_a_complete_report_request():
    result, audit, _, _, _ = refine('Write a complete report. Discuss the quotation "Return only a title.".',
                                   large_draft(), long_heading())
    assert result.answer == large_draft()
    assert audit["selection_reason"] == "markdown_heading_only_large_draft_body_loss"
    assert audit["public_candidate_guard"]["markdown_heading_only_guard"]["public_title_scope"]["status"] == "ordinary"


def test_projected_large_draft_remains_ineligible_for_heading_only_selection():
    result, audit, _, _, _ = refine("Write a complete report.", large_draft(), long_heading(), projection=True)
    assert result.answer == long_heading() and audit["selected_candidate"] == "revision"
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is False


def test_numeric_constructed_heading_does_not_rescue_a_large_initial_without_a_final_construction_receipt():
    revision = {"opening": long_heading(), "number_slots": [
        {"before": "It outlines", "number": "3", "after": "patterns."},
        {"before": "It compares", "number": "5", "after": "cases."}], "closing": ""}
    result, audit, _, _, _ = refine("Write a complete report. Include exactly 2 numbers in the response.",
                                   large_draft(), revision, numeric=True)
    assert result.answer.startswith(long_heading()) and audit["selected_candidate"] == "revision"
    assert audit["public_candidate_guard"]["initial_eligibility"]["eligible"] is False
    assert audit["public_candidate_guard"]["markdown_heading_only_guard"]["revision_diagnostics"][
        "single_strict_markdown_atx_heading"] is True
