import pytest
from pydantic import ValidationError

from jit_mas.public_literal_slots import literal_slots_plan, render
from jit_mas.schemas import PublicTask


def task(question):
    return PublicTask(task_id="opaque", question=question)


def test_public_exact_rules_compile_fixed_named_gaps_and_round_robin_insertions():
    plan = literal_slots_plan(task("Use word shore once, word tide twice."))
    assert plan.schedule == ("shore", "tide", "tide")
    assert plan.gap_names == ("gap_000", "gap_001", "gap_002", "gap_003")
    parsed = plan.model.model_validate({"gaps": {
        "gap_000": "By the", "gap_001": "the", "gap_002": "moves; the",
        "gap_003": "returns."}})
    answer = render(parsed)
    assert answer == "By the shore the tide moves; the tide returns."
    assert plan.source_plan.diagnose(answer)["status"] == "pass"


@pytest.mark.parametrize("gap", ["shore", "SHORE", "shores", "offshore", "shore-shore"])
def test_gap_text_declines_every_casefolded_broad_target_occurrence(gap):
    plan = literal_slots_plan(task("Use word shore once."))
    with pytest.raises(ValidationError):
        plan.model.model_validate({"gaps": {"gap_000": gap, "gap_001": ""}})


@pytest.mark.parametrize("question", [
    "Use word shore at least 2 times.",
    "Use word shore once. Use word tide at least 2 times.",
    "Use word shore once. Use word shore twice.",
    "Use word shore once. Use word offshore once.",
    "Use word Shore once. Use word shore once.",
    "Use word shore 128 times. Use word tide once.",
    "If helpful, use word shore once.",
])
def test_unsupported_mixed_inconsistent_overlapping_or_over_limit_rules_decline(question):
    assert literal_slots_plan(task(question)) is None


def test_duplicate_exact_rules_are_audited_without_duplicating_insertions():
    plan = literal_slots_plan(task("Use word shore twice. Use word shore twice."))
    assert plan.schedule == ("shore", "shore")
    assert len(plan.source_plan.rules) == 2


def test_render_revalidates_bypassed_gap_validation_and_preserves_layout():
    plan = literal_slots_plan(task("Use word shore once."))
    parsed = plan.model.model_validate({"gaps": {"gap_000": "First line\n\nA", "gap_001": ".\nNext line"}})
    assert render(parsed) == "First line\n\nA shore .\nNext line"
    parsed.gaps.__dict__["gap_000"] = "shore"
    with pytest.raises(ValidationError):
        render(parsed)


def test_zero_exact_count_requires_nonempty_complete_artifact_with_no_target():
    plan = literal_slots_plan(task("Use word shore 0 times."))
    assert plan.schedule == ()
    assert render(plan.model.model_validate({"gaps": {"gap_000": "A coastal view."}})) == "A coastal view."
    with pytest.raises(ValueError):
        render(plan.model.model_validate({"gaps": {"gap_000": ""}}))


def test_schema_requires_every_named_gap_and_has_no_extra_fields():
    plan = literal_slots_plan(task("Use word shore once."))
    assert plan.response_format()["json_schema"]["strict"] is True
    with pytest.raises(ValidationError):
        plan.model.model_validate({"gaps": {"gap_000": "A"}})
    with pytest.raises(ValidationError):
        plan.model.model_validate({"gaps": {"gap_000": "A", "gap_001": "view", "extra": ""}})
