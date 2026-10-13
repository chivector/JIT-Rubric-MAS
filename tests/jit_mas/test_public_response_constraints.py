import json
import re

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger
from jit_mas.compact_generation import build_messages, choose_construction, generate_one
from jit_mas.public_response_constraints import (compact_constraint_hints,
    leading_literal_plan, render_response_constraint, response_constraint_plan)
from jit_mas.schemas import PublicTask


def public_task(question):
    return PublicTask(task_id="METADATA_CANARY", question=question)


def boundary_task(keyword="vibrant"):
    return public_task("The second word in your response and the second to last word "
        f"in your response should be the word {keyword}. Write a short scene.")


def test_boundary_construction_preserves_content_and_has_exact_outer_word_positions():
    plan = response_constraint_plan(boundary_task())
    parsed = plan.model.model_validate({"first_word": "A",
        "middle": "scene unfolds.\n1. Copper glows through space; Tesla waits!",
        "final_word": "world"})
    answer = render_response_constraint(plan, parsed)
    words = re.findall(r"[A-Za-z0-9]+", answer)
    assert words[1] == words[-2] == "vibrant"
    assert parsed.middle in answer
    assert answer.endswith("vibrant world.")
    assert plan.response_format()["json_schema"]["strict"] is True
    assert plan.model.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("keyword", ["vibrant", "whisper", "despite"])
def test_boundary_constraints_activate_only_with_optional_flag_and_one_call(keyword):
    class Model:
        calls = 0

        def __call__(self, messages, **kwargs):
            self.calls += 1
            assert "METADATA_CANARY" not in json.dumps(messages)
            assert kwargs["response_format"]["json_schema"]["name"] == "PublicResponseConstraint"
            return json.dumps({"first_word": "A", "middle": "scene unfolds in a",
                               "final_word": "world"})

    task = boundary_task(keyword)
    assert choose_construction(task) == ("ordinary", None)
    provider = Model()
    result = generate_one(task, provider, BudgetLedger(max_calls=1, max_tokens=10000),
                          constraint_construction=True)
    assert result["complete"] is True
    assert result["answer"].split()[1] == result["answer"].split()[-2] == keyword
    assert provider.calls == 1


@pytest.mark.parametrize("question", [
    'Repeat this quotation: "The second word in your response and the second to last word in your response should be the word vibrant."',
    "If useful, the second word in your response and the second to last word in your response should be the word vibrant.",
    "The second word in your response and the second to last word in your response should be the word cannot.",
    "The second word in your response and the second to last word in your response should be the word vibrant in the title only.",
])
def test_ambiguous_or_nonlexical_boundary_rules_decline(question):
    assert response_constraint_plan(public_task(question)) is None


def test_split_ascii_contractions_cannot_occupy_boundary_word_slots():
    plan = response_constraint_plan(boundary_task())
    with pytest.raises(ValidationError):
        plan.model.model_validate({"first_word": "Cannot", "middle": "fit",
                                  "final_word": "world"})
    with pytest.raises(ValidationError):
        plan.model.model_validate({"first_word": "Two words", "middle": "fit",
                                  "final_word": "world"})


def test_letter_limit_schema_counts_entire_artifact_and_both_cases():
    task = public_task("Write a logic quiz. In your entire response, the letter t should appear at most once.")
    plan = response_constraint_plan(task)
    assert render_response_constraint(plan, plan.model.model_validate({
        "answer": "A sofa has a red arm. Is one red arm a sign of a red sofa? Try."}))
    for answer in ("Test title", "Tt", "t" * 3):
        with pytest.raises(ValidationError):
            plan.model.model_validate({"answer": answer})


def test_literal_constraints_can_compile_before_unrelated_question_with_if():
    question = ("Include keyword quickly once in your response, keyword bridge twice "
                "in your response. Explain whether and if so how immunity works.")
    task = public_task(question)
    assert choose_construction(task, literal=True) == ("ordinary", None)
    mode, plan = choose_construction(task, literal=True, constraint_construction=True)
    assert mode == "literal"
    assert [(rule.literal, rule.minimum) for rule in plan.source_plan.rules] == [
        ("quickly", 1), ("bridge", 2)]
    messages, _ = build_messages(task, construction=mode, plan=plan,
                                 constraint_construction=True)
    assert "Explain whether and if so how immunity works." in messages[1]["content"]
    gap_schema = plan.schema["$defs"]["PublicLiteralGaps"]["properties"]
    assert all(value["maxLength"] == 96 for value in gap_schema.values())
    invalid = {name: "" for name in plan.gap_names}
    invalid[plan.gap_names[0]] = "x" * 97
    with pytest.raises(ValidationError):
        plan.model.model_validate({"gaps": invalid})


@pytest.mark.parametrize("suffix", ["This count applies only to the title.",
    "Use keyword bridge twice.", "The rule is conditional on the genre."])
def test_leading_literal_fallback_does_not_drop_further_scope_or_rules(suffix):
    assert leading_literal_plan(public_task("Use word shore once. " + suffix)) is None


def test_constraint_hints_are_compact_and_leave_default_prompt_unchanged():
    task = public_task("Use at least 5 different coordinating conjunctions in the response. "
        "Use every standard punctuation mark at least once, including semicolons, colons, "
        "and the interrobang (?!). Explain a circuit.")
    ordinary, _ = build_messages(task)
    optimized, _ = build_messages(task, constraint_construction=True)
    assert "however, while" not in ordinary[0]["content"]
    assert "however, while" in optimized[0]["content"]
    assert "a separate question mark" in optimized[0]["content"]
    assert len(compact_constraint_hints(task)) == 2
    assert optimized[1] == ordinary[1]


def test_public_name_hints_retain_required_subset_and_do_not_assign_false_roles():
    task = public_task("Mention at least 3 different person names in the response, from "
        "this list of person names: Emma, Liam, Sophia, Noah . Explain nuclear physics.")
    hints = compact_constraint_hints(task)
    assert len(hints) == 1
    assert "Emma, Liam, Sophia." in hints[0]
    assert "do not invent historical roles" in hints[0]


def test_punctuation_schema_renders_all_required_marks_after_complete_body():
    task = public_task(
        "Use every standard punctuation mark at least once, including semicolons, "
        "colons, and the interrobang (?!). Write a short poem.")
    mode, plan = choose_construction(task, constraint_construction=True)
    assert mode == "response_constraint" and plan.kind == "punctuation"
    parsed = plan.model.model_validate({
        "body": "The ocean writes a brief poem.",
        "phrases": ["waves gather", "the shore listens", "the tide turns",
                    "a new line begins", "Can meaning remain", "The sea answers",
                    "What a pattern"],
    })
    answer = render_response_constraint(plan, parsed)
    assert parsed.body in answer
    assert all(mark in answer for mark in (".", ",", ";", ":", "?", "!"))
    assert "?!" in answer
    assert answer.count("?!") == 1


def test_punctuation_schema_rejects_tail_punctuation_and_preserves_unknown_mixes():
    task = public_task(
        "Use every standard punctuation mark at least once, including semicolons, "
        "colons, and the interrobang (?!). Write a short poem.")
    mode, plan = choose_construction(task, constraint_construction=True)
    with pytest.raises(ValidationError):
        plan.model.model_validate({
            "body": "Poem",
            "phrases": ["waves,", "shore", "tide", "light", "why", "yes", "what"],
        })
    mixed = public_task(
        "Use every standard punctuation mark at least once, including semicolons, "
        "colons, and the interrobang (?!). Use word shore twice.")
    assert response_constraint_plan(mixed) is None


def test_punctuation_default_path_is_unchanged():
    task = public_task(
        "Use every standard punctuation mark at least once, including semicolons, "
        "colons, and the interrobang (?!). Write a short poem.")
    assert choose_construction(task) == ("ordinary", None)


def test_optional_numeric_prompt_makes_required_values_and_list_label_use_explicit():
    task = public_task("Include exactly 7 numbers in the response. "
                      "What is ideological colonization?")
    mode, plan = choose_construction(task, numeric=True, numeric_layout="template")
    ordinary, _ = build_messages(task, construction=mode, plan=plan)
    optimized, _ = build_messages(task, construction=mode, plan=plan,
                                 constraint_construction=True)
    assert "as ordered list labels" not in ordinary[0]["content"]
    assert "as ordered list labels" in optimized[0]["content"]
    assert '"NUM_A":"1"' in optimized[0]["content"]
    assert '"NUM_G":"7"' in optimized[0]["content"]
    assert "an empty object is invalid" in optimized[0]["content"]


def test_optional_numeric_conjunction_prompt_does_not_let_list_labels_replace_clauses():
    task = public_task("Include exactly 15 numbers in the response. "
        "Use at least 5 different coordinating conjunctions in the response. Explain meaning.")
    mode, plan = choose_construction(task, numeric=True, numeric_layout="template")
    messages, _ = build_messages(task, construction=mode, plan=plan,
                                 constraint_construction=True)
    assert "Numbered labels do not replace the conjunction requirement" in messages[0]["content"]
    assert "Every listed point should be a grammatical clause" in messages[0]["content"]
    assert "at least 5 different words" in messages[0]["content"]


def test_optional_literal_prompt_explains_that_gaps_do_not_repeat_inserted_words():
    task = public_task("Write an image prompt. Use word blue three times, word river twice.")
    mode, plan = choose_construction(task, literal=True, constraint_construction=True)
    messages, _ = build_messages(task, construction=mode, plan=plan,
                                 constraint_construction=True)
    system = messages[0]["content"]
    assert "prose BETWEEN" in system
    assert "Never type any reserved word inside a gap" in system
    assert '"gap_000":""' in system
    assert '"gap_005":""' in system
