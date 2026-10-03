"""Public synthetic numeric construction tests; no benchmark checker.

No benchmark records, private evaluator modules, model calls or production
mutations are used. The production construction module is imported normally.
"""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


from jit_mas import public_numeric_slots


@pytest.fixture(scope="module")
def candidate():
    return public_numeric_slots


def public_task(rule, **extra):
    return {"question": "Write a fictional account of a woodland walk. " + rule,
            "constraints": [], **extra}


def get_plan(candidate, count=3, conjunction_count=None):
    rule = f"Include exactly {count} numbers."
    if conjunction_count is not None:
        rule += f" Use at least {conjunction_count} distinct coordinating conjunctions."
    result = candidate.numeric_plan(public_task(rule))
    assert result is not None, rule
    return result


def payload(count=3, conjunction_count=None):
    result = {
        "opening": "The party set out beneath quiet trees.",
        "number_slots": [{"before": "We saw", "number": "7", "after": "birds beside the path."}
                         for _ in range(count)],
        "closing": "Their calls followed us home.",
    }
    if conjunction_count is not None:
        result["conjunction_clauses"] = [
            {"before": "The woods were still", "after": "we kept walking."}
            for _ in range(conjunction_count)
        ]
    return result


def resolved_schema(schema, value):
    if "$ref" not in value:
        return value
    reference = value["$ref"]
    assert reference.startswith("#/$defs/")
    return schema["$defs"][reference.removeprefix("#/$defs/")]


@pytest.mark.parametrize("rule,expected_count", [
    ("Include exactly 4 numbers.", 4),
    ("Use exactly 9 numerals.", 9),
    ("Incorporate exactly 2 numeric literals.", 2),
    ("Insert exactly 6 numbers.", 6),
    ("Contain exactly 5 numbers.", 5),
    ("The answer must contain exactly 8 numerals.", 8),
    ("The answer contains exactly 11 numbers.", 11),
    ("INCLUDE EXACTLY 12 NUMBERS.", 12),
])
def test_public_count_aliases_activate_with_the_requested_array_length(candidate, rule, expected_count):
    plan = candidate.numeric_plan(public_task(rule))
    assert plan is not None
    field = plan.schema["properties"]["number_slots"]
    assert field["minItems"] == field["maxItems"] == expected_count
    assert plan.model.model_validate(payload(expected_count)) is not None


@pytest.mark.parametrize("count", [0, 1, 128])
def test_supported_count_boundaries_have_exact_lengths(candidate, count):
    plan = get_plan(candidate, count)
    schema = plan.schema["properties"]["number_slots"]
    assert schema["minItems"] == schema["maxItems"] == count
    parsed = plan.model.model_validate(payload(count))
    answer = candidate.render(parsed)
    assert len(re.findall(r"\d+", answer)) == count


def test_zero_numbers_still_requires_a_nonblank_finished_artifact(candidate):
    plan = get_plan(candidate, 0)
    assert candidate.render(plan.model.model_validate({
        "opening": "Leaves rustled beneath our steps.", "number_slots": [], "closing": ""}))
    for opening, closing in [("", ""), ("  ", "\t\n")]:
        with pytest.raises(ValidationError):
            plan.model.model_validate({"opening": opening, "number_slots": [], "closing": closing})


def test_zero_numbers_and_conjunctions_can_render_without_opening_or_closing(candidate):
    plan = get_plan(candidate, 0, 1)
    data = payload(0, 1)
    data["opening"] = data["closing"] = ""
    answer = candidate.render(plan.model.model_validate(data))
    assert answer.strip() and not re.search(r"\d", answer)
    assert re.search(r"\band\b", answer)


@pytest.mark.parametrize("adjective", ["different", "distinct"])
@pytest.mark.parametrize("count", [1, 4, 7])
def test_coordinating_conjunction_aliases_have_exact_lengths(candidate, adjective, count):
    rule = f"Include exactly 3 numbers. Use at least {count} {adjective} coordinating conjunctions."
    plan = candidate.numeric_plan(public_task(rule))
    assert plan is not None
    field = plan.schema["properties"]["conjunction_clauses"]
    assert field["minItems"] == field["maxItems"] == count


def test_strict_schema_has_only_required_slot_fields_and_no_model_chosen_conjunction(candidate):
    plan = get_plan(candidate, 3, 4)
    schema = plan.schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload(3, 4))
    assert set(schema["properties"]) == set(payload(3, 4))
    number_schema = resolved_schema(schema, schema["properties"]["number_slots"]["items"])
    conjunction_schema = resolved_schema(schema, schema["properties"]["conjunction_clauses"]["items"])
    for item_schema, fields in [(number_schema, {"before", "number", "after"}),
                               (conjunction_schema, {"before", "after"})]:
        assert item_schema["additionalProperties"] is False
        assert set(item_schema["required"]) == set(item_schema["properties"]) == fields
    response_format = plan.response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema


def test_conjunction_slots_are_absent_when_no_public_conjunction_rule_exists(candidate):
    plan = get_plan(candidate, 3)
    assert "conjunction_clauses" not in plan.schema["properties"]
    data = payload(3, 1)
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


def test_renderer_preserves_slot_order_and_inserts_registered_conjunction_literals(candidate):
    plan = get_plan(candidate, 3, 7)
    data = payload(3, 7)
    data["number_slots"] = [
        {"before": "At dawn we saw", "number": "-3", "after": "tracks in the mud."},
        {"before": "At noon we found", "number": "+8", "after": "small feathers."},
        {"before": "At dusk we counted", "number": "0", "after": "clouds above us."},
    ]
    labels = ["Alder", "Beech", "Cedar", "Dogwood", "Elm", "Fir", "Ginkgo"]
    data["conjunction_clauses"] = [
        {"before": label + " was nearby", "after": "we noticed its leaves."} for label in labels]
    answer = candidate.render(plan.model.model_validate(data))
    assert re.findall(r"[+-]?[0-9]+", answer) == ["-3", "+8", "0"]
    assert answer.index("At dawn") < answer.index("At noon") < answer.index("At dusk")
    for label, conjunction in zip(labels, ("and", "but", "or", "so", "yet", "for", "nor")):
        assert re.search(rf"{label} was nearby\s*,?\s+{conjunction}\s+we noticed its leaves\.", answer)
    assert answer.startswith(data["opening"]) and answer.endswith(data["closing"])
    assert all(field not in answer for field in ("number_slots", "conjunction_clauses"))


def test_repeated_values_still_render_as_the_requested_number_of_numeric_literals(candidate):
    plan = get_plan(candidate, 4)
    answer = candidate.render(plan.model.model_validate(payload(4)))
    assert re.findall(r"\d+", answer) == ["7"] * 4


@pytest.mark.parametrize("number", ["0", "-0", "+0", "3", "-47", "+82", "123456789012345678901234567890"])
def test_ascii_signed_canonical_integers_are_preserved(candidate, number):
    plan = get_plan(candidate, 1)
    data = payload(1)
    data["number_slots"][0]["number"] = number
    answer = candidate.render(plan.model.model_validate(data))
    assert re.findall(r"[+-]?[0-9]+", answer) == [number]


@pytest.mark.parametrize("number", [
    "", " ", " 3", "3 ", "3\n", "00", "01", "-01", "+00", "+-2", "++2", "--2",
    "2.0", ".5", "2e3", "2,000", "2_000", "0x10", "NaN", "\uff13", "\u0662", "\u22123",
    3, True, None,
])
def test_noncanonical_or_nonstring_number_slots_are_rejected(candidate, number):
    plan = get_plan(candidate, 1)
    data = payload(1)
    data["number_slots"][0]["number"] = number
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


def test_long_number_strings_need_no_integer_conversion(candidate):
    plan = get_plan(candidate, 1)
    data = payload(1)
    data["number_slots"][0]["number"] = "8" * 5000
    answer = candidate.render(plan.model.model_validate(data))
    assert re.findall(r"[0-9]+", answer) == ["8" * 5000]


@pytest.mark.parametrize("digit", ["9", "\uff19", "\u0669", "\u096f", "\U0001d7d1"])
@pytest.mark.parametrize("location", ["opening", "closing", "number.before", "number.after",
                                      "conjunction.before", "conjunction.after"])
def test_every_free_text_field_rejects_unicode_decimal_digits(candidate, digit, location):
    plan = get_plan(candidate, 1, 1)
    data = payload(1, 1)
    value = "A hidden digit " + digit + " appears here."
    if "." not in location:
        data[location] = value
    else:
        kind, field = location.split(".")
        data["number_slots" if kind == "number" else "conjunction_clauses"][0][field] = value
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize("location", ["number.before", "number.after", "conjunction.before", "conjunction.after"])
@pytest.mark.parametrize("value", ["", " \t\n", 12, True, None])
def test_context_slots_require_nonblank_strict_strings(candidate, location, value):
    plan = get_plan(candidate, 1, 1)
    data = payload(1, 1)
    kind, field = location.split(".")
    data["number_slots" if kind == "number" else "conjunction_clauses"][0][field] = value
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize("field", ["number_slots", "conjunction_clauses"])
@pytest.mark.parametrize("change", ["shorten", "lengthen", "tuple", "string"])
def test_array_lengths_and_types_cannot_relax_the_protocol(candidate, field, change):
    plan = get_plan(candidate, 3, 3)
    data = payload(3, 3)
    if change == "shorten":
        data[field].pop()
    elif change == "lengthen":
        data[field].append(copy.deepcopy(data[field][0]))
    elif change == "tuple":
        data[field] = tuple(data[field])
    else:
        data[field] = "slots"
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


def test_extra_fields_and_omitted_fields_cannot_bypass_construction(candidate):
    plan = get_plan(candidate, 1, 1)
    original = payload(1, 1)
    variants = []
    for field in original:
        data = copy.deepcopy(original)
        del data[field]
        variants.append(data)
    data = copy.deepcopy(original)
    data["answer"] = "An unchecked replacement artifact."
    variants.append(data)
    for field in ["number_slots", "conjunction_clauses"]:
        data = copy.deepcopy(original)
        data[field][0]["extra"] = "An unchecked addition."
        variants.append(data)
    for data in variants:
        with pytest.raises(ValidationError):
            plan.model.model_validate(data)


@pytest.mark.parametrize("mutation", ["append_number", "append_conjunction", "change_number", "hidden_digit"])
def test_render_revalidates_mutable_nested_slots(candidate, mutation):
    plan = get_plan(candidate, 1, 1)
    parsed = plan.model.model_validate(payload(1, 1))
    if mutation == "append_number":
        parsed.number_slots.append(parsed.number_slots[0])
    elif mutation == "append_conjunction":
        parsed.conjunction_clauses.append(parsed.conjunction_clauses[0])
    elif mutation == "change_number":
        object.__setattr__(parsed.number_slots[0], "number", "02")
    else:
        object.__setattr__(parsed.conjunction_clauses[0], "after", "We found \u0668 leaves.")
    with pytest.raises(ValidationError):
        candidate.render(parsed)


def test_render_rejects_unvalidated_payloads_and_constructed_invalid_models(candidate):
    plan = get_plan(candidate, 1)
    with pytest.raises(TypeError):
        candidate.render(payload(1))
    data = payload(1)
    data["number_slots"] = []
    with pytest.raises(ValidationError):
        candidate.render(plan.model.model_construct(**data))


@pytest.mark.parametrize("rule", [
    "Include exactly 129 numbers.",
    "Include exactly -1 numbers.",
    "Include exactly 999999999999999999999999999999999999 numbers.",
    "Include exactly three numbers.",
    "Include exactly \u0663 numbers.",
    "Include at least 3 numbers.",
    "Include at most 3 numbers.",
    "Include about 3 numbers.",
    "Include exactly 3.5 numbers.",
    "Include exactly 3 distinct numbers.",
    "Include exactly 3 even numbers.",
    "Include exactly 3 prime numbers.",
    "Include exactly 3 two-digit numbers.",
    "Include exactly 3 numbers per paragraph.",
    "Include exactly 3 numbers in a range.",
    "Include exactly 3 numbers from the date.",
    "Include exactly 3 numbers, all greater than zero.",
    "Include exactly 3 numbers, excluding the title.",
    "Use at least 4 distinct coordinating conjunctions.",
    "Include exactly 3 numbers. Use at least 0 different coordinating conjunctions.",
    "Include exactly 3 numbers. Use at least 8 distinct coordinating conjunctions.",
    "Include exactly 3 numbers. Use at least three distinct coordinating conjunctions.",
    "Include exactly 3 numbers. Use at least 2 distinct conjunctions.",
])
def test_unsupported_counts_qualifiers_and_ambiguous_rules_decline(candidate, rule):
    assert candidate.numeric_plan(public_task(rule)) is None


@pytest.mark.parametrize("question", [
    'Explain the sample instruction "Include exactly 5 numbers."',
    "The example instruction is: Include exactly 5 numbers.",
    "Discuss this example rather than following it:\n```text\nInclude exactly 5 numbers.\n```",
    "Describe this table:\n| Label | Example |\n| Sample | Include exactly 5 numbers. |",
])
def test_quoted_examples_code_blocks_and_tables_do_not_activate(candidate, question):
    assert candidate.numeric_plan({"question": question, "constraints": []}) is None


@pytest.mark.parametrize("extra_rule", [
    "Include exactly 4 numbers.",
    "Include at least 2 numbers.",
    "Include exactly 3 distinct numbers.",
    "Use at least 8 different coordinating conjunctions.",
    "Use at least 2 distinct conjunctions.",
    "Put the word Birch in the second sentence as the third word of that sentence.",
    "The fourth word of the second sentence must be Ash.",
    "Use the fifth word of a sentence for the number.",
])
def test_conflicting_or_unsupported_additional_public_rules_decline(candidate, extra_rule):
    assert candidate.numeric_plan({"question": "Include exactly 3 numbers.",
                                   "constraints": [extra_rule]}) is None


def test_identical_public_duplicates_retain_recoverable_source_spans(candidate):
    rule = "Include exactly 5 numbers."
    task = {"question": "Write a walk description. " + rule, "constraints": [rule]}
    plan = candidate.numeric_plan(task)
    assert plan is not None
    spans = plan.audit()["matched_public_spans"]
    assert len(spans) == 2
    assert {span["source"] for span in spans} == {"question", "constraints[0]"}
    for span in spans:
        source = task["question"] if span["source"] == "question" else task["constraints"][0]
        assert source[span["start"]:span["end"]] == span["text"]
        assert "exactly 5 numbers" in span["text"]


def test_ordinary_genre_and_style_requests_can_coexist_with_numeric_construction(candidate):
    task = {"question": "Write an English nature passage. Include exactly 4 numbers.",
            "constraints": ["Use natural, grammatical prose.", "Make the ending hopeful."]}
    plan = candidate.numeric_plan(task)
    assert plan is not None
    assert plan.schema["properties"]["number_slots"]["maxItems"] == 4


def test_conflicting_conjunction_minima_decline_conservative_combination(candidate):
    assert candidate.numeric_plan({"question": "Include exactly 3 numbers. Use at least 2 distinct coordinating conjunctions.",
                                   "constraints": ["Use at least 4 different coordinating conjunctions."]}) is None


def test_only_public_question_and_constraints_can_activate_the_candidate(candidate):
    rule = "Include exactly 5 numbers."
    assert candidate.numeric_plan({"question": "Write freely.", "task_id": rule,
                                   "metadata": rule, "evaluator": rule, "tools": [rule]}) is None
    task = {"question": "Write freely.", "constraints": [rule],
            "task_id": "ID_CANARY_SECRET", "metadata": {"private": "METADATA_CANARY_SECRET"},
            "evaluator": "EVALUATOR_CANARY_SECRET", "reference": "REFERENCE_CANARY_SECRET",
            "tools": ["TOOLS_CANARY_SECRET"]}
    plan = candidate.numeric_plan(task)
    assert plan is not None
    projection = json.dumps({"audit": plan.audit(), "schema": plan.schema,
                             "prompt": candidate.prepare_prompt_instruction(plan)})
    assert "CANARY_SECRET" not in projection
    for key in ["task_id", "metadata", "evaluator", "reference", "tools"]:
        altered = copy.deepcopy(task)
        altered[key] = "Include exactly 127 numbers."
        alternate = candidate.numeric_plan(altered)
        assert alternate is not None
        assert alternate.schema == plan.schema and alternate.audit() == plan.audit()


def test_object_tasks_project_the_same_public_fields_without_inspecting_other_attributes(candidate):
    mapping = {"question": "Include exactly 5 numbers.", "constraints": []}
    obj = SimpleNamespace(**mapping, task_id="OBJECT_CANARY_SECRET", reference="Include exactly 127 numbers.")
    first = candidate.numeric_plan(mapping)
    second = candidate.numeric_plan(obj)
    assert first is not None and second is not None
    assert first.schema == second.schema and first.audit() == second.audit()


@pytest.mark.parametrize("task", [
    {"question": 17, "constraints": []},
    {"question": "Include exactly 3 numbers.", "constraints": "Include exactly 4 numbers."},
    {"question": "Include exactly 3 numbers.", "constraints": [False]},
    {"question": "Include exactly 3 numbers.", "constraints": None},
])
def test_malformed_public_projection_declines_without_string_coercion(candidate, task):
    assert candidate.numeric_plan(task) is None


def test_prompt_explains_construction_without_claiming_other_requirements_are_proven(candidate):
    plan = get_plan(candidate, 4, 2)
    instruction = candidate.prepare_prompt_instruction(plan)
    assert all(field in instruction for field in payload(4, 2))
    assert "ASCII" in instruction
    assert "and" in instruction and "but" in instruction
    assert any(word in instruction.lower() for word in ("complete", "finished"))
    public_notice = json.dumps(plan.audit()).lower()
    assert "official" in public_notice and "checker" in public_notice
    assert "other" in public_notice and any(word in public_notice for word in ("fact", "grammar", "constraint"))


@pytest.mark.parametrize("question", [
    "I do not want you to include exactly 6 numbers.",
    "I don't want you to include exactly 6 numbers.",
    "I don’t want you to include exactly 6 numbers.",
    "Include exactly 6 numbers for each paragraph.",
    "Include exactly 6 numbers on every page.",
    "Include exactly 6 numbers, none repeated.",
    "Include exactly 6 numbers. Do not include any numbers.",
    "Include exactly 6 numbers. Numbers are not allowed.",
    "Include exactly 6 numbers. All numbers must be prime.",
    "Include exactly 6 numbers. Use only even numbers.",
    "Include exactly 6 numbers. Put Birch at word position 23.",
    "Include exactly 6 numbers. Put Birch in sentence number 8.",
    "Include exactly 6 numbers. Include exactly 2.5 numbers.",
    "Include exactly 6 numbers. Use at least 2.5 distinct coordinating conjunctions.",
])
def test_independent_review_found_ambiguities_use_normal_schema_fallback(candidate, question):
    assert candidate.numeric_plan({"question": question, "constraints": []}) is None


def test_rendered_artifact_agrees_with_independent_existing_public_metrics(candidate):
    from jit_mas.public_output_metrics import public_output_metrics
    task = public_task("Include exactly 4 numbers. Use at least 3 different coordinating conjunctions.")
    plan = candidate.numeric_plan(task)
    assert plan is not None
    data = payload(4, 3)
    literals = ["-8", "+42", "109", "0"]
    for slot, literal in zip(data["number_slots"], literals):
        slot["number"] = literal
    artifact = candidate.render(plan.model.model_validate(data))
    metrics = public_output_metrics(task, artifact)
    numeric = metrics["numeric_token_diagnostics"]
    assert numeric["total_numeric_tokens"] == 4
    assert [item["token"] for item in numeric["tokens"]] == literals
    assert metrics["decimal_digit_characters"] == 7
    assert metrics["fanboys_lexical_diagnostics"]["different_lexical_types_present"] >= 3
    assert artifact.startswith(data["opening"])
    assert artifact.endswith(data["closing"])


@pytest.mark.parametrize("numeric_rule", [
    "Include exactly 6 numbers.",
    "Include exactly 129 numbers.",
    "Include exactly six numbers.",
    "Include exactly 6 distinct numbers.",
    "Include exactly 6 numbers per paragraph.",
    "Include at most 6 numbers.",
    "Use several numbers.",
    "Do not include exactly 6 numbers.",
])
@pytest.mark.parametrize("positional_rule", [
    "Put the word Birch in the second sentence as the third word of that sentence.",
    "Put Birch at word position twenty-three.",
    "The two-hundredth word of the sentence should be Birch.",
])
def test_mixed_public_families_decline_before_any_schema_is_compiled(candidate, numeric_rule, positional_rule):
    task = {"question": numeric_rule, "constraints": [positional_rule]}
    assert candidate.public_construction_conflict(task) is True
    assert candidate.public_construction_conflict({"question": positional_rule,
                                                  "constraints": [numeric_rule]}) is True


@pytest.mark.parametrize("question,constraints,expected", [
    ("Write freely.", [], False),
    ("Include exactly 6 numbers.", [], False),
    ("Put Birch at word position twenty-three.", [], False),
    ("Write freely.", ["Include exactly 6 numbers.", "Put Birch at word position twenty-three."], True),
])
def test_conflict_detection_requires_both_public_families(candidate, question, constraints, expected):
    assert candidate.public_construction_conflict({"question": question, "constraints": constraints}) is expected


def test_private_metadata_cannot_create_construction_conflict(candidate):
    task = {"question": "Include exactly 6 numbers.", "constraints": [],
            "task_id": "Put Birch at word position twenty-three.",
            "reference": "The two-hundredth word should be Birch.",
            "metadata": {"constraints": ["Put Birch in the second sentence."]}}
    assert candidate.public_construction_conflict(task) is False


def test_position_compiler_none_does_not_mean_position_family_absent(candidate):
    from jit_mas.public_word_slots import position_plan
    task = {"question": "Include exactly 6 numbers.",
            "constraints": ["Put Birch at word position twenty-three."]}
    assert position_plan(task) is None
    assert candidate.numeric_plan(task) is None
    assert candidate.public_construction_conflict(task) is True


def test_word_plan_can_exist_even_when_numeric_compilation_declines_mixed_rules(candidate):
    from jit_mas.public_word_slots import position_plan
    task = {"question": "Include exactly 6 numbers.",
            "constraints": ["Put the word Birch in the second sentence as the third word of that sentence."]}
    assert candidate.numeric_plan(task) is None
    assert position_plan(task) is not None
    assert candidate.public_construction_conflict(task) is True
