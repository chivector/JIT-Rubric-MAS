"""Public synthetic construction tests; no benchmark records or checker calls."""

import re

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jit_mas.public_word_slots import position_plan, prepare_prompt_instruction, render
from jit_mas.schemas import PublicTask


def task(rule):
    return PublicTask(task_id="METADATA_CANARY", question="Write a nature passage. " + rule)


def data():
    return {
        "preceding_sentences": ["Birds gather!", "Wind rises?"],
        "prefix_words": ["We", "walk", "toward"],
        "keyword": "Maple",
        "suffix_words": ["trees", "together"],
        "following_sentences": ["The path is quiet."],
    }


def plan():
    return position_plan(task("Include keyword Maple in the 3rd sentence, as the 4th word of that sentence."))


def test_rendered_artifact_places_literal_in_actual_sentence_word_arrays():
    item = plan()
    answer = render(item.model.model_validate(data()))
    chunks = re.split(r"(?<=[.!?])\s+", answer)
    assert len(chunks) == 4
    assert chunks[2].split()[3] == "Maple"
    assert re.findall(r"\w+(?:['\u2019-]\w+)*", chunks[2])[3] == "Maple"
    assert answer == "Birds gather! Wind rises? We walk toward Maple trees together. The path is quiet."
    assert "process" not in answer and "prefix_words" not in answer


@pytest.mark.parametrize("rule", [
    'Put the word "copper" in the second sentence at the third word of this sentence.',
    "Place keyword 'copper' as the 3-rd word in the 2-nd sentence.",
    "USE term `copper` within the 2nd sentence, as the 3rd word in that sentence.",
    "Insert word \u201ccopper\u201d as the third word of the second sentence.",
    'The third word of the second sentence must be "copper".',
])
def test_common_imperative_variants_retain_exact_matched_public_span(rule):
    item = position_plan(task(rule))
    assert (item.keyword, item.sentence_index, item.word_index) == ("copper", 2, 3)
    span = item.matched_public_spans[0]
    assert span.text == rule.rstrip(".")
    assert task(rule).question[span.start:span.end] == span.text


def test_zero_prefix_and_preceding_sentence_counts_and_empty_suffix_are_usable():
    item = position_plan(task("Include keyword Sun in the 1st sentence as the 1st word of that sentence."))
    parsed = item.model.model_validate({"preceding_sentences": [], "prefix_words": [],
                                       "keyword": "Sun", "suffix_words": [], "following_sentences": []})
    assert render(parsed) == "Sun."


def test_schema_is_strict_and_exposes_required_lengths_and_literal():
    schema = plan().schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(data())
    assert schema["properties"]["preceding_sentences"]["minItems"] == 2
    assert schema["properties"]["preceding_sentences"]["maxItems"] == 2
    assert schema["properties"]["prefix_words"]["minItems"] == 3
    assert schema["properties"]["prefix_words"]["maxItems"] == 3
    assert schema["properties"]["keyword"]["const"] == "Maple"
    assert plan().response_format()["json_schema"]["strict"] is True


@pytest.mark.parametrize("field,value", [
    ("preceding_sentences", ["One sentence."]),
    ("prefix_words", ["one", "two"]),
    ("keyword", "maple"),
    ("suffix_words", [""]),
    ("prefix_words", ["one-two", "two", "three"]),
    ("suffix_words", ["two words"]),
    ("suffix_words", ["number7"]),
    ("suffix_words", ["\u6811"]),
    ("following_sentences", ["Two. Sentences."]),
    ("following_sentences", ["Missing ending"]),
    ("following_sentences", ["Line\nbreak."]),
    ("following_sentences", [" Leading space."]),
    ("following_sentences", ["Trailing space. "]),
    ("following_sentences", ["Only punctuation... "]),
    ("following_sentences", ["Not one\u3002 sentence."]),
    ("suffix_words", "words"),
])
def test_invalid_slots_cannot_be_rendered_as_success(field, value):
    payload = data()
    payload[field] = value
    with pytest.raises(ValidationError):
        plan().model.model_validate(payload)


def test_extra_fields_and_post_validation_mutation_are_rejected():
    payload = data()
    payload["answer"] = "Do not bypass slots"
    with pytest.raises(ValidationError):
        plan().model.model_validate(payload)
    parsed = plan().model.model_validate(data())
    parsed.prefix_words.append("extra")
    with pytest.raises(ValidationError):
        render(parsed)
    with pytest.raises(TypeError):
        render(data())


@pytest.mark.parametrize("rule", [
    "Include keyword dawn in the 129th sentence as the 2nd word of that sentence.",
    "Include keyword dawn in the 2nd sentence as the 257th word of that sentence.",
    "Include keyword dawn in the 0th sentence as the 2nd word of that sentence.",
    "Include keyword dawn in the second sentence as the twenty-first word of that sentence.",
    "The twenty-first word of the second sentence must be dawn.",
    "Include keyword dawn in the 9999999999999999th sentence as the 3rd word of that sentence.",
    "Include keyword two-words in the 2nd sentence as the 3rd word of that sentence.",
    "Include keyword dawn in the 2nd sentence.",
])
def test_unsupported_or_out_of_range_rules_decline_without_guessing(rule):
    assert position_plan(task(rule)) is None


def test_conflicting_positions_decline_but_identical_public_duplicates_are_audited():
    rule = "Include keyword Pine in the 2nd sentence as the 3rd word of that sentence."
    duplicate = {"question": rule, "constraints": [rule]}
    item = position_plan(duplicate)
    assert len(item.matched_public_spans) == 2
    assert item.matched_public_spans[1].source == "constraints[0]"
    assert position_plan({"question": rule, "constraints": [rule.replace("3rd", "4th")]}) is None
    assert position_plan({"question": rule, "constraints": [rule.replace("Pine", "Oak")]}) is None
    assert position_plan({"question": rule, "constraints": [
        "The fourth word of the second sentence must be Pine."]}) is None
    assert position_plan({"question": rule, "constraints": [
        "The twenty-first word of the second sentence must be Pine."]}) is None


def test_only_public_question_and_constraints_can_activate_or_enter_input():
    rule = "Include keyword Moss in the 2nd sentence as the 3rd word of that sentence."
    assert position_plan({"question": "Write freely.", "task_id": rule, "private_kwargs": rule}) is None
    item = position_plan({"question": "Write freely.", "constraints": [rule],
                          "task_id": "SECRET_ID", "reference": "PRIVATE_REFERENCE"})
    projected = str(item.audit()) + str(item.schema) + prepare_prompt_instruction(item)
    assert "SECRET_ID" not in projected and "PRIVATE_REFERENCE" not in projected
    assert "matched_public_spans" in item.audit()
    assert "official checker" in item.audit()["applicability"]
    assert "overrides any earlier instruction" in prepare_prompt_instruction(item)


def test_maximum_public_positions_produce_exact_schema_counts():
    item = position_plan(task("Include keyword Peak in the 128th sentence as the 256th word of that sentence."))
    assert item.schema["properties"]["preceding_sentences"]["maxItems"] == 127
    assert item.schema["properties"]["prefix_words"]["maxItems"] == 255


TREEBANK_LETTER_COMPOUNDS = ["cannot", "gimme", "gonna", "gotta", "lemme", "wanna"]


def alternate_case(word):
    return "".join(letter.upper() if index % 2 else letter for index, letter in enumerate(word))


@pytest.mark.parametrize("compound", TREEBANK_LETTER_COMPOUNDS)
@pytest.mark.parametrize("case", [str.lower, str.upper, str.title, alternate_case])
@pytest.mark.parametrize("field", ["prefix_words", "suffix_words"])
def test_general_treebank_compounds_fail_local_and_transported_word_slot_schema(compound, case, field):
    item = plan()
    payload = data()
    payload[field][0] = case(compound)
    validator = Draft202012Validator(item.schema)
    Draft202012Validator.check_schema(item.schema)
    errors = list(validator.iter_errors(payload))
    assert len(errors) == 1 and list(errors[0].path) == [field, 0]
    with pytest.raises(ValidationError):
        item.model.model_validate(payload)
    assert payload[field][0] == case(compound)  # Validation never rewrites text.


@pytest.mark.parametrize("compound", TREEBANK_LETTER_COMPOUNDS)
@pytest.mark.parametrize("case", [str.lower, str.upper, str.title, alternate_case])
def test_unsupported_literal_compound_declines_without_replacing_original_public_task(compound, case):
    public = task(f"Include keyword {case(compound)} in the second sentence as the third word of that sentence.")
    original = public.model_dump(mode="json")
    assert position_plan(public) is None
    assert public.model_dump(mode="json") == original


@pytest.mark.parametrize("word", [
    "Can", "not", "will", "wander", "cannotable", "GIMMELess", "GONNAbE",
    "gotten", "lemmas", "wannabe", "xCannot", "WANNAs",
])
def test_safe_words_and_compound_substrings_keep_existing_schema_and_render_behavior(word):
    item = plan()
    payload = data()
    payload["prefix_words"][1] = word
    Draft202012Validator(item.schema).validate(payload)
    parsed = item.model.model_validate(payload)
    answer = render(parsed)
    assert answer.split("toward Maple", 1)[0].endswith("We " + word + " ")
    assert parsed.model_dump(mode="json") == payload
    assert set(item.schema["required"]) == set(data())
    assert item.schema["properties"]["prefix_words"]["maxItems"] == 3


@pytest.mark.parametrize("field", ["prefix_words", "suffix_words"])
def test_render_revalidates_compound_inserted_by_model_construct(field):
    item = plan()
    payload = data()
    payload[field][0] = "gonna"
    unchecked = item.model.model_construct(**payload)
    with pytest.raises(ValidationError):
        render(unchecked)
    assert unchecked.model_dump(mode="python")[field][0] == "gonna"


@pytest.mark.parametrize("field", ["prefix_words", "suffix_words"])
def test_render_revalidates_compound_inserted_by_list_mutation(field):
    item = plan()
    parsed = item.model.model_validate(data())
    getattr(parsed, field)[0] = "WaNnA"
    with pytest.raises(ValidationError):
        render(parsed)
    assert getattr(parsed, field)[0] == "WaNnA"


def test_assignment_rejects_compounds_without_overwriting_previous_valid_words():
    parsed = plan().model.model_validate(data())
    original = parsed.model_dump(mode="json")
    with pytest.raises(ValidationError):
        parsed.suffix_words = ["GIMME"]
    assert parsed.model_dump(mode="json") == original


@pytest.mark.parametrize("word", ["\u0130", "\u0131", "\u017f", "\u212a"])
def test_case_insensitive_public_literal_matching_cannot_admit_non_ascii_keyword(word):
    assert position_plan(task(f"Include keyword {word} in the second sentence as the third word of that sentence.")) is None
