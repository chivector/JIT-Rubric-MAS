"""Offline JSON Schema/SDK compatibility for public numeric construction.

This imports an SDK schema helper only. No client is instantiated and no
request, socket, benchmark record or private evaluator is used. The candidate
and existing tests are never edited. Expected remote/local gaps are explicit.
"""

from __future__ import annotations

import copy
import re

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
import pytest
from pydantic import ValidationError as LocalValidationError
from jit_mas import public_numeric_slots


BOUNDARIES = [(0, None), (128, None), (0, 1), (0, 7), (128, 1), (128, 7)]


@pytest.fixture(scope="module")
def candidate():
    return public_numeric_slots


@pytest.fixture(scope="module")
def sdk_transform():
    helper = pytest.importorskip("openai.lib._pydantic")
    if not hasattr(helper, "to_strict_json_schema"):
        pytest.skip("Installed SDK has no offline strict-schema helper")
    return helper.to_strict_json_schema


def plan_for(candidate, number_count, conjunction_count=None):
    question = f"Include exactly {number_count} numbers."
    if conjunction_count is not None:
        question += f" Use at least {conjunction_count} distinct coordinating conjunctions."
    plan = candidate.numeric_plan({"question": question, "constraints": []})
    assert plan is not None
    return plan


def artifact(number_count, conjunction_count=None):
    data = {
        "opening": "The group followed a woodland trail.",
        "number_slots": [{"before": "We noticed", "number": "7", "after": "birds."}
                         for _ in range(number_count)],
        "closing": "The journey ended beneath quiet trees.",
    }
    if conjunction_count is not None:
        data["conjunction_clauses"] = [
            {"before": "The trail was long", "after": "we kept walking."}
            for _ in range(conjunction_count)
        ]
    return data


def object_schemas(value):
    if isinstance(value, dict):
        if value.get("type") == "object":
            yield value
        for child in value.values():
            yield from object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from object_schemas(child)


@pytest.mark.parametrize("number_count,conjunction_count", BOUNDARIES)
def test_all_objects_are_closed_and_required_all_at_count_boundaries(candidate, number_count, conjunction_count):
    schema = plan_for(candidate, number_count, conjunction_count).schema
    Draft202012Validator.check_schema(schema)
    objects = list(object_schemas(schema))
    assert len(objects) == (2 if conjunction_count is None else 3)
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
        assert all("default" not in field for field in obj["properties"].values())
    properties = schema["properties"]
    numbers = properties["number_slots"]
    assert numbers["minItems"] == numbers["maxItems"] == number_count
    if conjunction_count is None:
        assert "conjunction_clauses" not in properties
    else:
        conjunctions = properties["conjunction_clauses"]
        assert conjunctions["minItems"] == conjunctions["maxItems"] == conjunction_count


@pytest.mark.parametrize("number_count,conjunction_count", BOUNDARIES)
def test_sdk_strict_transform_preserves_entire_schema(candidate, sdk_transform, number_count, conjunction_count):
    plan = plan_for(candidate, number_count, conjunction_count)
    original = copy.deepcopy(plan.schema)
    converted = sdk_transform(plan.model)
    Draft202012Validator.check_schema(converted)
    assert converted == original
    assert plan.schema == original
    assert all(obj["additionalProperties"] is False and set(obj["required"]) == set(obj["properties"])
               for obj in object_schemas(converted))


@pytest.mark.parametrize("number_count,conjunction_count", [(0, None), (128, 7)])
def test_offline_completion_kwargs_preserve_caller_schema_over_default_json_object(candidate, number_count, conjunction_count):
    from openai._utils import transform
    from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
    from scripts.models.base import Model
    from scripts.models.openai_server import OpenAIServerModel

    plan = plan_for(candidate, number_count, conjunction_count)
    default_format = {"type": "json_object"}
    requested_format = plan.response_format()
    default_snapshot = copy.deepcopy(default_format)
    requested_snapshot = copy.deepcopy(requested_format)
    # Bypass OpenAIServerModel.__init__: no SDK client/transport/socket exists.
    model = object.__new__(OpenAIServerModel)
    Model.__init__(model, response_format=default_format)
    messages = [{"role": "user", "content": "Return the requested finished artifact."}]
    kwargs = model._prepare_completion_kwargs(
        messages, response_format=requested_format, model="offline-unused")
    assert kwargs["response_format"] == requested_snapshot
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == plan.schema
    assert kwargs["model"] == "offline-unused" and kwargs["messages"] == messages
    assert model.kwargs == {"response_format": default_snapshot}
    assert default_format == default_snapshot and requested_format == requested_snapshot
    # The SDK's request-data transform is pure; it does not submit a request.
    encoded = transform(kwargs, CompletionCreateParamsNonStreaming)
    assert encoded["response_format"] == requested_snapshot
    assert encoded["model"] == "offline-unused"
    assert "client" not in model.__dict__
    assert model.get_total_token_counts() == {"input_token_count": 0, "output_token_count": 0}


@pytest.mark.parametrize("number_count,conjunction_count", BOUNDARIES)
@pytest.mark.parametrize("schema_kind", ["native", "sdk"])
def test_valid_boundary_artifacts_pass_native_and_sdk_schemas(candidate, sdk_transform, number_count, conjunction_count, schema_kind):
    plan = plan_for(candidate, number_count, conjunction_count)
    schema = plan.schema if schema_kind == "native" else sdk_transform(plan.model)
    data = artifact(number_count, conjunction_count)
    Draft202012Validator(schema).validate(data)
    assert plan.model.model_validate(data) is not None


@pytest.mark.parametrize("number_count,conjunction_count", BOUNDARIES)
@pytest.mark.parametrize("schema_kind", ["native", "sdk"])
def test_exact_list_lengths_reject_all_available_short_and_long_variants(candidate, sdk_transform, number_count, conjunction_count, schema_kind):
    plan = plan_for(candidate, number_count, conjunction_count)
    schema = plan.schema if schema_kind == "native" else sdk_transform(plan.model)
    validator = Draft202012Validator(schema)
    variants = []
    original = artifact(number_count, conjunction_count)
    for field in ["number_slots", *(["conjunction_clauses"] if conjunction_count is not None else [])]:
        data = copy.deepcopy(original)
        data[field].append({"before": "We noticed", "number": "7", "after": "birds."}
                           if field == "number_slots" else {"before": "The day was warm", "after": "we walked."})
        variants.append(data)
        if original[field]:
            data = copy.deepcopy(original)
            data[field].pop()
            variants.append(data)
    for data in variants:
        with pytest.raises(SchemaValidationError):
            validator.validate(data)
        with pytest.raises(LocalValidationError):
            plan.model.model_validate(data)


@pytest.mark.parametrize("schema_kind", ["native", "sdk"])
@pytest.mark.parametrize("mutation", [
    "root_extra", "number_extra", "conjunction_extra", "root_missing",
    "number_missing", "conjunction_missing", "opening_integer", "closing_boolean",
    "number_integer", "number_boolean", "number_null", "before_integer",
    "after_boolean", "number_array_string", "conjunction_array_string", "conjunction_before_null",
])
def test_required_closed_objects_and_strict_types_reject_schema_bypasses(candidate, sdk_transform, schema_kind, mutation):
    plan = plan_for(candidate, 2, 2)
    schema = plan.schema if schema_kind == "native" else sdk_transform(plan.model)
    data = artifact(2, 2)
    if mutation == "root_extra":
        data["answer"] = "Unvalidated replacement."
    elif mutation == "number_extra":
        data["number_slots"][0]["extra"] = "Unexpected."
    elif mutation == "conjunction_extra":
        data["conjunction_clauses"][0]["conjunction"] = "and"
    elif mutation == "root_missing":
        del data["opening"]
    elif mutation == "number_missing":
        del data["number_slots"][0]["after"]
    elif mutation == "conjunction_missing":
        del data["conjunction_clauses"][0]["before"]
    elif mutation == "opening_integer":
        data["opening"] = 42
    elif mutation == "closing_boolean":
        data["closing"] = False
    elif mutation.startswith("number_") and mutation in {"number_integer", "number_boolean", "number_null"}:
        data["number_slots"][0]["number"] = {"number_integer": 7, "number_boolean": True, "number_null": None}[mutation]
    elif mutation == "before_integer":
        data["number_slots"][0]["before"] = 42
    elif mutation == "after_boolean":
        data["number_slots"][0]["after"] = False
    elif mutation == "number_array_string":
        data["number_slots"] = "Not an array."
    elif mutation == "conjunction_array_string":
        data["conjunction_clauses"] = "Not an array."
    else:
        data["conjunction_clauses"][0]["before"] = None
    with pytest.raises(SchemaValidationError):
        Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize("schema_kind", ["native", "sdk"])
def test_absent_conjunction_field_is_forbidden_remotely_and_locally(candidate, sdk_transform, schema_kind):
    plan = plan_for(candidate, 1)
    schema = plan.schema if schema_kind == "native" else sdk_transform(plan.model)
    data = artifact(1, 1)
    with pytest.raises(SchemaValidationError):
        Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize("digit", ["7", "\uff17", "\u0667", "\u096d", "\U0001d7cf"])
def test_python_jsonschema_rejects_decimal_digits_in_free_text(candidate, sdk_transform, digit):
    plan = plan_for(candidate, 1, 1)
    data = artifact(1, 1)
    data["conjunction_clauses"][0]["after"] = "We found " + digit + " feathers."
    for schema in [plan.schema, sdk_transform(plan.model)]:
        with pytest.raises(SchemaValidationError):
            Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


def test_unicode_digit_pattern_has_an_explicit_regex_dialect_limit(candidate):
    plan = plan_for(candidate, 0)
    text_pattern = plan.schema["properties"]["opening"]["pattern"]
    value = "We saw \u0667 birds."
    # ASCII \d behavior simulates one possible provider dialect; this is not a
    # claim about a particular remote provider and makes no provider request.
    assert re.search(text_pattern, value, flags=re.ASCII) is not None
    data = {"opening": value, "number_slots": [], "closing": ""}
    assert not Draft202012Validator(plan.schema).is_valid(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize("location", ["number.before", "number.after", "conjunction.before", "conjunction.after"])
def test_schema_min_length_does_not_express_local_nonblank_context_validation(candidate, sdk_transform, location):
    plan = plan_for(candidate, 1, 1)
    data = artifact(1, 1)
    kind, field = location.split(".")
    data["number_slots" if kind == "number" else "conjunction_clauses"][0][field] = " \t\n"
    for schema in [plan.schema, sdk_transform(plan.model)]:
        Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


def test_schema_does_not_express_local_nonempty_zero_number_artifact_validation(candidate, sdk_transform):
    plan = plan_for(candidate, 0)
    data = {"opening": "", "number_slots": [], "closing": ""}
    for schema in [plan.schema, sdk_transform(plan.model)]:
        Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)


def test_end_anchor_dialect_requires_local_fullmatch_for_number_strings(candidate, sdk_transform):
    plan = plan_for(candidate, 1)
    data = artifact(1)
    data["number_slots"][0]["number"] = "4\n"
    for schema in [plan.schema, sdk_transform(plan.model)]:
        Draft202012Validator(schema).validate(data)
    with pytest.raises(LocalValidationError):
        plan.model.model_validate(data)
