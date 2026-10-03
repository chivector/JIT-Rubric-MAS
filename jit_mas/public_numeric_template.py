"""Public numeric full-prose template construction; no model calls.

The production numeric parser supplies only public count/spans and its existing
conservative applicability rules. This optional layout leaves array and
named_objects production paths unchanged. Unique alphabetic placeholders reserve
numeric literals inside one complete prose artifact. Local validation proves
lexical cardinality; it cannot establish grammar, facts or remote enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from jit_mas.public_numeric_slots import (
    CONJUNCTION_REGISTRY, NumericPlan, _public_sources, numeric_plan as _public_numeric_plan,
)


_INTEGER = re.compile(r'[+-]?(?:0|[1-9][0-9]*)', re.ASCII)
_INTEGER_TEXT = Annotated[str, Field(strict=True, pattern=r'^[+-]?(?:0|[1-9][0-9]*)$')]
_TEMPLATE_TEXT = Annotated[str, Field(strict=True, min_length=1, pattern=r'^[^\d]*$')]
_MARKER_PREFIX = re.compile(r'<\s*NUM', re.IGNORECASE)
_FANBOYS = re.compile(r'\b(?:and|but|or|so|yet|for|nor)\b', re.IGNORECASE)


def _alphabetic_suffix(ordinal: int) -> str:
    """One-based alphabetic indices A..Z, AA..; no digits in marker names."""
    letters = []
    while ordinal:
        ordinal, remainder = divmod(ordinal - 1, 26)
        letters.append(chr(ord('A') + remainder))
    return ''.join(reversed(letters))


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True)


class _TemplateValues(_StrictRecord):
    @field_validator('*', check_fields=False)
    @classmethod
    def canonical_integer(cls, value):
        if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
            raise ValueError('Each numeric value must be one canonical ASCII signed integer string')
        return value


class _NumericTemplateResponse(_StrictRecord):
    _marker_keys: ClassVar[tuple[str, ...]] = ()
    _minimum_conjunction_types: ClassVar[int | None] = None

    @field_validator('answer_template', check_fields=False)
    @classmethod
    def template_text(cls, value):
        if not value.strip():
            raise ValueError('The complete artifact template must contain meaningful text')
        if any(character.isdecimal() for character in value):
            raise ValueError('Only numeric_values may contain Unicode decimal digits')
        return value

    @model_validator(mode='after')
    def template_markers_and_conjunctions(self):
        template = self.answer_template
        markers = tuple(f'<{key}>' for key in self._marker_keys)
        for marker in markers:
            if template.count(marker) != 1:
                raise ValueError(f'Each required numeric marker must occur exactly once: {marker}')
        # This also catches case changes, unknown suffixes, spacing changes and
        # unterminated/nested marker prefixes; no permissive marker repair.
        for prefix in _MARKER_PREFIX.finditer(template):
            if not any(template.startswith(marker, prefix.start()) for marker in markers):
                raise ValueError('Unknown or malformed reserved NUM marker in answer_template')
        if self._minimum_conjunction_types is not None:
            present = {word.casefold() for word in _FANBOYS.findall(template)}
            if len(present) < self._minimum_conjunction_types:
                raise ValueError('The complete template lacks the public minimum of distinct conjunction lexical types')
        return self


@dataclass(frozen=True)
class TemplatePlan:
    number_count: int
    conjunction_count: int | None
    marker_keys: tuple[str, ...]
    source_plan: NumericPlan
    model: type[BaseModel]
    layout: str = 'template'

    @property
    def schema(self):
        return self.model.model_json_schema()

    def response_format(self):
        return {'type': 'json_schema', 'json_schema': {
            'name': 'PublicNumericTemplate', 'strict': True, 'schema': self.schema}}

    def audit(self):
        result = self.source_plan.audit()
        result.pop('inserted_conjunction_literals', None)
        result.update(layout=self.layout, version='public-numeric-template-v1',
            marker_keys=list(self.marker_keys),
            markers=[f'<{key}>' for key in self.marker_keys],
            allowed_conjunction_literals=list(CONJUNCTION_REGISTRY),
            grammar_limit='Distinct FANBOYS occurrences in the complete template are lexical evidence only; grammatical coordination and every other public constraint still require normal review.',
            rendering='Revalidate the entire parsed model, then replace each unique marker with its canonical integer surrounded by spaces. No prose or conjunction clause is appended.',
            protocol='Optional template layout before submission, same fixed revision call. No API, output-prefix selection, saved-answer edit or private evaluation input.')
        return result


def template_plan(task) -> TemplatePlan | None:
    """Reuse public parsing, decline mixed/unsupported rules and reserved text."""
    source_plan = _public_numeric_plan(task)
    if source_plan is None:
        return None
    if any(_MARKER_PREFIX.search(text) for _, text in _public_sources(task)):
        # A public request might require a literal reserved marker. Do not guess
        # whether it is data, an output requirement or a construction placeholder.
        return None
    keys = tuple(f'NUM_{_alphabetic_suffix(i)}' for i in range(1, source_plan.number_count + 1))
    values = create_model('PublicNumericTemplateValues', __base__=_TemplateValues,
        **{key: (_INTEGER_TEXT, ...) for key in keys})
    model = create_model('PublicNumericTemplateResponse', __base__=_NumericTemplateResponse,
        answer_template=(_TEMPLATE_TEXT, ...), numeric_values=(values, ...))
    model._marker_keys = keys
    model._minimum_conjunction_types = source_plan.conjunction_count
    return TemplatePlan(source_plan.number_count, source_plan.conjunction_count,
                        keys, source_plan, model)


def render(parsed: BaseModel) -> str:
    """Strictly revalidate constructed/mutated instances; render no extra notes."""
    if not isinstance(parsed, _NumericTemplateResponse) or not type(parsed).model_fields:
        raise TypeError('Validate with template_plan(task).model before rendering')
    checked = type(parsed).model_validate(parsed.model_dump(mode='python'))
    answer = checked.answer_template
    for key in checked._marker_keys:
        answer = answer.replace(f'<{key}>', f' {getattr(checked.numeric_values, key)} ')
    return answer


def prepare_prompt_instruction(plan: TemplatePlan) -> str:
    markers = ', '.join(f'<{key}>' for key in plan.marker_keys) or '(no markers)'
    minimum = ''
    if plan.conjunction_count is not None:
        minimum = (f' The complete answer_template must naturally use at least {plan.conjunction_count} '
            'different coordinating conjunctions from and, but, or, so, yet, for, nor. '
            'Use genuine grammatical coordination in the prose; do not append an isolated word list.')
    return (
        'Return exactly one object with only answer_template and numeric_values. This template '
        'schema replaces any earlier answer or before/after-slot protocol. Write the entire '
        'finished user artifact as natural, coherent prose inside answer_template, preserving '
        'normal sentences, punctuation, meaningful content, supported facts and already satisfied '
        'public constraints. Do not split the prose into before/after fields or add process notes. '
        'answer_template must contain no Unicode decimal digits. Instead use each of these exact '
        'alphabetic numeric placeholders once and only once: ' + markers + '. '
        'No unknown, altered, duplicate or incomplete NUM placeholder is permitted. '
        'numeric_values is one closed object with exactly the required marker-name keys from the '
        'schema, without angle brackets. Each value is one canonical ASCII signed integer string; '
        'optional + or -, then digits; zero is allowed and leading zeros are forbidden. '
        'For zero requested numbers, numeric_values is {} and answer_template remains nonblank. '
        'The renderer replaces each placeholder with its value surrounded by spaces; it adds no '
        'sentences or conjunctions. Preserve useful numeric meaning without inventing unsupported '
        'facts or arbitrary values to fill placeholders.' + minimum +
        ' Return a single complete JSON object and stop after its closing brace. Local strict '
        'validation still applies; remote schema enforcement is not assumed. This construction '
        'proves limited numeric-literal cardinality and lexical conjunction presence, not factual '
        'accuracy, semantic number counting, grammar or all cross-constraints.')
