"""Public-only integer and conjunction construction protocol.

The optional construction path recognizes
only unambiguous English imperative counts from question/constraints. IDs,
scores, evaluators, tool metadata and attachments are unused. Quoted/examples
decline conservatively; this is not a general instruction/source classifier.
Construction proves a limited lexical invariant: N isolated ASCII integer
literals and, if explicitly requested, at least K distinct registry words.
It does not prove grammatical coordination, semantic number counts, factual
correctness, genre or any other public constraint. Positional combinations
decline; ordinary revision remains the fallback. No model calls are made here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator


MAX_NUMBERS = 128
CONJUNCTION_REGISTRY = ('and', 'but', 'or', 'so', 'yet', 'for', 'nor')
_INTEGER = re.compile(r'[+-]?(?:0|[1-9][0-9]*)', re.ASCII)
_NO_DECIMAL_TEXT = Annotated[str, Field(strict=True, pattern=r'^[^\d]*$')]
_CONTEXT_TEXT = Annotated[str, Field(strict=True, min_length=1, pattern=r'^[^\d]*$')]
_INTEGER_TEXT = Annotated[str, Field(strict=True, pattern=r'^[+-]?(?:0|[1-9][0-9]*)$')]

_COUNT = r'(?P<count>[0-9]+)'
_NUMBER_UNIT = r'(?:numbers|numerals|numeric\s+literals)'
_NUMBER_PATTERN = re.compile(
    r'\b(?:include(?:s)?|use|incorporate|insert|provide|put|contain(?:s)?)\s+'
    r'exactly\s+'+_COUNT+r'\s+'+_NUMBER_UNIT+
    r'(?P<scope>\s+(?:in|throughout)\s+(?:the|your|this)\s+'
    r'(?:response|answer|reply|text|output))?\b', re.IGNORECASE)
_CONJUNCTION_PATTERN = re.compile(
    r'\b(?:use|include(?:s)?|incorporate|employ)\s+at\s+least\s+'+_COUNT+
    r'\s+(?:different|distinct)\s+coordinating\s+conjunctions'
    r'(?:\s+(?:in|throughout)\s+(?:the|your|this)\s+'
    r'(?:response|answer|reply|text|output))?\b', re.IGNORECASE)
# Catch unsupported/qualified counts too; they must not be silently discarded.
_NUMERIC_COUNT_HINT = re.compile(
    r'\b(?:exactly|at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|'
    r'approximately|about|around)\s+[^\s;!?]+(?:\s+[\w-]+){0,5}\s+'
    r'(?:numbers?|numerals?|digits?|numeric\s+literals?)\b', re.IGNORECASE)
_CONJUNCTION_COUNT_HINT = re.compile(
    r'\b(?:exactly|at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|'
    r'approximately|about|around)\s+[^\s;!?]+(?:\s+[\w-]+){0,5}\s+'
    r'(?:coordinating\s+)?conjunctions?\b', re.IGNORECASE)
_POSITION_HINT = re.compile(
    r'\b(?:[0-9]+(?:\s*-?\s*(?:st|nd|rd|th))?|first|second|third|fourth|'
    r'fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|'
    r'fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|last)'
    r'\s+(?:word|sentence|paragraph|line|character)\b|'
    r'\b(?:word|sentence|paragraph|line|character)\s+'
    r'(?:(?:position|number|index)\s+)?[0-9]+\b', re.IGNORECASE)
_QUOTES = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”]*”|‘[^’]*’|(?<!\w)\x27[^\x27\n]*\x27')
_QUALIFIED_TAIL = re.compile(
    r'^\s*[,:(-]?\s*(?:per\b|each\b|(?:for|on)\s+(?:each|every)\b|'
    r'none\s+(?:repeated|identical|duplicate)\b|'
    r'in\s+(?:(?:a|the)\s+)?(?:each|every|words|order|range|date)\b|'
    r'(?:from|between|ranging|chosen|selected|excluding|except|excluding|other\s+than)\b|'
    r'(?:all\s+)?(?:distinct|different|unique|positive|negative|even|odd|prime|'
    r'whole|real|natural|roman|arabic|spelled|written|\w+-digit)\b|'
    r'(?:that|which)\s+(?:are|must|should)\b|'
    r'(?:all\s+)?(?:greater|less)\s+than\b|not\s+counting\b)', re.IGNORECASE)
_NUMBER_BAN = re.compile(
    r'\b(?:do\s+not|don[\x27’]t|never|must\s+not|should\s+not)\s+'
    r'(?:include|use|write|provide|add)\s+(?:(?:any|a)\s+)?'
    r'(?:numbers?|numerals?|digits?|numeric\s+literals?)\b|'
    r'\b(?:numbers?|numerals?|digits?)\s+(?:are\s+)?'
    r'(?:not\s+(?:allowed|permitted)|forbidden)\b', re.IGNORECASE)
_GLOBAL_NUMERIC_QUALIFIER = re.compile(
    r'\b(?:numbers?|numerals?|numeric\s+literals?)\s+'
    r'(?:(?:must|should|shall|are)\s+)?(?:all\s+)?(?:be\s+)?'
    r'(?:distinct|unique|different|positive|negative|even|odd|prime|'
    r'whole|real|natural|decimal|fractional|roman|arabic|spelled|written|\w+-digit)\b|'
    r'\b(?:use|include)\s+(?:only\s+)?(?:positive|negative|even|odd|prime|roman|'
    r'fractional|decimal|\w+-digit)\s+(?:numbers?|numerals?)\b', re.IGNORECASE)
_NUMERIC_FAMILY_DIRECTIVE = re.compile(
    r'\b(?:include(?:s)?|use|incorporate|insert|provide|put|contain(?:s)?|employ)\s+'
    r'[^\r\n;.!?]{0,120}\b(?:numbers?|numerals?|numeric\s+literals?)\b', re.IGNORECASE)
_POSITION_FAMILY_HINT = re.compile(
    r'\b(?:word|sentence|paragraph|line|character)\s+(?:position|index|ordinal)\b|'
    r'\b[\w-]+(?:st|nd|rd|th)\s+(?:word|sentence|paragraph|line|character)\b|'
    r'\b(?:keyword|word|term)\s+[^\r\n;.!?]{0,120}\b(?:in|within|into|as|at)\b'
    r'[^\r\n;.!?]{0,80}\b(?:sentence|word)\b', re.IGNORECASE)


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True)


def _no_decimal(value: str) -> str:
    # The local guarantee does not depend on the provider's regex dialect.
    if any(character.isdecimal() for character in value):
        raise ValueError('Only number slots may contain Unicode decimal digits')
    return value


class _NumberSlot(_StrictResponse):
    before: _CONTEXT_TEXT
    number: _INTEGER_TEXT
    after: _CONTEXT_TEXT

    @field_validator('before', 'after')
    @classmethod
    def context(cls, value):
        _no_decimal(value)
        if not value.strip():
            raise ValueError('Number context must be nonblank')
        return value

    @field_validator('number')
    @classmethod
    def integer_literal(cls, value):
        if _INTEGER.fullmatch(value) is None:
            raise ValueError('Use one canonical ASCII signed integer literal')
        return value


class _ConjunctionClause(_StrictResponse):
    before: _CONTEXT_TEXT
    after: _CONTEXT_TEXT

    @field_validator('before', 'after')
    @classmethod
    def context(cls, value):
        _no_decimal(value)
        if not value.strip():
            raise ValueError('Conjunction sides must be nonblank')
        return value


class _NumericSlotsResponse(_StrictResponse):
    _conjunction_literals: ClassVar[tuple[str, ...]] = ()
    _layout: ClassVar[str] = 'array'
    _number_slot_names: ClassVar[tuple[str, ...]] = ()

    @field_validator('opening', 'closing', check_fields=False)
    @classmethod
    def text_without_decimal_digits(cls, value):
        return _no_decimal(value)

    @model_validator(mode='after')
    def finished_artifact_not_empty(self):
        slots = getattr(self, 'number_slots', [])
        clauses = getattr(self, 'conjunction_clauses', [])
        if isinstance(slots, BaseModel):
            slots = slots.model_dump()
        if isinstance(clauses, BaseModel):
            clauses = clauses.model_dump()
        if not (getattr(self, 'opening', '').strip() or getattr(self, 'closing', '').strip()
                or slots or clauses):
            raise ValueError('The finished artifact must be nonempty')
        return self


@dataclass(frozen=True)
class PublicNumericSpan:
    source: str
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class NumericPlan:
    number_count: int
    conjunction_count: int | None
    matched_public_spans: tuple[PublicNumericSpan, ...]
    model: type[BaseModel]
    layout: Literal['array', 'named_objects'] = 'array'

    @property
    def schema(self) -> dict:
        return self.model.model_json_schema()

    def response_format(self) -> dict:
        return {'type': 'json_schema', 'json_schema': {
            'name': 'PublicNumericSlots', 'strict': True, 'schema': self.schema}}

    def audit(self) -> dict:
        return dict(version='public-numeric-slots-v1',
                    layout=self.layout,
                    number_count=self.number_count,
                    minimum_distinct_conjunction_count=self.conjunction_count,
                    inserted_conjunction_literals=list(self.model._conjunction_literals),
                    matched_public_spans=[vars(span).copy() for span in self.matched_public_spans],
                    limits={'numbers': [0, MAX_NUMBERS], 'explicit_conjunction_count': [1,7]},
                    counting_basis='Isolated canonical ASCII signed-integer literal strings, not digit characters, spelled-out number words or semantic numeric concepts. Unicode decimal digits are locally rejected in every other text field. This is not an official checker definition.',
                    applicability='One explicit unqualified total number count, optionally one explicit distinct coordinating-conjunction minimum. Repeated identical public rules are allowed; conflicting, qualified, quoted/example or position-combined rules decline. ASCII counts only.',
                    grammar_limit='Inserted registry words guarantee lexical types only, not genuine grammatical coordination. Writers must supply meaningful independent clauses; public review/judging still checks grammar and all other constraints.',
                    protocol='Optional flag, default off; substitute the existing fixed revision format before submission. No extra call, saved-answer edit, candidate selection or private evaluation input.')


def _public_sources(task) -> list[tuple[str,str]]:
    if isinstance(task,dict):
        question,constraints = task.get('question',''),task.get('constraints',[])
    else:
        question,constraints = getattr(task,'question',''),getattr(task,'constraints',[])
    if not isinstance(question,str) or not isinstance(constraints,(list,tuple)):
        return []
    if not all(isinstance(value,str) for value in constraints):
        return []
    return [('question',question),*[(f'constraints[{i}]',value) for i,value in enumerate(constraints)]]


def _count(text: str) -> int | None:
    if len(text)>3 or (len(text)>1 and text.startswith('0')):
        return None
    return int(text)


def public_construction_conflict(task) -> bool:
    """Detect simultaneous public rule families before either compiler runs.

    Presence is deliberately broader than successful compilation. Unsupported,
    qualified, over-limit, quoted or negated numeric/position rules can trigger
    conservative ordinary-schema fallback. False positives sacrifice this
    optional construction aid; they do not invent an interpretation. This
    helper is independent of feature flags and never reads draft/review/data IDs.
    Call it whenever either construction flag is enabled, before selecting any
    plan. A compiler returning None is not evidence that its rule is absent.
    """
    numeric = position = False
    for _,text in _public_sources(task):
        numeric = numeric or bool(_NUMERIC_COUNT_HINT.search(text)
                                  or _NUMBER_PATTERN.search(text)
                                  or _NUMERIC_FAMILY_DIRECTIVE.search(text))
        position = position or bool(_POSITION_HINT.search(text)
                                    or _POSITION_FAMILY_HINT.search(text))
    return numeric and position


def _instructional(text: str, start: int, end: int) -> bool:
    if any(start < quoted.end() and end > quoted.start() for quoted in _QUOTES.finditer(text)):
        return False
    line = text[text.rfind('\n',0,start)+1:text.find('\n',start) if '\n' in text[start:] else len(text)]
    if line.strip().startswith('|') or re.search(r'\b(?:example|sample|quotation|excerpt)(?:\s+\w+){0,4}\s*:',line,re.I):
        return False
    prefix = re.split(r'[.!?;\r\n]',text[:start])[-1].strip()
    prefix = re.sub(r'^(?:[-*]\s*|[0-9]+[.)]\s*)','',prefix)
    if re.search(r'\b(?:not|don[\x27’]t|never)\b',prefix,re.I):
        return False
    prefix = re.sub(r'\b(?:please|also)\s*$','',prefix,flags=re.I).strip()
    if not prefix:
        return True
    if re.fullmatch(r'(?:the|your|this)\s+(?:response|answer|reply|text|output)',prefix,re.I):
        return True
    if re.search(r'\b(?:must|should|required\s+to|need(?:s)?\s+to|make\s+sure\s+to|'
                 r'want\s+you\s+to)\s*$',prefix,re.I):
        return True
    if prefix.endswith((',',':')):
        return True
    if re.match(r'^(?:write|compose|respond|explain|return|provide|give|include|use|'
                r'create|draft|produce)\b',prefix,re.I) and re.search(r'\band\s*$',prefix,re.I):
        return True
    return bool(re.fullmatch(r'(?:in|throughout)\s+(?:the|your|this)\s+'
                             r'(?:response|answer|reply|text|output)',prefix,re.I))


def numeric_plan(task, *, layout: Literal['array', 'named_objects'] = 'array') -> NumericPlan | None:
    """Compile a conservative public imperative, or return ordinary-schema fallback."""
    if layout not in {'array', 'named_objects'}:
        raise ValueError('Numeric construction layout must be array or named_objects')
    if public_construction_conflict(task):
        return None
    number_rules,conjunction_rules,spans = [],[],[]
    for source,text in _public_sources(task):
        if (_POSITION_HINT.search(text) or _NUMBER_BAN.search(text)
                or _GLOBAL_NUMERIC_QUALIFIER.search(text)):
            return None
        numbers = list(_NUMBER_PATTERN.finditer(text))
        conjunctions = list(_CONJUNCTION_PATTERN.finditer(text))
        for pattern,matches in ((_NUMERIC_COUNT_HINT,numbers),(_CONJUNCTION_COUNT_HINT,conjunctions)):
            for hint in pattern.finditer(text):
                if not any(match.start()<=hint.start() and match.end()>=hint.end() for match in matches):
                    return None
        for kind,matches,rules in (('number_count',numbers,number_rules),
                                   ('conjunction_count',conjunctions,conjunction_rules)):
            for match in matches:
                count = _count(match['count'])
                if (count is None or not _instructional(text,match.start(),match.end())
                        or _QUALIFIED_TAIL.match(text[match.end():])):
                    return None
                if kind=='number_count' and not 0<=count<=MAX_NUMBERS:
                    return None
                if kind=='conjunction_count' and not 1<=count<=len(CONJUNCTION_REGISTRY):
                    return None
                rules.append(count)
                spans.append(PublicNumericSpan(source,match.start(),match.end(),match.group(),kind))
    if not number_rules or len(set(number_rules))!=1 or len(set(conjunction_rules))>1:
        return None
    number_count = number_rules[0]
    conjunction_count = conjunction_rules[0] if conjunction_rules else None
    fields = dict(opening=(_NO_DECIMAL_TEXT,...),
                  number_slots=(list[_NumberSlot],Field(...,min_length=number_count,max_length=number_count)),
                  closing=(_NO_DECIMAL_TEXT,...))
    if conjunction_count is not None:
        fields['conjunction_clauses'] = (list[_ConjunctionClause],Field(...,
                                            min_length=conjunction_count,max_length=conjunction_count))
    slot_names = tuple(f'slot_{index}' for index in range(1, number_count + 1))
    conjunction_literals = CONJUNCTION_REGISTRY[:conjunction_count] if conjunction_count else ()
    if layout == 'named_objects':
        numbers_model = create_model('PublicNamedNumberSlots', __base__=_StrictResponse,
            **{name: (_NumberSlot, ...) for name in slot_names})
        fields['number_slots'] = (numbers_model, ...)
        if conjunction_count is not None:
            clauses_model = create_model('PublicNamedConjunctionClauses', __base__=_StrictResponse,
                **{literal: (_ConjunctionClause, ...) for literal in conjunction_literals})
            fields['conjunction_clauses'] = (clauses_model, ...)
    model = create_model('PublicNumericSlotsResponse',__base__=_NumericSlotsResponse,**fields)
    model._conjunction_literals = conjunction_literals
    model._layout = layout
    model._number_slot_names = slot_names
    return NumericPlan(number_count,conjunction_count,tuple(spans),model,layout)


def render(parsed: BaseModel) -> str:
    """Revalidate even bypassed/mutated objects, then render one complete artifact."""
    if not isinstance(parsed,_NumericSlotsResponse) or not type(parsed).model_fields:
        raise TypeError('Validate with numeric_plan(task).model before rendering')
    checked = type(parsed).model_validate(parsed.model_dump(mode='python'))
    number_slots = (list(getattr(checked.number_slots, name) for name in checked._number_slot_names)
                    if checked._layout == 'named_objects' else checked.number_slots)
    parts = [checked.opening.strip()]
    parts.extend(' '.join((slot.before.strip(),slot.number,slot.after.strip())) for slot in number_slots)
    if checked._conjunction_literals:
        clauses = (list(getattr(checked.conjunction_clauses, literal) for literal in checked._conjunction_literals)
                   if checked._layout == 'named_objects' else checked.conjunction_clauses)
        parts.extend(' '.join((clause.before.strip(),literal,clause.after.strip()))
                     for literal,clause in zip(checked._conjunction_literals,clauses))
    parts.append(checked.closing.strip())
    return ' '.join(part for part in parts if part)


def prepare_prompt_instruction(plan: NumericPlan) -> str:
    if plan.layout == 'named_objects':
        numbers_instruction = (
            'number_slots must be one JSON object, never an array. Its exact required keys, '
            'in rendering order, are ' + ', '.join(plan.model._number_slot_names) + '. '
            'Complete every key exactly once; do not combine, omit or repeat entries. '
            if plan.number_count else 'number_slots must be the closed empty JSON object {}. ')
        clauses_instruction = ''
        if plan.conjunction_count is not None:
            clauses_instruction = (
                ' conjunction_clauses must be one JSON object, never an array, with exactly these '
                'required keys in rendering order: ' + ', '.join(plan.model._conjunction_literals) + '. '
                'Each value has before and after nonblank digit-free text. The renderer inserts '
                'the literal key between them. Supply suitable clauses so that each inserted '
                'word functions grammatically as a coordinating conjunction; lexical presence '
                'alone does not prove grammar.')
        return (
            'This revision uses only the supplied named-object numeric schema; no answer field, '
            'array, second object, repeated root field or process notes are permitted. Return one '
            'complete JSON object and stop immediately after its closing brace. '
            + numbers_instruction +
            'Each number-slot value is a complete object with before, number and after. before '
            'and after are nonblank meaningful context with no Unicode decimal digits. number '
            'Every entry must stand as a substantive local clause: before supplies its own '
            'subject and verb, and after supplies the remaining meaning and punctuation. '
            'Do not move the subject or verb into opening and leave before empty. Empty strings '
            'are forbidden in every before and after field, including conjunction entries. '
            'For conjunction entries, write a complete meaningful clause on each side of the '
            'inserted word. Opening and closing may be empty when the entries already form '
            'the finished artifact. The number field '
            'is one canonical ASCII signed integer string: optional + or -, then digits; zero '
            'is valid and leading zeros are forbidden. Put every numeric literal only inside '
            'its number field. Write a coherent local clause around each number rather than '
            'dropping alternating values while splitting long prose. opening and closing are '
            'digit-free text, may be empty, and carry the rest of the complete finished artifact. '
            'The renderer joins opening, every numbered entry, every conjunction entry, then '
            'closing with spaces, in the declared schema order regardless of returned key order. '
            + clauses_instruction +
            ' Preserve the original public request, useful supported facts and every already '
            'satisfied constraint. Do not invent numbers or factual claims to fill fields. '
            'With no numeric or conjunction entries, opening or closing must contain meaningful '
            'artifact text. This layout assists enumeration; remote schema enforcement is not '
            'assumed. Local strict validation and rendering prove only the stated lexical counts, '
            'not factual accuracy, grammatical coordination or other public constraints.')
    extra = ''
    if plan.conjunction_count is not None:
        extra = (f' conjunction_clauses must contain exactly {plan.conjunction_count} objects, '
                 'each with nonblank before and after text forming genuine coordinated clauses. '
                 'The renderer inserts, in this order, '+', '.join(plan.model._conjunction_literals)+
                 '. Write each pair so its inserted word functions grammatically as a coordinating '
                 'conjunction; in particular, for must give an explanation and nor needs suitable '
                 'negative coordination. A registry word merely appearing is not grammatical proof.')
    return (
        'For this revision, the numeric-slot schema overrides any earlier instruction to return '
        'an answer field. Do not return answer or process notes: return only the schema fields. '
        'Their rendered content must be the complete finished artifact requested by the user. '
        f'number_slots must contain exactly {plan.number_count} objects with before, number and after. '
        'Each number is one canonical ASCII signed integer string, with optional + or -, and '
        'digits only thereafter; zero is valid, leading zeros are forbidden. Each before and '
        'after is nonblank meaningful context without any Unicode decimal digit. opening and '
        'closing supply the rest of the finished artifact, may be empty, and cannot contain '
        'Unicode decimal digits. Place every actual numeric literal inside a number slot, '
        'including dates, amounts and list labels if those are necessary. Preserve required '
        'numbers and supported facts; do not invent arbitrary values merely to fill slots. '
        'The renderer joins opening, the number-context entries, optional conjunction clauses '
        'and closing with spaces, and separates each integer from context with spaces.'+extra+
        ' Preserve the public request, source-grounded meaning, language, genre and all already '
        'satisfied constraints. The count is numeric literals, not digit characters or spelled-out '
        'number words. This limited construction cannot guarantee semantic number counting, '
        'grammar, positional rules, completeness or other formatting constraints; those still '
        'require public review and normal judging.'
    )
