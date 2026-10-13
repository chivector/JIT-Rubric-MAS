"""Optional construction plans and compact hints from explicit public rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model

from .public_literal_constraints import public_literal_plan
from .public_word_slots import (PublicPositionSpan, _SINGLE_TOKEN_ASCII_PATTERN,
                               independent_positive_position_span)


_COMPOUNDS = {"cannot", "gimme", "gonna", "gotta", "lemme", "wanna"}
_BOUNDARY_RULE = re.compile(
    r"\bThe second word in your response and the second to last word in your "
    r"response should be the word (?P<keyword>[A-Za-z]+)\b", re.IGNORECASE)
_LETTER_RULE = re.compile(
    r"\b(?:In your entire response,\s*)?(?:the\s+)?letter\s+"
    r"(?P<letter>[A-Za-z])\s+should\s+appear\s+at\s+most\s+"
    r"(?P<count>once|twice|[0-9]{1,2}\s+times?)\b", re.IGNORECASE)
_CONJUNCTION_RULE = re.compile(
    r"\bUse at least (?P<count>[1-7](?:\.0)?) different coordinating "
    r"conjunctions in the response\b", re.IGNORECASE)
_PUNCTUATION_RULE = re.compile(
    r"\bUse every standard punctuation mark at least once, including semicolons, "
    r"colons, and the interrobang \(\?!\)", re.IGNORECASE)
_PERSON_RULE = re.compile(
    r"\bMention at least (?P<count>[0-9]{1,2}(?:\.0)?) different person names in "
    r"the response, from this list of person names:\s*"
    r"(?P<names>[A-Za-z]+(?:,\s*[A-Za-z]+)*)\s*", re.IGNORECASE)
_PUNCTUATION_SLOT = r"^[^.,;:?!\r\n]{1,160}$"
_LITERAL_HINT = re.compile(
    r"\b(?:word|keyword|term)\b[^\r\n.!?]{0,160}\b"
    r"(?:once|twice|thrice|[0-9]+\s+times?|at\s+least|at\s+most|exactly)\b",
    re.IGNORECASE)


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


@dataclass(frozen=True)
class ResponseConstraintPlan:
    kind: str
    model: type[BaseModel]
    instruction: str
    keyword: str | None = None

    def response_format(self):
        return {"type": "json_schema", "json_schema": {
            "name": "PublicResponseConstraint", "strict": True,
            "schema": self.model.model_json_schema()}}


def _sources(task):
    question = getattr(task, "question", "")
    constraints = getattr(task, "constraints", [])
    return [("question", question), *[(f"constraints[{index}]", value)
                                      for index, value in enumerate(constraints)]]


def _independent_matches(task, pattern):
    values = []
    for source, text in _sources(task):
        for match in pattern.finditer(text):
            span = PublicPositionSpan(source, match.start(), match.end(), match.group())
            if not independent_positive_position_span(text, span):
                return []
            values.append(match)
    return values


def response_constraint_plan(task):
    boundaries = _independent_matches(task, _BOUNDARY_RULE)
    letters = _independent_matches(task, _LETTER_RULE)
    punctuation = _independent_matches(task, _PUNCTUATION_RULE)
    conjunctions = _independent_matches(task, _CONJUNCTION_RULE)
    people = _independent_matches(task, _PERSON_RULE)
    literal = public_literal_plan(task)
    if boundaries and letters:
        return None
    if len(punctuation) == 1:
        if boundaries or letters or conjunctions or people or literal or any(
                _LITERAL_HINT.search(text) for _, text in _sources(task)):
            return None
        phrase = Annotated[str, Field(pattern=_PUNCTUATION_SLOT)]
        model = create_model("PublicPunctuationSlots", __base__=_StrictResponse,
            body=(str, Field(..., min_length=1, max_length=12000)),
            phrases=(list[phrase], Field(..., min_length=7, max_length=7)))
        return ResponseConstraintPlan(
            "punctuation", model,
            "Return one JSON object with exactly body and phrases. body must contain "
            "the full requested artifact, including any list "
            "or formatting the user requests. The renderer appends a natural punctuation "
            "tail from the seven phrases in order: phrase A, phrase B; phrase C: phrase D. "
            "phrase E? phrase F! phrase G?! phrases must contain exactly seven strings, "
            "each a short phrase "
            "with no periods, commas, semicolons, colons, question marks, exclamation "
            "marks, or line breaks; the inserted punctuation is mandatory. Make the "
            "phrases relevant to the body and grammatical in this sequence. For long "
            "requested lists, keep each item under forty words so every item fits. Do not "
            "describe the construction or repeat the instruction. Return one complete "
            "JSON object and stop.",
        )
    if len(boundaries) == 1:
        keyword = boundaries[0]["keyword"]
        if keyword.casefold() in _COMPOUNDS:
            return None
        model = create_model("PublicBoundaryWords", __base__=_StrictResponse,
            first_word=(str, Field(..., pattern=_SINGLE_TOKEN_ASCII_PATTERN, max_length=32)),
            middle=(str, Field(..., max_length=6000)),
            final_word=(str, Field(..., pattern=_SINGLE_TOKEN_ASCII_PATTERN, max_length=32)))
        return ResponseConstraintPlan("boundary_words", model,
            "Return one JSON object with first_word, middle, and final_word. "
            f"The renderer emits first_word, then {keyword!r}, then middle, then "
            f"{keyword!r}, then final_word and a period. first_word and final_word "
            "must each be one ASCII letter word without punctuation or spaces. "
            "Do not use cannot, gimme, gonna, gotta, lemme, or wanna in those fields. "
            "Write the requested artifact across these fields so the fixed words fit "
            "naturally. middle may contain punctuation, paragraphs or numbered items. "
            "Use a short opening and closing that carry actual requested content. "
            "No preface or explanation of the positions. Return the complete JSON once.",
            keyword)
    if len(letters) == 1:
        letter = letters[0]["letter"]
        quantifier = letters[0]["count"].casefold()
        count = {"once": 1, "twice": 2}.get(quantifier)
        if count is None:
            count = int(quantifier.split()[0])
        if count > 32:
            return None
        alphabet = letter.lower() + letter.upper()
        pattern = f"^[^{alphabet}]*(?:[{alphabet}][^{alphabet}]*){{0,{count}}}$"
        model = create_model("PublicLetterLimit", __base__=_StrictResponse,
            answer=(str, Field(..., min_length=1, max_length=6000, pattern=pattern)))
        return ResponseConstraintPlan("letter_limit", model,
            "Return one JSON object with answer containing the complete requested artifact. "
            f"Across answer, {letter!r} in either case may appear at most {count} times. "
            "Choose short synonyms that avoid this letter, including in headings. "
            "Do not repeat the instruction or explain the spelling constraint. "
            "Return the complete JSON once.")
    return None


def render_response_constraint(plan, parsed):
    checked = plan.model.model_validate(parsed.model_dump(mode="python"))
    if plan.kind == "letter_limit":
        return checked.answer
    if plan.kind == "punctuation":
        phrases = checked.phrases
        tail = (
            f"{phrases[0]}, {phrases[1]}; {phrases[2]}: {phrases[3]}. "
            f"{phrases[4]}? {phrases[5]}! {phrases[6]}?!"
        )
        return checked.body.rstrip() + "\n\n" + tail
    if plan.kind != "boundary_words":
        raise ValueError("Unknown public response construction")
    if any(value.casefold() in _COMPOUNDS
           for value in (checked.first_word, checked.final_word)):
        raise ValueError("Boundary fields must each contain one lexical word")
    parts = [checked.first_word, plan.keyword, checked.middle.strip(),
             plan.keyword, checked.final_word]
    return " ".join(value for value in parts if value) + "."


def compact_constraint_hints(task):
    hints = []
    conjunctions = _independent_matches(task, _CONJUNCTION_RULE)
    if len(conjunctions) == 1:
        count = int(float(conjunctions[0]["count"]))
        hints.append(f"Use at least {count} distinct words from and, but, or, so, yet, "
                     "for, nor as coordinating conjunctions in grammatical clauses; "
                     "however, while and because are not members of this list.")
    if len(_independent_matches(task, _PUNCTUATION_RULE)) == 1:
        hints.append("Use . , ; : and an interrobang ?! in the artifact. Also include "
                     "a separate question mark ? and a separate exclamation mark ! "
                     "outside that pair, for example in separate natural sentences.")
    people = _independent_matches(task, _PERSON_RULE)
    if len(people) == 1:
        names = people[0]["names"].split(",")
        names = list(dict.fromkeys(name.strip() for name in names))
        count = int(float(people[0]["count"]))
        if 0 < count <= len(names):
            hints.append("Mention every required name with this capitalization: "
                         + ", ".join(names[:count]) + ". They may be an audience "
                         "or hypothetical discussion group; do not invent historical "
                         "roles for them or replace them with other people.")
    return hints


def leading_literal_plan(task):
    """Allow unrelated later content after one leading standalone lexical rule."""
    if getattr(task, "constraints", []):
        return None
    question = getattr(task, "question", "")
    first, separator, remainder = question.partition(".")
    if not separator or not remainder.strip():
        return None
    if re.search(r"\b(?:word|keyword|term|count|requirement|rule|constraint|"
                 r"unless|conditional|only\s+if|provided\s+that)\b", remainder,
                 re.IGNORECASE):
        return None
    return public_literal_plan({"question": first + separator, "constraints": []})
