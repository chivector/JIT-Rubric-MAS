"""One-call exact literal construction from finite public instructions only.

The model writes named text gaps.  A deterministic renderer inserts the public
literals between those gaps in a fixed order.  Strict local validation forbids
casefolded literal substrings in gaps and verifies the complete artifact again.
This protocol guarantees only its finite lexical counts, not content quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from .public_literal_constraints import PublicLiteralPlan, public_literal_plan


MAX_LITERAL_OCCURRENCES = 128


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class _LiteralGaps(_StrictRecord):
    _forbidden_literals: ClassVar[tuple[str, ...]] = ()
    _whole_words: ClassVar[bool] = False

    @field_validator("*", check_fields=False)
    @classmethod
    def no_target_substrings(cls, value):
        folded = value.casefold()
        found = (any(re.search(r"\b" + re.escape(literal) + r"\b", value, re.IGNORECASE)
                     for literal in cls._forbidden_literals) if cls._whole_words else
                 any(literal in folded for literal in cls._forbidden_literals))
        if found:
            raise ValueError("Text gaps cannot contain any reserved literal substring")
        return value


class _LiteralSlotsResponse(_StrictRecord):
    _schedule: ClassVar[tuple[str, ...]] = ()
    _gap_names: ClassVar[tuple[str, ...]] = ()
    _source_plan: ClassVar[PublicLiteralPlan | None] = None
    _whole_words: ClassVar[bool] = False


@dataclass(frozen=True)
class LiteralSlotsPlan:
    literals: tuple[tuple[str, int], ...]
    schedule: tuple[str, ...]
    gap_names: tuple[str, ...]
    source_plan: PublicLiteralPlan
    model: type[BaseModel]

    @property
    def schema(self):
        return self.model.model_json_schema()

    def response_format(self):
        return {"type": "json_schema", "json_schema": {
            "name": "PublicLiteralSlots", "strict": True, "schema": self.schema}}

    def audit(self):
        return {"version": "public-literal-slots-v1",
                "literal_counts": [{"literal": literal, "count": count}
                                   for literal, count in self.literals],
                "schedule": list(self.schedule), "gap_names": list(self.gap_names),
                "source_plan": self.source_plan.audit(),
                "maximum_occurrences": MAX_LITERAL_OCCURRENCES,
                "counting": "case-insensitive whole words" if self.model._whole_words else "conservative literal substrings",
                "rendering": "Preserve named gap text and insert each scheduled literal with surrounding spaces.",
                "limitations": "Only exact finite lexical counts are established; grammar, facts, format and other constraints are not certified."}


def literal_slots_plan(task, *, source_plan=None, max_gap_chars=None, whole_words=False) -> LiteralSlotsPlan | None:
    """Compile all-exact, consistent and nonoverlapping finite public counts."""
    source = source_plan if source_plan is not None else public_literal_plan(task)
    if source is None or any(rule.operator != "==" for rule in source.rules):
        return None
    counts = {}
    for rule in source.rules:
        if rule.literal in counts and counts[rule.literal] != rule.minimum:
            return None
        counts[rule.literal] = rule.minimum
    folded = tuple(literal.casefold() for literal in counts)
    if any((left == right if whole_words else left in right or right in left)
           for index, left in enumerate(folded)
           for right in folded[index + 1:]):
        return None
    if sum(counts.values()) > MAX_LITERAL_OCCURRENCES:
        return None
    schedule = tuple(literal for ordinal in range(max(counts.values(), default=0))
                     for literal, count in counts.items() if ordinal < count)
    gap_names = tuple(f"gap_{index:03d}" for index in range(len(schedule) + 1))
    gap_fields = {}
    for index, name in enumerate(gap_names):
        position = (f"Text before inserted word {schedule[index]!r}."
                    if index < len(schedule) else "Text after the last inserted word.")
        options = {"max_length": max_gap_chars} if max_gap_chars is not None else {}
        gap_fields[name] = (str, Field(..., description=position +
                                      " Exclude every reserved word; empty text is allowed.",
                                      **options))
    gaps_model = create_model("PublicLiteralGaps", __base__=_LiteralGaps, **gap_fields)
    gaps_model._forbidden_literals = folded
    gaps_model._whole_words = whole_words
    model = create_model("PublicLiteralSlotsResponse", __base__=_LiteralSlotsResponse,
                         gaps=(gaps_model, ...))
    model._schedule = schedule
    model._gap_names = gap_names
    model._source_plan = source
    model._whole_words = whole_words
    return LiteralSlotsPlan(tuple(counts.items()), schedule, gap_names, source, model)


def render(parsed: BaseModel) -> str:
    """Revalidate and insert public literals without deleting or replacing prose."""
    if not isinstance(parsed, _LiteralSlotsResponse) or not type(parsed).model_fields:
        raise TypeError("Validate with literal_slots_plan(task).model before rendering")
    checked = type(parsed).model_validate(parsed.model_dump(mode="python"))
    parts = [getattr(checked.gaps, checked._gap_names[0])]
    for index, literal in enumerate(checked._schedule, 1):
        parts.extend((" ", literal, " ", getattr(checked.gaps, checked._gap_names[index])))
    answer = "".join(parts).strip()
    established = (all(len(re.findall(r"\b" + re.escape(rule.literal) + r"\b", answer,
                                     re.IGNORECASE)) == rule.minimum
                       for rule in checked._source_plan.rules) if checked._whole_words else
                   checked._source_plan.diagnose(answer)["status"] == "pass")
    if not answer or not established:
        raise ValueError("Rendered artifact does not establish every public exact literal count")
    return answer


def prepare_prompt_instruction(plan: LiteralSlotsPlan) -> str:
    insertions = ", ".join(f"{before} then {literal!r}" for before, literal
                           in zip(plan.gap_names, plan.schedule))
    restriction = (
        "Do not write any reserved word or capitalization variant as a complete word "
        "in any gap; longer words such as rhythms remain distinct from rhythm. "
        if plan.model._whole_words else
        "Do not write any reserved word, capitalization variant, or longer word containing "
        "one in any gap. "
    )
    return (
        "Return one JSON object containing only gaps. gaps is the closed object of named "
        "string fields required by the schema. Write the complete requested artifact in "
        "those text gaps; the renderer will insert reserved words between them. In order: "
        + insertions + (", then " if insertions else "") + plan.gap_names[-1] + ". "
        + restriction + "Empty gaps are allowed. Preserve needed punctuation and line breaks "
        "inside the gaps; each inserted word receives a space on both sides. Write coherent "
        "sentences across these insertion points, with useful content and every other public "
        "requirement satisfied. Return the complete JSON once and stop; no process notes."
    )
