"""Finite public literal-frequency constraints, never benchmark metadata.

Only positive English instructions about one ASCII word and an explicit digit
minimum are recognized. Unsupported or ambiguous frequency-like instructions
make the whole plan unknown. Counting is over the complete decoded artifact;
it does not use the bounded diagnostic vocabulary or certify semantic quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .public_word_slots import independent_positive_position_span
from .schemas import digest


PUBLIC_LITERAL_CONSTRAINT_VERSION = "public-literal-minimum-frequency-v1"
MAX_MINIMUM_OCCURRENCES = 128
COUNTING_BASIS = (
    "Complete decoded artifact only. PASS requires original-case standalone "
    "ASCII-letter lexical word occurrences within complete Unicode word/combining-mark "
    "units; internal ASCII/curly apostrophes and Unicode dash punctuation join such units. A separate "
    "casefolded strict count is diagnostic only. FAIL requires even the broader "
    "overlapping casefolded literal-substring occurrence count to be below the minimum. "
    "Other cases are UNKNOWN. The broad count does not prove word compliance, "
    "and these sufficient conditions are not a benchmark checker."
)
_LITERAL = (r'(?:[A-Za-z]{1,64}|"[A-Za-z]{1,64}"|\u201c[A-Za-z]{1,64}\u201d|'
            r"'[A-Za-z]{1,64}'|\u2018[A-Za-z]{1,64}\u2019|`[A-Za-z]{1,64}`)")
_ITEM_TEXT = (r"(?:the\s+)?(?:word|keyword|term)\s+" + _LITERAL
              + r"\s+at\s+least\s+[0-9]+\s+times?\b")
_GROUP = re.compile(
    r"\b(?:(?:make\s+sure\s+to|please)\s+)?(?:use|include)\s+" + _ITEM_TEXT
    + r"(?:\s*,?\s+and\s+(?:(?:use|include)\s+)?" + _ITEM_TEXT + r")*", re.IGNORECASE)
_ITEM = re.compile(
    r"\b(?:the\s+)?(?:word|keyword|term)\s+(?P<literal>" + _LITERAL + r")"
    + r"\s+at\s+least\s+(?P<minimum>[0-9]+)\s+times?\b", re.IGNORECASE)
_DIRECTIVE_SCAN = re.compile(r"\b(?:use|include)\s+(?:the\s+)?(?:word|keyword|term)\b",
                             re.IGNORECASE)
_FREQUENCY_HINT = re.compile(
    r"\b(?:word|keyword|term)\b[^\r\n.!?]{0,256}?\b"
    r"(?:at\s+least|at\s+most|exactly|no\s+(?:more|fewer)\s+than|times?|occurrences?)\b",
    re.IGNORECASE)
_UNSUPPORTED_FREQUENCY = re.compile(
    r"\b(?:repeat|mention|say|write|include|use|insert|add)\b[^\r\n.!?]{0,128}?"
    r"\b(?:(?:at\s+least|at\s+most|exactly|no\s+(?:more|fewer)\s+than)\s+)?"
    r"(?:[0-9]+|[A-Za-z]+)\s+times?\b|"
    r"\b(?:repeat|mention|say|write|include|use|insert|add)\b[^\r\n.!?]{0,128}?"
    r"\b(?:once|twice|thrice)\b", re.IGNORECASE)
_CONDITIONAL = re.compile(r"\b(?:if|unless|conditional(?:ly)?|provided\s+that|"
                           r"on\s+condition\s+that)\b", re.IGNORECASE)
_CASE_SCOPE = re.compile(r"\b(?:case[ -]sensitive|case[ -]insensitive|lower[ -]?case|"
                         r"upper[ -]?case|all\s+caps|capitali[sz](?:ation|e)|"
                         r"(?:capital|small)\s+letters)\b", re.IGNORECASE)
_UNSUPPORTED_SCOPE = re.compile(
    r"\b(?:this|the)\s+(?:requirement|rule|constraint|minimum|count)\s+"
    r"(?:applies?|refers?)\s+(?:only\s+)?to\b|"
    r"\b(?:only\s+count|count\s+only|not\s+counting|do\s+not\s+count)\b",
    re.IGNORECASE)
_EXAMPLE_HEADING = re.compile(
    r"(?im)^[ \t]*(?:examples?|samples?|quotations?|excerpts?|templates?|code|"
    r"quoted\s+(?:text|passage))[ \t]*:?[ \t]*$")
_EXAMPLE_INTRO = re.compile(
    r"\b(?:here\s+(?:is|are)|the\s+following|below\s+is|this\s+is)\s+"
    r"(?:(?:a|an|the)\s+)?(?:example|sample|quotation|excerpt|template|code)\b",
    re.IGNORECASE)


def _lexical_units(text):
    """Consume complete units so an ASCII prefix cannot pass by regex backtracking."""
    def word_character(character):
        return character.isalnum() or character == "_" or unicodedata.category(character).startswith("M")

    start, length = 0, len(text)
    while start < length:
        if not word_character(text[start]):
            start += 1
            continue
        end = start + 1
        while end < length:
            if word_character(text[end]):
                end += 1
            elif ((text[end] in "'\u2018\u2019" or unicodedata.category(text[end]) == "Pd")
                  and end + 1 < length
                  and word_character(text[end + 1])):
                end += 2
            else:
                break
        yield text[start:end]
        start = end


@dataclass(frozen=True)
class PublicLiteralSpan:
    source: str
    start: int
    end: int
    text: str

    def audit(self):
        return {"source": self.source, "start": self.start, "end": self.end,
                "text": self.text, "text_hash": digest(self.text)}


@dataclass(frozen=True)
class LiteralMinimumRule:
    literal: str
    minimum: int
    span: PublicLiteralSpan

    def audit(self):
        return {"literal": self.literal, "minimum_occurrences": self.minimum,
                "operator": ">=", "public_span": self.span.audit()}


@dataclass(frozen=True)
class PublicLiteralPlan:
    rules: tuple[LiteralMinimumRule, ...]
    instruction_spans: tuple[PublicLiteralSpan, ...]

    def audit(self):
        return {"version": PUBLIC_LITERAL_CONSTRAINT_VERSION, "status": "compiled",
                "counting_basis": COUNTING_BASIS,
                "rules": [rule.audit() for rule in self.rules],
                "instruction_spans": [span.audit() for span in self.instruction_spans],
                "limitations": "Only these finite positive literal minima are established; other public constraints and semantic correctness are not certified."}

    def diagnose(self, answer):
        if not isinstance(answer, str):
            return {"status": "unknown", "reason": "Decoded artifact is not text",
                    "counting_basis": COUNTING_BASIS, "artifact_hash": None, "rules": []}
        exact_counts = {rule.literal: 0 for rule in self.rules}
        folded_counts = {rule.literal.casefold(): 0 for rule in self.rules}
        for original in _lexical_units(answer):
            if original in exact_counts:
                exact_counts[original] += 1
            folded = original.casefold()
            if folded in folded_counts:
                folded_counts[folded] += 1
        folded_answer = answer.casefold()
        rows = []
        for rule in self.rules:
            exact = exact_counts[rule.literal]
            folded = folded_counts[rule.literal.casefold()]
            broad = sum(1 for _ in re.finditer(r"(?=" + re.escape(rule.literal.casefold()) + r")",
                                              folded_answer))
            status = ("pass" if exact >= rule.minimum else
                      "fail" if broad < rule.minimum else "unknown")
            rows.append({**rule.audit(), "exact_strict_count": exact,
                         "casefold_strict_count": folded,
                         "casefold_substring_count": broad, "status": status})
        status = ("pass" if all(row["status"] == "pass" for row in rows) else
                  "fail" if any(row["status"] == "fail" for row in rows) else "unknown")
        return {"status": status,
                "counting_basis": COUNTING_BASIS, "artifact_hash": digest(answer), "rules": rows}


def public_literal_plan(task) -> PublicLiteralPlan | None:
    """Compile every recognized frequency instruction, or conservatively decline.

    Supported instructions begin a standalone sentence with Use/Include,
    optionally Please/Make sure to. Further minima can be joined with and.
    Literal argument quotes are allowed; prose quotations, copied examples,
    negation, conditionals, count qualifiers and unsupported families decline.
    The shared scope helper remains a limited textual convention, not a general
    natural-language instruction classifier.
    """
    question = task.get("question", "") if isinstance(task, dict) else getattr(task, "question", "")
    constraints = task.get("constraints", []) if isinstance(task, dict) else getattr(task, "constraints", [])
    if (not isinstance(question, str) or not isinstance(constraints, (list, tuple))
            or not all(isinstance(value, str) for value in constraints)):
        return None
    sources = [("question", question), *[(f"constraints[{i}]", value)
                                          for i, value in enumerate(constraints)]]
    rules, spans = [], []
    for source, text in sources:
        groups = list(_GROUP.finditer(text))
        for hint in (*_DIRECTIVE_SCAN.finditer(text), *_FREQUENCY_HINT.finditer(text),
                     *_UNSUPPORTED_FREQUENCY.finditer(text)):
            if not any(group.start() <= hint.start() and hint.end() <= group.end() for group in groups):
                return None
        masked = list(text)
        items = []
        for group in groups:
            for item in _ITEM.finditer(text, group.start(), group.end()):
                count = item["minimum"]
                if (len(count) > 3 or (len(count) > 1 and count.startswith("0"))
                        or int(count) > MAX_MINIMUM_OCCURRENCES):
                    return None
                raw = item["literal"]
                literal = raw[1:-1] if raw[0] in "\"\u201c'\u2018`" else raw
                if not literal or not literal.isascii() or not literal.isalpha():
                    return None
                span = PublicLiteralSpan(source, item.start(), item.end(), item.group())
                items.append(LiteralMinimumRule(literal, int(count), span))
                start, end = item.span("literal")
                masked[start:end] = " " * (end - start)
        masked_text = "".join(masked)
        if (_CONDITIONAL.search(masked_text) or _CASE_SCOPE.search(masked_text)
                or _UNSUPPORTED_SCOPE.search(masked_text)):
            return None
        for group in groups:
            span = PublicLiteralSpan(source, group.start(), group.end(), group.group())
            before = masked_text[:span.start]
            if (_EXAMPLE_HEADING.search(before) or _EXAMPLE_INTRO.search(before)
                    or not independent_positive_position_span(masked_text, span)):
                return None
            spans.append(span)
        rules.extend(items)
    return PublicLiteralPlan(tuple(rules), tuple(spans)) if rules else None
