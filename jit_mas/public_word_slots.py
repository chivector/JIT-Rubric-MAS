"""A conservative public English word-position construction protocol.

Only question/constraints are inspected. One explicit, fully matched literal
word-position rule can activate; IDs, evaluator records and scores are unused.
This is an optional construction aid, not an official sentence/word checker.
Target-sentence words are ASCII letters only; sentence slots forbid internal
terminal punctuation/newlines and abbreviations. Other public constraints,
genre, completeness, facts and naturalness still need ordinary review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


MAX_SENTENCE_POSITION = 128
MAX_WORD_POSITION = 256
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
_ORDINAL = r"(?:[0-9]+(?:\s*-?\s*(?:st|nd|rd|th))?|" + "|".join(_ORDINALS) + r")"
_LITERAL = r"(?P<literal>[A-Za-z]+)"
_QUOTED_LITERAL = (r'(?:"' + _LITERAL + r'"|\u201c(?P<curly_literal>[A-Za-z]+)\u201d|'
                   r"'(?P<single_literal>[A-Za-z]+)'|\u2018(?P<curly_single_literal>[A-Za-z]+)\u2019|"
                   r"`(?P<code_literal>[A-Za-z]+)`|(?P<bare_literal>[A-Za-z]+))")
_INTRO = r"\b(?:include|place|put|insert|use)\s+(?:the\s+)?(?:keyword|word|term)\s+"
_PATTERNS = [
    re.compile(_INTRO + _QUOTED_LITERAL + r"\s+(?:in|within|into)\s+(?:the\s+)?"
               + r"(?P<sentence>" + _ORDINAL + r")\s+sentence\s*[,;:]?\s*"
               + r"(?:as|at)\s+(?:the\s+)?(?P<word>" + _ORDINAL + r")\s+word"
               + r"\s+(?:of|in)\s+(?:that|this|the)\s+sentence\b", re.IGNORECASE),
    re.compile(_INTRO + _QUOTED_LITERAL + r"\s+(?:as|at)\s+(?:the\s+)?"
               + r"(?P<word>" + _ORDINAL + r")\s+word\s+(?:in|of)\s+(?:the\s+)?"
               + r"(?P<sentence>" + _ORDINAL + r")\s+sentence\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?(?P<word>" + _ORDINAL + r")\s+word\s+(?:of|in)\s+"
               + r"(?:the\s+)?(?P<sentence>" + _ORDINAL + r")\s+sentence\s+"
               + r"(?:must|should)\s+be\s+" + _QUOTED_LITERAL + r"(?=$|[\s.!?])", re.IGNORECASE),
]
_INTRO_SCAN = re.compile(_INTRO, re.IGNORECASE)
_REVERSE_SCAN = re.compile(
    r"\b(?:the\s+)?[\w-]+\s+word\s+(?:of|in)\s+(?:the\s+)?[\w-]+\s+"
    r"sentence\s+(?:must|should)\s+be\s+", re.IGNORECASE
)
_POSITION_HINT = re.compile(r"\bword\b.*\bsentence\b|\bsentence\b.*\bword\b", re.IGNORECASE)
_ASCII_WORD = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z]+$")]
# No empty sentence, surrounding whitespace, internal terminator or line break.
# Unicode terminal punctuation is excluded too, to avoid chunk disagreements.
_SENTENCE = Annotated[str, Field(strict=True, pattern=(
    r"^[^\s.!?\u3002\uff01\uff1f](?:[^.!?\r\n\u3002\uff01\uff1f]*"
    r"[^\s.!?\u3002\uff01\uff1f])?[.!?]$"
))]


class _WordSlotsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


@dataclass(frozen=True)
class PublicPositionSpan:
    source: str
    start: int
    end: int
    text: str


def independent_positive_position_span(text, span):
    """Accept a standalone positive public rule under a limited textual convention.

    Literal keyword quotes inside the match are allowed. Quoted, negated,
    conditional, scoped and unsupported inline clauses decline construction.
    This conservative scope check is not a general natural-language parser.
    """
    before, after = text[:span.start], text[span.end:]
    if re.split(r"[.!?。！？]", before)[-1].strip():
        return False
    if re.match(r"\s*(?:[.!?。！？]|$)", after) is None:
        return False
    # Literal keyword quotes lie wholly inside the compiler span. Outer quotations,
    # inline code and open fenced blocks are not instructions for this actor.
    quotes = re.finditer(r'"[^"]*"|\u201c[^\u201d]*\u201d|'
                         r"'[^']*'|\u2018[^\u2019]*\u2019|`[^`]*`", text)
    if any(match.start() < span.start < match.end()
           or match.start() < span.end < match.end() for match in quotes):
        return False
    if before.count('"') % 2 or before.count("\u201c") > before.count("\u201d"):
        return False
    if any(before.count(fence) % 2 for fence in ("```", "~~~")):
        return False
    # Markdown permits unmarked lazy continuation lines within a blockquote.
    # A blank line provides a conservative paragraph boundary; do not infer
    # whether an unmarked line ends a still-open quoted paragraph.
    preceding_block = re.split(r"\r?\n[ \t]*\r?\n", before)[-1]
    if re.search(r"(?m)^[ \t]*>", preceding_block):
        return False
    if re.search(r"\b(?:examples?|samples?|quotation|excerpt|template|quoted\s+text|code)"
                 r"\b[^\r\n]*:", before, re.IGNORECASE):
        return False
    # A previous sentence can introduce literal payload without a colon or
    # quote delimiters. Require payload language and an explicit reproduction
    # mode; ordinary instructions to quote sources/references do not suffice.
    payload = r"\b(?:text|passages?|paragraphs?|sentences?|instructions?|wording|lines?|quotations?|excerpts?|templates?|samples?|examples?|messages?)\b"
    reproduce = r"\b(?:copy|repeat|reproduce|preserve|transcribe|render|print|return|output|include|write|keep|retain)\b"
    literal_mode = r"\b(?:verbatim|unchanged|word[\s-]+for[\s-]+word|exactly\s+as\s+(?:written|given|provided|shown)|as\s+(?:a\s+)?quotation)\b|\bexactly\s*$"
    quotation_declaration = r"\b(?:is|are|forms?)\s+(?:(?:a|an)\s+)?(?:quotation|quoted\s+(?:text|passage)|literal\s+(?:text|content))\b"
    for clause in re.split(r"[.!?\u3002\uff01\uff1f\r\n]", before):
        if re.search(payload, clause, re.IGNORECASE) and (
                (re.search(reproduce, clause, re.IGNORECASE)
                 and re.search(literal_mode, clause, re.IGNORECASE))
                or re.search(quotation_declaration, clause, re.IGNORECASE)):
            return False
    if re.search(r"\b(?:only\s+if|unless|provided\s+that|conditional(?:ly)?)\b", text,
                 re.IGNORECASE):
        return False
    return True


@dataclass(frozen=True)
class WordPositionPlan:
    keyword: str
    sentence_index: int
    word_index: int
    matched_public_spans: tuple[PublicPositionSpan, ...]
    model: type[BaseModel]

    @property
    def schema(self) -> dict:
        return self.model.model_json_schema()

    def response_format(self) -> dict:
        return {"type": "json_schema", "json_schema": {
            "name": "PublicWordSlots", "strict": True, "schema": self.schema}}

    def audit(self) -> dict:
        return {
            "version": "public-word-position-slots-v1",
            "keyword": self.keyword,
            "sentence_index_1_based": self.sentence_index,
            "word_index_1_based": self.word_index,
            "matched_public_spans": [vars(span).copy() for span in self.matched_public_spans],
            "applicability": (
                "One explicit matched standalone positive English literal-position rule only; target words "
                "are ASCII-letter tokens separated by single spaces. Sentence slots "
                "have one final . ! or ?, no internal terminators/newlines or abbreviations. "
                "This convention is not an official checker. Other constraints and facts "
                "are not established by the slot schema."
            ),
            "limits": {"sentence_position": MAX_SENTENCE_POSITION,
                       "word_position": MAX_WORD_POSITION},
        }


def _ordinal(text: str) -> int:
    value = text.casefold().strip()
    if value in _ORDINALS:
        return _ORDINALS[value]
    digits = re.match(r"[0-9]+", value).group()
    # Every supported maximum has three digits; decline huge/ambiguous tokens
    # without invoking unbounded integer conversion.
    return int(digits) if len(digits) <= 3 else 0


def _public_sources(task) -> list[tuple[str, str]]:
    if isinstance(task, dict):
        question, constraints = task.get("question", ""), task.get("constraints", [])
    else:
        question, constraints = getattr(task, "question", ""), getattr(task, "constraints", [])
    if not isinstance(question, str) or not isinstance(constraints, (list, tuple)):
        return []
    if not all(isinstance(item, str) for item in constraints):
        return []
    return [("question", question), *[(f"constraints[{index}]", value)
                                      for index, value in enumerate(constraints)]]


def position_plan(task) -> WordPositionPlan | None:
    """Return a dynamic schema only for one independent positive public rule.

    Digit ordinals and English first through twentieth are supported, in two
    imperative word orders and an explicit 'Nth word ... must be' order.
    Unsupported position-like imperatives decline the
    protocol rather than infer a missing literal or ordinal from metadata.
    """
    rules = []
    spans = []
    for source, text in _public_sources(task):
        matches = sorted((match for pattern in _PATTERNS for match in pattern.finditer(text)),
                         key=lambda match: match.start())
        for intro in _INTRO_SCAN.finditer(text):
            clause = re.split(r"[.!?\r\n]", text[intro.start():], maxsplit=1)[0]
            if _POSITION_HINT.search(clause) and not any(match.start() == intro.start() for match in matches):
                return None
        for reverse in _REVERSE_SCAN.finditer(text):
            if not any(match.start() == reverse.start() for match in matches):
                return None
        for match in matches:
            span = PublicPositionSpan(source, match.start(), match.end(), match.group())
            if not independent_positive_position_span(text, span):
                # Do not drop ambiguous matches and select a different public rule.
                return None
            groups = match.groupdict()
            keyword = next(value for key, value in groups.items()
                           if key.endswith("literal") and value is not None)
            sentence, word = _ordinal(groups["sentence"]), _ordinal(groups["word"])
            if not (1 <= sentence <= MAX_SENTENCE_POSITION and 1 <= word <= MAX_WORD_POSITION):
                return None
            rules.append((keyword, sentence, word))
            spans.append(span)
    if not rules or len(set(rules)) != 1:
        return None
    keyword, sentence, word = rules[0]
    model = create_model(
        "PublicWordSlotsResponse", __base__=_WordSlotsResponse,
        preceding_sentences=(list[_SENTENCE], Field(..., min_length=sentence - 1, max_length=sentence - 1)),
        prefix_words=(list[_ASCII_WORD], Field(..., min_length=word - 1, max_length=word - 1)),
        keyword=(Literal[keyword], ...),
        suffix_words=(list[_ASCII_WORD], ...),
        following_sentences=(list[_SENTENCE], ...),
    )
    return WordPositionPlan(keyword, sentence, word, tuple(spans), model)


def render(parsed: BaseModel) -> str:
    """Render a validated slot object; never edit facts or select candidates."""
    if not isinstance(parsed, _WordSlotsResponse) or not type(parsed).model_fields:
        raise TypeError("Validate the response with position_plan(task).model before rendering")
    # Revalidate even if a caller used model_construct or mutated list contents.
    checked = type(parsed).model_validate(parsed.model_dump(mode="python"))
    target = " ".join([*checked.prefix_words, checked.keyword, *checked.suffix_words]) + "."
    return " ".join([*checked.preceding_sentences, target, *checked.following_sentences])


def prepare_prompt_instruction(plan: WordPositionPlan) -> str:
    return (
        "For this revision, the word-slot schema overrides any earlier instruction to return "
        "an answer field. Do not include answer: return only the five required word-slot fields. "
        "Their rendered content must still be the user's complete finished artifact. "
        f"preceding_sentences must contain exactly {plan.sentence_index - 1} complete sentences. "
        f"prefix_words must contain exactly {plan.word_index - 1} ASCII-letter words; "
        f"keyword must be exactly {plan.keyword!r}. suffix_words contains the remaining "
        "ASCII-letter words of that sentence and may be empty. following_sentences contains "
        "the rest of the complete requested artifact and may be empty. Each sentence string "
        "must end with one . ! or ?, with no internal . ! ?, Unicode terminal punctuation, "
        "newlines, abbreviations or surrounding whitespace. Each word slot contains only "
        "A-Z or a-z, without punctuation, digits or spaces. Never use straight or curly "
        "apostrophes, apostrophe possessives, contractions or hyphenated words in "
        "prefix_words or suffix_words. Express possession with an of phrase and spell "
        "contracted words in full, rebuilding the prefix to its required length. In "
        "preceding_sentences and following_sentences, spell names and titles in full "
        "instead of using dotted abbreviations: write Doctor or Professor in full. "
        "A dotted title copied from the draft or critic is still invalid here; "
        "expanding the title preserves the person's identity. Use phrases such as "
        "the belongings of the owner instead of an apostrophe possessive. These "
        "construction requirements take priority over copying draft or review "
        "typography. Before returning, inspect every "
        "prefix_words/suffix_words entry character by character for ASCII letters only, "
        "and every sentence string for one final terminator and none inside. Count "
        f"the actual arrays again: exactly {plan.sentence_index - 1} preceding_sentences "
        f"entries and {plan.word_index - 1} prefix_words entries; do not infer these "
        "counts from fluent prose. The renderer joins target word "
        "slots with spaces, appends a period and joins all sentences with spaces. Preserve "
        "the public request, meaning, supported facts and already satisfied constraints; "
        "write the actual finished artifact, not process notes or a counting explanation. "
        "This construction has a limited counting convention and does not guarantee other "
        "format, length, factual or grammatical requirements."
    )
