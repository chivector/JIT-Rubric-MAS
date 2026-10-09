"""Mechanical diagnostics over the public artifact, never benchmark metadata.

These transparent counting conventions help a writer check its own draft. They
do not establish compliance with a task's unspecified or benchmark-specific
tokenization, sentence segmentation, or counting rules.
"""

from __future__ import annotations

import copy
import re
from collections import Counter

from .evidence import EVIDENCE_TASK_CONSTRAINT, public_instruction_question
from .schemas import digest


_WORDS = re.compile(r"\w+(?:['\u2019-]\w+)*", re.UNICODE)
_QUOTES = re.compile(r'''["\u201c]([^"\u201d\n]{1,64})["\u201d]|[\u2018]([^\u2019\n]{1,64})[\u2019]|`([^`\n]{1,64})`|(?<!\w)'([^'\n]{1,64})'(?!\w)''')
_CHUNKS = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|[\r\n]+")
_ENGLISH_WORDS = re.compile(
    r"(?<!\w)[A-Za-z]+(?:['\u2019-][A-Za-z]+)*(?!\w)", re.UNICODE
)
_NUMERIC_TOKENS = re.compile(
    r"(?<!\w)[+\-\u2212]?"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?"
    r"|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?:[eE][+\-]?[0-9]+)?(?!\w)"
)
_FANBOYS = ("for", "and", "nor", "but", "or", "yet", "so")

MAX_ENGLISH_FREQUENCY_TYPES = 1024
MAX_SENTENCE_CHUNKS = 128
MAX_SENTENCE_TOKEN_ITEMS = 4096
MAX_NUMERIC_TOKEN_ITEMS = 1024
PUBLIC_DIAGNOSTICS_TRANSPORT_VERSION = "public-artifact-diagnostics-transport-v1"
_POSITION_DETAIL_HINT = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|last|[0-9]+\s*-?\s*(?:st|nd|rd|th))\s+"
    r"(?:word|sentence|line|paragraph|character)\b|"
    r"\b(?:word|sentence|line|paragraph|character)\s+(?:position|index|number|[0-9])|"
    r"\b(?:each|every)\s+(?:sentence|line)\b[^\n.!?]{0,100}\bword\b|"
    r"第[^\n。！？]{0,16}[词詞字句段行]|[词詞字句段行][^\n。！？]{0,16}(?:位置|开头|開頭|结尾|結尾)",
    re.IGNORECASE,
)
_FREQUENCY_DETAIL_HINT = re.compile(
    r"\b(?:words?|keywords?|terms?|times?|occurrences?|frequenc\w*|repeat\w*|mention\w*)\b|"
    r"出现|出現|重复|重複|频率|頻率|次数|次數|单词|單詞|词语|詞語|关键词|關鍵詞",
    re.IGNORECASE,
)
_NUMERIC_DETAIL_HINT = re.compile(
    r"\b(?:numbers?|numerals?|digits?|numeric|integers?|decimals?)\b|数字|數字|数值|數值|数词|數詞",
    re.IGNORECASE,
)


def _public_diagnostic_request(task) -> dict:
    question = public_instruction_question(task)
    original = task.question if hasattr(task, "question") else task.get("question", "")
    constraints = task.constraints if hasattr(task, "constraints") else task.get("constraints", [])
    return {"question": question,
            "constraints": [value for value in constraints
                            if question == original or value != EVIDENCE_TASK_CONSTRAINT]}


def _english_frequencies(answer: str) -> dict:
    counts = Counter(match.group().casefold() for match in _ENGLISH_WORDS.finditer(answer))
    # Common words first; lexical ordering makes ties deterministic.
    retained = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
        :MAX_ENGLISH_FREQUENCY_TYPES
    ]
    return {
        "counting_basis": (
            "Standalone ASCII-letter words, casefolded; internal ASCII/curly apostrophes "
            "and hyphens join a word. Unicode word-character boundaries exclude embedded "
            "letters in alphanumeric or CJK identifiers. No stemming or semantic matching."
        ),
        "order": "descending count, then lexical word order",
        "total_word_occurrences": sum(counts.values()),
        "total_distinct_words": len(counts),
        "returned_distinct_words": len(retained),
        "max_returned_distinct_words": MAX_ENGLISH_FREQUENCY_TYPES,
        "truncated": len(retained) < len(counts),
        "omitted_distinct_words": len(counts) - len(retained),
        "frequencies": dict(retained),
    }


def _sentence_tokens(chunks: list[str]) -> dict:
    budget = MAX_SENTENCE_TOKEN_ITEMS
    retained_items = 0
    full_items_in_returned_chunks = 0
    rows = []
    for index, chunk in enumerate(chunks[:MAX_SENTENCE_CHUNKS], 1):
        whitespace = chunk.split()
        unicode_words = _WORDS.findall(chunk)
        full_items_in_returned_chunks += len(whitespace) + len(unicode_words)
        left, right = [], []
        # Share the budget between both token conventions, preserving prefixes.
        for token_index in range(max(len(whitespace), len(unicode_words))):
            if budget <= 0:
                break
            if token_index < len(whitespace):
                left.append(whitespace[token_index])
                budget -= 1
                retained_items += 1
            if token_index < len(unicode_words) and budget > 0:
                right.append(unicode_words[token_index])
                budget -= 1
                retained_items += 1
        rows.append({
            "chunk_index_1_based": index,
            "whitespace_word_count": len(whitespace),
            "unicode_word_count": len(unicode_words),
            "whitespace_tokens": left,
            "unicode_word_tokens": right,
            "whitespace_tokens_truncated": len(left) < len(whitespace),
            "unicode_word_tokens_truncated": len(right) < len(unicode_words),
        })
    return {
        "counting_basis": (
            "Chunks split at . ! ? and Chinese equivalents followed by whitespace, "
            "or at any line boundary. Abbreviations, decimal punctuation and line breaks "
            "can differ from linguistic sentences. Whitespace tokens use str.split and "
            "retain punctuation; Unicode-word tokens use word-character runs, joining "
            "internal apostrophes and hyphens. Array index plus one gives the word position "
            "under that representation, not an official checker position."
        ),
        "token_allocation": (
            "First chunks in source order; within each chunk, interleave whitespace and "
            "Unicode-word token prefixes. Both arrays share one global item budget; "
            "items are lexical tokens, not model-tokenizer tokens."
        ),
        "total_chunks": len(chunks),
        "returned_chunks": len(rows),
        "max_returned_chunks": MAX_SENTENCE_CHUNKS,
        "chunks_truncated": len(rows) < len(chunks),
        "omitted_chunks": len(chunks) - len(rows),
        "returned_token_items": retained_items,
        "max_returned_token_items": MAX_SENTENCE_TOKEN_ITEMS,
        "total_token_items_in_returned_chunks": full_items_in_returned_chunks,
        "token_items_truncated": retained_items < full_items_in_returned_chunks,
        "chunks": rows,
    }


def _numeric_tokens(answer: str) -> dict:
    tokens = []
    count = 0
    for match in _NUMERIC_TOKENS.finditer(answer):
        count += 1
        if len(tokens) < MAX_NUMERIC_TOKEN_ITEMS:
            tokens.append({"token": match.group(), "character_start": match.start(),
                           "character_end_exclusive": match.end()})
    return {
        "counting_basis": (
            "Standalone ASCII-digit numeric literals; optional leading +, ASCII -, or "
            "Unicode minus; optional decimal fraction, including leading-dot decimals; "
            "comma groups of exactly three digits join one literal; optional e/E exponent "
            "with sign joins one literal. Adjacent Unicode word characters exclude "
            "alphanumeric identifiers. Numeric values are not evaluated. This counts "
            "token-like literals, not digit characters or spelled-out number words. "
            "The separate decimal_digit_characters field counts Unicode decimal characters; "
            "its unit and character set differ and cannot replace this numeric-token count. "
            "Dates, ranges, fractions and times are not interpreted as semantic units "
            "and may yield multiple numeric literals. Malformed separators may also "
            "yield separate literals."
        ),
        "total_numeric_tokens": count,
        "returned_numeric_tokens": len(tokens),
        "max_returned_numeric_tokens": MAX_NUMERIC_TOKEN_ITEMS,
        "truncated": len(tokens) < count,
        "omitted_numeric_tokens": count - len(tokens),
        "tokens": tokens,
    }


def _fanboys(answer: str) -> dict:
    counts = Counter(match.group().casefold() for match in _ENGLISH_WORDS.finditer(answer))
    lexical_counts = {word: counts[word] for word in _FANBOYS}
    return {
        "counting_basis": (
            "Case-insensitive standalone lexical occurrences of the seven common English "
            "FANBOYS words under the English-word convention. This does not determine "
            "whether an occurrence functions grammatically as a coordinating conjunction; "
            "for and so, for example, can have other uses."
        ),
        "lexical_occurrence_counts": lexical_counts,
        "different_lexical_types_present": sum(count > 0 for count in lexical_counts.values()),
        "total_lexical_occurrences": sum(lexical_counts.values()),
    }


def public_output_metrics(task, answer: str) -> dict:
    """Describe decoded text using only its public request and constraints."""
    if not isinstance(answer, str):
        raise TypeError("The decoded artifact must be text")
    public_request = _public_diagnostic_request(task)
    request = "\n".join([public_request["question"], *public_request["constraints"]])
    terms = []
    for match in _QUOTES.finditer(request):
        term = next(value for value in match.groups() if value is not None).strip()
        if term and term not in terms:
            terms.append(term)
        if len(terms) == 32:
            break
    # Splitting convention is deliberately named, not treated as an official
    # sentence counter. Abbreviations and decimals may require another rule.
    chunks = [part for part in _CHUNKS.split(answer) if part.strip()]
    quoted_terms = []
    for term in terms:
        entry = {
            "term": term,
            "literal_substring_count": answer.count(term),
            "word_boundary_casefold_count": len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.IGNORECASE)),
        }
        if len(_WORDS.findall(term)) == 1:
            positions = []
            for index, sentence in enumerate(chunks, 1):
                matches = [position for position, word in enumerate(_WORDS.findall(sentence), 1)
                           if word.casefold() == term.casefold()]
                if matches:
                    positions.append({"sentence_chunk": index, "word_positions": matches})
            entry["positions_in_punctuation_or_line_chunks"] = positions[:64]
            entry["position_list_truncated"] = len(positions) > 64
        quoted_terms.append(entry)
    return {
        "schema_version": "public-artifact-diagnostics-v2",
        "scope": "decoded answer only; public request and constraints only; no benchmark checker",
        "counting_basis": "whitespace words use str.split; regex words use Unicode word runs with apostrophe/hyphen joins; chunks split on sentence punctuation followed by whitespace or line boundaries; these conventions may differ from the task's rules",
        "characters": len(answer),
        "non_whitespace_characters": sum(not char.isspace() for char in answer),
        "cjk_characters": len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", answer)),
        "whitespace_word_count": len(answer.split()),
        "regex_word_count": len(_WORDS.findall(answer)),
        "nonempty_lines": sum(bool(line.strip()) for line in answer.splitlines()),
        "blank_line_paragraphs": len([part for part in re.split(r"\n\s*\n", answer) if part.strip()]),
        "punctuation_or_line_chunks": len(chunks),
        "decimal_digit_characters": sum(char.isdecimal() for char in answer),
        "quoted_terms": quoted_terms,
        "english_word_frequencies": _english_frequencies(answer),
        "sentence_chunk_tokens": _sentence_tokens(chunks),
        "numeric_token_diagnostics": _numeric_tokens(answer),
        "fanboys_lexical_diagnostics": _fanboys(answer),
    }


def public_output_metrics_model_input(task, diagnostics: dict) -> dict:
    """Project mechanical details while retaining complete diagnostics in the audit."""
    public_request = _public_diagnostic_request(task)
    request = "\n".join([public_request["question"], *public_request["constraints"]])
    projected = copy.deepcopy(diagnostics)
    selection = {}
    frequencies = projected.get("english_word_frequencies")
    if frequencies is not None:
        detail = bool(_FREQUENCY_DETAIL_HINT.search(request))
        selection["english_word_frequencies"] = "full" if detail else "counts_only"
        if not detail:
            frequencies.pop("frequencies", None)
        frequencies["transport_detail"] = selection["english_word_frequencies"]
    numbers = projected.get("numeric_token_diagnostics")
    if numbers is not None:
        detail = bool(_NUMERIC_DETAIL_HINT.search(request))
        selection["numeric_token_diagnostics"] = "full" if detail else "counts_only"
        if not detail:
            numbers.pop("tokens", None)
        numbers["transport_detail"] = selection["numeric_token_diagnostics"]
    sentences = projected.get("sentence_chunk_tokens")
    if sentences is not None:
        positional = bool(_POSITION_DETAIL_HINT.search(request))
        position = None
        if positional:
            from .public_word_slots import position_plan

            position = position_plan(public_request)
        mode = "requested_word_slots" if position is not None else "full" if positional else "counts_only"
        selection["sentence_chunk_tokens"] = mode
        if mode != "full":
            original_rows = sentences.get("chunks", [])
            sentences["chunks"] = [{key: value for key, value in row.items()
                                     if key not in {"whitespace_tokens", "unicode_word_tokens"}}
                                    for row in original_rows]
            if position is not None:
                row = next((item for item in original_rows
                            if item["chunk_index_1_based"] == position.sentence_index), None)
                slot = {"sentence_chunk_index_1_based": position.sentence_index,
                        "word_index_1_based": position.word_index,
                        "requested_literal": position.keyword,
                        "chunk_observed": row is not None}
                for name in ("whitespace", "unicode_word"):
                    tokens = row.get(name + "_tokens", []) if row is not None else []
                    observed = position.word_index <= len(tokens)
                    slot[name + "_token_observed"] = observed
                    slot[name + "_token"] = tokens[position.word_index - 1] if observed else None
                    count_name = "whitespace_word_count" if name == "whitespace" else "unicode_word_count"
                    slot[count_name] = row.get(count_name) if row is not None else None
                sentences["requested_word_slots"] = [slot]
        sentences["transport_detail"] = mode
    projection = {
        "version": PUBLIC_DIAGNOSTICS_TRANSPORT_VERSION,
        "full_diagnostics_hash": digest(diagnostics),
        "public_request_hash": digest(public_request),
        "detail_selection": selection,
        "selection_basis": "Public question and constraint lexical hints route diagnostic detail; validated rendered fixed evidence is excluded from routing only. They do not classify instructions or establish compliance. Unsupported position hints retain all available token arrays.",
        "omission_policy": "Counts and counting conventions remain. Omitted lexical detail is available in the intact draft and complete audit; omission is not evidence of absence. Source truncation flags describe the complete diagnostics, independently of this projection.",
    }
    projected["transport_projection"] = projection
    projection["model_input_hash"] = digest(projected)
    return projected
