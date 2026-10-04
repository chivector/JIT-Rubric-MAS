"""Public artifact counts and literal arithmetic observations, never a grader.

Exact task quotations only establish substring presence, not requirement scope
or entailment. Arithmetic only evaluates printed decimal relations, not sources,
units, candidate membership or claims of independent verification.
"""

from __future__ import annotations

from decimal import Decimal
import operator
import re
import unicodedata

from .evidence import public_instruction_question
from .schemas import digest


PUBLIC_REVIEW_OBSERVATION_VERSION = "public-review-observations-v1"
MAX_COMPARISONS = 128
_NUMBER = r"[+-]?[0-9]{1,32}(?:\.[0-9]{1,32})?"
_RELATION = re.compile(
    r"(?=(?P<expression>(?<![\w.,+\-\u2212/*^%\u00d7\u00f7])(?P<left>" + _NUMBER + r")"
    r"\s*(?P<operator><=|>=|==|!=|<|>|=|\u2264|\u2265|\u2260)\s*"
    r"(?P<right>" + _NUMBER + r")(?![\w+\-\u2212/*^%\u00d7\u00f7]|\.[0-9]|,\s*[0-9])))")
_OPERATORS = {"<": operator.lt, ">": operator.gt, "<=": operator.le,
              ">=": operator.ge, "\u2264": operator.le, "\u2265": operator.ge,
              "=": operator.eq, "==": operator.eq, "!=": operator.ne, "\u2260": operator.ne}
_TASK_QUOTE = re.compile(r"\[TASK_QUOTE\]([\s\S]*?)\[/TASK_QUOTE\]")
_LAYOUT_TRANSFORM = re.compile(
    r"\b(?:single|one)\s+(?:line|paragraph)\b|\b(?:remove|strip|delete|omit|"
    r"eliminate|without|no)\b[^\n.!?]{0,64}\b(?:line\s*breaks?|newlines?|paragraphs?)\b|"
    r"\b(?:code|csv|tsv|json|xml|html|table|format|reformat|minify|serialize|verbatim|"
    r"paragraphs?|lines?|stanzas?|sentences?|headings?|bullets?|outline|markdown|"
    r"spacing|whitespace|compact|collapse|wrap(?:ped)?)\b|"
    r"(?:单行|一行|单段|一段|不要分段|不分段|不要换行|不换行|去掉换行|删除换行|"
    r"去除换行|代码|表格|格式|原样|逐字|段落|自然段|分段|换行|分行|空白|空格|"
    r"缩进|紧凑|压缩|一句|句子|标题|列表|分点|诗节|诗行)|(?:1|一|单|一个)\s*(?:段|行)", re.IGNORECASE)


def artifact_observations(answer):
    if not isinstance(answer, str):
        return {"status": "unknown", "reason": "Decoded artifact is not text"}
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n")
    stripped = normalized.strip()
    return {
        "status": "observed", "version": PUBLIC_REVIEW_OBSERVATION_VERSION,
        "artifact_hash": digest(answer), "decoded_characters": len(answer),
        "non_whitespace_characters": sum(not char.isspace() for char in answer),
        "han_ideograph_characters": sum(
            unicodedata.name(char, "").startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-"))
            for char in answer),
        "unicode_database_version": unicodedata.unidata_version,
        "nonempty_lines": sum(bool(line.strip()) for line in normalized.splitlines()),
        "paragraphs_separated_by_blank_lines": len(re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", stripped)) if stripped else 0,
        "line_break_events": normalized.count("\n"),
        "counting_basis": "Whole decoded artifact, including headings, contacts and notes; CRLF/CR normalize only for line/blank-line counts. Han ideographs are assigned Unicode ideograph characters, not words or tokens. No inferred body-only scope or exact threshold for approximate length.",
        "limitations": "Counts do not establish language, genre, semantic correctness or public requirement satisfaction.",
    }


def numeric_relation_observations(value):
    """Bounded exact Decimal comparisons of finite, printed literal relations."""
    rows = []
    truncated = False

    def walk(item, path):
        nonlocal truncated
        if isinstance(item, str):
            for match in _RELATION.finditer(item):
                prefix = item[:match.start("expression")].rstrip()
                suffix = item[match.end("expression"):].lstrip()
                if (prefix.endswith(("+", "-", "\u2212", "/", "*", "^", "%", "\u00d7", "\u00f7"))
                        or suffix.startswith(("+", "-", "\u2212", "/", "*", "^", "%", "\u00d7", "\u00f7"))
                        or (match["left"].startswith(("+", "-"))
                            and prefix and (prefix[-1].isalnum() or prefix[-1] in ")]"))
                        or re.search(r"[0-9]\s*[:,]$", prefix)
                        or re.match(r":\s*[+-]?[0-9]", suffix)):
                    continue
                if len(rows) == MAX_COMPARISONS:
                    truncated = True
                    return
                left, right = Decimal(match["left"]), Decimal(match["right"])
                rows.append({"path": path, "text_hash": digest(item),
                             "start": match.start("expression"), "end": match.end("expression"),
                             "expression": match["expression"], "left": match["left"],
                             "operator": match["operator"], "right": match["right"],
                             "arithmetic_truth": _OPERATORS[match["operator"]](left, right)})
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, path + [index])
                if truncated:
                    return
        elif isinstance(item, dict):
            for key, child in item.items():
                walk(child, path + [str(key)])
                if truncated:
                    return

    walk(value, [])
    return {"version": PUBLIC_REVIEW_OBSERVATION_VERSION, "comparisons": rows,
            "truncated": truncated, "maximum_comparisons": MAX_COMPARISONS,
            "false_arithmetic_relations": sum(not row["arithmetic_truth"] for row in rows),
            "limitations": "Only the printed finite decimal expressions are computed. Unsupported syntax and missing expressions remain unobserved. Units, entity binding, dates, quotation context, original thresholds and membership are not verified. A false relation can be correctly quoted as false; it is not automatically a defect."}


def review_basis_observations(task, review):
    question = public_instruction_question(task)
    constraints = task.constraints if hasattr(task, "constraints") else task.get("constraints", [])
    sources = [("question", question), *[(f"constraints[{i}]", text) for i, text in enumerate(constraints)]]
    records = []
    for index, issue in enumerate(review.get("issues", [])):
        basis = issue.get("public_basis", "")
        anchors = []
        for match in _TASK_QUOTE.finditer(basis):
            quote = match[1]
            hit = None
            if 8 <= len(quote) <= 512:
                for source, text in sources:
                    offset = text.find(quote)
                    if offset >= 0:
                        hit = {"source": source, "start": offset, "end": offset + len(quote)}
                        break
            anchors.append({"quote_hash": digest(quote), "characters": len(quote),
                            "state": "exact_public_substring" if hit else "unverified", "match": hit})
        records.append({"issue_index": index, "task_quote_observations": anchors,
                        "quote_state": "unknown" if not anchors else
                            "exact_public_substring" if any(a["match"] for a in anchors) else "unverified"})
    return {"version": PUBLIC_REVIEW_OBSERVATION_VERSION, "issues": records,
            "numeric_relations": numeric_relation_observations(review),
            "limitations": "Exact quote presence does not prove it is a positive instruction or entails the critique. Missing/unverified task anchors do not invalidate a factual correction based on public evidence. This is not a hard membership contract or evaluator."}


def empty_review_line_break_regression(task, draft, revision, review):
    """Detect public layout loss without acting as a semantic checker.

    The original guard only retained a draft when an empty review was followed
    by a byte-for-byte body with CR/LF removed.  In practice a structured JSON
    revision can validly parse while silently flattening a long, multi-paragraph
    artifact even when the review contains issues.  That failure is especially
    costly for writing and format-sensitive tasks.  Keep the strict legacy
    ``regression`` flag, and expose a separate ``severe_regression`` flag for a
    large ordinary artifact whose final body is reduced to one line.  The caller
    can retain the initial artifact for this narrow structural failure; no
    evaluator or private feedback is consulted.
    """
    question = public_instruction_question(task)
    constraints = task.constraints if hasattr(task, "constraints") else task.get("constraints", [])
    scope_known = not any(_LAYOUT_TRANSFORM.search(text) for text in [question, *constraints])
    observations = artifact_observations(draft)
    revision_observations = artifact_observations(revision)
    qualifies = (scope_known and review.get("issues") == []
                 and isinstance(draft, str) and isinstance(revision, str)
                 and observations.get("paragraphs_separated_by_blank_lines", 0) >= 2
                 and revision_observations.get("nonempty_lines") == 1
                 and draft != revision
                 and draft.replace("\r", "").replace("\n", "") == revision)
    # A parsed revision may alter wording while still destroying the layout of
    # a substantial ordinary answer.  Require a generous size threshold and at
    # least three separated paragraphs so short one-off prose and explicitly
    # formatted outputs retain the historical behavior.
    severe = (scope_known and isinstance(draft, str) and isinstance(revision, str)
              and observations.get("non_whitespace_characters", 0) >= 1000
              and observations.get("paragraphs_separated_by_blank_lines", 0) >= 3
              and observations.get("nonempty_lines", 0) >= 3
              and revision_observations.get("nonempty_lines") == 1
              and draft != revision
              # A tiny title/acknowledgement is handled by the existing
              # catastrophic-body guard.  This branch targets a substantive
              # rewrite whose only clear regression is layout flattening.
              and revision_observations.get("non_whitespace_characters", 0) /
                  max(1, observations.get("non_whitespace_characters", 0)) >= 0.35)
    return {"regression": qualifies, "severe_regression": severe,
            "layout_scope": "ordinary" if scope_known else "unknown",
            "draft": observations, "revision": revision_observations,
            "limitations": "The strict regression flag checks exact CR/LF deletion after an empty review. The severe flag checks only large ordinary multi-paragraph to one-line collapse; it does not assess semantic quality or all public format requirements, and explicit layout conversions suppress it."}
