"""Public, source-bound membership observations without a grader or model call.

This finite sidecar freezes the original request and diagnoses explicit decimal
thresholds against complete tables in the canonical fixed public evidence pack.
It does not infer semantic predicates, certify source truth, identify aliases,
or decide the final answer set. Contributor PASS labels remain unverified claims.
"""

from __future__ import annotations

import csv
import copy
import json
import re
from decimal import Decimal, InvalidOperation

from .evidence import EVIDENCE_QUESTION_HEADER, EVIDENCE_TASK_CONSTRAINT, public_instruction_question
from .schemas import PublicTask, digest


PUBLIC_MEMBERSHIP_VERSION = "public-membership-observations-v1"
PUBLIC_MEMBERSHIP_COMPACT_VERSION = "public-membership-input-compact-v1"
PUBLIC_MEMBERSHIP_ATTENTION_VERSION = "public-membership-attention-v1"
MAX_ATTENTION_CHECKS = 16
MAX_ATTENTION_CONTEXT_CHARACTERS_PER_SIDE = 256
MAX_CONDITIONS = 32
MAX_THRESHOLDS = 32
MAX_TABLES = 4
MAX_ROWS = 128
MAX_CLAIMS = 64
MAX_DECIMAL_CHARACTERS = 32
_DECIMAL = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?\Z", re.ASCII)
_BOUND = re.compile(
    r"(?P<operator>at\s+least|at\s+most|more\s+than|less\s+than|greater\s+than|"
    r">=|<=|>|<|=)\s*(?P<value>[+-]?[0-9]+(?:\.[0-9]+)?)"
    r"(?P<unit>\s*(?:%|(?:percent(?:age)?|kilograms?|kg|meters?|points?|USD|EUR|dollars?)\b))?",
    re.IGNORECASE | re.ASCII)
_OPERATOR = {"at least": ">=", "at most": "<=", "more than": ">",
             "greater than": ">", "less than": "<", ">=": ">=", "<=": "<=",
             ">": ">", "<": "<", "=": "="}
_AMBIGUOUS_SCOPE = re.compile(
    r"\b(?:if|unless|except|without|not|never|no|nor|example|sample|quotation|quoted|suppose|"
    r"hypothetical|alternatively|either|or)\b", re.IGNORECASE | re.ASCII)
_FINER_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"quarter|Q[1-4])\b|\b[12][0-9]{3}[-/][0-9]", re.IGNORECASE | re.ASCII)
_QUOTED = re.compile(r'```[\s\S]*?```|~~~[\s\S]*?~~~|"[^"\n]*"|'
                     r"(?<!\w)'[^'\n]*'(?!\w)|`[^`\n]*`|\u201c[^\u201d]*\u201d")
_ENTITY_HEADERS = {"entity", "name", "item", "geography", "state", "country",
                   "supplier", "organization", "product", "region", "city"}
_UNIT = {"%": "percent", "percent": "percent", "percentage": "percent",
         "pct": "percent", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
         "meter": "meters", "meters": "meters", "point": "points", "points": "points",
         "usd": "USD", "dollar": "USD", "dollars": "USD", "eur": "EUR"}
_LABEL = re.compile(r"\b(PASS|FAIL|UNKNOWN)\b", re.ASCII)
_COMPOUND_UNIT_SUFFIX = re.compile(
    r"\s*(?:[/^*\u00b2\u00b3\u00b7\u00d7]|per\b|square(?:d)?\b|cube(?:d)?\b|"
    r"[-\u2010\u2011\u2013\u2014\u2212][A-Za-z0-9])", re.IGNORECASE | re.ASCII)
_LIMITATIONS = (
    "Original spans freeze wording, not semantic entailment or instruction scope. "
    "Only finite positive decimal thresholds, unique complete-header token bindings, "
    "explicit source units and unambiguous requested year scopes are supported. "
    "Truth is conditional on the provided table and declared provenance, not independent "
    "source verification. No semantic condition, inferred rubric, upstream PASS label, "
    "entity mention, alias, historical applicability or complete membership is certified. "
    "UNKNOWN is neither a failure nor permission to invent a new exclusion."
)


def _span(source, text, start=0, end=None):
    end = len(text) if end is None else end
    selected = text[start:end]
    return {"source": source, "start": start, "end": end,
            "text": selected, "text_hash": digest(selected)}


def _public(task):
    if isinstance(task, PublicTask):
        return task.model_dump(mode="json")
    if not isinstance(task, dict):
        raise TypeError("Membership observations require PublicTask or its public object dump")
    # Never inspect arbitrary evaluator/metadata fields, including for diagnostics.
    public = {key: task.get(key) for key in ("task_id", "question", "constraints")}
    if not isinstance(public["question"], str):
        raise TypeError("Public question must be text")
    if not isinstance(public["constraints"], (list, tuple)):
        public["constraints"] = []
    public["constraints"] = [value for value in public["constraints"] if isinstance(value, str)]
    return public


def _decimal(value):
    if not isinstance(value, str) or len(value) > MAX_DECIMAL_CHARACTERS or not _DECIMAL.fullmatch(value):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _parts(text):
    """Keep offsets in raw text, including packs that contain literal \\n delimiters."""
    start = 0
    for separator in re.finditer(r"\\n|\r\n|\n|\r", text):
        yield start, separator.start(), text[start:separator.start()]
        start = separator.end()
    yield start, len(text), text[start:]


def _conditions(question, constraints):
    rows = []
    numbered = list(re.finditer(r"(?m)^[ \t]*[0-9]+[.)][ \t]+", question))
    if numbered:
        if question[:numbered[0].start()].strip():
            rows.append(("question", 0, numbered[0].start(), question))
        for i, item in enumerate(numbered):
            end = numbered[i + 1].start() if i + 1 < len(numbered) else len(question)
            rows.append(("question", item.start(), end, question))
    else:
        # Commas/conjunctions delimit finite numeric clauses, not inferred semantics.
        start = 0
        for separator in re.finditer(r"[,;](?![0-9])|\s+and\s+(?=(?:"
                                      r"had|have|has|whose|with)\b)", question, re.IGNORECASE):
            rows.append(("question", start, separator.start(), question))
            start = separator.end()
        rows.append(("question", start, len(question), question))
    rows.extend((f"constraints[{i}]", 0, len(value), value) for i, value in enumerate(constraints)
                if value != EVIDENCE_TASK_CONSTRAINT)
    conditions = []
    for source, start, end, text in rows:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        conditions.append({"condition_id": f"C{len(conditions) + 1:03}",
                           "original_span": _span(source, text, start, end),
                           "status": "unknown", "reason": "semantic_requirement_not_verified"})
    return conditions[:MAX_CONDITIONS], len(conditions) > MAX_CONDITIONS


def _thresholds(condition, original_text):
    span = condition["original_span"]
    text = span["text"]
    matches = list(_BOUND.finditer(text))
    if not matches:
        return []
    # Do not lose an if/example/quotation scope by splitting at a comma.
    ambiguous = _AMBIGUOUS_SCOPE.search(original_text) or any(
        quoted.start() <= span["start"] + match.start() < quoted.end()
        for quoted in _QUOTED.finditer(original_text) for match in matches)
    result = []
    for i, match in enumerate(matches):
        raw = match["value"]
        before, after = text[:match.start("value")], text[match.end("value"):]
        remainder = text[match.end():]
        following = re.match(r"\s+([A-Za-z]+)", remainder)
        unknown_unit = (not match["unit"] and following and following[1].lower() not in {
            "of", "in", "on", "for", "and", "with", "which", "who", "that", "meet", "meets"})
        compound_unit = bool(match["unit"] and _COMPOUND_UNIT_SUFFIX.match(remainder))
        unsupported = (bool(ambiguous) or _decimal(raw) is None
                       or (before and before[-1] in "./")
                       or bool(re.match(r"(?:[A-Za-z0-9_/]|\.[0-9]|,\s*[0-9])", after))
                       or (match.start() > 0 and text[match.start() - 1] in "!<>=")
                       or (match.start() > 0 and text[match.start() - 1].isalnum())
                       or bool(unknown_unit)
                       or compound_unit
                       or len(matches) != 1)
        operator = _OPERATOR[re.sub(r"\s+", " ", match["operator"].lower())]
        unit = (match["unit"] or "").strip().lower()
        result.append({"condition_id": condition["condition_id"],
                       "threshold_id": f"{condition['condition_id']}.N{i + 1}",
                       "operator": operator, "bound": raw,
                       "requested_unit": _UNIT.get(unit) if unit else None,
                       "original_span": {**_span(span["source"], text, match.start(), match.end()),
                                         "start": span["start"] + match.start(),
                                         "end": span["start"] + match.end()},
                       "condition_text": text,
                       "status": "unknown",
                       "reason": "unsupported_or_ambiguous_threshold" if unsupported else "no_bound_source_cell",
                       "supported": not unsupported})
    return result


def _pack(public):
    original = public_instruction_question(public)
    if original == public["question"]:
        return original, None
    return original, json.loads(public["question"].rpartition(EVIDENCE_QUESTION_HEADER)[2])


def _tokens(text):
    result = set()
    for token in re.findall(r"[A-Za-z]+|[0-9]+", text, re.ASCII):
        token = token.lower()
        token = {"grads": "graduate", "graduates": "graduate"}.get(token, token)
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result


def _cells(line, kind):
    if kind == "markdown":
        if not line.strip().startswith("|") or not line.strip().endswith("|"):
            return None
        pieces = list(re.finditer(r"(?<=\|)[^|]*(?=\|)", line))
        return [(match.group().strip(), match.start(), match.end()) for match in pieces]
    try:
        parsed = next(csv.reader([line], strict=True))
    except (csv.Error, StopIteration):
        return None
    if len(parsed) < 2:
        return None
    ranges, start, quoted = [], 0, False
    for i, character in enumerate(line):
        if character == '"':
            quoted = not quoted
        elif character == "," and not quoted:
            ranges.append((start, i))
            start = i + 1
    ranges.append((start, len(line)))
    if len(parsed) != len(ranges):
        return None
    return [(value.strip(), start, end) for value, (start, end) in zip(parsed, ranges)]


def _tables(source):
    segments = source.get("segments")
    if (source.get("status") != "ok" or source.get("truncated") is not False
            or not isinstance(source.get("source_id"), str) or not source["source_id"].strip()
            or not isinstance(source.get("source_sha256"), str)
            or not re.fullmatch(r"[a-fA-F0-9]{64}", source["source_sha256"])
            or not isinstance(segments, list) or len(segments) != 1):
        return []
    segment = segments[0]
    if (not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
            or segment.get("span_start") != 0
            or segment.get("span_end") != len(segment["text"])):
        return []
    text = segment["text"]
    lines = list(_parts(text))
    result, consumed = [], set()
    for i, (start, end, line) in enumerate(lines):
        if i in consumed:
            continue
        kind = "markdown" if line.strip().startswith("|") else "csv"
        header = _cells(line, kind)
        if (not header or header[0][0].casefold() not in _ENTITY_HEADERS
                or any(not value or not value.isascii() for value, _, _ in header)
                or len({value.casefold() for value, _, _ in header}) != len(header)):
            continue
        rows, incomplete, row_start = [], False, i + 1
        if kind == "markdown":
            if row_start >= len(lines):
                continue
            separator = _cells(lines[row_start][2], kind)
            if not separator or len(separator) != len(header) or not all(
                    re.fullmatch(r":?-{3,}:?", value) for value, _, _ in separator):
                continue
            row_start += 1
        for j in range(row_start, len(lines)):
            a, b, row_line = lines[j]
            if not row_line.strip():
                break
            cells = _cells(row_line, kind)
            if cells is None:
                if "," in row_line or row_line.strip().startswith(("|", '"')):
                    incomplete = True
                break
            if len(cells) != len(header) or not cells[0][0]:
                incomplete = True
                break
            consumed.add(j)
            rows.append({"entity": cells[0][0], "row_span": _span("segment", text, a, b),
                         "cells": [{"value": value, "start": a + c, "end": a + d}
                                   for value, c, d in cells]})
        if not rows:
            continue
        result.append({"format": kind, "source_id": source.get("source_id"),
                       "declared_source_sha256": source.get("source_sha256"),
                       "declared_date": source.get("date"), "locator": source.get("locator"),
                       "segment_hash": digest(text), "segment_index": 0,
                       "header_span": _span("segment", text, start, end),
                       "headers": [value for value, _, _ in header],
                       "header_cells": [{"value": value, "start": start + c, "end": start + d}
                                        for value, c, d in header],
                       "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS,
                       "incomplete": incomplete, "_text": text})
    return result


def _field(table, threshold):
    # A threshold's digits/unit are not evidence for a qualified column name.
    phrase = _BOUND.sub("", threshold["condition_text"])
    tokens = _tokens(phrase)
    matches = []
    for i, header in enumerate(table["headers"][1:], 1):
        field_tokens = _tokens(header) - {"pct", "percent", "percentage", "point"}
        if field_tokens and field_tokens <= tokens:
            matches.append(i)
    return matches[0] if len(matches) == 1 else None


def _unit(table, column):
    header, text = table["headers"][column], table["_text"]
    units = []
    for match in re.finditer(r"\(([^()]*)\)", header):
        unit = _UNIT.get(match[1].strip().lower())
        if unit:
            units.append((unit, {"source": "header", "text": match.group()}))
    pattern = (r"(?<![A-Za-z0-9_])" + re.escape(header)
               + r"\s+(?:uses|is\s+measured\s+in|units?\s*[:=])\s+([^.;\r\n]+)")
    for match in re.finditer(pattern, text, re.IGNORECASE | re.ASCII):
        declaration = match[1].strip().removesuffix(" units").strip()
        unit = _UNIT.get(declaration.lower())
        if unit is None and re.fullmatch(r"[A-Za-z ]+\bpoints?", declaration):
            unit = "points"
        if unit:
            units.append((unit, _span("segment", text, match.start(), match.end())))
    if header.lower().endswith("_pct"):
        for match in re.finditer(r"\bAll\s+_pct\s+columns\s+use\s+percent\s+units\b", text,
                                 re.IGNORECASE | re.ASCII):
            units.append(("percent", _span("segment", text, match.start(), match.end())))
    if not units or len({unit for unit, _ in units}) != 1:
        return None, []
    return units[0][0], [span for _, span in units]


def _date_scope(question, constraints, table):
    text = "\n".join([question, *constraints])
    if _FINER_DATE.search(text):
        return {"status": "unknown", "reason": "unsupported_finer_date_scope"}
    years = set(re.findall(r"\b[12][0-9]{3}\b", text))
    if len(years) > 1:
        return {"status": "unknown", "reason": "ambiguous_requested_year_scope"}
    date = table["declared_date"]
    if not isinstance(date, str) or not date.strip():
        return {"status": "unknown", "reason": "missing_declared_source_scope"}
    if years:
        source_years = set(re.findall(r"\b[12][0-9]{3}\b", date))
        if source_years != years:
            return {"status": "unknown", "reason": "source_requested_year_mismatch",
                    "requested_years": sorted(years), "declared_source_scope": date}
    return {"status": "bound", "requested_years": sorted(years), "declared_source_scope": date,
            "basis": "Declared source scope only; retrieval date is not an observation date and year agreement does not verify historical applicability."}


def _truth(value, operator, bound):
    return {">=": value >= bound, "<=": value <= bound, ">": value > bound,
            "<": value < bound, "=": value == bound}[operator]


def _diagnose_table(table, thresholds, question, constraints, draft):
    scope = _date_scope(question, constraints, table)
    bindings = []
    for threshold in thresholds:
        column = _field(table, threshold) if threshold["supported"] else None
        unit, unit_spans = _unit(table, column) if column is not None else (None, [])
        reason = (threshold["reason"] if not threshold["supported"] else
                  "ambiguous_or_missing_full_header_binding" if column is None else
                  "missing_or_conflicting_explicit_source_unit" if unit is None else
                  "requested_source_unit_mismatch" if threshold["requested_unit"] not in {None, unit} else
                  scope["reason"] if scope["status"] != "bound" else
                  "incomplete_table_rows" if table["incomplete"] else None)
        bindings.append({"threshold_id": threshold["threshold_id"], "condition_id": threshold["condition_id"],
                         "column_index": column, "field": table["headers"][column] if column is not None else None,
                         "unit": unit, "unit_evidence": unit_spans, "operator": threshold["operator"],
                         "bound": threshold["bound"], "status": "bound" if reason is None else "unknown",
                         "reason": reason,
                         "binding_basis": "Unique complete normalized ASCII header-token subset of the original clause; numeric header qualifiers are retained. No semantic entailment or unit conversion is inferred."})
    rows = []
    for row in table["rows"]:
        checks = []
        for binding in bindings:
            if binding["status"] != "bound":
                continue
            cell = row["cells"][binding["column_index"]]
            observed = _decimal(cell["value"])
            truth = (_truth(observed, binding["operator"], _decimal(binding["bound"]))
                     if observed is not None else None)
            checks.append({"threshold_id": binding["threshold_id"], "observed_value": cell["value"],
                           "cell_span": _span("segment", table["_text"], cell["start"], cell["end"]),
                           "arithmetic_truth": truth, "status": "unknown" if truth is None else "pass" if truth else "fail",
                           "reason": "unsupported_source_cell_decimal" if truth is None else "source_bound_decimal_comparison"})
        rows.append({"entity": row["entity"], "entity_cell_span": _span("segment", table["_text"],
                                                                       row["cells"][0]["start"], row["cells"][0]["end"]),
                     "row_span": row["row_span"], "checks": checks,
                     "numeric_status": "fail" if any(check["status"] == "fail" for check in checks) else
                                       "pass" if checks and len(checks) == len(thresholds) and all(check["status"] == "pass" for check in checks) else "unknown",
                     "membership_status": "unknown",
                     "entity_literal_present_in_draft": bool(isinstance(draft, str) and re.search(
                         r"(?<!\w)" + re.escape(row["entity"]) + r"(?!\w)", draft, re.IGNORECASE))})
    return {key: value for key, value in table.items() if key not in {"_text", "rows", "header_cells"}} | {
        "scope": scope, "bindings": bindings, "rows": rows,
        "membership_status": "unknown", "limitations": "Table numeric observations are not an entity-type, semantic-condition, source-truth, alias or complete-membership certificate."}


def _claims(public_materials, draft):
    material = public_materials if isinstance(public_materials, dict) else {}
    selected = []
    contributions = material.get("contributions", [])
    if isinstance(contributions, list):
        for i, contribution in enumerate(contributions):
            if not isinstance(contribution, dict):
                continue
            ledger = contribution.get("ledger", {})
            if not isinstance(ledger, dict):
                ledger = {}
            refs = ledger.get("source_references", [])
            refs = [{key: ref[key] for key in ("source_id", "locator") if isinstance(ref.get(key), str)}
                    for ref in refs if isinstance(ref, dict)] if isinstance(refs, list) else []
            for key in ("requirements", "outline"):
                values = ledger.get(key, [])
                if isinstance(values, list):
                    selected.extend((f"contributions[{i}].ledger.{key}[{j}]", value, refs)
                                    for j, value in enumerate(values) if isinstance(value, str))
            spans = ledger.get("evidence_spans", [])
            if isinstance(spans, list):
                selected.extend((f"contributions[{i}].ledger.evidence_spans[{j}].text", value["text"],
                                 [{"source_id": value["source_ref"]}] if isinstance(value.get("source_ref"), str) else [])
                                for j, value in enumerate(spans) if isinstance(value, dict) and isinstance(value.get("text"), str))
            if isinstance(contribution.get("answer"), str):
                selected.append((f"contributions[{i}].answer", contribution["answer"], refs))
    if isinstance(draft, str):
        selected.append(("draft", draft, []))
    claims = []
    for path, text, refs in selected:
        for start, end, line in _parts(text):
            labels = list(_LABEL.finditer(line))
            if not labels:
                continue
            claims.append({"claim_span": _span(path, text, start, end), "labels": [label[1] for label in labels],
                           "declared_source_references": refs, "status": "unknown",
                           "reason": "upstream_label_is_not_verified_evidence_or_a_hard_condition"})
    return claims[:MAX_CLAIMS], len(claims) > MAX_CLAIMS


def build_public_membership_observations(task, public_materials=None, draft=None):
    """Return a JSON sidecar from explicit public fields, never private records.

    The input material is the projection returned by ``public_materials``.
    Tool output and raw ledger text cannot manufacture a canonical fixed table.
    Whole-request and original constraint spans remain authoritative; any
    numbered/clause subdivision is a finite presentation convention, not NLP.
    """
    public = _public(task)
    question, pack = _pack(public)
    constraints = public["constraints"]
    conditions, conditions_truncated = _conditions(question, constraints)
    original_texts = {"question": question, **{f"constraints[{i}]": value for i, value in enumerate(constraints)}}
    thresholds = [threshold for condition in conditions
                  for threshold in _thresholds(condition, original_texts[condition["original_span"]["source"]])]
    thresholds_truncated = len(thresholds) > MAX_THRESHOLDS
    thresholds = thresholds[:MAX_THRESHOLDS]
    tables = []
    if pack:
        for source in pack["sources"]:
            if isinstance(source, dict):
                tables.extend(_tables(source))
    table_truncated = len(tables) > MAX_TABLES
    tables = [_diagnose_table(table, thresholds, question, constraints, draft) for table in tables[:MAX_TABLES]]
    claims, claims_truncated = _claims(public_materials, draft)
    return {"version": PUBLIC_MEMBERSHIP_VERSION,
            "original_request": _span("question", question),
            "original_constraints": [_span(f"constraints[{i}]", value) for i, value in enumerate(constraints)
                                     if value != EVIDENCE_TASK_CONSTRAINT],
            "hard_conditions": conditions, "numeric_thresholds": thresholds,
            "fixed_pack_status": "canonical_public_pack" if pack else "unknown",
            "fixed_pack_hash": digest(pack) if pack else None,
            "table_observations": tables, "upstream_claims": claims,
            "draft_hash": digest(draft) if isinstance(draft, str) else None,
            "membership_status": "unknown", "independently_verified": False,
            "truncated": conditions_truncated or thresholds_truncated or table_truncated or claims_truncated or any(table["truncated"] for table in tables),
            "limits": {"conditions": MAX_CONDITIONS, "thresholds": MAX_THRESHOLDS, "tables": MAX_TABLES, "rows_per_table": MAX_ROWS,
                       "claims": MAX_CLAIMS, "decimal_characters": MAX_DECIMAL_CHARACTERS},
            "limitations": _LIMITATIONS}


def _membership_input_anchor(span):
    """Raw offsets and the audit hash replace a repeated source-text copy."""
    result = {key: copy.deepcopy(span[key]) for key in ("source", "start", "end", "text_hash")
              if key in span}
    # Header unit fragments currently have source/text without offset fields.
    if "start" not in span or "end" not in span:
        if "text" in span:
            result["text"] = copy.deepcopy(span["text"])
    return result


def public_membership_model_input(observations, input_format="full"):
    """Project a rich public observation snapshot for the existing model calls.

    ``full`` returns an independent, unchanged copy. ``compact`` retains every
    already-observed row, rule, arithmetic result and UNKNOWN; it neither reruns
    the compiler nor infers an answer set. Save the rich snapshot in the audit
    before requesting a model: ``full_observation_hash`` identifies that snapshot,
    while model-input hashes must be taken over the actual projected payload.
    Original public task, pack and contributor material remain model inputs.
    """
    if input_format not in {"full", "compact"}:
        raise ValueError("Public membership input format must be full or compact")
    if not isinstance(observations, dict):
        raise TypeError("Membership input projection requires a public observation object")
    if input_format == "full":
        return copy.deepcopy(observations)
    expected_keys = {
        "version", "original_request", "original_constraints", "hard_conditions",
        "numeric_thresholds", "fixed_pack_status", "fixed_pack_hash", "table_observations",
        "upstream_claims", "draft_hash", "membership_status", "independently_verified",
        "truncated", "limits", "limitations",
    }
    if set(observations) != expected_keys or observations["version"] != PUBLIC_MEMBERSHIP_VERSION:
        raise ValueError("Compact input requires the supported rich public-membership builder output")
    thresholds = observations["numeric_thresholds"]
    threshold_ids = [threshold["threshold_id"] for threshold in thresholds]
    if len(set(threshold_ids)) != len(threshold_ids):
        raise ValueError("Compact input cannot map duplicate public threshold IDs")
    compact_thresholds = []
    for threshold in thresholds:
        compact_thresholds.append({key: copy.deepcopy(value) for key, value in threshold.items()
                                   if key not in {"condition_text", "original_span"}} | {
            "original_span": _membership_input_anchor(threshold["original_span"])})
    compact_conditions = [{key: copy.deepcopy(value) for key, value in condition.items()
                           if key != "original_span"} | {
                               "original_span": _membership_input_anchor(condition["original_span"])}
                          for condition in observations["hard_conditions"]]
    tables = []
    for table_index, table in enumerate(observations["table_observations"], 1):
        table_id = f"T{table_index:03}"
        bindings = []
        for binding in table["bindings"]:
            bindings.append({key: copy.deepcopy(value) for key, value in binding.items()
                             if key not in {"unit_evidence", "binding_basis"}} | {
                                 "unit_evidence": [_membership_input_anchor(span)
                                                   for span in binding["unit_evidence"]]})
        binding_ids = [binding["threshold_id"] for binding in bindings]
        if binding_ids != threshold_ids:
            raise ValueError("Compact input cannot drop, reorder or invent table threshold bindings")
        rows, unknown_cells = [], []
        for row_index, row in enumerate(table["rows"], 1):
            row_id = f"{table_id}.R{row_index:03}"
            check_ids = [check["threshold_id"] for check in row["checks"]]
            if len(set(check_ids)) != len(check_ids) or not set(check_ids) <= set(threshold_ids):
                raise ValueError("Compact input cannot map duplicate or unknown row check IDs")
            checks = {check["threshold_id"]: check for check in row["checks"]}
            values, truths = [], []
            for threshold_id in threshold_ids:
                check = checks.get(threshold_id)
                if check is None:
                    values.append(None)
                    truths.append(None)
                    continue
                value, truth, status = check["observed_value"], check["arithmetic_truth"], check["status"]
                if (truth is not None and type(truth) is not bool) or status != (
                        "unknown" if truth is None else "pass" if truth else "fail"):
                    raise ValueError("Compact input cannot reinterpret an inconsistent arithmetic check")
                values.append(copy.deepcopy(value))
                truths.append(truth)
                if status == "unknown":
                    unknown_cells.append({"row_id": row_id, "threshold_id": threshold_id,
                                          "reason": check["reason"],
                                          "cell_span": _membership_input_anchor(check["cell_span"])})
            if row["membership_status"] != "unknown":
                raise ValueError("Compact public observations cannot assert verified row membership")
            rows.append([row_id, row["entity"], [row["row_span"]["start"], row["row_span"]["end"]],
                         values, truths, row["numeric_status"], row["entity_literal_present_in_draft"]])
        tables.append({key: copy.deepcopy(value) for key, value in table.items()
                       if key not in {"rows", "bindings", "header_span"}} | {
            "table_id": table_id,
            "header_span": _membership_input_anchor(table["header_span"]),
            "bindings": bindings, "threshold_ids": list(threshold_ids),
            "row_fields": ["row_id", "entity", "source_span", "observed_values", "arithmetic_truth",
                           "numeric_status", "entity_literal_present_in_draft"],
            "rows": rows, "unknown_cells": unknown_cells,
            "coverage": {"observed_rows": len(table["rows"]),
                         "observed_checks": sum(len(row["checks"]) for row in table["rows"]),
                         "all_observed_rows_retained": True}})
    claims = [{key: copy.deepcopy(value) for key, value in claim.items() if key != "claim_span"} | {
        "claim_span": _membership_input_anchor(claim["claim_span"])} for claim in observations["upstream_claims"]]
    return {key: copy.deepcopy(observations[key]) for key in (
        "fixed_pack_status", "fixed_pack_hash", "draft_hash", "membership_status", "independently_verified",
        "truncated", "limits", "limitations")} | {
        "version": PUBLIC_MEMBERSHIP_COMPACT_VERSION,
        "observation_version": observations["version"], "full_observation_hash": digest(observations),
        "original_request": _membership_input_anchor(observations["original_request"]),
        "original_constraints": [_membership_input_anchor(span) for span in observations["original_constraints"]],
        "hard_conditions": compact_conditions, "numeric_thresholds": compact_thresholds,
        "table_observations": tables, "upstream_claims": claims,
        "matrix_convention": (
            "Each row array follows row_fields. observed_values/arithmetic_truth follow threshold_ids; "
            "null truth is UNKNOWN, including an unbound rule or unsupported cell. Missing checks do not "
            "fabricate observed values. TRUE/FALSE are only supplied-source arithmetic within the recorded "
            "field/unit/declared-date scope, not complete eligibility or independently verified history. "
            "All row membership remains UNKNOWN. An exact entity mention is a reason to inspect context, "
            "not an affirmative inclusion claim. Source spans are zero-based Unicode offsets with exclusive "
            "ends in the supplied segment; table/row IDs index the full hashed audit in original order. "
            "Original instruction/source/material span text stays in the original public inputs and rich audit. "
            "No entity, alias or final answer set is inferred by this projection."
        )}


def public_membership_attention_queue(observations, draft):
    """Focus existing public review on conditional FALSE observations.

    Only a bound FALSE cell and an actual literal draft mention enter this
    source-ordered queue. Mention context and original instruction scope remain
    for the reviewer to inspect: no candidate is automatically excluded. The
    full matrix/audit remain authoritative and retain UNKNOWN and unqueued rows.
    The finite queue cap bounds the additional review response, not coverage of
    the original task. Declared-date agreement is not historical verification.
    """
    # Reuse the projection's identity, check/ID and UNKNOWN integrity validation;
    # never compile new conditions or reinterpret the builder's arithmetic.
    public_membership_model_input(observations, "compact")
    if not isinstance(draft, str) or observations["draft_hash"] != digest(draft):
        raise ValueError("Attention checks require the original observed draft snapshot")
    thresholds = {row["threshold_id"]: row for row in observations["numeric_thresholds"]}
    conditions = {row["condition_id"]: row for row in observations["hard_conditions"]}
    items = []
    for table_index, table in enumerate(observations["table_observations"], 1):
        if table["scope"]["status"] != "bound" or table["incomplete"]:
            continue
        bindings = {row["threshold_id"]: row for row in table["bindings"]}
        for row_index, row in enumerate(table["rows"], 1):
            if not row["entity_literal_present_in_draft"]:
                continue
            mention = re.search(r"(?<!\w)" + re.escape(row["entity"]) + r"(?!\w)",
                                draft, re.IGNORECASE)
            if mention is None:
                continue
            line_start = draft.rfind("\n", 0, mention.start()) + 1
            line_end = draft.find("\n", mention.end())
            line_end = len(draft) if line_end < 0 else line_end
            context_start = max(line_start, mention.start() - MAX_ATTENTION_CONTEXT_CHARACTERS_PER_SIDE)
            context_end = min(line_end, mention.end() + MAX_ATTENTION_CONTEXT_CHARACTERS_PER_SIDE)
            row_id = f"T{table_index:03}.R{row_index:03}"
            for check in row["checks"]:
                binding = bindings[check["threshold_id"]]
                threshold = thresholds[check["threshold_id"]]
                if (binding["status"] != "bound" or threshold["supported"] is not True
                        or check["arithmetic_truth"] is not False or check["status"] != "fail"):
                    continue
                items.append({
                    "attention_id": f"{row_id}.{check['threshold_id']}",
                    "row_id": row_id, "entity": row["entity"],
                    "condition_id": threshold["condition_id"],
                    "threshold_id": check["threshold_id"],
                    "original_condition_span": copy.deepcopy(conditions[threshold["condition_id"]]["original_span"]),
                    "numeric_bound_span": copy.deepcopy(threshold["original_span"]),
                    "source": {key: copy.deepcopy(table[key]) for key in (
                        "source_id", "declared_source_sha256", "declared_date", "locator",
                        "segment_hash", "segment_index")},
                    "source_scope": copy.deepcopy(table["scope"]),
                    "row_span": copy.deepcopy(row["row_span"]),
                    "entity_cell_span": copy.deepcopy(row["entity_cell_span"]),
                    "binding": copy.deepcopy(binding), "observation": copy.deepcopy(check),
                    "draft_literal_span": _span("draft", draft, mention.start(), mention.end()),
                    "draft_context_span": _span("draft", draft, context_start, context_end),
                    "draft_context_truncated": context_start > line_start or context_end < line_end,
                    "membership_status": "unknown",
                })
    queued = items[:MAX_ATTENTION_CHECKS]
    return {
        "version": PUBLIC_MEMBERSHIP_ATTENTION_VERSION,
        "full_observation_hash": digest(observations), "draft_hash": digest(draft),
        "original_request": _membership_input_anchor(observations["original_request"]),
        "original_constraints": [_membership_input_anchor(span)
                                 for span in observations["original_constraints"]],
        "fixed_pack_hash": observations["fixed_pack_hash"], "items": queued,
        "draft_context_convention": (
            "Exact original-line text with at most 256 Unicode characters on either side "
            "of the first literal mention; the complete literal is retained. Context is a "
            "reading aid only. Inspect the complete draft for scope, other mentions or "
            "cross-line qualifications; a context window does not certify membership."
        ),
        "coverage": {"total_eligible_checks": len(items), "queued_checks": len(queued),
                     "maximum_queued_checks": MAX_ATTENTION_CHECKS,
                     "truncated": len(items) > len(queued),
                     "unqueued_attention_ids": [item["attention_id"] for item in items[len(queued):]],
                     "full_matrix_retained": True, "complete_membership_verified": False},
        "limitations": (
            "FALSE is arithmetic conditional on supplied field/unit/declared-date/source scope. "
            "A literal mention may be a quotation, exclusion, alias collision or unrelated use. "
            "Original scope and affirmative inclusion require public-context review; quote "
            "presence does not prove either. UNKNOWN is not failure. The capped queue does "
            "not certify exhaustive review, source truth, date applicability or membership."
        ),
    }
