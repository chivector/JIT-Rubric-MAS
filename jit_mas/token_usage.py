"""Portable token-usage summaries for formal experiment artifacts.

The provider usage fields are authoritative when they are present on a budget
record.  :class:`~jit_mas.budget.MeteredModel` marks a record as ``estimated``
when the provider did not return usage and the ledger had to settle it against
the request bounds.  This module keeps that distinction visible while making
it easy for experiment controllers to aggregate slot and campaign totals.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping


SCHEMA = "token-usage-v1"


def _int(value: Any, default: int = 0) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def empty_usage(*, unknown: bool = False) -> dict[str, Any]:
    """Return a JSON-safe zero usage record.

    ``usage_unknown`` is reserved for a failed callback which did not expose a
    ledger at all.  A zero-call callback used by an offline controller remains
    distinguishable from a provider call that returned no usage.
    """

    return {
        "schema": SCHEMA,
        "model_calls": 0,
        "actual_model_calls": 0,
        "estimated_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_tokens": 0,
        "unattributed_tokens": 0,
        "usage_unknown": bool(unknown),
        "source": "unknown" if unknown else "none",
        "by_stage": {},
    }


def _source(actual: int, estimated: int, unknown: bool = False) -> str:
    if unknown:
        return "unknown"
    if actual and estimated:
        return "mixed"
    if estimated:
        return "estimated"
    if actual:
        return "provider"
    return "none"


def _add_call(target: dict[str, Any], row: Mapping[str, Any], *, stage: str) -> None:
    input_tokens = _int(row.get("input_tokens"))
    output_tokens = _int(row.get("output_tokens"))
    total = input_tokens + output_tokens
    estimated = bool(row.get("estimated"))
    target["model_calls"] += 1
    target["input_tokens"] += input_tokens
    target["output_tokens"] += output_tokens
    target["total_tokens"] += total
    if estimated:
        target["estimated_model_calls"] += 1
        target["estimated_tokens"] += total
    else:
        target["actual_model_calls"] += 1

    stage_row = target["by_stage"].setdefault(stage, {
        "model_calls": 0,
        "actual_model_calls": 0,
        "estimated_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_tokens": 0,
    })
    stage_row["model_calls"] += 1
    stage_row["input_tokens"] += input_tokens
    stage_row["output_tokens"] += output_tokens
    stage_row["total_tokens"] += total
    if estimated:
        stage_row["estimated_model_calls"] += 1
        stage_row["estimated_tokens"] += total
    else:
        stage_row["actual_model_calls"] += 1


def summarize_budget(budget: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize one ``BudgetLedger.snapshot()``.

    Ledger records with ``estimated=False`` are provider-reported usage.  If a
    legacy or failure budget contains only aggregate ``tokens``/``model_calls``
    fields, those tokens are retained as ``unattributed_tokens`` and marked as
    estimated instead of silently presenting them as exact prompt/completion
    usage.
    """

    if not isinstance(budget, Mapping):
        return empty_usage(unknown=True)
    out = empty_usage()
    records = budget.get("records")
    model_records = [row for row in records or []
                     if isinstance(row, Mapping) and row.get("kind", "model") == "model"]
    if model_records:
        for row in model_records:
            _add_call(out, row, stage=str(row.get("stage") or "unknown"))
    else:
        # Some sealed failure/legacy records intentionally omit per-call rows.
        # Preserve their total and make its uncertainty explicit.
        calls = _int(budget.get("model_calls"))
        tokens = _int(budget.get("tokens"))
        if calls or tokens:
            out["model_calls"] = calls
            out["estimated_model_calls"] = calls
            out["total_tokens"] = tokens
            out["estimated_tokens"] = tokens
            out["unattributed_tokens"] = tokens
            out["source"] = "estimated"
            out["usage_unknown"] = True
            stage = out["by_stage"].setdefault("unknown", {
                "model_calls": 0,
                "actual_model_calls": 0,
                "estimated_model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_tokens": 0,
            })
            stage["model_calls"] = calls
            stage["estimated_model_calls"] = calls
            stage["total_tokens"] = tokens
            stage["estimated_tokens"] = tokens
    out["source"] = _source(out["actual_model_calls"], out["estimated_model_calls"],
                             out["usage_unknown"])
    return out


def merge_usage(usages: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Add token summaries while retaining provider/estimate provenance."""

    out = empty_usage()
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        for key in ("model_calls", "actual_model_calls", "estimated_model_calls",
                    "input_tokens", "output_tokens", "total_tokens",
                    "estimated_tokens", "unattributed_tokens"):
            out[key] += _int(usage.get(key))
        out["usage_unknown"] = out["usage_unknown"] or bool(usage.get("usage_unknown"))
        for stage, row in (usage.get("by_stage") or {}).items():
            if not isinstance(row, Mapping):
                continue
            dst = out["by_stage"].setdefault(stage, {
                "model_calls": 0,
                "actual_model_calls": 0,
                "estimated_model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_tokens": 0,
            })
            for key in dst:
                dst[key] += _int(row.get(key))
    out["source"] = _source(out["actual_model_calls"], out["estimated_model_calls"],
                             out["usage_unknown"])
    return out


def summarize_outcome(outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize generation/evaluation budgets present in an outcome.

    Normal pipeline outcomes expose one combined ``budget``.  Test-release
    scoring outcomes expose separate ``generation_budget`` and
    ``evaluation_budget`` fields, so both are included when present.
    """

    if not isinstance(outcome, Mapping):
        return empty_usage(unknown=True)
    budgets = []
    for key in ("budget", "generation_budget", "evaluation_budget"):
        value = outcome.get(key)
        if isinstance(value, Mapping):
            budgets.append(summarize_budget(value))
    if budgets:
        return merge_usage(budgets)
    existing = outcome.get("token_usage")
    if isinstance(existing, Mapping) and existing.get("schema") == SCHEMA:
        return deepcopy(dict(existing))
    return empty_usage(unknown=True)


def summarize_slot_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Find usage in a formal slot wrapper (or its nested pipeline result)."""

    if not isinstance(result, Mapping):
        return empty_usage(unknown=True)
    direct = result.get("token_usage")
    if isinstance(direct, Mapping) and direct.get("schema") == SCHEMA:
        return deepcopy(dict(direct))
    nested = result.get("outcome")
    return summarize_outcome(nested if isinstance(nested, Mapping) else result)


def usage_totals_from_slots(slots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the terminal token summaries stored in journal slot rows."""

    return merge_usage(summarize_slot_result(row.get("result"))
                       for row in slots if isinstance(row, Mapping)
                       and row.get("status") in {"complete", "incomplete", "failed", "missing"})


__all__ = [
    "SCHEMA", "empty_usage", "summarize_budget", "summarize_outcome",
    "summarize_slot_result", "merge_usage", "usage_totals_from_slots",
]
