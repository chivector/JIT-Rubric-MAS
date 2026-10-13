"""Per-round experiment telemetry for reproducible evolution curves."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


OPERATION_KINDS = (
    "add", "delete", "prune", "split", "merge", "specialize", "reorganize",
)
METRIC_FIELDS = ("model_calls", "tokens", "tool_calls", "cost")
PROFILE_BEHAVIOR_FIELDS = (
    "role", "capabilities", "prompt", "skills", "preferred_tools",
    "reasoning_strategy", "planning_strategy", "communication", "harness",
    "harness_content", "harness_code",
)
HARNESS_CHANGE_FIELDS = PROFILE_BEHAVIOR_FIELDS[:-2]
CHANGE_DOMAINS = ("harness", "legacy_harness", "memory", "profile_behavior", "harness_policy")


def _as_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _json_value(value: Any) -> Any:
    value = _as_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _budget_metrics(budget: Any) -> dict[str, Any]:
    budget = _as_dict(budget)
    if not isinstance(budget, Mapping):
        return {field: None for field in METRIC_FIELDS}
    return {field: _number(budget.get(field)) for field in METRIC_FIELDS}


def _stage_metrics(budget: Any) -> dict[str, dict[str, Any]]:
    budget = _as_dict(budget)
    stages = budget.get("by_stage") if isinstance(budget, Mapping) else None
    if not isinstance(stages, Mapping):
        return {}
    return {str(stage): _budget_metrics(metrics) for stage, metrics in stages.items()}


def _pool_profiles(snapshot: Any) -> dict[str, Any] | None:
    snapshot = _as_dict(snapshot)
    if snapshot is None:
        return None
    if isinstance(snapshot, Mapping):
        if "agent_pool" in snapshot:
            return _pool_profiles(snapshot["agent_pool"])
        profiles = snapshot.get("profiles")
        if profiles is None and "pool_agent_id" in snapshot:
            profiles = [snapshot]
    elif isinstance(snapshot, Sequence) and not isinstance(snapshot, (str, bytes)):
        profiles = snapshot
    else:
        return None
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        return None
    result = {}
    for index, profile in enumerate(profiles):
        profile = _as_dict(profile)
        if not isinstance(profile, Mapping):
            continue
        identity = profile.get("pool_agent_id", profile.get("agent_id", str(index)))
        result[str(identity)] = profile
    return result


def _content_hash(content: Any) -> str:
    encoded = json.dumps(_json_value(content), ensure_ascii=True,
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _harness_hashes(value: Any, *, include_memory: bool = False) -> dict[str, str] | None:
    """Hash explicit harness content/code or behavioral profile fields.

    Profile fallback excludes learned memory and identity/provenance/version.
    Direct harness maps exclude their top-level version. Their remaining
    content, including memory policy inside harness code/configuration, is exact.
    """
    value = _as_dict(value)
    if value is None:
        return None
    profiles = _pool_profiles(value)
    if profiles is not None:
        hashes = {}
        for identity, profile in profiles.items():
            content = profile.get("harness_content", profile.get("harness_code"))
            if content is None:
                fields = (*HARNESS_CHANGE_FIELDS, "memory") if include_memory else HARNESS_CHANGE_FIELDS
                content = {field: profile[field] for field in fields if field in profile}
            if content is None:
                continue
            hashes[identity] = _content_hash(content)
        return hashes
    if isinstance(value, Mapping):
        hashes = {}
        for identity, content in value.items():
            content = _as_dict(content)
            if isinstance(content, Mapping):
                content = {key: item for key, item in content.items() if key != "version"}
            hashes[str(identity)] = _content_hash(content)
        return hashes
    return None


def _profile_hashes(profiles: dict[str, Any] | None, fields: Sequence[str]) -> dict[str, str] | None:
    if profiles is None:
        return None
    return {identity: _content_hash({field: profile[field] for field in fields if field in profile})
            for identity, profile in profiles.items()}


def _memory_content(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, Mapping):
        return {key: _memory_content(item) for key, item in value.items()
                if key not in {"created_at", "version"}}
    if isinstance(value, list):
        return [_memory_content(item) for item in value]
    return value


def _memory_hashes(profiles: dict[str, Any] | None) -> dict[str, str] | None:
    if profiles is None:
        return None
    return {identity: _content_hash(_memory_content(profile.get("memory", [])))
            for identity, profile in profiles.items()}


def _changed_ids(before: dict[str, str] | None, after: dict[str, str] | None) -> list[str] | None:
    if before is None or after is None:
        return None
    return sorted(identity for identity in set(before) & set(after) if before[identity] != after[identity])


def call_statistics(budget: Any) -> dict[str, Any]:
    """Distinguish ledger reservations from audited physical provider attempts."""
    budget = _as_dict(budget)
    records = budget.get("records") if isinstance(budget, Mapping) else None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return {"api_attempts": None, "failed_api_attempts": None,
                "retry_count": None, "failed_calls": None}
    models = [record for record in records if isinstance(record, Mapping)
              and record.get("kind") == "model"]
    attempts = failed = retries = 0
    known = True
    for record in models:
        request = record.get("request")
        retry = request.get("transport_retry") if isinstance(request, Mapping) else None
        audit = retry.get("attempts") if isinstance(retry, Mapping) else None
        if not isinstance(audit, list):
            known = False
            continue
        attempts += len(audit)
        retries += max(0, len(audit) - 1)
        failed += sum(item.get("status") in {"retry", "failed"}
                      for item in audit if isinstance(item, Mapping))
    if (isinstance(budget, Mapping) and _number(budget.get("model_calls")) is not None
            and budget["model_calls"] != len(models)):
        known = False
    return {"api_attempts": attempts if known else None,
            "failed_api_attempts": failed if known else None,
            "retry_count": retries if known else None,
            "failed_calls": sum(bool(record.get("error")) for record in models)}


def _operation_kind(operation: Any) -> str | None:
    operation = _as_dict(operation)
    if not isinstance(operation, Mapping):
        return None
    kind = operation.get("kind")
    return str(kind).lower() if kind is not None else None


def _operations(value: Any) -> tuple[list[dict[str, Any]] | None, dict[str, int] | None]:
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, None
    items = []
    counts = Counter()
    for operation in value:
        normalized = _json_value(operation)
        kind = _operation_kind(operation)
        if kind in OPERATION_KINDS:
            counts[kind] += 1
        items.append({
            "operation_id": normalized.get("operation_id") if isinstance(normalized, Mapping) else None,
            "kind": kind,
        })
    return items, {kind: counts.get(kind, 0) for kind in OPERATION_KINDS}


class EvolutionTelemetry:
    """Collect immutable, nullable per-round rows for evolution curves.

    The caller supplies before/after snapshots and separately supplies the
    requested and applied operation lists. Missing measurements remain null;
    zero is emitted only when the caller explicitly supplies an empty list or
    a numeric zero.

    Harness changes exclude learned profile memory. Memory, profile behavior,
    and harness policy are separate measurements on existing members. The
    legacy harness measurement preserves v1 hashing for historical comparison.
    """

    def __init__(self):
        self._rows: list[dict[str, Any]] = []
        self._cumulative = {field: 0 for field in METRIC_FIELDS}
        self._cumulative_known = {field: True for field in METRIC_FIELDS}
        self._stage_cumulative: dict[str, dict[str, Any]] = {}
        self._stage_known: dict[str, dict[str, bool]] = {}
        self._operation_cumulative = {
            label: {kind: 0 for kind in OPERATION_KINDS}
            for label in ("requested", "applied")
        }
        self._operation_known = {label: True for label in ("requested", "applied")}
        self._round_count = 0
        self._failed_rounds = 0
        self._retry_count = 0
        self._retry_known = True
        self._failure_known = True
        self._stages_missing = False
        self._changes = {domain: 0 for domain in CHANGE_DOMAINS}
        self._changes_known = {domain: True for domain in CHANGE_DOMAINS}

    @property
    def rows(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._rows, ensure_ascii=True))

    def record_round(
        self,
        round_index: int,
        *,
        performance: Mapping[str, Any] | None = None,
        budget: Any = None,
        stage_budgets: Mapping[str, Any] | None = None,
        latency_seconds: int | float | None = None,
        pool_before: Any = None,
        pool_after: Any = None,
        operations_requested: Any = None,
        operations_applied: Any = None,
        harness_before: Any = None,
        harness_after: Any = None,
        outcome: Any = None,
    ) -> dict[str, Any]:
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        outcome = _as_dict(outcome)
        if not isinstance(outcome, Mapping):
            outcome = {}
        if budget is None:
            budget = outcome.get("budget")
        if performance is None:
            evaluation = outcome.get("evaluation")
            performance = evaluation if isinstance(evaluation, Mapping) else None
        metrics = _budget_metrics(budget)
        budget_mapping = _as_dict(budget)
        calls = call_statistics(budget)
        failed_calls = calls["failed_calls"]
        retry_count = calls["retry_count"]
        failed_attempts = outcome.get("failed_attempts")
        if isinstance(failed_attempts, Sequence) and not isinstance(failed_attempts, (str, bytes)):
            failed_attempt_count = len(failed_attempts)
        else:
            failed_attempt_count = failed_calls
        complete = outcome.get("complete")
        if complete is None:
            evaluation = outcome.get("evaluation")
            complete = evaluation.get("complete") if isinstance(evaluation, Mapping) else None
        if complete is not None:
            complete = bool(complete)
        failure = (None if complete is None else not complete)
        if retry_count is None:
            retry_count = _number(outcome.get("retry_count"))
        api_calls = calls["api_attempts"]
        if latency_seconds is None:
            budget_value = _as_dict(budget)
            latency_seconds = (budget_value.get("wall_seconds")
                               if isinstance(budget_value, Mapping) else None)
        latency_seconds = _number(latency_seconds)
        stages_available = (stage_budgets is not None or isinstance(budget_mapping, Mapping)
                            and isinstance(budget_mapping.get("by_stage"), Mapping))
        if stage_budgets is None:
            stage_budgets = _stage_metrics(budget)
        else:
            stage_budgets = {str(stage): _budget_metrics(value)
                             for stage, value in stage_budgets.items()}

        requested, requested_counts = _operations(operations_requested)
        applied, applied_counts = _operations(operations_applied)
        before_profiles = _pool_profiles(pool_before)
        after_profiles = _pool_profiles(pool_after)
        before_hashes = _harness_hashes(harness_before if harness_before is not None else pool_before)
        after_hashes = _harness_hashes(harness_after if harness_after is not None else pool_after)
        hash_pairs = {
            "harness": (before_hashes, after_hashes),
            "legacy_harness": (
                _harness_hashes(harness_before if harness_before is not None else pool_before, include_memory=True),
                _harness_hashes(harness_after if harness_after is not None else pool_after, include_memory=True)),
            "memory": (_memory_hashes(before_profiles), _memory_hashes(after_profiles)),
            "profile_behavior": (_profile_hashes(before_profiles, PROFILE_BEHAVIOR_FIELDS),
                                 _profile_hashes(after_profiles, PROFILE_BEHAVIOR_FIELDS)),
            "harness_policy": (_profile_hashes(before_profiles, ("harness",)),
                               _profile_hashes(after_profiles, ("harness",))),
        }
        added_ids = removed_ids = None
        if before_hashes is not None and after_hashes is not None:
            added_ids = sorted(set(after_hashes) - set(before_hashes))
            removed_ids = sorted(set(before_hashes) - set(after_hashes))
        pool_added_ids = pool_removed_ids = None
        if before_profiles is not None and after_profiles is not None:
            pool_added_ids = sorted(set(after_profiles) - set(before_profiles))
            pool_removed_ids = sorted(set(before_profiles) - set(after_profiles))
        change_metrics = {}
        for domain, (before, after) in hash_pairs.items():
            identities = _changed_ids(before, after)
            count = len(identities) if identities is not None else None
            if count is None:
                self._changes_known[domain] = False
            elif self._changes_known[domain]:
                self._changes[domain] += count
            change_metrics.update({
                f"{domain}_hashes_before": before,
                f"{domain}_hashes_after": after,
                f"{domain}_changed_agent_count": count,
                f"{domain}_changed_agent_ids": identities,
                f"cumulative_{domain}_changed_agent_count": (
                    self._changes[domain] if self._changes_known[domain] else None),
            })

        for field, value in metrics.items():
            if value is None:
                self._cumulative_known[field] = False
            elif self._cumulative_known[field]:
                self._cumulative[field] += value
        cumulative = {field: (self._cumulative[field] if self._cumulative_known[field] else None)
                      for field in METRIC_FIELDS}

        cumulative_stages: dict[str, dict[str, Any]] = {}
        for stage in set(self._stage_cumulative) | set(stage_budgets):
            current = stage_budgets.get(stage)
            if stage not in self._stage_cumulative:
                self._stage_cumulative[stage] = {field: 0 for field in METRIC_FIELDS}
                self._stage_known[stage] = {field: not self._stages_missing for field in METRIC_FIELDS}
            totals = self._stage_cumulative[stage]
            known = self._stage_known[stage]
            for field in METRIC_FIELDS:
                value = current.get(field) if current is not None else (0 if stages_available else None)
                if value is None:
                    known[field] = False
                elif known[field]:
                    totals[field] += value
            cumulative_stages[stage] = {field: (totals[field] if known[field] else None)
                                        for field in METRIC_FIELDS}
        self._stages_missing = self._stages_missing or not stages_available

        for label, counts in (("requested", requested_counts), ("applied", applied_counts)):
            if counts is None:
                self._operation_known[label] = False
            elif self._operation_known[label]:
                for kind, count in counts.items():
                    self._operation_cumulative[label][kind] += count

        self._round_count += 1
        if failure is True:
            self._failed_rounds += 1
        if failure is None:
            self._failure_known = False
        if retry_count is not None:
            self._retry_count += retry_count
        else:
            self._retry_known = False
        cumulative_failure_rate = (self._failed_rounds / self._round_count
                                    if self._round_count and self._failure_known else None)

        row = {
            "telemetry_schema_version": "evolution-telemetry-v2",
            "round_index": round_index,
            "performance": _json_value(performance) if performance is not None else None,
            "complete": complete,
            "failure": failure,
            "error": outcome.get("error") if outcome.get("error") is not None else None,
            "latency_seconds": latency_seconds,
            "api_calls": api_calls,
            "api_attempts": api_calls,
            "failed_api_attempts": calls["failed_api_attempts"],
            "failed_calls": failed_calls,
            "failed_attempts": failed_attempt_count,
            "retry_count": retry_count,
            "cumulative_failed_rounds": self._failed_rounds if self._failure_known else None,
            "cumulative_retry_count": self._retry_count if self._retry_known else None,
            "cumulative_failure_rate": cumulative_failure_rate,
            "model_calls": metrics["model_calls"],
            "tokens": metrics["tokens"],
            "tool_calls": metrics["tool_calls"],
            "cost": metrics["cost"],
            "cost_known": metrics["cost"] is not None,
            "cumulative": cumulative,
            "by_stage": stage_budgets or None,
            "cumulative_by_stage": cumulative_stages or None,
            "pool_agent_count_before": len(before_profiles) if before_profiles is not None else None,
            "pool_agent_count_after": len(after_profiles) if after_profiles is not None else None,
            "pool_added_agent_ids": pool_added_ids,
            "pool_removed_agent_ids": pool_removed_ids,
            "operations_requested": requested,
            "operations_applied": applied,
            "operation_counts_requested": requested_counts,
            "operation_counts_applied": applied_counts,
            "operation_counts_cumulative_requested": (
                self._operation_cumulative["requested"] if self._operation_known["requested"] else None),
            "operation_counts_cumulative_applied": (
                self._operation_cumulative["applied"] if self._operation_known["applied"] else None),
            "change_hash_content": {
                "harness": {"explicit_precedence": ["harness_content", "harness_code"],
                            "profile_fallback_fields": list(HARNESS_CHANGE_FIELDS),
                            "direct_map_excluded_fields": ["version"]},
                "legacy_harness": {"definition": "v1 harness hash retained for historical comparison",
                                   "profile_fallback_fields": [*HARNESS_CHANGE_FIELDS, "memory"]},
                "memory": {"profile_fields": ["memory"], "recursive_excluded_fields": ["created_at", "version"],
                           "lesson_provenance_included": True},
                "profile_behavior": {"profile_fields": list(PROFILE_BEHAVIOR_FIELDS)},
                "harness_policy": {"profile_fields": ["harness"]},
                "comparison": "Existing member intersection only; added and removed members are separate",
            },
            "harness_added_agent_ids": added_ids,
            "harness_removed_agent_ids": removed_ids,
            **change_metrics,
        }
        self._rows.append(json.loads(json.dumps(row, ensure_ascii=True)))
        return json.loads(json.dumps(row, ensure_ascii=True))

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        return path

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in self._rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=True, sort_keys=True)
                                 if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
        return path
