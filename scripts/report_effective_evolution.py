"""Export successful, integrity-checked evolution trajectories from a campaign.

The campaign database keeps every slot (including incomplete and failed calls).
This view intentionally filters the *effective* trajectory to completed model
receipts and valid candidate snapshots, while preserving raw slot counts and
the reason a candidate was excluded.  It is meant for the paper-facing Pool
and harness evolution tables and can be run while a campaign is still being
filled.

Examples
--------
    python -m scripts.report_effective_evolution --campaign tb_live_fallback
    python -m scripts.report_effective_evolution --campaign tds_live \
        --source writingbench --run-id 0 --output outputs/effective.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jit_mas.schemas import digest
from jit_mas.agent_pool import seed_pool


TERMINAL = {"complete", "incomplete", "failed", "missing"}
PROFILE_FIELDS = (
    "role", "capabilities", "prompt", "skills", "reasoning_strategy",
    "planning_strategy", "communication", "harness",
)
PROFILE_BEHAVIOR_FIELDS = tuple(field for field in PROFILE_FIELDS if field != "harness")
OPERATION_KINDS = ("add", "delete", "prune", "split", "merge", "specialize", "reorganize")


def _json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _budget_summary(budget: Any) -> dict[str, Any]:
    """Return stable aggregate telemetry without copying large call records."""
    budget = budget if isinstance(budget, dict) else {}
    by_stage = budget.get("by_stage") if isinstance(budget.get("by_stage"), dict) else {}
    stage_tokens = {}
    stage_calls = {}
    for stage, row in by_stage.items():
        if not isinstance(row, dict):
            continue
        stage_tokens[str(stage)] = _safe_int(row.get("tokens"))
        stage_calls[str(stage)] = _safe_int(row.get("model_calls"))
    records = budget.get("records") if isinstance(budget.get("records"), list) else []
    # A successful model receipt may contain transport retries.  Count the
    # physical attempts as well as the logical model call so the efficiency
    # view does not report a falsely perfect failure rate.
    retry_count = 0
    retry_failures = 0
    for row in records:
        if not isinstance(row, dict):
            continue
        retry = (row.get("request") or {}).get("transport_retry")
        attempts = retry.get("attempts") if isinstance(retry, dict) else None
        if not isinstance(attempts, list):
            continue
        retry_count += max(0, len(attempts) - 1)
        retry_failures += sum(
            1 for attempt in attempts[:-1]
            if isinstance(attempt, dict) and str(attempt.get("status", "")).lower() not in {"", "success", "ok"}
        )
    logical_failures = sum(1 for row in records if isinstance(row, dict) and row.get("error"))
    api_failures = logical_failures + retry_failures
    model_calls = _safe_int(budget.get("model_calls"))
    api_attempts = (model_calls + retry_count) if model_calls is not None else None
    return {
        "tokens": _safe_int(budget.get("tokens")),
        "model_calls": model_calls,
        "api_attempts": api_attempts,
        "api_failures": api_failures,
        "api_retry_count": retry_count,
        "api_failure_rate": (api_failures / api_attempts) if api_attempts else None,
        "tool_calls": _safe_int(budget.get("tool_calls")),
        "wall_seconds": budget.get("wall_seconds"),
        "cost": budget.get("cost"),
        "reserved_tokens": _safe_int(budget.get("reserved_tokens")),
        "stage_tokens": stage_tokens,
        "stage_model_calls": stage_calls,
    }


def _slot_axes(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": body.get("source"),
        "run_id": body.get("run_id"),
        "batch_index": body.get("batch_index"),
        "candidate_index": body.get("candidate_index"),
        "kind": body.get("kind"),
        "task_id": body.get("task_id"),
        "method": body.get("method"),
        "target": body.get("target"),
    }


def _profile_map(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    pool = (snapshot or {}).get("agent_pool") or {}
    rows = pool.get("profiles") or []
    return {
        str(row.get("pool_agent_id")): row for row in rows
        if isinstance(row, dict) and row.get("pool_agent_id")
    }


def _pool_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    known = isinstance(snapshot, dict)
    pool = (snapshot or {}).get("agent_pool") or {}
    profiles = pool.get("profiles") or []
    history = pool.get("structural_history") or []
    operations = Counter(str(row.get("kind")) for row in history if isinstance(row, dict))
    return {
        "known": known,
        "snapshot_version": (snapshot or {}).get("version"),
        "pool_version": pool.get("version"),
        "agent_count": len([row for row in profiles if isinstance(row, dict)]),
        "agent_ids": sorted(str(row.get("pool_agent_id")) for row in profiles if isinstance(row, dict) and row.get("pool_agent_id")),
        "profile_versions": {str(row.get("pool_agent_id")): row.get("version") for row in profiles if isinstance(row, dict) and row.get("pool_agent_id")},
        "experience_count": len((snapshot or {}).get("experiences") or []),
        "observation_count": len(pool.get("observations") or []),
        "applied_proposal_count": len((snapshot or {}).get("applied_proposals") or []),
        "structural_history_count": len(history),
        "operation_counts": {kind: int(operations.get(kind, 0)) for kind in OPERATION_KINDS},
    }


def _is_empty_serialized_pool(snapshot: dict[str, Any] | None) -> bool:
    if not isinstance(snapshot, dict):
        return False
    pool = snapshot.get("agent_pool") or {}
    return not any((pool.get("initialized"), pool.get("profiles"), pool.get("version"),
                    pool.get("applied_updates"), pool.get("structural_history"),
                    pool.get("observations")))


def _effective_base_snapshot(snapshot: dict[str, Any] | None,
                            successor: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, bool]:
    """Materialize the six deterministic seed roles for reporting only.

    The serialized checkpoint intentionally starts with an empty persistent
    pool.  Runtime ``effective_pool`` seeds the six roles on first use; that
    initialization is not a structural Add operation and should not appear as
    six newly added agents in a paper-facing report.
    """
    if not _is_empty_serialized_pool(snapshot):
        return snapshot, False
    # Synthetic/minimal fixtures may intentionally contain only a subset of
    # roles.  Materialize the seed only when the successor actually carries
    # the complete deterministic six-role roster used by the runtime.
    seed_ids = {row.pool_agent_id for row in seed_pool().profiles}
    successor_ids = set(_profile_map(successor))
    if not seed_ids <= successor_ids:
        return snapshot, False
    if not isinstance(snapshot, dict):
        return snapshot, False
    materialized = dict(snapshot)
    materialized["agent_pool"] = seed_pool().model_dump(mode="json")
    return materialized, True


def _pool_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {
            "known": False,
            "agent_count_before": None,
            "agent_count_after": len(_profile_map(after)) if isinstance(after, dict) else None,
            "agent_count_delta": None,
            "added_agent_ids": [], "removed_agent_ids": [], "changed_agent_ids": [],
            "changed_profile_fields": {}, "harness_modified_agent_ids": [],
            "harness_modification_count": None, "experience_count_delta": None,
            "profile_behavior_modified_agent_ids": [], "profile_behavior_modification_count": None,
            "observation_count_delta": None, "structural_operation_count_delta": None,
        }
    before_profiles, after_profiles = _profile_map(before), _profile_map(after)
    added = sorted(set(after_profiles) - set(before_profiles))
    removed = sorted(set(before_profiles) - set(after_profiles))
    changed: dict[str, list[str]] = {}
    for ident in sorted(set(before_profiles) & set(after_profiles)):
        fields = [field for field in PROFILE_FIELDS if before_profiles[ident].get(field) != after_profiles[ident].get(field)]
        if fields:
            changed[ident] = fields
    before_pool, after_pool = _pool_summary(before), _pool_summary(after)
    harness_ids = sorted(ident for ident, fields in changed.items() if "harness" in fields)
    behavior_ids = sorted(ident for ident, fields in changed.items() if any(field in PROFILE_BEHAVIOR_FIELDS for field in fields))
    return {
        "known": True,
        "agent_count_before": before_pool["agent_count"],
        "agent_count_after": after_pool["agent_count"],
        "agent_count_delta": after_pool["agent_count"] - before_pool["agent_count"],
        "added_agent_ids": added,
        "removed_agent_ids": removed,
        "changed_agent_ids": sorted(changed),
        "changed_profile_fields": changed,
        "harness_modified_agent_ids": harness_ids,
        "harness_modification_count": len(harness_ids),
        "profile_behavior_modified_agent_ids": behavior_ids,
        "profile_behavior_modification_count": len(behavior_ids),
        "experience_count_delta": after_pool["experience_count"] - before_pool["experience_count"],
        "observation_count_delta": after_pool["observation_count"] - before_pool["observation_count"],
        "structural_operation_count_delta": after_pool["structural_history_count"] - before_pool["structural_history_count"],
    }


def _operation_summary(snapshot: dict[str, Any] | None, decision: dict[str, Any] | None) -> dict[str, Any]:
    pool = (snapshot or {}).get("agent_pool") or {}
    applied = pool.get("structural_history") or []
    proposed = (decision or {}).get("pool_operations") or []
    return {
        "proposed_counts": {kind: sum(1 for row in proposed if isinstance(row, dict) and str(row.get("kind")) == kind) for kind in OPERATION_KINDS},
        "applied_counts": {kind: sum(1 for row in applied if isinstance(row, dict) and str(row.get("kind")) == kind) for kind in OPERATION_KINDS},
        "proposed": [row for row in proposed if isinstance(row, dict)],
        "applied": [row for row in applied if isinstance(row, dict)],
    }


def _valid_snapshot(candidate: dict[str, Any]) -> tuple[bool, str | None]:
    snapshot = candidate.get("snapshot")
    state_hash = candidate.get("state_hash")
    if not isinstance(snapshot, dict):
        return False, "snapshot_missing"
    if not isinstance(state_hash, str) or not state_hash:
        return False, "state_hash_missing"
    try:
        if digest(snapshot) != state_hash:
            return False, "state_hash_mismatch"
    except Exception as exc:  # pragma: no cover - defensive for damaged receipts
        return False, f"snapshot_unhashable:{type(exc).__name__}"
    return True, None


def _read_rows(database: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    database.row_factory = sqlite3.Row
    return list(database.execute(f"SELECT * FROM {table}"))


def _identity(metadata: dict[str, Any]) -> dict[str, Any]:
    execution = metadata.get("execution") or {}
    configuration = execution.get("configuration") or {}
    models = configuration.get("models") or {}
    model_summary = {}
    for role, value in models.items():
        if isinstance(value, dict):
            model_summary[str(role)] = {
                "model": value.get("model"),
                "endpoint": value.get("endpoint"),
                "key_env": value.get("key_env"),
            }
        else:
            model_summary[str(role)] = str(value)
    targeted = execution.get("targeted_optimization")
    if isinstance(targeted, dict):
        targeted = {
            "version": targeted.get("version"),
            "guide_hashes": {
                str(name): len(rows) if isinstance(rows, dict) else None
                for name, rows in (targeted.get("guide_hashes") or {}).items()
            },
        }
    return {
        "schema": metadata.get("schema"),
        "protocol_version": (metadata.get("protocol") or {}).get("version"),
        "initial_snapshot_hash": execution.get("initial_snapshot_hash"),
        "registration_material_sha256": execution.get("joint_manifest_sha256"),
        "code_fingerprint": execution.get("code_fingerprint"),
        "configuration_file_sha256": execution.get("configuration_file_sha256"),
        "launch_file_sha256": execution.get("launch_file_sha256"),
        "runner_sha256": execution.get("runner_sha256"),
        "backend": configuration.get("backend"),
        "execution_mode": configuration.get("execution_mode"),
        "evolving_agent_pool": configuration.get("evolving_agent_pool"),
        "models": model_summary,
        "targeted_optimization": targeted,
    }


def build_report(campaign: Path, *, source: str | None = None, run_id: int | None = None,
                 batch_index: int | None = None, include_noop: bool = False) -> dict[str, Any]:
    database_path = campaign / "campaign.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    migration_manifest = None
    migration_path = campaign / "migration_manifest.json"
    if migration_path.is_file():
        try:
            migration_manifest = _json(migration_path.read_text(encoding="utf-8"), {}) or {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            migration_manifest = {"read_error": True}
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        metadata_row = database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()
        metadata = _json(metadata_row[0], {}) if metadata_row else {}
        slots = _read_rows(database, "slots")
        checkpoints = {row["state_hash"]: _json(row["snapshot"], {}) for row in _read_rows(database, "checkpoints")}
        candidate_rows = _read_rows(database, "candidates") if _table_exists(database, "candidates") else []
        selection_rows = _read_rows(database, "batch_selections") if _table_exists(database, "batch_selections") else []

    selected_keys = set()
    for row in selection_rows:
        body = _json(row["body"], {}) or {}
        selected = body.get("selected") or {}
        if selected:
            selected_keys.add((row["source"], row["run_id"], row["batch_index"], selected.get("candidate_index")))

    slot_counts = Counter()
    receipt_identity_counts = Counter()
    successful_slots: list[dict[str, Any]] = []
    raw_slot_rows: list[dict[str, Any]] = []
    for row in sorted(slots, key=lambda item: int(item["ordinal"])):
        body = _json(row["body"], {}) or {}
        if source is not None and body.get("source") != source:
            continue
        if run_id is not None and body.get("run_id") != run_id:
            continue
        if batch_index is not None and body.get("batch_index") != batch_index:
            continue
        status = str(row["status"])
        kind = body.get("kind")
        slot_counts[f"{kind}:{status}"] += 1
        result = _json(row["result"], {}) or {}
        registration_hash = result.get("registration_hash")
        receipt_identity_counts[registration_hash or "missing"] += 1
        budget = _budget_summary(result.get("budget"))
        raw = {
            **_slot_axes(body),
            "ordinal": row["ordinal"],
            "slot_status": status,
            "complete_receipt": result.get("complete"),
            "score": result.get("score"),
            "execution_seconds": result.get("execution_seconds"),
            "after_snapshot_hash": result.get("after_snapshot_hash"),
            "state_hash": result.get("state_hash"),
            "budget": budget,
            "receipt_registration_hash": registration_hash,
        }
        raw_slot_rows.append(raw)
        if status == "complete" and kind in {"evolution", "validation", "test"} and result.get("complete") is True:
            successful_slots.append(raw)

    candidate_trajectories: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    candidate_operation_totals = Counter()
    candidate_applied_operation_totals = Counter()
    selected_operation_totals = Counter()
    structural_fallbacks: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: (str(item["source"]), int(item["run_id"]), int(item["batch_index"]), int(item["candidate_index"]))):
        if source is not None and row["source"] != source:
            continue
        if run_id is not None and row["run_id"] != run_id:
            continue
        if batch_index is not None and row["batch_index"] != batch_index:
            continue
        candidate = _json(row["body"], {}) or {}
        valid, invalid_reason = _valid_snapshot(candidate)
        selected = (row["source"], row["run_id"], row["batch_index"], row["candidate_index"]) in selected_keys
        no_op = bool(candidate.get("fallback_noop") or candidate.get("no_evidence"))
        effective = valid and candidate.get("status") == "complete" and (include_noop or not no_op)
        base_snapshot_raw = checkpoints.get(candidate.get("base_state_hash"))
        base_snapshot, seeded_base = _effective_base_snapshot(base_snapshot_raw, candidate.get("snapshot"))
        snapshot = candidate.get("snapshot") if valid else None
        trajectory = {
            "source": row["source"], "run_id": row["run_id"], "batch_index": row["batch_index"],
            "candidate_index": row["candidate_index"], "candidate_id": candidate.get("candidate_id"),
            "status": candidate.get("status"), "valid_snapshot": valid, "invalid_reason": invalid_reason,
            "selected": selected, "eligible": candidate.get("eligible"), "effective": effective,
            "fallback_noop": no_op, "fallback_reason": candidate.get("fallback_reason"),
            "no_evidence": bool(candidate.get("no_evidence")),
            "error_type": candidate.get("error_type"), "error": candidate.get("error"),
            "strategy": candidate.get("strategy"),
            "source_task_ids": candidate.get("source_task_ids") or [],
            "base_state_hash": candidate.get("base_state_hash"), "state_hash": candidate.get("state_hash"),
            "base_snapshot_found": isinstance(base_snapshot, dict),
            "base_pool_runtime_seeded": seeded_base,
            "pool_before_raw": _pool_summary(base_snapshot_raw),
            "pool_before": _pool_summary(base_snapshot), "pool_after": _pool_summary(snapshot),
            "pool_delta": _pool_delta(base_snapshot, snapshot) if valid else {},
            "operations": _operation_summary(snapshot, candidate.get("decision")) if valid else {},
            "experience_count": len((snapshot or {}).get("experiences") or []),
            "applied_proposal_count": len((snapshot or {}).get("applied_proposals") or []),
            "budget": _budget_summary(candidate.get("budget")),
            "decision_rationale": (candidate.get("decision") or {}).get("rationale"),
        }
        operation_summary = trajectory.get("operations") or {}
        for kind, count in (operation_summary.get("proposed_counts") or {}).items():
            candidate_operation_totals[str(kind)] += int(count or 0)
        for kind, count in (operation_summary.get("applied_counts") or {}).items():
            candidate_applied_operation_totals[str(kind)] += int(count or 0)
            if selected:
                selected_operation_totals[str(kind)] += int(count or 0)
        fallback = candidate.get("structural_fallback")
        if isinstance(fallback, dict):
            structural_fallbacks.append({
                "source": row["source"], "run_id": row["run_id"],
                "batch_index": row["batch_index"], "candidate_index": row["candidate_index"],
                "selected": selected, "effective": effective,
                "state_hash": candidate.get("state_hash"),
                **fallback,
            })
        if effective:
            candidate_trajectories.append(trajectory)
        else:
            excluded_candidates.append(trajectory)

    # Include native evolution receipts as one-row trajectories, even when no
    # batch candidate exists yet.  These show task-level quality and budgets.
    evolution_curve = []
    for row in successful_slots:
        if row["kind"] != "evolution":
            continue
        base = checkpoints.get(row.get("state_hash") or row.get("after_snapshot_hash"))
        pool = _pool_summary(base)
        evolution_curve.append({
            **{key: row.get(key) for key in ("source", "run_id", "batch_index", "task_id", "ordinal", "score", "execution_seconds")},
            "state_hash": row.get("state_hash"), "pool": pool, "budget": row.get("budget"),
        })

    old_registration_hash = ((migration_manifest or {}).get("old_registration_hash")
                             if isinstance(migration_manifest, dict) else None)
    new_registration_hash = ((migration_manifest or {}).get("new_registration_hash")
                             if isinstance(migration_manifest, dict) else None)
    inherited_receipts = int(receipt_identity_counts.get(old_registration_hash, 0)) if old_registration_hash else 0
    new_receipts = int(receipt_identity_counts.get(new_registration_hash, 0)) if new_registration_hash else 0
    return {
        "schema": "effective-evolution-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "identity": _identity(metadata),
        "migration": migration_manifest,
        "receipt_identity_counts": dict(receipt_identity_counts),
        "inherited_receipts": inherited_receipts,
        "new_receipts": new_receipts,
        "scope": {"source": source, "run_id": run_id, "batch_index": batch_index,
                  "include_noop": include_noop, "effective_definition": "complete + valid snapshot + non-noop"},
        "counts": {
            "slots_in_scope": len(raw_slot_rows),
            "successful_task_receipts": len(successful_slots),
            "effective_candidates": len(candidate_trajectories),
            "excluded_candidates": len(excluded_candidates),
            "selected_effective_candidates": sum(1 for row in candidate_trajectories if row["selected"]),
            "slot_statuses": dict(slot_counts),
        },
        "candidate_operation_totals": {
            kind: int(candidate_operation_totals.get(kind, 0)) for kind in OPERATION_KINDS
        },
        "candidate_applied_operation_totals": {
            kind: int(candidate_applied_operation_totals.get(kind, 0)) for kind in OPERATION_KINDS
        },
        "selected_operation_totals": {
            kind: int(selected_operation_totals.get(kind, 0)) for kind in OPERATION_KINDS
        },
        "structural_fallbacks": structural_fallbacks,
        "successful_task_receipts": successful_slots,
        "effective_trajectories": candidate_trajectories,
        "excluded_candidates": excluded_candidates,
        "evolution_curve": evolution_curve,
    }


def _table_exists(database: sqlite3.Connection, table: str) -> bool:
    return database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def write_outputs(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stem = output.with_suffix("")
    rows = report.get("effective_trajectories") or []
    fields = ["source", "run_id", "batch_index", "candidate_index", "candidate_id", "selected", "state_hash",
              "agent_count_before", "agent_count_after", "agent_count_delta", "harness_modification_count",
              "structural_operation_count_delta", "harness_modification_count", "profile_behavior_modification_count",
              "tokens", "model_calls", "api_attempts",
              "api_failures", "api_failure_rate", "execution_seconds"]
    with stem.with_name(stem.name + "_trajectories.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {**row, **(row.get("pool_delta") or {}), **(row.get("budget") or {})}
            writer.writerow({field: flat.get(field) for field in fields})
    receipt_fields = ["source", "run_id", "batch_index", "kind", "task_id", "ordinal", "score", "execution_seconds", "slot_status", "complete_receipt", "tokens", "model_calls", "api_attempts", "api_failures", "api_failure_rate"]
    with stem.with_name(stem.name + "_receipts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=receipt_fields, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("successful_task_receipts") or []:
            writer.writerow({**row, **(row.get("budget") or {})})
    structural_fields = [
        "source", "run_id", "batch_index", "candidate_index", "selected", "effective",
        "kind", "parent_agent_id", "child_agent_id", "evidence_count", "trigger",
        "state_hash", "source_task_ids",
    ]
    with stem.with_name(stem.name + "_structural_operations.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=structural_fields, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("structural_fallbacks") or []:
            writer.writerow({**row, "source_task_ids": json.dumps(row.get("source_task_ids") or [], ensure_ascii=False)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--include-noop", action="store_true")
    args = parser.parse_args()
    report = build_report(args.campaign.resolve(), source=args.source, run_id=args.run_id,
                          batch_index=args.batch_index, include_noop=args.include_noop)
    output = args.output or (args.campaign / "effective_evolution.json")
    write_outputs(report, output.resolve())
    print(json.dumps({"output": str(output.resolve()), "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
