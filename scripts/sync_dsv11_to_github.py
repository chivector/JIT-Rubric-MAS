"""Publish compact, credential-free campaign progress snapshots to GitHub.

The campaign itself owns its ``.r/<campaign>`` directory. This helper only reads its SQLite/status
files and commits a small JSON summary, so raw prompts, API credentials, and
large slot artifacts are never pushed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / ".r" / "dsv11"
OUTPUT = ROOT / "paper" / "experiments" / "live_results" / "dsv11_status.json"
REL_OUTPUT = OUTPUT.relative_to(ROOT).as_posix()


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile without external dependencies."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean_seconds": (sum(values) / len(values)) if values else None,
        "median_seconds": _percentile(values, 0.5),
        "p90_seconds": _percentile(values, 0.9),
    }


def _slot_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract one top-level slot's usage and latency, without nested duplicates.

    ``result["outcome"]["budget"]`` mirrors the top-level budget in many
    campaign versions. Only the top-level ``result["budget"]`` is read here.
    Model records are de-duplicated by call_id because some transports can
    include a repeated retry record.
    """
    budget = result.get("budget") or {}
    records = budget.get("records") or []
    model_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("kind") != "model":
            continue
        call_id = str(record.get("call_id") or f"record-{index}")
        model_records.setdefault(call_id, record)
    unique_records = list(model_records.values())
    input_tokens = sum(int(record.get("input_tokens") or 0) for record in unique_records)
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in unique_records)
    model_calls = budget.get("model_calls")
    if not isinstance(model_calls, (int, float)):
        model_calls = len(unique_records)
    tokens = budget.get("tokens")
    if not isinstance(tokens, (int, float)):
        tokens = input_tokens + output_tokens
    model_latencies = [
        float(record["wall_seconds"])
        for record in unique_records
        if isinstance(record.get("wall_seconds"), (int, float))
    ]
    slot_latency = result.get("execution_seconds")
    if not isinstance(slot_latency, (int, float)):
        slot_latency = budget.get("wall_seconds")
    return {
        "tokens": int(tokens),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_calls": int(model_calls),
        "slot_latency": float(slot_latency) if isinstance(slot_latency, (int, float)) else None,
        "model_latencies": model_latencies,
        "score": result.get("score") if isinstance(result.get("score"), (int, float)) else None,
    }


def _merge_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for complete top-level slots in one source/kind group."""
    scores = [row["score"] for row in rows if row.get("score") is not None]
    slot_latencies = [row["slot_latency"] for row in rows if row.get("slot_latency") is not None]
    model_latencies = [latency for row in rows for latency in row.get("model_latencies", [])]
    return {
        "sample_count": len(rows),
        "tokens": sum(row["tokens"] for row in rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "model_calls": sum(row["model_calls"] for row in rows),
        "score_mean": (sum(scores) / len(scores)) if scores else None,
        "slot_latency": _latency_summary(slot_latencies),
        "model_latency": _latency_summary(model_latencies),
    }


def _official_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Read only the compact official metric map from a completed outcome."""
    raw = ((result.get("outcome") or {}).get("evaluation") or {}).get("raw") or {}
    values = raw.get("official_metrics")
    if not isinstance(values, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _checkpoint_summary(con: sqlite3.Connection) -> dict[str, Any]:
    """Summarize the latest checkpoint for each source/run without reading slot artifacts."""
    latest: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for source, run_id, position, raw in con.execute(
        "select source, run_id, position, snapshot from checkpoints "
        "order by source, run_id, position"
    ):
        try:
            snapshot = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict):
            continue
        key = (str(source), int(run_id))
        latest[key] = (int(position), snapshot)
    rows: list[dict[str, Any]] = []
    for (source, run_id), (position, snapshot) in sorted(latest.items()):
        pool = snapshot.get("agent_pool") or {}
        rows.append(
            {
                "source": source,
                "run_id": run_id,
                "position": position,
                "state_version": snapshot.get("version"),
                "pool_size": len(pool.get("profiles") or []),
                "structural_history_count": len(pool.get("structural_history") or []),
                "applied_updates_count": len(pool.get("applied_updates") or []),
                "applied_proposals_count": len(snapshot.get("applied_proposals") or []),
                "experience_count": len(snapshot.get("experiences") or []),
            }
        )
    latest_overall = max(rows, key=lambda row: (row["position"], row["source"], row["run_id"])) if rows else None
    return {"latest_by_source_run": rows, "latest": latest_overall}


def _judge_summary(con: sqlite3.Connection) -> dict[str, Any]:
    """Expose non-secret Judge configuration/preflight state from registration."""
    try:
        row = con.execute("select body from metadata where key = 'registration'").fetchone()
    except sqlite3.OperationalError:
        return {"configured": False, "preflight_complete": False}
    if not row:
        return {"configured": False, "preflight_complete": False}
    try:
        registration = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return {"configured": False, "preflight_complete": False}
    execution = registration.get("execution") or {}
    configuration = execution.get("configuration") or {}
    judge_config = (configuration.get("models") or {}).get("judge") or {}
    judge_probe = (execution.get("served_models") or {}).get("judge") or {}
    calls = judge_probe.get("calls")
    preflight_complete = bool(
        isinstance(calls, (int, float))
        and calls > 0
        and judge_probe.get("content_nonempty")
        and judge_probe.get("finish_reason")
    )
    return {
        "configured": bool(judge_config),
        "model": judge_config.get("model"),
        "preflight_complete": preflight_complete,
        "preflight_calls": calls if isinstance(calls, (int, float)) else None,
        "preflight_returned_model": judge_probe.get("returned_model"),
    }


def _candidate_metrics(con: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], int | None]]:
    """Summarize candidate decisions and selected winners from batch tables."""
    candidates: defaultdict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    selected_indexes: dict[tuple[str, int, int], int | None] = {}
    try:
        candidate_rows = con.execute(
            "select source, run_id, batch_index, candidate_index, body "
            "from candidates order by source, run_id, batch_index, candidate_index"
        )
        for source, run_id, batch_index, candidate_index, raw in candidate_rows:
            try:
                body = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(body, dict):
                continue
            decision = body.get("decision") or {}
            if not isinstance(decision, dict):
                decision = {}
            pool_operations = decision.get("pool_operations") or []
            agent_updates = decision.get("agent_updates") or []
            candidates[(str(source), int(run_id), int(batch_index))].append(
                {
                    "candidate_index": int(candidate_index),
                    "pool_operation_count": len(pool_operations) if isinstance(pool_operations, list) else 0,
                    "agent_update_count": len(agent_updates) if isinstance(agent_updates, list) else 0,
                    "state_changed": bool(
                        body.get("base_state_hash")
                        and body.get("state_hash")
                        and body.get("base_state_hash") != body.get("state_hash")
                    ),
                    "fallback": bool(body.get("fallback_noop") or body.get("fallback_reason")),
                }
            )
    except sqlite3.OperationalError:
        return [], selected_indexes

    try:
        selection_rows = con.execute(
            "select source, run_id, batch_index, body from batch_selections "
            "order by source, run_id, batch_index"
        )
        for source, run_id, batch_index, raw in selection_rows:
            try:
                body = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(body, dict):
                continue
            selected = body.get("selected") or {}
            index = selected.get("candidate_index") if isinstance(selected, dict) else None
            selected_indexes[(str(source), int(run_id), int(batch_index))] = (
                int(index) if isinstance(index, int) else None
            )
    except sqlite3.OperationalError:
        pass

    summaries: list[dict[str, Any]] = []
    for (source, run_id, batch_index), rows in sorted(candidates.items()):
        selected_index = selected_indexes.get((source, run_id, batch_index))
        selected = next(
            (row for row in rows if row["candidate_index"] == selected_index),
            None,
        )
        summaries.append(
            {
                "source": source,
                "run_id": run_id,
                "batch_index": batch_index,
                "candidate_count": len(rows),
                "pool_operation_count": sum(row["pool_operation_count"] for row in rows),
                "agent_update_count": sum(row["agent_update_count"] for row in rows),
                "state_changed_count": sum(1 for row in rows if row["state_changed"]),
                "fallback_count": sum(1 for row in rows if row["fallback"]),
                "selected_candidate_index": selected_index,
                "selected_pool_operation_count": (
                    selected["pool_operation_count"] if selected is not None else None
                ),
                "selected_agent_update_count": (
                    selected["agent_update_count"] if selected is not None else None
                ),
            }
        )
    return summaries, selected_indexes


def _test_status_summary(con: sqlite3.Connection) -> dict[str, Any]:
    """Keep explicit formal-test progress separate from evolution/validation slots."""
    counts: defaultdict[str, int] = defaultdict(int)
    try:
        rows = con.execute("select status, body from slots")
    except sqlite3.OperationalError:
        return {"registered_slots": 0, "statuses": {}, "complete": False}
    for status, raw_body in rows:
        try:
            body = json.loads(raw_body)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(body, dict) and body.get("kind") == "test":
            counts[str(status)] += 1
    return {
        "registered_slots": sum(counts.values()),
        "statuses": dict(sorted(counts.items())),
        "complete": bool(counts) and counts.get("pending", 0) == 0 and counts.get("started", 0) == 0,
    }


def snapshot() -> dict[str, Any]:
    campaign_name = CAMPAIGN.name
    paper_dir = ROOT / ".r" / f"{campaign_name}_paper"
    status = _json(CAMPAIGN / "status.json", {})
    counts: dict[str, int] = {}
    selections: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    official_grouped: defaultdict[tuple[str, str], defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    checkpoint_summary: dict[str, Any] = {"latest_by_source_run": [], "latest": None}
    judge_summary: dict[str, Any] = {"configured": False, "preflight_complete": False}
    test_summary: dict[str, Any] = {"registered_slots": 0, "statuses": {}, "complete": False}
    candidate_summaries: list[dict[str, Any]] = []
    selected_candidate_ops: dict[tuple[str, int, int], dict[str, int | None]] = {}
    con = sqlite3.connect(CAMPAIGN / "campaign.sqlite", timeout=30)
    try:
        for key, value in con.execute("select status, count(*) from slots group by status"):
            counts[str(key)] = int(value)
        rows = con.execute(
            "select source, run_id, batch_index, body from batch_selections "
            "order by source, run_id, batch_index"
        ).fetchall()
        for source, run_id, batch_index, raw in rows:
            try:
                body = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(body, dict):
                continue
            selected = body.get("selected") or {}
            selections.append(
                {
                    "source": source,
                    "run_id": int(run_id),
                    "batch_index": int(batch_index),
                    "candidate_index": selected.get("candidate_index"),
                    "complete_evaluations": selected.get("complete_evaluations"),
                    "minimum_complete": selected.get("minimum_complete"),
                    "selection_utility": selected.get("selection_utility"),
                    "eligible": selected.get("eligible"),
                }
            )
        candidate_summaries, _selected_indexes = _candidate_metrics(con)
        for summary in candidate_summaries:
            selected_candidate_ops[
                (summary["source"], summary["run_id"], summary["batch_index"])
            ] = {
                "selected_pool_operation_count": summary["selected_pool_operation_count"],
                "selected_agent_update_count": summary["selected_agent_update_count"],
            }
        for selection in selections:
            selected_key = (selection["source"], selection["run_id"], selection["batch_index"])
            selected_metrics = selected_candidate_ops.get(selected_key, {})
            selection.update(selected_metrics)
        for _ordinal, slot_status, raw_body, raw_result in con.execute(
            "select ordinal, status, body, result from slots order by ordinal"
        ):
            if slot_status != "complete" or not raw_result:
                continue
            try:
                body = json.loads(raw_body)
                result = json.loads(raw_result)
            except (TypeError, json.JSONDecodeError):
                continue
            source = body.get("source")
            kind = body.get("kind")
            if not source or not kind or not isinstance(result, dict):
                continue
            grouped[(str(source), str(kind))].append(_slot_metrics(result))
            for metric, value in _official_metrics(result).items():
                official_grouped[(str(source), str(kind))][metric].append(value)
        checkpoint_summary = _checkpoint_summary(con)
        judge_summary = _judge_summary(con)
        test_summary = _test_status_summary(con)
    finally:
        con.close()

    by_source_kind: dict[str, Any] = {}
    official_by_source_kind: dict[str, Any] = {}
    for (source, kind), rows in sorted(grouped.items()):
        key = f"{source}/{kind}"
        by_source_kind[key] = _merge_metrics(rows)
        official = official_grouped.get((source, kind), {})
        if official:
            official_by_source_kind[key] = {
                "sample_count": len(rows),
                "means": {
                    metric: sum(values) / len(values)
                    for metric, values in sorted(official.items())
                    if values
                },
                "metric_sample_counts": {
                    metric: len(values) for metric, values in sorted(official.items()) if values
                },
            }

    # Count only explicit top-level statuses. ``incomplete`` is terminal for
    # progress accounting even though it is not a successful result.
    terminal = sum(counts.get(key, 0) for key in ("complete", "failed", "incomplete", "missing"))
    complete_rows = [row for rows in grouped.values() for row in rows]
    all_slot_latencies = [row["slot_latency"] for row in complete_rows if row.get("slot_latency") is not None]
    all_model_latencies = [latency for row in complete_rows for latency in row.get("model_latencies", [])]

    report = {
        "schema_version": f"{campaign_name}-live-v2",
        "campaign": campaign_name,
        "phase": status.get("phase"),
        "registered_slots": status.get("registered_slots"),
        "statuses": counts,
        "terminal_slots": terminal,
        "active": status.get("active", counts.get("started", 0)),
        "test_feedback_released": bool(status.get("test_feedback_released", False)),
        "test": test_summary,
        "judge": judge_summary,
        "registration_hash": status.get("registration_hash"),
        "batch_selections": selections,
        "candidate_metrics": candidate_summaries,
        "final_report_available": (CAMPAIGN / "full_report.json").exists(),
        "test_report_available": (CAMPAIGN / "test_report.json").exists(),
        "paper_summary_available": any(
            (paper_dir / name).exists()
            for name in ("paper_summary.json", "paper_summary.md")
        ),
        "metrics": {
            "complete_slot_count": len(complete_rows),
            "tokens": {
                "total": sum(row["tokens"] for row in complete_rows),
                "input": sum(row["input_tokens"] for row in complete_rows),
                "output": sum(row["output_tokens"] for row in complete_rows),
                "model_calls": sum(row["model_calls"] for row in complete_rows),
            },
            "latency": {
                "slot": _latency_summary(all_slot_latencies),
                "model": _latency_summary(all_model_latencies),
            },
            "by_source_kind": by_source_kind,
            "official_metrics_by_source_kind": official_by_source_kind,
        },
        "checkpoints": checkpoint_summary,
    }
    return report


def _write(report: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUTPUT)


def publish(report: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    comparable = dict(report)
    comparable.pop("observed_at", None)
    old_comparable = dict(previous or {})
    old_comparable.pop("observed_at", None)
    if comparable != old_comparable:
        report = dict(report)
        report["observed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write(report)
        _run("git", "add", "--", REL_OUTPUT)
        staged = _run("git", "diff", "--cached", "--quiet", "--", REL_OUTPUT, check=False)
        if staged.returncode != 0:
            _run("git", "commit", "--only", "-m", f"chore: update {CAMPAIGN.name} live results", "--", REL_OUTPUT)
    # Retry pushes even when the commit was created during a transient outage.
    pushed = _run("git", "push", "origin", "HEAD:main", check=False)
    if pushed.returncode != 0:
        raise RuntimeError((pushed.stderr or pushed.stdout).strip()[-1000:])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    previous: dict[str, Any] | None = _json(OUTPUT, None)
    while True:
        try:
            report = publish(snapshot(), previous)
            previous = report
            print(json.dumps({"statuses": report["statuses"], "batch_selections": len(report["batch_selections"])}, ensure_ascii=False), flush=True)
            done = (
                report["statuses"].get("pending", 0) == 0
                and report["statuses"].get("started", 0) == 0
                and report["final_report_available"]
            )
            if done:
                return
        except Exception as exc:  # keep syncing through temporary SQLite/GitHub/network errors
            print(f"sync error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
