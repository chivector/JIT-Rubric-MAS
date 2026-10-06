"""Publish compact, credential-free dsv11 progress snapshots to GitHub.

The campaign itself owns .r/dsv11. This helper only reads its SQLite/status
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


def snapshot() -> dict[str, Any]:
    status = _json(CAMPAIGN / "status.json", {})
    counts: dict[str, int] = {}
    selections: list[dict[str, Any]] = []
    con = sqlite3.connect(CAMPAIGN / "campaign.sqlite", timeout=30)
    try:
        for key, value in con.execute("select status, count(*) from slots group by status"):
            counts[str(key)] = int(value)
        rows = con.execute(
            "select source, run_id, batch_index, body from batch_selections "
            "order by source, run_id, batch_index"
        ).fetchall()
        for source, run_id, batch_index, raw in rows:
            body = json.loads(raw)
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
    finally:
        con.close()

    report = {
        "schema_version": "dsv11-live-v1",
        "campaign": "dsv11",
        "phase": status.get("phase"),
        "registered_slots": status.get("registered_slots"),
        "statuses": counts,
        "terminal_slots": sum(counts.get(k, 0) for k in ("complete", "failed", "missing")),
        "active": status.get("active", counts.get("started", 0)),
        "test_feedback_released": bool(status.get("test_feedback_released", False)),
        "registration_hash": status.get("registration_hash"),
        "batch_selections": selections,
        "final_report_available": (CAMPAIGN / "full_report.json").exists(),
        "paper_summary_available": any(
            (ROOT / ".r" / "dsv11_paper" / name).exists()
            for name in ("paper_summary.json", "paper_summary.md")
        ),
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
            _run("git", "commit", "-m", "chore: update dsv11 live results")
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
