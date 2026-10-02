"""Read-only monitor for the v33 ResearchRubrics method run.

The monitor never opens or mutates runtime state, keys, journals, or result
artifacts.  It writes one atomically replaced status snapshot and, only after a
verified terminal method run, invokes the sealed comparison audit once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import psutil


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v33"
LAUNCH = ROOT / "outputs/rr_deepseek_method_launch_20261002_v33/launch.json"
STATUS = ROOT / "outputs/rr_v33_monitor_status.json"
PID_FILE = ROOT / "outputs/rr_v33_monitor.pid"
BASELINE = ROOT / "outputs/rr_single_agent_run0_deepseekjudge_20261002_v29"
COMPARE = ROOT / "outputs/compare_rr_v33_v29.json"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def iso_timestamp(value: float | None):
    return None if value is None else datetime.fromtimestamp(value, timezone.utc).isoformat()


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def launch_state():
    launch = read_json(LAUNCH, {}) or {}
    pid = launch.get("method_pid")
    method_processes = []
    # The venv launcher can have a separate Python child. Match the actual
    # method command by output directory as well as module, rather than assuming
    # an old PID still identifies this experiment after PID reuse.
    method_path = str(METHOD).casefold()
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            command = " ".join(proc.info.get("cmdline") or []).casefold()
            if "scripts.run_rr_two_arm_pilot" in command and method_path in command:
                method_processes.append({"pid": proc.pid, "name": proc.info.get("name"),
                                         "created_at": iso_timestamp(proc.info.get("create_time"))})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    alive = bool(method_processes)
    return {"launch_file": str(LAUNCH), "method_pid": pid,
            "launched_pid_alive": process_alive(pid), "method_process_alive": alive,
            "actual_method_processes": method_processes, "launch_status": launch.get("status"),
            "started_at": launch.get("started_at"), "preflight_exit": launch.get("preflight_exit"),
            "code_commit": launch.get("code_commit")}


def terminal_files():
    rows = []
    for path in METHOD.rglob("complete.json"):
        value = read_json(path, {}) or {}
        rows.append({"run_dir": str(path.parent), "task_id": value.get("task_id"),
                     "mode": value.get("mode"), "kind": "complete", "status": value.get("status", "complete"),
                     "error": None})
    for path in METHOD.rglob("failure.json"):
        value = read_json(path, {}) or {}
        rows.append({"run_dir": str(path.parent), "task_id": value.get("task_id"),
                     "mode": value.get("mode"), "kind": "failure", "status": "failed",
                     "error": value.get("error")})
    # A run directory is the deduplication key; files may be observed more than
    # once while a process writes its final report.
    unique = {}
    for row in rows:
        unique[row["run_dir"]] = row
    return list(unique.values())


def deep_metrics(runs):
    finish_reasons = {}
    finish_reasons_by_stage = {}
    length_calls = []
    failure_categories = {}
    compactness_warnings = 0
    budgets_seen = set()
    budget_totals = {"model_calls": 0, "tokens": 0, "tool_calls": 0, "wall_seconds": 0.0}
    all_runs = {row["run_dir"]: row for row in runs}
    for path in METHOD.rglob("run_manifest.json"):
        key = str(path.parent)
        if key not in all_runs:
            manifest = read_json(path, {}) or {}
            task = manifest.get("comparison", {}).get("task", {})
            all_runs[key] = {"run_dir": key, "task_id": task.get("task_id"),
                             "mode": manifest.get("mode"), "kind": "in_progress"}
    for row in all_runs.values():
        run_dir = Path(row["run_dir"])
        budget_path = run_dir / "budget.json"
        budget_key = str(budget_path.resolve())
        if budget_path.is_file() and budget_key not in budgets_seen:
            budgets_seen.add(budget_key)
            budget = read_json(budget_path, {}) or {}
            for key in budget_totals:
                if isinstance(budget.get(key), (int, float)):
                    budget_totals[key] += budget[key]
            calls_seen = set()
            for record in budget.get("records", []):
                call_id = record.get("call_id")
                if call_id and call_id in calls_seen:
                    continue
                if call_id:
                    calls_seen.add(call_id)
                reason = (record.get("request") or {}).get("finish_reason")
                if reason:
                    finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
                    stage = record.get("stage", "unknown")
                    counts = finish_reasons_by_stage.setdefault(stage, {})
                    counts[reason] = counts.get(reason, 0) + 1
                    if reason == "length":
                        length_calls.append({"task_id": row.get("task_id"), "run_dir": str(run_dir),
                                             "stage": stage, "agent_id": record.get("agent_id"),
                                             "call_id": call_id, "output_tokens": record.get("output_tokens")})
        execution = read_json(run_dir / "execution.json", {}) or {}
        events = execution.get("metadata", {}).get("events", []) if isinstance(execution, dict) else []
        compactness_warnings += sum(event.get("kind") == "compactness_warning" for event in events)
        if row["kind"] == "failure":
            error = str(row.get("error") or "").lower()
            if "contextlimit" in error or "context window" in error:
                category = "context_limit"
            elif "complete json" in error or "truncated" in error or "length" in error:
                category = "truncation_or_json"
            elif "contributor answer exceeds" in error or "ledger." in error and "exceed" in error:
                category = "compactness_rejection"
            elif "reviewer" in error or "dependency" in error or "topology" in error:
                category = "review_topology"
            else:
                category = "other"
            failure_categories[category] = failure_categories.get(category, 0) + 1
    return {"unique_budget_files": len(budgets_seen), "budget_totals": budget_totals,
            "finish_reasons": finish_reasons, "finish_reasons_by_stage": finish_reasons_by_stage,
            "length_calls": length_calls, "failure_categories": failure_categories,
            "compactness_warning_events": compactness_warnings,
            "coverage": "Persisted budget.json once per distinct run directory; unfinished in-memory calls are unavailable."}


def latest_stages():
    latest = {}
    if not METHOD.is_dir():
        return latest
    for path in METHOD.rglob("*"):
        if path.is_file():
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            stage = path.relative_to(METHOD).parts[0]
            old = latest.get(stage)
            if old is None or stamp > old["mtime"]:
                latest[stage] = {"mtime": stamp, "timestamp": iso_timestamp(stamp),
                                 "path": str(path)}
    return latest


def report_state():
    report = read_json(METHOD / "ours/report.json", {}) or {}
    generation = read_json(METHOD / "generation_reports.json", {}) or {}
    metadata = read_json(METHOD / "pilot_metadata.json", {}) or {}
    interruption = read_json(METHOD / "interruption.json", None)
    return {"pilot_status": metadata.get("status"), "report_status": report.get("status"),
            "report_arm": report.get("arm"), "generation_status": {
                key: value.get("status") for key, value in generation.items()
                if isinstance(value, dict)}, "interruption_present": interruption is not None,
            "interruption": interruption}


def snapshot():
    runs = terminal_files()
    process = launch_state()
    reports = report_state()
    terminal_report = (reports["pilot_status"] in {"completed", "incomplete"}
                       or reports["report_status"] in {"completed", "submitted", "inconclusive", "failed"})
    interrupted = reports["interruption_present"]
    process_state = ("running" if process["method_process_alive"] else
                     "interrupted" if interrupted else
                     "terminal" if terminal_report else "vanished_or_incomplete")
    # A successful terminal method requires all independent signals: the actual
    # PID is gone, the pilot metadata says completed, the arm report is terminal,
    # and no interruption marker exists. A journal alone never qualifies.
    complete = (not process["method_process_alive"]
                and reports["pilot_status"] == "completed"
                and reports["report_status"] in {"submitted", "completed"}
                and not interrupted)
    return {"schema_version": "1.0", "observed_at": datetime.now(timezone.utc).isoformat(),
            "method_root": str(METHOD), "process": process, "reports": reports,
            "process_state": process_state, "method_success_verified": complete,
            "terminal_counts": {"total": len(runs),
                                "complete": sum(x["kind"] == "complete" for x in runs),
                                "failed": sum(x["kind"] == "failure" for x in runs)},
            "terminal_runs": runs, "deep_metrics": deep_metrics(runs),
            "latest_stage_files": latest_stages(), "comparison": {
                "requested": True, "invoked": COMPARE.is_file(), "path": str(COMPARE)}}


def atomic_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def maybe_compare(state):
    if not state["method_success_verified"] or COMPARE.exists():
        return
    command = [sys.executable, str(ROOT / "scripts/compare_rr_sealed_runs.py"),
               "--baseline", str(BASELINE), "--method", str(METHOD), "--output", str(COMPARE)]
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        state["comparison"].update({"invoked": True, "returncode": completed.returncode,
                                    "stderr": completed.stderr[-2000:]})
    except Exception as exc:
        state["comparison"].update({"invoked": True, "error": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--pid-file", type=Path, default=PID_FILE)
    args = parser.parse_args()
    deadline = time.monotonic() + 12 * 60 * 60
    atomic_write(args.pid_file, {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    while True:
        state = snapshot()
        state["monitor_deadline_hours"] = 12
        maybe_compare(state)
        if time.monotonic() >= deadline:
            state["monitor_status"] = "deadline_reached"
        atomic_write(args.status, state)
        if (args.once or state["method_success_verified"] or state["process_state"] == "interrupted"
                or time.monotonic() >= deadline):
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
