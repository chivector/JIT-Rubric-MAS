"""Run a targeted-compact experiment in independent, resumable shards.

The existing compact runner is intentionally simple and processes its plan
sequentially.  This wrapper keeps that runner unchanged while splitting a
full benchmark plan into independent child runs.  Every child has its own
ledger and output directory, so a failed shard can be retried without
discarding successful shards.  The merged report preserves the per-task
records and raw artifacts from each child.

This is for the explicitly labelled ``test_adapted``/targeted condition.  It
does not change the formal evolution campaign and does not inject reference
answers into model outputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = Path(__file__).resolve().with_name("run_targeted_compact_experiment.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.schemas import digest


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _absolute_plan_rows(plan: dict, plan_path: Path) -> list[dict]:
    """Return plan rows with data/evidence paths absolute for child plans."""
    rows = []
    for row in plan.get("benchmarks", []):
        copy = dict(row)
        for key in ("data", "evidence_dir"):
            if copy.get(key):
                value = Path(copy[key])
                copy[key] = str(value.resolve() if value.is_absolute() else (plan_path.parent / value).resolve())
        rows.append(copy)
    return rows


def _make_shard_plans(plan: dict, plan_path: Path, shard_root: Path, shard_count: int,
                      completed_keys: set[tuple[str, str]] | None = None) -> list[dict]:
    completed_keys = completed_keys or set()
    rows = _absolute_plan_rows(plan, plan_path)
    inventory: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(rows):
        for task_id in row.get("task_ids", []):
            inventory.append((row_index, row["name"], task_id))
    if not inventory:
        raise ValueError("The plan has no task_ids")
    shard_count = max(1, min(int(shard_count), len(inventory)))
    buckets: list[list[tuple[int, str, str]]] = [[] for _ in range(shard_count)]
    for index, item in enumerate(inventory):
        if (item[1], item[2]) not in completed_keys:
            buckets[index % shard_count].append(item)

    plans = []
    for shard_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        selected: dict[tuple[int, str], list[str]] = {}
        for row_index, benchmark, task_id in bucket:
            selected.setdefault((row_index, benchmark), []).append(task_id)
        shard_rows = []
        for row_index, row in enumerate(rows):
            task_ids = selected.get((row_index, row["name"]), [])
            if task_ids:
                shard_row = dict(row)
                shard_row["task_ids"] = task_ids
                shard_rows.append(shard_row)
        shard_plan = {
            "schema": "targeted-compact-shard-plan-v1",
            "parent_plan": str(plan_path),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "benchmarks": shard_rows,
        }
        path = shard_root / f"plan_{shard_index:03d}.json"
        _write_json(path, shard_plan)
        plans.append({"index": shard_index, "path": path, "tasks": len(bucket),
                      "task_ids": [item[2] for item in bucket]})
    return plans


def _record_sources(output: Path) -> list[tuple[dict, Path, int]]:
    """Read durable task records, including those in interrupted attempts."""
    found = []
    for record_path in sorted((output / "shards").glob("*/tasks/*/record.json")):
        try:
            row = json.loads(record_path.read_text(encoding="utf-8-sig"))
            if not row.get("benchmark") or not row.get("task_id"):
                continue
            found.append((row, record_path.parent, record_path.stat().st_mtime_ns))
        except (OSError, ValueError, TypeError):
            # A process may be midway through an atomic write; a later resume
            # will discover it once durable.
            continue
    return found


def _valid_complete(row: dict, source: Path) -> bool:
    if row.get("status") != "complete" or not row.get("complete"):
        return False
    answer = source / "answer.txt"
    if not answer.exists():
        return False
    try:
        return digest(answer.read_text(encoding="utf-8")) == row.get("answer_hash")
    except (OSError, UnicodeError):
        return False


def _selected_sources(output: Path) -> dict[tuple[str, str], tuple[dict, Path, int]]:
    chosen = {}
    for row, source, modified in _record_sources(output):
        key = (row["benchmark"], row["task_id"])
        previous = chosen.get(key)
        rank = (_valid_complete(row, source), modified)
        if previous is None or rank > (_valid_complete(previous[0], previous[1]), previous[2]):
            chosen[key] = (row, source, modified)
    return chosen


def _new_attempt(shard_root: Path, index: int) -> Path:
    # Short names avoid the Windows MAX_PATH boundary after task digests and
    # write_json's atomic temporary filename are appended.
    attempt_index = 1
    while True:
        attempt = shard_root / f".s{index:03d}.a{attempt_index:03d}"
        if not attempt.exists():
            return attempt
        attempt_index += 1


def _run_one_shard(index: int, plan_path: Path, config_path: Path, shard_root: Path,
                   canonical: Path, log_path: Path, resume: bool) -> dict:
    attempt = _new_attempt(shard_root, index)
    command = [sys.executable, str(BASE_RUNNER), "--config", str(config_path),
               "--plan", str(plan_path), "--output", str(attempt)]
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    # The base model's historical default exits the entire shard after five
    # provider failures.  Let the runner's per-task exception handler preserve
    # that failure and continue the independent task inventory instead.
    child_env["MODULAR_AGENT_API_FAILURE_ACTION"] = "raise"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command, "started_at": _utc_stamp(),
                                 "api_failure_action": "raise"}, ensure_ascii=False) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                   env=child_env, check=False)
    result = {"index": index, "status": "complete" if completed.returncode == 0 else "failed",
              "returncode": completed.returncode, "elapsed_seconds": time.monotonic() - started,
              "attempt": str(attempt), "log": str(log_path)}
    # Promote only into an unused canonical name.  Earlier canonical shards and
    # attempts are retained; all are merged and eligible for task-level reuse.
    if completed.returncode == 0 and (attempt / "report.json").exists():
        if not canonical.exists():
            attempt.rename(canonical)
            attempt = canonical
        report = json.loads((attempt / "report.json").read_text(encoding="utf-8"))
        result["tasks"] = len(report.get("records", []))
        result["shard"] = str(attempt)
    return result


def _merge(output: Path, parent_plan: Path, config_path: Path, shard_specs: list[dict],
           statuses: list[dict]) -> dict:
    selected = _selected_sources(output)
    records = []
    for key, (row, source, modified) in sorted(selected.items()):
        benchmark, task_id = key
        destination = output / "tasks" / benchmark / digest(task_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        row = dict(row)
        row["answer_file"] = (destination / "answer.txt").relative_to(output).as_posix() \
            if (destination / "answer.txt").exists() else None
        row["merged_from_shard"] = source.parent.parent.relative_to(output).as_posix()
        records.append(row)
    attempts = []
    for row, source, modified in _record_sources(output):
        attempt = dict(row)
        attempt["attempt_artifacts"] = source.relative_to(output).as_posix()
        attempts.append(attempt)
    plan = json.loads(parent_plan.read_text(encoding="utf-8-sig"))
    planned_tasks = sum(len(row.get("task_ids", [])) for row in plan.get("benchmarks", []))
    all_attempts_efficiency = {
        "task_attempts": len(attempts),
        "complete_task_attempts": sum(row.get("status") == "complete" for row in attempts),
        "tokens": sum(row.get("budget", {}).get("tokens", 0) for row in attempts),
        "model_calls": sum(row.get("budget", {}).get("model_calls", 0) for row in attempts),
        "failed_task_attempts": sum(row.get("status") == "failed" for row in attempts),
        "failed_model_calls": sum(bool(call.get("error")) for row in attempts
                                  for call in row.get("budget", {}).get("records", [])),
    }
    merged = {"schema": "targeted-compact-full-split-v2", "finished_at": _utc_stamp(),
              "condition": "test_adapted_targeted_compact_full_split", "base_runner": str(BASE_RUNNER),
              "base_runner_sha256": hashlib.sha256(BASE_RUNNER.read_bytes()).hexdigest(),
              "parent_plan": str(parent_plan), "config": str(config_path),
              "shards": sorted(statuses, key=lambda row: row["index"]), "records": records,
              "complete_tasks": sum(_valid_complete(row, source) for row, source, _ in selected.values()),
              "total_tasks": len(records), "planned_tasks": planned_tasks,
              "missing_tasks": planned_tasks - len(selected),
              "failed_shards": sum(row.get("status") == "failed" for row in statuses),
              "all_attempts_efficiency": all_attempts_efficiency,
              "selection_policy": "Reuse complete tasks regardless of shard exit; latest valid complete attempt, otherwise latest incomplete attempt; no score-based selection"}
    _write_json(output / "manifest.json", {key: value for key, value in merged.items() if key != "records"})
    _write_json(output / "report.json", merged)
    for name, values in (("records.jsonl", records), ("all_attempt_records.jsonl", attempts)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for record in values:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return merged


def run(config: str | Path, plan: str | Path, output: str | Path, shards: int = 8,
        concurrency: int | None = None, resume: bool = False, recover_only: bool = False) -> dict:
    config_path, plan_path, output = Path(config).resolve(), Path(plan).resolve(), Path(output).resolve()
    if not config_path.exists() or not plan_path.exists():
        raise FileNotFoundError("Config and plan must exist")
    if output.exists() and not (resume or recover_only):
        raise FileExistsError(f"Output already exists: {output}; pass --resume to reuse completed tasks")
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    parent_plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    completed_keys = {key for key, (row, source, _) in _selected_sources(output).items()
                      if _valid_complete(row, source)} if (resume or recover_only) else set()
    shard_specs = _make_shard_plans(parent_plan, plan_path, shard_root, max(1, int(shards)), completed_keys)
    statuses = []
    if recover_only or not shard_specs:
        return _merge(output, plan_path, config_path, shard_specs, statuses)
    concurrency = max(1, min(int(concurrency or len(shard_specs)), len(shard_specs)))
    print(json.dumps({"reused_complete_tasks": len(completed_keys),
                      "pending_tasks": sum(spec["tasks"] for spec in shard_specs),
                      "concurrency": concurrency}, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for spec in shard_specs:
            index = spec["index"]
            future = pool.submit(_run_one_shard, index, spec["path"], config_path, shard_root,
                                 shard_root / f"shard_{index:03d}",
                                 output / "logs" / f"shard_{index:03d}.log", resume)
            futures[future] = index
        for future in as_completed(futures):
            try:
                status = future.result()
            except Exception as error:
                status = {"index": futures[future], "status": "failed",
                          "returncode": None, "error": repr(error)}
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            # Persist partial merged results as independent shards finish.
            _merge(output, plan_path, config_path, shard_specs, statuses)
    return _merge(output, plan_path, config_path, shard_specs, statuses)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-only", action="store_true", help="Merge durable task artifacts without API calls")
    args = parser.parse_args(argv)
    result = run(args.config, args.plan, args.output, args.shards, args.concurrency, args.resume, args.recover_only)
    print(json.dumps({"output": args.output, "complete_tasks": result["complete_tasks"],
                      "total_tasks": result["total_tasks"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
