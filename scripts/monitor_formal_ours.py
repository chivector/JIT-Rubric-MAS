"""Continuously monitor a formal Ours output directory.

The monitor is intentionally read-only with respect to experiment receipts. It
writes one small ``monitor_status.json`` heartbeat and can be left running while
the native runner is resumed or while new benchmark reports are added.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from datetime import datetime, timezone


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path):
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        if line.strip():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def snapshot(root: Path):
    records = []
    record_paths = list(root.glob("**/records.jsonl"))
    for path in record_paths:
        records.extend({"source": str(path), **row} for row in read_jsonl(path))
    statuses = Counter(str(row.get("status", "unknown")) for row in records)
    active_slots = []
    for path in root.glob("**/slots/*/started.json"):
        if not (path.parent / "record.json").is_file():
            value = read_json(path, {}) or {}
            active_slots.append({"path": str(path), **{key: value.get(key) for key in
                ("benchmark", "phase", "round_index", "task_id", "started_at")}})
    pool_before = [row.get("agent_pool_size_before") for row in records
                   if isinstance(row.get("agent_pool_size_before"), int)]
    pool_after = [row.get("agent_pool_size_after") for row in records
                  if isinstance(row.get("agent_pool_size_after"), int)]
    operations = Counter()
    for row in records:
        for operation in row.get("operations_applied") or []:
            kind = operation.get("kind") if isinstance(operation, dict) else str(operation)
            if kind:
                operations[kind] += 1
    # A monitor heartbeat must not make stalled experiment receipts look fresh.
    files = [path for path in root.glob("**/*") if path.is_file()
             and not path.name.startswith("monitor")]
    latest = max((path.stat().st_mtime for path in files), default=None)
    campaign = read_json(root / "status.json", {}) or {}
    campaign_statuses = campaign.get("statuses") if isinstance(campaign.get("statuses"), dict) else {}
    campaign_active = campaign.get("active") if isinstance(campaign.get("active"), int) else 0
    campaign_slots = campaign.get("registered_slots") if isinstance(campaign.get("registered_slots"), int) else None
    campaign_terminal = campaign.get("terminal_slots") if isinstance(campaign.get("terminal_slots"), int) else None
    return {
        "schema": "formal-ours-monitor-v1",
        "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "records_files": len(record_paths),
        "records": len(records),
        "statuses": dict(statuses),
        "active_started": len(active_slots),
        "active_slots": active_slots,
        "pool_size_before_min": min(pool_before) if pool_before else None,
        "pool_size_before_max": max(pool_before) if pool_before else None,
        "pool_size_after_min": min(pool_after) if pool_after else None,
        "pool_size_after_max": max(pool_after) if pool_after else None,
        "applied_operations": dict(operations),
        "latest_receipt_mtime": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else None,
        "campaign_status": campaign_statuses or None,
        "campaign_registered_slots": campaign_slots,
        "campaign_active": campaign_active if campaign else None,
        "campaign_terminal_slots": campaign_terminal,
        "targeted_optimization": campaign.get("targeted_optimization") if campaign else None,
        "native_run_present": bool(record_paths or active_slots or campaign),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    status_path = root / "monitor_status.json"
    while True:
        value = snapshot(root)
        status_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if args.once:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
