"""Repair unavailable batch-meta candidates from a frozen public base state.

This is a bounded continuation utility for provider outages.  It never reads
VAL/TEST answers or private evaluator feedback.  Once all five EVO receipts in
an active batch are terminal, it creates three identical, evidence-free
candidate branches from that batch's already frozen checkpoint so the normal
VAL selector and TEST release can continue.  The receipt is marked
``public_default_repair`` and is therefore distinguishable from a model
generated evolution.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from jit_mas.schemas import digest, utc_now


ACTIVE_DEFAULT = {"deepsearchqa", "deepresearch_bench_ii"}
TERMINAL = {"complete", "failed", "missing", "incomplete"}
STRATEGIES = (
    "Conservative public continuation: retain the frozen base.",
    "Failure repair public continuation: retain the frozen base.",
    "Structure and cost public continuation: retain the frozen base.",
)


def _rows(database, query, values=()):
    database.row_factory = sqlite3.Row
    return database.execute(query, tuple(values)).fetchall()


def repair_once(campaign: Path, *, sources: set[str]) -> int:
    changed = 0
    database_path = campaign / "campaign.sqlite"
    with sqlite3.connect(database_path, timeout=60) as database:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA busy_timeout=60000")
        rows = _rows(database,
                     "SELECT * FROM slots WHERE json_extract(body,'$.kind')='batch_evolution' "
                     "AND json_extract(body,'$.source') IN ({}) AND status IN ('failed','complete')"
                     .format(",".join("?" for _ in sources)), sorted(sources))
        for row in rows:
            body = json.loads(row["body"])
            source, run_id, batch = body["source"], body["run_id"], body["batch_index"]
            evo = _rows(database,
                        "SELECT status FROM slots WHERE json_extract(body,'$.kind')='evolution' "
                        "AND json_extract(body,'$.source')=? "
                        "AND json_extract(body,'$.run_id')=? AND json_extract(body,'$.batch_index')=?",
                        (source, run_id, batch))
            if len(evo) != 5 or any(item["status"] not in TERMINAL for item in evo):
                continue
            existing = _rows(database,
                             "SELECT candidate_index FROM candidates WHERE source=? "
                             "AND run_id=? AND batch_index=?",
                             (source, run_id, batch))
            if existing:
                existing_bodies = _rows(database,
                    "SELECT body FROM candidates WHERE source=? AND run_id=? AND batch_index=?",
                    (source, run_id, batch))
                if not all((json.loads(item["body"]).get("status") in {"failed", "complete"}
                            and json.loads(item["body"]).get("snapshot") is None)
                           for item in existing_bodies):
                    continue
                database.execute("DELETE FROM candidates WHERE source=? AND run_id=? AND batch_index=?",
                                 (source, run_id, batch))
            checkpoint = database.execute(
                "SELECT state_hash,snapshot FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                (source, run_id, batch * 5)).fetchone()
            if checkpoint is None:
                continue
            state_hash, snapshot_text = checkpoint["state_hash"], checkpoint["snapshot"]
            snapshot = json.loads(snapshot_text)
            # Protocol is stored in registration metadata; derive the exact
            # batch task IDs from the slot bodies instead of trusting input.
            task_rows = _rows(database,
                "SELECT body FROM slots WHERE json_extract(body,'$.kind')='evolution' "
                "AND json_extract(body,'$.source')=? AND json_extract(body,'$.run_id')=? "
                "AND json_extract(body,'$.batch_index')=? ORDER BY ordinal",
                (source, run_id, batch))
            task_ids = [json.loads(item["body"])["task_id"] for item in task_rows]
            candidates = []
            for index, strategy in enumerate(STRATEGIES):
                candidate = {
                    "candidate_id": f"b{batch}c{index}",
                    "candidate_index": index,
                    "strategy": strategy,
                    "status": "complete",
                    "snapshot": snapshot,
                    "state_hash": digest(snapshot),
                    "source_task_ids": task_ids,
                    "base_state_hash": state_hash,
                    "public_default_repair": True,
                    "repair_reason": "batch_meta_unavailable_after_terminal_public_EVO_receipts",
                    "decision": {"rationale": "Public default continuation; retain frozen base."},
                    "budget": {"records": [], "tokens": 0, "model_calls": 0,
                               "reserved_tokens": 0, "usage_unknown": True, "cost": None},
                    "calls": [],
                }
                database.execute(
                    "INSERT INTO candidates(source,run_id,batch_index,candidate_index,body) "
                    "VALUES(?,?,?,?,?)",
                    (source, run_id, batch, index, json.dumps(candidate, ensure_ascii=False,
                                                              separators=(",", ":"))))
                candidates.append(candidate)
            result = {
                "task_id": body["task_id"], "candidates": candidates,
                "budget": {"records": [], "tokens": 0, "model_calls": 0,
                            "reserved_tokens": 0, "usage_unknown": True, "cost": None},
                "candidate_budgets": [item["budget"] for item in candidates],
                "reflection_budgets": [], "public_default_repair": True,
                "repair_reason": "batch_meta_unavailable_after_terminal_public_EVO_receipts",
                "state_hash": state_hash,
                "registration_hash": digest(json.loads(
                    database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()[0]
                )),
            }
            result_hash = digest(result)
            database.execute(
                "UPDATE slots SET status='complete',claim_token=NULL,owner=NULL,started_at=NULL,"
                "result=?,result_hash=?,finished_at=? WHERE slot_id=?",
                (json.dumps(result, ensure_ascii=False, separators=(",", ":")), result_hash,
                 utc_now(), row["slot_id"]))
            # A previous interrupted candidate receipt can have caused its
            # dependent VAL rows to be marked missing.  Reopen exactly those
            # rows after replacing all three candidate bodies; no score or
            # answer is copied by this repair.
            database.execute(
                "UPDATE slots SET status='pending',claim_token=NULL,owner=NULL,started_at=NULL,"
                "result=NULL,result_hash=NULL,finished_at=NULL "
                "WHERE json_extract(body,'$.kind')='validation' "
                "AND json_extract(body,'$.source')=? AND json_extract(body,'$.run_id')=? "
                "AND json_extract(body,'$.batch_index')=? AND status='missing'",
                (source, run_id, batch))
            changed += 1
        database.commit()
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--sources", default=",")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    sources = {item.strip() for item in args.sources.split(",") if item.strip()} or ACTIVE_DEFAULT
    while True:
        changed = repair_once(args.campaign.resolve(), sources=sources)
        if changed:
            print(json.dumps({"repaired_batches": changed, "campaign": str(args.campaign.resolve())}), flush=True)
        if not args.watch:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
