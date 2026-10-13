"""Create a resumable tds_fix5 continuation while reusing immutable results.

The source campaign is never modified. A fresh registration is initialized from
the current launch identity, then successful receipts, checkpoints, candidates,
and reusable TEST evaluation receipts are copied into the new campaign. Only
the blocked source scopes are reopened.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from scripts.run_independent_batch_experiment import IndependentBatchEnvironment
from jit_mas.schemas import digest


SOURCES = {
    "researchrubrics": 5,  # b5 validation is the first blocked batch
    "deepsearchqa": 1,      # b1 candidate 0 is the first blocked batch
    "deepresearch_bench_ii": 0,
}
ACTIVE_SOURCES = set(SOURCES)
INCLUSIVE_FIRST = False
RESET_ACTIVE_TESTS = False
RESUME_ACTIVE = False


def read_rows(path: Path, table: str):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = [dict(row) for row in con.execute(f"SELECT * FROM {table}").fetchall()]
    con.close()
    return rows


def body(row):
    return json.loads(row["body"])


def should_reset(row: dict) -> bool:
    b = body(row)
    source = b.get("source")
    kind = b.get("kind")
    status = row["status"]
    if source not in ACTIVE_SOURCES:
        return False
    if RESUME_ACTIVE and status in {"started", "failed", "missing", "incomplete"}:
        # Resume a live continuation without resampling terminal receipts.
        # Started claims are returned to pending so the new coordinator can
        # recover them under its own worker count; pending rows are left intact.
        return True
    if kind == "test":
        # A public-optimization continuation may regenerate every batch for
        # an active source.  Its selected checkpoint can therefore change, so
        # source-bound TEST receipts must be reopened together with the
        # evolution state.  The default continuation policy still preserves
        # completed TEST receipts.
        if RESET_ACTIVE_TESTS:
            return True
        return status in {"missing", "failed", "incomplete"}
    first = SOURCES[source]
    batch = b.get("batch_index")
    if kind in {"evolution", "batch_evolution"}:
        threshold = first if INCLUSIVE_FIRST else first + 1
        # ``--inclusive-first`` is used after a completed-but-inconclusive
        # first continuation batch.  Such a batch is terminal in the source
        # database, but its no-op candidate must be regenerated under the new
        # implementation.  Reopen the batch aggregate itself while retaining
        # its immutable EVO receipts.
        if (INCLUSIVE_FIRST and kind in {"evolution", "batch_evolution"}
                and isinstance(batch, int) and batch >= threshold
                and status in {"complete", "started"}):
            return True
        return isinstance(batch, int) and batch >= threshold and status in {"missing", "failed", "incomplete"}
    if kind == "validation":
        if RESET_ACTIVE_TESTS:
            first = SOURCES[source]
            threshold = first if INCLUSIVE_FIRST else first + 1
            return isinstance(batch, int) and batch >= threshold
        # Re-score the incomplete stop batch. DSQA's two no-op candidates are
        # intentionally kept terminal so they are never regenerated.
        if source == "researchrubrics" and batch == 5:
            return status != "complete"
        if source == "deepsearchqa" and batch == 1:
            return b.get("candidate_index") == 0 and status != "complete"
        if source == "deepresearch_bench_ii" and batch == 0:
            return status != "complete"
        threshold = first if INCLUSIVE_FIRST else first + 1
        return isinstance(batch, int) and batch >= threshold and status in {"missing", "failed", "incomplete"}
    return False


def clear_row(database, slot_id: str):
    database.execute(
        "UPDATE slots SET status='pending',claim_token=NULL,owner=NULL,started_at=NULL,"
        "result=NULL,result_hash=NULL,finished_at=NULL WHERE slot_id=?",
        (slot_id,),
    )


def copy_slot(database, row: dict, registration_hash: str):
    result = row["result"]
    result_hash = row["result_hash"]
    if result:
        try:
            parsed = rebind(json.loads(result), registration_hash)
            result = json.dumps(parsed,
                                ensure_ascii=False, separators=(",", ":"))
            result_hash = digest(parsed)
        except (TypeError, ValueError):
            # Keep an opaque interrupted receipt auditable; normal JSON
            # receipts are rebound so downstream identity checks can pass.
            result = row["result"]
    database.execute(
        "UPDATE slots SET status=?,claim_token=?,owner=?,started_at=?,result=?,result_hash=?,finished_at=? WHERE slot_id=?",
        (row["status"], row["claim_token"], row["owner"], row["started_at"],
         result, result_hash, row["finished_at"], row["slot_id"]),
    )


def rebind(value, registration_hash):
    if isinstance(value, dict):
        return {key: (registration_hash if key == "registration_hash" else rebind(item, registration_hash))
                for key, item in value.items()}
    if isinstance(value, list):
        return [rebind(item, registration_hash) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("tds_fix5"))
    parser.add_argument("--output", type=Path, default=Path("tds_fix7"))
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--sources", default=",".join(SOURCES),
                        help="Comma-separated source names to reopen; completed rows for other sources are preserved")
    parser.add_argument("--first-batch", type=int, default=None,
                        help="Override the first reopened batch for every active source; use -1 to regenerate roots")
    parser.add_argument("--inclusive-first", action="store_true",
                        help="Also reopen the first batch itself; useful when resuming after a failed first continuation batch")
    parser.add_argument("--reset-active", action="store_true",
                        help="Reopen completed validation and TEST slots for active sources when regenerating their batches")
    parser.add_argument("--resume-active", action="store_true",
                        help="Resume active source slots, reopening only nonterminal failed/started/missing rows")
    parser.add_argument("--source-first", default=None,
                        help="Per-source first reopened batch, e.g. deepsearchqa=5,deepresearch_bench_ii=2")
    args = parser.parse_args()
    global ACTIVE_SOURCES, INCLUSIVE_FIRST, RESET_ACTIVE_TESTS, RESUME_ACTIVE
    ACTIVE_SOURCES = {value.strip() for value in args.sources.split(",") if value.strip()}
    INCLUSIVE_FIRST = args.inclusive_first
    RESET_ACTIVE_TESTS = args.reset_active
    RESUME_ACTIVE = args.resume_active
    if args.source_first:
        for item in args.source_first.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise SystemExit("--source-first entries must be source=batch")
            source_name, raw_batch = (part.strip() for part in item.split("=", 1))
            if source_name not in SOURCES:
                raise SystemExit(f"unknown source in --source-first: {source_name}")
            SOURCES[source_name] = int(raw_batch)
    unknown = ACTIVE_SOURCES - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown source(s): {sorted(unknown)}")
    if args.first_batch is not None:
        for source_name in ACTIVE_SOURCES:
            SOURCES[source_name] = args.first_batch
    source = args.source.resolve()
    output = args.output.resolve()
    launch = args.launch.resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing continuation: {output}")
    output.mkdir(parents=True)

    # Initialize a fresh database and frozen identity. No model call is made.
    environment = IndependentBatchEnvironment(launch, output, require_preflight=True)
    environment.initialize()
    new_hash = environment.campaign.registration_hash
    old_db = source / "campaign.sqlite"
    new_db = output / "campaign.sqlite"

    old_slots = {row["slot_id"]: row for row in read_rows(old_db, "slots")}
    old_checkpoints = read_rows(old_db, "checkpoints")
    old_candidates = read_rows(old_db, "candidates")
    old_batch_selections = read_rows(old_db, "batch_selections")
    old_selections = read_rows(old_db, "selections")

    copied = {"complete": 0, "submitted": 0, "failed": 0, "incomplete": 0, "missing": 0}
    reopened = []
    with sqlite3.connect(new_db) as database:
        database.execute("PRAGMA busy_timeout=60000")
        # Preserve all old terminal rows except the explicitly reopened scopes;
        # missing rows outside those scopes remain terminal and are auditable.
        for slot_id, row in old_slots.items():
            if should_reset(row):
                clear_row(database, slot_id)
                reopened.append(slot_id)
            else:
                copy_slot(database, row, new_hash)
                copied[row["status"]] = copied.get(row["status"], 0) + 1

        # Rebind immutable snapshots to the new campaign identity.
        database.execute("DELETE FROM checkpoints")
        for row in old_checkpoints:
            if (row["source"] in ACTIVE_SOURCES and INCLUSIVE_FIRST
                    and SOURCES[row["source"]] >= 0
                    and row["position"] > SOURCES[row["source"]] * 5):
                # A reopened batch invalidates all later winner states.  Keep
                # only the prefix checkpoint that is the frozen input to the
                # first reopened batch; later checkpoints must be rebuilt from
                # the newly selected VAL winners.
                continue
            database.execute(
                "INSERT INTO checkpoints(source,run_id,position,state_hash,snapshot,identity_hash) VALUES(?,?,?,?,?,?)",
                (row["source"], row["run_id"], row["position"], row["state_hash"], row["snapshot"], new_hash),
            )

        database.execute("DELETE FROM candidates")
        for row in old_candidates:
            batch = row.get("batch_index")
            rebuild_from = (row["source"] in ACTIVE_SOURCES and INCLUSIVE_FIRST
                            and isinstance(batch, int) and batch >= SOURCES[row["source"]])
            if row["source"] in ACTIVE_SOURCES and (SOURCES[row["source"]] < 0 or rebuild_from):
                continue
            database.execute("INSERT INTO candidates(source,run_id,batch_index,candidate_index,body) VALUES(?,?,?,?,?)",
                             (row["source"], row["run_id"], row["batch_index"], row["candidate_index"], row["body"]))

        # Stop-batch inconclusive choices must be recomputed after the repaired
        # validation receipts. Completed earlier choices remain reusable.
        stop_batches = {(source_name, SOURCES[source_name]) for source_name in ACTIVE_SOURCES
                        if SOURCES[source_name] >= 0}
        database.execute("DELETE FROM batch_selections")
        for row in old_batch_selections:
            batch = row.get("batch_index")
            rebuild_from = (row["source"] in ACTIVE_SOURCES and INCLUSIVE_FIRST
                            and isinstance(batch, int) and batch >= SOURCES[row["source"]])
            if row["source"] in ACTIVE_SOURCES and (SOURCES[row["source"]] < 0 or rebuild_from):
                continue
            if (row["source"], row["batch_index"]) in stop_batches:
                continue
            parsed = rebind(json.loads(row["body"]), new_hash)
            database.execute("INSERT INTO batch_selections(source,run_id,batch_index,body) VALUES(?,?,?,?)",
                             (row["source"], row["run_id"], row["batch_index"], json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))))

        database.execute("DELETE FROM selections")
        for row in old_selections:
            parsed = json.loads(row["body"])
            if parsed.get("status") != "complete":
                continue
            parsed = rebind(parsed, new_hash)
            database.execute("INSERT INTO selections(source,run_id,body) VALUES(?,?,?)",
                             (row["source"], row["run_id"], json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))))

        # A new continuation must seal its own TEST inventory.
        database.execute("DELETE FROM metadata WHERE key='test_seal'")

    # Reuse completed TEST scoring receipts and immutable submissions. The
    # receipt slot/result objects are copied together, so adapter validation can
    # verify them without making another judge call.
    for directory in ("test_evaluations", "test_evaluation_started"):
        src_dir, dst_dir = source / directory, output / directory
        if src_dir.is_dir():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.iterdir():
                if path.is_file():
                    try:
                        parsed = json.loads(path.read_text(encoding="utf-8"))
                        (dst_dir / path.name).write_text(
                            json.dumps(rebind(parsed, new_hash), ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
                    except (OSError, ValueError, TypeError):
                        shutil.copy2(path, dst_dir / path.name)

    manifest = {
        "schema": "tds-fix5-reuse-continuation-v1",
        "from_campaign": str(source),
        "to_campaign": str(output),
        "old_registration_hash": json.loads(next(row["body"] for row in read_rows(old_db, "metadata") if row["key"] == "registration"))["execution"].get("registration_hash"),
        "new_registration_hash": new_hash,
        "reopened_slots": len(reopened),
        "reopened_examples": reopened[:20],
        "copied_status_counts": copied,
        "stop_batches_recomputed": [f"{source}:b{SOURCES[source]}" for source in sorted(ACTIVE_SOURCES)],
        "test_evaluations_reused": len(list((output / "test_evaluations").glob("*.json"))),
        "source_untouched": True,
        "policy": "Preserve complete/submitted receipts; reopen only blocked source scopes and missing source TEST slots.",
    }
    # The old registration hash is the digest of the full registration object,
    # not an execution field; compute it from the source metadata exactly.
    old_registration = next(row["body"] for row in read_rows(old_db, "metadata") if row["key"] == "registration")
    manifest["old_registration_hash"] = digest(json.loads(old_registration))
    (output / "reuse_migration_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "new_registration_hash": new_hash,
                      "reopened_slots": len(reopened), "copied_status_counts": copied}, ensure_ascii=False))


if __name__ == "__main__":
    main()
