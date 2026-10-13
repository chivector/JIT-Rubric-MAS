import json
import sqlite3

from jit_mas.schemas import digest
from scripts.report_effective_evolution import _budget_summary, build_report


def _snapshot(agent_ids=()):
    return {
        "agent_pool": {
            "profiles": [{"pool_agent_id": ident, "version": 1, "role": ident} for ident in agent_ids],
            "observations": [], "structural_history": [], "version": 1 if agent_ids else 0,
        },
        "version": 1 if agent_ids else 0, "experiences": [], "applied_proposals": [],
        "policy_versions": {}, "schema_version": "1.0",
    }


def _campaign(tmp_path, *, candidate_status="complete", corrupt=False):
    root = tmp_path / "campaign"
    root.mkdir(parents=True)
    db = sqlite3.connect(root / "campaign.sqlite")
    db.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE slots(slot_id TEXT, ordinal INTEGER, body TEXT, status TEXT,
                           claim_token TEXT, owner TEXT, started_at TEXT,
                           result TEXT, result_hash TEXT, finished_at TEXT);
        CREATE TABLE checkpoints(source TEXT, run_id INTEGER, position INTEGER,
                                 state_hash TEXT, snapshot TEXT, identity_hash TEXT);
        CREATE TABLE candidates(source TEXT, run_id INTEGER, batch_index INTEGER,
                                candidate_index INTEGER, body TEXT);
        CREATE TABLE batch_selections(source TEXT, run_id INTEGER, batch_index INTEGER, body TEXT);
    """)
    db.execute("INSERT INTO metadata VALUES (?, ?)", ("registration", json.dumps({"execution": {}})))
    before, after = _snapshot(), _snapshot(("writer", "critic"))
    before_hash, after_hash = digest(before), digest(after)
    db.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
               ("writingbench", 0, 0, before_hash, json.dumps(before), "identity"))
    candidate = {"candidate_id": "b0c0", "status": candidate_status,
                 "base_state_hash": before_hash, "state_hash": after_hash,
                 "snapshot": after, "decision": {"pool_operations": []},
                 "budget": {"tokens": 123, "model_calls": 2}}
    if corrupt:
        candidate["state_hash"] = "bad"
    db.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
               ("writingbench", 0, 0, 0, json.dumps(candidate)))
    db.commit()
    db.close()
    return root


def test_effective_view_accepts_only_integrity_checked_complete_candidates(tmp_path):
    report = build_report(_campaign(tmp_path))
    assert report["counts"]["effective_candidates"] == 1
    row = report["effective_trajectories"][0]
    assert row["pool_delta"]["agent_count_before"] == 0
    assert row["pool_delta"]["agent_count_after"] == 2
    assert row["pool_delta"]["agent_count_delta"] == 2

    corrupt = build_report(_campaign(tmp_path / "corrupt", corrupt=True))
    assert corrupt["counts"]["effective_candidates"] == 0
    assert corrupt["excluded_candidates"][0]["invalid_reason"] == "state_hash_mismatch"


def test_effective_view_excludes_fallback_noop_by_default(tmp_path):
    root = _campaign(tmp_path)
    db = sqlite3.connect(root / "campaign.sqlite")
    row = json.loads(db.execute("SELECT body FROM candidates").fetchone()[0])
    row["fallback_noop"] = True
    db.execute("UPDATE candidates SET body=?", (json.dumps(row),))
    db.commit(); db.close()
    report = build_report(root)
    assert report["counts"]["effective_candidates"] == 0
    assert report["excluded_candidates"][0]["fallback_noop"] is True
    assert build_report(root, include_noop=True)["counts"]["effective_candidates"] == 1


def test_budget_summary_counts_transport_retries_as_physical_attempts():
    summary = _budget_summary({
        "tokens": 12,
        "model_calls": 2,
        "records": [{
            "request": {"transport_retry": {
                "attempts": [{"status": "error"}, {"status": "success"}],
            }},
        }, {}],
    })
    assert summary["api_attempts"] == 3
    assert summary["api_retry_count"] == 1
    assert summary["api_failures"] == 1
    assert summary["api_failure_rate"] == 1 / 3
