import json
import sqlite3

import scripts.sync_dsv11_to_github as sync


def _make_campaign(tmp_path):
    root = tmp_path
    campaign = root / ".r" / "dsv12"
    campaign.mkdir(parents=True)
    (campaign / "status.json").write_text(
        json.dumps(
            {
                "phase": "full",
                "registered_slots": 4,
                "active": 0,
                "test_feedback_released": False,
            }
        ),
        encoding="utf-8",
    )
    (root / ".r" / "dsv12_paper").mkdir(parents=True)
    (root / ".r" / "dsv12_paper" / "paper_summary.json").write_text("{}", encoding="utf-8")
    db = sqlite3.connect(campaign / "campaign.sqlite")
    db.executescript(
        """
        create table slots (
            slot_id text primary key, ordinal integer, body text not null,
            status text not null, result text
        );
        create table batch_selections (
            source text, run_id integer, batch_index integer, body text
        );
        create table candidates (
            source text, run_id integer, batch_index integer,
            candidate_index integer, body text
        );
        create table checkpoints (
            source text, run_id integer, position integer,
            state_hash text, snapshot text, identity_hash text
        );
        create table metadata (key text primary key, body text not null);
        """
    )
    records = [
        {"kind": "model", "call_id": "call-1", "input_tokens": 10, "output_tokens": 2, "wall_seconds": 1.0},
        # A duplicate transport record must not be counted twice.
        {"kind": "model", "call_id": "call-1", "input_tokens": 999, "output_tokens": 999, "wall_seconds": 99.0},
        {"kind": "model", "call_id": "call-2", "input_tokens": 5, "output_tokens": 3, "wall_seconds": 3.0},
    ]
    result = {
        "budget": {"tokens": 20, "model_calls": 2, "records": records},
        # The nested copy is intentionally different; it must be ignored.
        "outcome": {
            "budget": {"tokens": 999999, "model_calls": 999},
            "evaluation": {"raw": {"official_metrics": {"Overall": 0.5, "Analysis": 0.25}}},
        },
        "execution_seconds": 4.0,
        "score": 0.75,
    }
    db.execute(
        "insert into slots values (?, ?, ?, ?, ?)",
        ("slot-0", 0, json.dumps({"source": "deepresearch_bench_ii", "kind": "validation"}), "complete", json.dumps(result)),
    )
    for ordinal, status in ((1, "incomplete"), (2, "failed"), (3, "pending")):
        db.execute(
            "insert into slots values (?, ?, ?, ?, ?)",
            (f"slot-{ordinal}", ordinal, json.dumps({"source": "writingbench", "kind": "validation"}), status, None),
        )
    checkpoint0 = {"version": 0, "experiences": [], "applied_proposals": [], "agent_pool": {"profiles": []}}
    checkpoint5 = {
        "version": 2,
        "experiences": [{"id": 1}],
        "applied_proposals": ["p1", "p2"],
        "agent_pool": {
            "profiles": [{"id": "a"}, {"id": "b"}],
            "structural_history": [],
            "applied_updates": ["u1", "u2"],
        },
    }
    db.execute("insert into checkpoints values (?, ?, ?, ?, ?, ?)", ("deepresearch_bench_ii", 0, 0, "h0", json.dumps(checkpoint0), "i0"))
    db.execute("insert into checkpoints values (?, ?, ?, ?, ?, ?)", ("deepresearch_bench_ii", 0, 5, "h5", json.dumps(checkpoint5), "i5"))
    db.execute(
        "insert into metadata values (?, ?)",
        (
            "registration",
            json.dumps(
                {
                    "execution": {
                        "configuration": {"models": {"judge": {"model": "deepseek-v4-flash-vision"}}},
                        "served_models": {
                            "judge": {
                                "calls": 1,
                                "content_nonempty": True,
                                "finish_reason": "stop",
                                "returned_model": "served-judge",
                            }
                        },
                    }
                }
            ),
        ),
    )
    candidate_body = {
        "base_state_hash": "h0",
        "state_hash": "h1",
        "decision": {"pool_operations": [{"op": "x"}], "agent_updates": [{"id": "u"}]},
        "status": "complete",
    }
    fallback_body = {
        "base_state_hash": "h0",
        "state_hash": "h0",
        "decision": {"pool_operations": [], "agent_updates": []},
        "fallback_noop": True,
        "fallback_reason": "invalid_batch_meta_decision",
        "status": "complete",
    }
    db.execute("insert into candidates values (?, ?, ?, ?, ?)", ("deepresearch_bench_ii", 0, 0, 0, json.dumps(candidate_body)))
    db.execute("insert into candidates values (?, ?, ?, ?, ?)", ("deepresearch_bench_ii", 0, 0, 1, json.dumps(fallback_body)))
    db.execute(
        "insert into batch_selections values (?, ?, ?, ?)",
        ("deepresearch_bench_ii", 0, 0, json.dumps({"selected": {"candidate_index": 1}})),
    )
    db.commit()
    db.close()
    return root, campaign


def test_snapshot_uses_top_level_budget_and_dynamic_campaign_paths(tmp_path, monkeypatch):
    root, campaign = _make_campaign(tmp_path)
    monkeypatch.setattr(sync, "ROOT", root)
    monkeypatch.setattr(sync, "CAMPAIGN", campaign)
    report = sync.snapshot()

    assert report["campaign"] == "dsv12"
    assert report["schema_version"] == "dsv12-live-v2"
    assert report["terminal_slots"] == 3  # complete + failed + incomplete
    assert report["paper_summary_available"] is True
    assert report["test_report_available"] is False
    assert report["test"] == {"registered_slots": 0, "statuses": {}, "complete": False}
    assert report["judge"] == {
        "configured": True,
        "model": "deepseek-v4-flash-vision",
        "preflight_complete": True,
        "preflight_calls": 1,
        "preflight_returned_model": "served-judge",
    }
    selection = report["batch_selections"][0]
    assert selection["selected_pool_operation_count"] == 0
    candidate_group = report["candidate_metrics"][0]
    assert candidate_group == {
        "source": "deepresearch_bench_ii",
        "run_id": 0,
        "batch_index": 0,
        "candidate_count": 2,
        "pool_operation_count": 1,
        "agent_update_count": 1,
        "state_changed_count": 1,
        "fallback_count": 1,
        "selected_candidate_index": 1,
        "selected_pool_operation_count": 0,
        "selected_agent_update_count": 0,
    }
    metrics = report["metrics"]
    assert metrics["complete_slot_count"] == 1
    assert metrics["tokens"] == {"total": 20, "input": 15, "output": 5, "model_calls": 2}
    assert metrics["latency"]["slot"]["median_seconds"] == 4.0
    assert metrics["latency"]["model"]["sample_count"] == 2
    group = metrics["by_source_kind"]["deepresearch_bench_ii/validation"]
    assert group["sample_count"] == 1
    assert group["tokens"] == 20
    assert metrics["official_metrics_by_source_kind"]["deepresearch_bench_ii/validation"]["means"] == {
        "Analysis": 0.25,
        "Overall": 0.5,
    }
    latest = report["checkpoints"]["latest"]
    assert latest["position"] == 5
    assert latest["pool_size"] == 2
    assert latest["applied_updates_count"] == 2
    assert latest["applied_proposals_count"] == 2


def test_percentile_empty_and_interpolation():
    assert sync._percentile([], 0.9) is None
    assert sync._percentile([1.0], 0.9) == 1.0
    assert sync._percentile([0.0, 10.0], 0.9) == 9.0
