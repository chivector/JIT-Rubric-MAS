from pathlib import Path
import json
import sqlite3

import pytest

from scripts.summarize_experiment_dashboard import (
    batch_measurements, comparable_profiles, historical_curves, nullable_cumulative, operation_curve,
    selected_summary, write_json,
)


def test_summary_combines_recovery_methods_without_dropping_tasks(tmp_path):
    directory = tmp_path / "researchrubrics_test33_recovered"
    directory.mkdir()
    records = [
        {"benchmark": "researchrubrics", "task_id": "task-one", "method": "original",
         "status": "complete", "complete": True, "score_result": {"score": 0.8, "complete": True},
         "budget": {"tokens": 10, "model_calls": 0, "records": [], "wall_seconds": 1, "by_stage": {}}},
        {"benchmark": "researchrubrics", "task_id": "task-two", "method": "recovery",
         "status": "complete", "complete": True, "score_result": {"score": 0.6, "complete": True},
         "budget": {"tokens": 20, "model_calls": 0, "records": [], "wall_seconds": 2, "by_stage": {}}},
    ]
    write_json(directory / "report.json", {"records": records})
    summary = selected_summary(tmp_path)
    assert len(summary) == 1
    assert summary[0]["sample_count"] == 2
    assert summary[0]["metrics"]["compliance"] == pytest.approx(0.7)
    assert summary[0]["efficiency"]["tokens"] == 30
    assert summary[0]["efficiency"]["actual_cost"] is None
    assert Path(summary[0]["source_path"]).name == "report.json"


def test_historical_curves_recompute_memory_behavior_and_policy_from_real_profiles(tmp_path):
    curve_dir = tmp_path / "summary/evolution_curve/writingbench"
    curve_dir.mkdir(parents=True)
    run_dir = tmp_path / "native_evolution"
    run_dir.mkdir()
    before = {"writer": {"version": 1, "prompt": "draft", "memory": [],
                          "harness": {"memory_policy": "full"}}}
    memory_after = {"writer": {"version": 2, "prompt": "draft", "memory": [{"instruction": "learned"}],
                                "harness": {"memory_policy": "full"}}}
    policy_after = {"writer": {"version": 3, "prompt": "draft", "memory": [{"instruction": "learned"}],
                                "harness": {"memory_policy": "recent"}}}
    records = [{"round_index": 1, "task_id": "writingbench:1", "harness_before": before, "harness_after": memory_after},
               {"round_index": 2, "task_id": "writingbench:2", "harness_before": memory_after, "harness_after": policy_after}]
    (run_dir / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    rows = [{"round_index": index, "task_id": f"writingbench:{index}", "phase": "evolution",
             "source_path": "native_evolution\\records.jsonl", "harness_changed_agent_count": 1,
             "harness_modification_count": 1, "cumulative_harness_changed_agent_count": index}
            for index in (1, 2)]
    (curve_dir / "native_evolution_native_evolution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    result = historical_curves(tmp_path)
    assert result[0]["harness_changed_agent_count"] == 0
    assert result[0]["harness_modification_count"] == 0
    assert result[0]["memory_changed_agent_count"] == 1
    assert result[0]["legacy_harness_changed_agent_count"] == 1
    assert result[1]["harness_policy_changed_agent_count"] == 1
    assert result[1]["profile_behavior_changed_agent_count"] == 1
    assert result[1]["cumulative_memory_changed_agent_count"] == 1
    assert result[1]["cumulative_harness_changed_agent_count"] == 1


def test_historical_missing_snapshots_remain_unknown_and_keep_legacy_counts(tmp_path):
    curve_dir = tmp_path / "summary/evolution_curve/writingbench"
    curve_dir.mkdir(parents=True)
    row = {"round_index": 1, "task_id": "writingbench:1", "phase": "evolution",
           "source_path": "missing/records.jsonl", "harness_changed_agent_count": 47,
           "cumulative_harness_changed_agent_count": 47}
    (curve_dir / "native_evolution_native_evolution.jsonl").write_text(json.dumps(row) + "\n")
    result = historical_curves(tmp_path)[0]
    assert result["harness_modification_count"] is None
    assert result["memory_changed_agent_count"] is None
    assert result["legacy_harness_changed_agent_count"] == 47
    assert result["snapshot_measurement_available"] is False


def test_historical_sqlite_snapshots_use_frozen_effective_seed_and_keep_skills_change(tmp_path):
    curve_dir = tmp_path / "summary/evolution_curve/writingbench"
    curve_dir.mkdir(parents=True)
    run_dir = tmp_path / "native_evolution/writingbench"
    run_dir.mkdir(parents=True)
    frozen_dir = run_dir / "runs/first"
    frozen_dir.mkdir(parents=True)
    seed = {"pool_agent_id": "writer", "role": "Writer", "capabilities": ["writing"],
            "prompt": "draft", "skills": {"genre": "match public request"}, "memory": [],
            "harness": {"memory_policy": "full"}}
    evolved = {**seed, "skills": {**seed["skills"], "source_check": "avoid placeholders"},
               "memory": [{"instruction": "learned"}]}
    write_json(frozen_dir / "frozen_plan.json", {"experience_hash": "seed-hash", "AgentPool": {"profiles": [seed]}})
    with sqlite3.connect(run_dir / "experience.sqlite") as database:
        database.execute("CREATE TABLE snapshots (version INTEGER PRIMARY KEY, body TEXT)")
        database.executemany("INSERT INTO snapshots VALUES (?,?)", [
            (0, json.dumps({"agent_pool": {"initialized": False, "profiles": []}})),
            (1, json.dumps({"agent_pool": {"initialized": True, "profiles": [evolved]}})),
        ])
    records = [{"round_index": index, "task_id": f"writingbench:{index}", "benchmark": "writingbench",
                "experience_version_before": 0, "experience_version_after": index - 1,
                "snapshot_before_hash": "seed-hash", "snapshot_after_hash": "seed-hash" if index == 1 else "after-hash",
                "harness_before": {"writer": {"prompt": "draft", "memory": [], "skills": seed["skills"]}}}
               for index in (1, 2)]
    (tmp_path / "native_evolution/records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    rows = [{"round_index": index, "task_id": f"writingbench:{index}", "benchmark": "writingbench", "phase": "evolution",
             "source_path": "native_evolution\\records.jsonl", "harness_changed_agent_count": index - 1}
            for index in (1, 2)]
    (curve_dir / "native_evolution_native_evolution.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = historical_curves(tmp_path)
    assert result[0]["profile_behavior_changed_agent_count"] == 0
    assert result[0]["memory_changed_agent_count"] == 0
    assert result[1]["profile_behavior_changed_agent_count"] == 1
    assert result[1]["memory_changed_agent_count"] == 1
    assert result[1]["harness_policy_changed_agent_count"] == 0
    assert any("experience.sqlite#version=1" in source for source in result[1]["state_snapshot_sources"])
    assert any("frozen_plan.json#AgentPool" in source for source in result[1]["state_snapshot_sources"])


def test_partial_batch_preserves_observed_usage_without_false_full_totals():
    recorded = {"tokens": 10, "model_calls": 1, "tool_calls": 0, "cost": None,
                "records": [{"kind": "model", "request": {"transport_retry": {"attempts": [{"status": "success"}]}}}],
                "by_stage": {"inference": {"tokens": 10, "model_calls": 1, "tool_calls": 0, "cost": None}}}
    budget, latency, observed = batch_measurements([
        {"result": {"budget": recorded, "execution_seconds": 2}}, {"result": {}}])
    assert budget["tokens"] is None
    assert budget["model_calls"] is None
    assert budget["by_stage"]["inference"]["tokens"] is None
    assert latency is None
    assert observed["observed_tokens"] == 10
    assert observed["unknown_tokens_slot_count"] == 1
    assert observed["observed_latency_seconds"] == 2
    assert observed["unknown_latency_slot_count"] == 1
    assert observed["api_attempts"] is None
    assert observed["observed_api_attempts"] == 1


def test_null_cumulative_gap_and_requested_operation_alias_remain_visible():
    assert nullable_cumulative([1, None, 2]) == [1, None, None]
    assert operation_curve([{"operation_counts_cumulative_requested": {"add": 2}}, {}], "proposed", "add") == [2, None]


def test_seed_initialization_metadata_is_not_reported_as_behavior_change():
    before = {"profiles": [{"pool_agent_id": "writer", "prompt": "draft", "memory": []}]}
    after = {"profiles": [{"pool_agent_id": "writer", "prompt": "draft", "memory": [],
                           "role": "writer", "capabilities": ["writing"]}]}
    first, second = comparable_profiles(before, after)
    assert first == second
    assert after["profiles"][0]["role"] == "writer"
