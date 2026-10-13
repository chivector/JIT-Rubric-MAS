import csv
import json

from jit_mas.experiment_telemetry import EvolutionTelemetry


def budget(*, tokens=10, calls=1, cost=None, stage="inference", error=""):
    return {
        "model_calls": calls, "tokens": tokens, "tool_calls": 0, "cost": cost,
        "wall_seconds": 1.25,
        "by_stage": {stage: {"model_calls": calls, "tokens": tokens,
                              "tool_calls": 0, "cost": cost}},
        "records": [{"kind": "model", "stage": stage, "error": error,
                     "request": {"transport_retry": {"attempts": (
                         [{"status": "retry"}, {"status": "success"}] if error
                         else [{"status": "success"}])}}}],
    }


def pools(harness="a", extra=False):
    profiles = [{"pool_agent_id": "writer", "harness": {"prompt": harness}}]
    if extra:
        profiles.append({"pool_agent_id": "critic", "harness": {"prompt": "critic"}})
    return {"version": 1, "profiles": profiles}


def test_round_preserves_official_metrics_and_requested_applied_operations():
    telemetry = EvolutionTelemetry()
    row = telemetry.record_round(
        0,
        performance={"Overall": 0.8, "InformationRecall": 0.75,
                     "Analysis": 0.9, "Presentation": 0.7},
        budget=budget(tokens=30, calls=2, cost=None),
        pool_before=pools(), pool_after=pools("b", extra=True),
        operations_requested=[{"operation_id": "x", "kind": "add"},
                              {"operation_id": "y", "kind": "merge"}],
        operations_applied=[{"operation_id": "x", "kind": "add"}],
        outcome={"evaluation": {"complete": True}},
    )
    assert row["performance"]["Overall"] == 0.8
    assert row["pool_agent_count_before"] == 1 and row["pool_agent_count_after"] == 2
    assert row["operation_counts_requested"]["merge"] == 1
    assert row["operation_counts_applied"]["merge"] == 0
    assert row["harness_changed_agent_count"] == 1
    assert row["harness_added_agent_ids"] == ["critic"]
    assert row["cost"] is None and row["cost_known"] is False


def test_cumulative_missingness_failure_retry_and_stage_metrics():
    telemetry = EvolutionTelemetry()
    first = telemetry.record_round(0, budget=budget(tokens=10, calls=1, cost=0.2),
                                   outcome={"complete": True})
    second = telemetry.record_round(1, budget=budget(tokens=4, calls=1, cost=None, error="429"),
                                    outcome={"complete": False, "error": "provider"},
                                    operations_requested=[])
    assert first["cumulative"]["tokens"] == 10
    assert second["cumulative"]["tokens"] == 14
    assert second["failure"] is True and second["failed_calls"] == 1
    assert second["retry_count"] == 1
    assert second["cumulative_failure_rate"] == 0.5
    assert second["operation_counts_requested"] == {kind: 0 for kind in (
        "add", "delete", "prune", "split", "merge", "specialize", "reorganize")}
    assert second["cumulative_by_stage"]["inference"]["tokens"] == 14
    assert first["retry_count"] == 0
    assert second["api_attempts"] == 2


def test_jsonl_and_csv_are_readable(tmp_path):
    telemetry = EvolutionTelemetry()
    telemetry.record_round(0, performance={"score": 1}, budget=budget())
    jsonl = telemetry.write_jsonl(tmp_path / "curve.jsonl")
    csv_path = telemetry.write_csv(tmp_path / "curve.csv")
    assert json.loads(jsonl.read_text(encoding="utf-8"))["round_index"] == 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["round_index"] == "0"
    assert json.loads(rows[0]["performance"]) == {"score": 1}


def test_unknown_pool_and_operations_remain_null():
    row = EvolutionTelemetry().record_round(0, budget=None)
    assert row["pool_agent_count_before"] is None
    assert row["operation_counts_requested"] is None
    assert row["model_calls"] is None
    assert row["cumulative"]["tokens"] is None


def test_snapshot_hashes_full_harness_without_counting_version_and_history():
    before = {"agent_pool": {"profiles": [{"pool_agent_id": "writer", "version": 1,
                                           "prompt": "old", "harness": {"memory": "full"}}]}}
    after = json.loads(json.dumps(before))
    after["agent_pool"]["profiles"][0].update(version=2, prompt="new")
    row = EvolutionTelemetry().record_round(0, pool_before=before, pool_after=after)
    assert row["pool_agent_count_before"] == 1
    assert row["harness_changed_agent_ids"] == ["writer"]


def test_pool_memory_updates_count_without_counting_agent_version():
    telemetry = EvolutionTelemetry()
    before = {"profiles": [{"pool_agent_id": "writer", "version": 1,
                            "memory": {"lesson": "old"}}]}
    version_only = json.loads(json.dumps(before))
    version_only["profiles"][0]["version"] = 2
    first = telemetry.record_round(0, pool_before=before, pool_after=version_only)
    after = json.loads(json.dumps(version_only))
    after["profiles"][0]["memory"]["lesson"] = "new"
    second = telemetry.record_round(1, pool_before=version_only, pool_after=after)
    assert first["harness_changed_agent_count"] == 0
    assert second["harness_changed_agent_ids"] == []
    assert second["memory_changed_agent_ids"] == ["writer"]
    assert second["profile_behavior_changed_agent_count"] == 0
    assert second["harness_policy_changed_agent_count"] == 0
    assert second["legacy_harness_changed_agent_ids"] == ["writer"]
    assert second["cumulative_harness_changed_agent_count"] == 0
    assert second["cumulative_memory_changed_agent_count"] == 1
    assert second["cumulative_legacy_harness_changed_agent_count"] == 1


def test_behavior_and_policy_changes_are_distinct_from_memory_and_provenance():
    telemetry = EvolutionTelemetry()
    before = {"profiles": [{"pool_agent_id": "writer", "version": 1, "role": "writer",
                            "prompt": "draft", "memory": [{"instruction": "be concise"}],
                            "source_task_ids": ["old"], "evidence": ["old"],
                            "harness": {"memory_policy": "full", "memory_window": 8}}]}
    prompt_after = json.loads(json.dumps(before))
    prompt_after["profiles"][0].update(prompt="revise", version=2, source_task_ids=["new"], evidence=["new"])
    first = telemetry.record_round(0, pool_before=before, pool_after=prompt_after)
    assert first["harness_changed_agent_ids"] == ["writer"]
    assert first["profile_behavior_changed_agent_ids"] == ["writer"]
    assert first["harness_policy_changed_agent_ids"] == []
    assert first["memory_changed_agent_ids"] == []
    assert first["profile_behavior_hashes_before"]["writer"] != first["profile_behavior_hashes_after"]["writer"]
    policy_after = json.loads(json.dumps(prompt_after))
    policy_after["profiles"][0]["harness"]["memory_policy"] = "recent"
    second = telemetry.record_round(1, pool_before=prompt_after, pool_after=policy_after)
    assert second["harness_policy_changed_agent_ids"] == ["writer"]
    assert second["memory_changed_agent_ids"] == []
    assert second["cumulative_profile_behavior_changed_agent_count"] == 2
    assert second["cumulative_harness_policy_changed_agent_count"] == 1
    assert second["cumulative_harness_changed_agent_count"] == 2


def test_memory_metadata_and_profile_provenance_do_not_change_content_hashes():
    before = {"profiles": [{"pool_agent_id": "writer", "version": 1, "role": "writer", "prompt": "draft",
                            "memory": [{"instruction": "be concise", "created_at": "earlier", "version": 1,
                                        "source_task_ids": ["task"], "evidence": ["trace"]}],
                            "source_task_ids": ["old"], "evidence": ["old"]}]}
    after = json.loads(json.dumps(before))
    after["profiles"][0].update(version=2, source_task_ids=["new"], evidence=["new"], parent_agent_id="critic")
    after["profiles"][0]["memory"][0].update(created_at="later", version=2)
    row = EvolutionTelemetry().record_round(0, pool_before=before, pool_after=after)
    for domain in ("harness", "memory", "profile_behavior", "harness_policy"):
        assert row[f"{domain}_changed_agent_count"] == 0
    assert row["legacy_harness_changed_agent_count"] == 1
    assert row["memory_hashes_before"] == row["memory_hashes_after"]
    assert row["change_hash_content"]["memory"]["recursive_excluded_fields"] == ["created_at", "version"]


def test_structural_membership_changes_do_not_count_as_content_modifications():
    before = {"profiles": [{"pool_agent_id": "writer", "prompt": "draft", "memory": []},
                            {"pool_agent_id": "removed", "prompt": "old"}]}
    after = {"profiles": [{"pool_agent_id": "writer", "prompt": "draft", "memory": []},
                           {"pool_agent_id": "added", "prompt": "new", "memory": [{"instruction": "new"}]}]}
    row = EvolutionTelemetry().record_round(0, pool_before=before, pool_after=after,
                                             operations_applied=[{"kind": "add"}, {"kind": "delete"}])
    assert row["pool_added_agent_ids"] == ["added"]
    assert row["pool_removed_agent_ids"] == ["removed"]
    assert row["harness_added_agent_ids"] == ["added"]
    assert row["operation_counts_applied"]["add"] == 1
    assert row["operation_counts_applied"]["delete"] == 1
    for domain in ("harness", "legacy_harness", "memory", "profile_behavior", "harness_policy"):
        assert row[f"{domain}_changed_agent_ids"] == []
        assert row[f"cumulative_{domain}_changed_agent_count"] == 0


def test_explicit_harness_maps_and_unknown_profile_cumulative_values_remain_separate():
    telemetry = EvolutionTelemetry()
    first = telemetry.record_round(0, harness_before={"writer": {"version": 1, "code": "old"}},
                                   harness_after={"writer": {"version": 2, "code": "new"}})
    assert first["harness_changed_agent_count"] == 1
    assert first["memory_changed_agent_count"] is None
    assert first["profile_behavior_changed_agent_count"] is None
    second = telemetry.record_round(1, pool_before=pools(), pool_after=pools())
    assert second["memory_changed_agent_count"] == 0
    assert second["cumulative_memory_changed_agent_count"] is None
    assert second["cumulative_profile_behavior_changed_agent_count"] is None
    assert second["cumulative_harness_changed_agent_count"] == 1
    assert telemetry.rows[0]["harness_changed_agent_count"] == 1


def test_history_rows_do_not_mutate_and_stage_absence_is_zero_if_snapshot_complete():
    telemetry = EvolutionTelemetry()
    telemetry.record_round(0, budget=budget(cost=0.1), operations_applied=[{"kind": "add"}])
    telemetry.record_round(1, budget=budget(cost=0.2, stage="evaluation"), operations_applied=[])
    rows = telemetry.rows
    assert rows[0]["operation_counts_cumulative_applied"]["add"] == 1
    assert rows[1]["cumulative_by_stage"]["inference"]["tokens"] == 10
    telemetry.record_round(2, budget=budget(cost=0.1), operations_applied=[{"kind": "add"}])
    assert telemetry.rows[0]["operation_counts_cumulative_applied"]["add"] == 1
