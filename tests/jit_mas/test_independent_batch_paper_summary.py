"""Release and state checks for the batch protocol's TEST paper summary."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.independent_batch_campaign import slot_registry
from jit_mas.independent_batch_protocol import SOURCES, TARGETS, build_protocol
from jit_mas.schemas import Experience, ExperienceSnapshot, digest
from jit_mas.token_usage import merge_usage, summarize_budget
import scripts.summarize_independent_paper as summary


def released_batch_campaign(directory, *, omit_defaults=False):
    directory.mkdir(parents=True)
    memberships = {source: {
        "evolution": [f"{source}:e{index}" for index in range(40)],
        "validation": [f"{source}:v{index}" for index in range(10)],
        "test": [f"{source}:t{index}" for index in range(33 if source == "researchrubrics" else 50)],
    } for source in SOURCES}
    memberships.update({target: {"test": [f"{target}:t{index}"
                        for index in range(40 if target == "deepresearch_bench_ii" else 50)]} for target in TARGETS})
    manifest = {"memberships": memberships, "joint_evolution_schedule": [
        {"run_id": run_id, "order_seed": run_id,
         "task_ids": [task_id for source in SOURCES for task_id in memberships[source]["evolution"]]}
        for run_id in range(3)]}
    protocol = build_protocol(manifest)
    registry = slot_registry(protocol)["artifacts"]
    execution = {name: name for name in ("launch_file_sha256", "configuration_file_sha256", "joint_manifest_sha256", "runner_sha256")}
    execution["initial_snapshot_hash"] = digest(ExperienceSnapshot())
    registration = {"protocol": protocol, "execution": execution, "schema": "independent-batch-campaign-v6"}
    registration_hash = digest(registration)
    adapter = {"schema": "independent-test-adapter-identity-v1", "protocol_sha256": protocol["protocol_sha256"],
               "adapter_sha256": hashlib.sha256(Path(summary.__file__).with_name("run_independent_test_release.py").read_bytes()).hexdigest(),
               **{name: execution[name] for name in ("launch_file_sha256", "configuration_file_sha256", "joint_manifest_sha256")},
               "evolution_runner_sha256": execution["runner_sha256"]}
    (directory / "test_adapter_identity.json").write_text(json.dumps(adapter), encoding="utf-8")
    states = {source: ExperienceSnapshot(version=1, experiences=[Experience(
        experience_id=f"{source}:lesson", bank="rubric", instruction="Synthetic batch lesson",
        applicability="Synthetic tasks", source_task_ids=memberships[source]["evolution"][:5],
        evidence=["feedback:synthetic"], created_at="2026-10-06T00:00:00+00:00")]).model_dump(mode="json") for source in SOURCES}
    if omit_defaults:
        for state in states.values():
            state.pop("policy_versions")
    budget = {"records": [], "model_calls": 0, "tokens": 0}
    usage = summarize_budget(budget)
    groups, results, usages = {}, {}, []
    with sqlite3.connect(directory / "campaign.sqlite") as database:
        database.executescript("CREATE TABLE metadata(key TEXT PRIMARY KEY, body TEXT);"
                               "CREATE TABLE slots(slot_id TEXT PRIMARY KEY, ordinal INTEGER, body TEXT, status TEXT,"
                               "result TEXT, result_hash TEXT, started_at TEXT, finished_at TEXT);"
                               "CREATE TABLE selections(source TEXT, run_id INTEGER, body TEXT);"
                               "CREATE TABLE checkpoints(source TEXT, run_id INTEGER, position INTEGER, state_hash TEXT, snapshot TEXT, identity_hash TEXT);")
        database.execute("INSERT INTO metadata VALUES('registration',?)", (json.dumps(registration),))
        for ordinal, slot in enumerate(registry):
            result, status = None, "pending"
            if slot["kind"] == "test":
                state_hash = execution["initial_snapshot_hash"] if slot["source"] == "static" else digest(states[slot["source"]])
                result = {"task_id": slot["task_id"], "evaluation": None, "state_hash": state_hash, "budget": budget}
                results[slot["slot_id"]] = digest(result)
                status = "failed"
                usages.append(usage)
                key = (slot["source"], slot["run_id"], slot["target"], slot["method"])
                group = groups.setdefault(key, {"source": key[0], "run_id": key[1], "target": key[2], "method": key[3],
                                               "scores": [], "complete": 0, "slots": 0, "mean_official_score": None})
                group["slots"] += 1
            database.execute("INSERT INTO slots VALUES(?,?,?,?,?,?,?,?)", (
                slot["slot_id"], ordinal, json.dumps(slot), status, json.dumps(result) if result else None,
                digest(result) if result else None, None, None))
        for trajectory in protocol["trajectories"]:
            source, run_id = trajectory["source"], trajectory["run_id"]
            snapshot = states[source]
            selected = {"position": 40, "candidate_index": 0, "state_hash": digest(snapshot)}
            database.execute("INSERT INTO selections VALUES(?,?,?)", (source, run_id, json.dumps({
                "source": source, "run_id": run_id, "selected": selected, "registration_hash": registration_hash})))
            database.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)", (
                source, run_id, 40, digest(snapshot), json.dumps(snapshot), registration_hash))
        database.execute("INSERT INTO metadata VALUES('test_seal',?)", (json.dumps({
            "registration_hash": registration_hash, "required_test_slots": 2751, "submission_hashes": results}),))
    (directory / "test_report.json").write_text(json.dumps({
        "schema": "independent-test-report-v5", "test_feedback_released": True,
        "slots": 2751, "evaluated": 2751, "complete": 0, "incomplete_or_failed": 2751,
        "token_usage": merge_usage(usages), "by_condition": list(groups.values())}), encoding="utf-8")
    return directory


def test_summary_accepts_batch_registry_and_multitask_selected_state(tmp_path):
    campaign = released_batch_campaign(tmp_path / "batch")
    registration, observations, usage = summary.load_test_data(campaign)
    assert registration["schema"] == "independent-batch-campaign-v6"
    assert len(observations) == 2751
    assert all(row["native_score"] is None for row in observations)
    assert usage["total_tokens"] == 0
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        counts = Counter(json.loads(body)["kind"] for body, in database.execute("SELECT body FROM slots"))
    assert counts == {"evolution": 360, "batch_evolution": 72, "validation": 2160, "test": 2751}


def test_summary_preserves_raw_checkpoint_identity_when_schema_adds_defaults(tmp_path):
    campaign = released_batch_campaign(tmp_path / "defaults", omit_defaults=True)
    registration, observations, _ = summary.load_test_data(campaign)
    assert len(observations) == 2751
    assert registration["schema"] == "independent-batch-campaign-v6"
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        state_hash, snapshot_json = database.execute("SELECT state_hash,snapshot FROM checkpoints LIMIT 1").fetchone()
    assert digest(json.loads(snapshot_json)) == state_hash
    assert digest(ExperienceSnapshot.model_validate_json(snapshot_json)) != state_hash


@pytest.mark.parametrize("tamper", ["registration", "inventory", "position", "source"])
def test_batch_summary_rejects_registry_or_final_state_changes(tmp_path, tamper):
    campaign = released_batch_campaign(tmp_path / tamper)
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        if tamper == "registration":
            registration = json.loads(database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()[0])
            registration["schema"] = "independent-campaign-v5"
            database.execute("UPDATE metadata SET body=? WHERE key='registration'", (json.dumps(registration),))
        elif tamper == "inventory":
            database.execute("DELETE FROM slots WHERE ordinal=0")
        elif tamper == "position":
            source, run_id, body = database.execute("SELECT source,run_id,body FROM selections LIMIT 1").fetchone()
            selection = json.loads(body)
            selection["selected"]["position"] = 35
            database.execute("UPDATE selections SET body=? WHERE source=? AND run_id=?", (json.dumps(selection), source, run_id))
        else:
            source, run_id, snapshot_json = database.execute("SELECT source,run_id,snapshot FROM checkpoints LIMIT 1").fetchone()
            snapshot = json.loads(snapshot_json)
            snapshot["experiences"][0]["source_task_ids"].append("outside:evo")
            state_hash = digest(ExperienceSnapshot.model_validate(snapshot))
            database.execute("UPDATE checkpoints SET snapshot=?,state_hash=? WHERE source=? AND run_id=?",
                             (json.dumps(snapshot), state_hash, source, run_id))
            selection = json.loads(database.execute("SELECT body FROM selections WHERE source=? AND run_id=?", (source, run_id)).fetchone()[0])
            selection["selected"]["state_hash"] = state_hash
            database.execute("UPDATE selections SET body=? WHERE source=? AND run_id=?", (json.dumps(selection), source, run_id))
    with pytest.raises(CheckpointIntegrityError):
        summary.load_test_data(campaign)
