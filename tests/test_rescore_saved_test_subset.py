import json
from pathlib import Path
import sqlite3

import pytest

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.config import MASConfig
from jit_mas.schemas import ExperienceSnapshot, digest
import scripts.rescore_saved_test_subset as rescore


def campaign_fixture(tmp_path):
    campaign = tmp_path / "source"
    campaign.mkdir()
    data = campaign / "synthetic.jsonl"
    data.write_text("\n".join(json.dumps({"sample_id": "rr:" + str(index), "prompt": "Synthetic public task " + str(index),
                    "rubrics": [{"criterion": "synthetic marker", "weight": 1}]}) for index in range(3)), encoding="utf-8")
    registration = {"schema": "synthetic-independent-campaign", "bounds": {},
                    "execution": {"configuration": MASConfig(backend="scripted").model_dump(mode="json"),
                                  "materials": {"researchrubrics": {"data": str(data),
                                                "dataset_sha256": rescore.file_hash(data)}}},
                    "protocol": {"sources": ["researchrubrics"], "target_test_tasks": {},
                                 "trajectories": [{"source": "researchrubrics", "run_id": 0,
                                                  "evolution_task_ids": [], "test_task_ids": ["rr:0", "rr:1", "rr:2"]}]}}
    snapshot = ExperienceSnapshot()
    selected = {"position": 40, "state_hash": digest(snapshot)}
    rows = []
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        database.executescript("""
            CREATE TABLE metadata(key TEXT, body TEXT);
            CREATE TABLE selections(source TEXT, run_id INTEGER, body TEXT);
            CREATE TABLE checkpoints(source TEXT, run_id INTEGER, position INTEGER, state_hash TEXT, snapshot TEXT, identity_hash TEXT);
            CREATE TABLE slots(slot_id TEXT, ordinal INTEGER, body TEXT, status TEXT, result TEXT, result_hash TEXT, started_at TEXT, finished_at TEXT);
        """)
        database.execute("INSERT INTO metadata VALUES('registration',?)", (json.dumps(registration),))
        database.execute("INSERT INTO selections VALUES('researchrubrics',0,?)", (json.dumps({"selected": selected}),))
        database.execute("INSERT INTO checkpoints VALUES('researchrubrics',0,40,?,?,?)",
                         (digest(snapshot), snapshot.model_dump_json(), digest(registration)))
        for ordinal, index in enumerate((2, 0, 1)):
            task_id = "rr:" + str(index)
            slot = {"slot_id": "test:" + task_id, "kind": "test", "source": "researchrubrics",
                    "target": "researchrubrics", "run_id": 0, "task_id": task_id}
            answer = "Boundary conditions: synthetic saved answer " + str(index)
            submission = {"answer": answer, "answer_hash": digest(answer)}
            path = campaign / (str(index) + ".json")
            path.write_text(json.dumps(submission), encoding="utf-8")
            result = {"task_id": task_id, "submission_path": str(path), "submission_hash": digest(submission),
                      "answer_hash": digest(answer), "state_hash": digest(snapshot),
                      "budget": {"model_calls": 2, "tokens": 100, "tool_calls": 0, "reserved_tokens": 0}}
            row = {**slot, "status": "submitted", "result": result, "result_hash": digest(result),
                   "started_at": "fixed", "finished_at": "fixed"}
            rows.append(row)
            database.execute("INSERT INTO slots VALUES(?,?,?,?,?,?,?,?)",
                             (slot["slot_id"], ordinal, json.dumps(slot), "submitted", json.dumps(result), digest(result), "fixed", "fixed"))
        seal = {"registration_hash": digest(registration), "required_test_slots": 3,
                "submission_hashes": {row["slot_id"]: row["result_hash"] for row in rows}}
        database.execute("INSERT INTO metadata VALUES('test_seal',?)", (json.dumps(seal),))
    return campaign, rows


def write_original_receipt(campaign, row, evaluation_budget, **values):
    receipt = {"slot": row, "submission_sha256": row["result"]["submission_hash"],
               "evaluation_budget": evaluation_budget, "complete": False,
               "status": "evaluation_failed", **values}
    receipt["receipt_sha256"] = digest(receipt)
    path = campaign / "test_evaluations" / (digest(row["slot_id"]) + ".json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_fixed_prefix_uses_registered_ordinal_and_rejects_answer_tampering(tmp_path):
    campaign, rows = campaign_fixture(tmp_path)
    prepared = rescore.prepare_subset(campaign, "researchrubrics", 2)
    assert [item["row"]["task_id"] for item in prepared["selected"]] == ["rr:2", "rr:0"]
    Path(rows[0]["result"]["submission_path"]).write_text('{"answer":"changed","answer_hash":"bad"}', encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="answer or state"):
        rescore.prepare_subset(campaign, "researchrubrics", 2)


def test_historical_judge_cost_is_deducted_and_unknown_requires_explicit_diagnosis(tmp_path):
    campaign, rows = campaign_fixture(tmp_path)
    write_original_receipt(campaign, rows[0], {"model_calls": 3, "tokens": 120, "tool_calls": 0})
    prepared = rescore.prepare_subset(campaign, "researchrubrics", 1)
    assert prepared["selected"][0]["cumulative_task_budget"]["tokens"] == 220
    assert prepared["selected"][0]["cumulative_task_budget"]["model_calls"] == 5
    path = write_original_receipt(campaign, rows[0], {"usage_unknown": True}, error_type="AttributeError")
    before = path.read_bytes()
    with pytest.raises(CheckpointIntegrityError, match="unknown"):
        rescore.prepare_subset(campaign, "researchrubrics", 1)
    prepared = rescore.prepare_subset(campaign, "researchrubrics", 1, recover_prejudge_path_error=True)
    diagnosis = prepared["selected"][0]["prejudge_failure_recovery"]
    assert diagnosis["original_evaluation_budget"] == {"usage_unknown": True}
    assert diagnosis["judge_calls_inferred_from_diagnosed_path"] == 0
    assert path.read_bytes() == before
    write_original_receipt(campaign, rows[0], {"usage_unknown": True}, error_type="TimeoutError")
    with pytest.raises(CheckpointIntegrityError, match="unknown"):
        rescore.prepare_subset(campaign, "researchrubrics", 1, recover_prejudge_path_error=True)


def test_interrupted_judgment_and_existing_output_cannot_trigger_calls(tmp_path, monkeypatch):
    campaign, rows = campaign_fixture(tmp_path)
    marker = campaign / "test_evaluation_started" / (digest(rows[0]["slot_id"]) + ".json")
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="Interrupted"):
        rescore.prepare_subset(campaign, "researchrubrics", 1)
    output = tmp_path / "existing-output"
    output.mkdir()
    monkeypatch.setattr(rescore, "make_scoring_pipeline", lambda *arguments: pytest.fail("Must not construct a scorer"))
    with pytest.raises(ValueError, match="new output"):
        rescore.run_subset(campaign, output, "researchrubrics", 1)


def test_judge_provider_rejects_actor_roles_before_native_creation():
    provider = object.__new__(rescore.JudgeOnlyModels)
    with pytest.raises(ValueError, match="only evaluator judge"):
        provider.create("exec", "actor", None, "execution")
    with pytest.raises(ValueError, match="only evaluator judge"):
        provider.create("judge", "judge", None, "inference")


def test_seal_tamper_rejected_before_evaluation(tmp_path):
    campaign, rows = campaign_fixture(tmp_path)
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        seal = json.loads(database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()[0])
        seal["submission_hashes"].pop(rows[0]["slot_id"])
        database.execute("UPDATE metadata SET body=? WHERE key='test_seal'", (json.dumps(seal),))
    with pytest.raises(CheckpointIntegrityError, match="seal differs"):
        rescore.prepare_subset(campaign, "researchrubrics", 1)


def test_submission_path_must_remain_inside_source_campaign(tmp_path):
    campaign, rows = campaign_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"answer": "outside", "answer_hash": digest("outside")}), encoding="utf-8")
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        result = json.loads(database.execute("SELECT result FROM slots WHERE slot_id=?", (rows[0]["slot_id"],)).fetchone()[0])
        result["submission_path"] = str(outside)
        database.execute("UPDATE slots SET result=?, result_hash=? WHERE slot_id=?",
                         (json.dumps(result), digest(result), rows[0]["slot_id"]))
        seal = json.loads(database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()[0])
        seal["submission_hashes"][rows[0]["slot_id"]] = digest(result)
        database.execute("UPDATE metadata SET body=? WHERE key='test_seal'", (json.dumps(seal),))
    with pytest.raises(CheckpointIntegrityError, match="escapes"):
        rescore.prepare_subset(campaign, "researchrubrics", 1)


def test_end_to_end_judge_only_subset_preserves_source_and_blocks_replay(tmp_path, monkeypatch):
    from jit_mas.offline import FixtureModels

    campaign, rows = campaign_fixture(tmp_path)
    provider = FixtureModels()
    monkeypatch.setattr(rescore, "JudgeOnlyModels", lambda config: provider)
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        registration = json.loads(database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()[0])
        registration["execution"]["configuration"]["models"] = {
            "judge": {"model": "synthetic", "endpoint": "https://example.invalid/v1", "key_env": "NO_NETWORK"}}
        database.execute("UPDATE metadata SET body=? WHERE key='registration'", (json.dumps(registration),))
        seal = json.loads(database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()[0])
        seal["registration_hash"] = digest(registration)
        database.execute("UPDATE metadata SET body=? WHERE key='test_seal'", (json.dumps(seal),))
        database.execute("UPDATE checkpoints SET identity_hash=?", (digest(registration),))
    before = {str(path): path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
    output = tmp_path / "rescore-v1"
    summary = rescore.run_subset(campaign, output, "researchrubrics", 2)
    assert summary["complete"] == 2
    assert summary["actor_calls"] == 0
    assert [result["task_id"] for result in summary["results"]] == ["rr:2", "rr:0"]
    assert len(provider.calls) == 2 and all(call["role"] == "judge" for call in provider.calls)
    assert all(Path(path).read_bytes() == contents for path, contents in before.items())
    assert len(list((output / "started").glob("*.json"))) == 2
    with pytest.raises(ValueError, match="new output"):
        rescore.run_subset(campaign, output, "researchrubrics", 2)
    assert len(provider.calls) == 2
