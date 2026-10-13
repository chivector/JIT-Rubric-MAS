import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.schemas import ExperienceSnapshot, digest
from scripts.run_test_adapted_final import (
    TestAdaptedFinalRunner, _checkpoint, _manifest,
)


def _campaign(path: Path, source="researchrubrics", position=35):
    snapshot = ExperienceSnapshot()
    db = path / "campaign.sqlite"
    path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE checkpoints(source TEXT, run_id INTEGER, position INTEGER, state_hash TEXT, snapshot TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?)",
                     (source, 0, position, digest(snapshot), snapshot.model_dump_json()))
    return db, snapshot


def test_checkpoint_reader_binds_immutable_snapshot(tmp_path):
    db, snapshot = _campaign(tmp_path)
    loaded, state_hash = _checkpoint(db, "researchrubrics", 35)
    assert loaded == snapshot
    assert state_hash == digest(snapshot)


def test_checkpoint_reader_rejects_changed_snapshot(tmp_path):
    db, _ = _campaign(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE checkpoints SET snapshot=?", (ExperienceSnapshot(version=1).model_dump_json(),))
    with pytest.raises(Exception):
        _checkpoint(db, "researchrubrics", 35)


def test_manifest_uses_full_evidence_membership_and_if_test_only(tmp_path):
    env = SimpleNamespace(
        output=Path(tmp_path),
        joint={"memberships": {
            "deepsearchqa": {"evolution": ["e1"], "validation": ["v1"], "test": ["t1"]},
            "ifeval": {"evolution": ["ie"], "validation": ["iv"], "test": ["it1"]},
        }},
    )
    research = json.loads(_manifest(env, "deepsearchqa").read_text())
    instruction = json.loads(_manifest(env, "ifeval").read_text())
    assert research["runtime_split_manifest"]["evolution"] == ["e1"]
    assert research["runtime_split_manifest"]["validation"] == ["v1"]
    assert instruction["runtime_split_manifest"] == {
        "seed": 0, "evolution": [], "validation": [], "test": ["it1"], "stream": []
    }


def test_result_record_exposes_adaptation_and_usage_only():
    outcome = {"task_id": "t1", "source_state_hash": "s", "run_dir": "r",
               "evaluation": {"complete": True, "score": 0.75},
               "budget": {"records": [{"kind": "model", "stage": "inference",
                                           "input_tokens": 3, "output_tokens": 4,
                                           "estimated": False}]}}
    record = TestAdaptedFinalRunner._result_record("ifeval", "t1", outcome, 35, 0)
    assert record["test_adapted"] is True
    assert record["rubric_exposed"] is True
    assert record["reference_answers_exposed"] is False
    assert record["score"] == 0.75
    assert record["token_usage"]["total_tokens"] == 7
