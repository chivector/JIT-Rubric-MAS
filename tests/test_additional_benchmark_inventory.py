import hashlib
import json
from pathlib import Path

import pytest

from jit_mas.schemas import digest
from scripts.prepare_additional_benchmark_inventory import RELEASES, build_inventory, project_rows


PUBLIC_FIELDS = {"task_id", "source_id", "source_row", "question_sha256", "group_id", "stratum"}


def test_public_projection_preserves_ids_and_omits_private_fields():
    records = [{"key": 92, "prompt": " A public\nquestion ",
                "instruction_id_list": ["PRIVATE_SENTINEL"], "kwargs": [{"N": 2, "n": 4}]},
               {"key": 7, "prompt": "a public question", "answer": "PRIVATE_SENTINEL"}]
    rows = project_rows("ifbench", records)
    assert [row["source_id"] for row in rows] == [92, 7]
    assert [row["source_row"] for row in rows] == [1, 2]
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert all(set(row) == PUBLIC_FIELDS for row in rows)
    assert "PRIVATE_SENTINEL" not in json.dumps(rows)


def test_writingbench_uses_only_public_language_and_domain():
    rows = project_rows("writingbench", [{"index": 13, "query": "A task",
                        "lang": "en", "domain1": "Education", "domain2": "Detailed",
                        "criteria": [{"score": "PRIVATE_SENTINEL"}]}])
    assert rows[0]["stratum"] == "en|Education"
    assert rows[0]["task_id"] == "writingbench:13"
    assert set(rows[0]) == PUBLIC_FIELDS


@pytest.mark.parametrize("records", [
    [{"key": 1, "prompt": "x"}, {"key": 1, "prompt": "y"}],
    [{"key": True, "prompt": "x"}],
    [{"key": 1, "prompt": None, "private": "PRIVATE_SENTINEL"}],
])
def test_invalid_rows_fail_without_echoing_record_values(records):
    with pytest.raises(ValueError) as error:
        project_rows("ifeval", records)
    assert "PRIVATE_SENTINEL" not in str(error.value)


def test_pinned_source_hash_is_required(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"key":1,"prompt":"not the pinned release"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_inventory("ifeval", path)


def test_inventory_builder_checks_count_and_hashes_output(tmp_path, monkeypatch):
    raw = b'{"key":5,"prompt":"public text","private":"PRIVATE_SENTINEL"}\n'
    path = tmp_path / "input.jsonl"
    path.write_bytes(raw)
    spec = {**RELEASES["ifeval"], "dataset_sha256": hashlib.sha256(raw).hexdigest(), "expected_tasks": 1}
    monkeypatch.setitem(RELEASES, "ifeval", spec)
    document = build_inventory("ifeval", path)
    assert document["rows"][0]["source_id"] == 5
    assert document["inventory_sha256"] == digest({k: v for k, v in document.items() if k != "inventory_sha256"})
    assert "PRIVATE_SENTINEL" not in json.dumps(document)
    monkeypatch.setitem(RELEASES, "ifeval", {**spec, "expected_tasks": 2})
    with pytest.raises(ValueError, match="count mismatch"):
        build_inventory("ifeval", path)


@pytest.mark.parametrize("name,count", [("writingbench", 1000), ("ifeval", 541), ("ifbench", 300)])
def test_committed_inventory_is_complete_public_and_self_authenticating(name, count):
    path = Path(__file__).resolve().parents[1] / "paper" / "experiments" / f"{name}_inventory_v4.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["benchmark"] == name
    assert len(document["rows"]) == count
    assert document["dataset_sha256"] == RELEASES[name]["dataset_sha256"]
    assert document["inventory_sha256"] == digest({k: v for k, v in document.items() if k != "inventory_sha256"})
    assert all(set(row) == PUBLIC_FIELDS for row in document["rows"])
    assert [row["source_row"] for row in document["rows"]] == list(range(1, count + 1))
    assert len({row["task_id"] for row in document["rows"]}) == count
    assert document["exposed_task_ids"] == (["writingbench:1"] if name == "writingbench" else [])
