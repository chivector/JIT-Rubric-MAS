import json
from pathlib import Path

import pytest

from scripts.execute_joint_experiment import JointExecutor
from jit_mas.experience import ExperienceStore


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/experiments/joint_protocol_v5.json"
MANIFEST = ROOT / "paper/experiments/joint_task_splits_v5.json"


def _bundle(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("backend: scripted\n", encoding="utf-8")
    benchmarks = {}
    for name in ("researchrubrics", "deepsearchqa", "writingbench", "deepresearch_bench_ii", "ifeval", "ifbench"):
        data = tmp_path / f"{name}.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        split = tmp_path / f"{name}.split.json"
        split.write_text("{}\n", encoding="utf-8")
        evidence = tmp_path / f"{name}.evidence"
        evidence.mkdir()
        (evidence / "manifest.json").write_text("{}\n", encoding="utf-8")
        benchmarks[name] = {"data": str(data), "splits": str(split), "evidence_dir": str(evidence)}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"protocol": str(PROTOCOL), "manifest": str(MANIFEST),
                                  "config": str(config), "benchmarks": benchmarks}, indent=2), encoding="utf-8")
    return bundle


def test_metadata_check_binds_all_inputs_without_provider(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    result = executor.check()
    assert result["ready"] is True
    assert result["provider_checked"] is False
    assert result["material_count"] == 6
    assert result["test_supported"] is False
    assert (tmp_path / "run" / "registration.json").is_file()


def test_slot_inventory_is_three_runs_and_no_test_slots(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    slots = executor._slots()
    assert len(slots) == 630
    assert sum(row["kind"] == "evolution" for row in slots.values()) == 180
    assert sum(row["kind"] == "validation" for row in slots.values()) == 450
    assert not any(row["kind"] == "test" for row in slots.values())


def test_global_runtime_manifest_uses_integer_order_seed(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    manifest = executor._global_manifest(0)
    assert manifest.seed == 20261001


def test_checkpoint_persistence_materializes_immutable_snapshot(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    store = ExperienceStore(tmp_path / "mutable.sqlite")
    try:
        record = executor._persist_checkpoint(store, 0, 0)
    finally:
        store.close()
    assert record["state_hash"]
    snapshot_path = Path(record["snapshot_path"])
    assert snapshot_path.is_file()
    frozen = ExperienceStore(snapshot_path, read_only=True)
    try:
        assert frozen.snapshot().version == 0
    finally:
        frozen.close()
    resumed_store = ExperienceStore(tmp_path / "mutable.sqlite", read_only=True)
    try:
        assert executor._persist_checkpoint(resumed_store, 0, 0)["state_hash"] == record["state_hash"]
    finally:
        resumed_store.close()


def test_assert_frozen_rejects_checkpoint_snapshot_change(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    store = ExperienceStore(tmp_path / "mutable.sqlite")
    try:
        record = executor._persist_checkpoint(store, 0, 0)
    finally:
        store.close()
    doc = executor._load_or_init()
    doc["checkpoints"]["run0:c0"] = record
    executor._journal().write_text(json.dumps(doc), encoding="utf-8")
    Path(record["snapshot_path"]).write_bytes(b"tampered")
    with pytest.raises(Exception, match="snapshot changed"):
        executor.assert_frozen()


def test_started_slot_is_consumed_as_failure_and_never_retried(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    slot_id = next(key for key, row in document["slots"].items() if row["kind"] == "evolution")
    first = executor._consume(document, slot_id, lambda: {"complete": True, "score": 1})
    assert first["status"] == "complete"
    called = []
    second = executor._consume(document, slot_id, lambda: called.append(True))
    assert second["status"] == "complete"
    assert called == []

    other = next(key for key, row in document["slots"].items() if row["kind"] == "evolution" and key != slot_id)
    document["slots"][other]["status"] = "started"
    recovered = executor._consume(document, other, lambda: (_ for _ in ()).throw(AssertionError("must not retry")))
    assert recovered["status"] == "failed"
    assert recovered["result"]["error_type"] == "InterruptedWithoutDurableOutcome"


def test_assert_frozen_rejects_data_change(tmp_path):
    bundle = _bundle(tmp_path)
    executor = JointExecutor(bundle, tmp_path / "run")
    executor.check()
    data = next(iter(executor.bundle["benchmarks"].values()))["data"]
    Path(data).write_text("changed\n", encoding="utf-8")
    with pytest.raises(Exception, match="changed"):
        executor.assert_frozen()


def test_test_mode_is_explicitly_unavailable(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    with pytest.raises(NotImplementedError, match="TestRelease"):
        executor.run_test()


def test_provider_run_requires_non_synthetic_preflight(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    with pytest.raises(ValueError, match="provider preflight"):
        executor.check(require_provider=True)
