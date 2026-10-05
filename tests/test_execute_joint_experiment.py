import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.execute_joint_experiment import JointExecutor
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import ChangeProposal, Experience
from jit_mas.schemas import digest


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


def test_existing_checkpoint_is_not_rewritten_after_live_store_advances(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    store = ExperienceStore(tmp_path / "mutable.sqlite")
    try:
        document = executor._load_or_init()
        initial = executor._persist_checkpoint(store, 0, 0)
        document["checkpoints"]["run0:c0"] = initial
        executor._journal().write_text(json.dumps(document), encoding="utf-8")
        store.commit(ChangeProposal(
            proposal_id="proposal-resume",
            source_task_id="synthetic-evo",
            base_version=0,
            experience=Experience(
                experience_id="experience-resume",
                bank="rubric",
                instruction="Use a compact checklist.",
                applicability="writing",
                source_task_ids=["synthetic-evo"],
                evidence=["synthetic-evidence"],
            ),
            diff="add checklist",
            rationale="resume test",
            evidence=["synthetic-evidence"],
            expected_benefit="stable output",
        ))
        preserved = executor._ensure_checkpoint(document, store, 0, 0)
        assert preserved == initial
        frozen = ExperienceStore(Path(initial["snapshot_path"]), read_only=True)
        try:
            assert frozen.snapshot().version == 0
        finally:
            frozen.close()
    finally:
        store.close()


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


def test_validation_reads_registered_checkpoint_after_live_store_advances(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    live = ExperienceStore(tmp_path / "live.sqlite")
    try:
        executor._ensure_checkpoint(document, live, 0, 0)
        task = executor._membership("researchrubrics", "validation")[0]
        seen = []

        class FakePipeline:
            def run_task(self, task_id, snapshot, **kwargs):
                seen.append(snapshot.version)
                return {"evaluation": {"complete": True, "score": 0.5}}

        dataset = SimpleNamespace(lower_bounds={task: 0.0}, upper_bounds={task: 1.0})
        # The live store is intentionally the mutable trajectory object.  A
        # fake pipeline would expose a later live version if _run_val used it;
        # the registered checkpoint must remain the only VAL input.
        live.db.execute("UPDATE state SET value='7' WHERE key='current'")
        live.db.commit()
        executor._run_val(document, {"researchrubrics": FakePipeline()},
                          {"researchrubrics": dataset}, "researchrubrics", 0, 0, task)
        assert seen == [0]
    finally:
        live.close()


def test_existing_checkpoint_is_not_rewritten_from_later_live_state(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    live = ExperienceStore(tmp_path / "live.sqlite")
    try:
        original = executor._ensure_checkpoint(document, live, 0, 15)
        # Simulate a trajectory that has advanced after C15.  Re-entering the
        # C15 loop must verify the immutable file and return it unchanged.
        live.db.execute("UPDATE state SET value='0' WHERE key='current'")
        live.db.commit()
        resumed = executor._ensure_checkpoint(document, live, 0, 15)
        assert resumed == original
        frozen = executor._open_checkpoint(document, 0, 15)
        try:
            assert frozen.snapshot().version == 0
        finally:
            frozen.close()
    finally:
        live.close()


def test_mid_c30_unresolved_prefix_fails_closed(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    live = ExperienceStore(tmp_path / "live.sqlite")
    try:
        executor._ensure_checkpoint(document, live, 0, 0)
        executor._ensure_checkpoint(document, live, 0, 15)
        executor._ensure_checkpoint(document, live, 0, 30)
        after = digest(live.snapshot())
        order = [row for row in document["slots"].values()
                 if row["kind"] == "evolution" and row["run_id"] == 0]
        order.sort(key=lambda row: row["ordinal"])
        for row in order[:30]:
            row.update(status="complete", result={"complete": True,
                       "after_state_sha256": after})
        order[30]["status"] = "started"
        with pytest.raises(Exception, match="Unresolved started EVO slot"):
            executor._assert_live_prefix(document, live, 0)
    finally:
        live.close()


def test_run_without_eligible_checkpoint_is_reported_inconclusive(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    record = executor._select_run(document, 0)
    assert record["status"] == "inconclusive"
    assert record["selected"] is None


def test_full_run_with_ineligible_replicate_writes_report_without_resampling(tmp_path, monkeypatch):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    check_inputs = executor.check
    monkeypatch.setattr(executor, "check", lambda **kwargs: check_inputs(require_provider=False))
    calls = {"evolution": 0, "validation": 0}

    class SyntheticPipeline:
        def __init__(self, store, benchmark, run_id):
            self.store, self.benchmark, self.run_id = store, benchmark, run_id

        def run(self, mode, task_ids):
            assert mode == "evolve" and len(task_ids) == 1
            calls["evolution"] += 1
            return [{"evaluation": {"complete": True, "score": 0.5}}]

        def run_task(self, task_id, snapshot, **kwargs):
            assert kwargs["mode"] == "validate" and kwargs["attribution"] is False
            calls["validation"] += 1
            complete = self.run_id != 0 or self.benchmark != "researchrubrics"
            return {"evaluation": {"complete": complete, "score": 0.5,
                                   "raw": {"native_mean": 5.5}}}

    def synthetic_runtime(run_id):
        store = ExperienceStore(executor.output / f"run{run_id}" / "experience.sqlite")
        names = ("researchrubrics", "deepsearchqa", "writingbench")
        pipelines = {name: SyntheticPipeline(store, name, run_id) for name in names}
        datasets = {name: SimpleNamespace(
            lower_bounds={task: 0.0 for task in executor._membership(name, "validation")},
            upper_bounds={task: 1.0 for task in executor._membership(name, "validation")})
            for name in names}
        return store, pipelines, datasets, executor._global_manifest(run_id)

    monkeypatch.setattr(executor, "_build_runtime", synthetic_runtime)
    report = executor.run_evolution_validation()
    assert report["status"] == "evo_val_inconclusive_test_pending"
    assert report["formal_test_ready"] is False
    assert report["test_feedback_released"] is False
    assert report["selections"]["0"]["selected"] is None
    assert all(report["selections"][str(run_id)]["selected"]["eligible"] for run_id in (1, 2))
    assert calls == {"evolution": 180, "validation": 450}
    journal = json.loads(executor._journal().read_text(encoding="utf-8"))
    assert len(journal["slots"]) == 630
    assert all(row["status"] in {"complete", "incomplete"} for row in journal["slots"].values())
    assert journal["status"] == report["status"]
    assert json.loads((executor.output / "evo_val_report.json").read_text(encoding="utf-8")) == report
    assert executor.run_evolution_validation() == report
    assert calls == {"evolution": 180, "validation": 450}


def test_journal_consume_is_thread_safe_for_independent_val_slots(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run", val_workers=4)
    executor.check()
    document = executor._load_or_init()
    slots = [sid for sid, row in document["slots"].items() if row["kind"] == "validation"][:8]

    def consume(slot_id):
        return executor._consume(document, slot_id,
                                 lambda: (time.sleep(0.01) or {"complete": True, "score": 1.0}))

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(consume, slots))
    assert all(row["status"] == "complete" for row in rows)
    assert all(document["slots"][sid]["status"] == "complete" for sid in slots)
    assert json.loads(executor._journal().read_text(encoding="utf-8"))["slots"]


def test_validation_worker_pool_keeps_fixed_slot_inventory(tmp_path):
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run", val_workers=3)
    calls, guard = [], threading.Lock()
    active = [0]
    maximum = [0]

    def fake_run_val(document, pipelines, datasets, benchmark, run_id, checkpoint, task):
        with guard:
            calls.append((benchmark, task))
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        time.sleep(0.01)
        with guard:
            active[0] -= 1

    executor._run_val = fake_run_val
    executor._run_validation_checkpoint({}, {}, {}, 0, 0)
    assert len(calls) == 30
    assert len(set(calls)) == 30
    assert maximum[0] >= 2


@pytest.mark.parametrize("kind", ["validation", "evolution"])
@pytest.mark.parametrize("error_type", [ValueError, TimeoutError, ConnectionError])
def test_failed_slot_consumes_one_callback_even_with_retry_environment(tmp_path, monkeypatch, kind, error_type):
    monkeypatch.setenv("JIT_MAS_SLOT_ATTEMPTS", "5")
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    slot_id = next(sid for sid, row in document["slots"].items() if row["kind"] == kind)
    calls = []

    def generate_once():
        calls.append(True)
        raise error_type("synthetic failure")

    row = executor._consume(document, slot_id, generate_once)
    assert row["status"] == "failed"
    assert row["attempts"] == 1
    assert row["result"]["error_type"] == error_type.__name__
    assert calls == [True]
    executor._consume(document, slot_id, generate_once)
    assert calls == [True]


def test_validation_parser_failure_is_single_shot_and_records_safe_code_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("JIT_MAS_SLOT_ATTEMPTS", "7")
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    slot_id = next(sid for sid, row in document["slots"].items() if row["kind"] == "validation")
    calls = []

    def parser_failure():
        private_key_value = "synthetic-secret-must-never-be-persisted"
        raise json.JSONDecodeError(private_key_value, private_key_value, 0)

    def generate_once():
        calls.append(True)
        parser_failure()

    row = executor._consume(document, slot_id, generate_once)
    diagnostic = row["result"]["failure_diagnostic"]
    assert calls == [True]
    assert row["attempts"] == 1
    assert diagnostic["error_type"] == "JSONDecodeError"
    assert any(frame["function"] == "parser_failure" for frame in diagnostic["frames"])
    assert all(set(frame) == {"file", "function", "line"} for frame in diagnostic["frames"])
    assert all(type(frame["line"]) is int and frame["line"] > 0 for frame in diagnostic["frames"])
    assert diagnostic["exception_message_recorded"] is False
    assert diagnostic["locals_recorded"] is False
    assert diagnostic["source_lines_recorded"] is False
    persisted = executor._journal().read_text(encoding="utf-8")
    assert "synthetic-secret-must-never-be-persisted" not in persisted
    assert "private_key_value" not in persisted


def test_pydantic_validation_error_does_not_repeat_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("JIT_MAS_SLOT_ATTEMPTS", "4")
    executor = JointExecutor(_bundle(tmp_path), tmp_path / "run")
    executor.check()
    document = executor._load_or_init()
    slot_id = next(sid for sid, row in document["slots"].items() if row["kind"] == "validation")
    calls = []

    def invalid_generated_record():
        calls.append(True)
        from jit_mas.schemas import PublicTask
        PublicTask.model_validate({"task_id": "generated", "question": 42})

    row = executor._consume(document, slot_id, invalid_generated_record)
    assert calls == [True]
    assert row["attempts"] == 1
    assert row["result"]["error_type"] == "ValidationError"
    assert row["result"]["failure_diagnostic"]["frames"]
