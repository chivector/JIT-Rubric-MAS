"""Offline checkpoint coordination, crash recovery, and real pipeline integration."""

import json
import sqlite3

import pytest

from jit_mas.checkpoints import (
    CheckpointIntegrityError, CheckpointRunner, UnresolvedSourceAttempt,
    select_checkpoint, snapshot_store,
)
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import ChangeProposal, Experience, ExperienceSnapshot, digest


def proposal(store, task_id):
    return ChangeProposal(
        proposal_id=f"proposal-{task_id}", source_task_id=task_id,
        base_version=store.snapshot().version,
        experience=Experience(
            experience_id=f"experience-{task_id}", bank="organization",
            instruction="Preserve source references in the single-pass shared ledger.",
            applicability="Tasks requiring evidence synthesis", source_task_ids=[task_id],
            evidence=[f"{task_id}:evidence"], created_at="2026-10-01T00:00:00+00:00"),
        diff="Preserve source references", rationale="Source reference was omitted",
        evidence=[f"{task_id}:evidence"], expected_benefit="Grounded synthesis")


def outcome(score=0.5, *, complete=True, calls=1, tokens=10):
    return {"evaluation": {"score": score, "complete": complete},
            "budget": {"model_calls": calls, "tokens": tokens}, "proposals": []}


@pytest.fixture
def store(tmp_path):
    state = ExperienceStore(tmp_path / "evolution.sqlite")
    yield state
    state.close()


def runner(tmp_path, store, *, evolve=None, evaluate=None, **kwargs):
    return CheckpointRunner(
        store=store, evolution_ids=kwargs.pop("evolution_ids", ["source-1", "source-2"]),
        validation_ids=kwargs.pop("validation_ids", ["val-1", "val-2"]),
        output_dir=tmp_path / "checkpoint-run", identity=kwargs.pop("identity", {"code": "frozen-code"}),
        lower_bounds=kwargs.pop("lower_bounds", {"val-1": -1.0, "val-2": -0.5}),
        evolve=evolve or (lambda task_id: outcome()),
        evaluate=evaluate or (lambda *args: outcome()),
        batch_size=kwargs.pop("batch_size", 1), **kwargs)


def test_latest_state_progression_full_validation_and_no_rollback(tmp_path, store):
    seen = []
    validation = []

    def evolve(task_id):
        seen.append(store.snapshot().version)
        store.commit(proposal(store, task_id))
        return outcome()

    def evaluate(task_id, snapshot, repeat, frozen):
        assert frozen.read_only
        assert frozen.path != store.path
        assert digest(snapshot) == digest(frozen.snapshot())
        validation.append((snapshot.version, task_id, repeat))
        return outcome(score=1 - snapshot.version / 3)

    coordinator = runner(tmp_path, store, evolve=evolve, evaluate=evaluate)
    result = coordinator.run()
    assert seen == [0, 1]
    assert store.snapshot().version == 2
    assert [row["position"] for row in result["checkpoints"]] == [0, 1, 2]
    assert validation == [(version, task, repeat)
                          for version in range(3) for task in ["val-1", "val-2"] for repeat in range(2)]
    assert result["selected"]["position"] == 0
    assert coordinator.selected_snapshot().version == 0
    assert result["accounting"]["model_calls_recorded"] == 14
    assert result["accounting"]["tokens_recorded"] == 140
    assert result["accounting"]["attempts_with_unknown_budget"] == 0


def test_identical_states_reuse_all_scores_without_extra_cost(tmp_path, store):
    calls = []
    coordinator = runner(tmp_path, store, evaluate=lambda *args: calls.append(args[:3]) or outcome())
    result = coordinator.run()
    assert len(calls) == 4
    assert [row["reused_from"] for row in result["checkpoints"]] == [None, 0, 0]
    assert result["accounting"]["actual_validation_slots"] == 4
    assert result["accounting"]["nominal_validation_slots"] == 12
    assert result["accounting"]["model_calls_recorded"] == 6
    assert result["selected"]["position"] == 0


def test_resume_complete_run_does_not_reissue_callbacks(tmp_path, store):
    original = runner(tmp_path, store).run()

    def forbidden(*args):
        pytest.fail("A resumed completed slot must not regenerate its artifact")

    resumed = runner(tmp_path, store, evolve=forbidden, evaluate=forbidden).run()
    assert resumed == original


def test_source_failure_and_no_proposal_consume_positions(tmp_path, store):
    seen = []

    def evolve(task_id):
        seen.append(task_id)
        if task_id == "source-1":
            error = RuntimeError("synthetic executor failure")
            error.jit_mas_run_failure = {"budget": {"model_calls": 3, "tokens": 30}}
            raise error
        return outcome()

    coordinator = runner(tmp_path, store, evolve=evolve)
    result = coordinator.run()
    assert seen == ["source-1", "source-2"]
    assert [row["status"] for row in coordinator.journal["sources"]] == ["failed", "complete"]
    assert [row["position"] for row in result["checkpoints"]] == [0, 1, 2]
    assert store.snapshot().version == 0
    assert result["accounting"]["model_calls_recorded"] == 8


def test_missing_utility_keeps_fixed_denominator_and_nullable_official_score(tmp_path, store):
    def evaluate(task_id, snapshot, repeat, frozen):
        return outcome(score=99, complete=False) if (task_id, repeat) == ("val-1", 0) else outcome(score=1)

    coordinator = runner(tmp_path, store, evaluate=evaluate, minimum_completion=0.75)
    result = coordinator.run()
    selected = result["selected"]
    assert selected["selection_utility"] == 0.5
    assert selected["complete_evaluations"] == 3
    assert selected["utility_is_conservative_surrogate"]
    missing = next(iter(coordinator.journal["validation_cache"].values()))["slots"][0]
    assert missing["official_score"] is None
    assert missing["selection_utility"] == -1
    assert missing["is_imputed_for_selection"]


def test_ninety_percent_is_36_of_40_and_no_eligible_is_inconclusive(tmp_path, store):
    ids = [f"val-{index}" for index in range(20)]
    coordinator = runner(tmp_path, store, validation_ids=ids, lower_bounds={key: 0 for key in ids},
                         evaluate=lambda task, snapshot, repeat, frozen: outcome(
                             complete=not (task in ids[:2] or (task == ids[2] and repeat == 0))))
    assert coordinator.minimum_complete == 36
    result = coordinator.run()
    assert result["status"] == "inconclusive"
    assert result["selected"] is None
    assert result["checkpoints"][0]["complete_evaluations"] == 35
    with pytest.raises(ValueError, match="eligible"):
        coordinator.selected_snapshot()


def test_fixed_global_maximum_tie_set_then_completion_position_hash():
    rows = [{"eligible": True, "selection_utility": value, "complete_evaluations": 40,
             "position": index, "state_hash": str(index)}
            for index, value in enumerate([0.0, 0.75e-12, 1.5e-12])]
    assert select_checkpoint(rows)["position"] == 1
    rows[2]["complete_evaluations"] = 41
    assert select_checkpoint(rows)["position"] == 2
    rows[2]["eligible"] = False
    assert select_checkpoint(rows)["position"] == 0


def test_non_multiple_batch_still_has_final_checkpoint(tmp_path, store):
    result = runner(tmp_path, store, evolution_ids=[f"source-{index}" for index in range(7)], batch_size=5).run()
    assert [row["position"] for row in result["checkpoints"]] == [0, 5, 7]


def test_frozen_store_is_independent_and_cannot_commit(tmp_path, store):
    initial = store.snapshot()
    frozen = snapshot_store(initial, tmp_path / "frozen.sqlite")
    try:
        with pytest.raises(PermissionError):
            frozen.commit(proposal(store, "source-1"))
        with pytest.raises(sqlite3.OperationalError):
            frozen.db.execute("UPDATE state SET value='99'")
        store.commit(proposal(store, "source-1"))
        assert frozen.snapshot() == initial
    finally:
        frozen.close()


def test_validation_mutation_is_fatal_not_a_scored_failure(tmp_path, store):
    def mutate(task_id, snapshot, repeat, frozen):
        store.commit(proposal(store, "illegal-validation-write"))
        return outcome()

    with pytest.raises(CheckpointIntegrityError, match="mutated"):
        runner(tmp_path, store, evaluate=mutate).run()


def test_resume_rejects_identity_and_external_state_changes(tmp_path, store):
    runner(tmp_path, store).run()
    with pytest.raises(CheckpointIntegrityError, match="identity"):
        runner(tmp_path, store, identity={"code": "changed"})
    store.commit(proposal(store, "external-change"))
    with pytest.raises(CheckpointIntegrityError, match="outside"):
        runner(tmp_path, store)


def test_interrupted_source_blocks_regeneration_then_can_record_failure(tmp_path, store):
    calls = []

    def interrupt(task_id):
        calls.append(task_id)
        raise KeyboardInterrupt()

    first = runner(tmp_path, store, evolve=interrupt)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    resumed = runner(tmp_path, store, evolve=lambda task_id: calls.append(task_id) or outcome())
    with pytest.raises(UnresolvedSourceAttempt):
        resumed.run()
    assert calls == ["source-1"]
    resumed.record_interrupted_source_failure("Inspected stopped process; no durable response", budget={"model_calls": 1, "tokens": 10})
    result = resumed.run()
    assert calls == ["source-1", "source-2"]
    assert result["status"] == "complete"


def test_interrupted_source_recovers_durable_pipeline_outcome_without_resampling(tmp_path, store):
    def interrupt(task_id):
        base = store.snapshot()
        store.save_task_run("evolve", task_id, "identity", base, "started")
        result = outcome()
        store.commit(proposal(store, task_id))
        result["next_experience_version"] = store.snapshot().version
        store.save_task_run("evolve", task_id, "identity", base, "complete", result)
        raise KeyboardInterrupt()

    first = runner(tmp_path, store, evolve=interrupt, evolution_ids=["source-1"])
    with pytest.raises(KeyboardInterrupt):
        first.run()
    recovered = runner(tmp_path, store, evolve=lambda *args: pytest.fail("Unexpected new source call"), evolution_ids=["source-1"])
    result = recovered.run()
    assert store.snapshot().version == 1
    assert result["status"] == "complete"
    assert recovered.journal["sources"][0]["recovered"]


def test_interrupted_validation_is_not_resampled(tmp_path, store):
    called = []

    def interrupt(task_id, snapshot, repeat, frozen):
        called.append((task_id, repeat))
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runner(tmp_path, store, evaluate=interrupt).run()
    resumed = runner(tmp_path, store, evaluate=lambda task, snapshot, repeat, frozen:
                     called.append((task, repeat)) or outcome(), minimum_completion=0.9)
    result = resumed.run()
    assert called.count(("val-1", 0)) == 1
    assert len(called) == 4
    assert result["status"] == "inconclusive"
    assert result["accounting"]["attempts_with_unknown_budget"] == 1


@pytest.mark.parametrize("kwargs", [
    {"evolution_ids": []}, {"validation_ids": []}, {"evolution_ids": ["val-1"]},
    {"batch_size": 0}, {"repeats": 0}, {"minimum_completion": 1.1},
    {"lower_bounds": {"val-1": float("nan"), "val-2": 0}},
])
def test_invalid_schedule_rejected_before_requests(tmp_path, store, kwargs):
    with pytest.raises(ValueError):
        runner(tmp_path, store, **kwargs)


def test_offline_real_jit_pipeline_evolves_validates_and_reuses(tmp_path, monkeypatch):
    from jit_mas.config import MASConfig
    from jit_mas.offline import FixtureModels
    from scripts.models.openai_server import OpenAIServerModel
    from scripts.run_jit_mas import make_pipeline

    monkeypatch.setattr(OpenAIServerModel, "__call__", lambda *args, **kwargs:
                        pytest.fail("Checkpoint integration must remain offline"))
    store = ExperienceStore(tmp_path / "state.sqlite")
    provider = FixtureModels()
    config = MASConfig(backend="scripted")
    pipeline = make_pipeline(config, store, tmp_path / "evolution", fixture_models=provider)
    try:
        def evaluate(task_id, snapshot, repeat, frozen):
            validation = make_pipeline(config, frozen, tmp_path / "validation", fixture_models=provider)
            return validation.run_task(task_id, snapshot, mode="validate", repeat=repeat, attribution=False)

        coordinator = CheckpointRunner(
            store=store, evolution_ids=pipeline.manifest.evolution,
            validation_ids=pipeline.manifest.validation, output_dir=tmp_path / "checkpoints",
            identity={"software_test_only": True, "code": pipeline.code_hash},
            lower_bounds={task: 0 for task in pipeline.manifest.validation},
            evolve=lambda task: pipeline.run("evolve", [task]), evaluate=evaluate, batch_size=5)
        report = coordinator.run()
        assert report["status"] == "complete"
        assert store.snapshot().version == 1
        assert [item["position"] for item in report["checkpoints"]] == [0, 1]
        assert report["accounting"]["actual_validation_slots"] == 8
        assert report["accounting"]["attempts_with_unknown_budget"] == 0
        validation_slots = [row for cached in coordinator.journal["validation_cache"].values() for row in cached["slots"]]
        assert all(row["complete_evaluation"] for row in validation_slots)
        assert all(row["outcome"]["proposals"] == [] for row in validation_slots)
        assert all(row["outcome"]["mode"] == "validate" for row in validation_slots)
        assert not any("update" in row["outcome"]["budget"]["by_stage"] for row in validation_slots)
        calls_before = len(provider.calls)
        assert coordinator.run() == report
        assert len(provider.calls) == calls_before
        assert json.loads((tmp_path / "checkpoints" / "checkpoint_journal.json").read_text())["status"] == "complete"
    finally:
        store.close()
