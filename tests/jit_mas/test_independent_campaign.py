"""Independent queue ordering, immutable state, and feedback barriers."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from jit_mas.agent_pool import seed_pool
from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.experience import ExperienceStore
from jit_mas.independent_campaign import IndependentCampaign, coordinator_lock
from jit_mas.independent_protocol import SOURCES, build_protocol
from jit_mas.schemas import ExperienceSnapshot, digest
from scripts.run_independent_experiment import IndependentEnvironment, validation_result


@pytest.fixture(scope="module")
def protocol():
    root = Path(__file__).resolve().parents[2]
    joint = json.loads((root / "paper/experiments/joint_task_splits_v5.json").read_text(encoding="utf-8"))
    return build_protocol(joint)


@pytest.fixture
def campaign(tmp_path, protocol):
    bounds = {task_id: [0, 1] for row in protocol["trajectories"] for task_id in row["validation_task_ids"]}
    return IndependentCampaign(tmp_path, protocol,
        {"initial_snapshot_hash": digest(ExperienceSnapshot()), "test": "synthetic-only"}, bounds)


def freeze_initial(campaign):
    for source in SOURCES:
        campaign.freeze_checkpoint(source, 0, 0, ExperienceSnapshot())


def result_for(claim, *, score=0.5, complete=True):
    result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"],
              "score": score, "complete": complete}
    if claim["kind"] == "evolution":
        result["after_snapshot_hash"] = claim["state_hash"]
    return result


def test_atomic_claim_prevents_duplicate_and_preserves_each_source_order(campaign):
    freeze_initial(campaign)
    with ThreadPoolExecutor(max_workers=8) as workers:
        claims = list(workers.map(lambda worker: campaign.claim(str(worker), phase="first-stage", kinds=("evolution",)), range(8)))
    accepted = [claim for claim in claims if claim]
    assert len(accepted) == 3
    assert len({claim["slot_id"] for claim in accepted}) == 3
    assert {claim["source"] for claim in accepted} == set(SOURCES)
    for claim in accepted:
        assert claim["task_id"] == campaign.trajectories[claim["source"], 0]["evolution_task_ids"][0]
    assert campaign.claim("extra", phase="first-stage", kinds=("evolution",)) is None
    campaign.finish(accepted[0], result_for(accepted[0]))
    next_claim = campaign.claim("next", phase="first-stage", kinds=("evolution",))
    assert next_claim["source"] == accepted[0]["source"]
    assert next_claim["task_id"] == campaign.trajectories[next_claim["source"], 0]["evolution_task_ids"][1]


def test_frozen_validation_is_ready_without_waiting_for_later_evolution(campaign):
    freeze_initial(campaign)
    source_claim = campaign.claim("evo", phase="first-stage", kinds=("evolution",))
    val_claim = campaign.claim("val", phase="first-stage", kinds=("validation",))
    assert val_claim["position"] == 0
    assert val_claim["state_hash"] == digest(ExperienceSnapshot())
    campaign.finish(source_claim, result_for(source_claim), status="failed")
    campaign.finish(val_claim, result_for(val_claim))
    with pytest.raises(ValueError, match="preceding EVO"):
        campaign.freeze_checkpoint(source_claim["source"], 0, 5, ExperienceSnapshot())


def test_finish_idempotent_and_rejects_replacement_or_unknown_claim(campaign):
    freeze_initial(campaign)
    claim = campaign.claim("val", phase="first-stage", kinds=("validation",))
    result = result_for(claim)
    campaign.finish(claim, result)
    campaign.finish(claim, result)
    with pytest.raises(CheckpointIntegrityError, match="replaced"):
        campaign.finish(claim, result_for(claim, score=0.9))
    other = {**claim, "claim_token": "forged"}
    with pytest.raises(ValueError, match="exclusive"):
        campaign.finish(other, result)


def test_complete_agent_pool_is_bound_and_other_sources_cannot_enter(campaign):
    snapshot = ExperienceSnapshot(agent_pool=seed_pool())
    campaign.freeze_checkpoint("researchrubrics", 0, 0, snapshot)
    loaded = campaign.checkpoint("researchrubrics", 0, 0)
    assert loaded["snapshot"].agent_pool == snapshot.agent_pool
    changed = snapshot.model_copy(deep=True)
    changed.agent_pool.profiles[0].prompt += " changed"
    with pytest.raises(CheckpointIntegrityError, match="replace"):
        campaign.freeze_checkpoint("researchrubrics", 0, 0, changed)
    changed = ExperienceSnapshot(agent_pool=seed_pool())
    changed.agent_pool.profiles[0].source_task_ids = [campaign.trajectories["deepsearchqa", 0]["evolution_task_ids"][0]]
    with pytest.raises(CheckpointIntegrityError, match="outside"):
        campaign.freeze_checkpoint("writingbench", 0, 0, changed)


def test_selection_requires_all_candidates_and_source_only_complete_scores(campaign):
    freeze_initial(campaign)
    with pytest.raises(ValueError, match="50 VAL"):
        campaign.select_trajectory("researchrubrics", 0)
    while claim := campaign.claim("evo", kinds=("evolution",)):
        campaign.finish(claim, result_for(claim))
        source, run_id = claim["source"], claim["run_id"]
        rows = campaign.records(kind="evolution", source=source, run_id=run_id)
        position = sum(row["status"] == "complete" for row in rows)
        if position in (5, 10, 15, 20):
            campaign.freeze_checkpoint(source, run_id, position, ExperienceSnapshot())
    while claim := campaign.claim("val", kinds=("validation",)):
        campaign.finish(claim, result_for(claim, score=0.2 + claim["position"] / 50))
    selected = campaign.select_trajectory("researchrubrics", 0)
    assert selected["selected"]["position"] == 20
    assert selected["selected"]["complete_evaluations"] == 10
    assert selected["selected"]["selection_utility"] == pytest.approx(0.6)
    assert campaign.select_trajectory("researchrubrics", 0) == selected


def test_test_feedback_waits_for_entire_registered_inventory(campaign):
    with pytest.raises(ValueError, match="global"):
        campaign.require_test_seal()
    with pytest.raises(ValueError, match="2,751"):
        campaign.seal_test_inventory()
    with sqlite3.connect(campaign.path) as database:
        for slot_id, body in database.execute("SELECT slot_id,body FROM slots").fetchall():
            slot = json.loads(body)
            if slot["kind"] != "test":
                continue
            result = {"task_id": slot["task_id"], "evaluation": None, "error_type": "SyntheticFailure"}
            database.execute("UPDATE slots SET status='failed',result=?,result_hash=? WHERE slot_id=?",
                             (json.dumps(result), digest(result), slot_id))
    seal = campaign.seal_test_inventory()
    assert len(seal["submission_hashes"]) == 2751
    assert campaign.require_test_seal() == seal
    with sqlite3.connect(campaign.path) as database:
        database.execute("UPDATE slots SET result='{}' WHERE slot_id=?", (next(iter(seal["submission_hashes"])),))
    with pytest.raises(CheckpointIntegrityError, match="outcome changed"):
        campaign.require_test_seal()


def test_submit_test_records_only_unscored_artifact(campaign, tmp_path):
    claim = campaign.claim("test-worker", kinds=("test",))
    assert claim is not None and claim["source"] == "static"
    slot_id = claim["slot_id"]
    submission = {"answer": "synthetic", "answer_hash": digest("synthetic"), "submitted_at": "now"}
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission), encoding="utf-8")
    campaign.submit_test(claim, path, answer_hash=submission["answer_hash"], budget={"model_calls": 1})
    record = next(row for row in campaign.records(kind="test") if row["slot_id"] == slot_id)
    assert record["status"] == "submitted"
    assert record["result"]["submission_hash"] == digest(submission)
    with pytest.raises(CheckpointIntegrityError):
        campaign.submit_test(claim, path, answer_hash=digest("other"))


def test_interrupted_claim_never_becomes_a_new_pending_attempt(campaign):
    freeze_initial(campaign)
    claim = campaign.claim("dead-worker", phase="first-stage", kinds=("validation",))
    assert campaign.interrupted_claims()[0]["claim_token"] == claim["claim_token"]
    failure = {**result_for(claim, score=None, complete=False), "budget": {"model_calls": 1}}
    campaign.finish(claim, failure, status="failed")
    claims = []
    while next_claim := campaign.claim("new-worker", phase="first-stage", kinds=("validation",)):
        claims.append(next_claim)
        campaign.finish(next_claim, result_for(next_claim))
    assert claim["slot_id"] not in {row["slot_id"] for row in claims}


def test_campaign_identity_changes_are_rejected(campaign):
    changed = copy.deepcopy(campaign.identity)
    changed["test"] = "changed"
    with pytest.raises(CheckpointIntegrityError, match="changed frozen"):
        IndependentCampaign(campaign.directory, campaign.protocol, changed, campaign.bounds)


def test_process_lock_blocks_duplicate_coordinator(tmp_path):
    with coordinator_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another campaign"):
            with coordinator_lock(tmp_path):
                pass
    with coordinator_lock(tmp_path):
        pass


def test_writingbench_selector_uses_native_mean_once():
    result = validation_result("writingbench", {"task_id": "wb", "evaluation": {
        "complete": True, "score": 0.5, "raw": {"native_mean": 5.5}}})
    assert result["score"] == 5.5


def test_synthetic_first_stage_runs_exactly_75_slots_asynchronously(tmp_path, protocol):
    bounds = {task_id: [0, 1] for row in protocol["trajectories"] for task_id in row["validation_task_ids"]}
    environment = object.__new__(IndependentEnvironment)
    environment.output = tmp_path
    environment.protocol, environment.workers = protocol, 2
    environment.identity = {"initial_snapshot_hash": digest(ExperienceSnapshot()), "synthetic_only": True}
    environment.bounds, environment.campaign = bounds, None
    environment.assert_frozen = lambda: None
    environment.recover = lambda: None
    initialized = threading.Event()
    overlapped = threading.Event()

    def initialize():
        environment.campaign = IndependentCampaign(tmp_path, protocol, environment.identity, bounds)
        freeze_initial(environment.campaign)

    def execute(claim):
        if claim["kind"] == "evolution":
            initialized.set()
            time.sleep(0.01)
        elif initialized.is_set():
            overlapped.set()
        environment.campaign.finish(claim, result_for(claim))
        if claim["kind"] == "evolution":
            rows = environment.campaign.records(kind="evolution", source=claim["source"], run_id=0)
            if sum(row["status"] == "complete" for row in rows) == 5:
                environment.campaign.freeze_checkpoint(claim["source"], 0, 5, ExperienceSnapshot())
        return {"slot_id": claim["slot_id"]}

    environment.initialize, environment.execute = initialize, execute
    report = environment.run(phase="first-stage")
    assert report["registered_slots"] == report["terminal_slots"] == 75
    assert report["evolution_validation_complete"]
    assert not report["test_feedback_released"]
    assert overlapped.is_set()
    assert environment.campaign.status()["statuses"]["pending"] == 3381 - 75


def test_recovery_consumes_interrupted_slot_without_calling_a_model(campaign):
    freeze_initial(campaign)
    claim = campaign.claim("dead", phase="first-stage", kinds=("evolution",))
    environment = object.__new__(IndependentEnvironment)
    environment.output, environment.campaign = campaign.directory, campaign
    environment.assert_frozen = lambda: None
    environment.freeze_ready = lambda: None
    state_path = environment.trajectory_directory(claim["source"], 0) / "experience.sqlite"
    store = ExperienceStore(state_path)
    store.close()
    budget_path = environment.slot_directory(claim) / "attempt" / "budget.json"
    budget_path.parent.mkdir(parents=True)
    budget_path.write_text(json.dumps({"model_calls": 2, "tokens": 400}), encoding="utf-8")
    environment.recover()
    row = next(row for row in campaign.records() if row["slot_id"] == claim["slot_id"])
    assert row["status"] == "failed"
    assert row["result"]["recovered"]
    assert row["result"]["budget"]["model_calls"] == 2
    assert row["result"]["after_snapshot_hash"] == digest(ExperienceSnapshot())
