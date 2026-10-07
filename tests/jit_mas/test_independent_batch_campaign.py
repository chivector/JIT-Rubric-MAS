import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.independent_batch_campaign import IndependentBatchCampaign, slot_registry
from jit_mas.independent_batch_protocol import SOURCES, build_protocol
from jit_mas.schemas import AgentPoolOperation, ExperienceSnapshot, digest


@pytest.fixture(scope="module")
def protocol():
    root = Path(__file__).resolve().parents[2]
    return build_protocol(json.loads((root / "paper/experiments/joint_task_splits_v6.json").read_text(encoding="utf-8")))


@pytest.fixture
def campaign(tmp_path, protocol):
    bounds = {task_id: [0, 1] for row in protocol["trajectories"] for task_id in row["validation_task_ids"]}
    instance = IndependentBatchCampaign(tmp_path, protocol,
        {"initial_snapshot_hash": digest(ExperienceSnapshot()), "software_test_only": True}, bounds)
    instance.freeze_checkpoint("researchrubrics", 0, 0, ExperienceSnapshot())
    return instance


def finish(instance, claim, score=0.5):
    result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"], "complete": True, "score": score}
    if claim["kind"] == "evolution":
        result["after_snapshot_hash"] = claim["state_hash"]
    instance.finish(claim, result)


def first_batch(instance):
    claims = [instance.claim("evo", kinds=("evolution",)) for _ in range(5)]
    for claim in claims:
        finish(instance, claim)
    return instance.claim("update", kinds=("batch_evolution",))


def candidates(instance, claim):
    receipts = []
    for index in range(3):
        state = ExperienceSnapshot(version=index + 1)
        receipts.append({"candidate_index": index, "status": "complete", "snapshot": state.model_dump(mode="json"),
                         "state_hash": digest(state), "budget": {"tokens": index}})
    instance.freeze_candidates(claim, receipts)
    finish(instance, claim)
    return receipts


def test_registry_has_task_and_update_inventories(protocol):
    registry = slot_registry(protocol)
    assert registry["counts"] == {"evolution": 360, "batch_evolution": 72, "validation": 2160, "test": 2751, "total": 5343}
    assert len({row["slot_id"] for row in registry["artifacts"]}) == 5343


def test_five_tasks_claim_in_parallel_and_update_waits_for_all(campaign):
    with ThreadPoolExecutor(max_workers=8) as workers:
        claims = list(workers.map(lambda index: campaign.claim(str(index), kinds=("evolution",)), range(8)))
    accepted = [claim for claim in claims if claim]
    assert len(accepted) == 5
    assert len({claim["slot_id"] for claim in accepted}) == 5
    assert {claim["state_hash"] for claim in accepted} == {digest(ExperienceSnapshot())}
    assert campaign.claim("update", kinds=("batch_evolution",)) is None
    for claim in accepted[:4]:
        finish(campaign, claim)
    assert campaign.claim("update", kinds=("batch_evolution",)) is None
    finish(campaign, accepted[4])
    assert campaign.claim("update", kinds=("batch_evolution",))["batch_index"] == 0
    assert campaign.claim("next", kinds=("evolution",)) is None


def test_next_batch_waits_for_all_thirty_val_and_uses_winner(campaign):
    update = first_batch(campaign)
    receipts = candidates(campaign, update)
    validation = [campaign.claim("val", kinds=("validation",)) for _ in range(30)]
    assert {claim["candidate_index"] for claim in validation} == {0, 1, 2}
    for claim in validation[:-1]:
        finish(campaign, claim, score=claim["candidate_index"] / 2)
    campaign.advance()
    assert campaign.checkpoint("researchrubrics", 0, 5) is None
    assert campaign.claim("next", kinds=("evolution",)) is None
    finish(campaign, validation[-1], score=validation[-1]["candidate_index"] / 2)
    campaign.advance()
    next_claim = campaign.claim("next", kinds=("evolution",))
    assert next_claim["batch_index"] == 1
    assert next_claim["state_hash"] == receipts[2]["state_hash"]
    assert campaign.selection("researchrubrics", 0) is None


def test_no_eligible_candidate_terminates_downstream_without_parent_fallback(campaign):
    update = first_batch(campaign)
    campaign.freeze_candidates(update, [{"candidate_index": index, "snapshot": None, "state_hash": None,
                                         "status": "failed", "budget": None} for index in range(3)])
    finish(campaign, update)
    campaign.advance()
    assert campaign.selection("researchrubrics", 0)["selected"] is None
    assert campaign.checkpoint("researchrubrics", 0, 5) is None
    remaining = campaign.records(source="researchrubrics", run_id=0)
    assert all(row["status"] in {"complete", "missing"} for row in remaining)


def test_evo_cannot_write_state_and_candidate_cannot_be_replaced(campaign):
    claim = campaign.claim("evo", kinds=("evolution",))
    result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"], "after_snapshot_hash": "changed"}
    with pytest.raises(CheckpointIntegrityError, match="without persistent"):
        campaign.finish(claim, result)
    finish(campaign, claim)
    for _ in range(4):
        finish(campaign, campaign.claim("evo", kinds=("evolution",)))
    update = campaign.claim("update", kinds=("batch_evolution",))
    receipts = candidates(campaign, update)
    receipts[0]["budget"]["tokens"] = 99
    with pytest.raises(CheckpointIntegrityError, match="replace or resample"):
        campaign.freeze_candidates(update, receipts)


def test_static_test_is_ready_and_selected_test_waits(campaign):
    claim = campaign.claim("static", kinds=("test",), prefer="test")
    assert claim["source"] == "static"
    assert claim["state_hash"] == digest(ExperienceSnapshot())
    assert not campaign.selection("researchrubrics", 0)


def test_candidate_validation_waits_for_batch_slot_commit(campaign):
    update = first_batch(campaign)
    receipts = [{"candidate_index": index, "snapshot": ExperienceSnapshot().model_dump(mode="json"),
                 "state_hash": digest(ExperienceSnapshot()), "status": "complete", "budget": {"tokens": 0}}
                for index in range(3)]
    campaign.freeze_candidates(update, receipts)
    assert campaign.claim("val", kinds=("validation",)) is None
    finish(campaign, update)
    assert campaign.claim("val", kinds=("validation",)) is not None


def test_candidate_rejects_nonprimary_operation_source_outside_evo(campaign):
    update = first_batch(campaign)
    allowed = campaign.trajectories["researchrubrics", 0]["evolution_task_ids"][0]
    state = ExperienceSnapshot()
    state.agent_pool.structural_history = [AgentPoolOperation(
        operation_id="operation", kind="reorganize", source_task_id=allowed,
        source_task_ids=[allowed, "outside:evo"], base_pool_version=0,
        evidence=["feedback:observed"], rationale="Observed change", expected_benefit="Coverage",
        token_cost_tradeoff="Recorded cost")]
    receipts = [{"candidate_index": index, "snapshot": state.model_dump(mode="json"),
                 "state_hash": digest(state), "status": "complete", "budget": {"tokens": 0}}
                for index in range(3)]
    with pytest.raises(CheckpointIntegrityError, match="source provenance"):
        campaign.freeze_candidates(update, receipts)


@pytest.mark.parametrize("workers", [32, 64])
def test_parallel_finishes_and_concurrent_reads_do_not_block_or_duplicate(campaign, workers):
    with campaign._database() as database:
        assert database.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert database.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
    with ThreadPoolExecutor(max_workers=workers) as pool:
        claims = list(pool.map(lambda index: campaign.claim(str(index), kinds=("test",)), range(workers)))
    assert len({claim["slot_id"] for claim in claims}) == workers
    barrier = threading.Barrier(workers)

    def commit(claim):
        barrier.wait(timeout=10)
        result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"], "evaluation": None,
                  "submission_path": "synthetic-submission.json"}
        campaign.finish(claim, result, status="submitted")
        return campaign.records(kind="validation", source="researchrubrics", run_id=0, batch_index=0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        reads = list(pool.map(commit, claims))
    assert all(len(rows) == 30 for rows in reads)
    assert sum(row["status"] == "submitted" for row in campaign.records(kind="test")) == workers
