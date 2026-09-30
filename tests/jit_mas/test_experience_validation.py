"""Paired rebuilds and persistence guards; no model calls or benchmark claims."""

import json
import sqlite3

import pytest

from jit_mas.budget import BudgetExceeded
from jit_mas.experience import ExperienceStore, candidate_snapshot, retrieve
from jit_mas.schemas import (
    ChangeProposal, EvaluationFeedback, Experience, ExperienceSnapshot,
    PublicTask, RubricFeedback, SplitManifest, digest,
)
from jit_mas.validation import PairedValidator, ValidationConfig


def proposal(base_version=0, proposal_id="proposal-1", experience_id="experience-1", replaces_id=None):
    return ChangeProposal(
        proposal_id=proposal_id, source_task_id="evolution-1", base_version=base_version,
        experience=Experience(
            experience_id=experience_id, bank="organization",
            instruction="Require the synthesizer to acknowledge each evidence handoff.",
            applicability="Tasks with independent evidence collection and synthesis",
            source_task_ids=["evolution-1"], evidence=["run-1:handoff-2"],
            created_at="2026-09-01T00:00:00+00:00",
        ),
        replaces_id=replaces_id,
        diff="Add evidence acknowledgement to collaboration policy",
        rationale="A verified handoff was omitted from the final answer",
        evidence=["run-1:handoff-2"], expected_benefit="Preserve supplied evidence",
        validation_plan="Rebuild baseline and candidate on disjoint validation tasks",
    )


def manifest():
    return SplitManifest(evolution=["evolution-1"], validation=["validation-a", "validation-b"],
                         test=["test-1"], stream=["stream-1", "stream-2"])


class Rebuild:
    """Recompute outcomes from the supplied candidate state on every call."""
    def __init__(self, *, old=(0, 0), new=(1, 0), weights=(5, -3), complete=True,
                 change_fingerprint=False, change_evaluator=False, fail=False):
        self.old, self.new, self.weights = old, new, weights
        self.complete = complete
        self.change_fingerprint, self.change_evaluator, self.fail = change_fingerprint, change_evaluator, fail
        self.calls = []

    def __call__(self, task_id, snapshot, repeat, label):
        self.calls.append((task_id, digest(snapshot), repeat, label))
        if self.fail:
            raise RuntimeError("Rebuild execution failed")
        improved = "proposal-1" in snapshot.accepted_proposals
        scores = self.new if improved else self.old
        rubrics = [
            RubricFeedback(rubric_id=f"{task_id}:r{index}", criterion=f"Criterion {index}",
                           weight=weight, score=score, verdict="Satisfied" if score else "Not Satisfied")
            for index, (weight, score) in enumerate(zip(self.weights, scores))
        ]
        denominator = sum(weight for weight in self.weights if weight > 0)
        score = sum(w * s for w, s in zip(self.weights, scores)) / denominator if denominator else 0
        feedback = EvaluationFeedback(
            task_id=task_id, evaluator_version="frozen-judge-v1" + ("-changed" if improved and self.change_evaluator else ""),
            rubrics=rubrics, score=score, complete=self.complete,
        )
        return {
            "task_id": task_id,
            "comparison_fingerprint": f"same-task-model-tools-budget:{task_id}:{repeat}" +
            (":changed" if improved and self.change_fingerprint else ""),
            "evaluation": feedback.model_dump(mode="json"),
        }


def validate(change=None, base=None, rebuild=None, config=None):
    change = change or proposal()
    base = base or ExperienceSnapshot()
    rebuild = rebuild or Rebuild()
    return PairedValidator(rebuild, manifest(), config).validate(change, base)


def test_paired_validation_rebuilds_both_states_for_each_task_and_repeat():
    rebuild = Rebuild()
    base, change = ExperienceSnapshot(), proposal()
    result = validate(change, base, rebuild, ValidationConfig(repeats=2))
    assert result.status == "accepted"
    assert len(result.pairs) == 4
    assert len(rebuild.calls) == 8
    for index in range(0, len(rebuild.calls), 2):
        old, new = rebuild.calls[index:index + 2]
        assert old[0] == new[0]
        assert old[2] == new[2]
        assert old[3] == "baseline" and new[3] == "candidate"
        assert old[1] == digest(base)
        assert old[1] != new[1]
    assert not base.experiences


def test_independent_task_count_not_rubric_or_repeat_count():
    rebuild = Rebuild(weights=(1,) * 20, old=(0,) * 20, new=(1,) * 20)
    validator = PairedValidator(rebuild, manifest(), ValidationConfig(min_tasks=2, repeats=4))
    result = validator.validate(proposal(), ExperienceSnapshot(), ["validation-a"])
    assert result.status == "pending"
    assert result.pairs == []
    assert rebuild.calls == []


@pytest.mark.parametrize("task_ids", [["test-1"], ["evolution-1"], ["validation-a", "validation-a"]])
def test_validation_disallows_other_splits_and_duplicate_tasks(task_ids):
    with pytest.raises(ValueError):
        PairedValidator(Rebuild(), manifest()).validate(proposal(), ExperienceSnapshot(), task_ids)


@pytest.mark.parametrize("rebuild", [
    Rebuild(complete=False), Rebuild(change_fingerprint=True),
    Rebuild(change_evaluator=True), Rebuild(fail=True),
])
def test_incomplete_or_incomparable_rebuilds_stay_pending(rebuild):
    assert validate(rebuild=rebuild).status == "pending"


def test_candidate_failure_preserves_completed_baseline_and_failed_run_budget():
    rebuild = Rebuild()
    calls = []

    def fail_candidate(task_id, snapshot, repeat, label):
        calls.append((task_id, label))
        if label == "candidate":
            error = RuntimeError("Candidate execution failed")
            error.jit_mas_run_failure = {
                "task_id": task_id, "run_dir": "failed-candidate",
                "budget": {"model_calls": 2, "tokens": 17, "tool_calls": 0}}
            raise error
        result = rebuild(task_id, snapshot, repeat, label)
        result["budget"] = {"model_calls": 3, "tokens": 31, "tool_calls": 0}
        return result

    result = validate(rebuild=fail_candidate)
    assert result.status == "pending"
    assert len(calls) == 4
    for pair in result.pairs:
        assert pair["baseline"]["evaluation"]["complete"]
        assert pair["baseline"]["budget"]["model_calls"] == 3
        assert "candidate" not in pair
        assert pair["failed_side"] == "candidate"
        assert pair["failed_run"]["budget"]["tokens"] == 17


@pytest.mark.parametrize("failed_side,expected_calls", [("baseline", 1), ("candidate", 2)])
def test_exhausted_budget_stops_dispatching_remaining_validation_tasks(failed_side, expected_calls):
    rebuild = Rebuild()
    calls = []

    def bounded(task_id, snapshot, repeat, label):
        calls.append((task_id, label))
        if label == failed_side:
            raise BudgetExceeded("Session budget exhausted")
        return rebuild(task_id, snapshot, repeat, label)

    result = validate(rebuild=bounded)
    assert result.status == "pending"
    assert len(calls) == expected_calls
    assert len(result.pairs) == 1
    assert result.pairs[0]["failed_side"] == failed_side
    assert "BudgetExceeded" in result.pairs[0]["error"]


def test_quality_loss_cannot_be_offset_by_other_tasks():
    normal = Rebuild()

    def mixed(task_id, snapshot, repeat, label):
        trial = Rebuild(old=(1, 0), new=(0, 0)) if task_id == "validation-a" else normal
        return trial(task_id, snapshot, repeat, label)

    assert validate(rebuild=mixed).status == "rejected"


def test_negative_criterion_penalty_blocks_otherwise_improved_total():
    result = validate(rebuild=Rebuild(old=(0, 0), new=(1, 1), weights=(5, -3)))
    assert result.pairs[0]["candidate"]["evaluation"]["score"] > result.pairs[0]["baseline"]["evaluation"]["score"]
    assert result.status == "rejected"


def test_removing_negative_behavior_counts_as_improvement():
    assert validate(rebuild=Rebuild(old=(1, 1), new=(1, 0))).status == "accepted"


def test_cost_only_or_no_quality_gain_is_not_accepted():
    assert validate(rebuild=Rebuild(old=(1, 0), new=(1, 0))).status == "rejected"


def test_changed_weights_are_not_comparable():
    rebuild = Rebuild()

    def altered(task_id, snapshot, repeat, label):
        result = rebuild(task_id, snapshot, repeat, label)
        if label == "candidate":
            result["evaluation"]["rubrics"][0]["weight"] = 1
        return result

    assert validate(rebuild=altered).status == "pending"


@pytest.mark.parametrize("mutation", ["failed_rubric", "wrong_task", "duplicate_rubric"])
def test_validation_rechecks_feedback_integrity(mutation):
    rebuild = Rebuild()

    def altered(task_id, snapshot, repeat, label):
        result = rebuild(task_id, snapshot, repeat, label)
        if label == "candidate":
            evaluation = result["evaluation"]
            if mutation == "failed_rubric":
                evaluation["rubrics"][0]["status"] = "error"
            elif mutation == "wrong_task":
                evaluation["task_id"] = "another-validation-task"
            else:
                evaluation["rubrics"][1]["rubric_id"] = evaluation["rubrics"][0]["rubric_id"]
        return result

    assert validate(rebuild=altered).status == "pending"


def test_commit_requires_recorded_accepted_validation_and_is_idempotent(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    change = proposal()
    result = validate(change, store.snapshot())
    try:
        with pytest.raises(ValueError, match="recorded"):
            store.commit(change, result)
        store.stage(change)
        assert not store.snapshot().experiences
        store.record_validation(result)
        first = store.commit(change, result)
        second = store.commit(change, result)
        assert first.version == second.version == 1
        assert len(store.snapshot().experiences) == 1
        assert first.experiences[0].validation_status == "accepted"
    finally:
        store.close()
    reopened = ExperienceStore(tmp_path / "experience.db")
    try:
        assert reopened.snapshot().version == 1
        assert reopened.snapshot().experiences[0].instruction == change.experience.instruction
    finally:
        reopened.close()


@pytest.mark.parametrize("rebuild", [Rebuild(complete=False), Rebuild(old=(1, 0), new=(0, 0))])
def test_pending_and_rejected_cannot_commit(tmp_path, rebuild):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot(), rebuild)
        store.record_validation(result)
        with pytest.raises(ValueError, match="accepted"):
            store.commit(change, result)
        assert store.snapshot().version == 0
    finally:
        store.close()


def test_partial_database_write_failure_rolls_back_entire_commit(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot())
        store.record_validation(result)
        store.db.execute("CREATE TRIGGER reject_commit BEFORE INSERT ON commits BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END")
        with pytest.raises(sqlite3.DatabaseError, match="simulated"):
            store.commit(change, result)
        assert store.snapshot().version == 0
        assert store.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 0
    finally:
        store.close()


def test_read_only_store_refuses_all_mutations(tmp_path):
    path = tmp_path / "experience.db"
    mutable = ExperienceStore(path)
    mutable.close()
    store = ExperienceStore(path, read_only=True)
    try:
        change = proposal()
        result = validate(change, store.snapshot())
        for operation in (lambda: store.stage(change), lambda: store.record_validation(result),
                          lambda: store.commit(change, result), lambda: store.rollback(0)):
            with pytest.raises(PermissionError, match="Frozen"):
                operation()
        assert store.snapshot().version == 0
    finally:
        store.close()


def test_rollback_restores_snapshot_and_freeze_is_immutable_export(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot())
        store.record_validation(result)
        accepted = store.commit(change, result)
        exported = store.freeze(tmp_path / "frozen.json")
        assert json.loads(exported.read_text())["version"] == 1
        with pytest.raises(FileExistsError):
            store.freeze(exported)
        old = store.rollback(0)
        assert old.version == store.snapshot().version == 0
        assert not old.experiences
        assert store.snapshot(1) == accepted
        assert json.loads(exported.read_text())["version"] == 1
    finally:
        store.close()


def test_stale_proposal_and_mutated_record_are_rejected(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot())
        mutated = change.model_copy(deep=True)
        mutated.experience.instruction = "Changed after staging"
        with pytest.raises(ValueError, match="different content"):
            store.stage(mutated)
        store.record_validation(result)
        with pytest.raises(ValueError, match="hashes|content"):
            store.commit(mutated, result)
        store.commit(change, result)
        with pytest.raises(ValueError, match="Stale"):
            store.stage(proposal(proposal_id="another", experience_id="another"))
    finally:
        store.close()


def test_retrieval_excludes_current_restricted_future_and_unaccepted_records():
    now = "2026-09-15T00:00:00+00:00"
    initial = proposal().experience
    entries = []
    for index, (source, status, timestamp) in enumerate([
        ("past-task", "accepted", "2026-09-01T00:00:00+00:00"),
        ("current", "accepted", "2026-09-01T00:00:00+00:00"),
        ("validation-a", "accepted", "2026-09-01T00:00:00+00:00"),
        ("future-task", "accepted", "2026-09-30T00:00:00+00:00"),
        ("pending-task", "staged", "2026-09-01T00:00:00+00:00"),
    ]):
        entry = initial.model_copy(deep=True)
        entry.experience_id = str(index)
        entry.source_task_ids, entry.validation_status, entry.created_at = [source], status, timestamp
        entries.append(entry)
    bank = ExperienceSnapshot(experiences=entries)
    found = retrieve(bank, PublicTask(task_id="current", question="A new task"),
                     excluded_task_ids=["validation-a"], before=now)
    assert [entry["source_task_ids"] for entry in found] == [["past-task"]]


def test_replacement_cannot_create_duplicate_experience_ids():
    a = proposal(experience_id="a").experience
    b = proposal(experience_id="b").experience
    base = ExperienceSnapshot(experiences=[a, b])
    with pytest.raises(ValueError):
        candidate_snapshot(base, proposal(experience_id="b", replaces_id="a"))


def test_reused_committed_proposal_id_cannot_hide_mutation(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot())
        store.record_validation(result)
        store.commit(change, result)
        changed = change.model_copy(deep=True)
        changed.experience.instruction = "Different content under the old accepted ID"
        with pytest.raises(ValueError):
            store.commit(changed, result)
    finally:
        store.close()


def test_repeat_validation_retains_canonical_timestamp_and_rejects_changed_evidence(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        store.stage(change)
        result = validate(change, store.snapshot())
        first = store.record_validation(result)
        repeated = result.model_copy(deep=True)
        repeated.created_at = "2026-10-01T00:00:00+00:00"
        canonical = store.record_validation(repeated)
        assert canonical == first
        changed = repeated.model_copy(deep=True)
        changed.pairs[0]["candidate"]["evaluation"]["score"] = 0.3
        with pytest.raises(ValueError, match="different content"):
            store.record_validation(changed)
        assert store.commit(change, canonical).version == 1
    finally:
        store.close()


def test_task_journal_survives_restart_and_rejects_baseline_or_policy_change(tmp_path):
    path = tmp_path / "experience.db"
    store = ExperienceStore(path)
    base = store.snapshot()
    store.save_task_run("stream", "stream-1", "policy-v1", base, "started")
    submitted = {"task_id": "stream-1", "answer_hash": "answer-v1", "proposals": []}
    store.save_task_run("stream", "stream-1", "policy-v1", base, "submitted", submitted)
    store.close()
    store = ExperienceStore(path)
    try:
        saved = store.task_run("stream", "stream-1")
        assert saved["status"] == "submitted"
        assert saved["baseline_version"] == 0
        assert saved["baseline_hash"] == digest(base)
        assert saved["outcome"] == submitted
        with pytest.raises(ValueError, match="changed policy or baseline"):
            store.save_task_run("stream", "stream-1", "policy-v2", base, "started")
        changed_base = candidate_snapshot(base, proposal())
        with pytest.raises(ValueError, match="changed policy or baseline"):
            store.save_task_run("stream", "stream-1", "policy-v1", changed_base, "started")
        store.save_task_run("stream", "stream-1", "policy-v1", base, "complete", submitted)
        assert store.task_run("stream", "stream-1")["status"] == "complete"
        assert store.task_run("stream", "unknown") is None
    finally:
        store.close()


def test_read_only_journal_cannot_change(tmp_path):
    path = tmp_path / "experience.db"
    store = ExperienceStore(path)
    base = store.snapshot()
    store.save_task_run("stream", "stream-1", "policy-v1", base, "started")
    store.close()
    store = ExperienceStore(path, read_only=True)
    try:
        assert store.task_run("stream", "stream-1")["status"] == "started"
        with pytest.raises(PermissionError):
            store.save_task_run("stream", "stream-1", "policy-v1", base, "complete", {})
    finally:
        store.close()


def test_candidate_hash_changes_before_version_is_committed():
    base = ExperienceSnapshot()
    trial = candidate_snapshot(base, proposal())
    assert trial.version == base.version
    assert digest(trial) != digest(base)
    assert trial.accepted_proposals == ["proposal-1"]
    assert base.accepted_proposals == []


def test_pipeline_journal_reuses_submission_and_enforces_stream_order(tmp_path, monkeypatch):
    from jit_mas.config import MASConfig
    from jit_mas.pipeline import MASPipeline

    # Exercise the real coordinating run()/store journal without a model backend.
    pipeline = MASPipeline.__new__(MASPipeline)
    pipeline.config = MASConfig(backend="scripted")
    pipeline.manifest = manifest()
    pipeline.tasks = {key: PublicTask(task_id=key, question=f"Question for {key}")
                      for key in pipeline.manifest.stream}
    pipeline.private_records = {key: {"task_id": key} for key in pipeline.tasks}
    pipeline.code_hash = "unchanged-code"
    pipeline.output_dir = tmp_path / "runs"
    pipeline.store = ExperienceStore(tmp_path / "experience.db")
    calls = []

    def execute(task_id, snapshot, **kwargs):
        calls.append((task_id, snapshot.version))
        return {"task_id": task_id, "experience_version": snapshot.version,
                "resumed": False, "proposals": [], "evaluation": {"complete": True}}

    pipeline.run_task = execute
    try:
        with pytest.raises(ValueError, match="manifest order"):
            pipeline.run("stream", ["stream-2"])
        first = pipeline.run("stream", ["stream-1", "stream-2"])
        assert calls == [("stream-1", 0), ("stream-2", 0)]
        replay = pipeline.run("stream", ["stream-1", "stream-2"])
        assert len(calls) == 2
        assert all(item["resumed"] for item in replay)
        assert [item["task_id"] for item in first] == [item["task_id"] for item in replay]
        with pytest.raises(ValueError, match="already submitted"):
            pipeline.run("stream", ["stream-1"], resume=False)
        monkeypatch.setattr("jit_mas.pipeline.code_fingerprint", lambda: "changed-code")
        with pytest.raises(ValueError, match="already submitted"):
            pipeline.run("stream", ["stream-1"])
    finally:
        pipeline.store.close()
