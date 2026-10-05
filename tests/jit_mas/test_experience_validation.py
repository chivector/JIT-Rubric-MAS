"""Direct experience updates, immutable persistence, and legacy data compatibility."""

import json
import sqlite3

import pytest

from jit_mas.experience import ExperienceStore, candidate_snapshot, retrieve
from jit_mas.schemas import (AgentEvolutionUpdate, AgentMemoryLesson, ChangeProposal,
                              Experience, ExperienceSnapshot, PublicTask, digest)


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
    )


def legacy_proposal(change):
    data = change.model_dump(mode="json")
    data["experience"]["validation_status"] = "staged"
    data["validation_plan"] = "Historical paired rebuild plan"
    data["validation_result"] = {}
    return data


def legacy_database(path, *, committed=False):
    change = proposal()
    base = ExperienceSnapshot().model_dump(mode="json")
    base["policy_versions"] = {"jit_mas": "1.0"}
    base["accepted_proposals"] = base.pop("applied_proposals")
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE snapshots(version INTEGER PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE staging(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE validations(id TEXT PRIMARY KEY, proposal_id TEXT, body TEXT NOT NULL);
            CREATE TABLE commits(proposal_id TEXT PRIMARY KEY, version INTEGER, validation_id TEXT);
            CREATE TABLE audit(id INTEGER PRIMARY KEY, body TEXT NOT NULL);
        """)
        db.execute("INSERT INTO snapshots VALUES(0,?)", (json.dumps(base),))
        db.execute("INSERT INTO staging VALUES(?,?)", (change.proposal_id, json.dumps(legacy_proposal(change))))
        # Invalid historical decision JSON proves direct updates never read it.
        db.execute("INSERT INTO validations VALUES('historical','proposal-1','opaque archived decision')")
        if committed:
            saved = candidate_snapshot(ExperienceSnapshot(), change).model_dump(mode="json")
            saved["version"] = 1
            saved["policy_versions"] = {"jit_mas": "1.0"}
            saved["accepted_proposals"] = saved.pop("applied_proposals")
            saved["experiences"][0]["validation_status"] = "accepted"
            db.execute("INSERT INTO snapshots VALUES(1,?)", (json.dumps(saved),))
            db.execute("INSERT INTO commits VALUES('proposal-1',1,'historical')")
        db.execute("INSERT INTO state VALUES('current',?)", ("1" if committed else "0",))
    return change


def test_direct_commit_has_no_quality_gate_and_is_idempotent(tmp_path):
    path = tmp_path / "experience.db"
    store = ExperienceStore(path)
    change = proposal()
    try:
        first = store.commit(change)
        second = store.commit(change)
        assert first == second
        assert first.version == store.snapshot().version == 1
        assert first.applied_proposals == [change.proposal_id]
        assert first.policy_versions["experience_update"] == "direct-v1"
        assert len(first.experiences) == 1
        assert "validation_status" not in first.experiences[0].model_dump()
        assert "validation_plan" not in change.model_dump()
        tables = {row[0] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not tables.intersection({"staging", "validations"})
        audit = [json.loads(row[0]) for row in store.db.execute("SELECT body FROM audit")]
        assert len(audit) == 1
        assert audit[0]["event"] == "direct_update"
        assert audit[0]["proposal_hash"] == digest(change)
        assert audit[0]["updated_hash"] == digest(first)
    finally:
        store.close()
    reopened = ExperienceStore(path)
    try:
        assert reopened.snapshot() == first
        assert reopened.commit(change) == first
    finally:
        reopened.close()


def test_candidate_is_a_pure_uncommitted_application():
    base, change = ExperienceSnapshot(), proposal()
    trial = candidate_snapshot(base, change)
    assert trial.version == base.version == 0
    assert digest(trial) != digest(base)
    assert trial.applied_proposals == [change.proposal_id]
    assert base.applied_proposals == [] and base.experiences == []
    assert change.experience.model_dump() == trial.experiences[0].model_dump()
    trial.experiences[0].instruction = "Changed copy"
    assert change.experience.instruction != trial.experiences[0].instruction


def test_partial_database_write_failure_rolls_back_entire_commit(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        store.db.execute("CREATE TRIGGER fail_commit BEFORE INSERT ON commits BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END")
        with pytest.raises(sqlite3.DatabaseError, match="simulated"):
            store.commit(proposal())
        assert store.snapshot().version == 0
        assert store.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        for table in ("proposals", "commits", "audit"):
            assert store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        store.db.execute("DROP TRIGGER fail_commit")
        assert store.commit(proposal()).version == 1
    finally:
        store.close()


def test_read_only_store_refuses_all_mutations(tmp_path):
    path = tmp_path / "experience.db"
    ExperienceStore(path).close()
    store = ExperienceStore(path, read_only=True)
    try:
        for operation in (lambda: store.commit(proposal()), lambda: store.rollback(0),
                          lambda: store.save_task_run("stream", "one", "policy", store.snapshot(), "complete")):
            with pytest.raises(PermissionError, match="Frozen"):
                operation()
        assert store.snapshot().version == 0
    finally:
        store.close()


def test_rollback_and_replay_never_resurrect_an_old_update(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        applied = store.commit(change)
        exported = store.freeze(tmp_path / "frozen.json")
        with pytest.raises(FileExistsError):
            store.freeze(exported)
        old = store.rollback(0)
        assert old.version == store.snapshot().version == 0
        assert not old.experiences
        assert store.commit(change) == applied
        assert store.snapshot().version == 0
        assert store.snapshot(1) == applied
        assert json.loads(exported.read_text())["version"] == 1
        branch = store.commit(proposal(proposal_id="new-branch", experience_id="branch-entry"))
        assert branch.version == 2
        assert branch.applied_proposals == ["new-branch"]
        assert [entry.experience_id for entry in branch.experiences] == ["branch-entry"]
    finally:
        store.close()


def test_stale_base_and_reused_committed_id_do_not_change_store(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        change = proposal()
        applied = store.commit(change)
        mutated = change.model_copy(deep=True)
        mutated.experience.instruction = "Different content under the old ID"
        with pytest.raises(ValueError, match="different content"):
            store.commit(mutated)
        with pytest.raises(ValueError, match="Stale"):
            store.commit(proposal(proposal_id="another", experience_id="another"))
        assert store.snapshot() == applied
        assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 1
    finally:
        store.close()


def test_other_connection_cannot_apply_a_stale_base(tmp_path):
    path = tmp_path / "experience.db"
    first, second = ExperienceStore(path), ExperienceStore(path)
    try:
        assert second.snapshot().version == 0
        first.commit(proposal())
        with pytest.raises(ValueError, match="Stale"):
            second.commit(proposal(proposal_id="competing", experience_id="competing"))
        assert first.snapshot() == second.snapshot()
    finally:
        first.close()
        second.close()


def test_replacement_preserves_other_entries_and_checks_collisions(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    try:
        store.commit(proposal(experience_id="a"))
        base = store.commit(proposal(base_version=1, proposal_id="b", experience_id="b"))
        with pytest.raises(ValueError, match="collides"):
            store.commit(proposal(base_version=2, proposal_id="collision", experience_id="b", replaces_id="a"))
        with pytest.raises(ValueError, match="does not exist"):
            store.commit(proposal(base_version=2, proposal_id="missing", replaces_id="absent"))
        with pytest.raises(ValueError, match="already exists"):
            store.commit(proposal(base_version=2, proposal_id="duplicate", experience_id="a"))
        assert store.snapshot() == base
        replaced = store.commit(proposal(base_version=2, proposal_id="replace-a", experience_id="c", replaces_id="a"))
        assert [entry.experience_id for entry in replaced.experiences] == ["b", "c"]
    finally:
        store.close()


def test_retrieval_excludes_current_restricted_and_future_sources():
    initial = proposal().experience
    entries = []
    for index, (source, timestamp) in enumerate([
        ("past-task", "2026-09-01T00:00:00+00:00"),
        ("current", "2026-09-01T00:00:00+00:00"),
        ("validation-a", "2026-09-01T00:00:00+00:00"),
        ("future-task", "2026-09-30T00:00:00+00:00"),
    ]):
        entry = initial.model_copy(deep=True)
        entry.experience_id = str(index)
        entry.source_task_ids, entry.created_at = [source], timestamp
        entries.append(entry)
    bank = ExperienceSnapshot(experiences=entries)
    found = retrieve(bank, PublicTask(task_id="current",
                     question="Plan independent evidence collection and synthesis for a new task."),
                     excluded_task_ids=["validation-a"], before="2026-09-15T00:00:00+00:00")
    assert [entry["source_task_ids"] for entry in found] == [["past-task"]]
    assert "validation_status" not in found[0]


def test_retrieval_rejects_eligible_source_outside_public_task_scope():
    entry = proposal().experience
    entry.source_task_ids = ["past-task"]
    bank = ExperienceSnapshot(experiences=[entry])
    found = retrieve(bank, PublicTask(task_id="current", question="A new task"),
                     excluded_task_ids=["validation-a"], before="2026-09-15T00:00:00+00:00")
    assert found == []


def test_legacy_snapshot_only_activates_previously_committed_members():
    data = proposal().experience.model_dump(mode="json")
    active = {**data, "validation_status": "accepted"}
    staged = {**data, "experience_id": "legacy-draft", "validation_status": "staged"}
    payload = {"version": 2, "accepted_proposals": ["old-update"], "experiences": [active, staged]}
    snapshot = ExperienceSnapshot.model_validate(payload)
    assert snapshot.applied_proposals == ["old-update"]
    assert [entry.experience_id for entry in snapshot.experiences] == ["experience-1"]
    assert "validation_status" not in snapshot.model_dump_json()
    assert "accepted_proposals" not in snapshot.model_dump_json()
    assert len(payload["experiences"]) == 2


def test_legacy_aliases_are_narrow_and_conflicts_are_errors():
    assert ChangeProposal.model_validate(legacy_proposal(proposal())) == proposal()
    with pytest.raises(ValueError, match="Conflicting"):
        ExperienceSnapshot.model_validate({"accepted_proposals": ["a"], "applied_proposals": ["b"]})
    with pytest.raises(ValueError, match="historical experience status"):
        Experience.model_validate({**proposal().experience.model_dump(), "validation_status": "unknown"})
    with pytest.raises(ValueError, match="historical validation plan"):
        ChangeProposal.model_validate({**legacy_proposal(proposal()), "validation_plan": 1})
    with pytest.raises(ValueError, match="historical validation result"):
        ChangeProposal.model_validate({**legacy_proposal(proposal()), "validation_result": []})
    with pytest.raises(ValueError):
        ExperienceSnapshot.model_validate({"quality_status": "verified"})


def test_legacy_staging_is_inactive_until_explicit_update_and_cannot_change_identity(tmp_path):
    path = tmp_path / "legacy.db"
    change = legacy_database(path)
    store = ExperienceStore(path)
    try:
        assert store.snapshot().experiences == []
        mutated = change.model_copy(deep=True)
        mutated.experience.instruction = "Altered historical draft"
        with pytest.raises(ValueError, match="different content"):
            store.commit(mutated)
        assert store.snapshot().version == 0
        assert store.commit(change).version == 1
        assert store.db.execute("SELECT body FROM validations").fetchone()[0] == "opaque archived decision"
        assert store.db.execute("SELECT COUNT(*) FROM validations").fetchone()[0] == 1
        assert store.db.execute("SELECT validation_id FROM commits").fetchone()[0] is None
        assert store.db.execute("SELECT COUNT(*) FROM staging").fetchone()[0] == 1
    finally:
        store.close()


def test_legacy_committed_identity_and_read_only_snapshots_survive_migration(tmp_path):
    path = tmp_path / "legacy.db"
    change = legacy_database(path, committed=True)
    readonly = ExperienceStore(path, read_only=True)
    try:
        assert readonly.snapshot().applied_proposals == [change.proposal_id]
        assert readonly.snapshot().experiences[0] == change.experience
        tables = {row[0] for row in readonly.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "proposals" not in tables
    finally:
        readonly.close()
    store = ExperienceStore(path)
    try:
        assert store.commit(change).version == 1
        mutated = change.model_copy(deep=True)
        mutated.rationale = "New rationale under historical identity"
        with pytest.raises(ValueError, match="different content"):
            store.commit(mutated)
        store.rollback(0)
        assert store.commit(change).version == 1
        assert store.snapshot().version == 0
        assert store.db.execute("SELECT body FROM validations").fetchone()[0] == "opaque archived decision"
    finally:
        store.close()


def test_missing_historical_proposal_body_cannot_bypass_identity_check(tmp_path):
    path = tmp_path / "legacy.db"
    change = legacy_database(path, committed=True)
    store = ExperienceStore(path)
    try:
        with store.db:
            store.db.execute("DELETE FROM staging")
        with pytest.raises(ValueError, match="immutable historical"):
            store.commit(change)
        assert store.snapshot().version == 1
    finally:
        store.close()


def test_new_update_versions_policy_without_relabeling_historical_snapshot(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path, committed=True)
    store = ExperienceStore(path)
    try:
        historical_body = store.db.execute("SELECT body FROM snapshots WHERE version=1").fetchone()[0]
        assert "experience_update" not in store.snapshot().policy_versions
        updated = store.commit(proposal(base_version=1, proposal_id="direct-update", experience_id="new-entry"))
        assert updated.policy_versions == {"jit_mas": "1.0", "experience_update": "direct-v1"}
        assert store.snapshot(1).policy_versions == {"jit_mas": "1.0"}
        assert store.db.execute("SELECT body FROM snapshots WHERE version=1").fetchone()[0] == historical_body
    finally:
        store.close()


def test_task_journal_survives_restart_and_preserves_baseline_binding(tmp_path):
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
        with pytest.raises(ValueError, match="changed policy or baseline"):
            store.save_task_run("stream", "stream-1", "policy-v1", candidate_snapshot(base, proposal()), "started")
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


def agent_update(task_id="evolution-1", update_id="agent-update-1", base_agent_version=1):
    return AgentEvolutionUpdate(
        update_id=update_id, pool_agent_id="writer", base_agent_version=base_agent_version,
        source_task_id=task_id,
        lessons=[AgentMemoryLesson(lesson_id="lesson-1", instruction="Preserve source links.",
                                   applicability="evidence synthesis", source_task_ids=[task_id],
                                   evidence=["run-1:evidence"], created_at="2026-10-01T00:00:00+00:00")],
        communication="Publish concise evidence-linked drafts.", evidence=["run-1:evidence"])


def test_agent_pool_update_is_atomic_idempotent_and_survives_restart(tmp_path):
    path = tmp_path / "evolution.db"
    store = ExperienceStore(path)
    try:
        first = store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                       updates=[agent_update()])
        assert first.version == 1 and first.agent_pool.version == 1
        writer = next(profile for profile in first.agent_pool.profiles if profile.pool_agent_id == "writer")
        assert writer.version == 2 and "lesson-1" in {lesson.lesson_id for lesson in writer.memory}
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                      updates=[agent_update()]) == first
        assert store.snapshot().version == 1
    finally:
        store.close()
    reopened = ExperienceStore(path)
    try:
        assert reopened.snapshot() == first
        with pytest.raises(ValueError, match="Stale"):
            reopened.commit_evolution(source_task_id="evolution-2", base_version=0,
                                      updates=[agent_update("evolution-2", "agent-update-2")])
    finally:
        reopened.close()


def test_agent_pool_update_rolls_back_as_one_snapshot_and_readonly_is_frozen(tmp_path):
    path = tmp_path / "evolution.db"
    store = ExperienceStore(path)
    try:
        applied = store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                         updates=[agent_update()])
        assert store.rollback(0).agent_pool.profiles == []
        assert store.snapshot().version == 0
        assert store.snapshot(1) == applied
    finally:
        store.close()
    frozen = ExperienceStore(path, read_only=True)
    try:
        assert frozen.snapshot(1) == applied
        with pytest.raises(PermissionError, match="Frozen"):
            frozen.commit_evolution(source_task_id="evolution-2", base_version=0,
                                    updates=[agent_update("evolution-2", "agent-update-2")])
    finally:
        frozen.close()


def test_dual_evolution_commits_both_layers_once_and_preserves_pool_on_meta_write(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.db")
    try:
        meta, update = proposal(), agent_update()
        evolved = store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                         proposal=meta, updates=[update], update_id="dual-1")
        assert evolved.version == evolved.agent_pool.version == 1
        assert evolved.applied_proposals == [meta.proposal_id]
        assert evolved.agent_pool.applied_updates == [update.update_id]
        assert store.commit(meta) == evolved
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                      proposal=meta, updates=[update], update_id="dual-1") == evolved
        next_meta = proposal(base_version=1, proposal_id="meta-2", experience_id="meta-2")
        following = store.commit(next_meta)
        assert following.version == 2
        assert following.agent_pool == evolved.agent_pool
        assert digest(candidate_snapshot(evolved, next_meta).agent_pool) == digest(evolved.agent_pool)
        store.rollback(0)
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                      proposal=meta, updates=[update], update_id="dual-1") == evolved
        assert store.snapshot().version == 0
    finally:
        store.close()


def test_partial_dual_evolution_write_failure_rolls_back_both_layers(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.db")
    try:
        store.db.execute("CREATE TRIGGER fail_evolution BEFORE INSERT ON evolution_commits BEGIN SELECT RAISE(ABORT, 'simulated evolution failure'); END")
        with pytest.raises(sqlite3.DatabaseError, match="simulated"):
            store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                    proposal=proposal(), updates=[agent_update()])
        assert store.snapshot() == ExperienceSnapshot()
        assert store.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        for table in ("proposals", "commits", "evolution_commits", "audit"):
            assert store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        store.db.execute("DROP TRIGGER fail_evolution")
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                       proposal=proposal(), updates=[agent_update()]).version == 1
    finally:
        store.close()


@pytest.mark.parametrize("violation", ["source", "lesson_source", "agent_version", "duplicate",
                                        "meta_source", "meta_version"])
def test_invalid_dual_evolution_batch_cannot_change_snapshot(tmp_path, violation):
    store = ExperienceStore(tmp_path / "evolution.db")
    change, update = proposal(), agent_update()
    updates = [update]
    if violation == "source":
        update.source_task_id = "another-task"
    elif violation == "lesson_source":
        update.lessons[0].source_task_ids = ["another-task"]
    elif violation == "agent_version":
        update.base_agent_version = 9
    elif violation == "duplicate":
        updates.append(update.model_copy(deep=True))
    elif violation == "meta_source":
        change.source_task_id = "another-task"
    else:
        change.base_version = 8
    try:
        with pytest.raises(ValueError):
            store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                    proposal=change, updates=updates)
        assert store.snapshot() == ExperienceSnapshot()
        assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 0
    finally:
        store.close()


def test_evolution_update_identity_cannot_be_rebound_or_reapplied(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.db")
    try:
        update = agent_update()
        first = store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                       updates=[update], update_id="batch-1")
        mutated = update.model_copy(deep=True)
        mutated.communication = "Different retained behavior"
        with pytest.raises(ValueError, match="different content"):
            store.commit_evolution(source_task_id="evolution-1", base_version=0,
                                    updates=[mutated], update_id="batch-1")
        with pytest.raises(ValueError, match="Duplicate"):
            store.commit_evolution(source_task_id="evolution-1", base_version=1,
                                    updates=[update], update_id="batch-2")
        assert store.snapshot() == first
        assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 1
    finally:
        store.close()


def test_empty_evolution_batch_preserves_current_version(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.db")
    try:
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0) == store.snapshot()
        store.commit_evolution(source_task_id="evolution-1", base_version=0, updates=[agent_update()])
        with pytest.raises(ValueError, match="Stale"):
            store.commit_evolution(source_task_id="evolution-2", base_version=0)
        assert store.commit_evolution(source_task_id="evolution-2", base_version=1).version == 1
    finally:
        store.close()
