"""Structural evolution is explicit, versioned and committed with both experience levels."""

import sqlite3

import pytest

from jit_mas.agent_pool import (
    apply_evolution, apply_updates, effective_pool, get_profile, resolve_profile,
    role_context, seed_pool, validate_bindings,
)
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import (
    AgentEvolutionUpdate, AgentMemoryLesson, AgentPoolObservation, AgentPoolOperation,
    AgentPoolSnapshot, AgentProfile, AgentSpec, ChangeProposal, Experience, ExperienceSnapshot,
    PublicTask, TeamSpec,
)


def profile(identity, *, parent=None, source="evolution-1", evidence="run-1:judge", version=1):
    return AgentProfile(pool_agent_id=identity, role=identity.title(), capabilities=[identity],
                        parent_agent_id=parent, prompt="Use the retained specialist harness.",
                        version=version, source_task_ids=[source], evidence=[evidence])


def operation(kind, targets=(), outputs=(), *, assignments=None, pool=None,
              operation_id="operation-1", source="evolution-1"):
    pool = pool if pool is not None else seed_pool()
    assignments = assignments or {}
    references = set(targets) | set(assignments)
    references.update(parent for parent in assignments.values() if parent is not None)
    references.update(output.parent_agent_id for output in outputs if output.parent_agent_id is not None)
    if kind in {"delete", "prune", "merge"}:
        for child in pool.profiles:
            if child.parent_agent_id in targets:
                references.add(child.pool_agent_id)
                parent_id = child.parent_agent_id
                while parent_id in targets:
                    parent_id = get_profile(pool, parent_id).parent_agent_id
                if parent_id is not None and parent_id not in {output.pool_agent_id for output in outputs}:
                    references.add(parent_id)
    references -= {output.pool_agent_id for output in outputs}
    if kind == "specialize":
        references.update(targets)
    return AgentPoolOperation(operation_id=operation_id, kind=kind, source_task_id=source,
        base_pool_version=pool.version, target_agent_ids=list(targets), profiles=list(outputs),
        parent_assignments=assignments,
        base_agent_versions={identity: get_profile(pool, identity).version for identity in references},
        evidence=["run-1:judge"], rationale="Repeated task failures show a role boundary issue.",
        expected_benefit="Improve future coverage and eliminate repeated work.",
        token_cost_tradeoff="Retain only roles whose performance gain warrants their token cost.")


def observation(*, source="evolution-1", identity="writer", temporary=False):
    return AgentPoolObservation(task_id=source, agent_id="task-worker", pool_agent_id=identity,
        profile_version=1, temporary=temporary, complete=True, submission_score=0.7,
        input_tokens=20, output_tokens=10, tool_calls=1, cost=0.002, evidence=["run-1:judge"])


def proposal():
    return ChangeProposal(proposal_id="meta-1", source_task_id="evolution-1", base_version=0,
        experience=Experience(experience_id="meta-lesson", instruction="Allocate verification coverage.",
            bank="organization", applicability="research tasks", source_task_ids=["evolution-1"],
            evidence=["run-1:judge"]), diff="Retain a design lesson.", rationale="Repeated evidence gaps.",
        evidence=["run-1:judge"], expected_benefit="Improve coverage.")


def test_intentionally_empty_pool_stays_empty_and_default_pool_is_seeded():
    assert len(effective_pool(AgentPoolSnapshot()).profiles) == 6
    empty = AgentPoolSnapshot(initialized=True)
    assert effective_pool(empty) == empty
    changed = apply_evolution(empty, operations=[operation("add", outputs=[profile("new")], pool=empty)],
                              source_task_id="evolution-1")
    assert [item.pool_agent_id for item in changed.profiles] == ["new"]
    removed = apply_evolution(seed_pool(), operations=[operation("prune", targets=[
        item.pool_agent_id for item in seed_pool().profiles])], source_task_id="evolution-1")
    assert removed.initialized and removed.profiles == []
    assert effective_pool(removed).profiles == []
    with pytest.raises(ValueError, match="Unknown"):
        apply_updates(removed, [AgentEvolutionUpdate(update_id="u", pool_agent_id="writer",
            base_agent_version=1, source_task_id="evolution-2", evidence=["run-2:judge"])],
            source_task_id="evolution-2")


def test_temporary_harness_runs_without_changing_pool_and_can_be_promoted_later():
    pool = AgentPoolSnapshot(initialized=True)
    temporary = profile("specialist")
    temporary.source_task_ids = []
    temporary.evidence = []
    agent = AgentSpec(agent_id="worker", role="Specialist", capability="specialist",
        pool_agent_id="specialist", pool_agent_version=1, temporary_profile=temporary,
        creation_rationale="No retained profile can satisfy this task's proof obligation.")
    team = TeamSpec(agents=[agent], synthesizer_id="worker")
    before = pool.model_copy(deep=True)
    validate_bindings(pool, team)
    assert resolve_profile(pool, agent) == temporary
    assert role_context(pool, agent, PublicTask(task_id="current", question="Prove a claim."))["temporary"]
    assert pool == before
    promoted = apply_evolution(pool, operations=[operation("add", outputs=[profile("specialist")], pool=pool)],
        observations=[observation(identity="specialist", temporary=True)], source_task_id="evolution-1")
    assert get_profile(promoted, "specialist").version == 1
    assert promoted.observations[0].temporary


def test_split_preserves_general_parent_and_inherits_mature_memory():
    pool = seed_pool()
    parent = get_profile(pool, "generalist")
    lesson = AgentMemoryLesson(lesson_id="historical", instruction="Check hidden assumptions.",
        applicability="proof tasks", source_task_ids=["older-task"], evidence=["older:judge"])
    parent.memory = [lesson]
    parent.source_task_ids = ["older-task"]
    parent.evidence = ["older:judge"]
    pool.profiles = [parent if item.pool_agent_id == "generalist" else item for item in pool.profiles]
    split = operation("split", ["generalist"], [profile("proof"), profile("numeric")], pool=pool)
    result = apply_evolution(pool, operations=[split], source_task_id="evolution-1")
    assert get_profile(result, "generalist") == parent
    for identity in ("proof", "numeric"):
        child = get_profile(result, identity)
        assert child.parent_agent_id == "generalist" and child.memory == [lesson]
        assert child.source_task_ids == ["older-task", "evolution-1"]
    assert result.version == 1 and result.structural_history == [split]
    assert get_profile(pool, "generalist") == parent


def test_merge_replaces_sources_preserves_lessons_and_reparents_children():
    pool = seed_pool()
    for identity in ("writer", "analyst"):
        retained = get_profile(pool, identity)
        retained.source_task_ids = [identity + "-task"]
        retained.evidence = [identity + ":judge"]
        retained.memory = [AgentMemoryLesson(lesson_id=identity + "-lesson",
            instruction="Check contribution completeness.", applicability=identity,
            source_task_ids=[identity + "-task"], evidence=[identity + ":judge"])]
        pool.profiles = [retained if item.pool_agent_id == identity else item for item in pool.profiles]
    child = profile("child", parent="writer", source="older-task", evidence="older:judge")
    pool.profiles = [*pool.profiles, child]
    merged = apply_evolution(pool, operations=[operation("merge", ["writer", "analyst"],
        [profile("integrator")], pool=pool)], source_task_id="evolution-1")
    assert {"writer", "analyst"}.isdisjoint(item.pool_agent_id for item in merged.profiles)
    assert {item.lesson_id for item in get_profile(merged, "integrator").memory} == {"writer-lesson", "analyst-lesson"}
    assert get_profile(merged, "child").parent_agent_id == "integrator"
    assert get_profile(merged, "child").version == 2
    reuse = operation("add", outputs=[profile("writer", source="evolution-2")], pool=merged,
                      operation_id="reuse", source="evolution-2")
    with pytest.raises(ValueError, match="reuse"):
        apply_evolution(merged, operations=[reuse], source_task_id="evolution-2")


def test_delete_and_prune_keep_children_and_remove_every_requested_profile():
    pool = AgentPoolSnapshot(initialized=True, profiles=[profile("root"), profile("middle", parent="root"),
                                                        profile("leaf", parent="middle")])
    result = apply_evolution(pool, operations=[operation("delete", ["middle"], pool=pool)],
                             source_task_id="evolution-1")
    assert get_profile(result, "leaf").parent_agent_id == "root"
    assert get_profile(result, "leaf").version == 2
    assert "middle" not in {item.pool_agent_id for item in result.profiles}


def test_specialization_and_reorganization_change_full_harness_and_version_once():
    pool = seed_pool()
    specialized = profile("analyst")
    specialized.role = "Mathematical Analyst"
    specialized.capabilities = ["proof", "symbolic analysis"]
    specialized.skills = {"proof_check": "Check every quantifier and inference."}
    specialization = operation("specialize", ["analyst"], [specialized], pool=pool)
    restructure = operation("reorganize", ["analyst"], assignments={"analyst": "generalist"}, pool=pool,
                            operation_id="restructure")
    result = apply_evolution(pool, operations=[specialization, restructure], source_task_id="evolution-1")
    retained = get_profile(result, "analyst")
    assert retained.role == "Mathematical Analyst" and retained.skills == specialized.skills
    assert retained.parent_agent_id == "generalist" and retained.version == 2 and result.version == 1


def local_update(identity, *, lesson_id):
    lesson = AgentMemoryLesson(lesson_id=lesson_id, instruction="Verify the finished contribution.",
        applicability="research tasks", source_task_ids=["evolution-1"], evidence=["run-1:judge"])
    return AgentEvolutionUpdate(update_id="update-" + identity, pool_agent_id=identity,
        base_agent_version=1, source_task_id="evolution-1", lessons=[lesson],
        skills={"current_" + identity: "Apply the new contribution check."}, evidence=["run-1:judge"])


def test_local_update_then_split_inherits_current_lessons_and_skills_once():
    pool = seed_pool()
    update = local_update("generalist", lesson_id="generalist-current")
    split = operation("split", ["generalist"], [profile("proof"), profile("numeric")], pool=pool)
    result = apply_evolution(pool, [update], [split], source_task_id="evolution-1")
    retained = get_profile(result, "generalist")
    assert retained.version == 2 and retained.memory == update.lessons
    assert retained.skills == update.skills and result.version == 1
    for identity in ("proof", "numeric"):
        child = get_profile(result, identity)
        assert child.version == 1 and child.parent_agent_id == "generalist"
        assert child.memory == update.lessons and child.skills == update.skills
        assert child.source_task_ids == ["evolution-1"]
    assert get_profile(pool, "generalist").version == 1


def test_local_updates_then_merge_inherit_both_current_harness_lessons():
    pool = seed_pool()
    updates = [local_update("writer", lesson_id="writer-current"),
               local_update("analyst", lesson_id="analyst-current")]
    merge = operation("merge", ["writer", "analyst"], [profile("integrator")], pool=pool)
    result = apply_evolution(pool, updates, [merge], source_task_id="evolution-1")
    merged = get_profile(result, "integrator")
    assert merged.version == 1 and result.version == 1
    assert {lesson.lesson_id for lesson in merged.memory} == {"writer-current", "analyst-current"}
    assert merged.skills == {**updates[0].skills, **updates[1].skills}
    assert merged.source_task_ids == ["evolution-1"]
    assert result.applied_updates == [item.update_id for item in updates]
    assert {"writer", "analyst"}.isdisjoint(item.pool_agent_id for item in result.profiles)


@pytest.mark.parametrize("kind", ["specialize", "delete", "prune"])
def test_local_update_cannot_conflict_with_specialization_or_removal(kind):
    pool = seed_pool()
    outputs = [profile("writer")] if kind == "specialize" else []
    change = operation(kind, ["writer"], outputs, pool=pool)
    with pytest.raises(ValueError, match="locally updated|Duplicate harness"):
        apply_evolution(pool, [local_update("writer", lesson_id="writer-current")],
                        [change], source_task_id="evolution-1")
    assert get_profile(pool, "writer").version == 1


def test_same_source_cannot_be_split_twice_in_one_structural_decision():
    pool = seed_pool()
    first = operation("split", ["generalist"], [profile("proof"), profile("numeric")], pool=pool)
    second = operation("split", ["generalist"], [profile("logic"), profile("coding")], pool=pool,
                       operation_id="second-split")
    with pytest.raises(ValueError, match="multiple structural"):
        apply_evolution(pool, operations=[first, second], source_task_id="evolution-1")


@pytest.mark.parametrize("invalid", ["cycle", "stale_pool", "stale_member", "unbound", "history", "evidence", "shape"])
def test_invalid_structural_batch_is_pure(invalid):
    pool = seed_pool()
    before = pool.model_copy(deep=True)
    change = operation("add", outputs=[profile("new")], pool=pool)
    if invalid == "cycle":
        change = operation("reorganize", ["writer", "analyst"], assignments={"writer": "analyst", "analyst": "writer"})
    elif invalid == "stale_pool":
        change.base_pool_version = 9
    elif invalid == "stale_member":
        change = operation("delete", ["writer"])
        change.base_agent_versions = {"writer": 9}
    elif invalid == "unbound":
        change = operation("delete", ["writer"])
        change.base_agent_versions = {}
    elif invalid == "history":
        change.profiles[0].source_task_ids = ["invented-task", "evolution-1"]
    elif invalid == "evidence":
        change.profiles[0].evidence = ["unknown:evidence"]
    else:
        change.kind = "merge"
    with pytest.raises(ValueError):
        apply_evolution(pool, operations=[change], source_task_id="evolution-1")
    assert pool == before


def test_atomic_structure_meta_and_observation_commit_survives_restart_and_rollback(tmp_path):
    path = tmp_path / "evolution.sqlite"
    store = ExperienceStore(path)
    change = operation("add", outputs=[profile("new")])
    measure = observation()
    meta = proposal()
    try:
        written = store.commit_evolution(source_task_id="evolution-1", base_version=0,
            proposal=meta, operations=[change], observations=[measure], update_id="batch-1")
        assert written.version == written.agent_pool.version == 1
        assert written.applied_proposals == ["meta-1"] and written.agent_pool.observations == [measure]
        assert get_profile(written.agent_pool, "new").source_task_ids == ["evolution-1"]
        assert store.commit_evolution(source_task_id="evolution-1", base_version=0,
            proposal=meta, operations=[change], observations=[measure], update_id="batch-1") == written
        altered = change.model_copy(deep=True)
        altered.expected_benefit = "Changed immutable decision."
        with pytest.raises(ValueError, match="different content"):
            store.commit_evolution(source_task_id="evolution-1", base_version=0,
                proposal=meta, operations=[altered], observations=[measure], update_id="batch-1")
        assert store.rollback(0) == ExperienceSnapshot()
        assert store.snapshot(1) == written
    finally:
        store.close()
    reopened = ExperienceStore(path)
    try:
        assert reopened.snapshot(1) == written
    finally:
        reopened.close()
    frozen = ExperienceStore(path, read_only=True)
    try:
        with pytest.raises(PermissionError, match="Frozen"):
            frozen.commit_evolution(source_task_id="evolution-1", base_version=0, operations=[change])
    finally:
        frozen.close()


def test_operation_only_and_observation_only_commits_are_persistent(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.sqlite")
    try:
        first = store.commit_evolution(source_task_id="evolution-1", base_version=0,
            operations=[operation("add", outputs=[profile("new")])])
        assert first.version == 1 and first.agent_pool.version == 1
        second = store.commit_evolution(source_task_id="evolution-2", base_version=1,
            observations=[observation(source="evolution-2")])
        assert second.version == 2 and second.agent_pool.version == 2
        assert second.agent_pool.profiles == first.agent_pool.profiles
    finally:
        store.close()


def test_new_profiles_can_be_reorganized_within_the_same_evolution_batch():
    pool = AgentPoolSnapshot(initialized=True)
    addition = operation("add", outputs=[profile("parent"), profile("child")], pool=pool)
    rearrangement = AgentPoolOperation(operation_id="rearrange", kind="reorganize",
        source_task_id="evolution-1", base_pool_version=0, target_agent_ids=["child"],
        parent_assignments={"child": "parent"}, evidence=["run-1:judge"],
        rationale="The child shares methods with its general parent.", expected_benefit="Reuse common methods.",
        token_cost_tradeoff="Organize the selection catalogue without extra execution roles.")
    result = apply_evolution(pool, operations=[addition, rearrangement], source_task_id="evolution-1")
    assert get_profile(result, "child").parent_agent_id == "parent"
    assert get_profile(result, "child").version == 1


def test_partial_dynamic_evolution_failure_rolls_back_both_layers(tmp_path):
    store = ExperienceStore(tmp_path / "evolution.sqlite")
    try:
        store.db.execute("CREATE TRIGGER fail_evolution BEFORE INSERT ON evolution_commits BEGIN SELECT RAISE(ABORT, 'simulated evolution failure'); END")
        with pytest.raises(sqlite3.DatabaseError, match="simulated"):
            store.commit_evolution(source_task_id="evolution-1", base_version=0,
                proposal=proposal(), operations=[operation("add", outputs=[profile("new")])],
                observations=[observation()])
        assert store.snapshot() == ExperienceSnapshot()
        for table in ("proposals", "commits", "evolution_commits", "audit"):
            assert store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        store.close()
