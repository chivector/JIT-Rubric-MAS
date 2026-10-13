"""Batch candidates retain source isolation and use one complete state update."""

import json
from pathlib import Path

import pytest

from jit_mas.agent_pool import apply_evolution, get_profile, seed_pool
from jit_mas.batch_evolution import (BatchEvolutionDecision, _inject_structural_fallback,
                                    _compact_candidate_payload, candidate_batch_snapshot, evolve_batch,
                                    materialize_batch_decision, materialize_batch_proposal_choices)
from jit_mas.budget import MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.schemas import (AgentEvolutionUpdate, AgentMemoryLesson, AgentPoolOperation, AgentProfile,
                            ChangeProposal, Experience, ExperienceSnapshot, SplitManifest, digest)
from scripts.run_jit_mas import make_pipeline


TASKS = [f"batch-task-{index}" for index in range(5)]


def test_structure_candidate_injects_evidence_bound_add(monkeypatch):
    monkeypatch.setenv("JIT_MAS_ENFORCE_STRUCTURAL_OPS", "1")
    base = ExperienceSnapshot(agent_pool=seed_pool())
    decision, audit = _inject_structural_fallback(
        BatchEvolutionDecision(rationale="No shape change proposed."),
        pool=base.agent_pool, task_ids=TASKS, observed_tasks=set(TASKS),
        valid_evidence_ids={"task::event-a", "task::event-b"},
        batch_id="researchrubrics:b9", candidate_index=2)
    assert audit["kind"] == "add"
    candidate = candidate_batch_snapshot(base, decision, source_task_ids=TASKS)
    assert len(candidate.agent_pool.profiles) == len(base.agent_pool.profiles) + 1
    assert candidate.agent_pool.structural_history[-1].kind == "add"


def update(*, sources=None):
    sources = sources or TASKS[:2]
    return AgentEvolutionUpdate(update_id="batch-writer", pool_agent_id="writer",
        base_agent_version=1, source_task_id=sources[0], source_task_ids=sources,
        lessons=[AgentMemoryLesson(lesson_id="coverage-lesson", instruction="Check public coverage before synthesis.",
            applicability="comparison tasks", source_task_ids=sources, evidence=["task::own-event"])],
        evidence=["task::own-event"])


def test_one_candidate_pool_update_retains_multiple_real_tasks_and_leaves_base_unchanged():
    base = ExperienceSnapshot(agent_pool=seed_pool())
    before = digest(base)
    candidate = candidate_batch_snapshot(base, BatchEvolutionDecision(agent_updates=[update()],
        rationale="Consolidate two observed role failures."), source_task_ids=TASKS)
    assert digest(base) == before
    assert candidate.version == 1 and candidate.agent_pool.version == 1
    assert get_profile(candidate.agent_pool, "writer").version == 2
    assert get_profile(candidate.agent_pool, "writer").source_task_ids == TASKS[:2]
    assert get_profile(candidate.agent_pool, "writer").memory[0].source_task_ids == TASKS[:2]


def test_batch_scopes_new_lessons_and_unions_counterevidence():
    lesson = AgentMemoryLesson(lesson_id="evidence-gap", instruction="Check the evidence gap.",
        applicability="research tasks", source_task_ids=TASKS[:2],
        evidence=["task::observed"], counterevidence=["task::counter"])
    update_with_gap = AgentEvolutionUpdate(update_id="batch-writer-gap", pool_agent_id="writer",
        base_agent_version=1, source_task_id=TASKS[0], source_task_ids=TASKS[:2],
        lessons=[lesson], evidence=["task::observed"])
    candidate = candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
        BatchEvolutionDecision(agent_updates=[update_with_gap], rationale="Retain supported evidence."),
        source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "writer").memory[0]
    assert saved.lesson_id == f"{TASKS[0]}:evidence-gap"
    assert saved.counterevidence == ["task::counter"]
    assert "task::counter" in get_profile(candidate.agent_pool, "writer").evidence


def test_batch_still_rejects_duplicate_lessons_within_one_update():
    repeated = update()
    repeated.lessons.append(repeated.lessons[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="duplicate identity"):
        candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
            BatchEvolutionDecision(agent_updates=[repeated], rationale="Malformed duplicate."),
            source_task_ids=TASKS)


def test_batch_scopes_new_lessons_in_retained_profile():
    specialist = AgentProfile(pool_agent_id="specialist", role="Specialist", capabilities=["comparison"],
        prompt="Inspect the task's boundary conditions.", source_task_ids=TASKS[:2],
        evidence=["task::observed"], memory=[AgentMemoryLesson(
            lesson_id="coverage", instruction="Check the coverage boundary.", applicability="comparison",
            source_task_ids=TASKS[:2], evidence=["task::observed"], counterevidence=["task::counter"])])
    operation = AgentPoolOperation(operation_id="retain-specialist-memory", kind="add",
        source_task_id=TASKS[0], source_task_ids=TASKS[:2], base_pool_version=0,
        profiles=[specialist], evidence=["task::observed"], rationale="Retain the observed specialist.",
        expected_benefit="Improve evidence coverage.", token_cost_tradeoff="Reuse the specialist.")
    candidate = candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
        BatchEvolutionDecision(pool_operations=[operation], rationale="Retain supported memory."),
        source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "specialist")
    assert saved.memory[0].lesson_id == f"{TASKS[0]}:coverage"
    assert "task::counter" in saved.evidence


def test_single_task_pool_path_does_not_accept_batch_provenance():
    with pytest.raises(ValueError, match="source task mismatch"):
        apply_evolution(seed_pool(), [update()], source_task_id=TASKS[0])
    with pytest.raises(ValueError, match="source task mismatch"):
        candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
            BatchEvolutionDecision(agent_updates=[update(sources=[TASKS[0], "held-out"])],
                                   rationale="Invalid source."), source_task_ids=TASKS)


def test_batch_add_retains_full_harness_with_real_multi_task_provenance():
    specialist = AgentProfile(pool_agent_id="specialist", role="Specialist", capabilities=["comparison"],
        prompt="Inspect the task's boundary conditions.", skills={"check": "Check coverage."},
        reasoning_strategy="Compare assumptions before conclusions.", planning_strategy="Allocate evidence checks.",
        communication="Publish concise evidence-backed findings.",
        source_task_ids=TASKS[:2], evidence=["task::own-event"])
    operation = AgentPoolOperation(operation_id="retain-specialist", kind="add", source_task_id=TASKS[0],
        source_task_ids=TASKS[:2], base_pool_version=0, profiles=[specialist], evidence=["task::own-event"],
        rationale="Two tasks lacked a suitable retained role.", expected_benefit="Improve evidence coverage.",
        token_cost_tradeoff="Reuse the specialist rather than rebuild it.")
    base = ExperienceSnapshot(agent_pool=seed_pool())
    candidate = candidate_batch_snapshot(base, BatchEvolutionDecision(pool_operations=[operation],
        rationale="Retain the observed specialist."), source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "specialist")
    assert saved.source_task_ids == TASKS[:2]
    assert saved.harness == specialist.harness and saved.skills == specialist.skills
    assert saved.reasoning_strategy == specialist.reasoning_strategy
    assert candidate.agent_pool.version == 1
    assert all(item.pool_agent_id != "specialist" for item in base.agent_pool.profiles)


def compact_operation(kind="specialize", *, patches=None, targets=None):
    targets = targets or ["writer"]
    return {
        "operation_id": "compact-structure", "kind": kind,
        "source_task_id": TASKS[0], "source_task_ids": TASKS[:2], "base_pool_version": 0,
        "target_agent_ids": targets, "base_agent_versions": {target: 1 for target in targets},
        "profile_patches": patches or [{
            "base_profile_id": "writer", "base_profile_version": 1,
            "pool_agent_id": "writer", "prompt": "Check public constraints before synthesis.",
            "skill_updates": {"coverage": "Check each requested dimension."},
        }],
        "evidence": ["task::own-event"], "rationale": "Repair observed coverage gaps.",
        "expected_benefit": "Preserve coverage.", "token_cost_tradeoff": "Reuse the existing prototype.",
    }


def test_compact_specialize_changes_policy_and_preserves_inherited_memory_and_skills():
    pool = seed_pool()
    writer = next(profile for profile in pool.profiles if profile.pool_agent_id == "writer")
    historical = AgentMemoryLesson(lesson_id="historical", instruction="Retain useful evidence.",
        applicability="research", source_task_ids=["earlier-task"], evidence=["earlier::event"])
    writer.memory = [historical]
    writer.source_task_ids = ["earlier-task"]
    writer.evidence = ["earlier::event"]
    base = ExperienceSnapshot(agent_pool=pool)
    before = digest(base)
    decision = BatchEvolutionDecision(pool_operations=[compact_operation()], rationale="Repair coverage.")
    candidate = candidate_batch_snapshot(base, decision, source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "writer")
    assert saved.version == 2
    assert saved.prompt == "Check public constraints before synthesis."
    assert saved.skills == {**writer.skills, "coverage": "Check each requested dimension."}
    assert saved.memory == [historical]
    assert saved.source_task_ids == ["earlier-task", *TASKS[:2]]
    assert digest(base) == before


def test_conditional_skill_extension_preserves_general_genre_instruction():
    base = ExperienceSnapshot(agent_pool=seed_pool())
    general = get_profile(base.agent_pool, "writer").skills["genre_control"]
    patch = {"base_profile_id": "writer", "base_profile_version": 1,
             "pool_agent_id": "writer", "skill_extensions": {
                 "genre_control": "For interchangeable-chapter fiction, verify independent chapter endings."}}
    candidate = candidate_batch_snapshot(base, BatchEvolutionDecision(
        pool_operations=[compact_operation(patches=[patch])], rationale="Keep broad capability with a conditional lesson."),
        source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "writer")
    assert saved.skills["genre_control"].startswith(general)
    assert "For interchangeable-chapter fiction" in saved.skills["genre_control"]
    assert get_profile(base.agent_pool, "writer").skills["genre_control"] == general
    assert saved.version == 2 and saved.source_task_ids == TASKS[:2]


@pytest.mark.parametrize("patch,message", [
    ({"skill_updates": {"genre_control": "Replace."},
      "skill_extensions": {"genre_control": "Extend."}}, "both replace and extend"),
    ({"skill_extensions": {"invented": "Extend."}}, "requires an inherited skill"),
])
def test_conditional_skill_extension_rejects_ambiguous_or_unknown_base(patch, message):
    operation = compact_operation(patches=[{"base_profile_id": "writer", "base_profile_version": 1,
                                           "pool_agent_id": "writer", **patch}])
    with pytest.raises(ValueError, match=message):
        candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
            BatchEvolutionDecision(pool_operations=[operation], rationale="Invalid extension."), source_task_ids=TASKS)


@pytest.mark.parametrize("kind", ["split", "merge"])
def test_compact_structural_patches_produce_complete_inherited_profiles(kind):
    targets = ["writer"] if kind == "split" else ["writer", "analyst"]
    identities = ["writer_child", "writer_checker"] if kind == "split" else ["combined"]
    patches = [{"base_profile_id": "writer", "base_profile_version": 1,
                "pool_agent_id": identity, "prompt": f"Perform the {identity} role."}
               for identity in identities]
    operation = compact_operation(kind, patches=patches, targets=targets)
    base = ExperienceSnapshot(agent_pool=seed_pool())
    candidate = candidate_batch_snapshot(base,
        BatchEvolutionDecision(pool_operations=[operation], rationale="Restructure observed roles."),
        source_task_ids=TASKS)
    for identity in identities:
        saved = get_profile(candidate.agent_pool, identity)
        assert saved.version == 1 and saved.source_task_ids == TASKS[:2]
        assert set(get_profile(base.agent_pool, "writer").skills) <= set(saved.skills)
    assert len(candidate.agent_pool.profiles) == (8 if kind == "split" else 5)
    assert candidate.agent_pool.structural_history[0].kind == kind


@pytest.mark.parametrize("field,value,message", [
    ("base_profile_id", "unknown", "unknown profile"),
    ("base_profile_version", 2, "base profile version"),
    ("pool_agent_id", "renamed_writer", "target identity"),
])
def test_compact_profile_patch_rejects_unknown_stale_or_renamed_specialization(field, value, message):
    operation = compact_operation()
    operation["profile_patches"][0][field] = value
    with pytest.raises(ValueError, match=message):
        candidate_batch_snapshot(ExperienceSnapshot(agent_pool=seed_pool()),
            BatchEvolutionDecision(pool_operations=[operation], rationale="Invalid reference."),
            source_task_ids=TASKS)


def test_compact_specialize_rejects_direct_update_conflict_with_actionable_correction():
    decision = BatchEvolutionDecision(agent_updates=[update()], pool_operations=[compact_operation()],
                                      rationale="Conflicting changes.")
    with pytest.raises(ValueError, match="Remove those direct agent_updates"):
        materialize_batch_decision(ExperienceSnapshot(agent_pool=seed_pool()), decision)


def test_opt_in_compact_specialize_retains_direct_update_lessons_and_policy_changes():
    base = ExperienceSnapshot(agent_pool=seed_pool())
    before = digest(base)
    self_update = update()
    self_update.planning_strategy = "Reserve capacity for final evidence coverage."
    decision = BatchEvolutionDecision(agent_updates=[self_update], pool_operations=[compact_operation()],
                                      rationale="Compose supported memory and specialization.")
    candidate = candidate_batch_snapshot(base, decision, source_task_ids=TASKS,
                                         compose_specializations=True)
    saved = get_profile(candidate.agent_pool, "writer")
    assert saved.prompt == "Check public constraints before synthesis."
    assert saved.skills["coverage"] == "Check each requested dimension."
    assert saved.planning_strategy == self_update.planning_strategy
    assert saved.memory[0].lesson_id == f"{TASKS[0]}:coverage-lesson"
    assert saved.memory[0].source_task_ids == TASKS[:2]
    assert candidate.agent_pool.applied_updates == [self_update.update_id]
    assert candidate.agent_pool.structural_history[0].kind == "specialize"
    assert saved.version == 2 and candidate.agent_pool.version == 1
    assert digest(base) == before


def test_compact_patch_cannot_replace_inherited_memory_or_provenance():
    operation = compact_operation()
    operation["profile_patches"][0]["source_task_ids"] = ["held-out"]
    with pytest.raises(ValueError, match="Extra inputs"):
        BatchEvolutionDecision(pool_operations=[operation], rationale="Invented history.")


def test_compact_merge_requires_explicit_resolution_of_conflicting_skills():
    pool = seed_pool()
    writer = next(profile for profile in pool.profiles if profile.pool_agent_id == "writer")
    analyst = next(profile for profile in pool.profiles if profile.pool_agent_id == "analyst")
    writer.skills["shared"] = "Preserve the requested wording."
    analyst.skills["shared"] = "Check the supporting evidence."
    patch = {"base_profile_id": "writer", "base_profile_version": 1, "pool_agent_id": "combined"}
    operation = compact_operation("merge", patches=[patch], targets=["writer", "analyst"])
    base = ExperienceSnapshot(agent_pool=pool)
    with pytest.raises(ValueError, match="resolve conflicting inherited skills"):
        candidate_batch_snapshot(base,
            BatchEvolutionDecision(pool_operations=[operation], rationale="Merge overlapping roles."),
            source_task_ids=TASKS)
    patch["skill_updates"] = {"shared": "Check support while preserving the requested wording."}
    candidate = candidate_batch_snapshot(base,
        BatchEvolutionDecision(pool_operations=[operation], rationale="Resolve the observed overlap."),
        source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "combined")
    assert saved.skills["shared"] == patch["skill_updates"]["shared"]


def test_compact_add_materializes_executed_retention_candidate_without_losing_memory():
    specialist = AgentProfile(pool_agent_id="specialist", role="Specialist", capabilities=["comparison"],
        prompt="Check source coverage.", source_task_ids=TASKS[:2], evidence=["task::own-event"],
        memory=[AgentMemoryLesson(lesson_id="coverage", instruction="Check every public dimension.",
            applicability="comparison", source_task_ids=TASKS[:2], evidence=["task::own-event"])])
    patch = {"base_profile_id": "specialist", "base_profile_version": 1, "pool_agent_id": "specialist"}
    operation = compact_operation("add", patches=[patch])
    operation["target_agent_ids"], operation["base_agent_versions"] = [], {}
    base = ExperienceSnapshot(agent_pool=seed_pool())
    decision = materialize_batch_decision(base,
        BatchEvolutionDecision(pool_operations=[operation], rationale="Retain an executed specialist."),
        retention_candidates=[specialist])
    candidate = candidate_batch_snapshot(base, decision, source_task_ids=TASKS)
    saved = get_profile(candidate.agent_pool, "specialist")
    assert saved.role == specialist.role and saved.capabilities == specialist.capabilities
    assert saved.memory[0].instruction == specialist.memory[0].instruction
    assert saved.memory[0].source_task_ids == TASKS[:2]


def test_compact_operation_cannot_mix_full_profiles_and_patches():
    operation = compact_operation()
    operation["profiles"] = [get_profile(seed_pool(), "writer").model_dump(mode="json")]
    with pytest.raises(ValueError, match="never both"):
        materialize_batch_decision(ExperienceSnapshot(agent_pool=seed_pool()),
            BatchEvolutionDecision(pool_operations=[operation], rationale="Conflicting representations."))


def proposal(identity, source):
    return ChangeProposal(proposal_id=identity, source_task_id=source, base_version=0,
        experience=Experience(experience_id=identity, bank="rubric", instruction="Check public coverage.",
            applicability="comparison", source_task_ids=[source], evidence=["task::own-event"]),
        diff="Retain a conditional coverage check.", rationale="Observed a coverage gap.",
        evidence=["task::own-event"], expected_benefit="Preserve public coverage.")


def test_proposal_choices_materialize_one_supported_choice_per_task_with_null_abstention():
    proposals = [proposal("first", TASKS[0]), proposal("alternative", TASKS[0]),
                 proposal("second", TASKS[1])]
    decision = BatchEvolutionDecision(
        proposal_choices={TASKS[0]: "alternative", TASKS[1]: None}, rationale="Retain the stronger proposal.")
    materialized = materialize_batch_proposal_choices(decision, source_task_ids=TASKS, proposals=proposals)
    assert materialized.proposal_ids == ["alternative"]
    assert materialized.proposal_choices is None
    assert decision.proposal_ids == [] and decision.proposal_choices[TASKS[0]] == "alternative"
    candidate = candidate_batch_snapshot(ExperienceSnapshot(), decision,
                                        source_task_ids=TASKS, proposals=proposals)
    assert candidate.applied_proposals == ["alternative"]


@pytest.mark.parametrize("choices,message", [
    ({"held-out": None}, "out-of-batch source_task_id"),
    ({TASKS[0]: "unknown"}, "unknown proposal"),
    ({TASKS[0]: "second"}, "source mismatch"),
])
def test_proposal_choices_reject_unknown_tasks_unknown_ids_and_cross_task_selection(choices, message):
    decision = BatchEvolutionDecision(proposal_choices=choices, rationale="Invalid selection.")
    with pytest.raises(ValueError, match=message):
        materialize_batch_proposal_choices(decision, source_task_ids=TASKS,
            proposals=[proposal("first", TASKS[0]), proposal("second", TASKS[1])])


@pytest.mark.parametrize("identities", [[], ["first"]])
def test_proposal_choices_cannot_be_combined_with_legacy_ids_even_when_empty(identities):
    with pytest.raises(ValueError, match="never both"):
        BatchEvolutionDecision(proposal_ids=identities, proposal_choices={TASKS[0]: "first"},
                               rationale="Conflicting formats.")


def test_legacy_duplicate_source_selection_error_names_source_and_allowed_ids():
    proposals = [proposal("first", TASKS[0]), proposal("alternative", TASKS[0])]
    with pytest.raises(ValueError, match="multiple proposals") as error:
        candidate_batch_snapshot(ExperienceSnapshot(),
            BatchEvolutionDecision(proposal_ids=["first", "alternative"], rationale="Duplicate source."),
            source_task_ids=TASKS, proposals=proposals)
    message = str(error.value)
    assert TASKS[0] in message and "allowed_proposal_ids" in message
    assert "first" in message and "alternative" in message and "proposal_choices" in message


class BatchFixtureModel(FixtureModel):
    def __call__(self, messages, **kwargs):
        # Record the effective output ceiling after MeteredModel applies the
        # batch-meta override.  This keeps the regression test at the wrapper
        # boundary instead of inspecting an implementation detail on the
        # fixture provider itself.
        if self.role == "global":
            payload = json.loads(messages[-1]["content"])
            if payload.get("phase") == "batch_evolution_integrate":
                self.provider.meta_max_tokens.append(kwargs.get("max_tokens"))
        return super().__call__(messages, **kwargs)

    def _phase(self, payload):
        if payload["phase"] == "batch_evolution_integrate":
            if self.provider.invalid_first and "response_correction" not in payload:
                return {"agent_updates": [], "pool_operations": []}
            if self.provider.fail_candidate and self.agent_id == "batch-meta-1":
                raise RuntimeError("Synthetic candidate provider failure")
            group = payload["agent_reflection_groups"]["writer"]
            tasks = group["source_task_ids"]
            evidence = group["valid_evidence_ids"]
            if self.provider.compact_specialize:
                operation = compact_operation()
                operation["source_task_id"], operation["source_task_ids"] = tasks[0], tasks
                operation["base_pool_version"] = payload["base_pool_version"]
                operation["evidence"] = ["unobserved::operation"] if self.provider.unknown_profile_evidence else evidence
                return {"pool_operations": [operation], "rationale": "Repair observed writer coverage."}
            counterevidence = (["unobserved::counterevidence"]
                               if self.provider.unknown_lesson_evidence else [])
            operations = []
            if self.provider.unknown_profile_evidence:
                specialist = AgentProfile(pool_agent_id="specialist", role="Specialist", capabilities=["comparison"],
                    prompt="Inspect coverage.", source_task_ids=tasks, evidence=evidence,
                    memory=[AgentMemoryLesson(lesson_id="coverage", instruction="Check coverage.",
                        applicability="comparison", source_task_ids=tasks, evidence=evidence,
                        counterevidence=["unobserved::profile-counter"])])
                operations = [AgentPoolOperation(operation_id="add-specialist", kind="add",
                    source_task_id=tasks[0], source_task_ids=tasks, base_pool_version=payload["base_pool_version"],
                    profiles=[specialist], evidence=evidence, rationale="Retain coverage role.",
                    expected_benefit="Better coverage.", token_cost_tradeoff="Reuse role.").model_dump(mode="json")]
            proposal_ids = [item["proposal_id"] for item in payload["candidate_proposals"][:1]]
            if self.provider.duplicate_source_proposals and proposal_ids:
                proposal_ids = proposal_ids + proposal_ids
            selection = {"proposal_ids": proposal_ids}
            if self.provider.proposal_choices:
                selection = {"proposal_choices": {
                    task_id: next(iter(payload["proposal_options_by_task"][task_id]), None)
                    for task_id in payload["source_task_ids"]}}
            update_id = ("writer-consolidated-update-1"
                         if self.provider.invented_update_id else group["update_id"])
            pool_agent_id = ("unknown-writer"
                             if self.provider.unknown_update_group else "writer")
            return {**selection,
                "agent_updates": [{"update_id": update_id, "pool_agent_id": pool_agent_id,
                    "base_agent_version": group["base_agent_version"],
                    "source_task_id": tasks[0], "source_task_ids": tasks,
                    "lessons": [{"lesson_id": "batch-coverage", "instruction": "Check the public task's boundary conditions.",
                        "applicability": "comparison", "source_task_ids": tasks, "evidence": evidence,
                        "counterevidence": counterevidence}],
                    "evidence": evidence}], "pool_operations": operations, "rationale": "Consolidate observed role lessons."}
        return super()._phase(payload)


class BatchFixtureModels(FixtureModels):
    def __init__(self, *, fail_candidate=False, invalid_first=False):
        super().__init__()
        self.fail_candidate = fail_candidate
        self.invalid_first = invalid_first
        self.duplicate_source_proposals = False
        self.meta_max_tokens = []
        self.unknown_lesson_evidence = False
        self.unknown_profile_evidence = False
        self.compact_specialize = False
        self.proposal_choices = False
        self.invented_update_id = False
        self.unknown_update_group = False

    def create(self, role, agent_id, ledger, stage):
        if role == "global":
            return MeteredModel(BatchFixtureModel(self, role, agent_id), ledger, stage, agent_id, 8192)
        return super().create(role, agent_id, ledger, stage)


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    monkeypatch.setattr("jit_mas.pipeline.code_fingerprint", lambda: "synthetic-frozen-code")
    monkeypatch.setattr("jit_mas.batch_evolution.code_fingerprint", lambda: "synthetic-frozen-code")
    store = ExperienceStore(tmp_path / "state.sqlite")
    models = BatchFixtureModels()
    pipeline = make_pipeline(MASConfig(backend="scripted", evolving_agent_pool=True), store,
                             tmp_path / "task-runs", fixture_models=models)
    original = pipeline.tasks["evolve-comparison"]
    private = pipeline.private_records["evolve-comparison"]
    pipeline.tasks = {task_id: original.model_copy(update={"task_id": task_id,
                      "question": f"Compare storage designs for service {index}."})
                      for index, task_id in enumerate(TASKS)}
    pipeline.private_records = {task_id: {**private, "sample_id": task_id, "task_id": task_id}
                                for task_id in TASKS}
    pipeline.manifest = SplitManifest(evolution=TASKS)
    base = store.snapshot()
    rows = []
    for task_id in TASKS:
        outcome = pipeline.run_task(task_id, base, mode="evolve", attribution=False)
        rows.append({"task_id": task_id, "status": "complete", "run_dir": outcome["run_dir"]})
    yield pipeline, store, models, base, rows
    store.close()


def test_batch_attribution_runs_after_five_frozen_tasks_and_candidates_share_same_base(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    assert not any(json.loads(call["messages"][-1]["content"]).get("phase", "").startswith("attribute")
                   for call in models.calls if call["role"] in {"global", "local"})
    before = digest(store.snapshot())
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert len(candidates) == 3 and all(item["status"] == "complete" for item in candidates)
    assert {item["base_state_hash"] for item in candidates} == {digest(base)}
    assert digest(store.snapshot()) == before
    for candidate in candidates:
        snapshot = ExperienceSnapshot.model_validate(candidate["snapshot"])
        assert snapshot.version == 1 and snapshot.agent_pool.version == 1
        assert get_profile(snapshot.agent_pool, "writer").source_task_ids == TASKS
        assert all(item.startswith(tuple(task_id + "::" for task_id in TASKS))
                   for item in get_profile(snapshot.agent_pool, "writer").evidence)
    phases = [json.loads(call["messages"][-1]["content"])["phase"]
              for call in models.calls if call["role"] in {"global", "local"}]
    assert phases.count("attribute_global") == 5
    assert phases.count("batch_evolution_integrate") == 3
    assert "evolution_integrate" not in phases


def test_batch_normalizes_model_generated_update_id_at_boundary(receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    models.invented_update_id = True
    monkeypatch.delenv("JIT_MAS_BATCH_INVALID_FALLBACK", raising=False)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")

    expected = digest({"batch": "b0", "member": "writer", "base": digest(base),
                       "sources": TASKS})
    assert all(item["status"] == "complete" for item in candidates)
    for candidate in candidates:
        update_record = candidate["decision"]["agent_updates"][0]
        assert update_record["pool_agent_id"] == "writer"
        assert update_record["update_id"] == expected
        assert update_record["update_id"] != "writer-consolidated-update-1"
    assert digest(store.snapshot()) == digest(base)


def test_batch_rejects_unknown_update_group_even_with_generated_update_id(
        receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    models.unknown_update_group = True
    monkeypatch.delenv("JIT_MAS_BATCH_INVALID_FALLBACK", raising=False)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")

    assert all(item["status"] == "failed" for item in candidates)
    assert all("identity" in item["error"] for item in candidates)
    assert digest(store.snapshot()) == digest(base)


def test_candidate_provider_failure_keeps_other_candidates_and_budget(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.fail_candidate = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert [item["status"] for item in candidates] == ["complete", "failed", "complete"]
    assert candidates[1]["snapshot"] is None and candidates[1]["budget"]["records"]
    assert digest(store.snapshot()) == digest(base)


def test_invalid_candidate_can_fallback_to_audited_noop_in_resilient_mode(
        receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    models.fail_candidate = True
    monkeypatch.setenv("JIT_MAS_BATCH_INVALID_FALLBACK", "1")
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert [item["status"] for item in candidates] == ["complete", "complete", "complete"]
    fallback = candidates[1]
    assert fallback["fallback_noop"] is True
    assert fallback["snapshot"] == base.model_dump(mode="json")
    assert fallback["state_hash"] == digest(base)
    assert fallback["error_type"] == "RuntimeError"
    assert fallback["budget"]["records"]
    assert digest(store.snapshot()) == digest(base)


def test_resilient_batch_sanitizer_keeps_one_proposal_per_source_and_audits_drop(
        receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    models.duplicate_source_proposals = True
    monkeypatch.setenv("JIT_MAS_BATCH_SANITIZE", "1")
    monkeypatch.delenv("JIT_MAS_BATCH_INVALID_FALLBACK", raising=False)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")

    assert [item["status"] for item in candidates] == ["complete"] * 3
    for candidate in candidates:
        assert len(candidate["decision"]["proposal_ids"]) == 1
        assert candidate["sanitized_proposals"] == [{
            "proposal_id": candidate["decision"]["proposal_ids"][0],
            "reason": "duplicate_source_task",
        }]
    assert digest(store.snapshot()) == digest(base)


def test_batch_duplicate_source_proposal_remains_strict_by_default(
        receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    models.duplicate_source_proposals = True
    monkeypatch.delenv("JIT_MAS_BATCH_SANITIZE", raising=False)
    monkeypatch.delenv("JIT_MAS_BATCH_INVALID_FALLBACK", raising=False)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")

    assert [item["status"] for item in candidates] == ["failed"] * 3
    assert all("duplicate proposals" in item["error"] for item in candidates)
    assert digest(store.snapshot()) == digest(base)


def test_proposal_choices_run_through_candidates_and_preserve_raw_choice_receipts(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.proposal_choices = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(candidate["status"] == "complete" for candidate in candidates)
    for candidate in candidates:
        assert len(candidate["decision"]["proposal_ids"]) == len(TASKS)
        assert candidate["decision"]["proposal_choices"] is None
        response = json.loads(candidate["calls"][0]["response"])
        assert set(response["proposal_choices"]) == set(TASKS)
        assert "proposal_ids" not in response
    assert digest(store.snapshot()) == digest(base)


def test_batch_meta_output_cap_reaches_metered_wrapper(receipts, tmp_path, monkeypatch):
    pipeline, store, models, base, rows = receipts
    monkeypatch.setenv("JIT_MAS_BATCH_META_MAX_TOKENS", "1234")
    monkeypatch.delenv("JIT_MAS_BATCH_SANITIZE", raising=False)
    monkeypatch.delenv("JIT_MAS_BATCH_INVALID_FALLBACK", raising=False)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")

    assert [item["status"] for item in candidates] == ["complete"] * 3
    assert models.meta_max_tokens == [1234, 1234, 1234]
    assert digest(store.snapshot()) == digest(base)


def test_candidate_gets_one_metered_json_repair_without_task_regeneration(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.invalid_first = True
    exec_before = sum(call["role"] == "exec" for call in models.calls)
    judge_before = sum(call["role"] == "judge" for call in models.calls)
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(item["status"] == "complete" and len(item["calls"]) == 2 for item in candidates)
    assert all(len(item["budget"]["records"]) == 2 for item in candidates)
    assert sum(call["role"] == "exec" for call in models.calls) == exec_before
    assert sum(call["role"] == "judge" for call in models.calls) == judge_before
    assert digest(store.snapshot()) == digest(base)


def test_batch_rejects_unobserved_lesson_counterevidence(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.unknown_lesson_evidence = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(item["status"] == "failed" for item in candidates)
    assert all("unavailable local evidence" in item["error"] for item in candidates)
    assert digest(store.snapshot()) == digest(base)


def test_batch_rejects_unobserved_retained_profile_counterevidence(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.unknown_profile_evidence = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(item["status"] == "failed" for item in candidates)
    assert all("retained profile cites unavailable evidence" in item["error"] for item in candidates)
    assert digest(store.snapshot()) == digest(base)


def test_compact_specialization_runs_through_candidate_evidence_validation(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.compact_specialize = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(candidate["status"] == "complete" for candidate in candidates)
    for candidate in candidates:
        operation = candidate["decision"]["pool_operations"][0]
        assert operation["profiles"][0]["pool_agent_id"] == "writer"
        assert "profile_patches" not in operation
        assert get_profile(ExperienceSnapshot.model_validate(candidate["snapshot"]).agent_pool, "writer").version == 2
    assert digest(store.snapshot()) == digest(base)


def test_compact_specialization_rejects_unobserved_operation_evidence(receipts, tmp_path):
    pipeline, store, models, base, rows = receipts
    models.compact_specialize = True
    models.unknown_profile_evidence = True
    candidates = evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert all(candidate["status"] == "failed" for candidate in candidates)
    assert all("unavailable source tasks or evidence" in candidate["error"] for candidate in candidates)
    assert digest(store.snapshot()) == digest(base)


def test_batch_rejects_mutated_task_base_before_reflection(receipts, tmp_path):
    pipeline, _, models, base, rows = receipts
    frozen_path = Path(rows[1]["run_dir"]) / "frozen_plan.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["experience_hash"] = "changed"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    before_calls = len(models.calls)
    with pytest.raises(ValueError, match="base state"):
        evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert len(models.calls) == before_calls


def test_batch_rejects_feedback_from_another_answer_before_reflection(receipts, tmp_path):
    pipeline, _, models, base, rows = receipts
    path = Path(rows[0]["run_dir"]) / "evaluation.json"
    feedback = json.loads(path.read_text(encoding="utf-8"))
    feedback["raw"]["submission_answer_hash"] = "wrong-answer"
    path.write_text(json.dumps(feedback), encoding="utf-8")
    before_calls = len(models.calls)
    with pytest.raises(ValueError, match="submitted task answer"):
        evolve_batch(pipeline, base, rows, tmp_path / "batch", batch_id="b0")
    assert len(models.calls) == before_calls


def test_all_failed_tasks_preserve_three_base_candidates_without_model_calls(receipts, tmp_path):
    pipeline, _, models, base, rows = receipts
    failed = [{"task_id": row["task_id"], "status": "failed"} for row in rows]
    before_calls = len(models.calls)
    candidates = evolve_batch(pipeline, base, failed, tmp_path / "batch", batch_id="b0")
    assert all(item["no_evidence"] and item["state_hash"] == digest(base) for item in candidates)
    assert len(models.calls) == before_calls


def test_compact_candidate_payload_drops_audit_noise_and_preserves_provenance():
    evidence = ["task-0::finding", "task-0::observation"]
    payload = {
        "task_reflections": [{
            "task_id": "task-0", "schema_version": "1.0",
            "findings": [{"finding_id": "f-1", "rubric_ids": ["r-1"],
                          "categories": ["execution"], "hypothesis": "repair",
                          "supporting_evidence": evidence, "opposing_evidence": [],
                          "alternatives": [], "component": "", "uncertainty": 0.25,
                          "success": False, "raw": {"request": "duplicate"}}],
            "observations": [{"task_id": "task-0", "agent_id": "writer",
                              "pool_agent_id": "writer", "profile_version": 1,
                              "temporary": False, "submission_score": None, "complete": True,
                              "input_tokens": 1, "output_tokens": 2, "tool_calls": 0,
                              "usage_available": False, "usage_estimated": True,
                              "evidence": evidence, "schema_version": "1.0",
                              "audit": {"raw_request": "duplicate"}}],
            "creations": [],
        }],
        "agent_reflection_groups": {"writer": {
            "source_task_ids": ["task-0"], "valid_evidence_ids": evidence,
            "base_agent_version": 1, "expected_update_id": "u-1", "update_id": "u-1",
            "records": [{"source_task_id": "task-0", "pool_agent_id": "writer",
                         "base_agent_version": 1, "update_id": "task-update",
                         "lessons": [{"lesson_id": "l-1", "instruction": "Check coverage",
                                      "applicability": "research", "capability": "review",
                                      "task_signals": ["comparison"], "source_task_ids": ["task-0"],
                                      "evidence": evidence, "counterevidence": [],
                                      "created_at": "audit"}],
                         "evidence": evidence, "prompt": None, "skills": None,
                         "schema_version": "1.0"}],
        }},
        "candidate_proposals": [{
            "proposal_id": "p-1", "source_task_id": "task-0", "base_version": 0,
            "diff": "Add a check.", "rationale": "Observed gap.",
            "expected_benefit": "Fewer misses.", "risks": [], "evidence": evidence,
            "schema_version": "1.0", "validation_plan": "audit",
            "experience": {"experience_id": "x-1", "bank": "execution", "version": 1,
                           "instruction": "Check coverage.", "applicability": "research",
                           "capability": "review", "task_signals": ["comparison"],
                           "source_task_ids": ["task-0"], "evidence": evidence,
                           "counterevidence": [], "created_at": "audit",
                           "schema_version": "1.0"},
        }],
        "agent_pool": {
            "version": 2, "initialized": True, "applied_updates": ["u-1"],
            "observations": [{"task_id": "task-0", "evidence": evidence}],
            "profiles": [{"pool_agent_id": "writer", "version": 1, "role": "Writer",
                          "capabilities": ["writing"], "prompt": "long duplicate prompt",
                          "skills": {"format": "Check format."}, "memory": [],
                          "source_task_ids": ["task-0"], "evidence": evidence,
                          "schema_version": "1.0"}],
            "structural_history": [{"operation_id": "op-1", "kind": "specialize",
                                    "source_task_id": "task-0", "source_task_ids": ["task-0"],
                                    "base_pool_version": 1, "target_agent_ids": ["writer"],
                                    "base_agent_versions": {"writer": 1}, "evidence": evidence,
                                    "rationale": "Observed gap", "expected_benefit": "Coverage",
                                    "token_cost_tradeoff": "Small", "profiles": [],
                                    "schema_version": "1.0"}],
        },
        "valid_evidence_ids": evidence,
    }

    compact = _compact_candidate_payload(payload)
    encoded = json.dumps(compact, ensure_ascii=False)

    assert len(encoded) < len(json.dumps(payload, ensure_ascii=False))
    assert compact["task_reflections"][0]["findings"][0]["supporting_evidence"] == evidence
    assert compact["task_reflections"][0]["observations"][0]["evidence"] == evidence
    assert compact["agent_reflection_groups"]["writer"]["valid_evidence_ids"] == evidence
    assert compact["agent_reflection_groups"]["writer"]["records"][0]["lessons"][0]["evidence"] == evidence
    assert compact["candidate_proposals"][0]["experience"]["evidence"] == evidence
    assert compact["agent_pool"]["structural_history"][0]["evidence"] == evidence
    assert "raw" not in compact["task_reflections"][0]["findings"][0]
    assert "prompt" not in compact["agent_pool"]["profiles"][0]
    assert "observations" not in compact["agent_pool"]
    assert "applied_updates" not in compact["agent_pool"]
