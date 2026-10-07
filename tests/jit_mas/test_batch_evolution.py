"""Batch candidates retain source isolation and use one complete state update."""

import json
from pathlib import Path

import pytest

from jit_mas.agent_pool import apply_evolution, get_profile, seed_pool
from jit_mas.batch_evolution import BatchEvolutionDecision, candidate_batch_snapshot, evolve_batch
from jit_mas.budget import MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.schemas import (AgentEvolutionUpdate, AgentMemoryLesson, AgentPoolOperation, AgentProfile,
                            ExperienceSnapshot, SplitManifest, digest)
from scripts.run_jit_mas import make_pipeline


TASKS = [f"batch-task-{index}" for index in range(5)]


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
            return {"proposal_ids": proposal_ids,
                "agent_updates": [{"update_id": group["update_id"], "pool_agent_id": "writer",
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
