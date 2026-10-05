"""Dynamic Pool changes happen only after feedback and survive frozen replay."""

import copy
import json
import threading
from pathlib import Path

import pytest

from jit_mas.budget import MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.pipeline import attribution_source_digest, submitted_source_digest
from jit_mas.schemas import AgentHarnessPolicy, AgentProfile, digest
from scripts.run_jit_mas import make_pipeline


def temporary_profile(identity="queue_specialist"):
    return AgentProfile(
        pool_agent_id=identity, role="Queue Protocol Specialist", capabilities=["comparison", "queue reliability"],
        parent_agent_id="generalist", prompt="Check queue delivery assumptions and failure recovery before synthesis.",
        skills={"delivery_check": "Compare delivery guarantees under failure and recovery."},
        reasoning_strategy="Test assumptions against a failure sequence.",
        planning_strategy="Inspect guarantees, costs and unresolved recovery gaps.",
        communication="Publish supported conclusions and unresolved assumptions.",
        harness=AgentHarnessPolicy(memory_policy="recent", memory_window=3),
    ).model_dump(mode="json")


class DynamicFixtureModel(FixtureModel):
    def _phase(self, payload):
        response = super()._phase(payload)
        phase = payload["phase"]
        if phase == "predict" and self.provider.temporary:
            candidate = next(item for item in response["candidates"] if item["agent_id"] in {"analyst", "composer"})
            retained = next((member for member in payload["agent_pool_catalogue"]
                             if member["pool_agent_id"] == "queue_specialist"), None)
            if retained is not None:
                candidate.update(pool_agent_id=retained["pool_agent_id"], pool_agent_version=retained["version"],
                                 role=retained["role"], temporary_profile=None, creation_rationale="")
            else:
                profile = temporary_profile()
                candidate.update(pool_agent_id=profile["pool_agent_id"], pool_agent_version=1,
                                 role=profile["role"], temporary_profile=profile,
                                 creation_rationale="No retained role has this delivery-failure harness; create one compact specialist.")
                unused = copy.deepcopy(candidate)
                unused.update(agent_id="unused", pool_agent_id="unused_specialist", depends_on=[])
                unused["temporary_profile"]["pool_agent_id"] = "unused_specialist"
                response["candidates"].append(unused)
        if phase == "reconcile" and self.provider.temporary:
            team = response["team"]
            team["agents"] = [agent for agent in team["agents"] if agent["agent_id"] != "unused"]
            team["synthesizer_id"] = team["agents"][-1]["agent_id"]
            team["total_max_calls"] = len(team["agents"])
            team["coverage"] = {rubric_id: [identity for identity in owners if identity != "unused"]
                                for rubric_id, owners in team["coverage"].items()}
            if "budget_plan" in team:
                team["budget_plan"]["agents"] = [estimate for estimate in team["budget_plan"]["agents"]
                                                  if estimate["agent_id"] != "unused"]
        if phase == "evolution_integrate":
            response.update(proposal_id=None, agent_update_ids=[], pool_operations=[])
            kind = self.provider.operation
            if kind == "add":
                profile = payload["retention_candidates"][0]
                operation = self.operation(payload, kind, profiles=[profile])
                operation["evidence"] = sorted({"evaluation:summary", *profile["evidence"],
                                               *(evidence for lesson in profile["memory"]
                                                 for evidence in lesson["evidence"] + lesson["counterevidence"])})
                response["pool_operations"] = [operation]
            elif kind == "prune":
                response["pool_operations"] = [self.operation(payload, kind, targets=["planner"])]
            elif kind == "merge":
                profile = temporary_profile("design_generalist")
                profile.update(parent_agent_id=None, role="Design Generalist", capabilities=["planning", "analysis"],
                               source_task_ids=[payload["source_task_id"]], evidence=["evaluation:summary"])
                response["pool_operations"] = [self.operation(payload, kind, targets=["planner", "generalist"],
                                                               profiles=[profile])]
            elif kind == "merge_selected":
                response["proposal_id"] = payload["candidate_proposals"][0]["proposal_id"]
                selected = [update for update in payload["agent_reflections"]
                            if update["pool_agent_id"] in {"analyst", "searcher"}]
                response["agent_update_ids"] = [update["update_id"] for update in selected]
                evidence = sorted({"evaluation:summary", *(identity for update in selected for identity in update["evidence"])})
                profile = temporary_profile("research_integrator")
                profile.update(parent_agent_id=None, role="Research Integrator", capabilities=["analysis", "research"],
                               skills={}, source_task_ids=[payload["source_task_id"]], evidence=evidence)
                operation = self.operation(payload, "merge", targets=["analyst", "searcher"], profiles=[profile])
                operation["evidence"] = evidence
                response["pool_operations"] = [operation]
            elif kind == "invalid_evidence":
                operation = self.operation(payload, "prune", targets=["planner"])
                operation["evidence"] = ["invented:history"]
                response["pool_operations"] = [operation]
            elif kind == "direct_temporary":
                response["agent_update_ids"] = [update["update_id"] for update in payload["agent_reflections"]
                                                if update["pool_agent_id"] == "queue_specialist"]
            elif kind == "retain_unused":
                profile = copy.deepcopy(next(row["agent"]["temporary_profile"] for row in payload["temporary_agents"]
                                             if not row["selected"]))
                profile.update(source_task_ids=[payload["source_task_id"]], evidence=["evaluation:summary"])
                response["pool_operations"] = [self.operation(payload, "add", profiles=[profile])]
        return response

    @staticmethod
    def operation(payload, kind, *, targets=(), profiles=()):
        members = {profile["pool_agent_id"]: profile for profile in payload["agent_pool"]["profiles"]}
        references = set(targets) | {profile["parent_agent_id"] for profile in profiles
                                     if profile["parent_agent_id"] is not None}
        return {"operation_id": payload["source_task_id"] + "-" + kind, "kind": kind,
                "source_task_id": payload["source_task_id"], "base_pool_version": payload["base_pool_version"],
                "target_agent_ids": list(targets), "profiles": list(profiles), "parent_assignments": {},
                "base_agent_versions": {identity: members[identity]["version"] for identity in references},
                "evidence": ["evaluation:summary"], "rationale": "Inspect observed task needs and role overhead.",
                "expected_benefit": "Retain useful coverage while reducing redundant role selection.",
                "token_cost_tradeoff": "Use actual role token usage; no measured causal gain is claimed."}


class DynamicFixtureModels(FixtureModels):
    def __init__(self, store, *, temporary=True, operation="add"):
        super().__init__()
        self.store, self.temporary, self.operation = store, temporary, operation
        self.observed_state_hashes = []
        self.store_thread_id = threading.get_ident()

    def create(self, role, agent_id, ledger, stage):
        if threading.get_ident() == self.store_thread_id:
            self.observed_state_hashes.append(digest(self.store.snapshot()))
        if role not in {"global", "local"}:
            return super().create(role, agent_id, ledger, stage)
        return MeteredModel(DynamicFixtureModel(self, role, agent_id), ledger, stage, agent_id, 8192)


@pytest.fixture
def dynamic_pipeline_factory(tmp_path):
    stores = []

    def create(**choices):
        store = ExperienceStore(tmp_path / f"state-{len(stores)}.sqlite")
        stores.append(store)
        models = DynamicFixtureModels(store, **choices)
        pipeline = make_pipeline(MASConfig(backend="scripted"), store,
                                 tmp_path / f"runs-{len(stores)}", fixture_models=models)
        return pipeline, store, models

    yield create
    for store in stores:
        store.close()


def artifact(outcome, name):
    return json.loads((Path(outcome["run_dir"]) / name).read_text(encoding="utf-8"))


def phases(models):
    return [json.loads(call["messages"][-1]["content"]) for call in models.calls
            if call["role"] in {"global", "local"}]


def profile_by_id(pool, identity):
    return next(profile for profile in pool.profiles if profile.pool_agent_id == identity)


@pytest.mark.parametrize("mode", ["construct", "evaluate"])
def test_temporary_execution_records_creation_without_mutating_pool(dynamic_pipeline_factory, mode):
    pipeline, store, models = dynamic_pipeline_factory()
    baseline = digest(store.snapshot())
    if mode == "construct":
        task_id = pipeline.manifest.evolution[0]
        outcome = pipeline.run_task(task_id, store.snapshot(), mode="evolve", attribution=False, resume=False)
    else:
        outcome = pipeline.run("evaluate", limit=1)[0]
    assert digest(store.snapshot()) == baseline
    assert set(models.observed_state_hashes) == {baseline}
    creations = artifact(outcome, "temporary_agents.json")["creations"]
    assert {row["agent"]["pool_agent_id"]: row["selected"] for row in creations} == {
        "queue_specialist": True, "unused_specialist": False}
    assert artifact(outcome, "execution.json")["metadata"]["temporary_agent_ids"] == ["analyst"]
    executed = [json.loads(call["messages"][1]["content"])["persistent_agent"]
                for call in models.calls if call["role"] == "exec" and call["agent_id"] == "analyst"]
    assert executed[0]["temporary"] and executed[0]["skills"] == temporary_profile()["skills"]
    assert not outcome["pool_operations"] and not outcome["pool_observations"]
    assert not any(payload["phase"] == "agent_evolve" for payload in phases(models))
    assert not (Path(outcome["run_dir"]) / "meta_evolution.json").exists()


@pytest.mark.parametrize("retain", [True, False])
def test_evolution_meta_decides_whether_to_retain_executed_temporary_harness(dynamic_pipeline_factory, retain):
    pipeline, store, models = dynamic_pipeline_factory(operation="add" if retain else "none")
    baseline = digest(store.snapshot())
    outcome = pipeline.run("evolve")[0]
    assert set(models.observed_state_hashes) == {baseline}
    pool = store.snapshot().agent_pool
    assert pool.version == store.snapshot().version == 1
    assert len(pool.observations) == 3 and len(outcome["pool_observations"]) == 3
    observation = next(item for item in pool.observations if item.pool_agent_id == "queue_specialist")
    assert observation.temporary and observation.input_tokens > 0 and observation.output_tokens > 0
    assert "unused_specialist" not in {profile.pool_agent_id for profile in pool.profiles}
    assert ("queue_specialist" in {profile.pool_agent_id for profile in pool.profiles}) == retain
    integration = next(payload for payload in phases(models) if payload["phase"] == "evolution_integrate")
    assert len(integration["agent_reflections"]) == 3
    assert len(integration["retention_candidates"]) == 1
    assert not outcome["agent_updates"] and not outcome["proposals"]
    if retain:
        retained = profile_by_id(pool, "queue_specialist")
        candidate = integration["retention_candidates"][0]
        assert retained.version == 1 and retained.parent_agent_id == "generalist"
        assert retained.memory and retained.source_task_ids == [outcome["task_id"]]
        assert all(retained.skills[key] == instruction for key, instruction in candidate["skills"].items())
        assert retained.harness.model_dump(mode="json") == candidate["harness"]
        assert len(outcome["agent_pool_operations"]) == 1 and pool.structural_history[0].kind == "add"
    else:
        assert pool.structural_history == [] and not outcome["pool_operations"]
    calls = len(models.calls)
    frozen_hash = digest(store.snapshot())
    assert pipeline.run("evolve")[0]["resumed"]
    assert len(models.calls) == calls and digest(store.snapshot()) == frozen_hash
    if retain:
        held_out = pipeline.run("evaluate", limit=1)[0]
        assert digest(store.snapshot()) == frozen_hash
        assert not (Path(held_out["run_dir"]) / "temporary_agents.json").exists()
        selected = next(agent for agent in artifact(held_out, "frozen_plan.json")["TeamSpec"]["agents"]
                        if agent["agent_id"] == "analyst")
        assert selected["pool_agent_id"] == "queue_specialist" and selected["pool_agent_version"] == 1
        assert selected["temporary_profile"] is None
        execution = [json.loads(call["messages"][1]["content"])["persistent_agent"]
                     for call in models.calls if call["role"] == "exec" and call["agent_id"] == "analyst"][-1]
        assert not execution["temporary"] and execution["memory"]


@pytest.mark.parametrize("kind", ["prune", "merge"])
def test_structure_only_decisions_commit_after_feedback_and_leave_frozen_execution_unchanged(dynamic_pipeline_factory, kind):
    pipeline, store, models = dynamic_pipeline_factory(temporary=False, operation=kind)
    outcome = pipeline.run("evolve")[0]
    assert outcome["proposals"] == outcome["agent_updates"] == outcome["experience_updates"] == []
    assert outcome["pool_operations"][0]["kind"] == kind
    pool = store.snapshot().agent_pool
    assert pool.structural_history[0].kind == kind and len(pool.observations) == 3
    retained_ids = {profile.pool_agent_id for profile in pool.profiles}
    assert "planner" not in retained_ids
    if kind == "merge":
        assert "generalist" not in retained_ids and "design_generalist" in retained_ids
    frozen = artifact(outcome, "frozen_plan.json")
    assert "planner" in {profile["pool_agent_id"] for profile in frozen["AgentPool"]["profiles"]}
    assert {agent["pool_agent_id"] for agent in frozen["TeamSpec"]["agents"]} == {"analyst", "searcher", "writer"}
    payloads = phases(models)
    assert max(index for index, payload in enumerate(payloads) if payload["phase"] == "agent_evolve") < next(
        index for index, payload in enumerate(payloads) if payload["phase"] == "evolution_integrate")


def test_joint_meta_local_learning_and_merge_commit_one_complete_snapshot(dynamic_pipeline_factory):
    pipeline, store, models = dynamic_pipeline_factory(temporary=False, operation="merge_selected")
    baseline = digest(store.snapshot())
    outcome = pipeline.run("evolve")[0]
    snapshot = store.snapshot()
    assert set(models.observed_state_hashes) == {baseline}
    assert snapshot.version == snapshot.agent_pool.version == 1
    assert len(snapshot.experiences) == 1 and snapshot.experiences[0].bank == "rubric"
    assert snapshot.applied_proposals == [outcome["proposals"][0]["proposal_id"]]
    assert {update["pool_agent_id"] for update in outcome["agent_updates"]} == {"analyst", "searcher"}
    members = {profile.pool_agent_id for profile in snapshot.agent_pool.profiles}
    assert {"analyst", "searcher"}.isdisjoint(members) and "research_integrator" in members
    merged = profile_by_id(snapshot.agent_pool, "research_integrator")
    lessons = {lesson["lesson_id"] for update in outcome["agent_updates"] for lesson in update["lessons"]}
    assert len(lessons) == 2 and {lesson.lesson_id for lesson in merged.memory} == lessons
    assert {"tradeoff_analysis", "set_reasoning", "evidence_selection", "coverage_search"} <= set(merged.skills)
    assert merged.version == 1 and merged.source_task_ids == [outcome["task_id"]]
    assert len(snapshot.agent_pool.structural_history) == len(outcome["pool_operations"]) == 1
    assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 1
    frozen = artifact(outcome, "frozen_plan.json")
    assert {agent["pool_agent_id"] for agent in frozen["TeamSpec"]["agents"]} == {"analyst", "searcher", "writer"}
    assert {profile["pool_agent_id"] for profile in frozen["AgentPool"]["profiles"]} >= {"analyst", "searcher"}
    assert "research_integrator" not in {profile["pool_agent_id"] for profile in frozen["AgentPool"]["profiles"]}


@pytest.mark.parametrize("operation", ["invalid_evidence", "direct_temporary", "retain_unused"])
def test_invalid_pool_decision_never_partially_commits(dynamic_pipeline_factory, operation):
    pipeline, store, models = dynamic_pipeline_factory(operation=operation)
    baseline = digest(store.snapshot())
    with pytest.raises(ValueError):
        pipeline.run("evolve")
    assert digest(store.snapshot()) == baseline
    assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 0
    assert any(payload["phase"] == "agent_evolve" for payload in phases(models))


def submitted_and_attributed(dynamic_pipeline_factory):
    source_pipeline, source_store, _ = dynamic_pipeline_factory()
    source = source_pipeline.run_task(source_pipeline.manifest.evolution[0], source_store.snapshot(),
                                     mode="evolve", attribution=False, resume=False)
    source_anchor = submitted_source_digest(source["run_dir"])
    continued, continued_store, continued_models = dynamic_pipeline_factory()
    attributed = continued.run("evolve", resume_source=source["run_dir"], resume_source_hash=source_anchor)[0]
    assert {call["role"] for call in continued_models.calls} <= {"global", "local"}
    assert (Path(attributed["run_dir"]) / "temporary_agents.json").read_bytes() == (
        Path(source["run_dir"]) / "temporary_agents.json").read_bytes()
    return source, source_anchor, attributed, continued_store


def test_temporary_retention_and_structure_replay_without_any_model_calls(dynamic_pipeline_factory, monkeypatch):
    source, source_anchor, attributed, continued_store = submitted_and_attributed(dynamic_pipeline_factory)
    attributed_anchor = attribution_source_digest(attributed["run_dir"])
    replay, replay_store, models = dynamic_pipeline_factory(operation="invalid_evidence")

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen dynamic Pool replay must not construct models")

    monkeypatch.setattr(models, "create", forbidden)
    outcome = replay.run("evolve", resume_source=source["run_dir"], resume_source_hash=source_anchor,
                         resume_attribution=attributed["run_dir"], resume_attribution_hash=attributed_anchor)[0]
    assert outcome["source_attribution_reused"] and outcome["budget"]["model_calls"] == 0
    assert not models.calls and digest(replay_store.snapshot()) == digest(continued_store.snapshot())
    assert outcome["pool_operations"] == attributed["pool_operations"]
    assert outcome["pool_observations"] == attributed["pool_observations"]
    assert outcome["agent_pool_operations"] == attributed["agent_pool_operations"]


@pytest.mark.parametrize("mutation", ["decision_operations", "observation_tokens"])
def test_frozen_dynamic_decision_rejects_tampering_before_model_calls(dynamic_pipeline_factory, monkeypatch, mutation):
    source, source_anchor, attributed, _ = submitted_and_attributed(dynamic_pipeline_factory)
    path = Path(attributed["run_dir"]) / "meta_evolution.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "decision_operations":
        document["decision"]["pool_operations"] = []
    else:
        document["pool_observations"][0]["input_tokens"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    attributed_anchor = attribution_source_digest(attributed["run_dir"])
    replay, store, models = dynamic_pipeline_factory()
    baseline = digest(store.snapshot())

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen decision validation precedes model construction")

    monkeypatch.setattr(models, "create", forbidden)
    with pytest.raises(ValueError):
        replay.run("evolve", resume_source=source["run_dir"], resume_source_hash=source_anchor,
                   resume_attribution=attributed["run_dir"], resume_attribution_hash=attributed_anchor)
    assert not models.calls and digest(store.snapshot()) == baseline


def test_submitted_temporary_creation_tamper_is_included_in_source_hash(dynamic_pipeline_factory, monkeypatch):
    source_pipeline, source_store, _ = dynamic_pipeline_factory(temporary=True, operation="add")
    source = source_pipeline.run_task(source_pipeline.manifest.evolution[0], source_store.snapshot(),
                                     mode="evolve", attribution=False, resume=False)
    anchor = submitted_source_digest(source["run_dir"])
    path = Path(source["run_dir"]) / "temporary_agents.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["creations"][0]["selected"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    replay, store, models = dynamic_pipeline_factory(temporary=False, operation="prune")
    baseline = digest(store.snapshot())

    def forbidden(*args, **kwargs):
        raise AssertionError("Source integrity validation precedes model construction")

    monkeypatch.setattr(models, "create", forbidden)
    with pytest.raises(ValueError, match="Submitted-source content hash mismatch"):
        replay.run("evolve", resume_source=source["run_dir"], resume_source_hash=anchor)
    assert not models.calls and digest(store.snapshot()) == baseline
