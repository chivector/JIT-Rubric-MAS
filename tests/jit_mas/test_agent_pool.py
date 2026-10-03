"""Focused invariants for the persistent, reusable Agent Pool."""

import json
import shutil

import pytest

from jit_mas.agent_pool import (
    apply_updates,
    get_profile,
    role_context,
    seed_pool,
    validate_bindings,
)
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.execution import (
    TeamExecutor,
    _role_messages,
    _role_tools,
)
from jit_mas.schemas import (
    AgentEvolutionUpdate,
    AgentHarnessPolicy,
    AgentMemoryLesson,
    AgentSpec,
    PublicTask,
    RubricGraph,
    TeamSpec,
)
from jit_mas.offline import FixtureModel, FixtureModels
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


def pooled_team(*, selected_skills=None, task_prompt=""):
    return TeamSpec(
        agents=[AgentSpec(
            agent_id="writer",
            role="Writer",
            capability="writing",
            pool_agent_id="writer",
            pool_agent_version=1,
            selected_skills=selected_skills or [],
            task_prompt=task_prompt,
            max_calls=1,
        )],
        synthesizer_id="writer",
        total_max_calls=1,
    )


def lesson(lesson_id, *, source="earlier-task", signals=None, capability="writing"):
    return AgentMemoryLesson(
        lesson_id=lesson_id,
        instruction="Preserve explicit uncertainty in the contribution.",
        applicability="technical writing tasks",
        capability=capability,
        task_signals=signals or ["technical writing"],
        source_task_ids=[source],
        evidence=["run-1:event-1"],
        created_at="2026-09-01T00:00:00+00:00",
    )


def test_role_context_is_scoped_and_does_not_mutate_profile():
    pool = seed_pool()
    writer = get_profile(pool, "writer", 1)
    writer.memory = [
        lesson("eligible", signals=["technical writing"]),
        lesson("current", source="current-task", signals=["technical writing"]),
        lesson("wrong-scope", signals=["numerical verification"], capability="verification"),
    ]
    pool.profiles[0] = writer
    task = PublicTask(task_id="current-task", question="Write a technical guide.")
    agent = pooled_team().agents[0]
    before = pool.model_copy(deep=True)
    context = role_context(pool, agent, task)
    assert [item["lesson_id"] for item in context["memory"]] == ["eligible"]
    context["skills"]["synthesis"] = "task override"
    assert pool == before


def test_pool_updates_preserve_identity_and_are_pure():
    pool = seed_pool()
    update = AgentEvolutionUpdate(
        update_id="update-1", pool_agent_id="writer", base_agent_version=1,
        source_task_id="evolution-task", lessons=[lesson("lesson-1", source="evolution-task")],
        communication="Publish concise drafts.", evidence=["run-1:event-1"],
    )
    updated = apply_updates(pool, [update], source_task_id="evolution-task")
    assert get_profile(pool, "writer").version == 1
    assert get_profile(updated, "writer").version == 2
    assert updated.version == 1 and updated.applied_updates == ["update-1"]
    with pytest.raises(ValueError, match="source task"):
        apply_updates(pool, [update], source_task_id="other-task")


def test_seed_pool_starts_unmature_and_bindings_require_exact_version():
    pool = seed_pool()
    assert all(not profile.source_task_ids and not profile.evidence for profile in pool.profiles)
    validate_bindings(pool, pooled_team())
    with pytest.raises(ValueError, match="Stale"):
        validate_bindings(pool, pooled_team().__class__(
            agents=[pooled_team().agents[0].model_copy(update={"pool_agent_version": 2})],
            synthesizer_id="writer", total_max_calls=1))
    with pytest.raises(ValueError, match="exact member identity"):
        validate_bindings(pool, TeamSpec(
            agents=[AgentSpec(agent_id="writer", role="Writer", capability="writing")],
            synthesizer_id="writer", total_max_calls=1))


def test_harness_reuse_skips_meta_generation_and_freezes_pool_sidecar():
    pool = seed_pool()
    task = PublicTask(task_id="pool-task", question="Write a concise technical guide.")
    model = type("Meta", (), {"calls": []})()
    synth = JITHarnessSynthesizer(meta_model=model)
    artifact = synth.synthesize(task, RubricGraph(rubrics=[]), pooled_team(
        selected_skills=["synthesis"], task_prompt="Emphasize limitations."), agent_pool=pool)
    try:
        assert model.calls == []
        assert artifact.selection["strategy"] == "pooled_agent_reuse"
        assert artifact.sidecar["agent_pool"]["profiles"][0]["prompt"]
        assert artifact.sidecar["team"]["agents"][0]["task_prompt"] == "Emphasize limitations."
        sidecar = artifact.sidecar
        sidecar["agent_pool"]["profiles"][0]["version"] = 99
        (artifact.path / "team.json").write_text(json.dumps(sidecar), encoding="utf-8")
        with pytest.raises(ValueError, match="sidecar changed"):
            artifact.verify_integrity()
    finally:
        for generated in synth._agents.values():
            path = generated.workspace_dir.resolve()
            if path.is_dir():
                shutil.rmtree(path)


def test_runtime_receives_frozen_profile_prompt_skills_and_task_adaptation():
    pool = seed_pool()
    task = PublicTask(task_id="runtime-pool-task", question="Write a concise guide.")
    team = pooled_team(selected_skills=["synthesis"], task_prompt="Mention open questions.")
    synth = JITHarnessSynthesizer()
    artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team, agent_pool=pool)
    retained_prompt = get_profile(pool, "writer").prompt
    pool.profiles[0].prompt = "A later profile revision."
    calls = []

    class Model:
        def __call__(self, messages, **kwargs):
            calls.append(messages)
            return ChatMessage(role="assistant", content=json.dumps({
                "answer": "A guide.", "evidence_ids": [], "checkpoints": {}}))

        def get_token_counts(self):
            return {"input_token_count": 1, "output_token_count": 1}

    try:
        result = TeamExecutor(lambda _aid: Model(), ledger=BudgetLedger()).execute(
            task, team, artifact)
        assert result.terminated_reason == "final_answer"
        payload = json.loads(calls[0][1]["content"])
        assert payload["persistent_agent"]["pool_agent_id"] == "writer"
        assert payload["persistent_agent"]["skills"]["synthesis"]
        assert payload["persistent_agent"]["prompt"] == retained_prompt
        assert "Mention open questions." in calls[0][0]["content"]
    finally:
        for generated in synth._agents.values():
            path = generated.workspace_dir.resolve()
            if path.is_dir():
                shutil.rmtree(path)


def test_sidecar_pool_binding_tampering_is_rejected_before_model_calls():
    pool = seed_pool()
    task = PublicTask(task_id="tamper-task", question="Write a guide.")
    team = pooled_team()
    synth = JITHarnessSynthesizer()
    artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team, agent_pool=pool)
    artifact.sidecar_hash = "tampered"
    try:
        with pytest.raises(ValueError, match="sidecar changed"):
            TeamExecutor(lambda _aid: pytest.fail("model must not run")).execute(task, team, artifact)
    finally:
        for generated in synth._agents.values():
            path = generated.workspace_dir.resolve()
            if path.is_dir():
                shutil.rmtree(path)


def test_harness_policy_limits_memory_and_orders_preferred_tools():
    persistent = {
        "harness": {"memory_policy": "recent", "memory_window": 1,
                     "tool_policy": "preferred_first"},
        "preferred_tools": ["search"],
    }
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "instruction"},
                {"role": "assistant", "content": "old"}, {"role": "user", "content": "new"}]
    assert [item["content"] for item in _role_messages(messages, persistent)] == ["system", "instruction", "new"]
    catalog = {name: name for name in ("search", "browse", "calc")}
    assert list(_role_tools({"calc", "search"}, catalog, persistent)) == ["search", "calc"]


def test_update_rejects_unprovenanced_lessons_and_does_not_mutate_on_failure():
    pool = seed_pool()
    update = AgentEvolutionUpdate(
        update_id="bad-update", pool_agent_id="writer", base_agent_version=1,
        source_task_id="evolution-task", lessons=[lesson("bad", source="other-task")],
        evidence=["run-1:event-1"],
    )
    before = pool.model_copy(deep=True)
    with pytest.raises(ValueError, match="provenance"):
        apply_updates(pool, [update], source_task_id="evolution-task")
    assert pool == before


def test_next_task_executes_agent_owned_internal_evolution(tmp_path):
    retained_prompt = "Build a coherent deliverable; check uncertainty before publishing it."
    retained_skill = "Read the draft against each public constraint before publishing."
    retained_reasoning = "Separate evidence from assumptions before selecting conclusions."
    retained_planning = "Plan the outline before drafting and verify the completed outline."
    retained_communication = "Publish compact handoffs with explicit unresolved uncertainties."

    class EvolvingFixtureModel(FixtureModel):
        def _phase(self, payload):
            response = super()._phase(payload)
            if payload["phase"] == "predict":
                for candidate in response["candidates"]:
                    candidate["task_prompt"] = "Current assignment: " + payload["task"]["question"]
            if payload["phase"] == "agent_evolve" and payload["agent_profile"]["pool_agent_id"] == "writer":
                response.update(
                    prompt=retained_prompt,
                    skills={**payload["agent_profile"]["skills"], "draft_check": retained_skill},
                    preferred_tools=["web_search"],
                    reasoning_strategy=retained_reasoning,
                    planning_strategy=retained_planning,
                    communication=retained_communication,
                    harness={"harness_id": "rubric_mas", "memory_policy": "recent",
                             "memory_window": 1, "tool_policy": "preferred_first"},
                )
            return response

    class EvolvingFixtureModels(FixtureModels):
        def create(self, role, agent_id, ledger, stage):
            if role not in {"global", "local"}:
                return super().create(role, agent_id, ledger, stage)
            return MeteredModel(EvolvingFixtureModel(self, role, agent_id), ledger, stage, agent_id, 8192)

    store = ExperienceStore(tmp_path / "state.sqlite")
    models = EvolvingFixtureModels()
    pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs", fixture_models=models)
    try:
        pipeline.run("evolve")
        writer = get_profile(store.snapshot().agent_pool, "writer", 2)
        assert writer.prompt == retained_prompt and writer.skills["draft_check"] == retained_skill
        assert writer.harness.memory_policy == "recent" and writer.harness.tool_policy == "preferred_first"
        pipeline.run("evaluate", limit=1)
        call = next(call for call in reversed(models.calls)
                    if call["role"] == "exec" and call["agent_id"] == "writer")
        payload = json.loads(call["messages"][1]["content"])
        profile = payload["persistent_agent"]
        assert profile["version"] == 2 and profile["prompt"] == retained_prompt
        assert profile["skills"]["draft_check"] == retained_skill
        assert profile["reasoning_strategy"] == retained_reasoning
        assert profile["planning_strategy"] == retained_planning
        assert profile["communication"] == retained_communication
        assert profile["harness"]["memory_window"] == 1
        assert profile["harness"]["tool_policy"] == "preferred_first"
        assert profile["preferred_tools"] == ["web_search"]
        task_prompt = payload["agent"]["task_prompt"]
        assert task_prompt.startswith("Domain scope and checks from the frozen plan:\nCurrent assignment: "
                                      + payload["public_task"]["question"])
        assert "Authoritative terminal assignment:" in task_prompt
        assert "deployment" in payload["agent"]["task_prompt"]
        assert retained_prompt in call["messages"][0]["content"]
        assert payload["public_task"]["question"] in payload["agent"]["task_prompt"]
    finally:
        store.close()


def test_pooled_iterative_harness_runs_two_turns_with_retained_policy():
    pool = seed_pool()
    writer = get_profile(pool, "writer", 1)
    writer.harness = AgentHarnessPolicy(memory_policy="recent", memory_window=1,
                                        tool_policy="preferred_first")
    pool.profiles[0] = writer
    task = PublicTask(task_id="iterative-pool-task", question="Write and revise a concise guide.")
    team = TeamSpec(execution_mode="iterative_shared_ledger", agents=[AgentSpec(
        agent_id="writer", role="Writer", capability="writing", pool_agent_id="writer",
        pool_agent_version=1, max_calls=None)], synthesizer_id="writer", total_max_calls=None)
    synth = JITHarnessSynthesizer()
    artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team, agent_pool=pool)
    calls = []

    class Model:
        def __call__(self, messages, **kwargs):
            calls.append(messages)
            return ChatMessage(role="assistant", content=json.dumps({
                "answer": "Initial draft." if len(calls) == 1 else "Revised complete guide.",
                "continue": len(calls) == 1, "evidence_ids": [], "checkpoints": {}}))

        def get_token_counts(self):
            return {"input_token_count": 2, "output_token_count": 2}

    ledger = BudgetLedger()
    try:
        result = TeamExecutor(lambda _aid: MeteredModel(Model(), ledger, "execution", "writer"),
                              ledger=ledger).execute(task, team, artifact)
        assert result.terminated_reason == "final_answer" and result.answer == "Revised complete guide."
        assert len(calls) == ledger.snapshot()["model_calls"] == 2
        assert len(result.sub_runs[0].trajectory) == 2
        assert calls[1][-2]["role"] == "assistant" and "Initial draft." in calls[1][-2]["content"]
        assert calls[1][-1]["role"] == "user" and "substantive contribution" in calls[1][-1]["content"]
        assert len(calls[1]) == 4
        assert json.loads(calls[1][1]["content"])["persistent_agent"]["harness"]["memory_policy"] == "recent"
        assert artifact.selection["strategy"] == "pooled_agent_reuse" and not artifact.meta_trajectory
    finally:
        for generated in synth._agents.values():
            path = generated.workspace_dir.resolve()
            if path.is_dir():
                shutil.rmtree(path)
