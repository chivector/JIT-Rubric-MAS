"""Joint meta/agent evolution and persistent role planning contracts."""

import json
from pathlib import Path

import pytest

from jit_mas.agent_pool import role_context, seed_pool
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentMemoryLesson, AgentSpec, LocalPlan, Prediction, PublicTask, RubricGraph, digest
from scripts.run_jit_mas import make_pipeline


def read_artifact(outcome, name):
    return json.loads((Path(outcome["run_dir"]) / name).read_text(encoding="utf-8"))


def test_joint_evolution_reuses_mature_agents_without_regenerating_harness(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    models = FixtureModels()
    pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs", fixture_models=models)
    try:
        first = pipeline.run("evolve")[0]
        snapshot = store.snapshot()
        assert snapshot.version == snapshot.agent_pool.version == 1
        assert snapshot.experiences and len(first["agent_pool_updates"]) == 3
        mature = {profile.pool_agent_id: profile for profile in snapshot.agent_pool.profiles}
        assert all(mature[member].version == 2 and mature[member].memory
                   for member in ("analyst", "searcher", "writer"))
        harness = read_artifact(first, "harness.json")
        assert harness["selection"]["strategy"] == "pooled_agent_reuse"
        assert harness["meta_trajectory"] == []
        assert read_artifact(first, "agent_evolution.json")["complete"]
        frozen_hash = digest(snapshot)
        following = pipeline.run("evaluate", limit=2)
        assert digest(store.snapshot()) == frozen_hash
        assert all(not row["agent_updates"] for row in following)
        next_team = read_artifact(following[0], "frozen_plan.json")["TeamSpec"]
        assert {agent["pool_agent_version"] for agent in next_team["agents"]} == {2}
        poem_team = read_artifact(following[1], "frozen_plan.json")["TeamSpec"]
        assert len(poem_team["agents"]) == 1 and poem_team["agents"][0]["pool_agent_id"] == "writer"
        writer_calls = [json.loads(call["messages"][1]["content"]) for call in models.calls
                        if call["role"] == "exec" and call["agent_id"] == "writer"]
        assert writer_calls[-1]["persistent_agent"]["memory"]
        calls = len(models.calls)
        assert pipeline.run("evolve")[0]["resumed"]
        assert len(models.calls) == calls and digest(store.snapshot()) == frozen_hash
    finally:
        store.close()


def test_pool_planning_exposes_catalogue_and_preserves_local_internal_choices():
    pool = seed_pool()
    candidate = AgentSpec(agent_id="task-writer", role="Writer", capability="writing",
                          pool_agent_id="writer", pool_agent_version=1)
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate])
    local = LocalPlan(agent_id=candidate.agent_id, capability=candidate.capability,
                      selected_skills=["synthesis"], reasoning_strategy="Check sources before drafting",
                      communication="Preserve explicit uncertainty")
    response = {"graph": {"rubrics": []}, "team": {
        "agents": [{**candidate.model_dump(mode="json"), "reasoning_strategy": "Meta replacement"}],
        "synthesizer_id": candidate.agent_id, "total_max_calls": 1}}
    calls = []
    def model(messages):
        calls.append(json.loads(messages[1]["content"]))
        return json.dumps(response)
    team = GlobalAnalyzer(model, agent_pool=pool).reconcile(
        PublicTask(task_id="new", question="Write a short explanation"), prediction, [local]).team
    assert team.agents[0].reasoning_strategy == local.reasoning_strategy
    assert team.agents[0].selected_skills == local.selected_skills
    assert all("memory" not in member and "prompt" not in member
               for member in calls[0]["agent_pool_catalogue"])


def test_pool_planning_rejects_identity_substitution():
    pool = seed_pool()
    candidate = AgentSpec(agent_id="a", role="Writer", capability="writing",
                          pool_agent_id="writer", pool_agent_version=1)
    response = {"graph": {"rubrics": []}, "team": {
        "agents": [{**candidate.model_dump(mode="json"), "pool_agent_id": "critic"}],
        "synthesizer_id": "a", "total_max_calls": 1}}
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response), agent_pool=pool)
    with pytest.raises(ValueError, match="cannot invent or replace"):
        analyzer.reconcile(PublicTask(task_id="new", question="Write"),
                           Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate]), [])


@pytest.mark.parametrize("selected_skills, expected", [(None, {"synthesis"}), ([], set())])
def test_local_skill_selection_preserves_explicit_empty_choice(selected_skills, expected):
    pool = seed_pool()
    candidate = AgentSpec(agent_id="writer", role="Writer", capability="writing",
                          pool_agent_id="writer", pool_agent_version=1)
    prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate])
    response = {"graph": {"rubrics": []}, "team": {
        "agents": [candidate.model_dump(mode="json")], "synthesizer_id": "writer", "total_max_calls": 1}}
    plan = LocalPlan(agent_id="writer", capability="writing", selected_skills=selected_skills)
    task = PublicTask(task_id="new", question="Write a short explanation")
    team = GlobalAnalyzer(lambda _: json.dumps(response), agent_pool=pool).reconcile(
        task, prediction, [plan]).team
    assert set(role_context(pool, team.agents[0], task)["skills"]) == expected


def test_local_pool_memory_is_scoped_before_planning():
    pool = seed_pool()
    profile = next(member for member in pool.profiles if member.pool_agent_id == "analyst")
    for identifier, source, applicability in (("current", "new", "comparison"),
                                              ("heldout", "test", "comparison"),
                                              ("other", "prior", "translation"),
                                              ("useful", "prior", "comparison")):
        profile.memory.append(AgentMemoryLesson(lesson_id=identifier, instruction="Conditional process advice",
            applicability=applicability, capability="comparison", source_task_ids=[source], evidence=["event"]))
    candidate = AgentSpec(agent_id="a", role="Analyst", capability="comparison",
                          pool_agent_id="analyst", pool_agent_version=1)
    payloads = []
    def model(messages):
        payloads.append(json.loads(messages[1]["content"]))
        return json.dumps({"agent_id": "a", "capability": "comparison"})
    analyzer = GlobalAnalyzer(model, agent_pool=pool, excluded_task_ids=["test"])
    analyzer.local_plan(PublicTask(task_id="new", question="Compare two designs"),
                        Prediction(graph=RubricGraph(rubrics=[]), candidates=[candidate]), candidate)
    assert [lesson["lesson_id"] for lesson in payloads[0]["agent_profile"]["memory"]] == ["useful"]


def test_invalid_self_reflection_never_commits_partial_joint_state(tmp_path):
    class InvalidReflection(FixtureModels):
        def create(self, role, agent_id, ledger, stage):
            model = super().create(role, agent_id, ledger, stage)
            if role != "local" or stage != "update":
                return model
            class CorruptResponse:
                def __call__(self, messages):
                    response = model(messages)
                    if json.loads(messages[1]["content"])["phase"] == "agent_evolve":
                        data = json.loads(response.content)
                        data["lessons"][0]["source_task_ids"].append("test-deployment")
                        response.content = json.dumps(data)
                    return response
            return CorruptResponse()
    store = ExperienceStore(tmp_path / "state.sqlite")
    pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs",
                             fixture_models=InvalidReflection())
    try:
        with pytest.raises(ValueError, match="invented source tasks"):
            pipeline.run("evolve")
        assert store.snapshot().version == 0 and not store.snapshot().agent_pool.profiles
        failed = list((tmp_path / "runs").glob("*/agent_evolution.json"))
        assert len(failed) == 1 and not json.loads(failed[0].read_text())["complete"]
    finally:
        store.close()
