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
        frozen_team = read_artifact(first, "frozen_plan.json")["TeamSpec"]
        assert {row["agent_id"] for row in frozen_team["budget_plan"]["agents"]} == {"analyst", "evidence", "writer"}
        planning_calls = read_artifact(first, "planning_calls.json")
        initial_budget = json.loads(planning_calls[0]["messages"][1]["content"])["limits"]["resource_budget"]
        final_budget = json.loads(planning_calls[-1]["messages"][1]["content"])["limits"]["resource_budget"]
        assert final_budget["remaining_tokens"] < initial_budget["remaining_tokens"]
        reflection_calls = read_artifact(first, "agent_evolution.json")["calls"]
        for reflection in reflection_calls:
            payload = json.loads(reflection["messages"][1]["content"])
            assert payload["resource_usage"][0]["input_tokens"] > 0
            assert payload["resource_usage"][0]["cost"] is None
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


def test_general_knowledge_policy_preserves_evolution_and_task_identity(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    models = FixtureModels()
    policy = "model_general_knowledge_allowed"
    pipeline = make_pipeline(MASConfig(backend="scripted"),
                             store, tmp_path / "runs", fixture_models=models, knowledge_policy=policy)
    try:
        outcome = pipeline.run("evolve")[0]
        assert store.snapshot().version == 1
        assert len(outcome["agent_pool_updates"]) == 3
        manifest = read_artifact(outcome, "run_manifest.json")
        assert manifest["comparison"]["knowledge_policy"] == policy
        execution = read_artifact(outcome, "execution.json")
        assert execution["metadata"]["knowledge_policy"] == policy
        assert execution["metadata"]["coordination"] == "single_pass_shared_ledger"
        team = read_artifact(outcome, "frozen_plan.json")["TeamSpec"]
        assert {agent["pool_agent_id"] for agent in team["agents"]} == {"analyst", "searcher", "writer"}
        assert all(agent["tools"] == [] for agent in team["agents"])
        calls = len(models.calls)
        assert pipeline.run("evolve")[0]["resumed"]
        assert len(models.calls) == calls
        pipeline.knowledge_policy = None
        with pytest.raises(ValueError, match="fresh store for changed policy"):
            pipeline.run("evolve")
        assert len(models.calls) == calls
    finally:
        store.close()


def test_general_knowledge_planning_adapts_research_without_mutating_persistent_pool():
    pool = seed_pool()
    researcher = next(profile for profile in pool.profiles if profile.pool_agent_id == "searcher")
    researcher.prompt = "Retain learned checks for conflicting assumptions."
    original_hash = digest(pool)
    candidates = [
        AgentSpec(agent_id="knowledge", role="Knowledge Researcher", capability="research",
                  pool_agent_id="searcher", pool_agent_version=1, max_calls=None,
                  checkpoints=["Separate remembered facts from uncertainty"]),
        AgentSpec(agent_id="writer", role="Writer", capability="writing",
                  pool_agent_id="writer", pool_agent_version=1, max_calls=None,
                  depends_on=["knowledge"]),
    ]
    calls = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        calls.append((messages[0]["content"], payload))
        if payload["phase"] == "predict":
            return Prediction(graph=RubricGraph(rubrics=[]), candidates=candidates).model_dump_json()
        if payload["phase"] == "local_plan":
            candidate = payload["candidate"]
            return LocalPlan(agent_id=candidate["agent_id"], capability=candidate["capability"],
                             depends_on=candidate["depends_on"], max_calls=None,
                             selected_skills=list(payload["agent_profile"]["skills"])).model_dump_json()
        return json.dumps({"graph": {"rubrics": []}, "team": {
            "execution_mode": "iterative_shared_ledger",
            "agents": payload["prediction"]["candidates"], "synthesizer_id": "writer",
            "total_max_calls": None}})

    analyzer = GlobalAnalyzer(model, agent_pool=pool, execution_mode="iterative_shared_ledger",
                              total_max_calls=None, knowledge_policy="model_general_knowledge_allowed")
    planned = analyzer.build(PublicTask(task_id="research", question="Explain current TTS evaluation."))
    assert {payload["phase"] for _, payload in calls} == {"predict", "local_plan", "reconcile"}
    for prompt, payload in calls:
        assert payload["limits"]["knowledge_policy"] == "model_general_knowledge_allowed"
        assert "no external retrieval or source-access tools" in prompt
        assert "Checkpoints must describe attainable checks" in prompt
        assert "Dynamic role selection" in prompt
        if "agent_pool_catalogue" in payload:
            member = next(item for item in payload["agent_pool_catalogue"] if item["pool_agent_id"] == "searcher")
            assert member["role"] == "Knowledge Researcher"
            assert "search" not in member["capabilities"]
            assert member["version"] == 1
        if payload["phase"] == "local_plan" and payload["candidate"]["agent_id"] == "knowledge":
            assert payload["agent_profile"]["prompt"].startswith("Organize relevant model general knowledge")
            assert researcher.prompt in payload["agent_profile"]["prompt"]
            assert payload["agent_profile"]["preferred_tools"] == []
    assert planned.team.agents[-1].depends_on == ["knowledge"]
    assert planned.team.execution_mode == "iterative_shared_ledger"
    assert digest(pool) == original_hash


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


@pytest.mark.parametrize("corrected", [True, False])
def test_pool_duplicate_correction_names_participations_without_rebinding(corrected):
    candidates = [
        AgentSpec(agent_id="analyst_1", role="Analyst", capability="analysis",
                  pool_agent_id="analyst", pool_agent_version=1),
        AgentSpec(agent_id="analyst_2", role="Analyst", capability="analysis",
                  pool_agent_id="analyst", pool_agent_version=1),
        AgentSpec(agent_id="writer_1", role="Writer", capability="writing",
                  pool_agent_id="writer", pool_agent_version=1),
    ]
    response = Prediction(graph=RubricGraph(rubrics=[]), candidates=candidates).model_dump(mode="json")
    original = json.loads(json.dumps(response))
    requests = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        requests.append(payload)
        if len(requests) == 2:
            correction = payload["response_correction"]
            assert json.loads(correction["previous_response"]) == original
            message = correction["validation_errors"][0]["message"]
            assert 'duplicate_pool_bindings={"analyst": ["analyst_1", "analyst_2"]}' in message
            assert "Changing agent_id does not create another pool member" in message
            assert "Reconciliation must retain" in message
            if corrected:
                return json.dumps({**response, "candidates": [response["candidates"][0],
                                                             response["candidates"][2]]})
        return json.dumps(response)

    analyzer = GlobalAnalyzer(model, agent_pool=seed_pool())
    task = PublicTask(task_id="new", question="Explain a technical comparison")
    if corrected:
        prediction = analyzer.predict(task)
        assert [(agent.agent_id, agent.pool_agent_id) for agent in prediction.candidates] == [
            ("analyst_1", "analyst"), ("writer_1", "writer")]
    else:
        with pytest.raises(ValueError, match="duplicate_pool_bindings"):
            analyzer.predict(task)
    assert len(requests) == len(analyzer.call_records) == 2
    assert response == original
    assert json.loads(analyzer.call_records[0]["response"]) == original


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
