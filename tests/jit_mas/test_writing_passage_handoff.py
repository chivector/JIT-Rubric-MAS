"""Synthetic writing passages reach reviewers and final synthesis without extra calls."""

import json
import shutil
from pathlib import Path

import pytest

from jit_mas.agent_pool import seed_pool
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import TeamExecutor
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
def test_writing_passage_guidance_reaches_all_planning_requests(mode):
    agents = [
        AgentSpec(agent_id="composer", role="Composer", capability="creative-writing",
                  rubric_ids=["voice"], max_calls=1),
        AgentSpec(agent_id="reviewer", role="Reviewer", capability="writing review",
                  rubric_ids=["voice"], depends_on=["composer"], max_calls=1),
        AgentSpec(agent_id="writer", role="Writer", capability="writing synthesis",
                  rubric_ids=["voice"], depends_on=["reviewer"], max_calls=1),
    ]
    graph = RubricGraph(rubrics=[{"rubric_id": "voice", "requirement": "Write in past tense"}])
    prediction = Prediction(graph=graph, candidates=agents)
    team = TeamSpec(execution_mode=mode, agents=agents, synthesizer_id="writer",
                    total_max_calls=3, max_parallel=1,
                    coverage={"voice": [agent.agent_id for agent in agents]},
                    primary={"voice": "composer"}, reviewers={"voice": ["reviewer"]})
    prompts = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        prompts.append((payload["phase"], " ".join(messages[0]["content"].split())))
        if payload["phase"] == "predict":
            return prediction.model_dump_json()
        if payload["phase"] == "local_plan":
            candidate = payload["candidate"]
            return json.dumps({key: candidate[key] for key in
                               ("agent_id", "capability", "rubric_ids", "depends_on", "max_calls")})
        return json.dumps({"graph": graph.model_dump(mode="json"),
                           "team": team.model_dump(mode="json")})

    result = GlobalAnalyzer(model, execution_mode=mode, total_max_calls=3).build(
        PublicTask(task_id="synthetic-writing-plan", question="Write a fictional scene in past tense."))
    assert {phase for phase, _ in prompts} == {"predict", "local_plan", "reconcile"}
    for _, prompt in prompts:
        assert "exact passage, its defect and effect" in prompt
        assert "existing" in prompt and "ledger.outline" in prompt
        assert "Fiction may invent within the public premise" in prompt
        assert "actual real-world factual claims still need appropriate support" in prompt
        assert "do not claim the finished prose was inspected" in prompt
    assert [agent.agent_id for agent in result.team.agents] == [agent.agent_id for agent in agents]
    assert result.team.total_max_calls == 3
    assert all(agent.max_calls == 1 and not agent.tools for agent in result.team.agents)
    assert result.team.agents[-1].depends_on == ["reviewer"]


@pytest.fixture
def writing_executor():
    synthesizer = JITHarnessSynthesizer()
    yield synthesizer
    workspace_root = Path(__file__).resolve().parents[2] / "scripts" / "workspaces"
    for agent in synthesizer._agents.values():
        generated_path = agent.workspace_dir.resolve()
        assert generated_path.parent == workspace_root.resolve()
        assert generated_path.name.startswith("mas_")
        if generated_path.is_dir():
            shutil.rmtree(generated_path)


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
@pytest.mark.parametrize("question,passage,defect,revision,final_answer", [
    (
        "Write a fictional scene in past tense about a clockwork fox finding a lantern. "
        "End with 'the light came home.'",
        "The fox will pause by the lantern, its brass ears tilted toward the rain.",
        "'will pause' changes to future tense and breaks the requested past-tense voice",
        "The fox paused by the lantern, its brass ears tilted toward the rain.",
        "The fox paused by the lantern, its brass ears tilted toward the rain. "
        "It carried the warm glow into its hollow tree, and the light came home.",
    ),
    (
        "Write one paragraph announcing the workshop. Supplied fact: it opened at 09:00. "
        "Do not invent attendance figures.",
        "At 09:00 the workshop opened to 5,000 visitors.",
        "'5,000 visitors' is an unsupported attendance figure prohibited by the public task",
        "At 09:00 the workshop opened.",
        "At 09:00 the workshop opened. Welcome to the workshop.",
    ),
])
def test_visible_passage_and_specific_revision_reach_final_writer(
    writing_executor, mode, question, passage, defect, revision, final_answer,
):
    task = PublicTask(task_id="synthetic-writing-handoff", question=question, tools=[])
    pool = seed_pool()
    agents = [
        AgentSpec(agent_id="composer", role="Composer", capability="writing contribution",
                  pool_agent_id="generalist", pool_agent_version=1, max_calls=1, max_tokens=512,
                  task_prompt="Supply an actual budgeted passage, not the complete final deliverable."),
        AgentSpec(agent_id="reviewer", role="Critic", capability="writing review",
                  pool_agent_id="critic", pool_agent_version=1, depends_on=["composer"], max_calls=1,
                  max_tokens=512),
        AgentSpec(agent_id="writer", role="Writer", capability="writing synthesis",
                  pool_agent_id="writer", pool_agent_version=1, depends_on=["reviewer"], max_calls=1,
                  max_tokens=512),
    ]
    team = TeamSpec(execution_mode=mode, agents=agents, synthesizer_id="writer",
                    total_max_calls=3, max_parallel=1)
    ledger = BudgetLedger(max_calls=3, max_tokens=40_000, max_tool_calls=0)
    observed = {}

    class Model:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def __call__(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            assert self.agent_id not in observed
            observed[self.agent_id] = payload
            assert payload["public_task"]["tools"] == []
            response = {"answer": passage, "evidence_ids": [], "checkpoints": {}}
            if self.agent_id != "writer":
                response["ledger"] = {"requirements": [], "outline": [],
                                      "evidence_spans": [], "source_references": []}
            if self.agent_id == "reviewer":
                source = next(item for item in payload["shared_ledger"]["contributions"]
                              if item["agent_id"] == "composer")
                assert source["answer"] == passage
                assert "Inspect actual upstream passages" in payload["persistent_agent"]["skills"][
                    "public_constraint_check"]
                response["answer"] = f"Passage: {passage}\nDefect: {defect}\nRevision: {revision}"
                response["ledger"]["outline"] = [response["answer"]]
            elif self.agent_id == "writer":
                sources = {item["agent_id"]: item for item in payload["shared_ledger"]["contributions"]}
                assert sources["composer"]["answer"] == passage
                assert passage in sources["reviewer"]["answer"]
                assert defect in sources["reviewer"]["answer"]
                assert revision in sources["reviewer"]["answer"]
                assert "budgeted actual passages" in payload["persistent_agent"]["skills"]["genre_control"]
                assert payload["shared_ledger"]["source_references"] == []
                assert payload["shared_ledger"]["evidence_spans"] == []
                response["answer"] = final_answer
            return ChatMessage(role="assistant", content=json.dumps(response))

        def get_token_counts(self):
            return {"input_token_count": 20, "output_token_count": 10}

    artifact = writing_executor.synthesize(task, RubricGraph(rubrics=[]), team, agent_pool=pool)
    result = TeamExecutor(lambda aid: MeteredModel(Model(aid), ledger, "execution", aid),
                          ledger=ledger, knowledge_policy="model_general_knowledge_allowed").execute(
                              task, team, artifact)
    assert result.terminated_reason == "final_answer", json.dumps([
        event for event in result.metadata.get("events", []) if event["kind"] == "execution_error"
    ])
    assert result.answer == final_answer
    assert "Passage:" not in result.answer and "Defect:" not in result.answer
    assert ledger.snapshot()["model_calls"] == 3
    assert ledger.snapshot()["tool_calls"] == 0
    assert set(observed) == {"composer", "reviewer", "writer"}
