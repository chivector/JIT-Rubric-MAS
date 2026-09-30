"""Task-specific role wording must not disable capability-scoped experience reuse."""

import json
import shutil

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import TeamExecutor
from jit_mas.experience import capability_matches, retrieve
from jit_mas.schemas import AgentSpec, Experience, ExperienceSnapshot, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


def test_capability_matching_is_scoped_not_role_id_or_exact_prose():
    assert capability_matches("technical verification", "Verification of CNN explanations")
    assert capability_matches("", "article synthesis")
    assert not capability_matches("technical verification", "creative storytelling")
    assert not capability_matches("analysis of results", "writing of poems")
    assert capability_matches("comparison_article_writing", "Technical article writing")
    assert not capability_matches("comparison_article_writing", "Numerical verification")


def advice():
    return Experience(experience_id="check-assumptions", bank="execution",
        instruction="State assumptions before applying a formula.", applicability="Technical explanations",
        capability="technical verification", source_task_ids=["earlier-task"],
        evidence=["earlier-run:e1"], validation_status="accepted")


def test_retrieve_uses_capability_overlap_and_keeps_task_exclusions():
    snapshot = ExperienceSnapshot(experiences=[advice()])
    task = PublicTask(task_id="later-task", question="Explain CNNs.")
    assert len(retrieve(snapshot, task, capability="Verification of CNN explanations")) == 1
    assert retrieve(snapshot, task, capability="creative storytelling") == []
    assert retrieve(snapshot, task, capability="technical verification", excluded_task_ids=["earlier-task"]) == []


def test_renamed_role_receives_scoped_advice_and_exact_completion_protocol():
    from jit_mas.bridge import JITHarnessSynthesizer

    team = TeamSpec(agents=[AgentSpec(agent_id="new-role", role="Reviewer",
        capability="Verification of CNN explanations", max_calls=1,
        checkpoints=["Assumptions checked"])], synthesizer_id="new-role")
    task = PublicTask(task_id="later-task", question="Explain CNNs.")
    synth = JITHarnessSynthesizer()

    class Model:
        def __call__(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            assert payload["execution_experiences"][0]["experience_id"] == "check-assumptions"
            assert "every exact name" in messages[0]["content"]
            return ChatMessage(role="assistant", content=json.dumps({
                "answer": "A checked explanation.", "checkpoints": {"Assumptions checked": True}}))

        def get_token_counts(self):
            return {"input_token_count": 5, "output_token_count": 3}

    try:
        artifact = synth.synthesize(task, RubricGraph(rubrics=[]), team, [advice()])
        ledger = BudgetLedger()
        result = TeamExecutor(lambda aid: MeteredModel(Model(), ledger,
            "execution", aid), ledger=ledger).execute(task, team, artifact)
        assert result.terminated_reason == "final_answer"
        assert result.sub_runs[0].metadata["agent_id"] == "new-role"
        assert ledger.snapshot()["model_calls"] == 1
    finally:
        for agent in synth._agents.values():
            path = agent.workspace_dir.resolve()
            assert path.parent.name == "workspaces" and path.name.startswith("mas_")
            if path.is_dir():
                shutil.rmtree(path)
