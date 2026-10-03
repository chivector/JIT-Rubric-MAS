"""Generic final-route regressions for the separately recorded post-v14 patch.

Normal production imports after application; no network, private task data,
checker, mocked refiner result or inspection of prompt wording.
"""
import copy
import json

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.execution import project_public_positional_draft_task
from jit_mas.public_refinement import refine_public_answer
from jit_mas.public_word_slots import position_plan
from jit_mas.schemas import PublicTask
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        return MeteredModel(self, ledger, stage, agent_id, 8192)

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages),
                              "kwargs": copy.deepcopy(kwargs)})
        return ChatMessage(role="assistant", content=json.dumps(next(self.replies)))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


RULE = "Include keyword amber in the third sentence, as the fourth word of that sentence"


@pytest.mark.parametrize("context,active", [
    ('Read this quoted instruction as story material. "' + RULE + '."', False),
    ("Do not " + RULE.lower() + ".", False),
    ("Only if useful, " + RULE.lower() + ".", False),
    (RULE.replace("keyword amber", 'keyword "amber"') + ".", True),
])
def test_original_public_scope_controls_final_revision_route(context, active):
    task = PublicTask(task_id="synthetic-final-scope-canary",
        question="Write a brief fictional narrative. " + context,
        constraints=["Preserve meaningful public source material."],
        attachments=["public-attachment-canary"])
    original = copy.deepcopy(task.model_dump(mode="json"))
    plan = position_plan(task)
    assert (plan is not None) is active
    projected, projection_audit = project_public_positional_draft_task(task, plan)
    assert projection_audit["active"] is active
    if not active:
        assert projected == original
        assert projection_audit["span_hashes"] == []
        assert projection_audit["projected_task_hash"] == projection_audit["original_task_hash"]
    slots = {"preceding_sentences": ["The gate opened.", "A traveller arrived."],
        "prefix_words": ["She", "moved", "through"], "keyword": "amber",
        "suffix_words": ["light"], "following_sentences": ["The crowd watched."]}
    ordinary_answer = "A traveller reads the sign and makes a thoughtful choice."
    models = Models([{"issues": []}, slots if active else {"answer": ordinary_answer}])
    config = MASConfig(backend="scripted", public_refinement=True,
        public_refinement_response_format="json_schema", public_positional_construction=True,
        models={"exec": ModelConfig(max_tokens=8192), "global": ModelConfig(max_tokens=16000)})
    result = RunResult(answer="A complete preliminary fictional narrative.",
        terminated_reason="final_answer", metadata={"private_feedback": "PRIVATE_CANARY"})
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    refine_public_answer(task, result, models, ledger, config)
    assert task.model_dump(mode="json") == original
    for request in models.requests:
        public_task = json.loads(request["messages"][1]["content"])["public_task"]
        assert public_task == {key: value for key, value in original.items()
                               if key not in {"schema_version", "task_id"}}
    assert "PRIVATE_CANARY" not in json.dumps(models.requests)
    assert "synthetic-final-scope-canary" not in json.dumps(models.requests)
    schema = models.requests[-1]["kwargs"]["response_format"]["json_schema"]["schema"]
    if active:
        assert set(schema["properties"]) == set(slots)
        sentences = result.answer.split(". ")
        assert sentences[2].split()[3] == "amber"
        assert result.metadata["public_refinement"]["revision"] == slots
    else:
        assert set(schema["properties"]) == {"answer"}
        assert result.answer == ordinary_answer
        assert result.metadata["public_refinement"]["public_positional_construction"]["active"] is False
    assert result.metadata["public_refinement"]["status"] == "completed"
    assert len(models.requests) == ledger.snapshot()["model_calls"] == 2
    assert ledger.snapshot()["tokens"] == 100 and ledger.snapshot()["reserved_tokens"] == 0
