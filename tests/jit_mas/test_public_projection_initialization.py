"""Synthetic executor/pipeline regressions for pre-role projection audit state."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.execution import (
    TeamAction, TeamExecutor, TeamMemory, TeamPlanning, TeamServices, TeamToolPolicy,
    compile_public_positional_draft_plan, content_hash,
)
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec, digest
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


RULE = "Place the word saffron as the fourth word in the third sentence"
DRAFT = (
    "Helpers reached the orchard safely. They brought a sturdy rope. "
    "They guided the stranded visitor toward the dry path. Everyone returned together."
)


def settings(mode="iterative_shared_ledger"):
    return MASConfig(
        backend="scripted", execution_mode=mode,
        team_max_calls=None if mode == "iterative_shared_ledger" else 16,
        max_model_calls=None, max_total_tokens=2_000_000, max_tool_calls=0,
        task_timeout=900, execution_timeout=900, max_repairs=0,
        planning_response_format="json_schema", public_refinement=True,
        public_refinement_guard=True, public_refinement_response_format="json_schema",
        public_membership_observations=True, public_membership_input_format="compact",
        public_construction_validation_retries=1, public_positional_construction=True,
        public_positional_draft_guidance=True, public_positional_draft_projection=True,
        public_numeric_construction=True, public_numeric_construction_layout="template",
    )


def public_task(supported=False):
    question = "Describe an orchard rescue with a careful ending."
    if supported:
        question += " " + RULE + ". Keep the rope and safe return."
    return PublicTask(task_id="synthetic-projection-state", question=question,
                      constraints=["Keep the answer in English."])


def loaded_modules(action=None):
    return {"action": action or TeamAction(), "memory": TeamMemory(),
            "planning": TeamPlanning(), "tool_policy": TeamToolPolicy(),
            "prompts": {"system_prompt": "Coordinate the complete public task."}}


def configure_executor(executor, task, config):
    executor.public_positional_draft_guidance_requested = True
    executor.public_positional_draft_plan = compile_public_positional_draft_plan(task, config)
    executor.public_positional_draft_projection_requested = True
    executor.public_membership_observations_requested = True
    executor.public_membership_input_format = "compact"


def executable_artifact(task, team, graph):
    return SimpleNamespace(
        backend="scripted", name="synthetic-projection-initialization", code_hash="synthetic",
        task_hash=content_hash(task), team_hash=content_hash(team),
        sidecar={"rubrics": graph.model_dump(mode="json"), "experiences": []},
        verify_integrity=lambda: None,
    )


class Reply:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return ChatMessage(role="assistant", content=json.dumps({
            "answer": DRAFT, "evidence_ids": [], "checkpoints": {}, "continue": False,
        }))

    def get_token_counts(self):
        return {"input_token_count": 20, "output_token_count": 30}


def run_executor(monkeypatch, mode, supported=False):
    task = public_task(supported)
    config = settings(mode)
    cap = None if mode == "iterative_shared_ledger" else 1
    team = TeamSpec(execution_mode=mode,
        agents=[AgentSpec(agent_id="author", role="Writer", capability="storytelling", max_calls=cap)],
        synthesizer_id="author", total_max_calls=cap)
    graph = RubricGraph(rubrics=[])
    reply = Reply()
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000, max_tool_calls=0, timeout_seconds=900)
    executor = TeamExecutor(lambda aid: MeteredModel(reply, ledger, "inference", aid, 8192),
                            ledger=ledger, timeout_seconds=900)
    configure_executor(executor, task, config)
    monkeypatch.setattr("jit_mas.execution.load_harness", lambda _: loaded_modules())
    result = executor.execute(task, team, executable_artifact(task, team, graph))
    return task, result, reply, ledger


def test_default_audit_exists_before_any_role_and_is_independent():
    task = public_task()
    before = task.model_dump(mode="json")
    first = TeamServices(lambda _: None, task)
    second = TeamServices(lambda _: None, task)
    audit = first.public_positional_draft_projection_audit
    assert audit["active"] is False and audit["final_constraints_retained"] is True
    assert audit["original_task_hash"] == audit["projected_task_hash"] == content_hash(before)
    assert audit["span_hashes"] == [] and audit["reason"]
    audit["span_hashes"].append({"synthetic": True})
    assert second.public_positional_draft_projection_audit["span_hashes"] == []
    assert task.model_dump(mode="json") == before


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
@pytest.mark.parametrize("supported", [False, True])
def test_actual_executor_succeeds_with_public_flags_and_both_plan_routes(monkeypatch, mode, supported):
    task, result, reply, ledger = run_executor(monkeypatch, mode, supported)
    assert result.terminated_reason == "final_answer" and result.answer == DRAFT
    assert len(reply.calls) == 1 and ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["tokens"] == 50 and ledger.snapshot()["reserved_tokens"] == 0
    audit = result.metadata["public_positional_draft_projection"]
    assert audit["active"] is supported and audit["final_constraints_retained"]
    assert result.metadata["public_positional_draft_guidance"]["active"] is supported
    payload = json.loads(reply.calls[0][1]["content"])
    assert (RULE in payload["public_task"]["question"]) is False
    if not supported:
        assert payload["public_task"] == task.model_dump(mode="json")
        assert audit["projected_task_hash"] == audit["original_task_hash"]
    assert result.metadata["public_membership_execution_audits"]


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
def test_primary_role_initialization_error_survives_before_projection_builder(monkeypatch, mode):
    def fail_before_prompt(*args):
        raise ValueError("synthetic-primary-role-initialization")

    monkeypatch.setattr("jit_mas.execution._role_prompt", fail_before_prompt)
    _, result, reply, ledger = run_executor(monkeypatch, mode)
    assert result.terminated_reason == "error" and result.answer is None
    assert not reply.calls and ledger.snapshot()["model_calls"] == 0
    assert result.metadata["public_positional_draft_projection"]["active"] is False
    assert result.sub_runs[0].metadata["error"] == "synthetic-primary-role-initialization"
    assert any(event["kind"] == "execution_error" and event["content"] ==
               "synthetic-primary-role-initialization" for event in result.metadata["events"])
    assert "AttributeError" not in json.dumps(result.full_dict())


@pytest.fixture
def pipeline_transport(monkeypatch):
    from scripts.models.openai_server import OpenAIServerModel

    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic pipeline must never issue a real API request")

    monkeypatch.setattr(OpenAIServerModel, "__call__", forbidden)
    original = FixtureModel.__call__
    observed = []

    def reply(self, messages, **kwargs):
        if self.role != "exec" and self.agent_id not in {"public-review", "public-revision"}:
            return original(self, messages, **kwargs)
        observed.append({"role": self.role, "agent_id": self.agent_id,
                         "messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        self.counts = {"input_token_count": 20, "output_token_count": 30}
        payload = json.loads(messages[-1]["content"])
        if self.agent_id == "public-review":
            value = {"issues": []}
        elif self.agent_id == "public-revision":
            if RULE in payload["public_task"]["question"]:
                value = {"preceding_sentences": ["Helpers reached the orchard safely.",
                         "They brought a sturdy rope."], "prefix_words": ["They", "stood", "beside"],
                         "keyword": "saffron", "suffix_words": ["trees", "until", "everyone", "returned"],
                         "following_sentences": ["They stored the rope safely."]}
            else:
                value = {"answer": DRAFT}
        else:
            synthesizer = payload["submission"] == "final_answer"
            value = {"answer": DRAFT if synthesizer else "A rope guided the visitor to a safe path.",
                     "evidence_ids": [], "checkpoints": {key: {
                         "status": "unverified", "reason": "Synthetic public check not independently verified.",
                         "evidence_ids": []} for key in payload["agent"]["checkpoints"]}, "continue": False}
            if not synthesizer:
                value["ledger"] = {"requirements": ["Keep the ending safe."],
                    "outline": ["Bring the rope and guide the visitor toward a dry path."],
                    "evidence_spans": [], "source_references": []}
        return ChatMessage(role="assistant", content=json.dumps(value))

    monkeypatch.setattr(FixtureModel, "__call__", reply)
    return observed


@pytest.mark.parametrize("supported", [False, True])
def test_full_pooled_pipeline_public_flags_no_plan_and_supported_plan(tmp_path, pipeline_transport, supported):
    output = tmp_path / "runs"
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        config = settings()
        provider = FixtureModels()
        pipeline = make_pipeline(config, store, output, fixture_models=provider,
                                 knowledge_policy="model_general_knowledge_allowed")
        task_id = pipeline.manifest.test[0]
        task = public_task(supported).model_copy(update={"task_id": task_id})
        original = task.model_dump(mode="json")
        pipeline.tasks[task_id] = task
        outcome = pipeline.run_task(task_id, store.snapshot(), defer_evaluation=True)
    finally:
        store.close()
    folder = Path(outcome["run_dir"])
    draft = json.loads((folder / "execution_draft.json").read_text(encoding="utf-8"))
    final = json.loads((folder / "execution.json").read_text(encoding="utf-8"))
    refinement = json.loads((folder / "public_refinement.json").read_text(encoding="utf-8"))
    harness = json.loads((folder / "harness.json").read_text(encoding="utf-8"))
    budget = json.loads((folder / "budget.json").read_text(encoding="utf-8"))
    submission = json.loads((folder / "submission.json").read_text(encoding="utf-8"))
    assert task.model_dump(mode="json") == original
    assert draft["terminated_reason"] == final["terminated_reason"] == "final_answer"
    assert draft["metadata"]["public_positional_draft_projection"]["active"] is supported
    assert refinement["public_input"]["public_task"]["question"] == original["question"]
    assert refinement["public_input"]["public_task"]["constraints"] == original["constraints"]
    assert len(refinement["calls"]) == 2 and refinement["status"] == "completed"
    assert harness["selection"]["strategy"] == "pooled_agent_reuse" and harness["repair_count"] == 0
    assert not any(row["agent_id"] == "meta" for row in budget["records"] if row["kind"] == "model")
    assert submission["answer"] == final["answer"]
    assert submission["answer_hash"] == digest(final["answer"])
    assert budget["reserved_tokens"] == 0
    assert not (folder / "evaluation.json").exists()
    assert {call["agent_id"] for call in pipeline_transport if call["role"] == "exec"} == {
        run["metadata"]["agent_id"] for run in draft["sub_runs"]}
    assert "PRIVATE_CANARY" not in json.dumps(pipeline_transport)


def test_full_pipeline_failure_retains_primary_run_with_no_role_reservation(tmp_path, monkeypatch, pipeline_transport):
    def fail_before_prompt(*args):
        raise ValueError("synthetic-primary-role-initialization")

    monkeypatch.setattr("jit_mas.execution._role_prompt", fail_before_prompt)
    output = tmp_path / "failed-runs"
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(settings(), store, output, fixture_models=FixtureModels(),
                                 knowledge_policy="model_general_knowledge_allowed")
        task_id = pipeline.manifest.test[0]
        pipeline.tasks[task_id] = public_task().model_copy(update={"task_id": task_id})
        with pytest.raises(RuntimeError):
            pipeline.run_task(task_id, store.snapshot(), defer_evaluation=True)
    finally:
        store.close()
    paths = list(output.rglob("failure.json"))
    assert len(paths) == 1
    failure = json.loads(paths[0].read_text(encoding="utf-8"))
    attempts = failure["jit_mas_failure"]["failed_execution_attempts"]
    assert len(attempts) == 1 and attempts[0]["role_execution_started"] is False
    run = attempts[0]["run"]
    assert run is not None and run["terminated_reason"] == "error" and run["answer"] is None
    assert run["metadata"]["public_positional_draft_projection"]["active"] is False
    assert any(child["metadata"].get("error") == "synthetic-primary-role-initialization"
               for child in run["sub_runs"])
    assert "synthetic-primary-role-initialization" in attempts[0]["error"]
    assert "public_positional_draft_projection_audit" not in attempts[0]["error"]
    assert not pipeline_transport
    role_ids = {child["metadata"]["agent_id"] for child in run["sub_runs"]}
    # Local planning uses these identities too; model records after reconciliation
    # must contain no role execution request or uncharged hidden reservation.
    budget = failure["budget"]
    assert not any(row["agent_id"] == "meta" for row in budget["records"] if row["kind"] == "model")
    assert all(row["stage"] != "execution" for row in budget["records"] if row["kind"] == "model")
    assert role_ids and budget["reserved_tokens"] == 0
    assert not list(output.rglob("submission.json")) and not list(output.rglob("evaluation.json"))
