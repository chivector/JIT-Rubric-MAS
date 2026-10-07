"""Public-only global refinement, shared resources and submission ordering."""

import copy
import json
from pathlib import Path

import pytest

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels
from jit_mas.public_refinement import PUBLIC_REFINEMENT_IDS, refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


DRAFT = "The clock keeps time. Its hands move slow."
REVISION = "The clock keeps time.\nIts hands move slow.\nThe room is still.\nThe hours go."
REVIEW = {"issues": [{"defect": "Two lines are missing.",
                      "public_basis": "The public task requests four lines.",
                      "repair": "Complete the four-line poem."}]}


class Responses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)})
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return ChatMessage(role="assistant", content=response if isinstance(response, str) else json.dumps(response))

    def get_token_counts(self):
        return {"input_token_count": 30, "output_token_count": 20}


class Models:
    def __init__(self, replies):
        self.responses = Responses(replies)
        self.created = []

    def create(self, role, agent_id, ledger, stage):
        self.created.append((role, agent_id, ledger, stage))
        return MeteredModel(self.responses, ledger, stage, agent_id, 16000)


def public_fixture():
    task = PublicTask(task_id="PRIVATE_ROUTING_BENCHMARK_ID", question="Write a four-line poem about a clock.",
                      constraints=['Use the word "clock" once.'])
    result = RunResult(answer=DRAFT, terminated_reason="final_answer", metadata={
        "call_counts": {"contributor": 1, "writer": 1}, "model_calls_used": 2,
        "private_feedback": "PRIVATE_CANARY", "raw_judge": {"score": "PRIVATE_CANARY"},
        "events": [{"kind": "evaluation", "content": "PRIVATE_CANARY"}],
        "artifacts": {
            "contributor": {"answer": "Use four plain-language lines.", "event_id": "public-event",
                "private_judge": "PRIVATE_CANARY", "checkpoint_reports": {"score": "PRIVATE_CANARY"},
                "ledger": {"requirements": ["Four lines."], "outline": ["A still room."],
                    "hidden_reference": "PRIVATE_CANARY",
                    "source_references": [{"source_id": "public-request", "locator": "public_task.question",
                                           "private_weight": "PRIVATE_CANARY"}],
                    "evidence_spans": [{"text": "four-line poem", "source_ref": "public-request",
                                       "private_criterion": "PRIVATE_CANARY"}]}},
            "writer": {"answer": DRAFT, "ledger": {}},
        }})
    return task, result


def refinement_config(cap=8192):
    return MASConfig(backend="scripted", public_refinement=True,
                     models={"exec": ModelConfig(max_tokens=cap), "global": ModelConfig(max_tokens=16000)})


@pytest.mark.parametrize("execution_cap,expected", [(4096, 4096), (16000, 16000)])
def test_two_global_calls_use_existing_ledger_and_public_data(execution_cap, expected):
    task, result = public_fixture()
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    # Earlier local execution costs remain on the same ledger.
    ticket = ledger.reserve("inference", "writer", 30, 20)
    ledger.settle(ticket, 30, 20)
    models = Models([REVIEW, {"answer": REVISION}])
    snapshots = []
    refine_public_answer(task, result, models, ledger, refinement_config(execution_cap),
                         synthesizer_id="writer", knowledge_policy="model_general_knowledge_allowed",
                         audit_writer=lambda audit: snapshots.append(copy.deepcopy(audit)))

    assert result.answer == REVISION
    assert [(role, agent, stage) for role, agent, _, stage in models.created] == [
        ("global", "public-review", "inference"), ("global", "public-revision", "inference")]
    assert all(shared is ledger for _, _, shared, _ in models.created)
    assert ledger.snapshot()["model_calls"] == 3
    assert ledger.snapshot()["tokens"] == 150
    assert result.metadata["call_counts"] == {"contributor": 1, "writer": 1}
    assert result.metadata["model_calls_used"] == 2
    requests = models.responses.requests
    assert len(requests) == 2
    for request in requests:
        assert request["kwargs"]["max_tokens"] == expected
        assert request["kwargs"]["response_format"] == {"type": "json_object"}
        assert 0 < request["kwargs"]["timeout"] <= ledger.timeout_seconds
        serialized = json.dumps(request["messages"])
        assert "PRIVATE_CANARY" not in serialized
        assert "PRIVATE_ROUTING_BENCHMARK_ID" not in serialized
        assert "Use four plain-language lines." in serialized
        assert "model_general_knowledge_allowed" in serialized
        payload = json.loads(request["messages"][1]["content"])
        assert payload["draft"] == DRAFT
        assert payload["public_diagnostics"]["characters"] == len(DRAFT)
    assert json.loads(requests[1]["messages"][1]["content"])["review"] == REVIEW

    audit = result.metadata["public_refinement"]
    assert audit == snapshots[-1]
    assert audit["status"] == "completed"
    assert audit["draft"] == DRAFT and audit["draft_hash"] == digest(DRAFT)
    assert audit["review"] == REVIEW and audit["review_hash"] == digest(REVIEW)
    assert audit["revision"] == {"answer": REVISION}
    assert audit["revision_hash"] == digest(audit["revision"])
    assert audit["answer_hash"] == digest(REVISION)
    assert audit["public_input_hash"] == digest(audit["public_input"])
    assert audit["audit_hash"] == digest({k: v for k, v in audit.items() if k != "audit_hash"})
    assert audit["revision_public_diagnostics"]["nonempty_lines"] == 4
    assert [row["agent_id"] for row in audit["budget_records"]] == list(PUBLIC_REFINEMENT_IDS)
    assert all(row["metered_calls"][0]["call_id"] == cost["call_id"]
               for row, cost in zip(audit["calls"], audit["budget_records"]))


def test_empty_review_still_submits_only_the_revision():
    task, result = public_fixture()
    # No quality comparison between the old draft and the returned revision.
    models = Models([{"issues": []}, {"answer": "A revised complete artifact."}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), refinement_config())
    assert result.answer == "A revised complete artifact."
    assert len(models.responses.requests) == 2


@pytest.mark.parametrize("bad_review", [
    "not json", '```json\n{"issues": []}\n```', '[]', '{"issues": [], "score": 9}',
    '{"issues": [], "issues": []}', '{"issues": [], "unused": NaN}',
    {"issues": [{"defect": " ", "public_basis": "request", "repair": "fix"}]},
    {"issues": [{"defect": "x", "public_basis": "request", "repair": "fix"}] * 9},
])
def test_invalid_review_is_charged_without_correction_or_revision(bad_review):
    task, result = public_fixture()
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = Models([bad_review])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, refinement_config())
    audit = result.metadata["public_refinement"]
    assert result.answer == DRAFT
    assert audit["status"] == "failed" and audit["draft"] == DRAFT
    assert len(models.responses.requests) == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["tokens"] == 50
    assert audit["calls"][0]["response"]
    assert audit["calls"][0]["status"] == "failed"
    assert audit["revision"] is None


@pytest.mark.parametrize("bad_revision", ["not json", {"answer": " "}, {"answer": 7},
                                          {"answer": "ok", "score": 10}])
def test_invalid_revision_preserves_review_and_both_costs(bad_revision):
    task, result = public_fixture()
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = Models([REVIEW, bad_revision])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, refinement_config())
    audit = result.metadata["public_refinement"]
    assert result.answer == DRAFT
    assert audit["review"] == REVIEW and audit["review_hash"] == digest(REVIEW)
    assert audit["status"] == "failed"
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["tokens"] == 100
    assert len(audit["budget_records"]) == 2
    assert len(models.responses.requests) == 2


def test_finite_common_call_budget_is_not_bypassed():
    task, result = public_fixture()
    ledger = BudgetLedger(1, 2_000_000, 0)
    models = Models([REVIEW])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, refinement_config())
    assert result.answer == DRAFT
    assert result.metadata["public_refinement"]["review"] == REVIEW
    assert len(models.responses.requests) == ledger.snapshot()["model_calls"] == 1


def test_token_budget_is_not_replaced_by_a_new_refinement_budget():
    task, result = public_fixture()
    ledger = BudgetLedger(None, 1, 0)
    models = Models([])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, refinement_config())
    assert not models.responses.requests
    assert ledger.snapshot()["model_calls"] == ledger.snapshot()["tokens"] == 0
    assert result.metadata["public_refinement"]["status"] == "failed"


def test_expired_task_deadline_prevents_refinement_requests(monkeypatch):
    task, result = public_fixture()
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=0.001)
    monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0.0)
    models = Models([])
    with pytest.raises(TimeoutError):
        refine_public_answer(task, result, models, ledger, refinement_config())
    assert not models.created and not models.responses.requests
    assert ledger.snapshot()["model_calls"] == 0
    assert result.metadata["public_refinement"]["status"] == "failed"


def test_transport_failure_preserves_estimated_charge():
    task, result = public_fixture()
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = Models([REVIEW, RuntimeError("synthetic transport failure")])
    with pytest.raises(RuntimeError, match="synthetic"):
        refine_public_answer(task, result, models, ledger, refinement_config())
    audit = result.metadata["public_refinement"]
    assert audit["review"] == REVIEW and audit["status"] == "failed"
    assert ledger.snapshot()["model_calls"] == 2
    assert audit["budget_records"][1]["estimated"] is True
    assert audit["budget_records"][1]["error"] == "RuntimeError"
    assert ledger.snapshot()["reserved_tokens"] == 0


class RefiningFixtureModels(FixtureModels):
    def __init__(self, output, *, fail_revision=False):
        super().__init__()
        self.output = output
        self.refinement = Models([REVIEW, "not json" if fail_revision else {"answer": REVISION}])
        self.judge_answers = []

    def create(self, role, agent_id, ledger, stage):
        if agent_id in PUBLIC_REFINEMENT_IDS:
            assert not list(self.output.glob("*/submission.json"))
            return self.refinement.create(role, agent_id, ledger, stage)
        model = super().create(role, agent_id, ledger, stage)
        if role == "judge":
            provider = self

            class Judge:
                def __call__(self, messages, **kwargs):
                    submission = json.loads(next(provider.output.glob("*/submission.json")).read_text())
                    audit = json.loads(next(provider.output.glob("*/public_refinement.json")).read_text())
                    assert audit["status"] == "completed"
                    assert submission["answer"] == REVISION
                    provider.judge_answers.append(submission["answer"])
                    return model(messages, **kwargs)

                def __getattr__(self, name):
                    return getattr(model, name)

            return Judge()
        return model


def make_refinement_pipeline(tmp_path, *, enabled=True, fail_revision=False):
    output = tmp_path / "runs"
    provider = RefiningFixtureModels(output, fail_revision=fail_revision)
    store = ExperienceStore(tmp_path / "state.sqlite")
    pipeline = make_pipeline(MASConfig(backend="scripted", public_refinement=enabled,
                                      max_agents=1, team_max_calls=1, max_model_calls=None),
                             store, output, fixture_models=provider)
    return pipeline, store, provider


def test_pipeline_preserves_draft_and_seals_revision_without_adding_pool_agents(tmp_path):
    pipeline, store, provider = make_refinement_pipeline(tmp_path)
    try:
        outcome = pipeline.run_task("test-poem", store.snapshot(), mode="evaluate", defer_evaluation=True)
        directory = Path(outcome["run_dir"])
        execution = json.loads((directory / "execution.json").read_text())
        draft = json.loads((directory / "execution_draft.json").read_text())
        audit = json.loads((directory / "public_refinement.json").read_text())
        submission = json.loads((directory / "submission.json").read_text())
        frozen = json.loads((directory / "frozen_plan.json").read_text())
        trace = json.loads((directory / "call_trace.json").read_text())
        assert draft["answer"] == audit["draft"]
        assert "public_refinement" not in draft["metadata"]
        assert execution["metadata"]["public_refinement"] == audit
        assert execution["answer"] == submission["answer"] == REVISION
        assert submission["answer_hash"] == outcome["answer_hash"] == digest(REVISION)
        assert len(frozen["TeamSpec"]["agents"]) == len(execution["sub_runs"]) == 1
        assert frozen["TeamSpec"]["agents"][0]["max_calls"] == 1
        assert execution["metadata"]["model_calls_used"] == 1
        assert len(trace["execution_calls"]) == 1
        assert [row["agent_id"] for row in trace["public_refinement_calls"]] == list(PUBLIC_REFINEMENT_IDS)
        assert outcome["status"] == "submitted_unscored"
        assert not provider.judge_answers and not (directory / "evaluation.json").exists()
        assert all("PRIVATE_CANARY" not in json.dumps(call) for call in provider.refinement.responses.requests)
        before = len(provider.refinement.responses.requests)
        again = pipeline.run_task("test-poem", store.snapshot(), mode="evaluate", defer_evaluation=True)
        assert again["resumed"] is True
        assert len(provider.refinement.responses.requests) == before
    finally:
        store.close()


def test_evaluator_sees_revision_only_after_submission(tmp_path):
    pipeline, store, provider = make_refinement_pipeline(tmp_path)
    try:
        outcome = pipeline.run_task("test-poem", store.snapshot(), mode="evaluate")
        assert provider.judge_answers == [REVISION]
        assert outcome["answer_hash"] == digest(REVISION)
    finally:
        store.close()


def test_failed_revision_writes_audit_costs_but_no_submission_or_evaluation(tmp_path):
    pipeline, store, provider = make_refinement_pipeline(tmp_path, fail_revision=True)
    try:
        with pytest.raises(ValueError) as caught:
            pipeline.run_task("test-poem", store.snapshot(), mode="evaluate", defer_evaluation=True)
        directory = Path(caught.value.jit_mas_run_failure["run_dir"])
        execution = json.loads((directory / "execution.json").read_text())
        audit = json.loads((directory / "public_refinement.json").read_text())
        budget = json.loads((directory / "budget.json").read_text())
        trace = json.loads((directory / "call_trace.json").read_text())
        assert execution["answer"] == audit["draft"]
        assert audit["status"] == "failed" and audit["review"] == REVIEW
        assert len(audit["budget_records"]) == len(trace["public_refinement_calls"]) == 2
        assert budget["tokens"] > 100 and budget["reserved_tokens"] == 0
        assert caught.value.jit_mas_run_failure["budget"]["tokens"] == budget["tokens"]
        assert not (directory / "submission.json").exists()
        assert not (directory / "evaluation.json").exists()
        assert not (directory / "complete.json").exists()
        assert not provider.judge_answers
    finally:
        store.close()


def test_default_off_preserves_existing_execution_and_budget(tmp_path):
    assert MASConfig().public_refinement is False
    pipeline, store, provider = make_refinement_pipeline(tmp_path, enabled=False)
    try:
        outcome = pipeline.run_task("test-poem", store.snapshot(), mode="evaluate", defer_evaluation=True)
        directory = Path(outcome["run_dir"])
        execution = json.loads((directory / "execution.json").read_text())
        assert not provider.refinement.created
        assert "public_refinement" not in execution["metadata"]
        assert not (directory / "execution_draft.json").exists()
        assert not (directory / "public_refinement.json").exists()
        assert outcome["budget"]["model_calls"] == sum(row["kind"] == "model"
                                                       for row in outcome["budget"]["records"])
    finally:
        store.close()
