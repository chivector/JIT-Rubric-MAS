"""Shared budget and typed-boundary regression checks without external clients."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.schemas import (
    AgentSpec, EvaluationFeedback, PredictedRubric, PublicTask, RubricFeedback,
    RubricGraph, TeamSpec,
)


def test_atomic_reservation_prevents_concurrent_oversubscription():
    ledger = BudgetLedger(max_calls=20, max_tokens=60)
    start = threading.Barrier(20)

    def reserve(index):
        start.wait(timeout=5)
        try:
            return ledger.reserve("inference", str(index), 10, 10)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        tickets = [ticket for ticket in pool.map(reserve, range(20)) if ticket is not None]
    assert len(tickets) == 3
    assert ledger.snapshot()["reserved_tokens"] == 60
    for ticket in tickets:
        ledger.settle(ticket, 8, 6)
    state = ledger.snapshot()
    assert state["reserved_tokens"] == 0 and state["tokens"] == 42
    assert state["model_calls"] == 3
    assert state["by_stage"]["inference"]["model_calls"] == 3
    assert len({row["call_id"] for row in state["records"]}) == 3


def test_unknown_and_partial_usage_reserve_conservatively_and_cost_is_unknown():
    ledger = BudgetLedger(max_calls=3, max_tokens=100)
    first = ledger.reserve("evaluation", "judge", 20, 10)
    second = ledger.reserve("update", "global", 20, 10)
    ledger.settle(first)
    ledger.settle(second, 5, None)
    state = ledger.snapshot()
    assert state["tokens"] == 45 and state["reserved_tokens"] == 0
    assert state["cost"] is None
    assert all(row["estimated"] for row in state["records"])
    assert state["by_stage"]["evaluation"]["tokens"] == 30
    assert state["by_stage"]["update"]["tokens"] == 15
    with pytest.raises(KeyError):
        ledger.settle(first)
    assert ledger.snapshot()["tokens"] == 45


def test_provider_overrun_is_recorded_then_blocks_next_request():
    ledger = BudgetLedger(max_calls=3, max_tokens=40)
    ticket = ledger.reserve("inference", "a", 10, 10)
    with pytest.raises(BudgetExceeded, match="exceeded"):
        ledger.settle(ticket, 25, 20)
    state = ledger.snapshot()
    assert state["tokens"] == 45 and state["reserved_tokens"] == 0
    assert state["model_calls"] == 1
    with pytest.raises(BudgetExceeded):
        ledger.reserve("inference", "b", 1, 1)


def test_call_and_tool_limits_are_shared_and_communication_does_not_rebill_tokens():
    ledger = BudgetLedger(max_calls=1, max_tokens=100, max_tool_calls=1)
    ticket = ledger.reserve("inference", "a", 10, 10)
    ledger.settle(ticket, 4, 5)
    with pytest.raises(BudgetExceeded, match="model-call"):
        ledger.reserve("update", "b", 1, 1)
    ledger.charge_tool("inference", "a", "lookup")
    with pytest.raises(BudgetExceeded, match="tool"):
        ledger.charge_tool("inference", "b", "lookup")
    ledger.charge_communication(120)
    assert ledger.snapshot()["tokens"] == 9
    assert ledger.snapshot()["by_stage"]["execution"]["communication_bytes"] == 120


@pytest.mark.parametrize("input_bound,output_bound", [(-1, 1), (1, -1)])
def test_negative_reservations_cannot_expand_budget(input_bound, output_bound):
    ledger = BudgetLedger(max_tokens=10)
    with pytest.raises(ValueError):
        ledger.reserve("inference", "a", input_bound, output_bound)
    assert ledger.snapshot()["reserved_tokens"] == 0


def test_metering_reuses_client_and_bounds_output_and_charges_failed_calls():
    calls = []

    def client(messages, **kwargs):
        calls.append(kwargs)
        return "answer"

    messages = [{"role": "user", "content": "public task"}]
    bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 64
    ledger = BudgetLedger(max_calls=3, max_tokens=1000)
    model = MeteredModel(client, ledger, "inference", "a", max_tokens=25)
    assert model(messages, max_tokens=100) == "answer"
    assert calls[0]["max_tokens"] == 25
    assert ledger.snapshot()["tokens"] == bound + 25
    assert model.calls[0]["messages"] == messages

    def failing_client(*args, **kwargs):
        raise TimeoutError("scripted timeout")

    with pytest.raises(TimeoutError):
        MeteredModel(failing_client, ledger, "evaluation", "judge", max_tokens=10)(messages)
    assert ledger.snapshot()["reserved_tokens"] == 0
    assert ledger.snapshot()["records"][-1]["error"] == "TimeoutError"
    assert ledger.snapshot()["tokens"] == bound * 2 + 35


def test_invalid_output_limit_never_reaches_model():
    calls = []
    model = MeteredModel(lambda *a, **kw: calls.append(kw), BudgetLedger(), "inference")
    with pytest.raises(ValueError):
        model([], max_tokens=-1)
    assert not calls


def test_unreadable_provider_usage_is_charged_as_unknown():
    class Client:
        def __call__(self, messages, **kwargs):
            return "answer"

        def get_token_counts(self):
            raise RuntimeError("Usage telemetry unavailable")

    ledger = BudgetLedger(max_calls=2, max_tokens=1000)
    model = MeteredModel(Client(), ledger, "inference", max_tokens=12)
    model([])
    state = ledger.snapshot()
    assert state["tokens"] == 2 + 64 + 12
    assert state["reserved_tokens"] == 0 and state["records"][0]["estimated"]


def make_agent(aid, deps=(), calls=2, tokens=128):
    return AgentSpec(agent_id=aid, role="Analyst", capability="compare sources",
                     depends_on=list(deps), max_calls=calls, max_tokens=tokens)


def test_execution_cycles_are_rejected_while_rubric_association_cycles_are_allowed():
    with pytest.raises(ValidationError, match="cycle"):
        TeamSpec(agents=[make_agent("a", ["b"], calls=1), make_agent("b", ["a"], calls=1)],
                 synthesizer_id="b", total_max_calls=2)
    graph = RubricGraph(rubrics=[{"rubric_id": rid, "requirement": rid} for rid in ["r1", "r2"]],
                        edges=[{"source": src, "target": dst, "relation": "support", "rationale": "related"}
                               for src, dst in [("r1", "r2"), ("r2", "r1")]])
    assert len(graph.edges) == 2


def test_agent_allocations_are_not_copies_of_total_team_budget():
    with pytest.raises(ValidationError, match="allocations"):
        TeamSpec(agents=[make_agent("a", calls=4), make_agent("b", ["a"], calls=4)],
                 synthesizer_id="b", total_max_calls=4, execution_mode="iterative_shared_ledger")
    spec = TeamSpec(agents=[make_agent("a"), make_agent("b", ["a"])],
                    synthesizer_id="b", total_max_calls=4, execution_mode="iterative_shared_ledger")
    assert sum(agent.max_calls for agent in spec.agents) == 4
    with pytest.raises(ValidationError):
        make_agent("a", tokens=0)


@pytest.mark.parametrize("calls", [2, None])
def test_single_pass_rejects_multiple_or_unlimited_role_calls(calls):
    with pytest.raises(ValidationError, match="single-pass execution requires AgentSpec.max_calls=1"):
        TeamSpec(agents=[make_agent("a", calls=calls)], synthesizer_id="a", total_max_calls=2)


def test_public_boundary_rejects_hidden_fields_and_importance_is_not_confidence():
    with pytest.raises(ValidationError, match="Extra inputs"):
        PublicTask(task_id="t", question="Public", rubrics=["hidden"])
    predicted = PredictedRubric(rubric_id="r", requirement="Evidence", importance=20, confidence=0.2)
    assert predicted.importance == 20 and predicted.confidence == 0.2
    with pytest.raises(ValidationError):
        PredictedRubric(rubric_id="r", requirement="Evidence", importance=-1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_cannot_cross_feedback_boundary(score):
    with pytest.raises(ValidationError):
        RubricFeedback(rubric_id="r", criterion="Evidence", weight=1, score=score)
    with pytest.raises(ValidationError):
        EvaluationFeedback(task_id="t", evaluator_version="test", rubrics=[], complete=True, score=score)
