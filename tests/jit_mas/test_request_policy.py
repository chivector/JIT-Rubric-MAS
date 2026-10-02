"""Synthetic context and shared request limits for registered native campaigns."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import ModelConfig
from jit_mas.request_policy import ContextLimitExceeded, RequestPolicyModel, prepare_context
from scripts.models.base import ChatMessage


def test_oldest_turn_policy_preserves_initial_and_latest_required_inputs():
    messages = [{"role": "system", "content": "Fixed system"},
                {"role": "user", "content": "Public task and frozen profile"},
                {"role": "assistant", "content": "old historical text " * 500},
                {"role": "user", "content": "old observation " * 500},
                {"role": "assistant", "content": "Latest draft"},
                {"role": "user", "content": "Latest tool observation"}]
    prepared, record = prepare_context(messages, output_tokens=16, context_window=200, margin=16,
                                       policy="oldest_turns")
    assert prepared == messages[:2] + messages[-2:]
    assert len(record["removed_message_hashes"]) == 2
    assert record["sent_input_tokens"] < record["original_input_tokens"]
    assert len(messages) == 6


def test_required_task_and_criteria_are_never_truncated():
    messages = [{"role": "system", "content": "Frozen evaluator"},
                {"role": "user", "content": json.dumps({"private_criteria": "required " * 1000})}]
    with pytest.raises(ContextLimitExceeded, match="Required initial or current"):
        prepare_context(messages, output_tokens=16, context_window=128, margin=16, policy="oldest_turns")
    with pytest.raises(ValueError, match="Context window"):
        ModelConfig(max_tokens=100, context_window=101, context_margin=1)


def test_native_requests_share_capacity_and_record_provider_identity():
    lock = threading.Lock()
    active = peak = 0
    gate = threading.BoundedSemaphore(2)

    class Model:
        def __call__(self, messages, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return ChatMessage(role="assistant", content='{"ok":true}',
                               raw=SimpleNamespace(model="served-model", system_fingerprint="fixed", id="synthetic",
                                                   choices=[SimpleNamespace(finish_reason="stop")]))

    models = [RequestPolicyModel(Model(), gate=gate, ledger=BudgetLedger(timeout_seconds=1),
                                 expected_model="served-model") for _ in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda model: model([{"role": "user", "content": "Synthetic"}]), models))
    assert peak == 2 and len(responses) == 6
    assert all(model.last_request_metadata["response_model"] == "served-model" for model in models)
    assert all(model.last_request_metadata["finish_reason"] == "stop" for model in models)
    assert any(model.last_request_metadata["queue_seconds"] > 0 for model in models)


def test_idle_request_queue_time_does_not_consume_task_deadline(monkeypatch):
    import jit_mas.budget as budget
    now = [0.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: now[0])
    ledger = BudgetLedger(timeout_seconds=10)
    ledger.begin_request_wait()
    now[0] = 100
    assert ledger.remaining_seconds() == 10
    ledger.end_request_wait(acquired=True)
    now[0] = 103
    assert ledger.remaining_seconds() == 7
    ledger.end_request()
    assert ledger.snapshot()["request_queue_idle_seconds"] == 100


def test_mismatched_served_identity_is_metered_without_retry():
    calls = []

    class Model:
        def __call__(self, messages, **kwargs):
            calls.append(messages)
            return ChatMessage(role="assistant", content="response", raw=SimpleNamespace(model="changed"))

    ledger = BudgetLedger()
    model = MeteredModel(RequestPolicyModel(Model(), expected_model="frozen"), ledger, "inference")
    with pytest.raises(ValueError, match="serving identity"):
        model([{"role": "user", "content": "Synthetic"}])
    assert len(calls) == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["records"][0]["request"]["response_model"] == "changed"


def test_context_policy_is_metered_with_the_actual_sent_messages():
    ledger = BudgetLedger()

    class Model:
        def __call__(self, messages, **kwargs):
            return ChatMessage(role="assistant", content="done")

    model = MeteredModel(Model(), ledger, "inference", max_tokens=16,
                         context_window=128, context_margin=16, context_policy="oldest_turns")
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": "past " * 1000}, {"role": "user", "content": "past observation"},
                {"role": "assistant", "content": "current"}, {"role": "user", "content": "new observation"}]
    model(messages)
    assert len(messages) == 5
    assert model.calls[0]["messages"] == messages
    assert ledger.snapshot()["records"][0]["context"]["removed_message_hashes"]
