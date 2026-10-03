"""Baseline compatibility with a shared token/time envelope; no API calls."""

import json
from pathlib import Path

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_methods import submit_method
from jit_mas.schemas import PublicTask
from scripts.kernel.runtime import AgentRuntime
from scripts.mas_baseline_methods import run_direct
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


class DirectProvider:
    def __init__(self):
        self.calls = []
        self.ledgers = []

    def create(self, role, agent_id, ledger, stage):
        assert (role, agent_id, stage) == ("exec", "direct", "inference")
        self.ledgers.append(ledger)
        return MeteredModel(self, ledger, stage, agent_id, 8192)

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatMessage(role="assistant", content="One final artifact.")

    def get_token_counts(self):
        return {"input_token_count": 20, "output_token_count": 5}


def test_direct_accepts_null_common_call_cap_and_preserves_one_call(tmp_path):
    provider = DirectProvider()
    outcome = run_direct(PublicTask(task_id="synthetic", question="Write one sentence."), None,
                         provider, lambda _: pytest.fail("Deferred submission must not create a judge"),
                         tmp_path / "direct", {"max_calls": None, "max_tokens": 2_000_000,
                                              "timeout_seconds": 900}, defer_evaluation=True)
    assert outcome["status"] == "submitted_unscored"
    assert outcome["budget"]["model_calls"] == len(provider.calls) == 1
    assert provider.ledgers[0].max_calls is None
    assert provider.ledgers[0].max_tokens == 2_000_000
    assert provider.ledgers[0].timeout_seconds == 900
    assert outcome["budget"]["tokens"] == 25
    assert not (Path(outcome["run_dir"]) / "evaluation.json").exists()


@pytest.mark.parametrize("field,value", [("max_calls", 0), ("max_calls", True),
                                          ("max_calls", 1.5), ("timeout_seconds", 0),
                                          ("timeout_seconds", float("inf")),
                                          ("timeout_seconds", True)])
def test_direct_rejects_invalid_envelopes_before_any_model_call(tmp_path, field, value):
    provider = DirectProvider()
    limits = {"max_calls": None, "max_tokens": 2_000_000, "timeout_seconds": 900, field: value}
    with pytest.raises(ValueError):
        run_direct(PublicTask(task_id="synthetic", question="Write one sentence."), None,
                   provider, lambda _: None, tmp_path / "invalid", limits, defer_evaluation=True)
    assert provider.calls == provider.ledgers == []
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("call_cap", [None, 200])
def test_native_upstream_harness_accepts_integer_step_interface_and_records_origin(tmp_path, monkeypatch, call_cap):
    config = MASConfig(backend="scripted", max_model_calls=call_cap, max_total_tokens=2_000_000,
                       task_timeout=900, execution_timeout=900)
    store = ExperienceStore(tmp_path / "state.sqlite")
    pipeline = make_pipeline(config, store, tmp_path / "unused")
    pipeline.evaluator_factory = lambda _: pytest.fail("Native submission must not create a judge")
    original = AgentRuntime.run
    observed = []

    def probe(runtime, *args, **kwargs):
        observed.append(runtime.max_steps)
        assert type(runtime.max_steps) is int
        assert len(range(runtime.max_steps)) == runtime.max_steps
        assert runtime.config["execution"]["model_call_budget"] == call_cap
        assert runtime.model.ledger.max_calls == call_cap
        assert runtime.model.ledger.timeout_seconds == 900
        return original(runtime, *args, **kwargs)

    monkeypatch.setattr(AgentRuntime, "run", probe)
    try:
        outcome = submit_method(pipeline, "test-deployment", store.snapshot(), method="jit_matched",
                                repeat=0, output_dir=tmp_path / "native")
        assert outcome["status"] == "submitted_unscored"
        expected_steps = 2_000_000 if call_cap is None else call_cap
        assert observed == [expected_steps]
        assert outcome["budget"]["reserved_tokens"] == 0
        assert outcome["budget"]["model_calls"] >= 2
        harness = json.loads((Path(outcome["run_dir"]) / "harness.json").read_text())
        recorded = harness["runtime_budget"]
        assert recorded["max_steps_interface"] == expected_steps
        assert recorded["model_call_cap"] == call_cap
        assert recorded["independent_step_cap"] is False
        assert recorded["max_steps_origin"] == ("derived_from_existing_token_envelope" if call_cap is None
                                                  else "configured_model_call_cap")
        assert not (Path(outcome["run_dir"]) / "evaluation.json").exists()
    finally:
        store.close()
