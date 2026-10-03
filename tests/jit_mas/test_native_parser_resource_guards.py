"""Synthetic parser-contract and hard-resource-stop regressions; no API calls."""
import json
from types import SimpleNamespace

import pytest
from json_repair import repair_json

from jit.meta_agent import MetaReActAgent, _repair_json_string_misuse
from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experiment_methods import _DeadlineModel, _NativeResourceStop, _native_jit
from jit_mas.schemas import PublicTask
from scripts.kernel.runtime import AgentRuntime
from scripts.models.base import ChatMessage


def parser_source(import_line="from json_repair import repair_json", expression="repair_json(text)"):
    return (import_line + "\ndef parse(text):\n"
            f"    parsed = {expression}\n"
            "    if isinstance(parsed, dict):\n"
            "        return parsed.get('tools', [])\n"
            "    elif isinstance(parsed, list):\n"
            "        return parsed\n"
            "    else:\n"
            "        return []\n")


def test_installed_repair_default_drops_all_tools_in_object_only_parser():
    raw = json.dumps({"think": "synthetic", "tools": [{"name": "final_answer", "arguments": {"answer": "artifact"}}]})
    repaired = repair_json(raw)
    assert isinstance(repaired, str) and not isinstance(repaired, (dict, list))
    assert repair_json(raw, return_objects=True)["tools"] == json.loads(repaired)["tools"]
    assert len(json.loads(repaired)["tools"]) == 1


@pytest.mark.parametrize("import_line,expression", [
    ("from json_repair import repair_json", "repair_json(text)"),
    ("from json_repair import repair_json as repair", "repair(text)"),
    ("import json_repair", "json_repair.repair_json(text)"),
    ("import json_repair as jr", "jr.repair_json(text, return_objects=False)"),
    ("from json_repair import repair_json", "repair_json(text, False)"),
])
def test_direct_default_string_object_consumption_is_rejected(import_line, expression):
    assert _repair_json_string_misuse(parser_source(import_line, expression), "action.py") == [3]


@pytest.mark.parametrize("source", [
    parser_source(expression="repair_json(text, return_objects=True)"),
    parser_source(expression="repair_json(text, True)"),
    parser_source("import json\nfrom json_repair import repair_json", "json.loads(repair_json(text))"),
    parser_source(expression="repair_json(text, return_objects=bool(text))"),
    parser_source(expression="repair_json(text, **{})"),
    parser_source("from another_package import repair_json"),
    parser_source().replace("def parse(text):", "def parse(text, repair_json):"),
    parser_source().replace("def parse(text):", "def repair_json(text):\n    return {}\ndef parse(text):"),
    parser_source().replace("    else:\n        return []", "    else:\n        import json\n        return json.loads(parsed).get('tools', [])"),
    parser_source().replace("isinstance(parsed, dict)", "isinstance(parsed, (str, dict))"),
    "from json_repair import repair_json\ndef parse(text):\n    return repair_json(text)\n",
])
def test_valid_decoding_dynamic_options_and_shadowed_bindings_are_accepted(source):
    assert _repair_json_string_misuse(source, "action.py") == []


def test_common_static_validator_rejects_parser_without_executing_generated_code(tmp_path):
    source = parser_source() + "\nraise RuntimeError('MUST NOT EXECUTE')\n"
    for filename in ("memory.py", "planning.py", "action.py", "tool_policy.py"):
        (tmp_path / filename).write_text(source if filename == "action.py" else "pass\n")
    (tmp_path / "prompt.yaml").write_text("system_prompt: synthetic\nplanning: {}\nsummary: {}\nfinal_answer: {}\nstep: {}\n")
    report = MetaReActAgent._static_harness_checks(SimpleNamespace(workspace_dir=tmp_path))
    assert "action.py:3" in report and "repair_json defaults to a JSON string" in report


class Provider:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.actual_calls = 0

    def __call__(self, messages, **kwargs):
        self.actual_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ChatMessage(role="assistant", content="synthetic artifact")

    def get_token_counts(self):
        return {"input_token_count": 10, "output_token_count": 5}


def swallowing_action(model):
    caught = 0
    for _ in range(100):
        try:
            return model([{"role": "user", "content": "synthetic"}]), caught
        except Exception:
            caught += 1
    pytest.fail("Generated Exception handler must not spin through exhausted resources")


def test_token_exhaustion_escapes_generated_exception_retry_and_preserves_failed_usage(tmp_path):
    provider = Provider([RuntimeError("synthetic transport failure")])
    ledger = BudgetLedger(max_calls=None, max_tokens=160, timeout_seconds=900)
    leaf = MeteredModel(provider, ledger, "inference", "native-executor", 16)
    snapshots = []

    def persist():
        snapshots.append(ledger.snapshot())
        (tmp_path / "budget.json").write_text(json.dumps(snapshots[-1]))

    model = _DeadlineModel(leaf, 900, ledger=ledger, on_settled=persist)
    with pytest.raises(_NativeResourceStop) as caught:
        swallowing_action(model)
    assert isinstance(caught.value.error, BudgetExceeded)
    assert provider.actual_calls == 1 and len(snapshots) == 2
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["model_calls"] == 1 and saved["tokens"] > 0 and saved["reserved_tokens"] == 0
    assert saved["records"][0]["error"] == "RuntimeError" and saved["records"][0]["estimated"]


@pytest.mark.parametrize("ledger_expired", [False, True])
def test_deadline_escapes_generated_exception_retry_before_any_provider_call(ledger_expired):
    provider = Provider()
    ledger = BudgetLedger(max_calls=None, max_tokens=10000, timeout_seconds=0 if ledger_expired else 900)
    callbacks = []
    model = _DeadlineModel(MeteredModel(provider, ledger, "inference", max_tokens=16), 900,
                           ledger=ledger, on_settled=lambda: callbacks.append(ledger.snapshot()))
    if not ledger_expired:
        model.deadline = 0
    with pytest.raises(_NativeResourceStop) as caught:
        swallowing_action(model)
    assert isinstance(caught.value.error, TimeoutError)
    assert provider.actual_calls == 0 and len(callbacks) == 1
    assert ledger.snapshot()["model_calls"] == 0


def test_transport_timeout_with_task_time_remaining_keeps_native_algorithm_retry():
    provider = Provider([TimeoutError("single request timeout")])
    ledger = BudgetLedger(max_calls=None, max_tokens=10000, timeout_seconds=900)
    leaf = MeteredModel(provider, ledger, "inference", max_tokens=16)
    callbacks = []
    result, caught = swallowing_action(_DeadlineModel(leaf, 900, ledger=ledger,
                                       on_settled=lambda: callbacks.append(ledger.snapshot())))
    assert result.content == "synthetic artifact" and caught == 1
    assert provider.actual_calls == 2 and len(callbacks) == 2
    assert ledger.snapshot()["model_calls"] == 2 and ledger.snapshot()["reserved_tokens"] == 0


@pytest.mark.parametrize("stop_kind,error_type", [("deadline", TimeoutError), ("tokens", BudgetExceeded)])
def test_native_boundary_restores_normal_failure_and_audit_is_written_before_return(tmp_path, monkeypatch, stop_kind, error_type):
    config = MASConfig(backend="scripted", max_model_calls=None, max_total_tokens=2_000_000,
                       task_timeout=900, execution_timeout=900)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000, timeout_seconds=900)
    pipeline = SimpleNamespace(config=config, tools={})

    def run(runtime, *args, **kwargs):
        runtime.model([{"role": "user", "content": "synthetic tool protocol"}])
        # Callback must persist before the algorithm returns or is interrupted.
        saved = json.loads((tmp_path / "model_calls.json").read_text())
        assert len(saved["generation"]) == len(saved["execution"]) == 1
        assert json.loads((tmp_path / "budget.json").read_text())["model_calls"] == 2
        if stop_kind == "deadline":
            runtime.model.deadline = 0
        else:
            ledger.max_tokens = ledger.snapshot()["tokens"]
        swallowing_action(runtime.model)

    monkeypatch.setattr(AgentRuntime, "run", run)
    with pytest.raises(error_type):
        _native_jit(pipeline, PublicTask(task_id="synthetic", question="Write a synthetic artifact."), ledger, tmp_path)
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["model_calls"] == 2 and saved["reserved_tokens"] == 0
    assert len(json.loads((tmp_path / "model_calls.json").read_text())["execution"]) == 1
