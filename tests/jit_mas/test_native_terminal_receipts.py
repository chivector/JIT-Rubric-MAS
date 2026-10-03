"""Native terminal contract checks use only public synthetic tool receipts."""

from types import SimpleNamespace

import pytest

from jit_mas import experiment_methods as methods
from scripts.kernel.types import RunResult, StepRecord, ToolCall
from scripts.tools.base import FinalAnswerTool


class Ledger:
    def snapshot(self):
        return {"model_calls": 6}


def receipt(arguments):
    events = []
    registry = SimpleNamespace(get=lambda name: FinalAnswerTool())
    observation = methods._native_final_tool_receipt(registry, arguments, Ledger(), events)
    return observation, events


def final_result(answer="Synthetic final artifact.", reason="final_answer_called"):
    step = StepRecord(tool_calls=[ToolCall(name="final_answer", arguments={"answer": answer})],
                      action_output=answer, observations=answer)
    return RunResult(answer=answer, terminated_reason=reason, trajectory=[step])


def test_confirmed_public_final_tool_alias_is_normalized_without_reauthoring_answer():
    result = final_result()
    original = result.answer
    observation, events = receipt({"answer": original})
    methods._normalize_native_final_termination(result, events, 6)
    assert result.answer == observation == original
    assert result.terminated_reason == "final_answer"
    assert result.metadata["native_terminal"]["normalized"] is True


def test_real_tool_error_cannot_be_normalized_as_a_final_submission():
    error, events = receipt({"args": "Synthetic attempted artifact."})
    result = final_result(error)
    result.trajectory[0].tool_calls[0].arguments = {"args": "Synthetic attempted artifact."}
    methods._normalize_native_final_termination(result, events, 6)
    assert events[0]["success"] is False and events[0]["error_type"] == "TypeError"
    assert result.terminated_reason == "final_answer_called"
    assert result.metadata["native_terminal"]["normalized"] is False


@pytest.mark.parametrize("change", ["no_receipt", "later_calls", "changed_answer", "step_error", "wrong_tool", "wrong_observation"])
def test_incomplete_or_contradictory_public_evidence_does_not_normalize(change):
    result = final_result()
    _, events = receipt({"answer": result.answer})
    calls = 6
    if change == "no_receipt":
        events = []
    elif change == "later_calls":
        calls = 7
    elif change == "changed_answer":
        result.answer = "Different artifact."
    elif change == "step_error":
        result.trajectory[0].error = ValueError("synthetic")
    elif change == "wrong_tool":
        result.trajectory[0].tool_calls[-1].name = "unknown"
    else:
        result.trajectory[0].observations = "Error executing tool 'final_answer': synthetic"
    methods._normalize_native_final_termination(result, events, calls)
    assert result.terminated_reason == "final_answer_called"
    assert result.metadata["native_terminal"]["normalized"] is False


def test_earlier_recovered_tool_error_does_not_negate_later_confirmed_submission():
    result = final_result()
    result.trajectory.insert(0, StepRecord(observations="Error executing tool '': missing"))
    _, events = receipt({"answer": result.answer})
    methods._normalize_native_final_termination(result, events, 6)
    assert result.terminated_reason == "final_answer"


@pytest.mark.parametrize("reason", ["error", "max_steps", "forced_final_answer", "final_answer"])
def test_only_observed_alias_is_normalized(reason):
    result = final_result(reason=reason)
    _, events = receipt({"answer": result.answer})
    methods._normalize_native_final_termination(result, events, 6)
    assert result.terminated_reason == reason
    assert result.metadata["native_terminal"]["normalized"] is False


def test_serialized_public_terminal_trace_requires_the_same_runtime_receipt():
    answer = "A complete synthetic serialized artifact."
    result = RunResult(answer=answer, terminated_reason="final_answer_called", trajectory=[{
        "tool_calls": [{"name": "final_answer", "arguments": {"answer": answer}}],
        "error": None, "action_output": answer, "observations": answer}])
    observation, events = receipt({"answer": answer})
    methods._normalize_native_final_termination(result, events, 6)
    assert result.answer == answer == observation
    assert result.terminated_reason == "final_answer"
    assert result.metadata["native_terminal"]["verified_final_tool_submission"] is True
