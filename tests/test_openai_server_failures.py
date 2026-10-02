"""Offline tests for provider failure policy; no network requests are made."""

import pytest

from scripts.models.openai_server import OpenAIServerModel


def test_failure_threshold_can_return_to_task_retry_layer(monkeypatch):
    monkeypatch.setattr(OpenAIServerModel, "_consecutive_api_failures", 0)
    monkeypatch.setattr(OpenAIServerModel, "_max_consecutive_api_failures", 2)
    monkeypatch.setenv("MODULAR_AGENT_API_FAILURE_ACTION", "raise")
    error = RuntimeError("transient gateway failure")

    # Below and at the threshold both calls remain catchable by the caller;
    # the second call must not terminate the process with SystemExit.
    OpenAIServerModel._record_api_failure_and_maybe_exit(error)
    OpenAIServerModel._record_api_failure_and_maybe_exit(error)
    assert OpenAIServerModel._consecutive_api_failures == 2


def test_unknown_failure_action_preserves_fail_fast_default(monkeypatch):
    monkeypatch.setattr(OpenAIServerModel, "_consecutive_api_failures", 0)
    monkeypatch.setattr(OpenAIServerModel, "_max_consecutive_api_failures", 1)
    monkeypatch.setenv("MODULAR_AGENT_API_FAILURE_ACTION", "invalid")
    with pytest.raises(SystemExit):
        OpenAIServerModel._record_api_failure_and_maybe_exit(RuntimeError("gateway"))
