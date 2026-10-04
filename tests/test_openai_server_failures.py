"""Offline tests for provider failure policy; no network requests are made."""

import pytest
from types import SimpleNamespace
import httpx
from openai import APIConnectionError

from scripts.models.base import Model
from scripts.models.openai_server import OpenAIServerModel


def test_transport_trust_env_is_client_only_and_never_request_body(monkeypatch):
    import openai

    captured = {}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            captured["http_client_kwargs"] = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    monkeypatch.setattr("httpx.Client", FakeHttpClient)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    model = OpenAIServerModel(
        model_id="judge",
        api_base="http://gateway.invalid/v1",
        api_key="test-key",
        http_trust_env=False,
    )

    assert captured["http_client_kwargs"]["trust_env"] is False
    assert captured["openai_kwargs"]["http_client"] is not None
    # The transport switch is consumed by the client constructor and cannot
    # leak into Model kwargs (which become completion request options).
    assert "http_trust_env" not in model.kwargs


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


def test_api_connection_failure_is_re_raised_at_threshold_for_task_recovery(monkeypatch):
    monkeypatch.setattr(OpenAIServerModel, "_consecutive_api_failures", 0)
    monkeypatch.setattr(OpenAIServerModel, "_max_consecutive_api_failures", 1)
    monkeypatch.setenv("MODULAR_AGENT_API_FAILURE_ACTION", "raise")
    error = APIConnectionError(request=httpx.Request("POST", "https://offline.example/v1"))

    def fail(**_kwargs):
        raise error

    model = OpenAIServerModel.__new__(OpenAIServerModel)
    Model.__init__(model)
    model.model_id = "offline-fixture"
    model.max_attempts = 1
    model.custom_role_conversions = None
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(APIConnectionError) as caught:
        model([{"role": "user", "content": "Offline request"}])
    assert caught.value is error
