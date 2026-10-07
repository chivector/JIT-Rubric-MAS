"""Offline tests for provider failure policy; no network requests are made."""

import pytest
from types import SimpleNamespace
import httpx
from openai import APIConnectionError, APIStatusError

from scripts.models.base import Model
from scripts.models.openai_server import (
    OpenAIServerModel, _estimate_input_tokens, _estimate_output_tokens,
)


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


def test_fallback_token_estimate_includes_tool_schema_and_native_arguments():
    prompt = {"messages": [{"role": "user", "content": "Use the tool"}],
              "tools": [{"type": "function", "function": {
                  "name": "lookup", "parameters": {"type": "object"}}}]}
    with_tools = _estimate_input_tokens(prompt)
    without_tools = _estimate_input_tokens({"messages": prompt["messages"]})
    assert with_tools > without_tools

    class Function:
        name = "lookup"
        arguments = '{"query":"token audit"}'

    class Call:
        function = Function()

    class Message:
        content = ""
        reasoning_content = "checking"
        tool_calls = [Call()]

    assert _estimate_output_tokens(Message()) > 0


def _status_error(status_code):
    request = httpx.Request("POST", "https://offline.example/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(f"HTTP {status_code}", response=response, body=None)


class _CompletionMessage:
    content = "ok"
    tool_calls = None
    reasoning = None
    reasoning_content = None

    def model_dump(self, include=None):
        return {"role": "assistant", "content": self.content}


class _CompletionResponse:
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
    choices = [SimpleNamespace(message=_CompletionMessage())]


def _fixture_model(events, *, max_tokens=16000, max_attempts=3):
    model = OpenAIServerModel.__new__(OpenAIServerModel)
    Model.__init__(model, max_tokens=max_tokens)
    model.model_id = "offline-fixture"
    model.max_attempts = max_attempts
    model.custom_role_conversions = None
    calls = []

    def create(**kwargs):
        calls.append(dict(kwargs))
        event = events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return model, calls


def test_http_507_retries_with_reduced_output_cap(monkeypatch):
    sleeps = []
    monkeypatch.setattr("scripts.models.openai_server.time.sleep", sleeps.append)
    model, calls = _fixture_model([_status_error(507), _CompletionResponse()])

    result = model([{"role": "user", "content": "Offline request"}])

    assert result.content == "ok"
    assert [call["max_tokens"] for call in calls] == [16000, 8000]
    assert sleeps == [30]
    assert model.last_retry_metadata["attempts"][0]["status_code"] == 507
    assert model.last_retry_metadata["attempts"][0]["next_max_tokens"] == 8000


def test_http_507_retry_floor_and_auth_fail_fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr("scripts.models.openai_server.time.sleep", sleeps.append)
    model, calls = _fixture_model(
        [_status_error(507), _status_error(507), _status_error(507)]
    )
    with pytest.raises(APIStatusError) as caught:
        model([{"role": "user", "content": "Offline request"}])
    assert caught.value.status_code == 507
    assert [call["max_tokens"] for call in calls] == [16000, 8000, 4096]
    assert sleeps == [30, 30]

    sleeps.clear()
    model, calls = _fixture_model([_status_error(401), _CompletionResponse()])
    with pytest.raises(APIStatusError) as caught:
        model([{"role": "user", "content": "Offline request"}])
    assert caught.value.status_code == 401
    assert len(calls) == 1
    assert sleeps == []
