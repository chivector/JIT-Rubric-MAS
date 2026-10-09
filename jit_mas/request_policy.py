"""Frozen context limits and process-wide request concurrency for native runs."""

from __future__ import annotations

import copy
import threading
import time

from scripts.kernel.token_counter import count_tokens_messages

from .schemas import digest


class ContextLimitExceeded(ValueError):
    pass


_gates = {}
_gates_lock = threading.Lock()


def request_gate(limit, *, endpoint=None):
    if limit is None:
        return None
    key = limit if endpoint is None else (endpoint.rstrip("/"), limit)
    with _gates_lock:
        if key not in _gates:
            _gates[key] = threading.BoundedSemaphore(limit)
        return _gates[key]


def prepare_context(messages, *, output_tokens, context_window=None, margin=2048,
                    policy="reject"):
    prepared = copy.deepcopy(messages)
    original_tokens = count_tokens_messages(prepared)
    removed = []
    if context_window is not None:
        input_limit = context_window - output_tokens - margin
        if input_limit <= 0:
            raise ContextLimitExceeded("Output reservation and margin exhaust the context window")
        while count_tokens_messages(prepared) > input_limit:
            if policy != "oldest_turns" or len(prepared) <= 4:
                raise ContextLimitExceeded("Required initial or current messages exceed the frozen context window")
            removed.append(digest(prepared.pop(2)))
    return prepared, {"estimator": "cl100k_base_messages_v1", "context_window": context_window,
                      "margin": margin, "policy": policy, "original_input_tokens": original_tokens,
                      "sent_input_tokens": count_tokens_messages(prepared),
                      "removed_message_hashes": removed, "sent_messages_hash": digest(prepared)}


class RequestPolicyModel:
    def __init__(self, model, *, gate=None, ledger=None, timeout=180, expected_model=None):
        self.model, self.gate, self.ledger = model, gate, ledger
        self.timeout, self.expected_model = timeout, expected_model
        self.last_request_metadata = {}

    def __call__(self, messages, **kwargs):
        started = time.monotonic()
        if self.ledger is not None:
            self.ledger.begin_request_wait()
        acquired = self.gate is None or self.gate.acquire(timeout=self.timeout)
        if not acquired:
            if self.ledger is not None:
                self.ledger.end_request_wait(acquired=False)
            raise TimeoutError("Task timeout while waiting for request capacity")
        waited = time.monotonic() - started
        if self.ledger is not None:
            self.ledger.end_request_wait(acquired=True)
        self.last_request_metadata = {"queue_seconds": waited}
        provider_invoked = False
        try:
            remaining = self.ledger.remaining_seconds() if self.ledger is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Task wall-clock budget exhausted before request")
            kwargs["timeout"] = min(kwargs.get("timeout", self.timeout), self.timeout,
                                    remaining if remaining is not None else self.timeout)
            provider_invoked = True
            response = self.model(messages, **kwargs)
            raw = getattr(response, "raw", None)
            identity = {"response_model": getattr(raw, "model", None),
                        "system_fingerprint": getattr(raw, "system_fingerprint", None),
                        "response_id": getattr(raw, "id", None)}
            choices = getattr(raw, "choices", None)
            if choices:
                finish_reason = getattr(choices[0], "finish_reason", None)
                if finish_reason is not None:
                    identity["finish_reason"] = finish_reason
            self.last_request_metadata.update(identity)
            if self.expected_model is not None and identity["response_model"] != self.expected_model:
                raise ValueError("Provider response model differs from the frozen serving identity")
            return response
        finally:
            retry_metadata = getattr(self.model, "last_retry_metadata", None) if provider_invoked else None
            if retry_metadata is not None:
                self.last_request_metadata["transport_retry"] = copy.deepcopy(retry_metadata)
            if self.ledger is not None:
                self.ledger.end_request()
            if self.gate is not None:
                self.gate.release()

    @property
    def _native_tool_registry(self):
        return getattr(self.model, "_native_tool_registry", {})

    @_native_tool_registry.setter
    def _native_tool_registry(self, value):
        self.model._native_tool_registry = value

    def __getattr__(self, name):
        return getattr(self.model, name)
