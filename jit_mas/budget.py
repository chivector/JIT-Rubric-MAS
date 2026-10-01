"""One thread-safe call/token ledger; sub-run serialization never adds charges."""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections import defaultdict
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, max_calls=200, max_tokens=2_000_000, max_tool_calls=100, *, timeout_seconds=None):
        finite = [value for value in (max_calls, max_tokens, max_tool_calls) if value is not None]
        if any(not isinstance(value, int) or value < 0 for value in finite):
            raise ValueError("Budgets cannot be negative")
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.max_tool_calls = max_tool_calls
        self._lock = threading.RLock()
        self._calls = self._tokens = self._reserved = self._tools = 0
        self._pending = {}
        self._records = []
        self._started = time.monotonic()
        self.timeout_seconds = timeout_seconds
        self._waiting_requests = self._active_requests = 0
        self._queue_pause_started = None
        self._queue_pause_seconds = 0.0

    def remaining_seconds(self):
        if self.timeout_seconds is None:
            return None
        with self._lock:
            paused = self._queue_pause_seconds
            if self._queue_pause_started is not None:
                paused += time.monotonic() - self._queue_pause_started
            return max(0.0, self.timeout_seconds - (time.monotonic() - self._started - paused))

    def begin_request_wait(self):
        with self._lock:
            self._waiting_requests += 1
            if not self._active_requests and self._queue_pause_started is None:
                self._queue_pause_started = time.monotonic()

    def end_request_wait(self, *, acquired):
        with self._lock:
            self._waiting_requests -= 1
            if acquired:
                self._active_requests += 1
            if self._queue_pause_started is not None and (self._active_requests or not self._waiting_requests):
                self._queue_pause_seconds += time.monotonic() - self._queue_pause_started
                self._queue_pause_started = None

    def end_request(self):
        with self._lock:
            self._active_requests -= 1
            if not self._active_requests and self._waiting_requests and self._queue_pause_started is None:
                self._queue_pause_started = time.monotonic()

    def reserve(self, stage, agent_id, input_bound, output_bound):
        if not isinstance(input_bound, int) or not isinstance(output_bound, int) or input_bound < 0 or output_bound <= 0:
            raise ValueError("Input bound must be nonnegative and output bound positive integers")
        total = input_bound + output_bound
        with self._lock:
            if self.remaining_seconds() == 0:
                raise TimeoutError("Task wall-clock budget exhausted")
            if self.max_calls is not None and self._calls >= self.max_calls:
                raise BudgetExceeded("Team model-call budget exhausted")
            if self._tokens + self._reserved + total > self.max_tokens:
                raise BudgetExceeded("Team token budget exhausted before request")
            ticket = uuid.uuid4().hex
            self._calls += 1
            self._reserved += total
            self._pending[ticket] = (stage, agent_id, input_bound, output_bound)
            return ticket

    def settle(self, ticket, input_tokens=None, output_tokens=None, *, error="", elapsed=0, metadata=None):
        with self._lock:
            stage, agent_id, inp, out = self._pending.pop(ticket)
            self._reserved -= inp + out
            estimated = input_tokens is None or output_tokens is None
            used_in = inp if input_tokens is None else max(0, int(input_tokens))
            used_out = out if output_tokens is None else max(0, int(output_tokens))
            self._tokens += used_in + used_out
            self._records.append({"call_id": ticket, "kind": "model", "stage": stage,
                                  "agent_id": agent_id, "input_tokens": used_in,
                                  "output_tokens": used_out, "estimated": estimated,
                                  "wall_seconds": elapsed, "error": error,
                                  **copy.deepcopy(metadata or {})})
            if self._tokens + self._reserved > self.max_tokens:
                raise BudgetExceeded("Provider usage exceeded reserved token bound")

    def charge_tool(self, stage="execution", agent_id="", tool_name=""):
        with self._lock:
            if self.remaining_seconds() == 0:
                raise TimeoutError("Task wall-clock budget exhausted")
            if self.max_tool_calls is not None and self._tools >= self.max_tool_calls:
                raise BudgetExceeded("Team tool budget exhausted")
            self._tools += 1
            self._records.append({"kind": "tool", "stage": stage,
                                  "agent_id": agent_id, "tool_name": tool_name})

    def charge_communication(self, amount, stage="execution"):
        with self._lock:
            self._records.append({"kind": "communication", "stage": stage,
                                  "bytes": max(0, int(amount))})

    def snapshot(self):
        with self._lock:
            groups = defaultdict(lambda: {"model_calls": 0, "tokens": 0,
                                          "tool_calls": 0, "communication_bytes": 0,
                                          "cost": None})
            for row in self._records:
                group = groups[row["stage"]]
                if row["kind"] == "model":
                    group["model_calls"] += 1
                    group["tokens"] += row["input_tokens"] + row["output_tokens"]
                elif row["kind"] == "tool":
                    group["tool_calls"] += 1
                else:
                    group["communication_bytes"] += row["bytes"]
            return {"model_calls": self._calls, "tokens": self._tokens,
                    "reserved_tokens": self._reserved, "tool_calls": self._tools,
                    "wall_seconds": time.monotonic() - self._started,
                    "request_queue_idle_seconds": self._queue_pause_seconds,
                    "task_timeout": self.timeout_seconds, "cost": None,
                    "by_stage": dict(groups), "records": copy.deepcopy(self._records)}


class MeteredModel:
    def __init__(self, model, ledger: BudgetLedger, stage: str, agent_id="", max_tokens=4096, *,
                 context_window=None, context_margin=2048, context_policy="reject"):
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model, self.ledger, self.stage = model, ledger, stage
        self.agent_id, self.max_tokens = agent_id, max_tokens
        self._lock = threading.RLock()
        self.calls = []
        self.context_window, self.context_margin = context_window, context_margin
        self.context_policy = context_policy

    def __call__(self, messages, **kwargs):
        limit = min(int(kwargs.pop("max_tokens", self.max_tokens)), self.max_tokens)
        if limit <= 0:
            raise ValueError("max_tokens must be positive")
        context_record = None
        if self.context_window is not None:
            from .request_policy import prepare_context
            prepared, context_record = prepare_context(messages, output_tokens=limit,
                context_window=self.context_window, margin=self.context_margin, policy=self.context_policy)
            messages[:] = prepared
        # UTF-8 bytes plus message framing conservatively bounds common tokenizers.
        bound = len(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8")) + 64
        with self._lock:
            ticket = self.ledger.reserve(self.stage, self.agent_id, bound, limit)
            start = time.monotonic()
            try:
                response = self.model(messages, max_tokens=limit, **kwargs)
            except BaseException as exc:
                self.ledger.settle(ticket, error=type(exc).__name__, elapsed=time.monotonic() - start,
                    metadata={"context": context_record, "request": copy.deepcopy(
                        getattr(self.model, "last_request_metadata", {}))})
                raise
            try:
                counts = self.model.get_token_counts() if hasattr(self.model, "get_token_counts") else {}
                counts = counts or {}
            except Exception:
                counts = {}
            content = getattr(response, "content", str(response))
            self.ledger.settle(ticket, counts.get("input_token_count"),
                               counts.get("output_token_count"), elapsed=time.monotonic() - start,
                               metadata={"context": context_record, "request": copy.deepcopy(
                                   getattr(self.model, "last_request_metadata", {}))})
            self.calls.append({"messages": copy.deepcopy(messages), "content": content,
                               "call_id": ticket,
                               **({"context": context_record} if context_record is not None else {}),
                               **({"request": copy.deepcopy(self.model.last_request_metadata)}
                                  if hasattr(self.model, "last_request_metadata") else {})})
            return response

    def __getattr__(self, name):
        return getattr(self.model, name)
