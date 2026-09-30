from .types import (
    Message, ToolCall, StepRecord, PlanState, SummaryState,
    TaskInput, MemoryView, ToolSelection, Directive, RunResult, RuntimeContext,
)
from .protocols import BaseMemory, BasePlanning, BaseAction, BaseToolPolicy
from .loader import load_harness

__all__ = [
    "Message", "ToolCall", "StepRecord", "PlanState", "SummaryState", "TaskInput",
    "MemoryView", "ToolSelection", "Directive", "RunResult", "RuntimeContext",
    "BaseMemory", "BasePlanning", "BaseAction", "BaseToolPolicy", "AgentRuntime", "load_harness",
]


def __getattr__(name):
    # Model clients import kernel utilities before the runtime's client import is ready.
    if name == "AgentRuntime":
        from .runtime import AgentRuntime
        globals()[name] = AgentRuntime
        return AgentRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
