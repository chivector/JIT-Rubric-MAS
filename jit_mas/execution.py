"""Single-pass and cooperative shared-ledger team primitives for JIT Action modules."""

from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from scripts.kernel.loader import load_harness
from scripts.kernel.monitoring import AgentLogger, LogLevel
from scripts.kernel.protocols import BaseAction, BaseMemory, BasePlanning, BaseToolPolicy
from scripts.kernel.runtime import AgentRuntime
from scripts.kernel.types import (
    Directive, MemoryView, PlanState, RunResult, RuntimeContext, StepRecord,
    SummaryState, TaskInput, ToolCall, ToolSelection,
)
from scripts.tools.registry import ToolRegistry
from jit_mas.experience import experience_applicability
from jit_mas.schemas import AgentPoolSnapshot, AgentSpec, PublicTask, RubricGraph, TeamSpec, utc_now


def _data(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_data(value), sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _bounded_call(call: Callable, timeout: float, *args, deadline_remaining=None, **kwargs):
    """Bound waiting on cooperative API/tool clients, without blocking shutdown.

    This is not an isolation boundary: a timed-out local callable can still be
    running. Untrusted generated code is therefore disabled unless unsafe-local
    was explicitly selected.
    """
    result = queue.Queue(maxsize=1)

    def invoke():
        try:
            result.put((True, call(*args, **kwargs)))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    while True:
        remaining = deadline_remaining() if deadline_remaining is not None else timeout
        if remaining <= 0:
            raise TimeoutError(f"call exceeded {timeout:g} seconds")
        try:
            ok, value = result.get(timeout=max(0.001, min(remaining, 0.1)
                                              if deadline_remaining is not None else remaining))
            break
        except queue.Empty as exc:
            if deadline_remaining is None:
                raise TimeoutError(f"call exceeded {timeout:g} seconds") from exc
    if not ok:
        if isinstance(value, (KeyboardInterrupt, SystemExit)):
            raise RuntimeError(f"client terminated: {value}") from value
        if not isinstance(value, Exception):
            raise RuntimeError(str(value)) from value
        raise value
    return value


class TeamMemory(BaseMemory):
    """Full observed history; an independent instance is created for each agent."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.steps = []
        self.messages = []

    def initialize(self, system_prompt, task):
        self.steps = []
        self.messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": task.task}]

    def build_context(self, plan=None):
        return MemoryView(messages=copy.deepcopy(self.messages))

    def update(self, step):
        self.steps.append(step)
        if step.model_input_messages is not None:
            self.messages = copy.deepcopy(step.model_input_messages)
        if step.model_output_messages is not None:
            output = step.model_output_messages
            self.messages.append({"role": "assistant", "content":
                                  str(getattr(output, "content", output) or "")})
        if step.observations:
            self.messages.append({"role": "user", "content": step.observations})

    def update_plan(self, plan):
        self.messages.append({"role": "user", "content": plan.plan})

    def update_summary(self, summary):
        self.messages.append({"role": "user", "content": summary.summary})

    def get_all_steps(self):
        return list(self.steps)


class TeamPlanning(BasePlanning):
    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.team = None

    def bind_team(self, team):
        self.team = _data(team)

    def init_plan(self, task, memory_view, tool_schemas, model):
        if self.team is None:
            raise ValueError("TeamSpec must be bound before execution")
        return PlanState(plan=json.dumps(self.team, ensure_ascii=False))

    def should_replan(self, step_number, step):
        return False

    def update_plan(self, task, step_number, memory_view, model):
        return SummaryState(summary="TeamSpec is frozen for this run.")

    def get_directive(self):
        return Directive(text="Follow the bound responsibilities and dependency DAG.")


class TeamToolPolicy(BaseToolPolicy):
    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.catalog = {}

    def initialize(self, tool_catalog, enable_skills=False):
        self.catalog = dict(tool_catalog)

    def select_tools(self, task, step_number, memory_view, plan=None):
        return ToolSelection(tools=dict(self.catalog))


@dataclass
class TeamServices:
    model_factory: Callable[[str], Any]
    public_task: Any
    rubrics: Any = None
    experiences: list = field(default_factory=list)
    ledger: Any = None
    timeout_seconds: float = 60.0
    run_id: str = field(default_factory=lambda: "run_" + uuid.uuid4().hex)
    events: list = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock)
    calls: int = 0
    cancelled: Any = field(default_factory=threading.Event)
    artifacts: dict = field(default_factory=dict)
    called_agents: set = field(default_factory=set)
    started: bool = False
    shared_ledger: dict = field(default_factory=dict)
    execution_mode: str = "single_pass"
    started_at: float = field(default_factory=time.monotonic)
    call_counts: dict = field(default_factory=dict)
    agent_pool: AgentPoolSnapshot | None = None
    task_seconds_at_start: float | None = field(init=False, default=None)

    def __post_init__(self):
        if self.ledger is not None and callable(getattr(self.ledger, "remaining_seconds", None)):
            self.task_seconds_at_start = self.ledger.remaining_seconds()

    def remaining_seconds(self):
        if self.task_seconds_at_start is not None:
            task_remaining = self.ledger.remaining_seconds()
            elapsed = self.task_seconds_at_start - task_remaining
            return max(0.0, min(self.timeout_seconds, self.task_seconds_at_start) - elapsed)
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started_at))

    def event(self, agent_id, kind, content, *, parents=(), recipient="", source=""):
        with self.lock:
            event = {
                "schema_version": "1.0", "run_id": self.run_id,
                "agent_id": agent_id, "event_id": f"{self.run_id}:e{len(self.events) + 1}",
                "timestamp": utc_now(), "kind": kind, "source": source or kind,
                "content": copy.deepcopy(content), "content_hash": content_hash(content),
                "artifact_version": 1, "parent_event_ids": list(parents),
                "recipient": recipient, "locator": f"events/{len(self.events)}",
            }
            self.events.append(event)
            return event["event_id"]

    def reserve_call(self, limit, agent_id, agent_limit=None):
        with self.lock:
            if self.cancelled.is_set():
                raise TimeoutError("team cancelled after timeout")
            if self.execution_mode == "single_pass" and agent_id in self.called_agents:
                raise RuntimeError("Single-pass execution permits only one model call per role")
            if agent_limit is not None and self.call_counts.get(agent_id, 0) >= agent_limit:
                raise RuntimeError(f"AgentSpec.max_calls exhausted for agent '{agent_id}'")
            if limit is not None and self.calls >= limit:
                raise RuntimeError("TeamSpec.total_max_calls exhausted")
            if self.remaining_seconds() <= 0:
                self.cancelled.set()
                raise TimeoutError("team execution timeout")
            self.called_agents.add(agent_id)
            self.calls += 1
            self.call_counts[agent_id] = self.call_counts.get(agent_id, 0) + 1


def validate_team(team, public_task=None):
    team = _data(team)
    agents = team["agents"]
    ids = [a["agent_id"] for a in agents]
    if not agents or len(set(ids)) != len(ids):
        raise ValueError("TeamSpec requires unique nonempty agents")
    if team["synthesizer_id"] not in ids:
        raise ValueError("unknown synthesizer")
    if team.get("max_parallel", 2) < 1 or (team.get("total_max_calls", 16) is not None and team["total_max_calls"] < 1):
        raise ValueError("team concurrency and call budgets must be positive")
    allowed = set(_data(public_task)["tools"]) if public_task is not None else None
    dependencies = {}
    for a in agents:
        deps = set(a.get("depends_on", []))
        if not deps.issubset(ids) or a["agent_id"] in deps:
            raise ValueError("unknown or self-referential agent dependency")
        if (a.get("max_calls", 3) is not None and a["max_calls"] < 1) or a.get("max_tokens", 4096) < 1:
            raise ValueError("agent budgets must be positive")
        if allowed is not None and not set(a.get("tools", [])).issubset(allowed):
            raise ValueError("agent requested tools outside PublicTask allowlist")
        dependencies[a["agent_id"]] = deps
    remaining = dict(dependencies)
    done = set()
    while remaining:
        ready = {aid for aid, deps in remaining.items() if deps.issubset(done)}
        if not ready:
            raise ValueError("execution dependencies must form a DAG")
        done.update(ready)
        remaining = {aid: deps for aid, deps in remaining.items() if aid not in ready}
    # All workers must be upstream of the unique final submission.
    pending = list(dependencies[team["synthesizer_id"]])
    ancestors = set()
    while pending:
        aid = pending.pop()
        if aid not in ancestors:
            ancestors.add(aid)
            pending.extend(dependencies[aid])
    if ancestors != set(ids) - {team["synthesizer_id"]}:
        raise ValueError("every worker must feed the final synthesizer through dependencies")
    return team


class ResponseProtocolError(ValueError):
    """Invalid output fails this task attempt without recalling a role."""


def _validate_completion_fields(parsed):
    if parsed.get("answer") is not None and (
            not isinstance(parsed["answer"], str) or not parsed["answer"].strip()):
        raise ResponseProtocolError("answer must be a nonempty string")
    if not isinstance(parsed.get("checkpoints", {}), dict):
        raise ResponseProtocolError("checkpoints must be an object")
    for check in parsed.get("checkpoints", {}).values():
        if isinstance(check, bool):
            continue
        if (not isinstance(check, dict)
                or not isinstance(check.get("status"), str)
                or check.get("status") not in {"completed", "passed", "failed", "unverified", "not_applicable"}
                or not isinstance(check.get("reason"), str) or not check["reason"].strip()
                or not isinstance(check.get("evidence_ids", []), list)
                or any(not isinstance(item, str) for item in check.get("evidence_ids", []))):
            raise ResponseProtocolError(
                "A checkpoint must be a boolean or {status, reason, evidence_ids}; "
                "status is completed, passed, failed, unverified, or not_applicable, with a nonempty reason")
    evidence = parsed.get("evidence_ids", [])
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise ResponseProtocolError("evidence_ids must be a list of exact event ID strings")


def _parse_response(response):
    content = getattr(response, "content", response)
    if isinstance(content, dict):
        parsed = content
    else:
        text = str(content or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ResponseProtocolError("Expected a complete JSON object; response may be truncated") from exc
    if not isinstance(parsed, dict):
        raise ResponseProtocolError("Response must be a JSON object, not a scalar or list")
    _validate_completion_fields(parsed)
    calls = parsed.get("tools", [])
    if not isinstance(calls, list) or any(not isinstance(call, dict) or
            not isinstance(call.get("name"), str) for call in calls):
        raise ResponseProtocolError("tools must be a list of named tool calls")
    # Validate every completion before executing any tool with possible side effects.
    for call in calls:
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ResponseProtocolError("Tool arguments must encode a complete JSON object") from exc
        if not isinstance(arguments, dict):
            raise ResponseProtocolError("Tool arguments must be an object")
        if call["name"] in {"final_answer", "complete"}:
            if "tools" in arguments:
                raise ResponseProtocolError("Nested tool requests in a completion are not supported")
            _validate_completion_fields(arguments)
    return parsed


def _upstream_agents(agent, team):
    by_id = {item["agent_id"]: item for item in team["agents"]}
    pending = list(agent.get("depends_on", []))
    ancestors = set()
    while pending:
        aid = pending.pop()
        if aid not in ancestors:
            ancestors.add(aid)
            pending.extend(by_id[aid].get("depends_on", []))
    return [item["agent_id"] for item in team["agents"] if item["agent_id"] in ancestors]


def _checkpoint_reports(agent, parsed, observed_ids):
    reports, missing = {}, []
    for name in agent.get("checkpoints", []):
        value = parsed.get("checkpoints", {}).get(name)
        if value is True:
            report = {"status": "passed", "reason": "", "evidence_ids": []}
        elif isinstance(value, dict):
            report = {"status": value["status"], "reason": value["reason"],
                      "evidence_ids": value.get("evidence_ids", [])}
            if not set(report["evidence_ids"]) <= observed_ids:
                raise ResponseProtocolError("Checkpoint cites evidence not observed by this agent")
        else:
            missing.append(name)
            continue
        reports[name] = {**report, "basis": "self_reported", "independently_verified": False}
    return reports, missing


def _contribution_ledger(value):
    if not isinstance(value, dict):
        raise ResponseProtocolError("Contributors must publish a structured ledger object")
    for key in ("requirements", "outline"):
        if not isinstance(value.get(key), list) or any(
                not isinstance(item, str) or not item.strip() for item in value[key]):
            raise ResponseProtocolError(f"ledger.{key} must be a list of nonempty strings")
    sources = value.get("source_references")
    spans = value.get("evidence_spans")
    if not isinstance(sources, list) or not isinstance(spans, list):
        raise ResponseProtocolError("ledger needs evidence_spans and source_references lists")
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict) or any(
                not isinstance(source.get(key), str) or not source[key].strip()
                for key in ("source_id", "locator")):
            raise ResponseProtocolError("Each source reference needs source_id and locator")
        if source["source_id"] in source_ids:
            raise ResponseProtocolError("Source IDs must be unique within a contribution")
        source_ids.add(source["source_id"])
    for span in spans:
        if (not isinstance(span, dict) or not isinstance(span.get("text"), str)
                or not span["text"].strip() or not isinstance(span.get("source_ref"), str)
                or span["source_ref"] not in source_ids):
            raise ResponseProtocolError("Each evidence span needs text and a declared source_ref")
    return copy.deepcopy({key: value[key] for key in
                          ("requirements", "outline", "evidence_spans", "source_references")})


def _shared_ledger(agent, team, services):
    upstream = _upstream_agents(agent, team)
    result = {"requirements": [], "outline": [], "evidence_spans": [],
              "source_references": [], "contributions": [], "tool_evidence": []}
    for aid in upstream:
        artifact = copy.deepcopy(services.artifacts[aid])
        contribution = artifact.pop("ledger")
        artifact["dependency_kind"] = "direct" if aid in agent.get("depends_on", []) else "transitive"
        result["contributions"].append(artifact)
        for key in ("requirements", "outline"):
            result[key].extend({"agent_id": aid, "text": item} for item in contribution[key])
        for key in ("evidence_spans", "source_references"):
            result[key].extend({**item, "agent_id": aid} for item in contribution[key])
        # Only published source observations, never another role's model history.
        result["tool_evidence"].extend(copy.deepcopy(event) for event in services.events
                                       if event["agent_id"] == aid and event["kind"] == "retrieved")
    return result


def _iterative_public_ledger(services, agent_id=None):
    result = {"requirements": [], "outline": [], "evidence_spans": [],
              "source_references": [], "contributions": [], "tool_evidence": [],
              "communications": []}
    with services.lock:
        artifacts = copy.deepcopy(services.artifacts)
        events = copy.deepcopy(services.events)
    for aid, artifact in artifacts.items():
        if aid == agent_id:
            continue
        contribution = artifact.get("ledger") or {"requirements": [], "outline": [],
                                                   "evidence_spans": [], "source_references": []}
        public = {key: copy.deepcopy(artifact.get(key))
                  for key in ("agent_id", "answer", "event_id", "version", "complete")}
        public["agent_id"] = aid
        public["ledger"] = copy.deepcopy(contribution)
        result["contributions"].append(public)
        for key in ("requirements", "outline"):
            result[key].extend({"agent_id": aid, "text": item} for item in contribution.get(key, []))
        for key in ("evidence_spans", "source_references"):
            result[key].extend({**item, "agent_id": aid} for item in contribution.get(key, []))
    result["tool_evidence"] = [event for event in events if event["kind"] == "retrieved"
                               and event["agent_id"] != agent_id]
    result["communications"] = [event for event in events if event["kind"] == "peer_message"
                                and (agent_id is None or event.get("recipient") == agent_id)]
    return result


def _ledger_evidence_ids(ledger):
    return {item["event_id"] for key in ("contributions", "tool_evidence", "communications")
            for item in ledger[key] if item.get("event_id")}


def _persistent_role(agent, services):
    if services.agent_pool is None:
        return None
    from .agent_pool import role_context
    return role_context(services.agent_pool, AgentSpec.model_validate(agent), services.public_task)


def _role_prompt(agent, ctx, persistent):
    if persistent is None:
        return str(ctx.prompt_templates.get("agent_prompt", ""))
    return (persistent["prompt"] + "\nRetained role policies:\n" + json.dumps({
        "skills": persistent["skills"], "reasoning_strategy": persistent["reasoning_strategy"],
        "planning_strategy": persistent["planning_strategy"],
        "communication": persistent["communication"]}, ensure_ascii=False)
        + "\nApply these methods within the public task, frozen collaboration topology and budgets. "
        "Your retained memory contains conditional process advice, not evidence for current claims. "
        "Choose tools and methods appropriate to this assignment.\nTask adaptation: "
        + agent.get("task_prompt", ""))


def _role_tools(allowed, catalog, persistent):
    if persistent is None or persistent["harness"]["tool_policy"] != "preferred_first":
        return {name: catalog[name] for name in sorted(allowed)}
    preferred = [name for name in persistent["preferred_tools"] if name in allowed]
    ordered = list(dict.fromkeys([*preferred, *sorted(allowed)]))
    return {name: catalog[name] for name in ordered}


def _role_messages(messages, persistent):
    if persistent is None or persistent["harness"]["memory_policy"] != "recent":
        return copy.deepcopy(messages)
    window = persistent["harness"]["memory_window"]
    return copy.deepcopy(messages[:2] + messages[2:][-window:])


def _run_agent_iterative(agent, team, ctx, services, *, state=None, one_turn=False,
                         defer_final_submission=False):
    if state is not None and state.get("initialized"):
        return _continue_agent_iterative(agent, team, ctx, services, state,
                                         one_turn=one_turn,
                                         defer_final_submission=defer_final_submission)
    aid = agent["agent_id"]
    persistent = _persistent_role(agent, services)
    synth = aid == team["synthesizer_id"]
    before = services.event(aid, "agent_started", {"role": agent["role"], "execution_mode": services.execution_mode})
    model = services.model_factory(aid)
    output_limit = agent.get("max_tokens", 4096)
    model_limit = getattr(model, "max_tokens", None)
    if isinstance(model_limit, int) and model_limit > 0:
        output_limit = min(output_limit, model_limit)
    allowed = set(agent.get("tools", []))
    catalog = ctx.tool_policy.select_tools("", 0, ctx.memory.build_context()).tools
    absent = allowed - set(catalog)
    if absent:
        raise ValueError(f"tools not installed: {sorted(absent)}")
    role_tools = _role_tools(allowed, catalog, persistent)
    model._native_tool_registry = role_tools
    peer_ids = [item["agent_id"] for item in team["agents"] if item["agent_id"] != aid]
    completion_example = {"answer": "The current complete deliverable." if synth else "Current contribution summary.",
                          "evidence_ids": [], "checkpoints": {}}
    if not synth:
        completion_example["ledger"] = {"requirements": [], "outline": [],
                                         "evidence_spans": [], "source_references": []}
    system = _role_prompt(agent, ctx, persistent) + (
        "\nExecution mode is iterative_shared_ledger. You may continue your role, revisit prior "
        "artifacts, ask peers through send_message, and request allowed tools. Tool results are "
        "returned in the next model turn. Publish honest public ledger updates. Return a complete "
        "JSON object each turn; set continue=true when another turn is needed, and finish with "
        "continue=false or a complete/final_answer tool. Token and timeout budgets are binding; "
        "do not wait for a fixed round count. Private conversation history stays private. "
        "Every tool request is {name, arguments}; arguments must be a JSON object conforming "
        "to its supplied schema. Cite only evidence IDs received in your ledger or observations. "
        "A terminal response must include the complete answer and all assigned checkpoints. "
        "send_message posts to an exact peer agent_id and reactivates a completed teammate. "
        "After asking a peer, the scheduler lets that peer respond before continuing your role. "
        "A completed contribution may be resumed to answer questions or revise its public artifact. "
        "The synthesizer submits only after all teammates and pending peer requests complete."
        " Use resource_budget to conserve shared tokens: avoid duplicate analysis and full draft "
        "copies, send concise evidence-linked messages, and reserve room for the complete final "
        "deliverable. Follow budget_policy's completion and quality-cost stopping guidance. "
        "budget_estimate is a forecast, not a fixed call or round limit."
    )
    shared_ledger = _iterative_public_ledger(services, aid)
    ledger_hash = content_hash(shared_ledger)
    observed_ids = _ledger_evidence_ids(shared_ledger)
    services.event(aid, "shared_ledger_read", {"ledger_hash": ledger_hash},
                   parents=sorted(observed_ids))
    if services.ledger is not None:
        services.ledger.charge_communication(
            len(json.dumps(shared_ledger, ensure_ascii=False).encode("utf-8")), stage="execution", agent_id=aid)
    instruction = {"public_task": _data(services.public_task), "agent": agent,
                   "shared_ledger": shared_ledger,
                   "rubrics": _data(services.rubrics) or {"rubrics": []},
                   "execution_experiences": [experience for item in services.experiences
                        if (experience := _data(item)).get("kind", experience.get("bank", "")) == "execution"
                        and experience_applicability(experience, services.public_task,
                                                     capability=agent["capability"])["matched"]],
                   "allowed_tool_schemas": ctx.get_tool_schemas(role_tools) if allowed else "[]",
                   "communication_tools": [{"name": "send_message",
                       "description": "Ask or inform a teammate, resuming its private role history if needed.",
                       "parameters": {"type": "object", "properties": {
                           "recipient": {"type": "string", "enum": peer_ids},
                           "content": {"type": "string", "minLength": 1}},
                           "required": ["recipient", "content"], "additionalProperties": False}}],
                   "coordination": "iterative_shared_ledger",
                   "scheduling": {"strategy": "cooperative_shared_ledger",
                                  "reactivate_completed_agents": True},
                   "submission": "final_answer" if synth else "contribution",
                   "completion_example": completion_example,
                   "budget_estimate": next((estimate for estimate in
                       (team.get("budget_plan") or {}).get("agents", [])
                       if estimate["agent_id"] == aid), None),
                   "budget_policy": {key: value for key, value in (team.get("budget_plan") or {}).items()
                                     if key != "agents"},
                   "output_budget": {"max_tokens_per_response": output_limit,
                                     "max_model_calls": None, "max_tool_calls": None}}
    if persistent is not None:
        instruction["persistent_agent"] = persistent
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)}]
    state = state if state is not None else {}
    state.update(initialized=True, persistent=persistent, model=model, before=before,
                 output_limit=output_limit, allowed=allowed, peer_ids=peer_ids,
                 messages=messages, trajectory=[], answer=None,
                 contribution={"requirements": [], "outline": [], "evidence_spans": [],
                               "source_references": []}, checkpoint_reports={}, evidence_ids=[],
                 pending_observation_ids=set(), observed_ids=observed_ids,
                 ledger_hash=ledger_hash, handled_message_ids=set(), awaiting_messages=[])
    return _continue_agent_iterative(agent, team, ctx, services, state,
                                     one_turn=one_turn,
                                     defer_final_submission=defer_final_submission)


def _continue_agent_iterative(agent, team, ctx, services, state, *, one_turn,
                              defer_final_submission):
    aid = agent["agent_id"]
    synth = aid == team["synthesizer_id"]
    persistent = state["persistent"]
    model = state["model"]
    before = state["before"]
    output_limit = state["output_limit"]
    allowed = state["allowed"]
    peer_ids = state["peer_ids"]
    messages = state["messages"]
    trajectory = state["trajectory"]
    answer = state["answer"]
    contribution = state["contribution"]
    checkpoint_reports = state["checkpoint_reports"]
    evidence_ids = state["evidence_ids"]
    pending_observation_ids = state["pending_observation_ids"]
    observed_ids = state["observed_ids"]
    ledger_hash = state["ledger_hash"]
    reason = "running"
    while not services.cancelled.is_set():
        remaining = services.remaining_seconds()
        if remaining <= 0:
            services.cancelled.set()
            reason = "timeout"
            break
        updated_ledger = _iterative_public_ledger(services, aid)
        updated_hash = content_hash(updated_ledger)
        if updated_hash != ledger_hash:
            update_content = "Updated shared ledger: " + json.dumps(updated_ledger, ensure_ascii=False)
            if len(messages) > 2 and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n" + update_content
            else:
                messages.append({"role": "user", "content": update_content})
            observed_ids.update(_ledger_evidence_ids(updated_ledger))
            ledger_hash = updated_hash
            services.event(aid, "shared_ledger_read", {"ledger_hash": ledger_hash},
                           parents=sorted(_ledger_evidence_ids(updated_ledger)))
            if services.ledger is not None:
                services.ledger.charge_communication(
                    len(update_content.encode("utf-8")), stage="execution", agent_id=aid)
        observed_ids.update(pending_observation_ids)
        pending_observation_ids.clear()
        state["handled_message_ids"].update(item["event_id"]
                                           for item in updated_ledger["communications"])
        if services.ledger is not None:
            instruction = json.loads(messages[1]["content"])
            instruction["resource_budget"] = services.ledger.resource_context()
            messages[1]["content"] = json.dumps(instruction, ensure_ascii=False)
        context_messages = _role_messages(messages, persistent)
        step = StepRecord(step_number=len(trajectory) + 1, model_input_messages=context_messages,
                          start_time=time.time())
        try:
            response = _bounded_call(model, min(services.timeout_seconds, remaining), context_messages,
                                     deadline_remaining=services.remaining_seconds,
                                     max_tokens=output_limit)
            step.model_output_messages = response
            usage = model.get_token_counts() if hasattr(model, "get_token_counts") else {}
            step.input_token_count = int(usage.get("input_token_count", 0))
            step.output_token_count = int(usage.get("output_token_count", 0))
            step.total_token_count = step.input_token_count + step.output_token_count
            output_event = services.event(aid, "model_output", step.full_dict()["model_output_messages"],
                                          parents=[before, *sorted(observed_ids)])
            parsed = _parse_response(response)
            if "continue" in parsed and not isinstance(parsed["continue"], bool):
                raise ResponseProtocolError("continue must be a boolean")
            tool_calls = parsed.get("tools", [])
            terminal = not bool(parsed.get("continue", False))
            observations = []
            merged = dict(parsed)
            for call in tool_calls:
                name = call["name"]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if name in {"complete", "final_answer"}:
                    if name == "final_answer" and not synth:
                        raise ResponseProtocolError("only the synthesizer may submit final_answer")
                    merged.update(args)
                    terminal = True
                    continue
                if name == "send_message":
                    recipient = args.get("recipient")
                    message = args.get("content", args.get("message", ""))
                    if (recipient not in peer_ids or not isinstance(message, str) or not message.strip()):
                        raise ResponseProtocolError("send_message requires an exact peer agent_id and nonempty content")
                elif name not in allowed:
                    raise PermissionError(f"tool '{name}' is not allowed for agent '{aid}'")
            merged["tools"] = []
            _validate_completion_fields(merged)
            evidence_ids = merged.get("evidence_ids", [])
            if not set(evidence_ids) <= observed_ids:
                raise ResponseProtocolError("Completion cites evidence not observed by this agent")
            checkpoint_reports, missing = _checkpoint_reports(agent, merged, observed_ids)
            if any(call["name"] not in {"complete", "final_answer"} for call in tool_calls):
                terminal = False
            if terminal and missing:
                raise ResponseProtocolError("Unconfirmed checkpoints: " + json.dumps(missing))
            if terminal and merged.get("answer") is None:
                raise ResponseProtocolError("A terminal response must include its complete answer")
            if not synth and merged.get("ledger") is not None:
                contribution = _contribution_ledger(merged["ledger"])
            for call in tool_calls:
                name = call["name"]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                step.tool_calls.append(ToolCall(name=name, arguments=copy.deepcopy(args)))
                if name in {"complete", "final_answer"}:
                    continue
                if services.cancelled.is_set():
                    raise TimeoutError("team cancelled before tool dispatch")
                remaining = services.remaining_seconds()
                if remaining <= 0:
                    raise TimeoutError("team execution timeout")
                if name == "send_message":
                    recipient = args["recipient"]
                    message = args.get("content", args.get("message", ""))
                    event_id = services.event(aid, "peer_message", {"content": message},
                                              parents=[output_event], recipient=recipient, source="send_message")
                    if services.ledger is not None:
                        services.ledger.charge_communication(len(message.encode("utf-8")), stage="execution", agent_id=aid)
                    observations.append({"event_id": event_id, "recipient": recipient, "content": message})
                    state["awaiting_messages"].append((recipient, event_id))
                    continue
                if services.ledger is not None:
                    services.ledger.charge_tool(stage="execution", agent_id=aid, tool_name=name)
                observation = _bounded_call(ctx.execute_tool, min(services.timeout_seconds, remaining), name, args,
                                            deadline_remaining=services.remaining_seconds)
                event_id = services.event(aid, "retrieved", {"tool": name, "arguments": args,
                                                               "output": observation},
                                          parents=[output_event], source=name)
                observations.append({"event_id": event_id, "output": observation})
            if merged.get("answer") is not None:
                answer = merged["answer"]
            if not synth and answer is None and not observations:
                raise ResponseProtocolError("Contributor must publish an answer or tool request")
            if answer is not None or contribution != {"requirements": [], "outline": [], "evidence_spans": [], "source_references": []}:
                artifact_event = services.event(aid, "artifact_published", answer or contribution,
                                                parents=[output_event])
                with services.lock:
                    services.artifacts[aid] = {"agent_id": aid, "answer": answer, "event_id": artifact_event,
                                           "content_hash": content_hash(answer or contribution),
                                           "version": len(trajectory) + 1, "complete": terminal,
                                           "checkpoint_reports": copy.deepcopy(checkpoint_reports),
                                           "ledger": copy.deepcopy(contribution)}
            step.action_output = answer
            step.observations = json.dumps(observations, ensure_ascii=False)
            messages.append({"role": "assistant", "content": getattr(response, "content", str(response))})
            if observations:
                messages.append({"role": "user", "content": "Public observations: " + json.dumps(observations, ensure_ascii=False)})
                pending_observation_ids.update(item["event_id"] for item in observations)
            if terminal:
                reason = "final_answer" if synth else "subtask_complete"
                step.end_time = time.time()
                step.duration = step.end_time - step.start_time
                trajectory.append(step)
                break
        except Exception as exc:
            step.error = exc
            step.observations = f"{type(exc).__name__}: {exc}"
            services.event(aid, "execution_error", step.observations, parents=[before])
            reason = "timeout" if isinstance(exc, TimeoutError) else "error"
            if isinstance(exc, TimeoutError):
                services.cancelled.set()
            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            trajectory.append(step)
            break
        step.end_time = time.time()
        step.duration = step.end_time - step.start_time
        trajectory.append(step)
        if one_turn:
            reason = "yielded"
            break
    if reason == "running":
        reason = "timeout" if services.cancelled.is_set() else "error"
    if reason not in {"final_answer", "subtask_complete", "yielded"}:
        answer = None
        with services.lock:
            services.artifacts.pop(aid, None)
    if reason == "final_answer" and answer is not None and synth and not defer_final_submission:
        services.event(aid, "final_answer", answer, parents=evidence_ids)
    state.update(answer=answer, contribution=contribution, checkpoint_reports=checkpoint_reports,
                 evidence_ids=evidence_ids, ledger_hash=ledger_hash)
    return RunResult(answer=answer, trajectory=trajectory, terminated_reason=reason,
                     metadata={"agent_id": aid, "role": agent["role"], "capability": agent["capability"],
                               "coordination": "iterative_shared_ledger", "model_calls": len(trajectory),
                               "scheduling": {"strategy": "cooperative_shared_ledger",
                                              "reactivate_completed_agents": True},
                               "checkpoint_reports": checkpoint_reports,
                               "observed_evidence_ids": sorted(observed_ids),
                               "pool_agent_id": agent.get("pool_agent_id"),
                               "pool_agent_version": agent.get("pool_agent_version"),
                               "event_ids": [e["event_id"] for e in services.events if e["agent_id"] == aid]})


def _run_agent(agent, team, ctx, services):
    if services.execution_mode == "iterative_shared_ledger":
        return _run_agent_iterative(agent, team, ctx, services)
    aid = agent["agent_id"]
    persistent = _persistent_role(agent, services)
    synth = aid == team["synthesizer_id"]
    before = services.event(aid, "agent_started", {"role": agent["role"]})
    shared_ledger = _shared_ledger(agent, team, services)
    observed_ids = {item["event_id"] for item in
                    shared_ledger["contributions"] + shared_ledger["tool_evidence"]}
    ledger_hash = content_hash(shared_ledger)
    if synth:
        services.shared_ledger = copy.deepcopy(shared_ledger)
        services.event(aid, "shared_ledger_ready", shared_ledger, parents=sorted(observed_ids))
    if synth or shared_ledger["contributions"]:
        read_event = services.event(aid, "shared_ledger_read", {"ledger_hash": ledger_hash},
                                    parents=sorted(observed_ids))
        observed_ids.add(read_event)
        if services.ledger is not None:
            services.ledger.charge_communication(len(json.dumps(shared_ledger, ensure_ascii=False).encode("utf-8")), agent_id=aid)
    rubric_data = _data(services.rubrics) or {"rubrics": []}
    primary_rubrics = [rid for rid, owner in team.get("primary", {}).items() if owner == aid]
    review_rubrics = [rid for rid, reviewers in team.get("reviewers", {}).items() if aid in reviewers]
    assigned_rubrics = set(agent.get("rubric_ids", [])) | set(primary_rubrics) | set(review_rubrics)
    relevant = [r for r in rubric_data.get("rubrics", [])
                if r["rubric_id"] in assigned_rubrics]
    experiences = [_data(e) for e in services.experiences]
    execution_experiences = [e for e in experiences if e.get("kind", e.get("bank", "")) == "execution"]
    experience_selection = [{"experience_id": e.get("experience_id"),
                             **experience_applicability(e, services.public_task,
                                                        capability=agent["capability"])}
                            for e in execution_experiences]
    local_experiences = [e for e, selection in zip(execution_experiences, experience_selection)
                         if selection["matched"]]
    model = services.model_factory(aid)
    output_limit = agent.get("max_tokens", 4096)
    model_limit = getattr(model, "max_tokens", None)
    if isinstance(model_limit, int) and model_limit > 0:
        output_limit = min(output_limit, model_limit)
    completion_example = {"answer": "The final requested artifact." if synth else "Brief contribution summary."}
    if not synth:
        completion_example["ledger"] = {"requirements": [], "outline": [],
                                         "evidence_spans": [], "source_references": []}
    completion_example.update(evidence_ids=[], checkpoints={name: {
        "status": "unverified", "reason": "Replace with the actual check result or limitation.",
        "evidence_ids": []} for name in agent.get("checkpoints", [])})
    instruction = {
        "public_task": _data(services.public_task), "agent": agent,
        "predicted_requirements": relevant, "shared_ledger": shared_ledger,
        "primary_rubrics": primary_rubrics, "review_rubrics": review_rubrics,
        "review_owners": {rid: team.get("primary", {}).get(rid) for rid in review_rubrics},
        "execution_experiences": local_experiences,
        "experience_selection": experience_selection,
        "coordination": "single_pass_shared_ledger",
        "submission": "final_answer" if synth else "subtask_complete",
        "completion_example": completion_example,
        "resource_budget": services.ledger.resource_context() if services.ledger is not None else None,
        "budget_estimate": next((estimate for estimate in
            (team.get("budget_plan") or {}).get("agents", [])
            if estimate["agent_id"] == aid), None),
        "budget_policy": {key: value for key, value in (team.get("budget_plan") or {}).items()
                          if key != "agents"},
        "output_budget": {"max_tokens_per_response": output_limit, "max_model_calls": 1},
    }
    if persistent is not None:
        instruction["persistent_agent"] = persistent
    system = _role_prompt(agent, ctx, persistent) + (
        "\nYou have an independent conversation. Follow only your assigned responsibilities. "
        "The role-specific completion_example is the authoritative output shape for your role; "
        "replace its example values with your own contribution, retaining the JSON field structure. "
        "Return one JSON object, not a prose or Markdown envelope. Escape newlines, quotes and "
        "backslashes inside JSON strings. Do not put JSON fields or a Markdown ledger inside answer. "
        "In checkpoints, use every exact name from agent.checkpoints "
        "as a key. Use true only for an actually completed self-check, or report an object such as "
        "{\"status\":\"unverified\",\"reason\":\"...\",\"evidence_ids\":[]}. "
        "Allowed statuses are completed, passed, failed, unverified, and not_applicable. "
        "Every object needs a nonempty reason. completed records that a self-check was performed; "
        "it is not a passing finding or independent verification. Report honest limitations, "
        "and do not claim success to satisfy a checkpoint. "
        "Missing checks or unexplained false values prevent completion; an explained failed "
        "or unverified check is a reported limitation, not verified success. Self-reported "
        "checks do not replace external evaluation. Address material limitations in your answer. "
        "You receive exactly one model call. There are no agent messaging or shared-memory lookup tools. "
        "Express missing input or disputed evidence in your answer and checkpoint reports. "
        "Never claim unobserved evidence or broadcast private conversations."
    )
    system += (
        "\nTreat predicted requirements, upstream drafts and generated task-specific hints as "
        "fallible planning hypotheses, not authoritative facts. The public task takes priority. "
        "Do not reproduce a factual error merely because it appears in a rubric or another agent's "
        "answer. For technical claims, distinguish definitions, assumptions and conclusions; check "
        "each inference and a simple numerical or limiting case when relevant. Do not present "
        "an unsupported step as a simplification. For nontechnical tasks, use domain-appropriate "
        "checks without imposing formulas or citations."
    )
    system += (
        f"\nEach complete JSON response has a hard ceiling of {output_limit} output tokens, "
        "including escaped prose, evidence IDs, checkpoints and closing braces. Plan a concise "
        f"answer well below this ceiling (roughly {max(1, output_limit // 2)} English words or fewer); "
        "token-to-word ratios vary, so leave room for structure and all required checks. "
        "Never start an answer too long to close its JSON object. Shorten wording, not factual accuracy."
    )
    if primary_rubrics:
        system += "\nYou are the primary owner for primary_rubrics: produce evidence or artifacts addressing each."
    if review_rubrics:
        system += (
            "\nYou are the assigned reviewer for review_rubrics. Independently examine the primary "
            "owner's upstream artifacts against those requirements. Resolve or explicitly report "
            "contradictions and missing evidence in this one contribution. Do not request peer input. "
            "Prioritize a few consequential errors over blanket approval. Identify the original "
            "claim, why it is wrong or uncertain, and a specific supported correction. For a proof "
            "or calculation, independently check its assumptions and the questionable step, not "
            "just whether the requested formula appears. Your review must identify what was "
            "checked and remaining uncertainty; do not declare all checks passed without support."
        )
    if synth:
        system += (
            "\nRead the structured shared ledger once, synthesize the final response from the "
            "analyst requirements and evidence spans, and do not initiate additional inter-agent "
            "communication. You cannot call external tools or request another role invocation. "
            "\nSynthesize a coherent final response: resolve contradictory artifacts, state unresolved "
            "gaps and uncertainty, and preserve relevant sources. Do not merely concatenate outputs. "
            "Cite evidence_ids only for artifacts actually incorporated; citations are optional for creative tasks. "
            "The answer must be the final deliverable itself, not an editing report, review preface, "
            "or draft followed by a second revised copy. Integrate relevant review corrections concisely."
            " Inspect the original ancestor artifacts as well as their reviews. A review can also "
            "be wrong: reconcile disputed claims using their evidence and stated assumptions, "
            "rather than following the most recent or most confident speaker. Preserve sound "
            "substantive content while fixing errors; do not substitute a thin summary for the "
            "requested deliverable or expose internal rubric IDs in it."
        )
    else:
        system += (
            "\nYou are a contributor, not the final Writer. Publish a brief summary in answer and "
            "a separate top-level ledger object; ledger is a sibling of answer, never text inside it. "
            "Keep substantive requirements, reasoning steps and evidence in their corresponding "
            "ledger fields, without repeating them in a competing full answer or repetitive lists. "
            "Publish exactly these structured ledger fields: "
            "requirements (list of strings), outline (list of strings), evidence_spans "
            "(list of {text, source_ref}), source_references (list of {source_id, locator}). "
            "Analytical responsibilities populate requirements and outline; evidence responsibilities "
            "populate source spans and provenance. Use empty lists when inapplicable, never invent "
            "sources. A source_ref must match a source_id in your own contribution; these are "
            "self-reported source claims, not verified citations. You may request a single batch "
            "of allowed external tools. Their raw results are published directly to the shared ledger "
            "for the writer; you will not receive a second model call to read them. Do not cite "
            "those future results as evidence you already observed."
        )
    memory = type(ctx.memory)(prompts=ctx.prompt_templates)
    memory.initialize(system, TaskInput(task=json.dumps(instruction, ensure_ascii=False)))
    allowed = set() if synth else set(agent.get("tools", []))
    catalog = ctx.tool_policy.select_tools("", 0, memory.build_context()).tools
    absent = allowed - set(catalog)
    if absent:
        raise ValueError(f"tools not installed: {sorted(absent)}")
    role_tools = _role_tools(allowed, catalog, persistent)
    model._native_tool_registry = role_tools
    trajectory = []
    answer = None
    reason = "error"
    evidence_ids = []
    checkpoint_reports = {}
    contribution = {"requirements": [], "outline": [], "evidence_spans": [], "source_references": []}
    # One response, optionally one external-tool batch, then publish. No feedback loop.
    if not services.cancelled.is_set():
        messages = memory.build_context().messages
        tool_schemas = ctx.get_tool_schemas(role_tools) if allowed else "[]"
        tools_left = (None if services.ledger is None or services.ledger.max_tool_calls is None else
                      max(0, services.ledger.max_tool_calls - services.ledger.snapshot()["tool_calls"]))
        messages.append({"role": "user", "content": "Allowed tool schemas: " + tool_schemas
                         + "\nThis is your only model call for this task. "
                         + (f"Remaining shared tool calls: {tools_left}. " if tools_left is not None else "")
                         + f"Return a complete JSON object within {output_limit} output tokens."
                         + "\nCompletion shape (replace example values, keep fields): "
                         + json.dumps(completion_example, ensure_ascii=False)
                         + ("\nAlternatively, an allowed contributor tool batch may use "
                            '{"tools":[{"name":"...","arguments":{}}]}; outputs go directly to the Writer.'
                            if allowed and tools_left != 0 else "\nNo external tool requests are available.")})
        step = StepRecord(step_number=1, model_input_messages=copy.deepcopy(messages),
                          start_time=time.time())
        observations = []
        try:
            response = _bounded_call(model, services.timeout_seconds, messages,
                                     max_tokens=output_limit)
            step.model_output_messages = response
            usage = model.get_token_counts() if hasattr(model, "get_token_counts") else {}
            step.input_token_count = int(usage.get("input_token_count", 0))
            step.output_token_count = int(usage.get("output_token_count", 0))
            step.total_token_count = step.input_token_count + step.output_token_count
            output_event = services.event(aid, "model_output", step.full_dict()["model_output_messages"],
                                          parents=[before])
            parsed = _parse_response(response)
            completions = [parsed]
            merged_completion = dict(parsed)
            for call in parsed.get("tools", []):
                if call["name"] == "final_answer" and not synth:
                    raise ResponseProtocolError("only the synthesizer may submit final_answer")
                if call["name"] in {"complete", "final_answer"}:
                    args = call.get("arguments", {})
                    merged_completion.update(json.loads(args) if isinstance(args, str) else args)
                    completions.append(dict(merged_completion))
            for completion in completions:
                if completion.get("answer") is not None:
                    if not set(completion.get("evidence_ids", [])) <= observed_ids:
                        raise ResponseProtocolError("Completion cites evidence not observed by this agent")
                    _checkpoint_reports(agent, completion, observed_ids)
            parsed = merged_completion
            tool_requests = [call for call in parsed.get("tools", [])
                             if call["name"] not in {"complete", "final_answer"}]
            for call in tool_requests:
                if call["name"] in {"send_message", "read_evidence", "raise_issue"}:
                    raise ResponseProtocolError("Inter-agent communication tools are unavailable in single-pass execution")
                if synth:
                    raise ResponseProtocolError("The writer only reads its completed ledger and submits once")
                if call["name"] not in allowed:
                    raise PermissionError(f"tool '{call['name']}' is not allowed for agent '{aid}'")
            if not synth:
                if parsed.get("ledger") is not None:
                    contribution = _contribution_ledger(parsed["ledger"])
                elif parsed.get("answer") is not None or not tool_requests:
                    raise ResponseProtocolError("Contributors must publish a structured ledger object")
            checkpoint_reports, missing = _checkpoint_reports(agent, parsed, observed_ids)
            if missing:
                raise ResponseProtocolError("Unconfirmed checkpoints: " + json.dumps(missing))
            for call in parsed.get("tools", []):
                name = call["name"]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                step.tool_calls.append(ToolCall(name=name, arguments=copy.deepcopy(args)))
                if name in {"final_answer", "complete"}:
                    parsed.update(args)
                    continue
                if services.cancelled.is_set():
                    raise TimeoutError("team cancelled before external tool dispatch")
                if services.ledger is not None:
                    services.ledger.charge_tool(stage="execution", agent_id=aid, tool_name=name)
                observation = _bounded_call(ctx.execute_tool, services.timeout_seconds, name, args)
                event_id = services.event(aid, "retrieved", {"tool": name, "arguments": args,
                                                           "output": observation},
                                          parents=[output_event], source=name)
                observation = {"event_id": event_id, "output": observation}
                observations.append(json.dumps(observation, ensure_ascii=False))
            if parsed.get("answer") is not None:
                evidence_ids = parsed.get("evidence_ids", [])
                answer = parsed["answer"]
            elif tool_requests and not synth:
                # Raw observations are public data, not an invented analyst answer.
                answer = json.dumps({"source_event_ids": [json.loads(item)["event_id"]
                                                          for item in observations]})
            else:
                raise ResponseProtocolError("A single-pass response must complete its assignment")
            step.action_output = answer
            reason = "final_answer" if synth else "subtask_complete"
        except ResponseProtocolError as exc:
            step.error = exc
            observations.append(f"ResponseProtocolError: {exc}. Task attempt failed; no role recall.")
            services.event(aid, "execution_error", observations[-1], parents=[before])
        except Exception as exc:
            step.error = exc
            observations.append(f"{type(exc).__name__}: {exc}")
            services.event(aid, "execution_error", observations[-1], parents=[before])
            reason = "error"
            if isinstance(exc, TimeoutError):
                services.cancelled.set()
        finally:
            step.observations = "\n".join(observations)
            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            trajectory.append(step)
            memory.update(step)
    if answer is not None:
        artifact_id = services.event(aid, "artifact_published", answer, parents=evidence_ids)
        services.artifacts[aid] = {"agent_id": aid, "answer": answer, "event_id": artifact_id,
                                   "content_hash": content_hash(answer), "version": 1,
                                   "checkpoint_reports": copy.deepcopy(checkpoint_reports),
                                   "ledger": contribution}
        if synth:
            services.event(aid, "final_answer", answer, parents=evidence_ids)
    return RunResult(answer=answer, trajectory=trajectory, terminated_reason=reason,
                     metadata={"agent_id": aid, "capability": agent["capability"],
                               "role": agent["role"], "rubric_ids": sorted(assigned_rubrics),
                               "primary_rubrics": primary_rubrics, "review_rubrics": review_rubrics,
                               "checkpoint_reports": checkpoint_reports,
                               "coordination": "single_pass_shared_ledger", "ledger_hash": ledger_hash,
                               "observed_evidence_ids": sorted(observed_ids),
                               "experience_selection": experience_selection,
                               "pool_agent_id": agent.get("pool_agent_id"),
                               "pool_agent_version": agent.get("pool_agent_version"),
                               "event_ids": [e["event_id"] for e in services.events if e["agent_id"] == aid]})


def _run_team_iterative(team, ctx, services):
    agents = {agent["agent_id"]: agent for agent in team["agents"]}
    states = {}
    statuses = {aid: "unstarted" for aid in agents}
    results = {}
    dispatch_order = {aid: 0 for aid in agents}
    dispatch_count = 0
    synthesizer = team["synthesizer_id"]
    while not services.cancelled.is_set():
        if services.remaining_seconds() <= 0:
            services.cancelled.set()
            break
        while True:
            failed = {aid for aid, status in statuses.items() if status == "failed"}
            blocked = [aid for aid, agent in agents.items() if statuses[aid] != "failed"
                       and failed.intersection(agent.get("depends_on", []))]
            if not blocked:
                break
            for aid in blocked:
                services.event(aid, "dependency_failed", {"failed": sorted(failed)})
                statuses[aid] = "failed"
                prior = results.get(aid)
                results[aid] = RunResult(terminated_reason="dependency_failed",
                                        trajectory=prior.trajectory if prior else [],
                                        metadata={"agent_id": aid})
                with services.lock:
                    services.artifacts.pop(aid, None)
        if statuses[synthesizer] == "failed":
            break
        with services.lock:
            communications = [copy.deepcopy(event) for event in services.events
                              if event["kind"] == "peer_message"]
        inbox = {}
        for event in communications:
            recipient = event["recipient"]
            if (recipient in agents and statuses[recipient] != "failed"
                    and event["event_id"] not in states.get(recipient, {}).get("handled_message_ids", set())):
                inbox.setdefault(recipient, []).append(event["event_id"])
        for aid, pending_ids in inbox.items():
            if statuses[aid] == "completed":
                statuses[aid] = "active"
                services.event(aid, "agent_resumed", {"reason": "peer_message",
                                                       "message_event_ids": pending_ids},
                               parents=pending_ids)
                with services.lock:
                    if aid in services.artifacts:
                        services.artifacts[aid]["complete"] = False
        waiting = {}
        for aid, state in states.items():
            unavailable = [recipient for recipient, event_id in state["awaiting_messages"]
                           if statuses.get(recipient) == "failed"]
            if unavailable and statuses[aid] != "failed":
                services.event(aid, "peer_unavailable", {"recipients": sorted(set(unavailable))})
                statuses[aid] = "failed"
                prior = results.get(aid)
                results[aid] = RunResult(terminated_reason="peer_unavailable",
                                        trajectory=prior.trajectory if prior else [],
                                        metadata={"agent_id": aid,
                                                  "unavailable_peers": sorted(set(unavailable))})
                with services.lock:
                    services.artifacts.pop(aid, None)
                state["awaiting_messages"] = []
                continue
            pending = [(recipient, event_id) for recipient, event_id in state["awaiting_messages"]
                       if not (event_id in states.get(recipient, {}).get("handled_message_ids", set())
                               and statuses[recipient] == "completed")]
            state["awaiting_messages"] = pending
            waiting[aid] = bool(pending)
        if all(status == "completed" for status in statuses.values()) and not inbox:
            current_ledger = _iterative_public_ledger(services, synthesizer)
            if content_hash(current_ledger) != states[synthesizer]["ledger_hash"]:
                statuses[synthesizer] = "active"
                services.event(synthesizer, "agent_resumed", {"reason": "updated_team_artifacts"})
                with services.lock:
                    services.artifacts[synthesizer]["complete"] = False
            else:
                final = results[synthesizer]
                services.event(synthesizer, "final_answer", final.answer,
                               parents=states[synthesizer]["evidence_ids"])
                for aid, result in results.items():
                    result.metadata["event_ids"] = [event["event_id"] for event in services.events
                                                    if event["agent_id"] == aid]
                return results
        ordered = sorted(agents, key=lambda aid: dispatch_order[aid])
        ready = [aid for aid in ordered if aid in inbox]
        ready.extend(aid for aid in ordered if aid not in ready and statuses[aid] == "active"
                     and not waiting.get(aid, False))
        ready.extend(aid for aid in ordered if aid not in ready and statuses[aid] == "unstarted"
                     and all(statuses[dependency] == "completed"
                             for dependency in agents[aid].get("depends_on", [])))
        if not ready:
            ready = [aid for aid in ordered if statuses[aid] == "active"]
        if not ready:
            break
        selected = ready[:team.get("max_parallel", 2)]
        for aid in selected:
            statuses[aid] = "active"
            dispatch_count += 1
            dispatch_order[aid] = dispatch_count
        with ThreadPoolExecutor(max_workers=team.get("max_parallel", 2)) as pool:
            futures = {pool.submit(_run_agent_iterative, agents[aid], team, ctx, services,
                                   state=states.setdefault(aid, {}), one_turn=True,
                                   defer_final_submission=True): aid for aid in selected}
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    results[aid] = future.result()
                except Exception as exc:
                    services.event(aid, "execution_error", str(exc))
                    results[aid] = RunResult(terminated_reason="error",
                                            trajectory=states[aid].get("trajectory", []),
                                            metadata={"agent_id": aid, "error": str(exc)})
                reason = results[aid].terminated_reason
                statuses[aid] = ("completed" if reason in {"subtask_complete", "final_answer"}
                                 else "active" if reason == "yielded" else "failed")
    for aid, status in statuses.items():
        if status != "failed":
            services.event(aid, "cancelled", "team ended without a final coordinated submission")
            prior = results.get(aid)
            results[aid] = RunResult(terminated_reason="cancelled",
                                    trajectory=prior.trajectory if prior else [],
                                    metadata={"agent_id": aid})
            with services.lock:
                services.artifacts.pop(aid, None)
    return results


def run_team(task, ctx, team, services):
    """Schedule a validated team with either legacy or iterative ledger coordination."""
    team = validate_team(team, services.public_task)
    with services.lock:
        if services.started:
            raise RuntimeError("A task team cannot be restarted")
        services.started = True
    ctx.planning.bind_team(team)
    plan = ctx.planning.init_plan(task, ctx.memory.build_context(), "", ctx.model)
    ctx.memory.update_plan(plan)
    pending = ({a["agent_id"]: a for a in team["agents"]}
               if services.execution_mode != "iterative_shared_ledger" else {})
    results = (_run_team_iterative(team, ctx, services)
               if services.execution_mode == "iterative_shared_ledger" else {})
    while pending and not services.cancelled.is_set():
        # Propagate a failed producer through every dependent before dispatching work.
        while True:
            failed = {aid for aid, result in results.items() if result.answer is None or
                      (services.execution_mode == "iterative_shared_ledger" and
                       result.terminated_reason not in {"subtask_complete", "final_answer"})}
            blocked = [aid for aid, agent in pending.items() if failed.intersection(agent.get("depends_on", []))]
            if not blocked:
                break
            for aid in blocked:
                services.event(aid, "dependency_failed", {"failed": sorted(failed)})
                results[aid] = RunResult(terminated_reason="dependency_failed", metadata={"agent_id": aid})
                del pending[aid]
        ready = [agent for agent in pending.values()
                 if set(agent.get("depends_on", [])).issubset(results)]
        if not ready:
            if pending:
                raise RuntimeError("No ready role in a validated single-pass dependency graph")
            break
        with ThreadPoolExecutor(max_workers=team.get("max_parallel", 2)) as pool:
            futures = {pool.submit(_run_agent, a, team, ctx, services): a for a in ready}
            for future in as_completed(futures):
                agent = futures[future]
                aid = agent["agent_id"]
                try:
                    results[aid] = future.result()
                except Exception as exc:
                    services.event(aid, "execution_error", str(exc))
                    results[aid] = RunResult(terminated_reason="error", metadata={"agent_id": aid,
                                                                                 "error": str(exc)})
                del pending[aid]
    for aid in pending:
        services.event(aid, "cancelled", "team timeout")
        results[aid] = RunResult(terminated_reason="cancelled", metadata={"agent_id": aid})
    if services.execution_mode == "iterative_shared_ledger":
        services.shared_ledger = _iterative_public_ledger(services)
    if services.ledger is not None:
        budget = services.ledger.snapshot()
        for agent in team["agents"]:
            aid = agent["agent_id"]
            own_records = [record for record in budget["records"] if record.get("agent_id") == aid]
            model_records = [record for record in own_records if record["kind"] == "model"]
            services.event(aid, "resource_usage", {
                "scope": "pre_submission_role_calls_including_local_planning",
                "model_calls": len(model_records),
                "input_tokens": sum(record["input_tokens"] for record in model_records),
                "output_tokens": sum(record["output_tokens"] for record in model_records),
                "tool_calls": sum(record["kind"] == "tool" for record in own_records),
                "communication_bytes": sum(record["bytes"] for record in own_records
                                           if record["kind"] == "communication"),
                "estimated": any(record["estimated"] for record in model_records), "cost": None})
            if aid in results:
                results[aid].metadata["event_ids"] = [event["event_id"] for event in services.events
                                                     if event["agent_id"] == aid]
        services.event("coordinator", "resource_usage", {
            "scope": "pre_submission_shared_task", "resource_budget": services.ledger.resource_context(),
            "by_stage": budget["by_stage"], "cost": None})
    final = results.get(team["synthesizer_id"],
                        RunResult(answer=None, terminated_reason="cancelled",
                                  metadata={"agent_id": team["synthesizer_id"]}))
    submitted = final.answer is not None and (services.execution_mode != "iterative_shared_ledger"
                                             or final.terminated_reason == "final_answer")
    return RunResult(answer=final.answer if submitted else None,
                     terminated_reason="final_answer" if submitted else "error",
                     sub_runs=[results.get(a["agent_id"],
                                           RunResult(terminated_reason="cancelled",
                                                     metadata={"agent_id": a["agent_id"]}))
                               for a in team["agents"]],
                     metadata={"run_id": services.run_id, "events": copy.deepcopy(services.events),
                               "artifacts": copy.deepcopy(services.artifacts),
                               "coordination": ("single_pass_shared_ledger"
                                                if services.execution_mode == "single_pass"
                                                else services.execution_mode),
                               **({"scheduling": {"strategy": "cooperative_shared_ledger",
                                                  "reactivate_completed_agents": True}}
                                  if services.execution_mode == "iterative_shared_ledger" else {}),
                               "shared_ledger": copy.deepcopy(services.shared_ledger),
                               "shared_ledger_hash": content_hash(services.shared_ledger),
                               "team_hash": content_hash(team), "model_calls_used": services.calls,
                               "timeout_seconds": services.timeout_seconds,
                               "call_counts": copy.deepcopy(services.call_counts)})


class TeamAction(BaseAction):
    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.team = None
        self.services = None

    def bind_team(self, team, services):
        self.team, self.services = team, services

    def run(self, task, ctx):
        if self.team is None or self.services is None:
            raise ValueError("rubric_mas requires a bound TeamSpec and TeamServices")
        return run_team(task, ctx, self.team, self.services)


class _CoordinatorModel:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("generated harness must use independently metered role models")


class _SinglePassModel:
    """Enforce the call cap even if generated Action code re-enters a role."""

    def __init__(self, model, services, agent, team):
        self.model, self.services, self.agent, self.team = model, services, agent, team

    def __call__(self, messages, **kwargs):
        aid = self.agent["agent_id"]
        self.services.reserve_call(self.team["total_max_calls"], aid, self.agent.get("max_calls"))
        self.services.event(aid, "agent_call", {
            "role": self.agent["role"], "call_index": self.services.call_counts.get(aid, 0),
            "execution_role": "writer" if aid == self.team["synthesizer_id"] else "contributor"})
        return self.model(messages, **kwargs)

    @property
    def _native_tool_registry(self):
        return getattr(self.model, "_native_tool_registry", {})

    @_native_tool_registry.setter
    def _native_tool_registry(self, value):
        self.model._native_tool_registry = value

    def __getattr__(self, name):
        return getattr(self.model, name)


class TeamExecutor:
    """Inject services into native JIT loader/runtime without modifying protocols."""

    def __init__(self, model_factory, tools=None, ledger=None, timeout_seconds=60.0,
                 unsafe_local=False):
        self.model_factory = model_factory
        self.tools = tools or {}
        self.ledger = ledger
        self.timeout_seconds = timeout_seconds
        self.unsafe_local = unsafe_local

    def execute(self, task, team, artifact, rubrics=None, experiences=()):
        if artifact.backend not in {"scripted", "native_jit"}:
            raise ValueError("unknown harness backend")
        task = PublicTask.model_validate(_data(task))
        team = TeamSpec.model_validate(_data(team))
        team_data = validate_team(team, task)
        if content_hash(_data(task)) != artifact.task_hash:
            raise ValueError("PublicTask differs from the frozen generation sidecar")
        if content_hash(team_data) != artifact.team_hash:
            raise ValueError("TeamSpec differs from the frozen generation sidecar")
        if artifact.backend == "native_jit" and not self.unsafe_local:
            raise RuntimeError("native_jit requires a trusted sandbox; none is configured. "
                               "Explicit unsafe-local is required for local generated code.")
        artifact.verify_integrity()
        pool_data = artifact.sidecar.get("agent_pool")
        agent_pool = AgentPoolSnapshot.model_validate(pool_data) if pool_data is not None else None
        if agent_pool is not None:
            from .agent_pool import validate_bindings
            validate_bindings(agent_pool, team)
        elif any(agent.pool_agent_id is not None for agent in team.agents):
            raise ValueError("Bound Agent Pool members require a frozen pool sidecar")
        if rubrics is None:
            rubrics = RubricGraph.model_validate(artifact.sidecar["rubrics"])
        elif content_hash(_data(rubrics)) != content_hash(artifact.sidecar["rubrics"]):
            raise ValueError("rubrics differ from the frozen generation sidecar")
        if not experiences:
            experiences = artifact.sidecar["experiences"]
        elif content_hash([_data(e) for e in experiences]) != content_hash(artifact.sidecar["experiences"]):
            raise ValueError("experiences differ from the frozen generation sidecar")
        loaded = load_harness(artifact.name)
        for key, protocol in [("memory", BaseMemory), ("planning", BasePlanning),
                              ("action", BaseAction), ("tool_policy", BaseToolPolicy)]:
            if not isinstance(loaded[key], protocol):
                raise TypeError(f"{key} must implement the original JIT protocol")
        if not callable(getattr(loaded["action"], "bind_team", None)):
            raise TypeError("generated Action must implement bind_team(team, services)")
        services = TeamServices(self.model_factory, task, rubrics, list(experiences),
                                self.ledger, self.timeout_seconds,
                                execution_mode=team_data.get("execution_mode", "single_pass"),
                                agent_pool=agent_pool)
        agents_by_id = {agent["agent_id"]: agent for agent in team_data["agents"]}

        def single_pass_model(agent_id):
            if agent_id not in agents_by_id:
                raise ValueError("Unknown role model")
            return _SinglePassModel(self.model_factory(agent_id), services,
                                    agents_by_id[agent_id], team_data)

        services.model_factory = single_pass_model
        try:
            return self._execute_bound(task, team_data, artifact, loaded, services, agents_by_id)
        except Exception as exc:
            exc.jit_mas_execution_started = services.calls > 0
            raise

    def _execute_bound(self, task, team_data, artifact, loaded, services, agents_by_id):
        loaded["action"].bind_team(team_data, services)
        # AgentRuntime.__init__ only adds a network client and default tools. Inject
        # existing clients here, then use its real lifecycle and Action dispatch.
        runtime = AgentRuntime.__new__(AgentRuntime)
        runtime.config = {"harness": artifact.name, "execution": {}}
        runtime.model = _CoordinatorModel()
        runtime.tool_registry = ToolRegistry()
        public_tools = set(_data(task)["tools"])
        runtime.tool_registry.register_batch({k: v for k, v in self.tools.items() if k in public_tools})
        runtime.memory, runtime.planning = loaded["memory"], loaded["planning"]
        runtime.action, runtime.tool_policy = loaded["action"], loaded["tool_policy"]
        runtime.harness_prompts = loaded["prompts"]
        runtime.tool_policy.initialize(runtime.tool_registry.get_all())
        runtime.logger = AgentLogger(level=LogLevel.OFF)
        runtime.max_steps = 1
        runtime.trace_dir, runtime._run_counter = "", 0
        try:
            result = runtime.run(_data(task)["question"])
            if not result.sub_runs or {r.metadata.get("agent_id") for r in result.sub_runs} != set(agents_by_id):
                raise RuntimeError("generated harness did not execute every bound role with local traces")
            if result.terminated_reason == "final_answer":
                if any((services.execution_mode == "single_pass" and len(run.trajectory) != 1) or
                        not run.trajectory or any(step.model_input_messages is None or
                        step.model_output_messages is None for step in run.trajectory)
                        for run in result.sub_runs):
                    raise RuntimeError("generated harness omitted full role traces")
                if services.calls != sum(len(run.trajectory) for run in result.sub_runs):
                    raise RuntimeError("generated harness role traces disagree with team call accounting")
        except Exception as exc:
            exc.jit_mas_execution_started = services.calls > 0
            raise
        result.metadata.update({"backend": artifact.backend, "harness": artifact.name,
                                "harness_hash": artifact.code_hash,
                                "unsafe_local": artifact.backend == "native_jit",
                                "software_test_only": artifact.backend == "scripted"})
        if services.agent_pool is not None:
            result.metadata["agent_pool_hash"] = content_hash(services.agent_pool)
        # Runtime sees a coordinator with no model calls. Sum disjoint leaf traces
        # here; the shared ledger is authoritative and is never charged again.
        result.metadata["input_token_count"] = sum(s.input_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["output_token_count"] = sum(s.output_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["total_token_count"] = (result.metadata["input_token_count"] +
                                                 result.metadata["output_token_count"])
        return result
