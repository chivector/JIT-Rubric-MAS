"""Bounded team primitives executed by an ordinary JIT Action module."""

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
from jit_mas.schemas import PublicTask, RubricGraph, TeamSpec, utc_now


def _data(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_data(value), sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _bounded_call(call: Callable, timeout: float, *args, **kwargs):
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
    try:
        ok, value = result.get(timeout=max(0.001, timeout))
    except queue.Empty as exc:
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
    messages: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    event_by_id: dict = field(default_factory=dict)

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
            self.event_by_id[event["event_id"]] = event
            return event["event_id"]

    def reserve_call(self, limit):
        with self.lock:
            if self.cancelled.is_set():
                raise TimeoutError("team cancelled after timeout")
            if self.calls >= limit:
                raise RuntimeError("TeamSpec.total_max_calls exhausted")
            self.calls += 1


def validate_team(team, public_task=None):
    team = _data(team)
    agents = team["agents"]
    ids = [a["agent_id"] for a in agents]
    if not agents or len(set(ids)) != len(ids):
        raise ValueError("TeamSpec requires unique nonempty agents")
    if team["synthesizer_id"] not in ids:
        raise ValueError("unknown synthesizer")
    if team.get("max_parallel", 2) < 1 or team.get("total_max_calls", 16) < 1:
        raise ValueError("team concurrency and call budgets must be positive")
    allowed = set(_data(public_task)["tools"]) if public_task is not None else None
    dependencies = {}
    for a in agents:
        deps = set(a.get("depends_on", []))
        if not deps.issubset(ids) or a["agent_id"] in deps:
            raise ValueError("unknown or self-referential agent dependency")
        if a.get("max_calls", 3) < 1 or a.get("max_tokens", 4096) < 1:
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


def _parse_response(response):
    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"answer": text} if text else {}
    return parsed if isinstance(parsed, dict) else {"answer": str(parsed)}


def _run_agent(agent, team, ctx, services):
    aid = agent["agent_id"]
    synth = aid == team["synthesizer_id"]
    before = services.event(aid, "agent_started", {"role": agent["role"]})
    dependencies = []
    for upstream in agent.get("depends_on", []):
        artifact = services.artifacts[upstream]
        sent = services.event(upstream, "message_sent", artifact, recipient=aid,
                              parents=[artifact["event_id"]])
        consumed = services.event(aid, "message_consumed", artifact, recipient=aid,
                                  parents=[sent])
        dependencies.append({**artifact, "consumed_event_id": consumed})
        if services.ledger is not None:
            services.ledger.charge_communication(len(json.dumps(artifact)))
    rubric_data = _data(services.rubrics) or {"rubrics": []}
    primary_rubrics = [rid for rid, owner in team.get("primary", {}).items() if owner == aid]
    review_rubrics = [rid for rid, reviewers in team.get("reviewers", {}).items() if aid in reviewers]
    assigned_rubrics = set(agent.get("rubric_ids", [])) | set(primary_rubrics) | set(review_rubrics)
    relevant = [r for r in rubric_data.get("rubrics", [])
                if r["rubric_id"] in assigned_rubrics]
    experiences = [_data(e) for e in services.experiences]
    local_experiences = [e for e in experiences
                         if e.get("kind", e.get("bank", "")) == "execution"
                         and (not e.get("capability") or e.get("capability") == agent["capability"])]
    instruction = {
        "public_task": _data(services.public_task), "agent": agent,
        "predicted_requirements": relevant, "upstream_artifacts": dependencies,
        "primary_rubrics": primary_rubrics, "review_rubrics": review_rubrics,
        "review_owners": {rid: team.get("primary", {}).get(rid) for rid in review_rubrics},
        "execution_experiences": local_experiences,
        "submission": "final_answer" if synth else "subtask_complete",
    }
    system = str(ctx.prompt_templates.get("agent_prompt", "")) + (
        "\nYou have an independent conversation. Follow only your assigned responsibilities. "
        "Use tools via {\"tools\":[{\"name\":\"...\",\"arguments\":{...}}]}. "
        "Complete with {\"answer\":\"...\",\"evidence_ids\":[],\"checkpoints\":{}}. "
        "Checkpoints must all be checked before completion. You may send_message(recipient,content), "
        "read_evidence(event_id), or raise_issue(content) to request missing input or dispute evidence. "
        "Never claim unobserved evidence or broadcast private conversations."
    )
    if primary_rubrics:
        system += "\nYou are the primary owner for primary_rubrics: produce evidence or artifacts addressing each."
    if review_rubrics:
        system += (
            "\nYou are the assigned reviewer for review_rubrics. Independently examine the primary "
            "owner's upstream artifacts against those requirements. Resolve or explicitly report "
            "contradictions and missing evidence; use raise_issue or request input when needed. "
            "Your review must identify what was checked and remaining uncertainty."
        )
    if synth:
        system += (
            "\nSynthesize a coherent final response: resolve contradictory artifacts, state unresolved "
            "gaps and uncertainty, and preserve relevant sources. Do not merely concatenate outputs. "
            "Cite evidence_ids only for artifacts actually incorporated; citations are optional for creative tasks."
        )
    memory = type(ctx.memory)(prompts=ctx.prompt_templates)
    memory.initialize(system, TaskInput(task=json.dumps(instruction, ensure_ascii=False)))
    model = services.model_factory(aid)
    allowed = set(agent.get("tools", []))
    catalog = ctx.tool_policy.select_tools("", 0, memory.build_context()).tools
    absent = allowed - set(catalog)
    if absent:
        raise ValueError(f"tools not installed: {sorted(absent)}")
    model._native_tool_registry = {name: catalog[name] for name in allowed}
    trajectory = []
    answer = None
    reason = "max_calls"
    evidence_ids = []
    for step_index in range(min(agent.get("max_calls", 3), ctx.max_steps)):
        messages = memory.build_context().messages
        with services.lock:
            incoming = services.messages.pop(aid, [])
        for message in incoming:
            consumed = services.event(aid, "message_consumed", message["content"],
                                      parents=[message["event_id"]])
            messages.append({"role": "user", "content": json.dumps(
                {"directed_message": message, "consumed_event_id": consumed})})
        tool_schemas = ctx.get_tool_schemas({name: catalog[name] for name in allowed}) if allowed else "[]"
        messages.append({"role": "user", "content": "Allowed tool schemas: " + tool_schemas})
        step = StepRecord(step_number=step_index + 1, model_input_messages=copy.deepcopy(messages),
                          start_time=time.time())
        observations = []
        try:
            services.reserve_call(team.get("total_max_calls", 16))
            response = _bounded_call(model, services.timeout_seconds, messages,
                                     max_tokens=agent.get("max_tokens", 4096))
            step.model_output_messages = response
            usage = model.get_token_counts() if hasattr(model, "get_token_counts") else {}
            step.input_token_count = int(usage.get("input_token_count", 0))
            step.output_token_count = int(usage.get("output_token_count", 0))
            step.total_token_count = step.input_token_count + step.output_token_count
            output_event = services.event(aid, "model_output", step.full_dict()["model_output_messages"],
                                          parents=[before])
            parsed = _parse_response(response)
            for call in parsed.get("tools", []):
                name = call["name"]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                step.tool_calls.append(ToolCall(name=name, arguments=copy.deepcopy(args)))
                if name in {"final_answer", "complete"}:
                    if name == "final_answer" and not synth:
                        raise PermissionError("only the synthesizer may submit final_answer")
                    parsed.update(args)
                    continue
                if services.ledger is not None:
                    services.ledger.charge_tool(stage="execution", agent_id=aid, tool_name=name)
                if name == "send_message":
                    recipient = args["recipient"]
                    if recipient not in {a["agent_id"] for a in team["agents"]}:
                        raise ValueError("unknown message recipient")
                    event_id = services.event(aid, "message_sent", args["content"],
                                              parents=[output_event], recipient=recipient)
                    with services.lock:
                        services.messages.setdefault(recipient, []).append({
                            "event_id": event_id, "sender": aid, "content": args["content"]})
                    if services.ledger is not None:
                        services.ledger.charge_communication(len(str(args["content"])))
                    observation = {"sent_event_id": event_id}
                elif name == "read_evidence":
                    event = services.event_by_id.get(args["event_id"])
                    if event is None or event["kind"] not in {"retrieved", "artifact_published", "issue"}:
                        raise PermissionError("only shared evidence and artifacts may be read")
                    services.event(aid, "evidence_read", event["content"],
                                   parents=[event["event_id"]])
                    observation = event
                elif name == "raise_issue":
                    observation = {"event_id": services.event(aid, "issue", args["content"],
                                                               parents=[output_event])}
                else:
                    if name not in allowed:
                        raise PermissionError(f"tool '{name}' is not allowed for agent '{aid}'")
                    observation = _bounded_call(ctx.execute_tool, services.timeout_seconds, name, args)
                    event_id = services.event(aid, "retrieved", {"tool": name, "arguments": args,
                                                               "output": observation},
                                              parents=[output_event], source=name)
                    observation = {"event_id": event_id, "output": observation}
                observations.append(json.dumps(observation, ensure_ascii=False))
            if parsed.get("answer") is not None:
                missing = [name for name in agent.get("checkpoints", [])
                           if parsed.get("checkpoints", {}).get(name) is not True]
                if missing:
                    observations.append("Unconfirmed checkpoints: " + json.dumps(missing))
                else:
                    evidence_ids = parsed.get("evidence_ids", [])
                    if any(e not in services.event_by_id for e in evidence_ids):
                        raise ValueError("completion cites unknown evidence event")
                    answer = parsed["answer"]
                    step.action_output = answer
                    reason = "final_answer" if synth else "subtask_complete"
            if not parsed:
                observations.append("Empty response; complete your assignment or request a tool.")
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
        if answer is not None or reason == "error":
            break
    if answer is not None:
        artifact_id = services.event(aid, "artifact_published", answer, parents=evidence_ids)
        services.artifacts[aid] = {"agent_id": aid, "answer": answer, "event_id": artifact_id,
                                   "content_hash": content_hash(answer), "version": 1}
        if synth:
            services.event(aid, "final_answer", answer, parents=evidence_ids)
    return RunResult(answer=answer, trajectory=trajectory, terminated_reason=reason,
                     metadata={"agent_id": aid, "capability": agent["capability"],
                               "role": agent["role"], "rubric_ids": sorted(assigned_rubrics),
                               "primary_rubrics": primary_rubrics, "review_rubrics": review_rubrics,
                               "event_ids": [e["event_id"] for e in services.events if e["agent_id"] == aid]})


def run_team(task, ctx, team, services):
    """Action-owned run loop: dependency scheduling followed by final synthesis."""
    team = validate_team(team, services.public_task)
    ctx.planning.bind_team(team)
    plan = ctx.planning.init_plan(task, ctx.memory.build_context(), "", ctx.model)
    ctx.memory.update_plan(plan)
    pending = {a["agent_id"]: a for a in team["agents"]}
    results = {}
    while pending and not services.cancelled.is_set():
        failed = {aid for aid, result in results.items() if result.answer is None}
        blocked = [aid for aid, agent in pending.items() if failed.intersection(agent.get("depends_on", []))]
        for aid in blocked:
            services.event(aid, "dependency_failed", {"failed": sorted(failed)})
            results[aid] = RunResult(terminated_reason="dependency_failed", metadata={"agent_id": aid})
            del pending[aid]
        ready = [agent for agent in pending.values()
                 if set(agent.get("depends_on", [])).issubset(results)]
        if not ready:
            if pending:
                continue  # Failed dependency propagation may span several layers.
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
    final = results[team["synthesizer_id"]]
    return RunResult(answer=final.answer, terminated_reason="final_answer" if final.answer is not None else "error",
                     sub_runs=[results[a["agent_id"]] for a in team["agents"]],
                     metadata={"run_id": services.run_id, "events": copy.deepcopy(services.events),
                               "artifacts": copy.deepcopy(services.artifacts),
                               "team_hash": content_hash(team), "model_calls_used": services.calls,
                               "timeout_seconds": services.timeout_seconds})


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
                                self.ledger, self.timeout_seconds)
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
        runtime.max_steps = max(a.get("max_calls", 3) for a in team_data["agents"])
        runtime.trace_dir, runtime._run_counter = "", 0
        result = runtime.run(_data(task)["question"])
        if not result.sub_runs or {r.metadata.get("agent_id") for r in result.sub_runs} != {
                a["agent_id"] for a in team_data["agents"]}:
            raise RuntimeError("generated harness did not execute every bound role with local traces")
        if result.terminated_reason == "final_answer":
            if any(not run.trajectory or any(step.model_input_messages is None or
                    step.model_output_messages is None for step in run.trajectory)
                    for run in result.sub_runs):
                raise RuntimeError("generated harness omitted full observed role traces")
            if services.calls != sum(len(run.trajectory) for run in result.sub_runs):
                raise RuntimeError("generated harness role traces disagree with team call accounting")
        result.metadata.update({"backend": artifact.backend, "harness": artifact.name,
                                "harness_hash": artifact.code_hash,
                                "unsafe_local": artifact.backend == "native_jit",
                                "software_test_only": artifact.backend == "scripted"})
        # Runtime sees a coordinator with no model calls. Sum disjoint leaf traces
        # here; the shared ledger is authoritative and is never charged again.
        result.metadata["input_token_count"] = sum(s.input_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["output_token_count"] = sum(s.output_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["total_token_count"] = (result.metadata["input_token_count"] +
                                                 result.metadata["output_token_count"])
        return result
