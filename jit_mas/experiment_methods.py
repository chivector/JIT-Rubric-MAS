"""Unscored experimental controls using JIT's real generation and runtime APIs."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import time
import uuid
from pathlib import Path

from jit.harness_ops import SECTION_TAG_TO_FILE, WORKSPACE_DIR, _parse_harness_response
from jit.meta_agent import MetaReActAgent
from jit.schemas import MetaAgentRequest
from scripts.kernel.loader import load_harness
from scripts.kernel.monitoring import AgentLogger, LogLevel
from scripts.kernel.runtime import AgentRuntime
from scripts.models.base import ChatMessage
from scripts.tools.registry import ToolRegistry

from .bridge import SEED_DIR, SynthesizedHarness, seed_response
from .budget import BudgetExceeded, BudgetLedger, MeteredModel
from .checkpoints import CheckpointIntegrityError
from .execution import TeamExecutor, content_hash
from .experience import retrieve
from .pipeline import write_json
from .planning import GlobalAnalyzer, JsonModelCalls, PREDICT_PROMPT
from .schemas import (AgentSpec, ExperienceSnapshot, PlannedTeam, Prediction, PublicTask,
                      RubricGraph, TeamSpec, digest, utc_now)


METHODS = {"jit_matched", "rubric_fixed", "G", "GO"}
QUALITY_PROMPT = (
    "Infer observable quality requirements from this public task and applicable historical "
    "experience. Return only a RubricGraph. Do not answer the task or propose agents, roles, "
    "assignments, teams, communication patterns, or execution topology. Requirements remain "
    "fallible hypotheses; preserve uncertainty and do not invent mandatory factual answers. "
    "No evaluator criteria, reference answers, or held-out feedback is available."
)


def _write_once(path, value):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _budget(config):
    return BudgetLedger(config.max_model_calls, config.max_total_tokens, config.max_tool_calls,
                        timeout_seconds=config.task_timeout)


def _analyzer(pipeline, ledger, *, explicit_rubrics=True, graph=None):
    config = pipeline.config
    kwargs = dict(max_agents=config.max_agents, max_parallel=config.max_parallel,
                  total_max_calls=config.team_max_calls, local_rounds=config.local_rounds,
                  explicit_rubrics=explicit_rubrics,
                  budget_context=ledger.resource_context,
                  execution_max_tokens=config.models["exec"].max_tokens if "exec" in config.models else 8192)
    cls = _FrozenQualityAnalyzer if graph is not None else GlobalAnalyzer
    return cls(pipeline.models.create("global", "global", ledger, "inference"),
               lambda aid: pipeline.models.create("local", aid, ledger, "inference"),
               **kwargs, **({"fixed_graph": graph} if graph is not None else {}))


class _FrozenQualityAnalyzer(GlobalAnalyzer):
    """Organization may change; the paired control's quality treatment may not."""

    def __init__(self, *args, fixed_graph, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_graph = fixed_graph.model_copy(deep=True)

    def _prompt(self, prompt):
        return super()._prompt(prompt) + (
            "\nCONTROL: The supplied quality graph is fixed. Copy its rubric and edge "
            "records exactly; do not add, remove, or rewrite requirements. Organize roles "
            "and responsibilities around this graph. Local quality additions are logged "
            "but not adopted in this paired fixed-guidance experiment."
        )

    def predict(self, task, experiences=()):
        def validate(item):
            self._validate_prediction(task, item)
            if digest(item.graph) != digest(self.fixed_graph):
                raise ValueError("Copy fixed_quality_graph exactly; only candidate organization may change")

        prediction = self.ask(self.global_model, "predict", self._prompt(PREDICT_PROMPT),
                              {"task": task, "experiences": experiences,
                               "fixed_quality_graph": self.fixed_graph, "limits": self._limits()},
                              Prediction, validate=validate, refresh_payload=self._refresh_limits)
        self.last_prediction = prediction.model_copy(deep=True)
        return prediction

    def local_plan(self, *args, **kwargs):
        plan = super().local_plan(*args, **kwargs)
        known = {rubric.rubric_id for rubric in self.fixed_graph.rubrics}
        # Raw local testimony remains in call_records; neither condition evolves R_star.
        return plan.model_copy(update={"additions": [],
                                       "rubric_ids": [rid for rid in plan.rubric_ids if rid in known]})

    def _validate_reconciliation(self, task, result):
        super()._validate_reconciliation(task, result)
        if digest(result.graph) != digest(self.fixed_graph):
            raise ValueError("The shared quality graph is immutable in this control")


class _QualityFixture:
    """Explicit synthetic transport for the new graph-only schema, never native."""

    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        from .offline import FixtureModel

        # Reuse the fixture's quality requirements, never expose its candidate roster.
        provider = type("FixtureProvider", (), {})()
        fixture = FixtureModel(provider, "global", "quality")
        raw = fixture._phase({"phase": "predict", "task": payload["task"],
                              "experiences": payload["experiences"],
                              "limits": {"max_agents": 3, "total_max_calls": 3}})["graph"]
        return ChatMessage(role="assistant", content=json.dumps(raw))

    def get_token_counts(self):
        return {"input_token_count": 100, "output_token_count": 80}


def _shared_quality(pipeline, task, snapshot, repeat, shared_dir, experiences):
    identity = {"task": task.model_dump(mode="json"), "snapshot": digest(snapshot), "repeat": repeat,
                "config": pipeline.config.model_dump(mode="json"), "code": pipeline.code_hash,
                "method_code": _file_hash(__file__), "prompt": QUALITY_PROMPT}
    key = digest(identity)
    path = Path(shared_dir) / f"{key}.json"
    marker = path.with_suffix(".started.json")
    if path.exists():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            graph = RubricGraph.model_validate(row["graph"])
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointIntegrityError("Shared R_star cache schema mismatch") from exc
        if (row.get("identity") != identity or row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
                or row.get("graph_hash") != digest(graph)):
            raise CheckpointIntegrityError("Shared R_star cache integrity mismatch")
        return graph, row, path, True
    if marker.exists():
        raise RuntimeError("Interrupted R_star preparation cannot be silently resampled")
    _write_once(marker, {"identity": identity})
    ledger = _budget(pipeline.config)
    calls = JsonModelCalls(max_corrections=0)
    if pipeline.config.backend == "scripted":
        model = MeteredModel(_QualityFixture(), ledger, "shared_rstar", "quality", 16000)
    else:
        model = pipeline.models.create("global", "quality", ledger, "shared_rstar")
    try:
        graph = calls.ask(model, "quality_only", QUALITY_PROMPT,
                          {"task": task, "experiences": experiences}, RubricGraph)
        if not graph.rubrics:
            raise ValueError("Shared quality guidance must contain at least one requirement")
        row = {"identity": identity, "graph": graph.model_dump(mode="json"), "graph_hash": digest(graph),
               "budget": ledger.snapshot(), "calls": calls.call_records,
               "software_test_only": pipeline.config.backend == "scripted"}
        row["sha256"] = digest(row)
        _write_once(path, row)
        return graph, row, path, False
    except BaseException as exc:
        _write_once(path.with_suffix(".failure.json"), {"identity": identity, "budget": ledger.snapshot(),
                    "calls": calls.call_records, "error_type": type(exc).__name__})
        raise


class _GuidedRoleModel:
    def __init__(self, model, graph, artifact, observed):
        self.model, self.graph, self.artifact, self.observed = model, graph, artifact, observed

    def __call__(self, messages, **kwargs):
        self.artifact.verify_integrity()
        augmented = copy.deepcopy(messages)
        payload = json.loads(augmented[1]["content"])
        # GO may assign requirement IDs, but actors in both treatments get the same
        # requirement text once, through this post-freeze field alone.
        payload["predicted_requirements"] = []
        payload["quality_guidance"] = self.graph.model_dump(mode="json")
        payload["quality_guidance_hash"] = digest(self.graph)
        augmented[1]["content"] = json.dumps(payload, ensure_ascii=False)
        augmented[0]["content"] += (
            "\nRead quality_guidance as fallible common requirements. Requirement text appears "
            "there once; responsibility/ownership IDs remain organization metadata. No additional "
            "role call or inter-agent communication is allowed."
        )
        self.observed.append({"agent_id": payload["agent"]["agent_id"],
                              "graph_hash": digest(self.graph), "frozen_harness_hash": self.artifact.code_hash,
                              "messages": copy.deepcopy(augmented)})
        # The caller's leaf trace must show the actual messages sent to the model.
        messages[:] = augmented
        return self.model(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class _GenerationPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, messages, **kwargs):
        messages = copy.deepcopy(messages)
        supplement = (
            "\nEXPERIMENT CONTROL: Do not copy requirement text into generated agent prompts "
            "or Python literals. Task-specific organization may reference requirement IDs. "
            "Actual common quality guidance is supplied only to role inputs after the harness "
            "has been frozen. Do not construct API clients; use the installed runtime models."
        )
        if isinstance(messages[0]["content"], list):
            messages[0]["content"].append({"type": "text", "text": supplement})
        else:
            messages[0]["content"] += supplement
        return self.model(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _fixed_team(config, graph):
    if config.max_agents < 3 or config.team_max_calls < 3:
        raise ValueError("rubric_fixed requires three agents within the common configured cap")
    ids = [rubric.rubric_id for rubric in graph.rubrics]
    max_tokens = config.models["exec"].max_tokens if "exec" in config.models else 8192
    agents = [AgentSpec(agent_id="analyst", role="Analyst", capability="task analysis",
                        rubric_ids=ids, responsibilities=["Publish requirements and a task outline to the ledger"],
                        max_calls=1, max_tokens=max_tokens),
              AgentSpec(agent_id="evidence", role="Evidence", capability="source verification",
                        rubric_ids=ids, responsibilities=["Publish source spans and provenance from the shared evidence"],
                        max_calls=1, max_tokens=max_tokens),
              AgentSpec(agent_id="writer", role="Writer", capability="synthesis", rubric_ids=ids,
                        responsibilities=["Read the completed ledger once and synthesize the final artifact"],
                        depends_on=["analyst", "evidence"], max_calls=1, max_tokens=max_tokens)]
    return TeamSpec(agents=agents, synthesizer_id="writer", max_parallel=min(2, config.max_parallel),
                    total_max_calls=3, coverage={rid: [agent.agent_id for agent in agents] for rid in ids},
                    primary={rid: "writer" for rid in ids}, selection_rationale="Preregistered fixed three-role baseline")


def _static_harness(task, graph, team, backend):
    name = "experiment_fixed_" + uuid.uuid4().hex
    directory = WORKSPACE_DIR.parent / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    files = _parse_harness_response(seed_response())
    for filename, text in files.items():
        (directory / filename).write_text(text, encoding="utf-8")
    sidecar = {"schema_version": "1.0", "task": task.model_dump(mode="json"),
               "rubrics": graph.model_dump(mode="json"), "team": team.model_dump(mode="json"),
               "experiences": [], "backend": backend}
    write_json(directory / "team.json", sidecar)
    return SynthesizedHarness(name, directory, backend, content_hash(team), content_hash(task),
                              content_hash(files), content_hash(sidecar), [],
                              selection={"kind": "fixed_checked_in_seed", "source": str(SEED_DIR)})


class _PublicNativeAdapter:
    def __init__(self, task, tools):
        self.task, self.tools = task, tools

    def format_task(self, item):
        return self.task.question + "\n\nPublic constraints:\n" + json.dumps(self.task.constraints)

    def get_tools(self):
        return list(self.tools)

    def get_task_tools(self, item):
        return self.tools


class _UpstreamFixture:
    def __init__(self, generation=False):
        self.generation = generation

    def __call__(self, messages, **kwargs):
        if self.generation:
            directory = SEED_DIR.parent / "flash_searcher"
            content = "\n\n".join(f"<<<{tag}>>>\n{(directory / filename).read_text(encoding='utf-8')}\n<<<END_{tag}>>>"
                                    for tag, filename in SECTION_TAG_TO_FILE.items())
        else:
            content = json.dumps({"think": "Synthetic native runtime software assertion",
                                  "tools": [{"name": "final_answer", "arguments": {
                                      "answer": "Synthetic native JIT artifact with explicit limitations."}}]})
        return ChatMessage(role="assistant", content=content)

    def get_token_counts(self):
        return {"input_token_count": 100, "output_token_count": 100}


class _NativeResourceStop(BaseException):
    """Escape generated Exception retries only for exhausted shared resources."""

    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


class _DeadlineModel:
    def __init__(self, model, timeout, *, ledger=None, on_settled=None):
        self.model, self.deadline = model, time.monotonic() + timeout
        self.ledger, self.on_settled = ledger, on_settled

    def expired(self):
        return (time.monotonic() >= self.deadline or
                (self.ledger is not None and self.ledger.remaining_seconds() == 0))

    def __call__(self, *args, **kwargs):
        try:
            if self.expired():
                raise _NativeResourceStop(TimeoutError("Native control task deadline exceeded"))
            try:
                return self.model(*args, **kwargs)
            except BudgetExceeded as exc:
                raise _NativeResourceStop(exc) from exc
            except Exception as exc:
                if self.expired():
                    raise _NativeResourceStop(TimeoutError("Native control task deadline exceeded")) from exc
                raise
        finally:
            if self.on_settled is not None:
                self.on_settled()

    def __getattr__(self, name):
        return getattr(self.model, name)


def _native_client_policy(files):
    """Reject direct unmetered client mistakes, without claiming code sandboxing."""
    forbidden = {"openai", "anthropic", "litellm", "requests", "httpx", "socket", "subprocess", "importlib"}
    findings = []
    for filename, source in files.items():
        if not filename.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(source, filename)):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".")[0] in forbidden or name.startswith(("urllib.request", "scripts.models.openai_server"))
                   for name in imports):
                findings.append(f"{filename}: direct client/network imports are forbidden; use ctx.model and ctx.execute_tool")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"exec", "eval", "__import__", "OpenAI", "AsyncOpenAI", "OpenAIServerModel", "AgentRuntime"}:
                    findings.append(f"{filename}: {name} bypasses the injected metered runtime")
    return "\n".join(sorted(set(findings)))


def _native_final_tool_receipt(registry, arguments, ledger, receipts):
    """Preserve AgentRuntime._execute_tool behavior while observing actual success."""
    event = {"event_id": f"native-final-tool:{len(receipts) + 1}",
             "tool_name": "final_answer", "arguments": copy.deepcopy(arguments),
             "model_calls_at_submission": ledger.snapshot()["model_calls"]}
    try:
        tool = registry.get("final_answer")
        value = tool(**arguments) if isinstance(arguments, dict) else tool(arguments)
        observation = str(value)
        event.update(success=True, answer=observation, observation=observation)
    except Exception as exc:
        observation = f"Error executing tool 'final_answer': {str(exc)}"
        event.update(success=False, error_type=type(exc).__name__, observation=observation)
    receipts.append(event)
    return observation


def _normalize_native_final_termination(result, receipts, current_model_calls):
    """Normalize the observed alias only when trusted tool and terminal trace agree.

    The runtime receipt records a successful registered tool invocation, not a
    test of the answer's meaning. Explicit non-tool terminal paths retain their
    original behavior. No draft, candidate or answer text is selected or edited.
    """
    original = result.terminated_reason
    last = receipts[-1] if receipts else {}
    answer = result.answer
    args = last.get("arguments")
    valid = (last.get("success") is True and isinstance(answer, str) and bool(answer.strip())
             and isinstance(args, dict) and set(args) == {"answer"}
             and args["answer"] == answer == last.get("answer") == last.get("observation")
             and last.get("model_calls_at_submission") == current_model_calls)

    def value(row, key):
        return row.get(key) if isinstance(row, dict) else getattr(row, key, None)

    steps = result.trajectory
    step = steps[-1] if steps else None
    calls = value(step, "tool_calls") or []
    tool = calls[-1] if calls else None
    valid = bool(valid and step is not None and value(step, "error") is None
                 and value(step, "action_output") == answer
                 and value(step, "observations") == answer
                 and value(tool, "name") == "final_answer"
                 and value(tool, "arguments") == args)
    normalized = original == "final_answer_called" and valid
    result.metadata["native_terminal"] = {
        "version": "confirmed-native-final-tool-v1", "original_terminated_reason": original,
        "verified_final_tool_submission": valid, "normalized": normalized,
        "rule": "observed final_answer_called alias; registered tool succeeded; answer and last terminal trace match; no later model calls",
        "tool_receipts": copy.deepcopy(receipts),
    }
    if normalized:
        result.terminated_reason = "final_answer"
    return result


def _native_jit(pipeline, task, ledger, run_dir):
    config = pipeline.config
    if config.backend == "native_jit" and not config.unsafe_local:
        raise RuntimeError("Native JIT control requires explicit unsafe-local or an installed sandbox")
    if config.backend == "scripted":
        meta = MeteredModel(_UpstreamFixture(True), ledger, "inference", "native-meta", 64000)
        execution = MeteredModel(_UpstreamFixture(), ledger, "inference", "native-executor", 8192)
    else:
        meta = pipeline.models.create("meta", "native-meta", ledger, "inference")
        execution = pipeline.models.create("exec", "native-executor", ledger, "inference")
    name = "experiment_native_" + uuid.uuid4().hex
    agent = MetaReActAgent({"model_id": "injected-metered-model", "api_base": "http://127.0.0.1:1/v1", "api_key": "EMPTY"},
                          {"meta_references": {"mode": "desc"}, "meta_review": {"enabled": False}},
                          workspace_name=name)
    agent.model = meta
    registry = ToolRegistry()
    registry.register_defaults(["final_answer"])
    registry.register_batch({key: value for key, value in pipeline.tools.items() if key in task.tools})
    adapter = _PublicNativeAdapter(task, registry.get_all())
    try:
        generated = agent.run(MetaAgentRequest(benchmark_adapter=adapter, item={"id": task.task_id},
                                             tools=list(registry.get_all()), generate_only=True, max_repairs=0))
    finally:
        write_json(run_dir / "model_calls.json", {"generation": meta.calls, "execution": execution.calls})
    repairs = []
    loaded = None
    for attempt in range(config.max_repairs + 1):
        emitted = {filename for call in meta.calls for filename in _parse_harness_response(call["content"])}
        missing = set(SECTION_TAG_TO_FILE.values()) - emitted
        error = (f"Generated response did not emit required files: {sorted(missing)}" if missing else
                 agent._static_harness_checks())
        if error.startswith("No static errors detected"):
            files = {filename: (agent.workspace_dir / filename).read_text(encoding="utf-8")
                     for filename in SECTION_TAG_TO_FILE.values()}
            error = _native_client_policy(files)
            if config.backend == "scripted":
                trusted = {filename: (SEED_DIR.parent / "flash_searcher" / filename).read_text(encoding="utf-8")
                           for filename in SECTION_TAG_TO_FILE.values()}
                if any(files[key].strip() != trusted[key].strip() for key in files):
                    raise ValueError("Offline native JIT executes only the trusted upstream fixture seed")
            if not error:
                try:
                    loaded = load_harness(name)
                except Exception as exc:
                    error = f"Native loader interface error: {type(exc).__name__}: {exc}"
            if not error:
                break
        if attempt == config.max_repairs:
            raise ValueError(error)
        try:
            repairs.append(agent._repair_harness(error, [{"error": error}], evaluation_result={}, validation_error=error))
        finally:
            write_json(run_dir / "model_calls.json", {"generation": meta.calls, "execution": execution.calls})
    files = {filename: (agent.workspace_dir / filename).read_text(encoding="utf-8")
             for filename in SECTION_TAG_TO_FILE.values()}
    # Upstream harnesses use integer max_steps in range/comparisons. When the
    # common call cap is absent, derive this interface bound from the existing
    # token envelope rather than silently adding an arbitrary step/call cap.
    # The injected metered model and deadline remain the actual resource guards.
    max_steps = config.max_model_calls if config.max_model_calls is not None else config.max_total_tokens
    runtime_budget = {"model_call_cap": config.max_model_calls, "max_total_tokens": config.max_total_tokens,
                      "task_timeout": config.task_timeout, "execution_timeout": config.execution_timeout,
                      "max_steps_interface": max_steps,
                      "max_steps_origin": "configured_model_call_cap" if config.max_model_calls is not None
                                          else "derived_from_existing_token_envelope",
                      "independent_step_cap": False,
                      "resource_guards": "shared metered token/time ledger and execution deadline"}
    write_json(run_dir / "harness.json", {"kind": "upstream_MetaReActAgent", "name": name,
               "path": str(agent.workspace_dir), "files": {key: digest(value) for key, value in files.items()},
               "meta_trajectory": generated.meta_agent_trajectory, "repairs": repairs,
               "runtime_budget": runtime_budget})
    if loaded is None:
        raise RuntimeError("Native loader did not produce a validated harness")
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.config = {"harness": name, "execution": {"model_call_budget": config.max_model_calls}}
    def persist_execution_audit():
        write_json(run_dir / "model_calls.json", {"generation": meta.calls, "execution": execution.calls})
        write_json(run_dir / "budget.json", ledger.snapshot())

    runtime.model = _DeadlineModel(execution, config.execution_timeout, ledger=ledger,
                                   on_settled=persist_execution_audit)
    runtime.tool_registry = registry
    runtime.memory, runtime.planning = loaded["memory"], loaded["planning"]
    runtime.action, runtime.tool_policy = loaded["action"], loaded["tool_policy"]
    runtime.harness_prompts = loaded["prompts"]
    runtime.tool_policy.initialize(registry.get_all())
    if hasattr(runtime.memory, "set_model"):
        runtime.memory.set_model(runtime.model)
    runtime.logger = AgentLogger(level=LogLevel.OFF)
    runtime.max_steps = max_steps
    runtime.trace_dir, runtime._run_counter = "", 0
    original_tool = runtime._execute_tool
    final_tool_receipts = []

    def execute_tool(name, arguments):
        if runtime.model.expired():
            raise _NativeResourceStop(TimeoutError("Native control task deadline exceeded"))
        if name == "final_answer":
            return _native_final_tool_receipt(registry, arguments, ledger, final_tool_receipts)
        try:
            ledger.charge_tool("inference", "native-executor", name)
        except BudgetExceeded as exc:
            raise _NativeResourceStop(exc) from exc
        return original_tool(name, arguments)

    runtime._execute_tool = execute_tool
    try:
        result = runtime.run(adapter.format_task({}))
        return _normalize_native_final_termination(result, final_tool_receipts,
                                                   ledger.snapshot()["model_calls"])
    except _NativeResourceStop as exc:
        raise exc.error from exc
    finally:
        persist_execution_audit()


def submit_method(pipeline, task_id, snapshot: ExperienceSnapshot, *, method, repeat, output_dir, shared_dir=None):
    """Submit one frozen test artifact; never construct a judge or read private rows."""
    if method not in METHODS:
        raise ValueError(f"Unsupported experimental control: {method}")
    if task_id not in pipeline.manifest.test:
        raise ValueError("Experimental control submissions are restricted to test tasks")
    if type(repeat) is not int or repeat < 0:
        raise ValueError("repeat must be a nonnegative integer")
    if method in {"jit_matched", "rubric_fixed"} and snapshot.experiences:
        raise ValueError("Static baselines must use the initial empty experience state")
    task = pipeline.tasks[task_id]
    if task.tools or pipeline.tools or pipeline.config.available_tools:
        raise ValueError("These controls require fixed shared evidence with no actor external tools")
    identity = {"method": method, "task": task.model_dump(mode="json"), "snapshot": digest(snapshot),
                "repeat": repeat, "config": pipeline.config.model_dump(mode="json"),
                "code": pipeline.code_hash, "method_code": _file_hash(__file__)}
    run_key = digest(identity)
    run_dir = Path(output_dir) / run_key
    if run_dir.exists():
        raise RuntimeError("Control submission already started; reuse its sealed outcome, do not resample")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run_manifest.json", identity)
    ledger = _budget(pipeline.config)
    shared_record = None
    try:
        if method == "jit_matched":
            result = _native_jit(pipeline, task, ledger, run_dir)
        else:
            experiences = retrieve(snapshot, task, excluded_task_ids=pipeline.manifest.validation + pipeline.manifest.test)
            if not pipeline.config.persistent_experience:
                experiences = []
            graph = None
            if method in {"G", "GO"}:
                graph, shared, path, reused = _shared_quality(
                    pipeline, task, snapshot, repeat,
                    shared_dir or Path(output_dir).parent / "shared_rstar", experiences)
                shared_record = {"path": str(path), "sha256": shared["sha256"], "graph_hash": shared["graph_hash"],
                                 "reused": reused, "preparation_budget": shared["budget"],
                                 "cost_semantics": "Preparation counted once by sha256; each condition deployment includes one preparation"}
                write_json(run_dir / "shared_rstar.json", shared_record)
            analyzer = _analyzer(pipeline, ledger, explicit_rubrics=method != "G", graph=graph if method == "GO" else None)
            try:
                if method == "rubric_fixed":
                    prediction = analyzer.predict(task, [])
                    planned = PlannedTeam(graph=prediction.graph, team=_fixed_team(pipeline.config, prediction.graph))
                else:
                    planned = analyzer.build(task, experiences, local_planning=pipeline.config.local_planning)
            finally:
                write_json(run_dir / "planning_calls.json", analyzer.call_records)
            write_json(run_dir / "frozen_plan.json", {"graph": planned.graph.model_dump(mode="json"),
                       "team": planned.team.model_dump(mode="json"), "hashes": {
                           "graph": digest(planned.graph), "team": digest(planned.team)}})
            synth = None
            if method == "rubric_fixed":
                artifact = _static_harness(task, planned.graph, planned.team, pipeline.config.backend)
            else:
                meta = pipeline.models.create("meta", "meta", ledger, "inference")
                synth = pipeline.synthesizer_factory(_GenerationPolicy(meta))
                artifact = synth.synthesize(task, planned.graph, planned.team, experiences=experiences)
            artifact.verify_integrity()
            write_json(run_dir / "harness.json", artifact.to_dict())
            observed = []

            def role_model(aid):
                model = pipeline.models.create("exec", aid, ledger, "inference")
                return _GuidedRoleModel(model, graph, artifact, observed) if graph is not None else model

            executor = TeamExecutor(role_model, tools=pipeline.tools, ledger=ledger,
                                    timeout_seconds=pipeline.config.execution_timeout,
                                    unsafe_local=pipeline.config.unsafe_local)
            try:
                result = (synth.execute_with_repair(executor, task, planned.team, artifact,
                                                   rubrics=planned.graph, experiences=experiences) if synth else
                          executor.execute(task, planned.team, artifact, rubrics=planned.graph))
            finally:
                write_json(run_dir / "harness.json", artifact.to_dict())
                write_json(run_dir / "guidance_injection.json", observed)
        write_json(run_dir / "execution.json", result.full_dict())
        if result.terminated_reason != "final_answer" or result.answer is None:
            raise RuntimeError("Control did not produce a final artifact")
        submission = {"answer": result.answer, "answer_hash": digest(result.answer), "submitted_at": utc_now()}
        write_json(run_dir / "submission.json", submission)
        outcome = {"run_key": run_key, "task_id": task_id, "mode": "evaluate", "method": method,
                   "repeat": repeat, "status": "submitted_unscored", "evaluation": None, "proposals": [],
                   "experience_updates": [], "experience_version": snapshot.version, "experience_hash": digest(snapshot),
                   "backend": pipeline.config.backend, "software_test_only": pipeline.config.backend == "scripted",
                   "answer_hash": submission["answer_hash"], "submitted_at": submission["submitted_at"],
                   "run_dir": str(run_dir), "budget": ledger.snapshot(), "shared_rstar": shared_record}
        if shared_record:
            outcome["logical_deployment_budget"] = {
                key: outcome["budget"][key] + shared_record["preparation_budget"][key]
                for key in ("model_calls", "tokens", "tool_calls")}
        write_json(run_dir / "complete.json", outcome)
        return outcome
    except BaseException as exc:
        failure = {"task_id": task_id, "mode": "evaluate", "method": method, "run_dir": str(run_dir),
                   "error_type": type(exc).__name__, "error": str(exc), "budget": ledger.snapshot(),
                   "shared_rstar": shared_record}
        write_json(run_dir / "failure.json", failure)
        exc.jit_mas_run_failure = failure
        raise
    finally:
        write_json(run_dir / "budget.json", ledger.snapshot())
