"""Task-conditioned JIT generation, five-block parsing, selection and repair."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, TemplateSyntaxError, meta

from jit.harness_ops import SECTION_TAG_TO_FILE, WORKSPACE_DIR, _parse_harness_response
from jit.meta_agent import MetaReActAgent
from jit.schemas import MetaAgentRequest
from jit.selector import _judge_case, _pick
from scripts.models.base import ChatMessage

from .execution import (TeamAction, TeamMemory, TeamPlanning, TeamServices,
                        TeamToolPolicy, _data, content_hash, run_team, validate_team)
from .schemas import PublicTask, RubricGraph, TeamSpec

SEED_DIR = Path(__file__).resolve().parents[1] / "harness_factory" / "harnesses" / "rubric_mas"

_MODULE_CONTRACTS = {
    "memory.py": ("MemoryStrategy", TeamMemory, {
        "initialize": (2, {}), "build_context": (1, {}), "update": (1, {}),
        "update_plan": (1, {}), "update_summary": (1, {}), "get_all_steps": (0, {}),
    }),
    "planning.py": ("PlanningStrategy", TeamPlanning, {
        "bind_team": (1, {}), "init_plan": (4, {}), "should_replan": (2, {}),
        "update_plan": (4, {}), "get_directive": (0, {}),
    }),
    "action.py": ("ActionStrategy", TeamAction, {"bind_team": (2, {}), "run": (2, {})}),
    "tool_policy.py": ("ToolPolicyStrategy", TeamToolPolicy, {
        "initialize": (1, {"enable_skills": False}), "select_tools": (4, {}),
    }),
}


def _mas_contract(execution_mode="single_pass") -> str:
    """Render the installed API, so the model need not invent an MAS framework."""
    lines = [
        "JIT-MAS BINDING CONTRACT FOR THIS REQUEST",
        "This extension specializes the generic JIT examples above. Keep the original five tagged "
        "blocks and native loader, but use the following installed public API for MAS execution.",
        "Exact module export names are mandatory; do not rename them TeamMemoryStrategy, "
        "TeamPlanningStrategy, TeamActionStrategy, or TeamToolPolicyStrategy.",
    ]
    for filename, (export, base, methods) in _MODULE_CONTRACTS.items():
        lines.extend([
            f"{filename}: from jit_mas.execution import {base.__name__}",
            f"class {export}({base.__name__}):",
            '    """Task-specific specialization; inherited methods are already executable."""',
            f"Installed {base.__name__} constructor: {inspect.signature(base)}",
        ])
        lines.extend(f"  {name}{inspect.signature(getattr(base, name))}" for name in methods)
    lines.extend([
        "These are API examples, not a required fixed team or a complete required output. "
        "Design task-appropriate module policies and agent_prompt text. Retain superclass invariants "
        "when overriding methods. Only call superclass methods actually listed in this installed API; "
        "generic JIT examples may use different methods. The supplied TeamSpec determines roles, "
        "allocation and dependencies.",
        "When overriding __init__, forward only parameters accepted by the installed base constructor "
        "shown above. In particular TeamPlanning accepts prompts, not summary_interval; any additional "
        "task-specific setting belongs on the generated subclass after super().__init__(prompts=prompts).",
        "Installed TeamPlanning implementation (already available, do not redefine this base):",
        inspect.getsource(TeamPlanning),
        "The generic JIT planning lifecycle above does not apply to this MAS binding. TeamSpec is "
        "already planned and frozen before task execution. Planning.init_plan and Planning.update_plan "
        "receive a blocked coordinator guard in their model parameter, NOT an LLM callable. Never "
        "invoke that parameter, a renamed parameter, or an alias of it. These methods may only "
        "deterministically read/serialize the bound TeamSpec and produce PlanState/SummaryState. "
        "Inherit TeamPlanning or delegate to super().init_plan(task, memory_view, tool_schemas, model); "
        "passing the guard through to this inherited implementation is valid. Keep should_replan "
        "false. Do not generate a new plan, run a model-based summary, or request another task pass.",
        "Installed TeamAction implementation (already available, do not redefine this base):",
        inspect.getsource(TeamAction),
        f"Installed helper: run_team{inspect.signature(run_team)} -> RunResult",
        "At runtime Action.bind_team(team, services) receives team as a plain validated TeamSpec "
        "dictionary and services as a TeamServices dataclass OBJECT. Use attribute access, e.g. "
        "services.model_factory(agent_id), services.public_task, services.rubrics, services.experiences. "
        "services is not a dict; services['model_factory'] and services.get(...) are invalid.",
        "services.public_task and services.rubrics are typed models (model_dump() is available). "
        "The runtime populates TeamServices; do not instantiate another services object, another "
        "API client or an unrelated execution framework.",
        f"TeamServices.event{inspect.signature(TeamServices.event)}",
        f"TeamServices.reserve_call{inspect.signature(TeamServices.reserve_call)}",
        "The recommended Action.run is return super().run(task, ctx). It already owns the complete "
        "single-pass execution, gives each role a separate Memory and metered model, enforces tool allowlists, "
        "schedules dependencies/concurrency, records full I/O and calls each role model exactly once. "
        "Custom actions must preserve all these contracts. Do not switch a shared memory's current_role.",
        "Use a task-conditioned forward-only collaboration topology: independent Analyst/Evidence "
        "contributors normally run in parallel, publish into a shared ledger, and the Writer "
        "(TeamSpec.synthesizer_id) reads the frozen ledger and writes the full deliverable once. "
        "Genuine forward data dependencies are allowed; do not hard-code agent IDs, a fixed domain "
        "team, or a particular answer. There is no draft-review-rewrite loop, clarification, "
        "negotiation, peer messaging, or second role-model call, even when max_calls is larger. "
        "The independent evaluator runs only after submission, outside this team.",
        "ctx.model is a coordinator guard in this MAS runner, not an execution model. Calling "
        "ctx.model(...) fails. Use inherited execution or properly budgeted role models obtained "
        "from services.model_factory. ctx.get_tool_schemas(tools=None) returns a JSON STRING; "
        "the Tool objects are in ctx.tool_policy.select_tools(...).tools, not that string.",
        "RunResult has a real sub_runs: list[RunResult] field. Never put sub_runs only inside "
        "metadata or return plain dictionaries instead of role RunResult values. Role metadata "
        "must include agent_id and each StepRecord preserves model_input_messages and "
        "model_output_messages. A successful outer result uses terminated_reason='final_answer'.",
        "prompt.yaml must define nonempty system_prompt and agent_prompt, plus planning, summary, "
        "final_answer and step mappings. system_prompt may use only Jinja tools/skills_prompt. "
        "agent_prompt is plain text, appended to each role's independent system message; task, role, "
        "shared_ledger and assigned rubric details arrive separately as structured JSON. "
        "Use shared_ledger, never the removed upstream_artifacts field.",
        "The rubric_mas role completion contract below replaces the generic think/tools ReAct envelope. "
        "Generated agent_prompt must not require think/tools-only responses, at least one tool call "
        "per turn, or final_answer from every role. It must permit direct answer/checkpoints JSON "
        "without any tool call. Do not copy conflicting generic output instructions into any template.",
        'The Writer completion JSON is {"answer":"...","evidence_ids":[],"checkpoints":{}}. '
        'A non-Writer contributor completes with {"answer":"brief contribution summary",'
        '"ledger":{"requirements":[],"outline":[],"evidence_spans":[],"source_references":[]},'
        '"evidence_ids":[],"checkpoints":{}}. requirements and outline are lists of strings; '
        'evidence_spans entries are {"text":"...","source_ref":"source-id"}; '
        'source_references entries are {"source_id":"source-id","locator":"..."}. '
        "All four ledger fields are required for a contributor completion and may be empty. "
        "Every span source_ref must match that contribution's source_references source_id. "
        "Do not invent provenance: state missing sources as a limitation and leave unsupported "
        "evidence fields empty. Contributors supply analysis and evidence, not competing full drafts. "
        "For each assigned checkpoint, use its EXACT name as a key with boolean true only after "
        "checking it, or use {status: 'completed'|'passed'|'failed'|'unverified'|'not_applicable', "
        "reason: '...', evidence_ids: []}. Every structured report requires a nonempty reason and "
        "an evidence_ids list containing only exact event IDs observed by that role; an empty list "
        "must not imply external verification. completed means the check was performed, a neutral "
        "self-report, not passed or independently verified. Report failed, unverified or "
        "not_applicable honestly when appropriate. These are self-reports, never independent "
        "verification. Missing checks or unexplained false values prevent completion. Do not require "
        "unconditional true values in generated prompts. Optional reasoning must "
        "not replace answer/checkpoints. Non-synthesizer roles finish locally using top-level answer "
        "or complete with the same answer/evidence_ids/checkpoints arguments, never final_answer. "
        "The synthesizer may also return top-level answer; only it may use final_answer to submit "
        "the team's final deliverable. complete/final_answer are terminal operations, not requests "
        "to bypass checkpoint or evidence validation.",
        'A contributor may emit one independent external tool batch as '
        '{"tools":[{"name":"...","arguments":{}}]}. Only a tool producer may omit answer. '
        "The runtime publishes the raw tool outputs into shared_ledger.tool_evidence without "
        "calling the contributor again or inventing analysis. These outputs were not observed "
        "by the producing model and must not be claimed as its already-read evidence. "
        "Tools must be in the current agent's allowlist and require remaining shared tool budget. "
        "Writer tool requests other than complete/final_answer are forbidden. send_message, "
        "read_evidence and raise_issue are removed, even with positive tool budget. With zero "
        "tool budget or no permitted tools, do not instruct agents to call tools. "
        "Instead report limitations in answer/checkpoints. Preserve relevant sources and "
        "uncertainty; do not force citations onto unrelated creative tasks.",
        "shared_ledger contains requirements, outline, evidence_spans, source_references, "
        "contributions (published artifact fields without duplicated ledger bodies), and tool_evidence. "
        "Aggregated ledger entries retain agent_id attribution; tool_evidence contains full retrieved events. "
        "The Writer receives all contributor artifacts, not private role histories; contributors "
        "see only forward dependencies, not peers' private conversations. The coordinator freezes "
        "and publishes the ledger with shared_ledger_ready/shared_ledger_read events, not messages. "
        "The Writer should resolve material disagreements and retain the full requested deliverable. Generated task hints "
        "and predicted rubrics are fallible; never force an unsupported formula or factual claim "
        "into the answer. Check assumptions and inferences when relevant to the task.",
        "Do not read team.json at module scope, hard-code current agent IDs or mutate evaluator, "
        "budget, stored experience, or task-global state. Generate all FIVE modules/config now.",
    ])
    if execution_mode == "iterative_shared_ledger":
        iterative_replacements = (
            ("single-pass execution", "cooperative shared-ledger execution"),
            ("gives each role a separate Memory and metered model", "gives each role private Memory and metered models"),
            ("schedules dependencies/concurrency, records full I/O and calls each role model exactly once",
             "schedules initial dependencies/concurrency, records every turn and preserves role histories across reactivation"),
            ("reads the frozen ledger and writes the full deliverable once", "reads ledger updates and revises before writing the full deliverable"),
            ("There is no draft-review-rewrite loop, clarification, negotiation, peer messaging, or second role-model call, even when max_calls is larger.",
             "Public clarification, peer messaging, artifact revision and additional role-model calls are allowed within the cooperative scheduler and configured budgets."),
            ("without another model call or inventing analysis.", "and may be followed by another role turn within configured budgets."),
            ("Only a tool producer may omit answer.", "Only a tool producer may omit answer on a nonterminal turn."),
        )
        def _replace_iterative(line, replacements):
            for old, new in replacements:
                line = line.replace(old, new)
            return line
        lines = [
            _replace_iterative(line, iterative_replacements) for line in lines
        ]
        lines.extend([
            "EXECUTION MODE: iterative_shared_ledger.",
            "The bound TeamSpec may omit model-call, tool-call and role-call ceilings. Preserve "
            "None values and rely on enforced token and timeout budgets as terminal conditions.",
            "Contributors and the Writer may revisit their public artifacts, communicate through "
            "auditable shared-ledger events, request allowed tools, receive results, and revise. "
            "Do not impose a fixed round limit or suppress a valid continuation. Private role "
            "histories remain private; only published artifacts, tool results and peer messages "
            "enter the shared ledger.",
            "ITERATIVE OVERRIDE: Any earlier single-pass, one-call, one-publication, frozen-ledger, "
            "no-negotiation or no-peer-message wording in this contract is historical and does not apply "
            "to this request. Use cooperative_shared_ledger scheduling: retain each role's private history, "
            "reactivate completed roles for public peer requests and revised artifacts, deliver tool results "
            "and ledger updates on subsequent turns, and let the synthesizer continue until a terminal answer "
            "is ready. Do not add a fixed round limit. Respect configured optional call ceilings when finite; "
            "when null, token and timeout budgets are the only terminal resource ceilings."
        ])
    return "\n\n".join(lines)


class _ContractModel:
    """Specialize generation/repair inputs without altering any model output."""

    def __init__(self, model, execution_mode="single_pass"):
        self.model = model
        self.execution_mode = execution_mode
        self.last_messages = None

    def __call__(self, messages, *args, **kwargs):
        messages = copy.deepcopy(messages)
        supplement = _mas_contract(self.execution_mode)
        ledger = getattr(self.model, "ledger", None)
        if ledger is not None:
            supplement += (
                "\nBudget-aware harness generation: preserve the bound team, reuse installed "
                "capabilities, avoid duplicate construction, and account for this generation or "
                "repair request within the shared budget. Future-stage reserves are planning "
                "estimates; runtime guards enforce the total ceiling.\nCurrent shared resource budget:\n"
                + json.dumps(ledger.resource_context(), ensure_ascii=False))
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is None:
            messages.insert(0, {"role": "system", "content": supplement})
        elif isinstance(system.get("content"), list):
            system["content"].append({"type": "text", "text": supplement})
        else:
            system["content"] = str(system.get("content", "")) + "\n\n" + supplement
        self.last_messages = copy.deepcopy(messages)
        return self.model(messages, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _record_actual_prompt(agent, event, *, normalized=False):
    messages = agent.model.last_messages
    if messages is None:
        return
    # MessageRole is a str Enum whose str() contains the enum name on some
    # Python versions; compare values instead to keep JIT's existing log keys.
    event["prompt" if normalized else "llm_prompt"] = {
        "system_prompt": next(("\n\n".join(p.get("text", "") for p in m["content"])
                                if isinstance(m["content"], list) else m["content"]
                                for m in messages if m["role"] == "system"), ""),
        "user_prompt": next(("\n\n".join(p.get("text", "") for p in m["content"])
                              if isinstance(m["content"], list) else m["content"]
                              for m in messages if m["role"] == "user"), ""),
    }
    event["model_input_messages"] = copy.deepcopy(messages)


def _symbol(node, imports):
    if isinstance(node, ast.Name):
        return imports.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _symbol(node.value, imports) + "." + node.attr
    return ""


def _method_signature(node):
    parameters = []
    positional = node.args.posonlyargs + node.args.args
    first_default = len(positional) - len(node.args.defaults)
    for index, argument in enumerate(positional):
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(node.args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = None if index >= first_default else inspect.Parameter.empty
        parameters.append(inspect.Parameter(argument.arg, kind, default=default))
    if node.args.vararg:
        parameters.append(inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        parameters.append(inspect.Parameter(argument.arg, inspect.Parameter.KEYWORD_ONLY,
            default=None if default is not None else inspect.Parameter.empty))
    if node.args.kwarg:
        parameters.append(inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _reference_path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _reference_path(node.value)
        return parent + (node.attr,) if parent else None
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        parent = _reference_path(node.value)
        if parent and isinstance(node.slice.value, (str, int)):
            return parent + (node.slice.value,)
    return None


def _coordinator_calls(method, parameter_refs=()):
    """Find common direct/straight-line alias mistakes, not prove code isolation."""
    aliases = set(parameter_refs)
    calls = []

    def guarded(value):
        path = _reference_path(value)
        return (path is not None and (path in aliases or path == ("ctx", "model"))) or (
            isinstance(value, ast.Attribute) and value.attr == "__call__" and guarded(value.value))

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            pass  # Nested scopes have their own arguments and alias bindings.

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Call(self, node):
            if guarded(node.func):
                calls.append(node)
            self.generic_visit(node)

        def visit_Assign(self, node):
            self.visit(node.value)
            source_guarded = guarded(node.value)
            for target in node.targets:
                path = _reference_path(target)
                if path is not None:
                    aliases.discard(path)
                    if source_guarded:
                        aliases.add(path)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

        def visit_If(self, node):
            self.visit(node.test)
            before = set(aliases)
            for statement in node.body:
                self.visit(statement)
            after_body = set(aliases)
            aliases.clear()
            aliases.update(before)
            for statement in node.orelse:
                self.visit(statement)
            aliases.update(after_body)

    visitor = Visitor()
    for statement in method.body:
        visitor.visit(statement)
    return calls


def _module_interface_errors(filename, tree):
    export, installed_base, requirements = _MODULE_CONTRACTS[filename]
    errors, imports, classes, aliases = [], {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports.update({alias.asname or alias.name: f"{node.module}.{alias.name}" for alias in node.names})
        elif isinstance(node, ast.Import):
            imports.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            aliases.update({target.id: node.value.id for target in node.targets if isinstance(target, ast.Name)})

    def resolve(name):
        seen = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    target = classes.get(resolve(export))
    exported_base = imports.get(resolve(export)) == f"jit_mas.execution.{installed_base.__name__}"
    if target is None and not exported_base:
        errors.append(f"{filename}: must export class {export}; found {', '.join(classes) or 'no classes'}")
        # Continue examining a misnamed class, so one repair sees all interface defects.
        target = next(iter(classes.values()), None)

    def lookup(class_node, method, seen=None):
        if class_node is None:
            return getattr(installed_base, method, None) if exported_base else None
        seen = set() if seen is None else seen
        if class_node.name in seen:
            return None
        seen.add(class_node.name)
        own = next((n for n in class_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == method), None)
        if own is not None:
            return own
        for base in class_node.bases:
            symbol = _symbol(base, imports)
            if symbol == f"jit_mas.execution.{installed_base.__name__}":
                return getattr(installed_base, method, None)
            inherited = classes.get(resolve(symbol))
            if inherited is not None:
                found = lookup(inherited, method, seen)
                if found is not None:
                    return found
        return None

    planning_guards = {}
    for method, (positional, keywords) in {"__init__": (0, {"prompts": {}}), **requirements}.items():
        implementation = lookup(target, method)
        if implementation is None:
            errors.append(f"{filename}: {export} must implement {method}{inspect.signature(getattr(installed_base, method))} "
                          f"or inherit {installed_base.__name__}")
            continue
        if isinstance(implementation, ast.AsyncFunctionDef):
            errors.append(f"{filename}: {export}.{method} must be synchronous")
            continue
        try:
            signature = (_method_signature(implementation) if isinstance(implementation, ast.FunctionDef)
                         else inspect.signature(implementation))
            signature.bind(None, *([None] * positional), **keywords)
            if filename == "planning.py" and method in {"init_plan", "update_plan"} and isinstance(
                    implementation, ast.FunctionDef):
                # Both installed methods receive the coordinator as their fourth
                # positional runtime argument, regardless of the generated name.
                marker = object()
                bound = signature.bind(None, None, None, None, marker)
                refs = []
                for name, value in bound.arguments.items():
                    if value is marker:
                        refs.append((name,))
                    elif isinstance(value, tuple):
                        refs.extend((name, index) for index, item in enumerate(value) if item is marker)
                planning_guards[id(implementation)] = refs
        except (TypeError, ValueError) as exc:
            errors.append(f"{filename}: incompatible {export}.{method}; expected calls compatible with "
                          f"{method}{inspect.signature(getattr(installed_base, method))}: {exc}")

    for class_node in classes.values():
        if len(class_node.bases) != 1 or _symbol(class_node.bases[0], imports) != (
                f"jit_mas.execution.{installed_base.__name__}"):
            continue
        constructor = next((node for node in class_node.body
                            if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
        if constructor is None:
            continue
        for node in ast.walk(constructor):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__init__" and isinstance(node.func.value, ast.Call)):
                continue
            receiver = node.func.value
            if not (isinstance(receiver.func, ast.Name) and receiver.func.id == "super"
                    and not receiver.args and not receiver.keywords):
                continue
            signature = inspect.signature(installed_base.__init__)
            keywords = {keyword.arg: None for keyword in node.keywords if keyword.arg is not None}
            expanded = any(isinstance(argument, ast.Starred) for argument in node.args)
            expanded = expanded or any(keyword.arg is None for keyword in node.keywords)
            try:
                # Unknown unpacked values cannot be checked statically, but explicit
                # unsupported keywords are still definite constructor mismatches.
                if expanded:
                    signature.bind_partial(None, **keywords)
                else:
                    signature.bind(None, *([None] * len(node.args)), **keywords)
            except TypeError as exc:
                errors.append(f"{filename}:{node.lineno}: incompatible {class_node.name} super().__init__; "
                              f"installed {installed_base.__name__}.__init__{signature}: {exc}")

    guarded_call_ids = set()
    for method in ast.walk(tree):
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in _coordinator_calls(method, planning_guards.get(id(method), ())):
                guarded_call_ids.add(id(call))
                errors.append(f"{filename}:{call.lineno}: ctx.model is a coordinator guard, as is the "
                              "model argument of Planning.init_plan/update_plan; do not invoke it or "
                              "its aliases. Inherit TeamPlanning or deterministically read the frozen "
                              "TeamSpec; superclass argument forwarding is valid")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (_reference_path(node.func) == ("ctx", "model") and id(node) not in guarded_call_ids):
                errors.append(f"{filename}:{node.lineno}: ctx.model is a coordinator guard; use inherited TeamAction "
                              "or the bound services.model_factory(agent_id) with budget guards")
            receiver = node.func.value
            service_receiver = (isinstance(receiver, ast.Name) and receiver.id == "services" or
                                isinstance(receiver, ast.Attribute) and receiver.attr == "services")
            if service_receiver and node.func.attr in {"get", "items", "keys", "values", "setdefault"}:
                errors.append(f"{filename}:{node.lineno}: TeamServices is a dataclass object; use attribute access")
        if isinstance(node, ast.Subscript):
            value = node.value
            if (isinstance(value, ast.Name) and value.id == "services" or
                    isinstance(value, ast.Attribute) and value.attr == "services"):
                errors.append(f"{filename}:{node.lineno}: TeamServices is not subscriptable; use services.field")
    if filename == "action.py" and target is not None:
        run_method = next((n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
        if run_method is not None:
            for node in ast.walk(run_method):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                    if _symbol(node.value.func, imports).rsplit(".", 1)[-1] != "RunResult":
                        continue
                    keywords = {kw.arg: kw.value for kw in node.value.keywords}
                    if "sub_runs" not in keywords and None not in keywords:
                        errors.append("action.py: outer RunResult must populate sub_runs=list[RunResult], "
                                      "not metadata['sub_runs']; inherited TeamAction.run already does this")
    return errors


def seed_response() -> str:
    """Trusted fixture response, still sent through JIT's real generation parser."""
    return "\n\n".join(f"<<<{tag}>>>\n{(SEED_DIR / name).read_text(encoding='utf-8')}"
                         f"\n<<<END_{tag}>>>" for tag, name in SECTION_TAG_TO_FILE.items())


class ScriptedHarnessModel:
    """Offline software fixture, never represented as a trained JIT checkpoint."""

    model_id = "scripted-harness-software-test"
    kwargs = {"max_tokens": 64000}

    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatMessage(role="assistant", content=seed_response())

    def get_token_counts(self):
        return {"input_token_count": 0, "output_token_count": 0}


@dataclass
class SynthesizedHarness:
    name: str
    path: Path
    backend: str
    team_hash: str
    task_hash: str
    code_hash: str
    sidecar_hash: str
    meta_trajectory: list = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    repair_count: int = 0

    @property
    def sidecar(self):
        return json.loads((self.path / "team.json").read_text(encoding="utf-8"))

    def verify_integrity(self):
        files = {name: (self.path / name).read_text(encoding="utf-8")
                 for name in SECTION_TAG_TO_FILE.values()}
        if content_hash(files) != self.code_hash:
            raise ValueError("generated harness changed after validation")
        if content_hash(self.sidecar) != self.sidecar_hash:
            raise ValueError("TeamSpec sidecar changed after generation")
        if self.backend == "scripted":
            expected = _parse_harness_response(seed_response())
            if files != expected:
                raise ValueError("scripted backend only executes the trusted checked-in seed")

    def to_dict(self):
        return {"name": self.name, "path": str(self.path), "backend": self.backend,
                "team_hash": self.team_hash, "task_hash": self.task_hash,
                "code_hash": self.code_hash, "sidecar_hash": self.sidecar_hash,
                "meta_trajectory": self.meta_trajectory, "selection": self.selection,
                "repair_count": self.repair_count}


class _PublicGenerationAdapter:
    def __init__(self, description, tools):
        self.description = description
        self.tools = tools

    def format_task(self, item):
        return self.description

    def get_tools(self):
        return []

    def get_task_tools(self, item):
        return self.tools


class JITHarnessSynthesizer:
    def __init__(self, backend="scripted", meta_model=None, meta_config=None, *,
                 candidates=1, max_repairs=2, selector_model=None, tools=None):
        if backend not in {"scripted", "native_jit"}:
            raise ValueError("backend must be scripted or native_jit")
        self.backend = backend
        self.meta_config = dict(meta_config or {})
        if backend == "native_jit":
            if not self.meta_config.get("api_base"):
                raise ValueError("native_jit requires an explicit meta endpoint (api_base)")
            if not self.meta_config.get("model_id"):
                raise ValueError("native_jit requires an explicit meta model_id")
        if not 1 <= candidates <= 8 or not 0 <= max_repairs <= 8:
            raise ValueError("candidates or max_repairs outside bounded limits")
        self.meta_model = meta_model
        if backend == "scripted" and meta_model is None:
            self.meta_model = ScriptedHarnessModel()
        self.candidates = candidates
        self.max_repairs = max_repairs
        self.selector_model = selector_model
        self.tools = tools or {}
        self._agents = {}
        self._generation_trajectories = {}

    def _attach_failure(self, exception, *, artifact=None, failed_attempts=(), names=None,
                        first_model_call=0):
        candidates = []
        for name in (self._agents if names is None else names):
            agent = self._agents[name]
            files = {}
            for filename in SECTION_TAG_TO_FILE.values():
                path = agent.workspace_dir / filename
                if path.is_file():
                    files[filename] = content_hash(path.read_text(encoding="utf-8"))
            candidates.append({"name": name, "path": str(agent.workspace_dir),
                               "file_hashes": files,
                               "meta_trajectory": copy.deepcopy(self._generation_trajectories.get(name, []))})
        calls = getattr(self.meta_model, "calls", [])
        # Store only public generation inputs and observed outputs, never client
        # configuration, credentials, or arbitrary exception object internals.
        exception.jit_mas_failure = {
            "artifact": artifact.to_dict() if artifact is not None else None,
            "failed_execution_attempts": copy.deepcopy(list(failed_attempts)),
            "meta_trajectory": copy.deepcopy(artifact.meta_trajectory) if artifact is not None else [],
            "observed_meta_calls": copy.deepcopy(calls[first_model_call:]) if isinstance(calls, list) else [],
            "generated_candidates": candidates,
        }

    def _agent(self, name):
        config = dict(self.meta_config)
        if self.meta_model is not None:
            # Constructing JIT's normal client does not issue requests. Replace it
            # with the injected metered/scripted transport before generation.
            config.setdefault("model_id", getattr(self.meta_model, "model_id", "injected-meta-model"))
            config.setdefault("api_base", "http://127.0.0.1:1/v1")
            config.setdefault("api_key", "EMPTY")
        agent = MetaReActAgent(config, {"meta_references": {"mode": "desc"},
                                       "meta_review": {"enabled": False}}, workspace_name=name)
        agent.model = _ContractModel(self.meta_model if self.meta_model is not None else agent.model,
                                     getattr(self, "_execution_mode", "single_pass"))
        self._agents[name] = agent
        return agent

    @staticmethod
    def _description(sidecar):
        reference = (SEED_DIR.parent.parent / "descriptions" / "rubric_mas.md").read_text(encoding="utf-8")
        mode_note = ""
        if sidecar.get("team", {}).get("execution_mode") == "iterative_shared_ledger":
            mode_note = (" The execution mode is iterative_shared_ledger: preserve unlimited call "
                         "ceilings represented by null, allow public ledger revisits, peer messages, "
                         "tool-result continuation and Writer revisions; token and timeout budgets "
                         "remain terminal and no fixed round cap may be introduced.")
        description = (
            "Generate a task-conditioned JIT MAS harness using the exact JIT-MAS binding contract "
            "appended to the system message. These are PUBLIC inputs only:\n"
            + json.dumps(sidecar, ensure_ascii=False, indent=2)
            + "\n\nMAS extension contract (preserve the original five tagged blocks):\n"
            + reference
            + "\nThe runtime binds Action.bind_team(team, services) after native loading. "
            "Reuse jit_mas.execution.TeamAction, TeamMemory, TeamPlanning, TeamToolPolicy "
            "as protocol-compatible bases, or implement the same bind_team and independent "
            "services.model_factory interface. TeamSpec must determine actual roles, tools, "
            "dependencies, checkpoints and budgets. Do not hard-code another team. "
            "Honor the team's budget_plan as a quality-cost planning estimate: prefer concise "
            "task-relevant context, reuse retained capabilities, avoid duplicate work, and preserve "
            "tokens for final synthesis and required downstream stages. Expected calls are estimates, "
            "not new execution ceilings. Runtime resource_budget comes from the shared ledger; "
            "never grant a fresh task budget to each role or bypass its guards. "
            "Do not load team.json at module scope or write any process-wide task state. "
            "Use services.public_task/rubrics/experiences as object attributes at run time. "
            "Export MemoryStrategy, PlanningStrategy, ActionStrategy, ToolPolicyStrategy exactly. "
            "Subclass the installed Team* implementations to reuse actual guards and full traces. "
            "Each action must return "
            "RunResult with each role's full observed I/O in sub_runs and immutable event metadata. "
            "Generate single-pass contributor ledger publication followed by one Writer model call; "
            "no task-internal negotiation or clarification. Only code/interface failures before any "
            "role-model invocation may be repaired; no evaluator feedback is available. "
            "Keep system_prompt, agent_prompt, planning, summary, final_answer, step in prompt.yaml."
            + mode_note
            + ("\nITERATIVE OVERRIDE: The mode note supersedes historical single-pass generation guidance. "
               "Generate cooperative_shared_ledger execution with private per-role histories, public peer "
               "requests, role reactivation, ledger/tool updates and Writer revisions. Do not generate a "
               "fixed round cap or suppress continuation; finite configured ceilings, token budgets and "
               "timeouts remain binding."
               if sidecar.get("team", {}).get("execution_mode") == "iterative_shared_ledger" else "")
        )
        if sidecar.get("team", {}).get("execution_mode") == "iterative_shared_ledger":
            description = description.replace(
                "Generate single-pass contributor ledger publication followed by one Writer model call; "
                "no task-internal negotiation or clarification.",
                "Generate cooperative shared-ledger publication with revisable role turns and public peer clarification; "
                "the Writer may continue and revise before terminal submission.")
            for old, new in (
                ("single-pass", "historical one-way"),
                ("one Writer model call", "Writer continuation and revision"),
                ("no task-internal negotiation or clarification", "public task-internal peer clarification is allowed"),
                ("no task-internal negotiation", "public task-internal peer clarification is allowed"),
                ("no draft-review-rewrite loop", "cooperative review and revision"),
                ("without another model call", "with additional role turns allowed"),
                ("second role-model call", "additional role-model turns"),
            ):
                description = description.replace(old, new)
        return description

    def _validate(self, agent):
        errors = []
        error = agent._static_harness_checks()
        if not error.startswith("No static errors detected"):
            errors.append(error)
        files = {}
        for filename in SECTION_TAG_TO_FILE.values():
            path = agent.workspace_dir / filename
            if path.is_file():
                files[filename] = path.read_text(encoding="utf-8")
        for filename in _MODULE_CONTRACTS:
            if filename not in files:
                continue
            try:
                tree = ast.parse(files[filename], filename)
            except SyntaxError:
                continue  # Already included in the original JIT static report.
            errors.extend(_module_interface_errors(filename, tree))
        try:
            prompts = yaml.safe_load(files.get("prompt.yaml", ""))
        except yaml.YAMLError:
            prompts = None
        if isinstance(prompts, dict):
            for key in ("system_prompt", "agent_prompt"):
                if not isinstance(prompts.get(key), str) or not prompts[key].strip():
                    errors.append(f"prompt.yaml: {key} must be a nonempty string")
            if isinstance(prompts.get("system_prompt"), str):
                try:
                    variables = meta.find_undeclared_variables(Environment().parse(prompts["system_prompt"]))
                    unknown = variables - {"tools", "skills_prompt"}
                    if unknown:
                        errors.append(f"prompt.yaml: system_prompt uses unavailable template variables {sorted(unknown)}")
                except TemplateSyntaxError as exc:
                    errors.append(f"prompt.yaml: invalid system_prompt template: {exc}")
        if errors:
            raise ValueError("JIT-MAS interface validation failed; fix all listed defects together:\n- "
                             + "\n- ".join(dict.fromkeys(errors)))
        if self.backend == "scripted" and files != _parse_harness_response(seed_response()):
            raise ValueError("scripted backend emitted untrusted code; use native_jit with isolation")
        return files

    def synthesize(self, task, rubrics, team, experiences=(), *, agent_pool=None):
        prior_names = set(self._agents)
        calls = getattr(self.meta_model, "calls", [])
        first_call = len(calls) if isinstance(calls, list) else 0
        try:
            return self._synthesize(task, rubrics, team, experiences, agent_pool=agent_pool)
        except BaseException as exc:
            self._attach_failure(exc, names=[name for name in self._agents if name not in prior_names],
                                 first_model_call=first_call)
            raise

    def _synthesize(self, task, rubrics, team, experiences=(), *, agent_pool=None):
        task = PublicTask.model_validate(_data(task))
        rubrics = RubricGraph.model_validate(_data(rubrics))
        team = TeamSpec.model_validate(_data(team))
        self._execution_mode = team.execution_mode
        team_data = validate_team(team, task)
        sidecar = {"schema_version": "1.0", "task": _data(task), "rubrics": _data(rubrics),
                   "team": team_data, "experiences": [_data(e) for e in experiences],
                   "backend": self.backend}
        ledger = getattr(self.meta_model, "ledger", None)
        if ledger is not None:
            sidecar["resource_budget_at_generation"] = ledger.resource_context()
        if agent_pool is not None:
            from .agent_pool import validate_bindings
            from .schemas import AgentPoolSnapshot

            pool = AgentPoolSnapshot.model_validate(_data(agent_pool))
            validate_bindings(pool, team)
            sidecar["agent_pool"] = pool.model_dump(mode="json")
            return self._reuse_pool_harness(sidecar)
        if any(agent.pool_agent_id is not None for agent in team.agents):
            raise ValueError("Bound Agent Pool members require a frozen pool sidecar")
        description = self._description(sidecar)
        public_tools = {name: tool for name, tool in self.tools.items() if name in _data(task)["tools"]}
        adapter = _PublicGenerationAdapter(description, public_tools)
        artifacts, rows, failures = [], [], []
        for rollout in range(self.candidates):
            name = "mas_" + uuid.uuid4().hex
            agent = self._agent(name)
            result = agent.run(MetaAgentRequest(benchmark_adapter=adapter,
                item={"id": _data(task)["task_id"]}, tools=[], generate_only=True,
                max_repairs=0, repair_only_on_error=True))
            if result.meta_agent_trajectory:
                _record_actual_prompt(agent, result.meta_agent_trajectory[-1], normalized=True)
            trajectory = list(result.meta_agent_trajectory)
            self._generation_trajectories[name] = trajectory
            repairs = 0
            while True:
                try:
                    files = self._validate(agent)
                    break
                except (ValueError, SyntaxError, FileNotFoundError, yaml.YAMLError) as exc:
                    if repairs >= self.max_repairs:
                        failures.append(f"candidate {rollout}: {exc}")
                        files = None
                        break
                    event = agent._repair_harness(str(exc), [{"error": str(exc)}],
                                                  evaluation_result={}, validation_error=str(exc))
                    _record_actual_prompt(agent, event)
                    event["stage"] = "protocol_repair"
                    trajectory.append(event)
                    repairs += 1
            if files is None:
                continue
            sidecar_path = agent.workspace_dir / "team.json"
            sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            artifact = SynthesizedHarness(name, agent.workspace_dir, self.backend,
                content_hash(team_data), content_hash(_data(task)), content_hash(files),
                content_hash(sidecar), trajectory, repair_count=repairs)
            artifacts.append(artifact)
            completion = "\n\n".join(f"<<<{tag}>>>\n{files[filename]}<<<END_{tag}>>>"
                                       for tag, filename in SECTION_TAG_TO_FILE.items())
            rows.append({"case": _data(task)["task_id"], "rollout": len(artifacts) - 1,
                         "n_sub": len(files), "protocol_score": 1.0, "user": description,
                         "completion": completion})
        if not artifacts:
            raise RuntimeError("JIT generation failed: " + "; ".join(failures))
        if len(rows) == 1:
            selection = _pick(rows, "protocol_score", "single_protocol_valid_candidate")
        else:
            model = self.selector_model or self.meta_model
            if model is None:
                raise ValueError("multiple candidates require an explicitly metered selector_model")
            verdict = _judge_case(_data(task)["task_id"], rows, model, 24000)
            if verdict["choice"] is None:
                raise ValueError("selector did not return a valid choice")
            for i, row in enumerate(rows):
                row["judge_score"] = float(i == verdict["choice"])
            selection = _pick(rows, "judge_score", "judge_pick")
            selection["judge_verdict"] = verdict
        selected = artifacts[selection["selected"][_data(task)["task_id"]]]
        selected.selection = selection
        selected.verify_integrity()
        return selected

    def _reuse_pool_harness(self, sidecar):
        name = "mas_" + uuid.uuid4().hex
        agent = self._agent(name)
        agent.workspace_dir.mkdir(parents=True, exist_ok=False)
        (agent.workspace_dir / "__init__.py").write_text("", encoding="utf-8")
        files = _parse_harness_response(seed_response())
        for filename, content in files.items():
            (agent.workspace_dir / filename).write_text(content, encoding="utf-8")
        files = self._validate(agent)
        (agent.workspace_dir / "team.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact = SynthesizedHarness(name, agent.workspace_dir, self.backend,
            content_hash(sidecar["team"]), content_hash(sidecar["task"]), content_hash(files),
            content_hash(sidecar), selection={"strategy": "pooled_agent_reuse",
                "agent_pool_hash": content_hash(sidecar["agent_pool"]),
                "temporary_agent_ids": [item["agent_id"] for item in sidecar["team"]["agents"]
                                        if item.get("temporary_profile") is not None]})
        artifact.verify_integrity()
        return artifact

    def repair(self, artifact, failure, failed_run=None):
        """Bounded exception-only repair. Task quality or evaluator scores are forbidden."""
        if not isinstance(failure, Exception):
            raise TypeError("repair requires an execution/interface exception, never a score")
        if _role_execution_started(failed_run, failure):
            raise RuntimeError("Single-pass execution already started; whole-team repair is forbidden") from failure
        if artifact.repair_count >= self.max_repairs:
            raise RuntimeError("JIT exception repair budget exhausted")
        artifact.verify_integrity()
        agent = self._agents.get(artifact.name) or self._agent(artifact.name)
        agent._last_task_description = self._description(artifact.sidecar)
        failed_cases = [{"error": str(failure)}]
        if failed_run is not None:
            failed_cases[0]["trajectory"] = [s.full_dict() for r in failed_run.sub_runs for s in r.trajectory]
        event = agent._repair_harness(str(failure), failed_cases, evaluation_result={},
                                     validation_error=str(failure))
        _record_actual_prompt(agent, event)
        event["stage"] = "execution_exception_repair"
        artifact.meta_trajectory.append(event)
        artifact.repair_count += 1
        artifact.code_hash = content_hash(self._validate(agent))
        artifact.verify_integrity()
        return artifact

    def execute_with_repair(self, executor, task, team, artifact, rubrics=None, experiences=()):
        failed_attempts = []
        while True:
            result = None
            try:
                result = executor.execute(task, team, artifact, rubrics=rubrics, experiences=experiences)
                errors = [str(step.error) for run in result.sub_runs for step in run.trajectory if step.error]
                errors += [run.metadata["error"] for run in result.sub_runs if run.metadata.get("error")]
                if errors and result.answer is None:
                    raise RuntimeError("; ".join(errors))
                result.metadata["failed_execution_attempts"] = failed_attempts
                result.metadata["repair_count"] = artifact.repair_count
                return result
            except BaseException as exc:
                failed_attempts.append({"error": str(exc), "run": result.full_dict() if result else None,
                                        "harness_hash": artifact.code_hash,
                                        "role_execution_started": _role_execution_started(result, exc)})
                # Safety gates and exhaustion are not repairable harness defects.
                if (_role_execution_started(result, exc) or not isinstance(exc, Exception)
                        or isinstance(exc, (PermissionError, TimeoutError)) or
                        any(term in str(exc).lower() for term in ("unsafe-local", "sandbox", "budget", "timeout",
                                                                "exhausted", "sidecar", "changed after"))):
                    self._attach_failure(exc, artifact=artifact, failed_attempts=failed_attempts)
                    raise
                try:
                    self.repair(artifact, exc, result)
                except BaseException as repair_error:
                    self._attach_failure(repair_error, artifact=artifact, failed_attempts=failed_attempts)
                    raise


def _role_execution_started(result=None, failure=None):
    """Conservatively prohibit replay after any role-model reservation or trace."""
    if getattr(failure, "jit_mas_execution_started", False):
        return True
    if result is None:
        return False
    return bool(result.metadata.get("model_calls_used", 0) or
                any(run.trajectory for run in result.sub_runs))
