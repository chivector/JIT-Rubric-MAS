"""Single-pass and cooperative shared-ledger team primitives for JIT Action modules."""

from __future__ import annotations

import copy
import hashlib
import json
import queue
import re
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
from jit_mas.planning import (
    QUALITY_ASSURANCE_PROMPT, knowledge_policy_prompt, knowledge_policy_role_adaptation,
)
from jit_mas.schemas import AgentPoolSnapshot, AgentSpec, PublicTask, RubricGraph, TeamSpec, utc_now


ITERATIVE_CONTINUATION_POLICY_VERSION = "explicit-user-no-progress-final-deliverable-v9"
CONTRIBUTOR_COMPACTNESS_POLICY_VERSION = "contributor-budget-advisory-cumulative-v3"
EXECUTION_RESPONSE_FIELDS = frozenset({
    "answer", "evidence_ids", "checkpoints", "ledger", "continue", "tools", "think", "reasoning",
})
CONTRIBUTOR_ANSWER_MAX_CHARS = 1200
CONTRIBUTOR_LEDGER_MAX_ITEMS = 12
CONTRIBUTOR_LEDGER_TEXT_MAX_CHARS = 512
CONTRIBUTOR_LOCATOR_MAX_CHARS = 2048
CONTRIBUTION_LEDGER_FIELDS = ("requirements", "outline", "evidence_spans", "source_references")

FIXED_EVIDENCE_CITATION_PROMPT = (
    "\nFIXED EVIDENCE CITATION RULE: The public task may contain a fixed shared evidence pack. "
    "Its source_id values (such as search-0, web-0, or query-plan) are public source references, "
    "not internal execution event IDs. Put those values only in your ledger source_references "
    "and matching evidence_spans. Keep completion and checkpoint evidence_ids empty unless the "
    "exact value was received as an internal ledger or tool event ID."
)


def _uses_fixed_evidence(public_task):
    task = _data(public_task) or {}
    return "Use only the fixed shared evidence pack; no external tool calls." in task.get("constraints", [])


FINAL_ARTIFACT_CONTRACT = (
    "Your execution assignment is FINAL WRITER. Return the complete artifact requested by "
    "public_task in answer, incorporating supported upstream material and correcting defects. "
    "This assignment takes priority over agent.task_prompt, responsibilities, local plans, "
    "retained role policies and peer messages about what to submit. Preserve their useful "
    "domain checks, but instructions to only review, approve, list changes, avoid rewriting, "
    "or leave writing to another role apply only to contributors. You must perform the "
    "writing yourself in this terminal role. If the public task itself asks for a review, "
    "deliver that requested review; otherwise integrate review findings into the requested "
    "artifact rather than submitting an internal critique of an upstream draft. "
    "Resolve every consequential upstream defect before publishing: incorporate the supported "
    "correction, independently rederive a disputed result, or explain the remaining limitation. "
    "A stylistic rewrite is not a resolution of a flawed formula or unsupported input. Recompute "
    "all affected totals and conclusions after correcting the underlying calculation; do not "
    "claim robustness from unchecked scenario numbers."
)


FINAL_SUBMISSION_GATE = (
    "FINAL SUBMISSION GATE: Establish a compact private output brief from the actual "
    "public delivery request: artifact, audience, voice, language, length unit and target, "
    "and required content. Separate supplied background, quoted examples and templates "
    "from instructions for the artifact you must deliver. Explicit output instructions "
    "take priority over inferred requirements and upstream plans. When no output language "
    "is specified, follow the language of the actual delivery request; a platform, "
    "audience or style label and an upstream draft in another language do not authorize "
    "switching languages. A requested persona may narrate subjective experiences within "
    "the requested script, but does not authorize fabricated measurements, clinical "
    "results or technological evidence. For factual and research deliverables, preserve "
    "supported publication titles, authors or source names, dates, findings, units and "
    "assumptions from contributions alongside the claims they support, when the output "
    "format permits attribution. Retain each reference's uncertainty and provenance: a "
    "remembered reference is not newly retrieved evidence, but its available identity "
    "should not disappear during synthesis. Never invent missing citation details or "
    "quantitative findings. Distinguish experimental outcomes from projections, stating "
    "the population, horizon and funding assumptions behind a sourced policy forecast. "
    "Allocate the requested length across required "
    "parts before drafting and check the actual decoded artifact after edits. For an "
    "approximate length, aim near the stated target in its stated units; the token "
    "ceiling is not an answer-length target. Merge repetition and shorten incidental "
    "examples while preserving every explicit deliverable. Keep the brief and counting "
    "notes outside answer, and do not claim a deterministic check without performing it. "
    "Before returning the terminal JSON, reread the original "
    "public_task.question and check the decoded answer in this order: deliver the requested "
    "artifact itself; satisfy every explicit output-only, language, count, length and format "
    "constraint; preserve every supported requested item without unsupported additions; "
    "recheck dates, units, denominators, formulas and source locators for consequential "
    "claims; use the supplied rubric criteria as a private checklist and map each criterion "
    "to at least one concrete, answer-relevant sentence, table row or explicitly supported "
    "limitation; retain named entities, quantities, counterexamples and requested format "
    "objects instead of replacing them with broad thematic summaries; identify any criterion "
    "still uncovered and repair it when supported by the task or evidence; remove process "
    "narration, internal rubric IDs and review metadata; and leave enough space for valid "
    "JSON closure. When an upstream artifact conflicts with the public task, follow the "
    "public task and repair the conflict before submitting."
)


PUBLIC_POSITIONAL_DRAFT_GUIDANCE_VERSION = "public-positional-internal-draft-v1"
PUBLIC_POSITIONAL_DRAFT_PROJECTION_VERSION = "public-positional-task-projection-v1"
PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT = (
    "\nPUBLIC POSITIONAL CONTENT DRAFT STAGE: This execution answer is internal content, "
    "not the final public submission. The already configured fixed public review and typed "
    "word-slot revision follow this completed draft and implement the supported public "
    "sentence and word position. This changes this stage's presentation task only; every "
    "public final-output constraint still applies to the final revision. Contributors "
    "preserve compact substantive material for the whole requested plot or content. The "
    "synthesizer returns one compact coherent content draft with the complete plot, causal "
    "progression, required examples and supported facts; a title, outline or promise is "
    "not a complete draft. Do not repeat paragraphs, sentences or draft copies to simulate "
    "an exact word slot, and do not claim the final position has been verified. Preserve "
    "meaningful requested repetition where it is itself a public requirement. Retain "
    "the role's exact JSON completion shape, substantive ledger and checkpoint honesty, "
    "leave space for JSON closure, and do not add calls or wait for another agent to "
    "perform this content task. The fixed final typed revision must still satisfy the "
    "position rule and all other public content, language, length and format constraints."
)


def compile_public_positional_draft_plan(public_task, config):
    """Select the optional draft stage from public text only; no model/tool calls."""
    if not all(getattr(config, flag, False) for flag in (
            "public_positional_draft_guidance", "public_refinement",
            "public_positional_construction")):
        return None
    from .public_numeric_slots import public_construction_conflict
    from .public_word_slots import position_plan

    if public_construction_conflict(public_task):
        return None
    return position_plan(public_task)


def _independent_public_position_span(text, span):
    """Use the same conservative public scope as final word-slot construction."""
    from .public_word_slots import independent_positive_position_span

    return independent_positive_position_span(text, span)


def project_public_positional_draft_task(public_task, plan):
    """Delete only verified compiler spans in an actor's independent public-task copy."""
    original = _data(public_task)
    projected = copy.deepcopy(original)
    audit = {
        "version": PUBLIC_POSITIONAL_DRAFT_PROJECTION_VERSION,
        "active": False, "final_constraints_retained": True,
        "original_task_hash": content_hash(original),
        "projected_task_hash": content_hash(original), "span_hashes": [],
        "scope": "Actor public_task.question and public_task.constraints only",
        "reason": "No single supported conflict-free public positional rule",
    }
    if plan is None:
        return projected, audit
    try:
        from .public_numeric_slots import public_construction_conflict
        from .public_word_slots import position_plan

        if (not isinstance(original, dict) or not isinstance(original.get("question"), str)
                or not isinstance(original.get("constraints", []), list)
                or not all(isinstance(item, str) for item in original.get("constraints", []))):
            raise ValueError("Invalid public question/constraints")
        if public_construction_conflict(public_task):
            raise ValueError("Conflicting public construction rules")
        expected = position_plan(public_task)
        if expected is None:
            raise ValueError("The original public task has no supported positional rule")
        spans = tuple(plan.matched_public_spans)
        if (not spans or (plan.keyword, plan.sentence_index, plan.word_index) !=
                (expected.keyword, expected.sentence_index, expected.word_index)
                or spans != expected.matched_public_spans):
            raise ValueError("The positional plan differs from the original public compiler result")
        grouped = {}
        for span in spans:
            if not isinstance(span.source, str):
                raise ValueError("Invalid public span source")
            if span.source == "question":
                text = original["question"]
                index = None
            else:
                match = re.fullmatch(r"constraints\[(0|[1-9][0-9]*)\]", span.source)
                if match is None:
                    raise ValueError("The span source is not an explicit public task field")
                index = int(match.group(1))
                text = original.get("constraints", [])[index]
            if (type(span.start) is not int or type(span.end) is not int
                    or not 0 <= span.start < span.end <= len(text)
                    or not isinstance(span.text, str) or text[span.start:span.end] != span.text):
                raise ValueError("The public span range or original text differs")
            if not _independent_public_position_span(text, span):
                raise ValueError("The positional span is not an independent positive public instruction")
            grouped.setdefault((span.source, index), []).append(span)
        span_hashes = []
        for (source, index), field_spans in grouped.items():
            ordered = sorted(field_spans, key=lambda span: span.start)
            if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
                raise ValueError("The compiler spans overlap")
            text = original["question"] if index is None else original["constraints"][index]
            for span in reversed(ordered):
                text = text[:span.start] + text[span.end:]
                span_hashes.append({"source": source, "start": span.start, "end": span.end,
                    "text_sha256": hashlib.sha256(span.text.encode("utf-8")).hexdigest()})
            if index is None:
                projected["question"] = text
            else:
                projected["constraints"][index] = text
        if not any(character.isalnum() for character in projected["question"]):
            raise ValueError("The projected question has no remaining substantive public request")
        audit.update(active=True, projected_task_hash=content_hash(projected),
                     span_hashes=span_hashes, reason=None)
        return projected, audit
    except Exception as exc:
        audit["reason"] = f"Projection declined: {type(exc).__name__}"
        return copy.deepcopy(original), audit


def _apply_public_positional_draft_guidance(system, instruction, services):
    plan = getattr(services, "public_positional_draft_plan", None)
    projection = None
    if getattr(services, "public_positional_draft_projection_requested", False):
        projected, audit = project_public_positional_draft_task(services.public_task, plan)
        services.public_positional_draft_projection_audit = audit
        if audit["active"]:
            instruction["public_task"] = projected
            projection = audit
    if plan is None:
        return system
    instruction["public_positional_draft_guidance"] = {
        "version": PUBLIC_POSITIONAL_DRAFT_GUIDANCE_VERSION,
        "stage": "internal_content_draft", "public_position_plan": plan.audit(),
        "final_constraints_retained": True,
    }
    if projection is not None:
        instruction["public_positional_draft_guidance"].pop("public_position_plan")
        instruction["public_positional_draft_guidance"]["projection_hash"] = (
            projection["projected_task_hash"])
    return system + PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT


def _execution_assignment(agent, synthesizer):
    """Expose delivery authority next to the frozen role scope without mutating it."""
    assignment = copy.deepcopy(agent)
    assignment["execution_role"] = "final_writer" if synthesizer else "contributor"
    if synthesizer:
        assignment["task_prompt"] = (
            "Domain scope and checks from the frozen plan:\n" + agent.get("task_prompt", "")
            + "\nAuthoritative terminal assignment:\n" + FINAL_ARTIFACT_CONTRACT
        )
    return assignment


def _contributor_handoff_prompt(*, closed_book=False):
    answer_limit = 2400 if closed_book else 1200
    item_limit = 24 if closed_book else 12
    return (
        f"Aim for answer <={answer_limit} characters as a compact summary, not the complete public deliverable. "
        "Put concrete facts, reasoning steps, intermediate results and limitations in the sibling ledger. "
        "Your output ledger is your own contribution, not a copy of the read-only shared_ledger "
        "aggregate. It permits exactly requirements, outline, evidence_spans and source_references; "
        "never include shared_ledger's contributions, tool_evidence or communications keys. "
        "requirements and outline are arrays of nonempty strings. An observed source reference is "
        "{source_id: nonempty string, locator: nonempty string}; each evidence span is "
        "{text: nonempty string, source_ref: the exact source_id declared in your own "
        "source_references}. Do not replace source_ref with source_id, evidence_id or an event ID. "
        "Exactly one source_references entry per distinct stable source_id in your contribution. "
        "When one observed dossier or fixed-pack source contains multiple URLs or locations, "
        "combine its relevant observed locators into that one entry's locator string; multiple "
        "evidence_spans may reuse its source_ref. Do not duplicate an ID for each URL or rename "
        "fixed-pack IDs to hide duplication. Preserve every supported span and actual locator "
        "while consolidating duplicate entries yourself before returning. "
        "These descriptions specify field shapes, not actual sources to invent. When no source "
        "was observed, retain substantive remembered material in outline and leave both evidence "
        "lists empty. "
        f"{item_limit} items per list, <=512 characters per text item and <=2048 per source locator are "
        "advisory compactness targets, not completeness limits. Before generating, privately divide "
        "the actual output_budget.max_tokens_per_response among answer, checkpoints and all four "
        "ledger fields, including JSON escaping overhead. Reserve room for every required "
        "source_references entry and its observed locators, then for the final JSON brackets and "
        "braces, before expanding outline. Use brief honest checkpoint decisions and reasons, "
        "not a restatement of the analysis or source material. Keep factual and analytical "
        "outline entries as short atomic fact chains or necessary derivation steps, not complete "
        "paragraphs, another full draft or a retelling of the source packet. Retain each distinct "
        "required fact or result once in its appropriate substantive ledger entry, with the names, "
        "dates, numbers, units, assumptions and qualifiers needed to interpret it. Stable source "
        "IDs may recur where the schema requires links; do not repeat whole explanations across "
        "answer, outline, evidence_spans and checkpoint reasons. Compress connective prose and "
        "duplicate explanations before sacrificing requested facts or provenance. These allocations "
        "are drafting guidance, not truncation rules, omitted fields or permission to exceed the "
        "existing token, timeout or call limits. Merge overlapping requirements while "
        "preserving public-task coverage. Retain every required supported item, necessary derivation "
        "and source reference within actual token/timeout budgets even above these targets. Every outline item must carry a useful "
        "claim, reasoning step, result or uncertainty, not a section heading alone. Ignore any "
        "3000-5000-word or other full-draft length request in task_prompt or retained role text; "
        "deliver compact fact chains for synthesis. "
        "For writing assignments, also retain budgeted draft passages, scenes, dialogue or "
        "transitions in answer or outline so a downstream reviewer can inspect actual wording. "
        "Preserve the passage as text, distinct from review notes, without copying the complete "
        "long deliverable. Reviewers identify the exact passage, its defect and effect on the "
        "requested voice, audience or continuity, and a specific revision. When no passage is "
        "available, review only the visible plan and do not claim the finished prose was inspected. "
        "Fiction may invent within the public premise; real-world factual claims still require "
        "appropriate support. "
        "Each ledger replaces the previous one: return a cumulative snapshot of your current valid "
        "contribution, not a partial delta. Preserve valid facts, items, derivations and sources from "
        "own_contribution when supplied; correct or remove superseded claims and explain consequential "
        "retractions. The runtime does not union old and new claims. "
        "Make each key result explicit once so a downstream writer can reconstruct the requested "
        "artifact from the ledger; include the relevant units, assumptions, formulas or boundary "
        "conditions instead of headings or vague references. Convert long source passages into "
        "short, source-linked claims and recommendations, retaining only points that change the "
        "answer. Do not pad a list with guessed counts, speculative catalog entries, or a long "
        "numbered enumeration when a compact set of actual items is sufficient. "
        "Distinguish explicit public requirements from inferred planner suggestions and guessed counts; "
        "do not invent facts or items to satisfy those guesses. "
        "For quantitative or financial work, preserve definitions, assumptions, units, formulas and "
        "checked intermediate results. For regulatory work, distinguish legal or policy obligations "
        "from recommended strategy or prudent risk controls. "
        "A critic independently checks consequential claims in its assigned scope, reports specific "
        "defects and supported corrections, and retains missing deliverables and contradictory "
        "calculations in the handoff. Judge each contributor by its actual assignment; check complete "
        "public-task coverage across the combined handoff and final writer."
    )


def _answer_check_prompt():
    return (
        "Predicted requirements are fallible: satisfy explicit public instructions, but do not invent "
        "facts or items to meet an inferred count. For numerical or financial claims, recompute derived "
        "figures from stated formulas, denominators, units and assumptions, including scenario changes. "
        "For regulatory claims, distinguish an actual obligation from a recommended strategy; avoid "
        "absolute necessity claims without support. Uncertainty labels do not establish unsupported "
        "numbers, citations or legal assertions. Compress copied source or ledger prose into the "
        "smallest set of actual, answer-relevant claims and recommendations. Never invent a long "
        "directory, catalog, numbered sequence, or repeated examples to appear comprehensive. Use "
        "representative items only when the public task requests examples or a sample. When it requests "
        "all qualifying items or a complete list, apply the same stated eligibility conditions to every "
        "item, deduplicate aliases, retain each supported qualifying item and its distinguishing identity, "
        "and privately organize candidates by every requested condition. Apply joint conditions "
        "conjunctively, preserve inclusive or strict comparison operators and the requested as-of date, "
        "and distinguish an unknown fact from a condition known to fail. For tabular evidence, map "
        "values by the actual column names, units and denominators, not by guessed column order; "
        "inspect every relevant row before claiming a complete filtered set. "
        "Check both missing items and unsupported additions. State an actual completeness limitation "
        "without replacing a supported full set with representative items. Remove duplicated claims before "
        "adding detail. Public-task length, count and output-only constraints determine answer length; "
        "a short answer must not be padded to consume the output-token allowance."
    )


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
    knowledge_policy: str | None = None
    task_seconds_at_start: float | None = field(init=False, default=None)

    def __post_init__(self):
        knowledge_policy_prompt(self.knowledge_policy)
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


_SUBSTANTIVE_TASK_VERBS = (
    "write", "explain", "analyz", "analyse", "analyze", "report", "design",
    "compare", "conduct", "discuss", "describe", "evaluate", "assess",
    "develop", "advise", "plan", "framework", "essay", "argument",
    "recommend", "compile a list",
    "撰写", "写作", "报告", "文章", "论文", "解释", "分析", "比较", "评估", "方案", "计划", "论证",
)
_SHORT_TASK_MARKERS = (
    "yes/no", "yes or no", "single word", "one word", "one sentence",
    "single number", "just the number",
    "what is the answer", "answer with a number",
    "一个词", "单个词", "一个字", "单个字", "一句话", "一个数字", "仅数字", "是或否",
)
def _requires_substantive_answer(public_task):
    """Return whether the public prompt requests a prose/artifact deliverable.

    This deliberately uses only the public question/constraints and is a conservative guard:
    short factual/format-specific prompts remain free to have short answers.
    """
    from .evidence import public_instruction_question

    task = _data(public_task) or {}
    question = public_instruction_question(task).strip().lower()
    constraints = " ".join(str(item) for item in task.get("constraints", [])).lower()
    if (re.match(r"^(?:please\s+)?name\s+the\b", question)
            and not re.search(r"\b(?:and|then|also)\s+(?:write|explain|analy[sz]e|discuss|describe)\b", question)):
        return False
    if not question or any(marker in question or marker in constraints
                           for marker in _SHORT_TASK_MARKERS):
        return False
    # Short imperatives such as "Write a researched guide" still request a
    # prose artifact; explicit short-format markers above protect prompts that
    # intentionally ask for a name, number, or one-sentence response.
    return any(verb in question for verb in _SUBSTANTIVE_TASK_VERBS)


def _heading_output_requested(public_task):
    from .evidence import public_instruction_question

    task = _data(public_task) or {}
    question = public_instruction_question(task).strip().lower()
    constraints = " ".join(str(item) for item in task.get("constraints", [])).lower()
    request = re.search(
        r"\b(?:write|create|compose|generate|draft|provide|give|return|suggest|choose)\s+"
        r"(?:(?:me|a|an|the|one|single|short|concise|catchy|creative|newspaper|suitable|"
        r"appropriate|effective)\s+)*(?:headline|title|heading)\b", question)
    additional_prose = re.search(
        r"\b(?:and|then|also)\s+(?:(?:write|create|compose|generate|draft|provide|give)\s+"
        r"(?:(?:a|an|the|full|complete|detailed|brief)\s+)*"
        r"(?:report|essay|article|analysis|plan|framework|summary)\b|"
        r"(?:explain|analy[sz]e|discuss|describe|evaluate|assess)\b|"
        r"(?:(?:a|an|the|full|complete|detailed|brief)\s+)*"
        r"(?:report|essay|article|analysis|plan|framework|summary)\b)", question)
    chinese_request = re.search(r"(?:撰写|写|拟|生成|提供|给出|返回|输出|推荐|起)[^。！？\n]{0,12}(?:标题|题目)", question)
    chinese_additional_prose = re.search(
        r"(?:并|以及|然后|同时|再)[^。！？\n]{0,12}(?:撰写|写|解释|分析|报告|文章|论文|方案|计划)", question)
    if (request or chinese_request) and not (additional_prose or chinese_additional_prose):
        return True
    return (re.search(r"\b(?:poem|poetry|verse|haiku)\b", question) is not None
            and re.search(r"\b(?:markdown\s+)?heading\s+lines?\b", question + " " + constraints) is not None
            and not additional_prose)


def _completion_quality_error(answer, public_task):
    """Detect an obvious non-answer while leaving domain quality to the judge.

    The check rejects a heading/status disclaimer in place of a requested report,
    essay, analysis, plan, or framework. It does not enforce a universal length,
    citations, or any hidden rubric requirement.
    """
    if not isinstance(answer, str) or not _requires_substantive_answer(public_task):
        return None
    # Retain separators before normalizing: a concise heading followed by real
    # prose is still a valid deliverable. Some gateways flatten newlines into
    # double spaces, so recognize those separators as well.
    chunks = [chunk.strip() for chunk in re.split(r"\n+|\s{2,}", answer) if chunk.strip()]
    text = " ".join(answer.split())
    if not text:
        return "The substantive public deliverable is empty"
    heading_output_requested = _heading_output_requested(public_task)

    def non_content(chunk):
        # Exclude source-status parentheticals from the title test, not from the
        # user's answer. They are retained unchanged in the raw trajectory.
        title_text = re.sub(
            r"[\[(][^\])]*(?:unverified|no external|not live|general knowledge|"
            r"no live search)[^\])]*[\])]", "", chunk, flags=re.I).strip()
        title = re.match(r"^(?:#{1,6}\s+|(?:title|subject|topic)\s*:)", title_text, re.I)
        # A sentence in the same line as a heading can contain substantive
        # analysis; only reject recognizable headings without sentence prose.
        if (title and not heading_output_requested
                and not re.search(r"[.!?](?:\s+[A-Z]|\s*$)", title_text.rstrip(".!?"))):
            return True
        statement = re.sub(r"^[*\s]+|[*\s]+$", "", chunk)
        # Match entire status/meta statements rather than a broad 'This report
        # is ...' prefix: an ordinary short report introduction may be useful.
        return re.fullmatch(
            r"(?:No (?:external|primary|live|retrieved) (?:sources?|evidence|search|retrieval) "
            r"(?:was|were|has been|have been)?\s*(?:observed|retrieved|accessed|verified|performed)"
            r"(?:\s+(?:in|during|for) this run)?[.!;]*|"
            r"(?:The|This) (?:essay|report|answer|response|deliverable) is "
            r"(?:(?:the|a)\s+)?(?:primary|final|complete|requested|full) "
            r"(?:answer|deliverable|artifact|response)(?:\s+(?:itself|requested by the public task))?[.!]*|"
            r"(?:All (?:specific |historical |factual )?(?:claims|citations|sources)|"
            r"(?:This|The) (?:report|essay|answer)) (?:are|is) "
            r"(?:remembered|unverified|based on (?:model )?general knowledge)"
            r"(?:\s+(?:and|/|,|;)\s*(?:remembered|unverified))?[.!]*|"
            r"I (?:have completed|am ready to (?:provide|submit)) (?:the|this) "
            r"(?:assignment|report|essay|answer|deliverable)[.!]*)",
            statement, re.I) is not None

    if chunks and all(non_content(chunk) for chunk in chunks):
        return ("The response is a title or completion/provenance statement rather than "
                "the substantive deliverable requested by the public task")
    return None


def _validate_completion_fields(parsed):
    unknown_fields = set(parsed) - EXECUTION_RESPONSE_FIELDS
    if unknown_fields:
        raise ResponseProtocolError(
            "Unknown execution response fields: " + json.dumps(sorted(unknown_fields)))
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
    unknown_fields = sorted(str(key) for key in value if key not in CONTRIBUTION_LEDGER_FIELDS)
    if unknown_fields:
        raise ResponseProtocolError("Unknown contributor ledger fields: " + json.dumps(unknown_fields))
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
    return copy.deepcopy({key: value[key] for key in CONTRIBUTION_LEDGER_FIELDS})


def _contributor_compactness_warnings(parsed, *, closed_book=False):
    """Measure budget overruns without editing a shape-validated contribution.

    Guided schemas may enforce compact output while decoding. JSON Object output
    uses the same limits as advisory targets so useful, legal content is retained.
    """
    warnings = []

    def measure(field, actual, suggested_max):
        if actual > suggested_max:
            warnings.append({"field": field, "actual": actual, "suggested_max": suggested_max})

    answer_limit = 2400 if closed_book else CONTRIBUTOR_ANSWER_MAX_CHARS
    item_limit = 24 if closed_book else CONTRIBUTOR_LEDGER_MAX_ITEMS
    text_limit = CONTRIBUTOR_LEDGER_TEXT_MAX_CHARS
    answer = parsed.get("answer")
    if isinstance(answer, str):
        measure("answer.characters", len(answer), answer_limit)
    ledger = parsed.get("ledger")
    if ledger is not None:
        for key in ("requirements", "outline", "source_references", "evidence_spans"):
            measure(f"ledger.{key}.items", len(ledger[key]), item_limit)
        for key in ("requirements", "outline"):
            for index, item in enumerate(ledger[key]):
                measure(f"ledger.{key}[{index}].characters", len(item), text_limit)
        for index, source in enumerate(ledger["source_references"]):
            measure(f"ledger.source_references[{index}].locator.characters", len(source["locator"]),
                    CONTRIBUTOR_LOCATOR_MAX_CHARS)
        for index, span in enumerate(ledger["evidence_spans"]):
                measure(f"ledger.evidence_spans[{index}].text.characters", len(span["text"]),
                        text_limit)
    return warnings


def _correctable_execution_shape_error(error, response, *, observed_ids, allowed_tools,
                                      peer_ids, synthesizer):
    """Permit one model-authored shape repair without recalling an authority violation."""
    prefixes = (
        "Expected a complete JSON object", "Response must be a JSON object",
        "Unknown execution response fields: ",
        "Unknown contributor ledger fields: ",
        "answer must be a nonempty string", "checkpoints must be an object",
        "Unconfirmed checkpoints: ",
        "A checkpoint must be", "evidence_ids must be a list",
        "tools must be a list of named tool calls", "Tool arguments must encode",
        "Tool arguments must be an object", "continue must be a boolean",
        "Contributors must publish a structured ledger object", "ledger.requirements must be",
        "ledger.outline must be", "ledger needs evidence_spans and source_references lists",
        "Each source reference needs source_id and locator", "Source IDs must be unique",
        "Each evidence span needs text and a declared source_ref",
    )
    if not str(error).startswith(prefixes):
        return False
    content = getattr(response, "content", response)
    try:
        if isinstance(content, dict):
            parsed = content
        else:
            text = str(content or "").strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # No actions or evidence from malformed JSON have been accepted.
        return True
    if not isinstance(parsed, dict):
        return True
    completions = [parsed]
    calls = parsed.get("tools", [])
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            continue
        name = call["name"]
        if name not in {*allowed_tools, "send_message", "complete", "final_answer"}:
            return False
        if name == "final_answer" and not synthesizer:
            return False
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(arguments, dict):
            continue
        if name in {"complete", "final_answer"}:
            completions.append(arguments)
        elif name == "send_message" and arguments.get("recipient") not in peer_ids:
            return False
    for completion in completions:
        citations = [completion.get("evidence_ids", [])]
        checkpoints = completion.get("checkpoints", {})
        if isinstance(checkpoints, dict):
            citations.extend(check.get("evidence_ids", []) for check in checkpoints.values()
                             if isinstance(check, dict))
        if any(isinstance(ids, list) and any(isinstance(item, str) and item not in observed_ids
                                          for item in ids) for ids in citations):
            return False
    return True


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
    """Build the complete current public snapshot, retaining its established shape."""
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


def _replace_iterative_ledger_snapshot(messages, ledger, state):
    """Keep one runtime snapshot in context without removing any other history.

    Only the exact fragment previously inserted here may be removed. Tool/peer
    observations and correction instructions can share that user message, and
    their original text must survive. Earlier model inputs remain deep-copied in
    StepRecord, so replacing a context snapshot does not rewrite the audit trail.
    """
    previous = state.get("ledger_snapshot_message")
    if previous is not None:
        prefix = state["ledger_snapshot_prefix"]
        suffix = state["ledger_snapshot_suffix"]
        for index, message in enumerate(messages[2:], 2):
            if message is previous:
                if message["content"] != prefix + suffix:
                    raise RuntimeError("Runtime ledger snapshot message changed unexpectedly")
                if prefix:
                    message["content"] = prefix
                else:
                    del messages[index]
                break
    content = "Updated shared ledger: " + json.dumps(ledger, ensure_ascii=False)
    if len(messages) > 2 and messages[-1]["role"] == "user":
        message = messages[-1]
        prefix = message["content"]
        suffix = "\n" + content
        message["content"] = prefix + suffix
    else:
        prefix, suffix = "", content
        message = {"role": "user", "content": content}
        messages.append(message)
    state.update(ledger_snapshot_message=message, ledger_snapshot_prefix=prefix,
                 ledger_snapshot_suffix=suffix)
    return content


def _ledger_evidence_ids(ledger):
    return {item["event_id"] for key in ("contributions", "tool_evidence", "communications")
            for item in ledger[key] if item.get("event_id")}


def _persistent_role(agent, services):
    if services.agent_pool is None:
        return None
    from .agent_pool import role_context
    context = role_context(services.agent_pool, AgentSpec.model_validate(agent), services.public_task)
    context.update(knowledge_policy_role_adaptation(context["pool_agent_id"], services.knowledge_policy,
                                                  prompt=context["prompt"]))
    return context


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
    tail = messages[2:]
    selected = tail[-window:] if window else []
    # A no-progress correction is a user turn paired with the draft it is
    # correcting. Retain that preceding assistant turn so a recent-memory
    # agent can revise the draft instead of seeing only the instruction.
    if (selected and selected[0].get("role") == "user"
            and isinstance(selected[0].get("content"), str)
            and selected[0]["content"].startswith(("Continue the assigned work", "Continuation correction:",
                                                  "Protocol correction:", "Deliverable correction:"))
            and len(tail) > len(selected)
            and tail[-len(selected) - 1].get("role") == "assistant"):
        selected = [tail[-len(selected) - 1], *selected]
    return copy.deepcopy(messages[:2] + selected)


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
                          "evidence_ids": [], "checkpoints": {name: {
                              "status": "unverified", "reason": "Replace with the actual check result or limitation.",
                              "evidence_ids": []} for name in agent.get("checkpoints", [])}}
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
        "A terminal response must include the role-appropriate answer and all assigned checkpoints: "
        "the synthesizer supplies the complete public deliverable, while a contributor supplies "
        "a compact summary and its substantive ledger. "
        "The role-specific completion_example is the authoritative output shape. Use every exact "
        "agent.checkpoints name as a key. Values are boolean true for an actually performed check "
        "or objects with status completed/passed/failed/unverified/not_applicable, a nonempty "
        "reason, and evidence_ids. Never use prose strings as checkpoint values. When no source "
        "was observed, ledger evidence_spans and source_references must be empty arrays; place "
        "remembered knowledge and limitations in the role-appropriate answer or ledger, never "
        "in fabricated evidence entries. "
        "send_message posts to an exact peer agent_id and reactivates a completed teammate. "
        "After asking a peer, the scheduler lets that peer respond before continuing your role. "
        "A completed contribution may be resumed to answer questions or revise its public artifact. "
        "The synthesizer submits only after all teammates and pending peer requests complete."
        " Use resource_budget to conserve shared tokens: avoid duplicate analysis and full draft "
        "copies, send concise evidence-linked messages, and reserve room for the complete final "
        "deliverable. Follow budget_policy's completion and quality-cost stopping guidance. "
        "budget_estimate is a forecast, not a fixed call or round limit."
    )
    if _uses_fixed_evidence(services.public_task):
        system += FIXED_EVIDENCE_CITATION_PROMPT
    if synth:
        system += (
            "\nYou are the final synthesizer even if your persistent role is a reviewer or critic. "
            "Return the complete artifact requested by public_task, incorporating the useful peer "
            "contributions and corrections. A review verdict, approval, change list, or instructions "
            "for a future writer cannot replace the requested deliverable."
            " A title or a completion/provenance statement alone cannot complete a prose/artifact "
            "assignment. Concise substantive answers and explicitly requested short formats remain valid."
        )
    shared_ledger = _iterative_public_ledger(services, aid)
    ledger_hash = content_hash(shared_ledger)
    observed_ids = _ledger_evidence_ids(shared_ledger)
    services.event(aid, "shared_ledger_read", {"ledger_hash": ledger_hash},
                   parents=sorted(observed_ids))
    if services.ledger is not None:
        services.ledger.charge_communication(
            len(json.dumps(shared_ledger, ensure_ascii=False).encode("utf-8")), stage="execution", agent_id=aid)
    instruction = {"public_task": _data(services.public_task),
                   "agent": _execution_assignment(agent, synth),
                   "shared_ledger": shared_ledger,
                   "own_contribution": None,
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
                                  "reactivate_completed_agents": True,
                                  "continuation_policy": ITERATIVE_CONTINUATION_POLICY_VERSION},
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
    if services.knowledge_policy is not None:
        instruction["knowledge_policy"] = services.knowledge_policy
        system += knowledge_policy_prompt(services.knowledge_policy)
        if services.knowledge_policy == "model_general_knowledge_allowed":
            system += (
                "\nContributors publish a compact substantive summary in answer and put supporting facts, "
                "key derivations and limitations in the structured ledger; the synthesizer publishes "
                "the complete requested deliverable in answer. This run has no external source "
                "observations, so publish evidence_spans=[] and source_references=[], including for "
                "evidence or research roles. Do not turn remembered knowledge into source entries. "
                "Keep requirements and outline as lists of nonempty strings."
            )
    if not synth:
        system += (
            "\nFINAL ROLE OVERRIDE (takes priority over task_prompt and retained role text): "
            "You are a contributor, never the final Writer. Do not write the complete public "
            "deliverable even if task_prompt says write or produce it. Publish a compact summary plus "
            "the substantive structured ledger. Compactness guidance is a soft target, not a limit on "
            "required content. A reviewer should prioritize three to five consequential findings and "
            "supported corrections while preserving additional omissions needed for correctness. "
            "own_contribution contains only your latest published artifact, not foreign private history."
        )
    else:
        system += (
            "\nFINAL WRITER OVERRIDE: Return one terminal complete public deliverable and cover each "
            "explicit public-task requirement. When the public task calls for substantive analysis, "
            "include a specific mechanism, example or calculation where appropriate and a clear conclusion. "
            "Exact short formats and output-only instructions take precedence over that presentation advice. Use a format "
            "appropriate to the task; do not repeat drafts, ledger bodies, review narration or "
            "protocol metadata, and leave enough room for valid JSON closure. "
            f"The complete JSON response has a hard ceiling of {output_limit} output tokens. "
            "For a long response, planning below about 80-85% of that ceiling leaves a protocol safety "
            "margin for checkpoints and JSON escaping; this is not a target answer length. Public-task "
            "length, count and output-only constraints take precedence, and short answers need no padding. "
            "Preserve every requested item, key argument and "
            "calculation; reduce duplicated prose and unnecessary lists before shortening substance."
        )
    system += "\n" + (_answer_check_prompt() if synth else
                          _contributor_handoff_prompt(
                              closed_book=services.knowledge_policy == "model_general_knowledge_allowed"))
    system += "\n" + QUALITY_ASSURANCE_PROMPT
    if synth:
        system += "\n" + FINAL_ARTIFACT_CONTRACT
        system += "\n" + FINAL_SUBMISSION_GATE
        instruction["terminal_assignment"] = FINAL_ARTIFACT_CONTRACT
    system = _apply_public_positional_draft_guidance(system, instruction, services)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)}]
    state = state if state is not None else {}
    state.update(initialized=True, persistent=persistent, model=model, before=before,
                 output_limit=output_limit, allowed=allowed, peer_ids=peer_ids,
                 messages=messages, trajectory=[], answer=None,
                 contribution={"requirements": [], "outline": [], "evidence_spans": [],
                               "source_references": []}, checkpoint_reports={}, evidence_ids=[],
                 pending_observation_ids=set(), observed_ids=observed_ids,
                 ledger_hash=ledger_hash, handled_message_ids=set(), awaiting_messages=[],
                 continuation_fingerprint=None, no_progress_correction_given=False,
                 protocol_correction_given=False,
                 answer_quality_correction_given=False)
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
            update_content = _replace_iterative_ledger_snapshot(messages, updated_ledger, state)
            observed_ids.update(_ledger_evidence_ids(updated_ledger))
            ledger_hash = updated_hash
            services.event(aid, "shared_ledger_read", {"ledger_hash": ledger_hash},
                           parents=sorted(_ledger_evidence_ids(updated_ledger)))
            if services.ledger is not None:
                services.ledger.charge_communication(
                    len(update_content.encode("utf-8")), stage="execution", agent_id=aid)
        elif state.get("ledger_snapshot_message") is not None:
            # Recent-memory roles also need the current snapshot on a turn with
            # no new peer publication. Move it alongside the newest observation
            # or correction instead of leaving it outside their history window.
            _replace_iterative_ledger_snapshot(messages, updated_ledger, state)
        observed_ids.update(pending_observation_ids)
        pending_observation_ids.clear()
        state["handled_message_ids"].update(item["event_id"]
                                           for item in updated_ledger["communications"])
        instruction = json.loads(messages[1]["content"])
        if state.get("ledger_snapshot_message") is not None:
            instruction["shared_ledger"] = {
                "snapshot_policy": "latest-full-runtime-snapshot-v1",
                "snapshot_location": "The latest user message containing Updated shared ledger: "
                                     "holds the complete current shared_ledger JSON; read it for "
                                     "contributions, cumulative ledgers, tool evidence and peer messages.",
                "snapshot_sha256": updated_hash,
            }
        with services.lock:
            instruction["own_contribution"] = copy.deepcopy(services.artifacts.get(aid))
        if services.ledger is not None:
            instruction["resource_budget"] = services.ledger.resource_context()
        messages[1]["content"] = json.dumps(instruction, ensure_ascii=False)
        context_messages = _role_messages(messages, persistent)
        external_input_hash = content_hash({"ledger_hash": updated_hash,
                                           "observed_evidence_ids": sorted(observed_ids)})
        step = StepRecord(step_number=len(trajectory) + 1, model_input_messages=context_messages,
                          start_time=time.time())
        try:
            response = _bounded_call(model, min(services.timeout_seconds, remaining), context_messages,
                                     deadline_remaining=services.remaining_seconds,
                                     max_tokens=output_limit,
                                     response_format={"type": "json_object"})
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
            if synth and terminal and merged.get("answer") is not None:
                quality_error = _completion_quality_error(merged["answer"], services.public_task)
                if quality_error:
                    raise ResponseProtocolError(quality_error)
            continuation_correction = False
            if not terminal and not tool_calls:
                fingerprint = (content_hash(parsed), external_input_hash)
                if fingerprint == state["continuation_fingerprint"]:
                    if state["no_progress_correction_given"]:
                        raise ResponseProtocolError(
                            "Repeated unchanged nonterminal response without new external input "
                            "after a continuation correction; task attempt failed with no progress")
                    continuation_correction = True
                    services.event(aid, "continuation_warning", {
                        "response_hash": fingerprint[0], "external_input_hash": external_input_hash,
                        "policy": ITERATIVE_CONTINUATION_POLICY_VERSION}, parents=[output_event])
                state["continuation_fingerprint"] = fingerprint
                state["no_progress_correction_given"] = continuation_correction
            else:
                state["continuation_fingerprint"] = None
                state["no_progress_correction_given"] = False
            if not synth and merged.get("ledger") is not None:
                contribution = _contribution_ledger(merged["ledger"])
                compactness_warnings = _contributor_compactness_warnings(
                    merged, closed_book=services.knowledge_policy == "model_general_knowledge_allowed")
                if compactness_warnings:
                    services.event(aid, "compactness_warning", {
                        "policy": CONTRIBUTOR_COMPACTNESS_POLICY_VERSION,
                        "measurements": compactness_warnings,
                        "content_preserved": True,
                    }, parents=[output_event])
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
            elif not terminal and content_hash(_iterative_public_ledger(services, aid)) == updated_hash:
                continuation = (
                    "Continue the assigned work with substantive facts, reasoning, results or revision; make a substantive contribution. "
                    "A title, heading-only outline or promise of future work does not complete the assignment. "
                    "Return one valid JSON object. When the role is actually complete, include its "
                    "role-appropriate answer (complete deliverable for the synthesizer; compact summary "
                    "plus fact-rich ledger for a contributor), assigned checkpoint reports, and set continue=false. "
                    "Preserve honest limitations; do not declare an incomplete artifact complete. "
                    + (_answer_check_prompt() if synth else
                       _contributor_handoff_prompt(
                           closed_book=services.knowledge_policy == "model_general_knowledge_allowed")))
                if continuation_correction:
                    continuation = (
                        "Continuation correction: your response repeated unchanged without new external input. "
                        "Produce substantive progress or complete the actual assigned contribution; for a "
                        "contributor put concrete facts, reasoning and results in the ledger and keep answer compact. "
                        "Another unchanged nonterminal response without new external input will fail this attempt. "
                        + continuation)
                messages.append({"role": "user", "content": continuation})
            if terminal:
                reason = "final_answer" if synth else "subtask_complete"
                step.end_time = time.time()
                step.duration = step.end_time - step.start_time
                trajectory.append(step)
                break
        except ResponseProtocolError as exc:
            # A model may miss the JSON/checkpoint/ledger shape contract. Give
            # the same role one explicit protocol correction before failing the
            # attempt; this is a normal iterative turn, not a harness repair or
            # a score-based retry.
            if (not state["answer_quality_correction_given"]
                    and (str(exc).startswith("The response is a title")
                         or str(exc).startswith("The substantive public deliverable"))
                    and services.remaining_seconds() > 0):
                state["answer_quality_correction_given"] = True
                step.error = exc
                step.observations = f"{type(exc).__name__}: {exc}"
                services.event(aid, "quality_warning", step.observations, parents=[before])
                messages.append({"role": "assistant", "content": getattr(response, "content", str(response))})
                messages.append({"role": "user", "content": (
                    "Deliverable correction: the previous answer was only a title, completion/status "
                    "statement, or provenance disclaimer. Continue the assigned public task and return "
                    "the actual substantive deliverable, with the required JSON fields and checkpoint "
                    "reports. Do not replace the requested report, essay, analysis, plan, or framework "
                    "with a promise to provide it. Preserve honest uncertainty and use only available "
                    "evidence. This is the single quality correction allowed for this role. "
                    + str(exc))})
                step.end_time = time.time()
                step.duration = step.end_time - step.start_time
                trajectory.append(step)
                if one_turn:
                    reason = "yielded"
                    break
                continue
            if (not state["protocol_correction_given"]
                    and services.remaining_seconds() > 0
                    and _correctable_execution_shape_error(
                        exc, response, observed_ids=observed_ids, allowed_tools=allowed,
                        peer_ids=peer_ids, synthesizer=synth)):
                state["protocol_correction_given"] = True
                step.error = exc
                step.observations = f"{type(exc).__name__}: {exc}"
                services.event(aid, "protocol_warning", step.observations, parents=[before])
                truncated = str(exc).startswith("Expected a complete JSON object; response may be truncated")
                if not truncated:
                    messages.append({"role": "assistant", "content": getattr(response, "content", str(response))})
                messages.append({"role": "user", "content": (
                    "Protocol correction: your previous JSON did not satisfy the execution contract. "
                    "Return exactly one JSON object with only these top-level fields: "
                    + json.dumps(sorted(EXECUTION_RESPONSE_FIELDS)) + ". "
                    "Keep the complete deliverable in answer, never split its prose across invented JSON keys. "
                    "Escape quotes inside the answer string. Every assigned checkpoint value must be either "
                    "true or an object with status set to completed, passed, failed, unverified, or "
                    "not_applicable, a nonempty reason string, and an evidence_ids array. Do not use "
                    "prose strings, status=complete, or embedded key-value text. Include all exact "
                    "checkpoint names and a role-appropriate answer when terminating (full deliverable only "
                    "for the synthesizer; compact summary plus ledger for a contributor). "
                    "Perform any missing checks you can support; otherwise report failed or unverified "
                    "with a specific reason. Do not invent a passed finding to finish. Preserve honest uncertainty. "
                    "Exact assigned checkpoint keys: " + json.dumps(agent.get("checkpoints", []), ensure_ascii=False) + ". "
                    "For a contributor, ledger permits exactly requirements, outline, evidence_spans and "
                    "source_references. Put substantive content from unsupported ledger keys into outline "
                    "strings or the role-appropriate answer; do not discard it. requirements and outline "
                    "are JSON arrays of nonempty strings (for example, \"outline\":[\"Claim and reason.\"]); "
                    "never put objects, numbers or nested arrays in those two fields. "
                    "evidence_spans and source_references are lists. "
                    "Each observed source entry needs nonempty source_id and locator; each evidence "
                    "span needs nonempty text and source_ref matching a declared source_id. "
                    "Exactly one source_references entry per distinct stable source_id in your contribution. "
                    "When one observed dossier or fixed-pack source contains multiple URLs or locations, "
                    "combine its relevant observed locators into that one entry's locator string; multiple "
                    "evidence_spans may reuse its source_ref. Do not duplicate an ID for each URL or rename "
                    "fixed-pack IDs to hide duplication. Preserve every supported span and actual locator "
                    "while consolidating duplicate entries yourself before returning. "
                    "If no sources were observed, ledger source_references/evidence_spans must be empty arrays. "
                    + ("This run is closed book: do not convert remembered citations into source objects. "
                       "Keep source_references=[] and evidence_spans=[]; move useful remembered claims, "
                       "citation details and limitations into ledger.outline as nonempty strings, "
                       "retaining the substantive material with its uncertainty. "
                       if services.knowledge_policy == "model_general_knowledge_allowed" else "") +
                    "For a contributor, put a compact summary in answer and substantive facts, reasoning and "
                    "results in ledger; for the synthesizer, put the complete public deliverable in answer. "
                    "Make the correction yourself; "
                    "the previous response has not been accepted or automatically edited. "
                    "If the previous JSON was incomplete or truncated, the failed assistant text is intentionally "
                    "omitted from this correction context; rewrite one complete JSON "
                    "response from the original task rather than continuing its broken tail or "
                    "reproducing it unchanged. Preserve the key arguments, calculations, required "
                    "items and honest limitations; delete duplicate passages, guessed directories "
                    "and speculative numbered catalogs, then close every JSON field. "
                    + (_answer_check_prompt() if synth else
                       "Apply the original system's advisory compactness and handoff guidance. "
                       "Return a compact summary and a cumulative current-valid ledger, preserving "
                       "required facts, checked intermediate results, necessary derivations, complete "
                       "supported sets and source references. Retract superseded claims explicitly. ") + " "
                    "Exact validation error: " + str(exc))})
                step.end_time = time.time()
                step.duration = step.end_time - step.start_time
                trajectory.append(step)
                if one_turn:
                    reason = "yielded"
                    break
                continue
            step.error = exc
            step.observations = f"{type(exc).__name__}: {exc}"
            services.event(aid, "execution_error", step.observations, parents=[before])
            reason = "error"
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
                                              "reactivate_completed_agents": True,
                                              "continuation_policy": ITERATIVE_CONTINUATION_POLICY_VERSION},
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
        "public_task": _data(services.public_task), "agent": _execution_assignment(agent, synth),
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
    if services.knowledge_policy is not None:
        instruction["knowledge_policy"] = services.knowledge_policy
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
        "checks do not replace external evaluation. Address material limitations in the role-appropriate "
        "answer or ledger. "
        "You receive exactly one model call. There are no agent messaging or shared-memory lookup tools. "
        "Express missing input or disputed evidence in the role-appropriate answer, ledger and checkpoint reports. "
        "Never claim unobserved evidence or broadcast private conversations."
    )
    if _uses_fixed_evidence(services.public_task):
        system += FIXED_EVIDENCE_CITATION_PROMPT
    if not synth:
        system += (
            "\nFINAL ROLE OVERRIDE (takes priority over task_prompt and retained role text): "
            "You are a contributor, never the final Writer. Do not write the complete public "
            "deliverable even if task_prompt says write or produce it. Publish a compact summary plus "
            "the substantive structured ledger. Compactness guidance is a soft target, not a limit on "
            "required content. A reviewer should prioritize three to five consequential findings and "
            "supported corrections while preserving additional omissions needed for correctness."
        )
    else:
        system += (
            "\nFINAL WRITER OVERRIDE: Return one terminal complete public deliverable and cover each "
            "explicit public-task requirement. When the public task calls for substantive analysis, "
            "include a specific mechanism, example or calculation where appropriate and a clear conclusion. "
            "Exact short formats and output-only instructions take precedence over that presentation advice. Use a format "
            "appropriate to the task; do not repeat drafts, ledger bodies, review narration or "
            "protocol metadata, and leave enough room for valid JSON closure. "
            "For a long response, planning below about 80-85% of the output-token ceiling leaves a "
            "protocol safety margin for checkpoints and escaping; this is not a target answer length. "
            "Public-task length, count and output-only constraints take precedence, and short answers "
            "need no padding. Preserve every requested "
            "item, key argument and calculation; reduce duplicated prose and unnecessary lists "
            "before shortening substance."
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
    if synth:
        system += (
            f"\nEach complete JSON response has a hard ceiling of {output_limit} output tokens, "
            "including escaped prose, evidence IDs, checkpoints and closing braces. Use the "
            "available budget for the complete deliverable, remove repetition, and leave enough "
            "room to close the JSON object."
        )
    else:
        system += (
            f"\nEach complete JSON response has a hard ceiling of {output_limit} output tokens. "
            "Remove repetition first while retaining facts, key derivations and honest limitations. "
            "This single-pass role must "
            "include its complete substantive contribution in this response."
        )
    system += knowledge_policy_prompt(services.knowledge_policy)
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
            " A title or a completion/provenance statement alone cannot complete a prose/artifact "
            "assignment. Concise substantive answers and explicitly requested short formats remain valid."
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
            "Keep substantive requirements, concrete claims, reasoning steps, intermediate results and evidence "
            "in their corresponding ledger fields, without repeating them in a competing full answer or repetitive lists. "
            "Every outline item must contain a useful claim, reasoning step, result or uncertainty; do not emit "
            "section headings alone. Distinguish explicit public requirements from inferred planner suggestions, "
            "especially guessed quantity targets. For quantitative or financial work, preserve definitions, "
            "assumptions, formulas, units and checked intermediate results. For regulatory work, distinguish "
            "a legal or policy obligation from a recommended strategy or prudent market-entry practice. "
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
    system += "\n" + (_answer_check_prompt() if synth else
                          _contributor_handoff_prompt(
                              closed_book=services.knowledge_policy == "model_general_knowledge_allowed"))
    system += "\n" + QUALITY_ASSURANCE_PROMPT
    if synth:
        system += "\n" + FINAL_ARTIFACT_CONTRACT
        system += "\n" + FINAL_SUBMISSION_GATE
        instruction["terminal_assignment"] = FINAL_ARTIFACT_CONTRACT
    system = _apply_public_positional_draft_guidance(system, instruction, services)
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
                                     max_tokens=output_limit,
                                     response_format={"type": "json_object"})
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
            if synth and parsed.get("answer") is not None:
                quality_error = _completion_quality_error(parsed["answer"], services.public_task)
                if quality_error:
                    raise ResponseProtocolError(quality_error)
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
                    compactness_warnings = _contributor_compactness_warnings(
                        parsed, closed_book=services.knowledge_policy == "model_general_knowledge_allowed")
                    if compactness_warnings:
                        services.event(aid, "compactness_warning", {
                            "policy": CONTRIBUTOR_COMPACTNESS_POLICY_VERSION,
                            "measurements": compactness_warnings,
                            "content_preserved": True,
                        }, parents=[output_event])
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
                               "contributor_compactness_policy": CONTRIBUTOR_COMPACTNESS_POLICY_VERSION,
                               "artifacts": copy.deepcopy(services.artifacts),
                               "coordination": ("single_pass_shared_ledger"
                                                if services.execution_mode == "single_pass"
                                                else services.execution_mode),
                               **({"scheduling": {"strategy": "cooperative_shared_ledger",
                                                  "reactivate_completed_agents": True,
                                                  "continuation_policy": ITERATIVE_CONTINUATION_POLICY_VERSION}}
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
                 unsafe_local=False, knowledge_policy=None):
        self.model_factory = model_factory
        self.tools = tools or {}
        self.ledger = ledger
        self.timeout_seconds = timeout_seconds
        self.unsafe_local = unsafe_local
        knowledge_policy_prompt(knowledge_policy)
        self.knowledge_policy = knowledge_policy

    def execute(self, task, team, artifact, rubrics=None, experiences=()):
        if artifact.backend not in {"scripted", "native_jit"}:
            raise ValueError("unknown harness backend")
        task = PublicTask.model_validate(_data(task))
        team = TeamSpec.model_validate(_data(team))
        if self.knowledge_policy == "model_general_knowledge_allowed" and (task.tools or self.tools):
            raise ValueError("General-knowledge execution requires an empty external tool allowlist")
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
                                agent_pool=agent_pool, knowledge_policy=self.knowledge_policy)
        services.public_positional_draft_plan = getattr(self, "public_positional_draft_plan", None)
        services.public_positional_draft_projection_requested = getattr(
            self, "public_positional_draft_projection_requested", False)
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
        if getattr(self, "public_positional_draft_guidance_requested", False):
            plan = services.public_positional_draft_plan
            result.metadata["public_positional_draft_guidance"] = {
                "version": PUBLIC_POSITIONAL_DRAFT_GUIDANCE_VERSION,
                "active": plan is not None, "final_constraints_retained": True,
                "public_position_plan": plan.audit() if plan is not None else None,
                "reason": None if plan is not None else
                    "No single supported conflict-free public positional rule",
            }
        if getattr(self, "public_positional_draft_projection_requested", False):
            result.metadata["public_positional_draft_projection"] = copy.deepcopy(
                services.public_positional_draft_projection_audit)
        if self.knowledge_policy is not None:
            result.metadata["knowledge_policy"] = self.knowledge_policy
        if services.agent_pool is not None:
            result.metadata["agent_pool_hash"] = content_hash(services.agent_pool)
        # Runtime sees a coordinator with no model calls. Sum disjoint leaf traces
        # here; the shared ledger is authoritative and is never charged again.
        result.metadata["input_token_count"] = sum(s.input_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["output_token_count"] = sum(s.output_token_count for r in result.sub_runs for s in r.trajectory)
        result.metadata["total_token_count"] = (result.metadata["input_token_count"] +
                                                 result.metadata["output_token_count"])
        return result
