"""Task-conditioned global/local planning over JIT's model callable protocol."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator

from .experience import experience_applicability
from .schemas import (
    AgentSpec, LocalPlan, PlannedTeam, Prediction, PublicTask, Record, RubricGraph, TeamSpec,
    AgentPoolSnapshot,
)

T = TypeVar("T", bound=BaseModel)


class _ReconciliationResponse(Record):
    graph: RubricGraph
    team: TeamSpec

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_plan_echo(cls, value: Any) -> Any:
        # Older scripted providers echoed plans; authors' validated originals remain authoritative.
        if isinstance(value, dict) and "local_plans" in value:
            TypeAdapter(list[LocalPlan]).validate_python(value["local_plans"])
            return {key: item for key, item in value.items() if key != "local_plans"}
        return value


def _pooled_reconciliation_schema(prediction: Prediction, *, max_agents: int) -> dict[str, Any]:
    """Bind generated selections and references to the original pooled candidates.

    Use complete anyOf object branches: the execution gateway does not implement
    if/then, and partial branches do not reliably inherit sibling properties.
    Semantic ownership, DAG and resource checks still run on the returned record.
    """
    schema = _ReconciliationResponse.model_json_schema()
    definitions = schema["$defs"]
    candidate_ids = [candidate.agent_id for candidate in prediction.candidates]
    original_agent = definitions["AgentSpec"]
    branches = []
    for candidate in prediction.candidates:
        branch = copy.deepcopy(original_agent)
        properties = branch["properties"]
        for field, value in (("agent_id", candidate.agent_id),
                             ("pool_agent_id", candidate.pool_agent_id),
                             ("pool_agent_version", candidate.pool_agent_version)):
            properties[field] = {"type": "integer" if type(value) is int else "string",
                                 "const": value}
            if field not in branch["required"]:
                branch["required"].append(field)
        other_ids = [aid for aid in candidate_ids if aid != candidate.agent_id]
        properties["depends_on"]["items"] = {"type": "string", "enum": other_ids} if other_ids \
            else {"type": "string"}
        if not other_ids:
            properties["depends_on"]["maxItems"] = 0
        branches.append(branch)
    definitions["AgentSpec"] = {"anyOf": branches}
    team = definitions["TeamSpec"]["properties"]
    team["agents"]["maxItems"] = min(team["agents"]["maxItems"], max_agents, len(candidate_ids))
    id_schema = {"type": "string", "enum": candidate_ids}
    team["synthesizer_id"] = copy.deepcopy(id_schema)
    team["primary"]["additionalProperties"] = copy.deepcopy(id_schema)
    for field in ("coverage", "reviewers"):
        assignment = team[field]["additionalProperties"]
        assignment["items"] = copy.deepcopy(id_schema)
    definitions["AgentBudgetEstimate"]["properties"]["agent_id"] = copy.deepcopy(id_schema)
    return schema


def as_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value


def knowledge_policy_prompt(knowledge_policy: str | None) -> str:
    if knowledge_policy is None:
        return ""
    if knowledge_policy != "model_general_knowledge_allowed":
        raise ValueError("unknown knowledge_policy")
    return ("\nKNOWLEDGE POLICY: model_general_knowledge_allowed. Use the public task and "
            "model general knowledge to complete the requested deliverable. This run has no "
            "external retrieval or source-access tools. Do not assign or wait for web searches, "
            "paper downloads, live data collection, URL verification or newly retrieved sources. "
            "Adapt research and evidence roles to organize relevant remembered knowledge, compare "
            "claims, check reasoning and flag consequential uncertainty. Preserve useful facts, "
            "examples, remembered citations and detail; distinguish them from sources actually "
            "observed in this run and never claim external retrieval or independent source verification. "
            "Checkpoints must describe attainable checks of the current contribution, not unavailable "
            "external research or hypothetical peer activity. If a requested check cannot be verified, "
            "report status=unverified with a reason; if a conditional check was not triggered, report "
            "status=not_applicable with a reason. A reported limitation permits completion. "
            "Source references and evidence spans may be empty when no source was observed. Cite "
            "evidence IDs only from actual delivered ledger events. Do not invent evidence to satisfy "
            "a checkpoint. Finish when the assigned artifact is complete; do not continue merely to "
            "wait for nonexistent tools or unsolicited future peer requests. Dynamic role selection, "
            "analysis, review, synthesis and useful peer clarification remain available. This policy "
            "takes precedence over retained retrieval habits, experience and generated role hints.")


def knowledge_policy_role_adaptation(pool_agent_id: str, knowledge_policy: str | None, *, prompt="") -> dict:
    knowledge_policy_prompt(knowledge_policy)
    if knowledge_policy is None or pool_agent_id != "searcher":
        return {}
    adapted_prompt = "Organize relevant model general knowledge with explicit assumptions and uncertainty."
    if prompt and prompt != "Retrieve and organize evidence with explicit provenance and uncertainty.":
        adapted_prompt += "\nRetained process advice, applicable only within the knowledge policy:\n" + prompt
    return {"role": "Knowledge Researcher",
            "capabilities": ["research", "evidence", "source comparison", "general knowledge"],
            "prompt": adapted_prompt,
            "preferred_tools": []}


class JsonModelCalls:
    """Each call is a fresh context; records contain only observable I/O."""

    def __init__(self, *, max_corrections: int = 1):
        if type(max_corrections) is not int or not 0 <= max_corrections <= 1:
            raise ValueError("At most one structured-response correction is allowed")
        self.max_corrections = max_corrections
        self.call_records: list[dict[str, Any]] = []
        self._records_lock = threading.Lock()

    @staticmethod
    def _validation_errors(exc: ValueError) -> list[dict]:
        if isinstance(exc, ValidationError):
            return [{"location": list(item["loc"]), "type": item["type"], "message": item["msg"]}
                    for item in exc.errors(include_input=False, include_url=False)]
        if isinstance(exc, json.JSONDecodeError):
            return [{"type": "json_decode", "message": exc.msg,
                     "line": exc.lineno, "column": exc.colno}]
        return [{"type": "contract", "message": str(exc)}]

    def ask(self, model: Callable, phase: str, instructions: str, payload: dict,
            schema: type[T], *, agent_id: str = "global",
            validate: Callable[[T], None] | None = None,
            refresh_payload: Callable[[], dict] | None = None,
            json_schema: dict[str, Any] | None = None) -> T:
        # Generation-time restrictions supplement, never replace, the record's
        # Pydantic and cross-field validation below.
        response_schema = copy.deepcopy(json_schema) if json_schema is not None else schema.model_json_schema()
        system = instructions + "\nReturn only one JSON object conforming to this JSON Schema:\n" \
            + json.dumps(response_schema)
        original_payload = {"phase": phase, "agent_id": agent_id, **as_json(payload)}
        correction = None
        for attempt in range(self.max_corrections + 1):
            request = copy.deepcopy(original_payload)
            if refresh_payload is not None:
                request.update(as_json(refresh_payload()))
            if correction is not None:
                request["response_correction"] = correction
            correction_system = (
                "\nThis is the sole contract correction turn. The response_correction field names "
                "the exact invalid cross-field assignments in your prior JSON. Fix those assignments "
                "in the returned object, then recheck every affected dependency and identity. "
                "Do not merely restate a correction in selection_rationale while retaining invalid fields. "
                "For a final-synthesizer DAG violation, final means a terminal node: no selected agent "
                "may depend on the final synthesizer, and all selected contributors must be its ancestors. "
                "If reviewers follow a writer, make a downstream selected role the final synthesizer "
                "and require its full deliverable; alternatively move reviews before the final writer. "
                "Never resolve it by adding an edge that makes a cycle."
                " For self-review, remove the named primary owner from reviewers for that rubric; "
                "choose a different selected downstream agent or omit the optional review. "
                "Do not change the primary owner merely to relabel a self-check as independent review."
                if correction is not None else "")
            corrected_system = (instructions + correction_system
                                + "\nReturn only one JSON object conforming to this JSON Schema:\n"
                                + json.dumps(response_schema)) if correction is not None else system
            messages = [{"role": "system", "content": corrected_system},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
            # Transport, authentication and budget failures are not output corrections.
            response = model(copy.deepcopy(messages))
            content = response if isinstance(response, str) else getattr(response, "content", None)
            record = {"phase": phase, "agent_id": agent_id, "attempt": attempt,
                      "messages": messages, "response": content}
            try:
                if not isinstance(content, str):
                    raise ValueError(f"{phase}: model response must contain JSON text")
                stripped = content.strip()
                if stripped.startswith("```") and stripped.endswith("```"):
                    stripped = "\n".join(stripped.splitlines()[1:-1])
                result = schema.model_validate(json.loads(stripped))
                if validate is not None:
                    validate(result)
            except ValueError as exc:
                errors = self._validation_errors(exc)
                record["validation_errors"] = errors
                if attempt == self.max_corrections:
                    raise
                correction = {
                    "instruction": "Your previous response failed the stated output contract. "
                    "Return one complete corrected JSON object using the original inputs and "
                    "these validation errors. Do not alter evidence, task constraints or budgets "
                    "to evade validation. This is the only correction attempt.",
                    "previous_response": content, "validation_errors": errors,
                }
            else:
                return result
            finally:
                with self._records_lock:
                    self.call_records.append(record)
        raise RuntimeError("Structured-response attempts unexpectedly exhausted")


PREDICT_PROMPT = """You organize an executable team for this public task before any answer
or evaluation exists. Infer task-specific, observable quality requirements, including implicit
requirements and prohibitions when justified. Importance is nonnegative planning priority;
confidence is uncertainty in the prediction, never a hidden evaluator weight. Distinguish
explicit, inferred, and experience-derived requirements. Link experience IDs when used.
Only a requirement actually stated in the public task may be labeled explicit. Predicted
rubrics are fallible planning hypotheses, not new instructions from the user. Do not turn
an unverified formula, algorithm choice, numerical threshold or stylistic preference into
a mandatory requirement. When such a choice needs checking, preserve its uncertainty in
confidence/applicability and ask for verification of the claim and its assumptions, not
obedience to the proposed answer. Independent review may challenge the predicted rubric
itself; factual correctness takes precedence over satisfying a mistaken prediction.
Propose capabilities and concrete responsibilities, not a permanent cast of named roles.
Prefer independent Analyst and Evidence contributions followed by one final Writer when
the task benefits from both. These are functions, not fixed role names or a fixed roster:
merge them into one agent for a simple task and retain real forward data dependencies.
Execution is single-pass: each selected agent receives one model call and publishes once.
Contributors publish a short answer and a structured ledger with requirements, outline,
evidence_spans and source_references; the Writer consumes the shared ledger and submits
the full deliverable. Plan contributors to supply compact, substantive facts, reasoning,
examples and uncertainty useful for synthesis, not competing full deliverables. A review
should identify consequential defects and supported corrections rather than rewrite the
whole answer. The final Writer must cover the public task, check consequential claims and
inferences, and use citations only when their details are supported or confidently known;
never guess an author, title, year, quotation or numerical result to appear well sourced.
External tool calls may be batched once by a contributor; their
results enter the ledger without another model turn. The Writer has tools=[]. Do not
plan send_message, read_evidence, raise_issue, debate, role revisits or iterative review.
One agent may cover several requirements, and several agents may contribute to one.
Use one agent when sufficient. Explain evidence needs and rubric relationships; a rubric
relationship is not automatically an execution dependency or a causal claim. Accepted
historical experience is conditional advice, not a source of task answers. Do not answer
the task. Choose only public tools and remain within the supplied limits. Candidate IDs
and rubric IDs must each be unique; every candidate rubric_id and every graph edge
endpoint must reference a rubric in this graph. Estimate each candidate's max_tokens
from its complete expected output, including JSON encoding and checkpoint overhead.
When limits.execution_max_tokens is supplied, every candidate.max_tokens must stay
within that per-response execution-model ceiling; a larger request cannot raise it.
Final synthesis must emit the full requested deliverable, even when its role is called
editor. Set every AgentSpec.max_calls=1 explicitly. There is no execution-time JSON
correction, second draft or communication round; fit a complete response in that call.
Responsibilities and checkpoints cannot require feedback from a later DAG role;
give any needed feedback incorporation to a downstream role instead."""

LOCAL_PLAN_PROMPT = """You are an independent candidate agent planning from your assigned
capability, the public task, and the initial quality graph. Inspect the draft critically.
Identify requirements you can own, needed inputs, concrete outputs, collaboration
dependencies, tools, resource needs, uncovered requirements, and risks. You may challenge
the global draft, add missed requirements, refine ambiguous ones, or recommend merging
redundant responsibilities. New requirements need new stable rubric IDs. Do not merely
confirm acceptance, impersonate other agents, or produce the final answer. Preserve your
assigned agent_id and capability. Your rubric_ids may reference only the initial graph
or your own additions. Use only public tools. Treat experience as conditional.
Copy immutable_identity.agent_id and immutable_identity.capability verbatim into your
response. These immutable fields are not summaries to rewrite. candidate.role is a
separate role label, not candidate.capability; never substitute the role label for the
capability, even when they describe similar responsibilities.
task.tools is the exact tool allowlist: when it is empty, tools must be []. Thinking,
writing, peer review and mathematical reasoning are capabilities, not callable tools.
depends_on must contain exact agent_id values from prediction.candidates, never your own
ID, role names, rubric IDs, artifact descriptions or prose. Describe artifacts in
required_inputs and expected_outputs. If an essential producer is missing, explain the
gap in uncovered/challenge instead of inventing a dependency ID.
Do not assume the global draft is factually correct. Challenge unsupported formulas,
algorithm choices, numerical targets and style restrictions, preserving uncertainty and
distinguishing explicit public instructions from inferred or experience-derived advice.
A review requires access to the actual draft or derivation, not just another reviewer's
verdict. Name the producer dependency and required artifact. State what would falsify a
suspect claim or reveal an omitted assumption; agreement between agents is not evidence.
Plan a compact contribution containing the facts, reasoning, examples and uncertainty
needed downstream, rather than a competing full deliverable or a source-status statement.
For review, identify consequential defects, explain their effect and give a specific
supported correction; do not plan a rewritten copy of the whole artifact. If you may
synthesize, plan to cover the public task and check consequential claims and inferences.
Use citations only when their details are supported or confidently known; never guess
an author, title, year, quotation or numerical result to fill an evidence gap.
Each selected role executes once in DAG order, with exactly one model call. A role cannot claim it will incorporate
feedback from its downstream reviewer later; propose a downstream synthesis responsibility
or a merge for reconciliation instead of an implicit second execution or backward edge.
In expected_outputs and risks, state the expected artifact length and whether the
candidate's output-token allocation can contain it plus valid JSON and checkpoints.
Use limits.execution_max_tokens, when present, as the actual output ceiling; requesting
more than the execution model supports cannot make a longer artifact fit.
If your role may synthesize the final answer, budget for the entire requested deliverable,
not merely editorial feedback. Set LocalPlan.max_calls=1; the reconciled AgentSpec.max_calls
must also be 1. Do not request correction calls or communication rounds. Contributors
publish a short answer and a ledger containing requirements, outline, evidence_spans and
source_references, all present even when empty. requirements and outline are string
lists; evidence_spans contain text and source_ref, and source_references contain source_id
and locator. Preserve real provenance and do not invent source observations.
Any external tool batch runs after that
single response and enters the shared ledger without a return call. The final Writer
uses tools=[] and consumes the shared ledger once. Plan only forward publication and
consumption, never send_message, read_evidence, raise_issue or a follow-up request.
local_rounds, when configured, belongs to pre-execution planning only."""

RECONCILE_PROMPT = """Reconcile the global draft with independent local plans for this task.
Consider every local addition, challenge, gap and resource request. Incorporate useful
discoveries, merge redundant responsibilities, resolve conflicting dependencies, and
record the selection rationale, including reasons for rejecting material local suggestions.
Resolve substantive challenges before preserving a predicted rubric: requirements not
stated by the public task remain fallible hypotheses. Do not enforce an unverified formula,
algorithm choice, numerical target or mandatory style merely because the global draft
predicted it. Preserve uncertainty in confidence/applicability and assign checks of the
claim and its assumptions. A reviewer may reject a mistaken rubric premise rather than
force the answer to conform to it. Conditional experience cannot override the public task.
Generate the revised graph and an executable TeamSpec. Preserve stable rubric IDs for
unchanged requirements. Coverage is many-to-many. The synthesizer reconciles conflicts
and gaps in the requested output genre, not just concatenates contributions; allocate
enough output tokens for synthesis and checks. Represent any merged responsibilities in the
agents array itself, not only in the selection rationale.
The synthesizer must independently emit the complete final deliverable requested by
the task, not just review notes, an editing preamble, or a pointer to an upstream draft.
Size its max_tokens for that final output plus JSON escaping, evidence IDs, checkpoints
and a margin for valid closure. max_tokens is a per-response output ceiling, not a
context allowance. Editing a full article still requires enough tokens to return the
full article; a genuinely short requested summary may need fewer tokens than its sources.
Budget a concise final response without duplicating drafts, review narration or preambles.
Give non-synthesizer roles compact, substantive material to produce: relevant facts,
reasoning, examples, tradeoffs and uncertainty, not competing full deliverables. A reviewer
should identify consequential defects with supported corrections, not rewrite the whole
artifact. Make these distinctions explicit in responsibilities and task_prompt. The
Writer must address all public task requirements and check consequential factual claims
and inferences against available evidence and assumptions. Do not require guessed citation
details: unsupported authors, titles, years, quotations and numerical results must not be
invented to give an appearance of evidence. Preserve useful content and honest uncertainty.
Set every AgentSpec.max_calls=1 explicitly. Execution provides exactly one model call
per selected role, with no JSON correction, second draft, debate or communication loop.
Unused team.total_max_calls is a ceiling, not permission to revisit a completed role.
Prefer task-conditioned Analyst/Evidence contributions in parallel, then a final Writer,
merging roles for simple tasks and retaining genuine forward data dependencies. The
synthesizer_id identifies that final Writer regardless of its role label; its tools=[].
Contributors publish a short answer plus a ledger with requirements, outline,
evidence_spans and source_references; one external tool batch may add raw evidence after
their response, without another model call. The Writer consumes the shared ledger once.
Do not allocate send_message, read_evidence, raise_issue or feedback request rounds.
The roster, dependencies, checkpoints and budgets should follow this task's requirements,
not a fixed workflow. Return only graph and team; the coordinator preserves the supplied
local plans separately, so do not echo them. Return no answer to the task and do not invent
evaluation feedback.

Before returning, check ALL cross-field constraints against the actual JSON you will emit:
- 1 <= len(team.agents) <= limits.max_agents. Agent IDs are unique. synthesizer_id
  references an existing selected agent; the synthesizer counts toward this limit. When
  the roster is full, assign synthesis to a selected agent instead of adding another.
- depends_on contains only selected agent IDs, with no self-dependency or cycle. A
  dependency means the downstream agent receives the upstream agent's public ledger contribution.
  Ensure the DAG provides the actual draft/derivation needed for review or synthesis,
  including its source producer as an ancestor; a reviewer's summary alone is insufficient
  to inspect a draft. Use required_inputs and expected_outputs to identify these artifacts.
- The synthesizer depends transitively on every other selected agent (or has no
  dependencies when it is the only agent). Rubric graph relationships are not this DAG.
  Each selected agent executes once in DAG order. A writer cannot synthesize before
  downstream reviewers run and then implicitly run again. Select an existing downstream
  agent for final synthesis and connect all contributors without cycles, or merge roles.
  Responsibilities and checkpoints may refer only to information available at that role's
  turn. An upstream writer cannot truthfully check that it incorporated future reviewer
  feedback. Assign incorporation to a downstream role; do not claim an impossible check.
- Rubric IDs are unique and every graph edge endpoint exists in graph.rubrics. Keys of
  coverage, primary and reviewers must be rubric IDs in this graph; their values must
  reference selected agent IDs. Every rubric has nonempty coverage and a primary owner
  in coverage[rubric_id]. Each agent.rubric_ids is exactly the set of rubric IDs whose
  coverage includes that agent.
  After finalizing coverage, mechanically derive every agent.rubric_ids as the sorted,
  deduplicated list of rid values for which agent.agent_id is in coverage[rid]. Recompute
  it after every merge or reassignment; do not copy an outdated candidate assignment.
- For each reviewer in reviewers[rubric_id], reviewer != primary[rubric_id], and that
  primary owner must be an ancestor of the reviewer through depends_on, directly or
  transitively. An upstream agent cannot review a downstream owner's future artifact.
  Do not add backward review dependencies that create cycles. reviewers is optional:
  omit a rubric or use an empty list when no independent downstream reviewer is feasible;
  put self-checks in checkpoints instead of claiming independent review. A reviewer is
  a single-pass consumer of published evidence, never an interactive feedback loop.
  SELF-REVIEW IS FORBIDDEN: compare the actual agent IDs in primary and reviewers for
  every rubric before returning; a different role label does not make the same ID
  independent. Omit the optional reviewer assignment when only self-checks are feasible.
- sum(agent.max_calls) <= team.total_max_calls <= limits.total_max_calls, and
  team.max_parallel <= limits.max_parallel. Every agent.max_calls=1. All agent tools
  must be in task.tools; the final synthesizer must have tools=[].
- If limits.execution_max_tokens is present, every agent.max_tokens must be <= that
  actual execution-model output ceiling, including the synthesizer. Fit the requested
  deliverable within this bound rather than claiming an unsupported larger allowance.
- For each agent's budget estimate, calculate and check the hard arithmetic constraint:
  expected_output_tokens <= agent.max_tokens * expected_model_calls. expected_output_tokens
  is the aggregate across the estimated turns, while agent.max_tokens is per response.
  This constraint still applies when agent.max_calls is null; null removes the optional
  call ceiling, not the per-response output ceiling. An iterative estimate may legitimately
  include multiple model calls. Keep expected calls within any finite role/team ceilings
  and the estimated input/output totals plus reserves within the remaining shared budget."""

POOL_ORGANIZATION_PROMPT = """\nEVOLVING AGENT POOL: Select reusable agents from agent_pool_catalogue.
Every selected candidate must carry the exact pool_agent_id and pool_agent_version of
one available pool member; copy its catalogue.version into pool_agent_version.
agent_id names its participation in this task; the pool
identity persists across tasks. Select and combine the smallest suitable roster. The
same pool member may participate only once. You control the task goals, responsibilities,
rubric assignments, communication topology, resource ceilings, and a short task_prompt.
Different agent_id values do not create additional copies of a pool member. If two
responsibilities need the same member, combine them into one participation; otherwise
choose distinct available pool members with suitable capabilities. A duplicate-binding
correction names every duplicate pool identity and its participating agent_id values.
Keep task_prompt limited to public task constraints and local role goals. Preserve the
member's established internal capabilities. Its local plan chooses skills, reasoning,
memory use, harness, and communication habits. Reconciliation may select or remove
candidates but cannot replace a selected member's pool identity or internal choices.
At reconciliation, team.agents must be a subset of prediction.candidates using the
exact original agent_id, pool_agent_id and pool_agent_version triples. Do not add a
new writer or rename a candidate to fill a gap; merge responsibilities into an existing
candidate and make an existing selected candidate the final synthesizer.
Reuse mature members rather than inventing a fresh role implementation for each task."""

POOL_LOCAL_PROMPT = """\nYou are the persistent agent described by agent_profile, adapting to this task.
Use your retained role instructions, skills, memory, reasoning strategy, harness, and
communication experience to choose how to perform your role within public constraints.
Return your selected_skills, reasoning_strategy, harness, and communication in the local
plan. These are your internal choices, not instructions to redesign the whole team.
Use selected_skills=null to retain your skill library or [] to use no retained skills.
Respect current tool allowlists and budgets. Retained memory is conditional experience,
not task evidence or permission to override the public task. Keep task_prompt short and
limited to this task's role goals; preserve your pool identity and version."""


BUDGET_AWARE_PROMPT = """\nBUDGET-AWARE ORGANIZATION: Optimize task quality together with total token
cost, not team size or call count alone. limits.resource_budget reports current shared
usage, pending reservations, remaining tokens/calls/tools and time. Planning, harness
generation, execution, evaluation and learning share these resources. Recheck the live
remaining budget after each planning step. Compare merging roles, mature-agent reuse,
independent checks and additional tool or peer turns by their expected quality benefit
and their input plus output token cost. Repeated context, tool observations, private
history and shared-ledger reads all contribute input tokens; communication bytes are a
separate diagnostic and must not be charged as extra tokens twice. Keep published
artifacts concise while preserving source provenance and the full final deliverable.
For reconciliation, emit team.budget_plan with one estimate per selected agent:
expected_model_calls, expected_input_tokens and expected_output_tokens are whole-task
execution estimates, including expected follow-up turns and reread context, not just
one response. For every role, calculate expected_output_tokens <= agent.max_tokens *
expected_model_calls before returning. The left side includes output from all estimated
turns; the right side is their combined response capacity. A null agent.max_calls does
not waive this arithmetic and does not force a one-call estimate. Multiple estimated
iterative calls remain legitimate within shared budgets and any finite call ceilings.
expected_tool_calls counts external tools; expected_communication_bytes
accounts for public artifacts and peer traffic. Include each role's cost rationale,
quality_cost_tradeoff and stopping_policy, and reserve future tokens/model calls for
remaining harness generation, evaluation and learning. Total estimated execution input
plus output tokens and future reserves must fit the actual remaining tokens; expected
calls and tools must fit remaining finite ceilings. A null call ceiling removes only
that ceiling, not the obligation to estimate cost. Estimates are not fixed iteration
caps. Stop when the deliverable is complete or further work has insufficient expected
quality benefit for its token cost, while respecting enforced budgets and preserving
the final response. Do not claim savings or numerical quality gains without evidence."""


class GlobalAnalyzer(JsonModelCalls):
    def __init__(self, global_model: Callable,
                 local_model_factory: Callable[[str], Callable] | None = None, *,
                 max_agents: int = 4, max_parallel: int = 2, local_rounds: int = 1,
                 total_max_calls: int | None = 16, explicit_rubrics: bool = True,
                 max_corrections: int = 1, execution_max_tokens: int | None = None,
                 execution_mode: str = "single_pass", agent_pool: AgentPoolSnapshot | None = None,
                 budget_context: Callable[[], dict] | None = None,
                 excluded_task_ids: Sequence[str] = (), knowledge_policy: str | None = None):
        super().__init__(max_corrections=max_corrections)
        if not 1 <= local_rounds <= 3 or max_agents < 1 or max_parallel < 1:
            raise ValueError("Planning requires positive limits and one to three local rounds")
        if execution_max_tokens is not None and (type(execution_max_tokens) is not int or execution_max_tokens < 1):
            raise ValueError("execution_max_tokens must be a positive integer or None")
        self.global_model = global_model
        self.local_model_factory = local_model_factory or (lambda _agent_id: global_model)
        self.max_agents = max_agents
        self.max_parallel = max_parallel
        self.local_rounds = local_rounds
        self.total_max_calls = total_max_calls
        self.explicit_rubrics = explicit_rubrics
        self.execution_max_tokens = execution_max_tokens
        if execution_mode not in {"single_pass", "iterative_shared_ledger"}:
            raise ValueError("unknown execution_mode")
        self.execution_mode = execution_mode
        self.agent_pool = agent_pool
        knowledge_policy_prompt(knowledge_policy)
        self.knowledge_policy = knowledge_policy
        self.budget_context = budget_context
        self.excluded_task_ids = set(excluded_task_ids)
        self.last_prediction: Prediction | None = None

    def _pool_catalogue(self):
        from .agent_pool import catalogue
        members = catalogue(self.agent_pool)
        for member in members:
            adaptation = knowledge_policy_role_adaptation(member["pool_agent_id"], self.knowledge_policy)
            member.update({key: value for key, value in adaptation.items() if key in member})
        return members

    def _agent_profile(self, candidate):
        if self.agent_pool is None:
            return None
        from .agent_pool import get_profile
        if not candidate.pool_agent_id or candidate.pool_agent_version is None:
            raise ValueError("Pooled candidates must select an exact persistent identity and version")
        profile = get_profile(self.agent_pool, candidate.pool_agent_id, candidate.pool_agent_version)
        return profile.model_copy(update=knowledge_policy_role_adaptation(
            profile.pool_agent_id, self.knowledge_policy, prompt=profile.prompt), deep=True)

    def _pool_bindings(self, agents):
        if self.agent_pool is None:
            return
        participants = {}
        for agent in agents:
            pool_id = self._agent_profile(agent).pool_agent_id
            participants.setdefault(pool_id, []).append(agent.agent_id)
        duplicates = {pool_id: agent_ids for pool_id, agent_ids in participants.items()
                      if len(agent_ids) > 1}
        if duplicates:
            raise ValueError("A pool member may participate only once in a task: "
                             f"duplicate_pool_bindings={json.dumps(duplicates, sort_keys=True)}. "
                             "Changing agent_id does not create another pool member. Merge or remove "
                             "duplicate participations and update their dependencies and rubric "
                             "assignments; prediction may instead select a distinct available "
                             "catalogue member. Reconciliation must retain each selected candidate's "
                             "original pool identity and version.")

    def _limits(self) -> dict:
        limits = {"max_agents": self.max_agents, "max_parallel": self.max_parallel,
                  "total_max_calls": self.total_max_calls, "execution_mode": self.execution_mode}
        if self.knowledge_policy is not None:
            limits["knowledge_policy"] = self.knowledge_policy
        if self.execution_max_tokens is not None:
            limits["execution_max_tokens"] = self.execution_max_tokens
        if self.budget_context is not None:
            limits["resource_budget"] = self._resource_budget()
        return limits

    def _resource_budget(self) -> dict:
        if self.budget_context is None:
            return {}
        state = as_json(self.budget_context())
        if not isinstance(state, dict):
            raise ValueError("budget_context must return a resource budget object")
        resource = {}
        for suffix, maximum_keys, used_keys in (
                ("tokens", ("max_total_tokens", "max_tokens"), ("used_tokens", "tokens")),
                ("model_calls", ("max_model_calls", "max_calls"), ("used_model_calls", "model_calls")),
                ("tool_calls", ("max_tool_calls",), ("used_tool_calls", "tool_calls"))):
            maximum = next((state[key] for key in maximum_keys if key in state), None)
            used = next((state[key] for key in used_keys if key in state), 0)
            reserved = state.get("reserved_tokens", 0) if suffix == "tokens" else 0
            resource["max_" + suffix] = maximum
            resource["used_" + suffix] = used
            if suffix == "tokens":
                resource["reserved_tokens"] = reserved
            resource["remaining_" + suffix] = (
                max(0, maximum - used - reserved) if maximum is not None
                else state.get("remaining_" + suffix))
        resource["communication_bytes"] = state.get("communication_bytes", sum(
            group.get("communication_bytes", 0) for group in state.get("by_stage", {}).values()))
        resource["remaining_seconds"] = state.get("remaining_seconds")
        return resource

    def _refresh_limits(self) -> dict:
        return {"limits": self._limits()}

    def _prompt(self, prompt: str) -> str:
        prompt += BUDGET_AWARE_PROMPT
        if self.execution_mode == "iterative_shared_ledger":
            # The planning constants also document the historical single-pass
            # control. Remove those prohibitions before adding the iterative
            # contract, so a model cannot receive two incompatible schedules.
            for old, new in (
                ("Execution is single-pass: each selected agent receives one model call and publishes once.",
                 "Execution may be iterative: each selected agent can receive additional model turns and revise a published artifact."),
                ("Each selected role executes once in DAG order, with exactly one model call. A role cannot claim it will incorporate\nfeedback from its downstream reviewer later; propose a downstream synthesis responsibility\nor a merge for reconciliation instead of an implicit second execution or backward edge.",
                 "The dependency DAG provides initial inputs; roles may later resume to answer public peer requests and incorporate new ledger evidence while preserving private history."),
                ("Set every AgentSpec.max_calls=1 explicitly. There is no execution-time JSON\ncorrection, second draft or communication round; fit a complete response in that call.",
                 "Set AgentSpec.max_calls to the configured ceiling, or null when the iterative task removes that optional ceiling. Token and timeout budgets remain binding."),
                ("Set LocalPlan.max_calls=1; the reconciled AgentSpec.max_calls\nmust also be 1. Do not request correction calls or communication rounds.",
                 "Set LocalPlan.max_calls to the configured iterative ceiling, or null when no role ceiling is requested. Peer communication and follow-up turns are allowed."),
                ("Each selected agent executes once in DAG order. A writer cannot synthesize before\ndownstream reviewers run and then implicitly run again. Select an existing downstream\nagent for final synthesis and connect all contributors without cycles, or merge roles.\nResponsibilities and checkpoints may refer only to information available at that role's\nturn. An upstream writer cannot truthfully check that it incorporated future reviewer\nfeedback. Assign incorporation to a downstream role; do not claim an impossible check.",
                 "The DAG orders initial availability, while cooperative scheduling may resume a role after peer requests or revised artifacts. Final synthesis waits for pending peer requests and terminal dependency outcomes."),
                ("A reviewer is\na single-pass consumer of published evidence, never an interactive feedback loop.",
                 "A reviewer may be resumed for public clarification or revision; private role history remains scoped to that role."),
                ("The Writer consumes the shared ledger once.",
                 "The Writer consumes ledger updates as they arrive and may revise before final submission."),
                ("Do not allocate send_message, read_evidence, raise_issue or feedback request rounds.",
                 "Allocate send_message only when public peer clarification is useful; it reactivates the addressed role without a fixed round count."),
                ("Any external tool batch runs after that\nsingle response and enters the shared ledger without another model call.",
                 "External tool results enter the shared ledger and may be followed by additional role turns."),
                ("one external tool batch may add raw evidence after their response, without another model call.",
                 "External tools may be requested across iterative turns within the configured tool and timeout budgets."),
                ("no task-internal negotiation or clarification.",
                 "task-internal peer clarification is allowed through the auditable shared ledger."),
                ("Do not plan send_message, read_evidence, raise_issue or a follow-up request.",
                 "Plan send_message only where peer clarification is task-relevant; follow-up requests must remain public and auditable."),
            ):
                prompt = prompt.replace(old, new)
            prompt += ("\nEXECUTION MODE: iterative_shared_ledger. Roles may revisit their work and "
                       "communicate through an auditable shared ledger. Contributors may request "
                       "allowed tools, receive tool results, and continue; the Writer may use tools "
                       "and revise its draft. Emit max_calls=null when no role ceiling is requested. "
                       "The team's total_max_calls must respect limits.total_max_calls; emit null "
                       "only when that configured ceiling is null. Configured call/tool ceilings, "
                       "token and timeout budgets remain binding. Plan termination on "
                       "completion, convergence, or those enforced budgets, never on a fixed round count.")
            prompt += ("\nITERATIVE OVERRIDE (takes precedence over any historical single-pass wording above): "
                       "Do not restrict a role to one call, one publication, one tool batch or one ledger read. "
                       "The cooperative scheduler may reactivate completed roles for public peer requests, "
                       "updated artifacts and clarification. Preserve each role's private history; expose only "
                       "published ledger events. The synthesizer must wait for pending requests and failed "
                       "dependencies, then submit one terminal answer. Do not introduce a fixed round count; "
                       "configured role/team call ceilings, token budgets and timeout remain binding.")
            # Normalize residual historical phrases whose source strings are
            # assembled from adjacent literals (and therefore contain spaces,
            # not source-line newlines).
            for old, new in (
                ("single-pass", "one-way historical"),
                ("exactly one model call", "one or more model calls"),
                ("max_calls=1", "a finite max_calls ceiling"),
                ("single-pass consumer", "forward consumer"),
                ("without another model call", "with additional role turns allowed"),
                ("never send_message", "send_message only when peer clarification is useful"),
                ("Do not allocate send_message", "Allocate send_message only when peer clarification is useful"),
                ("shared ledger once", "shared ledger as updates arrive"),
                ("consumes the shared ledger once", "consumes ledger updates before final submission"),
                ("no task-internal negotiation or clarification", "task-internal peer clarification is allowed"),
                ("no task-internal negotiation", "task-internal peer clarification is allowed"),
                ("The Writer has tools=[]", "The Writer may use allowed task tools"),
                ("The final Writer uses tools=[]", "The final Writer may use allowed task tools"),
                ("the final Writer uses tools=[]", "the final Writer may use allowed task tools"),
                ("the final synthesizer must have tools=[]", "the final synthesizer's tools must be within task.tools"),
                ("the synthesizer must have tools=[]", "the synthesizer's tools must be within task.tools"),
                ("its tools=[]", "its tools must be within task.tools"),
                ("External tool calls may be batched once by a contributor; their results enter the ledger without another model turn.",
                 "Allowed roles may request external tools across turns; results enter the ledger and can trigger another model turn."),
                ("Any external tool batch runs after that single response and enters the shared ledger without a return call.",
                 "External tool requests may span turns; results enter the shared ledger before later role turns."),
                ("one external tool batch may add raw evidence after their response, with additional role turns allowed.",
                 "External tools may add raw evidence across role turns."),
                ("Do not plan send_message, read_evidence, raise_issue, debate, role revisits or iterative review.",
                 "Plan public send_message clarification, role revisits and iterative review when task-relevant."),
                ("Plan only forward publication and consumption", "Plan public publication, clarification and revision"),
                ("External tool calls may be batched once by a contributor; their", "Allowed roles may request external tools across turns; their"),
                ("results enter the ledger without another model turn.", "results enter the ledger and can trigger another model turn."),
                ("Any external tool batch runs after that", "External tool requests may span turns and"),
                ("single response and enters the shared ledger without a return call.", "multiple turns and enter the shared ledger before later turns."),
                ("one external tool batch may add raw evidence after", "External tools may add raw evidence across turns after"),
                ("their response, with additional role turns allowed.", "their latest turn."),
                ("uses tools=[]", "may use allowed task tools"),
                ("Do not\nplan send_message", "Plan public send_message"),
                ("Plan only forward publication", "Plan public publication, clarification and revision"),
            ):
                prompt = prompt.replace(old, new)
        if self.explicit_rubrics:
            return prompt + knowledge_policy_prompt(self.knowledge_policy)
        return prompt + "\nABLATION: Do not explicitly predict rubrics. Return graph rubrics=[] " \
            "and edges=[], all rubric_ids=[] and additions=[], and coverage/primary/reviewers={}. " \
            "Still derive concrete responsibilities, dependencies, tools, checks and budgets " \
            "from the public task and applicable experience." + knowledge_policy_prompt(self.knowledge_policy)

    def predict(self, task: PublicTask, experiences: Sequence = ()) -> Prediction:
        task = PublicTask.model_validate(task)
        prompt = PREDICT_PROMPT + (POOL_ORGANIZATION_PROMPT if self.agent_pool is not None else "")
        prediction = self.ask(self.global_model, "predict", self._prompt(prompt),
                              {"task": task, "experiences": experiences,
                               **({"agent_pool_catalogue": self._pool_catalogue()} if self.agent_pool is not None else {}),
                               "limits": self._limits()}, Prediction,
                              validate=lambda item: self._validate_prediction(task, item),
                              refresh_payload=self._refresh_limits)
        self.last_prediction = prediction.model_copy(deep=True)
        return prediction

    def _validate_prediction(self, task: PublicTask, prediction: Prediction) -> None:
        if not self.explicit_rubrics and (prediction.graph.rubrics or prediction.graph.edges):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        ids = [agent.agent_id for agent in prediction.candidates]
        if len(ids) > self.max_agents or len(ids) != len(set(ids)):
            raise ValueError("Candidate count exceeds limit or contains duplicate IDs")
        remaining_calls = self._resource_budget().get("remaining_model_calls")
        if remaining_calls is not None and len(ids) > remaining_calls:
            raise ValueError("Candidate initial calls exceed remaining shared model-call budget")
        rubric_ids = {r.rubric_id for r in prediction.graph.rubrics}
        self._pool_bindings(prediction.candidates)
        for agent in prediction.candidates:
            self._check_agent(task, agent, rubric_ids)
            if self.agent_pool is not None:
                self._set_internal_policy(agent)

    def _set_internal_policy(self, agent, plan=None):
        profile = self._agent_profile(agent)
        for field, fallback in (("selected_skills", list(profile.skills)),
                                ("reasoning_strategy", profile.reasoning_strategy),
                                ("harness", profile.harness), ("communication", profile.communication)):
            value = getattr(plan, field) if plan is not None else None
            selected = fallback if value is None or (field != "selected_skills" and not value) else value
            setattr(agent, field, copy.deepcopy(selected))
        if plan is not None:
            agent.tools = list(plan.tools)

    def _check_agent(self, task: PublicTask, agent: AgentSpec, rubric_ids: set[str]):
        if not set(agent.tools) <= set(task.tools):
            raise ValueError(f"Agent {agent.agent_id} requested unavailable tools: "
                             f"requested_tools={agent.tools}; "
                             f"unavailable_tools={sorted(set(agent.tools) - set(task.tools))}; "
                             f"allowed_tools={sorted(set(task.tools))}. Use exact callable tool "
                             "names from task.tools; an empty allowlist requires tools=[]. "
                             "Reasoning, writing and review are capabilities, not tools.")
        if not set(agent.rubric_ids) <= rubric_ids:
            raise ValueError(f"Agent {agent.agent_id} references unknown rubrics")
        if self.execution_mode == "single_pass" and agent.max_calls != 1:
            raise ValueError(f"Agent {agent.agent_id}.max_calls={agent.max_calls}; "
                             "single-pass execution requires AgentSpec.max_calls=1 explicitly. "
                             "Do not allocate execution corrections or communication rounds.")
        if self.execution_max_tokens is not None and agent.max_tokens > self.execution_max_tokens:
            raise ValueError(f"Agent {agent.agent_id}.max_tokens={agent.max_tokens} exceeds "
                             f"limits.execution_max_tokens={self.execution_max_tokens}; "
                             "fit its complete output and JSON overhead within the actual ceiling")
        remaining_tokens = self._resource_budget().get("remaining_tokens")
        if remaining_tokens is not None and agent.max_tokens > remaining_tokens:
            raise ValueError(f"Agent {agent.agent_id}.max_tokens exceeds remaining shared token budget")
        if agent.tools and self._resource_budget().get("remaining_tool_calls") == 0:
            raise ValueError(f"Agent {agent.agent_id} requested tools with no remaining shared tool budget")

    def local_plan(self, task: PublicTask, prediction: Prediction, candidate: AgentSpec,
                   experiences: Sequence = ()) -> LocalPlan:
        local_experiences = []
        for experience in as_json(experiences):
            if experience_applicability(experience, task, capability=candidate.capability)["matched"]:
                local_experiences.append(experience)
        profile = self._agent_profile(candidate)
        if profile is not None:
            profile.memory = [lesson for lesson in profile.memory
                              if not (self.excluded_task_ids | {task.task_id}).intersection(lesson.source_task_ids)
                              and experience_applicability({**lesson.model_dump(mode="json"), "bank": "execution",
                                                            "task_signals": lesson.task_signals or [lesson.applicability]},
                                                           task, capability=candidate.capability)["matched"]]
        prompt = LOCAL_PLAN_PROMPT + (POOL_LOCAL_PROMPT if profile is not None else "")
        return self.ask(self.local_model_factory(candidate.agent_id), "local_plan",
                        self._prompt(prompt),
                        {"task": task, "prediction": prediction, "candidate": candidate,
                         **({"agent_profile": profile} if profile is not None else {}),
                         "immutable_identity": {"agent_id": candidate.agent_id,
                                                "capability": candidate.capability},
                         "experiences": local_experiences, "limits": self._limits()},
                        LocalPlan, agent_id=candidate.agent_id,
                        validate=lambda item: self._validate_local_plan(task, prediction, candidate, item),
                        refresh_payload=self._refresh_limits)

    def _validate_local_plan(self, task: PublicTask, prediction: Prediction,
                             candidate: AgentSpec, plan: LocalPlan) -> None:
        if plan.agent_id != candidate.agent_id or plan.capability != candidate.capability:
            expected = {"agent_id": candidate.agent_id, "capability": candidate.capability}
            actual = {"agent_id": plan.agent_id, "capability": plan.capability}
            raise ValueError("Local plan changed its agent identity or capability: "
                             f"expected_identity={json.dumps(expected, ensure_ascii=False)}; "
                             f"actual_identity={json.dumps(actual, ensure_ascii=False)}. "
                             "Copy both expected_identity fields verbatim into the corrected JSON. "
                             "candidate.role is not candidate.capability; do not substitute the "
                             "role label or paraphrase the capability.")
        violations = []
        profile = self._agent_profile(candidate)
        if profile is not None and not set(plan.selected_skills or []) <= set(profile.skills):
            violations.append("Local selected_skills must reference retained agent_profile.skills")
        if self.execution_mode == "single_pass" and plan.max_calls != 1:
            violations.append(f"LocalPlan.max_calls={plan.max_calls}; single-pass execution "
                              "requires LocalPlan.max_calls=1 explicitly. No execution correction "
                              "or communication round is available.")
        unavailable = sorted(set(plan.tools) - set(task.tools))
        if plan.tools and self._resource_budget().get("remaining_tool_calls") == 0:
            violations.append("Local plan requested tools with no remaining shared tool budget")
        if unavailable:
            violations.append("Local plan requested unavailable tools: "
                              f"unavailable_tools={unavailable}; allowed_tools={sorted(set(task.tools))}. "
                              "Use exact callable tool names from task.tools; an empty allowlist "
                              "requires tools=[]. Reasoning, writing and review are capabilities, "
                              "not tools.")
        candidate_ids = {agent.agent_id for agent in prediction.candidates}
        invalid_dependencies = sorted(set(plan.depends_on) - candidate_ids)
        self_dependency = plan.agent_id in plan.depends_on
        duplicate_dependencies = sorted({aid for aid in plan.depends_on
                                         if plan.depends_on.count(aid) > 1})
        if invalid_dependencies or self_dependency or duplicate_dependencies:
            violations.append("Local plan contains invalid execution dependencies: "
                              f"unknown_dependency_ids={invalid_dependencies}; "
                              f"self_dependency={self_dependency}; "
                              f"duplicate_dependency_ids={duplicate_dependencies}; "
                              f"allowed_dependency_ids={sorted(candidate_ids - {plan.agent_id})}. "
                              "depends_on contains only exact IDs of other prediction.candidates; "
                              "describe needed artifacts in required_inputs and missing producers "
                              "in uncovered/challenge, not as dependency IDs.")
        if not self.explicit_rubrics and (plan.rubric_ids or plan.additions):
            violations.append("Explicit rubrics are disabled for this ablation")
        known = {r.rubric_id for r in prediction.graph.rubrics}
        added = [r.rubric_id for r in plan.additions]
        if len(added) != len(set(added)) or known.intersection(added):
            violations.append("Local additions must use unique, new rubric IDs")
        if not set(plan.rubric_ids) <= known | set(added):
            violations.append("Local plan references unknown rubrics")
        if violations:
            raise ValueError("; ".join(violations))

    def reconcile(self, task: PublicTask, prediction: Prediction,
                  plans: Sequence[LocalPlan], experiences: Sequence = ()) -> PlannedTeam:
        prompt = RECONCILE_PROMPT + (POOL_ORGANIZATION_PROMPT if self.agent_pool is not None else "")
        response = self.ask(self.global_model, "reconcile", self._prompt(prompt),
                            {"task": task, "prediction": prediction, "local_plans": plans,
                             **({"agent_pool_catalogue": self._pool_catalogue()} if self.agent_pool is not None else {}),
                             "experiences": experiences, "limits": self._limits()},
                            _ReconciliationResponse,
                            validate=lambda item: self._validate_reconciled_pool(task, prediction, plans, item),
                            refresh_payload=self._refresh_limits,
                            json_schema=_pooled_reconciliation_schema(prediction, max_agents=self.max_agents)
                            if self.agent_pool is not None else None)
        # Local testimony belongs to its author, not the reconciler.
        result = PlannedTeam(graph=response.graph, team=response.team,
                             local_plans=[plan.model_copy(deep=True) for plan in plans])
        return result

    def _validate_reconciled_pool(self, task, prediction, plans, result):
        if self.agent_pool is None:
            self._validate_reconciliation(task, result)
            return
        candidates = {agent.agent_id: agent for agent in prediction.candidates}
        by_id = {plan.agent_id: plan for plan in plans}
        invalid_bindings = []
        for agent in result.team.agents:
            candidate = candidates.get(agent.agent_id)
            if candidate is None or (agent.pool_agent_id, agent.pool_agent_version) != (
                    candidate.pool_agent_id, candidate.pool_agent_version):
                invalid_bindings.append({"actual": {"agent_id": agent.agent_id,
                    "pool_agent_id": agent.pool_agent_id, "pool_agent_version": agent.pool_agent_version},
                    "expected": None if candidate is None else {"agent_id": candidate.agent_id,
                        "pool_agent_id": candidate.pool_agent_id,
                        "pool_agent_version": candidate.pool_agent_version}})
        if invalid_bindings:
            allowed = [{"agent_id": agent.agent_id, "pool_agent_id": agent.pool_agent_id,
                        "pool_agent_version": agent.pool_agent_version} for agent in prediction.candidates]
            raise ValueError("Reconciliation cannot invent or replace a candidate's pool identity: "
                             f"invalid_bindings={json.dumps(invalid_bindings, sort_keys=True)}; "
                             f"allowed_candidate_bindings={json.dumps(allowed, sort_keys=True)}. "
                             "Select a subset of these exact triples. Merge any missing responsibility "
                             "into an existing candidate; do not add, rename or rebind a role. "
                             "Update dependencies, synthesizer_id and all rubric assignments to "
                             "the selected existing IDs.")
        self._pool_bindings(result.team.agents)
        for agent in result.team.agents:
            self._set_internal_policy(agent, by_id.get(agent.agent_id))
        self._validate_reconciliation(task, result)

    def _validate_reconciliation(self, task: PublicTask, result: _ReconciliationResponse) -> None:
        team = result.team
        if not self.explicit_rubrics and (result.graph.rubrics or result.graph.edges
                                         or team.coverage or team.primary or team.reviewers):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        if (team.execution_mode != self.execution_mode or len(team.agents) > self.max_agents
                or team.max_parallel > self.max_parallel
                or (self.total_max_calls is not None
                    and (team.total_max_calls is None or team.total_max_calls > self.total_max_calls))):
            raise ValueError("Reconciled team exceeds configured resource limits: "
                             f"agents={len(team.agents)} <= {self.max_agents}, "
                             f"max_parallel={team.max_parallel} <= {self.max_parallel}, "
                             f"total_max_calls={team.total_max_calls} <= {self.total_max_calls} required")
        known = {r.rubric_id for r in result.graph.rubrics}
        if not set(team.coverage) <= known or not set(team.reviewers) <= known:
            raise ValueError("Team coverage references unknown rubrics")
        violations = []
        for agent in team.agents:
            try:
                self._check_agent(task, agent, known)
            except ValueError as exc:
                violations.append(str(exc))
            if self.execution_mode == "single_pass" and agent.agent_id == team.synthesizer_id and agent.tools:
                violations.append(f"Final Writer {agent.agent_id} must have tools=[]; "
                                  "move any external tool batch to an upstream contributor.")
            assigned = {rid for rid, owners in team.coverage.items() if agent.agent_id in owners}
            if assigned != set(agent.rubric_ids):
                raise ValueError("Agent assignments disagree with rubric coverage: "
                                 f"{agent.agent_id}.rubric_ids must be {sorted(assigned)}")
        if any(not team.coverage.get(rid) or rid not in team.primary for rid in known):
            raise ValueError("Every planned rubric requires coverage and a primary owner")
        dependencies = {agent.agent_id: set(agent.depends_on) for agent in team.agents}
        ancestors = set(dependencies[team.synthesizer_id])
        previous: set[str] = set()
        while ancestors != previous:
            previous = set(ancestors)
            ancestors.update(dep for aid in previous for dep in dependencies[aid])
        if ancestors != set(dependencies) - {team.synthesizer_id}:
            missing = sorted(set(dependencies) - {team.synthesizer_id} - ancestors)
            edges = sorted((upstream, downstream) for downstream, upstreams in dependencies.items()
                           for upstream in upstreams)
            upstreams = {upstream for upstream, _ in edges}
            terminals = sorted(set(dependencies) - upstreams)
            violations.append("Final synthesizer must depend on every contributing agent: "
                             f"synthesizer_id={team.synthesizer_id!r}; ancestors={sorted(ancestors)}; "
                             f"missing_contributor_ids={missing}; "
                             f"edges_upstream_to_downstream={edges}; terminal_candidates={terminals}. "
                             "Each agent executes once in DAG order. Select an existing downstream "
                             "agent and connect every missing contributor without a cycle, or merge "
                             "responsibilities and remove redundant agents. Changing only the "
                             "synthesizer_id may be insufficient; never add backward edges to an "
                             "upstream writer that already feeds its reviewers.")
        if violations:
            raise ValueError("; ".join(violations))
        self._validate_budget_plan(team)

    def _validate_budget_plan(self, team: TeamSpec) -> None:
        if self.budget_context is not None and team.budget_plan is None:
            raise ValueError("Budget-aware reconciliation requires an explicit team.budget_plan")
        if team.budget_plan is None:
            return
        resource = self._resource_budget()
        totals = team.budget_plan.totals()
        tokens = totals["input_tokens"] + totals["output_tokens"] + team.budget_plan.reserved_future_tokens
        calls = totals["model_calls"] + team.budget_plan.reserved_future_model_calls
        violations = []
        for name, requested in (("tokens", tokens), ("model_calls", calls),
                                ("tool_calls", totals["tool_calls"])):
            remaining = resource.get("remaining_" + name)
            if remaining is not None and requested > remaining:
                violations.append(f"Planned {name}={requested} exceeds remaining shared {name}={remaining}")
        if violations:
            raise ValueError("; ".join(violations))

    def build(self, task: PublicTask, experiences: Sequence = (), *,
              local_planning: bool = True) -> PlannedTeam:
        prediction = self.predict(task, experiences)
        initial = prediction.model_copy(deep=True)
        result = None
        for _round in range(self.local_rounds if local_planning else 1):
            plans = []
            if local_planning:
                with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                    futures = [pool.submit(self.local_plan, task, prediction, candidate,
                                           experiences) for candidate in prediction.candidates]
                    plans = [future.result() for future in futures]
            result = self.reconcile(task, prediction, plans, experiences)
            prediction = Prediction(graph=result.graph, candidates=result.team.agents)
        self.last_prediction = initial
        return result
