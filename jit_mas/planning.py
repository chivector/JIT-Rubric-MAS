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


def as_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value


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
            validate: Callable[[T], None] | None = None) -> T:
        system = instructions + "\nReturn only one JSON object conforming to this JSON Schema:\n" \
            + json.dumps(schema.model_json_schema())
        original_payload = {"phase": phase, "agent_id": agent_id, **as_json(payload)}
        correction = None
        for attempt in range(self.max_corrections + 1):
            request = copy.deepcopy(original_payload)
            if correction is not None:
                request["response_correction"] = correction
            messages = [{"role": "system", "content": system},
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
the full deliverable. External tool calls may be batched once by a contributor; their
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
- For each reviewer in reviewers[rubric_id], reviewer != primary[rubric_id], and that
  primary owner must be an ancestor of the reviewer through depends_on, directly or
  transitively. An upstream agent cannot review a downstream owner's future artifact.
  Do not add backward review dependencies that create cycles. reviewers is optional:
  omit a rubric or use an empty list when no independent downstream reviewer is feasible;
  put self-checks in checkpoints instead of claiming independent review. A reviewer is
  a single-pass consumer of published evidence, never an interactive feedback loop.
- sum(agent.max_calls) <= team.total_max_calls <= limits.total_max_calls, and
  team.max_parallel <= limits.max_parallel. Every agent.max_calls=1. All agent tools
  must be in task.tools; the final synthesizer must have tools=[].
- If limits.execution_max_tokens is present, every agent.max_tokens must be <= that
  actual execution-model output ceiling, including the synthesizer. Fit the requested
  deliverable within this bound rather than claiming an unsupported larger allowance."""


class GlobalAnalyzer(JsonModelCalls):
    def __init__(self, global_model: Callable,
                 local_model_factory: Callable[[str], Callable] | None = None, *,
                 max_agents: int = 4, max_parallel: int = 2, local_rounds: int = 1,
                 total_max_calls: int = 16, explicit_rubrics: bool = True,
                 max_corrections: int = 1, execution_max_tokens: int | None = None):
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
        self.last_prediction: Prediction | None = None

    def _limits(self) -> dict:
        limits = {"max_agents": self.max_agents, "max_parallel": self.max_parallel,
                  "total_max_calls": self.total_max_calls}
        if self.execution_max_tokens is not None:
            limits["execution_max_tokens"] = self.execution_max_tokens
        return limits

    def _prompt(self, prompt: str) -> str:
        if self.explicit_rubrics:
            return prompt
        return prompt + "\nABLATION: Do not explicitly predict rubrics. Return graph rubrics=[] " \
            "and edges=[], all rubric_ids=[] and additions=[], and coverage/primary/reviewers={}. " \
            "Still derive concrete responsibilities, dependencies, tools, checks and budgets " \
            "from the public task and applicable experience."

    def predict(self, task: PublicTask, experiences: Sequence = ()) -> Prediction:
        task = PublicTask.model_validate(task)
        prediction = self.ask(self.global_model, "predict", self._prompt(PREDICT_PROMPT),
                              {"task": task, "experiences": experiences,
                               "limits": self._limits()}, Prediction,
                              validate=lambda item: self._validate_prediction(task, item))
        self.last_prediction = prediction.model_copy(deep=True)
        return prediction

    def _validate_prediction(self, task: PublicTask, prediction: Prediction) -> None:
        if not self.explicit_rubrics and (prediction.graph.rubrics or prediction.graph.edges):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        ids = [agent.agent_id for agent in prediction.candidates]
        if len(ids) > self.max_agents or len(ids) != len(set(ids)):
            raise ValueError("Candidate count exceeds limit or contains duplicate IDs")
        rubric_ids = {r.rubric_id for r in prediction.graph.rubrics}
        for agent in prediction.candidates:
            self._check_agent(task, agent, rubric_ids)

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
        if agent.max_calls != 1:
            raise ValueError(f"Agent {agent.agent_id}.max_calls={agent.max_calls}; "
                             "single-pass execution requires AgentSpec.max_calls=1 explicitly. "
                             "Do not allocate execution corrections or communication rounds.")
        if self.execution_max_tokens is not None and agent.max_tokens > self.execution_max_tokens:
            raise ValueError(f"Agent {agent.agent_id}.max_tokens={agent.max_tokens} exceeds "
                             f"limits.execution_max_tokens={self.execution_max_tokens}; "
                             "fit its complete output and JSON overhead within the actual ceiling")

    def local_plan(self, task: PublicTask, prediction: Prediction, candidate: AgentSpec,
                   experiences: Sequence = ()) -> LocalPlan:
        local_experiences = []
        for experience in as_json(experiences):
            if experience_applicability(experience, task, capability=candidate.capability)["matched"]:
                local_experiences.append(experience)
        return self.ask(self.local_model_factory(candidate.agent_id), "local_plan",
                        self._prompt(LOCAL_PLAN_PROMPT),
                        {"task": task, "prediction": prediction, "candidate": candidate,
                         "immutable_identity": {"agent_id": candidate.agent_id,
                                                "capability": candidate.capability},
                         "experiences": local_experiences, "limits": self._limits()},
                        LocalPlan, agent_id=candidate.agent_id,
                        validate=lambda item: self._validate_local_plan(task, prediction, candidate, item))

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
        if plan.max_calls != 1:
            violations.append(f"LocalPlan.max_calls={plan.max_calls}; single-pass execution "
                              "requires LocalPlan.max_calls=1 explicitly. No execution correction "
                              "or communication round is available.")
        unavailable = sorted(set(plan.tools) - set(task.tools))
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
        response = self.ask(self.global_model, "reconcile", self._prompt(RECONCILE_PROMPT),
                            {"task": task, "prediction": prediction, "local_plans": plans,
                             "experiences": experiences, "limits": self._limits()},
                            _ReconciliationResponse,
                            validate=lambda item: self._validate_reconciliation(task, item))
        # Local testimony belongs to its author, not the reconciler.
        result = PlannedTeam(graph=response.graph, team=response.team,
                             local_plans=[plan.model_copy(deep=True) for plan in plans])
        return result

    def _validate_reconciliation(self, task: PublicTask, result: _ReconciliationResponse) -> None:
        team = result.team
        if not self.explicit_rubrics and (result.graph.rubrics or result.graph.edges
                                         or team.coverage or team.primary or team.reviewers):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        if (len(team.agents) > self.max_agents or team.max_parallel > self.max_parallel
                or team.total_max_calls > self.total_max_calls):
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
            if agent.agent_id == team.synthesizer_id and agent.tools:
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
