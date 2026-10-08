"""Post-submission semantic alignment and evidence-grounded collaborative analysis."""

from __future__ import annotations

import copy
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from pydantic import Field

from .planning import JsonModelCalls, as_json
from .schemas import (
    AttributionFinding, ChangeProposal, EvaluationFeedback, PublicTask, Record,
    RubricAlignment, RubricCreditAssignment, RubricGraph, TeamSpec, digest,
)


class Findings(Record):
    findings: list[AttributionFinding] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)


class AttributionOutline(Record):
    findings: list[AttributionFinding] = Field(default_factory=list)
    questions: dict[str, list[str]] = Field(default_factory=dict)
    rubric_assignments: dict[str, list[str]] = Field(default_factory=dict)
    rubric_assignment_rationale: dict[str, str] = Field(default_factory=dict)


class Proposals(Record):
    proposals: list[ChangeProposal] = Field(default_factory=list)


def feedback_view(feedback: EvaluationFeedback) -> dict:
    """Keep complete criterion feedback, not duplicated evaluator request/response audits."""
    feedback = EvaluationFeedback.model_validate(feedback)
    view = feedback.model_dump(mode="json", exclude={"raw", "rubrics"})
    if "quality_audit" in feedback.raw:
        view["quality_audit"] = copy.deepcopy(feedback.raw["quality_audit"])
    view["rubrics"] = []
    for rubric in feedback.rubrics:
        row = rubric.model_dump(mode="json", exclude={"raw"})
        # Quote verification and other semantic judge findings are not audit duplication.
        for field in ("axis", "confidence", "evidence_locations", "missing_elements", "error",
                      "quality_audit"):
            if field in rubric.raw:
                row[field] = copy.deepcopy(rubric.raw[field])
        row["signed_contribution"] = None if rubric.score is None else rubric.weight * rubric.score
        row["desired_score"] = 1.0 if rubric.weight > 0 else 0.0 if rubric.weight < 0 else None
        if rubric.status != "ok" or rubric.score is None:
            row["assessment"] = "evaluation_unavailable"
        elif rubric.weight < 0:
            row["assessment"] = "penalty_avoided" if rubric.score == 0 else "penalty_applied"
        elif rubric.weight == 0:
            row["assessment"] = "zero_weight"
        else:
            row["assessment"] = ("positive_criterion_met" if rubric.score == 1 else
                                 "positive_criterion_missed" if rubric.score == 0 else
                                 "positive_criterion_partial")
        view["rubrics"].append(row)
    return view


def build_credit_assignments(planned_graph: RubricGraph, team: TeamSpec,
                             planned_alignment: RubricAlignment,
                             feedback: EvaluationFeedback, *,
                             meta_assignments: dict[str, list[str]] | None = None,
                             meta_assignment_rationales: dict[str, str] | None = None
                             ) -> dict[str, RubricCreditAssignment]:
    """Route complete judge feedback by frozen responsibility and semantic alignment."""
    if not feedback.complete:
        raise ValueError("Rubric credit assignment requires complete evaluation feedback")
    predicted_ids = {rubric.rubric_id for rubric in planned_graph.rubrics}
    evaluated_ids = {rubric.rubric_id for rubric in feedback.rubrics}
    if len(evaluated_ids) != len(feedback.rubrics):
        raise ValueError("Evaluated rubric IDs must be unique for credit assignment")
    assigned_ids = {rubric_id for agent in team.agents for rubric_id in agent.rubric_ids}
    assigned_ids.update(team.coverage)
    assigned_ids.update(team.primary)
    assigned_ids.update(team.reviewers)
    if not assigned_ids <= predicted_ids:
        raise ValueError("Team responsibility references an unknown predicted rubric")
    predicted_to_evaluated: dict[str, set[str]] = {}
    for match in planned_alignment.matches:
        if (not match.predicted_ids or not match.evaluated_ids
                or not set(match.predicted_ids) <= predicted_ids
                or not set(match.evaluated_ids) <= evaluated_ids):
            raise ValueError("Alignment contains empty or unknown rubric references")
        for predicted_id in match.predicted_ids:
            predicted_to_evaluated.setdefault(predicted_id, set()).update(match.evaluated_ids)

    meta_assignments = meta_assignments or {}
    meta_assignment_rationales = meta_assignment_rationales or {}
    if not set(meta_assignments) <= {agent.agent_id for agent in team.agents}:
        raise ValueError("Meta rubric assignment references an unknown agent")
    if set(meta_assignments) != set(meta_assignment_rationales):
        raise ValueError("Every meta rubric assignment needs a rationale")
    for aid, rubric_ids in meta_assignments.items():
        if not rubric_ids or not set(rubric_ids) <= evaluated_ids:
            raise ValueError("Meta rubric assignment references an unknown evaluated rubric")
        if not meta_assignment_rationales[aid].strip():
            raise ValueError("Meta rubric assignment rationale cannot be empty")

    assignments = {}
    for agent in team.agents:
        basis: dict[str, list[str]] = {}
        for rubric_id in sorted(predicted_ids):
            sources = []
            if rubric_id in agent.rubric_ids:
                sources.append("agent_spec")
            if agent.agent_id in team.coverage.get(rubric_id, []):
                sources.append("coverage")
            if team.primary.get(rubric_id) == agent.agent_id:
                sources.append("primary")
            if agent.agent_id in team.reviewers.get(rubric_id, []):
                sources.append("reviewer")
            if sources:
                basis[rubric_id] = sources
        assigned_evaluated = {evaluated_id for rubric_id in basis
                              for evaluated_id in predicted_to_evaluated.get(rubric_id, ())}
        meta_evaluated = sorted(set(meta_assignments.get(agent.agent_id, ())) - assigned_evaluated)
        edges = [edge.model_copy(deep=True) for edge in planned_graph.edges
                 if edge.source in basis or edge.target in basis]
        assignments[agent.agent_id] = RubricCreditAssignment(
            agent_id=agent.agent_id, predicted_rubric_ids=list(basis),
            evaluated_rubric_ids=sorted(assigned_evaluated | set(meta_evaluated)), graph_edges=edges,
            assignment_basis=basis, meta_assigned_evaluated_rubric_ids=meta_evaluated,
            assignment_rationale=meta_assignment_rationales.get(agent.agent_id, ""))
    return assignments


def _compact_duplicate_event_content(events: Sequence[dict], local_execution: dict | None) -> list[dict]:
    """Remove only exact I/O duplicates from attribution payloads.

    ``local_execution`` is the authoritative complete trace.  Execution events
    repeat model outputs and submitted artifacts so that the event graph can be
    audited independently; sending both copies to a model wastes context.  Keep
    every event's identity and replace only content whose digest exactly matches
    a path in the complete local trace.  Unique content remains untouched, and
    persisted execution logs are never modified.
    """
    if not local_execution:
        return [copy.deepcopy(event) for event in events]
    references: dict[str, tuple[Any, str]] = {}

    def register(value, path):
        if value is not None:
            references.setdefault(digest(value), (value, path))

    for index, step in enumerate(local_execution.get("trajectory", [])):
        register(step.get("model_output_messages"),
                 f"local_execution.trajectory[{index}].model_output_messages")
        register(step.get("action_output"),
                 f"local_execution.trajectory[{index}].action_output")
    register(local_execution.get("answer"), "local_execution.answer")

    compacted = []
    for event in events:
        item = copy.deepcopy(event)
        if "content" in item:
            content_hash = digest(item["content"])
            duplicate = references.get(content_hash)
            if duplicate is not None and item["content"] == duplicate[0]:
                item.pop("content", None)
                item["content_ref"] = {"source": "local_execution", "path": duplicate[1],
                                        "content_hash": content_hash}
        compacted.append(item)
    return compacted


SIGNED_SCORE_PROMPT = """Interpret scoring by signed_contribution = weight * score, not
the verdict word alone. Positive-weight criteria reward satisfaction. Negative-weight
criteria describe behavior to avoid: Satisfied (score=1) applies a penalty, whereas
Not Satisfied (score=0) avoids that penalty and is the desired outcome. Do not claim
that absence of a negative-weight behavior lost points, or propose satisfying that
negative criterion to improve the official score. desired_score and assessment are
derived diagnostics; they never replace the original criterion, verdict or score.
An unavailable evaluation does not establish success or failure."""


EVIDENCE_QUALITY_PROMPT = """Separate the official score from answer quality and judge
reliability. Compare each criterion, verdict, reason and supplied answer evidence. If they
appear inconsistent, preserve both sides and classify the unresolved explanation as
external_or_uncertain; do not silently flip scores or learn advice to satisfy a suspected
judge error. quality_audit is risk-only: its flags request inspection, and
semantic_consistency=unassessed is not proof of a contradiction or an incorrect verdict.
An exact quote establishes where words occurred, not that the claim is factually true.
A checkpoint or agent statement that something was checked is only a self-report. To
claim a handoff was incorporated, identify the actual received artifact and resulting
answer change; a dependency path does not prove direct artifact consumption. In
particular an earlier writer cannot have incorporated a later review in a single-pass
DAG. For mathematical verification, cite the observable derivation, assumptions,
substitution, counterexample or tool output, including the tested claim and result.
Merely naming a theorem, repeating a conclusion, valid JSON, or a success flag does not
verify a calculation. Without observable checks retain the uncertainty, rather than
reporting verification or inventing a missing execution event."""


ALIGN_PROMPT = """Semantically align predicted quality requirements with independently
evaluated criteria. Determine matching by meaning, not equality of predicted and evaluated
IDs or literal text. Support paraphrases, partial
coverage, one-to-many splits, many-to-one merges and many-to-many mappings. Preserve
uncertain matches with their confidence and rationale. Distinguish prohibition from a
desired behavior; negative evaluator weights describe penalties and remain signed.
Report unmatched predictions without assuming they were wrong: evaluation can be
nonexhaustive. Include every omitted evaluated criterion and unmatched prediction.
Importance is a planner priority, not the evaluator weight: compare relative magnitude
only with a stated rationale. Do not modify either input graph or evaluation record.
Copy predicted_ids only from valid_predicted_ids and evaluated_ids only from
valid_evaluated_ids, preserving every character including long prefixes and suffixes.
Never invent, abbreviate, renumber or substitute criterion text for an ID. Each match
requires nonempty predicted_ids AND evaluated_ids; use missed_evaluated_ids and
unmatched_predicted_ids for omissions, never an empty-sided match. Evidence quotes with
evidence_locations.verified=false are not confirmed verbatim excerpts; confidence is
the evaluator's reported confidence, not proof of correctness.""" + "\n" + SIGNED_SCORE_PROMPT + "\n" + EVIDENCE_QUALITY_PROMPT

OUTLINE_PROMPT = """Develop an initial, evidence-supported attribution hypothesis after
submission and independent evaluation. Inspect both frozen quality predictions, their
semantic alignments, the team assignment, final answer, event index and shared artifacts.
Identify successes, failures and cross-rubric relationships. Allocate questions to relevant
agents, using coverage and execution dependencies. Do not invent unseen execution details.
Classify hypotheses with multiple labels when appropriate: prediction (missed or mistaken
requirements/priorities), organization (ownership, handoffs, review or budget), execution
(failure despite adequate organization), external_or_uncertain (tool/environment or
insufficient/conflicting evidence). An unowned missed requirement does not justify blaming
an arbitrary agent. Cite only supplied evidence IDs, record opposing evidence, alternatives
and uncertainty. This is a hypothesis, not an identified causal effect.
credit_assignments provides formal responsibility routes from the frozen team and semantic
alignment. To request an agent's reflection on an additional evaluated rubric, return its
exact evaluated ID in rubric_assignments[agent_id] with a nonempty explanation in
rubric_assignment_rationale[agent_id], grounded in the completed role or collaboration
topology. This may include missed criteria, but allocates reflection scope rather than
failure blame. Do not invent responsibility or alter the frozen plan. Criteria without a
supported local connection remain available only to the meta-agent's global analysis."""

COST_ATTRIBUTION_PROMPT = """Use observed resource_usage alongside team.budget_plan to inspect
token use, repeated context and redundant collaboration. Forecasts are not actual usage;
local planning and execution may be included in a role's recorded costs. Propose conditional
organization or execution lessons that conserve tokens without weakening public requirements
or factual checks. A shorter answer or fewer calls alone does not prove better quality-cost
efficiency; unknown prices remain unknown."""

FEEDBACK_IDS_PROMPT = """Copy rubric IDs exactly from the supplied graphs and feedback;
copy evidence references exactly from evidence_ids. Use only exact evidence IDs in
supporting_evidence and opposing_evidence: opposing_evidence is a list of evidence IDs,
not an explanation in prose. Put alternatives, explanations and uncertainty in their
dedicated text fields. A feedback:<rubric_id> reference is an evidence ID, not a rubric
ID. Quote verification flags describe whether the quoted text was found verbatim; do not treat unverified quotes as exact observations or change the recorded official score based on this diagnostic.""" + "\n" + SIGNED_SCORE_PROMPT + "\n" + EVIDENCE_QUALITY_PROMPT

LOCAL_ATTRIBUTION_PROMPT = """Analyze your own execution with your complete observable local
context, the relevant rubric feedback, global questions and upstream/downstream evidence.
Trace what was retrieved, read, sent, consumed and incorporated into the submitted answer;
these are different events. Your own previous claims are not verified evidence by themselves.
Explain successes and failures using event IDs, opposing evidence, alternative explanations
and uncertainty. Do not claim access to other agents' private histories. If a specific event
in the index is needed, return its event_id in evidence_requests; at most one additional
evidence exchange is allowed. Do not request hidden reasoning or invent missing evidence.
Suggest only supported hypotheses, not direct writes to experience or evaluator settings.
An event with content_ref repeats content already present at the supplied local_execution
path with the exact content_hash; inspect that complete value and cite the event_id.
This is a storage reference, not missing evidence or a summary of the content.
credit_assignment routes your frozen responsibilities to assigned_predicted_rubrics and
their semantically aligned judge feedback. Its graph_edges give cross-rubric context,
not additional ownership. Shared split/merge feedback does not identify your individual
causal contribution. Unassigned judge criteria remain the meta-agent's global concern."""

INTEGRATE_PROMPT = """Integrate the global outline and independent local analyses into
evidence-supported attribution hypotheses. Preserve disagreements and alternative
explanations; local self-reports are not authoritative. Assess cross-agent propagation and
missing handoffs. Multiple causes may share responsibility. For requirements not predicted
or assigned, examine prediction/organization before execution; never force blame on an
agent. Successes need evidence and applicability too. Cite supplied IDs only, including
counterevidence. Evidence gaps must remain external_or_uncertain. Do not claim experimental
causal identification from reflection. This phase produces findings, not experience writes."""

PROPOSE_PROMPT = """Propose compact, conditional improvements from evidence-supported
attribution findings. Each proposal changes one experience in exactly one bank: rubric
(task signals and quality prediction), organization (capability grouping, ownership,
dependencies, review and budget), or execution (a reusable capability-specific practice).
Store transferable instructions, never answers or hidden criterion text specific to the
source task. Do not copy a private source criterion's numerical threshold or target
algorithm names into a cross-task rule. Applicability and task_signals must depend on
public task features; future agents cannot assume access to private benchmark rubrics.
task_signals are necessary public conditions, not alternatives: use short observable
phrases, all applicable to the current public task. Distinguish the requested operation
(for example comparison or derivation) from generic article/writing format. A future
task must satisfy these signals before receiving the advice. Applicability states the
boundary; capability states which role can perform the practice, not a source role ID.
Describe a procedure and its verification, not a fixed checklist of source-specific
facts. For any count, derive the required amount from the future public task and needed
coverage; never turn a private source count of bullets, examples or dimensions into a
universal minimum. Verify a proposed instruction remains useful with different subjects
and a different rubric, and abstain if its benefit depends on knowing the source rubric.
Record applicability, task signals, capability signature for execution advice,
source task, evidence and counterevidence, concrete diff, old state version, expected benefit,
risks. Success findings can yield conditional
positive advice. Only supported findings justify a proposal; abstain when evidence is
insufficient. Cite evidence IDs from the findings and do not invent new evidence. The caller
selects which proposal, if any, to write after structural/evidence-reference checks under
the one-update-per-source policy; this phase only proposes candidates.
there is no acceptance decision, hold state, or independent-task promotion gate. Do not
claim an update has been experimentally validated. Do not edit evaluator,
hidden rubrics, split manifests, runtime or budget guards. Return at most three proposals.
Both proposal.evidence and proposal.experience.evidence must contain exact IDs from
valid_supporting_evidence_ids. experience.counterevidence must use exact IDs from
valid_counterevidence_ids. A finding_id identifies an analysis, not an observation:
refer to finding IDs only in rationale, never substitute them for evidence IDs. Follow
the finding's supporting_evidence references instead. Do not abbreviate or renumber IDs.
When scoring_context is available, ground any claimed score improvement in its signed
criterion interpretation rather than assuming every Not Satisfied verdict is a defect.""" + "\n" + SIGNED_SCORE_PROMPT + "\n" + EVIDENCE_QUALITY_PROMPT


_COUNT_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    range(21)))
_CONTENT_QUANTITY = (
    r"(?P<count>\d+|" + "|".join(_COUNT_WORDS) + r")\b"
    r"(?:[\s-]+\w+){0,5}?\s+(?P<unit>sentences?|bullets?|examples?|dimensions?|steps?|"
    r"similarities|differences|characteristics|points?|references?|cases?|sections?|tables?|charts?)\b"
)
_CONTENT_COUNT = re.compile(
    r"\b(?:at\s+least|at\s+most|no\s+fewer\s+than|no\s+more\s+than|exactly|"
    r"minimum(?:\s+of)?|maximum(?:\s+of)?|include|provide|list|enumerate|cover|present|contains?)\s+"
    + _CONTENT_QUANTITY, re.IGNORECASE)
_PUBLIC_CONTENT_COUNT = re.compile(r"\b" + _CONTENT_QUANTITY, re.IGNORECASE)


def _content_counts(text, *, public=False):
    pattern = _PUBLIC_CONTENT_COUNT if public else _CONTENT_COUNT
    counts = set()
    for match in pattern.finditer(text):
        value, unit = match.group("count").casefold(), match.group("unit").casefold()
        count = int(value) if value.isdigit() else _COUNT_WORDS[value]
        category = "similarity" if unit == "similarities" else unit.removesuffix("s")
        # Carrier words alone do not identify the required content. Keep an explicit
        # nearby comparison topic, but never equate unrelated method steps by number.
        if category in {"sentence", "bullet", "example", "point", "characteristic"}:
            clause = re.split(r"[.;\n]|\b(?:before|after|whereas)\b",
                              text[match.start():], maxsplit=1, flags=re.IGNORECASE)[0]
            words = set(re.findall(r"\w+", " ".join(clause.split()[:24]).casefold()))
            shared = bool(words.intersection({"similarity", "similarities", "shared", "commonality"}))
            different = bool(words.intersection({"difference", "differences", "contrasts"}))
            if shared != different:
                category = "similarity" if shared else "difference"
        counts.add((count, category))
    return counts


class RubricAttributor(JsonModelCalls):
    def __init__(self, global_model: Callable,
                 local_model_factory: Callable[[str], Callable] | None = None, *,
                 max_parallel: int = 2, local_attribution: bool = True,
                 max_corrections: int = 1):
        super().__init__(max_corrections=max_corrections)
        self.global_model = global_model
        self.local_model_factory = local_model_factory or (lambda _aid: global_model)
        self.max_parallel = max_parallel
        self.local_attribution = local_attribution
        self.last_global_outline: AttributionOutline | None = None
        self.last_local_findings: dict[str, Findings] = {}
        self.last_alignments: dict[str, RubricAlignment] = {}
        self.last_alignment_warnings: list[dict] = []
        self.last_credit_assignments: dict[str, RubricCreditAssignment] = {}
        self._scoring_context: dict | None = None

    def align(self, graph: RubricGraph, feedback: EvaluationFeedback) -> RubricAlignment:
        predicted = {r.rubric_id for r in graph.rubrics}
        evaluated = {r.rubric_id for r in feedback.rubrics}

        def validate(alignment):
            valid_matches = []
            for index, match in enumerate(alignment.matches):
                if not match.predicted_ids or not match.evaluated_ids:
                    # Provider aligners occasionally emit an empty-sided or
                    # truncated match despite the schema prompt. Preserve the
                    # auditable response as a warning, drop only that unusable
                    # edge, and let bookkeeping classify its IDs as unmatched.
                    self.last_alignment_warnings.append({
                        "index": index,
                        "predicted_ids": list(match.predicted_ids),
                        "evaluated_ids": list(match.evaluated_ids),
                        "reason": "empty_rubric_reference",
                    })
                    continue
                # Unknown IDs indicate a model hallucination rather than an
                # incomplete match. Keep the existing correction path so the
                # provider gets a chance to repair the exact identifier.
                if (not set(match.predicted_ids) <= predicted
                        or not set(match.evaluated_ids) <= evaluated):
                    raise ValueError("Alignment contains empty or unknown rubric references")
                valid_matches.append(match)
            alignment.matches[:] = valid_matches

        alignment = self.ask(self.global_model, "align", ALIGN_PROMPT,
                             {"prediction": graph, "feedback": feedback_view(feedback),
                              "valid_predicted_ids": [r.rubric_id for r in graph.rubrics],
                              "valid_evaluated_ids": [r.rubric_id for r in feedback.rubrics]}, RubricAlignment,
                             validate=validate)
        matched_predicted: set[str] = set()
        matched_evaluated: set[str] = set()
        for match in alignment.matches:
            matched_predicted.update(match.predicted_ids)
            matched_evaluated.update(match.evaluated_ids)
        # Unmatched sets are bookkeeping, not another semantic model judgment.
        alignment.missed_evaluated_ids = sorted(evaluated - matched_evaluated)
        alignment.unmatched_predicted_ids = sorted(predicted - matched_predicted)
        return alignment

    @staticmethod
    def _execution_view(result: Any) -> tuple[dict, dict[str, dict], list[dict]]:
        data = result.full_dict() if hasattr(result, "full_dict") else as_json(result)
        if not isinstance(data, dict):
            raise TypeError("Attribution needs RunResult or its full serialization")
        locals_by_id: dict[str, dict] = {}

        def collect(run: dict):
            aid = run.get("metadata", {}).get("agent_id")
            if aid:
                if aid in locals_by_id:
                    raise ValueError("Duplicate agent trace ID")
                locals_by_id[aid] = run
            for child in run.get("sub_runs", []):
                collect(child)

        collect(data)
        events = [as_json(event) for event in data.get("metadata", {}).get("events", [])]
        return data, locals_by_id, events

    def attribute(self, task: PublicTask, global_graph: RubricGraph,
                  planned_graph: RubricGraph, team: TeamSpec, result: Any,
                  feedback: EvaluationFeedback, *,
                  global_alignment: RubricAlignment | None = None,
                  planned_alignment: RubricAlignment | None = None) -> list[AttributionFinding]:
        if feedback.task_id != task.task_id:
            raise ValueError("Feedback task differs from submitted task")
        if not feedback.complete:
            raise ValueError("Attribution requires complete evaluation feedback")
        self.last_alignment_warnings = []
        data, local_runs, events = self._execution_view(result)
        global_alignment = global_alignment or self.align(global_graph, feedback)
        planned_alignment = planned_alignment or self.align(planned_graph, feedback)
        self.last_alignments = {"global": global_alignment, "planned": planned_alignment}
        self.last_credit_assignments = build_credit_assignments(planned_graph, team,
                                                               planned_alignment, feedback)
        event_by_id = {event["event_id"]: event for event in events}
        if len(event_by_id) != len(events):
            raise ValueError("Event IDs must be unique for attribution")
        event_index = [{key: value for key, value in event.items() if key != "content"}
                       for event in events]
        shared = [event for event in events if event.get("kind") in {
            "artifact", "artifact_created", "artifact_published", "submitted", "final_answer",
            "handoff", "message_sent", "resource_usage"}]
        submission_id = "submission:" + digest(data.get("answer"))
        evaluation = feedback_view(feedback)
        self._scoring_context = {"task_id": feedback.task_id, "criteria": [
            {key: copy.deepcopy(row[key]) for key in ("rubric_id", "criterion", "weight", "score",
                  "verdict", "reason", "signed_contribution", "desired_score", "assessment",
                  "quality_audit") if key in row}
            for row in evaluation["rubrics"]]}
        if "quality_audit" in evaluation:
            self._scoring_context["quality_audit"] = copy.deepcopy(evaluation["quality_audit"])
        context = {"task": task, "global_graph": global_graph, "planned_graph": planned_graph,
                   "team": team, "feedback": evaluation, "alignments": self.last_alignments,
                   "credit_assignments": self.last_credit_assignments,
                   "submission": {"evidence_id": submission_id, "answer": data.get("answer")},
                   "event_index": event_index, "shared_artifacts": shared,
                   "evidence_ids": [*event_by_id, submission_id,
                                    *["feedback:" + r.rubric_id for r in feedback.rubrics],
                                    "planning:global", "planning:planned", "planning:team"]}
        known_agents = {agent.agent_id for agent in team.agents}
        valid_evidence = set(context["evidence_ids"])
        valid_rubrics = ({r.rubric_id for r in planned_graph.rubrics}
                         | {r.rubric_id for r in global_graph.rubrics}
                         | {r.rubric_id for r in feedback.rubrics})

        def _validate_findings(findings, *, phase, unique_ids=False):
            finding_ids: set[str] = set()
            for index, finding in enumerate(findings.findings):
                finding_id = finding.finding_id
                if unique_ids and finding_id in finding_ids:
                    raise ValueError(f"{phase} findings[{index}] finding_id={finding_id!r}: "
                                     "Duplicate attribution finding ID")
                finding_ids.add(finding_id)
                invalid_agents = sorted(set(finding.agent_ids) - known_agents)
                invalid_rubrics = sorted(set(finding.rubric_ids) - valid_rubrics)
                if invalid_agents or invalid_rubrics:
                    raise ValueError(
                        f"{phase} findings[{index}] finding_id={finding_id!r}: "
                        "Attribution references unknown agents or rubrics; "
                        f"invalid_agent_ids={invalid_agents}; invalid_rubric_ids={invalid_rubrics}; "
                        f"valid_rubric_ids={sorted(valid_rubrics)}")
                for field in ("supporting_evidence", "opposing_evidence"):
                    references = getattr(finding, field)
                    invalid = sorted(set(references) - valid_evidence)
                    if invalid:
                        raise ValueError(
                            f"{phase} findings[{index}].{field} finding_id={finding_id!r}: "
                            "Attribution invented an evidence reference; "
                            f"invalid_evidence_ids={invalid}; valid_evidence_ids={sorted(valid_evidence)}")

        outline_attempts = 0

        def validate_outline(outline):
            nonlocal outline_attempts
            _validate_findings(outline, phase="attribute_global")
            outline_attempts += 1
            # Question allocation is advisory metadata for local attribution.
            # A model can occasionally echo a stale agent id from a prior
            # roster.  Preserve one correction turn for the normal case, then
            # discard only the stale keys deterministically so an invalid
            # attribution hint cannot consume the whole EVO task.  Findings,
            # rubric ids and proposals remain strictly validated below.
            unknown = set(outline.questions) - known_agents
            if unknown and outline_attempts == 1:
                raise ValueError("Global attribution addressed an unknown agent")
            if unknown:
                outline.questions = {aid: questions for aid, questions in outline.questions.items()
                                     if aid in known_agents}
            build_credit_assignments(planned_graph, team, planned_alignment, feedback,
                meta_assignments=outline.rubric_assignments,
                meta_assignment_rationales=outline.rubric_assignment_rationale)

        outline = self.ask(self.global_model, "attribute_global", OUTLINE_PROMPT + "\n" + COST_ATTRIBUTION_PROMPT + "\n" + FEEDBACK_IDS_PROMPT,
                           context, AttributionOutline, validate=validate_outline)
        self.last_global_outline = outline
        self.last_credit_assignments = build_credit_assignments(planned_graph, team,
            planned_alignment, feedback, meta_assignments=outline.rubric_assignments,
            meta_assignment_rationales=outline.rubric_assignment_rationale)
        context["credit_assignments"] = self.last_credit_assignments
        evaluated_to_predicted: dict[str, set[str]] = {}
        for match in planned_alignment.matches:
            for evaluated in match.evaluated_ids:
                evaluated_to_predicted.setdefault(evaluated, set()).update(match.predicted_ids)

        def analyze_agent(agent):
            aid = agent.agent_id
            assignment = self.last_credit_assignments[aid]
            if aid not in local_runs:
                return aid, Findings(findings=[AttributionFinding(
                    finding_id="missing-trace-" + aid, rubric_ids=assignment.predicted_rubric_ids,
                    categories=["external_or_uncertain"], agent_ids=[aid],
                    component="trace", hypothesis="Complete local observable trace is unavailable",
                    alternatives=["The agent may not have been scheduled"], uncertainty=1.0)])
            relevant = [rubric for rubric in evaluation["rubrics"]
                        if rubric["rubric_id"] in assignment.evaluated_rubric_ids]
            # Full local I/O is retained; only indexed, connected cross-agent events are shared.
            related = [event for event in events if event.get("agent_id") == aid
                       or event.get("recipient") == aid
                       or event.get("agent_id") in agent.depends_on and event.get("kind") in {
                           "artifact", "artifact_created", "artifact_published", "handoff", "message_sent"}]
            payload = {"task": task, "agent": agent, "local_execution": local_runs[aid],
                       "credit_assignment": assignment,
                       "assigned_predicted_rubrics": [rubric for rubric in planned_graph.rubrics
                           if rubric.rubric_id in assignment.predicted_rubric_ids],
                       "feedback": relevant, "questions": outline.questions.get(aid, []),
                       "global_findings": outline.findings,
                       "related_events": _compact_duplicate_event_content(related, local_runs[aid]),
                       "event_index": event_index, "submission": context["submission"],
                       "evidence_ids": context["evidence_ids"]}
            model = self.local_model_factory(aid)

            def validate_requests(findings):
                _validate_findings(findings, phase="attribute_local")
                requested = set(findings.evidence_requests)
                if len(requested) > 8 or not requested <= event_by_id.keys():
                    raise ValueError("Local attribution requested unavailable or too many events")

            def validate_followup(findings):
                _validate_findings(findings, phase="attribute_local_followup")
                if findings.evidence_requests:
                    raise ValueError("Local attribution exceeded its evidence exchange limit")

            findings = self.ask(model, "attribute_local", LOCAL_ATTRIBUTION_PROMPT + "\n" + FEEDBACK_IDS_PROMPT,
                                payload, Findings, agent_id=aid, validate=validate_requests)
            requested = list(dict.fromkeys(findings.evidence_requests))
            if requested:
                findings = self.ask(model, "attribute_local_followup",
                                    LOCAL_ATTRIBUTION_PROMPT + "\n" + FEEDBACK_IDS_PROMPT,
                                    {**payload, "prior_analysis": findings,
                                     "requested_evidence": _compact_duplicate_event_content(
                                         [event_by_id[eid] for eid in requested], local_runs[aid]),
                                     "remaining_exchanges": 0}, Findings, agent_id=aid,
                                    validate=validate_followup)
            return aid, findings

        self.last_local_findings = {}
        if self.local_attribution:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = [pool.submit(analyze_agent, agent) for agent in team.agents]
                self.last_local_findings = dict(future.result() for future in futures)
        def validate_final_findings(final):
            _validate_findings(final, phase="attribute_integrate", unique_ids=True)

        final = self.ask(self.global_model, "attribute_integrate", INTEGRATE_PROMPT + "\n" + COST_ATTRIBUTION_PROMPT + "\n" + FEEDBACK_IDS_PROMPT,
                         {**context, "global_outline": outline,
                          "local_findings": self.last_local_findings}, Findings,
                         validate=validate_final_findings)
        for finding in final.findings:
            if not finding.supporting_evidence:
                finding.categories = ["external_or_uncertain"]
                finding.uncertainty = max(finding.uncertainty, 0.9)
            if not finding.success:
                refs = set(finding.rubric_ids)
                evaluated_ids = {rubric.rubric_id for rubric in feedback.rubrics}
                mapped = set().union(*(evaluated_to_predicted.get(rid, set())
                    if rid in evaluated_ids else {rid} for rid in refs))
                owners = {assignment.agent_id for assignment in self.last_credit_assignments.values()
                          if mapped.intersection(assignment.predicted_rubric_ids)}
                if not owners and finding.supporting_evidence:
                    finding.agent_ids = []
                    finding.categories = [category for category in finding.categories
                                          if category != "execution"]
                    category = ("prediction" if refs.intersection(
                        planned_alignment.missed_evaluated_ids) else "organization")
                    if category not in finding.categories:
                        finding.categories.append(category)
        return final.findings

    def propose(self, task: PublicTask, findings: Sequence[AttributionFinding],
                base_version: int, experiences: Sequence = (), *,
                agent_reflections=None) -> list[ChangeProposal]:
        supported = [finding for finding in findings if finding.supporting_evidence
                     and any(category != "external_or_uncertain" for category in finding.categories)]
        if not supported:
            return []
        evidence = {eid for finding in supported for eid in finding.supporting_evidence}
        counterevidence = evidence | {eid for finding in supported for eid in finding.opposing_evidence}
        scoring = (self._scoring_context if self._scoring_context
                   and self._scoring_context["task_id"] == task.task_id else None)
        payload = {"task": task, "findings": supported,
                   "base_version": base_version, "experiences": experiences,
                   "valid_supporting_evidence_ids": sorted(evidence),
                   "valid_counterevidence_ids": sorted(counterevidence),
                   "scoring_context": scoring}
        allowed_banks = None
        prompt = PROPOSE_PROMPT + "\n" + FEEDBACK_IDS_PROMPT
        if agent_reflections is not None:
            allowed_banks = ["rubric", "organization"]
            payload.update(agent_reflections=agent_reflections, allowed_experience_banks=allowed_banks)
            prompt += ("\nThe meta-level update follows the supplied agent_reflections. "
                       "Integrate their supported lessons into rubric prediction or MAS organization "
                       "experience only, using allowed_experience_banks. Capability-specific "
                       "execution practices belong to the retained agent harness updates.")
        else:
            prompt += ("\nThis legacy caller writes only the first proposal after attribution; "
                       "additional proposals are recorded but not applied.")
        return self.ask(self.global_model, "propose", prompt, payload,
                             Proposals, validate=lambda result: self._validate_proposals(
                                 result.proposals, task, supported, base_version, experiences,
                                 scoring_context=scoring, allowed_banks=allowed_banks)).proposals

    @staticmethod
    def _validate_proposals(proposals, task, supported, base_version, experiences, *, scoring_context=None,
                            allowed_banks=None):
        if len(proposals) > 3:
            raise ValueError("At most three single-experience proposals are allowed")
        proposal_ids = [proposal.proposal_id for proposal in proposals]
        if any(not proposal_id.strip() for proposal_id in proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("Proposal IDs must be nonempty and unique")
        evidence = {eid for finding in supported for eid in finding.supporting_evidence}
        counterevidence = {eid for finding in supported for eid in finding.opposing_evidence}
        existing = {entry["experience_id"]: entry for entry in as_json(experiences)}
        private_counts = set()
        if scoring_context and scoring_context.get("task_id") == task.task_id:
            private_counts = set().union(*(_content_counts(row["criterion"])
                                          for row in scoring_context.get("criteria", [])))
            private_counts -= _content_counts(" ".join([task.question, *task.constraints]), public=True)
        for proposal in proposals:
            exp = proposal.experience
            if allowed_banks is not None and exp.bank not in allowed_banks:
                raise ValueError("Meta-level proposal must use the allowed experience banks: "
                                 + ", ".join(allowed_banks))
            if proposal.source_task_id != task.task_id or proposal.base_version != base_version:
                raise ValueError("Proposal source task or base version changed")
            if exp.source_task_ids != [task.task_id]:
                raise ValueError("Proposal must cite the observed source task")
            if not set(proposal.evidence + exp.evidence) <= evidence:
                raise ValueError("Proposal evidence is not supported by attribution: "
                    f"proposal_id={proposal.proposal_id}; "
                    f"invalid_proposal_evidence_ids={sorted(set(proposal.evidence) - evidence)}; "
                    f"invalid_experience_evidence_ids={sorted(set(exp.evidence) - evidence)}; "
                    f"valid_supporting_evidence_ids={sorted(evidence)}. "
                    "finding_id is for rationale, not an evidence reference.")
            if not set(exp.counterevidence) <= counterevidence | evidence:
                raise ValueError("Proposal invented counterevidence: "
                    f"invalid_counterevidence_ids={sorted(set(exp.counterevidence) - (counterevidence | evidence))}; "
                    f"valid_counterevidence_ids={sorted(counterevidence | evidence)}")
            if exp.bank == "execution" and not exp.capability.strip():
                raise ValueError("Execution experience requires a reusable capability signature")
            copied_counts = private_counts.intersection(_content_counts(exp.instruction))
            if copied_counts:
                raise ValueError("Experience contains a potential source-specific content-count threshold "
                    f"in a reusable instruction: count_categories={sorted(copied_counts)}. Describe a conditional "
                    "coverage/checking procedure and derive counts from the future public task; "
                    "do not copy hidden rubric minima. Matching is a conservative quantity/category "
                    "heuristic, not proof of leakage. Mathematical formula parameters are not "
                    "content-count thresholds.")
            if proposal.replaces_id and proposal.replaces_id not in existing:
                raise ValueError("Proposal replaces an unknown experience")
