"""Post-submission semantic alignment and evidence-grounded collaborative analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from pydantic import Field

from .planning import JsonModelCalls, as_json
from .schemas import (
    AttributionFinding, ChangeProposal, EvaluationFeedback, PublicTask, Record,
    RubricAlignment, RubricGraph, TeamSpec, digest,
)


class Findings(Record):
    findings: list[AttributionFinding] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)


class AttributionOutline(Record):
    findings: list[AttributionFinding] = Field(default_factory=list)
    questions: dict[str, list[str]] = Field(default_factory=dict)


class Proposals(Record):
    proposals: list[ChangeProposal] = Field(default_factory=list)


ALIGN_PROMPT = """Semantically align predicted quality requirements with independently
evaluated criteria. IDs or literal text need not match. Support paraphrases, partial
coverage, one-to-many splits, many-to-one merges and many-to-many mappings. Preserve
uncertain matches with their confidence and rationale. Distinguish prohibition from a
desired behavior; negative evaluator weights describe penalties and remain signed.
Report unmatched predictions without assuming they were wrong: evaluation can be
nonexhaustive. Include every omitted evaluated criterion and unmatched prediction.
Importance is a planner priority, not the evaluator weight: compare relative magnitude
only with a stated rationale. Do not modify either input graph or evaluation record."""

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
and uncertainty. This is a hypothesis, not an identified causal effect."""

LOCAL_ATTRIBUTION_PROMPT = """Analyze your own execution with your complete observable local
context, the relevant rubric feedback, global questions and upstream/downstream evidence.
Trace what was retrieved, read, sent, consumed and incorporated into the submitted answer;
these are different events. Your own previous claims are not verified evidence by themselves.
Explain successes and failures using event IDs, opposing evidence, alternative explanations
and uncertainty. Do not claim access to other agents' private histories. If a specific event
in the index is needed, return its event_id in evidence_requests; at most one additional
evidence exchange is allowed. Do not request hidden reasoning or invent missing evidence.
Suggest only supported hypotheses, not direct writes to experience or evaluator settings."""

INTEGRATE_PROMPT = """Integrate the global outline and independent local analyses into
evidence-supported attribution hypotheses. Preserve disagreements and alternative
explanations; local self-reports are not authoritative. Assess cross-agent propagation and
missing handoffs. Multiple causes may share responsibility. For requirements not predicted
or assigned, examine prediction/organization before execution; never force blame on an
agent. Successes need evidence and applicability too. Cite supplied IDs only, including
counterevidence. Evidence gaps must remain external_or_uncertain. Do not claim experimental
causal identification from reflection. No experience is accepted during this phase."""

PROPOSE_PROMPT = """Propose compact, conditional improvements from evidence-supported
attribution findings. Each proposal changes one experience in exactly one bank: rubric
(task signals and quality prediction), organization (capability grouping, ownership,
dependencies, review and budget), or execution (a reusable capability-specific practice).
Store transferable instructions, never answers or hidden criterion text specific to the
source task. Record applicability, task signals, capability signature for execution advice,
source task, evidence and counterevidence, concrete diff, old state version, expected benefit,
risks and a paired independent-task validation plan. Success findings can yield conditional
positive advice. Only supported findings justify a proposal; abstain when evidence is
insufficient. Cite evidence IDs from the findings and do not invent new evidence. All proposed
experience has validation_status=staged and validation_result remains empty. Proposing is
not accepting; only actual paired reruns may validate a change. Do not edit evaluator,
hidden rubrics, split manifests, runtime or budget guards. Return at most three proposals."""


class RubricAttributor(JsonModelCalls):
    def __init__(self, global_model: Callable,
                 local_model_factory: Callable[[str], Callable] | None = None, *,
                 max_parallel: int = 2, local_attribution: bool = True):
        super().__init__()
        self.global_model = global_model
        self.local_model_factory = local_model_factory or (lambda _aid: global_model)
        self.max_parallel = max_parallel
        self.local_attribution = local_attribution
        self.last_global_outline: AttributionOutline | None = None
        self.last_local_findings: dict[str, Findings] = {}
        self.last_alignments: dict[str, RubricAlignment] = {}

    def align(self, graph: RubricGraph, feedback: EvaluationFeedback) -> RubricAlignment:
        alignment = self.ask(self.global_model, "align", ALIGN_PROMPT,
                             {"prediction": graph, "feedback": feedback}, RubricAlignment)
        predicted = {r.rubric_id for r in graph.rubrics}
        evaluated = {r.rubric_id for r in feedback.rubrics}
        matched_predicted: set[str] = set()
        matched_evaluated: set[str] = set()
        for match in alignment.matches:
            if (not match.predicted_ids or not match.evaluated_ids
                    or not set(match.predicted_ids) <= predicted
                    or not set(match.evaluated_ids) <= evaluated):
                raise ValueError("Alignment contains empty or unknown rubric references")
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
        data, local_runs, events = self._execution_view(result)
        global_alignment = global_alignment or self.align(global_graph, feedback)
        planned_alignment = planned_alignment or self.align(planned_graph, feedback)
        self.last_alignments = {"global": global_alignment, "planned": planned_alignment}
        event_by_id = {event["event_id"]: event for event in events}
        if len(event_by_id) != len(events):
            raise ValueError("Event IDs must be unique for attribution")
        event_index = [{key: value for key, value in event.items() if key != "content"}
                       for event in events]
        shared = [event for event in events if event.get("kind") in {
            "artifact", "artifact_created", "artifact_published", "submitted", "final_answer",
            "handoff", "message_sent"}]
        submission_id = "submission:" + digest(data.get("answer"))
        context = {"task": task, "global_graph": global_graph, "planned_graph": planned_graph,
                   "team": team, "feedback": feedback, "alignments": self.last_alignments,
                   "submission": {"evidence_id": submission_id, "answer": data.get("answer")},
                   "event_index": event_index, "shared_artifacts": shared,
                   "evidence_ids": [*event_by_id, submission_id,
                                    *["feedback:" + r.rubric_id for r in feedback.rubrics],
                                    "planning:global", "planning:planned", "planning:team"]}
        outline = self.ask(self.global_model, "attribute_global", OUTLINE_PROMPT,
                           context, AttributionOutline)
        self.last_global_outline = outline
        known_agents = {agent.agent_id for agent in team.agents}
        if not set(outline.questions) <= known_agents:
            raise ValueError("Global attribution addressed an unknown agent")
        evaluated_to_predicted: dict[str, set[str]] = {}
        for match in planned_alignment.matches:
            for evaluated in match.evaluated_ids:
                evaluated_to_predicted.setdefault(evaluated, set()).update(match.predicted_ids)

        def analyze_agent(agent):
            aid = agent.agent_id
            if aid not in local_runs:
                return aid, Findings(findings=[AttributionFinding(
                    finding_id="missing-trace-" + aid, rubric_ids=agent.rubric_ids,
                    categories=["external_or_uncertain"], agent_ids=[aid],
                    component="trace", hypothesis="Complete local observable trace is unavailable",
                    alternatives=["The agent may not have been scheduled"], uncertainty=1.0)])
            relevant = [r for r in feedback.rubrics if
                        evaluated_to_predicted.get(r.rubric_id, set()).intersection(agent.rubric_ids)]
            # Full local I/O is retained; only indexed, connected cross-agent events are shared.
            related = [event for event in events if event.get("agent_id") == aid
                       or event.get("recipient") == aid
                       or event.get("agent_id") in agent.depends_on and event.get("kind") in {
                           "artifact", "artifact_created", "artifact_published", "handoff", "message_sent"}]
            payload = {"task": task, "agent": agent, "local_execution": local_runs[aid],
                       "feedback": relevant, "questions": outline.questions.get(aid, []),
                       "global_findings": outline.findings, "related_events": related,
                       "event_index": event_index, "submission": context["submission"],
                       "evidence_ids": context["evidence_ids"]}
            model = self.local_model_factory(aid)
            findings = self.ask(model, "attribute_local", LOCAL_ATTRIBUTION_PROMPT,
                                payload, Findings, agent_id=aid)
            requested = list(dict.fromkeys(findings.evidence_requests))
            if len(requested) > 8 or any(eid not in event_by_id for eid in requested):
                raise ValueError("Local attribution requested unavailable or too many events")
            if requested:
                findings = self.ask(model, "attribute_local_followup", LOCAL_ATTRIBUTION_PROMPT,
                                    {**payload, "prior_analysis": findings,
                                     "requested_evidence": [event_by_id[eid] for eid in requested],
                                     "remaining_exchanges": 0}, Findings, agent_id=aid)
                if findings.evidence_requests:
                    raise ValueError("Local attribution exceeded its evidence exchange limit")
            return aid, findings

        self.last_local_findings = {}
        if self.local_attribution:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = [pool.submit(analyze_agent, agent) for agent in team.agents]
                self.last_local_findings = dict(future.result() for future in futures)
        final = self.ask(self.global_model, "attribute_integrate", INTEGRATE_PROMPT,
                         {**context, "global_outline": outline,
                          "local_findings": self.last_local_findings}, Findings)
        valid_evidence = set(context["evidence_ids"])
        valid_rubrics = ({r.rubric_id for r in planned_graph.rubrics}
                         | {r.rubric_id for r in global_graph.rubrics}
                         | {r.rubric_id for r in feedback.rubrics})
        finding_ids: set[str] = set()
        for finding in final.findings:
            if finding.finding_id in finding_ids:
                raise ValueError("Duplicate attribution finding ID")
            finding_ids.add(finding.finding_id)
            if not set(finding.agent_ids) <= known_agents or not set(finding.rubric_ids) <= valid_rubrics:
                raise ValueError("Attribution references unknown agents or rubrics")
            if not set(finding.supporting_evidence + finding.opposing_evidence) <= valid_evidence:
                raise ValueError("Attribution invented an evidence reference")
            if not finding.supporting_evidence:
                finding.categories = ["external_or_uncertain"]
                finding.uncertainty = max(finding.uncertainty, 0.9)
            if not finding.success:
                refs = set(finding.rubric_ids)
                mapped = set().union(*(evaluated_to_predicted.get(rid, {rid}) for rid in refs))
                owners = set().union(*(set(team.coverage.get(rid, [])) for rid in mapped))
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
                base_version: int, experiences: Sequence = ()) -> list[ChangeProposal]:
        supported = [finding for finding in findings if finding.supporting_evidence
                     and any(category != "external_or_uncertain" for category in finding.categories)]
        if not supported:
            return []
        proposals = self.ask(self.global_model, "propose", PROPOSE_PROMPT,
                             {"task": task, "findings": supported,
                              "base_version": base_version, "experiences": experiences},
                             Proposals).proposals
        if len(proposals) > 3:
            raise ValueError("At most three single-experience proposals are allowed")
        evidence = {eid for finding in supported for eid in finding.supporting_evidence}
        counterevidence = {eid for finding in supported for eid in finding.opposing_evidence}
        existing = {entry["experience_id"]: entry for entry in as_json(experiences)}
        for proposal in proposals:
            exp = proposal.experience
            if proposal.source_task_id != task.task_id or proposal.base_version != base_version:
                raise ValueError("Proposal source task or base version changed")
            if exp.validation_status != "staged" or proposal.validation_result:
                raise ValueError("Model proposals cannot self-validate")
            if exp.source_task_ids != [task.task_id]:
                raise ValueError("Proposal must cite the observed source task")
            if not set(proposal.evidence + exp.evidence) <= evidence:
                raise ValueError("Proposal evidence is not supported by attribution")
            if not set(exp.counterevidence) <= counterevidence | evidence:
                raise ValueError("Proposal invented counterevidence")
            if exp.bank == "execution" and not exp.capability.strip():
                raise ValueError("Execution experience requires a reusable capability signature")
            if proposal.replaces_id and proposal.replaces_id not in existing:
                raise ValueError("Proposal replaces an unknown experience")
        return proposals
