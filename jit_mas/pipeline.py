"""The task-level closed loop. Private records enter only after submission."""

from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path

from pydantic import Field

from .attribution import RubricAttributor, _compact_duplicate_event_content, feedback_view
from .budget import BudgetLedger
from .config import MASConfig
from .experience import ExperienceStore, retrieve
from .planning import (GlobalAnalyzer, JsonModelCalls, knowledge_policy_prompt,
                       public_planning_context)
from .schemas import (AgentEvolutionUpdate, AgentPoolObservation, AgentPoolOperation, EvaluationFeedback,
                      ExperienceSnapshot, PlannedTeam, Prediction, PublicTask, Record, RubricFeedback,
                      RubricGraph, SplitManifest, digest, utc_now)

SUBMITTED_SOURCE_FILES = ("run_manifest.json", "frozen_plan.json", "planning_calls.json", "harness.json",
                          "execution.json", "submission.json", "evaluation.json", "budget.json")
ATTRIBUTION_SOURCE_FILES = SUBMITTED_SOURCE_FILES + (
    "attribution.json", "complete.json", "resume_provenance.json", "historical_source_budget.json")

AGENT_EVOLVE_PROMPT = """You are the persistent agent in agent_profile reflecting on your own
completed role. Improve how you perform that role using the supplied local trace,
own_events, numerical evaluation_summary, and evidence-supported attribution findings.
credit_assignment identifies the predicted requirements and related evaluated rubrics
assigned to your role in the completed MAS. Focus local improvements on this scope;
the total submission score is context, not responsibility for unrelated rubrics.
An own event with content_ref has the exact same content at the supplied local_execution
path and content_hash; inspect that complete value and cite the event_id. These references
remove duplicate copies only, not unique process evidence.
The evaluation_summary measures the submitted team's artifact, not your individual
contribution or a causal effect of your policy. It is post-submission feedback only.
You own your skills, memory, prompt, reasoning,
planning, harness, and communication habits; the Meta-Agent owns team selection and topology.
Use your observed resource_usage to identify redundant context, repeated work and
unnecessary communication. Retain conditional ways to conserve tokens while preserving
task constraints, factual checks and the complete deliverable. Token estimates and team
scores do not establish a causal quality-cost improvement; unknown prices remain unknown.
Keep identity, base_agent_version, source_task_id, and update_id exactly as supplied.
Distill short conditional process lessons, never a task answer, benchmark criterion,
reference answer, hidden evaluator threshold, or requirement to imitate one task's output.
Retained state will be used on new tasks. Preserve capabilities and useful existing skills;
when revising skills, provide the complete mapping. Optional changes may be null when
the evidence does not support a change. Lessons use the current source_task_id and exact
valid_evidence_ids. Include every lesson evidence and counterevidence ID in the update's top-level
evidence array. Cite only supplied process evidence or numerical feedback. Do not invent events or
claim that successful reflection establishes a causal improvement. If evidence is weak,
retain the existing policy and record only a cautious, conditional lesson.

Evidence whitelist (mechanical): valid_evidence_ids is the only allowed set. Before
returning, set update.evidence to a sorted, deduplicated nonempty list that contains
the union of every lesson's evidence and counterevidence, and may additionally contain
valid process observations supporting the update decision. Verify every item is in
valid_evidence_ids. Never use rubric IDs or feedback:<id> labels as evidence unless
that exact string appears in valid_evidence_ids; otherwise remove the lesson or omit
the unsupported reference. When no reliable lesson evidence remains, return lessons=[]
and preserve the existing policy; still cite one or more supplied observations in the
required top-level evidence (for example evaluation:summary or an own event) that
support retaining it, never an invented event. Extra decision evidence is allowed, but
every lesson reference must remain included in the top-level union."""


META_EVOLUTION_PROMPT = """Determine the final bi-level evolution after submission,
judge feedback, global attribution and every selected agent's local reflection.
Review the proposed meta experiences and full agent harness prototype updates together.
Meta-level experience improves rubric prediction and MAS structure design. Agent-level
updates improve role context, tools, memory, skills and installed harness policies.
Select at most one proposal_id from candidate_proposals and any supported update IDs
from agent_reflections. Use null and [] when the evidence supports no update. Preserve
the agents' submitted update contents, identities, versions and evidence. Direct updates
apply only to existing agent_pool members. Temporary reflections are retention candidates;
retain a temporary harness only through an explicit add operation in pool_operations.
During this post-execution evolution you may add, delete/prune, split, merge, specialize
or reorganize full harness prototypes. Construction and execution keep the Pool frozen.
Pool structure is optional: a flat collection or an emergent forest is valid. No seed
role is mandatory. Use historical observations, actual failures, submission feedback and
role token/tool usage to assess benefit and overhead; do not invent measured gains.
Operations may be empty. Every operation needs a unique operation_id, the current
source_task_id, base_pool_version, exact base_agent_versions for existing targets,
referenced existing parents and any child/retained ancestor affected by implicit
reparenting, evidence from valid_evidence_ids, rationale, expected_benefit
and token_cost_tradeoff. New profiles use version 1 and current-task provenance/evidence;
specialize preserves identity and its base version for the transaction to increment.
add creates profiles; delete/prune removes targets; split retains one general parent and
adds at least two children; merge replaces at least two targets with one new profile;
specialize replaces one target's complete harness; reorganize uses parent_assignments.
Include complete profile contents, retain supported inherited memories and histories,
and never fabricate experience. A temporary identity must retain its recorded role and
capabilities. Explain the joint decision in rationale.
Self-reflection and a team score do not establish causal improvement. Retain useful
existing practices and defer unsupported or conflicting changes. This determines the
contents of this evolution; it is not a held-out validation or promotion gate."""


class EvolutionDecision(Record):
    proposal_id: str | None = None
    agent_update_ids: list[str] = Field(default_factory=list)
    pool_operations: list[AgentPoolOperation] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


def submitted_source_digest(source_dir):
    source = Path(source_dir)
    names = SUBMITTED_SOURCE_FILES + tuple(name for name in ("temporary_agents.json",)
                                           if (source / name).is_file())
    return digest({name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in names})


def attribution_source_digest(source_dir):
    source = Path(source_dir)
    if source.is_file():
        if source.name != "attribution.json":
            raise ValueError("Attribution source file must be attribution.json")
        source = source.parent
    names = ATTRIBUTION_SOURCE_FILES + tuple(name for name in
        ("agent_evolution.json", "meta_evolution.json", "temporary_agents.json") if (source / name).is_file())
    return digest({name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ("jit_mas", "jit", "scripts/kernel", "scripts/models", "scripts/tools", "scripts/eval",
                      "benchmark/adapter", "harness_factory/descriptions", "harness_factory/harnesses"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix in (".py", ".yaml", ".txt", ".md"):
                files.append((path.relative_to(root).as_posix(), digest(path.read_text(encoding="utf-8"))))
    for filename in ("run_jit_mas.py", "mas_baseline_methods.py", "run_benchmark_experiment.py",
                     "run_benchmark_test.py", "prepare_benchmark_suite.py", "prepare_benchmark_evidence.py",
                     "summarize_benchmark_suite.py", "run_independent_experiment.py",
                     "preflight_independent_experiment.py", "verify_instruction_checkers.py",
                     "prepare_independent_protocol.py", "prepare_public_evidence.py"):
        path = root / "scripts" / filename
        if path.is_file():
            files.append((path.relative_to(root).as_posix(), digest(path.read_text(encoding="utf-8"))))
    return digest(files)


def input_fingerprints(task):
    fingerprints = []
    for attachment in task.attachments:
        path = Path(attachment)
        if "://" not in attachment and path.is_file():
            content = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content.update(chunk)
            fingerprints.append({"path": attachment, "hash": content.hexdigest()})
        else:
            fingerprints.append({"path": attachment, "unsnapshotted": True})
    return fingerprints


def convert_feedback(raw):
    return EvaluationFeedback(
        task_id=raw["task_id"], evaluator_version=raw["evaluator_version"],
        score=raw["score"] if raw["complete"] else None, complete=raw["complete"],
        aggregation=raw.get("aggregation", "sum(weight*score)/sum(positive weights)"),
        zero_denominator=raw.get("zero_denominator", False),
        source=raw.get("evaluator_source", "benchmark"), raw=raw,
        rubrics=[RubricFeedback(rubric_id=row["rubric_id"], criterion=row["criterion"],
                               weight=row["weight"], score=row["score"], verdict=row["verdict"],
                               reason=row.get("reasoning", ""), status=row["status"],
                               evidence=row.get("evidence_quotes", []), raw=row)
                 for row in raw["feedback"]])


class MASPipeline:
    def __init__(self, config: MASConfig, model_provider, evaluator_factory, synthesizer_factory,
                 tasks: dict[str, PublicTask], private_records: dict, manifest: SplitManifest,
                 store: ExperienceStore, output_dir, tools=None, knowledge_policy=None):
        self.config, self.models = config, model_provider
        self.evaluator_factory, self.synthesizer_factory = evaluator_factory, synthesizer_factory
        self.tasks, self.private_records, self.manifest = tasks, private_records, manifest
        self.store, self.output_dir, self.tools = store, Path(output_dir), tools or {}
        knowledge_policy_prompt(knowledge_policy)
        self.knowledge_policy = knowledge_policy
        self.code_hash = code_fingerprint()
        all_ids = set(manifest.evolution + manifest.validation + manifest.test + manifest.stream)
        if not all_ids <= tasks.keys() or not all_ids <= private_records.keys():
            raise ValueError("Every split task needs a public task and separate evaluation record")
        questions = [" ".join(tasks[key].question.split()).casefold() for key in all_ids]
        if len(questions) != len(set(questions)):
            raise ValueError("Normalized duplicate tasks across splits")

    def run_task(self, task_id, snapshot: ExperienceSnapshot, *, mode="evaluate", repeat=0,
                 attribution=False, resume=True, resume_source=None, resume_source_hash=None,
                 resume_attribution=None, resume_attribution_hash=None, defer_evaluation=False):
        split_name = {"evolve": "evolution", "validate": "validation", "evaluate": "test", "stream": "stream"}.get(mode)
        if split_name is None or task_id not in getattr(self.manifest, split_name):
            raise ValueError("Task does not belong to the explicitly selected split")
        if self.knowledge_policy == "model_general_knowledge_allowed" and (
                self.tasks[task_id].tools or self.tools or self.config.available_tools):
            raise ValueError("General-knowledge execution requires an empty external tool allowlist")
        if attribution and mode in ("validate", "evaluate"):
            raise ValueError("Held-out execution cannot perform attribution")
        if defer_evaluation and (mode != "evaluate" or attribution or resume_source is not None):
            raise ValueError("Deferred scoring is only valid for frozen test submissions")
        if type(repeat) is not int or repeat < 0:
            raise ValueError("Repeat must be a nonnegative integer")
        ledger = BudgetLedger(self.config.max_model_calls, self.config.max_total_tokens,
                              self.config.max_tool_calls, timeout_seconds=self.config.task_timeout)
        audit = {}
        try:
            if ((resume_attribution is None) != (resume_attribution_hash is None)
                    or resume_attribution is not None and resume_source is None):
                raise ValueError("Frozen attribution requires a source and an explicit attribution content hash")
            if resume_source is not None:
                if mode != "evolve" or not attribution:
                    raise ValueError("Submitted-source continuation requires evolution attribution")
                return self._resume_submitted_task(task_id, snapshot, ledger, audit,
                    resume_source, resume_source_hash, resume_attribution, resume_attribution_hash)
            return self._run_task(task_id, snapshot, ledger, audit, mode=mode, repeat=repeat,
                                  attribution=attribution, resume=resume, defer_evaluation=defer_evaluation)
        except BaseException as exc:
            run_dir = audit.get("run_dir", self.output_dir / "failures" / uuid.uuid4().hex)
            failure = {"task_id": task_id, "experience_hash": digest(snapshot), "mode": mode,
                       "error_type": type(exc).__name__, "error": str(exc), "time": utc_now(),
                       "run_dir": str(run_dir), "budget": ledger.snapshot()}
            exc.jit_mas_run_failure = {key: failure[key] for key in
                ("task_id", "mode", "run_dir", "error_type", "error", "budget")}
            for name in ("failed_attempts", "meta_trajectory", "artifact"):
                if hasattr(exc, name):
                    failure[name] = getattr(exc, name)
            if hasattr(exc, "jit_mas_failure"):
                failure["jit_mas_failure"] = exc.jit_mas_failure
            write_json(run_dir / "failure.json", failure)
            audit["run_dir"] = run_dir
            raise
        finally:
            if audit.get("run_dir") and not audit.get("cached"):
                write_json(audit["run_dir"] / "budget.json", ledger.snapshot())

    def _attribute(self, task, snapshot, ledger, run_dir, initial_graph, planned_graph, team, result, feedback,
                   pool=None):
        attributor = RubricAttributor(self.models.create("global", "global-post", ledger, "update"),
            lambda aid: self.models.create("local", aid, ledger, "update"),
            max_parallel=self.config.max_parallel, local_attribution=self.config.local_attribution)
        findings, proposals, agent_updates, complete = [], [], [], False
        pool_operations, observations = [], []
        integrated = False

        def save_attribution():
            write_json(run_dir / "attribution.json", {"complete": complete,
                "findings": [finding.model_dump(mode="json") for finding in findings],
                "alignments": {key: value.model_dump(mode="json") for key, value in attributor.last_alignments.items()},
                "credit_assignments": {key: value.model_dump(mode="json")
                                       for key, value in attributor.last_credit_assignments.items()},
                "global_outline": attributor.last_global_outline.model_dump(mode="json")
                                  if attributor.last_global_outline is not None else None,
                "evolution_integrated": integrated,
                "calls": attributor.call_records, "proposals": [proposal.model_dump(mode="json") for proposal in proposals]})

        try:
            findings = attributor.attribute(task, initial_graph, planned_graph, team, result, feedback)
            save_attribution()
            if pool is not None:
                agent_updates = self._evolve_agents(task, ledger, result, team, pool, run_dir)
            proposals = attributor.propose(task, findings, snapshot.version, snapshot.experiences,
                                          agent_reflections=agent_updates if pool is not None else None)
            if pool is not None:
                proposals, agent_updates, pool_operations, observations = self._integrate_evolution(
                    task, snapshot, ledger, run_dir, team, findings, feedback, proposals, agent_updates, result,
                    pool)
                integrated = True
            complete = True
            return proposals, agent_updates, pool_operations, observations
        finally:
            save_attribution()

    @staticmethod
    def _validate_evolution_decision(decision, proposals, updates, *, pool=None, task_id=None,
                                     observations=(), valid_ids=(), creations=()):
        if decision.proposal_id is not None and decision.proposal_id not in {
                proposal.proposal_id for proposal in proposals}:
            raise ValueError("Meta evolution selected an unknown proposal")
        if (len(decision.agent_update_ids) != len(set(decision.agent_update_ids))
                or not set(decision.agent_update_ids) <= {update.update_id for update in updates}):
            raise ValueError("Meta evolution selected unknown or duplicate agent updates")
        if pool is None:
            if decision.pool_operations:
                raise ValueError("Structural evolution requires a bound Agent Pool")
            return
        from .agent_pool import apply_evolution
        selected = [update for update in updates if update.update_id in decision.agent_update_ids]
        members = {profile.pool_agent_id for profile in pool.profiles}
        if any(update.pool_agent_id not in members for update in selected):
            raise ValueError("Temporary harness retention requires an Add operation")
        temporary = {row["agent"]["pool_agent_id"]: row for row in creations}
        for operation in decision.pool_operations:
            if not set(operation.evidence) <= set(valid_ids):
                raise ValueError("Pool operation cited unavailable evolution evidence")
            for profile in operation.profiles:
                record = temporary.get(profile.pool_agent_id)
                if record is not None:
                    original = record["agent"]["temporary_profile"]
                    if (operation.kind != "add" or not record["selected"]
                            or profile.role != original["role"]
                            or profile.capabilities != original["capabilities"]):
                        raise ValueError("Temporary retention must bind an executed creation's role and capabilities")
        apply_evolution(pool, selected, decision.pool_operations, observations,
                        source_task_id=task_id)

    @staticmethod
    def _pool_observations(task, team, result, feedback):
        _, _, events = RubricAttributor._execution_view(result)
        observations = []
        for agent in team.agents:
            usage = [event for event in events if event.get("agent_id") == agent.agent_id
                     and event.get("kind") == "resource_usage"]
            observations.append(AgentPoolObservation(task_id=task.task_id, agent_id=agent.agent_id,
                pool_agent_id=agent.pool_agent_id, profile_version=agent.pool_agent_version,
                temporary=agent.temporary_profile is not None, submission_score=feedback.score,
                complete=feedback.complete,
                input_tokens=sum(event["content"].get("input_tokens", 0) for event in usage),
                output_tokens=sum(event["content"].get("output_tokens", 0) for event in usage),
                tool_calls=sum(event["content"].get("tool_calls", 0) for event in usage),
                cost=sum(event["content"]["cost"] for event in usage)
                     if usage and all(event["content"].get("cost") is not None for event in usage) else None,
                usage_available=bool(usage),
                usage_estimated=not usage or any(event["content"].get("estimated", True) for event in usage),
                evidence=["planning:team", "evaluation:summary", *[event["event_id"] for event in usage]]))
        return observations

    @staticmethod
    def _pool_evidence(pool, result, feedback, creations):
        _, _, events = RubricAttributor._execution_view(result)
        valid = {event["event_id"] for event in events}
        valid.update({"planning:global", "planning:planned", "planning:team", "evaluation:summary",
                      "submission:" + digest(result.answer if hasattr(result, "answer") else result["answer"])})
        valid.update("feedback:" + rubric.rubric_id for rubric in feedback.rubrics)
        valid.update("temporary:" + row["agent"]["pool_agent_id"] for row in creations)
        for observation in pool.observations:
            valid.update(observation.evidence)
        for operation in pool.structural_history:
            valid.update(operation.evidence)
        for profile in pool.profiles:
            valid.update(profile.evidence)
            for lesson in profile.memory:
                valid.update(lesson.evidence + lesson.counterevidence)
        return valid

    @staticmethod
    def _retention_candidates(team, updates, task_id):
        by_member = {update.pool_agent_id: update for update in updates}
        candidates = []
        for agent in team.agents:
            if agent.temporary_profile is None:
                continue
            profile = agent.temporary_profile.model_copy(deep=True)
            update = by_member[profile.pool_agent_id]
            profile.memory = [lesson.model_copy(deep=True) for lesson in update.lessons]
            for field in ("prompt", "skills", "preferred_tools", "reasoning_strategy",
                          "planning_strategy", "communication", "harness"):
                value = getattr(update, field)
                if value is not None:
                    setattr(profile, field, value)
            profile.source_task_ids = [task_id]
            profile.evidence = list(update.evidence)
            candidates.append(profile)
        return candidates

    def _integrate_evolution(self, task, snapshot, ledger, run_dir, team, findings, feedback, proposals, updates, result, pool):
        from .agent_pool import effective_pool
        pool = effective_pool(pool)
        observations = self._pool_observations(task, team, result, feedback)
        creation_path = run_dir / "temporary_agents.json"
        creations = json.loads(creation_path.read_text(encoding="utf-8"))["creations"] if creation_path.is_file() else []
        valid_ids = self._pool_evidence(pool, result, feedback, creations)
        caller = JsonModelCalls(max_corrections=1)
        decision, complete = None, False
        try:
            decision = caller.ask(self.models.create("global", "meta-evolution", ledger, "update"),
                "evolution_integrate", META_EVOLUTION_PROMPT,
                {"task": task, "team": team, "base_version": snapshot.version,
                 "findings": findings, "candidate_proposals": proposals, "agent_reflections": updates,
                 "source_task_id": task.task_id, "agent_pool": pool,
                 "base_pool_version": pool.version, "pool_observations": observations,
                 "temporary_agents": creations, "valid_evidence_ids": sorted(valid_ids),
                 "retention_candidates": self._retention_candidates(team, updates, task.task_id),
                 "evaluation_summary": self._agent_reflection_evidence("", [], feedback)["evaluation_summary"]},
                EvolutionDecision, validate=lambda item: self._validate_evolution_decision(item, proposals, updates,
                    pool=pool, task_id=task.task_id, observations=observations,
                    valid_ids=valid_ids, creations=creations))
            complete = True
            return ([proposal for proposal in proposals if proposal.proposal_id == decision.proposal_id],
                    [update for update in updates if update.update_id in decision.agent_update_ids],
                    decision.pool_operations, observations)
        finally:
            write_json(run_dir / "meta_evolution.json", {"complete": complete,
                "candidate_proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                "agent_reflections": [update.model_dump(mode="json") for update in updates],
                "base_pool_hash": digest(pool),
                "pool_observations": [item.model_dump(mode="json") for item in observations],
                "decision": decision.model_dump(mode="json") if decision is not None else None,
                "calls": caller.call_records})

    def _evolve_agents(self, task, ledger, result, team, pool, run_dir):
        """Ask each selected pooled agent to distill evidence-scoped process lessons."""
        if pool is None:
            return []
        from .agent_pool import resolve_profile
        from .schemas import AttributionFinding
        attributed = json.loads((run_dir / "attribution.json").read_text(encoding="utf-8"))
        attributed_findings = [AttributionFinding.model_validate(item) for item in attributed["findings"]]
        feedback = EvaluationFeedback.model_validate_json((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        reflection_calls = []
        runs = result.full_dict() if hasattr(result, "full_dict") else result
        by_agent = {}
        def collect(run):
            aid = run.get("metadata", {}).get("agent_id")
            if aid:
                by_agent[aid] = run
            for child in run.get("sub_runs", []):
                collect(child)
        collect(runs)
        event_index = runs.get("metadata", {}).get("events", [])
        from .schemas import RubricCreditAssignment
        assignments = {key: RubricCreditAssignment.model_validate(value)
                       for key, value in attributed["credit_assignments"].items()}
        planned_graph = RubricGraph.model_validate(
            json.loads((run_dir / "frozen_plan.json").read_text(encoding="utf-8"))["R_planned"])
        updates = []
        for agent in team.agents:
            profile = resolve_profile(pool, agent)
            local = by_agent.get(agent.agent_id, {})
            assignment = assignments[agent.agent_id]
            evidence = self._agent_reflection_evidence(agent.agent_id, event_index, feedback, local,
                                                       evaluated_rubric_ids=assignment.evaluated_rubric_ids)
            valid_ids = set(evidence["valid_evidence_ids"])
            findings = [finding for finding in attributed_findings
                        if agent.agent_id in finding.agent_ids]
            payload = {"task": task, "agent": agent, "team": team, "agent_profile": profile,
                       "local_execution": local, "attribution_findings": findings,
                       "credit_assignment": assignment,
                       "assigned_predicted_rubrics": [rubric for rubric in planned_graph.rubrics
                                                      if rubric.rubric_id in assignment.predicted_rubric_ids],
                       **evidence,
                       "base_agent_version": profile.version, "source_task_id": task.task_id,
                       "update_id": digest({"task_id": task.task_id, "agent_id": agent.agent_id,
                                             "base_agent_version": profile.version,
                                             "evidence": sorted(valid_ids)})}
            model = self.models.create("local", agent.agent_id, ledger, "update")
            caller = JsonModelCalls(max_corrections=1)
            try:
                update = caller.ask(model, "agent_evolve", AGENT_EVOLVE_PROMPT, payload,
                                    AgentEvolutionUpdate, agent_id=agent.agent_id,
                                    validate=lambda item, p=profile, ids=valid_ids, uid=payload["update_id"]:
                                        self._validate_agent_update(item, p, task.task_id, ids, uid))
            finally:
                reflection_calls.extend(caller.call_records)
                write_json(run_dir / "agent_evolution.json", {"complete": False,
                    "calls": reflection_calls, "updates": [item.model_dump(mode="json") for item in updates]})
            updates.append(update)
        write_json(run_dir / "agent_evolution.json", {"complete": True,
            "calls": reflection_calls, "updates": [item.model_dump(mode="json") for item in updates]})
        return updates

    @staticmethod
    def _agent_reflection_evidence(agent_id, events, feedback, local_execution=None, *, evaluated_rubric_ids=None):
        own_events = [event for event in events if event.get("agent_id") == agent_id]
        summary = {"evidence_id": "evaluation:summary", "scope": "submission",
                   "score": feedback.score, "complete": feedback.complete,
                   "evaluator_version": feedback.evaluator_version, "aggregation": feedback.aggregation,
                   "causal_limit": "Team submission feedback does not establish individual contribution or causal improvement.",
                   "rubrics": [{"evidence_id": "feedback:" + rubric.rubric_id,
                                "rubric_id": rubric.rubric_id, "weight": rubric.weight,
                                "score": rubric.score,
                                "verdict": rubric.verdict if rubric.verdict.casefold() in
                                    {"yes", "no", "abstain", "pass", "fail", "true", "false", "missing", "error",
                                     "satisfied", "not satisfied"}
                                    else "unavailable",
                                "status": rubric.status}
                               for rubric in feedback.rubrics
                               if evaluated_rubric_ids is None or rubric.rubric_id in evaluated_rubric_ids]}
        valid_ids = {event["event_id"] for event in own_events if event.get("event_id")}
        valid_ids.update({"planning:team", summary["evidence_id"]})
        valid_ids.update(rubric["evidence_id"] for rubric in summary["rubrics"])
        return {"own_events": _compact_duplicate_event_content(own_events, local_execution),
                "evaluation_summary": summary,
                "resource_usage": [event["content"] for event in own_events
                                   if event.get("kind") == "resource_usage"],
                "valid_evidence_ids": sorted(valid_ids)}

    @staticmethod
    def _validate_agent_update(update, profile, task_id, valid_ids, update_id):
        if (update.pool_agent_id != profile.pool_agent_id or update.base_agent_version != profile.version
                or update.source_task_id != task_id or update.update_id != update_id):
            raise ValueError("Agent evolution changed immutable pool identity or source task")
        if not set(update.evidence) <= valid_ids:
            raise ValueError("Agent evolution cited unavailable process evidence")
        for lesson in update.lessons:
            if lesson.source_task_ids != [task_id]:
                raise ValueError(f"Agent memory lesson {lesson.lesson_id} invented source tasks; "
                                 f"expected_source_task_ids={[task_id]}")
            lesson_ids = set(lesson.evidence + lesson.counterevidence)
            invalid = sorted(lesson_ids - valid_ids)
            if invalid:
                raise ValueError(f"Agent memory lesson {lesson.lesson_id} cites unavailable evidence; "
                                 f"invalid_evidence_ids={invalid}. Correct or remove the lesson.")
            update.evidence = sorted(set(update.evidence) | lesson_ids)

    def _validate_creation_audit(self, documents, task, team):
        from .agent_pool import resolve_profile
        from .schemas import AgentPoolSnapshot

        predictions = [call for call in documents["planning_calls.json"]
                       if call.get("phase") == "predict" and not call.get("validation_errors")]
        prediction = Prediction.model_validate(json.loads(JsonModelCalls._response_json_text(
            predictions[-1]["response"]))) if predictions else None
        candidates = [agent for agent in prediction.candidates if agent.temporary_profile is not None] \
            if prediction is not None else []
        audit = documents.get("temporary_agents.json")
        if audit is None:
            if candidates or any(agent.temporary_profile is not None for agent in team.agents):
                raise ValueError("Submitted temporary harnesses require their creation audit")
            return
        pool_data = documents["frozen_plan.json"].get("AgentPool")
        if pool_data is None:
            raise ValueError("Temporary creation audit lacks a frozen Agent Pool")
        pool = AgentPoolSnapshot.model_validate(pool_data)
        analyzer = GlobalAnalyzer(None, None, agent_pool=pool, knowledge_policy=self.knowledge_policy)
        selected = {agent.agent_id: agent for agent in team.agents}
        expected = []
        for candidate in candidates:
            resolve_profile(pool, candidate)
            analyzer._set_internal_policy(candidate)
            expected.append({"task_id": task.task_id, "agent_id": candidate.agent_id,
                "selected": candidate.agent_id in selected, "agent": candidate.model_dump(mode="json"),
                "creation_evidence": ["planning:global", "planning:planned"],
                "baseline_pool_hash": digest(pool)})
            final = selected.get(candidate.agent_id)
            if final is not None and any(getattr(final, field) != getattr(candidate, field)
                    for field in ("pool_agent_id", "pool_agent_version", "temporary_profile", "creation_rationale")):
                raise ValueError("Temporary creation differs from the selected frozen harness")
        if audit != {"task_id": task.task_id, "creations": expected}:
            raise ValueError("Temporary creation audit differs from the recorded planning candidates")

    def _load_frozen_attribution(self, directory, anchor, source_hash, source_documents,
                                task, snapshot, initial, planned, team, result, feedback):
        from .schemas import AttributionFinding, ChangeProposal, RubricAlignment

        location = Path(directory).resolve()
        if attribution_source_digest(location) != anchor:
            raise ValueError("Frozen-attribution content hash mismatch")
        if location.is_file():
            location = location.parent
        documents = {name: json.loads((location / name).read_text(encoding="utf-8"))
                     for name in ATTRIBUTION_SOURCE_FILES}
        temporary_path = location / "temporary_agents.json"
        if temporary_path.is_file():
            documents["temporary_agents.json"] = json.loads(temporary_path.read_text(encoding="utf-8"))
        if documents.get("temporary_agents.json") != source_documents.get("temporary_agents.json"):
            raise ValueError("Frozen temporary creation audit differs from the submitted source")
        self._validate_creation_audit(documents, task, team)
        attributed, complete = documents["attribution.json"], documents["complete.json"]
        provenance, manifest = documents["resume_provenance.json"], documents["run_manifest.json"]
        if (attributed.get("complete") is not True or complete.get("source_attribution_reused")
                or attributed.get("proposals") != complete.get("proposals")
                or complete.get("task_id") != task.task_id or complete.get("mode") != "evolve"
                or complete.get("backend") != self.config.backend
                or complete.get("experience_hash") != digest(snapshot)
                or complete.get("experience_version") != snapshot.version
                or complete.get("answer_hash") != source_documents["submission.json"]["answer_hash"]
                or complete.get("evaluation") != feedback.model_dump(mode="json")
                or complete.get("plan_hashes") != source_documents["frozen_plan.json"]["hashes"]
                or complete.get("resume_provenance") != provenance
                or manifest.get("continuation") != provenance
                or manifest.get("comparison") != source_documents["run_manifest.json"]["comparison"]
                or provenance.get("source_hash") != source_hash
                or provenance.get("source_regenerated") is not False
                or provenance.get("source_rejudged") is not False
                or not provenance.get("continuation_code_hash")):
            raise ValueError("Frozen-attribution source/proposal/baseline binding mismatch")
        for name in SUBMITTED_SOURCE_FILES:
            if name not in {"budget.json", "run_manifest.json"} and documents[name] != source_documents[name]:
                raise ValueError("Frozen attribution belongs to different source evidence")
        if (documents["historical_source_budget.json"] != source_documents["budget.json"]
                or complete.get("historical_source_budget") != source_documents["budget.json"]):
            raise ValueError("Frozen-attribution historical source budget mismatch")
        budget = documents["budget.json"]
        if (set(budget["by_stage"]) - {"update"} or budget["reserved_tokens"]
                or any(budget[key] != complete["budget"][key]
                       for key in ("model_calls", "tokens", "tool_calls", "records"))):
            raise ValueError("Frozen-attribution budget is not a settled attribution-only budget")
        known_agents = {agent.agent_id for agent in team.agents}
        evaluated_ids = {rubric.rubric_id for rubric in feedback.rubrics}
        initial_ids = {rubric.rubric_id for rubric in initial.rubrics}
        planned_ids = {rubric.rubric_id for rubric in planned.rubrics}
        for label, predicted_ids in (("global", initial_ids), ("planned", planned_ids)):
            alignment = RubricAlignment.model_validate(attributed["alignments"][label])
            for match in alignment.matches:
                if (not match.predicted_ids or not match.evaluated_ids
                        or not set(match.predicted_ids) <= predicted_ids
                        or not set(match.evaluated_ids) <= evaluated_ids):
                    raise ValueError("Frozen-attribution alignment references unknown rubrics")
            matched_predicted = {rid for match in alignment.matches for rid in match.predicted_ids}
            matched_evaluated = {rid for match in alignment.matches for rid in match.evaluated_ids}
            if (set(alignment.unmatched_predicted_ids) != predicted_ids - matched_predicted
                    or set(alignment.missed_evaluated_ids) != evaluated_ids - matched_evaluated):
                raise ValueError("Frozen-attribution unmatched rubric bookkeeping mismatch")
        _, _, events = RubricAttributor._execution_view(result)
        event_ids = {event["event_id"] for event in events}
        if len(event_ids) != len(events):
            raise ValueError("Frozen-attribution source has duplicate event IDs")
        evidence = event_ids | {"submission:" + digest(result["answer"]),
            "planning:global", "planning:planned", "planning:team"} | {"feedback:" + rid for rid in evaluated_ids}
        findings = [AttributionFinding.model_validate(row) for row in attributed["findings"]]
        if len({finding.finding_id for finding in findings}) != len(findings):
            raise ValueError("Frozen attribution contains duplicate finding IDs")
        for finding in findings:
            if (not set(finding.agent_ids) <= known_agents
                    or not set(finding.rubric_ids) <= initial_ids | planned_ids | evaluated_ids
                    or not set(finding.supporting_evidence + finding.opposing_evidence) <= evidence):
                raise ValueError("Frozen-attribution finding invented an evidence/agent/rubric reference")
        supported = [finding for finding in findings if finding.supporting_evidence
                     and any(category != "external_or_uncertain" for category in finding.categories)]
        scoring_context = {"task_id": feedback.task_id, "criteria": feedback_view(feedback)["rubrics"]}
        proposals = [ChangeProposal.model_validate(row) for row in attributed["proposals"]]
        if len({proposal.proposal_id for proposal in proposals}) != len(proposals):
            raise ValueError("Frozen attribution contains duplicate proposal IDs")
        RubricAttributor._validate_proposals(proposals, task, supported, snapshot.version, snapshot.experiences,
                                            scoring_context=scoring_context)
        if [proposal.model_dump(mode="json") for proposal in proposals] != attributed["proposals"]:
            raise ValueError("Frozen-attribution proposal records were not canonical validated outputs")
        agent_updates = [AgentEvolutionUpdate.model_validate(item) for item in complete.get("agent_updates", [])]
        operations, observations = [], []
        if self.config.evolving_agent_pool and any(agent.pool_agent_id for agent in team.agents):
            from .attribution import AttributionOutline, build_credit_assignments
            from .agent_pool import resolve_profile
            from .schemas import AgentPoolSnapshot
            frozen_pool = source_documents["frozen_plan.json"].get("AgentPool")
            reflection_path = location / "agent_evolution.json"
            if frozen_pool is None or not reflection_path.is_file():
                raise ValueError("Pooled frozen attribution requires recorded self-evolution updates")
            reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
            candidate_updates = [AgentEvolutionUpdate.model_validate(item) for item in reflection.get("updates", [])]
            if reflection.get("complete") is not True or len(candidate_updates) != len(team.agents):
                raise ValueError("Frozen self-evolution did not collect every selected agent")
            outline = AttributionOutline.model_validate(attributed.get("global_outline") or {})
            assignments = build_credit_assignments(planned, team,
                RubricAlignment.model_validate(attributed["alignments"]["planned"]), feedback,
                meta_assignments=outline.rubric_assignments,
                meta_assignment_rationales=outline.rubric_assignment_rationale)
            recorded_assignments = attributed.get("credit_assignments")
            if recorded_assignments is not None and recorded_assignments != {
                    key: value.model_dump(mode="json") for key, value in assignments.items()}:
                raise ValueError("Frozen rubric credit assignment differs from the executed MAS")
            if attributed.get("evolution_integrated"):
                decision_path = location / "meta_evolution.json"
                if not decision_path.is_file():
                    raise ValueError("Frozen bi-level evolution lacks its final meta decision")
                meta = json.loads(decision_path.read_text(encoding="utf-8"))
                if meta.get("complete") is not True or meta.get("agent_reflections") != reflection["updates"]:
                    raise ValueError("Frozen meta evolution differs from the collected reflections")
                candidate_proposals = [ChangeProposal.model_validate(item) for item in meta["candidate_proposals"]]
                RubricAttributor._validate_proposals(candidate_proposals, task, supported,
                    snapshot.version, snapshot.experiences, scoring_context=scoring_context,
                    allowed_banks=["rubric", "organization"])
                decision = EvolutionDecision.model_validate(meta["decision"])
                creation_path = location / "temporary_agents.json"
                creations = json.loads(creation_path.read_text(encoding="utf-8")).get("creations", []) \
                    if creation_path.is_file() else []
                replay_pool = AgentPoolSnapshot.model_validate(frozen_pool)
                replay_observations = [AgentPoolObservation.model_validate(item)
                                       for item in meta.get("pool_observations", [])]
                if "base_pool_hash" in meta and meta["base_pool_hash"] != digest(replay_pool):
                    raise ValueError("Frozen evolution baseline Pool hash mismatch")
                if ("pool_observations" in meta
                        and replay_observations != self._pool_observations(task, team, result, feedback)):
                    raise ValueError("Frozen Pool observations differ from actual role execution and feedback")
                if (decision.pool_operations != [AgentPoolOperation.model_validate(item)
                        for item in complete.get("pool_operations", [])]
                        or replay_observations != [AgentPoolObservation.model_validate(item)
                            for item in complete.get("pool_observations", [])]):
                    raise ValueError("Frozen Pool evolution contents differ from the final meta decision")
                operations, observations = decision.pool_operations, replay_observations
                valid_ids = self._pool_evidence(replay_pool, result, feedback, creations)
                self._validate_evolution_decision(decision, candidate_proposals, candidate_updates,
                    pool=replay_pool, task_id=task.task_id, observations=replay_observations,
                    valid_ids=valid_ids, creations=creations)
                selected_proposals = [proposal for proposal in candidate_proposals
                                      if proposal.proposal_id == decision.proposal_id]
                selected_updates = [update for update in candidate_updates
                                    if update.update_id in decision.agent_update_ids]
                if selected_proposals != proposals or selected_updates != agent_updates:
                    raise ValueError("Frozen evolution contents differ from the final meta decision")
            elif (reflection.get("updates") != complete.get("agent_updates")
                    or complete.get("pool_operations") or complete.get("pool_observations")):
                raise ValueError("Frozen self-evolution records differ from the completed updates")
            pool = AgentPoolSnapshot.model_validate(frozen_pool)
            by_member = {item.pool_agent_id: item for item in candidate_updates}
            if len(by_member) != len(candidate_updates):
                raise ValueError("Frozen agent evolution contains duplicate members")
            for agent in team.agents:
                profile = resolve_profile(pool, agent)
                local_ids = set(self._agent_reflection_evidence(agent.agent_id, events, feedback,
                    evaluated_rubric_ids=assignments[agent.agent_id].evaluated_rubric_ids
                    if recorded_assignments is not None else None)["valid_evidence_ids"])
                expected_id = digest({"task_id": task.task_id, "agent_id": agent.agent_id,
                                      "base_agent_version": profile.version, "evidence": sorted(local_ids)})
                item = by_member.get(profile.pool_agent_id)
                if item is None:
                    raise ValueError("Frozen agent evolution omitted a selected pool member")
                self._validate_agent_update(item, profile, task.task_id, local_ids, expected_id)
        elif agent_updates or complete.get("pool_operations") or complete.get("pool_observations"):
            raise ValueError("Frozen agent evolution lacks a bound Agent Pool source")
        return (location, attributed, proposals, budget, provenance["continuation_code_hash"],
                agent_updates, operations, observations)

    def _resume_submitted_task(self, task_id, snapshot, ledger, audit, source_dir, source_hash,
                               attribution_dir=None, attribution_hash=None):
        from benchmark.adapter.researchrubrics import official_compliance_score
        from .bridge import SynthesizedHarness
        from .execution import content_hash
        from .schemas import TeamSpec

        source = Path(source_dir).resolve()
        if not source_hash or submitted_source_digest(source) != source_hash:
            raise ValueError("Submitted-source content hash mismatch")
        if snapshot.version != 0 or snapshot.experiences or snapshot.applied_proposals:
            raise ValueError("Submitted-source continuation requires a fresh empty baseline")
        documents = {name: json.loads((source / name).read_text(encoding="utf-8"))
                     for name in SUBMITTED_SOURCE_FILES}
        if (source / "temporary_agents.json").is_file():
            documents["temporary_agents.json"] = json.loads((source / "temporary_agents.json").read_text(encoding="utf-8"))
        manifest, frozen = documents["run_manifest.json"], documents["frozen_plan.json"]
        comparison, task = manifest["comparison"], self.tasks[task_id]
        expected = {"config": self.config.model_dump(mode="json"), "task": task.model_dump(mode="json"),
                    "private_hash": digest(self.private_records[task_id]), "repeat": 0,
                    "manifest": digest(self.manifest), "tools": sorted(self.tools),
                    "attachments": input_fingerprints(task)}
        if comparison.get("knowledge_policy") != self.knowledge_policy:
            raise ValueError("Submitted-source knowledge policy mismatch")
        if any(comparison.get(key) != value for key, value in expected.items()):
            raise ValueError("Submitted-source task/config/private/split binding mismatch")
        if (task_id not in self.manifest.evolution or manifest.get("mode") != "evolve"
                or manifest.get("backend") != self.config.backend
                or manifest.get("experience_hash") != digest(snapshot)
                or manifest.get("experience_version") != snapshot.version
                or frozen.get("experience_hash") != digest(snapshot)):
            raise ValueError("Submitted-source evolution baseline mismatch")
        for key in ("R_global", "R_planned", "TeamSpec"):
            if frozen.get("hashes", {}).get(key) != digest(frozen[key]):
                raise ValueError("Submitted-source frozen plan hash mismatch")
        initial = RubricGraph.model_validate(frozen["R_global"])
        planned = RubricGraph.model_validate(frozen["R_planned"])
        team = TeamSpec.model_validate(frozen["TeamSpec"])
        self._validate_creation_audit(documents, task, team)
        artifact_data = documents["harness.json"]
        artifact = SynthesizedHarness(**{**artifact_data, "path": Path(artifact_data["path"])})
        artifact.verify_integrity()
        sidecar = artifact.sidecar
        if (artifact.backend != self.config.backend or artifact.task_hash != content_hash(task)
                or artifact.team_hash != content_hash(team) or sidecar.get("task") != task.model_dump(mode="json")
                or sidecar.get("team") != team.model_dump(mode="json")
                or sidecar.get("rubrics") != planned.model_dump(mode="json") or sidecar.get("experiences") != []):
            raise ValueError("Submitted-source harness sidecar binding mismatch")
        if any(agent.pool_agent_id for agent in team.agents):
            from .schemas import AgentPoolSnapshot
            from .agent_pool import validate_bindings
            if (frozen.get("hashes", {}).get("AgentPool") != digest(frozen.get("AgentPool"))
                    or frozen.get("AgentPool") != sidecar.get("agent_pool")):
                raise ValueError("Submitted-source frozen Agent Pool binding mismatch")
            validate_bindings(AgentPoolSnapshot.model_validate(frozen["AgentPool"]), team)
        result, submission = documents["execution.json"], documents["submission.json"]
        metadata = result.get("metadata", {})
        if (result.get("terminated_reason") != "final_answer" or result.get("answer") != submission.get("answer")
                or submission.get("answer_hash") != digest(result.get("answer"))
                or metadata.get("team_hash") != artifact.team_hash or metadata.get("harness_hash") != artifact.code_hash
                or metadata.get("backend") != self.config.backend):
            raise ValueError("Submitted-source answer/execution binding mismatch")
        feedback = EvaluationFeedback.model_validate(documents["evaluation.json"])
        evaluator_id = self.evaluator_factory(None).evaluator_version
        if (not feedback.complete or feedback.task_id != task_id or feedback.score is None
                or feedback.evaluator_version != comparison.get("evaluator")
                or feedback.evaluator_version != evaluator_id
                or convert_feedback(feedback.raw).model_dump(mode="json") != feedback.model_dump(mode="json")):
            raise ValueError("Submitted-source evaluation identity or completeness mismatch")
        recorded_submission_hash = "submission_answer_hash" in feedback.raw
        if recorded_submission_hash and feedback.raw["submission_answer_hash"] != submission["answer_hash"]:
            raise ValueError("Submitted-source evaluation submission hash mismatch")
        rows = feedback.raw["feedback"]
        official = self.private_records[task_id]["rubrics"]
        if (len(rows) != len(official) or any(
                row.get("rubric_id") != rubric["rubric_id"] or row.get("criterion") != rubric["criterion"]
                or row.get("weight") != rubric["weight"] or row.get("status") != "ok"
                or not row.get("success") or row.get("score") not in (0, 1)
                or row.get("verdict") != ("Satisfied" if row.get("score") == 1 else "Not Satisfied")
                for row, rubric in zip(rows, official))
                or feedback.score != official_compliance_score(rows)):
            raise ValueError("Submitted-source official criterion evidence mismatch")
        frozen_attribution = (self._load_frozen_attribution(attribution_dir, attribution_hash, source_hash,
            documents, task, snapshot, initial, planned, team, result, feedback)
            if attribution_dir is not None else None)
        run_key = digest({"source": source_hash, "code": self.code_hash, "attempt": uuid.uuid4().hex})
        run_dir = self.output_dir / run_key
        audit["run_dir"] = run_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance = {"source_dir": str(source), "source_hash": source_hash,
            "source_code_hash": comparison["code"], "continuation_code_hash": self.code_hash,
            "source_submission_reused": True, "source_regenerated": False, "source_rejudged": False,
            "source_attribution_reused": frozen_attribution is not None,
            "evaluation_has_recorded_submission_hash": recorded_submission_hash,
            "evaluation_submission_binding": (
                "Recorded evaluation submission hash verified against the immutable submitted answer"
                if recorded_submission_hash else
                "Legacy evaluation has no recorded submission hash; explicit caller anchor trusted"),
            "file_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                            for name in SUBMITTED_SOURCE_FILES}}
        write_json(run_dir / "resume_provenance.json", provenance)
        write_json(run_dir / "run_manifest.json", {**manifest, "run_key": run_key, "continuation": provenance})
        for name in SUBMITTED_SOURCE_FILES:
            destination = "historical_source_budget.json" if name == "budget.json" else name
            if name != "run_manifest.json":
                (run_dir / destination).write_bytes((source / name).read_bytes())
        if (source / "temporary_agents.json").is_file():
            (run_dir / "temporary_agents.json").write_bytes((source / "temporary_agents.json").read_bytes())
        if frozen_attribution is None:
            pool = None
            if self.config.evolving_agent_pool and any(agent.pool_agent_id for agent in team.agents):
                from .schemas import AgentPoolSnapshot
                pool = AgentPoolSnapshot.model_validate(sidecar["agent_pool"])
            proposals, agent_updates, pool_operations, pool_observations = self._attribute(
                task, snapshot, ledger, run_dir, initial, planned, team, result, feedback, pool=pool)
        else:
            (attribution_location, _, proposals, prior_attribution_budget, attribution_code,
             agent_updates, pool_operations, pool_observations) = frozen_attribution
            provenance.update(attribution_source_dir=str(attribution_location), attribution_source_hash=attribution_hash,
                              attribution_code_hash=attribution_code, attribution_regenerated=False)
            (run_dir / "attribution.json").write_bytes((attribution_location / "attribution.json").read_bytes())
            for name in ("agent_evolution.json", "meta_evolution.json"):
                if (attribution_location / name).is_file():
                    (run_dir / name).write_bytes((attribution_location / name).read_bytes())
            write_json(run_dir / "historical_attribution_budget.json", prior_attribution_budget)
            write_json(run_dir / "resume_provenance.json", provenance)
            write_json(run_dir / "run_manifest.json", {**manifest, "run_key": run_key, "continuation": provenance})
            if attribution_source_digest(attribution_location) != attribution_hash:
                raise ValueError("Frozen-attribution files changed during continuation")
        if submitted_source_digest(source) != source_hash:
            raise ValueError("Submitted-source files changed during attribution continuation")
        budget = ledger.snapshot()
        outcome = {"run_key": run_key, "task_id": task_id, "mode": "evolve", "backend": self.config.backend,
            "software_test_only": self.config.backend == "scripted", "resumed": False,
            "source_submission_reused": True, "resume_provenance": provenance,
            "source_attribution_reused": frozen_attribution is not None,
            "historical_source_budget": documents["budget.json"],
            "experience_version": snapshot.version, "experience_hash": digest(snapshot),
            "comparison_fingerprint": digest(comparison), "answer_hash": submission["answer_hash"],
            "submitted_at": submission["submitted_at"], "source_evaluation_reused": True,
            "plan_hashes": frozen["hashes"],
            "evaluation": feedback.model_dump(mode="json"), "budget": budget,
            "proposals": [p.model_dump(mode="json") for p in proposals],
            "agent_updates": [item.model_dump(mode="json") for item in agent_updates],
            "pool_operations": [item.model_dump(mode="json") for item in pool_operations],
            "pool_observations": [item.model_dump(mode="json") for item in pool_observations],
            "run_dir": str(run_dir),
            "uncontrolled_variation": ["Source and any explicitly reused attribution retain their recorded historical code"],
            "costs": {"inference": {}, "external_evaluation": {},
                      "experience_update": budget["by_stage"].get("update", {})}}
        if frozen_attribution is None:
            outcome["attribution_continued_at"] = utc_now()
        else:
            outcome["attribution_reused_at"] = utc_now()
            outcome["historical_attribution_budget"] = prior_attribution_budget
        write_json(run_dir / "complete.json", outcome)
        return outcome

    def _run_task(self, task_id, snapshot, ledger, audit, *, mode, repeat, attribution, resume,
                  defer_evaluation=False):
        from .execution import TeamExecutor

        task = self.tasks[task_id]
        def model(role, agent_id, stage):
            return self.models.create(role, agent_id, ledger, stage)

        evaluator = self.evaluator_factory(model("judge", "judge", "evaluation"))
        comparison = {"config": self.config.model_dump(mode="json"), "code": self.code_hash,
                      "evaluator": evaluator.evaluator_version, "task": task.model_dump(mode="json"),
                      "private_hash": digest(self.private_records[task_id]), "repeat": repeat,
                      "manifest": digest(self.manifest), "tools": sorted(self.tools),
                      "attachments": input_fingerprints(task)}
        if self.knowledge_policy is not None:
            comparison["knowledge_policy"] = self.knowledge_policy
        comparison_hash = digest(comparison)
        run_key = digest({"comparison": comparison_hash, "snapshot": digest(snapshot),
                          "mode": mode, "attribution": attribution,
                          **({"defer_evaluation": True} if defer_evaluation else {})})
        cacheable = (self.config.backend == "scripted" or
                     (not task.tools and not any(a.get("unsnapshotted") for a in comparison["attachments"])))
        if not cacheable or not resume:
            run_key = digest({"identity": run_key, "attempt": uuid.uuid4().hex})
        run_dir = self.output_dir / run_key
        audit["run_dir"] = run_dir
        complete_path = run_dir / "complete.json"
        if resume and cacheable and complete_path.is_file():
            cached = json.loads(complete_path.read_text(encoding="utf-8"))
            if cached["run_key"] != run_key:
                raise ValueError("Cached run identity mismatch")
            cached["resumed"] = True
            audit["cached"] = True
            return cached
        if run_dir.exists():
            run_key = digest({"identity": run_key, "retry": uuid.uuid4().hex})
            run_dir = self.output_dir / run_key
            audit["run_dir"] = run_dir
            complete_path = run_dir / "complete.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_manifest.json", {"comparison": comparison,
            "experience_version": snapshot.version, "experience_hash": digest(snapshot),
            "backend": self.config.backend, "mode": mode, "run_key": run_key})
        experience = retrieve(snapshot, task, excluded_task_ids=self.manifest.validation + self.manifest.test)
        if not self.config.persistent_experience:
            experience = []
        pool = None
        if self.config.evolving_agent_pool and self.config.fixed_team is None:
            from .agent_pool import effective_pool, seed_pool, scope_pool
            pool = effective_pool(snapshot.agent_pool) if self.config.persistent_experience else seed_pool()
            excluded = set(self.manifest.validation + self.manifest.test)
            if any(excluded.intersection(profile.source_task_ids)
                   or any(excluded.intersection(lesson.source_task_ids) for lesson in profile.memory)
                   for profile in pool.profiles):
                raise ValueError("Agent Pool contains held-out source history")
            pool = scope_pool(pool, task, excluded_task_ids=excluded)
        analyzer = GlobalAnalyzer(model("global", "global", "inference"),
                                  lambda aid: model("local", aid, "inference"),
                                  max_agents=self.config.max_agents, max_parallel=self.config.max_parallel,
                                  local_rounds=self.config.local_rounds, total_max_calls=self.config.team_max_calls,
                                  explicit_rubrics=self.config.explicit_rubrics,
                                  execution_mode=self.config.execution_mode,
                                  execution_max_tokens=(self.config.models["exec"].max_tokens
                                                        if "exec" in self.config.models else None),
                                  agent_pool=pool,
                                  adaptive_budget_enforcement=self.config.adaptive_budget_enforcement,
                                  budget_context=ledger.resource_context,
                                  excluded_task_ids=self.manifest.validation + self.manifest.test,
                                  knowledge_policy=self.knowledge_policy)
        analyzer.public_planning_context = public_planning_context(task, self.config)
        analyzer.planning_response_format = self.config.planning_response_format
        if self.config.fixed_team is None:
            try:
                planned = analyzer.build(task, experience, local_planning=self.config.local_planning)
            finally:
                write_json(run_dir / "planning_calls.json", analyzer.call_records)
            prediction = analyzer.last_prediction
        else:
            if self.config.fixed_team.coverage or any(a.rubric_ids for a in self.config.fixed_team.agents):
                raise ValueError("Fixed MAS ablation uses explicit task-independent responsibilities without rubrics")
            prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=self.config.fixed_team.agents)
            planned = PlannedTeam(graph=prediction.graph, team=self.config.fixed_team)
        frozen = {"created_at": utc_now(), "experience_version": snapshot.version,
                  "experience_hash": digest(snapshot), "R_global": prediction.graph.model_dump(mode="json"),
                  "R_planned": planned.graph.model_dump(mode="json"), "TeamSpec": planned.team.model_dump(mode="json")}
        if pool is not None:
            frozen["AgentPool"] = pool.model_dump(mode="json")
        frozen["hashes"] = {key: digest(frozen[key]) for key in ("R_global", "R_planned", "TeamSpec")}
        if pool is not None:
            frozen["hashes"]["AgentPool"] = digest(frozen["AgentPool"])
        write_json(run_dir / "frozen_plan.json", frozen)
        write_json(run_dir / "planning_calls.json", analyzer.call_records)
        temporary_creations = []
        selected_ids = {agent.agent_id for agent in planned.team.agents}
        for candidate in prediction.candidates:
            if candidate.temporary_profile is None:
                continue
            temporary_creations.append({
                "task_id": task_id, "agent_id": candidate.agent_id,
                "selected": candidate.agent_id in selected_ids,
                "agent": candidate.model_dump(mode="json"),
                "creation_evidence": ["planning:global", "planning:planned"],
                "baseline_pool_hash": digest(pool) if pool is not None else None,
            })
        if temporary_creations:
            write_json(run_dir / "temporary_agents.json", {"task_id": task_id,
                "creations": temporary_creations})
        synth = self.synthesizer_factory(model("meta", "meta", "inference"))
        artifact = synth.synthesize(task, planned.graph, planned.team, experiences=experience,
                                    agent_pool=pool)
        write_json(run_dir / "harness.json", artifact.to_dict())
        executor = TeamExecutor(lambda aid: model("exec", aid, "inference"), tools=self.tools,
                                ledger=ledger, timeout_seconds=self.config.execution_timeout,
                                unsafe_local=self.config.unsafe_local,
                                knowledge_policy=self.knowledge_policy)
        executor.public_membership_observations_requested = self.config.public_membership_observations
        executor.public_membership_input_format = self.config.public_membership_input_format
        if self.config.public_membership_observations and self.config.public_membership_input_format == "compact":
            executor.public_membership_audit_writer = lambda records: write_json(
                run_dir / "public_membership_execution_audit.json",
                {"version": "public-membership-input-audit-v1", "records": records})
        if self.config.public_positional_draft_guidance:
            from .execution import compile_public_positional_draft_plan

            executor.public_positional_draft_guidance_requested = True
            executor.public_positional_draft_plan = compile_public_positional_draft_plan(
                task, self.config)
            executor.public_positional_draft_projection_requested = (
                self.config.public_positional_draft_projection)
        # JIT repair is limited to failures before any role call, never quality feedback.
        result = synth.execute_with_repair(executor, task, planned.team, artifact,
                                           rubrics=planned.graph, experiences=experience)
        write_json(run_dir / "harness.json", artifact.to_dict())
        write_json(run_dir / "execution.json", result.full_dict())
        call_trace = {"task_id": task_id, "run_key": run_key,
            "coordination": result.metadata.get("coordination"),
            "execution_calls": [{**event["content"], "agent_id": event["agent_id"],
                                 "event_id": event["event_id"], "timestamp": event["timestamp"]}
                                for event in result.metadata.get("events", [])
                                if event.get("kind") == "agent_call"],
            "shared_ledger_hash": result.metadata.get("shared_ledger_hash"), "evaluator_calls": []}
        write_json(run_dir / "call_trace.json", call_trace)
        if result.terminated_reason != "final_answer" or result.answer is None:
            write_json(run_dir / "budget.json", ledger.snapshot())
            raise RuntimeError("Team did not submit a final answer; official evaluation was not invoked")
        if self.config.public_refinement:
            from .public_refinement import PUBLIC_REFINEMENT_IDS, refine_public_answer

            write_json(run_dir / "execution_draft.json", result.full_dict())
            try:
                refine_public_answer(task, result, self.models, ledger, self.config,
                    knowledge_policy=self.knowledge_policy, synthesizer_id=planned.team.synthesizer_id,
                    audit_writer=lambda value: write_json(run_dir / "public_refinement.json", value))
            finally:
                # Preserve failures and charged calls before allowing submission.
                write_json(run_dir / "execution.json", result.full_dict())
                call_trace["public_refinement_calls"] = [row for row in ledger.snapshot()["records"]
                    if row.get("kind") == "model" and row.get("stage") == "inference"
                    and row.get("agent_id") in PUBLIC_REFINEMENT_IDS]
                write_json(run_dir / "call_trace.json", call_trace)
        submission = {"answer": result.answer, "answer_hash": digest(result.answer), "submitted_at": utc_now()}
        write_json(run_dir / "submission.json", submission)
        if defer_evaluation:
            outcome = {"run_key": run_key, "task_id": task_id, "mode": mode, "repeat": repeat,
                "status": "submitted_unscored", "backend": self.config.backend,
                "software_test_only": self.config.backend == "scripted", "resumed": False,
                "experience_version": snapshot.version, "experience_hash": digest(snapshot),
                "comparison_fingerprint": comparison_hash, "answer_hash": submission["answer_hash"],
                "submitted_at": submission["submitted_at"], "budget": ledger.snapshot(),
                "evaluation": None, "proposals": [], "pool_operations": [],
                "pool_observations": [], "run_dir": str(run_dir)}
            write_json(complete_path, outcome)
            return outcome
        # Sole transition at which the trusted coordinator opens private evaluation data.
        try:
            raw_feedback = evaluator.evaluate(str(result.answer), ground_truth=task_id,
                                              private_record=self.private_records[task_id])
        finally:
            # One evaluation may make many rubric-level calls; retain the actual settled records.
            call_trace["evaluator_calls"] = [record for record in ledger.snapshot()["records"]
                if record.get("stage") == "evaluation" and record.get("kind") == "model"]
            write_json(run_dir / "call_trace.json", call_trace)
        raw_feedback["submission_answer_hash"] = submission["answer_hash"]
        feedback = convert_feedback(raw_feedback)
        write_json(run_dir / "evaluation.json", feedback)
        proposals, agent_updates, pool_operations, pool_observations = [], [], [], []
        if attribution and feedback.complete:
            evolution_pool = pool if mode in ("evolve", "stream") and self.config.persistent_experience else None
            proposals, agent_updates, pool_operations, pool_observations = self._attribute(task, snapshot, ledger, run_dir, prediction.graph,
                planned.graph, planned.team, result, feedback, pool=evolution_pool)
        outcome = {"run_key": run_key, "task_id": task_id, "mode": mode, "backend": self.config.backend,
                   "software_test_only": self.config.backend == "scripted", "resumed": False,
                   "experience_version": snapshot.version, "experience_hash": digest(snapshot),
                   "comparison_fingerprint": comparison_hash, "answer_hash": submission["answer_hash"],
                   "submitted_at": submission["submitted_at"], "evaluated_at": utc_now(),
                   "plan_hashes": frozen["hashes"], "evaluation": feedback.model_dump(mode="json"),
                   "budget": ledger.snapshot(), "proposals": [p.model_dump(mode="json") for p in proposals],
                   "agent_updates": [item.model_dump(mode="json") for item in agent_updates],
                   "pool_operations": [item.model_dump(mode="json") for item in pool_operations],
                   "pool_observations": [item.model_dump(mode="json") for item in pool_observations],
                   "run_dir": str(run_dir), "uncontrolled_variation":
                   [] if self.config.backend == "scripted" else ["Provider sampling and live tool evidence are not snapshotted"]}
        stages = outcome["budget"]["by_stage"]
        outcome["costs"] = {"inference": stages.get("inference", {}),
                            "external_evaluation": stages.get("evaluation", {}),
                            "experience_update": stages.get("update", {})}
        write_json(complete_path, outcome)
        return outcome

    def run(self, mode, task_ids=None, *, limit=1, resume=True, resume_source=None, resume_source_hash=None,
            resume_attribution=None, resume_attribution_hash=None):
        from .schemas import ChangeProposal

        if mode not in ("evolve", "evaluate", "stream"):
            raise ValueError("Mode must be evolve, evaluate or stream")
        self.code_hash = code_fingerprint()
        allowed = getattr(self.manifest, {"evolve": "evolution", "evaluate": "test", "stream": "stream"}[mode])
        selected = list(allowed[:limit] if task_ids is None else task_ids)
        if not selected or not set(selected) <= set(allowed):
            raise ValueError("Tasks do not belong to the selected mode's split")
        if (resume_source is None) != (resume_source_hash is None):
            raise ValueError("Submitted-source continuation requires both path and content hash")
        if ((resume_attribution is None) != (resume_attribution_hash is None)
                or resume_attribution is not None and resume_source is None):
            raise ValueError("Frozen attribution requires a source and an explicit attribution content hash")
        if resume_source is not None and (mode != "evolve" or len(selected) != 1
                or self.store.snapshot().version != 0 or self.store.snapshot().experiences
                or self.store.task_run(mode, selected[0]) is not None):
            raise ValueError("Submitted-source continuation requires one evolution task and a fresh store")
        if mode == "stream" and selected != [item for item in allowed if item in selected]:
            raise ValueError("Stream order must match the manifest")
        frozen = self.store.snapshot()
        outcomes = []
        for task_id in selected:
            state = frozen if mode == "evaluate" else self.store.snapshot()
            update = mode != "evaluate" and self.config.persistent_experience
            identity = digest({"task": self.tasks[task_id], "private": self.private_records[task_id],
                               "config": self.config.model_dump(mode="json"), "code": self.code_hash,
                               "manifest": self.manifest.model_dump(mode="json"),
                               "attachments": input_fingerprints(self.tasks[task_id]),
                               **({"knowledge_policy": self.knowledge_policy}
                                  if self.knowledge_policy is not None else {})})
            journal = self.store.task_run(mode, task_id) if update else None
            if journal:
                if not resume or journal["identity"] != identity:
                    raise ValueError("Task was already submitted under this store; use a fresh store for changed policy")
                if journal["status"] == "complete":
                    outcome = journal["outcome"]
                    outcome["resumed"] = True
                    outcomes.append(outcome)
                    continue
                state = self.store.snapshot(journal["baseline_version"])
                if digest(state) != journal["baseline_hash"]:
                    raise ValueError("Task journal baseline changed")
            if update and mode == "stream":
                earlier = allowed[:allowed.index(task_id)]
                if any(not self.store.task_run(mode, key) or
                       self.store.task_run(mode, key)["status"] != "complete" for key in earlier):
                    raise ValueError("Stream tasks must be submitted in manifest order")
            if update and not journal:
                self.store.save_task_run(mode, task_id, identity, state, "started")
            outcome = (journal["outcome"] if journal and journal["outcome"] else
                       self.run_task(task_id, state, mode=mode, attribution=update, resume=resume,
                           **({"resume_source": resume_source, "resume_source_hash": resume_source_hash,
                               "resume_attribution": resume_attribution, "resume_attribution_hash": resume_attribution_hash}
                              if resume_source is not None else {})))
            if update:
                self.store.save_task_run(mode, task_id, identity, state, "submitted", outcome)
            outcome["experience_updates"] = []
            if update:
                proposal = (ChangeProposal.model_validate(outcome["proposals"][0])
                            if outcome["proposals"] else None)
                raw_agent_updates = [AgentEvolutionUpdate.model_validate(item)
                                     for item in outcome.get("agent_updates", [])]
                raw_operations = [AgentPoolOperation.model_validate(item)
                                  for item in outcome.get("pool_operations", [])]
                raw_observations = [AgentPoolObservation.model_validate(item)
                                    for item in outcome.get("pool_observations", [])]
                if proposal is not None or raw_agent_updates or raw_operations or raw_observations:
                    baseline = state.version
                    update_id = digest({"task_id": task_id, "baseline": digest(state),
                                        "proposal": proposal, "agent_updates": raw_agent_updates,
                                        "pool_operations": raw_operations, "pool_observations": raw_observations})
                    written = self.store.commit_evolution(source_task_id=task_id,
                        base_version=baseline, proposal=proposal, updates=raw_agent_updates,
                        operations=raw_operations, observations=raw_observations, update_id=update_id)
                    if proposal is not None:
                        outcome["experience_updates"].append({
                            "proposal_id": proposal.proposal_id, "source_task_id": proposal.source_task_id,
                            "base_version": proposal.base_version, "version": written.version,
                            "proposal_hash": digest(proposal), "snapshot_hash": digest(written),
                            "update_rule": "direct_after_attribution"})
                    if raw_agent_updates:
                        outcome["agent_pool_updates"] = [{
                            "update_id": item.update_id, "pool_agent_id": item.pool_agent_id,
                            "base_agent_version": item.base_agent_version,
                            "version": written.version, "snapshot_hash": digest(written),
                            "update_rule": "agent_self_reflection_after_attribution"}
                            for item in raw_agent_updates]
                    if raw_operations:
                        outcome["agent_pool_operations"] = [{
                            "operation_id": item.operation_id, "kind": item.kind,
                            "version": written.version, "snapshot_hash": digest(written),
                            "update_rule": "structural_evolution_after_reflection"}
                            for item in raw_operations]
            outcome["next_experience_version"] = self.store.snapshot().version
            if update:
                self.store.save_task_run(mode, task_id, identity, state, "complete", outcome)
            outcomes.append(outcome)
        write_json(self.output_dir / f"{mode}_report.json", outcomes)
        return outcomes
