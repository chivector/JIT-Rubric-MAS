"""Post-batch attribution and three independent full-state candidates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from pydantic import Field

from .agent_pool import apply_evolution, effective_pool, scope_pool
from .attribution import RubricAttributor
from .budget import BudgetLedger
from .experience import candidate_snapshot
from .planning import JsonModelCalls
from .schemas import (
    AgentEvolutionUpdate, AgentPoolObservation, AgentPoolOperation, ChangeProposal,
    EvaluationFeedback, ExperienceSnapshot, Record, RubricGraph, TeamSpec, digest,
)
from .pipeline import code_fingerprint, input_fingerprints, submitted_source_digest, write_json


STRATEGIES = (
    "Conservative: preserve established harnesses and consolidate only well-supported lessons.",
    "Failure repair: prioritize recurring rubric-prediction, coordination and role failures.",
    "Structure and cost: adjust Pool composition and harnesses to reduce evidenced redundancy while preserving coverage.",
)
BATCH_PROMPT = """Integrate this completed five-task batch into one candidate state.
The supplied rubric graphs, global/local attribution and agent reflections were
collected after all five tasks finished from one frozen state. No task was
executed using another task's reflection. Produce a single joint evolution from
base_version and agent_pool. Follow strategy; it is one of three preregistered proposal views, not
a different resource budget or an instruction to force updates.
Select supported proposal_ids exactly as supplied, at most one per observed
task. Group agent reflection records
by pool_agent_id; return at most one consolidated agent_update per existing Pool
member. Copy its bound base_agent_version and supplied update_id. source_task_id
must be one of that member's observed tasks; source_task_ids must be the distinct
real observed tasks supporting its update. Each new lesson cites only those real
task IDs and the exact namespaced evidence IDs available for that member. Preserve
the complete skills mapping when changing it. Keep observations separate from
causal improvement claims. Store conditional process knowledge, never answers,
private criterion text, hidden rubric thresholds or a desired benchmark score.
Pool operations may Add, Delete/Prune, Split, Merge, Specialize or Reorganize.
Bind the unchanged base_pool_version and every referenced existing member version.
Every operation has an actual observed source_task_id, source_task_ids containing
only supporting observed tasks, evidence from valid_evidence_ids, a rationale,
expected_benefit and token_cost_tradeoff. New full profiles use version 1;
specialize keeps the base member version. Preserve supported inherited history.
A member cannot receive both a direct update and a specialize operation. Retain a
temporary harness only by Add, with its recorded identity, role and capabilities.
No Pool shape is mandatory. Empty proposal_ids, agent_updates and pool_operations
are valid when evidence is insufficient. Candidate selection is performed later
on fixed VAL tasks; never read VAL or TEST answers or feedback here.

Output discipline: return a compact complete JSON object. Never echo the supplied
profiles, task reflections, evidence text or proposal bodies. Use terse rationales
and expected-benefit/tradeoff strings (under 160 characters each); use empty lists
when the evidence does not support a change. Hard output budget: at most two new
lessons per agent update, at most one pool operation, and never copy an inherited
profile or lesson into the response. If a supported change cannot fit compactly,
omit it rather than explaining it at length. Close every JSON string and brace
before the response limit."""


class BatchEvolutionDecision(Record):
    proposal_ids: list[str] = Field(default_factory=list)
    agent_updates: list[AgentEvolutionUpdate] = Field(default_factory=list)
    pool_operations: list[AgentPoolOperation] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


def _read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _ledger(pipeline):
    config = pipeline.config
    return BudgetLedger(config.max_model_calls, config.max_total_tokens,
                        config.max_tool_calls, timeout_seconds=config.task_timeout)


def _namespace(value, task_id):
    evidence_fields = {"evidence", "counterevidence", "supporting_evidence", "opposing_evidence"}
    if isinstance(value, list):
        return [_namespace(item, task_id) for item in value]
    if isinstance(value, dict):
        return {key: [f"{task_id}::{item}" for item in child]
                if key in evidence_fields and isinstance(child, list)
                else _namespace(child, task_id) for key, child in value.items()}
    return value


def _history_evidence(pool):
    evidence = {item for observation in pool.observations for item in observation.evidence}
    evidence.update(item for operation in pool.structural_history for item in operation.evidence)
    for profile in pool.profiles:
        evidence.update(profile.evidence)
        for lesson in profile.memory:
            evidence.update(lesson.evidence + lesson.counterevidence)
    return evidence


def _scope_batch_experience(proposal):
    """Make newly retained experience IDs unique across source tasks."""
    prefix = f"{proposal.source_task_id}:"
    experience_id = proposal.experience.experience_id
    if experience_id.startswith(prefix):
        return proposal
    experience = proposal.experience.model_copy(update={
        "experience_id": prefix + experience_id,
    })
    return proposal.model_copy(update={"experience": experience})


def _scope_batch_memory_update(update):
    """Scope newly proposed lessons and retain all cited lesson evidence.

    Batch meta models often use a short, role-local lesson ID (for example
    ``evidence_gap_handling``).  A later batch can legitimately propose a
    lesson with the same short ID, but AgentPool identities are persistent per
    member, so the unscoped ID would be rejected as a duplicate.  Prefixing by
    the update's real source task keeps identities stable for retries while
    preserving the source-task provenance that made the lesson eligible.

    Models also occasionally put a lesson's ``counterevidence`` in the lesson
    but omit it from the update-level evidence union.  ``apply_evolution``
    intentionally enforces that union; adding the cited IDs here lets valid
    batch evidence pass while the candidate validator still rejects IDs that
    were not observed for the member.
    """
    prefix = f"{update.source_task_id}:"
    lessons, evidence = [], list(update.evidence)
    for lesson in update.lessons:
        lesson_id = lesson.lesson_id
        if not lesson_id.startswith(prefix):
            lesson_id = prefix + lesson_id
        lessons.append(lesson.model_copy(update={"lesson_id": lesson_id}))
        evidence.extend(lesson.evidence)
        evidence.extend(lesson.counterevidence)
    return update.model_copy(update={"lessons": lessons,
                                     "evidence": list(dict.fromkeys(evidence))})


def _scope_batch_memory_operation(operation, pool):
    """Scope new memory in structural profiles while preserving inherited IDs."""
    inherited_ids = {lesson.lesson_id for profile in pool.profiles for lesson in profile.memory}
    prefix = f"{operation.source_task_id}:"
    profiles, evidence = [], list(operation.evidence)
    for profile in operation.profiles:
        lessons = []
        for lesson in profile.memory:
            inherited = lesson.lesson_id in inherited_ids
            lesson_id = lesson.lesson_id if inherited else (
                lesson.lesson_id if lesson.lesson_id.startswith(prefix)
                else prefix + lesson.lesson_id)
            lessons.append(lesson.model_copy(update={"lesson_id": lesson_id}))
            if not inherited:
                evidence.extend(lesson.evidence)
                evidence.extend(lesson.counterevidence)
        profiles.append(profile.model_copy(update={"memory": lessons}))
    return operation.model_copy(update={"profiles": profiles,
                                        "evidence": list(dict.fromkeys(evidence))})


def _load_task(pipeline, snapshot, row):
    task_id, directory = row["task_id"], Path(row["run_dir"])
    task = pipeline.tasks[task_id]
    manifest, frozen = _read(directory, "run_manifest.json"), _read(directory, "frozen_plan.json")
    comparison = manifest.get("comparison", {})
    if (manifest.get("mode") != "evolve" or manifest.get("experience_hash") != digest(snapshot)
            or frozen.get("experience_hash") != digest(snapshot)
            or comparison.get("code") != code_fingerprint()
            or comparison.get("manifest") != digest(pipeline.manifest)
            or comparison.get("repeat") != 0
            or comparison.get("tools") != sorted(pipeline.tools)
            or comparison.get("knowledge_policy") != pipeline.knowledge_policy
            or comparison.get("attachments") != input_fingerprints(task)
            or comparison.get("task") != task.model_dump(mode="json")
            or comparison.get("config") != pipeline.config.model_dump(mode="json")
            or comparison.get("private_hash") != digest(pipeline.private_records[task_id])):
        raise ValueError("Batch task receipt differs from its source, configuration or base state")
    for key in ("R_global", "R_planned", "TeamSpec", "AgentPool"):
        if frozen.get("hashes", {}).get(key) != digest(frozen.get(key)):
            raise ValueError("Batch task frozen plan identity changed")
    feedback = EvaluationFeedback.model_validate(_read(directory, "evaluation.json"))
    result, submission = _read(directory, "execution.json"), _read(directory, "submission.json")
    if (feedback.task_id != task_id or feedback.evaluator_version != comparison.get("evaluator")
            or feedback.evaluator_version != pipeline.evaluator_factory(None).evaluator_version
            or result.get("answer") != submission.get("answer")
            or submission.get("answer_hash") != digest(result.get("answer"))
            or feedback.raw.get("submission_answer_hash") != submission.get("answer_hash")):
        raise ValueError("Batch feedback is not bound to its submitted task answer")
    pool = scope_pool(effective_pool(snapshot.agent_pool), task,
                      excluded_task_ids=pipeline.manifest.validation + pipeline.manifest.test)
    if frozen["AgentPool"] != pool.model_dump(mode="json"):
        raise ValueError("Batch task uses a changed Agent Pool")
    documents = {"frozen_plan.json": frozen, "planning_calls.json": _read(directory, "planning_calls.json")}
    if (directory / "temporary_agents.json").is_file():
        documents["temporary_agents.json"] = _read(directory, "temporary_agents.json")
    pipeline._validate_creation_audit(documents, task, TeamSpec.model_validate(frozen["TeamSpec"]))
    return task, directory, frozen, feedback, result, pool


def _reflect_task(pipeline, snapshot, loaded, output):
    task, directory, frozen, feedback, result, pool = loaded
    ledger = _ledger(pipeline)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("frozen_plan.json", "evaluation.json", "temporary_agents.json"):
        source = directory / name
        if source.is_file():
            (output / name).write_bytes(source.read_bytes())
    attributor = RubricAttributor(pipeline.models.create("global", "batch-global", ledger, "update"),
        lambda agent_id: pipeline.models.create("local", agent_id, ledger, "update"),
        max_parallel=pipeline.config.max_parallel,
        local_attribution=pipeline.config.local_attribution)
    source_hash = submitted_source_digest(directory)
    record = {"task_id": task.task_id, "status": "failed", "run_dir": str(directory),
              "source_hash": source_hash}
    try:
        team = TeamSpec.model_validate(frozen["TeamSpec"])
        findings = attributor.attribute(task, RubricGraph.model_validate(frozen["R_global"]),
            RubricGraph.model_validate(frozen["R_planned"]), team, result, feedback)
        write_json(output / "attribution.json", {
            "complete": True, "findings": [item.model_dump(mode="json") for item in findings],
            "alignments": {key: value.model_dump(mode="json")
                           for key, value in attributor.last_alignments.items()},
            "alignment_warnings": list(attributor.last_alignment_warnings),
            "credit_assignments": {key: value.model_dump(mode="json")
                                   for key, value in attributor.last_credit_assignments.items()},
            "global_outline": (attributor.last_global_outline.model_dump(mode="json")
                               if attributor.last_global_outline is not None else None),
            "calls": attributor.call_records, "evolution_integrated": False})
        updates = pipeline._evolve_agents(task, ledger, result, team, pool, output)
        proposals = attributor.propose(task, findings, snapshot.version, snapshot.experiences,
                                      agent_reflections=updates)
        proposals = [pipeline._scope_proposal_id(task.task_id, item) for item in proposals]
        creations = (_read(output, "temporary_agents.json").get("creations", [])
                     if (output / "temporary_agents.json").is_file() else [])
        observations = pipeline._pool_observations(task, team, result, feedback)
        current_pool = pool.model_copy(update={"profiles": [], "observations": [], "structural_history": []})
        valid_ids = pipeline._pool_evidence(current_pool, result, feedback, creations)
        history_ids = _history_evidence(pool)
        record.update(status="complete", findings=_namespace([item.model_dump(mode="json") for item in findings], task.task_id),
            proposals=_namespace([item.model_dump(mode="json") for item in proposals], task.task_id),
            updates=_namespace([item.model_dump(mode="json") for item in updates], task.task_id),
            observations=_namespace([item.model_dump(mode="json") for item in observations], task.task_id),
            creations=creations,
            retention_candidates=_namespace([item.model_dump(mode="json") for item in
                pipeline._retention_candidates(team, updates, task.task_id)], task.task_id),
            valid_evidence_ids=sorted(history_ids | {f"{task.task_id}::{item}" for item in valid_ids}))
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        record["budget"] = ledger.snapshot()
        record["attribution_calls"] = attributor.call_records
        write_json(output / "reflection.json", record)
    if submitted_source_digest(directory) != source_hash:
        raise ValueError("Batch task source changed during reflection")
    return record


def candidate_batch_snapshot(base, decision, *, source_task_ids, proposals=(), observations=()):
    """Apply one consolidated candidate without mutating its base or a store."""
    base = ExperienceSnapshot.model_validate(base)
    decision = BatchEvolutionDecision.model_validate(decision)
    tasks = list(source_task_ids)
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("Batch evolution requires five distinct real EVO task IDs")
    proposal_map = {item.proposal_id: item for item in proposals}
    if (len(decision.proposal_ids) != len(set(decision.proposal_ids))
            or not set(decision.proposal_ids) <= set(proposal_map)):
        raise ValueError("Batch decision selected unknown or duplicate proposals")
    sources = [proposal_map[identity].source_task_id for identity in decision.proposal_ids]
    if len(sources) != len(set(sources)):
        raise ValueError("Batch decision cannot retain multiple proposals from one task")
    candidate = base.model_copy(deep=True)
    for identity in decision.proposal_ids:
        proposal = proposal_map[identity]
        if proposal.source_task_id not in tasks or not set(proposal.experience.source_task_ids) <= set(tasks):
            raise ValueError("Batch experience invented source tasks")
        candidate = candidate_snapshot(candidate, proposal)
    # Meta-model output is batch-local and may use short lesson identities or
    # omit counterevidence from the update-level evidence union.  Normalize it
    # only at this batch boundary; direct single-task evolution continues to
    # enforce the strict schema unchanged.
    updates = [_scope_batch_memory_update(item) for item in decision.agent_updates]
    pool = effective_pool(base.agent_pool)
    operations = [_scope_batch_memory_operation(item, pool)
                  for item in decision.pool_operations]
    candidate.agent_pool = apply_evolution(base.agent_pool, updates,
        operations, observations, source_task_id=tasks[0], source_task_ids=tasks)
    if digest(candidate) != digest(base):
        candidate.version = base.version + 1
        candidate.policy_versions = {**candidate.policy_versions, "experience_update": "batch-v1",
                                     "agent_pool": "dynamic-batch-evolution-v1"}
    return ExperienceSnapshot.model_validate(candidate.model_dump(mode="json"))


def _groups(records, base, batch_id):
    profiles = {item.pool_agent_id: item for item in effective_pool(base.agent_pool).profiles}
    groups = {}
    for record in records:
        for update in record.get("updates", []):
            if update["pool_agent_id"] not in profiles:
                continue
            group = groups.setdefault(update["pool_agent_id"], {
                "agent_profile": profiles[update["pool_agent_id"]], "records": [], "source_task_ids": []})
            group["records"].append(update)
            group["source_task_ids"].append(record["task_id"])
    for member, group in groups.items():
        group["update_id"] = digest({"batch": batch_id, "member": member,
                                     "base": digest(base), "sources": group["source_task_ids"]})
        group["valid_evidence_ids"] = sorted({item for update in group["records"]
                                              for item in update["evidence"]})
    return groups


def _pool_view(pool):
    value = pool.model_dump(mode="json")
    value["observations"] = [{key: item[key] for key in ("task_id", "pool_agent_id", "submission_score",
        "complete", "input_tokens", "output_tokens", "tool_calls", "cost", "usage_available")}
        for item in value["observations"]]
    value["structural_history"] = [{key: item[key] for key in ("operation_id", "kind", "source_task_id",
        "source_task_ids", "target_agent_ids", "rationale", "expected_benefit", "token_cost_tradeoff", "evidence")}
        for item in value["structural_history"]]
    return value


def _candidate(pipeline, base, task_ids, records, output, batch_id, index):
    try:
        max_corrections = max(1, int(os.environ.get("JIT_MAS_BATCH_META_CORRECTIONS", "1")))
    except ValueError:
        max_corrections = 1
    ledger, caller = _ledger(pipeline), JsonModelCalls(max_corrections=max_corrections)
    groups = _groups(records, base, batch_id)
    proposals = [_scope_batch_experience(ChangeProposal.model_validate(item))
                 for record in records for item in record.get("proposals", [])]
    observations = [AgentPoolObservation.model_validate(item) for record in records
                    for item in record.get("observations", [])]
    valid_ids = {item for record in records for item in record.get("valid_evidence_ids", [])}
    pool = effective_pool(base.agent_pool)
    creations = [item for record in records for item in record.get("creations", [])]
    observed_tasks = {record["task_id"] for record in records if record["status"] == "complete"}
    result = {"candidate_id": f"{batch_id}c{index}", "candidate_index": index,
              "strategy": STRATEGIES[index], "status": "failed", "snapshot": None,
              "state_hash": None, "source_task_ids": task_ids, "base_state_hash": digest(base)}

    sanitized_proposals = []

    def _sanitize_proposals(decision):
        """Keep at most one supported proposal per source task in resilient mode."""
        if os.environ.get("JIT_MAS_BATCH_SANITIZE") != "1":
            return decision
        selected, seen_sources = [], set()
        proposal_map = {item.proposal_id: item for item in proposals}
        for proposal_id in decision.proposal_ids:
            proposal = proposal_map.get(proposal_id)
            if proposal is None or proposal.source_task_id not in task_ids:
                sanitized_proposals.append({"proposal_id": proposal_id,
                                            "reason": "unknown_or_out_of_batch"})
                continue
            if proposal.source_task_id in seen_sources:
                sanitized_proposals.append({"proposal_id": proposal_id,
                                            "reason": "duplicate_source_task"})
                continue
            seen_sources.add(proposal.source_task_id)
            selected.append(proposal_id)
        if selected != decision.proposal_ids:
            decision.proposal_ids[:] = selected
        return decision

    def validate(decision):
        _sanitize_proposals(decision)
        for update in decision.agent_updates:
            group = groups.get(update.pool_agent_id)
            if (group is None or update.update_id != group["update_id"]
                    or update.base_agent_version != group["agent_profile"].version
                    or not update.source_task_ids or not set(update.source_task_ids) <= set(group["source_task_ids"])
                    or not set(update.evidence) <= set(group["valid_evidence_ids"])):
                raise ValueError("Batch agent update changed its identity, source tasks or local evidence")
            lesson_evidence = {item for lesson in update.lessons
                               for item in lesson.evidence + lesson.counterevidence}
            if not lesson_evidence <= set(group["valid_evidence_ids"]):
                raise ValueError("Batch agent lesson cites unavailable local evidence")
        for operation in decision.pool_operations:
            if (not operation.source_task_ids or not set(operation.source_task_ids) <= observed_tasks
                    or not set(operation.evidence) <= valid_ids):
                raise ValueError("Batch Pool operation cites unavailable source tasks or evidence")
            inherited_lessons = {lesson.lesson_id for profile in pool.profiles for lesson in profile.memory}
            for profile in operation.profiles:
                profile_evidence = {item for lesson in profile.memory
                                    if lesson.lesson_id not in inherited_lessons
                                    for item in lesson.evidence + lesson.counterevidence}
                if not profile_evidence <= valid_ids:
                    raise ValueError("Batch retained profile cites unavailable evidence")
                matching = [item for item in creations
                            if item["agent"]["pool_agent_id"] == profile.pool_agent_id]
                if matching and (operation.kind != "add" or not any(item["selected"]
                        and profile.role == item["agent"]["temporary_profile"]["role"]
                        and profile.capabilities == item["agent"]["temporary_profile"]["capabilities"]
                        for item in matching)):
                    raise ValueError("Batch retained harness differs from its executed temporary creation")
                if matching and not set(operation.source_task_ids) <= {
                        item["task_id"] for item in matching if item["selected"]}:
                    raise ValueError("Batch temporary retention invented source tasks")
        return candidate_batch_snapshot(base, decision, source_task_ids=task_ids,
                                        proposals=proposals, observations=observations)

    try:
        supported = [record for record in records if record["status"] == "complete"]
        if not supported:
            decision = BatchEvolutionDecision(rationale="No complete attribution evidence; preserve the batch base.")
            candidate = base.model_copy(deep=True)
            result["no_evidence"] = True
        else:
            meta_model = pipeline.models.create("global", f"batch-meta-{index}", ledger, "update")
            try:
                meta_cap = int(os.environ.get("JIT_MAS_BATCH_META_MAX_TOKENS", "0"))
            except ValueError:
                meta_cap = 0
            if meta_cap > 0:
                # Only the compact batch integration response receives the
                # larger ceiling; ordinary task planning keeps its frozen cap.
                meta_model.max_tokens = meta_cap
            decision = caller.ask(meta_model,
                "batch_evolution_integrate", BATCH_PROMPT,
                {"batch_id": batch_id, "source_task_ids": task_ids, "base_version": base.version,
                 "meta_experiences": base.experiences,
                 "agent_pool": _pool_view(pool), "base_pool_version": pool.version,
                 "strategy": STRATEGIES[index], "task_reflections": [
                     {key: record[key] for key in ("task_id", "findings", "observations", "creations")}
                     for record in supported],
                 "agent_reflection_groups": {member: {
                     **{key: value for key, value in group.items() if key != "agent_profile"},
                     "base_agent_version": group["agent_profile"].version} for member, group in groups.items()},
                 "candidate_proposals": proposals,
                 "valid_evidence_ids": sorted(valid_ids),
                 "retention_candidates": [item for record in records
                                            for item in record.get("retention_candidates", [])]},
                BatchEvolutionDecision, validate=validate)
            candidate = validate(decision)
            if sanitized_proposals:
                result["sanitized_proposals"] = list(sanitized_proposals)
        result.update(status="complete", snapshot=candidate.model_dump(mode="json"),
                      state_hash=digest(candidate), decision=decision.model_dump(mode="json"))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        # A malformed meta decision must never make an entire source
        # unevaluable when the experiment explicitly opts into resilient
        # production mode.  Preserve the base snapshot as a transparent
        # no-op candidate and retain the failure details for diagnostics;
        # strict/default mode keeps the historical failed-candidate behavior.
        if os.environ.get("JIT_MAS_BATCH_INVALID_FALLBACK") == "1":
            result.update(status="complete", fallback_noop=True,
                          fallback_reason="invalid_batch_meta_decision",
                          snapshot=base.model_dump(mode="json"),
                          state_hash=digest(base),
                          decision=BatchEvolutionDecision(
                              rationale="Invalid batch meta decision; preserve the validated base state."
                          ).model_dump(mode="json"))
    finally:
        result["budget"] = ledger.snapshot()
        result["calls"] = caller.call_records
        write_json(output / f"candidate{index}.json", result)
    return result


def evolve_batch(pipeline, base_snapshot, task_runs, output_dir, *, batch_id):
    """Reflect five terminal EVO receipts, then return three parallel candidates."""
    base = ExperienceSnapshot.model_validate(base_snapshot)
    task_ids = [row["task_id"] for row in task_runs]
    if (len(task_ids) != 5 or len(set(task_ids)) != 5
            or not set(task_ids) <= set(pipeline.manifest.evolution)
            or any(row.get("status") not in {"complete", "incomplete", "failed", "missing"}
                   for row in task_runs)):
        raise ValueError("Batch evolution requires five terminal receipts from its EVO split")
    loaded = [_load_task(pipeline, base, row) for row in task_runs if row["status"] == "complete"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(_reflect_task, pipeline, base, item,
                                  output / "reflections" / digest(item[0].task_id))
                   for item in loaded if item[3].complete]
        records = [future.result() for future in futures]
        candidates = list(workers.map(lambda index: _candidate(pipeline, base, task_ids, records,
            output, batch_id, index), range(3)))
    write_json(output / "batch_evolution.json", {"batch_id": batch_id,
        "source_task_ids": task_ids, "base_state_hash": digest(base),
        "task_statuses": [{"task_id": row["task_id"], "status": row["status"]} for row in task_runs],
        "reflections": records, "candidates": candidates,
        "no_intra_batch_evolution": True})
    if digest(base) != digest(base_snapshot):
        raise ValueError("Batch evolution mutated its base snapshot")
    return candidates
