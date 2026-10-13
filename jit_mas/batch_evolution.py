"""Post-batch attribution and three independent full-state candidates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path

from pydantic import Field, model_validator

from .agent_pool import apply_evolution, effective_pool, scope_pool
from .attribution import RubricAttributor
from .budget import BudgetLedger
from .experience import candidate_snapshot
from .planning import JsonModelCalls
from .schemas import (
    AgentEvolutionUpdate, AgentPoolObservation, AgentPoolOperation, ChangeProposal,
    AgentHarnessPolicy, AgentProfile, EvaluationFeedback, ExperienceSnapshot, Record,
    RubricGraph, TeamSpec, digest,
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
Prefer proposal_choices: one source_task_id key per observed task maps to one
supplied proposal_id for that SAME task, or null to retain none. Select only IDs
listed in proposal_options_by_task for that key. Omit proposal_ids entirely when
using proposal_choices; never return both formats, even an empty proposal_ids
list. Legacy proposal_ids is accepted only without proposal_choices and may
select at most one proposal per observed task. Group agent reflection records
by pool_agent_id; return at most one consolidated agent_update per existing Pool
member. The host supplies one deterministic expected_update_id for each group;
copy that value character-for-character and never invent a descriptive
consolidated ID (such as analyst-consolidated-update-1). Copy its bound
base_agent_version. source_task_id
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
For a supported structural change, use profile_patches instead of copying full
profiles. Each patch binds base_profile_id and its exact base_profile_version
from agent_pool or retention_candidates, names the output pool_agent_id, and
contains only changed policy fields. skill_updates replaces named instructions
while preserving other skills; skill_extensions appends conditional guidance
to an existing skill without removing its general instruction. Never put the
same key in both mappings. The host restores unchanged skills, memory and history.
When retaining a general-purpose member, preserve its genre-independent skills.
Add narrowly supported genre advice as a separate skill with an explicit task
condition; do not replace general genre control with a rule for one genre.
Specialize patches reference their target and keep its identity and base version.
Split patches reference the retained parent and name new child identities; Merge
patches reference a target and name one new output identity. Add may retain an
executed retention_candidate with its recorded identity, role and capabilities,
or derive a new child from an existing Pool member. Bind every affected existing
target and parent in base_agent_versions. Never write memory, source_task_ids,
evidence or a new version inside a patch; the operation supplies current-task
provenance and the host preserves validated inherited history.
A specialize operation replaces direct agent_updates for that target: omit ALL
direct updates whose pool_agent_id is specialized. Put supported policy changes
in its profile_patch instead; use direct updates when retaining new lessons.
No Pool shape is mandatory. Empty proposal_choices, agent_updates and pool_operations
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


class BatchProfilePatch(Record):
    base_profile_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    base_profile_version: int = Field(ge=1)
    pool_agent_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    role: str | None = Field(default=None, min_length=1)
    capabilities: list[str] | None = Field(default=None, min_length=1)
    parent_agent_id: str | None = None
    prompt: str | None = Field(default=None, min_length=1, max_length=6000)
    skill_updates: dict[str, str] = Field(default_factory=dict)
    skill_extensions: dict[str, str] = Field(default_factory=dict)
    preferred_tools: list[str] | None = None
    reasoning_strategy: str | None = None
    planning_strategy: str | None = None
    communication: str | None = None
    harness: AgentHarnessPolicy | None = None


class BatchPoolOperation(AgentPoolOperation):
    profile_patches: list[BatchProfilePatch] = Field(default_factory=list)


class BatchEvolutionDecision(Record):
    proposal_ids: list[str] = Field(default_factory=list)
    proposal_choices: dict[str, str | None] | None = None
    agent_updates: list[AgentEvolutionUpdate] = Field(default_factory=list)
    pool_operations: list[AgentPoolOperation | BatchPoolOperation] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def exclusive_proposal_formats(cls, value):
        if (isinstance(value, dict) and value.get("proposal_choices") is not None
                and "proposal_ids" in value):
            raise ValueError("Batch decision must use proposal_choices or proposal_ids, never both")
        return value


def _proposal_options(proposals, task_ids):
    return {task_id: [proposal.proposal_id for proposal in proposals
                      if proposal.source_task_id == task_id] for task_id in task_ids}


def materialize_batch_proposal_choices(decision, *, source_task_ids, proposals):
    """Resolve one task-bound choice into the existing proposal ID representation."""
    decision = BatchEvolutionDecision.model_validate(decision)
    choices = decision.proposal_choices
    if choices is None:
        return decision
    task_ids = list(source_task_ids)
    options = _proposal_options(proposals, task_ids)
    proposal_map = {proposal.proposal_id: proposal for proposal in proposals}
    selected = []
    for task_id, proposal_id in choices.items():
        if task_id not in options:
            raise ValueError(f"Batch proposal_choices contains an out-of-batch source_task_id: {task_id}")
        if proposal_id is None:
            continue
        proposal = proposal_map.get(proposal_id)
        if proposal is None:
            raise ValueError(
                f"Batch proposal_choices selected an unknown proposal for source_task_id={task_id}; "
                f"proposal_id={proposal_id}; allowed_proposal_ids={options[task_id]}")
        if proposal.source_task_id != task_id:
            raise ValueError(
                f"Batch proposal_choices source mismatch: source_task_id={task_id}; "
                f"proposal_id={proposal_id}; actual_source_task_id={proposal.source_task_id}; "
                f"allowed_proposal_ids={options[task_id]}")
        selected.append(proposal_id)
    return decision.model_copy(update={"proposal_ids": selected, "proposal_choices": None})


def materialize_batch_decision(base, decision, *, retention_candidates=(),
                               compose_specializations=False):
    """Restore compact structural policies before the existing pool validation."""
    decision = BatchEvolutionDecision.model_validate(decision)
    pool = effective_pool(ExperienceSnapshot.model_validate(base).agent_pool)
    profiles = {profile.pool_agent_id: profile for profile in pool.profiles}
    retained = {}
    for candidate in retention_candidates:
        profile = AgentProfile.model_validate(candidate)
        previous = retained.get(profile.pool_agent_id)
        if previous is not None and previous != profile:
            raise ValueError("Batch profile patch has an ambiguous retention candidate")
        retained[profile.pool_agent_id] = profile
    specialized = {target for operation in decision.pool_operations
                   if operation.kind == "specialize" for target in operation.target_agent_ids}
    conflicts = sorted(specialized & {update.pool_agent_id for update in decision.agent_updates})
    if conflicts and not compose_specializations:
        raise ValueError(
            "Batch direct agent_updates conflict with specialize targets: "
            + ", ".join(conflicts)
            + ". Remove those direct agent_updates and keep the specialize profile_patches, "
            "or remove their specialize operations; do not silently drop either change.")
    operations = []
    for operation in decision.pool_operations:
        patches = getattr(operation, "profile_patches", ())
        if not patches:
            operations.append(AgentPoolOperation.model_validate(
                operation.model_dump(mode="json", exclude={"profile_patches"})))
            continue
        if operation.profiles:
            raise ValueError("Batch Pool operation must use profiles or profile_patches, never both")
        outputs = []
        for patch in patches:
            reference = profiles.get(patch.base_profile_id) or retained.get(patch.base_profile_id)
            if reference is None:
                raise ValueError("Batch profile patch references an unknown profile")
            if reference.version != patch.base_profile_version:
                raise ValueError("Batch profile patch changed its bound base profile version")
            from_pool = patch.base_profile_id in profiles
            if operation.kind not in {"add", "split", "merge", "specialize"}:
                raise ValueError("Batch profile patches require Add, Split, Merge or Specialize")
            if operation.kind != "add" and (
                    not from_pool or patch.base_profile_id not in operation.target_agent_ids):
                raise ValueError("Batch structural profile patch must reference an existing target")
            if not from_pool and patch.pool_agent_id != reference.pool_agent_id:
                raise ValueError("Batch retained profile patch changed its executed identity")
            if operation.kind == "specialize" and patch.pool_agent_id != reference.pool_agent_id:
                raise ValueError("Batch specialize profile patch changed its target identity")
            value = reference.model_dump(mode="json")
            value.update(pool_agent_id=patch.pool_agent_id,
                         version=reference.version if operation.kind == "specialize" else 1,
                         source_task_ids=list(operation.source_task_ids),
                         evidence=list(operation.evidence))
            if from_pool:
                value["memory"] = []
            if operation.kind == "split" or operation.kind == "add" and from_pool:
                value["parent_agent_id"] = reference.pool_agent_id
            elif operation.kind == "merge":
                value["parent_agent_id"] = None
                inherited_skills, conflicting_skills = {}, set()
                for target in operation.target_agent_ids:
                    source = profiles.get(target)
                    if source is None:
                        raise ValueError("Batch merge profile patch has an unknown target")
                    for skill_id, instruction in source.skills.items():
                        if skill_id in inherited_skills and inherited_skills[skill_id] != instruction:
                            conflicting_skills.add(skill_id)
                        inherited_skills[skill_id] = instruction
                if not conflicting_skills <= set(patch.skill_updates):
                    raise ValueError("Batch merge profile patch must resolve conflicting inherited skills")
                value["skills"] = inherited_skills
            for field in ("role", "capabilities", "prompt", "preferred_tools",
                          "reasoning_strategy", "planning_strategy", "communication", "harness"):
                changed = getattr(patch, field)
                if changed is not None:
                    value[field] = changed
            if "parent_agent_id" in patch.model_fields_set:
                value["parent_agent_id"] = patch.parent_agent_id
            if set(patch.skill_updates) & set(patch.skill_extensions):
                raise ValueError("Batch skill patch cannot both replace and extend the same key")
            value["skills"] = {**value["skills"], **patch.skill_updates}
            for skill_id, instruction in patch.skill_extensions.items():
                if skill_id not in reference.skills:
                    raise ValueError("Batch skill extension requires an inherited skill")
                value["skills"][skill_id] = (reference.skills[skill_id].rstrip()
                    + "\n\nAdditional conditional guidance (apply only when the task signals this condition): "
                    + instruction.strip())
            outputs.append(AgentProfile.model_validate(value))
        document = operation.model_dump(mode="json", exclude={"profile_patches"})
        document["profiles"] = outputs
        operations.append(AgentPoolOperation.model_validate(document))
    return decision.model_copy(update={"pool_operations": operations})


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


def candidate_batch_snapshot(base, decision, *, source_task_ids, proposals=(), observations=(),
                             compose_specializations=False):
    """Apply one consolidated candidate without mutating its base or a store."""
    base = ExperienceSnapshot.model_validate(base)
    decision = materialize_batch_decision(base, decision,
                                        compose_specializations=compose_specializations)
    tasks = list(source_task_ids)
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("Batch evolution requires five distinct real EVO task IDs")
    decision = materialize_batch_proposal_choices(decision, source_task_ids=tasks, proposals=proposals)
    proposal_map = {item.proposal_id: item for item in proposals}
    if (len(decision.proposal_ids) != len(set(decision.proposal_ids))
            or not set(decision.proposal_ids) <= set(proposal_map)):
        raise ValueError("Batch decision selected unknown or duplicate proposals")
    sources = [proposal_map[identity].source_task_id for identity in decision.proposal_ids]
    if len(sources) != len(set(sources)):
        options = _proposal_options(proposals, tasks)
        duplicates = {source: {"selected_proposal_ids": [
            identity for identity in decision.proposal_ids if proposal_map[identity].source_task_id == source],
            "allowed_proposal_ids": options.get(source, [])}
            for source in dict.fromkeys(sources) if sources.count(source) > 1}
        raise ValueError("Batch decision cannot retain multiple proposals from one task; "
                         "choose one allowed ID per source_task_id using proposal_choices: "
                         + json.dumps(duplicates, sort_keys=True))
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
        operations, observations, source_task_id=tasks[0], source_task_ids=tasks,
        compose_specializations=compose_specializations)
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


def _compact_value(value):
    """Copy JSON-compatible values while dropping empty audit placeholders.

    Batch reflections are persisted as full receipts.  The meta model only needs
    semantic values and evidence IDs; ``schema_version``, timestamps and null/
    empty defaults are bookkeeping noise.  Keep false/zero values because they
    are meaningful (for example ``success=False`` and ``uncertainty=0``).
    """
    if isinstance(value, dict):
        return {key: _compact_value(child) for key, child in value.items()
                if key != "schema_version" and child is not None
                and child != "" and child != [] and child != {}}
    if isinstance(value, list):
        return [_compact_value(child) for child in value
                if child is not None and child != "" and child != [] and child != {}]
    return copy.deepcopy(value)


def _compact_fields(value, fields):
    """Return an allowlisted, non-empty copy of a reflection record."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {}
    return _compact_value({field: value[field] for field in fields if field in value})


def _compact_lesson(value, *, include_source_tasks=False):
    fields = ("lesson_id", "instruction", "applicability", "capability", "task_signals",
              "evidence", "counterevidence")
    if include_source_tasks:
        fields += ("source_task_ids",)
    return _compact_fields(value, fields)


def _compact_harness(value):
    return _compact_fields(value, ("harness_id", "memory_policy", "memory_window", "tool_policy"))


def _compact_finding(value):
    # Findings contain the actionable attribution, while persisted alignment,
    # request and raw audit fields are intentionally excluded from the payload.
    return _compact_fields(value, ("finding_id", "rubric_ids", "categories", "component",
                                   "hypothesis", "alternatives", "uncertainty", "success",
                                   "supporting_evidence", "opposing_evidence"))


def _compact_observation(value):
    return _compact_fields(value, ("task_id", "agent_id", "pool_agent_id", "profile_version",
                                   "temporary", "submission_score", "complete", "input_tokens",
                                   "output_tokens", "tool_calls", "usage_available", "usage_estimated",
                                   "evidence"))


def _compact_creation(value):
    """Keep temporary-harness identity and selection evidence without prompts."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {}
    result = _compact_fields(value, ("task_id", "agent_id", "selected", "agent",
                                     "creation_evidence"))
    agent = value.get("agent")
    if hasattr(agent, "model_dump"):
        agent = agent.model_dump(mode="json")
    if isinstance(agent, dict):
        result["agent"] = _compact_fields(
            agent, ("agent_id", "pool_agent_id", "pool_agent_version", "role", "capability",
                    "rubric_ids", "responsibilities", "tools", "depends_on", "temporary_profile",
                    "creation_rationale"))
        if isinstance(agent.get("temporary_profile"), dict):
            result["agent"]["temporary_profile"] = _compact_profile(agent["temporary_profile"])
    return _compact_value(result)


def _compact_task_reflection(value):
    if not isinstance(value, dict):
        return {}
    result = {"task_id": value.get("task_id"),
              "findings": [_compact_finding(item) for item in value.get("findings", [])],
              "observations": [_compact_observation(item)
                               for item in value.get("observations", [])],
              "creations": [_compact_creation(item) for item in value.get("creations", [])]}
    # Findings, observations and temporary-harness identity are semantic input;
    # attribution/transport receipts and duplicate task payloads are omitted.
    return _compact_value(result)


def _compact_group_record(value):
    if not isinstance(value, dict):
        return {}
    result = _compact_fields(value, ("source_task_id", "lessons", "prompt", "skills",
                                     "preferred_tools", "reasoning_strategy", "planning_strategy",
                                     "communication", "harness", "evidence"))
    if "lessons" in result:
        result["lessons"] = [_compact_lesson(item) for item in value.get("lessons", [])]
    if "harness" in result:
        result["harness"] = _compact_harness(value.get("harness"))
    return _compact_value(result)


def _compact_reflection_group(value):
    if not isinstance(value, dict):
        return {}
    result = _compact_fields(value, ("source_task_ids", "valid_evidence_ids", "records",
                                     "base_agent_version", "expected_update_id", "update_id"))
    if "records" in result:
        result["records"] = [_compact_group_record(item) for item in value.get("records", [])]
    return _compact_value(result)


def _compact_proposal(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {}
    result = _compact_fields(value, ("proposal_id", "source_task_id", "diff", "rationale",
                                     "expected_benefit", "risks", "evidence", "experience"))
    experience = value.get("experience")
    if isinstance(experience, dict):
        result["experience"] = _compact_fields(
            experience, ("instruction", "applicability", "capability", "task_signals",
                          "evidence", "counterevidence", "source_task_ids"))
    return _compact_value(result)


def _compact_profile(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {}
    # The full role prompt is repeated in every task harness and is restored by
    # the host for profile patches.  Keep policy/identity and all provenance;
    # omit only that duplicate prompt and schema/audit defaults.
    result = _compact_fields(value, ("pool_agent_id", "version", "role", "capabilities",
                                     "parent_agent_id", "skills", "preferred_tools",
                                     "reasoning_strategy", "planning_strategy", "communication",
                                     "harness", "memory", "source_task_ids", "evidence"))
    if "harness" in result:
        result["harness"] = _compact_harness(value.get("harness"))
    if "memory" in result:
        result["memory"] = [_compact_lesson(item, include_source_tasks=True)
                             for item in value.get("memory", [])]
    return _compact_value(result)


def _compact_structural_history(value):
    if not isinstance(value, dict):
        return {}
    return _compact_fields(value, ("operation_id", "kind", "source_task_id", "source_task_ids",
                                    "base_pool_version", "target_agent_ids", "base_agent_versions",
                                    "parent_assignments", "evidence", "rationale",
                                    "expected_benefit", "token_cost_tradeoff"))


def _pool_view(pool):
    """Return the compact, provenance-bound Pool context for the batch meta call."""
    value = pool.model_dump(mode="json") if hasattr(pool, "model_dump") else copy.deepcopy(pool)
    if not isinstance(value, dict):
        return {}
    # Observations and applied update IDs are audit ledgers repeated in each
    # task reflection.  Structural history and profile provenance remain so
    # the model can bind every proposed operation to the frozen base.
    return {
        "version": value.get("version", 0),
        "profiles": [_compact_profile(item) for item in value.get("profiles", [])],
        "structural_history": [_compact_structural_history(item)
                                for item in value.get("structural_history", [])],
    }


def _compact_candidate_payload(payload):
    """Compact a batch meta request without changing provenance semantics.

    This boundary is intentionally pure: callers can persist and inspect the
    original reflection receipts, while the model receives only an allowlisted
    semantic projection.  Evidence and task IDs are copied character-for-
    character, never rewritten or inferred.
    """
    if not isinstance(payload, dict):
        return copy.deepcopy(payload)
    compacted = copy.deepcopy(payload)
    if "task_reflections" in compacted:
        compacted["task_reflections"] = [_compact_task_reflection(item)
                                          for item in payload.get("task_reflections", [])]
    if "agent_reflection_groups" in compacted:
        compacted["agent_reflection_groups"] = {
            member: _compact_reflection_group(group)
            for member, group in payload.get("agent_reflection_groups", {}).items()
        }
    if "candidate_proposals" in compacted:
        compacted["candidate_proposals"] = [_compact_proposal(item)
                                             for item in payload.get("candidate_proposals", [])]
    if "agent_pool" in compacted:
        compacted["agent_pool"] = _pool_view(payload["agent_pool"])
    if "retention_candidates" in compacted:
        compacted["retention_candidates"] = [_compact_profile(item)
                                              for item in payload.get("retention_candidates", [])]
    return compacted


def _inject_structural_fallback(decision, *, pool, task_ids, observed_tasks,
                                valid_evidence_ids, batch_id, candidate_index):
    """Add one evidence-bound Pool shape change for the structural candidate.

    The meta model is allowed to choose no Pool operation, which made the
    preregistered ``Structure and cost`` view degenerate into another
    specialize-only view in the live run.  When the explicit runtime flag is
    enabled, keep the model decision intact and add a small, auditable child
    harness only when the batch has at least two completed tasks and real
    evidence IDs.  The operation is validated by the normal host path before
    it can become a candidate snapshot; it is never applied to the base.
    """
    if os.environ.get("JIT_MAS_ENFORCE_STRUCTURAL_OPS") != "1" or candidate_index != 2:
        return decision, None
    shape_kinds = {"add", "delete", "prune", "split", "merge"}
    if any(operation.kind in shape_kinds for operation in decision.pool_operations):
        return decision, None
    observed = sorted(set(observed_tasks) & set(task_ids))
    evidence = sorted(set(valid_evidence_ids))
    if len(observed) < 2 or not evidence:
        return decision, None
    parent = next((profile for profile in pool.profiles if profile.pool_agent_id == "critic"), None)
    parent = parent or (pool.profiles[0] if pool.profiles else None)
    if parent is None:
        return decision, None
    source_task_ids = observed
    source_task_id = source_task_ids[0]
    child_id = f"{parent.pool_agent_id}_struct_{batch_id.replace(':', '_')}_c{candidate_index}"
    operation = {
        "operation_id": f"structural-fallback:{batch_id}:c{candidate_index}",
        "kind": "add",
        "source_task_id": source_task_id,
        "source_task_ids": source_task_ids,
        "base_pool_version": pool.version,
        "target_agent_ids": [],
        "base_agent_versions": {parent.pool_agent_id: parent.version},
        "parent_assignments": {},
        "evidence": evidence[:2],
        "rationale": "Repeated batch evidence supports a narrowly scoped reviewer child for the structural candidate.",
        "expected_benefit": "Separate recurring cross-task review from the parent role while preserving inherited skills.",
        "token_cost_tradeoff": "Adds one Pool member and one planning context, trading small setup cost for focused review coverage.",
        "profile_patches": [{
            "base_profile_id": parent.pool_agent_id,
            "base_profile_version": parent.version,
            "pool_agent_id": child_id,
            "skill_updates": {
                "structural_review": (
                    "When recurring evaluator evidence spans a batch, isolate the repeated gap, "
                    "check it independently, and hand a concise correction to the synthesizer.")
            },
            "skill_extensions": {},
        }],
    }
    updated = decision.model_copy(update={
        "pool_operations": [*decision.pool_operations, BatchPoolOperation.model_validate(operation)],
        "rationale": decision.rationale + " Added an evidence-bound structural child for the structure-and-cost view.",
    })
    return updated, {"kind": "add", "parent_agent_id": parent.pool_agent_id,
                     "child_agent_id": child_id, "source_task_ids": source_task_ids,
                     "evidence_count": len(evidence), "trigger": "structure_strategy_no_shape_operation"}


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
              "state_hash": None, "source_task_ids": task_ids, "base_state_hash": digest(base),
              "compose_specialize_updates": pipeline.config.compose_specialize_updates}

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
        # The batch host owns the deterministic group binding. Models may
        # replace the supplied digest with a descriptive ``*-consolidated-*``
        # label even when all other provenance fields are valid. Repair only
        # that binding here; the source/evidence/version checks below remain
        # strict, and unknown pool members are never normalized.
        for index, update in enumerate(decision.agent_updates):
            group = groups.get(update.pool_agent_id)
            if group is not None and update.update_id != group["update_id"]:
                decision.agent_updates[index] = update.model_copy(
                    update={"update_id": group["update_id"]})
        normalized_choices = materialize_batch_proposal_choices(
            decision, source_task_ids=task_ids, proposals=proposals)
        decision.proposal_choices = None
        decision.proposal_ids = normalized_choices.proposal_ids
        normalized = materialize_batch_decision(base, decision, retention_candidates=[
            item for record in records for item in record.get("retention_candidates", [])],
            compose_specializations=pipeline.config.compose_specialize_updates)
        decision.pool_operations = normalized.pool_operations
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
                                        proposals=proposals, observations=observations,
                                        compose_specializations=pipeline.config.compose_specialize_updates)

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
            evolution_prompt = BATCH_PROMPT
            if pipeline.config.compose_specialize_updates:
                evolution_prompt = evolution_prompt.replace(
                    "A member cannot receive both a direct update and a specialize operation.",
                    "A member may receive both a direct update and a specialize operation: the host retains\nvalidated update lessons and provenance, then applies explicit specialization policy changes.")
                evolution_prompt = evolution_prompt.replace(
                    "A specialize operation replaces direct agent_updates for that target: omit ALL\ndirect updates whose pool_agent_id is specialized. Put supported policy changes\nin its profile_patch instead; use direct updates when retaining new lessons.",
                    "Keep supported direct agent_updates alongside specialize profile_patches when the\nmember has new lessons. The host composes them in one version increment; explicit\nprofile_patch changes take precedence and inherited self-update memory is retained.")
            candidate_payload = _compact_candidate_payload({
                "batch_id": batch_id, "source_task_ids": task_ids, "base_version": base.version,
                "meta_experiences": base.experiences,
                "agent_pool": _pool_view(pool), "base_pool_version": pool.version,
                "compose_specialize_updates": pipeline.config.compose_specialize_updates,
                "strategy": STRATEGIES[index], "task_reflections": [
                    {key: record[key] for key in ("task_id", "findings", "observations", "creations")}
                    for record in supported],
                "agent_reflection_groups": {member: {
                    **{key: value for key, value in group.items() if key != "agent_profile"},
                    "base_agent_version": group["agent_profile"].version,
                    "expected_update_id": group["update_id"]} for member, group in groups.items()},
                "candidate_proposals": proposals,
                "proposal_options_by_task": _proposal_options(proposals, task_ids),
                "valid_evidence_ids": sorted(valid_ids),
                "retention_candidates": [item for record in records
                                           for item in record.get("retention_candidates", [])]})
            decision = caller.ask(meta_model,
                "batch_evolution_integrate", evolution_prompt, candidate_payload,
                BatchEvolutionDecision, validate=validate)
            decision, structural_fallback = _inject_structural_fallback(
                decision, pool=pool, task_ids=task_ids, observed_tasks=observed_tasks,
                valid_evidence_ids=valid_ids, batch_id=batch_id, candidate_index=index)
            if structural_fallback is not None:
                result["structural_fallback"] = structural_fallback
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
    # A five-task batch can produce several large attribution requests at the
    # same time.  Some OpenAI-compatible gateways accept the small task calls
    # but reset connections when all five high-context reflections overlap.
    # Do not silently convert that transport failure into ``no evidence``:
    # retry failed reflections one at a time, in a separate durable directory.
    # The retry is opt-out so frozen historical campaigns remain unchanged.
    try:
        reflection_retries = max(0, int(os.environ.get("JIT_MAS_BATCH_REFLECTION_RETRIES", "1")))
    except ValueError:
        reflection_retries = 1
    if reflection_retries and records:
        by_task = {record.get("task_id"): record for record in records}
        for loaded_item in loaded:
            task_id = loaded_item[0].task_id
            previous = by_task.get(task_id)
            if previous is not None and previous.get("status") == "complete":
                continue
            for retry_index in range(reflection_retries):
                retry_root = output / "reflections_retry" / f"{digest(task_id)}_r{retry_index}"
                retry_record = _reflect_task(pipeline, base, loaded_item, retry_root)
                by_task[task_id] = retry_record
                if retry_record.get("status") == "complete":
                    break
        records = [by_task[item[0].task_id] for item in loaded
                   if item[3].complete and item[0].task_id in by_task]
    try:
        candidate_workers = max(1, int(os.environ.get("JIT_MAS_BATCH_CANDIDATE_WORKERS", "1")))
    except ValueError:
        candidate_workers = 1
    with ThreadPoolExecutor(max_workers=min(3, candidate_workers)) as workers:
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
