"""Persistent role identities, scoped memory and agent-owned evolution policies."""

from __future__ import annotations

from .experience import experience_applicability
from .schemas import (
    AgentEvolutionUpdate,
    AgentPoolObservation,
    AgentPoolOperation,
    AgentPoolSnapshot,
    AgentProfile,
    AgentSpec,
    PublicTask,
    TeamSpec,
)


def seed_pool() -> AgentPoolSnapshot:
    roles = {
        "writer": ("Writer", ["writing", "synthesis", "creative-writing"],
                   "Build the requested deliverable in its genre, using evidence for factual claims where relevant.",
                   {"synthesis": "Resolve conflicting contributions, preserve useful detail and sources.",
                    "format_control": "Check the decoded final answer against explicit language, length, "
                                      "format, required wording and forbidden-content constraints; "
                                      "keep protocol metadata and review notes outside the artifact.",
                    "genre_control": "Follow the requested genre and audience: develop motivated scenes "
                                     "and continuity for fiction, reasons and relevant counterarguments "
                                     "within the requested genre and length for argumentative writing, "
                                     "or concrete benefits and actions for practical copy. "
                                     "Do not impose research-report conventions on creative work. "
                                     "When assigned a writing contribution, provide budgeted actual "
                                     "passages, scenes, dialogue or transitions in existing answer or "
                                     "outline fields so a downstream reviewer can inspect wording. "
                                     "Keep passages distinct from review notes and avoid duplicating "
                                     "the complete long deliverable. As final writer, integrate useful "
                                     "passages and specific revisions into one coherent artifact. "
                                     "Fiction may invent within the public premise; actual real-world "
                                     "factual claims still need appropriate support."}),
        "searcher": ("Searcher", ["search", "source verification", "research", "evidence"],
                     "Retrieve and organize evidence with explicit provenance and uncertainty.",
                     {"evidence_selection": "Prefer direct sources and distinguish observed facts from inference.",
                      "coverage_search": "Map requested facets to sources, check dates and scope, "
                                         "search gaps and contradictory evidence, and preserve exact "
                                         "locators. Do not treat a truncated page as evidence of absence."}),
        "critic": ("Critic", ["review", "verification", "criticism"],
                   "Check consequential claims independently and provide supported corrections.",
                   {"claim_check": "Check assumptions, counterexamples and conflicting evidence before accepting a claim.",
                    "public_constraint_check": "Inspect the actual artifact against every explicit "
                                               "public constraint, including counts and format; "
                                               "identify the defect and a concrete correction. "
                                               "Keep critique outside the final requested artifact. "
                                               "Inspect actual upstream passages against the requested genre, "
                                               "audience and voice. Identify the exact passage, its defect and "
                                               "effect on continuity or purpose, and a specific revision; do "
                                               "not merely approve intended style. If no prose is available, "
                                               "review only the visible plan and do not claim the finished "
                                               "prose was inspected. Fictional events within the public premise "
                                               "need no factual citation; real-world claims still need support."}),
        "planner": ("Planner", ["planning", "requirements", "decomposition"],
                    "Turn the public task into feasible requirements and a concrete work plan.",
                    {"decomposition": "Identify required inputs, dependencies, constraints and observable completion conditions."}),
        "analyst": ("Analyst", ["analysis", "comparison", "explanation", "reasoning"],
                    "Analyze the problem with explicit assumptions and supported reasoning.",
                    {"tradeoff_analysis": "Compare alternatives under stated conditions and test important inferences.",
                     "set_reasoning": "For exhaustive requests, apply consistent inclusion conditions, "
                                      "deduplicate aliases, retain all supported distinct members and "
                                      "separate unresolved gaps from confirmed answers."}),
        "generalist": ("Generalist", ["general task solving", "specialized tasks"],
                       "Solve the assigned role using public requirements and report limitations.",
                       {"task_check": "Check the public constraints and the completeness of your contribution."}),
    }
    return AgentPoolSnapshot(initialized=True, profiles=[AgentProfile(
        pool_agent_id=key, role=role, capabilities=capabilities, prompt=prompt,
        skills=skills,
        reasoning_strategy="Choose reasoning suited to the task's genre. For factual analysis, distinguish "
                           "assumptions, evidence and conclusions and identify material uncertainty. "
                           "Invent within the requested premise for fiction.",
        planning_strategy="Choose methods appropriate to the assigned goal and available inputs.",
        communication="Publish substantive contributions with sources and unresolved gaps where relevant; "
                      "keep review notes in the internal handoff and honor the final artifact's format.")
        for key, (role, capabilities, prompt, skills) in roles.items()])


def effective_pool(pool: AgentPoolSnapshot) -> AgentPoolSnapshot:
    if (pool.initialized or pool.profiles or pool.version or pool.applied_updates
            or pool.structural_history or pool.observations):
        result = pool.model_copy(deep=True)
        result.initialized = True
        return result
    return seed_pool()


def catalogue(pool: AgentPoolSnapshot) -> list[dict]:
    """Return task-safe selection metadata for the meta-level planner.

    The catalogue describes the reusable harness contract without exposing a
    role prompt or its conditional execution memory.  The selected role later
    receives those private fields through :func:`role_context`, after the
    planner has bound an exact pool identity and version.
    """
    return [{
        "pool_agent_id": profile.pool_agent_id,
        "parent_agent_id": profile.parent_agent_id,
        "version": profile.version,
        "role": profile.role,
        "capabilities": list(profile.capabilities),
        "skills": sorted(profile.skills),
        "preferred_tools": list(profile.preferred_tools),
        "reasoning_strategy": profile.reasoning_strategy,
        "planning_strategy": profile.planning_strategy,
        "communication": profile.communication,
        "harness": profile.harness.model_dump(mode="json"),
        "completed_tasks": len(profile.source_task_ids),
        "memory_lessons": len(profile.memory),
    } for profile in pool.profiles]


def scope_pool(pool: AgentPoolSnapshot, task: PublicTask, *, excluded_task_ids=()) -> AgentPoolSnapshot:
    scoped = pool.model_copy(deep=True)
    excluded = set(excluded_task_ids) | {task.task_id}
    for profile in scoped.profiles:
        profile.memory = [lesson for lesson in profile.memory
                          if not excluded.intersection(lesson.source_task_ids)]
    return scoped


def get_profile(pool: AgentPoolSnapshot, pool_agent_id: str,
                version: int | None = None) -> AgentProfile:
    profile = next((item for item in pool.profiles if item.pool_agent_id == pool_agent_id), None)
    if profile is None:
        raise ValueError(f"Unknown Agent Pool member: {pool_agent_id}")
    if version is not None and profile.version != version:
        raise ValueError(f"Stale Agent Pool member version: {pool_agent_id}")
    return profile.model_copy(deep=True)


def resolve_profile(pool: AgentPoolSnapshot, agent: AgentSpec) -> AgentProfile:
    if agent.temporary_profile is not None:
        profile = AgentProfile.model_validate(agent.temporary_profile.model_dump(mode="json"))
        if agent.pool_agent_id != profile.pool_agent_id or agent.pool_agent_version != profile.version:
            raise ValueError("Temporary profile identity must match its task binding")
        if any(item.pool_agent_id == profile.pool_agent_id for item in pool.profiles):
            raise ValueError("Temporary profile identity conflicts with an Agent Pool member")
        if profile.version != 1 or profile.memory or profile.source_task_ids or profile.evidence:
            raise ValueError("Temporary profiles must be new harnesses without invented history")
        if profile.parent_agent_id is not None:
            get_profile(pool, profile.parent_agent_id)
        if not agent.creation_rationale.strip():
            raise ValueError("Temporary agent creation requires a rationale")
        return profile
    if not agent.pool_agent_id or agent.pool_agent_version is None:
        raise ValueError("Every pooled role needs an exact member identity and version")
    return get_profile(pool, agent.pool_agent_id, agent.pool_agent_version)


def validate_bindings(pool: AgentPoolSnapshot, team: TeamSpec) -> None:
    identities = [agent.pool_agent_id or (agent.temporary_profile.pool_agent_id
                  if agent.temporary_profile else None) for agent in team.agents]
    if len(identities) != len(set(identities)):
        raise ValueError("A pooled member may participate only once in a task")
    for agent in team.agents:
        profile = resolve_profile(pool, agent)
        if not set(agent.selected_skills or []) <= set(profile.skills):
            raise ValueError("Agent selected skills outside its retained skill library")


def role_context(pool: AgentPoolSnapshot, agent: AgentSpec, task: PublicTask) -> dict:
    profile = resolve_profile(pool, agent)
    memory = []
    for lesson in profile.memory:
        if task.task_id in lesson.source_task_ids:
            continue
        scoped = {**lesson.model_dump(mode="json"), "bank": "execution",
                  "task_signals": lesson.task_signals or [lesson.applicability]}
        if experience_applicability(scoped, task, capability=agent.capability)["matched"]:
            memory.append(lesson.model_dump(mode="json"))
    selected = list(profile.skills) if agent.selected_skills is None else agent.selected_skills
    return {"pool_agent_id": profile.pool_agent_id, "version": profile.version,
            "temporary": agent.temporary_profile is not None,
            "parent_agent_id": profile.parent_agent_id,
            "role": profile.role, "capabilities": profile.capabilities, "prompt": profile.prompt,
            "skills": {key: profile.skills[key] for key in selected}, "memory": memory,
            "reasoning_strategy": agent.reasoning_strategy or profile.reasoning_strategy,
            "planning_strategy": profile.planning_strategy,
            "communication": agent.communication or profile.communication,
            "harness": (agent.harness or profile.harness).model_dump(mode="json"),
            "preferred_tools": profile.preferred_tools,
            "completed_tasks": len(profile.source_task_ids)}


def apply_updates(pool: AgentPoolSnapshot, updates, source_task_id: str) -> AgentPoolSnapshot:
    return apply_evolution(pool, updates, (), (), source_task_id=source_task_id)


def _validate_hierarchy(profiles: dict[str, AgentProfile]) -> None:
    for profile in profiles.values():
        ancestors = {profile.pool_agent_id}
        parent_id = profile.parent_agent_id
        while parent_id is not None:
            if parent_id not in profiles:
                raise ValueError("Agent Pool hierarchy contains an unknown parent")
            if parent_id in ancestors:
                raise ValueError("Agent Pool hierarchy contains a cycle")
            ancestors.add(parent_id)
            parent_id = profiles[parent_id].parent_agent_id


def _inherited_profile(profile, sources, operation, source_task_id):
    inherited_lessons = {}
    inherited_skills = {}
    conflicting_skills = set()
    inherited_tasks = []
    inherited_evidence = set()
    for source in sources:
        for skill_id, instruction in source.skills.items():
            if skill_id in inherited_skills and inherited_skills[skill_id] != instruction:
                conflicting_skills.add(skill_id)
            inherited_skills[skill_id] = instruction
        inherited_tasks.extend(source.source_task_ids)
        inherited_evidence.update(source.evidence)
        for lesson in source.memory:
            previous = inherited_lessons.get(lesson.lesson_id)
            if previous is not None and previous != lesson:
                raise ValueError("Inherited memory has conflicting lesson identities")
            inherited_lessons[lesson.lesson_id] = lesson
            inherited_tasks.extend(lesson.source_task_ids)
            inherited_evidence.update(lesson.evidence + lesson.counterevidence)
    if source_task_id not in profile.source_task_ids:
        raise ValueError("Structural profile requires current task provenance")
    if not set(profile.source_task_ids) <= set(inherited_tasks) | {source_task_id}:
        raise ValueError("Structural profile invented historical task provenance")
    if (not profile.evidence or not any(item.strip() for item in profile.evidence)
            or not set(profile.evidence) <= inherited_evidence | set(operation.evidence)):
        raise ValueError("Structural profile has invalid evidence provenance")
    known_lessons = dict(inherited_lessons)
    for lesson in profile.memory:
        previous = inherited_lessons.get(lesson.lesson_id)
        if previous is not None:
            if previous != lesson:
                raise ValueError("Structural operation rewrote inherited memory provenance")
        elif (lesson.source_task_ids != [source_task_id]
              or not set(lesson.evidence + lesson.counterevidence) <= set(operation.evidence)):
            raise ValueError("Structural memory lesson has invalid provenance")
        known_lessons[lesson.lesson_id] = lesson
    result = profile.model_copy(deep=True)
    if operation.kind != "specialize" and not conflicting_skills <= set(profile.skills):
        raise ValueError("Structural operation must resolve conflicting inherited skills")
    if operation.kind != "specialize":
        result.skills = {**inherited_skills, **profile.skills}
    result.memory = [lesson.model_copy(deep=True) for lesson in known_lessons.values()]
    result.source_task_ids = list(dict.fromkeys([*inherited_tasks, source_task_id]))
    result.evidence = list(dict.fromkeys([
        *(evidence for source in sources for evidence in source.evidence),
        *profile.evidence, *operation.evidence]))
    return result


def apply_evolution(pool: AgentPoolSnapshot, updates=(), operations=(), observations=(), *,
                    source_task_id: str) -> AgentPoolSnapshot:
    if not isinstance(source_task_id, str) or not source_task_id.strip():
        raise ValueError("Agent evolution requires a nonempty source task ID")
    changes = [AgentEvolutionUpdate.model_validate(
        update.model_dump(mode="json") if hasattr(update, "model_dump") else update)
        for update in (updates or ())]
    structures = [AgentPoolOperation.model_validate(
        operation.model_dump(mode="json") if hasattr(operation, "model_dump") else operation)
        for operation in (operations or ())]
    measurements = [AgentPoolObservation.model_validate(
        observation.model_dump(mode="json") if hasattr(observation, "model_dump") else observation)
        for observation in (observations or ())]
    if not changes and not structures and not measurements:
        return pool.model_copy(deep=True)
    result = effective_pool(pool)
    base = {profile.pool_agent_id: profile.model_copy(deep=True) for profile in result.profiles}
    profiles = {identity: profile.model_copy(deep=True) for identity, profile in base.items()}
    _validate_hierarchy(profiles)
    changed = set()
    local_changed = set()
    update_ids = set(result.applied_updates)
    for update in changes:
        if update.source_task_id != source_task_id:
            raise ValueError("Agent evolution source task mismatch")
        if update.update_id in update_ids or update.pool_agent_id in changed:
            raise ValueError("Duplicate Agent evolution update or member")
        if update.pool_agent_id not in base:
            raise ValueError(f"Unknown Agent Pool member: {update.pool_agent_id}")
        profile = profiles[update.pool_agent_id]
        if profile.version != update.base_agent_version:
            raise ValueError(f"Stale Agent Pool member version: {update.pool_agent_id}")
        if source_task_id in profile.source_task_ids:
            raise ValueError("Agent already learned from this source task")
        known_lessons = {lesson.lesson_id for lesson in profile.memory}
        for lesson in update.lessons:
            if (lesson.lesson_id in known_lessons or lesson.source_task_ids != [source_task_id]
                    or not set(lesson.evidence + lesson.counterevidence) <= set(update.evidence)):
                raise ValueError("Agent memory lesson has invalid provenance or duplicate identity")
            known_lessons.add(lesson.lesson_id)
            profile.memory.append(lesson.model_copy(deep=True))
        for field in ("prompt", "skills", "preferred_tools", "reasoning_strategy",
                      "planning_strategy", "communication", "harness"):
            value = getattr(update, field)
            if value is not None:
                setattr(profile, field, value)
        profile.source_task_ids.append(source_task_id)
        profile.evidence = list(dict.fromkeys([*profile.evidence, *update.evidence]))
        result.applied_updates.append(update.update_id)
        changed.add(update.pool_agent_id)
        local_changed.add(update.pool_agent_id)
        update_ids.add(update.update_id)
    historical_ids = set(base)
    operation_ids = {operation.operation_id for operation in result.structural_history}
    for operation in result.structural_history:
        historical_ids.update(operation.target_agent_ids)
        historical_ids.update(profile.pool_agent_id for profile in operation.profiles)
    specialized = set()
    structural_targets = set()
    created_ids = set()
    for operation in structures:
        if operation.source_task_id != source_task_id:
            raise ValueError("Agent Pool operation source task mismatch")
        if operation.base_pool_version != result.version:
            raise ValueError("Stale Agent Pool operation base version")
        if operation.operation_id in operation_ids:
            raise ValueError("Duplicate Agent Pool operation ID")
        if not all(item.strip() for item in operation.evidence):
            raise ValueError("Agent Pool operation requires nonempty evidence provenance")
        if not all(value.strip() for value in (operation.rationale, operation.expected_benefit,
                                               operation.token_cost_tradeoff)):
            raise ValueError("Agent Pool operation requires a rationale and quality/cost tradeoff")
        targets = operation.target_agent_ids
        if len(targets) != len(set(targets)) or any(identity not in profiles for identity in targets):
            raise ValueError("Agent Pool operation has duplicate or unknown targets")
        outputs = operation.profiles
        output_ids = [profile.pool_agent_id for profile in outputs]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("Agent Pool operation has duplicate output identities")
        assignments = operation.parent_assignments
        references = set(targets) | set(assignments)
        references.update(parent for parent in assignments.values() if parent is not None)
        references.update(profile.parent_agent_id for profile in outputs if profile.parent_agent_id is not None)
        if operation.kind in {"delete", "prune", "merge"}:
            for child in profiles.values():
                if child.parent_agent_id in targets:
                    references.add(child.pool_agent_id)
                    parent_id = child.parent_agent_id
                    while parent_id in targets:
                        parent_id = profiles[parent_id].parent_agent_id
                    if parent_id is not None and parent_id not in output_ids:
                        references.add(parent_id)
        existing_references = references - set(output_ids) - created_ids
        if operation.kind == "specialize":
            existing_references.update(targets)
        if any(identity not in profiles for identity in existing_references):
            raise ValueError("Agent Pool operation references an unknown member")
        if set(operation.base_agent_versions) != existing_references:
            raise ValueError("Agent Pool operation must bind exact versions for every referenced member")
        for identity, version in operation.base_agent_versions.items():
            if identity in base and version != profiles[identity].version:
                raise ValueError(f"Stale Agent Pool operation member version: {identity}")
        if operation.kind == "add":
            if targets or not outputs:
                raise ValueError("Add requires new profiles and no targets")
        elif operation.kind in {"delete", "prune"}:
            if not targets or outputs:
                raise ValueError("Delete/Prune requires targets and no output profiles")
        elif operation.kind == "split":
            if len(targets) != 1 or len(outputs) < 2:
                raise ValueError("Split requires one retained parent and at least two new child profiles")
        elif operation.kind == "merge":
            if len(targets) < 2 or len(outputs) != 1:
                raise ValueError("Merge requires at least two targets and one new profile")
        elif operation.kind == "specialize":
            if len(targets) != 1 or len(outputs) != 1 or output_ids != targets:
                raise ValueError("Specialize requires one target and a same-identity output")
            if targets[0] in local_changed or targets[0] in specialized:
                raise ValueError("Duplicate harness changes for a specialized Agent Pool member")
            specialized.add(targets[0])
        elif operation.kind == "reorganize":
            if outputs or not assignments or set(targets) != set(assignments):
                raise ValueError("Reorganize requires targets matching its parent assignments and no profiles")
        if operation.kind != "reorganize":
            if set(targets) & structural_targets:
                raise ValueError("Agent Pool member cannot receive multiple structural changes in one evolution batch")
            if operation.kind in {"delete", "prune"} and set(targets) & local_changed:
                raise ValueError("Agent Pool member cannot be locally updated and removed in one evolution batch")
            structural_targets.update(targets)
        if operation.kind != "specialize":
            if any(identity in historical_ids for identity in output_ids):
                raise ValueError("Agent Pool cannot reuse an existing or retired profile identity")
            if any(profile.version != 1 for profile in outputs):
                raise ValueError("New Agent Pool profiles must start at version 1")
        elif outputs[0].version != profiles[targets[0]].version:
            raise ValueError("Specialized profile changed its base version")
        sources = [profiles[identity].model_copy(deep=True) for identity in targets]
        if operation.kind == "add":
            pending = list(outputs)
            while pending:
                ready = [output for output in pending
                         if output.parent_agent_id is None or output.parent_agent_id in profiles]
                if not ready:
                    raise ValueError("Agent Pool hierarchy contains a cycle or unknown parent")
                for output in ready:
                    inherited = [profiles[output.parent_agent_id]] if output.parent_agent_id is not None else []
                    profiles[output.pool_agent_id] = _inherited_profile(output, inherited, operation, source_task_id)
                    pending.remove(output)
        elif operation.kind == "split":
            for output in outputs:
                profile = _inherited_profile(output, sources, operation, source_task_id)
                if profile.parent_agent_id is None:
                    profile.parent_agent_id = targets[0]
                profiles[profile.pool_agent_id] = profile
        elif operation.kind in {"merge", "specialize"}:
            output = _inherited_profile(outputs[0], sources, operation, source_task_id)
            if operation.kind == "merge" and output.parent_agent_id is None:
                parent_ids = {profile.parent_agent_id for profile in sources} - set(targets)
                if len(parent_ids) == 1:
                    output.parent_agent_id = next(iter(parent_ids))
            profiles[output.pool_agent_id] = output
            if operation.kind == "specialize":
                changed.add(output.pool_agent_id)
        removed = set(targets) if operation.kind in {"delete", "prune", "merge"} else set()
        for identity in removed:
            del profiles[identity]
        for profile in profiles.values():
            if profile.parent_agent_id in removed:
                parent_id = outputs[0].pool_agent_id if operation.kind == "merge" else profile.parent_agent_id
                while parent_id in removed:
                    parent_id = next(source.parent_agent_id for source in sources if source.pool_agent_id == parent_id)
                profile.parent_agent_id = parent_id
                if profile.pool_agent_id in base:
                    changed.add(profile.pool_agent_id)
        for identity, parent_id in assignments.items():
            if identity not in profiles or (parent_id is not None and parent_id not in profiles):
                raise ValueError("Agent Pool parent assignment references a removed or unknown member")
            if profiles[identity].parent_agent_id != parent_id:
                profiles[identity].parent_agent_id = parent_id
                if identity in base:
                    changed.add(identity)
        historical_ids.update(output_ids)
        if operation.kind != "specialize":
            created_ids.update(output_ids)
        operation_ids.add(operation.operation_id)
        result.structural_history.append(operation.model_copy(deep=True))
    _validate_hierarchy(profiles)
    measurement_keys = {(item.task_id, item.agent_id) for item in result.observations}
    for observation in measurements:
        if observation.task_id != source_task_id:
            raise ValueError("Agent Pool observation changed its source task")
        key = (observation.task_id, observation.agent_id)
        if key in measurement_keys:
            raise ValueError("Duplicate Agent Pool observation")
        if observation.temporary:
            if observation.pool_agent_id in base or observation.profile_version != 1:
                raise ValueError("Temporary observation changed its profile identity or version")
        elif (observation.pool_agent_id not in base
              or base[observation.pool_agent_id].version != observation.profile_version):
            raise ValueError("Agent Pool observation changed its bound member version")
        measurement_keys.add(key)
        result.observations.append(observation.model_copy(deep=True))
    for identity in changed.intersection(profiles):
        profiles[identity].version = base[identity].version + 1 if identity in base else profiles[identity].version + 1
    result.profiles = list(profiles.values())
    result.version += 1
    return AgentPoolSnapshot.model_validate(result.model_dump(mode="json"))
