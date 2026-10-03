"""Persistent role identities, scoped memory and agent-owned evolution policies."""

from __future__ import annotations

from .experience import experience_applicability
from .schemas import (
    AgentEvolutionUpdate,
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
    return AgentPoolSnapshot(profiles=[AgentProfile(
        pool_agent_id=key, role=role, capabilities=capabilities, prompt=prompt,
        skills=skills,
        reasoning_strategy="Choose reasoning suited to the task's genre. For factual analysis, distinguish "
                           "assumptions, evidence and conclusions and identify material uncertainty. "
                           "Invent within the requested premise for fiction.",
        planning_strategy="Choose methods appropriate to the assigned goal and available inputs.",
        communication="Publish substantive contributions with sources and unresolved gaps where relevant; "
                      "keep review notes in the internal handoff and honor the final artifact's format.")
        for key, (role, capabilities, prompt, skills) in roles.items()])


def catalogue(pool: AgentPoolSnapshot) -> list[dict]:
    return [{"pool_agent_id": profile.pool_agent_id, "version": profile.version,
             "role": profile.role, "capabilities": list(profile.capabilities),
             "completed_tasks": len(profile.source_task_ids),
             "memory_lessons": len(profile.memory)} for profile in pool.profiles]


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


def validate_bindings(pool: AgentPoolSnapshot, team: TeamSpec) -> None:
    identities = [agent.pool_agent_id for agent in team.agents]
    if len(identities) != len(set(identities)):
        raise ValueError("A pooled member may participate only once in a task")
    for agent in team.agents:
        if not agent.pool_agent_id or agent.pool_agent_version is None:
            raise ValueError("Every pooled role needs an exact member identity and version")
        profile = get_profile(pool, agent.pool_agent_id, agent.pool_agent_version)
        if not set(agent.selected_skills or []) <= set(profile.skills):
            raise ValueError("Agent selected skills outside its retained skill library")


def role_context(pool: AgentPoolSnapshot, agent: AgentSpec, task: PublicTask) -> dict:
    profile = get_profile(pool, agent.pool_agent_id, agent.pool_agent_version)
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
            "role": profile.role, "capabilities": profile.capabilities, "prompt": profile.prompt,
            "skills": {key: profile.skills[key] for key in selected}, "memory": memory,
            "reasoning_strategy": agent.reasoning_strategy or profile.reasoning_strategy,
            "planning_strategy": profile.planning_strategy,
            "communication": agent.communication or profile.communication,
            "harness": (agent.harness or profile.harness).model_dump(mode="json"),
            "preferred_tools": profile.preferred_tools,
            "completed_tasks": len(profile.source_task_ids)}


def apply_updates(pool: AgentPoolSnapshot, updates, source_task_id: str) -> AgentPoolSnapshot:
    changes = [AgentEvolutionUpdate.model_validate(
        update.model_dump(mode="json") if hasattr(update, "model_dump") else update)
        for update in (updates or ())]
    if not changes:
        return pool.model_copy(deep=True)
    result = (pool if pool.profiles else seed_pool()).model_copy(deep=True)
    changed = set()
    update_ids = set(result.applied_updates)
    for update in changes:
        if update.source_task_id != source_task_id:
            raise ValueError("Agent evolution source task mismatch")
        if update.update_id in update_ids or update.pool_agent_id in changed:
            raise ValueError("Duplicate Agent evolution update or member")
        profile = get_profile(result, update.pool_agent_id, update.base_agent_version)
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
        profile.version += 1
        profile.source_task_ids.append(source_task_id)
        profile.evidence = list(dict.fromkeys([*profile.evidence, *update.evidence]))
        result.profiles = [profile if item.pool_agent_id == profile.pool_agent_id else item
                           for item in result.profiles]
        result.applied_updates.append(update.update_id)
        changed.add(update.pool_agent_id)
        update_ids.add(update.update_id)
    if changes:
        result.version += 1
    return result
