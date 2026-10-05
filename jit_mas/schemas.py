"""Validated public, evaluation, planning, and experience boundaries."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"Cannot hash {type(item).__name__}")
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=encode,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: str = "1.0"


class PublicTask(Record):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    attachments: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


class PrivateEvaluationRecord(Record):
    task_id: str
    criteria: list[dict[str, Any]]
    reference: Any = None
    source: str = "benchmark"
    evaluator_version: str
    raw: dict[str, Any] = Field(default_factory=dict)


class PredictedRubric(Record):
    rubric_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    source: Literal["explicit", "inferred", "experience"] = "inferred"
    importance: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    confidence: float = Field(default=0.5, ge=0, le=1)
    expected_evidence: list[str] = Field(default_factory=list)
    applicability: str = "current task"
    prohibition: bool = False
    experience_ids: list[str] = Field(default_factory=list)


class RubricEdge(Record):
    source: str
    target: str
    relation: Literal["prerequisite", "support", "overlap", "tradeoff"]
    rationale: str
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class RubricGraph(Record):
    rubrics: list[PredictedRubric]
    edges: list[RubricEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [r.rubric_id for r in self.rubrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rubric IDs")
        if any(e.source not in ids or e.target not in ids for e in self.edges):
            raise ValueError("Rubric edge refers to an unknown rubric")
        return self


class AgentHarnessPolicy(Record):
    harness_id: Literal["rubric_mas"] = "rubric_mas"
    memory_policy: Literal["full", "recent"] = "full"
    memory_window: int = Field(default=8, ge=1, le=100)
    tool_policy: Literal["all_allowed", "preferred_first"] = "all_allowed"


class AgentMemoryLesson(Record):
    lesson_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=3000)
    applicability: str = Field(min_length=1)
    capability: str = ""
    task_signals: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    counterevidence: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class AgentProfile(Record):
    pool_agent_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    version: int = Field(default=1, ge=1)
    role: str = Field(min_length=1)
    capabilities: list[str] = Field(min_length=1)
    parent_agent_id: str | None = None
    prompt: str = Field(min_length=1, max_length=6000)
    skills: dict[str, str] = Field(default_factory=dict)
    preferred_tools: list[str] = Field(default_factory=list)
    memory: list[AgentMemoryLesson] = Field(default_factory=list)
    reasoning_strategy: str = ""
    planning_strategy: str = ""
    communication: str = ""
    harness: AgentHarnessPolicy = Field(default_factory=AgentHarnessPolicy)
    source_task_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_history(self):
        ids = [lesson.lesson_id for lesson in self.memory]
        if len(ids) != len(set(ids)) or len(self.source_task_ids) != len(set(self.source_task_ids)):
            raise ValueError("Agent history must have distinct lessons and tasks")
        return self


class AgentPoolOperation(Record):
    operation_id: str = Field(min_length=1)
    kind: Literal["add", "delete", "prune", "split", "merge", "specialize", "reorganize"]
    source_task_id: str = Field(min_length=1)
    base_pool_version: int = Field(ge=0)
    target_agent_ids: list[str] = Field(default_factory=list)
    base_agent_versions: dict[str, int] = Field(default_factory=dict)
    profiles: list[AgentProfile] = Field(default_factory=list)
    parent_assignments: dict[str, str | None] = Field(default_factory=dict)
    evidence: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_benefit: str = Field(min_length=1)
    token_cost_tradeoff: str = Field(min_length=1)


class AgentPoolObservation(Record):
    task_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    pool_agent_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    temporary: bool = False
    submission_score: float | None = Field(default=None, allow_inf_nan=False)
    complete: bool
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    usage_available: bool = False
    usage_estimated: bool = True
    evidence: list[str] = Field(min_length=1)


class AgentPoolSnapshot(Record):
    version: int = Field(default=0, ge=0)
    initialized: bool = False
    profiles: list[AgentProfile] = Field(default_factory=list)
    applied_updates: list[str] = Field(default_factory=list)
    observations: list[AgentPoolObservation] = Field(default_factory=list)
    structural_history: list[AgentPoolOperation] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_agents(self):
        ids = [profile.pool_agent_id for profile in self.profiles]
        if len(ids) != len(set(ids)) or len(self.applied_updates) != len(set(self.applied_updates)):
            raise ValueError("Agent Pool IDs and update IDs must be unique")
        parents = {profile.pool_agent_id: profile.parent_agent_id for profile in self.profiles}
        for agent_id in parents:
            visited = {agent_id}
            parent = parents[agent_id]
            while parent is not None:
                if parent not in parents:
                    raise ValueError("Agent Pool parent refers to an unknown member")
                if parent in visited:
                    raise ValueError("Agent Pool hierarchy contains a cycle")
                visited.add(parent)
                parent = parents[parent]
        return self


class AgentEvolutionUpdate(Record):
    update_id: str = Field(min_length=1)
    pool_agent_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    base_agent_version: int = Field(ge=1)
    source_task_id: str = Field(min_length=1)
    lessons: list[AgentMemoryLesson] = Field(default_factory=list)
    prompt: str | None = Field(default=None, min_length=1, max_length=6000)
    skills: dict[str, str] | None = None
    preferred_tools: list[str] | None = None
    reasoning_strategy: str | None = None
    planning_strategy: str | None = None
    communication: str | None = None
    harness: AgentHarnessPolicy | None = None
    evidence: list[str] = Field(min_length=1)


class AgentSpec(Record):
    agent_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    role: str
    capability: str
    rubric_ids: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    max_calls: int | None = Field(default=1, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    checkpoints: list[str] = Field(default_factory=list)
    pool_agent_id: str | None = None
    pool_agent_version: int | None = Field(default=None, ge=1)
    temporary_profile: AgentProfile | None = None
    creation_rationale: str = ""
    task_prompt: str = ""
    selected_skills: list[str] | None = None
    reasoning_strategy: str = ""
    harness: AgentHarnessPolicy | None = None
    communication: str = ""

    @model_validator(mode="after")
    def bound_identity(self):
        if (self.pool_agent_id is None) != (self.pool_agent_version is None):
            raise ValueError("Pool identity and version must be bound together")
        if self.temporary_profile is not None:
            if (self.pool_agent_id != self.temporary_profile.pool_agent_id
                    or self.pool_agent_version != self.temporary_profile.version):
                raise ValueError("Temporary harness identity must match its task binding")
            if not self.creation_rationale.strip():
                raise ValueError("Temporary harness creation requires a rationale")
            if (self.temporary_profile.version != 1 or self.temporary_profile.memory
                    or self.temporary_profile.source_task_ids or self.temporary_profile.evidence):
                raise ValueError("Temporary harness cannot invent learned history")
        return self


class AgentBudgetEstimate(Record):
    agent_id: str = Field(min_length=1)
    expected_model_calls: int = Field(ge=1)
    expected_input_tokens: int = Field(ge=0)
    expected_output_tokens: int = Field(ge=1)
    expected_tool_calls: int = Field(default=0, ge=0)
    expected_communication_bytes: int = Field(default=0, ge=0)
    rationale: str = Field(min_length=1)


class TeamBudgetPlan(Record):
    agents: list[AgentBudgetEstimate] = Field(min_length=1)
    reserved_future_tokens: int = Field(default=0, ge=0)
    reserved_future_model_calls: int = Field(default=0, ge=0)
    quality_cost_tradeoff: str = Field(min_length=1)
    stopping_policy: str = Field(min_length=1)

    def totals(self) -> dict[str, int]:
        return {
            "model_calls": sum(agent.expected_model_calls for agent in self.agents),
            "input_tokens": sum(agent.expected_input_tokens for agent in self.agents),
            "output_tokens": sum(agent.expected_output_tokens for agent in self.agents),
            "tool_calls": sum(agent.expected_tool_calls for agent in self.agents),
            "communication_bytes": sum(agent.expected_communication_bytes for agent in self.agents),
        }


class TeamSpec(Record):
    execution_mode: Literal["single_pass", "iterative_shared_ledger"] = "single_pass"
    agents: list[AgentSpec] = Field(min_length=1, max_length=16)
    synthesizer_id: str
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    primary: dict[str, str] = Field(default_factory=dict)
    reviewers: dict[str, list[str]] = Field(default_factory=dict)
    selection_rationale: str = ""
    termination: str = "All required artifacts and final synthesis submitted"
    max_parallel: int = Field(default=2, ge=1, le=16)
    total_max_calls: int | None = Field(default=16, ge=1)
    budget_plan: TeamBudgetPlan | None = None

    @model_validator(mode="after")
    def validate_team(self):
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)) or self.synthesizer_id not in ids:
            raise ValueError("Duplicate agent ID or unknown synthesizer")
        if self.execution_mode == "single_pass" and any(a.max_calls != 1 for a in self.agents):
            raise ValueError("single-pass execution requires AgentSpec.max_calls=1")
        if self.total_max_calls is not None and all(a.max_calls is not None for a in self.agents) and sum(a.max_calls for a in self.agents) > self.total_max_calls:
            raise ValueError("Agent allocations exceed team call budget")
        if self.budget_plan is not None:
            estimates = {estimate.agent_id: estimate for estimate in self.budget_plan.agents}
            if len(estimates) != len(self.budget_plan.agents) or set(estimates) != set(ids):
                raise ValueError("Budget estimates must identify every selected agent exactly once")
            for agent in self.agents:
                estimate = estimates[agent.agent_id]
                if self.execution_mode == "single_pass" and estimate.expected_model_calls != 1:
                    raise ValueError("single-pass budget estimates require expected_model_calls=1")
                if agent.max_calls is not None and estimate.expected_model_calls > agent.max_calls:
                    raise ValueError("Expected agent calls exceed the configured role ceiling")
                if estimate.expected_output_tokens > agent.max_tokens * estimate.expected_model_calls:
                    raise ValueError("Expected output tokens exceed the role response ceilings")
                if estimate.expected_tool_calls and not agent.tools:
                    raise ValueError("Expected external tool calls require allowed agent tools")
            if (self.total_max_calls is not None
                    and self.budget_plan.totals()["model_calls"] > self.total_max_calls):
                raise ValueError("Expected calls exceed the team call ceiling")
        remaining = {a.agent_id: set(a.depends_on) for a in self.agents}
        if any(not deps <= set(ids) for deps in remaining.values()):
            raise ValueError("Unknown execution dependency")
        done: set[str] = set()
        while remaining:
            ready = [key for key, deps in remaining.items() if deps <= done]
            if not ready:
                raise ValueError("Execution dependencies contain a cycle")
            for key in ready:
                done.add(key)
                del remaining[key]
        for assignments in (self.coverage, self.reviewers):
            if any(not set(owners) <= set(ids) for owners in assignments.values()):
                raise ValueError("Unknown rubric owner/reviewer")
        if any(owner not in ids for owner in self.primary.values()):
            raise ValueError("Unknown primary owner")
        for rid, owner in self.primary.items():
            if owner not in self.coverage.get(rid, []):
                raise ValueError("Primary owner must be in coverage")
        dependencies = {a.agent_id: set(a.depends_on) for a in self.agents}
        review_violations = []
        for rid, reviewers in self.reviewers.items():
            owner = self.primary.get(rid)
            if not owner:
                raise ValueError("Reviewed rubric needs a primary owner")
            for reviewer in reviewers:
                ancestors = set(dependencies[reviewer])
                previous = set()
                while ancestors != previous:
                    previous = set(ancestors)
                    ancestors.update(parent for aid in previous for parent in dependencies[aid])
                if reviewer == owner or owner not in ancestors:
                    review_violations.append(
                        f"rubric_id={rid!r}; reviewer_id={reviewer!r}; "
                        f"primary_owner_id={owner!r}; reviewer_ancestor_ids={sorted(ancestors)}")
        if review_violations:
            raise ValueError("Independent reviewer must depend on the primary owner's artifact: "
                             + " | ".join(review_violations) + ". "
                             "Choose an independent downstream reviewer with the primary owner "
                             "as an ancestor through depends_on, or omit this optional review "
                             "assignment. Do not add a backward dependency or cycle; self-checks "
                             "belong in checkpoints, not reviewers.")
        return self


class LocalPlan(Record):
    agent_id: str
    capability: str
    rubric_ids: list[str] = Field(default_factory=list)
    additions: list[PredictedRubric] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    max_calls: int | None = Field(default=1, ge=1)
    uncovered: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    challenge: str = ""
    selected_skills: list[str] | None = None
    reasoning_strategy: str = ""
    harness: AgentHarnessPolicy | None = None
    communication: str = ""


class Prediction(Record):
    graph: RubricGraph
    candidates: list[AgentSpec] = Field(min_length=1)


class PlannedTeam(Record):
    graph: RubricGraph
    team: TeamSpec
    local_plans: list[LocalPlan] = Field(default_factory=list)


class EvidenceRef(Record):
    run_id: str
    agent_id: str
    event_id: str
    timestamp: str = Field(default_factory=utc_now)
    source: str = "message"
    artifact_version: int = 1
    content_hash: str = ""
    locator: str = ""
    parent_event_ids: list[str] = Field(default_factory=list)


class TeamEvent(EvidenceRef):
    kind: str
    content: Any = None
    recipient: str = ""


class RubricFeedback(Record):
    rubric_id: str
    criterion: str
    weight: float = Field(allow_inf_nan=False)
    score: float | None = Field(default=None, allow_inf_nan=False)
    verdict: str = ""
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    status: Literal["ok", "missing", "error"] = "ok"
    raw: dict[str, Any] = Field(default_factory=dict)


class EvaluationFeedback(Record):
    task_id: str
    evaluator_version: str
    rubrics: list[RubricFeedback]
    score: float | None = Field(allow_inf_nan=False)
    complete: bool
    aggregation: str = "sum(weight*score)/sum(positive weights)"
    zero_denominator: bool = False
    source: str = "benchmark"
    raw: dict[str, Any] = Field(default_factory=dict)


class AlignmentMatch(Record):
    predicted_ids: list[str]
    evaluated_ids: list[str]
    relation: Literal["equivalent", "partial", "split", "merge", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    rationale: str
    importance_difference: float | None = None


class RubricAlignment(Record):
    matches: list[AlignmentMatch] = Field(default_factory=list)
    missed_evaluated_ids: list[str] = Field(default_factory=list)
    unmatched_predicted_ids: list[str] = Field(default_factory=list)


class RubricCreditAssignment(Record):
    agent_id: str = Field(min_length=1)
    predicted_rubric_ids: list[str] = Field(default_factory=list)
    evaluated_rubric_ids: list[str] = Field(default_factory=list)
    graph_edges: list[RubricEdge] = Field(default_factory=list)
    assignment_basis: dict[str, list[Literal["agent_spec", "coverage", "primary", "reviewer"]]] = Field(
        default_factory=dict)
    meta_assigned_evaluated_rubric_ids: list[str] = Field(default_factory=list)
    assignment_rationale: str = ""
    causal_identification: Literal[False] = False


class AttributionFinding(Record):
    finding_id: str
    rubric_ids: list[str]
    categories: list[Literal["prediction", "organization", "execution", "external_or_uncertain"]]
    agent_ids: list[str] = Field(default_factory=list)
    component: str = ""
    hypothesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    opposing_evidence: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    success: bool = False


class Experience(Record):
    experience_id: str
    bank: Literal["rubric", "organization", "execution"]
    instruction: str = Field(min_length=1, max_length=3000)
    applicability: str
    capability: str = ""
    task_signals: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    counterevidence: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def read_legacy_status(cls, value):
        if isinstance(value, dict) and "validation_status" in value:
            value = dict(value)
            status = value.pop("validation_status")
            if not isinstance(status, str) or status not in {"staged", "accepted"}:
                raise ValueError("Unknown historical experience status")
        return value


class ChangeProposal(Record):
    proposal_id: str
    source_task_id: str
    base_version: int = Field(ge=0)
    experience: Experience
    replaces_id: str | None = None
    diff: str
    rationale: str
    evidence: list[str] = Field(min_length=1)
    expected_benefit: str
    risks: list[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def read_legacy_metadata(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "validation_plan" in value and not isinstance(value.pop("validation_plan"), str):
                raise ValueError("Invalid historical validation plan")
            if "validation_result" in value and not isinstance(value.pop("validation_result"), dict):
                raise ValueError("Invalid historical validation result")
        return value


class ExperienceSnapshot(Record):
    version: int = 0
    experiences: list[Experience] = Field(default_factory=list)
    policy_versions: dict[str, str] = Field(default_factory=lambda: {
        "jit_mas": "1.0", "experience_update": "direct-v1"})
    applied_proposals: list[str] = Field(default_factory=list)
    agent_pool: AgentPoolSnapshot = Field(default_factory=AgentPoolSnapshot)

    @model_validator(mode="before")
    @classmethod
    def read_legacy_snapshot(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "accepted_proposals" in value:
            old = value.pop("accepted_proposals")
            if "applied_proposals" in value and value["applied_proposals"] != old:
                raise ValueError("Conflicting historical proposal IDs")
            value["applied_proposals"] = old
        if isinstance(value.get("experiences"), list):
            # Historical drafts were not bank members and must not become active.
            value["experiences"] = [entry for entry in value["experiences"]
                                    if not (isinstance(entry, dict)
                                            and entry.get("validation_status") == "staged")]
        return value


class SplitManifest(Record):
    seed: int = 0
    evolution: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    stream: list[str] = Field(default_factory=list)
    deduplication: str = "task_id and normalized public question SHA256"

    @model_validator(mode="after")
    def disjoint(self):
        groups = [self.evolution, self.validation, self.test, self.stream]
        flat = [item for group in groups for item in group]
        if len(flat) != len(set(flat)):
            raise ValueError("Splits must contain distinct, complete tasks")
        return self
