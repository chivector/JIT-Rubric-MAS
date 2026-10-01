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


class AgentSpec(Record):
    agent_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    role: str
    capability: str
    rubric_ids: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    max_calls: int = Field(default=1, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    checkpoints: list[str] = Field(default_factory=list)


class TeamSpec(Record):
    agents: list[AgentSpec] = Field(min_length=1, max_length=16)
    synthesizer_id: str
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    primary: dict[str, str] = Field(default_factory=dict)
    reviewers: dict[str, list[str]] = Field(default_factory=dict)
    selection_rationale: str = ""
    termination: str = "All required artifacts and final synthesis submitted"
    max_parallel: int = Field(default=2, ge=1, le=16)
    total_max_calls: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def validate_team(self):
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)) or self.synthesizer_id not in ids:
            raise ValueError("Duplicate agent ID or unknown synthesizer")
        if sum(a.max_calls for a in self.agents) > self.total_max_calls:
            raise ValueError("Agent allocations exceed team call budget")
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
                    raise ValueError("Independent reviewer must depend on the primary owner's artifact")
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
    max_calls: int = Field(default=1, ge=1)
    uncovered: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    challenge: str = ""


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
