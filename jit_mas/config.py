"""Explicit per-role endpoints. No credential or execution-model fallback."""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from .budget import MeteredModel
from .schemas import Record, TeamSpec


class ModelConfig(Record):
    model: str = ""
    endpoint: str = ""
    key_env: str = ""
    max_tokens: int = Field(default=4096, ge=1)
    timeout: float = Field(default=120, gt=0)
    temperature: float = 0
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["none", "low", "high", "max"] | None = None
    context_window: int | None = Field(default=None, ge=1)
    context_margin: int = Field(default=2048, ge=0)
    context_policy: Literal["reject", "oldest_turns"] = "reject"
    expected_response_model: str | None = None
    # Keep the provider transport policy explicit and out of the completion
    # request body.  Windows httpx may otherwise inherit the system proxy.
    http_trust_env: bool = True

    @model_validator(mode="after")
    def context_limits(self):
        if self.context_window is not None and self.max_tokens + self.context_margin >= self.context_window:
            raise ValueError("Context window must exceed the output reservation and margin")
        return self

    def check(self, role):
        if not self.model or not self.endpoint or "${" in self.model or "${" in self.endpoint:
            raise ValueError(f"native_jit requires explicit {role} model and endpoint")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid {role} endpoint URL")
        if not self.key_env or not os.environ.get(self.key_env):
            raise ValueError(f"Missing API credential environment variable for {role}: {self.key_env}")


class MASConfig(Record):
    backend: Literal["scripted", "native_jit"] = "native_jit"
    execution_mode: Literal["single_pass", "iterative_shared_ledger"] = "single_pass"
    unsafe_local: bool = False
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    max_agents: int = Field(default=4, ge=1, le=16)
    max_parallel: int = Field(default=2, ge=1, le=16)
    judge_parallel: int = Field(default=1, ge=1, le=16)
    team_max_calls: int | None = Field(default=16, ge=1)
    max_model_calls: int | None = Field(default=100, ge=1)
    max_total_tokens: int = Field(default=2_000_000, ge=1)
    max_tool_calls: int | None = Field(default=64, ge=0)
    max_repairs: int = Field(default=2, ge=0, le=5)
    candidates: int = Field(default=1, ge=1, le=4)
    execution_timeout: float = Field(default=120, gt=0)
    task_timeout: float | None = Field(default=None, gt=0)
    max_inflight_requests: int | None = Field(default=None, ge=1)
    local_planning: bool = True
    # Constrain generated planning records while keeping semantic graph validation.
    planning_response_format: Literal["json_object", "json_schema"] = "json_object"
    # Two public-only global review/revision calls before submission; same ledger.
    public_refinement: bool = False
    # Select a valid draft when an ordinary rewrite locally fails or loses its body.
    # This public-only selection never inspects evaluator scores or rescues execution.
    public_refinement_guard: bool = False
    public_revision_mode: Literal["full", "patch"] = "full"
    public_membership_observations: bool = False
    public_membership_input_format: Literal["full", "compact"] = "full"
    # Surface conditional numeric contradictions for explicit public review.
    # Lexical mentions are attention cues, never certified answer membership.
    public_membership_attention_checks: bool = False
    public_construction_validation_retries: int = Field(default=0, ge=0, le=1)
    public_refinement_response_format: Literal["json_object", "json_schema", "json_schema_review"] = "json_object"
    public_positional_construction: bool = False
    public_positional_draft_guidance: bool = False
    public_positional_draft_projection: bool = False
    public_numeric_construction: bool = False
    public_numeric_construction_layout: Literal["array", "named_objects", "template"] = "array"
    public_construction_response_format: Literal["json_schema", "json_object"] = "json_schema"
    public_revision_frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    # These rounds reconcile plans before execution, never rerun task contributors.
    local_rounds: int = Field(default=1, ge=1, le=3)
    local_attribution: bool = True
    persistent_experience: bool = True
    evolving_agent_pool: bool = True
    # Compatibility with frozen configs that explicitly disabled the removed
    # difficulty-to-roster rule; new runs cannot re-enable that rule.
    adaptive_budget_enforcement: Literal[False] = False
    explicit_rubrics: bool = True
    fixed_team: TeamSpec | None = None
    seed: int = 0
    available_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def removed_promotion_settings(cls, value):
        if isinstance(value, dict) and {"validation", "max_validation_tasks"}.intersection(value):
            raise ValueError("Experience updates are direct after attribution; remove obsolete "
                             "validation and max_validation_tasks settings")
        return value

    @model_validator(mode="after")
    def fixed_team_limits(self):
        if self.public_membership_attention_checks and not (
                self.public_refinement and self.public_membership_observations):
            raise ValueError("Public membership attention checks require public_refinement "
                             "and public_membership_observations")
        if self.public_revision_mode == "patch" and not self.public_refinement:
            raise ValueError("Public patch revision requires public_refinement")
        if self.public_refinement_guard and not self.public_refinement:
            raise ValueError("Public refinement guard requires public_refinement")
        if self.public_positional_draft_projection and not all((
                self.public_refinement, self.public_positional_construction,
                self.public_positional_draft_guidance)):
            raise ValueError("Public positional draft projection requires public_refinement, "
                             "public_positional_construction and public_positional_draft_guidance")
        if self.public_positional_draft_guidance and (
                not self.public_refinement or not self.public_positional_construction):
            raise ValueError("Public positional draft guidance requires public_refinement "
                             "and public_positional_construction")
        if self.public_positional_construction and (
                not self.public_refinement or self.public_refinement_response_format not in {
                    "json_schema", "json_schema_review"}):
            raise ValueError("Public positional construction requires public_refinement and json_schema or json_schema_review output")
        if self.public_numeric_construction and (
                not self.public_refinement or self.public_refinement_response_format not in {
                    "json_schema", "json_schema_review"}):
            raise ValueError("Public numeric construction requires public_refinement and json_schema or json_schema_review output")
        team = self.fixed_team
        if team and (len(team.agents) > self.max_agents or team.max_parallel > self.max_parallel
                     or (self.team_max_calls is not None
                         and (team.total_max_calls is None or team.total_max_calls > self.team_max_calls))):
            raise ValueError("Fixed team exceeds configured agent, concurrency or call limits")
        if team and team.execution_mode != self.execution_mode:
            raise ValueError("Fixed team execution_mode must match MASConfig.execution_mode")
        return self

    def check_native(self):
        for role in ("meta", "global", "local", "exec", "judge"):
            if role not in self.models:
                raise ValueError(f"native_jit requires explicit {role} model and endpoint")
            self.models[role].check(role)
        if not self.unsafe_local:
            raise RuntimeError("No trusted generated-code sandbox is configured. native_jit fails closed; "
                               "--unsafe-local explicitly permits execution with host access.")


class NativeModels:
    def __init__(self, config: MASConfig):
        config.check_native()
        self.config = config

    def create(self, role, agent_id, ledger, stage):
        from scripts.models.openai_server import OpenAIServerModel

        cfg = self.config.models[role]
        options = {}
        if role in {"global", "local"}:
            # Planning always returns a structured record. Execution shares its
            # provider with direct-answer and native harness calls, so its JSON
            # format is requested only at the strict protocol call sites.
            options["response_format"] = {"type": "json_object"}
        if cfg.frequency_penalty is not None:
            options["frequency_penalty"] = cfg.frequency_penalty
        if cfg.thinking is not None:
            options["extra_body"] = {"thinking": {"type": cfg.thinking}}
        if cfg.reasoning_effort is not None:
            options["reasoning_effort"] = cfg.reasoning_effort
        attempts = max(1, int(os.getenv("JIT_MAS_MODEL_ATTEMPTS", "1")))
        model = OpenAIServerModel(model_id=cfg.model, api_base=cfg.endpoint,
                                  api_key=os.environ[cfg.key_env], temperature=cfg.temperature,
                                  max_tokens=cfg.max_tokens, max_attempts=attempts,
                                  http_trust_env=cfg.http_trust_env, **options)
        model.client = model.client.with_options(max_retries=0, timeout=cfg.timeout)
        from .request_policy import RequestPolicyModel, request_gate
        model = RequestPolicyModel(model, gate=request_gate(self.config.max_inflight_requests),
            ledger=ledger, timeout=cfg.timeout, expected_model=cfg.expected_response_model)
        return MeteredModel(model, ledger, stage, agent_id, cfg.max_tokens,
                            context_window=cfg.context_window, context_margin=cfg.context_margin,
                            context_policy=cfg.context_policy)
