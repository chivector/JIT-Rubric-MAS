"""Run one isolated ResearchRubrics direct-vs-JIT-MAS pilot in parallel.

This pilot uses the frozen v5 ResearchRubrics membership (20 EVO, 10 VAL and
33 TEST) for one run. Shared public evidence is required by default. An
explicit --closed-book flag registers a separate protocol deviation.
Credentials are read from environment variables and are never written to the
experiment artifacts.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import os
import logging
import re
import shutil
from pathlib import Path
import time

from jit_mas.checkpoints import CheckpointIntegrityError, CheckpointRunner, snapshot_store
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.experience import ExperienceStore
from jit_mas.execution import ITERATIVE_CONTINUATION_POLICY_VERSION
from jit_mas.independent_protocol import normalize_score
from jit_mas.pipeline import code_fingerprint, write_json as _write_json
from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest, utc_now
from scripts.mas_baseline_methods import JudgeEnvelopeModel, run_direct
from scripts.benchmark_jit_mas_live import SafeTransport
from scripts.run_jit_mas import make_pipeline
from jit_mas.test_release import TestRelease, score_with_pipeline


def write_json(path, value):
    for attempt in range(5):
        try:
            return _write_json(path, value)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest(joint_path):
    joint = _read(joint_path)
    membership = joint["memberships"]["researchrubrics"]
    schedules = [row for row in joint.get("joint_evolution_schedule", []) if row.get("run_id") == 0]
    if len(schedules) != 1:
        raise ValueError("Frozen joint manifest must contain exactly one run0 schedule")
    evolution_ids = set(membership["evolution"])
    scheduled = [task_id for task_id in schedules[0]["task_ids"] if task_id in evolution_ids]
    if len(scheduled) != len(evolution_ids) or set(scheduled) != evolution_ids:
        raise ValueError("run0 schedule does not contain exactly the RR evolution membership")
    manifest = SplitManifest(
        seed=20261001,
        evolution=scheduled,
        validation=membership["validation"],
        test=membership["test"],
    )
    if (len(manifest.evolution), len(manifest.validation), len(manifest.test)) != (20, 10, 33):
        raise ValueError("Frozen ResearchRubrics membership is not 20/10/33")
    return manifest


def _config(args):
    generator = dict(model=args.exec_model, endpoint=args.exec_endpoint,
                      key_env="RR_EXEC_API_KEY", max_tokens=16000, timeout=args.timeout,
                      temperature=0, thinking="disabled", reasoning_effort="none",
                      context_window=131072, context_margin=2048,
                      context_policy="oldest_turns")
    judge = dict(model=args.judge_model, endpoint=args.judge_endpoint,
                 key_env="RR_JUDGE_API_KEY", max_tokens=getattr(args, "judge_max_tokens", 16000), timeout=args.timeout,
                 temperature=0, thinking="disabled", reasoning_effort="none",
                 context_window=131072, context_margin=2048,
                 context_policy="oldest_turns")
    return MASConfig(
        backend="native_jit", execution_mode="iterative_shared_ledger", unsafe_local=True,
        models={"meta": ModelConfig(**generator), "global": ModelConfig(**generator),
                "local": ModelConfig(**generator),
                "exec": ModelConfig(**{**generator, "max_tokens": 8192}),
                "judge": ModelConfig(**judge)},
        max_agents=3, max_parallel=2, team_max_calls=None, max_model_calls=None,
        max_total_tokens=2_000_000, max_tool_calls=None, max_repairs=2, candidates=1,
        max_inflight_requests=getattr(args, "max_inflight_requests", 2),
        execution_timeout=900, task_timeout=900, local_planning=True, local_rounds=1,
        local_attribution=True, persistent_experience=True, evolving_agent_pool=True,
        explicit_rubrics=True, available_tools=[])


def _pipeline(config, store, output, args, manifest, evidence_dir):
    split_path = Path(args.split_path)
    pipeline = make_pipeline(config, store, output, data=args.data, splits=split_path,
                             benchmark="researchrubrics", evidence_dir=evidence_dir,
                             knowledge_policy=_knowledge_policy(args))
    pipeline.models = SafeModels(pipeline.models, config,
                                 structured_output=getattr(args, "structured_output", "none"),
                                 structured_policy=_structured_output_policy(args))
    pipeline.synthesizer_factory = lambda meta: JITHarnessSynthesizer(
        backend="native_jit", meta_model=meta,
        meta_config={"model_id": config.models["meta"].model,
                     "api_base": config.models["meta"].endpoint, "api_key": "INJECTED",
                     "max_tokens": config.models["meta"].max_tokens},
        candidates=config.candidates, max_repairs=config.max_repairs,
        selector_model=meta, tools=pipeline.tools)
    judge_parallel = getattr(args, "judge_parallel", 2)
    if judge_parallel > 1:
        from benchmark.adapter.researchrubrics import ResearchRubricsAdapter

        def evaluator_factory(judge):
            ledger = judge.ledger if judge is not None else None
            next_index = 0

            def independent_judge():
                nonlocal next_index
                if ledger is None:
                    raise ValueError("A task ledger is required for judgment")
                agent_id = f"judge_rubric_{next_index}"
                next_index += 1
                return pipeline.models.create("judge", agent_id, ledger, "evaluation")

            spec = config.models["judge"]
            return ResearchRubricsAdapter(
                judge=judge, judge_id=spec.model, judge_api_base=spec.endpoint,
                judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout,
                max_attempts=getattr(args, "judge_attempts", 1), judge_factory=independent_judge,
                max_parallel_judgments=judge_parallel)

        pipeline.evaluator_factory = evaluator_factory
    return pipeline


def _knowledge_policy(args):
    return "model_general_knowledge_allowed" if getattr(args, "closed_book", False) else None


def _structured_output_policy(args=None):
    return {"planning_string_max_length": getattr(args, "planning_string_max_length", 2048),
            "planning_communication_max_length": getattr(args, "planning_communication_max_length", 1024),
            "planning_array_max_items": getattr(args, "planning_array_max_items", 64),
            "execution_schema_version": "rr-execution-v4",
            "execution_answer_characters_per_token": 2,
            "execution_checkpoint_reason_max_length": 512,
            "execution_ledger_text_max_length": 2048,
            "reference_schema_version": "rr-reference-v2"}


def _bounded_planning_schema(schema, policy):
    bounded = copy.deepcopy(schema)

    def visit(node, field=""):
        if isinstance(node, dict):
            if node.get("type") == "string":
                limit = policy["planning_communication_max_length"] if field == "communication" \
                    else policy["planning_string_max_length"]
                node["maxLength"] = min(node.get("maxLength", limit), limit)
            if node.get("type") == "array":
                limit = policy["planning_array_max_items"]
                node["maxItems"] = min(node.get("maxItems", limit), limit)
            for key, value in node.items():
                if key == "properties":
                    for name, child in value.items():
                        visit(child, name)
                elif isinstance(value, (dict, list)):
                    visit(value, field)
        elif isinstance(node, list):
            for child in node:
                visit(child, field)

    visit(bounded)
    return bounded


def _reference_schema(schema, payload):
    bounded = copy.deepcopy(schema)
    definitions = bounded.get("$defs", {})

    def restrict_array(node, values):
        if not isinstance(node, dict):
            return
        allowed = sorted(set(values))
        node["items"] = {"type": "string", "enum": allowed} if allowed else {"type": "string"}
        if not allowed:
            node["maxItems"] = 0

    phase = payload.get("phase")
    limits = payload.get("limits", {})
    if limits.get("execution_mode") == "iterative_shared_ledger" and limits.get("total_max_calls") is None:
        for definition in definitions.values():
            properties = definition.get("properties", {})
            for field in ("max_calls", "total_max_calls"):
                if field in properties:
                    properties[field] = {"type": "null"}
        if "max_calls" in bounded.get("properties", {}):
            bounded["properties"]["max_calls"] = {"type": "null"}
    if phase == "align":
        properties = definitions.get("AlignmentMatch", {}).get("properties", {})
        for field, key in (("predicted_ids", "valid_predicted_ids"),
                           ("evaluated_ids", "valid_evaluated_ids")):
            restrict_array(properties.get(field), payload.get(key, []))
            if field in properties:
                properties[field]["minItems"] = 1
        properties = bounded.get("properties", {})
        restrict_array(properties.get("unmatched_predicted_ids"), payload.get("valid_predicted_ids", []))
        restrict_array(properties.get("missed_evaluated_ids"), payload.get("valid_evaluated_ids", []))
    if phase in {"attribute_global", "attribute_integrate"}:
        agent_ids = [agent["agent_id"] for agent in payload.get("team", {}).get("agents", [])]
        properties = bounded.get("properties", {})
        questions = properties.get("questions")
        if isinstance(questions, dict):
            value_schema = questions.get("additionalProperties", {"type": "array", "items": {"type": "string"}})
            questions["properties"] = {agent_id: copy.deepcopy(value_schema) for agent_id in agent_ids}
            questions["additionalProperties"] = False
        findings = definitions.get("AttributionFinding", {}).get("properties", {})
        restrict_array(findings.get("agent_ids"), agent_ids)
        rubric_ids = [rubric["rubric_id"] for key in ("global_graph", "planned_graph", "feedback")
                      for rubric in payload.get(key, {}).get("rubrics", [])]
        restrict_array(findings.get("rubric_ids"), rubric_ids)
    if phase in {"attribute_global", "attribute_local", "attribute_local_followup", "attribute_integrate"}:
        findings = definitions.get("AttributionFinding", {}).get("properties", {})
        for field in ("supporting_evidence", "opposing_evidence"):
            restrict_array(findings.get(field), payload.get("evidence_ids", []))
        properties = bounded.get("properties", {})
        event_ids = [event["event_id"] for event in payload.get("event_index", [])]
        restrict_array(properties.get("evidence_requests"), event_ids if phase == "attribute_local" else [])
        if phase == "attribute_local" and "evidence_requests" in properties:
            properties["evidence_requests"]["maxItems"] = min(8, len(event_ids))
    if phase == "propose":
        properties = definitions.get("ChangeProposal", {}).get("properties", {})
        source_id = payload["task"]["task_id"]
        base_version = payload["base_version"]
        if "proposal_id" in properties:
            properties["proposal_id"]["pattern"] = "^" + re.escape(f"{source_id}:{base_version}:") + "[A-Za-z0-9_-]+$"
        for field, value in (("source_task_id", source_id), ("base_version", base_version)):
            if field in properties:
                properties[field]["const"] = value
        restrict_array(properties.get("evidence"), payload.get("valid_supporting_evidence_ids", []))
        experience = definitions.get("Experience", {}).get("properties", {})
        restrict_array(experience.get("evidence"), payload.get("valid_supporting_evidence_ids", []))
        restrict_array(experience.get("counterevidence"), payload.get("valid_counterevidence_ids", []))
        if "source_task_ids" in experience:
            experience["source_task_ids"]["const"] = [source_id]
        if "proposals" in bounded.get("properties", {}):
            bounded["properties"]["proposals"]["maxItems"] = 3
    if phase == "agent_evolve":
        properties = bounded.get("properties", {})
        for field in ("base_agent_version", "source_task_id", "update_id"):
            if field in properties:
                properties[field]["const"] = payload[field]
        if "pool_agent_id" in properties:
            properties["pool_agent_id"]["const"] = payload["agent_profile"]["pool_agent_id"]
        restrict_array(properties.get("evidence"), payload.get("valid_evidence_ids", []))
        lesson = definitions.get("AgentMemoryLesson", {}).get("properties", {})
        for field in ("evidence", "counterevidence"):
            restrict_array(lesson.get(field), payload.get("valid_evidence_ids", []))
        if "source_task_ids" in lesson:
            lesson["source_task_ids"]["const"] = [payload["source_task_id"]]
    return bounded


def _execution_response_schema(messages, max_tokens=None):
    text = {"type": "string", "minLength": 1}
    answer_text = {"type": "string"}

    instruction = {}
    payloads = []
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        try:
            candidate = json.loads(content)
        except (TypeError, ValueError):
            candidate = None
            for marker in ("Updated shared ledger: ", "Public observations: "):
                if content.startswith(marker):
                    try:
                        candidate = json.loads(content[len(marker):])
                    except (TypeError, ValueError):
                        pass
                    break
        if candidate is not None:
            payloads.append(candidate)
        if isinstance(candidate, dict) and isinstance(candidate.get("agent"), dict):
            instruction = candidate

    observed_ids = set()
    def collect_observed(value):
        if isinstance(value, dict):
            if isinstance(value.get("event_id"), str):
                observed_ids.add(value["event_id"])
            for key in ("observed_evidence_ids", "allowed_evidence_ids"):
                explicit = value.get(key)
                if isinstance(explicit, list):
                    observed_ids.update(item for item in explicit if isinstance(item, str))
            for key in ("observed_evidence_ids", "allowed_evidence_ids", "shared_ledger",
                        "contributions", "tool_evidence", "communications", "observations"):
                if key in value:
                    collect_observed(value[key])
        elif isinstance(value, list):
            for item in value:
                collect_observed(item)

    for payload in payloads:
        collect_observed(payload)
    ids = {"type": "array", "maxItems": len(observed_ids),
           "items": ({"type": "string", "enum": sorted(observed_ids)}
                     if observed_ids else {"type": "string"})}

    checkpoint = {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["completed", "passed", "failed", "unverified", "not_applicable"]},
        "reason": {**text, "maxLength": 512}, "evidence_ids": ids}, "required": ["status", "reason"], "additionalProperties": True}
    checkpoint_names = []
    agent = instruction.get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("checkpoints"), list):
        checkpoint_names = [name for name in agent["checkpoints"] if isinstance(name, str)]
    checkpoint_properties = {name: {"anyOf": [{"type": "boolean"}, checkpoint]}
                             for name in checkpoint_names}
    checkpoint_object = {"type": "object", "properties": checkpoint_properties,
                         "additionalProperties": False}
    if checkpoint_names:
        checkpoint_object["required"] = checkpoint_names

    source_ref = {"type": "string", "minLength": 1, "pattern": r"^[^,\s]+$"}
    ledger_text = {**text, "maxLength": 2048}
    ledger = {"type": "object", "properties": {
        "requirements": {"type": "array", "items": ledger_text},
        "outline": {"type": "array", "items": ledger_text},
        "source_references": {"type": "array", "items": {"type": "object", "properties": {
            "source_id": source_ref, "locator": text}, "required": ["source_id", "locator"], "additionalProperties": True}},
        "evidence_spans": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "maxLength": 2048}, "source_ref": source_ref}, "required": ["text", "source_ref"], "additionalProperties": True}}},
        "required": ["requirements", "outline", "source_references", "evidence_spans"], "additionalProperties": False}
    if instruction.get("knowledge_policy") == "model_general_knowledge_allowed":
        for field in ("source_references", "evidence_spans"):
            ledger["properties"][field]["maxItems"] = 0
    answer_schema = {"type": "string"}
    if isinstance(max_tokens, int) and max_tokens > 0:
        answer_schema["maxLength"] = max_tokens * 2
    completion = {"answer": {"anyOf": [answer_schema, {"type": "null"}]}, "evidence_ids": ids,
                  "checkpoints": checkpoint_object,
                  "ledger": {"anyOf": [ledger, {"type": "null"}]},
                  "continue": {"type": "boolean"}, "think": {}}
    terminal_tool = {"type": "object", "properties": {
        "name": {"type": "string", "enum": ["complete", "final_answer"]},
        "arguments": {"type": "object", "properties": completion, "additionalProperties": False}},
        "required": ["name", "arguments"], "additionalProperties": True}
    if checkpoint_names:
        terminal_tool["properties"]["arguments"]["required"] = ["answer", "checkpoints"]
    tool_shapes = [terminal_tool]
    allowed = set(agent.get("tools", [])) if isinstance(agent, dict) else set()
    allowed.difference_update({"complete", "final_answer", "send_message"})
    if allowed:
        tool_shapes.append({"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(allowed)},
            "arguments": {"type": "object", "additionalProperties": True}},
            "required": ["name"], "additionalProperties": True})
    tool_shapes.append({"type": "object", "properties": {
        "name": {"type": "string", "const": "send_message"},
        "arguments": {"type": "object", "properties": {
            "recipient": text, "content": text, "message": text}, "required": ["recipient"],
            "anyOf": [{"required": ["content"]}, {"required": ["message"]}], "additionalProperties": True}},
        "required": ["name", "arguments"], "additionalProperties": True})
    schema = {"type": "object", "properties": {
        **completion, "tools": {"type": "array", "items": {"anyOf": tool_shapes}}},
        "additionalProperties": False}
    if checkpoint_names:
        # Guided decoding on the execution gateway supports anyOf, but rejects
        # if/then. A turn either reports checkpoints, explicitly continues, or
        # dispatches a tool. Terminal tool arguments enforce their own contract.
        schema["anyOf"] = [
            {"required": ["checkpoints"]},
            {"required": ["continue"], "properties": {"continue": {"const": True}}},
            {"required": ["tools"], "properties": {"tools": {"minItems": 1}}},
        ]
    return schema


class StructuredOutputModel:
    def __init__(self, model, role, agent_id, mode, *, policy=None):
        if mode not in {"none", "json_object", "json_schema"}:
            raise ValueError("Unknown structured output mode")
        self.model, self.role, self.agent_id, self.mode = model, role, agent_id, mode
        self.policy = dict(policy or _structured_output_policy())

    def __call__(self, messages, **kwargs):
        if self.mode != "none" and self.role in {"global", "local"}:
            payload = next((json.loads(message["content"]) for message in messages
                            if message.get("role") == "user" and isinstance(message.get("content"), str)
                            and message["content"].lstrip().startswith("{")), {})
            marker = "Return only one JSON object conforming to this JSON Schema:\n"
            for message in messages:
                content = message.get("content")
                if message.get("role") == "system" and isinstance(content, str) and marker in content:
                    schema = json.loads(content.rsplit(marker, 1)[1])
                    schema = _bounded_planning_schema(schema, self.policy)
                    schema = _reference_schema(schema, payload)
                    kwargs["response_format"] = (
                        {"type": "json_schema", "json_schema": {
                            "name": schema.get("title", "StructuredResponse"), "strict": True, "schema": schema}}
                        if self.mode == "json_schema" else {"type": "json_object"})
                    break
        elif self.mode != "none" and self.role == "exec" and self.agent_id != "direct":
            kwargs["response_format"] = (
                {"type": "json_schema", "json_schema": {
                    "name": "ExecutionResponse", "strict": True, "schema": _execution_response_schema(
                        messages, max_tokens=kwargs.get("max_tokens"))}}
                if self.mode == "json_schema" else {"type": "json_object"})
        return self.model(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class SafeModels:
    def __init__(self, provider, config, *, structured_output="none", structured_policy=None):
        self.provider, self.config, self.structured_output = provider, config, structured_output
        self.structured_policy = dict(structured_policy or _structured_output_policy())

    def create(self, role, agent_id, ledger, stage):
        model = StructuredOutputModel(self.provider.create(role, agent_id, ledger, stage),
                                      role, agent_id, self.structured_output, policy=self.structured_policy)
        model = SafeTransport(model, os.environ[self.config.models[role].key_env])
        return JudgeEnvelopeModel(model) if role == "judge" else model


def _safe_error(error):
    message = str(error)
    for name in ("RR_EXEC_API_KEY", "RR_JUDGE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runner_hash():
    return _sha256_file(Path(__file__))


class _CredentialSafeFormatter(logging.Formatter):
    def format(self, record):
        return _safe_error(super().format(record))


def _configure_logging():
    logging.basicConfig(level=logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(_CredentialSafeFormatter("%(levelname)s:%(name)s:%(message)s"))


def _preflight(args, manifest, config, evidence_dir, *, require_credentials):
    """Validate the pinned RR inventory without making model requests."""
    data_path = Path(args.data).resolve()
    joint_path = Path(args.joint_manifest).resolve()
    if not data_path.is_file() or not joint_path.is_file():
        raise ValueError("Pinned data and joint manifest must exist")
    from jit_mas.benchmarks import load_benchmark
    dataset = load_benchmark("researchrubrics", data_path, available_tools=config.available_tools)
    raw_sha256 = _sha256_file(data_path)
    expected_sha256 = _read(joint_path)["benchmarks"]["researchrubrics"]["dataset_sha256"]
    if raw_sha256 != expected_sha256 or dataset.dataset_sha256 != raw_sha256:
        raise ValueError("RR data bytes do not match the frozen joint manifest")
    expected = set(manifest.evolution + manifest.validation + manifest.test)
    actual = set(dataset.tasks)
    if not expected <= actual:
        raise ValueError(f"Pinned RR split references missing task IDs ({len(expected - actual)})")
    if len(actual) < len(expected):
        raise ValueError("Pinned RR data contains fewer tasks than the registered inventory")
    if require_credentials:
        config.check_native()
    elif config.backend != "native_jit":
        raise ValueError("The pilot requires native_jit")
    for role in ("meta", "global", "local", "exec", "judge"):
        config.models[role].check(role) if require_credentials else _check_model_identity(config.models[role], role)
    if evidence_dir is not None:
        from jit_mas.evidence import load_evidence_tasks
        selected = {task_id: dataset.tasks[task_id] for task_id in expected}
        load_evidence_tasks(selected, evidence_dir, expected_count=len(selected))
        evidence_manifest = _read(evidence_dir / "manifest.json")
        source_count = notice_count = 0
        for entry in evidence_manifest["tasks"].values():
            pack = _read(evidence_dir / entry["file"])
            usable = sum(row.get("kind") == "web" and row.get("status") == "ok"
                         and bool(str(row.get("text", "")).strip()) for row in pack.get("sources", []))
            source_count += bool(usable)
            notice_count += not bool(usable)
        evidence_coverage = {"tasks_with_usable_public_sources": source_count,
                             "tasks_with_retrieval_notice_only": notice_count,
                             "task_count": len(expected)}
    else:
        evidence_coverage = None
    return {
        "data_sha256": raw_sha256,
        "joint_manifest_sha256": _sha256_file(joint_path),
        "dataset_sha256": dataset.dataset_sha256,
        "task_count": len(dataset.tasks),
        "selected_task_count": len(expected),
        "evidence_mode": "shared_public_evidence" if evidence_dir else "closed_book_public_tasks",
        "evidence_coverage": evidence_coverage,
    }


def _check_model_identity(spec, role):
    if not spec.model or not spec.endpoint or "${" in spec.model or "${" in spec.endpoint:
        raise ValueError(f"native_jit requires explicit {role} model and endpoint")
    from urllib.parse import urlparse
    parsed = urlparse(spec.endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid {role} endpoint URL")


def _assert_identity(args):
    identity = getattr(args, "frozen_identity", None)
    if identity is None:
        return
    if (identity["code"] != code_fingerprint()
            or identity["runner"] != _runner_hash()
            or identity["config"] != digest(_config(args))
            or identity["judge_parallel"] != getattr(args, "judge_parallel", 2)
            or identity.get("structured_output", "none") != getattr(args, "structured_output", "none")
            or identity.get("structured_policy", _structured_output_policy()) != _structured_output_policy(args)
            or identity.get("knowledge_policy", _knowledge_policy(args)) != _knowledge_policy(args)
            or any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
                   for path, expected in identity["files"].items())):
        raise CheckpointIntegrityError("Frozen pilot code, data or evidence changed")


def _score_rows(outcomes, failures=(), expected_ids=()):
    by_id = {outcome.get("task_id"): outcome for outcome in outcomes}
    failed_by_id = {failure.get("task_id"): failure for failure in failures}
    rows = []
    task_ids = list(expected_ids) or list(dict.fromkeys([*by_id, *failed_by_id]))
    for task_id in task_ids:
        outcome = by_id.get(task_id)
        if outcome is None:
            failure = failed_by_id.get(task_id, {})
            rows.append({"task_id": task_id, "score": None, "complete": False,
                         "run_dir": None, "budget": failure.get("budget"),
                         "error_type": failure.get("error_type")})
            continue
        feedback = outcome.get("evaluation") or {}
        rows.append({"task_id": task_id, "score": feedback.get("score") if feedback.get("complete") is True else None,
                     "complete": feedback.get("complete") is True,
                     "run_dir": outcome.get("run_dir"), "budget": outcome.get("budget")})
    scores = [row["score"] for row in rows if row["complete"] and isinstance(row["score"], (int, float))]
    return {"tasks": rows, "completed": sum(row["complete"] for row in rows),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "denominator": len(rows), "scored_denominator": len(scores)}


def _test_release(output, manifest, reports):
    inventory = [{"slot_id": f"{arm}:{task_id}", "task_id": task_id,
                  "method": arm, "repeat": 0}
                 for arm in reports for task_id in manifest.test]
    release = TestRelease(output / "test_release", inventory)
    for arm, report in reports.items():
        outcomes = {row["task_id"]: row for row in report.get("submitted_outcomes", [])}
        failures = {row["task_id"]: row for row in report.get(
            "failures" if arm == "baseline" else "test_failures", [])}
        for task_id in manifest.test:
            slot_id = f"{arm}:{task_id}"
            if task_id in outcomes:
                release.record(slot_id, outcomes[task_id])
            else:
                failure = failures.get(task_id, {})
                release.record_failure(slot_id, error_type=failure.get("error_type", "NoSelectedCheckpoint"),
                                       budget=failure.get("budget", {"usage_unknown": True}))
    release.seal()
    return release


def _score_deferred(args, config, manifest, evidence_dir, output, reports):
    release = _test_release(output, manifest, reports)

    def score_arm(arm):
        report = reports[arm]
        original_status = report.get("status")
        scoring_started = time.monotonic()
        store = ExperienceStore(output / arm / "experience.sqlite", read_only=True)
        before = digest(store.snapshot())
        try:
            pipeline = _pipeline(config, store, output / arm / "scoring", args, manifest, evidence_dir)
            cached = {}
            if getattr(args, "reuse_evaluations_from", []):
                evaluator_version = pipeline.evaluator_factory(None).evaluator_version
                cached = _cached_evaluations(args, release, arm, evaluator_version)
            results = []
            for task_id in manifest.test:
                _assert_identity(args)
                result = release.evaluate(f"{arm}:{task_id}", lambda record: cached[task_id]
                                          if task_id in cached else score_with_pipeline(pipeline, record))
                results.append(result)
                write_json(output / arm / "scoring_progress.json", {"results": results})
            scored = [row["official_score"] for row in results
                      if row.get("complete") and row.get("official_score") is not None]
            report["results" if arm == "baseline" else "test"] = {
                "tasks": results, "completed": len(scored), "denominator": len(manifest.test),
                "scored_denominator": len(scored),
                "mean_score": math.fsum(scored) / len(scored) if scored else None,
                "all_task_mean": math.fsum(scored) / len(manifest.test) if len(scored) == len(manifest.test) else None}
            report["status"] = original_status if original_status in ("inconclusive", "failed") else (
                "completed" if len(scored) == len(manifest.test) else "incomplete")
            report["experience_store_unchanged_by_test"] = digest(store.snapshot()) == before
            report["scoring_seconds"] = time.monotonic() - scoring_started
            report["finished_at"] = utc_now()
            write_json(output / arm / "report.json", report)
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=min(2, len(reports))) as workers:
        futures = [workers.submit(score_arm, arm) for arm in reports]
        for future in futures:
            future.result()


def _failure_report(error, arm):
    return {"arm": arm, "status": "failed", "error_type": type(error).__name__,
            "error": _safe_error(error), "submitted_outcomes": [],
            "failures": [], "test_failures": [], "finished_at": utc_now()}


def _run_arm(function, args, config, manifest, evidence_dir, output):
    try:
        return function(args, config, manifest, evidence_dir, output)
    except BaseException as error:
        report_path = output / "report.json"
        report = _read(report_path) if report_path.exists() else {}
        failed = _failure_report(error, output.name)
        failed.update({key: value for key, value in report.items()
                       if key not in ("status", "error", "error_type", "finished_at")})
        write_json(report_path, failed)
        return failed


def _cached_evaluations(args, release, arm, evaluator_version):
    cached = {}
    for source in getattr(args, "reuse_evaluations_from", []):
        directory = Path(source).resolve() / "test_release" / "evaluations"
        for path in sorted(directory.glob("*.json")):
            row = _read(path)
            task_id = row["slot"]["task_id"]
            if row["slot"]["method"] != arm or task_id in cached or row.get("complete") is not True:
                continue
            if row.get("evaluation", {}).get("evaluator_version") != evaluator_version:
                continue
            record = release._read(release._path(f"{arm}:{task_id}"))
            if record["status"] != "submitted" or row.get("answer_hash") != record["submission"]["answer_hash"]:
                raise ValueError("Cached evaluation answer differs from the sealed answer")
            if row.get("evaluation_budget", {}).get("reserved_tokens", 0):
                raise ValueError("Cached evaluation has unsettled calls")
            copied = dict(row)
            copied["evaluation_reused_from"] = str(path)
            cached[task_id] = copied
    return cached


def _baseline_reuse(args, manifest, config):
    source = getattr(args, "reuse_baseline_from", None)
    if source is None:
        return {}
    if args.arm != "baseline" or not args.closed_book:
        raise ValueError("Baseline reuse requires an isolated closed-book baseline")
    source = Path(source).resolve()
    metadata = _read(source / "pilot_metadata.json")
    if (metadata["manifest"]["test"] != manifest.test
            or metadata["execution_model"] != config.models["exec"].model
            or metadata["data_sha256"] != _sha256_file(args.data)
            or metadata.get("knowledge_policy") != "model_general_knowledge_allowed"):
        raise ValueError("Reused baseline has a different data, split, model or knowledge policy")
    source_exec = metadata.get("config", {}).get("models", {}).get("exec")
    if source_exec is not None and source_exec != config.models["exec"].model_dump(mode="json"):
        raise ValueError("Reused baseline execution configuration differs")
    reports = _read(source / "generation_reports.json")
    imported = {}
    for saved in reports["baseline"].get("submitted_outcomes", []):
        outcome = dict(saved)
        task_id = outcome["task_id"]
        if task_id not in manifest.test or task_id in imported:
            raise ValueError("Reused baseline task IDs must be unique frozen TEST members")
        if outcome.get("status") != "submitted_unscored" or outcome.get("evaluation") is not None:
            raise ValueError("Only unscored baseline generation outcomes can be reused")
        run_dir = Path(outcome["run_dir"]).resolve()
        if not run_dir.is_relative_to(source) or not run_dir.is_dir():
            run_dir = source / "baseline" / "tasks" / task_id
        submission = _read(run_dir / "submission.json")
        if digest(submission["answer"]) != submission["answer_hash"] or outcome["answer_hash"] != submission["answer_hash"]:
            raise ValueError("Reused baseline answer hash mismatch")
        from jit_mas.test_release import remaining_task_budget
        remaining_task_budget(config, outcome)
        outcome["_reuse_source_dir"] = str(run_dir)
        imported[task_id] = outcome
    return imported


def _import_baseline_outcome(outcome, output):
    imported = dict(outcome)
    source = Path(imported.pop("_reuse_source_dir"))
    destination = output / "tasks" / imported["task_id"]
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("complete.json", "*.tmp"))
    imported["run_dir"] = str(destination.resolve())
    imported["generation_reused_from"] = str(source)
    write_json(destination / "complete.json", imported)
    return imported


def run_baseline(args, config, manifest, evidence_dir, output):
    started = time.monotonic()
    started_at = utc_now()
    output.mkdir(parents=True, exist_ok=False)
    store = ExperienceStore(output / "experience.sqlite")
    try:
        pipeline = _pipeline(config, store, output / "pipeline", args, manifest, evidence_dir)
        outcomes = []
        failures = []
        report = {"arm": "single_agent_direct", "status": "running", "started_at": started_at,
                  "manifest": manifest.model_dump(mode="json"), "submitted_outcomes": outcomes,
                  "failures": failures, "experience_updates": False}
        write_json(output / "report.json", report)
        for task_id in manifest.test:
            _assert_identity(args)
            reusable = getattr(args, "baseline_reuse", {}).get(task_id)
            if reusable is not None:
                outcomes.append(_import_baseline_outcome(reusable, output))
                write_json(output / "report.json", report)
                continue
            task_dir = output / "tasks" / task_id
            write_json(output / "started" / f"{task_id}.json", {"task_id": task_id, "started_at": utc_now()})
            try:
                outcome = run_direct(
                    pipeline.tasks[task_id], None, pipeline.models,
                    pipeline.evaluator_factory, task_dir,
                    {"max_calls": 1, "max_tokens": config.max_total_tokens,
                     "output_tokens": config.models["exec"].max_tokens}, defer_evaluation=True)
                outcomes.append(outcome)
            except Exception as error:
                budget_path = task_dir / "budget.json"
                failures.append({"task_id": task_id, "error_type": type(error).__name__,
                                 "error": _safe_error(error),
                                 "budget": _read(budget_path) if budget_path.exists() else {"usage_unknown": True}})
            write_json(output / "report.json", report)
        report = {"arm": "single_agent_direct", "status": "submitted", "started_at": started_at,
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "one-run ResearchRubrics TEST, direct single-agent, closed-book unless evidence supplied",
                  "manifest": manifest.model_dump(mode="json"), "results": _score_rows(outcomes, failures, manifest.test),
                  "failures": failures, "submitted_outcomes": outcomes, "experience_version": store.snapshot().version,
                  "experience_updates": False}
        write_json(output / "report.json", report)
        return report
    finally:
        store.close()


def _validate_checkpoint(config, store, args, manifest, evidence_dir, output, snapshot,
                         label, task_id, repeat=0, frozen=None):
    checkpoint_dir = output / "checkpoints" / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    owned = frozen is None
    if owned:
        frozen = snapshot_store(snapshot, checkpoint_dir / f"{digest(snapshot)}.sqlite")
    before = digest(store.snapshot())
    state_hash = digest(snapshot)
    slot_dir = checkpoint_dir / "slots" / task_id
    write_json(slot_dir / "started.json", {"task_id": task_id, "repeat": repeat,
                                           "state_hash": state_hash, "started_at": utc_now()})
    try:
        _assert_identity(args)
        pipeline = _pipeline(config, frozen, checkpoint_dir / "runs", args, manifest, evidence_dir)
        outcome = pipeline.run_task(task_id, snapshot, mode="validate", repeat=repeat,
                                    attribution=False, resume=False)
        if outcome.get("experience_updates") or outcome.get("agent_pool_updates"):
            raise CheckpointIntegrityError("VAL attempted to update the frozen experience state")
        _assert_identity(args)
        write_json(slot_dir / "result.json", outcome)
        return outcome
    except Exception as error:
        failure = {**getattr(error, "jit_mas_run_failure", {}), "task_id": task_id,
                   "repeat": repeat, "state_hash": state_hash,
                   "error_type": type(error).__name__, "error": _safe_error(error)}
        write_json(slot_dir / "failed.json", failure)
        raise
    finally:
        changed = (digest(store.snapshot()) != before or digest(snapshot) != state_hash
                   or digest(frozen.snapshot()) != state_hash)
        if owned:
            frozen.close()
        if changed:
            raise CheckpointIntegrityError("VAL mutated its frozen snapshot or EVO trajectory")


class _NormalizedCheckpointRunner(CheckpointRunner):
    def __init__(self, *, upper_bounds, **kwargs):
        task_ids = list(kwargs["validation_ids"])
        self.upper_bounds = {task_id: float(upper_bounds[task_id]) for task_id in task_ids}
        lower_bounds = kwargs["lower_bounds"]
        if any(not math.isfinite(self.upper_bounds[task_id])
               or self.upper_bounds[task_id] < float(lower_bounds[task_id]) for task_id in task_ids):
            raise ValueError("VAL theoretical bounds must be finite and ordered")
        identity = dict(kwargs["identity"])
        identity.update(validation_upper_bounds=self.upper_bounds,
                        normalization="(score-L_t)/(U_t-L_t); missing=0; zero span=0")
        kwargs["identity"] = identity
        super().__init__(**kwargs)

    def _save(self):
        for cached in self.journal.get("validation_cache", {}).values():
            for row in cached["slots"]:
                task_id = row["task_id"]
                row["theoretical_bounds"] = [self.lower_bounds[task_id], self.upper_bounds[task_id]]
                row["selection_utility"] = 0.0
                if row.get("complete_evaluation"):
                    try:
                        row["selection_utility"] = normalize_score(row["official_score"], row["theoretical_bounds"])
                    except ValueError as error:
                        row.update(status="incomplete", complete_evaluation=False,
                                   is_imputed_for_selection=True, official_score=None,
                                   normalization_error=str(error))
        super()._save()


def run_ours(args, config, manifest, evidence_dir, output):
    started = time.monotonic()
    started_at = utc_now()
    output.mkdir(parents=True, exist_ok=False)
    store = ExperienceStore(output / "experience.sqlite")
    try:
        write_json(output / "report.json", {"arm": "ours_evolve_then_test", "status": "evolving",
                                           "started_at": started_at, "submitted_outcomes": [],
                                           "test_failures": []})
        pipeline = _pipeline(config, store, output / "evolution", args, manifest, evidence_dir)
        dataset = pipeline.benchmark_dataset
        identity = {"arm": "ours_evolve_then_test", "run_id": 0,
                    "manifest": manifest.model_dump(mode="json"),
                    "config": config.model_dump(mode="json"), "dataset_sha256": dataset.dataset_sha256,
                    "frozen_identity": getattr(args, "frozen_identity", None),
                    "selection": "fixed theoretical RR bounds, 9/10 complete, maximum normalized mean"}

        def evolve(task_id):
            _assert_identity(args)
            outcome = pipeline.run("evolve", [task_id], resume=False)
            _assert_identity(args)
            return outcome

        def evaluate(task_id, snapshot, repeat, frozen):
            label = f"C{len(runner.journal['sources'])}"
            return _validate_checkpoint(config, store, args, manifest, evidence_dir, output,
                                        snapshot, label, task_id, repeat, frozen)

        runner = _NormalizedCheckpointRunner(
            store=store, evolution_ids=manifest.evolution, validation_ids=manifest.validation,
            output_dir=output / "checkpoints", identity=identity,
            lower_bounds=dataset.lower_bounds, upper_bounds=dataset.upper_bounds,
            evolve=evolve, evaluate=evaluate, batch_size=5, repeats=1,
            minimum_completion=0.9)
        checkpoint_result = runner.run()
        evolution = [row["outcome"] for row in runner.journal["sources"] if row.get("outcome")]
        evolution_failures = [{"task_id": row["task_id"], **row["failure"]}
                              for row in runner.journal["sources"] if row["status"] == "failed"]
        checkpoints = {f"C{row['position']}": {
            **row, "validation_slots": runner.journal["validation_cache"][row["cache_key"]]["slots"]}
            for row in checkpoint_result["checkpoints"]}
        selected_row = checkpoint_result["selected"]
        selected = runner.selected_snapshot() if selected_row is not None else None
        selected_label = f"C{selected_row['position']}" if selected_row is not None else None
        selected_snapshot_path = output / "selected_snapshot.json"
        if selected is not None:
            write_json(selected_snapshot_path, selected.model_dump(mode="json"))
        test_outcomes = []
        test_failures = []
        report = {"arm": "ours_evolve_then_test", "status": "submitting" if selected is not None else "inconclusive",
                  "started_at": started_at, "manifest": manifest.model_dump(mode="json"),
                  "checkpoints": checkpoints, "checkpoint_selection": checkpoint_result,
                  "selected": selected_row, "selected_checkpoint": selected_label,
                  "selected_snapshot": str(selected_snapshot_path) if selected is not None else None,
                  "selected_snapshot_hash": digest(selected) if selected is not None else None,
                  "submitted_outcomes": test_outcomes, "test_failures": test_failures,
                  "experience_version": store.snapshot().version}
        write_json(output / "report.json", report)
        if selected is None:
            for task_id in manifest.test:
                failure = {"task_id": task_id, "status": "missing", "budget": None,
                           "error_type": "NoEligibleCheckpoint",
                           "error": "All five fixed VAL candidates failed the 9/10 completeness threshold"}
                test_failures.append(failure)
                write_json(output / "test" / "slots" / task_id / "missing.json", failure)
                write_json(output / "report.json", report)
        else:
            state_hash = digest(selected)
            trajectory_hash = digest(store.snapshot())
            test_state = snapshot_store(selected, output / "test" / "selected_state.sqlite")
            try:
                test_pipeline = _pipeline(config, test_state, output / "test" / "runs",
                                          args, manifest, evidence_dir)
                for task_id in manifest.test:
                    slot = output / "test" / "slots" / task_id
                    write_json(slot / "started.json", {"task_id": task_id, "started_at": utc_now(),
                                                         "state_hash": state_hash})
                    try:
                        _assert_identity(args)
                        outcome = test_pipeline.run_task(task_id, selected, mode="evaluate", repeat=0,
                                                         attribution=False, resume=False,
                                                         defer_evaluation=True)
                        if (outcome.get("evaluation") is not None
                                or outcome.get("status") != "submitted_unscored"
                                or outcome.get("experience_updates") or outcome.get("agent_pool_updates")):
                            raise CheckpointIntegrityError("TEST must return one unscored immutable submission")
                        _assert_identity(args)
                        test_outcomes.append(outcome)
                        write_json(slot / "submitted.json", outcome)
                        write_json(output / "report.json", report)
                    except CheckpointIntegrityError:
                        raise
                    except Exception as error:
                        failure = {**getattr(error, "jit_mas_run_failure", {}), "task_id": task_id,
                                   "status": "failed", "error_type": type(error).__name__,
                                   "error": _safe_error(error), "state_hash": state_hash}
                        failure.setdefault("budget", None)
                        test_failures.append(failure)
                        write_json(slot / "failed.json", failure)
                        write_json(output / "report.json", report)
                    finally:
                        if (digest(selected) != state_hash or digest(test_state.snapshot()) != state_hash
                                or digest(store.snapshot()) != trajectory_hash):
                            raise CheckpointIntegrityError("TEST mutated its selected state or EVO trajectory")
            finally:
                test_state.close()
        report = {"arm": "ours_evolve_then_test", "status": "submitted" if selected is not None else "inconclusive", "started_at": started_at,
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "one-run ResearchRubrics 20 EVO + 10 VAL + 33 TEST, closed-book unless evidence supplied",
                  "manifest": manifest.model_dump(mode="json"), "checkpoints": checkpoints,
                  "checkpoint_selection": checkpoint_result, "selected": selected_row,
                  "selected_checkpoint": selected_label,
                  "selected_snapshot": str(selected_snapshot_path) if selected is not None else None,
                  "selected_snapshot_hash": digest(selected) if selected is not None else None,
                  "evolution": _score_rows(evolution, evolution_failures, manifest.evolution),
                  "evolution_failures": evolution_failures,
                  "test": _score_rows(test_outcomes, test_failures, manifest.test),
                  "submitted_outcomes": test_outcomes,
                  "test_failures": test_failures, "experience_version": store.snapshot().version,
                  "test_feedback_updates_experience": False}
        write_json(output / "report.json", report)
        return report
    finally:
        store.close()


def _cost_summary(directory):
    seen = {}
    unknown_budgets = 0
    unsettled_calls = 0
    for path in Path(directory).rglob("budget.json"):
        budget = _read(path)
        if budget.get("usage_unknown"):
            unknown_budgets += 1
        settled = sum(row.get("kind") == "model" for row in budget.get("records", []))
        unsettled_calls += max(0, budget.get("model_calls", settled) - settled)
        for row in budget.get("records", []):
            call_id = row.get("call_id")
            if row.get("kind") == "model" and call_id:
                if call_id in seen and seen[call_id] != row:
                    raise CheckpointIntegrityError("A model-call ledger record changed across artifacts")
                seen[call_id] = row
    for path in Path(directory).rglob("scoring_progress.json"):
        for result in _read(path).get("results", []):
            for row in result.get("evaluation_budget", {}).get("records", []):
                if row.get("kind") == "model" and row.get("call_id"):
                    if row["call_id"] in seen and seen[row["call_id"]] != row:
                        raise CheckpointIntegrityError("A scoring ledger record changed across artifacts")
                    seen[row["call_id"]] = row
    stages = {}
    for row in seen.values():
        group = stages.setdefault(row["stage"], {"model_calls": 0, "input_tokens": 0,
                                                "output_tokens": 0, "estimated_calls": 0,
                                                "request_seconds": 0.0})
        group["model_calls"] += 1
        group["input_tokens"] += row["input_tokens"]
        group["output_tokens"] += row["output_tokens"]
        group["estimated_calls"] += bool(row.get("estimated"))
        group["request_seconds"] += row.get("wall_seconds", 0)
    return {"model_calls": len(seen), "tokens": sum(row["input_tokens"] + row["output_tokens"] for row in seen.values()),
            "by_stage": stages, "unknown_budgets": unknown_budgets, "unsettled_calls": unsettled_calls,
            "monetary_cost": None}


def _paired_comparison(reports, manifest):
    if set(reports) != {"baseline", "ours"}:
        return {"available": False, "reason": "Both arms are required for paired comparison"}
    by_arm = {arm: {row["slot"]["task_id"]: row for row in reports[arm].get(
        "results" if arm == "baseline" else "test", {}).get("tasks", []) if "slot" in row}
              for arm in ("baseline", "ours")}
    paired = []
    for task_id in manifest.test:
        baseline = by_arm["baseline"].get(task_id, {})
        ours = by_arm["ours"].get(task_id, {})
        complete = bool(baseline.get("complete") and ours.get("complete"))
        paired.append({"task_id": task_id, "baseline": baseline.get("official_score"),
                       "ours": ours.get("official_score"), "complete_pair": complete,
                       "difference": ours["official_score"] - baseline["official_score"] if complete else None})
    differences = [row["difference"] for row in paired if row["complete_pair"]]
    return {"tasks": paired, "denominator": len(manifest.test), "complete_pairs": len(differences),
            "mean_difference_complete_pairs": math.fsum(differences) / len(differences) if differences else None,
            "all_task_mean_difference": math.fsum(differences) / len(manifest.test)
            if len(differences) == len(manifest.test) else None,
            "wins": sum(value > 1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
            "losses": sum(value < -1e-12 for value in differences)}


def main(argv=None):
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--joint-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exec-endpoint", default="https://composure-presuming-comrade.ngrok-free.dev/v1")
    parser.add_argument("--exec-model", default="deepseek-v4-flash-vision")
    parser.add_argument("--judge-endpoint", default="https://hk.xty.app/v1")
    parser.add_argument("--judge-model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--judge-parallel", type=int, default=2)
    parser.add_argument("--max-inflight-requests", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=16000)
    parser.add_argument("--judge-attempts", type=int, default=1)
    parser.add_argument("--structured-output", choices=("none", "json_object", "json_schema"),
                        default="json_schema",
                        help="Constrain JIT planning responses and execution JSON; direct answers stay text.")
    parser.add_argument("--planning-string-max-length", type=int, default=2048)
    parser.add_argument("--planning-communication-max-length", type=int, default=1024)
    parser.add_argument("--planning-array-max-items", type=int, default=64)
    parser.add_argument("--arm", choices=("both", "baseline", "ours"), default="both",
                        help="Run both arms in parallel, or isolate one arm for recovery/debugging.")
    parser.add_argument("--reuse-baseline-from", help="Import sealed baseline answers and fill only missing generations")
    parser.add_argument("--reuse-evaluations-from", action="append", default=[],
                        help="Reuse complete judgments only when evaluator and answer identities match")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--closed-book", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_inflight_requests <= 16:
        parser.error("--max-inflight-requests must be between 1 and 16")
    if not 1 <= args.judge_parallel <= 16:
        parser.error("--judge-parallel must be between 1 and 16")
    if args.judge_max_tokens < 1 or not 1 <= args.judge_attempts <= 3:
        parser.error("Judge token limit must be positive and attempts must be between 1 and 3")
    if any(value < 1 for value in (args.planning_string_max_length,
                                  args.planning_communication_max_length, args.planning_array_max_items)):
        parser.error("Structured planning limits must be positive")
    if not Path(args.data).is_file() or not Path(args.joint_manifest).is_file():
        parser.error("Pinned data and joint manifest must exist")
    manifest = _manifest(args.joint_manifest)
    config = _config(args)
    args.baseline_reuse = _baseline_reuse(args, manifest, config)
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    if evidence_dir is None and not args.closed_book:
        parser.error("Shared public evidence is required; --closed-book explicitly registers a protocol deviation")
    if evidence_dir is not None and args.closed_book:
        parser.error("Choose shared evidence or closed-book, not both")
    if evidence_dir is not None and not evidence_dir.is_dir():
        parser.error("Evidence directory does not exist")
    preflight = _preflight(args, manifest, config, evidence_dir, require_credentials=not args.check_only)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "rr_split.json"
    args.split_path = str(split_path)
    write_json(split_path, manifest.model_dump(mode="json"))
    files = [Path(args.data).resolve(), Path(args.joint_manifest).resolve(), split_path]
    if evidence_dir:
        files.extend(sorted(path for path in evidence_dir.rglob("*") if path.is_file()))
    if args.reuse_baseline_from:
        source = Path(args.reuse_baseline_from).resolve()
        files.extend([source / "pilot_metadata.json", source / "generation_reports.json"])
        for outcome in args.baseline_reuse.values():
            files.extend(sorted(path for path in Path(outcome["_reuse_source_dir"]).rglob("*") if path.is_file()))
    for source in args.reuse_evaluations_from:
        files.extend(sorted((Path(source).resolve() / "test_release" / "evaluations").glob("*.json")))
    args.frozen_identity = {"code": code_fingerprint(), "runner": _runner_hash(), "config": digest(config),
                            "judge_parallel": args.judge_parallel,
                            "structured_output": args.structured_output,
                            "structured_policy": _structured_output_policy(args),
                            "knowledge_policy": _knowledge_policy(args),
                            "continuation_policy": ITERATIVE_CONTINUATION_POLICY_VERSION,
                            "files": {str(path): _sha256_file(path) for path in files}}
    metadata = {"version": "rr-two-arm-pilot-v2", "manifest": manifest.model_dump(mode="json"),
                "data": str(Path(args.data).resolve()), **preflight,
                "execution_model": args.exec_model, "execution_endpoint": args.exec_endpoint,
                "judge_model": args.judge_model, "judge_endpoint": args.judge_endpoint,
                "judge_parallel": args.judge_parallel, "process_request_cap": config.max_inflight_requests,
                "judge_attempts": args.judge_attempts,
                "structured_output": args.structured_output,
                "structured_policy": _structured_output_policy(args),
                "continuation_policy": ITERATIVE_CONTINUATION_POLICY_VERSION,
                "model_attempts": os.getenv("JIT_MAS_MODEL_ATTEMPTS", "1"),
                "consecutive_failure_limit": os.getenv("MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES", "5"),
                "tls_verify": os.getenv("JIT_MAS_TLS_VERIFY", "default"),
                "tls_endpoint": os.getenv("JIT_MAS_TLS_ENDPOINT"),
                "disable_keepalive": os.getenv("JIT_MAS_DISABLE_KEEPALIVE", "0"),
                "baseline_generation_source": str(Path(args.reuse_baseline_from).resolve()) if args.reuse_baseline_from else None,
                "reused_generation_count": len(args.baseline_reuse),
                "evaluation_cache_sources": [str(Path(source).resolve()) for source in args.reuse_evaluations_from],
                "config": config.model_dump(mode="json"), "frozen_identity": args.frozen_identity,
                "protocol_scope": "RR-only run0 projection of v5; not the full mixed-benchmark campaign",
                "parallel_arms": args.arm == "both", "selected_arm": args.arm,
                "repeats": 1, "test_feedback_updates_experience": False,
                "knowledge_policy": ("model_general_knowledge_allowed" if args.closed_book
                                      else "fixed_shared_evidence_only"),
                "status": "preflight_passed" if args.check_only else "running",
                "started_at": utc_now(), "monetary_cost": None}
    write_json(output / "pilot_metadata.json", metadata)
    if args.check_only:
        print(json.dumps({"status": "preflight_passed", "paid_requests": 0, **preflight}, ensure_ascii=True))
        return 0
    if args.arm == "both":
        with ThreadPoolExecutor(max_workers=2) as workers:
            baseline = workers.submit(_run_arm, run_baseline, args, config, manifest, evidence_dir, output / "baseline")
            ours = workers.submit(_run_arm, run_ours, args, config, manifest, evidence_dir, output / "ours")
            reports = {"baseline": baseline.result(), "ours": ours.result()}
    elif args.arm == "baseline":
        reports = {"baseline": _run_arm(run_baseline, args, config, manifest, evidence_dir, output / "baseline")}
    else:
        reports = {"ours": _run_arm(run_ours, args, config, manifest, evidence_dir, output / "ours")}
    write_json(output / "generation_reports.json", reports)
    _score_deferred(args, config, manifest, evidence_dir, output, reports)
    for arm, report in reports.items():
        report["cost"] = _cost_summary(output / arm)
        write_json(output / arm / "report.json", report)
    metadata["status"] = "completed" if all(report.get("status") == "completed" for report in reports.values()) else "incomplete"
    metadata["finished_at"] = utc_now()
    write_json(output / "pilot_metadata.json", metadata)
    write_json(output / "comparison.json", {"metadata": metadata, "reports": reports,
                                            "paired": _paired_comparison(reports, manifest)})
    summary = {"status": metadata["status"], "output": str(output),
               "mean_scope": "complete-only; fixed-task means are null until all tasks complete"}
    if "baseline" in reports:
        summary.update(baseline_mean=reports["baseline"]["results"]["mean_score"],
                       baseline_completed=reports["baseline"]["results"]["completed"])
    if "ours" in reports:
        summary.update(ours_mean=reports["ours"]["test"]["mean_score"],
                       ours_completed=reports["ours"]["test"]["completed"])
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
