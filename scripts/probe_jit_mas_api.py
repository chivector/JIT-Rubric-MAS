"""Bounded live API smoke test; generated native code is never executed.

Credentials are accepted through stdin JSON or an environment variable. The
default runs connectivity and planning only; later stages are explicit opt-ins.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from benchmark.adapter.researchrubrics import ResearchRubricsAdapter
from jit_mas.attribution import RubricAttributor
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import TeamExecutor
from jit_mas.experience import ExperienceStore
from jit_mas.pipeline import convert_feedback, write_json
from jit_mas.planning import GlobalAnalyzer, as_json
from jit_mas.schemas import PublicTask, digest, utc_now
from scripts.models.openai_server import OpenAIServerModel


STAGES = ("connectivity", "planning", "synthesis", "execution", "evaluation", "attribution")
DEPENDENCIES = {"synthesis": "planning", "execution": "planning",
                "evaluation": "execution", "attribution": "evaluation"}
DEFAULT_STAGES = ("connectivity", "planning")


@dataclass(frozen=True)
class ProbeCredentials:
    endpoint: str
    model: str
    api_key: str = field(repr=False)

    def __post_init__(self):
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("endpoint must be an HTTP(S) base URL without credentials or query")
        if not self.model.strip() or not self.api_key.strip():
            raise ValueError("model and API key must be nonempty")


def redact(value, secret):
    """Sanitize all persisted provider output, including exception messages."""
    return json.loads(json.dumps(as_json(value), ensure_ascii=True, default=str).replace(
        json.dumps(secret, ensure_ascii=True)[1:-1], "[REDACTED]"))


class CredentialSafeModel:
    """Reject reflected credentials before JIT can write generated source files."""

    def __init__(self, model, secret):
        self.model, self.secret = model, secret

    def __call__(self, messages, **kwargs):
        response = self.model(messages, **kwargs)
        for field_name in ("content", "reasoning_content"):
            content = getattr(response, field_name, None)
            if isinstance(content, str) and self.secret in content:
                raise ValueError("Provider response contains credential text; discarded before persistence")
        return response

    def __getattr__(self, name):
        return getattr(self.model, name)


def smoke_task():
    return PublicTask(
        task_id="live-api-smoke-comparison",
        question=("A small service deploys once a week. Compare rolling deployment and "
                  "blue-green deployment for a team with limited infrastructure budget. "
                  "Recommend one approach and explain when that recommendation would change."),
        constraints=["Use at most 180 words.", "Use only the given scenario and general knowledge.",
                     "Do not call tools, browse, or invent measurements."],
    )


def private_smoke_record(task_id):
    return {"task_id": task_id, "sample_id": task_id,
            "source": "custom-api-smoke; not a ResearchRubrics dataset sample",
            "rubrics": [
                {"rubric_id": "comparison", "criterion": "Compare rolling and blue-green "
                 "deployment in terms of infrastructure cost and rollback behavior.",
                 "weight": 1, "axis": "comparison"},
                {"rubric_id": "recommendation", "criterion": "Recommend an approach for a "
                 "small team with limited infrastructure budget and state a condition that "
                 "would change the recommendation.", "weight": 1, "axis": "reasoning"},
            ]}


def run_probe(credentials, output_dir, stages=DEFAULT_STAGES, *, model_factory=None,
              request_timeout: float = 45, structured_max_tokens: int = 4096):
    """Run selected stages, stop on first failure, and always save metered usage.

    model_factory is an offline-test injection point. Each invocation receives
    (ledger, stage, agent_id, max_tokens) and must return a metered JIT model.
    """
    selected = set(stages)
    if not selected or selected - set(STAGES):
        raise ValueError("Select at least one known stage")
    if not 0 < request_timeout <= 300:
        raise ValueError("request_timeout must be positive and at most 300 seconds")
    if (not isinstance(structured_max_tokens, int) or isinstance(structured_max_tokens, bool)
            or not 1 <= structured_max_tokens <= 16000):
        raise ValueError("structured_max_tokens must be an integer from 1 to 16000")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Use a new empty output directory for each probe")
    ledger = BudgetLedger(max_calls=20, max_tokens=500_000, max_tool_calls=0)
    clients = []
    report = {"started_at": utc_now(), "endpoint": credentials.endpoint,
              "model": credentials.model, "backend": "trusted_seed_live_models",
              "scope": "custom smoke, not benchmark results or performance evidence",
              "model_source": "injected_offline_test" if model_factory else "live_api",
              "native_generated_code_executed": False, "experience_committed": False,
              "limits": {"max_calls": 20, "max_tokens": 500_000, "request_timeout": request_timeout,
                         "structured_max_tokens": structured_max_tokens,
                         "model_attempts": 1, "sdk_retries": 0, "max_tool_calls": 0},
              "stages": {stage: {"status": "not_requested"} for stage in STAGES}}
    task = smoke_task()
    analyzer = planned = result = submission = feedback = None

    def save(name, value):
        write_json(output_dir / name, redact(value, credentials.api_key))

    def model(stage, aid, max_tokens=4096):
        if stage in {"planning", "attribution"}:
            max_tokens = structured_max_tokens
        if model_factory:
            return CredentialSafeModel(model_factory(ledger, stage, aid, max_tokens), credentials.api_key)
        json_options = ({"response_format": {"type": "json_object"}}
                        if stage in {"planning", "attribution"} else {})
        raw = OpenAIServerModel(model_id=credentials.model, api_base=credentials.endpoint,
                                api_key=credentials.api_key, max_attempts=1,
                                max_tokens=max_tokens, timeout=request_timeout, **json_options)
        raw.client = raw.client.with_options(max_retries=0, timeout=request_timeout)
        clients.append(raw.client)
        return CredentialSafeModel(MeteredModel(raw, ledger, stage, aid, max_tokens), credentials.api_key)

    # Provider exception bodies can echo secrets. The CLI emits only our scrubbed
    # report; suppress lower-level transport logging during this bounded probe.
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    failed = False
    try:
        save("public_task.json", task)
        for stage in STAGES:
            if stage not in selected:
                continue
            row = report["stages"][stage] = {"status": "running", "started_at": utc_now()}
            report["budget"] = ledger.snapshot()
            save("report.json", report)
            dependency = DEPENDENCIES.get(stage)
            if failed or (dependency and report["stages"][dependency]["status"] != "passed"):
                row.update(status="skipped", reason="earlier_failure" if failed else
                           "required stage not selected: " + dependency)
                continue
            started = time.monotonic()
            try:
                if stage == "connectivity":
                    response = model(stage, "probe", 256)([
                        {"role": "system", "content": 'Return only the JSON object {"ok": true}.'},
                        {"role": "user", "content": "Check JSON response compatibility."},
                    ], response_format={"type": "json_object"})
                    content = json.loads(response.content)
                    if content != {"ok": True}:
                        raise ValueError("Connectivity response did not match the requested JSON")
                    save("connectivity.json", content)
                elif stage == "planning":
                    analyzer = GlobalAnalyzer(model(stage, "global"),
                        lambda aid: model(stage, aid), max_agents=2, max_parallel=1,
                        total_max_calls=4)
                    try:
                        planned = analyzer.build(task)
                    finally:
                        save("planning_calls.json", analyzer.call_records)
                    save("frozen_plan.json", {"created_at": utc_now(),
                        "R_global": analyzer.last_prediction.graph,
                        "R_planned": planned.graph, "TeamSpec": planned.team,
                        "team_hash": digest(planned.team)})
                    row["agent_count"] = len(planned.team.agents)
                elif stage == "synthesis":
                    synthesizer = JITHarnessSynthesizer(
                        backend="native_jit", meta_model=model(stage, "meta", 16000),
                        meta_config={"api_base": credentials.endpoint,
                                     "model_id": credentials.model, "api_key": "INJECTED"},
                        candidates=1, max_repairs=1)
                    artifact = synthesizer.synthesize(task, planned.graph, planned.team)
                    save("native_harness.json", artifact.to_dict())
                    row.update(backend="native_jit", verification="static_only", executed=False)
                elif stage == "execution":
                    seed = JITHarnessSynthesizer(backend="scripted", max_repairs=0)
                    artifact = seed.synthesize(task, planned.graph, planned.team)
                    save("trusted_seed_harness.json", artifact.to_dict())
                    executor = TeamExecutor(lambda aid: model(stage, aid), ledger=ledger,
                                            timeout_seconds=max(200, request_timeout * 4 + 20),
                                            unsafe_local=False)
                    result = executor.execute(task, planned.team, artifact, rubrics=planned.graph)
                    save("execution.json", result.full_dict())
                    if result.terminated_reason != "final_answer" or result.answer is None:
                        raise RuntimeError("Execution did not submit a final answer")
                    submission = {"answer": result.answer, "answer_hash": digest(result.answer),
                                  "submitted_at": utc_now()}
                    save("submission.json", submission)
                    row.update(backend="trusted_seed_live_models", artifact_backend=artifact.backend,
                               independent_agent_count=len(result.sub_runs))
                elif stage == "evaluation":
                    frozen = json.loads((output_dir / "submission.json").read_text(encoding="utf-8"))
                    if digest(result.answer) != frozen["answer_hash"]:
                        raise ValueError("Submitted answer changed before evaluation")
                    evaluator = ResearchRubricsAdapter(judge=model(stage, "judge"),
                        judge_id=credentials.model, judge_timeout=request_timeout, max_attempts=1)
                    raw = evaluator.evaluate(str(result.answer), ground_truth=task.task_id,
                                             private_record=private_smoke_record(task.task_id))
                    feedback = convert_feedback(raw)
                    save("evaluation.json", feedback)
                    if not feedback.complete:
                        raise RuntimeError("At least one independent rubric evaluation failed")
                    row.update(evaluated_at=utc_now(), criterion_count=len(feedback.rubrics),
                               score=feedback.score, submitted_at=submission["submitted_at"])
                elif stage == "attribution":
                    attributor = RubricAttributor(model(stage, "global-post"),
                        lambda aid: model(stage, aid), max_parallel=1)
                    try:
                        findings = attributor.attribute(task, analyzer.last_prediction.graph,
                            planned.graph, planned.team, result, feedback)
                        proposals = attributor.propose(task, findings, 0, [])
                    finally:
                        save("attribution_calls.json", attributor.call_records)
                    save("attribution.json", {"findings": findings, "proposals": proposals,
                                               "alignments": attributor.last_alignments})
                    if credentials.api_key in json.dumps(as_json(proposals)):
                        raise ValueError("Provider output contains credential text; staging refused")
                    store = ExperienceStore(output_dir / "staged_experience.sqlite")
                    try:
                        for proposal in proposals:
                            store.stage(proposal)
                        row.update(staged_proposals=len(proposals),
                                   experience_version=store.snapshot().version,
                                   validation_status="not_run")
                    finally:
                        store.close()
                row["status"] = "passed"
            except (Exception, SystemExit) as exc:
                failed = True
                row.update(status="failed", error_type=type(exc).__name__,
                           error=str(exc)[:4000], http_status=getattr(exc, "status_code", None))
            finally:
                row["wall_seconds"] = time.monotonic() - started
                report["budget"] = ledger.snapshot()
                save("report.json", report)
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        logging.disable(previous_logging)
        report.update(finished_at=utc_now(), budget=ledger.snapshot())
        report["status"] = "failed" if failed else (
            "incomplete" if any(r["status"] == "skipped" for r in report["stages"].values()) else "passed")
        save("report.json", report)
    return redact(report, credentials.api_key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--key-env", help="Environment variable containing the API key; never the key itself")
    parser.add_argument("--output-dir", default="outputs/api_probe_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--request-timeout", type=float, default=45,
                        help="Per-request timeout in seconds (greater than 0, at most 300)")
    parser.add_argument("--structured-max-tokens", type=int, default=4096,
                        help="Planning and attribution output-token limit (1 to 16000)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES))
    args = parser.parse_args(argv)
    try:
        if args.key_env:
            credentials = ProbeCredentials(args.endpoint or "", args.model or "", os.getenv(args.key_env, ""))
        else:
            supplied = json.load(sys.stdin)
            credentials = ProbeCredentials(supplied["endpoint"], supplied["model"], supplied["api_key"])
        report = run_probe(credentials, args.output_dir, args.stages,
                           request_timeout=args.request_timeout,
                           structured_max_tokens=args.structured_max_tokens)
    except (Exception, SystemExit) as exc:
        # Input validation failures must not print supplied values or traceback.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": report["status"], "report": str(Path(args.output_dir) / "report.json"),
                      "model_calls": report["budget"]["model_calls"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
