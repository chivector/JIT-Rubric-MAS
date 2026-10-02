"""Bounded, reviewed native JIT evolution and held-out ResearchRubrics smoke.

Credentials arrive through stdin JSON, never command arguments or config files.
This is a closed-book smoke experiment, not a full benchmark reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
from pathlib import Path

from scripts.probe_jit_mas_api import CredentialSafeModel, ProbeCredentials, redact
from benchmark.adapter.researchrubrics import DATASET_REVISION, ResearchRubricsAdapter
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.experience import ExperienceStore
from jit_mas.pipeline import MASPipeline, write_json
from jit_mas.review import FileReviewGate
from jit_mas.schemas import PublicTask, SplitManifest, digest, utc_now
from scripts.models.openai_server import OpenAIServerModel


OFFICIAL_DATA_SHA256 = "ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb"
ROLE_TOKENS = {"meta": 16000, "global": 16000, "local": 16000, "exec": 8192, "judge": 4096}


class SessionLedger(BudgetLedger):
    def __init__(self, path, *, max_calls, max_tokens):
        super().__init__(max_calls=max_calls, max_tokens=max_tokens, max_tool_calls=0)
        self.path = Path(path)

    def reserve(self, *args, **kwargs):
        with self._lock:
            ticket = super().reserve(*args, **kwargs)
            write_json(self.path, self.snapshot())
            return ticket

    def settle(self, *args, **kwargs):
        with self._lock:
            try:
                return super().settle(*args, **kwargs)
            finally:
                write_json(self.path, self.snapshot())


class SafeTransport(CredentialSafeModel):
    def __call__(self, messages, **kwargs):
        try:
            return super().__call__(messages, **kwargs)
        except Exception as exc:
            # Provider errors can reflect an Authorization value into diagnostics.
            message = redact(str(exc), self.secret)
            raise RuntimeError(f"{type(exc).__name__}: {message}") from None


class LiveModels:
    def __init__(self, credentials, session, timeout=120, non_thinking=False):
        self.credentials, self.session, self.timeout = credentials, session, timeout
        self.non_thinking = non_thinking
        self.clients = []
        self._lock = threading.Lock()

    def create(self, role, agent_id, ledger, stage):
        limit = ROLE_TOKENS[role]
        options = {"response_format": {"type": "json_object"}} if role in {"global", "local"} else {}
        if self.non_thinking:
            options.update(extra_body={"thinking": {"type": "disabled"}}, reasoning_effort="none")
        raw = OpenAIServerModel(model_id=self.credentials.model, api_base=self.credentials.endpoint,
            api_key=self.credentials.api_key, temperature=0, max_attempts=1,
            max_tokens=limit, timeout=self.timeout, **options)
        raw.client = raw.client.with_options(max_retries=0, timeout=self.timeout)
        with self._lock:
            self.clients.append(raw.client)
        transport = SafeTransport(raw, self.credentials.api_key)
        session_model = MeteredModel(transport, self.session, stage, agent_id, limit)
        # These are separate session/task ledgers, not two charges in one ledger.
        return MeteredModel(session_model, ledger, stage, agent_id, limit)

    def close(self):
        for client in self.clients:
            client.close()


class ReviewingExecutor:
    def __init__(self, executor, gate):
        self.executor, self.gate = executor, gate

    def execute(self, task, team, artifact, **kwargs):
        self.gate.wait(artifact)
        return self.executor.execute(task, team, artifact, **kwargs)


class ReviewedSynthesizer(JITHarnessSynthesizer):
    def __init__(self, gate, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate

    def execute_with_repair(self, executor, *args, **kwargs):
        # The base repair loop re-enters this wrapper after every source change.
        return super().execute_with_repair(ReviewingExecutor(executor, self.gate), *args, **kwargs)


def load_inputs(data, splits):
    content = Path(data).read_bytes()
    if hashlib.sha256(content).hexdigest() != OFFICIAL_DATA_SHA256:
        raise ValueError("This smoke requires the unmodified pinned official dataset")
    manifest = SplitManifest.model_validate_json(Path(splits).read_text(encoding="utf-8"))
    if len(manifest.evolution) != 1 or len(manifest.test) != 1 or manifest.stream:
        raise ValueError("Smoke split must contain 1 evolution and 1 held-out task")
    selected = set(manifest.evolution + manifest.validation + manifest.test)
    dataset = ResearchRubricsAdapter()
    rows = dataset.load_dataset(str(data))
    tasks = {row["task_id"]: PublicTask(task_id=row["task_id"], question=row["question"],
        attachments=row.get("attachments", []), constraints=row.get("explicit_constraints", []),
        tools=[]) for row in rows if row["task_id"] in selected}
    if set(tasks) != selected or any(task.attachments for task in tasks.values()):
        raise ValueError("All selected tasks must exist and be attachment-free")
    private = {task_id: dataset.private_record(task_id) for task_id in selected}
    return tasks, private, manifest


def _confirm_closed_loop(report, snapshot):
    receipts = [item for row in report.get("evolution", [])
                for item in row.get("experience_updates", [])]
    applied_ids = set(snapshot.applied_proposals)
    applied = {item["proposal_id"] for item in receipts
                   if item.get("update_rule") == "direct_after_attribution"
                   and item.get("proposal_id") in applied_ids
                   and item.get("source_task_id") in {
                       row["task_id"] for row in report.get("evolution", [])}}
    updated_ids = {item["experience_id"] for row in report.get("evolution", [])
                   for item in row.get("proposal_experiences", [])
                   if item["proposal_id"] in applied}
    current_ids = {entry.experience_id for entry in snapshot.experiences}
    retrieved = updated_ids.intersection(
        current_ids, report.get("held_out_retrieved_experience_ids", []))
    report["held_out_retrieved_updated_experience_ids"] = sorted(retrieved)
    report["experience_update_complete"] = bool(applied)
    report["full_mechanism_exercised"] = bool(
        report["stages"] == {"evolution": "completed", "held_out": "completed"}
        and report["experience_update_complete"]
        and report.get("evolution") and report.get("held_out")
        and all(row["evaluation_complete"] for row in report["evolution"] + report["held_out"]))
    report["updated_experience_reuse_demonstrated"] = bool(
        report["full_mechanism_exercised"] and retrieved)
    report["closed_loop_confirmed"] = report["updated_experience_reuse_demonstrated"]


def _usage_accounting(budget):
    records = [row for row in budget.get("records", []) if row["kind"] == "model"]
    known = [row for row in records if not row["estimated"]]
    estimated = [row for row in records if row["estimated"]]
    return {
        "model_attempts": budget["model_calls"],
        "attempts_with_complete_token_counters": len(known),
        "attempts_without_complete_token_counters": len(estimated),
        "pending_model_attempts": max(0, budget["model_calls"] - len(records)),
        "known_usage_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in known),
        "estimated_reserved_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in estimated),
        "pending_reserved_tokens": budget.get("reserved_tokens", 0),
        "physical_http_requests": None,
        "note": "Model attempts are ledger reservations and may fail before HTTP. Known tokens are counters returned by the model adapter, which may itself estimate missing provider usage; they are not verified billed usage. Estimated reserved tokens are settled conservative charges without complete counters, not confirmed consumption. Physical HTTP count is not instrumented.",
    }




def run_benchmark(credentials, data, splits, output_dir, *, unsafe_local=False,
                  max_calls=220, max_tokens=2_000_000, request_timeout=120, non_thinking=False,
                  resume_source=None, resume_source_hash=None, resume_attribution=None, resume_attribution_hash=None,
                  resume_heldout=None, resume_heldout_hash=None):
    if not unsafe_local:
        raise PermissionError("Explicit --unsafe-local approval is required; review is not a sandbox")
    if not 1 <= max_calls <= 220 or not 1 <= max_tokens <= 2_000_000:
        raise ValueError("Smoke session is limited to 220 model calls and 2000000 tokens")
    if not 0 < request_timeout <= 300:
        raise ValueError("Request timeout must be in (0, 300]")
    if (resume_source is None) != (resume_source_hash is None):
        raise ValueError("--resume-source requires an explicit --resume-source-hash")
    if ((resume_attribution is None) != (resume_attribution_hash is None)
            or resume_attribution is not None and resume_source is None):
        raise ValueError("--resume-attribution requires --resume-source and an explicit --resume-attribution-hash")
    if resume_heldout is not None or resume_heldout_hash is not None:
        raise ValueError("Held-out continuation from the historical paired-validation protocol is unsupported; "
                         "start a fresh direct-update smoke. No prior accept/hold/reject result is reused.")
    tasks, private, manifest = load_inputs(data, splits)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory and a fresh experience store")
    session = SessionLedger(output / "session_budget.json", max_calls=max_calls, max_tokens=max_tokens)
    config = MASConfig(backend="native_jit", unsafe_local=True,
        models={role: ModelConfig(model=credentials.model, endpoint=credentials.endpoint,
            key_env="STDIN_ONLY_NOT_EXPORTED", max_tokens=limit, timeout=request_timeout,
            thinking="disabled" if non_thinking else None,
            reasoning_effort="none" if non_thinking else None)
            for role, limit in ROLE_TOKENS.items()},
        max_agents=3, max_parallel=1, team_max_calls=6, max_model_calls=90,
        max_total_tokens=600_000, max_tool_calls=0, max_repairs=2, candidates=1,
        execution_timeout=max(240, request_timeout * 6 + 30))
    models = LiveModels(credentials, session, request_timeout, non_thinking)
    gate = FileReviewGate(output / "reviews", timeout_seconds=900)
    report = {"started_at": utc_now(), "status": "running", "scope": "closed-book official-task smoke",
        "benchmark": "ResearchRubrics", "dataset_revision": DATASET_REVISION,
        "dataset_sha256": OFFICIAL_DATA_SHA256, "manifest": manifest.model_dump(mode="json"),
        "model": credentials.model, "endpoint": credentials.endpoint,
        "execution_mode": "reviewed_native_jit", "unsafe_local": True,
        "non_thinking_requested": non_thinking,
        "resume_source": str(Path(resume_source).resolve()) if resume_source is not None else None,
        "resume_source_hash": resume_source_hash,
        "resume_attribution": str(Path(resume_attribution).resolve()) if resume_attribution is not None else None,
        "resume_attribution_hash": resume_attribution_hash,
        "security_note": "Manual hash-bound review is not isolation; host access is authorized.",
        "completion_note": "status=completed means the runner finished. full_mechanism_exercised requires a direct experience update and completed source/held-out evaluations; closed_loop_confirmed additionally requires that updated experience to be retrieved by the held-out planner. This is not evidence of quality improvement.",
        "closed_loop_confirmed": False,
        "update_rule": "direct_after_attribution",
        "unused_validation_task_ids": list(manifest.validation),
        "full_mechanism_exercised": False, "updated_experience_reuse_demonstrated": False,
        "experience_update_complete": False,
        "new_experience_applied": False, "cross_task_experience_available": False,
        "held_out_retrieved_experience_ids": [], "held_out_retrieved_updated_experience_ids": [],
        "limits": {"session_model_calls": max_calls, "session_tokens": max_tokens,
                   "max_agents": config.max_agents, "team_max_calls": config.team_max_calls,
                   "tool_calls": 0, "automatic_transport_retries": 0,
                   "structured_output_corrections_per_phase": 1,
                   "max_harness_repairs_per_candidate": config.max_repairs,
                   "role_output_limits": ROLE_TOKENS},
        "stages": {"evolution": "not_started", "held_out": "not_started"}}

    def save():
        report["budget"] = session.snapshot()
        report["usage_accounting"] = _usage_accounting(report["budget"])
        write_json(output / "report.json", redact(report, credentials.api_key))

    def make_pipeline(store, path):
        def evaluator(judge):
            return ResearchRubricsAdapter(judge=judge, judge_id=credentials.model,
                judge_timeout=request_timeout, judge_max_tokens=ROLE_TOKENS["judge"], max_attempts=1)

        def synthesizer(meta):
            return ReviewedSynthesizer(gate, backend="native_jit", meta_model=meta,
                meta_config={"model_id": credentials.model, "api_base": credentials.endpoint,
                             "api_key": "INJECTED"}, candidates=1, max_repairs=config.max_repairs)

        return MASPipeline(config, models, evaluator, synthesizer, tasks, private, manifest,
                           store, path)

    store = None
    old_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        write_json(output / "config.json", config)
        store_path = output / "experience.sqlite"
        store = ExperienceStore(store_path)
        report["stages"]["evolution"] = "running"
        save()
        continuation = ({"resume_source": resume_source, "resume_source_hash": resume_source_hash,
                         "resume_attribution": resume_attribution, "resume_attribution_hash": resume_attribution_hash}
                        if resume_source is not None else {})
        evolved = make_pipeline(store, output / "evolution").run("evolve", limit=1, resume=False, **continuation)
        if any(not r["evaluation"]["complete"] for r in evolved):
            raise RuntimeError("Source evaluation was incomplete; held-out execution was not started")
        report["evolution"] = [{"task_id": r["task_id"], "score": r["evaluation"]["score"],
            "evaluation_complete": r["evaluation"]["complete"], "proposals": len(r["proposals"]),
            "proposal_experiences": [{"proposal_id": p["proposal_id"],
                                      "experience_id": p["experience"]["experience_id"]} for p in r["proposals"]],
            "source_submission_reused": r.get("source_submission_reused", False),
            "source_attribution_reused": r.get("source_attribution_reused", False),
            "resume_provenance": r.get("resume_provenance"),
            "historical_source_usage": (_usage_accounting(r["historical_source_budget"])
                                        if "historical_source_budget" in r else None),
            "historical_attribution_usage": (_usage_accounting(r["historical_attribution_budget"])
                                             if "historical_attribution_budget" in r else None),
            "experience_updates": r["experience_updates"], "run_dir": r["run_dir"]} for r in evolved]
        report["stages"]["evolution"] = "completed"
        snapshot = store.snapshot()
        report["experience_version"] = snapshot.version
        report["experience_count"] = len(snapshot.experiences)
        report["new_experience_applied"] = snapshot.version > 0
        report["cross_task_experience_available"] = bool(snapshot.experiences)
        _confirm_closed_loop(report, snapshot)
        store.close()
        store = ExperienceStore(store_path, read_only=True)
        report["stages"]["held_out"] = "running"
        save()
        tested = make_pipeline(store, output / "held_out").run("evaluate", limit=1, resume=False)
        report["held_out"] = [{"task_id": r["task_id"], "score": r["evaluation"]["score"],
            "evaluation_complete": r["evaluation"]["complete"],
            "experience_version": r["experience_version"], "run_dir": r["run_dir"]} for r in tested]
        if any(not r["evaluation"]["complete"] for r in tested):
            raise RuntimeError("Held-out evaluation was incomplete")
        report["stages"]["held_out"] = "completed"
        report["status"] = "completed"
        used_ids = []
        for outcome in tested:
            calls = json.loads((Path(outcome["run_dir"]) / "planning_calls.json").read_text(encoding="utf-8"))
            for call in calls:
                if call["phase"] == "predict":
                    payload = json.loads(call["messages"][-1]["content"])
                    used_ids.extend(item["experience_id"] for item in payload["experiences"])
        report["held_out_retrieved_experience_ids"] = sorted(set(used_ids))
        _confirm_closed_loop(report, snapshot)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=redact(str(exc), credentials.api_key))
        for stage, state in report["stages"].items():
            if state == "running":
                report["stages"][stage] = "failed"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if store is not None:
            store.close()
        try:
            models.close()
        finally:
            logging.disable(old_logging)
            report["finished_at"] = utc_now()
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unsafe-local", action="store_true")
    parser.add_argument("--max-calls", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--non-thinking", action="store_true",
                        help="Request DeepSeek thinking=disabled and reasoning_effort=none")
    parser.add_argument("--resume-source", help="Reuse a verified, immutable source submission/evaluation for new attribution only")
    parser.add_argument("--resume-source-hash", help="Explicit submitted_source_digest anchor; never inferred automatically")
    parser.add_argument("--resume-attribution", help="Reuse complete frozen attribution from this source; no new attribution calls")
    parser.add_argument("--resume-attribution-hash", help="Explicit attribution_source_digest anchor; requires --resume-source")
    parser.add_argument("--resume-heldout", help="Removed historical paired-validation continuation; fails before model calls")
    parser.add_argument("--resume-heldout-hash", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        import sys
        supplied = json.load(sys.stdin)
        credentials = ProbeCredentials(supplied["endpoint"], supplied["model"], supplied["api_key"])
        report = run_benchmark(credentials, args.data, args.splits, args.output_dir,
            unsafe_local=args.unsafe_local, max_calls=args.max_calls, max_tokens=args.max_tokens,
            request_timeout=args.request_timeout, non_thinking=args.non_thinking,
            resume_source=args.resume_source, resume_source_hash=args.resume_source_hash,
            resume_attribution=args.resume_attribution, resume_attribution_hash=args.resume_attribution_hash,
            resume_heldout=args.resume_heldout, resume_heldout_hash=args.resume_heldout_hash)
        print(json.dumps({"status": report["status"], "report": str(Path(args.output_dir) / "report.json"),
                          "model_attempts": report["budget"]["model_calls"]}))
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
