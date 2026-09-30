"""Bounded, reviewed native JIT loop on four pinned ResearchRubrics tasks.

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
from jit_mas.pipeline import MASPipeline, code_fingerprint, write_json
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
    if (len(manifest.evolution) != 1 or len(manifest.validation) != 2
            or len(manifest.test) != 1 or manifest.stream):
        raise ValueError("Smoke split must contain 1 evolution, 2 validation and 1 held-out task")
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


def _fully_scored_pair(pair):
    if pair.get("error"):
        return False
    sides = [pair.get(side, {}) for side in ("baseline", "candidate")]
    evaluations = [side.get("evaluation", {}) for side in sides]
    if any(not item.get("complete") or item.get("score") is None for item in evaluations):
        return False
    for item in evaluations:
        rubrics = item.get("rubrics", [])
        if (not rubrics or len({row["rubric_id"] for row in rubrics}) != len(rubrics)
                or any(row.get("status") != "ok" or row.get("score") is None for row in rubrics)):
            return False
    return bool(sides[0].get("comparison_fingerprint")) and (
        sides[0]["comparison_fingerprint"] == sides[1].get("comparison_fingerprint")
        and evaluations[0].get("evaluator_version") == evaluations[1].get("evaluator_version")
        and {row["rubric_id"]: row["weight"] for row in evaluations[0]["rubrics"]}
        == {row["rubric_id"]: row["weight"] for row in evaluations[1]["rubrics"]})


def _validation_progress(outcomes, task_ids, repeats):
    validations = [validation for outcome in outcomes for validation in outcome["validations"]]
    pairs = [pair for validation in validations for pair in validation["pairs"]]
    expected = {(task_id, repeat) for task_id in task_ids for repeat in range(repeats)}
    complete = any(
        validation["status"] in {"accepted", "rejected"}
        and expected
        and {(pair["task_id"], pair["repeat"]) for pair in validation["pairs"]} == expected
        and all(_fully_scored_pair(pair) for pair in validation["pairs"])
        for validation in validations)
    return {"paired_validation_attempted": bool(pairs),
            "paired_validation_attempted_pairs": len(pairs),
            "paired_validation_fully_scored_pairs": sum(_fully_scored_pair(pair) for pair in pairs),
            "paired_validation_executed": bool(complete),
            "paired_validation_complete": bool(complete)}


def _confirm_closed_loop(report, snapshot):
    accepted_ids = {entry.experience_id for entry in snapshot.experiences
                    if entry.validation_status == "accepted"}
    retrieved = accepted_ids.intersection(report.get("held_out_retrieved_experience_ids", []))
    report["held_out_retrieved_accepted_experience_ids"] = sorted(retrieved)
    report["full_mechanism_exercised"] = bool(
        report["stages"] == {"evolution": "completed", "held_out": "completed"}
        and report.get("paired_validation_complete")
        and report.get("evolution") and report.get("held_out")
        and all(row["evaluation_complete"] for row in report["evolution"] + report["held_out"]))
    report["accepted_experience_reuse_demonstrated"] = bool(
        report["full_mechanism_exercised"] and report.get("new_experience_accepted") and retrieved)
    report["closed_loop_confirmed"] = report["accepted_experience_reuse_demonstrated"]


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


def _heldout_evidence_files(directory):
    root = Path(directory).resolve()
    required = [root / name for name in ("report.json", "config.json", "session_budget.json", "experience.sqlite")]
    if any(not path.is_file() for path in required):
        raise ValueError("Held-out continuation requires the complete prior report and store")
    if any(path.stat().st_size for path in root.glob("experience.sqlite-*") if path.is_file()):
        raise ValueError("Held-out continuation requires a settled SQLite store without side journals")
    files = set(required) | set(root.rglob("*.json"))
    if any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("Held-out evidence must remain inside its source directory")
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def heldout_continuation_digest(directory):
    """Caller-pinned immutable report, store, and all saved JSON evidence."""
    return digest(_heldout_evidence_files(directory))


def _load_heldout_continuation(directory, anchor, config, tasks, private, manifest):
    root = Path(directory).resolve()
    files = _heldout_evidence_files(root)
    if digest(files) != anchor:
        raise ValueError("Held-out continuation evidence hash mismatch")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    old = read(root / "report.json")
    if (old.get("status") != "failed"
            or old.get("stages") != {"evolution": "completed", "held_out": "failed"}
            or old.get("held_out") or not old.get("paired_validation_complete")
            or old.get("accepted_experience_count") != 0 or old.get("accepted_experience_version") != 0
            or old.get("dataset_sha256") != OFFICIAL_DATA_SHA256
            or old.get("dataset_revision") != DATASET_REVISION
            or old.get("manifest") != manifest.model_dump(mode="json")
            or read(root / "config.json") != config.model_dump(mode="json")):
        raise ValueError("Held-out continuation requires identical config/splits and completed rejected pairs")
    budget = read(root / "session_budget.json")
    usage = _usage_accounting(budget)
    if (any(old.get("budget", {}).get(key) != value for key, value in budget.items() if key != "wall_seconds")
            or usage["pending_model_attempts"] or usage["pending_reserved_tokens"]):
        raise ValueError("Prior session budget must be complete and settled")
    current_code = code_fingerprint()

    def run_dir(value):
        path = Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("Prior run reference escaped its evidence directory")
        return path

    def check_run(outcome, task_id, *, paired):
        path = run_dir(outcome["run_dir"])
        complete = read(path / "complete.json")
        evaluation = read(path / "evaluation.json")
        submission = read(path / "submission.json")
        execution = read(path / "execution.json")
        comparison = read(path / "run_manifest.json")["comparison"]
        official = private[task_id]["rubrics"]
        rows = evaluation["rubrics"]
        if (complete["task_id"] != task_id or evaluation["task_id"] != task_id
                or not evaluation["complete"] or complete["evaluation"] != evaluation
                or len(rows) != len(official)
                or any(any(row[key] != target[key] for key in ("rubric_id", "criterion", "weight"))
                       or row["status"] != "ok" or row["score"] is None
                       for row, target in zip(rows, official))
                or comparison["task"] != tasks[task_id].model_dump(mode="json")
                or comparison["private_hash"] != digest(private[task_id])
                or comparison["config"] != config.model_dump(mode="json")
                or comparison["manifest"] != digest(manifest)
                or submission["answer_hash"] != digest(submission["answer"])
                or submission["answer"] != execution["answer"]
                or complete["answer_hash"] != submission["answer_hash"]):
            raise ValueError("Prior scored task evidence is inconsistent")
        denominator = sum(row["weight"] for row in rows if row["weight"] > 0)
        numerator = sum(row["weight"] * row["score"] for row in rows)
        expected_score = numerator / denominator if denominator else 0.0
        if evaluation["score"] != expected_score:
            raise ValueError("Prior official score does not match its unchanged criteria")
        binding = evaluation["raw"].get("submission_answer_hash")
        if binding != submission["answer_hash"] and (paired or "submission_answer_hash" in evaluation["raw"]):
            raise ValueError("Prior evaluation submission hash mismatch")
        if paired and (complete != outcome or comparison["code"] != current_code
                       or outcome["comparison_fingerprint"] != digest(comparison)):
            raise ValueError("Paired evidence must use the identical current runtime fingerprint")
        if not paired and (outcome["score"] != evaluation["score"] or not outcome["evaluation_complete"]):
            raise ValueError("Source report differs from its immutable evaluation")

    store = ExperienceStore(root / "experience.sqlite", read_only=True)
    try:
        snapshot = store.snapshot()
        validations = [json.loads(row[0]) for row in store.db.execute("SELECT body FROM validations")]
        if (snapshot.version != 0 or snapshot.experiences or snapshot.accepted_proposals
                or store.db.execute("SELECT count(*) FROM commits").fetchone()[0]
                or len(validations) != 1 or validations[0]["status"] != "rejected"
                or validations[0]["config_hash"] != digest(config.validation)
                or validations[0]["baseline_hash"] != digest(snapshot)):
            raise ValueError("Held-out continuation only supports an unchanged empty uncommitted bank")
        progress = _validation_progress([{"validations": validations}], manifest.validation, config.validation.repeats)
        if not progress["paired_validation_complete"] or any(old.get(k) != v for k, v in progress.items()):
            raise ValueError("Prior paired validation must be complete and unchanged")
        for pair in validations[0]["pairs"]:
            for side in ("baseline", "candidate"):
                outcome = pair[side]
                if outcome["experience_hash"] != validations[0][side + "_hash"]:
                    raise ValueError("Prior pair used a different experience snapshot")
                check_run(outcome, pair["task_id"], paired=True)
        if len(old.get("evolution", [])) != 1 or old["evolution"][0]["task_id"] != manifest.evolution[0]:
            raise ValueError("Prior source report does not match the evolution split")
        expected_validations = [{"status": item["status"], "reason": item["reason"], "pairs": len(item["pairs"])}
                                for item in validations]
        if old["evolution"][0]["validations"] != expected_validations:
            raise ValueError("Prior report validation summary differs from its stored decision")
        check_run(old["evolution"][0], manifest.evolution[0], paired=False)
        failures = list((root / "held_out").glob("*/failure.json"))
        if (len(failures) != 1 or any((root / "held_out").rglob("submission.json"))
                or any((root / "held_out").rglob("evaluation.json"))
                or any((root / "held_out").rglob("complete.json"))):
            raise ValueError("Held-out continuation requires exactly one failed unsubmitted task")
        failure = read(failures[0])
        comparison = read(failures[0].with_name("run_manifest.json"))["comparison"]
        if (failure["task_id"] != manifest.test[0] or failure["experience_hash"] != digest(snapshot)
                or comparison["code"] != current_code or comparison["config"] != config.model_dump(mode="json")
                or comparison["task"] != tasks[manifest.test[0]].model_dump(mode="json")
                or comparison["private_hash"] != digest(private[manifest.test[0]])
                or comparison["manifest"] != digest(manifest)):
            raise ValueError("Failed held-out task differs from the frozen continuation inputs")
    finally:
        store.close()
    if _heldout_evidence_files(root) != files:
        raise ValueError("Prior evidence changed during continuation verification")
    return old, {"source_dir": str(root), "evidence_hash": anchor,
                 "report_sha256": files["report.json"], "file_sha256": files,
                 "runtime_fingerprint": current_code, "prior_session_usage": usage,
                 "source_regenerated": False, "attribution_regenerated": False,
                 "paired_validation_rerun": False, "held_out_previous_submission": False}


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
    if ((resume_heldout is None) != (resume_heldout_hash is None)
            or resume_heldout is not None and (resume_source is not None or resume_attribution is not None)):
        raise ValueError("--resume-heldout requires its explicit hash and cannot combine with source/attribution resume")
    if resume_heldout is not None and (max_calls > 39 or max_tokens > 521_350):
        raise ValueError("Held-out continuation is limited to 39 attempts and 521350 tokens")
    tasks, private, manifest = load_inputs(data, splits)
    output = Path(output_dir)
    if resume_heldout is not None and output.resolve().is_relative_to(Path(resume_heldout).resolve()):
        raise ValueError("Held-out continuation output must be outside the immutable prior run")
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
        execution_timeout=max(240, request_timeout * 6 + 30), max_validation_tasks=2)
    prior = None
    if resume_heldout is not None:
        prior, provenance = _load_heldout_continuation(
            resume_heldout, resume_heldout_hash, config, tasks, private, manifest)
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
        "completion_note": "status=completed means the runner finished. full_mechanism_exercised allows a legitimate validation rejection; closed_loop_confirmed additionally requires an accepted experience retrieved by the held-out planner.",
        "closed_loop_confirmed": False,
        "full_mechanism_exercised": False, "accepted_experience_reuse_demonstrated": False,
        "paired_validation_attempted": False, "paired_validation_attempted_pairs": 0,
        "paired_validation_fully_scored_pairs": 0, "paired_validation_executed": False,
        "paired_validation_complete": False,
        "new_experience_accepted": False, "cross_task_accepted_experience_available": False,
        "held_out_retrieved_experience_ids": [], "held_out_retrieved_accepted_experience_ids": [],
        "limits": {"session_model_calls": max_calls, "session_tokens": max_tokens,
                   "max_agents": config.max_agents, "team_max_calls": config.team_max_calls,
                   "tool_calls": 0, "automatic_transport_retries": 0,
                   "structured_output_corrections_per_phase": 1,
                   "max_harness_repairs_per_candidate": config.max_repairs,
                   "role_output_limits": ROLE_TOKENS},
        "stages": {"evolution": "not_started", "held_out": "not_started"}}
    if prior is not None:
        report["held_out_continuation"] = provenance
        report["evolution"] = prior["evolution"]
        report.update({key: value for key, value in prior.items() if key.startswith("paired_validation_")})
        report["stages"]["evolution"] = "completed"

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
        store_path = (Path(resume_heldout).resolve() / "experience.sqlite" if prior is not None
                      else output / "experience.sqlite")
        store = ExperienceStore(store_path, read_only=prior is not None)
        if prior is None:
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
            "source_submission_reused": r.get("source_submission_reused", False),
            "source_attribution_reused": r.get("source_attribution_reused", False),
            "resume_provenance": r.get("resume_provenance"),
            "historical_source_usage": (_usage_accounting(r["historical_source_budget"])
                                        if "historical_source_budget" in r else None),
            "historical_attribution_usage": (_usage_accounting(r["historical_attribution_budget"])
                                             if "historical_attribution_budget" in r else None),
            "validations": [{"status": v["status"], "reason": v["reason"], "pairs": len(v["pairs"])}
                            for v in r["validations"]], "run_dir": r["run_dir"]} for r in evolved]
            report["stages"]["evolution"] = "completed"
            report.update(_validation_progress(evolved, manifest.validation, config.validation.repeats))
        snapshot = store.snapshot()
        report["accepted_experience_version"] = snapshot.version
        report["accepted_experience_count"] = len(snapshot.experiences)
        report["new_experience_accepted"] = snapshot.version > 0
        report["cross_task_accepted_experience_available"] = bool(snapshot.experiences)
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
        if prior is not None and heldout_continuation_digest(resume_heldout) != resume_heldout_hash:
            raise RuntimeError("Immutable prior evidence changed during held-out continuation")
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
    parser.add_argument("--resume-heldout", help="Continue only a failed, unsubmitted held-out task; preserve complete prior pairs")
    parser.add_argument("--resume-heldout-hash", help="Explicit heldout_continuation_digest anchor for all prior saved evidence")
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
