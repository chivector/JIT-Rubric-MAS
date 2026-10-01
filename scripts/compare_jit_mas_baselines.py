"""Preregistered small direct/fixed-team-JIT/JIT-MAS comparison.

Live workers import a frozen checkout, accept both credentials through stdin,
and retain failures. Screening, exposed-task development and confirmation tasks
are separate phases. This is not a full benchmark or experience acceptance test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


VERSION = "jit-mas-method-baselines-v1"
ARMS = ("direct", "fixed_team_jit", "jit_mas")
PHASES = ("screening", "development", "holdout")
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_registration(path):
    registration = read_json(path)
    if registration.get("schema_version") != VERSION or registration.get("arms") != list(ARMS):
        raise ValueError("Unexpected comparison version or arms")
    rows = registration.get("tasks", [])
    ids = [row["public_task"]["task_id"] for row in rows]
    if len(ids) != len(set(ids)) or not 2 <= len(ids) <= 8:
        raise ValueError("Require distinct preregistered screening and holdout tasks")
    if set(ids).intersection(registration.get("excluded_task_ids", [])):
        raise ValueError("Selected tasks overlap the recorded exposure exclusions")
    if {row["split"] for row in rows} != {"screening", "holdout"}:
        raise ValueError("Both screening and holdout selections must be frozen")
    for row in rows:
        task = row["public_task"]
        if task.get("attachments") or task.get("tools") or digest(task) != row["public_task_hash"]:
            raise ValueError("Public-only closed-book task changed")
        if set(task) != {"schema_version", "task_id", "question", "attachments", "constraints", "tools", "capabilities"}:
            raise ValueError("Public task contains unexpected fields")
    settings = registration["settings"]
    if settings.get("repeats") != 1 or settings.get("tools") != []:
        raise ValueError("This pilot requires one run per task and no live task tools")
    for field, maximum in (("max_calls_per_arm", 500), ("max_tokens_per_arm", 5_000_000)):
        value = settings[field]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("Runaway-call guard exceeds this small pilot's limits")
    timeout = settings.get("timeout")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("Invalid request timeout")
    for role in ("execution", "judge"):
        identity = settings[role]
        endpoint = urlsplit(identity["endpoint"])
        if (set(identity) != {"model", "endpoint", "non_thinking"}
                or not isinstance(identity["model"], str) or not identity["model"].strip()
                or endpoint.scheme != "https" or not endpoint.hostname or endpoint.username
                or endpoint.password or endpoint.query or endpoint.fragment
                or type(identity["non_thinking"]) is not bool):
            raise ValueError("Invalid public model identity")
    if settings["execution"]["model"] == settings["judge"]["model"]:
        raise ValueError("This pilot requires a different judge model identifier")
    return registration


def load_runtime(root):
    # Import the existing isolation/binding helpers without first importing scripts.*.
    root = Path(root).resolve()
    spec = importlib.util.spec_from_file_location("_baseline_binding", root / "scripts/compare_jit_mas_quality.py")
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    runtime = binding.import_runtime(root)
    methods = importlib.import_module("scripts.mas_baseline_methods")
    return runtime, methods, binding


def phase_tasks(registration, phase):
    split = "screening" if phase == "development" else phase
    return [row for row in registration["tasks"] if row["split"] == split]


def prepare(args, runtime, binding):
    registration = load_registration(args.registration)
    if file_hash(args.data) != registration["dataset_sha256"]:
        raise ValueError("Dataset differs from preregistration")
    selected = {row["public_task"]["task_id"]: row for row in phase_tasks(registration, args.phase)}
    tasks, private = {}, {}
    split_item = importlib.import_module("benchmark.adapter.researchrubrics").split_item
    for line in Path(args.data).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        task_id = raw.get("sample_id")
        if task_id not in selected:
            continue
        public, record = split_item(raw)
        task = runtime.PublicTask(task_id=task_id, question=public["question"],
            attachments=public.get("attachments", []), constraints=public.get("explicit_constraints", []), tools=[])
        if task_id in tasks or digest(task.model_dump(mode="json")) != selected[task_id]["public_task_hash"]:
            raise ValueError("Dataset public task differs from frozen selection")
        tasks[task_id], private[task_id] = task, record
    if set(tasks) != set(selected):
        raise ValueError("Missing selected tasks")
    tasks = {key: tasks[key] for key in selected}
    runtime_binding = binding.bind_runtime(args.runtime_root, "candidate", runtime)
    if args.expected_runtime_fingerprint and args.expected_runtime_fingerprint != runtime_binding["fingerprint"]:
        raise ValueError("Runtime differs from the approved preflight")
    counts = {key: len(private[key]["rubrics"]) for key in tasks}
    if registration["settings"]["max_calls_per_arm"] < sum(counts.values()) + len(tasks):
        raise ValueError("Budget cannot cover even direct answers and their rubric calls")
    for name in ("system_prompt.txt", "user_prompt.txt"):
        relative = "benchmark/adapter/researchrubrics_prompts/" + name
        pinned = subprocess.check_output(["git", "-C", str(args.runtime_root), "show",
            binding.BASELINE + ":" + relative], text=True, encoding="utf-8")
        if (Path(args.runtime_root) / relative).read_text(encoding="utf-8") != pinned:
            raise ValueError("Official judge prompt differs from the baseline release")
    return registration, tasks, private, {
        "runtime": runtime_binding, "registration_sha256": file_hash(args.registration),
        "phase": args.phase, "rubric_counts": counts,
        "official_prompt_hashes": {name: file_hash(Path(args.runtime_root) /
            "benchmark/adapter/researchrubrics_prompts" / name)
            for name in ("system_prompt.txt", "user_prompt.txt")},
    }


def make_config(runtime, methods, registration, arm):
    settings = registration["settings"]
    configs = {}
    for role, limit in runtime.ROLE_TOKENS.items():
        identity = settings["judge" if role == "judge" else "execution"]
        configs[role] = runtime.ModelConfig(model=identity["model"], endpoint=identity["endpoint"],
            key_env="STDIN_ONLY_NOT_EXPORTED", max_tokens=limit, timeout=settings["timeout"],
            temperature=0, thinking="disabled" if identity["non_thinking"] else None,
            reasoning_effort="none" if identity["non_thinking"] else None)
    return runtime.MASConfig(backend="native_jit", unsafe_local=True, models=configs,
        max_agents=3, max_parallel=2, team_max_calls=3,
        max_model_calls=settings["max_calls_per_arm"], max_total_tokens=settings["max_tokens_per_arm"],
        max_tool_calls=0, max_repairs=2, candidates=1, execution_timeout=settings["timeout"] * 3 + 30,
        local_planning=arm == "jit_mas", explicit_rubrics=arm == "jit_mas",
        persistent_experience=False, local_attribution=False,
        fixed_team=methods.fixed_team() if arm == "fixed_team_jit" else None)


def claim_protocol(study, phase, protocol):
    path = Path(study) / phase / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=True)
    except FileExistsError:
        if read_json(path) != protocol:
            raise ValueError("Methods must share the same phase protocol")


class ModelRouter:
    def __init__(self, runtime, methods, execution, judge, session, settings):
        self.execution = runtime.LiveModels(execution, session, settings["timeout"], settings["execution"]["non_thinking"])
        self.judge = runtime.LiveModels(judge, session, settings["timeout"], settings["judge"]["non_thinking"])
        self.methods, self.judges = methods, []

    def create(self, role, agent_id, ledger, stage):
        if role == "judge":
            result = self.methods.JudgeEnvelopeModel(self.judge.create(role, agent_id, ledger, stage))
            self.judges.append(result)
            return result
        return self.execution.create(role, agent_id, ledger, stage)

    def close(self):
        try:
            self.execution.close()
        finally:
            self.judge.close()


def run_arm(args, runtime, methods, binding, prepared, credentials):
    if not args.unsafe_local or not args.expected_runtime_fingerprint:
        raise PermissionError("Live runs need unsafe-local and an approved runtime fingerprint")
    registration, tasks, private, checks = prepared
    settings = registration["settings"]
    for name in ("execution", "judge"):
        if credentials[name].model != settings[name]["model"] or credentials[name].endpoint != settings[name]["endpoint"]:
            raise ValueError("Credential endpoints/models differ from the frozen protocol")
    study = Path(args.registration).resolve().parent
    config = make_config(runtime, methods, registration, args.arm)

    def evaluator(judge):
        return runtime.ResearchRubricsAdapter(judge=judge, judge_id=settings["judge"]["model"],
            judge_api_base=settings["judge"]["endpoint"], judge_timeout=settings["timeout"],
            judge_max_tokens=runtime.ROLE_TOKENS["judge"], max_attempts=1)

    protocol = {"version": VERSION, "phase": args.phase, "settings": settings,
        "registration_sha256": checks["registration_sha256"], "dataset_sha256": registration["dataset_sha256"],
        "runtime_fingerprint": checks["runtime"]["fingerprint"], "launcher_sha256": file_hash(__file__),
        "configs": {arm: make_config(runtime, methods, registration, arm).model_dump(mode="json") for arm in ARMS},
        "task_ids": list(tasks), "evaluator_version": evaluator(None).evaluator_version,
        "official_prompt_hashes": checks["official_prompt_hashes"], "role_tokens": runtime.ROLE_TOKENS,
        "judge_envelope_policy": "strip only one complete JSON-object Markdown fence; preserve raw; no semantic changes",
        "scope": "exposed-task development" if args.phase == "development" else "preregistered small method pilot",
        "equal_compute": False, "experience_mode": "empty_read_only_evaluate"}
    claim_protocol(study, args.phase, protocol)
    output = study / args.phase / args.arm
    output.mkdir()  # Refuse replacing or retrying a consumed arm, even if it failed.
    session = runtime.SessionLedger(output / "session_budget.json", max_calls=settings["max_calls_per_arm"],
                                     max_tokens=settings["max_tokens_per_arm"])
    report = {"version": VERSION, "arm": args.arm, "phase": args.phase, "status": "running",
        "started_at": runtime.utc_now(), "protocol": protocol, "protocol_hash": digest(protocol),
        "preflight": checks, "config": config.model_dump(mode="json"), "tasks": [],
        "runtime_unchanged": False, "experience_store_unchanged": False,
        "security_note": "Exact-hash manual review before unsafe-local code; not a sandbox.",
        "method_note": "Fixed roster still uses task-conditioned native JIT harness generation."
            if args.arm == "fixed_team_jit" else args.arm}
    models = store = None
    old_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    def sanitize(value):
        for credential in credentials.values():
            value = runtime.redact(value, credential.api_key)
        return value

    def save():
        report["budget"] = session.snapshot()
        report["usage_accounting"] = runtime._usage_accounting(report["budget"])
        write_json(output / "session_budget.json", report["budget"])
        write_json(output / "report.json", sanitize(report))

    try:
        save()
        initial = runtime.ExperienceStore(output / "experience.sqlite")
        initial.close()
        store_hash = file_hash(output / "experience.sqlite")
        store = runtime.ExperienceStore(output / "experience.sqlite", read_only=True)
        if store.snapshot().experiences or store.snapshot().version:
            raise ValueError("Baseline pilot requires an empty bank")
        models = ModelRouter(runtime, methods, credentials["execution"], credentials["judge"], session, settings)
        gate = runtime.FileReviewGate(output / "reviews", timeout_seconds=900)

        def synthesizer(meta):
            return runtime.ReviewedSynthesizer(gate, backend="native_jit", meta_model=meta,
                meta_config={"model_id": settings["execution"]["model"], "api_base": settings["execution"]["endpoint"],
                             "api_key": "INJECTED"}, candidates=1, max_repairs=2)

        pipeline = runtime.MASPipeline(config, models, evaluator, synthesizer, tasks, private,
            runtime.SplitManifest(test=list(tasks)), store, output / "evaluate")
        for task_id, task in tasks.items():
            row = {"task_id": task_id, "status": "failed"}
            try:
                if binding.bind_runtime(args.runtime_root, "candidate", runtime) != checks["runtime"]:
                    raise RuntimeError("Frozen runtime changed")
                if args.arm == "direct":
                    outcome = methods.run_direct(task, private[task_id], models, evaluator,
                        output / "evaluate" / task_id,
                        {"max_calls": settings["max_calls_per_arm"], "max_tokens": settings["max_tokens_per_arm"]})
                else:
                    outcome = pipeline.run("evaluate", task_ids=[task_id], limit=1, resume=False)[0]
                row.update(outcome=outcome, status="completed" if outcome["evaluation"]["complete"] else "incomplete")
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=sanitize(str(exc)))
            report["tasks"].append(row)
            write_json(output / "judge_envelopes.json", sanitize([judge.calls for judge in models.judges]))
            save()
        report["runtime_unchanged"] = binding.bind_runtime(args.runtime_root, "candidate", runtime) == checks["runtime"]
        report["experience_store_unchanged"] = file_hash(output / "experience.sqlite") == store_hash
        if file_hash(args.registration) != checks["registration_sha256"]:
            raise RuntimeError("Registration changed during execution")
        report["status"] = "completed" if report["runtime_unchanged"] and report["experience_store_unchanged"] and all(
            row["status"] == "completed" for row in report["tasks"]) else "failed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=sanitize(str(exc)))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        try:
            for resource in (models, store):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        report.update(status="failed", cleanup_error=sanitize(str(exc)))
        finally:
            logging.disable(old_logging)
            report["finished_at"] = runtime.utc_now()
            save()
            report["evidence_sha256"] = binding.evidence_hashes(output)
            write_json(output / "report.json", sanitize(report))
    return report


def summarize(study, phase, binding):
    directory = Path(study).resolve() / phase
    registration = load_registration(Path(study) / "pre_registration.json")
    protocol = read_json(directory / "protocol.json")
    expected_ids = [row["public_task"]["task_id"] for row in phase_tasks(registration, phase)]
    if (protocol["registration_sha256"] != file_hash(Path(study) / "pre_registration.json")
            or protocol["phase"] != phase or protocol["task_ids"] != expected_ids
            or protocol["settings"] != registration["settings"]
            or protocol["dataset_sha256"] != registration["dataset_sha256"]):
        raise ValueError("Registration or phase changed")
    reports = {arm: read_json(directory / arm / "report.json") for arm in ARMS}
    for arm, report in reports.items():
        if (report["arm"] != arm or report["phase"] != phase or report["protocol"] != protocol
                or report["protocol_hash"] != digest(protocol)
                or report["preflight"]["phase"] != phase
                or report["preflight"]["registration_sha256"] != protocol["registration_sha256"]
                or report["preflight"]["runtime"]["fingerprint"] != protocol["runtime_fingerprint"]
                or report["config"] != protocol["configs"][arm]
                or report["evidence_sha256"] != binding.evidence_hashes(directory / arm)):
            raise ValueError("Method evidence or common protocol changed")
        if (report["budget"]["reserved_tokens"] or report["usage_accounting"]["pending_model_attempts"]
                or report["budget"]["model_calls"] > protocol["settings"]["max_calls_per_arm"]
                or report["budget"]["tokens"] > protocol["settings"]["max_tokens_per_arm"]
                or read_json(directory / arm / "session_budget.json") != report["budget"]):
            raise ValueError("Cannot summarize pending model calls")
        if [row["task_id"] for row in report["tasks"]] != protocol["task_ids"]:
            raise ValueError("Incomplete or reordered task inventory")
        for row in report["tasks"]:
            if "outcome" not in row:
                if row["status"] == "completed":
                    raise ValueError("Completed task has no evidence")
                continue
            outcome = row["outcome"]
            run_dir = Path(outcome["run_dir"]).resolve()
            if not run_dir.is_relative_to(directory / arm):
                raise ValueError("Task evidence escaped its method directory")
            submission, evaluation = read_json(run_dir / "submission.json"), read_json(run_dir / "evaluation.json")
            completed = read_json(run_dir / "complete.json")
            # Read-only support for old empty-bank reports, never a promotion path.
            update_key = "experience_updates" if "experience_updates" in outcome else "validations"
            expected = completed if arm == "direct" else {**completed, update_key: [], "next_experience_version": 0}
            if (outcome != expected or outcome["task_id"] != row["task_id"]
                    or read_json(run_dir / "execution.json")["answer"] != submission["answer"]
                    or submission["answer_hash"] != digest(submission["answer"])
                    or evaluation != outcome["evaluation"]
                    or (row["status"] == "completed") != bool(evaluation["complete"])
                    or evaluation["evaluator_version"] != protocol["evaluator_version"]
                    or evaluation["raw"]["submission_answer_hash"] != submission["answer_hash"]):
                raise ValueError("Scoring is not bound to its submitted answer and judge")
            if arm != "direct":
                comparison = read_json(run_dir / "run_manifest.json")["comparison"]
                if (comparison["code"] != report["preflight"]["runtime"]["native_code_fingerprint"]
                        or comparison["config"] != report["config"]):
                    raise ValueError("Task runtime or config differs from the phase protocol")
    complete = all(report["status"] == "completed" and report["runtime_unchanged"]
                   and report["experience_store_unchanged"]
                   and all(row["status"] == "completed" for row in report["tasks"])
                   for report in reports.values())
    rows = [{"task_id": task_id, "scores": {arm: next(row for row in reports[arm]["tasks"]
             if row["task_id"] == task_id).get("outcome", {}).get("evaluation", {}).get("score")
             for arm in ARMS}} for task_id in protocol["task_ids"]]
    means = {arm: sum(row["scores"][arm] for row in rows) / len(rows) if complete else None for arm in ARMS}
    return {"version": VERSION, "phase": phase, "comparison_complete": complete, "tasks": rows,
        "mean_scores": means, "jit_minus_baseline": {arm: means["jit_mas"] - means[arm] if complete else None
                                                     for arm in ARMS if arm != "jit_mas"},
        "costs": {arm: report["usage_accounting"] for arm, report in reports.items()},
        "interpretation": "Small one-run method pilot, unequal compute. No significance, SOTA or experience-learning claim."
            + (" Development tasks were already exposed." if phase == "development" else "")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--phase", choices=PHASES, default="screening")
    parser.add_argument("--runtime-root", default=str(ROOT))
    parser.add_argument("--registration")
    parser.add_argument("--data")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--expected-runtime-fingerprint")
    parser.add_argument("--unsafe-local", action="store_true")
    parser.add_argument("--summarize")
    args = parser.parse_args(argv)
    if args.summarize:
        spec = importlib.util.spec_from_file_location("_baseline_binding", ROOT / "scripts/compare_jit_mas_quality.py")
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
        print(json.dumps(summarize(args.summarize, args.phase, binding)))
        return 0
    if not args.registration or not args.data:
        parser.error("--registration and --data are required")
    runtime, methods, binding = load_runtime(args.runtime_root)
    prepared = prepare(args, runtime, binding)
    if args.preflight_only:
        print(json.dumps(prepared[-1]))
        return 0
    if not args.arm:
        parser.error("--arm is required for a live run")
    supplied = json.load(sys.stdin)
    credentials = {name: runtime.ProbeCredentials(**supplied[name]) for name in ("execution", "judge")}
    report = run_arm(args, runtime, methods, binding, prepared, credentials)
    print(json.dumps({"arm": args.arm, "phase": args.phase, "status": report["status"],
                      "usage": report["usage_accounting"]}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
