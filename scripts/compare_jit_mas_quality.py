"""Two frozen tasks, one reviewed native-JIT A/B arm per isolated process.

Use --preflight-only without credentials before authorizing live calls. Live
credentials are stdin JSON only. --summarize reads existing artifacts only.
This compares frozen implementations independently of experience updates.
--development-regression explicitly runs only the candidate on already exposed
tasks in a new, parent-anchored study; this is not a blind or A/B comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys


VERSION = "jit-mas-quality-comparison-v1"
BASELINE = "a1ee6cd"
DATA_SHA256 = "ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb"
TASK_HASHES = {
    "6847465956a0f6376a605414": "9be0900e76a4a4ab072727de2399ad2b9249cf2169ba1ab0829dab2b1cfb20e2",
    "6847465956a0f6376a605434": "cc5f05b42b7e51702cb2ade2adadc56784b889b8a8c81dd479743669497505ec",
}
ROLE_TOKENS = {"meta": 16000, "global": 16000, "local": 16000, "exec": 8192, "judge": 4096}
COMPARISON_STUDY = "implementation_comparison"
DEVELOPMENT_STUDY = "development_regression"
DEVELOPMENT_SCOPE = ("candidate-only development regression on already exposed tasks; "
                     "not blind testing, not an A/B comparison, not a SOTA claim or full benchmark")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluation_config(config):
    """Only inactive legacy update-gate settings differ between frozen A/B runtimes."""
    return {key: value for key, value in config.items() if key not in {"validation", "max_validation_tasks"}}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def study_metadata(registration, *, development_regression=False, arm=None):
    kind = registration.get("study_kind", COMPARISON_STUDY)
    if development_regression:
        if kind != DEVELOPMENT_STUDY or arm != "candidate":
            raise ValueError("Development regression requires matching registration and candidate arm only")
        parent = registration.get("parent_study")
        if (not isinstance(parent, dict) or set(parent) != {"study_id", "summary_sha256"}
                or not isinstance(parent["study_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", parent["study_id"])
                or not isinstance(parent["summary_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", parent["summary_sha256"])):
            raise ValueError("Development registration needs parent_study study_id and summary_sha256")
        return {"study_kind": kind, "scope": DEVELOPMENT_SCOPE, "parent_study": dict(parent),
                "parent_anchor_verification": "structure only; parent files are not read"}
    if kind != COMPARISON_STUDY or "parent_study" in registration:
        raise ValueError("Development registration cannot be used as an implementation comparison")
    return {"study_kind": kind,
            "scope": "two-task implementation A/B; not experience evolution or a full benchmark"}


def load_registration(path, *, development_regression=False, arm=None):
    registration = read_json(path)
    study_metadata(registration, development_regression=development_regression, arm=arm)
    rows = registration.get("tasks", [])
    if (registration.get("schema_version") != VERSION or registration.get("baseline_commit") != BASELINE
            or registration.get("dataset_sha256") != DATA_SHA256
            or [row.get("task_id") for row in rows] != list(TASK_HASHES)):
        raise ValueError("Comparison requires the two frozen public-only selections")
    for row in rows:
        public = {key: row[key] for key in ("task_id", "question", "attachments")}
        if row["attachments"] or digest(public) != TASK_HASHES[row["task_id"]] or row["public_task_hash"] != digest(public):
            raise ValueError("Preregistered public task changed")
    return registration


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def bind_runtime(root, arm, runtime):
    root = Path(root).resolve()
    head = git(root, "rev-parse", "HEAD")
    baseline = git(root, "rev-parse", BASELINE + "^{commit}")
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if arm == "baseline" and (head != baseline or status):
        raise ValueError("Baseline requires the clean a1ee6cd checkout")
    files = git(root, "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    source = {name: file_hash(root / name) for name in sorted(set(files))
              if Path(name).suffix in {".py", ".yaml", ".yml", ".txt", ".md", ".toml", ".json"}
              and (root / name).is_file()}
    binding = {"commit": head, "baseline_commit": baseline, "dirty": bool(status),
               "source_sha256": source, "native_code_fingerprint": runtime.code_fingerprint()}
    return {**binding, "fingerprint": digest(binding)}


def import_runtime(root):
    """Worker only: no repository from the launcher's process is a fallback."""
    root = Path(root).resolve()
    launcher_root = Path(__file__).resolve().parents[1]
    if any(name == "jit_mas" or name.startswith(("jit_mas.", "scripts.", "benchmark.")) for name in sys.modules):
        raise RuntimeError("Runtime imports require a fresh worker process")
    sys.path[:] = [str(root)] + [entry for entry in sys.path if entry and
        Path(entry).resolve() not in {root, launcher_root, launcher_root / "scripts"}]
    os.chdir(root)
    runtime = importlib.import_module("scripts.benchmark_jit_mas_live")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"scripts", "jit_mas", "jit", "benchmark", "harness_factory"}:
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise RuntimeError("A runtime module escaped the selected checkout")
    if runtime.ROLE_TOKENS != ROLE_TOKENS:
        raise ValueError("Both arms require identical role output limits")
    return runtime


def load_tasks(runtime, registration, data):
    if file_hash(data) != DATA_SHA256:
        raise ValueError("Pinned official dataset bytes changed")
    tasks, private = {}, {}
    split = importlib.import_module("benchmark.adapter.researchrubrics").split_item
    for line in Path(data).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        task_id = raw.get("sample_id")
        if task_id not in TASK_HASHES:
            continue
        public = {"task_id": task_id, "question": raw["prompt"], "attachments": raw.get("attachments", [])}
        if task_id in tasks or digest(public) != TASK_HASHES[task_id]:
            raise ValueError("Dataset public task differs from preregistration")
        # Selection is already frozen; private records are coordinator-owned.
        public_row, private[task_id] = split(raw)
        tasks[task_id] = runtime.PublicTask(**public, constraints=public_row.get("explicit_constraints", []), tools=[])
    if set(tasks) != set(TASK_HASHES):
        raise ValueError("Frozen tasks are missing from the dataset")
    manifest = runtime.SplitManifest(test=list(TASK_HASHES))
    counts = {task_id: len(private[task_id]["rubrics"]) for task_id in TASK_HASHES}
    return tasks, private, manifest, counts


def preflight(args, runtime):
    if not 1 <= args.max_calls <= 120 or not 1 <= args.max_tokens <= 1_000_000:
        raise ValueError("Each arm is capped at 120 calls and 1000000 tokens")
    if not 0 < args.request_timeout <= 300:
        raise ValueError("Request timeout must be in (0, 300]")
    registration = load_registration(args.registration,
        development_regression=getattr(args, "development_regression", False), arm=args.arm)
    tasks, private, manifest, counts = load_tasks(runtime, registration, args.data)
    binding = bind_runtime(args.runtime_root, args.arm, runtime)
    if args.expected_runtime_fingerprint and binding["fingerprint"] != args.expected_runtime_fingerprint:
        raise ValueError("Runtime differs from the explicitly approved fingerprint")
    # At least predict, one local plan, reconcile, JIT generation and execution
    # per task. Extra roles, retries, repairs and judge input tokens cost more.
    judge_calls = sum(counts.values())
    minimum_calls = judge_calls + 5 * len(tasks)
    if args.max_calls < minimum_calls:
        raise ValueError(f"Insufficient call budget: theoretical floor is {minimum_calls}; tasks cannot be replaced")
    prompts = {}
    for name in ("system_prompt.txt", "user_prompt.txt"):
        relative = "benchmark/adapter/researchrubrics_prompts/" + name
        text = (Path(args.runtime_root) / relative).read_text(encoding="utf-8")
        pinned = subprocess.check_output(["git", "-C", str(args.runtime_root), "show", BASELINE + ":" + relative],
                                         text=True, encoding="utf-8")
        if text != pinned:
            raise ValueError("Official judge prompt differs from the baseline release")
        prompts[name] = digest(text)
    return registration, tasks, private, manifest, {
        **study_metadata(registration, development_regression=getattr(args, "development_regression", False), arm=args.arm),
        "runtime": binding, "registration_sha256": file_hash(args.registration),
        "official_prompt_hashes": prompts, "rubric_counts": counts,
        "minimum_judge_calls": judge_calls,
        "remaining_nonjudge_calls": args.max_calls - judge_calls,
        "theoretical_minimum_calls": minimum_calls,
        "budget_note": (f"Floor only, not a completion guarantee. {judge_calls} judge calls leave at most "
            f"{args.max_calls - judge_calls} nonjudge calls for {len(tasks)} tasks under the {args.max_calls}-call "
            f"limit. The theoretical floor is {minimum_calls} calls including one local planning call per task. "
            f"The {args.max_tokens}-token limit includes input reservations. No automatic budget expansion."),
    }


def claim_protocol(directory, protocol):
    development = protocol.get("study_kind") == DEVELOPMENT_STUDY
    name = "development_protocol.json" if development else "comparison_protocol.json"
    incompatible = "comparison_protocol.json" if development else "development_protocol.json"
    if (directory / incompatible).exists() or development and (directory / "baseline").exists():
        raise ValueError("Development and comparison studies cannot share protocol or arm directories")
    path = directory / name
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, sort_keys=True, ensure_ascii=True, indent=2)
    except FileExistsError:
        if read_json(path) != protocol:
            raise ValueError("Both arms must use identical registration, model, config and official judge")


def evidence_hashes(directory):
    root = Path(directory).resolve()
    paths = list(root.rglob("*.json")) + list(root.glob("experience.sqlite"))
    result = {}
    for path in sorted(paths):
        if path.name == "report.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Evidence escaped its arm directory")
        result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def run_arm(args, runtime, credentials, prepared):
    if not args.unsafe_local or not args.expected_runtime_fingerprint:
        raise PermissionError("Live execution needs --unsafe-local and a preflight-approved --expected-runtime-fingerprint")
    registration, tasks, private, manifest, checks = prepared
    study_info = study_metadata(registration,
        development_regression=getattr(args, "development_regression", False), arm=args.arm)
    study = Path(args.output_dir).resolve()
    if study != Path(args.registration).resolve().parent:
        raise ValueError("Arm outputs must share the preregistration directory; no rerun directories")
    config = runtime.MASConfig(backend="native_jit", unsafe_local=True,
        models={role: runtime.ModelConfig(model=credentials.model, endpoint=credentials.endpoint,
            key_env="STDIN_ONLY_NOT_EXPORTED", max_tokens=limit, timeout=args.request_timeout,
            thinking="disabled" if args.non_thinking else None,
            reasoning_effort="none" if args.non_thinking else None) for role, limit in ROLE_TOKENS.items()},
        max_agents=3, max_parallel=1, team_max_calls=6, max_model_calls=args.max_calls,
        max_total_tokens=args.max_tokens, max_tool_calls=0, max_repairs=2, candidates=1,
        execution_timeout=max(240, args.request_timeout * 6 + 30))

    def evaluator(judge):
        return runtime.ResearchRubricsAdapter(judge=judge, judge_id=credentials.model,
            judge_timeout=args.request_timeout, judge_max_tokens=ROLE_TOKENS["judge"], max_attempts=1)

    protocol = {"version": VERSION, **study_info, "registration_sha256": checks["registration_sha256"],
        "dataset_sha256": DATA_SHA256, "public_tasks": TASK_HASHES,
        "private_task_hashes": {key: runtime.digest(value) for key, value in private.items()},
        "config": evaluation_config(config.model_dump(mode="json")), "evaluator_version": evaluator(None).evaluator_version,
        "official_prompt_hashes": checks["official_prompt_hashes"], "role_output_limits": ROLE_TOKENS,
        "mode": "evaluate", "repeats": 1, "tools": [], "quality_audit_mode": "off",
        "legacy_runtime_compatibility": "Read-only old snapshots/outcomes are supported. Inactive validation and max_validation_tasks settings are omitted from execution-config comparison; runtime fingerprints remain distinct, and historical evidence is not rewritten.",
        "max_calls_per_arm": args.max_calls, "max_tokens_per_arm": args.max_tokens,
        "non_thinking": args.non_thinking, "launcher_sha256": file_hash(__file__)}
    claim_protocol(study, protocol)
    output = study / args.arm
    output.mkdir()  # A consumed arm cannot be retried, even after failure.
    session = runtime.SessionLedger(output / "session_budget.json", max_calls=args.max_calls, max_tokens=args.max_tokens)
    report = {"version": VERSION, "arm": args.arm, "status": "running", "started_at": runtime.utc_now(),
        **study_info,
        "experience_updates_enabled": False, "protocol": protocol, "protocol_hash": digest(protocol),
        "preflight": checks, "runtime_root": str(Path(args.runtime_root).resolve()), "tasks": [],
        "experience_store_unchanged": False, "runtime_unchanged": False,
        "security_note": "Explicit unsafe-local execution; hash-bound manual review is not a sandbox."}
    models = store = None
    old_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    def save():
        report["budget"] = session.snapshot()
        report["usage_accounting"] = runtime._usage_accounting(report["budget"])
        write_json(output / "session_budget.json", report["budget"])
        write_json(output / "report.json", runtime.redact(report, credentials.api_key))

    try:
        write_json(output / "config.json", protocol["config"])
        save()
        store_path = output / "experience.sqlite"
        fresh = runtime.ExperienceStore(store_path)
        fresh.close()
        store_hash = file_hash(store_path)
        store = runtime.ExperienceStore(store_path, read_only=True)
        snapshot = store.snapshot()
        # A frozen historical baseline may still name its empty bookkeeping field this way.
        applied = getattr(snapshot, "applied_proposals", getattr(snapshot, "accepted_proposals", []))
        if snapshot.version != 0 or snapshot.experiences or applied:
            raise ValueError("Quality comparison requires an empty experience bank")
        models = runtime.LiveModels(credentials, session, args.request_timeout, args.non_thinking)
        gate = runtime.FileReviewGate(output / "reviews", timeout_seconds=900)

        def synthesizer(meta):
            return runtime.ReviewedSynthesizer(gate, backend="native_jit", meta_model=meta,
                meta_config={"model_id": credentials.model, "api_base": credentials.endpoint, "api_key": "INJECTED"},
                candidates=1, max_repairs=config.max_repairs)

        pipeline = runtime.MASPipeline(config, models, evaluator, synthesizer, tasks, private,
                                       manifest, store, output / "evaluate")
        for task_id in TASK_HASHES:
            entry = {"task_id": task_id, "status": "failed"}
            try:
                if bind_runtime(args.runtime_root, args.arm, runtime) != checks["runtime"]:
                    raise RuntimeError("Runtime changed after preflight")
                outcomes = pipeline.run("evaluate", task_ids=[task_id], limit=1, resume=False)
                if len(outcomes) != 1 or outcomes[0]["task_id"] != task_id:
                    raise RuntimeError("Unexpected task or count from evaluate")
                outcome = outcomes[0]
                entry.update(outcome=outcome, status="completed" if outcome["evaluation"]["complete"] else "incomplete")
            except Exception as exc:
                entry.update(error_type=type(exc).__name__, error=runtime.redact(str(exc), credentials.api_key))
            report["tasks"].append(entry)
            save()
        report["experience_store_unchanged"] = file_hash(store_path) == store_hash and store.snapshot() == snapshot
        report["runtime_unchanged"] = (bind_runtime(args.runtime_root, args.arm, runtime) == checks["runtime"]
            and file_hash(__file__) == protocol["launcher_sha256"]
            and file_hash(args.registration) == checks["registration_sha256"])
        report["status"] = ("completed" if report["experience_store_unchanged"] and report["runtime_unchanged"]
            and len(report["tasks"]) == 2 and all(row["status"] == "completed" for row in report["tasks"]) else "failed")
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=runtime.redact(str(exc), credentials.api_key))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        try:
            if store is not None:
                store.close()
            if models is not None:
                models.close()
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=runtime.redact(str(exc), credentials.api_key))
        finally:
            logging.disable(old_logging)
            report["finished_at"] = runtime.utc_now()
            save()
            report["evidence_sha256"] = evidence_hashes(output)
            write_json(output / "report.json", runtime.redact(report, credentials.api_key))
    return report


def summarize(study):
    study = Path(study).resolve()
    if (study / "development_protocol.json").exists():
        raise ValueError("Development regression cannot be summarized as an A/B comparison")
    load_registration(study / "pre_registration.json")
    protocol = read_json(study / "comparison_protocol.json")
    if protocol.get("study_kind", COMPARISON_STUDY) != COMPARISON_STUDY:
        raise ValueError("Development protocol cannot be summarized as an A/B comparison")
    if protocol["registration_sha256"] != file_hash(study / "pre_registration.json"):
        raise ValueError("Registration changed after execution")
    reports = {arm: read_json(study / arm / "report.json") for arm in ("baseline", "candidate")}
    for arm, report in reports.items():
        if (report.get("study_kind", COMPARISON_STUDY) != COMPARISON_STUDY
                or report["arm"] != arm or report["protocol"] != protocol or report["protocol_hash"] != digest(protocol)
                or report["evidence_sha256"] != evidence_hashes(study / arm)):
            raise ValueError("Arm evidence or comparison protocol changed")
        if (report["budget"]["model_calls"] > protocol["max_calls_per_arm"]
                or report["budget"]["tokens"] > protocol["max_tokens_per_arm"]
                or report["budget"]["reserved_tokens"]):
            raise ValueError("Budget exceeded or pending: no normal score comparison")
        seen = [row["task_id"] for row in report["tasks"]]
        if len(set(seen)) != len(seen) or not set(seen) <= set(TASK_HASHES):
            raise ValueError("Unexpected or duplicated comparison tasks")
        for row in report["tasks"]:
            if "outcome" not in row:
                continue
            outcome = row["outcome"]
            path = Path(outcome["run_dir"]).resolve()
            if not path.is_relative_to(study / arm):
                raise ValueError("Task evidence escaped its arm")
            comparison = read_json(path / "run_manifest.json")["comparison"]
            evaluation = read_json(path / "evaluation.json")
            submission = read_json(path / "submission.json")
            completed = read_json(path / "complete.json")
            # MASPipeline.run adds these two fields after writing complete.json.
            # Read-only compatibility with the pinned pre-removal baseline artifacts.
            update_field = "experience_updates" if "experience_updates" in outcome else "validations"
            expected = {**completed, update_field: [], "next_experience_version": 0}
            if (expected != outcome or outcome["evaluation"] != evaluation
                    or comparison["code"] != report["preflight"]["runtime"]["native_code_fingerprint"]
                    or evaluation_config(comparison["config"]) != evaluation_config(protocol["config"])
                    or evaluation["evaluator_version"] != protocol["evaluator_version"]
                    or evaluation["raw"].get("submission_answer_hash") != submission["answer_hash"]
                    or read_json(path / "execution.json")["answer"] != submission["answer"]):
                raise ValueError("Task evidence is not bound to its runtime, judge or submission")
    rows = []
    for task_id in TASK_HASHES:
        sides = {arm: next((item for item in report["tasks"] if item["task_id"] == task_id), {})
                 for arm, report in reports.items()}
        scored = all(side.get("status") == "completed" and reports[arm]["status"] == "completed"
                     and reports[arm]["runtime_unchanged"] and reports[arm]["experience_store_unchanged"]
                     for arm, side in sides.items())
        scores = {arm: side.get("outcome", {}).get("evaluation", {}).get("score") for arm, side in sides.items()}
        rows.append({"task_id": task_id, "status": "scored" if scored else "incomplete", "scores": scores,
                     "candidate_minus_baseline": scores["candidate"] - scores["baseline"] if scored else None})
    complete = all(report["status"] == "completed" for report in reports.values()) and all(row["status"] == "scored" for row in rows)
    return {"version": VERSION, "study_kind": COMPARISON_STUDY, "comparison_complete": complete, "tasks": rows,
        "mean_score_delta": sum(row["candidate_minus_baseline"] for row in rows) / 2 if complete else None,
        "combined_model_calls": sum(report["budget"]["model_calls"] for report in reports.values()),
        "combined_tokens": sum(report["budget"]["tokens"] for report in reports.values()),
        "runtime_fingerprints": {arm: report["preflight"]["runtime"]["fingerprint"] for arm, report in reports.items()},
        "experience_updates_enabled": False,
        "interpretation": "Two preregistered tasks, one run per implementation. No significance or general-quality claim; same-model judges may be inconsistent. Failures are retained, not screened out."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--runtime-root")
    parser.add_argument("--registration")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-runtime-fingerprint")
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--non-thinking", action="store_true")
    parser.add_argument("--unsafe-local", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--development-regression", action="store_true",
        help="Candidate-only development regression on already exposed tasks; not a blind or A/B comparison")
    parser.add_argument("--summarize", metavar="STUDY_DIRECTORY")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            if args.development_regression:
                raise ValueError("Development regression has no A/B summary mode")
            print(json.dumps(summarize(args.summarize), ensure_ascii=True))
            return 0
        if any(getattr(args, key) is None for key in ("arm", "runtime_root", "registration", "data", "output_dir")):
            raise ValueError("Specify arm, runtime-root, registration, data and output-dir")
        for key in ("runtime_root", "registration", "data", "output_dir"):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
        if not args.worker:
            child_args = ["--worker", "--arm", args.arm]
            for key in ("runtime_root", "registration", "data", "output_dir", "max_calls", "max_tokens", "request_timeout"):
                child_args += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            for key in ("non_thinking", "unsafe_local", "preflight_only", "development_regression"):
                if getattr(args, key):
                    child_args.append("--" + key.replace("_", "-"))
            if args.expected_runtime_fingerprint:
                child_args += ["--expected-runtime-fingerprint", args.expected_runtime_fingerprint]
            supplied = "" if args.preflight_only else sys.stdin.read()
            result = subprocess.run([sys.executable, "-I", "-u", str(Path(__file__).resolve()), *child_args],
                input=supplied, text=True, encoding="utf-8", capture_output=True, cwd=args.runtime_root)
            # Never relay arbitrary child/provider stdout or stderr.
            safe = json.loads(result.stdout.splitlines()[-1])
            if set(safe) - {"status", "error_type", "preflight", "report", "model_attempts"}:
                raise RuntimeError("Unexpected worker output")
            print(json.dumps(safe, ensure_ascii=True))
            return result.returncode
        runtime = import_runtime(args.runtime_root)
        prepared = preflight(args, runtime)
        if args.preflight_only:
            print(json.dumps({"status": "preflight_only", "preflight": prepared[-1]}))
            return 0
        supplied = json.load(sys.stdin)
        credentials = runtime.ProbeCredentials(supplied["endpoint"], supplied["model"], supplied["api_key"])
        report = run_arm(args, runtime, credentials, prepared)
        print(json.dumps({"status": report["status"], "report": str(Path(args.output_dir) / args.arm / "report.json"),
                          "model_attempts": report["budget"]["model_calls"]}))
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
