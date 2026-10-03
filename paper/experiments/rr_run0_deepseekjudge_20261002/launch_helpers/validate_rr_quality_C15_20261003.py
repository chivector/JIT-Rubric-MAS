"""One fixed ten-VAL engineering comparison using the immutable v35 C15 state."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime/rr_quality_C15_20261003"
OUTPUT = ROOT / "outputs/rr_quality_C15_20261003"
SOURCE = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v35"
SOURCE_RUNTIME = ROOT / ".runtime/rr_deepseek_judge_20261002_v35"
JOURNAL = SOURCE / "ours/checkpoints/checkpoint_journal.json"
SOURCE_STORE = SOURCE / "ours/experience.sqlite"
DATA = ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl"
JOINT = ROOT / "paper/experiments/joint_task_splits_v5.json"
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"
VERSION = 13


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(directory, entries):
    changed = [relative for relative, expected in entries.items()
               if not (directory / relative).is_file()
               or file_hash(directory / relative) != expected]
    if changed:
        raise RuntimeError("Frozen source files changed or disappeared: " + repr(changed))


def c15_anchor(journal):
    matches = [row for row in journal["checkpoints"] if row["position"] == 15]
    if len(matches) != 1 or matches[0]["state_version"] != VERSION:
        raise RuntimeError("A unique completed C15 with historical state version 13 is required")
    checkpoint = matches[0]
    cache = journal["validation_cache"][checkpoint["cache_key"]]
    return {"checkpoint": checkpoint, "cache": cache,
            "identity": journal["identity"]}


def main():
    sys.dont_write_bytecode = True
    if RUNTIME.exists() or OUTPUT.exists():
        raise SystemExit("Refusing to replace an existing C15 diagnostic; interrupted slots are not replayed")
    if not os.environ.get("RR_EXEC_API_KEY"):
        raise SystemExit("Inject RR_EXEC_API_KEY through the process environment")
    initial_journal = read_json(JOURNAL)
    source_anchor = c15_anchor(initial_journal)
    source_manifest_path = SOURCE_RUNTIME / "source_manifest.json"
    source_entries = read_json(source_manifest_path)
    verify_manifest(SOURCE_RUNTIME, source_entries)
    immutable_files = {str(path): file_hash(path) for path in
                       (DATA, JOINT, source_manifest_path, SOURCE / "rr_split.json")}
    source_identity = source_anchor["identity"]["execution"]
    for filename, expected in source_identity["frozen_identity"]["files"].items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise RuntimeError("Original input identity mismatch: " + filename)
    for name in ("benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"):
        if not (ROOT / name).is_dir():
            raise FileNotFoundError(ROOT / name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    RUNTIME.mkdir(parents=True)
    for name in ("benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"):
        shutil.copytree(ROOT / name, RUNTIME / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "workspaces"))
    code_entries = {path.relative_to(RUNTIME).as_posix(): file_hash(path)
                    for path in RUNTIME.rglob("*") if path.is_file()}
    (RUNTIME / "source_manifest.json").write_text(
        json.dumps(code_entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chdir(RUNTIME)
    sys.path.insert(0, str(RUNTIME))
    from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
    from jit_mas.experience import ExperienceStore
    from jit_mas.independent_protocol import normalize_score, validation_summary
    from jit_mas.schemas import SplitManifest, digest
    from scripts.run_rr_two_arm_pilot import (
        _config, _configure_logging, _manifest, _pipeline, _safe_error, write_json,
    )

    os.environ.update(RR_JUDGE_API_KEY=os.environ["RR_EXEC_API_KEY"],
        JIT_MAS_MODEL_ATTEMPTS="5", MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES="5",
        MODULAR_AGENT_API_FAILURE_ACTION="raise", JIT_MAS_DISABLE_KEEPALIVE="1",
        JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
        PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    _configure_logging()
    original_manifest = _manifest(JOINT)
    task_ids = list(original_manifest.validation)
    if len(task_ids) != 10 or len(set(task_ids)) != 10:
        raise RuntimeError("The original v5 validation set must contain exactly ten unique tasks")
    if source_anchor["identity"]["validation_ids"] != task_ids:
        raise RuntimeError("Original checkpoint VAL membership/order differs from fixed v5")
    original_slots = source_anchor["cache"]["slots"]
    if len(original_slots) != 10 or [row["task_id"] for row in original_slots] != task_ids:
        raise RuntimeError("Original C15 must contain all ten fixed VAL slots in order")
    state_hash = source_anchor["checkpoint"]["state_hash"]
    if digest(source_anchor["cache"]["snapshot"]) != state_hash:
        raise CheckpointIntegrityError("Original C15 cache snapshot hash mismatch")
    history, bounds, artifact_hashes = [], {}, {}
    for row in original_slots:
        task_id = row["task_id"]
        lower = source_anchor["identity"]["lower_bounds"][task_id]
        upper = source_identity["validation_upper_bounds"][task_id]
        bounds[task_id] = [lower, upper]
        if row["repeat"] != 0 or row["theoretical_bounds"] != bounds[task_id]:
            raise CheckpointIntegrityError("Original VAL repeat/bounds mismatch: " + task_id)
        complete = row.get("complete_evaluation") is True
        score = row.get("official_score") if complete else None
        utility = normalize_score(score, bounds[task_id]) if complete else 0.0
        if not math.isclose(utility, row["selection_utility"], abs_tol=1e-12):
            raise CheckpointIntegrityError("Original VAL utility mismatch: " + task_id)
        outcome = row.get("outcome", {})
        if complete:
            evaluation = outcome["evaluation"]
            if evaluation.get("complete") is not True or evaluation.get("score") != score:
                raise CheckpointIntegrityError("Original complete score is not bound to evaluation: " + task_id)
            directory = Path(outcome["run_dir"])
            submission = read_json(directory / "submission.json")
            if digest(submission["answer"]) != submission["answer_hash"] or submission["answer_hash"] != outcome["answer_hash"]:
                raise CheckpointIntegrityError("Original submitted answer hash mismatch: " + task_id)
            for filename in ("run_manifest.json", "submission.json", "evaluation.json", "complete.json", "budget.json"):
                path = directory / filename
                artifact_hashes[str(path)] = file_hash(path)
        history.append({"task_id": task_id, "status": row["status"], "complete": complete,
            "score": score, "official_score": score, "theoretical_bounds": bounds[task_id],
            "selection_utility": utility, "answer_hash": outcome.get("answer_hash"),
            "run_dir": outcome.get("run_dir"),
            "evaluator_version": outcome.get("evaluation", {}).get("evaluator_version")})
    historical_summary = validation_summary(history, task_ids, bounds)
    if not math.isclose(historical_summary["selection_utility"], source_anchor["checkpoint"]["selection_utility"], abs_tol=1e-12):
        raise CheckpointIntegrityError("Original C15 aggregate utility mismatch")
    source_store = ExperienceStore(SOURCE_STORE, read_only=True)
    frozen = None
    rows = []
    integrity_ok = False
    status = "registered"
    try:
        snapshot = source_store.snapshot(VERSION)
        if digest(snapshot) != state_hash:
            raise CheckpointIntegrityError("Read-only source snapshot(13) differs from original C15")
        OUTPUT.mkdir(parents=True)
        frozen_path = OUTPUT / "states" / (state_hash + ".sqlite")
        frozen = snapshot_store(snapshot, frozen_path)
        frozen_file_hash = file_hash(frozen_path)
        split_path = OUTPUT / "validation_split.json"
        manifest = SplitManifest(seed=original_manifest.seed, validation=task_ids)
        write_json(split_path, manifest)
        args = SimpleNamespace(data=DATA, split_path=split_path, exec_model=MODEL,
            exec_max_tokens=12288, exec_endpoint=ENDPOINT, judge_model=MODEL,
            judge_endpoint=ENDPOINT, timeout=180, judge_max_tokens=4096,
            judge_attempts=2, judge_parallel=1, max_inflight_requests=1, closed_book=True,
            structured_output="json_schema_planning", planning_string_max_length=2048,
            planning_communication_max_length=1024, planning_array_max_items=64)
        config = _config(args)
        if config.model_dump(mode="json") != source_identity["config"]:
            raise CheckpointIntegrityError("Diagnostic configuration/resource budgets differ from original C15")
        pipeline = _pipeline(config, frozen, OUTPUT / "runs", args, manifest, None)
        evaluator_version = pipeline.evaluator_factory(None).evaluator_version
        if any(row["complete"] and row["evaluator_version"] != evaluator_version for row in history):
            raise CheckpointIntegrityError("Judge identity/score rules differ from original C15")
        current_inputs = {str(split_path): file_hash(split_path),
                          str(RUNTIME / "source_manifest.json"): file_hash(RUNTIME / "source_manifest.json"),
                          str(Path(__file__).resolve()): file_hash(__file__)}
        current_inputs.update(immutable_files)
        source_anchor_hash = digest(source_anchor)
        registration = {"purpose": "Fixed ten-public-VAL quality diagnostic; never a replacement C15 or TEST comparison.",
            "original_split": "validation", "task_ids": task_ids, "task_count": 10,
            "source_checkpoint": "v35:C15", "source_state_version": VERSION,
            "state_hash": state_hash, "source_store": str(SOURCE_STORE),
            "source_journal": str(JOURNAL), "source_journal_sha256_at_registration": file_hash(JOURNAL),
            "source_c15_anchor_hash": source_anchor_hash, "source_manifest_sha256": file_hash(source_manifest_path),
            "source_artifact_hashes": artifact_hashes, "runtime": str(RUNTIME),
            "code_commit": commit, "code_hash": pipeline.code_hash, "code_manifest": code_entries,
            "code_manifest_sha256": file_hash(RUNTIME / "source_manifest.json"),
            "runner_sha256": file_hash(__file__), "input_hashes": current_inputs,
            "frozen_state_file_sha256": frozen_file_hash, "config": config.model_dump(mode="json"),
            "judge_model": MODEL, "judge_endpoint": ENDPOINT, "evaluator_version": evaluator_version,
            "exec_max_tokens": 12288, "judge_max_tokens": 4096, "judge_attempts": 2,
            "judge_parallel": 1, "max_inflight_requests": 1, "model_attempts": 5,
            "repeats": 1, "resume": False, "attribution": False, "state_updates_allowed": False,
            "test_calls": 0, "minimum_complete": 9, "missing_selection_utility": 0,
            "knowledge_policy": pipeline.knowledge_policy, "historical_C15": history,
            "historical_summary": historical_summary, "registered_at": datetime.now(timezone.utc).isoformat()}
        write_json(OUTPUT / "registration.json", registration)

        def audit_integrity():
            verify_manifest(SOURCE_RUNTIME, source_entries)
            verify_manifest(RUNTIME, code_entries)
            for filename, expected in {**current_inputs, **artifact_hashes}.items():
                if file_hash(filename) != expected:
                    raise CheckpointIntegrityError("Registered input/source artifact changed: " + filename)
            if digest(c15_anchor(read_json(JOURNAL))) != source_anchor_hash:
                raise CheckpointIntegrityError("Original C15 journal anchor changed")
            if (digest(source_store.snapshot(VERSION)) != state_hash or digest(snapshot) != state_hash
                    or digest(frozen.snapshot()) != state_hash or file_hash(frozen_path) != frozen_file_hash):
                raise CheckpointIntegrityError("Source or isolated historical state changed")

        def summarize():
            diagnostic = validation_summary(rows, task_ids, bounds)
            pairs = []
            for historical in history:
                current = next((row for row in rows if row["task_id"] == historical["task_id"]), None)
                paired = bool(current and current["complete"] and historical["complete"])
                pairs.append({"task_id": historical["task_id"], "historical_score": historical["score"],
                    "diagnostic_score": current["score"] if current else None, "complete_pair": paired,
                    "difference": current["score"] - historical["score"] if paired else None})
            differences = [pair["difference"] for pair in pairs if pair["complete_pair"]]
            valid = (diagnostic["eligible"] and integrity_ok and len(rows) == 10
                     and status in {"completed", "inconclusive"})
            accounting = {name: math.fsum(row.get("budget", {}).get(name, 0) for row in rows)
                          for name in ("model_calls", "tokens", "tool_calls", "wall_seconds")}
            reasons = Counter()
            length_calls = []
            for row in rows:
                run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
                synthesizer = (read_json(run_dir / "frozen_plan.json")["TeamSpec"]["synthesizer_id"]
                               if run_dir and (run_dir / "frozen_plan.json").exists() else None)
                for record in row.get("budget", {}).get("records", []):
                    if record.get("kind") != "model":
                        continue
                    reason = record.get("request", {}).get("finish_reason")
                    if reason:
                        reasons[reason] += 1
                    if reason == "length":
                        length_calls.append({"task_id": row["task_id"], "agent_id": record.get("agent_id"),
                            "synthesizer_id": synthesizer, "stage": record.get("stage"),
                            "call_id": record.get("call_id"), "output_tokens": record.get("output_tokens"),
                            "terminal_synthesizer": record.get("agent_id") == synthesizer})
            summary = {"status": status, "scope": "fixed_ten_public_VAL_engineering_diagnostic",
                "state_hash": state_hash, "tasks": [{key: value for key, value in row.items() if key != "budget"} for row in rows],
                "attempted_tasks": len(rows), "validation": diagnostic, "valid_summary": valid,
                "valid_selection_utility": diagnostic["selection_utility"] if valid else None,
                "historical_summary": historical_summary, "paired": pairs,
                "complete_pairs": len(differences),
                "mean_score_difference_complete_pairs": math.fsum(differences) / len(differences)
                    if valid and historical_summary["eligible"] and len(differences) >= 9 else None,
                "mean_score_difference_all_ten": math.fsum(differences) / 10 if valid and len(differences) == 10 else None,
                "selection_utility_difference": diagnostic["selection_utility"] - historical_summary["selection_utility"]
                    if valid and historical_summary["eligible"] else None,
                "accounting": {**accounting, "unknown_budget_slots": sum(not row.get("budget") for row in rows), "cost": None},
                "finish_reasons": dict(reasons), "length_calls": length_calls,
                "source_and_frozen_integrity_verified": integrity_ok, "experience_unchanged": integrity_ok,
                "test_calls": 0, "updated_at": datetime.now(timezone.utc).isoformat()}
            write_json(OUTPUT / "summary.json", summary)

        audit_integrity()
        integrity_ok = True
        status = "running"
        summarize()
        for task_id in task_ids:
            audit_integrity()
            write_json(OUTPUT / "slots" / task_id / "started.json", {"task_id": task_id,
                "state_hash": state_hash, "repeat": 0, "started_at": datetime.now(timezone.utc).isoformat()})
            row = {"task_id": task_id, "complete": False, "score": None,
                   "official_score": None, "selection_utility": 0.0, "theoretical_bounds": bounds[task_id]}
            try:
                outcome = pipeline.run_task(task_id, snapshot, mode="validate", repeat=0,
                                            attribution=False, resume=False)
                if outcome.get("proposals") or outcome.get("agent_updates"):
                    raise CheckpointIntegrityError("Held-out diagnostic attempted an experience update")
                feedback = outcome["evaluation"]
                row.update(run_dir=outcome["run_dir"], budget=outcome["budget"], answer_hash=outcome["answer_hash"],
                           status="complete" if feedback.get("complete") is True else "incomplete",
                           evaluator_version=feedback["evaluator_version"])
                if feedback["task_id"] != task_id or feedback["evaluator_version"] != evaluator_version:
                    raise CheckpointIntegrityError("Diagnostic evaluation identity mismatch")
                if feedback.get("complete") is True:
                    utility = normalize_score(feedback["score"], bounds[task_id])
                    row.update(complete=True, score=feedback["score"], official_score=feedback["score"], selection_utility=utility)
                write_json(OUTPUT / "slots" / task_id / "result.json", outcome)
            except CheckpointIntegrityError:
                raise
            except Exception as error:
                failure = getattr(error, "jit_mas_run_failure", {})
                row.update(status="failed", complete=False, score=None, official_score=None,
                    selection_utility=0.0, error_type=type(error).__name__, error=_safe_error(error),
                    run_dir=failure.get("run_dir", row.get("run_dir")), budget=failure.get("budget", row.get("budget", {})))
                write_json(OUTPUT / "slots" / task_id / "failed.json", row)
            rows.append(row)
            audit_integrity()
            summarize()
            print(json.dumps({key: row.get(key) for key in ("task_id", "status", "complete", "score")}), flush=True)
        status = "completed" if validation_summary(rows, task_ids, bounds)["eligible"] else "inconclusive"
        audit_integrity()
        summarize()
    except BaseException as error:
        integrity_ok = False
        if OUTPUT.exists():
            write_json(OUTPUT / "interruption.json", {"error_type": type(error).__name__,
                "error": _safe_error(error), "attempted_tasks": len(rows),
                "state_hash": state_hash, "ended_at": datetime.now(timezone.utc).isoformat()})
            if "summarize" in locals():
                status = "interrupted"
                summarize()
        raise
    finally:
        if frozen is not None:
            frozen.close()
        source_store.close()


if __name__ == "__main__":
    main()
