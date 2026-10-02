"""Conservative continuation for an interrupted isolated RR ``ours`` arm.

This helper is deliberately separate from the pilot runner.  It may only be
called after the launcher child processes have disappeared.  It reuses the
checkpoint journal and the existing ExperienceStore task journals: a durable
``complete``/``submitted`` evolution is resumed, an unresolved evolution is
closed as an interruption, and a validation slot that was ``started`` is
closed as an interruption.  No slot is sampled a second time.

The helper is not a general resume command.  It refuses changed code, data,
configuration, judge identity, experiment identity, or any evidence that TEST
generation has begun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _process_exists(pid: int) -> bool:
    """Return true for a live PID; fail closed when the PID cannot be checked."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    import psutil  # type: ignore
    try:
        return bool(psutil.pid_exists(pid) and psutil.Process(pid).is_running())
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _assert_processes_gone(launch: dict):
    live = []
    for key in ("preflight_pid", "method_pid"):
        pid = launch.get(key)
        if isinstance(pid, int) and _process_exists(pid):
            live.append(f"{key}={pid}")
    if live:
        raise RuntimeError("Refusing recovery while launcher child is live: " + ", ".join(live))
    # Also detect a relaunched pilot whose PID is not in the original launcher
    # metadata.  Only command lines that name this output are relevant.
    import psutil  # type: ignore
    marker = str(launch.get("method_output", "")).casefold()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or []).casefold()
            if marker and marker in command and "run_rr_two_arm_pilot" in command:
                raise RuntimeError(f"Another pilot process owns this output: {process.pid}")
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue


def _test_started(ours: Path, output: Path) -> bool:
    """Detect any durable TEST activity before opening the mutable store."""
    report = ours / "report.json"
    if report.is_file():
        body = _read(report)
        if body.get("submitted_outcomes") or body.get("test_failures"):
            return True
    test = ours / "test"
    if test.is_dir() and any(path.is_file() for path in test.rglob("*")):
        return True
    release = output / "test_release"
    return release.exists() and any(path.is_file() for path in release.rglob("*"))


def _same(a, b, name):
    if a != b:
        raise RuntimeError(f"Frozen {name} changed")


def _args_from_metadata(root: Path, output: Path, metadata: dict, launch: dict):
    identity = metadata["frozen_identity"]
    files = identity["files"]
    data = Path(metadata["data"]).resolve()
    joint = root / "paper/experiments/joint_task_splits_v5.json"
    split = output / "rr_split.json"
    for path, label in ((data, "data"), (joint, "joint manifest"), (split, "split")):
        expected = files.get(str(path))
        if expected is None or _sha256(path) != expected:
            raise RuntimeError(f"Frozen {label} bytes changed or are missing")
    endpoint = metadata["execution_endpoint"]
    model = metadata["execution_model"]
    models = metadata["config"]["models"]
    policy = metadata["structured_policy"]
    return SimpleNamespace(
        data=data, joint_manifest=joint, split_path=str(split),
        exec_model=model, exec_endpoint=endpoint,
        exec_max_tokens=int(models["exec"]["max_tokens"]), timeout=float(models["exec"]["timeout"]),
        judge_model=metadata["judge_model"], judge_endpoint=metadata["judge_endpoint"],
        judge_max_tokens=int(models["judge"]["max_tokens"]), judge_attempts=int(metadata["judge_attempts"]),
        judge_parallel=int(metadata["judge_parallel"]),
        max_inflight_requests=int(metadata["process_request_cap"]),
        closed_book=metadata["knowledge_policy"] == "model_general_knowledge_allowed",
        structured_output=metadata["structured_output"],
        planning_string_max_length=int(policy["planning_string_max_length"]),
        planning_communication_max_length=int(policy["planning_communication_max_length"]),
        planning_array_max_items=int(policy["planning_array_max_items"]), frozen_identity=identity,
        arm="ours", evidence_dir=None, reuse_evaluations_from=[],
    )


def _verify_runtime(runtime: Path):
    """Verify every frozen source byte before importing the continuation."""
    manifest_path = runtime / "source_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Frozen runtime source manifest is missing")
    manifest = _read(manifest_path)
    for relative, expected in manifest.items():
        path = runtime / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen runtime file changed: {relative}")
    expected_paths = {str(Path(relative)).replace("\\", "/") for relative in manifest}
    source_suffixes = {".py", ".yaml", ".yml", ".txt", ".md"}
    for path in runtime.rglob("*"):
        if not path.is_file() or path.suffix not in source_suffixes:
            continue
        relative = path.relative_to(runtime)
        if any(part in {"__pycache__", "workspaces"} for part in relative.parts):
            continue
        if relative.as_posix() not in expected_paths:
            raise RuntimeError(f"Unlisted importable runtime file: {relative}")


def _audit(root: Path, output: Path, launch_path: Path | None = None):
    launch_path = launch_path or root / "outputs/rr_deepseek_method_launch_20261002_v34/launch.json"
    metadata_path = output / "pilot_metadata.json"
    ours = output / "ours"
    if not launch_path.is_file() or not metadata_path.is_file() or not ours.is_dir():
        raise RuntimeError("v34 launch metadata, pilot metadata, or ours output is missing")
    launch, metadata = _read(launch_path), _read(metadata_path)
    _same(Path(launch["method_output"]).resolve(), output.resolve(), "output location")
    _assert_processes_gone(launch)
    if _test_started(ours, output):
        raise RuntimeError("TEST generation already started; manual sealed recovery is required")
    if metadata.get("selected_arm") != "ours" or metadata.get("parallel_arms"):
        raise RuntimeError("This helper only accepts the isolated v34 ours arm")
    if metadata.get("formal_experience_initial_state") not in (None, "empty"):
        raise RuntimeError("The continuation is not an empty-state v34 run")
    return launch, metadata, ours


def _restore_transport_environment(metadata: dict, launch: dict):
    mapping = {"JIT_MAS_MODEL_ATTEMPTS": metadata["model_attempts"],
               "MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES": metadata["consecutive_failure_limit"],
               "MODULAR_AGENT_API_FAILURE_ACTION": launch["api_failure_action"],
               "JIT_MAS_DISABLE_KEEPALIVE": metadata["disable_keepalive"],
               "JIT_MAS_TLS_VERIFY": metadata["tls_verify"],
               "JIT_MAS_TLS_ENDPOINT": metadata["tls_endpoint"]}
    for name, value in mapping.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def _recover_validation_result(ours: Path, cache: dict, task_id: str, snapshot, repeat: int):
    """Read only the exact checkpoint slot and verify its durable documents."""
    from jit_mas.checkpoints import CheckpointIntegrityError
    from jit_mas.schemas import digest

    state_hash = digest(snapshot)
    if cache.get("state_hash") != state_hash or digest(cache["snapshot"]) != state_hash:
        raise CheckpointIntegrityError("Validation recovery cache snapshot changed")
    checkpoint = ours / "checkpoints" / f"C{cache['first_position']}"
    slot = checkpoint / "slots" / task_id
    result_path = slot / "result.json"
    if not result_path.is_file():
        # A completed pipeline may have been written immediately before the
        # coordinator was stopped.  It is not safe to guess which run belongs
        # to this slot, so a missing coordinator result remains an interruption.
        return None
    result = _read(result_path)
    score = (result.get("evaluation") or {}).get("score")
    if (result.get("task_id") != task_id or result.get("mode") != "validate"
            or result.get("experience_hash") != state_hash
            or result.get("experience_version") != snapshot.version
            or result.get("repeat", repeat) != repeat
            or result.get("experience_updates") or result.get("agent_pool_updates")
            or (result.get("evaluation") or {}).get("complete") is not True
            or not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score)):
        raise CheckpointIntegrityError("Durable validation result is incomplete or unbound")
    run_dir = Path(result.get("run_dir", "")).resolve()
    runs = (checkpoint / "runs").resolve()
    if run_dir == runs or runs not in run_dir.parents:
        raise CheckpointIntegrityError("Validation result points outside its checkpoint")
    required = ("complete.json", "run_manifest.json", "submission.json", "evaluation.json", "budget.json")
    if any(not (run_dir / name).is_file() for name in required):
        raise CheckpointIntegrityError("Validation result has incomplete durable documents")
    complete, manifest = _read(run_dir / "complete.json"), _read(run_dir / "run_manifest.json")
    submission, evaluation = _read(run_dir / "submission.json"), _read(run_dir / "evaluation.json")
    if (complete != result or manifest.get("experience_hash") != state_hash
            or manifest.get("experience_version") != snapshot.version
            or manifest.get("mode") != "validate"
            or manifest.get("run_key") != result.get("run_key")
            or (manifest.get("comparison") or {}).get("task", {}).get("task_id") != task_id
            or (manifest.get("comparison") or {}).get("repeat") != repeat
            or digest(manifest.get("comparison")) != result.get("comparison_fingerprint")
            or digest(submission.get("answer")) != submission.get("answer_hash")
            or submission.get("answer_hash") != result.get("answer_hash")
            or evaluation != result.get("evaluation")
            or evaluation.get("raw", {}).get("submission_answer_hash") != result.get("answer_hash")
            or _read(run_dir / "budget.json") != result.get("budget")):
        raise CheckpointIntegrityError("Validation durable documents or answer hash changed")
    return result


def _run(root: Path, output: Path, launch_path: Path | None = None):
    launch, metadata, ours = _audit(root, output, launch_path)
    runtime = Path(launch["runtime"]).resolve()
    if not runtime.is_dir():
        raise RuntimeError("Frozen runtime is missing")
    _verify_runtime(runtime)
    _restore_transport_environment(metadata, launch)
    # Runtime imports must precede the workspace imports.  This is what makes
    # _assert_identity compare against the code that v34 actually froze.
    sys.path.insert(0, str(runtime))
    from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
    from jit_mas.experience import ExperienceStore
    from jit_mas.pipeline import code_fingerprint
    from scripts.run_rr_two_arm_pilot import (
        _NormalizedCheckpointRunner, _assert_identity, _config, _manifest,
        _pipeline, _safe_error, _score_rows, _validate_checkpoint, write_json,
    )
    from jit_mas.schemas import digest, utc_now

    args = _args_from_metadata(root, output, metadata, launch)
    if code_fingerprint() != metadata["frozen_identity"]["code"]:
        raise RuntimeError("Frozen runtime code fingerprint changed")
    config = _config(args)
    if digest(config) != metadata["frozen_identity"]["config"]:
        raise RuntimeError("Frozen configuration changed")
    _same(args.exec_max_tokens, launch["exec_max_tokens"], "execution token cap")
    _same(args.judge_model, launch["judge_model"], "launcher judge model")
    _same(config.model_dump(mode="json"), metadata["config"], "full parameters")
    config.check_native()
    _assert_identity(args)
    manifest = _manifest(args.joint_manifest)
    _same(manifest.model_dump(mode="json"), metadata["manifest"], "manifest membership")
    _assert_processes_gone(launch)
    store = ExperienceStore(ours / "experience.sqlite")
    journal_path = ours / "checkpoints/checkpoint_journal.json"
    journal = _read(journal_path)
    expected = (journal["sources"][-1].get("after_hash")
                if journal["sources"] and journal["sources"][-1]["status"] != "started"
                else digest(journal["initial_snapshot"]))
    if not journal["sources"] or journal["sources"][-1]["status"] == "started":
        # A started source may have committed an update only at the pipeline's
        # atomic completion boundary.  Without a durable outcome, refuse to
        # guess what happened rather than silently retaining a partial update.
        if not journal["sources"] and digest(store.snapshot()) != expected:
            raise RuntimeError("Initial experience snapshot differs from checkpoint journal")
        if journal["sources"]:
            row = journal["sources"][-1]
            current = store.snapshot()
            prior = store.task_run("evolve", row["task_id"])
            durable_complete = bool(prior and prior["status"] in ("complete", "submitted")
                                    and prior["outcome"])
            if digest(current) != row["before_hash"] and not durable_complete:
                raise RuntimeError("Unresolved EVO slot changed the experience store; manual audit required")
            if digest(current) != row["before_hash"] and prior and prior["status"] == "complete":
                outcome = prior["outcome"]
                if outcome.get("next_experience_version") != current.version:
                    raise RuntimeError("Completed EVO receipt does not match the current experience version")
                receipts = outcome.get("experience_updates", []) + outcome.get("agent_pool_updates", [])
                if receipts and any(receipt.get("snapshot_hash") != digest(current) for receipt in receipts):
                    raise RuntimeError("Completed EVO receipt does not match the current experience hash")
    elif digest(store.snapshot()) != expected:
        raise RuntimeError("Experience snapshot hash differs from checkpoint journal")

    pipeline = _pipeline(config, store, ours / "evolution", args, manifest, None)
    identity = {"arm": "ours_evolve_then_test", "run_id": 0,
                "manifest": manifest.model_dump(mode="json"), "config": config.model_dump(mode="json"),
                "dataset_sha256": pipeline.benchmark_dataset.dataset_sha256,
                "frozen_identity": args.frozen_identity,
                "selection": "fixed theoretical RR bounds, 9/10 complete, maximum normalized mean"}

    def evolve(task_id):
        _assert_identity(args)
        result = pipeline.run("evolve", [task_id], resume=True)
        _assert_identity(args)
        return result[0]

    def evaluate(task_id, snapshot, repeat, frozen):
        label = f"C{len(runner.journal['sources'])}"
        return _validate_checkpoint(config, store, args, manifest, None, ours,
                                    snapshot, label, task_id, repeat, frozen)

    def recover_validation(task_id, snapshot, repeat, frozen):
        """Reuse a fully written validation result, without judging again."""
        matching = [cache for cache in runner.journal["validation_cache"].values()
                    if cache["state_hash"] == digest(snapshot)]
        if len(matching) != 1:
            raise CheckpointIntegrityError("Ambiguous validation recovery cache")
        return _recover_validation_result(ours, matching[0], task_id, snapshot, repeat)

    runner = _NormalizedCheckpointRunner(
        store=store, evolution_ids=manifest.evolution, validation_ids=manifest.validation,
        output_dir=ours / "checkpoints", identity=identity,
        lower_bounds=pipeline.benchmark_dataset.lower_bounds,
        upper_bounds=pipeline.benchmark_dataset.upper_bounds,
        evolve=evolve, evaluate=evaluate, batch_size=5, repeats=1,
        minimum_completion=0.9, recover_evaluation=recover_validation)
    # A source with no durable task outcome is closed once, with no replacement
    # model call.  A submitted/complete task is recovered by CheckpointRunner.
    if runner.journal["sources"] and runner.journal["sources"][-1]["status"] == "started":
        row = runner.journal["sources"][-1]
        prior = store.task_run("evolve", row["task_id"])
        if not prior or prior["status"] not in ("complete", "submitted"):
            runner.record_interrupted_source_failure("Launcher stopped; no durable EVO outcome", budget=None)
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
    selected_snapshot_path = ours / "selected_snapshot.json"
    if selected is not None:
        write_json(selected_snapshot_path, selected.model_dump(mode="json"))
    test_outcomes, test_failures = [], []
    report = {"arm": "ours_evolve_then_test", "status": "submitting" if selected is not None else "inconclusive",
              "started_at": metadata.get("started_at", utc_now()), "manifest": manifest.model_dump(mode="json"),
              "checkpoints": checkpoints, "checkpoint_selection": checkpoint_result,
              "selected": selected_row, "selected_checkpoint": selected_label,
              "selected_snapshot": str(selected_snapshot_path) if selected is not None else None,
              "selected_snapshot_hash": digest(selected) if selected is not None else None,
              "submitted_outcomes": test_outcomes, "test_failures": test_failures,
              "experience_version": store.snapshot().version}
    write_json(ours / "report.json", report)
    if selected is None:
        for task_id in manifest.test:
            failure = {"task_id": task_id, "status": "missing", "budget": None,
                       "error_type": "NoEligibleCheckpoint",
                       "error": "All five fixed VAL candidates failed the 9/10 completeness threshold"}
            test_failures.append(failure)
            write_json(ours / "test/slots" / task_id / "missing.json", failure)
    else:
        state_hash, trajectory_hash = digest(selected), digest(store.snapshot())
        test_state = snapshot_store(selected, ours / "test/selected_state.sqlite")
        try:
            test_pipeline = _pipeline(config, test_state, ours / "test/runs", args, manifest, None)
            for task_id in manifest.test:
                slot = ours / "test/slots" / task_id
                write_json(slot / "started.json", {"task_id": task_id, "started_at": utc_now(), "state_hash": state_hash})
                try:
                    _assert_identity(args)
                    outcome = test_pipeline.run_task(task_id, selected, mode="evaluate", repeat=0,
                                                     attribution=False, resume=False, defer_evaluation=True)
                    if (outcome.get("evaluation") is not None
                            or outcome.get("status") != "submitted_unscored"
                            or outcome.get("experience_updates")
                            or outcome.get("agent_pool_updates")):
                        raise CheckpointIntegrityError("TEST must return one unscored immutable submission")
                    test_outcomes.append(outcome)
                    write_json(slot / "submitted.json", outcome)
                except CheckpointIntegrityError:
                    raise
                except Exception as error:
                    failure = {"task_id": task_id, "status": "failed", "error_type": type(error).__name__,
                               "error": _safe_error(error), "state_hash": state_hash, "budget": None}
                    test_failures.append(failure)
                    write_json(slot / "failed.json", failure)
                finally:
                    if digest(selected) != state_hash or digest(test_state.snapshot()) != state_hash or digest(store.snapshot()) != trajectory_hash:
                        raise CheckpointIntegrityError("TEST mutated its selected state or EVO trajectory")
        finally:
            test_state.close()
    report = {"arm": "ours_evolve_then_test", "status": "submitted" if selected is not None else "inconclusive",
              "started_at": report["started_at"], "elapsed_seconds": None,
              "scope": "one-run ResearchRubrics 20 EVO + 10 VAL + 33 TEST, closed-book unless evidence supplied",
              "manifest": manifest.model_dump(mode="json"), "checkpoints": checkpoints,
              "checkpoint_selection": checkpoint_result, "selected": selected_row,
              "selected_checkpoint": selected_label, "selected_snapshot": str(selected_snapshot_path) if selected is not None else None,
              "selected_snapshot_hash": digest(selected) if selected is not None else None,
              "evolution": _score_rows(evolution, evolution_failures, manifest.evolution),
              "evolution_failures": evolution_failures, "test": _score_rows(test_outcomes, test_failures, manifest.test),
              "submitted_outcomes": test_outcomes, "test_failures": test_failures,
              "experience_version": store.snapshot().version, "test_feedback_updates_experience": False,
              "recovered": True}
    write_json(ours / "report.json", report)
    # Re-enter the normal deferred-judgment sealing path after generation.  It
    # records every TEST slot before releasing any judge request and updates
    # the ordinary pilot artifacts, while the read-only scoring store keeps
    # the selected experience immutable.
    from scripts.run_rr_two_arm_pilot import _score_deferred
    write_json(output / "generation_reports.json", {"ours": report})
    _score_deferred(args, config, manifest, None, output, {"ours": report})
    report = _read(ours / "report.json")
    metadata = _read(output / "pilot_metadata.json")
    metadata.update(status="completed" if report.get("status") == "completed" else "incomplete",
                    recovered=True, finished_at=utc_now())
    write_json(output / "pilot_metadata.json", metadata)
    write_json(output / "comparison.json", {"metadata": metadata, "reports": {"ours": _read(ours / "report.json")},
                                               "paired": {"available": False, "reason": "Only ours arm was resumed"}})
    store.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/rr_jit_mas_run0_deepseekjudge_20261002_v34"))
    args = parser.parse_args(argv)
    report = _run(args.root.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "recovered": True,
                      "test_completed": report["test"]["completed"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
