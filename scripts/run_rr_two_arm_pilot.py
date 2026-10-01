"""Run one isolated ResearchRubrics direct-vs-JIT-MAS pilot in parallel.

This pilot uses the frozen v5 ResearchRubrics membership (20 EVO, 10 VAL and
33 TEST) for one run. Shared public evidence is required by default. An
explicit --closed-book flag registers a separate protocol deviation.
Credentials are read from environment variables and are never written to the
experiment artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import os
import logging
from pathlib import Path
import time

from jit_mas.checkpoints import CheckpointIntegrityError, CheckpointRunner, snapshot_store
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.experience import ExperienceStore
from jit_mas.independent_protocol import normalize_score
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest, utc_now
from scripts.mas_baseline_methods import JudgeEnvelopeModel, run_direct
from scripts.benchmark_jit_mas_live import SafeTransport
from scripts.run_jit_mas import make_pipeline
from jit_mas.test_release import TestRelease, score_with_pipeline


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
                 key_env="RR_JUDGE_API_KEY", max_tokens=16000, timeout=args.timeout,
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
        max_inflight_requests=2,
        execution_timeout=900, task_timeout=900, local_planning=True, local_rounds=1,
        local_attribution=True, persistent_experience=True, evolving_agent_pool=True,
        explicit_rubrics=True, available_tools=[])


def _pipeline(config, store, output, args, manifest, evidence_dir):
    split_path = Path(args.split_path)
    pipeline = make_pipeline(config, store, output, data=args.data, splits=split_path,
                             benchmark="researchrubrics", evidence_dir=evidence_dir)
    pipeline.models = SafeModels(pipeline.models, config)
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
                max_attempts=1, judge_factory=independent_judge,
                max_parallel_judgments=judge_parallel)

        pipeline.evaluator_factory = evaluator_factory
    return pipeline


class SafeModels:
    def __init__(self, provider, config):
        self.provider, self.config = provider, config

    def create(self, role, agent_id, ledger, stage):
        model = SafeTransport(self.provider.create(role, agent_id, ledger, stage),
                              os.environ[self.config.models[role].key_env])
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
                 for arm in ("baseline", "ours") for task_id in manifest.test]
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
            results = []
            for task_id in manifest.test:
                _assert_identity(args)
                result = release.evaluate(f"{arm}:{task_id}", lambda record: score_with_pipeline(pipeline, record))
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

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(score_arm, arm) for arm in ("baseline", "ours")]
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
    parser.add_argument("--evidence-dir")
    parser.add_argument("--closed-book", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.judge_parallel < 1 or args.judge_parallel > 2:
        parser.error("--judge-parallel must be 1 or 2 under the frozen request cap")
    if not Path(args.data).is_file() or not Path(args.joint_manifest).is_file():
        parser.error("Pinned data and joint manifest must exist")
    manifest = _manifest(args.joint_manifest)
    config = _config(args)
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
    args.frozen_identity = {"code": code_fingerprint(), "runner": _runner_hash(), "config": digest(config),
                            "judge_parallel": args.judge_parallel,
                            "files": {str(path): _sha256_file(path) for path in files}}
    metadata = {"version": "rr-two-arm-pilot-v2", "manifest": manifest.model_dump(mode="json"),
                "data": str(Path(args.data).resolve()), **preflight,
                "execution_model": args.exec_model, "execution_endpoint": args.exec_endpoint,
                "judge_model": args.judge_model, "judge_endpoint": args.judge_endpoint,
                "judge_parallel": args.judge_parallel, "process_request_cap": config.max_inflight_requests,
                "config": config.model_dump(mode="json"), "frozen_identity": args.frozen_identity,
                "protocol_scope": "RR-only run0 projection of v5; not the full mixed-benchmark campaign",
                "parallel_arms": True, "repeats": 1, "test_feedback_updates_experience": False,
                "knowledge_policy": ("model_general_knowledge_allowed" if args.closed_book
                                      else "fixed_shared_evidence_only"),
                "status": "preflight_passed" if args.check_only else "running",
                "started_at": utc_now(), "monetary_cost": None}
    write_json(output / "pilot_metadata.json", metadata)
    if args.check_only:
        print(json.dumps({"status": "preflight_passed", "paid_requests": 0, **preflight}, ensure_ascii=True))
        return 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        baseline = workers.submit(_run_arm, run_baseline, args, config, manifest, evidence_dir, output / "baseline")
        ours = workers.submit(_run_arm, run_ours, args, config, manifest, evidence_dir, output / "ours")
        reports = {"baseline": baseline.result(), "ours": ours.result()}
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
    print(json.dumps({"status": metadata["status"], "output": str(output),
                      "baseline_mean": reports["baseline"]["results"]["mean_score"],
                      "ours_mean": reports["ours"]["test"]["mean_score"],
                      "mean_scope": "complete-only; fixed-task means are null until all tasks complete",
                      "baseline_completed": reports["baseline"]["results"]["completed"],
                      "ours_completed": reports["ours"]["test"]["completed"]}, ensure_ascii=True))
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
