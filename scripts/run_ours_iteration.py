"""Focused Ours-only development iterations with immutable references and sealed scoring."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import yaml

from jit_mas.benchmarks import ALL_BENCHMARK_NAMES, load_benchmark
from jit_mas.budget import BudgetLedger
from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.config import MASConfig
from jit_mas.evidence import evidence_pack_path, load_evidence_tasks
from jit_mas.experience import ExperienceStore
from jit_mas.independent_campaign import coordinator_lock
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.pipeline import code_fingerprint, convert_feedback, write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from jit_mas.test_release import TestRelease, remaining_task_budget
from scripts.env_config import resolve_env_placeholders
from scripts.eval.config import load_dotenv
from scripts.run_benchmark_experiment import file_hash
from scripts.run_jit_mas import make_pipeline
from scripts.run_joint_experiment import normalize_score


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def select_tasks(manifest, benchmarks, task_ids, partition, limit):
    selected = {name: [] for name in benchmarks}
    for value in task_ids:
        if "=" in value:
            name, task_id = value.split("=", 1)
        elif len(benchmarks) == 1:
            name, task_id = benchmarks[0], value
        else:
            raise ValueError("Multiple benchmarks require --task-id BENCHMARK=TASK_ID")
        if name not in selected or task_id not in manifest["memberships"][name][partition]:
            raise ValueError("Task is outside the selected benchmark/partition membership")
        if task_id in selected[name]:
            raise ValueError("A task may occur only once per iteration")
        selected[name].append(task_id)
    for name in benchmarks:
        if not task_ids:
            selected[name] = manifest["memberships"][name][partition][:limit]
        if not selected[name]:
            raise ValueError(f"No selected {partition} tasks for {name}")
    return selected


def reference_rows(document):
    if isinstance(document.get("rows"), list):
        return document["rows"]
    rows = []
    for slot_id, value in document.get("slots", {}).items():
        result = value.get("result") or {}
        feedback = result.get("outcome", {}).get("evaluation") or {}
        score = result.get("score")
        if score is None:
            score = feedback.get("score")
        if value.get("benchmark") == "writingbench":
            score = feedback.get("raw", {}).get("native_mean")
        rows.append({"task_id": value.get("task_id", result.get("task_id")),
                     "method": value.get("method", "ours"), "slot_id": slot_id,
                     "score": score, "complete": result.get("complete", feedback.get("complete", False))})
    return rows


def project_evidence(tasks, source, destination):
    manifest = read_json(source / "manifest.json")
    if manifest.get("manifest_sha256") != digest({key: value for key, value in manifest.items()
                                                if key != "manifest_sha256"}):
        raise CheckpointIntegrityError("Original evidence manifest hash mismatch")
    projected = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "tasks", "count"}}
    projected.update(count=len(tasks), tasks={task_id: manifest["tasks"][task_id] for task_id in tasks})
    projected["manifest_sha256"] = digest(projected)
    destination.mkdir(parents=True, exist_ok=True)
    files = {str(source / "manifest.json"): file_hash(source / "manifest.json")}
    for task_id in tasks:
        original, copied = evidence_pack_path(source, task_id), evidence_pack_path(destination, task_id)
        expected = projected["tasks"][task_id]["file_sha256"]
        if file_hash(original) != expected:
            raise CheckpointIntegrityError("Original evidence pack hash mismatch")
        if copied.exists() and file_hash(copied) != expected:
            raise CheckpointIntegrityError("Projected evidence pack changed")
        if not copied.exists():
            shutil.copy2(original, copied)
        files[str(original)] = files[str(copied)] = expected
    path = destination / "manifest.json"
    if path.exists() and read_json(path) != projected:
        raise CheckpointIntegrityError("Projected evidence manifest changed")
    if not path.exists():
        write_json(path, projected)
    files[str(path)] = file_hash(path)
    return load_evidence_tasks(tasks, destination, expected_count=len(tasks)), files


def score_submission(pipeline, record):
    slot, submission = record["slot"], record["submission"]
    if digest(submission["answer"]) != submission["answer_hash"]:
        raise CheckpointIntegrityError("Saved answer integrity failure")
    config = pipeline.config
    used = record["outcome"]["budget"]
    active_seconds = max(0.0, used.get("wall_seconds", 0) - used.get("request_queue_idle_seconds", 0))
    remaining_seconds = max(0.0, config.task_timeout - active_seconds) if config.task_timeout else None
    ledger = BudgetLedger(**remaining_task_budget(config, record["outcome"]), timeout_seconds=remaining_seconds)
    try:
        if remaining_seconds is not None and remaining_seconds <= 0:
            raise TimeoutError("Original task time budget exhausted before deferred scoring")
        judge = pipeline.models.create("judge", "judge", ledger, "evaluation")
        evaluator = pipeline.evaluator_factory(judge)
        raw = evaluator.evaluate(str(submission["answer"]), ground_truth=slot["task_id"],
                                 private_record=pipeline.private_records[slot["task_id"]])
        raw["submission_answer_hash"] = submission["answer_hash"]
        feedback = convert_feedback(raw)
    except Exception as error:
        error.evaluation_budget = ledger.snapshot()
        raise
    return {"slot": slot, "official_score": feedback.score, "complete": feedback.complete,
            "evaluation": feedback.model_dump(mode="json"), "evaluated_at": utc_now(),
            "evaluation_budget": ledger.snapshot(), "generation_budget": used,
            "answer_hash": submission["answer_hash"]}


class OursIteration:
    def __init__(self, args):
        self.output = args.output.resolve()
        self.bundle_path = args.bundle.resolve()
        bundle = read_json(self.bundle_path)
        base = self.bundle_path.parent
        self.config_path = (args.config.resolve() if args.config else resolve_path(bundle["config"], base))
        config = MASConfig.model_validate(resolve_env_placeholders(
            yaml.safe_load(self.config_path.read_text(encoding="utf-8"))))
        if config.backend != "native_jit" or config.candidates != 1:
            raise ValueError("Ours iterations require native_jit and exactly one candidate")
        self.workers, self.judge_workers = args.workers, args.judge_workers
        inflight = args.max_inflight or config.max_inflight_requests or max(self.workers, self.judge_workers)
        judge_parallel = args.judge_parallel or config.judge_parallel
        self.config = config.model_copy(update={"max_inflight_requests": inflight,
                                                "judge_parallel": judge_parallel})
        manifest_path = resolve_path(bundle["manifest"], base)
        manifest = read_json(manifest_path)
        selected = select_tasks(manifest, args.benchmark, args.task_id, args.partition, args.limit)
        overrides = {}
        for value in args.evidence:
            name, separator, path = value.partition("=")
            if not separator or name not in selected or name in overrides:
                raise ValueError("Use unique --evidence BENCHMARK=DIRECTORY arguments")
            overrides[name] = Path(path).resolve()
        self.files = {str(path): file_hash(path) for path in
                      (self.bundle_path, self.config_path, manifest_path)}
        self.materials, tasks = {}, []
        for name, task_ids in selected.items():
            item = bundle["benchmarks"][name]
            data = resolve_path(item["data"], base)
            dataset = load_benchmark(name, data, available_tools=self.config.available_tools)
            if dataset.dataset_sha256 != manifest["benchmarks"][name]["dataset_sha256"]:
                raise CheckpointIntegrityError("Pinned dataset hash mismatch")
            self.files[str(data)] = file_hash(data)
            public = {task_id: dataset.tasks[task_id] for task_id in task_ids}
            original_evidence = overrides.get(name) or (resolve_path(item["evidence_dir"], base)
                                                        if item.get("evidence_dir") else None)
            evidence = self.output / "evidence" / name if original_evidence else None
            if evidence:
                public, evidence_files = project_evidence(public, original_evidence, evidence)
                self.files.update(evidence_files)
            checker = resolve_path(item["checker_source_root"], base) if item.get("checker_source_root") else None
            checker_identity = PinnedInstructionChecker(checker, name).identity if checker else None
            self.materials[name] = {"data": str(data), "evidence": str(evidence) if evidence else None,
                                    "original_evidence": str(original_evidence) if original_evidence else None,
                                    "knowledge_mode": "frozen_evidence" if evidence else "benchmark_public_input",
                                    "checker": str(checker) if checker else None,
                                    "checker_identity": checker_identity, "task_ids": task_ids}
            for task_id in task_ids:
                tasks.append({"benchmark": name, "task_id": task_id, "method": "ours",
                              "original_partition": args.partition, "public_task_hash": digest(public[task_id]),
                              "private_record_hash": digest(dataset.private_records[task_id]),
                              "bounds": [1.0, 10.0] if name == "writingbench" else
                                        [dataset.lower_bounds[task_id], dataset.upper_bounds[task_id]]})
        self.references = []
        for path in args.reference:
            path = path.resolve()
            self.files[str(path)] = file_hash(path)
            document = read_json(path)
            identity_path = path.parent / "registration.json"
            identity = read_json(identity_path) if identity_path.is_file() else {}
            if identity:
                self.files[str(identity_path)] = file_hash(identity_path)
            self.references.append({"path": str(path), "sha256": self.files[str(path)],
                                    "rows": reference_rows(document),
                                    "aggregate": document.get("by_condition", document.get("arms")),
                                    "knowledge_mode": identity.get("knowledge_mode"),
                                    "source_identity": identity.get("sources", identity.get("materials")),
                                    "configuration": identity.get("configuration")})
        if args.state:
            self.state = args.state.resolve()
            self.files[str(self.state)] = file_hash(self.state)
            store = ExperienceStore(self.state, read_only=True)
        else:
            self.state = self.output / "states" / "initial.sqlite"
            store = (ExperienceStore(self.state, read_only=True) if self.state.exists() else
                     snapshot_store(ExperienceSnapshot(), self.state))
        try:
            self.snapshot = store.snapshot()
            if not args.state and digest(self.snapshot) != digest(ExperienceSnapshot()):
                raise CheckpointIntegrityError("Initial state must remain empty")
        finally:
            store.close()
        for task in tasks:
            task["experience_hash"] = digest(self.snapshot)
            task["slot_id"] = digest(task)
        for name, material in self.materials.items():
            path = self.output / "splits" / f"{name}.json"
            document = {"test": material["task_ids"]}
            if path.exists() and read_json(path) != document:
                raise CheckpointIntegrityError("Development runtime split changed")
            if not path.exists():
                write_json(path, document)
            self.files[str(path)] = file_hash(path)
        self.registration = {"schema": "ours-development-iteration-v1", "formal_result": False,
                             "partition": args.partition, "files": self.files,
                             "configuration": self.config.model_dump(mode="json"),
                             "code": code_fingerprint(), "runner_sha256": file_hash(Path(__file__)),
                             "state_path": str(self.state), "state_hash": digest(self.snapshot),
                             "materials": self.materials, "slots": tasks,
                             "workers": self.workers, "judge_workers": self.judge_workers,
                             "references": self.references,
                             "policy": "Ours only; consume once; seal all outputs before scoring; references read only"}
        self.registration["registration_hash"] = digest(self.registration)
        path = self.output / "registration.json"
        if path.exists() and read_json(path) != self.registration:
            raise CheckpointIntegrityError("Iteration identity changed; use a new output directory")
        if not path.exists():
            write_json(path, self.registration)
        self.release = TestRelease(self.output / "release", tasks)

    def assert_frozen(self):
        if read_json(self.output / "registration.json") != self.registration:
            raise CheckpointIntegrityError("Iteration registration changed")
        release = TestRelease(self.release.directory)
        if release.inventory != self.release.inventory:
            raise CheckpointIntegrityError("Iteration inventory changed")
        if (code_fingerprint() != self.registration["code"] or
                file_hash(Path(__file__)) != self.registration["runner_sha256"] or
                any(file_hash(Path(path)) != expected for path, expected in self.files.items())):
            raise CheckpointIntegrityError("Iteration code, inputs or reference files changed")
        store = ExperienceStore(self.state, read_only=True)
        try:
            if digest(store.snapshot()) != self.registration["state_hash"]:
                raise CheckpointIntegrityError("Iteration state changed")
        finally:
            store.close()
        if (self.release.directory / "seal.json").exists():
            self.release.seal()
        for name, material in self.materials.items():
            if material["checker"]:
                if PinnedInstructionChecker(material["checker"], name).identity != material["checker_identity"]:
                    raise CheckpointIntegrityError("Instruction checker changed")

    def pipeline(self, slot):
        material = self.materials[slot["benchmark"]]
        store = ExperienceStore(self.state, read_only=True)
        try:
            pipeline = make_pipeline(self.config, store, self.output / "runs" / slot["slot_id"],
                                     data=material["data"], splits=self.output / "splits" / f"{slot['benchmark']}.json",
                                     benchmark=slot["benchmark"], evidence_dir=material["evidence"],
                                     checker_source_root=material["checker"])
            if digest(pipeline.tasks[slot["task_id"]]) != slot["public_task_hash"]:
                raise CheckpointIntegrityError("Registered public task changed")
            return pipeline, store
        except BaseException:
            store.close()
            raise

    def submit(self, slot):
        slot_id = slot["slot_id"]
        if self.release._path(slot_id).exists():
            return
        started = self.release._path(slot_id, "submission_started")
        if started.exists():
            outcomes = list((self.output / "runs" / slot_id).glob("*/complete.json"))
            if len(outcomes) == 1:
                self.release.record(slot_id, read_json(outcomes[0]))
            else:
                budgets = list((self.output / "runs" / slot_id).rglob("budget.json"))
                observed = [read_json(path) for path in budgets]
                budget = (observed[0] if len(observed) == 1 else
                          {"usage_unknown": True, "observed_budgets": observed})
                self.release.record_failure(slot_id, error_type="InterruptedSubmission",
                                            budget=budget)
            return
        write_json(started, {"slot_hash": digest(slot), "started_at": utc_now()})
        store = None
        try:
            pipeline, store = self.pipeline(slot)
            outcome = pipeline.run_task(slot["task_id"], self.snapshot, mode="evaluate", repeat=0,
                                        attribution=False, defer_evaluation=True, resume=False)
            self.release.record(slot_id, outcome)
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            failure = getattr(error, "jit_mas_run_failure", {})
            self.release.record_failure(slot_id, error_type=type(error).__name__,
                                        budget=failure.get("budget", {"usage_unknown": True}))
        finally:
            if store is not None:
                store.close()
        print(json.dumps({"stage": "generation", "task_id": slot["task_id"],
                          "status": read_json(self.release._path(slot_id))["status"]}), flush=True)

    def score(self, slot):
        slot_id = slot["slot_id"]
        output = self.release._path(slot_id, "evaluations")
        if output.exists():
            return
        if self.release._path(slot_id, "evaluation_started").exists():
            write_json(output, {"slot": slot, "official_score": None, "complete": False,
                                "status": "interrupted_evaluation", "evaluation_budget": {"usage_unknown": True}})
            return
        if read_json(self.release._path(slot_id))["status"] == "failed":
            self.release.evaluate(slot_id, lambda record: None)
            return
        pipeline, store = self.pipeline(slot)
        try:
            result = self.release.evaluate(slot_id, lambda record: score_submission(pipeline, record))
        finally:
            store.close()
        print(json.dumps({"stage": "evaluation", "task_id": slot["task_id"],
                          "score": result.get("official_score"), "complete": result.get("complete")}), flush=True)

    def run(self):
        self.assert_frozen()
        self.config.check_native()
        slots = list(self.release.slots.values())
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.submit, slots))
        self.assert_frozen()
        self.release.seal()
        with ThreadPoolExecutor(max_workers=self.judge_workers) as pool:
            list(pool.map(self.score, slots))
        self.assert_frozen()
        report = self.summary()
        write_json(self.output / "summary.json", report)
        return report

    def summary(self):
        self.assert_frozen()
        rows, by_benchmark = [], {}
        costs = {"model_calls": 0, "tokens": 0, "unknown_budgets": 0}
        for slot_id, slot in self.release.slots.items():
            submission = self.release._path(slot_id)
            evaluation = self.release._path(slot_id, "evaluations")
            record = read_json(submission) if submission.exists() else {}
            result = read_json(evaluation) if evaluation.exists() else {}
            if record and record.get("slot") != slot:
                raise CheckpointIntegrityError("Submission slot differs from registered task")
            if result and result.get("slot") != slot:
                raise CheckpointIntegrityError("Evaluation slot differs from registered task")
            if record.get("status") == "submitted":
                answer_hash = digest(record["submission"]["answer"])
                if answer_hash != record["submission"]["answer_hash"]:
                    raise CheckpointIntegrityError("Saved submission answer changed")
                if result.get("answer_hash", answer_hash) != answer_hash:
                    raise CheckpointIntegrityError("Evaluation answer differs from sealed submission")
            complete = result.get("complete") is True
            if complete and record.get("status") != "submitted":
                raise CheckpointIntegrityError("Failed generation cannot have a complete evaluation")
            score = result.get("official_score") if complete else None
            if complete and slot["benchmark"] == "writingbench":
                score = result.get("evaluation", {}).get("raw", {}).get("native_mean", score)
            normalized = normalize_score(score, slot["bounds"]) if score is not None else None
            reference_scores = []
            for reference in self.references:
                for previous in reference["rows"]:
                    previous_score = previous.get("native_score", previous.get("score"))
                    if (previous.get("task_id") == slot["task_id"] and previous.get("complete") is True
                            and isinstance(previous_score, (int, float)) and not isinstance(previous_score, bool)):
                        identity = (reference.get("source_identity") or {}).get(slot["benchmark"], {})
                        public = identity.get("public_tasks", {}).get(slot["task_id"], {})
                        previous_hash = previous.get("public_task_hash", public.get("actor_hash"))
                        reference_scores.append({"path": reference["path"], "method": previous.get("method"),
                                                 "score": previous_score,
                                                 "public_task_match": previous_hash == slot["public_task_hash"] if previous_hash else None,
                                                 "configuration_match": reference.get("configuration") == self.registration["configuration"]
                                                                        if reference.get("configuration") else None,
                                                 "delta": score - previous_score if score is not None else None})
            row = {**slot, "generation_status": record.get("status", "pending"),
                   "evaluation_status": result.get("status", "complete" if complete else "pending"),
                   "native_score": score, "normalized_score": normalized, "complete": complete,
                   "generation_error_type": record.get("error_type"),
                   "evaluation_error_type": result.get("error_type"), "reference_scores": reference_scores}
            rows.append(row)
            group = by_benchmark.setdefault(slot["benchmark"], {"slots": 0, "complete": 0, "scores": [], "normalized": []})
            group["slots"] += 1
            group["complete"] += complete
            if score is not None:
                group["scores"].append(score)
                group["normalized"].append(normalized)
            generation = record.get("outcome", {}).get("budget", record.get("budget", {}))
            for budget in (generation, result.get("evaluation_budget", {})):
                costs["model_calls"] += budget.get("model_calls", 0)
                costs["tokens"] += budget.get("tokens", 0)
                costs["unknown_budgets"] += bool(budget.get("usage_unknown") or budget.get("reserved_tokens", 0))
        for name, group in by_benchmark.items():
            scores = group.pop("scores")
            normalized = group.pop("normalized")
            group["native_mean_complete"] = sum(scores) / len(scores) if scores else None
            group["normalized_mean_complete"] = sum(normalized) / len(normalized) if normalized else None
            terminal = all(row["evaluation_status"] != "pending" for row in rows if row["benchmark"] == name)
            group["normalized_mean_failure_zero"] = sum(normalized) / group["slots"] if terminal else None
        report = {"schema": "ours-development-summary-v1", "formal_result": False,
                "registration_hash": self.registration["registration_hash"],
                "partition": self.registration["partition"], "sealed": (self.release.directory / "seal.json").exists(),
                "all_complete": all(row["complete"] for row in rows), "rows": rows,
                "by_benchmark": by_benchmark, "costs": costs, "references": self.references,
                "reference_policy": "Fixed read-only results; inspect recorded configuration/public-task matches before interpreting deltas",
                "output": str(self.output)}
        existing = self.output / "summary.json"
        if existing.exists() and read_json(existing) != report:
            raise CheckpointIntegrityError("Saved iteration summary differs from durable results")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "register", "summary"), default="run")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=ALL_BENCHMARK_NAMES, action="append", required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--partition", choices=("evolution", "validation", "test"), default="evolution")
    parser.add_argument("--limit", type=int, default=2, help="Tasks per benchmark when no explicit IDs are supplied")
    parser.add_argument("--state", type=Path, help="Read-only checkpoint; default is an empty initial snapshot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--max-inflight", type=int, help="Process-wide limit across all actor and judge requests")
    parser.add_argument("--judge-parallel", type=int,
                        help="Concurrent independent rubric judge calls per scored task")
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[], help="BENCHMARK=DIRECTORY override; selected packs are copied unchanged")
    args = parser.parse_args(argv)
    if args.limit < 1 or any(not 1 <= value <= 128 for value in (args.workers, args.judge_workers)):
        parser.error("limit must be positive and worker counts must be between 1 and 128")
    if args.max_inflight is not None and not 1 <= args.max_inflight <= 128:
        parser.error("max-inflight must be between 1 and 128")
    if args.judge_parallel is not None and not 1 <= args.judge_parallel <= 64:
        parser.error("judge-parallel must be between 1 and 64")
    if len(args.benchmark) != len(set(args.benchmark)):
        parser.error("benchmarks must be unique")
    if args.mode == "summary" and not (args.output / "registration.json").is_file():
        parser.error("summary requires an existing registered iteration")
    load_dotenv()
    with coordinator_lock(args.output):
        iteration = OursIteration(args)
        report = iteration.run() if args.mode == "run" else iteration.summary()
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
