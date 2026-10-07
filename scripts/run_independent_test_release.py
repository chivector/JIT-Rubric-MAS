"""Run the independent v5 TEST artifact release and deferred scoring.

TEST generation is source-aware: a selected source checkpoint is used only for
that source/run, while static methods use the empty initial snapshot.  Every
submission is committed to :class:`IndependentCampaign`; judging is available
only after the campaign's complete 2,751-slot seal.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import threading

from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.experience import ExperienceStore
from jit_mas.independent_campaign import TERMINAL, coordinator_lock
from jit_mas.independent_protocol import SOURCES
from jit_mas.pipeline import write_json
from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest, utc_now
from jit_mas.token_usage import merge_usage, summarize_budget, summarize_outcome
from scripts.mas_baseline_methods import run_direct
from scripts.run_benchmark_experiment import file_hash
from scripts.run_independent_experiment import IndependentEnvironment
from scripts.run_jit_mas import make_pipeline
from jit_mas.test_release import score_with_pipeline


MAX_WORKERS = 64


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class IndependentTestReleaseRunner:
    """Coordinator for independent unscored TEST submission and scoring."""

    def __init__(self, launch_config, output, *, require_preflight=True, environment=None):
        self.environment = environment or IndependentEnvironment(launch_config, output,
                                                                 require_preflight=require_preflight)
        self.environment.initialize()
        self.campaign = self.environment.campaign
        self.output = Path(output).resolve()
        self._state_lock = threading.Lock()
        self._split_lock = threading.Lock()
        self._identity_path = self.output / "test_adapter_identity.json"
        self._assert_adapter_identity(create=True)

    def _adapter_identity(self):
        return {
            "schema": "independent-test-adapter-identity-v1",
            "adapter_sha256": file_hash(Path(__file__)),
            "launch_file_sha256": self.environment.identity["launch_file_sha256"],
            "configuration_file_sha256": self.environment.identity["configuration_file_sha256"],
            "joint_manifest_sha256": self.environment.identity["joint_manifest_sha256"],
            "evolution_runner_sha256": self.environment.identity["runner_sha256"],
            "protocol_sha256": self.campaign.protocol["protocol_sha256"],
        }

    def _assert_adapter_identity(self, *, create=False):
        identity = self._adapter_identity()
        if self._identity_path.exists():
            recorded = _read(self._identity_path)
            if recorded != identity:
                raise CheckpointIntegrityError("Independent TEST adapter identity changed")
        elif create:
            write_json(self._identity_path, identity)
        else:
            raise CheckpointIntegrityError("Independent TEST adapter identity is missing")
        return identity

    def _recover_interrupted_tests(self):
        for claim in self.campaign.interrupted_claims():
            if claim["kind"] != "test":
                continue
            slot_root = self.output / "test_runs" / digest(claim["slot_id"])
            completed = list(slot_root.glob("**/complete.json"))
            if len(completed) > 1:
                raise CheckpointIntegrityError("Interrupted TEST slot has multiple durable outcomes")
            if completed:
                outcome = _read(completed[0])
                if (outcome.get("task_id") != claim["task_id"]
                        or outcome.get("experience_hash") != claim["state_hash"]
                        or outcome.get("status") != "submitted_unscored"
                        or outcome.get("evaluation") is not None
                        or outcome.get("experience_updates") or outcome.get("proposals")):
                    raise CheckpointIntegrityError("Interrupted TEST outcome changed its frozen condition")
                path = completed[0].parent / "submission.json"
                self.campaign.submit_test(claim, path, answer_hash=outcome["answer_hash"],
                                          budget=outcome.get("budget"),
                                          metadata={"recovered": True,
                                                    "token_usage": summarize_outcome(outcome)})
                continue
            budgets = list(slot_root.glob("**/budget.json"))
            if len(budgets) > 1:
                raise CheckpointIntegrityError("Interrupted TEST slot has multiple attempt budgets")
            budget = _read(budgets[0]) if budgets else {"usage_unknown": True}
            result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"],
                      "complete": False, "score": None, "evaluation": None,
                      "error_type": "InterruptedWithoutDurableOutcome", "budget": budget,
                      "retry_policy": "Consumed original slot; no task resampling",
                      "recovered": True, "finished_at": utc_now()}
            self.campaign.finish(claim, result, status="failed")

    def _select_ready_trajectories(self):
        sources = tuple(self.campaign.protocol.get("sources", SOURCES))
        run_ids = tuple(self.campaign.protocol.get("run_ids", sorted({row["run_id"] for row in self.campaign.protocol["trajectories"]})))
        for source in sources:
            for run_id in run_ids:
                if self.campaign.selection(source, run_id) is not None:
                    continue
                try:
                    self.campaign.select_trajectory(source, run_id)
                except ValueError:
                    records = self.campaign.records(source=source, run_id=run_id)
                    relevant = [row for row in records if row["kind"] in {"evolution", "validation"}]
                    if relevant and all(row["status"] in TERMINAL for row in relevant):
                        raise

    def _test_tasks(self, target):
        sources = tuple(self.campaign.protocol.get("sources", SOURCES))
        if target in sources:
            run_id = next(iter(self.campaign.protocol.get("run_ids", (0,))))
            return list(self.campaign.trajectories[target, run_id]["test_task_ids"])
        return list(self.campaign.protocol["target_test_tasks"][target])

    def _split_path(self, target):
        path = self.output / "test_runtime_splits" / f"{target}.json"
        # ``make_pipeline`` validates runtime splits with the strict
        # ``SplitManifest`` schema.  Keep the adapter-specific wrapper outside
        # that schema; putting ``version``/``benchmark`` beside the split lists
        # makes Pydantic reject every TEST slot before any model call.
        manifest = SplitManifest(seed=0, evolution=[], validation=[],
                                 test=self._test_tasks(target), stream=[])
        document = {"runtime_split_manifest": manifest.model_dump(mode="json")}
        with self._split_lock:
            if path.exists() and _read(path) != document:
                raise CheckpointIntegrityError("Independent TEST runtime split changed")
            if not path.exists():
                write_json(path, document)
        return path

    def _ready_state(self, slot):
        if slot["source"] == "static":
            return ExperienceSnapshot()
        selection = self.campaign.selection(slot["source"], slot["run_id"])
        if not selection or not selection.get("selected"):
            return None
        selected = selection["selected"]
        checkpoint = self.campaign.checkpoint(slot["source"], slot["run_id"], selected["position"])
        if checkpoint["state_hash"] != selected["state_hash"]:
            raise CheckpointIntegrityError("Selected checkpoint hash changed")
        expected = slot.get("state_hash") or (slot.get("result") or {}).get("state_hash")
        if expected and checkpoint["state_hash"] != expected:
            raise CheckpointIntegrityError("TEST slot state differs from selected source state")
        allowed = set(self.campaign.trajectories[slot["source"], slot["run_id"]]["evolution_task_ids"])
        snapshot = checkpoint["snapshot"]
        observed = {source for entry in snapshot.experiences for source in entry.source_task_ids}
        profiles = list(snapshot.agent_pool.profiles)
        for operation in snapshot.agent_pool.structural_history:
            observed.add(operation.source_task_id)
            observed.update(operation.source_task_ids)
            profiles.extend(operation.profiles)
        for profile in profiles:
            observed.update(profile.source_task_ids)
            observed.update(source for lesson in profile.memory for source in lesson.source_task_ids)
        observed.update(observation.task_id for observation in snapshot.agent_pool.observations)
        if not observed <= allowed:
            raise CheckpointIntegrityError("TEST state contains history outside its source EVO allowlist")
        return snapshot

    def _pipeline(self, slot, snapshot):
        target = slot["target"]
        material = self.environment.materials[target]
        state_path = self.output / "test_states" / f"{digest(snapshot)}.sqlite"
        with self._state_lock:
            frozen = snapshot_store(snapshot, state_path) if not state_path.exists() else None
        if frozen is not None:
            frozen.close()
        store = ExperienceStore(state_path, read_only=True)
        if digest(store.snapshot()) != digest(snapshot):
            store.close()
            raise CheckpointIntegrityError("TEST state snapshot changed")
        output = self.output / "test_runs" / digest(slot["slot_id"])
        try:
            pipeline = make_pipeline(self.environment.config, store, output,
                                     data=material["data"], splits=self._split_path(target),
                                     benchmark=target, evidence_dir=material.get("evidence_dir"),
                                     checker_source_root=material.get("checker_source_root"))
        except BaseException:
            store.close()
            raise
        return pipeline, store, output

    def _submit_one(self, slot):
        snapshot = self._ready_state(slot)
        if snapshot is None:
            raise ValueError("TEST slot has no selected source state")
        if digest(snapshot) != slot["state_hash"]:
            raise CheckpointIntegrityError("TEST snapshot does not match inventory state")
        pipeline, store, output = self._pipeline(slot, snapshot)
        try:
            if slot["method"] == "direct":
                config = self.environment.config
                outcome = run_direct(pipeline.tasks[slot["task_id"]], None, pipeline.models,
                                     pipeline.evaluator_factory, output / "direct",
                                     {"max_calls": config.max_model_calls,
                                      "max_tokens": config.max_total_tokens,
                                      "max_tool_calls": config.max_tool_calls,
                                      "output_tokens": min(8192, config.models["exec"].max_tokens),
                                      "timeout_seconds": config.task_timeout},
                                     defer_evaluation=True)
            elif slot["method"] in {"jit_matched", "rubric_fixed"}:
                from jit_mas.experiment_methods import submit_method
                outcome = submit_method(pipeline, slot["task_id"], store.snapshot(),
                                        method=slot["method"], repeat=0, output_dir=output,
                                        shared_dir=self.output / "shared_rstar")
            else:
                outcome = pipeline.run_task(slot["task_id"], store.snapshot(), mode="evaluate",
                                            repeat=0, attribution=False,
                                            defer_evaluation=True, resume=False)
            if slot["method"] == "direct":
                outcome["experience_hash"] = digest(snapshot)
            if (outcome.get("experience_hash") != digest(snapshot)
                    or outcome.get("evaluation") is not None
                    or outcome.get("experience_updates") or outcome.get("proposals")):
                raise CheckpointIntegrityError("TEST outcome changed its frozen unscored condition")
            outcome["token_usage"] = summarize_outcome(outcome)
            run_dir = Path(outcome["run_dir"])
            submission = _read(run_dir / "submission.json")
            write_json(run_dir / "complete.json", outcome)
            return outcome, run_dir / "submission.json", submission
        finally:
            store.close()

    def _submit_claim(self, claim):
        try:
            outcome, path, submission = self._submit_one(claim)
            self.campaign.submit_test(claim, path, answer_hash=submission["answer_hash"],
                                      budget=outcome.get("budget"),
                                      metadata={"method": claim["method"],
                                                "target": claim["target"],
                                                "token_usage": outcome.get("token_usage")})
            return "submitted"
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            failure = getattr(error, "jit_mas_run_failure", {})
            result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"],
                      "complete": False, "score": None, "evaluation": None,
                      "error_type": type(error).__name__,
                      "budget": failure.get("budget"), "finished_at": utc_now()}
            self.campaign.finish(claim, result, status="failed")
            return "failed"

    def submit(self, *, workers=1):
        if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"workers must be an integer between 1 and {MAX_WORKERS}")
        with coordinator_lock(self.output):
            self._assert_adapter_identity()
            self._recover_interrupted_tests()
            self._select_ready_trajectories()
            states = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                active = set()
                while True:
                    while len(active) < workers:
                        claim = self.campaign.claim("test-worker", kinds=("test",), prefer="test")
                        if claim is None:
                            break
                        active.add(pool.submit(self._submit_claim, claim))
                    if not active:
                        break
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in finished:
                        states.append(future.result())
                        active.remove(future)
            self._select_ready_trajectories()
        test_records = self.campaign.records(kind="test")
        terminal = sum(row["status"] in TERMINAL for row in test_records)
        test_count = len(test_records)
        return {"submitted": states.count("submitted"), "failed": states.count("failed"),
                "claimed": len(states), "test_slots": len(test_records),
                "terminal": terminal, "sealed": terminal == len(test_records) == test_count}

    def seal(self):
        self._assert_adapter_identity()
        return self.campaign.seal_test_inventory()

    def _evaluation_path(self, slot_id):
        return self.output / "test_evaluations" / f"{digest(slot_id)}.json"

    def _score_one(self, row):
        path = self._evaluation_path(row["slot_id"])
        if path.exists():
            result = _read(path)
            expected = result.get("receipt_sha256")
            body = {key: value for key, value in result.items() if key != "receipt_sha256"}
            if (result.get("slot") != row or expected != digest(body)
                    or result.get("submission_sha256") != row["result"].get("submission_hash")):
                raise CheckpointIntegrityError("Independent TEST evaluation receipt changed")
            return result
        started = self.output / "test_evaluation_started" / f"{digest(row['slot_id'])}.json"
        if started.exists():
            raise RuntimeError("Interrupted TEST judgment cannot be silently resampled")
        write_json(started, {"slot_result_sha256": row["result_hash"], "started_at": utc_now()})
        store = None
        try:
            submission = _read(row["result"]["submission_path"])
            snapshot = self._ready_state(row)
            pipeline, store, _ = self._pipeline(row, snapshot)
            record = {"slot": row, "submission": submission,
                      "outcome": {"budget": row["result"].get("budget") or {}}}
            result = score_with_pipeline(pipeline, record)
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            result = {"slot": row, "official_score": None, "complete": False,
                      "status": "evaluation_failed", "error_type": type(error).__name__,
                      "generation_budget": row["result"].get("budget"),
                      "evaluation_budget": getattr(error, "evaluation_budget", {"usage_unknown": True})}
        finally:
            if store is not None:
                store.close()
        result["submission_sha256"] = row["result"]["submission_hash"]
        result["token_usage"] = summarize_outcome(result)
        result["receipt_sha256"] = digest(result)
        write_json(path, result)
        return result

    def score(self, *, workers=1):
        if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"workers must be an integer between 1 and {MAX_WORKERS}")
        self._assert_adapter_identity()
        self.campaign.require_test_seal()
        test_rows = self.campaign.records(kind="test")
        claims = [row for row in test_rows if row["status"] == "submitted"]

        with coordinator_lock(self.output):
            if workers == 1:
                results = [self._score_one(row) for row in claims]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(self._score_one, claims))
        by_condition = {}
        for row in test_rows:
            if row["status"] == "submitted":
                continue
            slot = row
            key = (slot["source"], slot["run_id"], slot["target"], slot["method"])
            summary = by_condition.setdefault(
                key, {"source": slot["source"], "run_id": slot["run_id"],
                      "target": slot["target"], "method": slot["method"],
                      "scores": [], "complete": 0, "slots": 0})
            summary["slots"] += 1
        for result in results:
            slot = result["slot"]
            key = (slot["source"], slot["run_id"], slot["target"], slot["method"])
            row = by_condition.setdefault(key, {"source": slot["source"], "run_id": slot["run_id"],
                                                "target": slot["target"], "method": slot["method"],
                                                "scores": [], "complete": 0, "slots": 0})
            row["slots"] += 1
            if result.get("complete"):
                row["complete"] += 1
                row["scores"].append(result.get("official_score"))
        for row in by_condition.values():
            row["mean_official_score"] = (sum(row["scores"]) / row["slots"]
                                           if row["complete"] == row["slots"] else None)
        complete = sum(bool(r.get("complete")) for r in results)
        report_schema = ("independent-test-report-v7" if self.campaign.protocol.get("version", "").endswith("-v7")
                         else "independent-test-report-v5")
        report = {"schema": report_schema, "slots": len(test_rows),
                  "evaluated": len(test_rows), "complete": complete,
                  "incomplete_or_failed": len(test_rows) - complete,
                  "token_usage": merge_usage([
                      *[summarize_outcome(result) for result in results],
                      *[summarize_budget(row["result"].get("budget")) for row in test_rows
                        if row["status"] != "submitted"]]),
                  "by_condition": list(by_condition.values()), "test_feedback_released": True}
        report_path = self.output / "test_report.json"
        if report_path.exists() and _read(report_path) != report:
            raise CheckpointIntegrityError("Independent TEST report changed")
        if not report_path.exists():
            write_json(report_path, report)
        return report

    def status(self):
        self._assert_adapter_identity()
        return self.campaign.status()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("submit", "seal", "score", "status"), required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        runner = IndependentTestReleaseRunner(args.launch_config, args.output,
                                              require_preflight=args.mode != "status")
        result = (runner.submit(workers=args.workers) if args.mode == "submit" else
                  runner.seal() if args.mode == "seal" else
                  runner.score(workers=args.workers) if args.mode == "score" else runner.status())
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError, CheckpointIntegrityError) as error:
        parser.exit(2, f"Independent TEST release: {error}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
