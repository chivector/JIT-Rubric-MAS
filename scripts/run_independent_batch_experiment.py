"""Execute independent 40-task trajectories with frozen five-task batches."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import time

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.experience import ExperienceStore
from jit_mas.independent_batch_campaign import IndependentBatchCampaign
from jit_mas.independent_batch_protocol import SOURCES, RUN_IDS, build_protocol
from jit_mas.independent_campaign import TERMINAL, coordinator_lock
from jit_mas.pipeline import write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from scripts.eval.config import load_dotenv
from scripts.run_independent_experiment import IndependentEnvironment, _read, validation_result


def batch_budget(reflections, candidates, *, interrupted=False):
    budgets = [row.get("budget") for row in [*reflections, *candidates]]
    known = [budget for budget in budgets if isinstance(budget, dict)]
    return {"records": [record for budget in known for record in budget.get("records", [])],
            "tokens": sum(budget.get("tokens", 0) for budget in known),
            "model_calls": sum(budget.get("model_calls", 0) for budget in known),
            "reserved_tokens": sum(budget.get("reserved_tokens", 0) for budget in known),
            "usage_unknown": interrupted or len(known) != len(budgets)
                             or any(budget.get("usage_unknown") for budget in known), "cost": None}


class IndependentBatchEnvironment(IndependentEnvironment):
    protocol_builder = staticmethod(build_protocol)
    runner_path = Path(__file__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity["selection"] = "five frozen EVO tasks; three same-base candidates; fixed VAL winner before next batch"
        self.identity["transport"] = self._transport_identity()

    @staticmethod
    def _transport_identity():
        return {"model_attempts": max(1, int(os.getenv("JIT_MAS_MODEL_ATTEMPTS", "1"))),
                "disable_keepalive": os.getenv("JIT_MAS_DISABLE_KEEPALIVE", "0") == "1"}

    def assert_frozen(self):
        super().assert_frozen()
        if self.identity["transport"] != self._transport_identity():
            raise CheckpointIntegrityError("Frozen transport retry or connection policy changed")

    def initialize(self):
        self.campaign = IndependentBatchCampaign(self.output, self.protocol, self.identity, self.bounds)
        sources = tuple(self.protocol.get("sources", SOURCES))
        run_ids = tuple(self.protocol.get("run_ids", RUN_IDS))
        for source in sources:
            for run_id in run_ids:
                self._split(source, run_id)
                if self.campaign.checkpoint(source, run_id, 0) is None:
                    self.campaign.freeze_checkpoint(source, run_id, 0, ExperienceSnapshot())

    def _claim_state(self, claim):
        if claim["kind"] == "validation":
            candidate = self.campaign.candidate(claim["source"], claim["run_id"],
                                                claim["batch_index"], claim["candidate_index"])
            snapshot = ExperienceSnapshot.model_validate(candidate["snapshot"])
        else:
            snapshot = self.campaign.checkpoint(claim["source"], claim["run_id"], claim["batch_index"] * 5)["snapshot"]
        if digest(snapshot) != claim["state_hash"]:
            raise CheckpointIntegrityError("Claim no longer matches its frozen input state")
        return snapshot

    def _batch_artifacts(self, claim):
        root = self.slot_directory(claim)
        aggregate = _read(root / "batch_evolution.json") if (root / "batch_evolution.json").is_file() else None
        tasks = self.campaign.trajectories[claim["source"], claim["run_id"]]["batches"][claim["batch_index"]]["evolution_task_ids"]
        if aggregate and (aggregate.get("batch_id") != claim["task_id"]
                or aggregate.get("base_state_hash") != claim["state_hash"]
                or aggregate.get("source_task_ids") != tasks or aggregate.get("no_intra_batch_evolution") is not True):
            raise CheckpointIntegrityError("Durable batch artifact changed its frozen identity")
        reflections = aggregate["reflections"] if aggregate else [_read(path) for path in sorted(root.glob("reflections/*/reflection.json"))]
        candidates = []
        for index in range(3):
            recorded = self.campaign.candidate(claim["source"], claim["run_id"], claim["batch_index"], index)
            path = root / f"candidate{index}.json"
            candidate = recorded or (_read(path) if path.is_file() else None)
            if candidate is None and aggregate:
                candidate = next((row for row in aggregate["candidates"] if row["candidate_index"] == index), None)
            if candidate is not None:
                if (candidate.get("candidate_index") != index
                        or candidate.get("base_state_hash") != claim["state_hash"]
                        or candidate.get("source_task_ids") != tasks):
                    raise CheckpointIntegrityError("Durable candidate changed its batch source or input state")
            else:
                candidate = {"candidate_index": index, "status": "failed", "snapshot": None,
                             "state_hash": None, "budget": None, "error_type": "InterruptedBatchEvolution",
                             "base_state_hash": claim["state_hash"], "source_task_ids": tasks}
            candidates.append(candidate)
        return reflections, candidates, aggregate is None

    def _batch_result(self, claim, reflections, candidates, *, interrupted=False):
        return {"task_id": claim["task_id"], "candidates": [
                    {key: value for key, value in candidate.items() if key != "snapshot"} for candidate in candidates],
                "budget": batch_budget(reflections, candidates, interrupted=interrupted),
                "candidate_budgets": [candidate.get("budget") for candidate in candidates],
                "reflection_budgets": [reflection.get("budget") for reflection in reflections]}

    def execute(self, claim):
        self.assert_frozen()
        start = time.monotonic()
        source, run_id = claim["source"], claim["run_id"]
        snapshot = self._claim_state(claim)
        store = self._frozen_store({"snapshot": snapshot, "state_hash": claim["state_hash"]})
        try:
            pipeline = self._pipeline(source, run_id, store, self.slot_directory(claim))
            if claim["kind"] == "batch_evolution":
                from jit_mas.batch_evolution import evolve_batch

                batch_runs = [row for row in self.campaign.records(kind="evolution", source=source, run_id=run_id)
                              if row["batch_index"] == claim["batch_index"]]
                tasks = [{"task_id": row["task_id"], "status": row["status"],
                          "run_dir": row["result"].get("outcome", {}).get("run_dir"),
                          "input_state_hash": row["result"]["state_hash"],
                          "after_state_hash": row["result"]["after_snapshot_hash"]} for row in batch_runs]
                candidates = evolve_batch(pipeline, snapshot, tasks, self.slot_directory(claim),
                                          batch_id=claim["task_id"])
                self.campaign.freeze_candidates(claim, candidates)
                reflections, candidates, interrupted = self._batch_artifacts(claim)
                result = self._batch_result(claim, reflections, candidates, interrupted=interrupted)
                status = "complete"
            else:
                mode = "evolve" if claim["kind"] == "evolution" else "validate"
                outcome = pipeline.run_task(claim["task_id"], snapshot, mode=mode, attribution=False, repeat=0)
                if digest(store.snapshot()) != claim["state_hash"] or digest(snapshot) != claim["state_hash"]:
                    raise CheckpointIntegrityError("Task execution mutated its frozen batch state")
                result = validation_result(source, outcome)
                if claim["kind"] == "evolution":
                    result["after_snapshot_hash"] = claim["state_hash"]
                status = "complete" if result["complete"] else "incomplete"
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            failure = getattr(error, "jit_mas_run_failure", {})
            result = {"task_id": claim["task_id"], "complete": False, "score": None,
                      "error_type": type(error).__name__, "error": str(error), "failure": failure,
                      "budget": failure.get("budget"), "evaluation": None}
            status = "failed"
            if claim["kind"] == "evolution":
                result["after_snapshot_hash"] = claim["state_hash"]
            elif claim["kind"] == "batch_evolution":
                reflections, candidates, interrupted = self._batch_artifacts(claim)
                self.campaign.freeze_candidates(claim, candidates)
                result.update(self._batch_result(claim, reflections, candidates, interrupted=interrupted))
        finally:
            store.close()
        result.update(state_hash=claim["state_hash"], execution_seconds=time.monotonic() - start,
                      registration_hash=self.campaign.registration_hash)
        self.assert_frozen()
        self.campaign.finish(claim, result, status=status)
        return {"slot_id": claim["slot_id"], "status": status}

    def recover(self):
        self.assert_frozen()
        for claim in self.campaign.interrupted_claims():
            if claim["kind"] == "test":
                continue
            outcome = self._durable_outcome(claim) if claim["kind"] != "batch_evolution" else None
            if outcome:
                result = validation_result(claim["source"], outcome)
                status = "complete" if result["complete"] else "incomplete"
            else:
                result, status = self._interrupted_failure(claim), "failed"
            if claim["kind"] == "evolution":
                result["after_snapshot_hash"] = claim["state_hash"]
            if claim["kind"] == "batch_evolution":
                reflections, candidates, interrupted = self._batch_artifacts(claim)
                self.campaign.freeze_candidates(claim, candidates)
                result = self._batch_result(claim, reflections, candidates, interrupted=interrupted)
                status = "complete" if any(candidate.get("snapshot") is not None for candidate in candidates) else "failed"
            result.update(state_hash=claim["state_hash"], recovered=True, registration_hash=self.campaign.registration_hash)
            self.campaign.finish(claim, result, status=status)
        self.campaign.advance()

    def run(self, *, phase="full"):
        from scripts.run_independent_test_release import IndependentTestReleaseRunner

        self.initialize()
        release = IndependentTestReleaseRunner(self.launch_path, self.output, environment=self)
        with coordinator_lock(self.output):
            self.recover()
            release._recover_interrupted_tests()
            with ThreadPoolExecutor(max_workers=self.workers) as workers:
                active = {}
                while True:
                    self.campaign.advance()
                    while len(active) < self.workers:
                        claim = self.campaign.claim("batch-worker", phase=phase)
                        if claim is None and phase == "full":
                            claim = self.campaign.claim("test-worker", kinds=("test",), prefer="test")
                        if claim is None:
                            break
                        callback = release._submit_claim if claim["kind"] == "test" else self.execute
                        active[workers.submit(callback, claim)] = claim
                    report = self.campaign.status(phase=phase)
                    report.update(active=len(active), updated_at=utc_now(), test_feedback_released=False)
                    write_json(self.output / "status.json", report)
                    if not active:
                        break
                    finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                        active.pop(future)
            self.campaign.advance()
            relevant = [row for row in self.campaign.records() if row["kind"] != "test"]
            if phase == "first-stage":
                relevant = [row for row in relevant if row["run_id"] == 0 and row["batch_index"] == 0]
            if any(row["status"] not in TERMINAL for row in relevant):
                raise RuntimeError("No runnable work remains but the batch dependency inventory is unfinished")
            report = self.campaign.status(phase=phase)
            report.update(evolution_validation_complete=True, test_feedback_released=False)
            write_json(self.output / f"{phase}_report.json", report)
        if phase == "full":
            release.seal()
            report["test"] = release.score(workers=self.workers)
            from scripts.summarize_independent_paper import summarize

            from scripts.summarize_independent_paper import markdown

            paper = summarize(self.output)
            paper_dir = self.output.parent / f"{self.output.name}_paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            write_json(paper_dir / "paper_summary.json", paper)
            (paper_dir / "paper_summary.md").write_text(markdown(paper), encoding="utf-8")
            report["paper"] = {"directory": str(paper_dir), "registration_sha256": paper["registration_sha256"]}
            write_json(self.output / "full_report.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run", "status"), required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("first-stage", "full"), default="full")
    args = parser.parse_args(argv)
    load_dotenv()
    environment = IndependentBatchEnvironment(args.launch_config, args.output, require_preflight=args.mode == "run")
    if args.mode == "run":
        result = environment.run(phase=args.phase)
    elif args.mode == "status":
        environment.initialize()
        result = environment.campaign.status(phase=args.phase)
    else:
        workload = environment.protocol.get("workload", {})
        result = {"inputs_ready": True, "model_calls": 0, "protocol_sha256": environment.protocol["protocol_sha256"],
                  "task_slots": workload.get("total_task_slots"),
                  "batch_evolution_operations": workload.get("trajectories", 0) * 8,
                  "candidate_generation_operations": workload.get("trajectories", 0) * 24}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
