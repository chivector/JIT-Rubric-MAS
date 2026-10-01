"""Run independent v5 trajectories with asynchronous immutable validation."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import threading
import time

import yaml

from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.independent_campaign import IndependentCampaign, TERMINAL, coordinator_lock
from jit_mas.independent_protocol import (
    BENCHMARKS, CHECKPOINT_POSITIONS, RUN_IDS, SOURCES, build_protocol,
)
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import ExperienceSnapshot, digest
from scripts.eval.config import load_dotenv
from scripts.env_config import resolve_env_placeholders
from scripts.run_benchmark_experiment import evidence_identity, file_hash
from scripts.run_jit_mas import make_pipeline


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(value, base):
    location = Path(value)
    return location.resolve() if location.is_absolute() else (base / location).resolve()


def validation_result(source, outcome):
    feedback = outcome.get("evaluation", {})
    score = feedback.get("score")
    if source == "writingbench":
        score = feedback.get("raw", {}).get("native_mean")
    return {"task_id": outcome["task_id"], "score": score,
            "complete": feedback.get("complete") is True, "outcome": outcome,
            "budget": outcome.get("budget"), "experience_updates": outcome.get("experience_updates", [])}


class IndependentEnvironment:
    """Validate inputs before requests and create isolated per-slot pipelines."""

    def __init__(self, launch_path, output, *, require_preflight=True):
        self.launch_path = Path(launch_path).resolve()
        self.output = Path(output).resolve()
        self.document = _read(self.launch_path)
        base = self.launch_path.parent
        self.config_path = _path(self.document["configuration"], base)
        self.joint_path = _path(self.document.get("joint_manifest", "../../paper/experiments/joint_task_splits_v5.json"), base)
        raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.config = MASConfig.model_validate(resolve_env_placeholders(raw_config))
        self.joint = _read(self.joint_path)
        self.protocol = build_protocol(self.joint)
        self.workers = self.document.get("workers", 2)
        if type(self.workers) is not int or self.workers < 2:
            raise ValueError("Independent scheduling requires at least two task workers")
        self._validate_configuration()
        from jit_mas.benchmarks import load_benchmark

        self.datasets, self.materials, self.bounds = {}, {}, {}
        registrations = self.document.get("benchmarks", {})
        if set(registrations) != set(BENCHMARKS):
            raise ValueError("Launch must bind the complete six-benchmark inventory")
        for name in BENCHMARKS:
            registered = registrations[name]
            data = _path(registered["data"], base)
            dataset = load_benchmark(name, data, available_tools=self.config.available_tools)
            if dataset.dataset_sha256 != self.joint["benchmarks"][name]["dataset_sha256"]:
                raise CheckpointIntegrityError("Dataset bytes differ from the pinned independent protocol")
            membership = self.joint["memberships"][name]
            selected = {task_id for partition in ("evolution", "validation", "test")
                        for task_id in membership.get(partition, [])}
            if not selected <= set(dataset.tasks):
                raise ValueError("Frozen benchmark membership refers to unavailable tasks")
            evidence = _path(registered["evidence_dir"], base) if registered.get("evidence_dir") else None
            evidence_hash = None
            if name in {"researchrubrics", "deepsearchqa", "deepresearch_bench_ii"}:
                if evidence is None:
                    raise ValueError("Research benchmarks require frozen shared public evidence")
                evidence_hash = evidence_identity(evidence)
            if evidence is not None:
                from jit_mas.evidence import load_evidence_tasks

                load_evidence_tasks({task_id: dataset.tasks[task_id] for task_id in selected},
                                    evidence, expected_count=len(selected))
                evidence_hash = evidence_hash or evidence_identity(evidence)
            self.datasets[name] = dataset
            self.materials[name] = {"data": str(data), "evidence_dir": str(evidence) if evidence else None,
                                    "dataset_sha256": dataset.dataset_sha256, "evidence": evidence_hash}
            if name in SOURCES:
                for task_id in membership["validation"]:
                    self.bounds[task_id] = ([1.0, 10.0] if name == "writingbench" else
                                            [dataset.lower_bounds[task_id], dataset.upper_bounds[task_id]])
        self.served_models = self.document.get("served_models", {})
        self.preflight = self.document.get("preflight", {})
        if require_preflight:
            self._validate_preflight(base)
            self.config.check_native()
        self.identity = {
            "software_test_only": False, "configuration": self.config.model_dump(mode="json"),
            "configuration_file_sha256": file_hash(self.config_path),
            "launch_file_sha256": file_hash(self.launch_path), "code_fingerprint": code_fingerprint(),
            "runner_sha256": file_hash(__file__), "joint_manifest_sha256": file_hash(self.joint_path),
            "materials": self.materials, "served_models": self.served_models,
            "preflight": self.preflight, "workers": self.workers,
            "initial_snapshot_hash": digest(ExperienceSnapshot()),
            "retry_policy": "recover durable outcome; unresolved started slots fail without resampling",
            "selection": "all 20 EVO and 50 VAL terminal before source-only normalized selection",
        }
        self.campaign = None
        self._state_lock = threading.Lock()

    def _validate_configuration(self):
        config = self.config
        if config.backend != "native_jit" or config.execution_mode != "iterative_shared_ledger":
            raise ValueError("Independent v5 requires native iterative shared-ledger execution")
        if any(value is not None for value in (config.team_max_calls, config.max_model_calls, config.max_tool_calls)):
            raise ValueError("Independent v5 forbids fixed role/team/task model or tool call caps")
        if not all((config.persistent_experience, config.evolving_agent_pool, config.explicit_rubrics,
                    config.local_planning, config.local_attribution)) or config.fixed_team is not None:
            raise ValueError("Independent trajectories require the complete evolving Meta-Agent and Agent Pool")
        if config.available_tools:
            raise ValueError("This frozen-evidence launch does not permit actor external tools")
        if not config.unsafe_local:
            raise ValueError("Native harness execution must be explicitly configured with unsafe_local")

    def _validate_preflight(self, base):
        if self.preflight.get("passed") is not True or not self.preflight.get("report_path"):
            raise ValueError("Formal execution requires a completed synthetic preflight report")
        report_path = _path(self.preflight["report_path"], base)
        report = _read(report_path)
        file_digest = file_hash(report_path)
        content_digest = digest({key: value for key, value in report.items() if key != "report_sha256"})
        expected_digest = self.preflight.get("report_sha256")
        if expected_digest not in {file_digest, content_digest}:
            raise CheckpointIntegrityError("Frozen synthetic preflight report hash mismatch")
        if report.get("report_sha256") and report["report_sha256"] != content_digest:
            raise CheckpointIntegrityError("Synthetic preflight self-hash mismatch")
        if report.get("passed") is not True or report.get("synthetic_only") is not True:
            raise ValueError("Preflight must pass on synthetic tasks without experimental data")
        if report.get("configuration_sha256") != digest(self.config.model_dump(mode="json")):
            raise CheckpointIntegrityError("Preflight was performed under a different execution configuration")
        if report.get("code_fingerprint") != code_fingerprint():
            raise CheckpointIntegrityError("Preflight was performed under a different code identity")
        if set(self.served_models) != {"meta", "global", "local", "exec", "judge"}:
            raise ValueError("Launch must freeze observed serving identity for every model role")
        for role, spec in self.config.models.items():
            observed = self.served_models.get(role, {})
            if observed.get("requested_model") != spec.model or not observed.get("returned_model"):
                raise ValueError("Every model role requires its requested and observed served model identity")
            if observed.get("synthetic") is True:
                raise ValueError("Formal execution requires provider-observed model identities, not synthetic preflight identities")

    def assert_frozen(self):
        changed = (file_hash(self.launch_path) != self.identity["launch_file_sha256"]
                   or file_hash(self.config_path) != self.identity["configuration_file_sha256"]
                   or file_hash(self.joint_path) != self.identity["joint_manifest_sha256"]
                   or code_fingerprint() != self.identity["code_fingerprint"]
                   or file_hash(__file__) != self.identity["runner_sha256"])
        for material in self.materials.values():
            changed = changed or file_hash(material["data"]) != material["dataset_sha256"]
            if material["evidence_dir"]:
                changed = changed or evidence_identity(material["evidence_dir"]) != material["evidence"]
        if changed:
            raise CheckpointIntegrityError("Frozen campaign code, configuration, data or evidence changed")

    def trajectory_directory(self, source, run_id):
        return self.output / "trajectories" / source / f"run{run_id}"

    def slot_directory(self, claim):
        return self.output / "slots" / digest(claim["slot_id"])

    def _split(self, source, run_id):
        trajectory = self.campaign.trajectories[source, run_id]
        path = self.trajectory_directory(source, run_id) / "runtime_split.json"
        document = {"seed": trajectory["order_seed"], "evolution": trajectory["evolution_task_ids"],
                    "validation": trajectory["validation_task_ids"], "test": trajectory["test_task_ids"], "stream": []}
        if path.exists() and _read(path) != document:
            raise CheckpointIntegrityError("Runtime trajectory membership changed")
        if not path.exists():
            write_json(path, document)
        return path

    def initialize(self):
        self.campaign = IndependentCampaign(self.output, self.protocol, self.identity, self.bounds)
        for source in SOURCES:
            for run_id in RUN_IDS:
                self._split(source, run_id)
                state_path = self.trajectory_directory(source, run_id) / "experience.sqlite"
                store = ExperienceStore(state_path)
                try:
                    recorded = self.campaign.records(kind="evolution", source=source, run_id=run_id)
                    started = any(row["status"] != "pending" for row in recorded)
                    if not started and (store.snapshot().version or store.snapshot().experiences
                                        or store.snapshot().agent_pool.profiles):
                        raise CheckpointIntegrityError("A new trajectory must begin from an empty experience store")
                    if self.campaign.checkpoint(source, run_id, 0) is None:
                        if started:
                            raise CheckpointIntegrityError("Started trajectory is missing its frozen initial state")
                        self.campaign.freeze_checkpoint(source, run_id, 0, store.snapshot())
                finally:
                    store.close()

    def _pipeline(self, source, run_id, store, output):
        material = self.materials[source]
        return make_pipeline(self.config, store, output, data=material["data"],
                             splits=self._split(source, run_id), benchmark=source,
                             evidence_dir=material["evidence_dir"])

    def _frozen_store(self, checkpoint):
        with self._state_lock:
            return snapshot_store(checkpoint["snapshot"], self.output / "states" / f"{checkpoint['state_hash']}.sqlite")

    def execute(self, claim):
        self.assert_frozen()
        start = time.monotonic()
        source, run_id = claim["source"], claim["run_id"]
        state_path = self.trajectory_directory(source, run_id) / "experience.sqlite"
        store = None
        try:
            if claim["kind"] == "evolution":
                store = ExperienceStore(state_path)
                if digest(store.snapshot()) != claim["state_hash"]:
                    raise CheckpointIntegrityError("EVO cannot execute from a changed trajectory state")
                pipeline = self._pipeline(source, run_id, store, self.slot_directory(claim))
                outcome = pipeline.run("evolve", [claim["task_id"]])[0]
                result = {"task_id": claim["task_id"], "outcome": outcome,
                          "budget": outcome.get("budget"), "after_snapshot_hash": digest(store.snapshot())}
                status = "complete"
            elif claim["kind"] == "validation":
                checkpoint = self.campaign.checkpoint(source, run_id, claim["position"])
                store = self._frozen_store(checkpoint)
                before = digest(store.snapshot())
                pipeline = self._pipeline(source, run_id, store, self.slot_directory(claim))
                outcome = pipeline.run_task(claim["task_id"], checkpoint["snapshot"],
                                            mode="validate", attribution=False, repeat=0)
                if digest(store.snapshot()) != before or digest(checkpoint["snapshot"]) != before:
                    raise CheckpointIntegrityError("VAL mutated its immutable complete checkpoint")
                result = validation_result(source, outcome)
                status = "complete" if result["complete"] else "incomplete"
            else:
                raise ValueError("TEST generation uses the separately sealed submission interface")
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            failure = getattr(error, "jit_mas_run_failure", {})
            result = {"task_id": claim["task_id"], "complete": False, "score": None,
                      "error_type": type(error).__name__, "failure": failure,
                      "budget": failure.get("budget"), "evaluation": None}
            if claim["kind"] == "evolution":
                if store is None:
                    store = ExperienceStore(state_path)
                result["after_snapshot_hash"] = digest(store.snapshot())
            status = "failed"
        finally:
            if store is not None:
                store.close()
        result.update(state_hash=claim["state_hash"], execution_seconds=time.monotonic() - start,
                      registration_hash=self.campaign.registration_hash)
        self.assert_frozen()
        self.campaign.finish(claim, result, status=status)
        self.freeze_ready()
        return {"slot_id": claim["slot_id"], "status": status}

    def freeze_ready(self):
        for source in SOURCES:
            for run_id in RUN_IDS:
                evolution = self.campaign.records(kind="evolution", source=source, run_id=run_id)
                terminal = 0
                for row in evolution:
                    if row["status"] not in TERMINAL:
                        break
                    terminal += 1
                if terminal in CHECKPOINT_POSITIONS[1:] and self.campaign.checkpoint(source, run_id, terminal) is None:
                    store = ExperienceStore(self.trajectory_directory(source, run_id) / "experience.sqlite", read_only=True)
                    try:
                        self.campaign.freeze_checkpoint(source, run_id, terminal, store.snapshot())
                    finally:
                        store.close()

    def _durable_outcome(self, claim):
        completed = []
        for path in self.slot_directory(claim).glob("*/complete.json"):
            outcome = _read(path)
            if (outcome.get("task_id") == claim["task_id"]
                    and outcome.get("experience_hash") == claim["state_hash"]):
                manifest = _read(path.parent / "run_manifest.json")
                comparison = manifest.get("comparison", {})
                if (comparison.get("code") != self.identity["code_fingerprint"]
                        or comparison.get("config") != self.config.model_dump(mode="json")):
                    raise CheckpointIntegrityError("Interrupted outcome belongs to a different execution identity")
                completed.append(outcome)
        if len(completed) > 1:
            raise CheckpointIntegrityError("Interrupted slot contains multiple final outcomes")
        return completed[0] if completed else None

    def recover(self):
        self.assert_frozen()
        for claim in self.campaign.interrupted_claims():
            source, run_id = claim["source"], claim["run_id"]
            outcome = self._durable_outcome(claim)
            store = None
            try:
                if claim["kind"] == "evolution":
                    store = ExperienceStore(self.trajectory_directory(source, run_id) / "experience.sqlite")
                    prior = store.task_run("evolve", claim["task_id"])
                    if prior is None and outcome is not None:
                        raise CheckpointIntegrityError("Durable EVO outcome lacks its baseline transaction journal")
                    if prior is None and digest(store.snapshot()) != claim["state_hash"]:
                        raise CheckpointIntegrityError("Interrupted EVO store no longer matches its claimed input state")
                    if prior and prior["baseline_hash"] != claim["state_hash"]:
                        raise CheckpointIntegrityError("Interrupted EVO baseline differs from its exclusive claim")
                    if prior and prior["status"] == "complete":
                        outcome = prior["outcome"]
                    elif prior and (prior["status"] == "submitted" or outcome is not None):
                        if outcome is not None and prior["status"] != "submitted":
                            store.save_task_run("evolve", claim["task_id"], prior["identity"],
                                                store.snapshot(prior["baseline_version"]), "submitted", outcome)
                        pipeline = self._pipeline(source, run_id, store, self.slot_directory(claim))
                        outcome = pipeline.run("evolve", [claim["task_id"]])[0]
                    if outcome is not None:
                        result = {"task_id": claim["task_id"], "outcome": outcome,
                                  "budget": outcome.get("budget"), "after_snapshot_hash": digest(store.snapshot())}
                        status = "complete"
                    else:
                        result = self._interrupted_failure(claim)
                        result["after_snapshot_hash"] = digest(store.snapshot())
                        status = "failed"
                elif claim["kind"] == "validation" and outcome is not None:
                    result = validation_result(source, outcome)
                    status = "complete" if result["complete"] else "incomplete"
                else:
                    result, status = self._interrupted_failure(claim), "failed"
                result.update(state_hash=claim["state_hash"], recovered=True,
                              registration_hash=self.campaign.registration_hash)
                self.campaign.finish(claim, result, status=status)
            finally:
                if store is not None:
                    store.close()
        self.freeze_ready()

    def _interrupted_failure(self, claim):
        budgets = [_read(path) for path in self.slot_directory(claim).glob("*/budget.json")]
        if len(budgets) > 1:
            raise CheckpointIntegrityError("Interrupted slot has multiple independent attempt budgets")
        return {"task_id": claim["task_id"], "complete": False, "score": None,
                "error_type": "InterruptedWithoutDurableOutcome", "evaluation": None,
                "budget": budgets[0] if budgets else None,
                "retry_policy": "Consumed original slot; no task resampling"}

    def run(self, *, phase="first-stage"):
        with coordinator_lock(self.output):
            self.initialize()
            self.recover()
            with ThreadPoolExecutor(max_workers=self.workers) as workers:
                active = {}
                submitted = 0
                while True:
                    while len(active) < self.workers:
                        preference = "validation" if submitted % self.workers == self.workers - 1 else "evolution"
                        claim = self.campaign.claim(f"worker{len(active)}", phase=phase, prefer=preference)
                        if claim is None:
                            break
                        active[workers.submit(self.execute, claim)] = claim
                        submitted += 1
                    if not active:
                        break
                    finished, _ = wait(active, return_when=FIRST_COMPLETED, timeout=1)
                    for future in finished:
                        future.result()
                        active.pop(future)
            if phase == "full":
                for source in SOURCES:
                    for run_id in RUN_IDS:
                        self.campaign.select_trajectory(source, run_id)
            report = self.campaign.status(phase=phase)
            completed = report["terminal_slots"] if phase == "first-stage" else sum(
                row["status"] in TERMINAL for row in self.campaign.records()
                if row["kind"] in {"evolution", "validation"})
            report.update(evolution_validation_complete=(completed == (75 if phase == "first-stage" else 630)),
                          test_feedback_released=False, output=str(self.output))
            write_json(self.output / f"{phase}_report.json", report)
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "run", "status"], required=True)
    parser.add_argument("--launch-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["first-stage", "full"], default="first-stage")
    args = parser.parse_args(argv)
    load_dotenv()
    if args.mode == "status":
        path = args.output.resolve() / "campaign.sqlite"
        import sqlite3

        database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            registration = json.loads(database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()[0])
        finally:
            database.close()
        campaign = IndependentCampaign(args.output, registration["protocol"], registration["execution"], registration["bounds"])
        report = campaign.status(phase=args.phase)
    else:
        if args.launch_config is None:
            parser.error("--launch-config is required for check and run")
        with redirect_stdout(sys.stderr):
            environment = IndependentEnvironment(args.launch_config, args.output, require_preflight=args.mode == "run")
            report = (environment.run(phase=args.phase) if args.mode == "run" else
                      {"inputs_ready": True, "formal_launch_ready": False, "model_calls": 0,
                       "protocol_sha256": environment.protocol["protocol_sha256"],
                       "registered_slots": 3381, "first_stage_slots": 75})
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
