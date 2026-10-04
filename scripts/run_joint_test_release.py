"""Run the frozen joint v5 TEST release (submission, seal, then scoring).

The joint v5 controller has a different condition inventory from the older
benchmark-suite runner.  This adapter therefore uses the shared
``jit_mas.test_release.TestRelease`` primitive with an explicit 1,911-slot
inventory:

* ``ours_selected``: three selected joint checkpoints for every TEST task;
* ``ours_initial``, ``direct``, ``jit_matched`` and ``rubric_fixed``: one
  initial-state artifact for every TEST task.

No judge is constructed during ``submit``.  ``seal`` must succeed for all
slots before ``score`` can create a judge.  Interrupted submissions are
consumed as failures only with an explicit ``--resolve-interrupted`` reason;
the runner never resamples a slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.pipeline import write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from jit_mas.test_release import TestRelease, score_with_pipeline
from scripts.execute_joint_experiment import BENCHMARKS, JointExecutor, RUN_IDS
from scripts.mas_baseline_methods import run_direct
from scripts.run_benchmark_experiment import file_hash
from scripts.run_jit_mas import make_pipeline


STATIC_METHODS = ("ours_initial", "direct", "jit_matched", "rubric_fixed")
ALL_METHODS = ("ours_selected",) + STATIC_METHODS


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_or_verify(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Frozen TEST registration changed: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _source_checkpoint(report: dict[str, Any], run_id: int, *, base_dir: Path | None = None) -> dict[str, Any]:
    selected = report.get("selections", {}).get(str(run_id), {}).get("selected")
    if not isinstance(selected, dict) or not selected.get("eligible"):
        raise ValueError(f"Run {run_id} has no eligible selected checkpoint")
    path = selected.get("snapshot_path")
    state_hash = selected.get("state_hash")
    file_sha = selected.get("snapshot_file_sha256")
    if not isinstance(path, str) or not isinstance(state_hash, str):
        raise ValueError(f"Run {run_id} selected checkpoint has no immutable snapshot")
    path = str((base_dir / path).resolve()) if base_dir is not None and not Path(path).is_absolute() else path
    if not Path(path).is_file():
        raise ValueError(f"Run {run_id} selected checkpoint has no immutable snapshot")
    if not isinstance(file_sha, str) or file_hash(Path(path)) != file_sha:
        raise CheckpointIntegrityError(f"Run {run_id} selected checkpoint file hash mismatch")
    store = ExperienceStore(path, read_only=True)
    try:
        snapshot = store.snapshot()
        if digest(snapshot) != state_hash:
            raise CheckpointIntegrityError(f"Run {run_id} selected checkpoint state hash mismatch")
    finally:
        store.close()
    return {"position": selected["position"], "snapshot_path": str(Path(path).resolve()),
            "snapshot_file_sha256": file_sha, "state_hash": state_hash}


def _selected_checkpoints(evo_val_dir: Path) -> dict[int, dict[str, Any]]:
    report_path = evo_val_dir / "evo_val_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing completed EVO/VAL report: {report_path}")
    report = _read(report_path)
    if report.get("status") != "evo_val_complete_test_pending" or report.get("formal_test_ready") is not False:
        raise ValueError("EVO/VAL report is not the sealed pre-TEST release state")
    return {run_id: _source_checkpoint(report, run_id, base_dir=evo_val_dir) for run_id in RUN_IDS}


def _initial_snapshot(path: Path) -> dict[str, Any]:
    snapshot = ExperienceSnapshot()
    frozen = (ExperienceStore(path, read_only=True) if path.exists()
              else snapshot_store(snapshot, path))
    try:
        observed = frozen.snapshot()
        if digest(observed) != digest(snapshot):
            raise CheckpointIntegrityError("Initial TEST snapshot changed")
    finally:
        frozen.close()
    return {"snapshot_path": str(path.resolve()), "snapshot_file_sha256": file_hash(path),
            "state_hash": digest(snapshot), "position": 0}


class JointTestReleaseRunner:
    """Coordinator for the joint v5 test release and delayed judge phase."""

    def __init__(self, bundle: str | Path, evo_val_dir: str | Path, campaign: str | Path):
        self.bundle_path = Path(bundle).resolve()
        self.evo_val_dir = Path(evo_val_dir).resolve()
        self.campaign = Path(campaign).resolve()
        self.executor = JointExecutor(self.bundle_path, self.evo_val_dir)
        self.bundle = self.executor.bundle
        self.materials: dict[str, dict[str, Any]] = {}
        self.config: MASConfig | None = None

    def _check_release_inputs(self) -> None:
        # This check is metadata-only, but require the bundle's explicit
        # readiness bit so an ``--allow-incomplete`` audit bundle cannot enter
        # a paid TEST campaign.
        if self.bundle.get("formal_ready") is not True:
            raise ValueError("Joint bundle is not formal-ready; complete data/evidence/checker/provider preflight first")
        self.executor.check(require_provider=True)
        self.executor.assert_frozen()
        registration_path = self.campaign / "registration.json"
        if registration_path.is_file():
            registration = _read(registration_path)
            expected_runner = registration.get("test_runner_sha256")
            if expected_runner and expected_runner != file_hash(Path(__file__)):
                raise CheckpointIntegrityError("TEST release runner changed after registration")
        self.materials = self.executor._identity["materials"]
        self.config = self.executor.config

    def _inventory(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        self._check_release_inputs()
        selected = _selected_checkpoints(self.evo_val_dir)
        initial = _initial_snapshot(self.campaign / "states" / "initial.sqlite")
        manifest = self.executor.manifest
        slots: list[dict[str, Any]] = []
        for run_id in RUN_IDS:
            state = selected[run_id]
            for benchmark in BENCHMARKS:
                for task_id in self.executor._membership(benchmark, "test"):
                    condition_id = f"ours_selected:run{run_id}:{benchmark}"
                    slots.append({"condition_id": condition_id, "method": "ours_selected", "run_id": run_id,
                                  "benchmark": benchmark, "task_id": task_id,
                                  "snapshot_path": state["snapshot_path"], "experience_hash": state["state_hash"],
                                  "snapshot_file_sha256": state["snapshot_file_sha256"], "source_checkpoint": state["position"]})
        for method in STATIC_METHODS:
            for benchmark in BENCHMARKS:
                for task_id in self.executor._membership(benchmark, "test"):
                    condition_id = f"{method}:initial:{benchmark}"
                    slots.append({"condition_id": condition_id, "method": method, "run_id": 0,
                                  "benchmark": benchmark, "task_id": task_id,
                                  "snapshot_path": initial["snapshot_path"], "experience_hash": initial["state_hash"],
                                  "snapshot_file_sha256": initial["snapshot_file_sha256"], "source_checkpoint": 0})
        for row in slots:
            row["slot_id"] = digest(row)
        if len(slots) != 1911 or len({row["slot_id"] for row in slots}) != 1911:
            raise ValueError("Joint v5 TEST inventory must contain exactly 1,911 unique slots")
        return slots, selected

    def register(self) -> dict[str, Any]:
        slots, selected = self._inventory()
        release = TestRelease(self.campaign, slots)
        registration = {"schema": "joint-test-release-v5", "bundle_sha256": file_hash(self.bundle_path),
                        "evo_val_dir": str(self.evo_val_dir), "registration_sha256": digest(release.inventory),
                        "test_runner_sha256": file_hash(Path(__file__)),
                        "slot_count": len(slots), "selected_checkpoints": selected,
                        "test_feedback_released": False, "test_resampling": False,
                        "selection_uses_test": False}
        _write_once_or_verify(self.campaign / "registration.json", registration)
        return {"registered": True, "slot_count": len(slots), "inventory_hash": release.inventory["inventory_hash"],
                "registration_sha256": digest(registration)}

    def _release(self) -> TestRelease:
        return TestRelease(self.campaign)

    def _slot_material(self, slot: dict[str, Any]) -> tuple[object, ExperienceStore, Path]:
        assert self.config is not None
        material = self.materials[slot["benchmark"]]
        store = ExperienceStore(slot["snapshot_path"], read_only=True)
        output = self.campaign / "runs" / digest(slot["slot_id"])[:12]
        item = self.bundle["benchmarks"][slot["benchmark"]]
        pipeline = make_pipeline(self.config, store, output,
                                 data=material["data"], splits=material["splits"],
                                 benchmark=slot["benchmark"], evidence_dir=material["evidence"],
                                 checker_source_root=material["checker"])
        return pipeline, store, output

    def _submit_one(self, slot: dict[str, Any], *, shared_dir: Path) -> dict[str, Any]:
        pipeline, store, output = self._slot_material(slot)
        try:
            method = slot["method"]
            if method == "direct":
                assert self.config is not None
                outcome = run_direct(pipeline.tasks[slot["task_id"]], None, pipeline.models,
                                     pipeline.evaluator_factory, output / "direct",
                                     {"max_calls": self.config.max_model_calls,
                                      "max_tokens": self.config.max_total_tokens,
                                      "max_tool_calls": self.config.max_tool_calls,
                                      "output_tokens": min(8192, self.config.models["exec"].max_tokens),
                                      "timeout_seconds": self.config.task_timeout},
                                     defer_evaluation=True)
                # ``run_direct`` is deliberately benchmark-agnostic and does
                # not know the frozen experience state; bind it here before
                # TestRelease verifies the registered condition.
                outcome["experience_hash"] = slot["experience_hash"]
                write_json(Path(outcome["run_dir"]) / "complete.json", outcome)
            elif method in {"jit_matched", "rubric_fixed"}:
                from jit_mas.experiment_methods import submit_method
                outcome = submit_method(pipeline, slot["task_id"], store.snapshot(), method=method,
                                        repeat=0, output_dir=output, shared_dir=shared_dir)
            else:
                outcome = pipeline.run_task(slot["task_id"], store.snapshot(), mode="evaluate",
                                            repeat=0, attribution=False, defer_evaluation=True, resume=False)
            if outcome.get("experience_hash") != slot["experience_hash"]:
                raise CheckpointIntegrityError("TEST outcome experience hash differs from registered state")
            return outcome
        finally:
            store.close()

    def submit(self, *, resolve_interrupted: str | None = None) -> dict[str, Any]:
        self._check_release_inputs()
        release = self._release()
        shared_dir = self.campaign / "shared_rstar"
        completed = failed = recovered = 0
        for slot_id, slot in release.slots.items():
            record_path = release._path(slot_id)
            if record_path.exists():
                completed += 1
                continue
            started = release._path(slot_id, "submission_started")
            if started.exists():
                if resolve_interrupted is None:
                    raise RuntimeError(f"Interrupted TEST slot requires explicit resolution: {slot_id}")
                release.record_failure(slot_id, error_type="InterruptedSubmission",
                                       budget={"usage_unknown": True, "reason": resolve_interrupted})
                failed += 1
                continue
            write_json(started, {"slot_hash": digest(slot), "started_at": utc_now()})
            try:
                outcome = self._submit_one(slot, shared_dir=shared_dir)
                release.record(slot_id, outcome)
                completed += 1
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                failure = getattr(exc, "jit_mas_run_failure", {})
                release.record_failure(slot_id, error_type=type(exc).__name__,
                                       budget=failure.get("budget", {"usage_unknown": True}))
                failed += 1
        return {"submitted_or_preexisting": completed, "recovered": recovered, "failed": failed,
                "slots": len(release.slots), "sealed": (self.campaign / "seal.json").is_file()}

    def seal(self) -> dict[str, Any]:
        release = self._release()
        seal = release.seal()
        body = {"schema": "joint-test-seal-v5", **seal, "required_test_slots": 1911,
                "feedback_released": False}
        _write_once_or_verify(self.campaign / "test_seal.json", body)
        return body

    def score(self) -> dict[str, Any]:
        self._check_release_inputs()
        release = self._release()
        if not (self.campaign / "seal.json").is_file():
            raise ValueError("Cannot score before every TEST slot is sealed")
        scored = complete = failed = 0
        by_condition: dict[str, dict[str, Any]] = {}
        for slot_id, slot in release.slots.items():
            pipeline, store, _ = self._slot_material(slot)
            try:
                result = release.evaluate(slot_id, lambda record, pipeline=pipeline: score_with_pipeline(pipeline, record))
                scored += 1
                complete += bool(result.get("complete"))
                failed += not bool(result.get("complete"))
                key = f"{slot['method']}:{slot['benchmark']}:run{slot['run_id']}"
                summary = by_condition.setdefault(key, {
                    "method": slot["method"], "benchmark": slot["benchmark"],
                    "run_id": slot["run_id"], "scores": [], "complete": 0,
                    "slots": 0})
                summary["slots"] += 1
                if result.get("complete"):
                    summary["complete"] += 1
                score = result.get("official_score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    summary["scores"].append(float(score))
            finally:
                store.close()
        report = {"schema": "joint-test-report-v5", "slots": len(release.slots),
                  "evaluated": scored, "complete": complete,
                  "incomplete_or_failed": failed, "feedback_released": True,
                  "by_condition": [
                      {**row, "mean_official_score": (sum(row["scores"]) / len(row["scores"])
                                                       if row["scores"] else None)}
                      for row in by_condition.values()]}
        _write_once_or_verify(self.campaign / "test_report.json", report)
        return report

    def status(self) -> dict[str, Any]:
        if not (self.campaign / "inventory.json").is_file():
            raise ValueError("Joint TEST release is not registered")
        release = self._release()
        submitted = failed = evaluated = 0
        for slot_id in release.slots:
            if release._path(slot_id).is_file():
                row = _read(release._path(slot_id))
                submitted += row.get("status") == "submitted"
                failed += row.get("status") == "failed"
            if release._path(slot_id, "evaluations").is_file():
                evaluated += 1
        return {"slots": len(release.slots), "submitted": submitted, "failed": failed,
                "evaluated": evaluated, "sealed": (self.campaign / "seal.json").is_file(),
                "feedback_released": bool(evaluated)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("register", "submit", "seal", "score", "status"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evo-val-dir", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--resolve-interrupted")
    args = parser.parse_args(argv)
    try:
        runner = JointTestReleaseRunner(args.bundle, args.evo_val_dir, args.campaign)
        result = (runner.register() if args.mode == "register" else
                  runner.submit(resolve_interrupted=args.resolve_interrupted) if args.mode == "submit" else
                  runner.seal() if args.mode == "seal" else
                  runner.score() if args.mode == "score" else runner.status())
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError, CheckpointIntegrityError) as exc:
        parser.exit(2, f"Joint TEST release: {exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
