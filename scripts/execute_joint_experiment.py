"""Execute the real (non-TEST) portion of the frozen joint v5 protocol.

This module is intentionally small and explicit.  It owns the frozen-input
check, a single-coordinator journal, and the three shared per-run experience
stores.  Model requests are made only by :meth:`JointExecutor.run_evolution_validation`;
``check`` never constructs a provider and ``run_test`` is deliberately refused
until the separate TestRelease integration is available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import yaml

from jit_mas.checkpoints import CheckpointIntegrityError, select_checkpoint, snapshot_store
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.independent_campaign import coordinator_lock
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import SplitManifest, digest
from jit_mas.token_usage import summarize_budget, summarize_outcome, usage_totals_from_slots
from scripts.env_config import resolve_env_placeholders
from scripts.run_benchmark_experiment import evidence_identity, file_hash
from scripts.run_jit_mas import make_pipeline
from scripts.run_joint_experiment import CHECKPOINTS, RUN_IDS, SOURCES, _stage_order


BENCHMARKS = SOURCES + ("deepresearch_bench_ii", "ifeval", "ifbench")
TERMINAL = frozenset({"complete", "incomplete", "failed", "missing"})


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _score(name: str, outcome: Mapping[str, Any]) -> float | None:
    evaluation = outcome.get("evaluation") or {}
    if evaluation.get("complete") is not True:
        return None
    raw = evaluation.get("raw") or {}
    value = raw.get("native_mean") if name == "writingbench" else evaluation.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _failure_diagnostic(exc: Exception) -> dict[str, Any]:
    """Record code locations only, never exception text, source lines or locals."""
    root = Path(__file__).resolve().parents[1]
    frames = []
    traceback = exc.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        filename = code.co_filename
        if filename.startswith("<"):
            filename = filename if filename in {"<stdin>", "<string>"} else "<dynamic>"
        else:
            path = Path(filename)
            try:
                filename = path.resolve().relative_to(root).as_posix()
            except ValueError:
                # External absolute paths may contain user-controlled directory
                # names.  The basename suffices to locate a dependency frame.
                filename = path.name
        frames.append({"file": filename, "function": code.co_name,
                       "line": traceback.tb_lineno})
        traceback = traceback.tb_next
    return {"schema": "exception-code-locations-v1", "error_type": type(exc).__name__,
            "frames": frames, "exception_message_recorded": False,
            "locals_recorded": False, "source_lines_recorded": False}


def _failure_token_usage(exc: Exception) -> dict[str, Any]:
    """Recover ledger usage attached by ``MASPipeline`` on a failed task."""
    failure = getattr(exc, "jit_mas_run_failure", None)
    if isinstance(failure, Mapping) and isinstance(failure.get("budget"), Mapping):
        return summarize_budget(failure["budget"])
    return summarize_budget(None)


class JointExecutor:
    """Frozen joint executor.  The bundle is the only source of runtime inputs."""

    def __init__(self, bundle: str | Path, output: str | Path, *, val_workers: int = 1):
        if type(val_workers) is not int or not 1 <= val_workers <= 16:
            raise ValueError("val_workers must be an integer between 1 and 16")
        self.bundle_path = Path(bundle).resolve()
        self.output = Path(output).resolve()
        self.val_workers = val_workers
        self._journal_lock = threading.RLock()
        self.bundle = _read(self.bundle_path)
        self.base = self.bundle_path.parent
        self.protocol_path = _path(self.bundle["protocol"], self.base)
        self.manifest_path = _path(self.bundle["manifest"], self.base)
        self.protocol = _read(self.protocol_path)
        self.manifest = _read(self.manifest_path)
        self.config_path = _path(self.bundle["config"], self.base)
        raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.config = MASConfig.model_validate(resolve_env_placeholders(raw_config))
        self._identity = None

    def _membership(self, benchmark: str, part: str) -> list[str]:
        values = self.manifest.get("memberships", {}).get(benchmark, {}).get(part)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"Invalid frozen {benchmark}/{part} membership")
        return list(values)

    def _protocol_check(self):
        if self.protocol.get("version") != "jit-compose-joint-protocol-v5":
            raise ValueError("Wrong joint protocol version")
        if self.manifest.get("version") != "jit-compose-joint-subset-v5":
            raise ValueError("Wrong joint split manifest version")
        if self.protocol.get("status") != "SUBSET_PROTOCOL_FROZEN_NOT_RUN" or self.manifest.get("status") != "SUBSET_PROTOCOL_FROZEN_NOT_RUN":
            raise ValueError("Joint inputs must remain the frozen pre-run registration")
        if self.protocol.get("split_manifest_sha256") != self.manifest.get("manifest_sha256"):
            raise ValueError("Protocol/manifest hash binding mismatch")
        if _hash_json({k: v for k, v in self.manifest.items() if k != "manifest_sha256"}) != self.manifest.get("manifest_sha256"):
            raise ValueError("Split manifest self-hash mismatch")
        if _hash_json({k: v for k, v in self.protocol.items() if k != "protocol_sha256"}) != self.protocol.get("protocol_sha256"):
            raise ValueError("Protocol self-hash mismatch")
        totals = self.manifest.get("totals", {})
        if totals.get("evolution") != 60 or totals.get("validation") != 30:
            raise ValueError("Joint runtime must contain 60 EVO and 30 VAL tasks per run")
        expected_evo = {task for benchmark in SOURCES for task in self._membership(benchmark, "evolution")}
        if len(expected_evo) != 60:
            raise ValueError("Frozen source EVO memberships must contain exactly 60 unique tasks")
        for run_id in RUN_IDS:
            if set(_stage_order(self.manifest, run_id)) != expected_evo:
                raise ValueError(f"Run {run_id} schedule does not match frozen EVO memberships")
        for benchmark in SOURCES:
            if len(self._membership(benchmark, "validation")) != 10:
                raise ValueError(f"Frozen {benchmark} VAL membership must contain 10 tasks")

    def check(self, *, require_provider: bool = False) -> dict[str, Any]:
        """Validate all frozen paths without making model requests."""
        self._protocol_check()
        if require_provider:
            preflight = self.bundle.get("preflight", {})
            if preflight.get("passed") is not True or preflight.get("synthetic_only") is True:
                raise ValueError("Formal joint execution requires a passed non-synthetic provider preflight")
            self.config.check_native()
            observed = self.bundle.get("served_models", {})
            for role, spec in self.config.models.items():
                identity = observed.get(role, {})
                if identity.get("requested_model") != spec.model or not identity.get("returned_model") or identity.get("synthetic") is True:
                    raise ValueError(f"Provider preflight has no observed non-synthetic identity for {role}")
        registrations = self.bundle.get("benchmarks")
        if not isinstance(registrations, dict) or set(registrations) != set(BENCHMARKS):
            raise ValueError("Bundle must bind all six benchmarks")
        materials = {}
        for name in BENCHMARKS:
            item = registrations[name]
            data = _path(item["data"], self.base)
            splits = _path(item["splits"], self.base)
            if not data.is_file() or not splits.is_file():
                raise FileNotFoundError(f"Missing frozen {name} data/split path")
            evidence = _path(item["evidence_dir"], self.base) if item.get("evidence_dir") else None
            if name in {"researchrubrics", "deepsearchqa", "deepresearch_bench_ii"} and evidence is None:
                raise ValueError(f"{name} requires a frozen evidence directory")
            if evidence is not None and not evidence.is_dir():
                raise FileNotFoundError(f"Missing evidence directory for {name}")
            checker = _path(item["checker_source_root"], self.base) if item.get("checker_source_root") else None
            if name in {"ifeval", "ifbench"} and require_provider and checker is None:
                raise ValueError(f"{name} requires a pinned checker source root")
            if checker is not None and not checker.is_dir():
                raise FileNotFoundError(f"Checker source root for {name} is not a directory")
            materials[name] = {"data": str(data), "data_sha256": file_hash(data),
                               "splits": str(splits), "splits_sha256": file_hash(splits),
                               "evidence": str(evidence) if evidence else None,
                               "evidence_identity": evidence_identity(evidence) if evidence else None,
                               "checker": str(checker) if checker else None,
                               "checker_sha256": evidence_identity(checker) if checker and checker.is_dir() else None}
        identity = {"schema": "joint-execution-bundle-v1", "bundle_sha256": file_hash(self.bundle_path),
                    "protocol_sha256": file_hash(self.protocol_path), "manifest_sha256": file_hash(self.manifest_path),
                    "config_sha256": file_hash(self.config_path), "configuration": self.config.model_dump(mode="json"),
                    "code_fingerprint": code_fingerprint(), "runner_sha256": file_hash(Path(__file__)),
                    "materials": materials, "seed": self.bundle.get("seed", 20261001)}
        self._identity = identity
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "registration.json"
        if path.exists() and _read(path) != identity:
            raise CheckpointIntegrityError("Frozen joint registration changed")
        if not path.exists():
            write_json(path, identity)
        return {"ready": True, "provider_checked": require_provider, "registration_sha256": digest(identity),
                "protocol_sha256": identity["protocol_sha256"], "material_count": len(materials),
                "test_supported": False}

    def assert_frozen(self):
        if self._identity is None:
            self.check(require_provider=False)
        current = self._identity
        if file_hash(self.bundle_path) != current["bundle_sha256"] or file_hash(self.protocol_path) != current["protocol_sha256"]:
            raise CheckpointIntegrityError("Joint bundle or protocol changed")
        if file_hash(self.manifest_path) != current["manifest_sha256"] or file_hash(self.config_path) != current["config_sha256"]:
            raise CheckpointIntegrityError("Joint manifest or configuration changed")
        if code_fingerprint() != current["code_fingerprint"] or file_hash(Path(__file__)) != current["runner_sha256"]:
            raise CheckpointIntegrityError("Joint code identity changed")
        for name, material in current["materials"].items():
            if file_hash(Path(material["data"])) != material["data_sha256"] or file_hash(Path(material["splits"])) != material["splits_sha256"]:
                raise CheckpointIntegrityError(f"Frozen {name} data/split changed")
            if material["evidence"] and evidence_identity(Path(material["evidence"])) != material["evidence_identity"]:
                raise CheckpointIntegrityError(f"Frozen {name} evidence changed")
            if material["checker"] and evidence_identity(Path(material["checker"])) != material["checker_sha256"]:
                raise CheckpointIntegrityError(f"Frozen {name} checker changed")
        journal = self._journal()
        if journal.is_file():
            checkpoints = _read(journal).get("checkpoints", {})
            for key, checkpoint in checkpoints.items():
                path = checkpoint.get("snapshot_path")
                expected = checkpoint.get("snapshot_file_sha256")
                if path and expected and (not Path(path).is_file() or file_hash(Path(path)) != expected):
                    raise CheckpointIntegrityError(f"Frozen {key} snapshot changed")
                if path and checkpoint.get("state_hash"):
                    frozen = ExperienceStore(path, read_only=True)
                    try:
                        if digest(frozen.snapshot()) != checkpoint["state_hash"]:
                            raise CheckpointIntegrityError(f"Frozen {key} state hash mismatch")
                    finally:
                        frozen.close()

    def _slots(self) -> dict[str, dict[str, Any]]:
        slots = {}
        for run in RUN_IDS:
            for ordinal, task in enumerate(_stage_order(self.manifest, run)):
                sid = f"evo:run{run}:{ordinal:02d}:{task}"
                slots[sid] = {"slot_id": sid, "kind": "evolution", "run_id": run, "ordinal": ordinal, "task_id": task, "status": "pending"}
            for checkpoint in CHECKPOINTS:
                for benchmark in SOURCES:
                    for task in self._membership(benchmark, "validation"):
                        sid = f"val:run{run}:c{checkpoint}:{benchmark}:{task}"
                        slots[sid] = {"slot_id": sid, "kind": "validation", "run_id": run, "checkpoint": checkpoint, "benchmark": benchmark, "task_id": task, "status": "pending"}
        return slots

    def _journal(self) -> Path:
        return self.output / "evo_val_journal.json"

    def _load_or_init(self) -> dict[str, Any]:
        slots = self._slots()
        identity = {"registration_sha256": digest(self._identity), "slot_ids": sorted(slots),
                    "val_workers": self.val_workers}
        path = self._journal()
        if path.exists():
            doc = _read(path)
            if doc.get("identity") != identity or set(doc.get("slots", {})) != set(slots):
                raise CheckpointIntegrityError("Joint journal identity or slot inventory changed")
            return doc
        doc = {"schema": "joint-evo-val-journal-v1", "identity": identity, "slots": slots,
               "checkpoints": {}, "selections": {}, "status": "running",
               "token_usage": usage_totals_from_slots(slots.values())}
        write_json(path, doc)
        return doc

    def _consume(self, doc, slot_id, callback):
        # Slot state is shared by optional VAL workers.  Hold the lock only for
        # durable journal transitions; model generation and judging remain
        # outside the critical section so independent VAL slots can overlap.
        with self._journal_lock:
            row = doc["slots"][slot_id]
            if row["status"] in TERMINAL:
                return row
            if row["status"] == "started":
                row.update({"status": "failed", "result": {
                    "error_type": "InterruptedWithoutDurableOutcome", "score": None,
                    "token_usage": summarize_budget(None)}, "finished_at": time.time()})
                doc["token_usage"] = usage_totals_from_slots(doc["slots"].values())
                write_json(self._journal(), doc)
                return row
            row.update({"status": "started", "started_at": time.time(), "attempts": 1})
            write_json(self._journal(), doc)
        # A read-only VAL callback still generates a task artifact.  Repeating
        # it would resample that artifact, contrary to the frozen one-artifact
        # protocol.  Both EVO and VAL therefore consume their callback once.
        try:
            result = callback()
            status = "complete" if result.get("complete", True) else "incomplete"
        except Exception as exc:
            result = {"error_type": type(exc).__name__, "score": None, "complete": False,
                      "failure_diagnostic": _failure_diagnostic(exc),
                      "token_usage": _failure_token_usage(exc)}
            status = "failed"
        if "token_usage" not in result:
            nested = result.get("outcome") if isinstance(result, Mapping) else None
            result["token_usage"] = summarize_outcome(nested if isinstance(nested, Mapping) else result)
        with self._journal_lock:
            row = doc["slots"][slot_id]
            row.update({"status": status, "result": result, "result_sha256": digest(result), "finished_at": time.time()})
            doc["token_usage"] = usage_totals_from_slots(doc["slots"].values())
            write_json(self._journal(), doc)
            return row

    def _global_manifest(self, run_id: int) -> SplitManifest:
        evo = _stage_order(self.manifest, run_id)
        val = [task for benchmark in SOURCES for task in self._membership(benchmark, "validation")]
        test = [task for benchmark in BENCHMARKS for task in self._membership(benchmark, "test")]
        # ``membership_seed`` in the frozen v5 document is a textual
        # generator label, while the runtime SplitManifest schema requires an
        # integer order seed.  Runtime membership is already bound above, so
        # use the registered first order seed as the deterministic descriptor.
        return SplitManifest(seed=20261001, evolution=evo,
                             validation=val, test=test, stream=[])

    def _build_runtime(self, run_id: int):
        from jit_mas.benchmarks import load_benchmark
        registrations = self.bundle["benchmarks"]
        datasets, materials = {}, self._identity["materials"]
        merged_tasks, merged_private = {}, {}
        for name in BENCHMARKS:
            dataset = load_benchmark(name, materials[name]["data"], available_tools=self.config.available_tools)
            datasets[name] = dataset
            merged_tasks.update(dataset.tasks)
            merged_private.update(dataset.private_records)
        global_manifest = self._global_manifest(run_id)
        store = ExperienceStore(self.output / f"run{run_id}" / "experience.sqlite")
        pipelines = {}
        for name in BENCHMARKS:
            item = registrations[name]
            # make_pipeline validates and loads the benchmark-specific adapter;
            # replace only the task maps/manifest with the joint public union so
            # retrieval cannot see another benchmark's held-out tasks.
            pipeline = make_pipeline(self.config, store, self.output / f"run{run_id}" / name,
                                     data=materials[name]["data"], splits=materials[name]["splits"],
                                     benchmark=name, evidence_dir=materials[name]["evidence"],
                                     checker_source_root=materials[name]["checker"])
            # Evidence packs can replace the public task attachment set.  Keep
            # those verified task objects in the global union before rebinding
            # every pipeline, otherwise research tasks would silently lose the
            # frozen evidence packet.
            merged_tasks.update(pipeline.tasks)
            merged_private.update(pipeline.private_records)
            pipelines[name] = pipeline
        for pipeline in pipelines.values():
            pipeline.tasks, pipeline.private_records, pipeline.manifest = merged_tasks, merged_private, global_manifest
        return store, pipelines, datasets, global_manifest

    def _bounds(self, name, dataset, task):
        if name == "writingbench":
            return 1.0, 10.0
        return float(dataset.lower_bounds[task]), float(dataset.upper_bounds[task])

    def _run_val(self, doc, pipelines, datasets, name, run_id, checkpoint, task):
        sid = f"val:run{run_id}:c{checkpoint}:{name}:{task}"
        def invoke():
            # Validation is a read-only observation of the exact checkpoint.
            # The mutable trajectory may already be several stages ahead when a
            # run is resumed, so never source VAL from ``pipeline.store``.
            frozen = self._open_checkpoint(doc, run_id, checkpoint)
            try:
                snapshot = frozen.snapshot()
                out = pipelines[name].run_task(task, snapshot, mode="validate", attribution=False, repeat=0)
            finally:
                frozen.close()
            value = _score(name, out)
            lower, upper = self._bounds(name, datasets[name], task)
            norm = None if value is None else (0.0 if upper == lower else max(0.0, min(1.0, (value-lower)/(upper-lower))))
            return {"task_id": task, "score": value, "normalized": norm, "complete": value is not None,
                    "outcome": out, "token_usage": summarize_outcome(out)}
        return self._consume(doc, sid, invoke)

    def _run_validation_checkpoint(self, doc, pipelines, datasets, run_id: int, checkpoint: int) -> None:
        """Evaluate one fixed VAL checkpoint, optionally overlapping read-only slots."""
        jobs = [(benchmark, task)
                for benchmark in SOURCES
                for task in self._membership(benchmark, "validation")]
        if self.val_workers == 1:
            for benchmark, task in jobs:
                self._run_val(doc, pipelines, datasets, benchmark, run_id, checkpoint, task)
            return
        with ThreadPoolExecutor(max_workers=min(self.val_workers, len(jobs)),
                                thread_name_prefix=f"val-run{run_id}-c{checkpoint}") as pool:
            futures = [pool.submit(self._run_val, doc, pipelines, datasets,
                                   benchmark, run_id, checkpoint, task)
                       for benchmark, task in jobs]
            # Consume all futures so the checkpoint never advances while a
            # sibling slot is still running; each slot remains single-shot.
            for future in as_completed(futures):
                future.result()

    def _open_checkpoint(self, doc, run_id: int, position: int) -> ExperienceStore:
        """Open and verify one immutable checkpoint for a VAL observation."""
        key = f"run{run_id}:c{position}"
        record = doc.get("checkpoints", {}).get(key)
        if not isinstance(record, dict):
            raise CheckpointIntegrityError(f"Missing frozen checkpoint {key}")
        path = Path(record.get("snapshot_path", ""))
        expected_file = record.get("snapshot_file_sha256")
        expected_state = record.get("state_hash")
        if not path.is_file() or not expected_file or file_hash(path) != expected_file:
            raise CheckpointIntegrityError(f"Frozen {key} snapshot file hash mismatch")
        frozen = ExperienceStore(path, read_only=True)
        observed = frozen.snapshot()
        if not expected_state or digest(observed) != expected_state:
            frozen.close()
            raise CheckpointIntegrityError(f"Frozen {key} state hash mismatch")
        return frozen

    def _ensure_checkpoint(self, doc, store: ExperienceStore, run_id: int, position: int) -> dict[str, Any]:
        """Keep an existing snapshot immutable; never compare it with later live state."""
        key = f"run{run_id}:c{position}"
        existing = doc.get("checkpoints", {}).get(key)
        if existing is not None:
            frozen = self._open_checkpoint(doc, run_id, position)
            frozen.close()
            return existing
        record = self._persist_checkpoint(store, run_id, position)
        doc["checkpoints"][key] = record
        write_json(self._journal(), doc)
        return record

    def _assert_live_prefix(self, doc, store: ExperienceStore, run_id: int) -> None:
        """Fail closed if the mutable store and durable EVO journal disagree."""
        order = _stage_order(self.manifest, run_id)
        latest_hash = None
        c0 = doc.get("checkpoints", {}).get(f"run{run_id}:c0")
        if c0:
            latest_hash = c0.get("state_hash")
        for ordinal, task in enumerate(order):
            sid = f"evo:run{run_id}:{ordinal:02d}:{task}"
            row = doc["slots"].get(sid)
            if row is None or row.get("status") == "pending":
                break
            if row.get("status") == "started":
                raise CheckpointIntegrityError(f"Unresolved started EVO slot {sid}; refusing resampling")
            if row.get("status") not in TERMINAL:
                raise CheckpointIntegrityError(f"Unknown EVO slot status for {sid}")
            if row.get("status") == "complete":
                after = (row.get("result") or {}).get("after_state_sha256")
                if not after:
                    raise CheckpointIntegrityError(f"Complete EVO slot {sid} has no durable state hash")
                latest_hash = after
        if latest_hash is not None and digest(store.snapshot()) != latest_hash:
            raise CheckpointIntegrityError(f"Mutable run{run_id} state disagrees with durable EVO prefix")

    def run_evolution_validation(self):
        """Run all 3×(C0 VAL → 60 EVO/30 VAL) trajectories once.

        This method requires native credentials in the bundle configuration and
        therefore is never called by the offline tests.
        """
        self.check(require_provider=True)
        self.assert_frozen()
        lock = coordinator_lock(self.output)
        with lock:
            doc = self._load_or_init()
            for run_id in RUN_IDS:
                store, pipelines, datasets, _ = self._build_runtime(run_id)
                try:
                    # C0 is a registered whole-state candidate.  Persist the
                    # empty initial state before its first validation pass so
                    # selection never silently excludes the baseline.
                    self._ensure_checkpoint(doc, store, run_id, 0)
                    self._assert_live_prefix(doc, store, run_id)
                    for checkpoint in CHECKPOINTS:
                        self._run_validation_checkpoint(doc, pipelines, datasets, run_id, checkpoint)
                        if checkpoint == 60:
                            self._ensure_checkpoint(doc, store, run_id, checkpoint)
                            break
                        order = _stage_order(self.manifest, run_id)
                        start, end = checkpoint, checkpoint + 15
                        for ordinal in range(start, end):
                            task = order[ordinal]
                            sid = f"evo:run{run_id}:{ordinal:02d}:{task}"
                            benchmark = next(name for name in SOURCES if task in self._membership(name, "evolution"))
                            def invoke(task=task, pipeline=pipelines[benchmark]):
                                outcome = pipeline.run("evolve", [task])[0]
                                return {"complete": True, "task_id": task, "outcome": outcome,
                                        "after_state_sha256": digest(pipeline.store.snapshot()),
                                        "token_usage": summarize_outcome(outcome)}
                            self._consume(doc, sid, invoke)
                        self._assert_live_prefix(doc, store, run_id)
                        self._ensure_checkpoint(doc, store, run_id, checkpoint + 15)
                finally:
                    store.close()
                # The final validation at C60 is performed by the loop above;
                # selection is recorded only after every checkpoint is terminal.
            selections = {str(run_id): self._select_run(doc, run_id) for run_id in RUN_IDS}
            doc["status"] = ("evo_val_complete_test_pending"
                              if all((row.get("selected") or {}).get("eligible")
                                     for row in selections.values())
                              else "evo_val_inconclusive_test_pending")
            write_json(self._journal(), doc)
            report = {"schema": "joint-evo-val-report-v1", "status": doc["status"],
                      "registration_sha256": doc["identity"]["registration_sha256"],
                      "selections": doc["selections"], "formal_test_ready": False,
                      "test_note": "TEST requires TestRelease integration",
                      "test_feedback_released": False,
                      "token_usage": usage_totals_from_slots(doc["slots"].values())}
            write_json(self.output / "evo_val_report.json", report)
            return report

    def _persist_checkpoint(self, store: ExperienceStore, run_id: int, position: int) -> dict[str, Any]:
        """Materialize a read-only snapshot artifact for later TestRelease use."""
        snapshot = store.snapshot()
        path = self.output / f"run{run_id}" / "checkpoints" / f"c{position:03d}.sqlite"
        # A crash after the atomic snapshot rename but before the journal write
        # is resumable when the existing immutable bytes describe the same
        # state; a different state remains a hard integrity failure.
        frozen = (ExperienceStore(path, read_only=True) if path.exists()
                  else snapshot_store(snapshot, path))
        try:
            observed = frozen.snapshot()
            if digest(observed) != digest(snapshot):
                raise CheckpointIntegrityError("Persisted checkpoint snapshot changed")
        finally:
            frozen.close()
        return {"state_hash": digest(snapshot), "snapshot_path": str(path), "snapshot_file_sha256": file_hash(path)}

    def _select_run(self, doc, run_id):
        candidates = []
        for checkpoint in CHECKPOINTS:
            rows = [row for row in doc["slots"].values() if row["kind"] == "validation" and row["run_id"] == run_id and row["checkpoint"] == checkpoint]
            by_benchmark = {}
            for benchmark in SOURCES:
                b = [row for row in rows if row["benchmark"] == benchmark]
                complete = [row for row in b if row["status"] == "complete" and row.get("result", {}).get("normalized") is not None]
                by_benchmark[benchmark] = {"complete": len(complete), "slots": len(b), "mean_normalized": (sum(r["result"]["normalized"] for r in complete) / len(b) if len(b) == 10 else None)}
            eligible = all(v["slots"] == 10 and v["complete"] >= 9 and v["mean_normalized"] is not None for v in by_benchmark.values())
            utility = sum(v["mean_normalized"] for v in by_benchmark.values()) / 3 if eligible else None
            checkpoint_record = doc.get("checkpoints", {}).get(f"run{run_id}:c{checkpoint}")
            state_hash = checkpoint_record.get("state_hash") if checkpoint_record else None
            candidates.append({"position": checkpoint, "eligible": eligible and bool(state_hash), "selection_utility": utility,
                               "complete_evaluations": sum(v["complete"] for v in by_benchmark.values()), "by_benchmark": by_benchmark,
                               "state_hash": state_hash,
                               "snapshot_path": checkpoint_record.get("snapshot_path") if checkpoint_record else None,
                               "snapshot_file_sha256": checkpoint_record.get("snapshot_file_sha256") if checkpoint_record else None})
        selected = select_checkpoint(candidates, tolerance=1e-12)
        record = {"candidates": candidates, "selected": selected,
                  "selection_uses_test": False,
                  "status": "selected" if selected is not None else "inconclusive"}
        doc["selections"][str(run_id)] = record
        return record

    def run_test(self):
        raise NotImplementedError("TEST execution is intentionally unavailable until unified TestRelease wiring is reviewed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-workers", type=int, default=1,
                        help="Optional concurrent read-only VAL workers (1 preserves sequential order)")
    parser.add_argument("--mode", choices=("check", "run", "test"), required=True)
    args = parser.parse_args(argv)
    executor = JointExecutor(args.bundle, args.output, val_workers=args.val_workers)
    if args.mode == "check":
        result = executor.check(require_provider=False)
    elif args.mode == "run":
        result = executor.run_evolution_validation()
    else:
        executor.run_test()
        result = {}
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
