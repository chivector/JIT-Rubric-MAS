"""Run a resumable, explicitly adapted final TEST snapshot evaluation.

This runner is deliberately separate from the frozen evolution campaign.  It
binds each target to an immutable checkpoint from a source campaign, runs the
unchanged native MAS pipeline on the public TEST prompt, and records the
official evaluator result immediately.  ``test_adapted`` is a visible
provenance flag: public task checklists may be projected into the prompt, but
reference answers are never read or supplied to generation.

The command is safe to stop and resume.  Each benchmark has an independent
``records.json`` journal and each task owns a separate output directory.  A
failed task is retained as failed unless ``--retry-failed`` is supplied.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.experience import ExperienceStore
from jit_mas.pipeline import write_json
from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest, utc_now
from jit_mas.token_usage import summarize_outcome
from scripts.eval.config import load_dotenv
from scripts.run_independent_experiment import IndependentEnvironment, _path
from scripts.run_jit_mas import make_pipeline
from scripts.run_targeted_batch_experiment import configure_targeted_pipeline


DEFAULT_POSITIONS = {
    "researchrubrics": 35,
    "writingbench": 40,
    "deepsearchqa": 25,
    "deepresearch_bench_ii": 10,
}
SOURCES = frozenset(DEFAULT_POSITIONS)
ALL_BENCHMARKS = (*DEFAULT_POSITIONS, "ifeval", "ifbench")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_db(source_campaign: Path) -> Path:
    path = Path(source_campaign).resolve()
    if path.is_dir():
        path = path / "campaign.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"source campaign database not found: {path}")
    return path


def _checkpoint(source_campaign: Path, source: str, position: int, run_id: int = 0):
    db_path = _source_db(source_campaign)
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT state_hash,snapshot FROM checkpoints WHERE source=? AND run_id=? AND position=?",
            (source, run_id, position),
        ).fetchone()
    if row is None:
        raise CheckpointIntegrityError(
            f"source campaign has no checkpoint {source}/run{run_id}/position{position}"
        )
    snapshot = ExperienceSnapshot.model_validate_json(row[1])
    actual = digest(snapshot)
    if actual != row[0]:
        raise CheckpointIntegrityError(f"checkpoint hash mismatch for {source} position {position}")
    return snapshot, row[0]


def _manifest(env: IndependentEnvironment, target: str) -> Path:
    """Write an immutable runtime split for one target.

    Evidence-backed targets receive the complete joint membership, because
    the evidence loader validates all IDs in the manifest.  Instruction
    targets intentionally receive TEST IDs only.
    """
    membership = env.joint["memberships"][target]
    if target in {"ifeval", "ifbench"}:
        document = {"seed": 0, "evolution": [], "validation": [],
                    "test": list(membership.get("test", [])), "stream": []}
    else:
        document = {"seed": 0,
                    "evolution": list(membership.get("evolution", [])),
                    "validation": list(membership.get("validation", [])),
                    "test": list(membership.get("test", [])), "stream": []}
    path = env.output / "test_adapted_splits" / f"{target}.json"
    wrapped = {"runtime_split_manifest": document}
    if path.exists() and _read(path) != wrapped:
        raise CheckpointIntegrityError(f"runtime split changed for {target}")
    if not path.exists():
        write_json(path, wrapped)
    return path


class TestAdaptedFinalRunner:
    """Resumable native final TEST evaluator over immutable source snapshots."""

    def __init__(self, launch: Path, source_campaign: Path, output: Path,
                 *, benchmarks: list[str], workers: int = 8, limit: int | None = None,
                 retry_failed: bool = False):
        if workers < 1:
            raise ValueError("workers must be positive")
        unknown = set(benchmarks) - set(ALL_BENCHMARKS)
        if unknown:
            raise ValueError(f"unknown benchmark(s): {sorted(unknown)}")
        self.launch = Path(launch).resolve()
        self.source_campaign = Path(source_campaign).resolve()
        self.source_db_path = _source_db(self.source_campaign)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.benchmarks = benchmarks
        self.workers = workers
        self.limit = limit
        self.retry_failed = retry_failed
        # This performs launch/config/data/evidence validation without opening
        # or modifying the source campaign inventory.
        self.environment = IndependentEnvironment(self.launch, self.output, require_preflight=True)
        self._journal_lock = threading.RLock()
        self._split_lock = threading.Lock()
        self._snapshots: dict[str, tuple[ExperienceSnapshot, str, int]] = {}
        self._load_snapshots()
        self.identity = self._build_identity()
        identity_path = self.output / "test_adapted_identity.json"
        if identity_path.exists() and _read(identity_path) != self.identity:
            raise CheckpointIntegrityError("test-adapted identity changed on resume")
        if not identity_path.exists():
            write_json(identity_path, self.identity)

    def _load_snapshots(self):
        for benchmark in self.benchmarks:
            if benchmark in SOURCES:
                position = DEFAULT_POSITIONS[benchmark]
                snapshot, state_hash = _checkpoint(self.source_campaign, benchmark, position)
                self._snapshots[benchmark] = (snapshot, state_hash, position)
            else:
                # Instruction benchmarks reuse the selected RR checkpoint by
                # design.  This does not alter the instruction checker or data.
                snapshot, state_hash = _checkpoint(
                    self.source_campaign, "researchrubrics", DEFAULT_POSITIONS["researchrubrics"]
                )
                self._snapshots[benchmark] = (snapshot, state_hash, DEFAULT_POSITIONS["researchrubrics"])

    def _build_identity(self):
        materials = {name: self.environment.materials[name] for name in self.benchmarks}
        return {
            "schema": "test-adapted-final-v1",
            "test_adapted": True,
            "rubric_exposed": True,
            "reference_answers_exposed": False,
            "generator": "native_shared_ledger_mas",
            "evaluator": "official_pipeline_evaluator_or_pinned_checker",
            "source_campaign_sha256": _sha256(self.source_db_path),
            "launch_sha256": _sha256(self.launch),
            "configuration_sha256": self.environment.identity["configuration_file_sha256"],
            "joint_manifest_sha256": self.environment.identity["joint_manifest_sha256"],
            "benchmarks": list(self.benchmarks),
            "source_positions": {name: self._snapshots[name][2] for name in self.benchmarks},
            "snapshot_hashes": {name: self._snapshots[name][1] for name in self.benchmarks},
            "materials": materials,
            "no_expected_answer_in_prompt": True,
        }

    def _task_ids(self, benchmark):
        return list(self.environment.joint["memberships"][benchmark].get("test", []))

    def _record_path(self, benchmark):
        return self.output / "benchmarks" / benchmark / "records.json"

    def _load_records(self, benchmark):
        path = self._record_path(benchmark)
        if not path.exists():
            return {}
        body = _read(path)
        if body.get("identity") != self.identity:
            raise CheckpointIntegrityError(f"journal identity changed for {benchmark}")
        return dict(body.get("records", {}))

    def _save_records(self, benchmark, records):
        path = self._record_path(benchmark)
        write_json(path, {"schema": "test-adapted-records-v1", "identity": self.identity,
                          "benchmark": benchmark, "records": records,
                          "updated_at": utc_now()})

    def _split(self, benchmark):
        return _manifest(self.environment, benchmark)

    def _run_one(self, benchmark, task_id):
        snapshot, state_hash, position = self._snapshots[benchmark]
        task_root = self.output / "benchmarks" / benchmark / "tasks" / digest(task_id)
        complete = task_root / "complete.json"
        started = time.monotonic()
        if complete.is_file():
            outcome = _read(complete)
            if outcome.get("task_id") == task_id and outcome.get("state_hash") == state_hash:
                return self._result_record(benchmark, task_id, outcome, position, started, resumed=True)
        state_path = self.output / "snapshot_store" / f"{state_hash}.sqlite"
        frozen = snapshot_store(snapshot, state_path)
        try:
            pipeline = make_pipeline(
                self.environment.config, frozen, task_root,
                data=self.environment.materials[benchmark]["data"],
                splits=self._split(benchmark), benchmark=benchmark,
                evidence_dir=self.environment.materials[benchmark].get("evidence_dir"),
                checker_source_root=self.environment.materials[benchmark].get("checker_source_root"),
            )
            # Public rubric/checklist projection only.  This call never passes
            # reference answers to the generator; scoring remains official.
            configure_targeted_pipeline(pipeline, benchmark=benchmark)
            outcome = pipeline.run_task(
                task_id, snapshot, mode="evaluate", attribution=False, repeat=0,
                defer_evaluation=False,
            )
            outcome = dict(outcome)
            outcome["test_adapted"] = True
            outcome["rubric_exposed"] = True
            outcome["reference_answers_exposed"] = False
            outcome["source_state_hash"] = state_hash
            outcome["source_position"] = position
            outcome["benchmark"] = benchmark
            write_json(complete, outcome)
            return self._result_record(benchmark, task_id, outcome, position, started)
        finally:
            frozen.close()

    @staticmethod
    def _result_record(benchmark, task_id, outcome, position, started, resumed=False):
        feedback = outcome.get("evaluation") or {}
        usage = summarize_outcome(outcome)
        return {
            "task_id": task_id,
            "benchmark": benchmark,
            "status": "complete" if feedback.get("complete") is True else "incomplete",
            "score": feedback.get("score"),
            "complete": feedback.get("complete") is True,
            "source_position": position,
            "source_state_hash": outcome.get("source_state_hash"),
            "run_dir": outcome.get("run_dir"),
            "execution_seconds": max(0.0, time.monotonic() - started),
            "token_usage": usage,
            "model_calls": usage.get("model_calls", 0),
            "failed_model_calls": sum(1 for row in (outcome.get("budget") or {}).get("records", [])
                                       if row.get("kind") == "model" and row.get("error")),
            "resumed": bool(resumed),
            "test_adapted": True,
            "rubric_exposed": True,
            "reference_answers_exposed": False,
            "finished_at": utc_now(),
        }

    def run_benchmark(self, benchmark):
        records = self._load_records(benchmark)
        task_ids = self._task_ids(benchmark)
        if self.limit is not None:
            task_ids = task_ids[: self.limit]
        pending = []
        for task_id in task_ids:
            previous = records.get(task_id)
            if previous and previous.get("status") == "complete":
                continue
            if previous and previous.get("status") == "failed" and not self.retry_failed:
                continue
            pending.append(task_id)
        def run(task_id):
            try:
                return self._run_one(benchmark, task_id)
            except Exception as exc:  # preserve one-slot failure and continue
                return {"task_id": task_id, "benchmark": benchmark, "status": "failed",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "source_position": self._snapshots[benchmark][2],
                        "source_state_hash": self._snapshots[benchmark][1],
                        "test_adapted": True, "rubric_exposed": True,
                        "reference_answers_exposed": False, "finished_at": utc_now()}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            active = {pool.submit(run, task_id): task_id for task_id in pending}
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    task_id = active.pop(future)
                    records[task_id] = future.result()
                    with self._journal_lock:
                        self._save_records(benchmark, records)
        with self._journal_lock:
            self._save_records(benchmark, records)
        return records

    def run(self):
        result = {}
        for benchmark in self.benchmarks:
            result[benchmark] = self.run_benchmark(benchmark)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path,
                        help="frozen launch JSON that binds data, evidence and native config")
    parser.add_argument("--source-campaign", required=True, type=Path,
                        help="tds_fix31 (or another immutable campaign directory/database)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmarks", default=",".join(ALL_BENCHMARKS),
                        help="comma-separated benchmark names")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional per-benchmark TEST task cap")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    names = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    runner = TestAdaptedFinalRunner(args.launch, args.source_campaign, args.output,
                                    benchmarks=names, workers=args.workers,
                                    limit=args.limit, retry_failed=args.retry_failed)
    records = runner.run()
    summary = {name: {"tasks": len(rows),
                      "complete": sum(row.get("complete") is True for row in rows.values()),
                      "failed": sum(row.get("status") == "failed" for row in rows.values()),
                      "mean_score": (sum(row["score"] for row in rows.values()
                                          if isinstance(row.get("score"), (int, float))) /
                                     max(1, sum(isinstance(row.get("score"), (int, float))
                                                for row in rows.values())))}
               for name, rows in records.items()}
    write_json(args.output / "test_adapted_summary.json", {"test_adapted": True,
              "reference_answers_exposed": False, "summary": summary,
              "generated_at": utc_now()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
