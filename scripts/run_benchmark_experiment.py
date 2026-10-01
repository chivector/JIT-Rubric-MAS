"""Registered benchmark evolution with periodic whole-validation checkpoints."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

from jit_mas.benchmarks import BENCHMARK_NAMES, load_benchmark
from jit_mas.checkpoints import CheckpointIntegrityError, CheckpointRunner
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_profile import validate_execution_profile
from jit_mas.experiment_splits import ORDER_SEEDS, validate_split
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import digest
from scripts.eval.config import load_dotenv
from scripts.run_jit_mas import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "paper" / "experiments" / "benchmark_suite_v3.json"
EVOLVING_METHODS = ("ours_selected", "no_explicit_rubrics", "global_only_planning", "global_only_attribution")


def file_hash(path):
    content = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content.update(chunk)
    return content.hexdigest()


def dataset_file(benchmark, data):
    path = Path(data).resolve()
    if path.is_dir():
        path /= {"researchrubrics": "processed_data.jsonl", "deepsearchqa": "DSQA-full.csv",
                 "deepresearch_bench_ii": "tasks_and_rubrics.jsonl"}[benchmark]
    return path


def evidence_identity(directory):
    location = Path(directory).resolve()
    if not location.is_dir():
        raise ValueError("--evidence-dir must be an existing frozen evidence directory")
    files = {path.relative_to(location).as_posix(): file_hash(path)
             for path in sorted(location.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Frozen evidence directory is empty")
    return {"files": files, "sha256": digest(files)}


def load_registration(args, *, available_tools=()):
    if not args.benchmark or not args.data or not args.splits:
        raise ValueError("Registered runs require --benchmark, --data and --splits")
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    if suite.get("version") != "jit-compose-benchmark-suite-v3":
        raise ValueError("Expected the registered benchmark suite v3")
    if list(suite.get("order_seeds", [])) != list(ORDER_SEEDS):
        raise ValueError("Suite evolution order seeds differ from the registered implementation")
    specification = suite.get("benchmarks", {}).get(args.benchmark)
    if specification is None:
        raise ValueError("Benchmark is not registered in this suite")
    dataset = load_benchmark(args.benchmark, args.data, available_tools=available_tools)
    if suite.get("release_hashes", {}).get(args.benchmark) != dataset.dataset_sha256:
        raise ValueError("Dataset bytes do not match the suite's pinned release hash")
    split = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    manifest = validate_split(split, dataset, specification)
    if split["counts"] != specification["counts"] or len(dataset.tasks) != specification["expected_tasks"]:
        raise ValueError("Dataset or split counts differ from the registered benchmark")
    return suite, specification, dataset, split, manifest


def _method_config(config, method):
    expected = {"explicit_rubrics": method != "no_explicit_rubrics",
                "local_planning": method != "global_only_planning",
                "local_attribution": method != "global_only_attribution"}
    if not config.persistent_experience or config.fixed_team is not None:
        raise ValueError("Evolving conditions require direct persistent experience and a dynamic team")
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"Config {key} does not match registered method {method}")


@dataclass
class ExperimentEnvironment:
    args: argparse.Namespace
    config: MASConfig
    store: ExperienceStore
    pipeline: object
    dataset: object
    suite: dict
    split: dict
    identity: dict
    evolution_ids: list[str]
    validation_ids: list[str]
    lower_bounds: dict[str, float]
    batch_size: int
    output_dir: Path
    smoke: bool

    def close(self):
        self.store.close()

    def assert_frozen(self):
        try:
            changed = (code_fingerprint() != self.identity["code_fingerprint"]
                       or file_hash(__file__) != self.identity["runner_sha256"]
                       or self.config.model_dump(mode="json") != self.identity["configuration"])
            if not self.smoke:
                changed = changed or any(file_hash(path) != expected for path, expected in (
                    (self.args.suite, self.identity["suite_file_sha256"]),
                    (self.args.config, self.identity["configuration_file_sha256"]),
                    (self.args.splits, self.identity["split_file_sha256"]),
                    (dataset_file(self.args.benchmark, self.args.data), self.identity["dataset_sha256"])))
                changed = changed or evidence_identity(self.args.evidence_dir) != self.identity["evidence"]
        except (OSError, ValueError) as exc:
            raise CheckpointIntegrityError("Frozen experiment inputs became unavailable or invalid") from exc
        if changed:
            raise CheckpointIntegrityError("Code, configuration, data, split or evidence changed during a frozen experiment")

    def validation_pipeline(self, frozen_store):
        kwargs = ({"fixture_models": self.pipeline.models} if self.smoke else
                  {"data": self.args.data, "splits": self.args.splits,
                   "benchmark": self.args.benchmark, "evidence_dir": self.args.evidence_dir})
        return make_pipeline(self.config, frozen_store, self.output_dir / "validation", **kwargs)

    def checkpoint_runner(self):
        def evaluate(task_id, snapshot, repeat, frozen_store):
            self.assert_frozen()
            pipeline = self.validation_pipeline(frozen_store)
            return pipeline.run_task(task_id, snapshot, mode="validate", repeat=repeat, attribution=False)

        def evolve(task_id):
            self.assert_frozen()
            return self.pipeline.run("evolve", [task_id])

        return CheckpointRunner(
            store=self.store, evolution_ids=self.evolution_ids, validation_ids=self.validation_ids,
            output_dir=self.output_dir / "checkpoints", identity=self.identity,
            lower_bounds=self.lower_bounds,
            evolve=evolve, evaluate=evaluate,
            batch_size=self.batch_size, repeats=2,
            minimum_completion=self.suite.get("checkpoint_selection", {}).get("minimum_complete_fraction", 0.9))


def build_environment(args):
    """Validate a frozen registration before constructing any native model provider."""
    output = Path(args.output).resolve()
    smoke = bool(args.smoke)
    if not smoke:
        if not args.config:
            raise ValueError("Native registered evolution requires --config")
        if not args.evidence_dir:
            raise ValueError("Native registered evolution requires --evidence-dir")
        if not args.unsafe_local:
            raise ValueError("Native generated-code execution requires explicit --unsafe-local")
        if args.run_id not in (0, 1, 2):
            raise ValueError("--run-id must be one of the three registered runs: 0, 1, 2")
        raw = os.path.expandvars(Path(args.config).read_text(encoding="utf-8"))
        config = MASConfig.model_validate(yaml.safe_load(raw))
        if config.backend != "native_jit":
            raise ValueError("Registered results require native_jit; use --smoke for synthetic software tests")
        config.unsafe_local = True
        _method_config(config, args.method)
        suite, specification, dataset, split, manifest = load_registration(args, available_tools=config.available_tools)
        validate_execution_profile(config, suite)
        if args.method not in specification["methods"]:
            raise ValueError("Method is not registered for this benchmark")
        if not manifest.evolution or not manifest.validation:
            raise ValueError("This benchmark is transfer-only; no target evolution or validation is allowed")
        schedules = [row for row in split["evolution_schedule"] if row["run_id"] == args.run_id]
        if len(schedules) != 1 or schedules[0]["order_seed"] != ORDER_SEEDS[args.run_id]:
            raise ValueError("Missing or invalid registered evolution run order")
        evolution = schedules[0]["task_ids"]
        if len(evolution) != len(set(evolution)) or set(evolution) != set(manifest.evolution):
            raise ValueError("Registered source order must contain every evolution task exactly once")
        batch_size = specification["batch_size"]
        expected_positions = list(range(0, len(evolution) + 1, batch_size))
        if split["checkpoint_positions"] != expected_positions or len(evolution) % batch_size:
            raise ValueError("Checkpoint positions differ from the registered batch schedule")
        if schedules[0]["batches"] != [evolution[start:start + batch_size] for start in range(0, len(evolution), batch_size)]:
            raise ValueError("Frozen source batches do not match the registered source order")
        validation = manifest.validation
        bounds = {key: dataset.lower_bounds[key] for key in validation}
        evidence = evidence_identity(args.evidence_dir)
        identity = {"software_test_only": False, "benchmark": args.benchmark, "method": args.method,
                    "run_id": args.run_id, "order_seed": ORDER_SEEDS[args.run_id],
                    "dataset_sha256": dataset.dataset_sha256,
                    "suite_file_sha256": file_hash(args.suite), "split_file_sha256": file_hash(args.splits),
                    "split_manifest_sha256": split["manifest_sha256"],
                    "configuration": config.model_dump(mode="json"), "configuration_file_sha256": file_hash(args.config),
                    "evidence": evidence}
    else:
        if any((args.data, args.splits, args.evidence_dir, args.config)):
            raise ValueError("--smoke uses only labelled built-in synthetic fixtures; do not supply native inputs")
        if args.method != "ours_selected":
            raise ValueError("The minimal smoke checks the full method only")
        config = MASConfig(backend="scripted")
        suite = {"checkpoint_selection": {"minimum_complete_fraction": 0.9}}
        split, dataset, evolution, validation, bounds, batch_size = {}, None, [], [], {}, 5
        identity = {"software_test_only": True, "benchmark": "synthetic-software-fixture", "method": args.method,
                    "run_id": 0, "configuration": config.model_dump(mode="json")}
    identity.update(code_fingerprint=code_fingerprint(), runner_sha256=file_hash(__file__))
    state_path = Path(args.state).resolve() if args.state else output / "experience.sqlite"
    store = ExperienceStore(state_path)
    try:
        journal_path = output / "checkpoints" / "checkpoint_journal.json"
        if not journal_path.exists() and (store.snapshot().version != 0 or store.snapshot().experiences):
            raise ValueError("A new registered trajectory must begin with an empty experience store")
        kwargs = {} if smoke else {"data": args.data, "splits": args.splits,
                                  "benchmark": args.benchmark, "evidence_dir": args.evidence_dir}
        pipeline = make_pipeline(config, store, output / "evolution", **kwargs)
        if smoke:
            evolution = list(pipeline.manifest.evolution)
            validation = list(pipeline.manifest.validation)
            bounds = {key: 0.0 for key in validation}
            split = {"runtime_split_manifest": pipeline.manifest.model_dump(mode="json")}
        environment = ExperimentEnvironment(args, config, store, pipeline, dataset, suite, split, identity,
                                            list(evolution), list(validation), bounds, batch_size, output, smoke)
        # Constructing the coordinator verifies an existing journal identity before any request.
        environment.checkpoint_runner()
        return environment
    except BaseException:
        store.close()
        raise


def export_public(args):
    suite, specification, dataset, split, manifest = load_registration(args)
    destination = Path(args.output)
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an exported public task inventory")
    if sidecar.exists():
        raise FileExistsError("Refusing to overwrite an existing public export identity")
    # This is the typed public projection, never raw dataset rows or private metadata.
    write_json(destination, [dataset.tasks[key].model_dump(mode="json") for key in sorted(dataset.tasks)])
    write_json(sidecar, {"benchmark": dataset.name, "dataset_sha256": dataset.dataset_sha256,
                        "manifest_sha256": split["manifest_sha256"], "public_file_sha256": file_hash(destination),
                        "task_count": len(dataset.tasks)})
    return {"mode": "export-public", "benchmark": dataset.name, "tasks": len(dataset.tasks),
            "output": str(destination.resolve()), "identity": str(sidecar.resolve())}


def status(output):
    path = Path(output) / "checkpoints" / "checkpoint_journal.json"
    journal = json.loads(path.read_text(encoding="utf-8"))
    return {"status": journal["status"], "source_slots": len(journal["sources"]),
            "completed_checkpoints": [row["position"] for row in journal["checkpoints"]],
            "selected_position": journal["selected_position"],
            "accounting": journal.get("accounting"), "journal_path": str(path.resolve())}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["evolve", "status", "export-public"])
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES)
    parser.add_argument("--data", help="Local pinned benchmark release; never downloaded implicitly")
    parser.add_argument("--splits", help="Frozen benchmark-split-v3 JSON")
    parser.add_argument("--run-id", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--method", choices=EVOLVING_METHODS, default="ours_selected")
    parser.add_argument("--config", help="Explicit native MASConfig YAML; credentials via environment variables")
    parser.add_argument("--state", help="Dedicated trajectory SQLite file; default OUTPUT/experience.sqlite")
    parser.add_argument("--output", required=True, help="Run directory, or JSON file for export-public")
    parser.add_argument("--evidence-dir", help="Verified immutable public evidence pack directory")
    parser.add_argument("--unsafe-local", action="store_true", help="Explicitly allow generated Python host access")
    parser.add_argument("--smoke", action="store_true", help="Offline synthetic fixture override, never a benchmark result")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    environment = None
    try:
        if args.mode == "status":
            report = status(args.output)
        elif args.mode == "export-public":
            if args.smoke:
                raise ValueError("export-public requires a real registered dataset; it issues no API calls")
            report = export_public(args)
        else:
            with redirect_stdout(sys.stderr):
                environment = build_environment(args)
                report = environment.checkpoint_runner().run()
            report.update(mode="evolve", software_test_only=environment.smoke,
                          benchmark=environment.identity["benchmark"], method=args.method, run_id=args.run_id,
                          paid_requests=0 if environment.smoke else None)
            write_json(environment.output_dir / "experiment_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.exit(2, f"Registered benchmark experiment: {exc}\n")
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
