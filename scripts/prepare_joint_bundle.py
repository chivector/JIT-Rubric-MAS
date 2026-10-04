"""Build a content-addressed input bundle for the frozen joint v5 run.

The command is deliberately a *registration* step.  It never constructs a
model client and never executes a task.  It projects the frozen six-source
membership into the split format consumed by :func:`make_pipeline`, checks
that every selected task exists in its pinned data file, and records the
byte identity of all files that will be used by the executor.

Evidence and provider preflight are fail-closed by default.  ``--allow-
incomplete`` is intended only for an offline audit or for producing a report
of missing materials; an incomplete bundle cannot be used for a formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jit_mas.benchmarks import load_benchmark
from jit_mas.schemas import digest
from scripts.run_benchmark_experiment import evidence_identity, file_hash


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "researchrubrics", "deepsearchqa", "writingbench",
    "deepresearch_bench_ii", "ifeval", "ifbench",
)
SOURCE_BENCHMARKS = {"researchrubrics", "deepsearchqa", "deepresearch_bench_ii"}
CHECKER_BENCHMARKS = {"ifeval", "ifbench"}

DEFAULT_DATA = {
    "researchrubrics": ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl",
    "deepsearchqa": ROOT / "dataset/deepsearchqa/DSQA-pinned-v3.csv",
    "writingbench": ROOT / "dataset/writingbench/benchmark_all.jsonl",
    "deepresearch_bench_ii": ROOT / "dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl",
    "ifeval": ROOT / "dataset/ifeval/ifeval_input_data.jsonl",
    "ifbench": ROOT / "dataset/ifbench/IFBench_test.jsonl",
}
DEFAULT_CHECKER = {
    "ifeval": ROOT / "outputs/independent_v5_preparation/checkers/Google-IFEval-e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
    "ifbench": ROOT / "outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _relative(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix() if path.resolve().is_relative_to(base.resolve()) else str(path.resolve())


def _parse_assignments(values: list[str], *, allowed: set[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use BENCHMARK=PATH")
        name, raw = value.split("=", 1)
        if name not in allowed:
            raise ValueError(f"Unknown {label} benchmark: {name}")
        if name in result:
            raise ValueError(f"Duplicate {label} benchmark: {name}")
        result[name] = _resolve(raw)
    return result


def _manifest_membership(manifest: dict[str, Any], benchmark: str, partition: str) -> list[str]:
    value = manifest.get("memberships", {}).get(benchmark, {}).get(partition)
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise ValueError(f"Invalid frozen {benchmark}/{partition} membership")
    return list(value)


def _split_projection(manifest: dict[str, Any], benchmark: str) -> dict[str, Any]:
    """Return a SplitManifest-compatible projection of the global v5 split."""
    return {
        "schema_version": "jit-compose-joint-benchmark-split-v5",
        # v5 records a textual membership-generation label.  SplitManifest's
        # runtime seed is an integer order seed; this projection is consumed
        # only for task membership, so use the registered first order seed.
        "seed": 20261001,
        "evolution": _manifest_membership(manifest, benchmark, "evolution"),
        "validation": _manifest_membership(manifest, benchmark, "validation"),
        "test": _manifest_membership(manifest, benchmark, "test"),
        "stream": [],
    }


def _pack_ids(directory: Path) -> set[str]:
    """Recover task IDs from immutable evidence filenames only.

    Evidence filenames are SHA256(task_id).json.  The original task IDs are
    intentionally not recoverable from filenames, so coverage is checked by
    using ``manifest.json`` when present and otherwise reported as unknown.
    """
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return set()
    doc = _read(manifest)
    ids = doc.get("task_ids")
    if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
        return set(ids)
    tasks = doc.get("tasks")
    if isinstance(tasks, dict) and all(isinstance(x, str) for x in tasks):
        return set(tasks)
    return set()


def _evidence_record(directory: Path, required_ids: list[str], allow_incomplete: bool) -> dict[str, Any]:
    if not directory.is_dir():
        if allow_incomplete:
            return {"path": str(directory), "exists": False, "complete": False, "missing_count": len(required_ids)}
        raise FileNotFoundError(f"Missing evidence directory: {directory}")
    identity = evidence_identity(directory)
    listed = _pack_ids(directory)
    missing = sorted(set(required_ids) - listed) if listed else list(required_ids)
    record = {"path": str(directory), "exists": True, "identity": identity,
              "manifest_task_count": len(listed), "required_task_count": len(required_ids),
              "missing_count": len(missing), "complete": not missing,
              "missing_task_ids": missing if len(missing) <= 20 else {"count": len(missing), "sha256": hashlib.sha256("\n".join(missing).encode()).hexdigest()}}
    if missing and not allow_incomplete:
        raise ValueError(f"Evidence coverage incomplete ({len(missing)} missing tasks): {directory}")
    return record


def build_bundle(*, protocol: Path, manifest: Path, config: Path, output_dir: Path,
                 data: dict[str, Path], evidence: dict[str, Path], checker: dict[str, Path],
                 preflight: Path | None = None, served_models: Path | None = None,
                 allow_incomplete: bool = False) -> dict[str, Any]:
    protocol_doc, manifest_doc = _read(protocol), _read(manifest)
    if protocol_doc.get("version") != "jit-compose-joint-protocol-v5" or manifest_doc.get("version") != "jit-compose-joint-subset-v5":
        raise ValueError("Only frozen joint v5 protocol and manifest are accepted")
    if protocol_doc.get("split_manifest_sha256") != manifest_doc.get("manifest_sha256"):
        raise ValueError("Protocol/manifest hash binding mismatch")
    if digest({k: v for k, v in protocol_doc.items() if k != "protocol_sha256"}) != protocol_doc.get("protocol_sha256"):
        raise ValueError("Protocol self-hash mismatch")
    if digest({k: v for k, v in manifest_doc.items() if k != "manifest_sha256"}) != manifest_doc.get("manifest_sha256"):
        raise ValueError("Manifest self-hash mismatch")
    missing_data = [name for name in BENCHMARKS if name not in data]
    if missing_data:
        raise ValueError(f"Missing data bindings: {', '.join(missing_data)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_records: dict[str, Any] = {}
    readiness: dict[str, Any] = {"data": {}, "evidence": {}, "checkers": {}}
    for name in BENCHMARKS:
        path = data[name]
        if not path.is_file():
            if allow_incomplete:
                readiness["data"][name] = {"exists": False}
            else:
                raise FileNotFoundError(f"Missing {name} data: {path}")
        else:
            dataset = load_benchmark(name, path)
            expected = set(_manifest_membership(manifest_doc, name, "evolution")) | set(_manifest_membership(manifest_doc, name, "validation")) | set(_manifest_membership(manifest_doc, name, "test"))
            missing = sorted(expected - set(dataset.tasks))
            if missing and not allow_incomplete:
                raise ValueError(f"{name} data misses {len(missing)} frozen task IDs")
            actual_hash = file_hash(path)
            expected_hash = manifest_doc.get("benchmarks", {}).get(name, {}).get("dataset_sha256")
            hash_match = expected_hash is None or actual_hash == expected_hash
            if not hash_match and not allow_incomplete:
                raise ValueError(f"{name} data hash differs from frozen manifest")
            readiness["data"][name] = {"exists": True, "sha256": actual_hash, "expected_sha256": expected_hash,
                                        "hash_match": hash_match, "task_count": len(dataset.tasks), "missing_selected_count": len(missing)}
        split_path = output_dir / f"{name}.runtime_split.json"
        split_doc = _split_projection(manifest_doc, name)
        encoded = json.dumps(split_doc, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        if split_path.exists() and split_path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace existing split projection: {split_path}")
        split_path.write_text(encoded, encoding="utf-8", newline="\n")

        record: dict[str, Any] = {"data": _relative(path, output_dir), "splits": _relative(split_path, output_dir)}
        if name in SOURCE_BENCHMARKS:
            if name not in evidence:
                if not allow_incomplete:
                    raise ValueError(f"Missing evidence binding for {name}")
                readiness["evidence"][name] = {"exists": False, "complete": False, "missing_binding": True}
            else:
                required = _manifest_membership(manifest_doc, name, "evolution") + _manifest_membership(manifest_doc, name, "validation") + _manifest_membership(manifest_doc, name, "test")
                ev = _evidence_record(evidence[name], required, allow_incomplete)
                readiness["evidence"][name] = ev
                record["evidence_dir"] = _relative(evidence[name], output_dir)
        if name in CHECKER_BENCHMARKS:
            root = checker.get(name)
            if root is None or not root.is_dir():
                if not allow_incomplete:
                    raise FileNotFoundError(f"Missing pinned checker root for {name}")
                readiness["checkers"][name] = {"exists": False}
            else:
                readiness["checkers"][name] = {"exists": True, "identity": evidence_identity(root)}
                record["checker_source_root"] = _relative(root, output_dir)
        benchmark_records[name] = record

    preflight_doc = _read(preflight) if preflight else {"passed": False, "synthetic_only": True, "missing": "provider preflight"}
    if not isinstance(preflight_doc, dict):
        raise ValueError("Provider preflight must be a JSON object")
    served = _read(served_models) if served_models else {}
    bundle = {
        "schema": "joint-execution-bundle-v2",
        "protocol": _relative(protocol, output_dir), "manifest": _relative(manifest, output_dir),
        "config": _relative(config, output_dir), "benchmarks": benchmark_records,
        "preflight": preflight_doc, "served_models": served,
        "seed": manifest_doc.get("membership_seed", 0),
        "readiness": readiness,
        "formal_ready": (
            all(v.get("exists") and v.get("hash_match", False) and not v.get("missing_selected_count", 0) for v in readiness["data"].values())
            and all(readiness["evidence"].get(name, {}).get("complete") is True for name in SOURCE_BENCHMARKS)
            and all(readiness["checkers"].get(name, {}).get("exists") is True for name in CHECKER_BENCHMARKS)
            and preflight_doc.get("passed") is True
            and preflight_doc.get("synthetic_only") is not True
        ),
    }
    bundle_path = output_dir / "bundle.json"
    encoded = json.dumps(bundle, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if bundle_path.exists() and bundle_path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"Refusing to replace existing bundle: {bundle_path}")
    bundle_path.write_text(encoded, encoding="utf-8", newline="\n")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper/experiments/joint_protocol_v5.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper/experiments/joint_task_splits_v5.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", action="append", default=[], metavar="BENCHMARK=PATH")
    parser.add_argument("--evidence", action="append", default=[], metavar="BENCHMARK=PATH")
    parser.add_argument("--checker", action="append", default=[], metavar="BENCHMARK=PATH")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--served-models", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = dict(DEFAULT_DATA)
        data.update(_parse_assignments(args.data, allowed=set(BENCHMARKS), label="--data"))
        evidence = _parse_assignments(args.evidence, allowed=SOURCE_BENCHMARKS, label="--evidence")
        checker = dict(DEFAULT_CHECKER)
        checker.update(_parse_assignments(args.checker, allowed=CHECKER_BENCHMARKS, label="--checker"))
        result = build_bundle(protocol=_resolve(args.protocol), manifest=_resolve(args.manifest), config=_resolve(args.config), output_dir=_resolve(args.output_dir), data=data, evidence=evidence, checker=checker, preflight=_resolve(args.preflight) if args.preflight else None, served_models=_resolve(args.served_models) if args.served_models else None, allow_incomplete=args.allow_incomplete)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"Bundle preparation: {exc}\n")
    print(json.dumps({"bundle": str((_resolve(args.output_dir) / "bundle.json")), "formal_ready": result["formal_ready"], "readiness": result["readiness"]}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
