"""Freeze or check a metadata-only v5 subset; no model calls or evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.joint_subset import (build_subset, protocol_for, render_assignments,
                                  validate_subset, workload)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_drbii_strata(parent, path):
    dataset = load_benchmark("deepresearch_bench_ii", path)
    if dataset.dataset_sha256 != parent["benchmarks"]["deepresearch_bench_ii"]["dataset_sha256"]:
        raise ValueError("DRBII raw bytes differ from the pinned parent release")
    return [{"task_id": key, "language": meta["language"], "theme": meta["theme"]}
            for key, meta in sorted(dataset.public_metadata.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--parent", type=Path,
                        help="Pinned v4 manifest; defaults to --directory/joint_task_splits_v4.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--drbii-data", type=Path,
                        default=Path("dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl"))
    parser.add_argument("--check", type=Path, help="Reconstruct with independent public anchors; no raw data needed")
    args = parser.parse_args(argv)
    parent = read_json(args.parent or args.directory / "joint_task_splits_v4.json")
    if args.check:
        document = validate_subset(read_json(args.check), parent)
        if read_json(args.check.parent / "joint_protocol_v5.json") != protocol_for(document):
            raise ValueError("Subset protocol differs from fixed v5 rules")
        if (args.check.parent / "task_assignments_v5.md").read_text(encoding="utf-8") != render_assignments(document):
            raise ValueError("Readable subset assignments differ from the JSON manifest")
        print(json.dumps({"valid": True, "totals": document["totals"],
                          "workload": workload(), "model_calls": 0}))
        return 0
    document = build_subset(parent, public_drbii_strata(parent, args.drbii_data))
    validate_subset(document, parent)
    output = args.output_dir or args.directory
    files = {
        "joint_task_splits_v5.json": json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        "joint_protocol_v5.json": json.dumps(protocol_for(document), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        "task_assignments_v5.md": render_assignments(document),
    }
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite different frozen subset artifacts; version the amendment")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = output / name
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"manifest_sha256": document["manifest_sha256"],
                      "totals": document["totals"], "model_calls": 0}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
