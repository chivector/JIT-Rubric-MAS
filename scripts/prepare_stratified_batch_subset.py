"""Freeze or verify a public-only domain-balanced independent batch amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jit_mas.independent_batch_protocol import build_protocol, validate_protocol
from jit_mas.stratified_subset import (
    build_stratified_subset, render_assignments, validate_stratified_subset,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_rr_domains(parent, path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != parent["benchmarks"]["researchrubrics"]["dataset_sha256"]:
        raise ValueError("ResearchRubrics raw bytes differ from the pinned parent release")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(({"task_id": row["sample_id"], "domain": row["domain"]}
                   for row in rows), key=lambda row: row["task_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rr-data", type=Path,
                        default=Path("outputs/researchrubrics_live_20260930/processed_data.jsonl"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    parent = read_json(args.directory / "joint_task_splits_v4.json")
    previous = read_json(args.directory / "joint_task_splits_v5.json")
    exposures = read_json(args.directory / "test_exposure_20261004.json")
    output = args.output_dir or args.directory
    manifest_path = output / "joint_task_splits_v6_stratified.json"
    if args.check:
        document = validate_stratified_subset(read_json(manifest_path), parent, previous, exposures)
    else:
        document = build_stratified_subset(parent, previous,
                                           public_rr_domains(parent, args.rr_data), exposures)
        validate_stratified_subset(document, parent, previous, exposures)
    protocol = build_protocol(document)
    validate_protocol(protocol, document)
    files = {
        "joint_task_splits_v6_stratified.json": json.dumps(
            document, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        "task_assignments_v6_stratified.md": render_assignments(document),
        "independent_protocol_v6_stratified.json": json.dumps(
            protocol, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
    }
    for name, content in files.items():
        path = output / name
        if args.check or path.exists():
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Frozen stratified artifact differs from reconstruction: {name}")
    if not args.check:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = output / name
            if not path.exists():
                path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"valid": True, "manifest_sha256": document["manifest_sha256"],
                      "totals": document["totals"], "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
