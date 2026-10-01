"""Freeze a benchmark's public-only task manifest; never call a model."""

import argparse
import json
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.experiment_splits import build_split, cross_benchmark_overlaps, validate_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="paper/experiments/benchmark_suite_v3.json")
    parser.add_argument("--benchmark", choices=["researchrubrics", "deepsearchqa", "deepresearch_bench_ii"])
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--exposures", help="JSON list of prior-exposed public task IDs")
    parser.add_argument("--check", nargs="+", help="Validate frozen manifests and cross-benchmark exact-question overlap")
    args = parser.parse_args(argv)
    if args.check:
        documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.check]
        for document in documents:
            validate_split(document)
        overlaps = cross_benchmark_overlaps(documents)
        print(json.dumps({"valid_manifests": len(documents), "cross_benchmark_overlap_groups": len(overlaps)}))
        if overlaps:
            raise ValueError("Cross-benchmark overlap; amend registration before formal execution")
        return 0
    if not all((args.benchmark, args.data, args.output)):
        parser.error("--benchmark, --data and --output are required for preparation")
    suite_path = Path(args.suite)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    specification = suite["benchmarks"][args.benchmark]
    inherited = (json.loads((suite_path.parent / specification["inherited_split"]).read_text(encoding="utf-8"))
                 if specification.get("inherited_split") else None)
    exposed = json.loads(Path(args.exposures).read_text(encoding="utf-8")) if args.exposures else []
    if not isinstance(exposed, list) or any(not isinstance(key, str) for key in exposed):
        raise ValueError("Exposure inventory must be a list of task IDs")
    dataset = load_benchmark(args.benchmark, args.data)
    if dataset.dataset_sha256 != suite["release_hashes"][args.benchmark]:
        raise ValueError("Input file differs from the pinned author release")
    document = build_split(dataset, specification, inherited=inherited, exposed_ids=exposed)
    validate_split(document, dataset, specification)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({"benchmark": dataset.name, "counts": document["counts"],
                      "manifest_sha256": document["manifest_sha256"], "model_calls": 0}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Dataset parser errors must not echo held-out rows or secrets.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
