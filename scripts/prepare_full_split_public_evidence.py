"""Prepare deterministic public evidence for an official full-test split.

This helper selects task IDs from a frozen split manifest and delegates to the
existing Bing RSS/Jina Reader evidence builder.  It never reads private
records, reference answers, or evaluator outputs.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import build_evidence_manifest
from scripts.prepare_public_bing_evidence import _build_one


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        choices=("deepsearchqa", "deepresearch_bench_ii", "researchrubrics"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True,
                        help="Frozen split JSON containing runtime_split_manifest.test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pages", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32 or not 1 <= args.max_pages <= 8:
        parser.error("workers must be 1..32 and max-pages must be 1..8")
    dataset = load_benchmark(args.benchmark, args.data, available_tools=[])
    split = json.loads(Path(args.split).read_text(encoding="utf-8-sig"))
    selected_ids = list(split["runtime_split_manifest"]["test"])
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Split test inventory must be nonempty and unique")
    unknown = set(selected_ids) - set(dataset.tasks)
    if unknown:
        raise ValueError(f"Split contains unknown task IDs: {sorted(unknown)[:3]}")
    tasks = {task_id: dataset.tasks[task_id] for task_id in selected_ids}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = {"created": 0, "reused": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, task, output, max_pages=args.max_pages): task_id
                   for task_id, task in tasks.items()}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as error:
                counts["failed"] += 1
                failures.append({"task_id": task_id, "type": type(error).__name__,
                                 "message": str(error)[:300]})
    manifest = build_evidence_manifest(tasks, output)
    summary = {"benchmark": args.benchmark, "split": str(Path(args.split).resolve()),
               "task_count": len(tasks), **counts,
               "manifest_sha256": manifest["manifest_sha256"],
               "public_only": True, "answer_data_used": False,
               "failures": failures}
    (output / "full_split_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
