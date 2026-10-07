"""Complete the frozen v7 DRB-II public evidence directory with retries.

This helper only performs deterministic public Bing RSS/Jina retrieval.  It
never calls a model API and refuses to write a manifest until every selected
v7 task has a valid evidence pack.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import build_evidence_manifest, evidence_pack_path, load_evidence_pack
from scripts.prepare_public_bing_evidence import _build_one


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("paper/experiments/joint_task_splits_v7_ours_only.json"))
    parser.add_argument("--data", type=Path, default=Path("dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 1 <= args.attempts <= 10:
        raise ValueError("workers and attempts out of range")
    root = Path(__file__).resolve().parents[1]
    split = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_ids = []
    for part in ("evolution", "validation", "test"):
        selected_ids.extend(split["memberships"]["deepresearch_bench_ii"][part])
    dataset = load_benchmark("deepresearch_bench_ii", str(args.data), available_tools=[])
    selected = {task_id: dataset.tasks[task_id] for task_id in dict.fromkeys(selected_ids)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def valid(task_id):
        try:
            load_evidence_pack(evidence_pack_path(output, task_id), selected[task_id])
            return True
        except Exception:
            return False

    pending = [task_id for task_id in selected if not valid(task_id)]
    log = []
    for attempt in range(1, args.attempts + 1):
        if not pending:
            break
        next_pending = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_build_one, selected[task_id], output, max_pages=args.max_pages): task_id
                       for task_id in pending}
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    log.append({"task_id": task_id, "attempt": attempt, "status": result})
                except Exception as exc:
                    next_pending.append(task_id)
                    log.append({"task_id": task_id, "attempt": attempt,
                                "status": "failed", "error_type": type(exc).__name__,
                                "error": str(exc)[:300]})
        pending = [task_id for task_id in next_pending if not valid(task_id)]
        if pending:
            time.sleep(min(30, 2 ** (attempt - 1)))

    (output / "retry_log.json").write_text(json.dumps({"selected": len(selected), "pending": pending,
                                                          "events": log}, ensure_ascii=True, indent=2) + "\n",
                                                     encoding="utf-8")
    if pending:
        raise RuntimeError(json.dumps({"pending": pending}, ensure_ascii=True))
    manifest = build_evidence_manifest(selected, output)
    print(json.dumps({"tasks": len(selected), "manifest_sha256": manifest["manifest_sha256"],
                      "output": str(output)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
