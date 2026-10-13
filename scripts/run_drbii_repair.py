"""Run an isolated DRBII continuation with smaller in-memory judge batches.

This wrapper keeps the v7 registration and frozen source files unchanged.  It
only adjusts the evaluator class in this process before the runner is created;
the campaign itself records a new registration hash and remains resumable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jit_mas.benchmarks import DeepResearchBenchIIEvaluator
from scripts.run_independent_batch_experiment import IndependentBatchEnvironment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("first-stage", "full"), default="full")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-attempts", type=int, default=None,
                        help="Explicitly set JIT_MAS_MODEL_ATTEMPTS before model construction")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 50:
        raise SystemExit("batch-size must be between 1 and 50")
    if args.model_attempts is not None:
        if args.model_attempts < 1:
            raise SystemExit("model-attempts must be positive")
        os.environ["JIT_MAS_MODEL_ATTEMPTS"] = str(args.model_attempts)
    DeepResearchBenchIIEvaluator.batch_size = args.batch_size
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "repair_manifest.json").write_text(
        json.dumps({
            "schema": "drbii-small-judge-batch-repair-v1",
            "source": "deepresearch_bench_ii",
            "batch_size": args.batch_size,
            "reason": "50-item judge calls produced incomplete rubric receipts; this isolated continuation uses smaller parseable calls",
            "launch": str(args.launch.resolve()),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = IndependentBatchEnvironment(args.launch.resolve(), args.output.resolve(), require_preflight=True)
    report = environment.run(phase=args.phase)
    print(json.dumps({"output": str(args.output.resolve()), "batch_size": args.batch_size,
                      "status": report.get("statuses"), "test": report.get("test")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
