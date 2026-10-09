"""Run a frozen, one-call public generation subset for IFEval or IFBench.

The command never retries, edits, or rescored-guides an answer.  It generates
from ``PublicTask.question`` and ``constraints`` only, then invokes the pinned
checker after generation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.budget import MeteredModel
from jit_mas.compact_generation import (generate_one, make_ledger, frozen_identity,
                                        code_identity, SYSTEM_PROMPT, PROMPT_VERSION, safe_error)
from jit_mas.config import MASConfig
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.pipeline import convert_feedback, write_json, code_fingerprint
from jit_mas.request_policy import RequestPolicyModel, request_gate
from jit_mas.schemas import digest, utc_now
from jit_mas.benchmarks import load_benchmark
from scripts.models.openai_server import OpenAIServerModel


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_identity() -> dict:
    files = ("jit_mas/compact_generation.py", "jit_mas/public_literal_constraints.py",
             "jit_mas/public_literal_slots.py", "jit_mas/public_numeric_template.py",
             "jit_mas/public_numeric_slots.py", "jit_mas/public_word_slots.py",
             "jit_mas/budget.py", "jit_mas/request_policy.py", "jit_mas/config.py",
             "jit_mas/benchmarks.py", "jit_mas/instruction_checkers.py",
             "jit_mas/schemas.py", "jit_mas/pipeline.py",
             "scripts/models/openai_server.py", "scripts/models/base.py",
             "scripts/kernel/token_counter.py", "scripts/run_compact_benchmark_subset.py")
    hashes = {name: _hash_file(ROOT / name) for name in files}
    return {"files": hashes, "sha256": digest(hashes),
            "runtime_code_fingerprint": code_fingerprint()}


def _dataset_file(path: Path, benchmark: str) -> Path:
    if path.is_dir():
        path = path / {"ifeval": "ifeval_input_data.jsonl",
                       "ifbench": "IFBench_test.jsonl"}[benchmark]
    if not path.is_file():
        raise ValueError("--dataset must be an existing dataset file or directory")
    return path


def _load_config(path: Path) -> MASConfig:
    raw = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("--config must contain an object")
    return MASConfig.model_validate(raw)


def _model(config: MASConfig, ledger):
    spec = config.models.get("exec")
    if spec is None:
        raise ValueError("--config must define models.exec")
    spec.check("exec")
    options = {}
    if spec.frequency_penalty is not None:
        options["frequency_penalty"] = spec.frequency_penalty
    if spec.thinking is not None:
        options["extra_body"] = {"thinking": {"type": spec.thinking}}
    if spec.reasoning_effort is not None:
        options["reasoning_effort"] = spec.reasoning_effort
    provider = OpenAIServerModel(model_id=spec.model, api_base=spec.endpoint,
                                 api_key=os.environ[spec.key_env],
                                 temperature=spec.temperature, max_attempts=1,
                                 http_trust_env=spec.http_trust_env, **options)
    provider.client = provider.client.with_options(max_retries=0, timeout=spec.timeout)
    wrapped = RequestPolicyModel(provider,
        gate=request_gate(spec.max_inflight_requests or config.max_inflight_requests,
                          endpoint=spec.endpoint), ledger=ledger, timeout=spec.timeout,
        expected_model=spec.expected_response_model)
    return MeteredModel(wrapped, ledger, "inference", "compact-generation",
                        spec.max_tokens, context_window=spec.context_window,
                        context_margin=spec.context_margin,
                        context_policy=spec.context_policy)


def _selected_ids(dataset, requested, count):
    if count is not None and count < 1:
        raise ValueError("--count must be positive")
    values = []
    if count is not None:
        values.extend(list(dataset.tasks)[:count])
    if requested:
        values.extend(requested)
    if not values:
        raise ValueError("provide --task-id or --count")
    selected = []
    for task_id in values:
        if task_id not in dataset.tasks:
            raise ValueError(f"Unknown task id: {task_id}")
        if task_id not in selected:
            selected.append(task_id)
    return selected


def _record_without_answer(record):
    safe = copy.deepcopy(record)
    safe.pop("answer", None)
    safe.pop("raw_response", None)
    return safe


def run(args) -> dict:
    if args.benchmark not in {"ifeval", "ifbench"}:
        raise ValueError("--benchmark must be ifeval or ifbench")
    config_path = Path(args.config).resolve()
    data_path = _dataset_file(Path(args.dataset).resolve(), args.benchmark)
    dataset = load_benchmark(args.benchmark, data_path)
    checker = PinnedInstructionChecker(Path(args.checker_source).resolve(), args.benchmark)
    checker.assert_frozen()
    task_ids = _selected_ids(dataset, args.task_id, args.count)
    config = _load_config(config_path)
    if "exec" not in config.models:
        raise ValueError("--config must define models.exec")
    config.models["exec"].check("exec")
    config_hash = _hash_file(config_path)
    dataset_hash = _hash_file(data_path)
    code = _code_identity()
    prompt_identity = digest({"version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
                              "benchmark": args.benchmark,
                              "numeric": args.numeric_construction,
                              "positional": args.positional_construction,
                              "literal": args.literal_construction,
                              "numeric_layout": args.numeric_layout})
    identity = frozen_identity(config_bytes=config_path.read_bytes(),
        dataset_sha256=dataset_hash, checker_identity=checker.identity,
        prompt_hash=prompt_identity, runner_sha256=code_identity(Path(__file__)))
    manifest = {"schema": "compact-generation-run-v1", "started_at": utc_now(),
                "benchmark": args.benchmark, "task_ids": task_ids,
                "dataset_sha256": dataset_hash, "config_sha256": config_hash,
                "code_identity": code,
                "generation_settings": config.models["exec"].model_dump(mode="json",
                    exclude={"endpoint", "key_env"}),
                "resolved_configuration_sha256": digest(config.model_dump(mode="json")),
                "checker_identity": checker.identity, "identity": identity,
                "construction_flags": {"numeric": args.numeric_construction,
                                        "positional": args.positional_construction,
                                        "literal": args.literal_construction,
                                        "numeric_layout": args.numeric_layout},
                "one_call_per_task": True, "retry_policy": "none",
                "scoring": "fixed checker after generation only"}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    (output / "answers").mkdir()
    (output / "raw_responses").mkdir()
    model = None
    rows = []
    try:
        for task_id in task_ids:
            if (_hash_file(config_path) != config_hash or _hash_file(data_path) != dataset_hash
                    or _code_identity() != code):
                raise RuntimeError("Frozen configuration, dataset or code changed during the run")
            checker.assert_frozen()
            task = dataset.tasks[task_id]
            ledger = make_ledger(config.max_total_tokens, timeout=config.task_timeout or 900)
            row = {"task_id": task_id, "benchmark": args.benchmark,
                   "question_sha256": digest(task.question),
                   "constraints_sha256": digest(task.constraints),
                   "private_record_sha256": digest(dataset.private_records[task_id])}
            try:
                model = _model(config, ledger)
                generated = generate_one(task, model, ledger,
                    numeric=args.numeric_construction,
                    positional=args.positional_construction,
                    literal=args.literal_construction,
                    numeric_layout=args.numeric_layout,
                    max_tokens=config.models["exec"].max_tokens)
                row.update(generated)
                if generated["raw_response"] is not None:
                    raw_path = output / "raw_responses" / f"{digest(task_id)}.txt"
                    raw_text = generated["raw_response"]
                    if not isinstance(raw_text, str):
                        raw_text = json.dumps(raw_text, ensure_ascii=False)
                    raw_path.write_text(raw_text, encoding="utf-8")
                    row["raw_response_file"] = str(raw_path.relative_to(output))
                if generated["answer"] is not None:
                    answer_path = output / "answers" / f"{digest(task_id)}.txt"
                    answer_path.write_text(generated["answer"], encoding="utf-8")
                    row["answer_file"] = str(answer_path.relative_to(output))
                if generated["complete"]:
                    try:
                        raw = dataset.evaluator(None, checker=checker).evaluate(
                            generated["answer"], ground_truth=task_id,
                            private_record=dataset.private_records[task_id])
                        feedback = convert_feedback(raw)
                        row["score_result"] = {"complete": feedback.complete,
                                                "score": feedback.score,
                                                "evaluation": feedback.model_dump(mode="json")}
                    except Exception as exc:
                        row["score_result"] = {"complete": False, "score": None,
                                                "status": "evaluation_failed",
                                                "error": safe_error(exc)}
                else:
                    row["score_result"] = {"complete": False, "score": None,
                                            "status": ("generation_incomplete" if generated["answer"]
                                                       is not None else "generation_failed")}
            except Exception as exc:
                row.setdefault("answer", None)
                row.setdefault("answer_hash", None)
                row.update({"error": safe_error(exc),
                            "budget": ledger.snapshot()})
                row.setdefault("score_result", {"complete": False, "score": None,
                                                  "status": "generation_failed"})
            finally:
                if model is not None:
                    try:
                        model.model.client.close()
                    except Exception as exc:
                        row["client_close_error"] = safe_error(exc)
                    model = None
            write_json(output / "records" / f"{digest(task_id)}.json",
                       _record_without_answer(row))
            rows.append(_record_without_answer(row))
    finally:
        manifest["finished_at"] = utc_now()
        manifest["records"] = rows
        scores = [row["score_result"]["score"] for row in rows
                  if row.get("score_result", {}).get("complete")]
        requested = len(task_ids)
        mean_all_tasks = sum(scores) / requested
        manifest["summary"] = {"count": len(rows), "requested": requested,
                                "attempted": len(rows), "scored": len(scores),
                                "unattempted": requested - len(rows),
                                "failed": requested - len(scores),
                                "completion_rate": len(scores) / requested,
                                "mean_all_tasks": mean_all_tasks, "mean": mean_all_tasks,
                                "mean_scored": sum(scores) / len(scores) if scores else None,
                                "generation_calls": sum(row.get("budget", {}).get("model_calls", 0)
                                                         for row in rows),
                                "generation_tokens": sum(row.get("budget", {}).get("tokens", 0)
                                                          for row in rows)}
        write_json(output / "report.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checker-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True, choices=["ifeval", "ifbench"])
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--count", type=int)
    parser.add_argument("--numeric-construction", action="store_true")
    parser.add_argument("--positional-construction", action="store_true")
    parser.add_argument("--literal-construction", action="store_true")
    parser.add_argument("--numeric-layout", choices=["array", "template"], default="array")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"compact generation: {safe_error(exc)['message']}\n")
    print(json.dumps(result["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
