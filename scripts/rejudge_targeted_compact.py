"""Re-evaluate frozen targeted-compact answers without regenerating them.

Example (credentials are read from the configuration's environment variables)::

    python -m scripts.rejudge_targeted_compact --config config.yaml --plan plan.json \
        --source tdrbii_full_20261012 --output tdrbii_rejudge --workers 8 \
        --judge-max-tokens 16384 --rubric-batch-size 10 --resume

The native benchmark evaluator, criterion text and aggregation stay unchanged.
DRBII judgments that already succeeded for the same frozen answer are retained;
only missing judgments are requested. Each attempt and its usage remain on disk.
Use --dry-run to inspect the answer inventory without loading clients or calling
an API. This script never performs evolution or answer generation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import DRBII_DIMENSIONS, _json_response
from jit_mas.budget import BudgetLedger
from jit_mas.config import NativeModels
from jit_mas.pipeline import convert_feedback
from jit_mas.schemas import digest, utc_now
from scripts.mas_baseline_methods import JudgeEnvelopeModel
from scripts.run_high_score_native_experiment import _failure, _redact
from scripts.run_targeted_compact_experiment import _clients_close, load_inputs


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    """Publish complete JSON atomically, including a credential-safe copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_redact(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _raw_evaluation(value):
    if not isinstance(value, dict):
        return None
    return value.get("raw") if isinstance(value.get("raw"), dict) else value


def _source_artifacts(source, benchmark, task_id):
    identity = digest(task_id)
    candidates = [source / "tasks" / benchmark / identity, source / "tasks" / identity]
    for directory in candidates:
        answer_path = directory / "answer.txt"
        record = read_json(directory / "record.json", {})
        if answer_path.is_file() and record.get("task_id", task_id) == task_id:
            answer = answer_path.read_text(encoding="utf-8-sig")
            if not answer.strip():
                raise ValueError("Frozen answer is empty: " + task_id)
            if record.get("answer_hash") not in (None, digest(answer)):
                raise ValueError("Frozen answer hash differs from source receipt: " + task_id)
            return {"directory": directory, "answer_file": answer_path, "answer": answer,
                    "answer_hash": digest(answer), "record": record,
                    "evaluation": read_json(directory / "evaluation.json")}
    raise FileNotFoundError("No frozen targeted-compact answer artifact for " + task_id)


def _native_items(private):
    return [(axis, item) for axis in DRBII_DIMENSIONS
            for item in private.get("rubric", {}).get(axis, [])]


def _rubric_id(task_id, axis, index, item):
    return f"{task_id}:{axis}:{index}:{digest(item)[:12]}"


def _compatible(raw, evaluator, task_id):
    return (isinstance(raw, dict) and raw.get("task_id") == task_id
            and raw.get("judge_model") == evaluator.judge_id
            and raw.get("evaluator_source") == evaluator.source)


def _seed_cache(evaluation, evaluator, task_id, private, answer, cache, provenance):
    """Retain only validated native judgments with exact task/criterion identity."""
    raw = _raw_evaluation(evaluation)
    if not _compatible(raw, evaluator, task_id):
        return 0
    expected = {_rubric_id(task_id, axis, index, item): item
                for index, (axis, item) in enumerate(_native_items(private))}
    count = 0
    for row in raw.get("feedback", []):
        identity = row.get("rubric_id")
        if (identity not in expected or row.get("criterion") != expected[identity]
                or row.get("status") != "ok" or type(row.get("native_score")) is not int
                or row["native_score"] not in (-1, 0, 1)
                or not isinstance(row.get("reasoning"), str)):
            continue
        evidence = "\n".join(row.get("evidence_quotes") or [])
        cache[identity] = {"rubric_item": row["criterion"], "score": row["native_score"],
                           "reason": row["reasoning"], "evidence": evidence,
                           "_provenance": provenance}
        count += 1
    return count


class CachedRubricJudge:
    """Serve successful DRBII items and judge missing items with the same prompt.

    Cache keys include the native occurrence ordinal, so duplicate criteria keep
    the evaluator's original last-occurrence-wins semantics.
    """

    def __init__(self, model, task_id, private, cache, directory):
        self.model, self.task_id, self.cache = model, task_id, cache
        self.items, self.position = _native_items(private), 0
        self.directory, self.calls = directory, []

    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        requested = payload["rubric_items"]
        start, self.position = self.position, self.position + len(requested)
        batch = self.items[start:self.position]
        if requested != [item for _, item in batch]:
            raise ValueError("Native rubric order differs from cache inventory")
        identities = [_rubric_id(self.task_id, axis, start + index, item)
                      for index, (axis, item) in enumerate(batch)]
        missing = [index for index, identity in enumerate(identities) if identity not in self.cache]
        event = {"batch_index": len(self.calls), "rubric_ids": identities,
                 "cached_rubric_ids": [identity for identity in identities if identity in self.cache],
                 "requested_rubric_ids": [identities[index] for index in missing],
                 "started_at": utc_now(), "status": "cached" if not missing else "started"}
        self.calls.append(event)
        began = time.monotonic()
        try:
            if missing:
                subset = copy.deepcopy(payload)
                subset["rubric_items"] = [requested[index] for index in missing]
                sent = copy.deepcopy(messages)
                sent[-1]["content"] = json.dumps(subset, ensure_ascii=False, allow_nan=False)
                event["messages_sha256"] = digest(sent)
                result = _json_response(self.model(sent, **kwargs))
                rows = result.get("results")
                if (not isinstance(rows, list) or len(rows) != len(missing)
                        or any(not isinstance(row, dict) for row in rows)
                        or Counter(row.get("rubric_item") for row in rows)
                        != Counter(subset["rubric_items"])):
                    raise ValueError("Judge criterion coverage mismatch")
                if any(type(row.get("score")) is not int or row["score"] not in (-1, 0, 1)
                       or not isinstance(row.get("reason"), str)
                       or not isinstance(row.get("evidence"), str) for row in rows):
                    raise ValueError("Invalid native rubric judgment")
                by_text = defaultdict(deque)
                for row in rows:
                    by_text[row["rubric_item"]].append(row)
                for index in missing:
                    value = dict(by_text[requested[index]].popleft())
                    value["_provenance"] = str(self.directory)
                    self.cache[identities[index]] = value
                event["status"] = "complete"
            return {"results": [{key: value for key, value in self.cache[identity].items()
                                  if not key.startswith("_")} for identity in identities]}
        except Exception as error:
            event.update(status="failed", failure=_failure(error))
            raise
        finally:
            event.update(finished_at=utc_now(), latency_seconds=time.monotonic() - began)
            write_json(self.directory / "rubric_cache.json", self.cache)
            write_json(self.directory / "cache_trace.json", self.calls)


def _sum_budgets(budgets):
    keys = ("model_calls", "tokens", "tool_calls", "wall_seconds", "request_queue_idle_seconds")
    result = {key: sum(value.get(key, 0) or 0 for value in budgets) for key in keys}
    result["records"] = [row for value in budgets for row in value.get("records", [])]
    costs = [value.get("cost") for value in budgets]
    result["cost"] = sum(costs) if costs and all(isinstance(cost, (int, float)) for cost in costs) else None
    result["input_tokens"] = sum(row.get("input_tokens", 0) for row in result["records"])
    result["output_tokens"] = sum(row.get("output_tokens", 0) for row in result["records"])
    result["failed_model_calls"] = sum(bool(row.get("error")) for row in result["records"]
                                       if row.get("kind") == "model")
    result["model_call_failure_rate"] = (result["failed_model_calls"] / result["model_calls"]
                                          if result["model_calls"] else 0.0)
    result["estimated_model_calls"] = sum(bool(row.get("estimated")) for row in result["records"]
                                          if row.get("kind") == "model")
    return result


def _rejudge_one(config, material, task_id, source, output, *, resume, task_attempts, batch_size):
    benchmark, dataset = material["benchmark"], material["dataset"]
    private, frozen = dataset.private_records[task_id], _source_artifacts(source, benchmark, task_id)
    directory = output / "tasks" / benchmark / digest(task_id)
    previous = read_json(directory / "record.json")
    if previous and previous.get("answer_hash") != frozen["answer_hash"]:
        raise ValueError("Resume answer hash mismatch: " + task_id)
    if resume and previous and previous.get("complete"):
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "answer.txt").write_text(frozen["answer"], encoding="utf-8")
    write_json(directory / "source.json", {"source_answer": str(frozen["answer_file"]),
        "source_record": str(frozen["directory"] / "record.json"), "answer_hash": frozen["answer_hash"],
        "dataset_sha256": dataset.dataset_sha256, "criteria_sha256": digest(private),
        "generation_preserved": True, "evaluation_criteria_preserved": True})
    provider = NativeModels(config)
    cache, final_evaluation = {}, None
    attempt_root = directory / "attempts"
    existing = sorted(attempt_root.glob("*/record.json"))
    source_record = frozen["record"]
    for offset in range(task_attempts):
        attempt_index = len(existing) + offset + 1
        attempt = attempt_root / f"{attempt_index:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        ledger = BudgetLedger(max_calls=config.max_model_calls, max_tokens=config.max_total_tokens,
                              max_tool_calls=0, timeout_seconds=config.task_timeout)
        models, cache_judges = [], []
        failure = evaluation = None
        began = time.monotonic()
        write_json(attempt / "started.json", {"task_id": task_id, "started_at": utc_now(),
                                              "answer_hash": frozen["answer_hash"]})
        print(json.dumps({"benchmark": benchmark, "task_id": task_id, "attempt": attempt_index,
                          "status": "rejudging"}), flush=True)
        try:
            def judge_factory():
                model = JudgeEnvelopeModel(provider.create("judge", "targeted-compact-rejudge", ledger, "evaluation"))
                models.append(model)
                if benchmark == "deepresearch_bench_ii":
                    judge = CachedRubricJudge(model, task_id, private, cache, attempt)
                    cache_judges.append(judge)
                    return judge
                return model

            spec = config.models["judge"]
            evaluator = dataset.evaluator(judge_factory(), judge_id=spec.model, judge_api_base=spec.endpoint,
                judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout,
                max_parallel_judgments=config.judge_parallel, judge_factory=judge_factory)
            if benchmark == "deepresearch_bench_ii":
                evaluator.batch_size = batch_size
                if not cache:
                    _seed_cache(frozen["evaluation"], evaluator, task_id, private, frozen["answer"],
                                cache, str(frozen["directory"] / "evaluation.json"))
                    for path in existing:
                        _seed_cache(read_json(path.parent / "evaluation.json"), evaluator, task_id,
                                    private, frozen["answer"], cache, str(path.parent / "evaluation.json"))
            raw = evaluator.evaluate(frozen["answer"], ground_truth=task_id, private_record=private)
            evaluation = convert_feedback(raw).model_dump(mode="json")
            write_json(attempt / "evaluation.json", evaluation)
            final_evaluation = evaluation
        except Exception as error:
            failure = _failure(error)
        finally:
            budget = ledger.snapshot()
            budget["wall_seconds"] = time.monotonic() - began
            close_errors = _clients_close(models)
            write_json(attempt / "call_trace.json", [{"stage": "evaluation", "calls": model.calls,
                "metered_calls": getattr(model.model, "calls", [])} for model in models])
            write_json(attempt / "cache_trace.json", [event for judge in cache_judges for event in judge.calls])
            write_json(attempt / "record.json", {"benchmark": benchmark, "task_id": task_id,
                "attempt": attempt_index, "complete": bool((evaluation or {}).get("complete")),
                "answer_hash": frozen["answer_hash"], "score": (evaluation or {}).get("score"),
                "budget": budget, "failure": failure, "client_close_errors": close_errors,
                "latency_seconds": budget["wall_seconds"], "finished_at": utc_now()})
        if evaluation and evaluation["complete"]:
            break
    attempts = [read_json(path, {}) for path in sorted(attempt_root.glob("*/record.json"))]
    final_evaluation = final_evaluation or next((read_json(path.parent / "evaluation.json")
                                               for path in reversed(sorted(attempt_root.glob("*/record.json")))
                                               if (path.parent / "evaluation.json").is_file()), None)
    if final_evaluation:
        write_json(directory / "evaluation.json", final_evaluation)
    budget = _sum_budgets([row["budget"] for row in attempts if "budget" in row])
    complete = bool((final_evaluation or {}).get("complete"))
    row = {"benchmark": benchmark, "task_id": task_id, "phase": "test",
        "method": "targeted-compact-frozen-answer-rejudge", "condition": "test_adapted_targeted_compact_full_split",
        "status": "complete" if complete else "incomplete", "complete": complete,
        "answer_hash": frozen["answer_hash"], "answer_file": (directory / "answer.txt").relative_to(output).as_posix(),
        "score_result": {"complete": complete, "score": (final_evaluation or {}).get("score"),
                         "evaluation": final_evaluation},
        "outcome": {"evaluation": final_evaluation} if final_evaluation else None,
        "budget": budget, "latency_seconds": budget["wall_seconds"],
        "source_budget": source_record.get("budget"), "source_latency_seconds": source_record.get("latency_seconds"),
        "source_generation_checks": source_record.get("generation_checks"),
        "source_generation_finish_reason": source_record.get("generation_finish_reason"),
        "source_generation_complete": bool(source_record.get("generation_checks", {}).get("complete", True))
            and source_record.get("generation_finish_reason") not in {"length", "content_filter", "tool_calls", "function_call"},
        "attempt_count": len(attempts), "cached_rubric_count": len(cache),
        "source_evaluation_reuse": "successful exact rubric occurrences for matching judge model and evaluator source",
        "failure": attempts[-1].get("failure") if attempts else None, "finished_at": utc_now()}
    write_json(directory / "record.json", row)
    print(json.dumps({"task_id": task_id, "status": row["status"], "score": row["score_result"]["score"],
                      "tokens": budget["tokens"], "model_calls": budget["model_calls"]}), flush=True)
    return row


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def _report(output, materials, rows, started):
    summaries = []
    for material in materials:
        selected = [row for row in rows if row["benchmark"] == material["benchmark"]]
        complete = [row for row in selected if row.get("complete")]
        scores = [row["score_result"]["score"] for row in complete]
        metrics = defaultdict(list)
        for row in complete:
            raw = _raw_evaluation(row["score_result"]["evaluation"]) or {}
            for key, value in raw.get("official_metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
            for key in ("precision", "recall", "f1"):
                if isinstance(raw.get(key), (int, float)):
                    metrics[key].append(raw[key])
        budget = _sum_budgets([row.get("budget", {}) for row in selected])
        latencies = [row["latency_seconds"] for row in selected if isinstance(row.get("latency_seconds"), (int, float))]
        summaries.append({"benchmark": material["benchmark"], "tasks": len(material["task_ids"]),
            "answers_reused": len(selected), "complete": len(complete), "incomplete": len(selected) - len(complete),
            "mean_score_completed": statistics.mean(scores) if scores else None,
            "full_split_score": statistics.mean(scores) if len(complete) == len(material["task_ids"]) else None,
            "official_metrics_completed": {key: statistics.mean(value) for key, value in metrics.items()},
            "new_evaluation_budget": budget, "mean_new_evaluation_latency_seconds": statistics.mean(latencies) if latencies else None,
            "p95_new_evaluation_latency_seconds": _percentile(latencies, .95)})
    report = {"schema": "targeted-compact-rejudge-report-v1", "finished_at": utc_now(),
              "wall_seconds_this_invocation": time.monotonic() - started, "benchmarks": summaries, "records": rows}
    write_json(output / "report.json", report)
    (output / "records.jsonl").write_text("".join(json.dumps(_redact(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with (output / "benchmark_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        keys = ["benchmark", "tasks", "answers_reused", "complete", "incomplete", "mean_score_completed", "full_split_score",
                "model_calls", "tokens", "input_tokens", "output_tokens", "failed_model_calls", "model_call_failure_rate", "cost",
                "mean_new_evaluation_latency_seconds", "p95_new_evaluation_latency_seconds"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, summary["new_evaluation_budget"].get(key)) for key in keys})
    return report


def run(config, materials, source, output, *, workers=8, resume=False, task_attempts=2,
        batch_size=10, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    inventory = []
    for material in materials:
        for task_id in material["task_ids"]:
            frozen = _source_artifacts(source, material["benchmark"], task_id)
            inventory.append({"benchmark": material["benchmark"], "task_id": task_id,
                              "answer_hash": frozen["answer_hash"], "answer_file": str(frozen["answer_file"]),
                              "dataset_sha256": material["dataset"].dataset_sha256,
                              "criteria_sha256": digest(material["dataset"].private_records[task_id])})
    if dry_run:
        result = {"dry_run": True, "api_calls": 0, "tasks": len(inventory), "inventory": inventory}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Nonempty output requires --resume")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    old = read_json(manifest_path)
    identity = digest({"inventory": inventory, "judge": config.models["judge"].model_dump(mode="json"),
                       "batch_size": batch_size})
    if old and old.get("run_identity") != identity:
        raise ValueError("Resume inventory, judge configuration or rubric batch size changed")
    manifest = old or {"schema": "targeted-compact-rejudge-v1", "started_at": utc_now(),
        "run_identity": identity, "source": str(source), "inventory": inventory,
        "configuration": config.model_dump(mode="json"), "runner_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
        "generation": "reused immutable source answers; zero generation calls",
        "evaluation": "unchanged native benchmark criteria, prompt and scoring aggregation",
        "rubric_batch_size": batch_size, "condition": "test_adapted_targeted_compact_full_split"}
    manifest.update(workers=workers, task_attempts_per_invocation=task_attempts, resumed_at=utc_now())
    write_json(manifest_path, manifest)
    config.check_native()
    began, results = time.monotonic(), {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_rejudge_one, config, material, task_id, source, output, resume=resume,
            task_attempts=task_attempts, batch_size=batch_size): (material["benchmark"], task_id)
            for material in materials for task_id in material["task_ids"]}
        for future in as_completed(jobs):
            benchmark, task_id = jobs[future]
            try:
                results[(benchmark, task_id)] = future.result()
            except Exception as error:
                results[(benchmark, task_id)] = {"benchmark": benchmark, "task_id": task_id,
                    "status": "failed", "complete": False, "failure": _failure(error)}
            write_json(output / "progress.json", {"finished_at": utc_now(), "tasks": len(inventory),
                "finished": len(results), "complete": sum(bool(row.get("complete")) for row in results.values()),
                "wall_seconds": time.monotonic() - began})
    rows = [results[(item["benchmark"], item["task_id"])] for item in inventory]
    return _report(output, materials, rows, began)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "plan", "source", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rubric-batch-size", type=int, default=10)
    parser.add_argument("--task-attempts", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-timeout", type=float, default=300)
    parser.add_argument("--max-inflight", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.workers, args.rubric_batch_size, args.task_attempts, args.judge_max_tokens) <= 0 or args.judge_timeout <= 0:
        parser.error("Workers, batch size, attempts, token limit and timeout must be positive")
    if args.max_inflight is not None and args.max_inflight <= 0:
        parser.error("--max-inflight must be positive")
    config, materials = load_inputs(args.config, args.plan)
    spec = config.models["judge"].model_dump(mode="json")
    spec.update(max_tokens=args.judge_max_tokens, timeout=args.judge_timeout)
    if args.max_inflight is not None:
        spec["max_inflight_requests"] = args.max_inflight
    config.models["judge"] = type(config.models["judge"]).model_validate(spec)
    return run(config, materials, args.source, args.output, workers=args.workers, resume=args.resume,
               task_attempts=args.task_attempts, batch_size=args.rubric_batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
