"""Score saved compact answers with both pinned author instruction metrics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import load_benchmark
from jit_mas.compact_generation import safe_error
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.schemas import digest, utc_now

_CHECKER_LOCK = threading.RLock()


def _check_variant(checker, prediction, private_record, variant):
    import langdetect

    record = copy.deepcopy(private_record)
    prompt = record["prompt"]
    checker_kwargs = record["kwargs"]
    if checker.benchmark == "ifbench":
        checker_kwargs = [{key: value for key, value in options.items() if value is not None}
                          for options in checker_kwargs]
    example = checker.author.InputExample(
        key=0, instruction_id_list=record["instruction_id_list"],
        prompt=prompt, kwargs=checker_kwargs)
    evaluate = getattr(checker.author, "test_instruction_following_" + variant)
    started = time.perf_counter()
    with _CHECKER_LOCK:
        previous_state = random.getstate()
        previous_seed = langdetect.DetectorFactory.seed
        try:
            random.seed(int(digest({"prompt": prompt,
                                    "instruction_id_list": record["instruction_id_list"],
                                    "kwargs": record["kwargs"]}), 16))
            langdetect.DetectorFactory.seed = 0
            result = evaluate(example, {prompt: prediction})
        finally:
            random.setstate(previous_state)
            langdetect.DetectorFactory.seed = previous_seed
    verdicts = list(result.follow_instruction_list)
    if (len(verdicts) != len(private_record["instruction_id_list"])
            or not verdicts or any(type(verdict) is not bool for verdict in verdicts)
            or type(result.follow_all_instructions) is not bool
            or result.follow_all_instructions != all(verdicts)):
        raise ValueError("Author checker returned inconsistent instruction verdicts")
    return {"complete": True, "follow_all_instructions": result.follow_all_instructions,
            "follow_instruction_list": verdicts, "pass_count": sum(verdicts),
            "check_count": len(verdicts), "latency_seconds": time.perf_counter() - started}


def _aggregate(rows):
    metrics = {}
    scored_metrics = {}
    failure_zero_metrics = {}
    counts = {}
    total_instructions = sum(row["instruction_count"] for row in rows)
    for variant in ("strict", "loose"):
        results = [row[variant + "_result"] for row in rows
                   if row[variant + "_result"] and row[variant + "_result"].get("complete")]
        passed_prompts = sum(result["follow_all_instructions"] for result in results)
        passed_instructions = sum(result["pass_count"] for result in results)
        checked_instructions = sum(result["check_count"] for result in results)
        prompt_key = "prompt_level_" + variant + "_accuracy"
        instruction_key = "instruction_level_" + variant + "_accuracy"
        scored_metrics[prompt_key] = passed_prompts / len(results) if results else None
        scored_metrics[instruction_key] = (passed_instructions / checked_instructions
                                           if checked_instructions else None)
        failure_zero_metrics[prompt_key] = passed_prompts / len(rows)
        failure_zero_metrics[instruction_key] = passed_instructions / total_instructions
        metrics[prompt_key] = scored_metrics[prompt_key] if len(results) == len(rows) else None
        metrics[instruction_key] = (scored_metrics[instruction_key]
                                    if len(results) == len(rows) else None)
        counts[variant] = {"scored": len(results), "failed": len(rows) - len(results),
                           "passed_prompts": passed_prompts,
                           "passed_instructions": passed_instructions,
                           "checked_instructions": checked_instructions}
    return {"complete": all(counts[variant]["failed"] == 0 for variant in counts),
            "requested": len(rows), "instruction_count": total_instructions,
            "official_metrics": metrics, "scored_metrics": scored_metrics,
            "all_tasks_zero_for_failures": failure_zero_metrics, "counts": counts}


def run(args):
    started_at = utc_now()
    started = time.perf_counter()
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()
    if output_path == report_path or output_path.exists():
        raise FileExistsError("--output must be a new file distinct from the source report")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema") != "compact-generation-run-v1":
        raise ValueError("--report must be a compact-generation-run-v1 report")
    benchmark = report.get("benchmark")
    if benchmark not in {"ifeval", "ifbench"}:
        raise ValueError("The report benchmark must be IFEval or IFBench")
    data_path = Path(args.dataset).resolve()
    if data_path.is_dir():
        data_path = data_path / {"ifeval": "ifeval_input_data.jsonl",
                                 "ifbench": "IFBench_test.jsonl"}[benchmark]
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if dataset_hash != report.get("dataset_sha256"):
        raise ValueError("Dataset hash differs from the source report")
    dataset = load_benchmark(benchmark, data_path)
    task_ids = report.get("task_ids")
    if (not isinstance(task_ids, list) or not task_ids
            or any(not isinstance(task_id, str) for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)
            or any(task_id not in dataset.tasks for task_id in task_ids)):
        raise ValueError("Report contains an invalid task inventory")
    source_rows = {}
    for record in report.get("records", []):
        task_id = record.get("task_id")
        if task_id not in task_ids or task_id in source_rows:
            raise ValueError("Report contains an unknown or duplicate task record")
        source_rows[task_id] = record
    checker = PinnedInstructionChecker(Path(args.checker_source).resolve(), benchmark)
    checker.assert_frozen()
    rows = []
    for task_id in task_ids:
        private_record = dataset.private_records[task_id]
        record = source_rows.get(task_id)
        row = {"task_id": task_id,
               "instruction_id_list": list(private_record["instruction_id_list"]),
               "instruction_count": len(private_record["instruction_id_list"]),
               "private_record_sha256": digest(private_record),
               "strict_result": None, "loose_result": None}
        if record is None:
            row["status"] = "not_attempted"
        elif not record.get("complete"):
            row["status"] = "generation_incomplete"
            row["generation_error"] = record.get("error")
        elif not record.get("answer_file"):
            row["status"] = "answer_missing"
        else:
            if record.get("private_record_sha256") != digest(private_record):
                raise ValueError("Private record hash differs from the source report: " + task_id)
            answer_path = (report_path.parent / record["answer_file"]).resolve()
            if not answer_path.is_relative_to(report_path.parent):
                raise ValueError("Answer path must stay inside the source report directory")
            row["answer_file"] = str(answer_path)
            row["answer_hash"] = record.get("answer_hash")
            if not answer_path.is_file():
                row["status"] = "answer_missing"
            else:
                prediction = answer_path.read_text(encoding="utf-8")
                if digest(prediction) != record.get("answer_hash"):
                    raise ValueError("Answer hash differs from the source report: " + task_id)
                for variant in ("strict", "loose"):
                    try:
                        row[variant + "_result"] = _check_variant(
                            checker, prediction, private_record, variant)
                    except Exception as exc:
                        row[variant + "_result"] = {"complete": False, "error": safe_error(exc)}
                row["status"] = ("scored" if all(row[variant + "_result"]["complete"]
                                                for variant in ("strict", "loose"))
                                 else "evaluation_failed")
        rows.append(row)
    checker.assert_frozen()
    if (report_path.read_bytes() != report_bytes
            or hashlib.sha256(data_path.read_bytes()).hexdigest() != dataset_hash):
        raise RuntimeError("Source report or dataset changed during scoring")
    result = {"schema": "compact-official-metrics-v1", "benchmark": benchmark,
              "started_at": started_at, "finished_at": utc_now(),
              "source_report": str(report_path),
              "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
              "dataset_sha256": dataset_hash, "task_ids": task_ids,
              "checker_identity": copy.deepcopy(checker.identity),
              "checker_variants": ["strict", "loose"], "records": rows,
              "summary": _aggregate(rows),
              "efficiency": {"api_calls": 0, "tokens": 0, "cost": 0,
                             "latency_seconds": time.perf_counter() - started}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checker-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, "offline official scoring: " + safe_error(exc)["message"] + "\n")
    print(json.dumps(result["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
