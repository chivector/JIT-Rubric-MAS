"""Summarize high-score experiment records and export evolution curve data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.experiment_telemetry import OPERATION_KINDS, EvolutionTelemetry, call_statistics


ALIASES = {
    "rr": "researchrubrics", "research_rubrics": "researchrubrics",
    "dsqa": "deepsearchqa", "deepsearch_qa": "deepsearchqa",
    "drbii": "deepresearch_bench_ii", "drb_ii": "deepresearch_bench_ii",
    "deepresearchbenchii": "deepresearch_bench_ii", "writing_bench": "writingbench",
}
INSTRUCTION_METRICS = (
    "prompt_strict_accuracy", "instruction_strict_accuracy",
    "prompt_loose_accuracy", "instruction_loose_accuracy",
)
BENCHMARK_METRICS = {
    "deepresearch_bench_ii": ("Overall", "InformationRecall", "Analysis", "Presentation"),
    "ifeval": INSTRUCTION_METRICS,
    "ifbench": INSTRUCTION_METRICS,
    "writingbench": ("nativeMean", "normalized_score"),
    "researchrubrics": ("compliance",),
    "deepsearchqa": ("Precision", "Recall", "F1"),
}


def number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def benchmark_name(value, task_id=""):
    value = value or task_id.split(":", 1)[0]
    key = re.sub(r"[\s-]+", "_", str(value).lower())
    return ALIASES.get(key, key)


def operation_counts(operations):
    if not isinstance(operations, list):
        return None
    counts = {kind: 0 for kind in OPERATION_KINDS}
    for operation in operations:
        if isinstance(operation, dict) and operation.get("kind") in counts:
            counts[operation["kind"]] += 1
    return counts


def operation_attempts(path):
    document = read_json(path)
    attempts = []
    for call in document.get("calls", []):
        if not isinstance(call, dict) or call.get("phase") != "evolution_integrate":
            continue
        response = call.get("response")
        parse_error = None
        if isinstance(response, str):
            text = response.strip()
            if text.startswith("```") and text.endswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            try:
                response = json.loads(text)
            except ValueError as error:
                response = None
                parse_error = type(error).__name__
        operations = response.get("pool_operations") if isinstance(response, dict) else None
        errors = call.get("validation_errors") or []
        valid = False if errors or parse_error else (True if isinstance(operations, list) else None)
        summaries = None
        if isinstance(operations, list):
            summaries = []
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                encoded = json.dumps(operation, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()
                summaries.append({
                    "operation_id": operation.get("operation_id"), "kind": operation.get("kind"),
                    "source_task_id": operation.get("source_task_id"),
                    "target_agent_ids": operation.get("target_agent_ids"),
                    "profile_agent_ids": [profile.get("pool_agent_id") for profile in operation.get("profiles", [])
                                          if isinstance(profile, dict)],
                    "operation_sha256": hashlib.sha256(encoded).hexdigest(),
                })
        attempts.append({"attempt": call.get("attempt"), "valid": valid,
                         "validation_errors": errors, "parse_error": parse_error,
                         "pool_operations": summaries})
    return attempts


def evolution_operations(record):
    attempts = record.get("evolution_operation_attempts")
    proposed = [operation for attempt in attempts for operation in attempt.get("pool_operations") or []] \
        if isinstance(attempts, list) else None
    if proposed is None:
        proposed = record.get("operations_requested")
    unique = None
    if isinstance(proposed, list):
        unique = {}
        for operation in proposed:
            identity = operation.get("operation_id") or operation.get("operation_sha256")
            if identity is None:
                identity = json.dumps(operation, sort_keys=True, ensure_ascii=True)
            unique.setdefault(identity, operation)
        unique = list(unique.values())
    return {"evolution_operation_attempts": attempts,
            "operation_counts_proposed": operation_counts(proposed),
            "operation_counts_proposed_unique": operation_counts(unique),
            "operation_counts_validated": operation_counts(record.get("operations_requested")),
            "operation_counts_applied": operation_counts(record.get("operations_applied"))}


def aggregate_operations(records):
    groups = defaultdict(list)
    for record in records:
        key = (record["benchmark"], record.get("run_id", "root"), record.get("method", "native"),
               record.get("phase", "test"))
        groups[key].append(record)
    results = []
    for key, rows in sorted(groups.items()):
        result = dict(zip(("benchmark", "run_id", "method", "phase"), key))
        result["task_count"] = len(rows)
        evidence = [evolution_operations(row) for row in rows]
        for label in ("proposed", "proposed_unique", "validated", "applied"):
            counts = [item["operation_counts_" + label] for item in evidence]
            result["operation_counts_" + label] = (
                {kind: sum(item[kind] for item in counts) for kind in OPERATION_KINDS}
                if all(item is not None for item in counts) else None)
            result["operation_counts_" + label + "_known_tasks"] = sum(item is not None for item in counts)
        results.append(result)
    return results


def load_records(directory):
    directory = Path(directory)
    records = []
    seen = set()
    native_paths = sorted(directory.rglob("records.jsonl"))
    canonical = [(path, path.parent, "native") for path in native_paths]
    native_roots = {path.parent for path in native_paths}
    for path in sorted(directory.rglob("report.json")):
        if any(path.is_relative_to(root) for root in native_roots):
            continue
        document = read_json(path)
        if isinstance(document, dict) and isinstance(document.get("records"), list):
            canonical.append((path, path.parent, "compact"))
    canonical_roots = {root for _, root, _ in canonical}
    for path in sorted(directory.rglob("records/*.json")):
        parent = path.parent.parent
        if any(path.is_relative_to(root) for root in canonical_roots):
            continue
        canonical.append((path, parent, "compact"))
    supplements = {}
    for path in sorted(directory.rglob("official_metrics.json")):
        document = read_json(path)
        if document.get("schema") != "compact-official-metrics-v1":
            continue
        source = Path(document.get("source_report", path.parent / "report.json"))
        supplements[source.resolve()] = {row["task_id"]: row for row in document.get("records", [])
                                         if isinstance(row, dict) and row.get("task_id")}
    for path, parent, method in canonical:
        if path.suffix == ".jsonl":
            documents = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                         if line.strip()]
        else:
            document = read_json(path)
            documents = document["records"] if path.name == "report.json" else [document]
            if path.name == "report.json" and isinstance(document.get("task_ids"), list):
                documents = list(documents)
                recorded_tasks = {row.get("task_id") for row in documents if isinstance(row, dict)}
                for task_id in document["task_ids"]:
                    if isinstance(task_id, str) and task_id and task_id not in recorded_tasks:
                        documents.append({"task_id": task_id, "benchmark": document.get("benchmark"),
                                          "status": "not_attempted", "complete": False, "budget": None,
                                          "score_result": {"complete": False, "score": None}})
                        recorded_tasks.add(task_id)
        for document in documents:
            if (not isinstance(document, dict) or not document.get("task_id")
                    or not any(key in document for key in ("budget", "outcome", "score_result"))):
                continue
            row = dict(document)
            supplemental = supplements.get(path.resolve(), {}).get(row["task_id"])
            if supplemental:
                for name in ("strict_result", "loose_result", "official_metrics", "instruction_count",
                             "instruction_id_list"):
                    if name in supplemental:
                        row[name] = supplemental[name]
            failure = row.get("failure") if isinstance(row.get("failure"), dict) else {}
            run_failure = failure.get("run_failure") if isinstance(failure.get("run_failure"), dict) else {}
            run_dir = run_failure.get("run_dir")
            outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
            meta_path = Path(outcome.get("run_dir") or run_dir) / "meta_evolution.json" \
                if outcome.get("run_dir") or run_dir else None
            if meta_path is not None and meta_path.is_file():
                row["evolution_operation_attempts"] = operation_attempts(meta_path)
                row["evolution_operation_attempts_source"] = str(meta_path)
            if run_dir and not isinstance(row.get("outcome"), dict):
                location = Path(run_dir)
                evaluation_path = location / "evaluation.json"
                if evaluation_path.is_file():
                    row["evaluation"] = read_json(evaluation_path)
                    row["evaluation_recovered_from"] = str(evaluation_path)
                if not isinstance(row.get("budget"), dict):
                    budget_path = location / "budget.json"
                    if isinstance(run_failure.get("budget"), dict):
                        row["budget"] = run_failure["budget"]
                    elif budget_path.is_file():
                        row["budget"] = read_json(budget_path)
            row["benchmark"] = benchmark_name(row.get("benchmark"), row["task_id"])
            row.setdefault("phase", "test")
            relative_parent = str(parent.relative_to(directory))
            row.setdefault("run_id", "root" if relative_parent == "." else relative_parent)
            row.setdefault("method", method)
            identity = (str(parent.resolve()), row["benchmark"], row["task_id"], row["phase"],
                        row.get("round_index"), row.get("attempt_id", row.get("attempt", 0)))
            if identity in seen:
                continue
            seen.add(identity)
            row["source_path"] = str(path.relative_to(directory))
            records.append(row)
    return records


def evaluation(record):
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else record
    result = outcome.get("evaluation") or record.get("score_result") or outcome.get("score_result")
    if not isinstance(result, dict):
        return {}
    return result


def complete(record):
    if record.get("status") in {"failed", "incomplete", "interrupted", "generation_incomplete",
                                "generation_failed", "evaluation_failed", "not_attempted"}:
        return False
    result = evaluation(record)
    value = result.get("complete")
    if value is None:
        outcome = record.get("outcome")
        value = outcome.get("complete") if isinstance(outcome, dict) else record.get("complete")
    return value if type(value) is bool else None


def metric_layers(record):
    layers = []
    def visit(value):
        if not isinstance(value, dict):
            return
        layers.append(value)
        for name in ("outcome", "score_result", "evaluation", "raw", "official_metrics"):
            visit(value.get(name))
    visit(record)
    return layers


def find_value(layers, names):
    for layer in reversed(layers):
        for name in names:
            value = number(layer.get(name))
            if value is not None:
                return value
    return None


def instruction_result(layers, mode, default_mode):
    names = (f"{mode}_result", f"{mode}_evaluation", f"evaluation_{mode}", f"{mode}_checker")
    for layer in layers:
        for name in names:
            result = layer.get(name)
            if isinstance(result, dict):
                return result
    if mode == default_mode:
        return next((layer for layer in layers if "pass_count" in layer and "check_count" in layer), {})
    return {}


def task_metrics(record):
    benchmark = record["benchmark"]
    layers = metric_layers(record)
    result = evaluation(record)
    if benchmark == "deepresearch_bench_ii":
        dimensions = result.get("dimensions") or result.get("raw", {}).get("dimensions", {})
        dimensions = dimensions if isinstance(dimensions, dict) else {}
        metrics = {"Overall": find_value(layers, ["Overall", "score", "official_score"])}
        for name, axis in (("InformationRecall", "info_recall"), ("Analysis", "analysis"),
                           ("Presentation", "presentation")):
            value = find_value(layers, [name])
            if value is None and isinstance(dimensions.get(axis), dict):
                value = number(dimensions[axis].get("score"))
            metrics[name] = value
        return metrics
    if benchmark in ("ifeval", "ifbench"):
        default_mode = "strict" if benchmark == "ifeval" else "loose"
        metrics = {}
        for mode in ("strict", "loose"):
            checked = instruction_result(layers, mode, default_mode)
            prompt = find_value(layers, [f"prompt_{mode}_accuracy", f"prompt_level_{mode}_accuracy",
                                         f"prompt_level_accuracy_{mode}"])
            instruction = find_value(layers, [f"instruction_{mode}_accuracy",
                                              f"instruction_level_{mode}_accuracy",
                                              f"instruction_level_accuracy_{mode}"])
            if prompt is None:
                prompt = number(checked.get("score"))
            if prompt is None and type(checked.get("follow_all_instructions")) is bool:
                prompt = float(checked["follow_all_instructions"])
            if prompt is None and mode == default_mode:
                prompt = number(result.get("score"))
            if instruction is None:
                instruction = number(checked.get("instruction_accuracy"))
            if instruction is None:
                passed, total = number(checked.get("pass_count")), number(checked.get("check_count"))
                instruction = passed / total if passed is not None and total else None
            metrics[f"prompt_{mode}_accuracy"] = prompt
            metrics[f"instruction_{mode}_accuracy"] = instruction
        return metrics
    if benchmark == "writingbench":
        return {"nativeMean": find_value(layers, ["nativeMean", "native_mean"]),
                "normalized_score": find_value(layers, ["normalized_score", "score", "official_score"])}
    if benchmark == "researchrubrics":
        return {"compliance": find_value(layers, ["compliance", "official_compliance", "score", "official_score"])}
    if benchmark == "deepsearchqa":
        return {"Precision": find_value(layers, ["Precision", "precision"]),
                "Recall": find_value(layers, ["Recall", "recall"]),
                "F1": find_value(layers, ["F1", "f1", "score", "official_score"])}
    return {"score": find_value(layers, ["score", "official_score"])}


def aggregate_metrics(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record.get("run_id", "root"), record.get("method", "native"),
                record.get("phase", "test"))].append(record)
    result = {}
    for (benchmark, run_id, method, phase), rows in sorted(groups.items()):
        metrics = BENCHMARK_METRICS.get(benchmark, ("score",))
        values = {name: [] for name in metrics}
        scored_values = {name: [] for name in metrics}
        strict_counts = {mode: [] for mode in ("strict", "loose")}
        for row in rows:
            scored = evaluation(row).get("complete") is True
            observed = task_metrics(row)
            if scored:
                for name in metrics:
                    if observed.get(name) is not None:
                        scored_values[name].append(observed[name])
            if scored or complete(row) is not False:
                for name in metrics:
                    if observed.get(name) is not None:
                        values[name].append(observed[name])
            if benchmark in ("ifeval", "ifbench"):
                layers = metric_layers(row)
                default = "strict" if benchmark == "ifeval" else "loose"
                for mode in strict_counts:
                    checked = instruction_result(layers, mode, default)
                    passed, total = number(checked.get("pass_count")), number(checked.get("check_count"))
                    if total is None:
                        total = number(row.get("instruction_count"))
                    if total is None and isinstance(row.get("instruction_id_list"), list):
                        total = len(row["instruction_id_list"])
                    checked_complete = checked.get("complete")
                    checked_scored = (checked_complete is True or checked_complete is None
                                      and passed is not None and scored)
                    if total is not None and total > 0:
                        strict_counts[mode].append((passed if checked_scored and passed is not None else 0,
                                                  total, checked_scored))
        summaries = {}
        for name, observed in values.items():
            observed_mean = mean(observed) if observed else None
            all_tasks_mean = sum(observed) / len(rows) if observed else None
            scored_observed = scored_values[name]
            summaries[name] = {"mean": all_tasks_mean, "mean_observed": observed_mean,
                               "mean_all_tasks": all_tasks_mean, "failure_as_zero": all_tasks_mean,
                               "mean_scored_all": mean(scored_observed) if scored_observed else None,
                               "mean_all_tasks_scored_missing_as_zero": sum(scored_observed) / len(rows),
                               "scored_tasks": len(scored_observed),
                               "observed_tasks": len(observed), "requested_tasks": len(rows),
                               "complete": len(observed) == len(rows)}
        for mode, counts in strict_counts.items():
            if counts:
                name = f"instruction_{mode}_accuracy"
                observed_counts = [pair for pair in counts if pair[2]]
                weighted_observed = (sum(pair[0] for pair in observed_counts) / sum(pair[1] for pair in observed_counts)
                                     if observed_counts else None)
                weighted_all = (sum(pair[0] for pair in counts) / sum(pair[1] for pair in counts)
                                if len(counts) == len(rows) else None)
                summaries[name] = {"mean": weighted_all, "mean_observed": weighted_observed,
                                   "mean_all_tasks": weighted_all, "failure_as_zero": weighted_all,
                                   "mean_scored_all": weighted_observed,
                                   "mean_all_tasks_scored_missing_as_zero": weighted_all,
                                   "scored_tasks": len(observed_counts),
                                   "observed_tasks": len(observed_counts), "requested_tasks": len(rows),
                                   "complete": len(observed_counts) == len(rows), "aggregation": "instruction-weighted",
                                   "passed_instructions": sum(pair[0] for pair in counts),
                                   "total_instructions": sum(pair[1] for pair in counts) if len(counts) == len(rows) else None,
                                   "observed_instructions": sum(pair[1] for pair in observed_counts)}
        condition = f"{run_id}:{method}:{phase}"
        result.setdefault(benchmark, {})[condition] = {
            "run_id": run_id, "method": method, "phase": phase,
            "task_count": len(rows), "complete_tasks": sum(complete(row) is True for row in rows),
            "execution_complete_tasks": sum(complete(row) is True for row in rows),
            "failed_tasks": sum(complete(row) is False for row in rows),
            "scored_tasks": sum(evaluation(row).get("complete") is True for row in rows),
            "evaluation_complete_tasks": sum(evaluation(row).get("complete") is True for row in rows),
            "metrics": summaries,
        }
    return result


def record_budgets(record):
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    main = record.get("budget") or outcome.get("budget")
    budgets = {"main": main} if isinstance(main, dict) else {}
    for label in ("generation", "evaluation", "retrieval", "evolution"):
        value = record.get(label + "_budget") or outcome.get(label + "_budget")
        if isinstance(value, dict) and value != main:
            budgets[label] = value
    return budgets


def combined_budget(record):
    budgets = list(record_budgets(record).values())
    if not budgets:
        return None
    combined = {name: sum(budget[name] for budget in budgets)
                if all(number(budget.get(name)) is not None for budget in budgets) else None
                for name in ("tokens", "model_calls", "tool_calls", "cost")}
    combined["records"] = [call for budget in budgets for call in budget.get("records", [])]
    stages = {}
    for label, budget in record_budgets(record).items():
        for stage, values in (budget.get("by_stage") or {label: budget}).items():
            group = stages.setdefault(stage, {name: 0 for name in ("tokens", "model_calls", "tool_calls", "cost")})
            for name in group:
                value = number(values.get(name))
                group[name] = group[name] + value if group[name] is not None and value is not None else None
    combined["by_stage"] = stages
    latency = record.get("latency_seconds")
    if latency is None:
        latency = (sum(budget["wall_seconds"] for budget in budgets)
                   if all(number(budget.get("wall_seconds")) is not None for budget in budgets) else None)
    combined["wall_seconds"] = latency
    return combined


def model_records(budget):
    records = budget.get("records") if isinstance(budget, dict) else None
    return [record for record in records or []
            if isinstance(record, dict) and record.get("kind") == "model"]


def measured_sum(rows, field):
    values = [number(row.get(field)) if isinstance(row, dict) else None for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def token_statistics(budgets):
    calls = [record for budget in budgets for record in model_records(budget)]
    declared_calls = measured_sum(budgets, "model_calls")
    declared_tokens = measured_sum(budgets, "tokens")
    observed_input = sum(number(record.get("input_tokens")) or 0 for record in calls)
    observed_output = sum(number(record.get("output_tokens")) or 0 for record in calls)
    detailed = (declared_calls is not None and declared_calls == len(calls)
                and measured_sum(calls, "input_tokens") is not None
                and measured_sum(calls, "output_tokens") is not None
                and declared_tokens == observed_input + observed_output)
    result = {"input_tokens": observed_input if detailed else None,
              "output_tokens": observed_output if detailed else None,
              "observed_input_tokens": observed_input, "observed_output_tokens": observed_output,
              "unattributed_model_calls": (declared_calls - len(calls)
                                            if declared_calls is not None and declared_calls >= len(calls)
                                            else None),
              "unattributed_tokens": (declared_tokens - observed_input - observed_output
                                       if declared_tokens is not None
                                       and declared_tokens >= observed_input + observed_output else None)}
    partitions = {
        "actual": [record for record in calls if record.get("estimated") is False],
        "estimated": [record for record in calls if record.get("estimated") is True],
        "unknown_provenance": [record for record in calls if type(record.get("estimated")) is not bool],
    }
    for label, records in partitions.items():
        inp, out = measured_sum(records, "input_tokens"), measured_sum(records, "output_tokens")
        result[label + "_model_calls"] = len(records)
        result[label + "_input_tokens"] = inp
        result[label + "_output_tokens"] = out
        result[label + "_tokens"] = inp + out if inp is not None and out is not None else None
    result["token_usage_complete"] = detailed and not partitions["unknown_provenance"]
    unknown_rows = 0
    for budget in budgets:
        records = model_records(budget)
        inp, out = measured_sum(records, "input_tokens"), measured_sum(records, "output_tokens")
        known = (isinstance(budget, dict) and number(budget.get("model_calls")) == len(records)
                 and inp is not None and out is not None and number(budget.get("tokens")) == inp + out
                 and all(type(record.get("estimated")) is bool for record in records))
        unknown_rows += not known
    result["token_usage_unknown_rows"] = unknown_rows
    if not result["token_usage_complete"]:
        source = "unknown"
    elif partitions["actual"] and partitions["estimated"]:
        source = "mixed"
    elif partitions["estimated"]:
        source = "estimated"
    else:
        source = "provider" if calls else "none"
    result["token_usage_source"] = source
    return result


def call_efficiency(budgets):
    audits = [call_statistics(budget) for budget in budgets]
    result = {field: measured_sum(audits, field)
              for field in ("api_attempts", "failed_api_attempts", "failed_calls", "retry_count")}
    result["api_calls"] = result["api_attempts"]
    attempts, failed = result["api_attempts"], result["failed_api_attempts"]
    result["api_failure_rate"] = failed / attempts if attempts and failed is not None else None
    logical = measured_sum(budgets, "model_calls")
    calls = [record for budget in budgets for record in model_records(budget)]
    latencies = sorted(number(record.get("wall_seconds")) for record in calls
                       if number(record.get("wall_seconds")) is not None)
    settled = logical is not None and logical == len(calls)
    result["model_call_failure_rate"] = (result["failed_calls"] / logical
                                          if settled and logical and result["failed_calls"] is not None else None)
    result["call_wall_seconds_sum"] = sum(latencies) if settled and len(latencies) == len(calls) else None
    result["call_wall_seconds_mean"] = mean(latencies) if latencies else None
    result["call_wall_seconds_p50"] = latencies[(len(latencies) - 1) // 2] if latencies else None
    result["call_wall_seconds_p95"] = latencies[math.ceil((len(latencies) - 1) * 0.95)] if latencies else None
    result["call_latency_known_calls"] = len(latencies)
    queues = [number((record.get("request") or {}).get("queue_seconds")) for record in calls]
    result["request_queue_seconds_sum"] = sum(queues) if settled and all(value is not None for value in queues) else None
    return result


def estimated_cost(budget, rates):
    if not rates:
        return None
    records = budget.get("records") if isinstance(budget, dict) else None
    if not isinstance(records, list):
        return None
    cost = 0.0
    for record in records:
        if record.get("kind") != "model":
            continue
        request = record.get("request") or {}
        model = request.get("response_model") or record.get("model")
        rate = rates.get("models", {}).get(model) or rates.get("default")
        if not isinstance(rate, dict):
            return None
        inp, out = number(record.get("input_tokens")), number(record.get("output_tokens"))
        in_rate = number(rate.get("input_per_million"))
        out_rate = number(rate.get("output_per_million"))
        if None in (inp, out, in_rate, out_rate):
            return None
        cost += (inp * in_rate + out * out_rate) / 1_000_000
    if number(budget.get("model_calls")) is not None:
        settled = sum(record.get("kind") == "model" for record in records)
        if settled != budget["model_calls"]:
            return None
    return cost


def summarize_efficiency(records, rates=None):
    rates = rates or {}
    groups = defaultdict(list)
    conditions = defaultdict(list)
    for record in records:
        groups[record["benchmark"]].append(record)
        key = (record["benchmark"], record.get("run_id", "root"), record.get("method", "native"),
               record.get("phase", "test"))
        conditions[key].append(record)
    def summarize(rows):
        budgets = [combined_budget(row) for row in rows]
        totals = {}
        for field in ("tokens", "model_calls", "tool_calls", "cost"):
            observed = [number(budget.get(field)) if budget else None for budget in budgets]
            totals[field if field != "cost" else "actual_cost"] = (
                sum(observed) if all(value is not None for value in observed) else None)
            totals[f"{field}_known_rows"] = sum(value is not None for value in observed)
        totals.update(token_statistics(budgets))
        totals.update(call_efficiency(budgets))
        costs = [estimated_cost(budget, rates) for budget in budgets]
        totals["estimated_cost"] = sum(costs) if all(value is not None for value in costs) else None
        totals["estimated_cost_known_rows"] = sum(value is not None for value in costs)
        totals["cost_evidence"] = {"actual_source": "budget.cost",
                                    "actual_cost_known_rows": totals["cost_known_rows"],
                                    "actual_cost_missing_reason": "provider billing not recorded" if totals["actual_cost"] is None else None,
                                    "rate_card_supplied": bool(rates),
                                    "estimated_source": "per-call token counters and supplied rates" if rates else None,
                                    "estimated_cost_missing_reason": ("unsettled or unpriced calls" if rates
                                                                       else "no rate card supplied")
                                        if totals["estimated_cost"] is None else None,
                                    "estimated_cost_known_rows": totals["estimated_cost_known_rows"]}
        latencies = [number(budget.get("wall_seconds")) if budget else None for budget in budgets]
        measured = sorted(value for value in latencies if value is not None)
        totals["latency_seconds_sum"] = sum(measured) if len(measured) == len(rows) else None
        totals["latency_seconds_mean"] = mean(measured) if measured else None
        totals["latency_seconds_p50"] = measured[(len(measured) - 1) // 2] if measured else None
        totals["latency_seconds_p95"] = measured[math.ceil((len(measured) - 1) * 0.95)] if measured else None
        totals["latency_known_rows"] = len(measured)
        status = [complete(row) for row in rows]
        totals.update(task_count=len(rows), complete_tasks=sum(value is True for value in status),
                      execution_complete_tasks=sum(value is True for value in status),
                      scored_tasks=sum(evaluation(row).get("complete") is True for row in rows),
                      evaluation_complete_tasks=sum(evaluation(row).get("complete") is True for row in rows),
                      failed_tasks=sum(value is False for value in status),
                      unknown_status_tasks=sum(value is None for value in status),
                      task_failure_rate=(sum(value is False for value in status) / len(rows)
                                         if all(value is not None for value in status) and rows else None))
        totals["by_stage"] = stage_efficiency(budgets, rates)
        return totals
    return {"total": summarize(records),
            "by_benchmark": {benchmark: summarize(rows) for benchmark, rows in sorted(groups.items())},
            "by_condition": [{"benchmark": key[0], "run_id": key[1], "method": key[2], "phase": key[3],
                              **summarize(rows)} for key, rows in sorted(conditions.items())],
            "measurement_sources": {
                "token_provenance": "budget.records[].estimated: false=provider usage, true=estimated usage",
                "api_calls": "audited provider attempts in budget.records[].request.transport_retry.attempts",
                "model_calls": "logical ledger reservations, including unsettled calls",
                "task_latency": "record.latency_seconds, falling back to budget.wall_seconds",
                "stage_latency": "sum and distribution of per-model call wall_seconds; calls can overlap",
            },
            "pricing": rates if rates else None,
            "cost_currency": rates.get("currency", "USD") if rates else None}


def stage_efficiency(budgets, rates):
    groups = defaultdict(list)
    unknown = 0
    for budget in budgets:
        if not isinstance(budget, dict) or not isinstance(budget.get("by_stage"), dict):
            unknown += 1
            continue
        if number(budget.get("model_calls")) != len(model_records(budget)):
            unknown += 1
        for stage, values in budget.get("by_stage", {}).items():
            records = [record for record in budget.get("records", []) if record.get("stage") == stage]
            groups[stage].append({**values, "records": records})
    result = {}
    for stage, rows in groups.items():
        totals = {}
        for name in ("tokens", "model_calls", "tool_calls", "cost"):
            values = [number(row.get(name)) for row in rows]
            totals["actual_cost" if name == "cost" else name] = (
                sum(values) if all(value is not None for value in values) and not unknown else None)
        costs = [estimated_cost(row, rates) for row in rows]
        totals["estimated_cost"] = sum(costs) if all(value is not None for value in costs) and not unknown else None
        measured = rows + [None] * unknown
        totals.update(token_statistics(measured))
        totals.update(call_efficiency(measured))
        totals["budget_rows"] = len(rows)
        totals["unknown_budget_rows"] = unknown
        totals["cost_evidence"] = {"actual_source": "budget.by_stage.cost",
                                    "actual_cost_known_rows": sum(number(row.get("cost")) is not None for row in rows),
                                    "rate_card_supplied": bool(rates),
                                    "estimated_source": "stage per-call token counters and supplied rates" if rates else None}
        result[stage] = totals
    return result


def curve_records(records, destination, *, phase_only=None):
    groups = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record.get("run_id", "root"), record.get("method", "native"))].append(record)
    paths = {}
    for (benchmark, run_id, method), rows in sorted(groups.items()):
        telemetry = EvolutionTelemetry()
        phases = {"evolve": 0, "evolution": 0, "stream": 0, "validate": 1,
                  "validation": 1, "test": 2, "evaluate": 2}
        ordered = sorted(rows, key=lambda row: (phases.get(row.get("phase", ""), 3),
                                                row.get("round_index") if row.get("round_index") is not None
                                                else len(rows), row["task_id"]))
        exported = []
        operation_totals = {label: {kind: 0 for kind in OPERATION_KINDS}
                            for label in ("proposed", "proposed_unique", "validated", "applied")}
        operation_known = {label: True for label in operation_totals}
        for index, record in enumerate(ordered):
            outcome = dict(record.get("outcome") or {})
            outcome["complete"] = complete(record)
            outcome["error"] = record.get("error") or outcome.get("error")
            budget = combined_budget(record)
            row = telemetry.record_round(
                record.get("round_index", index), performance=task_metrics(record),
                budget=budget, latency_seconds=record.get("latency_seconds"),
                pool_before=record.get("pool_before"), pool_after=record.get("pool_after"),
                operations_requested=record.get("operations_requested"),
                operations_applied=record.get("operations_applied"),
                harness_before=record.get("harness_before"), harness_after=record.get("harness_after"),
                outcome=outcome)
            row.update(benchmark=benchmark, task_id=record["task_id"],
                       phase=record.get("phase", "test"), round_index=record.get("round_index"),
                       sample_index=index, source_path=record["source_path"], run_id=run_id, method=method)
            row["token_usage"] = token_statistics([budget])
            for name in ("input_tokens", "output_tokens", "actual_tokens", "estimated_tokens"):
                row[name] = row["token_usage"][name]
            for label in ("before", "after"):
                if record.get("agent_pool_size_" + label) is not None:
                    row["pool_agent_count_" + label] = record["agent_pool_size_" + label]
            for name in ("harness_modification_count", "harness_added_agent_ids", "harness_deleted_agent_ids",
                         "construction_repair_count", "harness_added_ids", "harness_deleted_ids",
                         "harness_repair_count", "harness_meta_trajectory", "harness_failed_attempts"):
                if name in record:
                    row[name] = record[name]
            row["evaluation_complete"] = evaluation(record).get("complete")
            row["score_per_round"] = find_value(metric_layers(record), ["score", "official_score", "Overall"])
            row.update(evolution_operations(record))
            row["evolution_operation_attempts_source"] = record.get("evolution_operation_attempts_source")
            for label in operation_totals:
                counts = row["operation_counts_" + label]
                if counts is None:
                    operation_known[label] = False
                else:
                    for kind in OPERATION_KINDS:
                        operation_totals[label][kind] += counts[kind]
                        row[f"operation_{label}_{kind}"] = counts[kind]
                cumulative = dict(operation_totals[label]) if operation_known[label] else None
                row["operation_counts_cumulative_" + label] = cumulative
                if cumulative is not None:
                    row.update({f"cumulative_operation_{label}_{kind}": value
                                for kind, value in cumulative.items()})
            row.update({"performance_" + key: value for key, value in task_metrics(record).items()})
            row.update({"cumulative_" + key: value for key, value in row["cumulative"].items()})
            for label in ("requested", "applied"):
                counts = row["operation_counts_" + label]
                if counts is not None:
                    row.update({"operation_" + label + "_" + key: value for key, value in counts.items()})
            exported.append(row)
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        suffix = "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", phase_only) if phase_only else ""
        target = Path(destination) / "evolution_curve" / benchmark / f"{safe_run}_{method}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = target.with_suffix(".jsonl")
        csv_path = target.with_suffix(".csv")
        jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                     for row in exported), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in exported for key in row}))
            writer.writeheader()
            for row in exported:
                writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict))
                                 else value for key, value in row.items()})
        paths.setdefault(benchmark, {})[f"{run_id}:{method}"] = {"jsonl": str(jsonl_path), "csv": str(csv_path)}
    if phase_only is None:
        for phase in sorted({record.get("phase", "test") for record in records}):
            phase_paths = curve_records([record for record in records if record.get("phase", "test") == phase],
                                        destination, phase_only=phase)
            for benchmark, conditions in phase_paths.items():
                for condition, values in conditions.items():
                    paths[benchmark][condition].setdefault("phases", {})[phase] = values
    return paths


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(directory, destination=None, *, rates=None):
    directory = Path(directory)
    destination = Path(destination) if destination is not None else directory / "summary"
    records = load_records(directory)
    if not records:
        raise ValueError("No task records found")
    destination.mkdir(parents=True, exist_ok=True)
    efficiency = summarize_efficiency(records, rates)
    operation_summary = aggregate_operations(records)
    efficiency["evolution_operations"] = operation_summary
    metrics = aggregate_metrics(records)
    curves = curve_records(records, destination)
    write_json(destination / "efficiency_summary.json", efficiency)
    write_json(destination / "benchmark_metrics.json", metrics)
    write_json(destination / "evolution_operations.json", operation_summary)
    lines = ["# High-score experiment results", "", f"Input: `{directory}`", "",
             f"Tasks: {len(records)}", "",
             "Scored means use `mean_scored_all` across completed evaluations, including execution or commit failures. The full-task denominator counts unavailable scores as zero; an entirely unscored condition remains null. Execution completion is counted separately.",
             "", "| Benchmark | Phase | Execution complete | Scored | Scored mean | Full-task denominator (missing=0) |",
             "|---|---|---:|---:|---|---|"]
    for benchmark, phases in metrics.items():
        for condition, values in phases.items():
            measured = ", ".join(f"{name}={value['mean_scored_all']:.6f}" if value["mean_scored_all"] is not None else f"{name}=null"
                                 for name, value in values["metrics"].items())
            full = ", ".join(f"{name}={value['mean_all_tasks_scored_missing_as_zero']:.6f}"
                             if value["mean_all_tasks_scored_missing_as_zero"] is not None else f"{name}=null"
                             for name, value in values["metrics"].items())
            lines.append(f"| {benchmark} | {condition} | {values['execution_complete_tasks']}/{values['task_count']} | "
                         f"{values['scored_tasks']}/{values['task_count']} | {measured} | {full} |")
    lines += ["", "| Benchmark | Tokens | Model calls | API attempts | Mean latency (s) | Failures | Actual cost | Estimated cost |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for benchmark, values in efficiency["by_benchmark"].items():
        lines.append(f"| {benchmark} | {values['tokens']} | {values['model_calls']} | {values['api_attempts']} | "
                     f"{values['latency_seconds_mean']} | {values['failed_tasks']}/{values['task_count']} | "
                     f"{values['actual_cost']} | {values['estimated_cost']} |")
    lines += ["", "Evolution curves: `evolution_curve/<benchmark>/<run_id>_<method>.jsonl` and `.csv`.",
              "Phase-specific curves add `_<phase>` and reset cumulative measurements for that phase.",
              "Proposed operation attempts, unique proposed IDs, validated requests and applied operations are separate in `evolution_operations.json` and curve columns.",
              "Missing measurements remain null. Estimated cost is computed from the supplied rates and observed token counters."]
    (destination / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"records": len(records), "output": str(destination), "curves": curves,
            "efficiency_summary": efficiency, "benchmark_metrics": metrics,
            "evolution_operations": operation_summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output")
    parser.add_argument("--rates", help="JSON with models/default input_per_million and output_per_million rates")
    args = parser.parse_args(argv)
    result = summarize(args.directory, args.output, rates=read_json(args.rates) if args.rates else None)
    print(json.dumps({"records": result["records"], "output": result["output"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
