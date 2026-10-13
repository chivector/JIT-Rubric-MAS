"""Create local benchmark tables and plots from recorded experiment receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import csv
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean

from jit_mas.experiment_telemetry import EvolutionTelemetry, call_statistics
from scripts.summarize_high_score_experiment import (
    aggregate_metrics, combined_budget, load_records, summarize_efficiency, task_metrics,
)


CHANGE_DOMAINS = ("harness", "memory", "profile_behavior", "harness_policy")


def number(value):
    return value if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None


def complete_sum(values):
    values = [number(value) for value in values]
    return sum(values) if values and all(value is not None for value in values) else None


def observed_sum(values):
    values = [value for item in values if (value := number(item)) is not None]
    return sum(values) if values else None


def recorded_profiles(record, label):
    pool = record.get("pool_" + label)
    if not isinstance(pool, dict):
        snapshot = record.get("snapshot_" + label)
        pool = snapshot.get("agent_pool") if isinstance(snapshot, dict) else None
    harness = record.get("harness_" + label)
    profiles = {profile["pool_agent_id"]: dict(profile) for profile in (pool or {}).get("profiles", [])
                if isinstance(profile, dict) and profile.get("pool_agent_id")}
    if isinstance(harness, dict):
        for identity, content in harness.items():
            if isinstance(content, dict):
                profiles[identity] = {**content, **profiles.get(identity, {}), "pool_agent_id": identity}
    return {"profiles": list(profiles.values())} if isinstance(pool, dict) or isinstance(harness, dict) else None


def comparable_profiles(before, after):
    if before is None or after is None:
        return before, after
    before = {"profiles": [dict(profile) for profile in before["profiles"]]}
    after = {"profiles": [dict(profile) for profile in after["profiles"]]}
    before_ids = {profile["pool_agent_id"]: profile for profile in before["profiles"]}
    after_ids = {profile["pool_agent_id"]: profile for profile in after["profiles"]}
    for identity in before_ids.keys() & after_ids.keys():
        first, second = before_ids[identity], after_ids[identity]
        for field in first.keys() ^ second.keys():
            first.pop(field, None)
            second.pop(field, None)
    return before, after


SELECTED = (
    ("ResearchRubrics TEST33", "researchrubrics_test33_recovered", None, "TEST33"),
    ("IFBench TEST40", "ifbench_v11_final", None, "TEST40"),
    ("IFEval TEST40", "ifeval_v11", None, "TEST40"),
    ("DeepSearchQA ACT", "research_v5_enhanced", "deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c", "one task"),
    ("DeepSearchQA country", "research_v6_focused", "deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c", "one task"),
    ("DeepResearch Bench II", "research_v5_enhanced", "deepresearch_bench_ii:1", "one task"),
    ("WritingBench 104 selected", "writing_v9_editorial_scored2", None, "one candidate"),
    ("WritingBench 104 final", "writing_v9_final_scored", None, "one candidate"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def compact_efficiency(rows):
    full = summarize_efficiency(rows)["total"]
    fields = ("tokens", "input_tokens", "output_tokens", "actual_tokens", "estimated_tokens",
              "model_calls", "api_attempts", "failed_api_attempts", "retry_count", "api_failure_rate",
              "task_failure_rate", "latency_seconds_mean", "latency_seconds_p95", "latency_seconds_sum",
              "actual_cost", "estimated_cost", "task_count", "complete_tasks", "failed_tasks")
    result = {key: full.get(key) for key in fields}
    generation = [value.get("tokens") for name, value in full["by_stage"].items()
                  if name not in ("evaluation", "update")]
    result["generation_tokens"] = complete_sum(generation) if generation else (0 if full.get("tokens") is not None else None)
    result["evaluation_tokens"] = full["by_stage"].get("evaluation", {}).get("tokens")
    result["update_tokens"] = full["by_stage"].get("update", {}).get("tokens")
    return result


def selected_summary(targeted):
    rows = []
    for label, run, task_id, scope in SELECTED:
        source = targeted / run
        if not source.is_dir():
            continue
        records = load_records(source)
        if task_id:
            records = [row for row in records if row["task_id"] == task_id]
        if not records:
            raise ValueError(f"Selected task missing: {run} {task_id}")
        metrics = aggregate_metrics([{**row, "run_id": run, "method": "selected", "phase": "test"}
                                     for row in records])
        condition = next(iter(next(iter(metrics.values())).values()))
        rows.append({"benchmark": label, "run": run, "scope": scope, "sample_count": len(records),
                     "metrics": {key: value["mean"] for key, value in condition["metrics"].items()},
                     "efficiency": compact_efficiency(records), "source_path": str(source / records[0]["source_path"])})
    return rows


def historical_curves(history):
    paths = sorted((history / "summary/evolution_curve").glob("*/*_evolution.jsonl"))
    curves, cached_sources, cached_snapshots, cached_seed_pools = [], {}, {}, {}

    def source_records(source_name, benchmark):
        normalized = Path(str(source_name or "").replace("\\", "/"))
        candidates = [
            history / normalized,
            history / normalized.parent.name / benchmark / normalized.name,
            history / "native_evolution" / benchmark / normalized.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def frozen_seed_pool(snapshot_hash, benchmark):
        key = (snapshot_hash, benchmark)
        if key in cached_seed_pools:
            return cached_seed_pools[key]
        if not snapshot_hash:
            cached_seed_pools[key] = None
            return None
        root = history / "native_evolution" / benchmark / "runs"
        pool = None
        if root.is_dir():
            for plan_path in root.glob("*/frozen_plan.json"):
                try:
                    plan = read_json(plan_path)
                except (OSError, ValueError, TypeError):
                    continue
                if plan.get("experience_hash") != snapshot_hash:
                    continue
                candidate = plan.get("AgentPool")
                if isinstance(candidate, dict) and candidate.get("profiles"):
                    pool = (candidate, str(plan_path))
                    break
        cached_seed_pools[key] = pool
        return pool

    for path in paths:
        telemetry = EvolutionTelemetry()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for original in sorted(rows, key=lambda row: row.get("round_index", 0)):
            if original.get("phase") != "evolution":
                continue
            row = {**original, "curve_kind": "historical task evolution", "curve_source": str(path)}
            benchmark = row.get("benchmark", path.parent.name)
            source_path = source_records(row.get("source_path"), benchmark)
            source_path_key = source_path or history / str(row.get("source_path", "")).replace("\\", "/")
            if source_path_key not in cached_sources:
                cached_sources[source_path_key] = ([json.loads(line) for line in source_path_key.read_text(encoding="utf-8-sig").splitlines()
                                                    if line.strip()] if source_path_key.is_file() else [])
            record = next((item for item in cached_sources[source_path_key]
                           if item.get("task_id") == row.get("task_id")
                           and item.get("round_index") == row.get("round_index")), {})
            record = dict(record)
            database_candidates = ([source_path.parent / "experience.sqlite",
                                    source_path.parent / benchmark / "experience.sqlite"] if source_path else [])
            snapshot_database = next((candidate for candidate in database_candidates if candidate.is_file()), None)
            if snapshot_database not in cached_snapshots:
                if snapshot_database is not None and snapshot_database.is_file():
                    with closing(sqlite3.connect(snapshot_database.resolve().as_uri() + "?mode=ro", uri=True)) as database:
                        cached_snapshots[snapshot_database] = {version: json.loads(body)
                                                               for version, body in database.execute("SELECT version,body FROM snapshots")}
                else:
                    cached_snapshots[snapshot_database] = {}
            snapshot_sources = []
            for label in ("before", "after"):
                version = record.get("experience_version_" + label)
                snapshot = cached_snapshots[snapshot_database].get(version)
                seed = None
                if version == 0 and snapshot is not None:
                    seed = frozen_seed_pool(record.get("snapshot_" + label + "_hash"), benchmark)
                    if seed is not None:
                        snapshot = {**snapshot, "agent_pool": seed[0]}
                if snapshot is not None:
                    record["pool_" + label] = snapshot.get("agent_pool")
                    if snapshot_database is not None:
                        snapshot_sources.append(f"{snapshot_database}#version={version}")
                    if seed is not None:
                        snapshot_sources.append(seed[1] + "#AgentPool")
            before, after = comparable_profiles(recorded_profiles(record, "before"), recorded_profiles(record, "after"))
            measured = telemetry.record_round(row.get("round_index", 0), pool_before=before, pool_after=after)
            for domain in CHANGE_DOMAINS:
                for key in (f"{domain}_hashes_before", f"{domain}_hashes_after", f"{domain}_changed_agent_count",
                            f"{domain}_changed_agent_ids", f"cumulative_{domain}_changed_agent_count"):
                    row[key] = measured[key]
            for key in ("harness_hashes_before", "harness_hashes_after", "harness_changed_agent_count",
                        "harness_changed_agent_ids", "cumulative_harness_changed_agent_count"):
                row["legacy_" + key] = original.get(key)
            row.update(telemetry_schema_version=measured["telemetry_schema_version"], change_hash_content=measured["change_hash_content"],
                       harness_modification_count=measured["harness_changed_agent_count"],
                       snapshot_measurement_source=str(source_path_key) if record else None,
                       state_snapshot_sources=snapshot_sources,
                       snapshot_measurement_available=bool(record) and measured["harness_changed_agent_count"] is not None,
                       snapshot_comparison_fields="Fields observed on both sides of each existing member; one-sided metadata omitted")
            curves.append(row)
    return curves


def released_campaign_summary(directory):
    report = directory / "test_report.json"
    if not report.is_file() or read_json(report).get("test_feedback_released") is not True:
        return []
    groups = defaultdict(list)
    with closing(sqlite3.connect((directory / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)) as database:
        for body, status, result in database.execute("SELECT body,status,result FROM slots"):
            slot = json.loads(body)
            if slot["kind"] != "test":
                continue
            receipt = directory / "test_evaluations" / (slot_hash(slot["slot_id"]) + ".json")
            value = read_json(receipt) if receipt.is_file() else {"complete": False, "status": status,
                "generation_budget": json.loads(result).get("budget") if result else None}
            value = dict(value)
            value.update(benchmark=slot["target"], task_id=slot["task_id"], run_id=f"{slot['source']}-C40",
                         method="targeted-v7", phase="test")
            if value.get("complete"):
                value["status"] = "complete"
            groups[slot["source"], slot["target"]].append(value)
    summaries = []
    for (source, target), records in sorted(groups.items()):
        metric_groups = aggregate_metrics(records)
        condition = next(iter(next(iter(metric_groups.values())).values()))
        summaries.append({"benchmark": target, "run": f"v7-{source}-C40", "scope": "v7 TEST",
            "sample_count": len(records), "complete_count": condition["scored_tasks"],
            "metrics": {key: value["mean"] if value["complete"] else None for key, value in condition["metrics"].items()},
            "efficiency": compact_efficiency(records), "source_path": str(report)})
    return summaries


def slot_hash(value):
    from jit_mas.schemas import digest
    return digest(value)


def batch_measurements(slots):
    budgets = [row["result"].get("budget") for row in slots]
    observed = [budget for budget in budgets if isinstance(budget, dict)]
    metrics = ("tokens", "model_calls", "tool_calls", "cost")
    budget = {key: complete_sum([item.get(key) if isinstance(item, dict) else None for item in budgets])
              for key in metrics}
    budget["records"] = [record for item in observed for record in item.get("records", [])]
    budget["usage_unknown"] = any(not isinstance(item, dict) or item.get("usage_unknown") for item in budgets)
    if budget["usage_unknown"]:
        budget.update({key: None for key in metrics})
    budget["by_stage"] = {}
    stage_names = {stage for item in observed for stage in (item.get("by_stage") or {})}
    for stage in stage_names:
        budget["by_stage"][stage] = {key: complete_sum([
            item["by_stage"].get(stage, {}).get(key, 0) if isinstance(item, dict)
            and isinstance(item.get("by_stage"), dict) else None for item in budgets]) for key in metrics}
    calls = [call_statistics(item) for item in budgets]
    latency = [row["result"].get("execution_seconds") for row in slots]
    observations = {f"observed_{key}": observed_sum([item.get(key) for item in observed])
                    for key in metrics}
    observations.update({f"unknown_{key}_slot_count": sum(not isinstance(item, dict) or number(item.get(key)) is None
                                                         for item in budgets) for key in metrics})
    for key in ("api_attempts", "failed_api_attempts", "retry_count"):
        values = [item[key] for item in calls]
        observations[f"observed_{key}"] = observed_sum(values)
        observations[f"unknown_{key}_slot_count"] = sum(value is None for value in values)
        observations[key] = complete_sum(values)
    observations.update(observed_latency_seconds=observed_sum(latency),
                        unknown_latency_slot_count=sum(number(item) is None for item in latency),
                        unknown_budget_slot_count=len(budgets) - len(observed),
                        usage_unknown=budget["usage_unknown"])
    return budget, complete_sum(latency), observations


def campaign_curves(directory):
    database_path = directory / "campaign.sqlite"
    if not database_path.is_file():
        return [], [], []
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        database.execute("BEGIN")
        slots = [{**json.loads(row["body"]), "status": row["status"],
                  "result": json.loads(row["result"]) if row["result"] else {},
                  "started_at": row["started_at"], "finished_at": row["finished_at"]}
                 for row in database.execute("SELECT * FROM slots ORDER BY ordinal")]
        checkpoints = {(row["source"], row["run_id"], row["position"]): json.loads(row["snapshot"])
                       for row in database.execute("SELECT * FROM checkpoints")}
        candidates = {(row["source"], row["run_id"], row["batch_index"], row["candidate_index"]): json.loads(row["body"])
                      for row in database.execute("SELECT * FROM candidates")}
        selections = {(row["source"], row["run_id"], row["batch_index"]): json.loads(row["body"])
                      for row in database.execute("SELECT * FROM batch_selections")}
    curves, candidate_rows, slot_rows = [], [], []
    for row in slots:
        result = row["result"]
        slot_rows.append({**{key: row.get(key) for key in ("source", "run_id", "kind", "batch_index",
            "candidate_index", "target", "task_id", "slot_id", "status", "started_at", "finished_at")},
            "score": result.get("score"), "execution_seconds": result.get("execution_seconds"),
            "budget": result.get("budget"), "error_type": result.get("error_type")})
    pairs = sorted({(row["source"], row["run_id"]) for row in slots})
    for source, run_id in pairs:
        telemetry = EvolutionTelemetry()
        for batch in range(8):
            batch_slots = [row for row in slots if (row["source"], row["run_id"], row.get("batch_index"))
                           == (source, run_id, batch) and row["kind"] != "test"]
            finished = [row for row in batch_slots if row["status"] not in ("pending", "started")]
            if not any(row["status"] in ("complete", "incomplete", "failed") for row in finished):
                continue
            selection = selections.get((source, run_id, batch), {})
            selected = selection.get("selected") or {}
            winner = candidates.get((source, run_id, batch, selected.get("candidate_index")), {})
            budget, latency, observations = batch_measurements(batch_slots)
            before = checkpoints.get((source, run_id, batch * 5))
            after = checkpoints.get((source, run_id, (batch + 1) * 5))
            from jit_mas.agent_pool import effective_pool
            from jit_mas.schemas import AgentPoolSnapshot
            effective_before = (effective_pool(AgentPoolSnapshot.model_validate(before["agent_pool"])).model_dump(mode="json")
                                if before is not None else None)
            effective_after = (effective_pool(AgentPoolSnapshot.model_validate(after["agent_pool"])).model_dump(mode="json")
                               if after is not None else None)
            operations = winner.get("decision", {}).get("pool_operations") if winner else None
            evo_scores = [row["result"].get("score") for row in finished if row["kind"] == "evolution"
                          and row["result"].get("complete")]
            evo_scores = [score for score in evo_scores if isinstance(score, (float, int))]
            val_scores = [row["result"].get("score") for row in finished if row["kind"] == "validation"
                          and row["result"].get("complete")]
            val_scores = [score for score in val_scores if isinstance(score, (float, int))]
            performance = {"winner_validation_score": selected.get("selection_utility"),
                           "evolution_task_mean": mean(evo_scores) if evo_scores else None,
                           "all_candidate_validation_mean": mean(val_scores) if val_scores else None}
            row = telemetry.record_round((batch + 1) * 5, performance=performance, budget=budget,
                latency_seconds=latency,
                pool_before=effective_before, pool_after=effective_after, operations_requested=operations,
                operations_applied=operations if after is not None else None,
                harness_before=effective_before, harness_after=effective_after,
                outcome={"complete": bool(selected) if selection else None})
            row.update(benchmark=source, run_id=f"v7-run{run_id}", phase="evolution", batch_index=batch,
                       curve_kind="v7 batch evolution", curve_source=str(database_path),
                       batch_terminal_slots=len(finished), batch_registered_slots=len(batch_slots),
                       batch_status=selection.get("status", "in progress"),
                       winner_candidate_index=selected.get("candidate_index"),
                       score_per_round=performance["winner_validation_score"],
                       latency_definition="sum of registered slot durations; parallel work may overlap; total null if any slot duration unknown",
                       **performance, **observations)
            row["api_calls"] = row["api_attempts"]
            row["stored_pool_agent_count_before"] = len(before["agent_pool"]["profiles"]) if before is not None else None
            row["stored_pool_agent_count_after"] = len(after["agent_pool"]["profiles"]) if after is not None else None
            row.update({"cumulative_" + key: value for key, value in row["cumulative"].items()})
            for kind, value in (row.get("operation_counts_applied") or {}).items():
                row["operation_applied_" + kind] = value
            for kind, value in (row.get("operation_counts_requested") or {}).items():
                row["operation_requested_" + kind] = value
            row["harness_modification_count"] = row.get("harness_changed_agent_count")
            curves.append(row)
            for index in range(3):
                candidate = candidates.get((source, run_id, batch, index))
                if candidate is None:
                    continue
                val = [item for item in finished if item["kind"] == "validation"
                       and item.get("candidate_index") == index]
                scores = [item["result"]["score"] for item in val if item["result"].get("complete")]
                candidate_rows.append({"benchmark": source, "run_id": run_id, "round_index": (batch + 1) * 5,
                    "candidate_index": index, "status": candidate["status"],
                    "selected": index == selected.get("candidate_index") if selected else None,
                    "validation_complete": len(scores), "validation_mean": mean(scores) if scores else None,
                    "state_hash": candidate.get("state_hash"), "decision": candidate.get("decision"),
                    "budget": candidate.get("budget"), "error_type": candidate.get("error_type")})
    return curves, candidate_rows, slot_rows


def scalar_rows(rows):
    return [{key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            | {f"performance_{key}": value for key, value in (row.get("performance") or {}).items()}
            for row in rows]


def nullable_cumulative(values):
    total, known, result = 0, True, []
    for value in values:
        measured = number(value)
        if measured is None:
            known = False
        elif known:
            total += measured
        result.append(total if known else None)
    return result


def operation_curve(rows, label, operation):
    aliases = ("proposed", "requested") if label == "proposed" else (label,)
    values = []
    for row in rows:
        value = None
        for alias in aliases:
            key = "operation_counts_cumulative_" + alias
            if isinstance(row.get(key), dict):
                value = row[key].get(operation)
                break
        values.append(number(value))
    return values


def plot_curves(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = defaultdict(list)
    for row in rows:
        groups[row["benchmark"], row["run_id"], row["curve_kind"]].append(row)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
                         "figure.facecolor": "white", "font.size": 10})
    paths = []
    for (benchmark, run_id, kind), curve in groups.items():
        curve.sort(key=lambda row: row["round_index"])
        rounds = [row["round_index"] for row in curve]
        figure, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
        figure.suptitle(f"{benchmark}  |  {run_id}  |  {kind}", fontsize=16)
        panels = axes.flat

        def values(field):
            return [row.get(field) if isinstance(row.get(field), (float, int)) else np.nan for row in curve]

        scores = values("score_per_round")
        rolling = [mean([value for value in scores[max(0, index - 4):index + 1] if math.isfinite(value)])
                   if any(math.isfinite(value) for value in scores[max(0, index - 4):index + 1]) else np.nan
                   for index in range(len(scores))]
        panels[0].plot(rounds, scores, "o-", alpha=0.5, markersize=3, label="Observed score")
        panels[0].plot(rounds, rolling, linewidth=2, label="5-point rolling mean")
        panels[0].set_title("Performance")
        panels[0].legend(fontsize=8)
        for axis, field, title, scale in ((panels[1], "tokens", "Tokens per round (thousands)", 1000),
                (panels[2], "latency_seconds", "Sum of slot durations (seconds)" if kind.startswith("v7") else "Duration per round (seconds)", 1),
                (panels[3], "cumulative_tokens", "Cumulative tokens (millions)", 1000000),
                (panels[4], "pool_agent_count_after", "Agent Pool members", 1),
                (panels[5], "model_calls", "Logical model calls", 1),
                (panels[8], "cumulative_failure_rate", "Cumulative uncommitted round rate", 1)):
            data = values(field)
            if field == "cumulative_tokens" and not any(math.isfinite(value) for value in data):
                data = [row.get("cumulative", {}).get("tokens", np.nan) for row in curve]
            axis.plot(rounds, np.array(data, dtype=float) / scale, "o-", markersize=3)
            axis.set_title(title)
        for operation in ("add", "delete", "split", "merge"):
            data = [value if value is not None else np.nan for value in operation_curve(curve, "applied", operation)]
            panels[6].plot(rounds, data, label=operation, linewidth=1.8)
        panels[6].set_title("Cumulative applied Pool operations")
        panels[6].legend(fontsize=8, ncol=2)
        modification = [row.get("harness_modification_count", row.get("harness_changed_agent_count")) for row in curve]
        modification = [value if isinstance(value, (int, float)) else np.nan for value in modification]
        panels[7].bar(rounds, modification, color="#778da9", label="Changes per round")
        cumulative = [value if value is not None else np.nan for value in nullable_cumulative(modification)]
        panels[7].plot(rounds, cumulative, color="#bc6c25", label="Cumulative changes")
        panels[7].set_title("Agent harness changes")
        panels[7].legend(fontsize=8)
        for axis in panels:
            axis.set_xlabel("Evolution tasks completed" if kind.startswith("v7") else "Evolution round")
            axis.set_xlim(max(0, min(rounds) - 1), max(rounds) + 1)
        stem = f"evolution_{benchmark}_{run_id}"
        for suffix in ("png", "svg", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180)
            paths.append(str(path))
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
        for domain, label in (("memory", "Memory"), ("profile_behavior", "Profile behavior"),
                              ("harness_policy", "Harness policy")):
            data = values(f"cumulative_{domain}_changed_agent_count")
            axis.plot(rounds, data, "o-", markersize=3, label=label)
        axis.set(title=f"{benchmark}: cumulative changes to existing agents", xlabel="Evolution round",
                 ylabel="Changed agent observations")
        axis.legend(fontsize=9)
        path = output / f"agent_changes_{benchmark}_{run_id}.png"
        figure.savefig(path, dpi=180)
        paths.append(str(path))
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
        stages = sorted({stage for row in curve for stage in (row.get("by_stage") or {})})
        if stages:
            axis.stackplot(rounds, *[[value / 1000 if (value := number((row.get("by_stage") or {}).get(stage, {}).get("tokens")))
                                     is not None else np.nan for row in curve] for stage in stages], labels=stages, alpha=0.8)
            axis.legend(loc="upper left", ncol=len(stages), fontsize=9)
        else:
            axis.text(0.5, 0.5, "Stage-level counters not available", transform=axis.transAxes, ha="center")
        axis.set(title=f"{benchmark}: token allocation by stage", xlabel="Evolution round", ylabel="Tokens (thousands)")
        path = output / f"token_stages_{benchmark}_{run_id}.png"
        figure.savefig(path, dpi=180)
        paths.append(str(path))
        plt.close(figure)
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        figure.suptitle(f"{benchmark}: reliability and structural proposals", fontsize=14)
        api_rates = [row["failed_api_attempts"] / row["api_attempts"] if row.get("api_attempts")
                     and isinstance(row.get("failed_api_attempts"), (int, float)) else np.nan for row in curve]
        axes[0, 0].plot(rounds, api_rates, "o-", markersize=3, color="#ad2831")
        axes[0, 0].set_title("API attempt failure rate")
        axes[0, 1].plot(rounds, values("actual_tokens"), label="Provider tokens")
        axes[0, 1].plot(rounds, values("estimated_tokens"), label="Estimated tokens")
        axes[0, 1].set_title("Token counter provenance")
        axes[0, 1].legend(fontsize=9)
        proposed_data = []
        for operation in ("add", "delete", "split", "merge", "specialize"):
            data = [value if value is not None else np.nan for value in operation_curve(curve, "proposed", operation)]
            if any(math.isfinite(value) for value in data):
                axes[1, 0].plot(rounds, data, label=operation)
                proposed_data.append(operation)
        axes[1, 0].set_title("Cumulative proposed Pool operations")
        if proposed_data:
            axes[1, 0].legend(fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5, "Proposal counts unavailable", transform=axes[1, 0].transAxes, ha="center")
        zeroed = [score if row.get("evaluation_complete", row.get("complete")) is True else
                  0 if row.get("evaluation_complete", row.get("complete")) is False else np.nan
                  for row, score in zip(curve, scores)]
        axes[1, 1].plot(rounds, scores, "o-", markersize=3, label="Observed score")
        axes[1, 1].plot(rounds, zeroed, ":", label="Confirmed unscored as zero")
        axes[1, 1].set_title("Scoring coverage")
        axes[1, 1].legend(fontsize=9)
        for axis in axes.flat:
            axis.set_xlabel("Evolution round")
        path = output / f"reliability_{benchmark}_{run_id}.png"
        figure.savefig(path, dpi=180)
        paths.append(str(path))
        plt.close(figure)
    return paths


def build(root, destination, campaign=None):
    destination.mkdir(parents=True, exist_ok=True)
    targeted = root / "outputs/targeted_optimization_20261010"
    selected = selected_summary(targeted)
    if campaign:
        selected.extend(released_campaign_summary(campaign.resolve()))
    run_rows = []
    for directory in sorted(targeted.iterdir()):
        if not directory.is_dir() or not ((directory / "records.jsonl").is_file() or (directory / "report.json").is_file()):
            continue
        records = load_records(directory)
        for benchmark in sorted({row["benchmark"] for row in records}):
            subset = [row for row in records if row["benchmark"] == benchmark]
            run_rows.append({"benchmark": benchmark, "run": directory.name, "scope": "recorded tasks",
                             **compact_efficiency(subset), "source_path": str(directory)})
    metrics = [{"benchmark": row["benchmark"], "run": row["run"], "scope": row["scope"],
                "task_count": row["sample_count"], "metric": key, "value": value, "source": row["source_path"]}
               for row in selected for key, value in row["metrics"].items()]
    curves = historical_curves(root / "outputs/high_score_experiment_20261009")
    candidate_rows, slot_rows = [], []
    if campaign:
        batch_curves, candidate_rows, slot_rows = campaign_curves(campaign.resolve())
        curves.extend(batch_curves)
    charts = plot_curves(curves, destination)
    suspension_path = root / "outputs/other_benchmarks_v7_20261010/campaign_suspension.json"
    suspension = read_json(suspension_path) if campaign and suspension_path.is_file() else None
    campaign_status = read_json(campaign / "status.json") if campaign and (campaign / "status.json").is_file() else None
    document = {"summary_rows": selected, "metric_rows": metrics, "run_rows": run_rows,
                "evolution_rows": scalar_rows(curves), "candidate_rows": candidate_rows,
                "campaign_slots": slot_rows, "charts": charts,
                "campaign_status_snapshot": campaign_status, "campaign_suspension": suspension,
                "definitions": {"cost": "Unavailable: endpoints did not return billing and no rate card was supplied",
                    "score": "Each benchmark retains its own native metric; single-task rows are not full-benchmark scores",
                    "historical_curve": "40 sequential WritingBench EVO tasks; different tasks per round",
                    "v7_curve": "Each point summarizes one five-task batch and its three candidates; in-progress batches are labelled",
                    "missing": "Missing observations remain null; unavailable USD costs are not zero",
                    "change_domains": "Memory, full profile behavior (without learned memory), and execution harness policy are distinct",
                    "legacy_harness": "Historical v1 hash includes learned memory; retained only for compatibility",
                    "campaign_totals": "Full batch totals require every registered slot measurement; observed_* are measured subsets",
                    "batch_latency": "Sum of slot durations; concurrent slots can overlap. It is not wall-clock campaign latency"}}
    write_json(destination / "dashboard.json", document)
    write_csv(destination / "official_metrics.csv", metrics)
    write_csv(destination / "efficiency.csv", [{key: value for key, value in row.items() if key not in ("metrics", "efficiency")}
              | row["efficiency"] for row in selected])
    write_csv(destination / "all_runs.csv", run_rows)
    write_csv(destination / "evolution_curves.csv", scalar_rows(curves))
    write_csv(destination / "batch_candidates.csv", candidate_rows)
    write_csv(destination / "campaign_slots.csv", slot_rows)
    (destination / "evolution_curves.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in curves), encoding="utf-8")
    lines = ["# 实验指标与进化曲线", "", "各行保留实际样本规模。成本未返回，记为空值。所有文件仅保存本地。", "",
             "| Benchmark | 样本 | 指标 | Token | API attempts | 平均延迟(s) |", "|---|---:|---|---:|---:|---:|"]
    for row in selected:
        efficiency = row["efficiency"]
        metric_text = "; ".join(f"{key}={value:.6f}" for key, value in row["metrics"].items() if value is not None) or "未完成评分"
        latency = efficiency['latency_seconds_mean']
        latency_text = f"{latency:.3f}" if isinstance(latency, (int, float)) else "缺失"
        lines.append(f"| {row['benchmark']} | {row['sample_count']} | {metric_text} | {efficiency['tokens']} | {efficiency['api_attempts']} | {latency_text} |")
    lines.extend(["", "## 随进化轮数变化", "", "历史 WritingBench 有 40 轮真实 EVO 数据。逐轮题目不同，分数曲线表示当轮任务表现；同一固定 VAL 的比较在新 v7 批次曲线中单列。",
                  "", "曲线含分数、Token、延迟、累积 Token、Agent Pool、API 调用、结构操作、harness 修改及失败率。未发生的操作保留为 0；缺失记录保留为空。", ""])
    for chart in charts:
        if chart.endswith(".png"):
            lines.extend([f"![{Path(chart).stem}]({Path(chart).name})", ""])
    lines.extend(["## 数据文件", "", "- `experiment_metrics.xlsx`：指标总表、官方指标、进化记录、所有版本效率。",
                  "- `evolution_curves.csv` / `.jsonl`：逐轮原始曲线数据。", "- `batch_candidates.csv`：三候选评分与选择。",
                  "- `campaign_slots.csv`：当前 campaign 全部槽位（含失败/缺失/未执行）。",
                  "- `all_runs.csv`：保留全部已记录版本，便于核对迭代费用。", ""])
    if campaign and (campaign / "status.json").is_file():
        status = read_json(campaign / "status.json")
        suspension_text = (f"运行状态：{suspension.get('status', suspension.get('reason', 'stopped_for_mechanism_repair'))}；"
                           "SQLite started 槽位为中断记录，不能视为仍在执行。" if suspension else
                           "运行状态依据保存的 status 快照，未单独确认进程状态。")
        lines.extend(["## 当前 v7 campaign", "", suspension_text,
                      f"状态快照：{status.get('updated_at', 'unknown')}；槽位状态：{status.get('statuses', {})}。",
                      f"原始记录：`{campaign}`。", "", "新曲线含进行中批次；尚未选出 winner 的分数与 Pool 状态为空。",
                      "Pool 曲线使用 effective Pool（空持久状态执行时有 6 个种子 agent）；stored Pool 数量另列。",
                      "批次 Token、API、延迟总量只在全部注册槽位已有测量时输出；observed_* 为已测子集，unknown_* 记录缺失量。",
                      "memory、行为配置与执行 policy 修改分别记录；旧含 memory 的 harness 计数仅作为 legacy 对照。", ""])
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary_rows": len(selected), "evolution_points": len(curves), "charts": len(charts), "output": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve(), args.campaign), ensure_ascii=True))


if __name__ == "__main__":
    main()
