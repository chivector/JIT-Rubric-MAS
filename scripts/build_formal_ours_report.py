"""Build a provenance-first Chinese report for the formal Ours campaign.

The script is intentionally read-only with respect to experiment inputs.  It
combines the dashboard, high-score telemetry and mechanism audit, then emits a
fresh report directory containing machine-readable tables and paper figures.
Missing measurements remain null; this is important for interrupted runs and
for endpoints which do not expose billing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jit_mas.experiment_telemetry import EvolutionTelemetry, OPERATION_KINDS
from scripts.summarize_high_score_experiment import (
    aggregate_metrics, aggregate_operations, benchmark_name, combined_budget,
    complete, evaluation, evolution_operations, summarize_efficiency, task_metrics,
    token_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD = ROOT / "outputs" / "benchmark_dashboard_20261010"
DEFAULT_AUDIT = ROOT / "outputs" / "evolution_mechanism_audit_20261010"
DEFAULT_HIGHSCORE = ROOT / "outputs" / "high_score_experiment_20261009"
OUTPUT_REL = Path("outputs") / "formal_ours_20261010" / "report"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any, digits: int = 4) -> str:
    number = scalar(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def load_sources(root: Path, dashboard: Path, audit: Path, highscore: Path) -> tuple[dict[str, Any], list[Path]]:
    document = read_json(dashboard / "dashboard.json", {}) or {}
    audit_raw = read_json(audit / "audit_raw.json", {}) or {}
    document["mechanism_audit"] = audit_raw
    document["audit_readme"] = (audit / "方法进化检查与验证说明.md").read_text(encoding="utf-8", errors="replace") if (audit / "方法进化检查与验证说明.md").is_file() else ""
    document["high_score_results"] = (highscore / "RESULTS_ZH.md").read_text(encoding="utf-8", errors="replace") if (highscore / "RESULTS_ZH.md").is_file() else ""
    sources: list[Path] = []
    for path in (dashboard / "dashboard.json", audit / "audit_raw.json", audit / "方法进化检查与验证说明.md",
                 highscore / "RESULTS_ZH.md", highscore / "summary" / "efficiency_summary.json",
                 highscore / "summary" / "benchmark_metrics.json", highscore / "summary" / "evolution_operations.json"):
        if path.is_file():
            sources.append(path)
    replay = root / "outputs" / "formal_ours_20261010" / "mechanism_replay" / "report.json"
    if replay.is_file():
        sources.append(replay)
    return document, sources


def discover_new_results(base: Path, output: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Read each terminal task once, preferring durable JSONL over summaries.

    A runner's report is an aggregate receipt, not an additional task. Different
    run directories remain distinct even when their slot IDs or task IDs match.
    Failed/interrupted tasks are retained with their recorded usage.
    """
    rows: list[dict[str, Any]] = []
    files: list[Path] = []
    if not base.is_dir():
        return rows, files
    base, output = base.resolve(), output.resolve()
    native = [path for path in sorted(base.rglob("records.jsonl")) if output not in path.parents]
    paths = native + [path for path in sorted(base.rglob("report.json"))
                      if output not in path.parents and not any(root.parent in path.parents for root in native)]
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        if path.name == "records.jsonl":
            records = read_jsonl(path)
        else:
            data = read_json(path, {})
            # Ignore report-level totals and mechanism-replay summaries.
            records = data.get("records", []) if isinstance(data, dict) else []
        accepted = False
        for index, record in enumerate(records if isinstance(records, list) else []):
            if not isinstance(record, dict) or not record.get("task_id"):
                continue
            if not any(key in record for key in ("budget", "outcome", "score_result", "evaluation")):
                continue
            phase = record.get("phase", "test")
            identity = (str(path.parent), record.get("slot_id") or record["task_id"], phase,
                        record.get("round_index"), record.get("attempt_id", record.get("attempt", 0)))
            if identity in seen:
                continue
            seen.add(identity)
            accepted = True
            rows.append({**record, "benchmark": benchmark_name(record.get("benchmark"), record["task_id"]),
                         "phase": phase, "run_id": record.get("run_id", path.parent.relative_to(base).as_posix()),
                         "method": record.get("method", "native"), "source": str(path),
                         "source_path": str(path), "record_index": index, "data_status": "new/formal_native"})
        if accepted:
            files.append(path)
            report = path.parent / "report.json"
            if report.is_file():
                files.append(report)
    return rows, files


def new_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    efficiencies = {(row["benchmark"], row["run_id"], row["method"], row["phase"]): row
                    for row in summarize_efficiency(rows)["by_condition"]}
    result: list[dict[str, Any]] = []
    for benchmark, conditions in aggregate_metrics(rows).items():
        for condition in conditions.values():
            run, method, phase = (condition[key] for key in ("run_id", "method", "phase"))
            items = [row for row in rows if (row["benchmark"], row["run_id"], row.get("method", "native"), row["phase"])
                     == (benchmark, run, method, phase)]
            result.append({"benchmark": benchmark, "run": run, "run_id": run, "method": method,
                           "scope": phase, "sample_count": condition["task_count"],
                           "complete_count": condition["execution_complete_tasks"],
                           "scored_count": condition["evaluation_complete_tasks"],
                           "failed_count": condition["failed_tasks"],
                           "metrics": {key: value["mean_scored_all"] for key, value in condition["metrics"].items()},
                           "metric_details": condition["metrics"],
                           "score_aggregation": "completed official evaluations; missing remains null",
                           "efficiency": efficiencies[benchmark, run, method, phase],
                           "source": sorted({str(item.get("source")) for item in items})})
    return result


def new_curve_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["benchmark"], record["run_id"], record.get("method", "native"), record["phase"]].append(record)
    result = []
    for (benchmark, run, method, phase), items in sorted(groups.items()):
        telemetry = EvolutionTelemetry()
        operation_totals = {label: {kind: 0 for kind in OPERATION_KINDS}
                            for label in ("proposed", "proposed_unique", "validated", "applied")}
        operation_known = {label: True for label in operation_totals}
        for index, record in enumerate(sorted(items, key=lambda row: (row.get("round_index", 0), row["task_id"]))):
            outcome = {**(record.get("outcome") or {}), "complete": complete(record)}
            budget = combined_budget(record)
            before, after = record.get("pool_before"), record.get("pool_after")
            # Empty persistent Pools have six effective seed profiles at execution.
            try:
                from jit_mas.agent_pool import effective_pool
                from jit_mas.schemas import AgentPoolSnapshot
                before = effective_pool(AgentPoolSnapshot.model_validate(before)).model_dump(mode="json") if before is not None else None
                after = effective_pool(AgentPoolSnapshot.model_validate(after)).model_dump(mode="json") if after is not None else None
            except (TypeError, ValueError):
                pass
            row = telemetry.record_round(record.get("round_index", index), performance=task_metrics(record),
                                         budget=budget, latency_seconds=record.get("latency_seconds"),
                                         pool_before=before, pool_after=after,
                                         operations_requested=record.get("operations_requested"),
                                         operations_applied=record.get("operations_applied"), outcome=outcome)
            row.update(data_status="new/formal_native", benchmark=benchmark, run_id=run, method=method,
                       phase=phase, task_id=record["task_id"], source_path=record.get("source_path"),
                       curve_kind="new native task trajectory", evaluation_complete=evaluation(record).get("complete"),
                       experience_version_before=record.get("experience_version_before"),
                       experience_version_after=record.get("experience_version_after"))
            row.update({"performance_" + key: value for key, value in task_metrics(record).items()})
            row.update({"cumulative_" + key: value for key, value in row["cumulative"].items()})
            row.update(token_statistics([budget]))
            operation_evidence = evolution_operations(record)
            row.update(operation_evidence)
            for label in operation_totals:
                counts = operation_evidence["operation_counts_" + label]
                if counts is None:
                    operation_known[label] = False
                else:
                    for kind, count in counts.items():
                        operation_totals[label][kind] += count
                        row[f"operation_{label}_{kind}"] = count
                for kind in OPERATION_KINDS:
                    row[f"cumulative_operation_{label}_{kind}"] = (operation_totals[label][kind]
                                                                 if operation_known[label] else None)
            for label in ("before", "after"):
                if scalar(record.get("agent_pool_size_" + label)) is not None:
                    row["pool_agent_count_" + label] = record["agent_pool_size_" + label]
            result.append(row)
    return result


def discover_batch_reports(base: Path, output: Path) -> list[dict[str, Any]]:
    """Collect aggregate batch campaign status without treating it as tasks."""
    reports = []
    seen_campaigns = set()
    for path in sorted(base.rglob("full_report.json")):
        if output in path.parents:
            continue
        data = read_json(path, {})
        if not isinstance(data, dict) or "registered_slots" not in data or "statuses" not in data:
            continue
        data = {**data, "source": str(path), "campaign": path.parent.name,
                "data_status": "new/formal_batch"}
        reports.append(data)
        seen_campaigns.add(path.parent.resolve())
    # Live campaigns may only have status.json until the aggregate report is
    # written; include that status without turning it into task-level scores.
    for path in sorted(base.rglob("status.json")):
        if output in path.parents or path.parent.resolve() in seen_campaigns:
            continue
        data = read_json(path, {})
        if not isinstance(data, dict) or "registered_slots" not in data or "statuses" not in data:
            continue
        reports.append({**data, "source": str(path), "campaign": path.parent.name,
                        "data_status": "new/formal_batch", "aggregate_kind": "live_status"})
    return reports


def latex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    def clean(value: Any) -> str:
        text = "N/A" if value is None or value == "" else str(value)
        substitutions = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "_": "\\_",
                         "#": "\\#", "$": "\\$", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}",
                         "^": "\\textasciicircum{}"}
        return "".join(substitutions.get(char, char) for char in text)
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule",
             " & ".join(title for _, title in columns) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(clean(row.get(key)) for key, _ in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_figures(curves: list[dict[str, Any]], output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    generated: list[str] = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        if scalar(row.get("round_index")) is not None:
            key = tuple(str(row.get(field, "")) for field in
                        ("data_status", "benchmark", "run_id", "method", "phase", "curve_kind"))
            groups[key].append(row)
    figure_index = []
    for key, history in sorted(groups.items()):
        history.sort(key=lambda row: scalar(row.get("round_index")) or 0)
        x = [scalar(row.get("round_index")) for row in history]
        historical = key[0].startswith("historical")
        is_primary = historical and key[1:5] == ("writingbench", "native_evolution", "native", "evolution")
        suffix = "" if is_primary else "_" + re.sub(r"[^A-Za-z0-9]+", "_", "_".join(key[:5])).strip("_")
        identity = f"{'Historical' if historical else 'New native'}: {key[1]} | {key[2]} | {key[4]} | n={len(history)}"
        is_batch = "batch" in key[-1]
        if is_batch:
            performance = [("performance_winner_validation_score", "selected VAL native score"),
                           ("performance_evolution_task_mean", "EVO native mean (different tasks)")]
        else:
            # The generic score and normalized score are usually identical.
            performance = [("performance_normalized_score", "normalized official score")]
            if not any(scalar(row.get("performance_normalized_score")) is not None for row in history):
                performance = [("performance_compliance", "compliance"), ("performance_F1", "F1"),
                               ("performance_Overall", "Overall"), ("score_per_round", "official score")]
        specs = [
            ("performance_curve", performance, "Official evaluation trace", "Score"),
            ("pool_size_curve", [("pool_agent_count_before", "before"), ("pool_agent_count_after", "after")], "Effective Agent Pool size", "Agents"),
            ("structure_operations_curve", [(f"cumulative_operation_applied_{kind}", kind) for kind in OPERATION_KINDS], "Applied structural operations", "Cumulative operations"),
            ("change_domains_curve", [("cumulative_memory_changed_agent_count", "learned memory"), ("cumulative_profile_behavior_changed_agent_count", "profile behavior"), ("cumulative_harness_policy_changed_agent_count", "harness policy"), ("cumulative_legacy_harness_changed_agent_count", "legacy hash (includes memory)")], "Change domains on existing members", "Cumulative member changes"),
        ]
        for stem, fields, title, ylabel in specs:
            fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=160)
            plotted = False
            for field, label in fields:
                values = [scalar(row.get(field)) for row in history]
                if field == "cumulative_legacy_harness_changed_agent_count":
                    values = [value if value is not None else scalar(row.get("legacy_cumulative_harness_changed_agent_count"))
                              for row, value in zip(history, values)]
                if not any(value is not None for value in values):
                    continue
                ax.plot(x, [value if value is not None else float("nan") for value in values], marker="o", linewidth=1.4, markersize=3, label=label)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(title + "\n" + identity, fontsize=10)
            ax.set_xlabel("Batch end position" if is_batch else "Task round (different tasks)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            if stem == "performance_curve":
                ax.text(0, -0.23, "Cross-task trace; this curve does not estimate improvement on a fixed evaluation set.",
                        transform=ax.transAxes, fontsize=8)
            fig.tight_layout()
            for extension in ("png", "pdf", "svg"):
                target = output / f"{stem}{suffix}.{extension}"
                fig.savefig(target, bbox_inches="tight")
                generated.append(str(target))
            figure_index.append({"stem": stem + suffix, "trajectory": list(key), "rows": len(history)})
            plt.close(fig)
        # Tokens and seconds need independent panels; cumulative and per-task
        # token scales also differ by a factor of the number of rounds.
        fields = [("tokens", "Tokens / task"), ("cumulative_tokens", "Cumulative tokens"),
                  ("latency_seconds", "Task latency (s)")]
        observed = [(field, label) for field, label in fields
                    if any(scalar(row.get(field)) is not None for row in history)]
        if observed:
            fig, axes = plt.subplots(len(observed), 1, figsize=(8.2, 2.25 * len(observed) + 0.6), dpi=160,
                                     sharex=True, squeeze=False)
            for (field, label), axis in zip(observed, axes[:, 0]):
                axis.plot(x, [scalar(row.get(field)) if scalar(row.get(field)) is not None else float("nan")
                              for row in history], marker="o", markersize=3, linewidth=1.3)
                axis.set_ylabel(label)
                axis.grid(alpha=0.25)
                if field != "latency_seconds":
                    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axes[0, 0].set_title("Efficiency trace\n" + identity, fontsize=10)
            axes[-1, 0].set_xlabel("Batch end position" if is_batch else "Task round (different tasks)")
            fig.tight_layout()
            for extension in ("png", "pdf", "svg"):
                target = output / f"token_latency_curve{suffix}.{extension}"
                fig.savefig(target, bbox_inches="tight")
                generated.append(str(target))
            figure_index.append({"stem": "token_latency_curve" + suffix, "trajectory": list(key), "rows": len(history)})
            plt.close(fig)
    write_json(output / "figure_index.json", figure_index)
    return generated


def build(root: Path, output: Path, dashboard: Path, audit: Path, highscore: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    document, source_files = load_sources(root, dashboard, audit, highscore)
    budget_audit = read_json(output / "budget_awareness_audit.json", {}) or {}
    preflight = read_json(root / "outputs" / "formal_ours_20261010" / "formal_v7_preflight.json", {}) or {}
    new_rows, new_files = discover_new_results(root / "outputs" / "formal_ours_20261010", output)
    batch_root = root / "outputs" / "formal_ours_20261010"
    batch_reports = discover_batch_reports(batch_root, output)
    # Live fallback campaigns are deliberately kept at repository root to
    # avoid Windows MAX_PATH limits. Include shallow root-level ``tb*`` and
    # ``tds*`` campaign receipts while keeping them as aggregate status
    # evidence rather than task-level scores.
    seen_batch_sources = {str(row.get("source")) for row in batch_reports}
    campaign_roots = {path for pattern in ("tb*", "tds*") for path in root.glob(pattern)}
    for campaign_root in sorted(campaign_roots):
        if not campaign_root.is_dir():
            continue
        for receipt in (campaign_root / "full_report.json", campaign_root / "status.json"):
            if not receipt.is_file() or str(receipt) in seen_batch_sources:
                continue
            data = read_json(receipt, {})
            if not isinstance(data, dict) or "registered_slots" not in data or "statuses" not in data:
                continue
            batch_reports.append({**data, "source": str(receipt),
                                  "campaign": campaign_root.name,
                                  "data_status": "new/formal_batch",
                                  "aggregate_kind": "live_status" if receipt.name == "status.json" else None})
            seen_batch_sources.add(str(receipt))
    source_files.extend(new_files)
    source_files.extend(Path(row["source"]) for row in batch_reports)
    summary_rows = [{"data_status": "historical/dashboard", **row} for row in document.get("summary_rows", []) if isinstance(row, dict)]
    new_summaries = [{"data_status": "new/formal_native", **row} for row in new_summary(new_rows)]
    summary_rows.extend(new_summaries)
    metric_rows = [{"data_status": "historical/dashboard", **row} for row in document.get("metric_rows", []) if isinstance(row, dict)]
    for row in new_summaries:
        for name, value in row["metrics"].items():
            detail = row["metric_details"][name]
            metric_rows.append({"data_status": "new/formal_native", "benchmark": row["benchmark"],
                                "run": row["run_id"], "scope": row["scope"], "task_count": row["sample_count"],
                                "scored_count": detail["scored_tasks"], "metric": name, "value": value,
                                "full_task_denominator_missing_as_zero": detail["mean_all_tasks_scored_missing_as_zero"],
                                "source": row["source"]})
    curve_rows = [{"data_status": "historical/dashboard", **row} for row in document.get("evolution_rows", []) if isinstance(row, dict)]
    new_curves = new_curve_rows(new_rows)
    curve_rows.extend(new_curves)
    audit_raw = document.get("mechanism_audit", {}) or {}
    historical = audit_raw.get("historical_writingbench", {}) or {}
    stopped = audit_raw.get("stopped_v7_campaign", {}) or {}
    actual_changes = historical.get("actual_profile_field_change_counts", {}) or {}
    operations = historical.get("applied_operations", {}) or {}
    mechanism_rows = [{"data_status": "historical/audit", "benchmark": "writingbench", "run": "native_evolution",
                       "scope": "evolution", "domain": key, "count": value} for key, value in actual_changes.items()]
    operation_rows = [{"data_status": "historical/audit", "benchmark": "writingbench", "run": "native_evolution",
                       "scope": "evolution", "operation": key, "applied_count": value}
                      for key, value in operations.items()]
    new_operation_summary = aggregate_operations(new_rows)
    for condition in new_operation_summary:
        for kind in OPERATION_KINDS:
            row = {"data_status": "new/formal_native", "benchmark": condition["benchmark"], "run": condition["run_id"],
                   "scope": condition["phase"], "operation": kind, "task_count": condition["task_count"]}
            for label in ("proposed", "proposed_unique", "validated", "applied"):
                counts = condition["operation_counts_" + label]
                row[label + "_count"] = counts.get(kind) if counts is not None else None
                row[label + "_known_tasks"] = condition["operation_counts_" + label + "_known_tasks"]
            operation_rows.append(row)
    curve_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in new_curves:
        curve_groups[row["benchmark"], row["run_id"], row["phase"]].append(row)
    for (benchmark, run, phase), items in sorted(curve_groups.items()):
        for domain in ("memory", "profile_behavior", "harness_policy", "legacy_harness"):
            values = [scalar(row.get(domain + "_changed_agent_count")) for row in items]
            mechanism_rows.append({"data_status": "new/formal_native", "benchmark": benchmark, "run": run,
                                   "scope": phase, "domain": domain, "count": sum(values) if all(value is not None for value in values) else None,
                                   "known_tasks": sum(value is not None for value in values), "task_count": len(items)})
    candidates = [row for row in document.get("candidate_rows", []) if isinstance(row, dict)]
    candidate_rows = [{"data_status": "historical/candidate", "benchmark": row.get("benchmark"), "run_id": row.get("run_id"),
                       "round_index": row.get("round_index"), "candidate_index": row.get("candidate_index"),
                       "status": row.get("status"), "selected": row.get("selected"),
                       "validation_complete": row.get("validation_complete"), "validation_mean": row.get("validation_mean"),
                       "error_type": row.get("error_type"), "state_hash": row.get("state_hash")} for row in candidates]

    summary_table_rows = []
    for row in summary_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        efficiency = row.get("efficiency") if isinstance(row.get("efficiency"), dict) else {}
        summary_table_rows.append({"data_status": row.get("data_status"), "benchmark": row.get("benchmark"),
                                   "run": row.get("run", row.get("run_id")), "scope": row.get("scope"), "n": row.get("sample_count"),
                                   "complete": row.get("complete_count", efficiency.get("complete_tasks")),
                                   "scored": row.get("scored_count", efficiency.get("evaluation_complete_tasks", efficiency.get("complete_tasks"))),
                                   "metric": "; ".join(f"{key}={fmt(value, 4)}" for key, value in metrics.items()) or fmt(row.get("score_mean")),
                                   "tokens": efficiency.get("tokens") if efficiency else row.get("tokens"),
                                   "latency": efficiency.get("latency_seconds_mean") if efficiency else row.get("latency_seconds_mean")})
    write_json(output / "report.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "summary_rows": summary_rows,
                                         "metric_rows": metric_rows, "candidate_rows": candidate_rows,
                                         "evolution_rows": curve_rows, "new_native_rows": new_rows,
                                         "budget_awareness_audit": budget_audit,
                                         "formal_v7_preflight": preflight,
                                         "mechanism": {"historical": historical, "stopped_v7": stopped,
                                                       "new_operation_summary": new_operation_summary,
                                                       "change_rows": mechanism_rows, "operation_rows": operation_rows,
                                                       "batch_campaigns": batch_reports},
                                         "definitions": {**document.get("definitions", {}),
                                             "new_metrics": "Completed official evaluations are averaged; unavailable scores remain null. Full-task missing-as-zero values are exported separately.",
                                             "new_task_counts": "Each task receipt counted once per run directory; report-level aggregates excluded; failed/interrupted tasks retained.",
                                             "new_curves": "Run, benchmark, phase and historical/new status have separate cumulative series.",
                                             "new_latency": "Mean observed task latency; known-row coverage is exported in efficiency.csv. Concurrent task sums are not campaign wall time."}})
    write_csv(output / "benchmark_summary.csv", summary_table_rows)
    write_csv(output / "official_metrics.csv", metric_rows)
    write_csv(output / "candidate_audit.csv", candidate_rows)
    write_csv(output / "evolution_curve.csv", curve_rows)
    efficiency_rows = [{"data_status": "historical/dashboard", **row} for row in document.get("run_rows", []) if isinstance(row, dict)]
    efficiency_rows.extend({"data_status": "new/formal_native", "benchmark": row["benchmark"], "run": row["run_id"],
                            "scope": row["scope"], **row["efficiency"]} for row in new_summaries)
    write_csv(output / "efficiency.csv", efficiency_rows)
    write_csv(output / "mechanism_changes.csv", mechanism_rows)
    write_csv(output / "structural_operations.csv", operation_rows)

    paper_table = latex_table(summary_table_rows, [("data_status", "Evidence"), ("benchmark", "Benchmark"), ("run", "Run"), ("scope", "Scope"), ("n", "$n$"), ("metric", "Native metric"), ("tokens", "Tokens"), ("latency", "Mean latency (s)")])
    (output / "table_benchmarks.tex").write_text("% Generated by build_formal_ours_report.py\n" + paper_table, encoding="utf-8")
    paper_eff = [{"data_status": row.get("data_status"), "benchmark": row.get("benchmark"), "run": row.get("run", row.get("run_id")),
                  "scope": row.get("scope"), "tokens": row.get("efficiency", {}).get("tokens") if isinstance(row.get("efficiency"), dict) else row.get("tokens"),
                  "api_attempts": row.get("efficiency", {}).get("api_attempts") if isinstance(row.get("efficiency"), dict) else None,
                  "failure_rate": row.get("efficiency", {}).get("api_failure_rate") if isinstance(row.get("efficiency"), dict) else None,
                  "cost": row.get("efficiency", {}).get("estimated_cost") if isinstance(row.get("efficiency"), dict) else None}
                 for row in summary_rows]
    (output / "table_efficiency.tex").write_text("% Generated by build_formal_ours_report.py\n" + latex_table(paper_eff, [("data_status", "Evidence"), ("benchmark", "Benchmark"), ("run", "Run"), ("scope", "Scope"), ("tokens", "Tokens"), ("api_attempts", "API attempts"), ("failure_rate", "API failure"), ("cost", "Estimated USD")]), encoding="utf-8")
    (output / "table_mechanism.tex").write_text("% Generated by build_formal_ours_report.py\n" + latex_table(mechanism_rows, [("data_status", "Evidence"), ("benchmark", "Benchmark"), ("run", "Run"), ("scope", "Scope"), ("domain", "Change domain"), ("count", "Count")]) + "\n" + latex_table(operation_rows, [("data_status", "Evidence"), ("benchmark", "Benchmark"), ("run", "Run"), ("scope", "Scope"), ("operation", "Operation"), ("applied_count", "Applied")]), encoding="utf-8")

    figures = make_figures(curve_rows, output)
    mechanism_note = ("历史 40 轮记录显示 40 次尝试、26 次提交、Pool 6→6；memory 更新 47 次，profile behavior/skills 1 次，"
                      "harness policy 0 次，Add/Delete/Prune/Split/Merge/Specialize/Reorganize 实际应用均为 0。"
                      "旧 harness 计数包含 memory hash，不能解释为 Python harness 代码编辑次数。")
    new_mechanism_note = []
    for condition in new_operation_summary:
        proposed = condition["operation_counts_proposed"] or {}
        applied = condition["operation_counts_applied"] or {}
        proposed_text = ", ".join(f"{kind}={proposed.get(kind, 0)}" for kind in OPERATION_KINDS if proposed.get(kind, 0)) or "none"
        applied_text = ", ".join(f"{kind}={applied.get(kind, 0)}" for kind in OPERATION_KINDS if applied.get(kind, 0)) or "none"
        new_mechanism_note.append(f"{condition['benchmark']}/{condition['run_id']}/{condition['phase']}: proposed {proposed_text}; applied {applied_text}")
    if new_mechanism_note:
        mechanism_note += " 新 native receipt：" + "；".join(new_mechanism_note) + "。"
    batch_note = ""
    if batch_reports:
        parts = []
        for report in batch_reports:
            parts.append(f"{report['campaign']} registered={report.get('registered_slots')} statuses={report.get('statuses')}")
        batch_note = "批次 campaign aggregate（不拆成重复 task receipt）：" + "；".join(parts) + "。"
    report_lines = ["# Ours 正式实验结果与机制审计报告", "", f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
                    "## 结论边界", "", "本报告将历史 dashboard、高分版本记录、机制审计和 `formal_ours_20261010` 下新 native 记录分开统计。历史记录与定向 targeted test 不代表完整官方 leaderboard；单题结果不外推为 benchmark 均值。缺失、未评分和费用未知保留为 null/—，不以 0 填充。"]
    if preflight.get("inputs_ready"):
        benches = ", ".join(f"{row.get('name')} EVO={row.get('evolution_tasks')} TEST={row.get('test_tasks')}" for row in preflight.get("benchmarks", []))
        report_lines.extend(["", "正式 v7 本地预检已通过（0 次模型调用）：" + benches + "。本地可运行计划为 `formal_v7_plan_local.json`。"])
    report_lines.extend(["", f"本次目录读取到 {len(new_rows)} 条独立任务 receipt（含失败/中断）；生成 report 与 records.jsonl 不重复计数。新 native 的评测完成数和执行完成数分别记录，不能把已调用但未评分的槽位当作结果。",
                         "", "## Benchmark 汇总", "", "| Evidence | Benchmark | Run | Scope | n | Scored | Native metric | Tokens | Mean latency (s) |", "|---|---|---|---|---:|---:|---|---:|---:|"])
    for row in summary_table_rows:
        report_lines.append(f"| {row['data_status']} | {row['benchmark']} | {row['run'] or '—'} | {row['scope']} | {row['n'] if row['n'] is not None else '—'} | {row['scored'] if row['scored'] is not None else '—'} | {row['metric']} | {row['tokens'] if row['tokens'] not in (None, '') else '—'} | {fmt(row['latency'], 3)} |")
    report_lines.extend(["", "## 机制审计", "", mechanism_note, "", batch_note,
                         f"停止的 v7 campaign 共注册 {stopped.get('registered_slots', '—')} slots；状态为 {stopped.get('statuses', {})}，batch/final selection={stopped.get('batch_selections', '—')}/{stopped.get('final_selections', '—')}。这些 pending/started 槽位不计为完成评测。"])
    if budget_audit:
        changed = budget_audit.get("tasks_with_live_budget_change")
        total = budget_audit.get("task_count")
        rate = budget_audit.get("live_budget_change_rate")
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "—"
        report_lines.extend(["", "## Budget-aware planning", "", f"对 {total if total is not None else '—'} 个 native task 的 `planning_calls.json` 与 `budget.json` 做了只读审计：{changed if changed is not None else '—'} 个任务在首次与末次规划之间看到剩余 token 预算变化（{rate_text}），{budget_audit.get('plans_with_explicit_budget', '—')} 个任务生成了显式 TeamBudgetPlan，超出当时剩余 token 的计划为 {budget_audit.get('plans_exceeding_remaining_tokens', '—')}。这些记录能够说明 live ledger 预算被传入规划及计划可行性，不能单独证明 allocation 随任务/剩余预算自适应，也不能识别预算机制带来的分数收益或资源节省。`adaptive_budget_enforcement` 是难度到 roster cap 的兼容字段，不把它误报成已启用的自适应机制。", "见 `budget_awareness_audit.json/.csv/.md`。"])
    report_lines.extend(["", "## Candidate 与费用", "", "候选状态、验证完成数、错误类型和 state hash 见 `candidate_audit.csv`。端点未返回账单且未提供价目表，`actual_cost` 与 `estimated_cost` 保持 null；失败调用保留其 token/attempt 记录。官方指标仅对已完成的评测取均值；缺分按零的完整槽位分母结果在 `official_metrics.csv` 的单独列中记录。", "", "## 曲线和图表", "", "历史、新 native、各 benchmark、run 和阶段使用独立曲线与累计值，不跨轨迹连接。历史逐轮曲线使用不同 WritingBench 任务，不能把跨题分数连线解释为固定任务持续提升。固定题配对 pilot 另保留在机制审计目录。图中 memory、profile behavior、harness policy 与 legacy hash 分开绘制；Token 与秒使用独立面板。"])
    for figure in figures:
        if Path(figure).suffix == ".png":
            report_lines.append(f"\n![{Path(figure).stem}]({Path(figure).name})")
    report_lines.extend(["", "## 数据文件", "", "- `report.json`：汇总、曲线、机制和新 native 记录。", "- `benchmark_summary.csv` / `official_metrics.csv`：官方指标与样本规模。", "- `efficiency.csv`：Token、API attempts、失败率、latency、费用。", "- `candidate_audit.csv`：候选生成/验证/选择状态。", "- `mechanism_changes.csv` / `structural_operations.csv`：memory、行为、policy 与结构操作分域统计。", "- `table_*.tex`：论文表格；`*.png`、`*.pdf`、`*.svg`：静态图。", "", "## Targeted test tuning", "", "定向 checklist、source supplement、lexical repair 和事实矩阵等属于 targeted optimization；报告保留其原始 scope 和来源，不把针对题目/评测器的修复结果包装成未调参的通用能力。"])
    (output / "report_zh.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_entries = []
    for path in sorted(set(source_files)):
        if path.is_file():
            manifest_entries.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path), "role": "input"})
    for path in sorted(output.iterdir()):
        if path.name == "provenance_manifest.json" or not path.is_file():
            continue
        manifest_entries.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path), "role": "generated"})
    write_json(output / "provenance_manifest.json", {"schema": "formal-ours-report-v2", "generated_at": datetime.now(timezone.utc).isoformat(),
                                                      "root": str(root), "dashboard": str(dashboard), "audit": str(audit), "highscore": str(highscore),
                                                      "historical_vs_new": {"historical_summary_rows": len(summary_rows) - len(new_summaries), "new_native_task_rows": len(new_rows), "new_native_summary_rows": len(new_summaries), "new_batch_campaigns": len(batch_reports)},
                                                      "entries": manifest_entries})
    return {"output": str(output), "summary_rows": len(summary_rows), "new_native_rows": len(new_rows), "curve_rows": len(curve_rows), "figures": len(figures), "sources": len(source_files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--highscore", type=Path, default=DEFAULT_HIGHSCORE)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_REL)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve(), args.dashboard.resolve(), args.audit.resolve(), args.highscore.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
