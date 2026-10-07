"""Read released independent TEST receipts and report paper statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sqlite3
from statistics import mean

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.independent_batch_campaign import slot_registry as batch_slot_registry, state_sources
from jit_mas.independent_protocol import RUN_IDS, SOURCES, slot_registry
from jit_mas.schemas import ExperienceSnapshot, digest
from jit_mas.token_usage import merge_usage, summarize_budget, summarize_outcome


COMPARATORS = ("ours_initial", "jit_matched", "rubric_fixed")
FAMILY_SIZES = {"primary_main": 6, "primary_diagnostic": 6,
                "secondary_main": 12, "secondary_diagnostic": 12}
REGISTERED_TEST_SLOTS = 2751


def _registered_test_slots(registry):
    return int(registry.get("counts", {}).get("test", sum(
        row.get("kind") == "test" for row in registry.get("artifacts", []))))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise CheckpointIntegrityError(message)


def budget_audit(budget):
    valid = isinstance(budget, dict) and any(key in budget for key in ("tokens", "model_calls", "records"))
    records = budget.get("records") or [] if valid else []
    malformed = any(not isinstance(record, dict) or any(type(record.get(key)) is not int or record[key] < 0
        for key in ("input_tokens", "output_tokens")) for record in records
        if not isinstance(record, dict) or record.get("kind", "model") == "model")
    reserved = budget.get("reserved_tokens", 0) if isinstance(budget, dict) else 0
    unsettled = type(reserved) is not int or reserved != 0
    return {"unknown": not valid or malformed or unsettled or bool(budget.get("usage_unknown")) if isinstance(budget, dict) else True,
            "unsettled": unsettled, "reserved_tokens": reserved if type(reserved) is int and reserved >= 0 else None}


def cost_accounting(rows):
    rows, stages = list(rows), {}
    usage = merge_usage(row.get("token_usage") for row in rows)
    audits = [row.get("budget_audit", {"known_unattempted": False, "stages": {}}) for row in rows]
    unknown = sum(any(stage["unknown"] for stage in row["stages"].values()) for row in audits)
    unsettled = sum(any(stage["unsettled"] for stage in row["stages"].values()) for row in audits)
    for name in ("generation", "evaluation"):
        budgets = [row["stages"][name] for row in audits if name in row["stages"]]
        stages[name] = {"attempted_slots": len(budgets), "unknown_budget_slots": sum(row["unknown"] for row in budgets),
                       "unsettled_budget_slots": sum(row["unsettled"] for row in budgets),
                       "recorded_reserved_tokens": sum(row["reserved_tokens"] or 0 for row in budgets)}
    return {"recorded_tokens": usage["total_tokens"], "estimated_tokens": usage["estimated_tokens"],
            "provider_tokens_lower_bound": usage["total_tokens"] - usage["estimated_tokens"],
            "total_tokens_exact": usage["total_tokens"] if not (unknown or unsettled or usage["estimated_tokens"]) else None,
            "recorded_usage_lower_bound": bool(unknown or unsettled),
            "unknown_budget_slots": unknown, "unsettled_budget_slots": unsettled, "by_stage": stages,
            "known_unattempted_missing_slots": sum(row["known_unattempted"] for row in audits)}


def load_test_data(directory):
    directory = Path(directory).resolve()
    with closing(sqlite3.connect((directory / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)) as database:
        database.execute("BEGIN")
        metadata = {key: json.loads(body) for key, body in database.execute("SELECT key,body FROM metadata")}
        rows = database.execute(
            "SELECT body,status,result,result_hash,started_at,finished_at FROM slots ORDER BY ordinal").fetchall()
        selections = {(source, run_id): json.loads(body) for source, run_id, body in database.execute(
            "SELECT source,run_id,body FROM selections")}
        checkpoints = {(source, run_id, position): (state_hash, snapshot, identity_hash)
                       for source, run_id, position, state_hash, snapshot, identity_hash in database.execute(
                           "SELECT source,run_id,position,state_hash,snapshot,identity_hash FROM checkpoints")}
    registration = metadata["registration"]
    protocol = registration["protocol"]
    batch_protocol = protocol.get("version", "").startswith("jit-compose-independent-batch-protocol-v")
    v7 = protocol.get("version") == "jit-compose-independent-batch-protocol-v7"
    if batch_protocol:
        expected_schema = "independent-batch-campaign-v7" if v7 else "independent-batch-campaign-v6"
        require(registration.get("schema") == expected_schema, "Batch campaign registration schema mismatch")
    registry = (batch_slot_registry if batch_protocol else slot_registry)(protocol)["artifacts"]
    registered_test_slots = sum(row.get("kind") == "test" for row in registry)
    require([json.loads(row[0]) for row in rows] == registry, "Registered slot inventory changed")
    tests = [{**json.loads(body), "status": status, "result": json.loads(result) if result else {},
              "result_hash": result_hash, "started_at": started, "finished_at": finished}
             for body, status, result, result_hash, started, finished in rows
             if json.loads(body)["kind"] == "test"]
    seal = metadata.get("test_seal")
    require(seal is not None, "Global TEST seal is required before reading evaluation receipts")
    require(len(tests) == registered_test_slots and all(row["status"] in {"submitted", "failed", "missing"} for row in tests),
            "Global TEST inventory is not terminal")
    require(seal == {"registration_hash": digest(registration), "required_test_slots": registered_test_slots,
                     "submission_hashes": {row["slot_id"]: row["result_hash"] for row in tests}},
            "Global TEST seal differs from registered inventory")
    report_path = directory / "test_report.json"
    require(report_path.is_file(), "Published test_report.json is required before reading evaluation receipts")
    published = read_json(report_path)
    require(published.get("schema") in {"independent-test-report-v5", "independent-test-report-v7"}
            and published.get("test_feedback_released") is True
            and published.get("slots") == published.get("evaluated") == len(tests),
            "Published TEST report does not release the complete inventory")
    execution = registration["execution"]
    adapter = {"schema": "independent-test-adapter-identity-v1",
               "adapter_sha256": hashlib.sha256(Path(__file__).with_name("run_independent_test_release.py").read_bytes()).hexdigest(),
               "protocol_sha256": registration["protocol"]["protocol_sha256"],
               **{name: execution[name] for name in ("launch_file_sha256", "configuration_file_sha256", "joint_manifest_sha256")},
               "evolution_runner_sha256": execution["runner_sha256"]}
    require(read_json(directory / "test_adapter_identity.json") == adapter, "TEST adapter execution identity mismatch")
    if any(row["source"] == "static" for row in tests):
        require(execution["initial_snapshot_hash"] == digest(ExperienceSnapshot()), "Static initial state identity mismatch")
    states = {}
    for key, selection in selections.items():
        require(selection.get("registration_hash") == digest(registration)
                and (selection.get("source"), selection.get("run_id")) == key, "Selection registration identity mismatch")
        selected = selection.get("selected")
        if selected:
            if batch_protocol:
                require(selected.get("position") == 40, "Batch TEST state must be the final batch winner")
            checkpoint = checkpoints.get((*key, selected["position"]))
            require(checkpoint is not None and checkpoint[0] == selected["state_hash"]
                    and checkpoint[2] == digest(registration)
                    and digest(json.loads(checkpoint[1])) == checkpoint[0],
                    "Selected checkpoint identity mismatch")
            snapshot = ExperienceSnapshot.model_validate_json(checkpoint[1])
            if batch_protocol:
                trajectory = next((row for row in protocol["trajectories"]
                                   if (row["source"], row["run_id"]) == key), None)
                require(trajectory is not None and state_sources(snapshot)
                        <= set(trajectory["evolution_task_ids"]), "Selected state contains history outside source EVO")
            states[key] = checkpoint[0]
    expected_paths = {f"{digest(row['slot_id'])}.json" for row in tests if row["status"] == "submitted"}
    require({path.name for path in (directory / "test_evaluations").glob("*.json")} == expected_paths,
            "Evaluation receipt inventory differs from submitted slots")
    groups, observations, usages = {}, [], []
    for row in tests:
        result = row["result"]
        require(digest(result) == row["result_hash"], "Slot result hash mismatch")
        require(result.get("task_id") == row["task_id"] and result.get("evaluation") is None,
                "Slot result changed its task or unscored submission")
        if row["status"] == "missing":
            require(result.get("registration_hash") == digest(registration)
                    and (row["source"], row["run_id"]) in selections
                    and selections[row["source"], row["run_id"]].get("selected") is None
                    and result.get("error_type") == "NoEligibleSelectedState",
                    "Missing slot registration identity mismatch")
        else:
            expected_state = execution["initial_snapshot_hash"] if row["source"] == "static" else states.get((row["source"], row["run_id"]))
            require(expected_state is not None and result.get("state_hash") == expected_state,
                    "TEST result state differs from selected or initial state")
        receipt, score, status, official_metrics = None, None, row["status"], None
        usage = summarize_budget(result.get("budget"))
        unattempted = status == "missing" and row["started_at"] is None
        audit = {"known_unattempted": unattempted, "stages": {} if unattempted else {"generation": budget_audit(result.get("budget"))}}
        if status == "submitted":
            submission = read_json(result["submission_path"])
            require(digest(submission) == result.get("submission_hash")
                    and digest(submission.get("answer")) == result.get("answer_hash") == submission.get("answer_hash")
                    and isinstance(submission.get("answer"), str) and submission["answer"].strip()
                    and submission.get("evaluation") is None,
                    "Submission hash mismatch")
            receipt = read_json(directory / "test_evaluations" / f"{digest(row['slot_id'])}.json")
            require(receipt.get("receipt_sha256") == digest({key: value for key, value in receipt.items()
                                                           if key != "receipt_sha256"})
                    and receipt.get("slot") == row
                    and receipt.get("submission_sha256") == result["submission_hash"]
                    and receipt.get("generation_budget") == result.get("budget"), "Evaluation receipt hash or binding mismatch")
            usage = receipt.get("token_usage")
            audit["stages"]["evaluation"] = budget_audit(receipt.get("evaluation_budget"))
            require(usage == summarize_outcome({key: value for key, value in receipt.items() if key != "token_usage"}),
                    "Evaluation receipt token usage mismatch")
            status = "complete" if receipt.get("complete") is True else "failed"
            if status == "complete":
                evaluation = receipt.get("evaluation", {})
                official = receipt.get("official_score")
                official_metrics = evaluation.get("official_metrics")
                if row["target"] == "deepresearch_bench_ii" and not isinstance(official_metrics, dict):
                    dimensions = evaluation.get("dimensions", {})
                    official_metrics = {
                        "Overall": official,
                        "InformationRecall": (dimensions.get("info_recall") or {}).get("score"),
                        "Analysis": (dimensions.get("analysis") or {}).get("score"),
                        "Presentation": (dimensions.get("presentation") or {}).get("score"),
                    }
                require(evaluation.get("complete") is True and evaluation.get("score") == official
                        and evaluation.get("task_id") == row["task_id"]
                        and evaluation.get("evaluator_version") == evaluation.get("raw", {}).get("evaluator_version")
                        and isinstance(evaluation.get("evaluator_version"), str)
                        and receipt.get("answer_hash") == evaluation.get("raw", {}).get("submission_answer_hash") == result["answer_hash"],
                        "Complete evaluation binding mismatch")
                score = evaluation.get("raw", {}).get("native_mean") if row["target"] == "writingbench" else official
                require(type(score) in (int, float) and math.isfinite(score)
                        and type(official) in (int, float) and math.isfinite(official), "Complete scores must be finite")
                if row["target"] == "writingbench":
                    require(1 <= score <= 10 and math.isclose((score - 1) / 9, official,
                                                              rel_tol=1e-12, abs_tol=1e-12),
                            "WritingBench native and normalized scores disagree")
                elif row["target"] in {"deepsearchqa", "deepresearch_bench_ii", "ifeval", "ifbench"}:
                    require(0 <= official <= 1, "Benchmark score is outside its [0,1] range")
        usages.append(usage)
        key = (row["source"], row["run_id"], row["target"], row["method"])
        group = groups.setdefault(key, {"source": key[0], "run_id": key[1], "target": key[2], "method": key[3],
                                       "scores": [], "complete": 0, "slots": 0})
        group["slots"] += 1
        if status == "complete":
            group["complete"] += 1
            group["scores"].append(receipt["official_score"])
        observations.append({**{name: row[name] for name in ("source", "run_id", "target", "method", "task_id")},
                             "native_score": score, "official_metrics": official_metrics,
                             "status": status, "token_usage": usage, "budget_audit": audit})
    for group in groups.values():
        group["mean_official_score"] = sum(group["scores"]) / group["slots"] if group["complete"] == group["slots"] else None
    published_groups = {(row["source"], row["run_id"], row["target"], row["method"]): row
                        for row in published.get("by_condition", [])}
    require(len(published_groups) == len(published.get("by_condition", [])) and published_groups == groups,
            "Published TEST condition inventory or scores differ from receipts")
    complete = sum(row["status"] == "complete" for row in observations)
    total_usage = merge_usage(usages)
    require(published.get("complete") == complete and published.get("incomplete_or_failed") == len(tests) - complete
            and published.get("token_usage") == total_usage, "Published TEST totals differ from receipts")
    return registration, observations, total_usage


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def task_statistics(values, *, seed, iterations, inferential):
    if not values:
        return {"mean_difference": None, "bootstrap_95_ci": None, "sign_flip_pvalue": None}
    generator = random.Random(seed)
    sampled = sorted(mean(generator.choices(values, k=len(values))) for _ in range(iterations))
    probability = None
    if inferential:
        threshold = abs(sum(values)) - 1e-12
        if len(values) <= 16:
            flips = itertools.product((-1, 1), repeat=len(values))
            extreme = sum(abs(sum(value * sign for value, sign in zip(values, signs))) >= threshold for signs in flips)
            probability = extreme / 2 ** len(values)
        else:
            extreme = sum(abs(sum(value * generator.choice((-1, 1)) for value in values)) >= threshold
                          for _ in range(iterations))
            probability = (extreme + 1) / (iterations + 1)
    return {"mean_difference": mean(values), "bootstrap_95_ci": [percentile(sampled, 0.025), percentile(sampled, 0.975)],
            "sign_flip_pvalue": probability}


def paired_comparisons(observations, *, seed, iterations):
    # Ours-only v7 deliberately has no baseline rows or inferential contrasts.
    if not any(row["method"] in COMPARATORS for row in observations):
        return []
    scores = {(row["source"], row["run_id"], row["target"], row["method"], row["task_id"]): row["native_score"]
              for row in observations}
    require(len(scores) == len(observations), "Duplicate condition-task observations")
    conditions = sorted({(row["source"], row["target"]) for row in observations if row["method"] == "ours_selected"})
    comparisons = []
    for source, target in conditions:
        task_ids = sorted({key[4] for key in scores if key[:1] == (source,) and key[2] == target})
        for comparator in COMPARATORS:
            tasks, run_rows = [], []
            for task_id in task_ids:
                baseline = scores.get(("static", None, target, comparator, task_id))
                differences = [None if baseline is None or (selected := scores.get(
                    (source, run_id, target, "ours_selected", task_id))) is None else selected - baseline for run_id in RUN_IDS]
                tasks.append({"task_id": task_id, "run_differences": differences,
                              "mean_difference_three_runs": mean(differences) if all(value is not None for value in differences) else None})
            values = [row["mean_difference_three_runs"] for row in tasks if row["mean_difference_three_runs"] is not None]
            full = len(values) == len(tasks)
            for run_id in RUN_IDS:
                complete_pairs = [row["run_differences"][run_id] for row in tasks if row["run_differences"][run_id] is not None]
                run_rows.append({"run_id": run_id, "complete_pairs": len(complete_pairs),
                                 "mean_difference_all_complete": mean(complete_pairs) if len(complete_pairs) == len(tasks) else None,
                                 "mean_difference_complete_pair_descriptive": mean(complete_pairs) if complete_pairs else None})
            condition_seed = seed + int(digest([source, target, comparator])[:16], 16)
            stats = task_statistics(values, seed=condition_seed, iterations=iterations, inferential=full)
            family = ("primary" if comparator == "ours_initial" else "secondary") + (
                "_diagnostic" if target in {"ifeval", "ifbench"} else "_main")
            comparisons.append({"source": source, "target": target, "comparator": comparator,
                "scope": "source_test" if source == target else "migration_test", "family": family,
                "registered_tasks": len(tasks), "complete_task_clusters": len(values), "all_complete": full,
                "mean_difference_all_complete": stats["mean_difference"] if full else None,
                "complete_pair_descriptive": {"label": "Descriptive complete three-run task pairs; excludes missing pairs",
                                              **stats}, "bootstrap_95_ci_all_complete": stats["bootstrap_95_ci"] if full else None,
                "sign_flip_pvalue": stats["sign_flip_pvalue"], "holm_pvalue": None,
                "runs": run_rows, "paired_tasks": tasks})
    for family, size in FAMILY_SIZES.items():
        members = [row for row in comparisons if row["family"] == family]
        require(len(members) == size, "Comparison family differs from registered 6/6/12/12 inventory")
        ordered = sorted((row for row in members if row["sign_flip_pvalue"] is not None), key=lambda row: row["sign_flip_pvalue"])
        previous = 0
        for index, row in enumerate(ordered):
            previous = max(previous, min(1, (size - index) * row["sign_flip_pvalue"]))
            row["holm_pvalue"] = previous
    return comparisons


def summarize(directory, *, seed=0, iterations=10000):
    if type(iterations) is not int or iterations < 100:
        raise ValueError("At least 100 fixed bootstrap/sign-flip iterations are required")
    registration, observations, usage = load_test_data(directory)
    groups = defaultdict(list)
    for row in observations:
        groups[row["source"], row["run_id"], row["target"], row["method"]].append(row)
    conditions = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        values = [row["native_score"] for row in rows if row["native_score"] is not None]
        statuses = Counter(row["status"] for row in rows)
        metric_rows = [row.get("official_metrics") for row in rows
                       if isinstance(row.get("official_metrics"), dict)]
        official_metric_means = None
        if metric_rows:
            names = ("Overall", "InformationRecall", "Analysis", "Presentation")
            official_metric_means = {name: (mean([item[name] for item in metric_rows
                                                  if isinstance(item.get(name), (int, float))])
                                            if any(isinstance(item.get(name), (int, float)) for item in metric_rows)
                                            else None)
                                     for name in names}
        conditions.append({"source": key[0], "run_id": key[1], "target": key[2], "method": key[3],
            "registered_slots": len(rows), "complete": statuses["complete"], "failed": statuses["failed"],
            "missing": statuses["missing"], "native_mean_all_complete": mean(values) if len(values) == len(rows) else None,
            "native_mean_complete_descriptive": mean(values) if values else None,
            "token_usage": merge_usage(row["token_usage"] for row in rows),
            "cost_accounting": cost_accounting(rows),
            "official_metric_means": official_metric_means,
            "task_scores": [{**{name: row[name] for name in ("task_id", "native_score", "status")},
                             "official_metrics": row.get("official_metrics")} for row in rows]})
    protocol = registration["protocol"]
    ours_only = protocol.get("version") == "jit-compose-independent-batch-protocol-v7"
    return {"schema": "independent-paper-summary-v1", "registration_sha256": digest(registration),
        "protocol_sha256": registration["protocol"]["protocol_sha256"], "seed": seed, "iterations": iterations,
        "statistical_unit": ("Task; one registered run per source; no cross-method inference" if ours_only else
                             "Task; three run differences are averaged within each source-target-task before resampling"),
        "inference": ("Descriptive Ours-only results; no comparator or p-value claims" if ours_only else
                      "Two-sided sign-flip; exact up to 16 tasks, otherwise Monte Carlo with plus-one correction"),
        "missing_policy": ("Null; full means require every registered task" if ours_only else
                           "Null; full means and inferential tests require every registered task and all three runs"),
        "cost_policy": ("Each physical TEST slot counted once; no static baseline artifacts" if ours_only else
                        "Each physical TEST slot counted once; static artifacts shared across sources and runs"),
        "family_sizes": {} if ours_only else FAMILY_SIZES, "conditions": conditions, "token_usage": usage, "cost_accounting": cost_accounting(observations),
        "comparisons": paired_comparisons(observations, seed=seed, iterations=iterations)}


def markdown(report):
    def display(value):
        return "null" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = ["# Independent TEST paper summary", "", report["statistical_unit"] + ".",
             report["missing_policy"] + ".", report["cost_policy"] + ".", "",
             "## Native scores", "", "| Source | Run | Target | Method | Complete | Failed | Missing | Full mean | Descriptive complete mean |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in report["conditions"]:
        lines.append("| " + " | ".join(display(row[key]) for key in ("source", "run_id", "target", "method", "complete", "failed", "missing", "native_mean_all_complete", "native_mean_complete_descriptive")) + " |")
    drb_rows = [row for row in report["conditions"]
                if row["target"] == "deepresearch_bench_ii" and row.get("official_metric_means")]
    if drb_rows:
        lines += ["", "## DeepResearch Bench II official metrics", "",
                  "| Source | Run | Method | Overall | InformationRecall | Analysis | Presentation |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for row in drb_rows:
            metrics = row["official_metric_means"]
            lines.append("| " + " | ".join(display(value) for value in
                (row["source"], row["run_id"], row["method"], metrics.get("Overall"),
                 metrics.get("InformationRecall"), metrics.get("Analysis"), metrics.get("Presentation"))) + " |")
    lines += ["", "## Paired differences", "", "Complete-pair results are descriptive when any registered pair is missing.", "",
              "| Source | Target | Comparator | Task clusters | Full difference | Full 95% CI | Descriptive difference | Sign-flip p | Holm p |",
              "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for row in report["comparisons"]:
        interval = row["bootstrap_95_ci_all_complete"]
        cells = [row["source"], row["target"], row["comparator"], f"{row['complete_task_clusters']}/{row['registered_tasks']}",
                 row["mean_difference_all_complete"], "null" if interval is None else ", ".join(display(value) for value in interval),
                 row["complete_pair_descriptive"]["mean_difference"], row["sign_flip_pvalue"], row["holm_pvalue"]]
        lines.append("| " + " | ".join(display(value) for value in cells) + " |")
    costs = report["cost_accounting"]
    lines += ["", f"Seed: {report['seed']}; iterations: {report['iterations']}; Holm families: 6/6/12/12.",
              f"TEST recorded usage: {report['token_usage']['model_calls']} calls, {costs['recorded_tokens']} tokens; exact total: {display(costs['total_tokens_exact'])}.",
              f"Provider token lower bound: {costs['provider_tokens_lower_bound']}; estimated tokens: {costs['estimated_tokens']}; unknown slots: {costs['unknown_budget_slots']}; unsettled slots: {costs['unsettled_budget_slots']}.",
              "Generation/evaluation unknown slots: " + "/".join(str(costs["by_stage"][stage]["unknown_budget_slots"]) for stage in ("generation", "evaluation")) + ".",
              "Per-run and per-task scores, differences, cost provenance, and missing statuses are retained in JSON.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output_dir.resolve()
    if output == campaign or campaign in output.parents:
        raise ValueError("Write the analysis outside the campaign directory")
    report = summarize(campaign, seed=args.seed, iterations=args.iterations)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "paper_summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote independent paper summary to {output}")


if __name__ == "__main__":
    main()
