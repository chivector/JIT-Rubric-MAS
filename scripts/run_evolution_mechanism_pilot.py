"""Replay validated batch reflections and compare frozen states on fixed tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import time

from jit_mas.agent_pool import effective_pool
from jit_mas.batch_evolution import _candidate
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_telemetry import EvolutionTelemetry
from jit_mas.pipeline import code_fingerprint, submitted_source_digest, write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from scripts.run_independent_experiment import validation_result
from scripts.run_jit_mas import make_pipeline
from scripts.run_targeted_batch_experiment import configure_targeted_pipeline


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def configure_supplemental_sources(pipeline, supplemental_path):
    if not supplemental_path:
        return pipeline
    from jit_mas.evidence import EVIDENCE_QUESTION_HEADER

    supplement = read_json(supplemental_path)
    tasks = dict(pipeline.tasks)
    for task_id, payload in supplement.get("tasks", {}).items():
        if task_id not in tasks:
            raise ValueError("Supplemental source task is outside the loaded inventory")
        sources = payload.get("sources", [])
        if not sources or any(source.get("status") != "ok" for source in sources):
            raise ValueError("Source supplement must contain successfully fetched sources")
        normalized_sources = []
        for source in sources:
            excerpts = []
            if isinstance(source.get("excerpt"), str):
                excerpts.append(source["excerpt"])
            for window in source.get("excerpt_windows", []):
                if isinstance(window, dict) and isinstance(window.get("text"), str):
                    excerpts.append(window["text"])
            excerpts = list(dict.fromkeys(excerpts))
            if not excerpts:
                raise ValueError("Source supplement source has no extracted excerpt")
            normalized_sources.append({
                "source_id": source.get("source_id") or source.get("id"),
                "title": source.get("title"),
                "date": source.get("date"),
                "locator": source.get("final_url") or source.get("url"),
                "status": "ok",
                "segments": [{"text": excerpt} for excerpt in excerpts],
            })
        original = tasks[task_id]
        question, _, _ = original.question.partition(EVIDENCE_QUESTION_HEADER)
        derived = payload.get("derived_calculations")
        if isinstance(derived, dict) and ("nba player" in original.question.lower()
                                         and "influential" in original.question.lower()):
            # The fixed NBA matrix is calculated from the verified fact receipt
            # on the host.  Preserve any policy text from the source supplement,
            # but never allow a model-authored total to become the canonical one.
            from jit_mas.targeted_nba import build_host_calculations
            derived = dict(derived)
            derived["host_calculations"] = build_host_calculations()
        if derived:
            question += "\n\n## Deterministic calculation support\n" + json.dumps(derived, ensure_ascii=False)
        # Match the canonical fixed-evidence pack exactly so public diagnostic
        # routing can remove source text from lexical hints while retaining the
        # immutable evidence in the actual task and audit.
        from jit_mas.evidence import EVIDENCE_SCOPE
        pack = {"task_id": task_id, "sources": normalized_sources, "scope": EVIDENCE_SCOPE}
        question += EVIDENCE_QUESTION_HEADER + json.dumps(pack, ensure_ascii=False,
                                                          sort_keys=True, separators=(",", ":"))
        tasks[task_id] = original.model_copy(update={"question": question})
        # Keep deterministic support on the host as well as in the public task.
        # The model can explain the matrix, while the audit receipt can check
        # whether its final answer retained the fixed rows and host winner.
        if derived and isinstance(derived, dict):
            host = derived.get("host_calculations")
            if isinstance(host, dict):
                if not hasattr(pipeline, "targeted_host_calculations"):
                    pipeline.targeted_host_calculations = {}
                pipeline.targeted_host_calculations[task_id] = host
    pipeline.tasks = tasks
    pipeline.source_supplement = {"path": str(supplemental_path), "hash": digest(supplement),
        "preparation": supplement.get("preparation"),
        "scope": "Fetched source and host-calculation augmentation; old evidence receipts preserved separately"}
    return pipeline


def replay_candidates(pipeline, campaign, source, output):
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        base_row = database.execute("SELECT snapshot FROM checkpoints WHERE source=? AND position=0", (source,)).fetchone()
        rows = database.execute("SELECT body,result FROM slots WHERE status='complete'").fetchall()
    base = ExperienceSnapshot.model_validate_json(base_row[0])
    slot = next(json.loads(body) for body, _ in rows if json.loads(body)["source"] == source
                and json.loads(body)["kind"] == "batch_evolution" and json.loads(body)["batch_index"] == 0)
    batch_directory = campaign / "slots" / digest(slot["slot_id"])
    records = [read_json(path) for path in sorted((batch_directory / "reflections").glob("*/reflection.json"))]
    if len(records) != 5 or not any(row["status"] == "complete" for row in records):
        raise ValueError("Pilot replay requires five recorded reflections and some complete attribution evidence")
    for record in records:
        if record.get("source_hash") and submitted_source_digest(record["run_dir"]) != record["source_hash"]:
            raise ValueError("Replay source receipt changed")
    task_ids = read_json(campaign / "trajectories" / source / "run0/runtime_split.json")["evolution"][:5]
    if set(task_ids) != {row["task_id"] for row in records}:
        raise ValueError("Replay task membership changed")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "replay_manifest.json", {"kind": "post-batch integration repair pilot",
        "source_campaign": str(campaign), "base_state_hash": digest(base), "source_task_ids": task_ids,
        "source_receipts": [{key: record.get(key) for key in ("task_id", "status", "run_dir", "source_hash", "error")} for record in records],
        "code_fingerprint": code_fingerprint(), "created_at": utc_now(),
        "original_reflection_budgets": [record["budget"] for record in records],
        "new_budget_scope": "New integration and fixed-task evaluation; source reflection budgets are recorded separately."})
    candidates = [_candidate(pipeline, base, task_ids, records, output, slot["task_id"] + "-repair", index)
                  for index in range(3)]
    write_json(output / "replay_results.json", {"candidates": candidates})
    return base, candidates


def run_task(config, material, split, source, task_id, label, snapshot, output, supplemental_path=None):
    start = time.monotonic()
    store = ExperienceStore(output / "state.sqlite")
    before = digest(snapshot)
    pipeline = make_pipeline(config, store, output / "runs", data=material["data"], splits=split,
        benchmark=source, evidence_dir=material.get("evidence_dir"))
    pipeline = configure_targeted_pipeline(configure_supplemental_sources(pipeline, supplemental_path), benchmark=source)
    result = {"benchmark": source, "task_id": task_id, "condition": label, "input_state_hash": before,
              "status": "failed", "complete": False, "started_at": utc_now()}
    try:
        mode = "validate" if task_id in pipeline.manifest.validation else "evaluate"
        outcome = pipeline.run_task(task_id, snapshot, mode=mode, attribution=False, repeat=0)
        result.update(validation_result(source, outcome))
        host_calculations = getattr(pipeline, "targeted_host_calculations", {}).get(task_id)
        if host_calculations:
            from jit_mas.targeted_nba import validate_host_calculations
            submission_path = Path(outcome["run_dir"]) / "submission.json"
            answer = read_json(submission_path).get("answer", "")
            result["host_calculation_audit"] = validate_host_calculations(answer, host_calculations)
            write_json(Path(outcome["run_dir"]) / "host_calculation_audit.json",
                       result["host_calculation_audit"])
        result["status"] = "complete" if result["complete"] else "incomplete"
        frozen = read_json(Path(outcome["run_dir"]) / "frozen_plan.json")
        result["team"] = frozen["TeamSpec"]
        result["pool_bindings"] = [{key: agent.get(key) for key in ("agent_id", "pool_agent_id", "pool_agent_version")}
                                   for agent in frozen["TeamSpec"]["agents"]]
    except Exception as error:
        failure = getattr(error, "jit_mas_run_failure", {})
        result.update(error_type=type(error).__name__, error=str(error), budget=failure.get("budget"), failure=failure)
    finally:
        result.update(latency_seconds=time.monotonic() - start, finished_at=utc_now(),
                      state_unchanged=digest(snapshot) == before)
        store.close()
        write_json(output / "result.json", result)
    return result


def run(args):
    root = Path(__file__).resolve().parents[1]
    launch = read_json(args.launch)
    config = MASConfig.model_validate(read_json(launch["configuration"]))
    config.check_native()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split = args.split.resolve() if args.split else campaign / "trajectories" / args.source / "run0/runtime_split.json"
    material = launch["benchmarks"][args.source]
    state_store = ExperienceStore(output / "integration_state.sqlite")
    pipeline = make_pipeline(config, state_store, output / "integration_pipeline", data=material["data"],
        splits=split, benchmark=args.source, evidence_dir=material.get("evidence_dir"))
    pipeline = configure_targeted_pipeline(configure_supplemental_sources(pipeline, args.supplemental_sources), benchmark=args.source)
    conditions = [("seed", ExperienceSnapshot())]
    if args.replay:
        base, candidates = replay_candidates(pipeline, campaign, args.source, output / "replay")
        conditions = [("seed", base)] + [(f"repair-c{row['candidate_index']}",
            ExperienceSnapshot.model_validate(row["snapshot"])) for row in candidates if row["status"] == "complete"]
    if args.snapshot:
        store = ExperienceStore(args.snapshot, read_only=True)
        try:
            conditions.append((args.candidate_label if args.snapshot_version is not None else "historical-evolved",
                               store.snapshot(args.snapshot_version)))
        finally:
            store.close()
    if args.existing_candidate:
        candidate = read_json(args.existing_candidate)
        conditions.append((args.candidate_label, ExperienceSnapshot.model_validate(candidate["snapshot"])))
    state_store.close()
    task_ids = args.task_id or read_json(split)["validation"][:2]
    if not set(task_ids) <= set(pipeline.manifest.validation + pipeline.manifest.test):
        raise ValueError("Pilot tasks must be in the registered validation/test inventory")
    for label, snapshot in conditions:
        write_json(output / f"{label}_snapshot.json", snapshot.model_dump(mode="json"))
    write_json(output / "manifest.json", {"source": args.source, "task_ids": task_ids,
        "conditions": [{"label": label, "state_hash": digest(snapshot)} for label, snapshot in conditions],
        "code_fingerprint": code_fingerprint(), "configuration": config.model_dump(mode="json"),
        "source_campaign": str(campaign), "guide_identity": pipeline.targeted_optimization,
        "created_at": utc_now(), "paired_fixed_tasks": True,
        "seed_executed": not args.skip_seed,
        "reference_scope": "seed state; baseline receipts are separate when seed execution is skipped"})
    if args.supplemental_sources:
        manifest = read_json(output / "manifest.json")
        manifest["source_supplement"] = pipeline.source_supplement
        write_json(output / "manifest.json", manifest)
    jobs = [(label, snapshot, task_id) for label, snapshot in conditions for task_id in task_ids
            if label != "seed" or not args.skip_seed]
    if not jobs:
        raise ValueError("Pilot requires at least one executed condition")
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        results = list(workers.map(lambda job: run_task(config, material, split, args.source, job[2],
            job[0], job[1], output / job[0] / digest(job[2])[:12], args.supplemental_sources), jobs))
    telemetry = EvolutionTelemetry()
    curves = []
    for index, (label, snapshot) in enumerate(conditions):
        selected = [result for result in results if result["condition"] == label]
        if not selected:
            continue
        scores = [result["score"] for result in selected if result["complete"]]
        complete = len(scores) == len(task_ids)
        budgets = [result.get("budget") for result in selected]
        known = [budget for budget in budgets if isinstance(budget, dict)]
        budget = {"tokens": sum(item["tokens"] for item in known) if len(known) == len(budgets) else None,
            "model_calls": sum(item["model_calls"] for item in known) if len(known) == len(budgets) else None,
            "records": [record for item in known for record in item.get("records", [])]}
        raw_mean = sum(scores) / len(scores) if complete else None
        score = (raw_mean - 1) / 9 if args.source == "writingbench" and raw_mean is not None else raw_mean
        baseline = effective_pool(conditions[0][1].agent_pool)
        evolved = effective_pool(snapshot.agent_pool)
        operations = snapshot.agent_pool.structural_history[len(baseline.structural_history):]
        curve = telemetry.record_round(index, performance={"fixed_task_mean": score, "native_mean": raw_mean},
            budget=budget, latency_seconds=sum(item["latency_seconds"] for item in selected),
            pool_before=baseline, pool_after=evolved, operations_requested=operations,
            operations_applied=operations, outcome={"complete": complete})
        curve.update(condition=label, compared_to="seed", score_per_round=score,
                     semantics="Parallel candidate conditions; not a chronological evolution trajectory.")
        curves.append(curve)
    write_json(output / "report.json", {"records": results, "state_comparisons": curves,
        "complete": all(result["complete"] for result in results), "finished_at": utc_now()})
    print(json.dumps([{key: row.get(key) for key in ("condition", "task_id", "score", "complete", "latency_seconds", "error")}
                      for row in results], ensure_ascii=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("researchrubrics", "deepsearchqa", "writingbench"), required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-version", type=int)
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--supplemental-sources", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--existing-candidate", type=Path)
    parser.add_argument("--candidate-label", default="prior-candidate")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=2)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
