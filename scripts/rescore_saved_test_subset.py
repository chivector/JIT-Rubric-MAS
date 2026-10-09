"""Score a fixed prefix of sealed independent TEST answers in a new directory."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import ALL_BENCHMARK_NAMES, load_benchmark
from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.config import MASConfig, NativeModels
from jit_mas.independent_batch_campaign import state_sources
from jit_mas.independent_campaign import TERMINAL
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from jit_mas.test_release import remaining_task_budget, score_with_pipeline
from scripts.eval.config import load_dotenv
from scripts.run_benchmark_experiment import evidence_identity, file_hash


class JudgeOnlyModels(NativeModels):
    def __init__(self, config):
        config.models["judge"].check("judge")
        self.config = config

    def create(self, role, agent_id, ledger, stage):
        if role != "judge" or stage != "evaluation":
            raise ValueError("Saved TEST rescoring permits only evaluator judge calls")
        return super().create(role, agent_id, ledger, stage)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def saved_submission_path(campaign, value):
    path = Path(value)
    if not path.is_absolute():
        path = campaign / path
    resolved = path.resolve()
    try:
        resolved.relative_to(campaign)
    except ValueError as error:
        raise CheckpointIntegrityError("Saved submission path escapes the original campaign") from error
    if path.is_symlink() or not resolved.is_file():
        raise CheckpointIntegrityError("Saved submission path is missing or symlinked")
    return resolved


def historical_budget(campaign, row, recover_prejudge_path_error):
    receipt_path = campaign / "test_evaluations" / f"{digest(row['slot_id'])}.json"
    started = campaign / "test_evaluation_started" / receipt_path.name
    if started.exists() and not receipt_path.exists():
        raise CheckpointIntegrityError("Interrupted original judgment cannot be silently resampled")
    budget = dict(row["result"]["budget"])
    receipt_hash, recovery = None, None
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != digest(body) or receipt.get("slot") != row
                or receipt.get("submission_sha256") != row["result"]["submission_hash"]):
            raise CheckpointIntegrityError("Original evaluation receipt identity changed")
        previous = receipt.get("evaluation_budget")
        if (recover_prejudge_path_error and receipt.get("status") == "evaluation_failed"
                and receipt.get("error_type") == "AttributeError"
                and isinstance(row["result"].get("submission_path"), str)
                and previous == {"usage_unknown": True}):
            recovery = {"diagnosis": "string submission_path passed to _read before judge construction",
                        "original_evaluation_budget": previous,
                        "judge_calls_inferred_from_diagnosed_path": 0,
                        "accounting_status": "diagnostic inference; original unknown accounting remains unchanged"}
            previous = {"model_calls": 0, "tokens": 0, "tool_calls": 0}
        if (not isinstance(previous, dict) or previous.get("usage_unknown")
                or previous.get("reserved_tokens", 0)):
            raise CheckpointIntegrityError("Original evaluation cost is unknown or unsettled")
        for name in ("model_calls", "tokens", "tool_calls"):
            value = previous.get(name)
            if type(value) is not int or value < 0:
                raise CheckpointIntegrityError("Original evaluation accounting is invalid")
            budget[name] = budget.get(name, 0) + value
        receipt_hash = file_hash(receipt_path)
    return budget, receipt_path, receipt_hash, recovery


def prepare_subset(campaign, target, count, *, recover_prejudge_path_error=False):
    campaign = Path(campaign).resolve()
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    with closing(sqlite3.connect((campaign / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        registration = json.loads(database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()[0])
        sealed = database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()
        if sealed is None:
            raise CheckpointIntegrityError("Original TEST inventory is not sealed")
        seal = json.loads(sealed[0])
        records = []
        for saved in database.execute("SELECT * FROM slots WHERE json_extract(body,'$.kind')='test' ORDER BY ordinal"):
            row = {**json.loads(saved["body"]), "status": saved["status"],
                   "result": json.loads(saved["result"]) if saved["result"] else None,
                   "result_hash": saved["result_hash"], "started_at": saved["started_at"],
                   "finished_at": saved["finished_at"]}
            if row["status"] not in TERMINAL or digest(row["result"]) != row["result_hash"]:
                raise CheckpointIntegrityError("Original TEST inventory is unfinished or changed")
            records.append(row)
        if (seal.get("registration_hash") != digest(registration)
                or seal.get("required_test_slots") != len(records)
                or seal.get("submission_hashes") != {row["slot_id"]: row["result_hash"] for row in records}):
            raise CheckpointIntegrityError("Original TEST seal differs from its registered inventory")
        candidates = [row for row in records if row["target"] == target and row["status"] == "submitted"]
        if len(candidates) < count:
            raise ValueError("count exceeds saved submitted answers for target")
        config = MASConfig.model_validate(registration["execution"]["configuration"])
        selected = []
        for row in candidates[:count]:
            source = row["source"]
            position, state_hash = None, digest(ExperienceSnapshot())
            if source != "static":
                selection = json.loads(database.execute("SELECT body FROM selections WHERE source=? AND run_id=?", (source, row["run_id"])).fetchone()[0])["selected"]
                if not selection:
                    raise CheckpointIntegrityError("Saved TEST source has no selected state")
                position, state_hash = selection["position"], selection["state_hash"]
                checkpoint = database.execute("SELECT * FROM checkpoints WHERE source=? AND run_id=? AND position=?", (source, row["run_id"], position)).fetchone()
                snapshot = ExperienceSnapshot.model_validate_json(checkpoint["snapshot"])
                trajectory = next(item for item in registration["protocol"]["trajectories"]
                                  if item["source"] == source and item["run_id"] == row["run_id"])
                if (checkpoint["state_hash"] != state_hash or digest(snapshot) != state_hash
                        or checkpoint["identity_hash"] != digest(registration)
                        or not state_sources(snapshot) <= set(trajectory["evolution_task_ids"])):
                    raise CheckpointIntegrityError("Saved TEST selected checkpoint identity changed")
            target_tasks = (trajectory["test_task_ids"] if source == target else
                            prepared_tasks(registration["protocol"], target, row["run_id"]))
            if row["task_id"] not in target_tasks:
                raise CheckpointIntegrityError("Saved answer is outside the target TEST partition")
            submission_path = saved_submission_path(campaign, row["result"]["submission_path"])
            submission = read_json(submission_path)
            if (row["result"].get("state_hash") != state_hash
                    or digest(submission) != row["result"]["submission_hash"]
                    or digest(submission["answer"]) != submission["answer_hash"]
                    or row["result"].get("answer_hash") != submission["answer_hash"]):
                raise CheckpointIntegrityError("Saved TEST answer or state identity changed")
            cumulative, receipt_path, receipt_hash, recovery = historical_budget(campaign, row, recover_prejudge_path_error)
            remaining_task_budget(config, {"budget": cumulative})
            selected.append({"row": row, "submission": submission, "submission_path": str(submission_path),
                             "submission_file_sha256": file_hash(submission_path),
                             "selected_position": position, "selected_state_hash": state_hash,
                             "cumulative_task_budget": cumulative, "original_receipt_path": str(receipt_path),
                             "original_receipt_sha256": receipt_hash, "prejudge_failure_recovery": recovery})
    provenance_path = campaign / "recovery_runtime_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else None
    if provenance:
        specification = config.models["judge"]
        specification.context_window = provenance["runtime_context_window"]
        specification.context_margin = provenance["runtime_context_margin"]
        specification.context_policy = provenance["runtime_context_policy"]
        config = MASConfig.model_validate(config.model_dump(mode="json"))
    return {"campaign": str(campaign), "registration": registration, "seal": seal,
            "config": config, "target": target, "selected": selected,
            "runtime_provenance": provenance, "candidate_count": len(candidates)}


def prepared_tasks(protocol, target, run_id):
    if target in protocol.get("sources", ()):
        return next(item["test_task_ids"] for item in protocol["trajectories"]
                    if item["source"] == target and item["run_id"] == run_id)
    return protocol["target_test_tasks"][target]


def make_scoring_pipeline(prepared):
    target, config = prepared["target"], prepared["config"]
    material = prepared["registration"]["execution"]["materials"][target]
    dataset = load_benchmark(target, material["data"])
    if dataset.dataset_sha256 != material["dataset_sha256"]:
        raise CheckpointIntegrityError("Pinned evaluator dataset changed")
    checker = None
    if target in {"ifeval", "ifbench"}:
        if evidence_identity(material["checker_source_root"]) != material["checker_source"]:
            raise CheckpointIntegrityError("Pinned author checker source changed")
        checker = PinnedInstructionChecker(material["checker_source_root"], target)
    provider = None if checker is not None else JudgeOnlyModels(config)
    specification = config.models["judge"]

    def evaluator_factory(judge):
        arguments = {"judge_id": specification.model, "judge_api_base": specification.endpoint,
                     "judge_max_tokens": specification.max_tokens, "judge_timeout": specification.timeout,
                     "checker": checker}
        if target == "researchrubrics":
            arguments["max_parallel_judgments"] = config.judge_parallel
            if config.judge_parallel > 1:
                arguments["judge_factory"] = (lambda: None) if judge is None else (
                    lambda: provider.create("judge", "saved-test-rubric", judge.ledger, "evaluation"))
        return dataset.evaluator(judge, **arguments)

    models = provider or SimpleNamespace(create=lambda *arguments: None)
    return SimpleNamespace(config=config, models=models, evaluator_factory=evaluator_factory,
                           private_records=dataset.private_records,
                           manifest=SimpleNamespace(test=[item["row"]["task_id"] for item in prepared["selected"]]))


def run_subset(campaign, output, target, count, *, recover_prejudge_path_error=False):
    output, campaign = Path(output).resolve(), Path(campaign).resolve()
    if output.exists() or output == campaign or campaign in output.parents:
        raise ValueError("Rescoring requires a new output directory outside the original campaign")
    prepared = prepare_subset(campaign, target, count, recover_prejudge_path_error=recover_prejudge_path_error)
    pipeline = make_scoring_pipeline(prepared)
    registration = {"schema": "saved-test-subset-rescore-v1", "created_at": utc_now(),
                    "original_campaign": str(campaign), "original_registration_hash": digest(prepared["registration"]),
                    "original_seal": prepared["seal"], "target": target, "count": count,
                    "selection_policy": "first count submitted target slots in immutable ordinal order; no score filtering",
                    "original_configuration": prepared["registration"]["execution"]["configuration"],
                    "runtime_configuration": prepared["config"].model_dump(mode="json"),
                    "runtime_provenance": prepared["runtime_provenance"],
                    "evaluator_version": pipeline.evaluator_factory(None).evaluator_version,
                    "transport_environment": {name: os.getenv(name) for name in (
                        "JIT_MAS_MODEL_ATTEMPTS", "JIT_MAS_DISABLE_KEEPALIVE", "JIT_MAS_RETRY_BASE_SECONDS",
                        "JIT_MAS_RETRY_MAX_SECONDS", "JIT_MAS_RETRY_JITTER_SECONDS")},
                    "code_fingerprint": code_fingerprint(), "runner_sha256": file_hash(__file__),
                    "actor_calls": 0, "records": prepared["selected"]}
    registration["registration_hash"] = digest(registration)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", registration)
    results = []
    for item in prepared["selected"]:
        row, submission = item["row"], item["submission"]
        key = digest(row["slot_id"])
        write_json(output / "started" / f"{key}.json", {"slot_id": row["slot_id"], "started_at": utc_now(),
                                                        "registration_hash": registration["registration_hash"]})
        try:
            if (file_hash(item["submission_path"]) != item["submission_file_sha256"]
                    or (item["original_receipt_sha256"] is not None and
                        file_hash(item["original_receipt_path"]) != item["original_receipt_sha256"])):
                raise CheckpointIntegrityError("Original source files changed before rescoring")
            result = score_with_pipeline(pipeline, {"slot": row, "submission": submission,
                        "outcome": {"budget": row["result"]["budget"],
                                    "task_generation_budget": item["cumulative_task_budget"]}})
        except Exception as error:
            result = {"slot": row, "complete": False, "official_score": None,
                      "status": "evaluation_failed", "error_type": type(error).__name__,
                      "evaluation_budget": getattr(error, "evaluation_budget", {"usage_unknown": True})}
        if (file_hash(item["submission_path"]) != item["submission_file_sha256"]
                or (item["original_receipt_sha256"] is not None and
                    file_hash(item["original_receipt_path"]) != item["original_receipt_sha256"])):
            raise CheckpointIntegrityError("Original source files changed during rescoring")
        result.update(registration_hash=registration["registration_hash"], source_answer_hash=submission["answer_hash"],
                      original_receipt_sha256=item["original_receipt_sha256"], actor_calls=0)
        result["receipt_sha256"] = digest(result)
        write_json(output / "evaluations" / f"{key}.json", result)
        results.append(result)
    summary = {"schema": "saved-test-subset-rescore-summary-v1", "registration_hash": registration["registration_hash"],
               "target": target, "count": count, "complete": sum(bool(result["complete"]) for result in results),
               "actor_calls": 0, "finished_at": utc_now(), "original_files_unchanged": True,
               "results": [{"slot_id": result["slot"]["slot_id"], "task_id": result["slot"]["task_id"],
                            "complete": result["complete"], "score": result["official_score"],
                            "judge_calls": result["evaluation_budget"].get("model_calls"),
                            "judge_tokens": result["evaluation_budget"].get("tokens")} for result in results]}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=ALL_BENCHMARK_NAMES, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--recover-prejudge-path-error", action="store_true",
                        help="Explicitly attribute legacy AttributeError receipts to the diagnosed pre-judge string-path bug")
    arguments = parser.parse_args(argv)
    load_dotenv()
    try:
        summary = run_subset(arguments.campaign, arguments.output, arguments.target, arguments.count,
                             recover_prejudge_path_error=arguments.recover_prejudge_path_error)
    except Exception as error:
        parser.exit(2, f"Saved TEST rescoring failed: {type(error).__name__}; original artifacts are unchanged\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if summary["complete"] == summary["count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
