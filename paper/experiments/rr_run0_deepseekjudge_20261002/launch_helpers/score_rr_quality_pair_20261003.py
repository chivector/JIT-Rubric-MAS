"""Judge two previously sealed answers on one public VAL engineering input."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime/rr_quality_preflight_20261003_v4"
OUTPUT = ROOT / "outputs/rr_quality_pair_20261003"
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"

if not os.environ.get("RR_EXEC_API_KEY"):
    raise SystemExit("RR_EXEC_API_KEY must be injected through the process environment")
if OUTPUT.exists():
    raise SystemExit("Refusing to overwrite the engineering score pair")
os.environ.update(RR_JUDGE_API_KEY=os.environ["RR_EXEC_API_KEY"], JIT_MAS_MODEL_ATTEMPTS="5",
                  MODULAR_AGENT_API_FAILURE_ACTION="raise", JIT_MAS_DISABLE_KEEPALIVE="1",
                  JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
                  PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
sys.path.insert(0, str(RUNTIME))
from jit_mas.budget import BudgetLedger
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import digest
from scripts.run_rr_two_arm_pilot import _config, _configure_logging, _manifest, _pipeline, _safe_error, write_json

_configure_logging()
manifest = _manifest(ROOT / "paper/experiments/joint_task_splits_v5.json")
task_id = manifest.validation[9]
source_rows = []
for tag in ("20261003_v2", "20261003_v4"):
    source = ROOT / ("outputs/rr_quality_preflight_" + tag)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed" and summary["judge_requests"] == 0
    assert summary["experience_unchanged"] and metadata["task_ids"] == [task_id]
    assert len(summary["outcomes"]) == 1 and summary["outcomes"][0]["task_id"] == task_id
    submission_path = Path(summary["outcomes"][0]["run_dir"]) / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    run_manifest = json.loads((submission_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["comparison"]["task"]["task_id"] == task_id
    answer_hash = digest(submission["answer"])
    assert answer_hash == submission["answer_hash"] == summary["outcomes"][0]["answer_hash"]
    source_rows.append({"tag": tag, "source_commit": metadata["source_commit"],
                        "submission_path": str(submission_path), "answer_hash": answer_hash,
                        "submission_file_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
                        "answer_text_sha256": hashlib.sha256(submission["answer"].encode("utf-8")).hexdigest(),
                        "answer": submission["answer"]})

OUTPUT.mkdir(parents=True)
write_json(OUTPUT / "registered_pair.json", {
    "purpose": "Single public VAL engineering diagnostic; not a full VAL or TEST comparison.",
    "task_id": task_id, "original_split": "validation", "state_updates_allowed": False,
    "judge_model": MODEL, "judge_endpoint": ENDPOINT, "generation_calls": 0,
    "judge_attempts": 2, "score_repeats": 1,
    "answers": [{name: value for name, value in row.items() if name != "answer"} for row in source_rows],
    "registered_at": datetime.now(timezone.utc).isoformat()})
args = SimpleNamespace(data=ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl",
    split_path=ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v35/rr_split.json",
    exec_model=MODEL, exec_max_tokens=12288, exec_endpoint=ENDPOINT,
    judge_model=MODEL, judge_endpoint=ENDPOINT, timeout=180, judge_max_tokens=4096,
    judge_attempts=2, judge_parallel=1, max_inflight_requests=1, closed_book=True,
    structured_output="json_schema_planning", planning_string_max_length=2048,
    planning_communication_max_length=1024, planning_array_max_items=64)
config = _config(args)
store = ExperienceStore(OUTPUT / "state.sqlite")
initial = digest(store.snapshot())
outcomes = []
try:
    pipeline = _pipeline(config, store, OUTPUT / "unused_generation", args, manifest, None)
    evaluator_versions = set()
    for row in source_rows:
        ledger = BudgetLedger(config.max_model_calls, config.max_total_tokens,
                              config.max_tool_calls, timeout_seconds=config.task_timeout)
        try:
            judge = pipeline.models.create("judge", "judge", ledger, "evaluation")
            evaluator = pipeline.evaluator_factory(judge)
            evaluation = evaluator.evaluate(row["answer"], ground_truth=task_id,
                                              private_record=pipeline.private_records[task_id])
            evaluator_versions.add(evaluation["evaluator_version"])
            write_json(OUTPUT / (row["tag"] + ".evaluation.json"), evaluation)
            outcomes.append({"tag": row["tag"], "answer_hash": row["answer_hash"],
                "status": evaluation["status"], "complete": evaluation["complete"],
                "score": evaluation["score"] if evaluation["complete"] else None,
                "failed_count": evaluation["failed_count"],
                "evaluator_version": evaluation["evaluator_version"]})
        except Exception as error:
            outcomes.append({"tag": row["tag"], "answer_hash": row["answer_hash"],
                             "status": "failed", "complete": False, "score": None,
                             "error": _safe_error(error)})
        finally:
            write_json(OUTPUT / (row["tag"] + ".budget.json"), ledger.snapshot())
        assert digest(store.snapshot()) == initial
        write_json(OUTPUT / "summary.json", {"status": "running", "outcomes": outcomes})
        print(json.dumps(outcomes[-1], ensure_ascii=False), flush=True)
    complete = all(row["complete"] for row in outcomes) and len(evaluator_versions) == 1
    write_json(OUTPUT / "summary.json", {
        "status": "completed" if complete else "incomplete", "task_id": task_id,
        "scope": "single_public_VAL_engineering_pair", "outcomes": outcomes,
        "paired_score_difference": outcomes[1]["score"] - outcomes[0]["score"] if complete else None,
        "experience_unchanged": digest(store.snapshot()) == initial,
        "generation_calls": 0, "ended_at": datetime.now(timezone.utc).isoformat()})
finally:
    store.close()
