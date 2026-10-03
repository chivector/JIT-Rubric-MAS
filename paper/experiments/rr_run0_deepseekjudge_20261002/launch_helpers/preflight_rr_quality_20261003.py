"""Unscored public VAL probes for isolated quality and protocol repairs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="20261003")
parser.add_argument("--val-indices", type=int, nargs="+", default=[10, 3])
options = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]+", options.tag):
    raise SystemExit("Tag must contain only letters, numbers, underscores and hyphens")
RUNTIME = ROOT / (".runtime/rr_quality_preflight_" + options.tag)
OUTPUT = ROOT / ("outputs/rr_quality_preflight_" + options.tag)
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"

if not os.environ.get("RR_EXEC_API_KEY"):
    raise SystemExit("RR_EXEC_API_KEY must be injected through the process environment")
for destination in (RUNTIME, OUTPUT):
    if destination.exists():
        raise SystemExit("Refusing to overwrite preflight artifacts: " + str(destination))
os.environ.update(RR_JUDGE_API_KEY=os.environ["RR_EXEC_API_KEY"],
                  JIT_MAS_MODEL_ATTEMPTS="5", MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES="5",
                  MODULAR_AGENT_API_FAILURE_ACTION="raise", JIT_MAS_DISABLE_KEEPALIVE="1",
                  JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
                  PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
RUNTIME.mkdir(parents=True)
OUTPUT.mkdir(parents=True)
for name in ("benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"):
    shutil.copytree(ROOT / name, RUNTIME / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "workspaces"))
source_hashes = {path.relative_to(RUNTIME).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in RUNTIME.rglob("*") if path.is_file()}
(RUNTIME / "source_manifest.json").write_text(json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8")
sys.path.insert(0, str(RUNTIME))
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import SplitManifest, digest
from scripts.run_rr_two_arm_pilot import _config, _manifest, _pipeline, _safe_error, write_json

original = _manifest(ROOT / "paper/experiments/joint_task_splits_v5.json")
if any(index < 1 or index > len(original.validation) for index in options.val_indices):
    raise SystemExit("VAL index must identify a public validation input")
task_ids = [original.validation[index - 1] for index in options.val_indices]
manifest = SplitManifest(test=task_ids)
split_path = OUTPUT / "preflight_split.json"
write_json(split_path, manifest)
args = SimpleNamespace(data=ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl",
    split_path=split_path, exec_model=MODEL, exec_max_tokens=12288, exec_endpoint=ENDPOINT,
    judge_model=MODEL, judge_endpoint=ENDPOINT, timeout=180, judge_max_tokens=4096,
    judge_attempts=2, judge_parallel=1, max_inflight_requests=1, closed_book=True,
    structured_output="json_schema_planning", planning_string_max_length=2048,
    planning_communication_max_length=1024, planning_array_max_items=64)
config = _config(args)
store = ExperienceStore(OUTPUT / "state.sqlite")
initial = digest(store.snapshot())
metadata = {"purpose": "Unscored engineering probes on fixed public VAL inputs; not formal scores.",
    "original_split": "validation", "task_ids": task_ids, "formal_score": None,
    "state_updates_allowed": False, "judge_requests": 0, "status": "running",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "execution_model": MODEL, "output_cap": 12288,
    "started_at": datetime.now(timezone.utc).isoformat()}
write_json(OUTPUT / "metadata.json", metadata)
outcomes = []
try:
    pipeline = _pipeline(config, store, OUTPUT / "runs", args, manifest, None)
    for task_id in task_ids:
        try:
            result = pipeline.run_task(task_id, store.snapshot(), mode="evaluate", resume=False,
                                       defer_evaluation=True)
            assert result["evaluation"] is None
            assert digest(store.snapshot()) == initial
            records = result["budget"]["records"]
            assert not any(row.get("stage") == "evaluation" for row in records)
            outcomes.append({"task_id": task_id, "status": "passed", "formal_score": None,
                "run_dir": result["run_dir"], "answer_hash": result["answer_hash"],
                "budget": {key: result["budget"][key] for key in ("model_calls", "tokens", "wall_seconds")},
                "finish_reasons": [row["request"].get("finish_reason") for row in records if row.get("request")]})
        except Exception as error:
            outcomes.append({"task_id": task_id, "status": "failed", "formal_score": None,
                "error_type": type(error).__name__, "error": _safe_error(error)})
        write_json(OUTPUT / "summary.json", {"status": "running", "judge_requests": 0,
            "formal_score": None, "experience_unchanged": digest(store.snapshot()) == initial,
            "outcomes": outcomes})
        print(json.dumps(outcomes[-1], ensure_ascii=False), flush=True)
    status = "passed" if all(row["status"] == "passed" for row in outcomes) else "failed"
    write_json(OUTPUT / "summary.json", {"status": status, "judge_requests": 0,
        "formal_score": None, "experience_unchanged": digest(store.snapshot()) == initial,
        "outcomes": outcomes, "ended_at": datetime.now(timezone.utc).isoformat()})
    metadata.update(status=status, ended_at=datetime.now(timezone.utc).isoformat())
    write_json(OUTPUT / "metadata.json", metadata)
finally:
    store.close()
