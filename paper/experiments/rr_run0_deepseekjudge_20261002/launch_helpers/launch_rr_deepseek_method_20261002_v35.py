"""Freeze and run v35 from empty experience; credentials enter only through stdin/env."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"
TAG = "v35"
RUNTIME = ROOT / f".runtime/rr_deepseek_judge_20261002_{TAG}"
OUTPUT = ROOT / f"outputs/rr_jit_mas_run0_deepseekjudge_20261002_{TAG}"
PREFLIGHT = ROOT / f"outputs/rr_closed_book_preflight_deepseekjudge_20261002_{TAG}"
LOGDIR = ROOT / f"outputs/rr_deepseek_method_launch_20261002_{TAG}"

key = os.environ.get("RR_EXEC_API_KEY")
if not key:
    key = json.loads(sys.stdin.readline())["execution_api_key"]
env = os.environ.copy()
env.update(RR_EXEC_API_KEY=key, RR_JUDGE_API_KEY=key,
    JIT_MAS_MODEL_ATTEMPTS="5", MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES="5",
    MODULAR_AGENT_API_FAILURE_ACTION="raise", JIT_MAS_DISABLE_KEEPALIVE="1",
    JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
    PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
for path in (RUNTIME, OUTPUT, PREFLIGHT, LOGDIR):
    if path.exists():
        raise SystemExit("Refusing to replace existing v35 artifact: " + str(path))
RUNTIME.mkdir(parents=True)
for name in ("benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"):
    shutil.copytree(ROOT / name, RUNTIME / name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "workspaces"))
freeze = {str(path.relative_to(RUNTIME)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in RUNTIME.rglob("*") if path.is_file()}
(RUNTIME / "source_manifest.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
LOGDIR.mkdir()
python = str(ROOT / ".venv/Scripts/python.exe")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def launch(command, name, cwd):
    with (LOGDIR / (name + ".stdout.log")).open("w", encoding="utf-8") as out, \
         (LOGDIR / (name + ".stderr.log")).open("w", encoding="utf-8") as err:
        return subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW)

def save(state):
    (LOGDIR / "launch.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

preflight = launch([python, str(ROOT / "outputs/preflight_rr_closed_book_deepseek_planning_20261002_v35.py"),
    str(RUNTIME), str(PREFLIGHT)], "preflight", ROOT)
state = {"status": "method_preflight_running", "started_at": datetime.now(timezone.utc).isoformat(),
    "preflight_pid": preflight.pid, "runtime": str(RUNTIME), "preflight_output": str(PREFLIGHT),
    "method_output": str(OUTPUT), "code_commit": commit, "judge_model": MODEL,
    "judge_parallel": 1, "max_inflight_requests": 1, "structured_output": "json_schema_planning",
    "exec_max_tokens": 12288, "model_attempts": 5, "api_failure_action": "raise",
    "formal_experience_initial_state": "empty", "protocol": "closed_book_exploratory_deepseek_self_judge",
    "comparison_baseline": str(ROOT / "outputs/rr_single_agent_run0_deepseekjudge_20261002_v29"),
    "baseline_exec_max_tokens": 8192, "baseline_rerun": False}
save(state)
print(json.dumps(state), flush=True)
code = preflight.wait()
state["preflight_exit"] = code
if code:
    state["status"] = "method_preflight_failed"
    save(state)
    print(json.dumps(state), flush=True)
    raise SystemExit(code)
command = [python, "-m", "scripts.run_rr_two_arm_pilot",
    "--data", str(ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl"),
    "--joint-manifest", str(ROOT / "paper/experiments/joint_task_splits_v5.json"), "--closed-book",
    "--exec-endpoint", ENDPOINT, "--exec-model", MODEL, "--exec-max-tokens", "12288",
    "--judge-endpoint", ENDPOINT, "--judge-model", MODEL, "--judge-max-tokens", "4096",
    "--judge-attempts", "2", "--judge-parallel", "1", "--max-inflight-requests", "1",
    "--structured-output", "json_schema_planning", "--arm", "ours", "--output", str(OUTPUT)]
method = launch(command, "ours", RUNTIME)
state.update(status="method_running", method_pid=method.pid)
save(state)
print(json.dumps(state), flush=True)
code = method.wait()
state.update(status="method_process_finished", method_exit=code,
    ended_at=datetime.now(timezone.utc).isoformat())
save(state)
print(json.dumps(state), flush=True)
raise SystemExit(code)

