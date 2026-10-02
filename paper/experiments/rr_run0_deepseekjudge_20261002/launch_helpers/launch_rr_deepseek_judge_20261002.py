"""Launch isolated run0 arms with stdin-only credentials and a frozen checkout."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime/rr_deepseek_judge_20261002_v27"
BASELINE = ROOT / "outputs/rr_single_agent_run0_deepseekjudge_20261002_v26"
METHOD = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v27"
PREFLIGHT = ROOT / "outputs/rr_closed_book_preflight_deepseekjudge_20261002_v27"
SOURCE = ROOT / "outputs/rr_single_agent_run0_gpt56sol_20261002_v15"
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


credentials = json.loads(sys.stdin.readline())
key = credentials["execution_api_key"]
env = os.environ.copy()
env.update(RR_EXEC_API_KEY=key, RR_JUDGE_API_KEY=key,
    JIT_MAS_MODEL_ATTEMPTS="2", MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES="5",
    JIT_MAS_DISABLE_KEEPALIVE="1", JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
    PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
RUNTIME.mkdir(parents=True, exist_ok=False)
for name in ["benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"]:
    shutil.copytree(ROOT / name, RUNTIME / name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "workspaces"))
freeze = {str(path.relative_to(RUNTIME)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in RUNTIME.rglob("*") if path.is_file()}
save(RUNTIME / "source_manifest.json", freeze)
python = str(ROOT / ".venv/Scripts/python.exe")
common = [python, "-m", "scripts.run_rr_two_arm_pilot",
    "--data", str(ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl"),
    "--joint-manifest", str(ROOT / "paper/experiments/joint_task_splits_v5.json"),
    "--closed-book", "--exec-endpoint", ENDPOINT, "--exec-model", MODEL,
    "--judge-endpoint", ENDPOINT, "--judge-model", MODEL,
    "--judge-max-tokens", "4096", "--judge-attempts", "2",
    "--judge-parallel", "8", "--max-inflight-requests", "8",
    "--structured-output", "json_schema"]
baseline_cmd = common + ["--arm", "baseline", "--reuse-baseline-from", str(SOURCE), "--output", str(BASELINE)]
method_cmd = common + ["--arm", "ours", "--output", str(METHOD)]

# A tiny availability probe uses the same service/model as the proposed judge.
# Its output contains identity/status only, never credentials or raw exceptions.
import requests
requests.packages.urllib3.disable_warnings()
try:
    response = requests.post(ENDPOINT + "/chat/completions", headers={"Authorization": "Bearer " + key},
        json={"model": MODEL, "messages": [{"role": "user", "content": "Return exactly {\"ok\":true} as JSON."}],
              "max_tokens": 64, "temperature": 0, "response_format": {"type": "json_object"}},
        timeout=40, verify=False)
    probe = {"http_status": response.status_code, "requested_model": MODEL, "formal_score": None}
    if response.ok:
        data = response.json()
        probe.update(response_model=data.get("model"), finish_reason=data.get("choices", [{}])[0].get("finish_reason"))
    save(ROOT / "outputs/rr_deepseek_judge_probe_20261002_v26.json", probe)
    if not response.ok:
        print(json.dumps({"status": "probe_failed", **probe}), flush=True)
        raise SystemExit(1)
except requests.RequestException as exc:
    print(json.dumps({"status": "probe_failed", "error_type": type(exc).__name__}), flush=True)
    raise SystemExit(1)

logdir = ROOT / "outputs/rr_deepseek_judge_launch_20261002_v27"
logdir.mkdir(exist_ok=False)
def launch(command, name, cwd):
    stdout = (logdir / (name + ".stdout.log")).open("w", encoding="utf-8")
    stderr = (logdir / (name + ".stderr.log")).open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW)
    stdout.close(); stderr.close()
    return process

baseline = None
existing = json.loads((ROOT / "outputs/rr_deepseek_judge_launch_20261002_v26/launch.json").read_text(encoding="utf-8"))
baseline_pid = existing["baseline_pid"]
save(logdir / "launch.json", {"protocol": "closed_book_exploratory", "judge_model": MODEL,
    "baseline_pid": baseline_pid, "baseline_output": str(BASELINE), "baseline_generation_reused": 33,
    "baseline_old_judgments_reused": 0, "status": "baseline_scoring_and_method_preflight",
    "runtime": str(RUNTIME), "method_output": str(METHOD)})
print(json.dumps({"status": "baseline_continues", "pid": baseline_pid, "output": str(BASELINE)}), flush=True)

preflight = launch([python, str(ROOT / "outputs/preflight_rr_closed_book_20261002.py"),
    str(RUNTIME), str(PREFLIGHT)], "preflight", ROOT)
preflight_code = preflight.wait()
if preflight_code != 0:
    state = json.loads((logdir / "launch.json").read_text(encoding="utf-8"))
    state.update(status="method_preflight_failed", preflight_exit=preflight_code)
    save(logdir / "launch.json", state)
    print(json.dumps({"status": "method_preflight_failed", "baseline_pid": baseline_pid,
        "output": str(PREFLIGHT)}), flush=True)
    raise SystemExit(1)
method = launch(method_cmd, "ours", RUNTIME)
state = json.loads((logdir / "launch.json").read_text(encoding="utf-8"))
state.update(status="parallel_running", method_pid=method.pid, preflight_exit=0)
save(logdir / "launch.json", state)
print(json.dumps({"status": "parallel_running", "baseline_pid": baseline_pid,
    "method_pid": method.pid, "baseline_output": str(BASELINE), "method_output": str(METHOD)}), flush=True)
