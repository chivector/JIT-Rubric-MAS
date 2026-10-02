"""Fresh DeepSeek-v4-flash serial launch helper; credentials read once from stdin JSON."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TAG = "v28"
RUNTIME = ROOT / f".runtime/rr_deepseek_judge_20261002_{TAG}"
BASELINE = ROOT / f"outputs/rr_single_agent_run0_deepseekjudge_20261002_{TAG}"
METHOD = ROOT / f"outputs/rr_jit_mas_run0_deepseekjudge_20261002_{TAG}"
PREFLIGHT = ROOT / f"outputs/rr_closed_book_preflight_deepseekjudge_20261002_{TAG}"
LOGDIR = ROOT / f"outputs/rr_deepseek_judge_launch_20261002_{TAG}"
SOURCE = ROOT / "outputs/rr_single_agent_run0_gpt56sol_20261002_v15"
ENDPOINT = "https://composure-presuming-comrade.ngrok-free.dev/v1"
MODEL = "deepseek-v4-flash-vision"
PYTHON = str(ROOT / ".venv/Scripts/python.exe")

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def launch(command, name, cwd):
    out = (LOGDIR / (name + ".stdout.log")).open("w", encoding="utf-8")
    err = (LOGDIR / (name + ".stderr.log")).open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    out.close(); err.close()
    return process

credentials = json.loads(sys.stdin.readline())
key = credentials["execution_api_key"]
env = os.environ.copy()
env.update(RR_EXEC_API_KEY=key, RR_JUDGE_API_KEY=key,
          JIT_MAS_MODEL_ATTEMPTS="2", MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES="5",
          JIT_MAS_DISABLE_KEEPALIVE="1", JIT_MAS_TLS_VERIFY="0", JIT_MAS_TLS_ENDPOINT=ENDPOINT,
          PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
for path in (RUNTIME, BASELINE, METHOD, PREFLIGHT, LOGDIR):
    if path.exists():
        raise SystemExit(f"refusing existing path: {path}")
RUNTIME.mkdir(parents=True)
for name in ["benchmark", "configs", "harness_factory", "jit", "jit_mas", "scripts"]:
    shutil.copytree(ROOT / name, RUNTIME / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "workspaces"))
freeze = {str(path.relative_to(RUNTIME)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in RUNTIME.rglob("*") if path.is_file()}
save(RUNTIME / "source_manifest.json", freeze)
COMMON = [PYTHON, "-m", "scripts.run_rr_two_arm_pilot",
    "--data", str(ROOT / "outputs/researchrubrics_live_20260930/processed_data.jsonl"),
    "--joint-manifest", str(ROOT / "paper/experiments/joint_task_splits_v5.json"),
    "--closed-book", "--exec-endpoint", ENDPOINT, "--exec-model", MODEL,
    "--judge-endpoint", ENDPOINT, "--judge-model", MODEL, "--judge-max-tokens", "4096",
    "--judge-attempts", "2", "--judge-parallel", "1", "--max-inflight-requests", "1",
    "--structured-output", "json_schema"]

# Probe service/model before spending requests; never persist credentials or body.
import requests
requests.packages.urllib3.disable_warnings()
try:
    response = requests.post(ENDPOINT + "/chat/completions", headers={"Authorization": "Bearer " + key},
        json={"model": MODEL, "messages": [{"role": "user", "content": "Return exactly {ok:true} as JSON."}],
              "max_tokens": 64, "temperature": 0, "response_format": {"type": "json_object"}},
        timeout=40, verify=False)
except requests.RequestException as exc:
    save(ROOT / f"outputs/rr_deepseek_judge_probe_20261002_{TAG}.json", {"http_status": None, "requested_model": MODEL, "error_type": type(exc).__name__})
    raise SystemExit(1)
probe = {"http_status": response.status_code, "requested_model": MODEL, "formal_score": None}
if response.ok:
    data = response.json(); probe.update(response_model=data.get("model"), finish_reason=data.get("choices", [{}])[0].get("finish_reason"))
save(ROOT / f"outputs/rr_deepseek_judge_probe_20261002_{TAG}.json", probe)
if not response.ok:
    print(json.dumps({"status": "probe_failed", **probe}), flush=True); raise SystemExit(1)
LOGDIR.mkdir(parents=True)
preflight_cmd = [PYTHON, str(ROOT / "outputs/preflight_rr_closed_book_deepseek_serial_20261002.py"), str(RUNTIME), str(PREFLIGHT)]
preflight = launch(preflight_cmd, "preflight", ROOT)
preflight_code = preflight.wait()
if preflight_code:
    save(LOGDIR / "launch.json", {"status": "method_preflight_failed", "preflight_exit": preflight_code,
                                   "runtime": str(RUNTIME), "preflight_output": str(PREFLIGHT)})
    print(json.dumps({"status": "method_preflight_failed", "preflight_exit": preflight_code}), flush=True)
    raise SystemExit(1)
# Baseline answer generation is reused from sealed v15; only DeepSeek scoring is performed.
baseline = launch(COMMON + ["--arm", "baseline", "--reuse-baseline-from", str(SOURCE), "--output", str(BASELINE)], "baseline", RUNTIME)
method = launch(COMMON + ["--arm", "ours", "--output", str(METHOD)], "ours", RUNTIME)
save(LOGDIR / "launch.json", {"status": "parallel_running", "baseline_pid": baseline.pid, "method_pid": method.pid,
                               "runtime": str(RUNTIME), "preflight_output": str(PREFLIGHT),
                               "baseline_output": str(BASELINE), "method_output": str(METHOD),
                               "judge_model": MODEL, "judge_parallel": 1, "max_inflight_requests": 1,
                               "baseline_generation_reused": 33, "baseline_old_judgments_reused": 0})
print(json.dumps({"status": "parallel_running", "baseline_pid": baseline.pid, "method_pid": method.pid,
                  "baseline_output": str(BASELINE), "method_output": str(METHOD)}), flush=True)

