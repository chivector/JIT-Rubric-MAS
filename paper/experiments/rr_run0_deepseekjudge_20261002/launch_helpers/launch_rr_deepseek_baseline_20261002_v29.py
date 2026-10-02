"""Fresh baseline-only scoring launch; credentials from stdin, frozen v28 runtime."""
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.runtime/rr_deepseek_judge_20261002_v28'
OUTPUT=ROOT/'outputs/rr_single_agent_run0_deepseekjudge_20261002_v29'
LOGDIR=ROOT/'outputs/rr_deepseek_baseline_launch_20261002_v29'
SOURCE=ROOT/'outputs/rr_single_agent_run0_gpt56sol_20261002_v15'
ENDPOINT='https://composure-presuming-comrade.ngrok-free.dev/v1';MODEL='deepseek-v4-flash-vision'
key=json.loads(sys.stdin.readline())['execution_api_key'];env=os.environ.copy()
env.update(RR_EXEC_API_KEY=key,RR_JUDGE_API_KEY=key,JIT_MAS_MODEL_ATTEMPTS='2',MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES='5',
 JIT_MAS_DISABLE_KEEPALIVE='1',JIT_MAS_TLS_VERIFY='0',JIT_MAS_TLS_ENDPOINT=ENDPOINT,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
if OUTPUT.exists() or LOGDIR.exists():raise SystemExit('refusing existing v29 paths')
if not (RUNTIME/'source_manifest.json').exists():raise SystemExit('missing frozen v28 runtime')
LOGDIR.mkdir()
command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','scripts.run_rr_two_arm_pilot','--data',str(ROOT/'outputs/researchrubrics_live_20260930/processed_data.jsonl'),
 '--joint-manifest',str(ROOT/'paper/experiments/joint_task_splits_v5.json'),'--closed-book','--exec-endpoint',ENDPOINT,'--exec-model',MODEL,
 '--judge-endpoint',ENDPOINT,'--judge-model',MODEL,'--judge-max-tokens','4096','--judge-attempts','2','--judge-parallel','1','--max-inflight-requests','1',
 '--structured-output','json_schema','--arm','baseline','--reuse-baseline-from',str(SOURCE),'--output',str(OUTPUT)]
out=(LOGDIR/'baseline.stdout.log').open('w',encoding='utf-8');err=(LOGDIR/'baseline.stderr.log').open('w',encoding='utf-8')
p=subprocess.Popen(command,cwd=RUNTIME,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);out.close();err.close()
metadata={'status':'baseline_scoring_running','pid':p.pid,'output':str(OUTPUT),'runtime':str(RUNTIME),'judge_model':MODEL,'judge_parallel':1,'max_inflight_requests':1,'sealed_answers_reused':33,'old_evaluations_reused':0}
(LOGDIR/'launch.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(metadata),flush=True)
