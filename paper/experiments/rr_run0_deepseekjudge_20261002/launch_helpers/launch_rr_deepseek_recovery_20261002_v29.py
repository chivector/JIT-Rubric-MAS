"""v29 recovery modes: baseline/method/diagnostic; credentials read from stdin JSON."""
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODE=sys.argv[1]
if MODE not in {'baseline','method','diagnostic'}:raise SystemExit('mode must be baseline, method, or diagnostic')
credentials=json.loads(sys.stdin.readline());key=credentials['execution_api_key']
if MODE in {'baseline','diagnostic'}:
 script=ROOT/'outputs'/('launch_rr_deepseek_baseline_20261002_v29.py' if MODE=='baseline' else 'diagnose_rr_deepseek_transport_20261002_v29.py')
 completed=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(script)],input=json.dumps(credentials)+'\n',text=True,cwd=ROOT)
 raise SystemExit(completed.returncode)
RUNTIME=ROOT/'.runtime/rr_deepseek_judge_20261002_v28'
OUTPUT=ROOT/'outputs/rr_jit_mas_run0_deepseekjudge_20261002_v29'
PREFLIGHT=ROOT/'outputs/rr_closed_book_preflight_deepseekjudge_20261002_v29'
LOGDIR=ROOT/'outputs/rr_deepseek_method_launch_20261002_v29'
ENDPOINT='https://composure-presuming-comrade.ngrok-free.dev/v1';MODEL='deepseek-v4-flash-vision'
env=os.environ.copy();env.update(RR_EXEC_API_KEY=key,RR_JUDGE_API_KEY=key,JIT_MAS_MODEL_ATTEMPTS='5',MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES='5',
 JIT_MAS_DISABLE_KEEPALIVE='1',JIT_MAS_TLS_VERIFY='0',JIT_MAS_TLS_ENDPOINT=ENDPOINT,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
for path in [OUTPUT,PREFLIGHT,LOGDIR]:
 if path.exists():raise SystemExit('refusing existing v29 method path: '+str(path))
if not (RUNTIME/'source_manifest.json').exists():raise SystemExit('missing frozen v28 runtime')
LOGDIR.mkdir()
def save(data): (LOGDIR/'launch.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def launch(command,name,cwd):
 out=(LOGDIR/(name+'.stdout.log')).open('w',encoding='utf-8');err=(LOGDIR/(name+'.stderr.log')).open('w',encoding='utf-8')
 p=subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);out.close();err.close();return p
python=str(ROOT/'.venv/Scripts/python.exe')
p=launch([python,str(ROOT/'outputs/preflight_rr_closed_book_deepseek_serial_20261002.py'),str(RUNTIME),str(PREFLIGHT)],'preflight',ROOT)
state={'status':'method_preflight_running','preflight_pid':p.pid,'runtime':str(RUNTIME),'preflight_output':str(PREFLIGHT),'method_output':str(OUTPUT),'model_attempts':5,'judge_model':MODEL,'judge_parallel':1,'max_inflight_requests':1};save(state)
code=p.wait()
if code:
 state.update(status='method_preflight_failed',preflight_exit=code);save(state);print(json.dumps(state),flush=True);raise SystemExit(1)
command=[python,'-m','scripts.run_rr_two_arm_pilot','--data',str(ROOT/'outputs/researchrubrics_live_20260930/processed_data.jsonl'),
 '--joint-manifest',str(ROOT/'paper/experiments/joint_task_splits_v5.json'),'--closed-book','--exec-endpoint',ENDPOINT,'--exec-model',MODEL,
 '--judge-endpoint',ENDPOINT,'--judge-model',MODEL,'--judge-max-tokens','4096','--judge-attempts','2','--judge-parallel','1','--max-inflight-requests','1',
 '--structured-output','json_schema','--arm','ours','--output',str(OUTPUT)]
method=launch(command,'ours',RUNTIME);state.update(status='method_running',method_pid=method.pid,preflight_exit=0);save(state);print(json.dumps(state),flush=True)
