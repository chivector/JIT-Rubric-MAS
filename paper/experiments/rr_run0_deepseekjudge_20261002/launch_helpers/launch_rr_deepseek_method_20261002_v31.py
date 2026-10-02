"""v31 planning-schema / JSON-object execution recovery; stdin credential JSON."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ENDPOINT='https://composure-presuming-comrade.ngrok-free.dev/v1';MODEL='deepseek-v4-flash-vision';TAG='v31'
RUNTIME=ROOT/f'.runtime/rr_deepseek_judge_20261002_{TAG}';OUTPUT=ROOT/f'outputs/rr_jit_mas_run0_deepseekjudge_20261002_{TAG}';PREFLIGHT=ROOT/f'outputs/rr_closed_book_preflight_deepseekjudge_20261002_{TAG}';LOGDIR=ROOT/f'outputs/rr_deepseek_method_launch_20261002_{TAG}'
key=json.loads(sys.stdin.readline())['execution_api_key'];env=os.environ.copy();env.update(RR_EXEC_API_KEY=key,RR_JUDGE_API_KEY=key,JIT_MAS_MODEL_ATTEMPTS='5',MODULAR_AGENT_MAX_CONSECUTIVE_API_FAILURES='5',JIT_MAS_DISABLE_KEEPALIVE='1',JIT_MAS_TLS_VERIFY='0',JIT_MAS_TLS_ENDPOINT=ENDPOINT,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
for p in [RUNTIME,OUTPUT,PREFLIGHT,LOGDIR]:
 if p.exists():raise SystemExit('refusing existing v31 path: '+str(p))
RUNTIME.mkdir(parents=True)
for name in ['benchmark','configs','harness_factory','jit','jit_mas','scripts']:
 shutil.copytree(ROOT/name,RUNTIME/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','workspaces'))
freeze={str(p.relative_to(RUNTIME)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUNTIME.rglob('*') if p.is_file()};(RUNTIME/'source_manifest.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
LOGDIR.mkdir();python=str(ROOT/'.venv/Scripts/python.exe')
def launch(cmd,name,cwd):
 out=(LOGDIR/(name+'.stdout.log')).open('w',encoding='utf-8');err=(LOGDIR/(name+'.stderr.log')).open('w',encoding='utf-8');p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);out.close();err.close();return p
def save(x):(LOGDIR/'launch.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=launch([python,str(ROOT/'outputs/preflight_rr_closed_book_deepseek_planning_20261002_v31.py'),str(RUNTIME),str(PREFLIGHT)],'preflight',ROOT); state={'status':'method_preflight_running','preflight_pid':p.pid,'runtime':str(RUNTIME),'preflight_output':str(PREFLIGHT),'method_output':str(OUTPUT),'judge_model':MODEL,'judge_parallel':1,'max_inflight_requests':1,'structured_output':'json_schema_planning','model_attempts':5};save(state);code=p.wait()
if code:state.update(status='method_preflight_failed',preflight_exit=code);save(state);print(json.dumps(state));raise SystemExit(1)
cmd=[python,'-m','scripts.run_rr_two_arm_pilot','--data',str(ROOT/'outputs/researchrubrics_live_20260930/processed_data.jsonl'),'--joint-manifest',str(ROOT/'paper/experiments/joint_task_splits_v5.json'),'--closed-book','--exec-endpoint',ENDPOINT,'--exec-model',MODEL,'--judge-endpoint',ENDPOINT,'--judge-model',MODEL,'--judge-max-tokens','4096','--judge-attempts','2','--judge-parallel','1','--max-inflight-requests','1','--structured-output','json_schema_planning','--arm','ours','--output',str(OUTPUT)]
m=launch(cmd,'ours',RUNTIME);state.update(status='method_running',method_pid=m.pid,preflight_exit=0);save(state);print(json.dumps(state))
