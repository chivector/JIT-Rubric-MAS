"""Credential-safe transport diagnostics; key read from stdin, no bodies logged."""
import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENDPOINT='https://composure-presuming-comrade.ngrok-free.dev/v1'
MODEL='deepseek-v4-flash-vision'
key=json.loads(sys.stdin.readline())['execution_api_key']
rows=[]
payload={'model':MODEL,'messages':[{'role':'user','content':'Reply with a JSON object containing ok:true.'}], 'max_tokens':64,'temperature':0,'response_format':{'type':'json_object'}}
def safe(value):
 text=str(value).replace(key,'[REDACTED]')
 import re
 return re.sub(r'Bearer\s+\S+','Bearer [REDACTED]',text)[:1000]
def failure(exc):
 causes=[]; current=exc
 for _ in range(5):
  if current is None:break
  causes.append({'type':type(current).__name__,'message':safe(current)})
  current=current.__cause__ or current.__context__
 return causes
import requests,httpx,openai
requests.packages.urllib3.disable_warnings()
for name in ('requests','httpx','openai_sdk'):
 for index in range(3):
  started=time.monotonic(); row={'transport':name,'index':index}
  try:
   if name=='requests':
    r=requests.post(ENDPOINT+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,verify=False,timeout=40)
    row['http_status']=r.status_code
    if r.ok:row['response_model']=r.json().get('model')
   elif name=='httpx':
    with httpx.Client(verify=False,limits=httpx.Limits(max_keepalive_connections=0,max_connections=100),timeout=40) as client:
     r=client.post(ENDPOINT+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload)
     row['http_status']=r.status_code
     if r.is_success:row['response_model']=r.json().get('model')
   else:
    with openai.OpenAI(base_url=ENDPOINT,api_key=key,max_retries=0,timeout=40,http_client=httpx.Client(verify=False,limits=httpx.Limits(max_keepalive_connections=0,max_connections=100))) as client:
     response=client.chat.completions.create(**payload)
     row.update(http_status=200,response_model=response.model)
  except Exception as exc:row['error_chain']=failure(exc)
  row['elapsed_seconds']=round(time.monotonic()-started,3);rows.append(row)
  print(json.dumps(row),flush=True)
path=ROOT/'outputs/rr_deepseek_transport_diagnostics_20261002_v29.json'
path.write_text(json.dumps({'requested_model':MODEL,'endpoint':ENDPOINT,'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
