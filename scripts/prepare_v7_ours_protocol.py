"""Freeze the v7 Ours-only batched protocol from its manifest; no model calls."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jit_mas.independent_batch_protocol import build_protocol,validate_protocol

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,default=Path('paper/experiments/joint_task_splits_v7_ours_only.json')); p.add_argument('--output',type=Path,default=Path('paper/experiments/independent_protocol_v7_ours_only.json')); p.add_argument('--check',action='store_true'); a=p.parse_args(); m=json.loads(a.manifest.read_text(encoding='utf-8')); protocol=build_protocol(m); validate_protocol(protocol,m); content=json.dumps(protocol,indent=2,ensure_ascii=True,allow_nan=False)+'\n';
 if a.check:
  if not a.output.exists() or a.output.read_text(encoding='utf-8') != content: raise ValueError('Frozen v7 protocol differs from manifest')
 elif a.output.exists():
  if a.output.read_text(encoding='utf-8') != content: raise ValueError('Refusing to overwrite changed v7 protocol')
 else:
  a.output.write_text(content,encoding='utf-8',newline='\n')
 print(json.dumps({'valid':True,'protocol_sha256':protocol['protocol_sha256'],'manifest_sha256':protocol['parent_manifest_sha256'],'workload':protocol['workload'],'model_calls':0},ensure_ascii=True))
if __name__=='__main__': main()
