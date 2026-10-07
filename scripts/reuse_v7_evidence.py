"""Find and copy byte-identical validated public evidence packs from old runs."""
from __future__ import annotations
import json, shutil
from pathlib import Path
from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import evidence_pack_path, load_evidence_pack

root = Path(__file__).resolve().parents[1]
split = json.loads((root/'paper/experiments/joint_task_splits_v7_ours_only.json').read_text())
ids = []
for part in ('evolution','validation','test'):
    ids.extend(split['memberships']['deepresearch_bench_ii'][part])
data = load_benchmark('deepresearch_bench_ii', str(root/'dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl'), available_tools=[])
out = root/'.runtime/formal_v7_ours_assets_20261006/evidence_drbii'; out.mkdir(parents=True, exist_ok=True)
missing = []
for tid in dict.fromkeys(ids):
    try: load_evidence_pack(evidence_pack_path(out, tid), data.tasks[tid]); continue
    except Exception: pass
    missing.append(tid)
found = {}
for manifest_path in root/'.runtime'.rglob('manifest.json'):
    try: doc=json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception: continue
    rows=doc.get('tasks')
    if not isinstance(rows,dict): continue
    for tid in list(missing):
        row=rows.get(tid)
        if not isinstance(row,dict) or tid in found: continue
        rel=row.get('file')
        if not isinstance(rel,str): continue
        src=(manifest_path.parent/rel).resolve()
        if not src.is_file(): continue
        try: load_evidence_pack(src, data.tasks[tid])
        except Exception: continue
        dst=evidence_pack_path(out, tid)
        shutil.copy2(src,dst)
        found[tid]=str(src)
print(json.dumps({'missing_before':len(missing),'copied':len(found),'ids':sorted(found)}, ensure_ascii=True, indent=2))
