"""Reuse exact DSQA v7 task evidence packs from prior manifests."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import evidence_pack_path
from jit_mas.schemas import digest
from scripts.prepare_public_bing_evidence import _validate_pack_quality

root = Path(__file__).resolve().parents[1]
split = json.loads((root/'paper/experiments/joint_task_splits_v7_ours_only.json').read_text(encoding='utf-8'))
ids = list(dict.fromkeys(sum([split['memberships']['deepsearchqa'][k] for k in ('evolution','validation','test')], [])))
data = load_benchmark('deepsearchqa', str(root/'dataset/deepsearchqa/DSQA-pinned-v3.csv'), available_tools=[])
out = root/'.runtime/formal_v7_ours_assets_20261006/evidence_dsqa_v7'
out.mkdir(parents=True, exist_ok=True)
found = {}
for mp in (root/'.runtime').rglob('manifest.json'):
    try:
        doc=json.loads(mp.read_text(encoding='utf-8'))
    except Exception:
        continue
    rows=doc.get('tasks')
    if not isinstance(rows,dict):
        continue
    for tid in ids:
        if tid in found or tid not in rows:
            continue
        rel=rows[tid].get('file')
        if not isinstance(rel,str):
            continue
        src=(mp.parent/rel).resolve()
        if not src.is_file():
            continue
        try:
            pack=json.loads(src.read_text(encoding='utf-8'))
        except Exception:
            continue
        if pack.get('task_id') != tid or pack.get('status') != 'complete':
            continue
        try:
            _validate_pack_quality(pack)
        except ValueError:
            # A structurally valid but empty/lexical-only pack is not safe to
            # reuse. The retry helper will retrieve a replacement.
            continue
        dst=evidence_pack_path(out,tid)
        shutil.copy2(src,dst)
        found[tid]=str(src)
entries={}
for tid in found:
    p=evidence_pack_path(out,tid)
    raw=p.read_bytes()
    pack=json.loads(raw.decode('utf-8'))
    entries[tid]={'file':p.name,'file_sha256':hashlib.sha256(raw).hexdigest(),
                  'task_sha256':pack.get('task_sha256'),'pack_sha256':pack.get('pack_sha256')}
from jit_mas.evidence import BUILDER_VERSION, PREVALIDATED_RENDERER
manifest={'version':BUILDER_VERSION,'count':len(entries),
          'renderer_validation':PREVALIDATED_RENDERER,'tasks':dict(sorted(entries.items()))}
manifest['manifest_sha256']=digest(manifest)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
print(json.dumps({'selected':len(ids),'copied':len(found),'missing':len(ids)-len(found)},ensure_ascii=True))
