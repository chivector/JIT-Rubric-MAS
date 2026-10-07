"""Assemble the v7 evidence manifest from already validated pack JSON files."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from jit_mas.evidence import BUILDER_VERSION, PREVALIDATED_RENDERER, evidence_pack_path
from jit_mas.schemas import digest

root = Path(__file__).resolve().parents[1]
split = json.loads((root/'paper/experiments/joint_task_splits_v7_ours_only.json').read_text(encoding='utf-8'))
ids = []
for part in ('evolution','validation','test'):
    ids.extend(split['memberships']['deepresearch_bench_ii'][part])
ids = list(dict.fromkeys(ids))
out = root/'.runtime/formal_v7_ours_assets_20261006/evidence_drbii'
entries = {}
for tid in ids:
    path = evidence_pack_path(out, tid)
    raw = path.read_bytes()
    pack = json.loads(raw.decode('utf-8'))
    if pack.get('task_id') != tid or pack.get('status') != 'complete':
        raise RuntimeError(f'bad pack identity: {tid}')
    entries[tid] = {'file': path.name,
                    'file_sha256': hashlib.sha256(raw).hexdigest(),
                    'task_sha256': pack.get('task_sha256'),
                    'pack_sha256': pack.get('pack_sha256')}
manifest = {'version': BUILDER_VERSION, 'count': len(entries),
            'renderer_validation': PREVALIDATED_RENDERER, 'tasks': dict(sorted(entries.items()))}
manifest['manifest_sha256'] = digest(manifest)
(out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(',', ':'))+'\n', encoding='utf-8')
print(json.dumps({'tasks': len(entries), 'manifest_sha256': manifest['manifest_sha256']}, ensure_ascii=True))
