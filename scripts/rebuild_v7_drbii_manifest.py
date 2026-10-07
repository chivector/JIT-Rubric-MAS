"""Validate and rebuild the complete v7 DRB-II evidence manifest."""
from __future__ import annotations
import json
from pathlib import Path
from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import build_evidence_manifest, evidence_pack_path, load_evidence_pack

root = Path(__file__).resolve().parents[1]
split = json.loads((root/'paper/experiments/joint_task_splits_v7_ours_only.json').read_text(encoding='utf-8'))
ids = []
for part in ('evolution','validation','test'):
    ids.extend(split['memberships']['deepresearch_bench_ii'][part])
ids = list(dict.fromkeys(ids))
data = load_benchmark('deepresearch_bench_ii', str(root/'dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl'), available_tools=[])
out = root/'.runtime/formal_v7_ours_assets_20261006/evidence_drbii'
selected = {tid: data.tasks[tid] for tid in ids}
bad = []
for tid, task in selected.items():
    try:
        load_evidence_pack(evidence_pack_path(out, tid), task)
    except Exception as exc:
        bad.append({'task_id': tid, 'error': type(exc).__name__})
if bad:
    raise RuntimeError(json.dumps({'invalid': bad}, ensure_ascii=True))
manifest = build_evidence_manifest(selected, out)
if len(manifest.get('tasks', {})) != len(selected):
    raise RuntimeError('manifest task count mismatch after rebuild')
print(json.dumps({'tasks': len(selected), 'manifest_sha256': manifest['manifest_sha256']}, ensure_ascii=True))
