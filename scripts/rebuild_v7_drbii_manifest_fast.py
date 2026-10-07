"""Rebuild the v7 DRB-II evidence manifest after packs were validated.

This checks each immutable pack's task identity and pack hash, then writes the
manifest without re-tokenizing the 90 rendered packs a second time.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import BUILDER_VERSION, PREVALIDATED_RENDERER, evidence_pack_path
from jit_mas.schemas import digest

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "paper/experiments/joint_task_splits_v7_ours_only.json"
DATA = ROOT / "dataset/deepresearch_bench_ii/tasks_and_rubrics.jsonl"
OUT = ROOT / ".runtime/formal_v7_ours_assets_20261006/evidence_drbii"

def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = []
    for part in ("evolution", "validation", "test"):
        ids.extend(split["memberships"]["deepresearch_bench_ii"][part])
    ids = list(dict.fromkeys(ids))
    dataset = load_benchmark("deepresearch_bench_ii", str(DATA), available_tools=[])
    entries, invalid = {}, []
    for task_id in ids:
        path = evidence_pack_path(OUT, task_id)
        try:
            raw = path.read_bytes()
            pack = json.loads(raw.decode("utf-8"))
            if pack.get("task_id") != task_id:
                raise ValueError("task_id mismatch")
            if pack.get("status") != "complete":
                raise ValueError("pack is not complete")
            if pack.get("task_sha256") != digest(dataset.tasks[task_id].model_dump(mode="json")):
                raise ValueError("task hash mismatch")
            if pack.get("pack_sha256") != digest({k: v for k, v in pack.items() if k != "pack_sha256"}):
                raise ValueError("pack hash mismatch")
            entries[task_id] = {"file": path.name,
                                "file_sha256": hashlib.sha256(raw).hexdigest(),
                                "task_sha256": pack["task_sha256"],
                                "pack_sha256": pack["pack_sha256"]}
        except Exception as exc:
            invalid.append({"task_id": task_id, "error": type(exc).__name__, "detail": str(exc)})
    if invalid:
        raise RuntimeError(json.dumps({"invalid": invalid}, ensure_ascii=True))
    manifest = {"version": BUILDER_VERSION, "count": len(entries),
                "renderer_validation": PREVALIDATED_RENDERER, "tasks": dict(sorted(entries.items()))}
    manifest["manifest_sha256"] = digest(manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"tasks": len(entries), "manifest_sha256": manifest["manifest_sha256"], "invalid": 0}, ensure_ascii=True))

if __name__ == "__main__":
    main()
