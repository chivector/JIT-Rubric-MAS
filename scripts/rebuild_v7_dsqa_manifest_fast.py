"""Rebuild the v7 DSQA evidence manifest from immutable packs.

Preparation already validates each pack before writing it.  This helper only
reassembles the manifest from those packs and verifies the embedded task/pack
hashes and file digests, avoiding a second renderer replay for 100 tasks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import BUILDER_VERSION, PREVALIDATED_RENDERER, _public_task, evidence_pack_path
from jit_mas.schemas import digest
from scripts.prepare_public_bing_evidence import _validate_pack_quality


SPLIT = Path("paper/experiments/joint_task_splits_v7_ours_only.json")
DATA = Path("dataset/deepsearchqa/DSQA-pinned-v3.csv")
OUT = Path(".runtime/formal_v7_ours_assets_20261006/evidence_dsqa_v7")


def main() -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = (split["memberships"]["deepsearchqa"]["evolution"]
           + split["memberships"]["deepsearchqa"]["validation"]
           + split["memberships"]["deepsearchqa"]["test"])
    dataset = load_benchmark("deepsearchqa", str(DATA), available_tools=[])
    tasks = {task_id: dataset.tasks[task_id] for task_id in ids}
    entries = {}
    invalid = []
    for task_id, task in sorted(tasks.items()):
        path = evidence_pack_path(OUT, task_id)
        try:
            raw = path.read_bytes()
            pack = json.loads(raw.decode("utf-8"))
            expected_pack_hash = digest({k: v for k, v in pack.items() if k != "pack_sha256"})
            expected_task_hash = digest(_public_task(task))
            if (pack.get("task_id") != task_id or pack.get("status") != "complete"
                    or pack.get("task_sha256") != expected_task_hash
                    or pack.get("pack_sha256") != expected_pack_hash):
                raise ValueError("embedded identity/hash mismatch")
            _validate_pack_quality(pack)
            entries[task_id] = {
                "file": path.name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "task_sha256": pack["task_sha256"],
                "pack_sha256": pack["pack_sha256"],
            }
        except Exception as exc:  # report every invalid task, then fail atomically
            invalid.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    if invalid:
        print(json.dumps({"count": len(tasks), "valid": len(entries), "invalid": invalid}, indent=2))
        return 1
    manifest = {
        "version": BUILDER_VERSION,
        "count": len(entries),
        "renderer_validation": PREVALIDATED_RENDERER,
        "tasks": entries,
    }
    manifest["manifest_sha256"] = digest(manifest)
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(entries), "manifest_sha256": manifest["manifest_sha256"], "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
