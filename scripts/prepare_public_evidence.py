"""Build immutable evidence packs with unauthenticated public retrieval only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.evidence import EvidenceConfig, EvidencePackBuilder, build_evidence_manifest, evidence_pack_path, save_evidence_pack
from jit_mas.public_retrieval import PublicRetriever, RetrievalConfig
from jit_mas.schemas import PublicTask


class LocalQueryPlanner:
    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        question = payload["public_task"]["question"].strip()
        return {"queries": [question]}


def read_tasks(path):
    content = Path(path).read_text(encoding="utf-8-sig")
    rows = json.loads(content) if content.lstrip().startswith("[") else [json.loads(line) for line in content.splitlines() if line.strip()]
    tasks = {}
    for row in rows:
        task = PublicTask.model_validate(row)
        if task.task_id in tasks:
            raise ValueError("Duplicate public task ID")
        tasks[task.task_id] = task
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-pages", type=int, default=8)
    args = parser.parse_args(argv)
    evidence_config = EvidenceConfig(max_pages=args.max_pages)
    tasks = read_tasks(args.tasks)
    if args.task_id:
        unknown = set(args.task_id) - set(tasks)
        if unknown:
            parser.error("Unknown public task ID")
        tasks = {task_id: tasks[task_id] for task_id in dict.fromkeys(args.task_id)}
    output = Path(args.output)
    retriever = PublicRetriever(RetrievalConfig(max_results=5))
    created = 0
    for task_id, task in tasks.items():
        destination = evidence_pack_path(output, task_id)
        if destination.exists():
            continue
        ledger = BudgetLedger(max_calls=1, max_tokens=10000, max_tool_calls=12 + len(task.attachments))
        planner = MeteredModel(LocalQueryPlanner(), ledger, "evidence_preparation", "local_query_planner", 4096)
        builder = EvidencePackBuilder(planner, retriever.search, retriever.fetch,
                                      config=evidence_config, ledger=ledger, identity={"retrieval": retriever.identity,
                                                               "network": "bing-rss-then-duckduckgo-html"})
        pack = builder.build(task)
        save_evidence_pack(pack, destination)
        created += 1
    manifest = build_evidence_manifest(tasks, output)
    print(json.dumps({"tasks": len(tasks), "created": created,
                      "manifest_sha256": manifest["manifest_sha256"],
                      "retriever": retriever.identity}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
