"""Prepare immutable shared evidence from public tasks, only on explicit request.

Use ``python -m scripts.prepare_benchmark_evidence --help``. No benchmark answer
or evaluator record is accepted. This command does not run task executors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import ModelConfig
from jit_mas.evidence import (EvidencePackBuilder, build_evidence_manifest,
                              evidence_pack_path, load_evidence_pack, save_evidence_pack)
from jit_mas.schemas import PublicTask


def _read_tasks(path):
    content = Path(path).read_text(encoding="utf-8-sig")
    if content.lstrip().startswith("["):
        rows = json.loads(content)
    else:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide a nonempty PublicTask JSON array or JSONL file")
    tasks = {}
    for row in rows:
        task = PublicTask.model_validate(row)
        if task.task_id in tasks:
            raise ValueError("Duplicate public task ID")
        tasks[task.task_id] = task
    return tasks


def _public_attachment_loader(root, reader):
    allowed_root = Path(root).resolve() if root else None

    def load(locator):
        if locator.startswith(("http://", "https://")):
            return reader(locator)
        if allowed_root is None:
            raise ValueError("Local public attachments require --attachment-root")
        path = Path(locator)
        path = path.resolve() if path.is_absolute() else (allowed_root / path).resolve()
        if not path.is_relative_to(allowed_root):
            raise ValueError("Public attachment is outside the explicitly configured root")
        if path.suffix.lower() not in {".txt", ".md", ".csv", ".json", ".jsonl", ".html", ".htm"}:
            raise ValueError("Provide a publicly extracted UTF-8 text attachment")
        return path.read_text(encoding="utf-8-sig")

    return load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="PublicTask-only JSON array or JSONL")
    parser.add_argument("--output", required=True, help="New immutable evidence directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-id", action="append")
    selection.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly invoke paid model and retrieval APIs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--public-excluded-urls", help="Publicly specified JSON mapping from task ID to excluded URLs")
    parser.add_argument("--attachment-root", help="Explicit root containing only public attachment text")
    args = parser.parse_args(argv)
    if not args.allow_network:
        parser.error("Network preparation requires --allow-network; no model or retrieval request was made")
    try:
        tasks = _read_tasks(args.tasks)
        if args.task_id:
            if not set(args.task_id) <= set(tasks):
                raise ValueError("Unknown requested public task ID")
            tasks = {task_id: tasks[task_id] for task_id in dict.fromkeys(args.task_id)}
        excluded = json.loads(Path(args.public_excluded_urls).read_text(encoding="utf-8")) if args.public_excluded_urls else {}
        if not isinstance(excluded, dict) or any(not isinstance(value, list) for value in excluded.values()):
            raise ValueError("Public exclusions must map task IDs to URL lists")
        config = ModelConfig(model=args.model, endpoint=args.endpoint, key_env=args.key_env,
                             max_tokens=4096, timeout=args.timeout, temperature=0)
        config.check("evidence_query_planner")
        if not os.environ.get("SERPER_API_KEY") or not os.environ.get("JINA_API_KEY"):
            raise ValueError("Evidence preparation requires SERPER_API_KEY and JINA_API_KEY")
        from scripts.models.openai_server import OpenAIServerModel
        from scripts.tools import search_tools
        from scripts.tools.search_tools import read_page, web_search_google_serper

        def search(query):
            results, error = web_search_google_serper(query, serp_num=5, max_retries=1)
            return {"results": results, "error": error}

        loader = _public_attachment_loader(args.attachment_root, read_page)
        created = reused = 0
        for task_id, task in tasks.items():
            path = evidence_pack_path(args.output, task_id)
            if path.exists():
                load_evidence_pack(path, task)
                reused += 1
                continue
            model = OpenAIServerModel(model_id=config.model, api_base=config.endpoint,
                                      api_key=os.environ[config.key_env], temperature=0,
                                      max_tokens=4096, max_attempts=1)
            model.client = model.client.with_options(max_retries=0, timeout=config.timeout)
            ledger = BudgetLedger(max_calls=1, max_tokens=2_000_000, max_tool_calls=12 + len(task.attachments))
            metered = MeteredModel(model, ledger, "evidence_preparation", "query_planner", 4096)
            builder = EvidencePackBuilder(metered, search, read_page, ledger=ledger,
                                          identity={"query_model": config.model, "query_endpoint": config.endpoint,
                                                    "temperature": 0, "timeout": config.timeout,
                                                    "search": "google-serper-top5-one-attempt",
                                                    "crawl": "jina-read_page-raw-no-model-extraction",
                                                    "search_tool_code_sha256": hashlib.sha256(
                                                        Path(search_tools.__file__).read_bytes()).hexdigest()})
            pack = builder.build(task, public_excluded_urls=excluded.get(task_id, []), attachment_loader=loader)
            save_evidence_pack(pack, path)
            load_evidence_pack(path, task)
            created += 1
        manifest = build_evidence_manifest(tasks, args.output)
        print(json.dumps({"tasks": manifest["count"], "created": created, "reused": reused,
                          "manifest_sha256": manifest["manifest_sha256"],
                          "output": str(Path(args.output).resolve()), "task_execution": False}, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.exit(2, f"Evidence preparation: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
