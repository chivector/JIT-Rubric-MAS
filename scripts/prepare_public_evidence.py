"""Build immutable evidence packs with unauthenticated public retrieval only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.evidence import EvidenceConfig, EvidencePackBuilder, build_evidence_manifest, evidence_pack_path, save_evidence_pack
from jit_mas.public_retrieval import PublicRetriever, RetrievalConfig
from jit_mas.request_policy import RequestPolicyModel, request_gate
from jit_mas.schemas import PublicTask
from scripts.env_config import resolve_env_placeholders
from scripts.benchmark_jit_mas_live import SafeTransport


class LocalQueryPlanner:
    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        question = payload["public_task"]["question"].strip()
        return {"queries": [question]}


def read_planner_config(path):
    config = MASConfig.model_validate(resolve_env_placeholders(
        yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))))
    if config.backend != "native_jit" or "meta" not in config.models:
        raise ValueError("Query planning requires an explicit native_jit meta model")
    config.models["meta"].check("evidence_query_planner")
    return config


def query_planner(config, ledger, output_cap):
    if config is None:
        return MeteredModel(LocalQueryPlanner(), ledger, "evidence_preparation",
                            "local_query_planner", output_cap)
    from scripts.models.openai_server import OpenAIServerModel

    model_config = config.models["meta"]
    options = {"response_format": {"type": "json_object"}}
    if model_config.frequency_penalty is not None:
        options["frequency_penalty"] = model_config.frequency_penalty
    if model_config.thinking is not None:
        options["extra_body"] = {"thinking": {"type": model_config.thinking}}
    if model_config.reasoning_effort is not None:
        options["reasoning_effort"] = model_config.reasoning_effort
    model = OpenAIServerModel(model_id=model_config.model, api_base=model_config.endpoint,
                              api_key=os.environ[model_config.key_env], temperature=0,
                              max_tokens=output_cap, max_attempts=1,
                              http_trust_env=model_config.http_trust_env, **options)
    model.client = model.client.with_options(max_retries=0, timeout=model_config.timeout)
    client = model.client
    model = SafeTransport(model, os.environ[model_config.key_env])
    concurrency = model_config.max_inflight_requests or config.max_inflight_requests
    model = RequestPolicyModel(model, gate=request_gate(concurrency, endpoint=model_config.endpoint),
                               ledger=ledger, timeout=model_config.timeout,
                               expected_model=model_config.expected_response_model)
    planner = MeteredModel(model, ledger, "evidence_preparation", "query_planner", output_cap,
                           context_window=model_config.context_window,
                           context_margin=model_config.context_margin,
                           context_policy=model_config.context_policy)
    planner.query_client = client
    return planner


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
    parser.add_argument("--planner-config", type=Path,
                        help="MASConfig JSON/YAML; use its meta model for one metered public query plan")
    args = parser.parse_args(argv)
    planner_config = read_planner_config(args.planner_config) if args.planner_config else None
    output_cap = min(4096, planner_config.models["meta"].max_tokens) if planner_config else 4096
    evidence_config = EvidenceConfig(max_pages=args.max_pages, query_max_tokens=output_cap)
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
        ledger = BudgetLedger(max_calls=1,
                              max_tokens=planner_config.max_total_tokens if planner_config else 10000,
                              max_tool_calls=12 + len(task.attachments))
        planner = query_planner(planner_config, ledger, output_cap)
        planner_identity = ({"query_model": planner_config.models["meta"].model_dump(mode="json"),
                             "query_max_attempts": 1, "query_temperature": 0}
                            if planner_config else {"query_planner": "local_full_question_v1"})
        builder = EvidencePackBuilder(planner, retriever.search, retriever.fetch,
                                      config=evidence_config, ledger=ledger,
                                      identity={"retrieval": retriever.identity,
                                                "network": "bing-rss-then-duckduckgo-html",
                                                **planner_identity})
        try:
            pack = builder.build(task)
        finally:
            client = getattr(planner, "query_client", None)
            if client is not None:
                client.close()
        save_evidence_pack(pack, destination)
        created += 1
    manifest = build_evidence_manifest(tasks, output)
    print(json.dumps({"tasks": len(tasks), "created": created,
                      "manifest_sha256": manifest["manifest_sha256"],
                      "retriever": retriever.identity}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
