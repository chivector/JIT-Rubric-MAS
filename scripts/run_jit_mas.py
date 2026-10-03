"""Bounded JIT-MAS CLI. Smoke uses only explicitly synthetic local transports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from scripts.eval.config import load_dotenv

from jit_mas.config import MASConfig, NativeModels
from jit_mas.experience import ExperienceStore
from jit_mas.pipeline import MASPipeline
from jit_mas.schemas import PublicTask, SplitManifest
from jit_mas.benchmarks import ALL_BENCHMARK_NAMES


def make_pipeline(config, store, output, *, data=None, splits=None, fixture_models=None,
                  benchmark="researchrubrics", evidence_dir=None, checker=None, knowledge_policy=None,
                  checker_source_root=None):
    from benchmark.adapter.researchrubrics import ResearchRubricsAdapter
    from jit_mas.bridge import JITHarnessSynthesizer

    tools, dataset = {}, None
    if config.backend == "scripted":
        from jit_mas.offline import FixtureModels, fixture_dataset
        if data or splits:
            raise ValueError("scripted mode only supports labelled synthetic fixtures")
        tasks, private, manifest = fixture_dataset()
        provider = fixture_models or FixtureModels()
        judge_id = "scripted-software-fixture-v1"
    else:
        if not data or not splits:
            raise ValueError("native_jit requires --data and --splits")
        if checker_source_root is not None:
            if benchmark not in {"ifeval", "ifbench"} or checker is not None:
                raise ValueError("Checker source is only valid for an instruction benchmark without an injected checker")
            from jit_mas.instruction_checkers import PinnedInstructionChecker

            checker = PinnedInstructionChecker(checker_source_root, benchmark)
        if benchmark in {"ifeval", "ifbench"} and checker is None:
            raise ValueError("IFEval and IFBench require an injected checker or --instruction-checker-source")
        provider = NativeModels(config)
        from jit_mas.benchmarks import load_benchmark
        dataset = load_benchmark(benchmark, data, available_tools=config.available_tools)
        tasks, private = dataset.tasks, dataset.private_records
        split_document = json.loads(Path(splits).read_text(encoding="utf-8"))
        if split_document.get("version") == "jit-compose-benchmark-split-v3":
            from jit_mas.experiment_splits import validate_split
            manifest = validate_split(split_document, dataset)
        else:
            manifest = SplitManifest.model_validate(split_document.get("runtime_split_manifest", split_document))
        if evidence_dir:
            from jit_mas.evidence import load_evidence_tasks
            selected_ids = set(manifest.evolution + manifest.validation + manifest.test + manifest.stream)
            selected_tasks = {task_id: tasks[task_id] for task_id in selected_ids}
            evidence_tasks = load_evidence_tasks(selected_tasks, evidence_dir, expected_count=len(selected_tasks))
            tasks = {**tasks, **evidence_tasks}
            config = config.model_copy(update={"available_tools": []})
        from scripts.tools.registry import ToolRegistry
        registry = ToolRegistry()
        names = sorted({tool for task in tasks.values() for tool in task.tools})
        supported = {"web_search", "crawl_page", "wiki_search", "final_answer"}
        if not set(names) <= supported:
            raise ValueError("JIT-MAS native tools currently require metered capability adapters; "
                             f"unsupported tools: {sorted(set(names) - supported)}")
        if names:
            registry.register_defaults(names)
        tools = registry.get_all()
        judge_id = config.models["judge"].model

    def evaluator_factory(judge):
        kwargs = {}
        if config.backend == "native_jit":
            spec = config.models["judge"]
            kwargs = {"judge_api_base": spec.endpoint, "judge_max_tokens": spec.max_tokens,
                      "judge_timeout": spec.timeout}
        if dataset is not None:
            return dataset.evaluator(judge, judge_id=judge_id, checker=checker, **kwargs)
        return ResearchRubricsAdapter(judge=judge, judge_id=judge_id, **kwargs)

    def synthesizer_factory(meta):
        meta_config = {}
        if config.backend == "native_jit":
            spec = config.models["meta"]
            meta_config = {"model_id": spec.model, "api_base": spec.endpoint,
                           "api_key": os.environ[spec.key_env], "max_tokens": spec.max_tokens}
        return JITHarnessSynthesizer(backend=config.backend, meta_model=meta,
            meta_config=meta_config, candidates=config.candidates, max_repairs=config.max_repairs,
            selector_model=meta, tools=tools)

    pipeline = MASPipeline(config, provider, evaluator_factory, synthesizer_factory, tasks, private,
                           manifest, store, output, tools=tools, knowledge_policy=knowledge_policy)
    pipeline.benchmark_dataset = dataset
    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["smoke", "evolve", "evaluate", "stream", "freeze", "rollback"])
    parser.add_argument("--config", help="YAML MASConfig; environment references are expanded")
    parser.add_argument("--data", help="Pinned local benchmark dataset file")
    parser.add_argument("--benchmark", default="researchrubrics",
                        choices=ALL_BENCHMARK_NAMES)
    parser.add_argument("--evidence-dir", help="Verified immutable shared public evidence packs")
    parser.add_argument("--instruction-checker-source",
                        help="Pinned google-research root for IFEval or IFBench checkout root; prepared resources required")
    parser.add_argument("--splits", help="Explicit task-level SplitManifest JSON")
    parser.add_argument("--state", default="outputs/jit_mas/experience.sqlite")
    parser.add_argument("--output", default="outputs/jit_mas/runs")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--version", type=int, help="Stored snapshot version for rollback")
    parser.add_argument("--unsafe-local", action="store_true", help="Permit native generated Python with host access")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    if args.limit < 1:
        parser.error("--limit must be positive")
    config = MASConfig()
    if args.config:
        raw = os.path.expandvars(Path(args.config).read_text(encoding="utf-8"))
        config = MASConfig.model_validate(yaml.safe_load(raw))
    if args.mode == "smoke":
        config = MASConfig(backend="scripted")
    if args.unsafe_local:
        config.unsafe_local = True
    store = None
    try:
        if args.mode not in ("smoke", "freeze", "rollback") and config.backend == "native_jit":
            config.check_native()
        if args.mode == "evaluate" and not Path(args.state).is_file():
            initial = ExperienceStore(args.state)
            initial.close()
        store = ExperienceStore(args.state, read_only=args.mode == "evaluate")
        if args.mode == "freeze":
            print(store.freeze(args.output))
            return 0
        if args.mode == "rollback":
            if args.version is None:
                raise ValueError("rollback requires --version")
            print(json.dumps({"restored_version": store.rollback(args.version).version}))
            return 0
        pipeline = make_pipeline(config, store, args.output, data=args.data, splits=args.splits,
                                 benchmark=args.benchmark, evidence_dir=args.evidence_dir,
                                 checker_source_root=args.instruction_checker_source)
        if args.mode == "smoke":
            evolved = pipeline.run("evolve")
            evaluated = pipeline.run("evaluate", limit=2)
            report = {"software_test_only": True, "paid_requests": 0,
                      "evolution_tasks": len(evolved), "held_out_fixture_tasks": len(evaluated),
                      "experience_version": store.snapshot().version,
                      "experience_updates": [u for o in evolved for u in o["experience_updates"]],
                      "agent_pool_version": store.snapshot().agent_pool.version,
                      "agent_pool_members": len(store.snapshot().agent_pool.profiles),
                      "agent_pool_updates": [receipt for outcome in evolved
                                             for receipt in outcome.get("agent_pool_updates", [])],
                      "output": str(Path(args.output).resolve())}
        else:
            results = pipeline.run(args.mode, args.task_id, limit=args.limit, resume=not args.no_resume)
            report = {"mode": args.mode, "backend": config.backend, "tasks": len(results),
                      "experience_version": store.snapshot().version,
                      "agent_pool_version": store.snapshot().agent_pool.version,
                      "results": [{"task_id": r["task_id"], "score": r["evaluation"]["score"],
                                   "complete": r["evaluation"]["complete"], "run_dir": r["run_dir"]} for r in results]}
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as exc:
        parser.exit(2, f"JIT-MAS: {exc}\n")
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
