"""Run native benchmark tasks with durable efficiency and evolution receipts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.agent_pool import effective_pool
from jit_mas.benchmarks import ALL_BENCHMARK_NAMES, load_benchmark
from jit_mas.checkpoints import snapshot_store
from jit_mas.config import MASConfig
from jit_mas.evidence import load_evidence_tasks
from jit_mas.experience import ExperienceStore
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import AgentPoolSnapshot, ExperienceSnapshot, SplitManifest, digest, utc_now
from scripts.env_config import resolve_env_placeholders
from scripts.mas_baseline_methods import JUDGE_ENVELOPE_VERSION, JudgeEnvelopeModel
from scripts.run_jit_mas import make_pipeline


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, base):
    location = Path(value)
    return location.resolve() if location.is_absolute() else (base / location).resolve()


def _redact(value):
    secrets = [secret for name, secret in os.environ.items()
               if len(secret) >= 4 and any(part in name.upper()
                                          for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def _failure(error):
    return _redact({"type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(),
                    "run_failure": getattr(error, "jit_mas_run_failure", None),
                    "provider_failure": getattr(error, "jit_mas_failure", None)})


def _pool(snapshot):
    return snapshot.agent_pool.model_dump(mode="json")


def _harness(pool):
    pool = effective_pool(AgentPoolSnapshot.model_validate(pool)).model_dump(mode="json")
    fields = ("version", "prompt", "skills", "memory", "preferred_tools", "reasoning_strategy",
              "planning_strategy", "communication", "harness")
    return {profile["pool_agent_id"]: {field: profile.get(field) for field in fields}
            for profile in pool.get("profiles", [])}


def _changes(before, after):
    old_ids = {item["operation_id"] for item in before.get("structural_history", [])}
    applied = [item for item in after.get("structural_history", [])
               if item["operation_id"] not in old_ids]
    old_harness, new_harness = _harness(before), _harness(after)
    old_profiles, new_profiles = old_harness or {}, new_harness or {}
    common = old_profiles.keys() & new_profiles.keys()
    changed = [identity for identity in sorted(common)
               if digest({key: value for key, value in old_profiles[identity].items() if key != "version"})
               != digest({key: value for key, value in new_profiles[identity].items() if key != "version"})]
    return {"operations_applied": applied,
            "harness_before": old_harness, "harness_after": new_harness,
            "harness_changed_agent_ids": changed,
            "harness_modification_count": len(changed),
            "harness_added_agent_ids": sorted(new_profiles.keys() - old_profiles.keys()),
            "harness_deleted_agent_ids": sorted(old_profiles.keys() - new_profiles.keys())}


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_redact(value), ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _records(path, *, recover=False):
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    records, offset = [], 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not recover or index != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                    raise
                partial = path.with_name(path.name + ".partial-" + hashlib.sha256(line).hexdigest()[:12])
                partial.write_bytes(line)
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                return records
        offset += len(line)
    if recover and content and not content.endswith((b"\n", b"\r")):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return records


def _slot_id(name, phase, position, task_id):
    return digest({"benchmark": name, "phase": phase, "round_index": position, "task_id": task_id})


def load_inputs(config_path, plan_path):
    config_path, plan_path = Path(config_path).resolve(), Path(plan_path).resolve()
    config = MASConfig.model_validate(resolve_env_placeholders(
        yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))))
    if config.backend != "native_jit":
        raise ValueError("This runner requires backend: native_jit")
    plan = _read(plan_path)
    rows = plan.get("benchmarks") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Plan requires a nonempty benchmarks array")
    materials, names = [], set()
    for row in rows:
        name = row.get("name")
        if name not in ALL_BENCHMARK_NAMES or name in names:
            raise ValueError("Plan benchmark names must be supported and unique")
        names.add(name)
        data = _path(row["data"], plan_path.parent)
        dataset = load_benchmark(name, data, available_tools=config.available_tools)
        split = SplitManifest(evolution=row.get("evolution_ids", []), test=row.get("test_ids", []))
        selected = split.evolution + split.test
        if not selected or not set(selected) <= set(dataset.tasks):
            raise ValueError(f"Plan has no tasks or unknown task IDs for {name}")
        attached, evidence_identity = {}, None
        evidence = _path(row["evidence_dir"], plan_path.parent) if row.get("evidence_dir") else None
        if evidence:
            manifest = _read(evidence / "manifest.json")
            evidence_ids = set(manifest.get("tasks", {}))
            if not set(selected) <= evidence_ids or not evidence_ids <= set(dataset.tasks):
                raise ValueError(f"Evidence manifest lacks selected public tasks for {name}")
            attached = load_evidence_tasks({identity: dataset.tasks[identity]
                                           for identity in evidence_ids}, evidence)
            evidence_identity = {"directory": str(evidence), "manifest_sha256": _hash(evidence / "manifest.json"),
                                 "files": {entry["file"]: entry["file_sha256"]
                                           for entry in manifest["tasks"].values()}}
        checker_root = _path(row["checker_source"], plan_path.parent) if row.get("checker_source") else None
        checker = None
        if name in {"ifeval", "ifbench"}:
            if checker_root is None:
                raise ValueError(f"Plan requires checker_source for {name}")
            checker = PinnedInstructionChecker(checker_root, name)
            checker.assert_frozen()
        elif checker_root is not None:
            raise ValueError("checker_source only applies to IFEval and IFBench")
        initial_path = _path(row["initial_snapshot"], plan_path.parent) if row.get("initial_snapshot") else None
        initial = ExperienceSnapshot.model_validate(_read(initial_path)) if initial_path else None
        materials.append({"name": name, "data": data, "dataset": dataset, "split": split,
                          "attached": attached, "checker": checker, "evidence": evidence_identity,
                          "initial_snapshot": initial, "initial_snapshot_path": initial_path})
    identity = {"schema": "high-score-native-experiment-v1", "config_sha256": _hash(config_path),
                "resolved_configuration_sha256": digest(config), "plan_sha256": _hash(plan_path),
                "code_fingerprint": code_fingerprint(), "runner_sha256": _hash(__file__),
                "materials": [{"name": item["name"], "data": str(item["data"]),
                               "dataset_sha256": item["dataset"].dataset_sha256,
                               "split": item["split"].model_dump(mode="json"),
                               "evidence": item["evidence"],
                               "initial_snapshot_sha256": digest(item["initial_snapshot"])
                                   if item["initial_snapshot"] is not None else None,
                               "checker": item["checker"].identity if item["checker"] else None}
                              for item in materials],
                "transport": {"model_attempts": os.environ.get("JIT_MAS_MODEL_ATTEMPTS", "1"),
                              "disable_keepalive": os.environ.get("JIT_MAS_DISABLE_KEEPALIVE", "0")},
                "judge_envelope": JUDGE_ENVELOPE_VERSION}
    return config, materials, identity


def _track_clients(provider):
    original = provider.create
    clients, lock = [], threading.Lock()

    def create(*args, **kwargs):
        model = original(*args, **kwargs)
        transport = model
        visited = set()
        while hasattr(transport, "model") and id(transport) not in visited:
            visited.add(id(transport))
            transport = transport.model
        client = getattr(transport, "client", None)
        if client is not None:
            with lock:
                clients.append(client)
        role = args[0] if args else kwargs.get("role")
        return JudgeEnvelopeModel(model) if role == "judge" else model

    provider.create = create
    return clients


def _close_clients(clients):
    errors = []
    while clients:
        client = clients.pop()
        try:
            client.close()
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})
    return _redact(errors)


def _requested(outcome, failure):
    if outcome is not None:
        return outcome.get("pool_operations", [])
    directory = ((failure or {}).get("run_failure") or {}).get("run_dir")
    if directory:
        path = Path(directory) / "meta_evolution.json"
        if path.is_file():
            return (_read(path).get("decision") or {}).get("pool_operations", [])
    return []


def _harness_artifact(outcome, failure):
    directory = (outcome or {}).get("run_dir") or (((failure or {}).get("run_failure") or {}).get("run_dir"))
    artifact = None
    details = {}
    if directory:
        location = Path(directory)
        if (location / "harness.json").is_file():
            artifact = _read(location / "harness.json")
        if (location / "failure.json").is_file():
            details = _read(location / "failure.json")
            artifact = artifact or details.get("artifact")
    trajectory = (artifact or {}).get("meta_trajectory") or details.get("meta_trajectory", [])
    return {"harness_artifact": artifact,
            "harness_repair_count": (artifact or {}).get("repair_count"),
            "harness_meta_trajectory": trajectory,
            "harness_failed_attempts": details.get("failed_attempts", [])}


def _interrupted_record(started, store, runs):
    journal = store.task_run("evolve", started["task_id"]) if started["phase"] == "evolution" else None
    outcome = (journal or {}).get("outcome")
    locations = []
    for manifest_path in runs.glob("*/run_manifest.json"):
        manifest = _read(manifest_path)
        task = (manifest.get("comparison") or {}).get("task", {})
        mode = "evolve" if started["phase"] == "evolution" else "evaluate"
        if (task.get("task_id") == started["task_id"] and manifest.get("mode") == mode
                and manifest.get("experience_hash") == started["snapshot_before_hash"]):
            locations.append(manifest_path.parent)
    if len(locations) > 1:
        raise ValueError("Interrupted slot matches multiple run directories")
    directory = Path(outcome["run_dir"]) if outcome else locations[0] if locations else None
    recovered = {}
    budget = (outcome or {}).get("budget")
    if directory:
        if (directory / "complete.json").is_file():
            outcome = outcome or _read(directory / "complete.json")
        if (directory / "failure.json").is_file():
            recovered = _read(directory / "failure.json")
        if (directory / "budget.json").is_file():
            budget = _read(directory / "budget.json")
        else:
            budget = budget or (outcome or {}).get("budget") or recovered.get("budget")
    failure = {"type": "InterruptedSlot", "message": "Recovered interrupted slot without resampling",
               "run_failure": recovered or ({"run_dir": str(directory)} if directory else None)}
    after = store.snapshot()
    return {**started, "finished_at": utc_now(), "status": "interrupted",
            "outcome": outcome, "budget": budget, "latency_seconds": None,
            "failure": failure, "pool_after": _pool(after), "experience_version_after": after.version,
            "snapshot_after_hash": digest(after), "operations_requested": _requested(outcome, failure),
            "agent_pool_size_before": len(effective_pool(AgentPoolSnapshot.model_validate(started["pool_before"])).profiles),
            "agent_pool_size_after": len(effective_pool(after.agent_pool).profiles),
            **_changes(started["pool_before"], _pool(after)), **_harness_artifact(outcome, failure)}


def execute_slot(pipeline, store, clients, output, name, phase, position, task_id):
    before = store.snapshot()
    slot_id = _slot_id(name, phase, position, task_id)
    started = {"slot_id": slot_id, "benchmark": name, "phase": phase, "round_index": position,
               "task_id": task_id, "started_at": utc_now(), "snapshot_before_hash": digest(before),
               "experience_version_before": before.version, "pool_before": _pool(before)}
    slot_dir = output / "slots" / slot_id
    write_json(slot_dir / "started.json", started)
    print(json.dumps({"benchmark": name, "phase": phase, "round": position,
                      "task_id": task_id, "status": "started"}), flush=True)
    log = io.StringIO()
    outcome = failure = interrupted = None
    status = "failed"
    clock_start = time.monotonic()
    try:
        with redirect_stdout(log), redirect_stderr(log):
            if phase == "evolution":
                outcome = pipeline.run("evolve", [task_id])[0]
            else:
                outcome = pipeline.run_task(task_id, before, mode="evaluate", attribution=False)
        status = "complete" if outcome["evaluation"].get("complete") is True else "incomplete"
    except BaseException as error:
        failure = _failure(error)
        if not isinstance(error, Exception):
            status, interrupted = "interrupted", error
        journal = store.task_run("evolve", task_id) if phase == "evolution" else None
        if journal:
            outcome = journal.get("outcome")
    finally:
        latency = time.monotonic() - clock_start
        after = store.snapshot()
        close_errors = _close_clients(clients)
        raw_failure = (failure or {}).get("run_failure") or {}
        budget = (outcome or {}).get("budget") or raw_failure.get("budget")
        record = {**started, "finished_at": utc_now(), "status": status, "outcome": outcome,
                  "failure": failure, "budget": budget, "latency_seconds": latency,
                  "experience_version_after": after.version, "snapshot_after_hash": digest(after),
                  "pool_after": _pool(after),
                  "agent_pool_size_before": len(effective_pool(before.agent_pool).profiles),
                  "agent_pool_size_after": len(effective_pool(after.agent_pool).profiles),
                  "operations_requested": _requested(outcome, failure),
                  "client_close_errors": close_errors, **_changes(_pool(before), _pool(after)),
                  **_harness_artifact(outcome, failure)}
        (slot_dir / "stdout_stderr.log").write_text(_redact(log.getvalue()), encoding="utf-8")
        write_json(slot_dir / "record.json", _redact(record))
        _append(output / "records.jsonl", record)
        print(json.dumps({"benchmark": name, "phase": phase, "round": position,
                          "task_id": task_id, "status": status,
                          "score": (outcome or {}).get("evaluation", {}).get("score"),
                          "tokens": (budget or {}).get("tokens"),
                          "latency_seconds": latency}), flush=True)
    if interrupted is not None:
        raise interrupted
    return record


def summarize(records):
    summary = {}
    for record in records:
        name = record["benchmark"]
        phase = record["phase"]
        group = summary.setdefault(name, {}).setdefault(phase, {
            "attempted": 0, "complete": 0, "failed": 0, "incomplete": 0, "scores": [],
            "latency_seconds": 0, "model_calls": 0, "tokens": 0})
        group["attempted"] += 1
        group[record["status"] if record["status"] in {"complete", "incomplete"} else "failed"] += 1
        evaluation = (record.get("outcome") or {}).get("evaluation", {})
        if record["status"] == "complete" and evaluation.get("score") is not None:
            group["scores"].append(evaluation["score"])
        group["latency_seconds"] += record.get("latency_seconds") or 0
        group["model_calls"] += (record.get("budget") or {}).get("model_calls") or 0
        group["tokens"] += (record.get("budget") or {}).get("tokens") or 0
        group["mean_score_complete"] = (sum(group["scores"]) / len(group["scores"])
                                        if group["scores"] else None)
        group["failure_rate"] = group["failed"] / group["attempted"]
        group["incomplete_rate"] = (group["failed"] + group["incomplete"]) / group["attempted"]
    return summary


def run(config, materials, identity, output, *, resume=False):
    config.check_native()
    output = Path(output).resolve()
    if output.exists():
        if not resume or _read(output / "manifest.json").get("identity") != identity:
            raise ValueError("Output already exists or resume identity differs")
    else:
        output.mkdir(parents=True)
        write_json(output / "manifest.json", {"identity": identity, "started_at": utc_now(),
                   "configuration": config.model_dump(mode="json"),
                   "evolution": "Per-task direct committed MAS evolution; TEST does not update state"})
    records = _records(output / "records.jsonl", recover=resume)
    seen = {row["slot_id"] for row in records}
    for material in materials:
        name = material["name"]
        directory = output / name
        write_json(directory / "runtime_split.json", material["split"])
        state_path = directory / "experience.sqlite"
        if not state_path.exists() and material.get("initial_snapshot") is not None:
            seeded = snapshot_store(material["initial_snapshot"], state_path)
            seeded.close()
        store = ExperienceStore(state_path)
        clients = []
        try:
            pipeline = make_pipeline(config, store, directory / "runs", data=material["data"],
                                     splits=directory / "runtime_split.json", benchmark=name,
                                     checker=material["checker"])
            pipeline.tasks.update({identity: material["attached"][identity]
                                   for identity in material["split"].evolution + material["split"].test
                                   if identity in material["attached"]})
            clients = _track_clients(pipeline.models)
            for phase, task_ids in (("evolution", material["split"].evolution),
                                    ("test", material["split"].test)):
                for position, task_id in enumerate(task_ids, 1):
                    slot_id = _slot_id(name, phase, position, task_id)
                    if slot_id in seen:
                        continue
                    record_path = output / "slots" / slot_id / "record.json"
                    if record_path.is_file():
                        record = _read(record_path)
                        _append(output / "records.jsonl", record)
                    elif (output / "slots" / slot_id / "started.json").exists():
                        started = _read(output / "slots" / slot_id / "started.json")
                        record = _interrupted_record(started, store, directory / "runs")
                        write_json(record_path, _redact(record))
                        _append(output / "records.jsonl", record)
                    else:
                        record = execute_slot(pipeline, store, clients, output, name, phase, position, task_id)
                    records.append(record)
                    seen.add(slot_id)
                    write_json(output / "report.json", {"identity": identity, "updated_at": utc_now(),
                               "summary": summarize(records), "record_count": len(records)})
        finally:
            _close_clients(clients)
            store.close()
    report = {"identity": identity, "finished_at": utc_now(), "summary": summarize(records),
              "record_count": len(records), "records_file": "records.jsonl"}
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate local inputs without model requests or outputs")
    args = parser.parse_args(argv)
    try:
        config, materials, identity = load_inputs(args.config, args.plan)
        if args.check:
            print(json.dumps({"inputs_ready": True, "model_calls": 0,
                              "benchmarks": [{"name": item["name"],
                                  "evolution_tasks": len(item["split"].evolution),
                                  "test_tasks": len(item["split"].test)} for item in materials]}, ensure_ascii=True))
            return 0
        if not args.output:
            raise ValueError("--output is required unless --check is used")
        result = run(config, materials, identity, args.output, resume=args.resume)
        print(json.dumps({"finished": True, "record_count": result["record_count"],
                          "summary": result["summary"]}, ensure_ascii=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, "native experiment: " + _redact(str(error)) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
