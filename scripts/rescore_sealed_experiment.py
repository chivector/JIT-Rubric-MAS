"""Immutable-answer rescoring with explicit judge identity and append-only attempts.

This runner never calls an answer generator. Private evaluator records are used
only inside the registered benchmark adapters; status emits metadata only.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
from urllib.parse import urlsplit

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "jit_mas").is_dir() and (path / "scripts").is_dir())
sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import load_benchmark
from scripts.run_rr_two_arm_pilot import _configure_logging
import re
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.pipeline import convert_feedback
from jit_mas.request_policy import RequestPolicyModel, request_gate
from jit_mas.schemas import digest, utc_now
from scripts.mas_baseline_methods import JudgeEnvelopeModel
from scripts.models.openai_server import OpenAIServerModel
from scripts.probe_jit_mas_api import CredentialSafeModel
from benchmark.adapter.researchrubrics import ResearchRubricsAdapter

VERSION = "sealed-rescoring-v2"
FINAL_VERSION = "sealed-rescoring-evaluations-v2"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_bindings(values, name):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name} binding must be NAME=PATH")
        key, path = value.split("=", 1)
        if not key or not path:
            raise ValueError(f"{name} binding must be NAME=PATH")
        if name == "source" and not LABEL_RE.fullmatch(key):
            raise ValueError("Source labels may contain only letters, digits, dot, underscore and hyphen")
        if key in result:
            raise ValueError(f"Duplicate {name} binding: {key}")
        result[key] = Path(path).expanduser()
    return result


def release_path(source):
    source = Path(source).resolve()
    if (source / "inventory.json").is_file() and (source / "seal.json").is_file():
        return source
    candidate = source / "test_release"
    if (candidate / "inventory.json").is_file() and (candidate / "seal.json").is_file():
        return candidate
    candidate = source / "release"
    if (candidate / "inventory.json").is_file() and (candidate / "seal.json").is_file():
        return candidate
    raise ValueError(f"sealed release is missing below source: {source}")


def relocate_data(path, source):
    path = Path(path)
    if path.is_file():
        return path.resolve()
    candidates = []
    root = Path(__file__).resolve().parents[1]
    basename = path.name
    for base in (Path(source).resolve(), root / "outputs", root / "dataset", root):
        if base.is_file() and base.name == basename:
            candidates.append(base.resolve())
        elif base.is_dir():
            candidates.extend(p.resolve() for p in base.rglob(basename) if p.is_file())
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(f"Cannot safely relocate pinned data file: {path}")
    return unique[0]


def verify_release(release):
    inventory = read_json(release / "inventory.json")
    seal = read_json(release / "seal.json")
    if digest(inventory.get("slots")) != inventory.get("inventory_hash"):
        raise ValueError("source inventory hash mismatch")
    if seal.get("inventory_hash") != inventory.get("inventory_hash"):
        raise ValueError("source seal inventory hash mismatch")
    slots = inventory.get("slots")
    if not isinstance(slots, list) or not slots or len({row.get("slot_id") for row in slots}) != len(slots):
        raise ValueError("source inventory has invalid or duplicate slots")
    records = {}
    for slot in slots:
        slot_id = slot["slot_id"]
        path = release / "submissions" / f"{digest(slot_id)}.json"
        if not path.is_file():
            raise ValueError(f"sealed submission is missing: {slot_id}")
        record = read_json(path)
        if record.get("slot") != slot:
            raise ValueError(f"sealed submission slot mismatch: {slot_id}")
        if seal.get("submission_hashes", {}).get(slot_id) != digest(record):
            raise ValueError(f"sealed submission hash mismatch: {slot_id}")
        if record.get("status") == "submitted":
            submission = record.get("submission") or {}
            if digest(submission.get("answer")) != submission.get("answer_hash"):
                raise ValueError(f"sealed answer hash mismatch: {slot_id}")
        elif record.get("status") != "failed":
            raise ValueError(f"unknown sealed submission status: {slot_id}")
        records[slot_id] = (path, record)
    if set(seal.get("submission_hashes", {})) != set(records):
        raise ValueError("source seal does not cover exactly the inventory")
    return inventory, records


def metadata_for(source, release):
    root = release.parent
    registration_path = root / "registration.json"
    metadata_path = root / "pilot_metadata.json"
    if registration_path.is_file():
        registration = read_json(registration_path)
        return {"kind": "development", "path": root, "registration": registration,
                "metadata_hash": file_hash(registration_path)}
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    return {"kind": "rr", "path": root, "metadata": metadata,
            "metadata_hash": file_hash(metadata_path) if metadata_path.is_file() else None}


def data_bindings_for(source, info, data_bindings):
    result = {}
    if info["kind"] == "development":
        for benchmark, identity in info["registration"].get("sources", {}).items():
            result[benchmark] = data_bindings.get(f"{source}:{benchmark}",
                                                  data_bindings.get(benchmark, Path(identity["data_path"])))
    else:
        benchmark = "researchrubrics"
        metadata = info["metadata"]
        result[benchmark] = data_bindings.get(f"{source}:{benchmark}",
                                              data_bindings.get(benchmark, Path(metadata.get("data", ""))))
    return {key: relocate_data(value, info["path"]) for key, value in result.items()}


def checker_for(source, benchmark, info):
    if benchmark not in {"ifeval", "ifbench"}:
        return None
    if info["kind"] != "development":
        raise ValueError(f"{source}:{benchmark} has no pinned author checker registration")
    identity = info["registration"]["sources"].get(benchmark, {}).get("checker") or {}
    if not identity.get("source_root"):
        raise ValueError(f"{source}:{benchmark} is missing pinned author checker")
    from jit_mas.instruction_checkers import PinnedInstructionChecker
    checker = PinnedInstructionChecker(identity["source_root"], benchmark)
    if checker.identity != identity.get("identity") or digest(checker.identity) != identity.get("identity_sha256"):
        raise ValueError(f"{source}:{benchmark} author checker identity changed")
    checker.assert_frozen()
    return checker


def _load_prepared_sources(source_bindings, data_bindings):
    prepared = {}
    for label, source in source_bindings.items():
        release = release_path(source)
        inventory, records = verify_release(release)
        info = metadata_for(source, release)
        paths = data_bindings_for(label, info, data_bindings)
        for benchmark, path in paths.items():
            expected = (info["registration"]["sources"][benchmark].get("dataset_sha256")
                        if info["kind"] == "development" else info["metadata"].get("dataset_sha256"))
            if expected and file_hash(path) != expected:
                raise ValueError(f"Pinned source data hash changed: {label}:{benchmark}")
        datasets = {benchmark: load_benchmark(benchmark, path) for benchmark, path in paths.items()}
        checkers = {benchmark: checker_for(label, benchmark, info) for benchmark in datasets}
        prepared[label] = {"source": source.resolve(), "release": release, "inventory": inventory,
                           "records": records, "info": info, "datasets": datasets, "checkers": checkers,
                           "data_paths": paths}
    return prepared


def output_slots(prepared):
    slots = []
    for label, item in prepared.items():
        for source_slot in item["inventory"]["slots"]:
            row = dict(source_slot)
            row["slot_id"] = f"{label}:{source_slot['slot_id']}"
            row["source_label"] = label
            row["source_slot_id"] = source_slot["slot_id"]
            row["benchmark"] = row.get("benchmark") or ("researchrubrics" if item["info"]["kind"] == "rr" else None)
            if not row["benchmark"]:
                raise ValueError(f"Cannot infer benchmark for {label}:{source_slot['slot_id']}")
            slots.append(row)
    return slots


def binding_for_slot(slot, prepared):
    item = prepared[slot["source_label"]]
    benchmark = slot["benchmark"]
    if slot["task_id"] not in item["datasets"][benchmark].tasks:
        raise ValueError(f"Task is absent from pinned data: {slot['slot_id']}")
    return {"dataset": item["datasets"][benchmark], "benchmark": benchmark,
            "checker": item["checkers"].get(benchmark), "task_id": slot["task_id"]}


def write_json(path, value):
    """Atomically replace only derived files; immutable records use create_json."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def create_json(path, value):
    target = Path(path)
    if target.exists():
        if read_json(target) != value:
            raise ValueError("Immutable record changed: " + target.name)
        return
    write_json(target, value)


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}))
    except FileExistsError:
        raise RuntimeError("Output is locked; inspect the owner process before restoring a stopped run") from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def code_identity():
    files = [Path(__file__), ROOT / "jit_mas/budget.py",
             ROOT / "jit_mas/request_policy.py", ROOT / "jit_mas/benchmarks.py",
             ROOT / "jit_mas/pipeline.py", ROOT / "jit_mas/schemas.py",
             ROOT / "jit_mas/instruction_checkers.py", ROOT / "benchmark/adapter/researchrubrics.py",
             ROOT / "scripts/mas_baseline_methods.py", ROOT / "scripts/probe_jit_mas_api.py",
             ROOT / "scripts/models/base.py", ROOT / "scripts/models/openai_server.py",
             ROOT / "scripts/kernel/token_counter.py"]
    files += sorted((ROOT / "benchmark/adapter/researchrubrics_prompts").glob("*.txt"))
    return {str(path.resolve()): file_hash(path) for path in files}


def validate_registration(info):
    if info["kind"] == "development":
        registration = info["registration"]
        expected = registration.get("registration_hash")
        body = {key: value for key, value in registration.items() if key != "registration_hash"}
        if not expected or digest(body) != expected:
            raise ValueError("Source registration hash mismatch")


def prepare_sources(source_bindings, data_bindings):
    # Validate registration metadata before the legacy loader touches datasets.
    for source in source_bindings.values():
        release = release_path(source)
        info = metadata_for(source, release)
        validate_registration(info)
        if info["kind"] == "rr" and not info["metadata"].get("dataset_sha256"):
            raise ValueError("RR source metadata must pin dataset_sha256")
        if info["kind"] == "development":
            if any(not row.get("dataset_sha256") for row in info["registration"]["sources"].values()):
                raise ValueError("Development source registration must pin every dataset hash")
    prepared = _load_prepared_sources(source_bindings, data_bindings)
    for item in prepared.values():
        if item["info"]["kind"] == "development":
            if item["inventory"]["slots"] != item["info"]["registration"].get("slots"):
                raise ValueError("Source inventory differs from registered slots")
    return prepared


def source_identity(prepared):
    return {label: {"path": str(item["source"]), "release": str(item["release"]),
            "inventory_sha256": file_hash(item["release"] / "inventory.json"),
            "seal_sha256": file_hash(item["release"] / "seal.json"),
            "inventory_hash": item["inventory"]["inventory_hash"], "kind": item["info"]["kind"],
            "metadata_sha256": item["info"]["metadata_hash"],
            "data": {key: {"path": str(path), "sha256": file_hash(path)}
                     for key, path in item["data_paths"].items()}}
            for label, item in prepared.items()}


def assert_sources_frozen(prepared, expected):
    if source_identity(prepared) != expected:
        raise ValueError("Registered source, dataset or metadata bytes changed")
    for item in prepared.values():
        verify_release(item["release"])
        info = metadata_for(item["source"], item["release"])
        validate_registration(info)
        if info["metadata_hash"] != item["info"]["metadata_hash"]:
            raise ValueError("Source metadata hash changed")
        for checker in item["checkers"].values():
            if checker is not None:
                checker.assert_frozen()


def pinned_error(error, secret=""):
    # Exception strings may contain judge prompts/private criteria: retain type only.
    return type(error).__name__


def task_identity(slot, answer_hash):
    return digest({"source_label": slot["source_label"], "source_slot_id": slot["source_slot_id"],
                   "benchmark": slot["benchmark"], "task_id": slot["task_id"],
                   "answer_hash": answer_hash})


def read_history(directories):
    """Pin metadata and raw hashes; never expose evaluator bodies to status."""
    receipts, entries = [], {}
    for directory in directories:
        parent = Path(directory).resolve()
        config = read_json(parent / "config.json")
        inventory, records = verify_release(parent / "test_release")
        prior_is_v2 = config.get("version") == VERSION
        if prior_is_v2:
            validate_config(config)
            verify_final_seal(parent, required=False)
            for receipt in config["history"]:
                for relative, expected in receipt["files"].items():
                    if file_hash(Path(receipt["path"]) / relative) != expected:
                        raise ValueError("Ancestor history bytes changed")
                receipts.append(receipt)
        # Legacy history is immutable-by-hash from this registration onward, not
        # retroactively described as originally evaluation-sealed or model-pinned.
        file_hashes = {name: file_hash(parent / name) for name in (
            "config.json", "test_release/inventory.json", "test_release/seal.json")}
        final = parent / "final_seal.json"
        if final.exists():
            file_hashes["final_seal.json"] = file_hash(final)
        for slot in inventory["slots"]:
            key = slot["slot_id"]
            record = records[key][1]
            answer_hash = (record.get("submission") or {}).get("answer_hash")
            evaluation = parent / "test_release/evaluations" / (digest(key) + ".json")
            started = parent / "test_release/evaluation_started" / (digest(key) + ".json")
            if not evaluation.exists() and not started.exists():
                continue
            if prior_is_v2:
                entries.setdefault(key, []).extend(copy.deepcopy(
                    config["history_metadata"].get(key, [])))
            row = read_json(evaluation) if evaluation.exists() else None
            if row is not None:
                if row.get("slot") != slot:
                    raise ValueError("History evaluation slot mismatch")
                if row.get("answer_hash") not in (None, answer_hash):
                    raise ValueError("History evaluation answer hash mismatch")
                budget = row.get("evaluation_budget") or {}
                tokens, calls = int(budget.get("tokens", 0)), int(budget.get("model_calls", 0))
                status = row.get("status")
                entry = {"path": str(parent), "slot_id": key, "answer_hash": answer_hash,
                         "task_identity": task_identity(slot, answer_hash), "status": status,
                         "complete": row.get("complete") is True, "tokens": tokens, "model_calls": calls,
                         "estimated": any(r.get("estimated") for r in budget.get("records", [])),
                         "unknown_usage": False, "evaluation_sha256": file_hash(evaluation),
                         "judge": config.get("judge"), "legacy_unsealed_history": not prior_is_v2}
                if prior_is_v2:
                    validate_evaluation(parent, slot, record, config, row)
                    previous_calls = int(budget.get("model_calls", 0))
                    if previous_calls < 0:
                        raise ValueError("History model call count cannot be negative")
            else:
                marker = read_json(started)
                if marker.get("record_hash") != digest(record):
                    raise ValueError("History started marker submission hash mismatch")
                tokens = int(marker.get("budget_tokens") or config.get("judge", {}).get("task_budget_tokens", 0))
                if tokens <= 0:
                    raise ValueError("Interrupted history lacks a positive frozen token budget")
                entry = {"path": str(parent), "slot_id": key, "answer_hash": answer_hash,
                         "task_identity": task_identity(slot, answer_hash), "status": "interrupted_unknown_usage",
                         "complete": False, "tokens": tokens, "model_calls": 0,
                         "estimated": True, "unknown_usage": True, "evaluation_sha256": None,
                         "judge": config.get("judge"), "legacy_unsealed_history": not prior_is_v2}
            if evaluation.exists():
                file_hashes[str(evaluation.relative_to(parent))] = file_hash(evaluation)
            if started.exists():
                file_hashes[str(started.relative_to(parent))] = file_hash(started)
            entries.setdefault(key, []).append(entry)
        receipts.append({"path": str(parent), "files": file_hashes,
                         "legacy_unsealed_history": not prior_is_v2})
    # Importing both a parent and its successor must not charge the parent twice.
    unique_receipts = {}
    for receipt in receipts:
        key = receipt["path"]
        if key in unique_receipts and unique_receipts[key] != receipt:
            raise ValueError("Conflicting ancestor history identities")
        unique_receipts[key] = receipt
    for key, rows in entries.items():
        unique = {}
        for row in rows:
            identity = (row["path"], row["slot_id"])
            if identity in unique and unique[identity] != row:
                raise ValueError("Conflicting history attempt identities")
            unique[identity] = row
        entries[key] = list(unique.values())
    return list(unique_receipts.values()), entries


def history_cost(entries):
    # read_history flattens unique ancestor attempts; every failed or interrupted
    # attempt is charged once, whether one or several parents were supplied.
    return {"tokens": sum(row["tokens"] for row in entries),
            "model_calls": sum(row["model_calls"] for row in entries),
            "unknown_usage_count": sum(row["unknown_usage"] for row in entries),
            "estimated": any(row["estimated"] for row in entries)}


def judge_options(args):
    models = sorted(set(args.expected_response_model or []))
    if not models or any(not isinstance(value, str) or not value.strip() for value in models):
        raise ValueError("At least one exact --expected-response-model is required")
    return {"model": args.judge_model, "endpoint": args.judge_endpoint,
            "key_env": args.judge_key_env, "expected_response_models": models,
            "max_tokens": args.judge_max_tokens, "timeout": args.timeout,
            "transport_attempts": args.transport_attempts, "parse_attempts": 1,
            "rubric_parallel": args.rubric_parallel, "process_parallel": args.process_parallel,
            "task_budget_tokens": args.budget_tokens, "task_timeout_seconds": args.task_timeout,
            "retry_delay_seconds": args.retry_delay, "temperature": 0,
            "sdk_retries": 0, "adapter_revision": "existing-diagnostic-v1-no-score-changes"}


def config_body(config):
    return {key: value for key, value in config.items() if key != "registration_hash"}


def validate_config(config):
    if config.get("version") != VERSION or digest(config_body(config)) != config.get("registration_hash"):
        raise ValueError("Rescore registration hash mismatch")
    if not config.get("judge", {}).get("expected_response_models"):
        raise ValueError("Rescore registration lacks response model pin")


def initialize_output(output, prepared, args):
    output = Path(output).resolve()
    all_slots = output_slots(prepared)
    if args.slot:
        selected = set(args.slot)
        unknown = selected - {row["slot_id"] for row in all_slots}
        if unknown:
            raise ValueError("Selected slot is absent from frozen source inventory")
        slots = [slot for slot in all_slots if slot["slot_id"] in selected]
    else:
        slots = all_slots
    if not slots:
        raise ValueError("No slots selected")
    history_receipts, history = read_history(args.prior_attempt)
    options = judge_options(args)
    retry, interrupted, rejudge = set(args.retry_slot), set(args.retry_interrupted_slot), set(args.rejudge_slot)
    selected_ids = {slot["slot_id"] for slot in slots}
    if (retry | interrupted | rejudge) - selected_ids:
        raise ValueError("Retry/rejudge selection is absent from this attempt inventory")
    records = {}
    policy = {}
    for slot in slots:
        original = copy.deepcopy(prepared[slot["source_label"]]["records"][slot["source_slot_id"]][1])
        original["slot"] = slot
        records[slot["slot_id"]] = original
        answer_hash = (original.get("submission") or {}).get("answer_hash")
        past = history.get(slot["slot_id"], [])
        if any(row["task_identity"] != task_identity(slot, answer_hash) for row in past):
            raise ValueError("History input identity differs from current sealed answer")
        complete = [row for row in past if row["complete"]]
        last = past[-1] if past else None
        if complete:
            if slot["slot_id"] not in rejudge:
                raise ValueError("Previously complete slot requires explicit --rejudge-slot in a newly registered attempt")
            policy[slot["slot_id"]] = "explicit_complete_rejudge_new_attempt"
        elif last and last["unknown_usage"]:
            if slot["slot_id"] not in interrupted:
                raise ValueError("Interrupted prior slot requires explicit --retry-interrupted-slot")
            policy[slot["slot_id"]] = "explicit_interrupted_retry_full_budget_reserved"
        elif last:
            if slot["slot_id"] not in retry:
                raise ValueError("Failed prior slot requires explicit --retry-slot")
            policy[slot["slot_id"]] = "explicit_failed_retry_new_attempt"
        else:
            policy[slot["slot_id"]] = "first_attempt"
    body = {"version": VERSION, "judge": options, "sources": source_identity(prepared),
            "slot_count": len(slots), "slot_policy": policy, "history": history_receipts,
            "history_metadata": history, "code_sha256": code_identity(),
            "campaign_budget_tokens": args.campaign_budget_tokens,
            "registered_at": utc_now()}
    config_path = output / "config.json"
    if config_path.exists():
        old = read_json(config_path)
        validate_config(old)
        if {key: value for key, value in config_body(old).items() if key != "registered_at"} != {
                key: value for key, value in body.items() if key != "registered_at"}:
            raise ValueError("Rescore output configuration changed")
        config = old
    else:
        config = body | {"registration_hash": digest(body)}
        create_json(config_path, config)
    release = output / "test_release"
    inventory = {"slots": slots, "inventory_hash": digest(slots)}
    create_json(release / "inventory.json", inventory)
    for slot in slots:
        create_json(release / "submissions" / (digest(slot["slot_id"]) + ".json"), records[slot["slot_id"]])
    create_json(release / "seal.json", {"inventory_hash": inventory["inventory_hash"],
                "submission_hashes": {key: digest(value) for key, value in records.items()}})
    verify_release(release)
    return output, slots, config


def assert_frozen(output, prepared, config):
    validate_config(read_json(Path(output) / "config.json"))
    if read_json(Path(output) / "config.json") != config or code_identity() != config["code_sha256"]:
        raise ValueError("Registered configuration or evaluator source code changed")
    assert_sources_frozen(prepared, config["sources"])
    verify_release(Path(output) / "test_release")
    for receipt in config["history"]:
        for relative, expected in receipt["files"].items():
            if file_hash(Path(receipt["path"]) / relative) != expected:
                raise ValueError("Prior attempt history bytes changed")


class ResponseModelPolicy(RequestPolicyModel):
    def __init__(self, model, *, expected_models, **kwargs):
        super().__init__(model, expected_model=None, **kwargs)
        self.expected_models = tuple(expected_models)

    def __call__(self, messages, **kwargs):
        response = super().__call__(messages, **kwargs)
        if self.last_request_metadata.get("response_model") not in self.expected_models:
            raise ValueError("Provider response model differs from registered exact identity set")
        return response


def transport_retriable(error):
    from openai import APIConnectionError, APIStatusError
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        return getattr(error, "status_code", None) in {429, 500, 502, 503, 504}
    # An HTTP body that cannot be parsed as JSON is a transport failure, unlike
    # malformed JSON inside a successful model message (adapter-owned validation).
    return isinstance(error, json.JSONDecodeError)


class TransportRetryModel:
    def __init__(self, model, *, attempts, retry_delay, ledger, sleeper=time.sleep):
        self.model, self.attempts, self.retry_delay = model, attempts, retry_delay
        self.ledger, self.sleeper = ledger, sleeper

    def __call__(self, messages, **kwargs):
        for index in range(self.attempts):
            try:
                return self.model(messages, **kwargs)
            except Exception as exc:
                if not transport_retriable(exc) or index + 1 == self.attempts:
                    raise
                remaining = self.ledger.remaining_seconds()
                delay = min(30, self.retry_delay * (2 ** index))
                if remaining is not None and remaining <= delay:
                    raise TimeoutError("Task deadline cannot accommodate transport retry") from None
                self.sleeper(delay)

    def __getattr__(self, name):
        return getattr(self.model, name)


class JudgeFactory:
    def __init__(self, options, secret):
        self.options, self.secret = options, secret
        self.gate = request_gate(options["process_parallel"] * options["rubric_parallel"])

    def create(self, ledger):
        cfg = self.options
        raw = OpenAIServerModel(model_id=cfg["model"], api_base=cfg["endpoint"], api_key=self.secret,
                                max_attempts=1, max_tokens=cfg["max_tokens"], timeout=cfg["timeout"], temperature=0)
        raw.client = raw.client.with_options(max_retries=0, timeout=cfg["timeout"])
        policy = ResponseModelPolicy(CredentialSafeModel(raw, self.secret), gate=self.gate, ledger=ledger,
                                     timeout=cfg["timeout"], expected_models=cfg["expected_response_models"])
        metered = MeteredModel(policy, ledger, "evaluation", "judge", cfg["max_tokens"])
        retried = TransportRetryModel(metered, attempts=cfg["transport_attempts"],
                                      retry_delay=cfg["retry_delay_seconds"], ledger=ledger)
        return JudgeEnvelopeModel(CredentialSafeModel(retried, self.secret))


def build_evaluator(binding, factory, ledger):
    cfg = factory.options
    if binding["benchmark"] == "researchrubrics":
        return ResearchRubricsAdapter(judge_id=cfg["model"], judge_api_base=cfg["endpoint"],
                judge_max_tokens=cfg["max_tokens"], judge_timeout=cfg["timeout"], max_attempts=1,
                judge_factory=lambda: factory.create(ledger), max_parallel_judgments=cfg["rubric_parallel"])
    kwargs = {"judge_id": cfg["model"], "judge_api_base": cfg["endpoint"],
              "judge_max_tokens": cfg["max_tokens"], "judge_timeout": cfg["timeout"]}
    if binding["benchmark"] in {"ifeval", "ifbench"}:
        kwargs["checker"] = binding["checker"]
    return binding["dataset"].evaluator(factory.create(ledger), **kwargs)


def evaluation_body(row):
    return {key: value for key, value in row.items() if key != "result_hash"}


def validate_evaluation(output, slot, record, config, row=None):
    path = Path(output) / "test_release/evaluations" / (digest(slot["slot_id"]) + ".json")
    row = read_json(path) if row is None else row
    if row.get("result_hash") != digest(evaluation_body(row)):
        raise ValueError("Evaluation result hash mismatch")
    expected_answer = (record.get("submission") or {}).get("answer_hash")
    if row.get("slot") != slot or row.get("answer_hash") != expected_answer:
        raise ValueError("Evaluation input identity mismatch")
    if row.get("registration_hash") != config["registration_hash"] or row.get("record_hash") != digest(record):
        raise ValueError("Evaluation registration/submission binding mismatch")
    if row.get("judge") != config["judge"]:
        raise ValueError("Evaluation judge policy mismatch")
    return row


def score_slot(slot, output, prepared, config, secret, *, factory_class=JudgeFactory):
    assert_frozen(output, prepared, config)
    release = Path(output) / "test_release"
    record = read_json(release / "submissions" / (digest(slot["slot_id"]) + ".json"))
    evaluation = release / "evaluations" / (digest(slot["slot_id"]) + ".json")
    if evaluation.exists():
        return validate_evaluation(output, slot, record, config)
    marker = release / "evaluation_started" / (digest(slot["slot_id"]) + ".json")
    if marker.exists():
        raise RuntimeError("started-without-result: create a new explicit attempt; never delete the marker")
    cfg = config["judge"]
    previous = history_cost(config["history_metadata"].get(slot["slot_id"], []))
    remaining_tokens = cfg["task_budget_tokens"] - previous["tokens"]
    ledger = BudgetLedger(max_calls=None, max_tokens=max(0, remaining_tokens), max_tool_calls=0,
                          timeout_seconds=cfg["task_timeout_seconds"])
    base = {"version": VERSION, "slot": slot, "registration_hash": config["registration_hash"],
            "record_hash": digest(record), "answer_hash": (record.get("submission") or {}).get("answer_hash"),
            "judge": cfg, "history_budget": previous, "evaluated_at": utc_now()}
    if record.get("status") == "failed":
        row = base | {"official_score": None, "complete": False, "status": "submission_failed",
                      "error_type": record.get("error_type"), "evaluation_budget": ledger.snapshot()}
    elif remaining_tokens <= 0:
        row = base | {"official_score": None, "complete": False, "status": "history_budget_exhausted",
                      "error_type": "BudgetExceeded", "evaluation_budget": ledger.snapshot()}
    else:
        create_json(marker, {"slot": slot, "registration_hash": config["registration_hash"],
                    "record_hash": digest(record), "budget_tokens": remaining_tokens,
                    "history_tokens": previous["tokens"], "started_at": utc_now()})
        try:
            binding = binding_for_slot(slot, prepared)
            evaluator = build_evaluator(binding, factory_class(cfg, secret), ledger)
            raw = evaluator.evaluate(record["submission"]["answer"], ground_truth=slot["task_id"],
                                     private_record=binding["dataset"].private_records[slot["task_id"]])
            raw["submission_answer_hash"] = base["answer_hash"]
            feedback = convert_feedback(raw)
            row = base | {"official_score": feedback.score, "complete": feedback.complete,
                          "status": "complete" if feedback.complete else "evaluation_incomplete",
                          "evaluation": feedback.model_dump(mode="json"), "evaluation_budget": ledger.snapshot()}
        except BaseException as exc:
            row = base | {"official_score": None, "complete": False, "status": "evaluation_failed",
                          "error_type": pinned_error(exc, secret), "evaluation_budget": ledger.snapshot()}
    # Never overwrite a result, including incomplete attempts. A retry creates
    # a separate output directory and explicitly registers this history.
    row["result_hash"] = digest(row)
    assert_frozen(output, prepared, config)
    create_json(evaluation, row)
    return row


def metadata_summary(slots, rows, config):
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    current_tokens = sum(row["evaluation_budget"]["tokens"] for row in rows)
    current_calls = sum(row["evaluation_budget"]["model_calls"] for row in rows)
    prior_tokens = sum(history_cost(config["history_metadata"].get(slot["slot_id"], []))["tokens"] for slot in slots)
    return {"version": VERSION, "registration_hash": config["registration_hash"],
            "slot_count": len(slots), "terminal_count": len(rows),
            "complete_count": sum(row.get("complete") is True for row in rows), "statuses": statuses,
            "current_attempt_tokens": current_tokens, "current_attempt_model_calls": current_calls,
            "history_tokens": prior_tokens, "cumulative_tokens": prior_tokens + current_tokens,
            "scores_exposed": False}


def seal_evaluations(output, slots, config):
    inventory, records = verify_release(Path(output) / "test_release")
    rows = [validate_evaluation(output, slot, records[slot["slot_id"]][1], config) for slot in slots]
    directory = Path(output) / "test_release/evaluations"
    names = {digest(slot["slot_id"]) + ".json" for slot in slots}
    if {path.name for path in directory.glob("*.json")} != names:
        raise ValueError("Evaluation directory does not exactly cover the registered inventory")
    summary = metadata_summary(slots, rows, config)
    create_json(Path(output) / "summary.json", summary)
    body = {"version": FINAL_VERSION, "registration_hash": config["registration_hash"],
            "config_sha256": file_hash(Path(output) / "config.json"),
            "inventory_sha256": file_hash(Path(output) / "test_release/inventory.json"),
            "submission_seal_sha256": file_hash(Path(output) / "test_release/seal.json"),
            "summary_sha256": file_hash(Path(output) / "summary.json"),
            "evaluations": {slot["slot_id"]: file_hash(directory / (digest(slot["slot_id"]) + ".json")) for slot in slots}}
    seal = body | {"final_seal_hash": digest(body)}
    create_json(Path(output) / "final_seal.json", seal)
    verify_final_seal(output, required=True)
    return summary


def verify_final_seal(output, required=True):
    output = Path(output)
    path = output / "final_seal.json"
    if not path.exists():
        if required:
            raise ValueError("Evaluation final seal is missing")
        return False
    seal = read_json(path)
    body = {key: value for key, value in seal.items() if key != "final_seal_hash"}
    if seal.get("version") != FINAL_VERSION or digest(body) != seal.get("final_seal_hash"):
        raise ValueError("Final evaluation seal hash mismatch")
    config = read_json(output / "config.json")
    validate_config(config)
    inventory, records = verify_release(output / "test_release")
    if seal.get("registration_hash") != config["registration_hash"]:
        raise ValueError("Final seal registration mismatch")
    for relative, key in (("config.json", "config_sha256"), ("test_release/inventory.json", "inventory_sha256"),
                          ("test_release/seal.json", "submission_seal_sha256"), ("summary.json", "summary_sha256")):
        if file_hash(output / relative) != seal.get(key):
            raise ValueError("Final seal file hash mismatch")
    expected = {slot["slot_id"] for slot in inventory["slots"]}
    if set(seal.get("evaluations", {})) != expected:
        raise ValueError("Final seal evaluation inventory mismatch")
    directory = output / "test_release/evaluations"
    if {path.name for path in directory.glob("*.json")} != {digest(key) + ".json" for key in expected}:
        raise ValueError("Final seal evaluation coverage mismatch")
    for slot in inventory["slots"]:
        path = directory / (digest(slot["slot_id"]) + ".json")
        if file_hash(path) != seal["evaluations"][slot["slot_id"]]:
            raise ValueError("Final sealed evaluation changed")
        validate_evaluation(output, slot, records[slot["slot_id"]][1], config)
    return True


def status_metadata(output):
    """No dataset loading, credentials, private rubric fields or score output."""
    output = Path(output).resolve()
    config = read_json(output / "config.json")
    validate_config(config)
    inventory, records = verify_release(output / "test_release")
    rows, interrupted = [], 0
    for slot in inventory["slots"]:
        key = digest(slot["slot_id"]) + ".json"
        if (output / "test_release/evaluations" / key).exists():
            rows.append(validate_evaluation(output, slot, records[slot["slot_id"]][1], config))
        elif (output / "test_release/evaluation_started" / key).exists():
            interrupted += 1
    result = metadata_summary(inventory["slots"], rows, config)
    result.update(output=str(output), final_sealed=verify_final_seal(output, required=False),
                  started_without_result=interrupted, untouched=len(inventory["slots"]) - len(rows) - interrupted)
    return result


def run(args):
    if args.mode == "status":
        print(json.dumps(status_metadata(args.output), ensure_ascii=True))
        return 0
    source_bindings = parse_bindings(args.source, "source")
    if not source_bindings:
        raise ValueError("At least one source is required")
    prepared = prepare_sources(source_bindings, parse_bindings(args.data, "data"))
    with output_lock(args.output):
        output, slots, config = initialize_output(args.output, prepared, args)
        assert_frozen(output, prepared, config)
        if args.mode == "check":
            print(json.dumps({"ready": True, "slots": len(slots), "api_calls": 0,
                              "registration_hash": config["registration_hash"]}))
            return 0
        if verify_final_seal(output, required=False):
            print(json.dumps(status_metadata(output)))
            return 0
        # Preflight all markers before scheduling any API calls. The old runner
        # could raise from one future after already scheduling unrelated slots.
        pending = []
        spent = 0
        for slot in slots:
            key = digest(slot["slot_id"]) + ".json"
            evaluation = output / "test_release/evaluations" / key
            if evaluation.exists():
                row = validate_evaluation(output, slot, read_json(output / "test_release/submissions" / key), config)
                spent += row["evaluation_budget"]["tokens"]
            elif (output / "test_release/evaluation_started" / key).exists():
                raise RuntimeError("started-without-result blocks scheduling; use an explicit new attempt")
            else:
                pending.append(slot)
        history_total = sum(history_cost(config["history_metadata"].get(slot["slot_id"], []))["tokens"] for slot in slots)
        # Reserve the entire remaining per-slot envelope before concurrent work;
        # this conservative campaign bound cannot be exceeded by hidden retries.
        reservation = sum(max(0, config["judge"]["task_budget_tokens"] - history_cost(
                config["history_metadata"].get(slot["slot_id"], []))["tokens"]) for slot in pending)
        if history_total + spent + reservation > config["campaign_budget_tokens"]:
            raise ValueError("Campaign budget cannot reserve every pending task including failure history")
        secret = os.environ.get(args.judge_key_env)
        if not secret:
            raise ValueError("Missing judge credential environment variable")
        # Override historical process-wide SystemExit only for this runner; each
        # bounded transport failure is returned to its registered attempt ledger.
        os.environ["MODULAR_AGENT_API_FAILURE_ACTION"] = "raise"
        with ThreadPoolExecutor(max_workers=config["judge"]["process_parallel"]) as pool:
            futures = [pool.submit(score_slot, slot, output, prepared, config, secret) for slot in pending]
            for future in as_completed(futures):
                future.result()
        assert_frozen(output, prepared, config)
        summary = seal_evaluations(output, slots, config)
        print(json.dumps(summary, ensure_ascii=True))
        return 0 if summary["complete_count"] == summary["slot_count"] else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("check", "run", "status"), default="run")
    result.add_argument("--source", action="append", default=[])
    result.add_argument("--data", action="append", default=[])
    result.add_argument("--output", required=True)
    result.add_argument("--judge-model", default="gpt-5.6-sol")
    result.add_argument("--judge-endpoint", default=os.environ.get("RESCORE_JUDGE_ENDPOINT"))
    result.add_argument("--judge-key-env", default="RESCORE_JUDGE_API_KEY")
    result.add_argument("--expected-response-model", action="append", default=[])
    result.add_argument("--judge-max-tokens", type=int, default=4096)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--task-timeout", type=float, default=900)
    result.add_argument("--budget-tokens", type=int, default=500000)
    result.add_argument("--campaign-budget-tokens", type=int, required=False, default=33000000)
    result.add_argument("--transport-attempts", type=int, default=3)
    result.add_argument("--retry-delay", type=float, default=5)
    result.add_argument("--rubric-parallel", type=int, default=1)
    result.add_argument("--process-parallel", type=int, default=1)
    result.add_argument("--prior-attempt", action="append", default=[])
    result.add_argument("--slot", action="append", default=[])
    result.add_argument("--retry-slot", action="append", default=[])
    result.add_argument("--retry-interrupted-slot", action="append", default=[])
    result.add_argument("--rejudge-slot", action="append", default=[])
    return result


def validate_args(args):
    if args.mode == "status":
        return
    if not args.judge_endpoint:
        raise ValueError("An explicit --judge-endpoint or RESCORE_JUDGE_ENDPOINT is required")
    endpoint = urlsplit(args.judge_endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password:
        raise ValueError("Judge endpoint must be an absolute HTTP(S) URL without credentials")
    if not 1 <= args.judge_max_tokens <= 32768:
        raise ValueError("Judge max tokens must be between 1 and 32768")
    if min(args.budget_tokens, args.campaign_budget_tokens) <= 0:
        raise ValueError("Token budgets must be positive")
    if args.timeout <= 0 or args.task_timeout <= 0 or args.retry_delay < 0:
        raise ValueError("Timeouts must be positive and retry delay nonnegative")
    if not 1 <= args.transport_attempts <= 3 or not 1 <= args.rubric_parallel <= 8 or not 1 <= args.process_parallel <= 8:
        raise ValueError("Transport attempts must be 1..3 and parallelism 1..8")
    judge_options(args)


def main(argv=None):
    args = parser().parse_args(argv)
    _configure_logging()
    try:
        validate_args(args)
        return run(args)
    except Exception as exc:
        # No private exception strings or provider request bodies in console.
        print(json.dumps({"rescore_failed": True, "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
