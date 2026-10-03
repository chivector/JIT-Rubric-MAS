"""Small, sealed three-method diagnostics on fixed source tasks.

Source EVO is the default; exposed TEST requires an explicit partition flag.
This command does not run or modify the formal EVO/VAL/TEST protocol. Each
iteration freezes its inputs, consumes every generation slot once, seals all
answers (including failures), then applies the same benchmark self-judge.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

from jit_mas.benchmarks import load_benchmark, normalized_question_hash
from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetLedger
from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.config import MASConfig, ModelConfig, NativeModels
from jit_mas.evidence import apply_evidence_pack, evidence_pack_path, load_evidence_pack
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_methods import _native_jit
from jit_mas.independent_protocol import normalize_score
from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.pipeline import MASPipeline, code_fingerprint, convert_feedback, write_json
from jit_mas.schemas import ExperienceSnapshot, SplitManifest, digest, utc_now
from jit_mas.test_release import TestRelease, remaining_task_budget
from scripts.benchmark_jit_mas_live import SafeTransport
from scripts.mas_baseline_methods import JudgeEnvelopeModel, run_direct


VERSION = "source-evo-development-pilot-v1"
ARMS = ("direct", "ours", "native_jit")
SOURCES = ("researchrubrics", "deepsearchqa", "deepresearch_bench_ii", "writingbench", "ifeval", "ifbench")
INSTRUCTION_SOURCES = {"ifeval", "ifbench"}
PARTITIONS = ("source_evo", "exposed_test")
KEY_ENV = "JIT_BENCHMARK_API_KEY"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / "paper/experiments/joint_task_splits_v5.json"
CLOSED_BOOK_CONSTRAINT = (
    "Development diagnostic: model general knowledge is allowed. No browsing or "
    "external tools are available. Do not claim to have searched, opened unavailable "
    "attachments, measured, or independently verified unavailable sources. State "
    "material factual uncertainty; preserve the requested genre and final artifact."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_identity():
    return {"runtime": code_fingerprint(), "runner": file_hash(__file__)}


def safe_error(error):
    value = str(error)
    secret = os.environ.get(KEY_ENV)
    return value.replace(secret, "[REDACTED]") if secret else value


def common_config(endpoint, model, expected_response_model=None, *, request_timeout=180):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or not model.strip()):
        raise ValueError("Explicit credential-free HTTPS endpoint and model required")
    common = dict(model=model, endpoint=endpoint, key_env=KEY_ENV,
                  timeout=request_timeout, temperature=0, thinking=None, reasoning_effort=None,
                  expected_response_model=expected_response_model,
                  context_window=131072, context_margin=2048, context_policy="oldest_turns")
    return MASConfig(
        backend="native_jit", execution_mode="iterative_shared_ledger", unsafe_local=True,
        models={role: ModelConfig(**{**common, "context_policy": "reject" if role == "judge" else "oldest_turns"},
                                  max_tokens=8192 if role in {"exec", "judge"} else 16000)
                for role in ("meta", "global", "local", "exec", "judge")},
        max_agents=3, max_parallel=2, team_max_calls=None, max_model_calls=None,
        max_total_tokens=2_000_000, max_tool_calls=None, max_repairs=2, candidates=1,
        max_inflight_requests=2, execution_timeout=900, task_timeout=900,
        local_planning=True, local_rounds=1, local_attribution=True,
        persistent_experience=True, evolving_agent_pool=True, explicit_rubrics=True,
        available_tools=[])


def select_sources(splits, counts, partition="source_evo"):
    """Select a frozen prefix; TEST exposure never changes the formal manifest."""
    if (splits.get("version") != "jit-compose-joint-subset-v5"
            or splits.get("manifest_sha256") != digest({k: v for k, v in splits.items()
                                                        if k != "manifest_sha256"})):
        raise ValueError("Invalid joint v5 split identity")
    if partition not in PARTITIONS:
        raise ValueError("Unknown development partition")
    limit = 2 if partition == "exposed_test" else 20
    if not counts or set(counts) - set(SOURCES) or any(type(n) is not int or not 1 <= n <= limit for n in counts.values()):
        if partition == "exposed_test":
            raise ValueError("Select 1..2 explicitly exposed TEST tasks per supported source")
        raise ValueError("Select 1..20 EVO tasks per supported source")
    index_rows = splits["task_index"]
    index = {row["task_id"]: row for row in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("Duplicate frozen task identity")
    if partition == "exposed_test":
        for source in counts:
            membership = splits["memberships"][source]["test"]
            observed = [row["task_id"] for row in index_rows
                        if row["benchmark"] == source and row["partition"] == "test"]
            if len(membership) != len(set(membership)) or set(observed) != set(membership):
                raise ValueError("Frozen task-index TEST rows differ from TEST membership")
        consumed, selected = Counter(), []
        for row in index_rows:
            source = row["benchmark"]
            if source not in counts or row["partition"] != "test" or consumed[source] >= counts[source]:
                continue
            selected.append({"task_id": row["task_id"], "benchmark": source,
                             "source_row": row["source_row"], "question_sha256": row["question_sha256"],
                             "formal_partition": "test", "original_partition": row["partition"],
                             "parent_partition": row.get("parent_partition", "test"),
                             "original_test_slice": row.get("test_slice"),
                             "former_development": row.get("former_development", False)})
            consumed[source] += 1
        if dict(consumed) != counts:
            raise ValueError("Insufficient frozen TEST positions")
        return selected
    schedules = [row for row in splits["joint_evolution_schedule"] if row["run_id"] == 0]
    if len(schedules) != 1:
        raise ValueError("Exactly one frozen run-0 schedule required")
    sequence = schedules[0]["task_ids"]
    if len(sequence) != len(set(sequence)):
        raise ValueError("Duplicate source positions")
    for source in counts:
        expected = set(splits["memberships"][source]["evolution"])
        observed = {key for key in sequence if index[key]["benchmark"] == source}
        if observed != expected:
            raise ValueError("Frozen source schedule differs from EVO membership")
    consumed, selected = Counter(), []
    for key in sequence:
        row = index[key]
        source = row["benchmark"]
        if source in counts and consumed[source] < counts[source]:
            if row["partition"] != "evolution" or key not in splits["memberships"][source]["evolution"]:
                raise ValueError("Development selection must contain source EVO tasks only")
            selected.append({"task_id": key, "benchmark": source, "source_row": row["source_row"],
                             "question_sha256": row["question_sha256"], "formal_partition": "evolution"})
            consumed[source] += 1
    if dict(consumed) != counts:
        raise ValueError("Insufficient source EVO positions")
    return selected


def selected_dataset(source, data_path, selected):
    """Normalize selected source rows only; unselected JSONL records are not parsed."""
    rows_needed = {row["source_row"] for row in selected}
    location = Path(data_path)
    rows = []
    if location.suffix.lower() == ".csv":
        with location.open(encoding="utf-8-sig", newline="") as handle:
            for position, row in enumerate(csv.DictReader(handle), 1):
                if position in rows_needed:
                    rows.append(row)
                if position >= max(rows_needed):
                    break
    else:
        with location.open(encoding="utf-8-sig") as handle:
            position = 0
            for line in handle:
                if not line.strip():
                    continue
                position += 1
                if position in rows_needed:
                    rows.append(json.loads(line))
                if position >= max(rows_needed):
                    break
    if len(rows) != len(rows_needed):
        raise ValueError("Selected source rows are missing")
    # The existing adapter owns normalization and private projection. The temporary
    # file contains selected rows only and is removed before any model call.
    with tempfile.TemporaryDirectory(prefix="jit-development-selected-") as directory:
        projection = Path(directory) / "selected.json"
        write_json(projection, rows)
        dataset = load_benchmark(source, projection)
    expected = {row["task_id"]: row for row in selected}
    if set(dataset.tasks) != set(expected):
        raise ValueError("Selected source row identity differs from the frozen manifest")
    for task_id, task in dataset.tasks.items():
        if normalized_question_hash(task.question) != expected[task_id]["question_sha256"]:
            raise ValueError("Selected public question differs from the frozen manifest")
        if task.tools or task.attachments:
            raise ValueError("This initial diagnostic supports text-only tasks with no external tools")
    return dataset


def source_material(source, identity, selected, knowledge_mode):
    dataset = selected_dataset(source, identity["data_path"], selected)
    tasks = {}
    for task_id, task in dataset.tasks.items():
        if knowledge_mode == "closed_book":
            tasks[task_id] = task.model_copy(update={"constraints": [*task.constraints, CLOSED_BOOK_CONSTRAINT]})
        else:
            pack = load_evidence_pack(evidence_pack_path(identity["evidence_path"], task_id), task)
            tasks[task_id] = apply_evidence_pack(task, pack)
    return dataset, tasks


def source_checker(source, identity):
    if source not in INSTRUCTION_SOURCES:
        return None
    record = identity.get("checker")
    if not record:
        raise ValueError("Instruction sources require a pinned local author checker")
    checker = PinnedInstructionChecker(record["source_root"], source)
    if checker.identity != record["identity"] or digest(checker.identity) != record["identity_sha256"]:
        raise CheckpointIntegrityError("Frozen author checker identity changed")
    checker.assert_frozen()
    return checker


def validate_development_config(config):
    """Keep registration credential-free and forbid unregistered candidate choice."""
    if config.candidates != 1:
        raise ValueError("Development diagnostics require exactly one harness candidate")
    for spec in config.models.values():
        parsed = urlsplit(spec.endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or not spec.model.strip()):
            raise ValueError("Every configured role requires a credential-free HTTPS endpoint and explicit model")
        if spec.key_env != KEY_ENV:
            raise ValueError("Every configured role must use the designated credential environment name")


def register(campaign, *, splits_path, data_paths, counts, config, iteration,
             knowledge_mode="closed_book", evidence_paths=None, previous_campaign=None,
             partition="source_evo", checker_paths=None):
    campaign = Path(campaign).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", iteration):
        raise ValueError("Use a simple explicit iteration identifier")
    if knowledge_mode not in {"closed_book", "frozen_evidence"}:
        raise ValueError("Unknown shared knowledge mode")
    validate_development_config(config)
    splits = read_json(splits_path)
    selected = select_sources(splits, counts, partition)
    if set(data_paths) != set(counts):
        raise ValueError("Supply exactly one pinned data file for each selected source")
    evidence_paths = evidence_paths or {}
    checker_paths = checker_paths or {}
    if set(checker_paths) != set(counts) & INSTRUCTION_SOURCES:
        raise ValueError("Supply one pinned checker source for each instruction benchmark only")
    if knowledge_mode == "frozen_evidence" and set(evidence_paths) != set(counts):
        raise ValueError("Frozen evidence mode requires a shared pack for every source")
    if knowledge_mode == "closed_book" and evidence_paths:
        raise ValueError("Closed-book registration cannot supply evidence packs")
    identities = {}
    for source, path in data_paths.items():
        location = Path(path).resolve()
        actual = file_hash(location)
        if actual != splits["benchmarks"][source]["dataset_sha256"]:
            raise ValueError("Dataset bytes differ from the frozen source release")
        identity = {"data_path": str(location), "dataset_sha256": actual}
        selected_source = [row for row in selected if row["benchmark"] == source]
        if knowledge_mode == "frozen_evidence":
            directory = Path(evidence_paths[source]).resolve()
            identity.update(evidence_path=str(directory), evidence_files={
                evidence_pack_path(directory, row["task_id"]).name:
                file_hash(evidence_pack_path(directory, row["task_id"])) for row in selected_source})
        dataset, tasks = source_material(source, identity, selected_source, knowledge_mode)
        checker = None
        if source in INSTRUCTION_SOURCES:
            checker = PinnedInstructionChecker(checker_paths[source], source)
            checker.assert_frozen()
            identity["checker"] = {"source_root": str(Path(checker_paths[source]).resolve()),
                                   "identity": checker.identity, "identity_sha256": digest(checker.identity)}
        identity["source_public_projection_sha256"] = digest({
            key: dataset.tasks[key].model_dump(mode="json") for key in sorted(dataset.tasks)})
        identity["public_tasks"] = {key: {"canonical_hash": digest(dataset.tasks[key]),
                                              "actor_hash": digest(task), "task": task.model_dump(mode="json"),
                                              "bounds": [dataset.lower_bounds[key], dataset.upper_bounds[key]]}
                                    for key, task in tasks.items()}
        identity["evaluator_version"] = dataset.evaluator(None, judge_id=config.models["judge"].model,
            judge_api_base=config.models["judge"].endpoint, judge_max_tokens=config.models["judge"].max_tokens,
            judge_timeout=config.models["judge"].timeout, checker=checker).evaluator_version
        identity["primary_metric"] = {"ifeval": "prompt_level_strict_accuracy",
                                      "ifbench": "prompt_level_loose_accuracy",
                                      "deepresearch_bench_ii": "deepresearch_bench_ii_satisfaction"}.get(source, "native_benchmark_score")
        identities[source] = identity
    previous = None
    if previous_campaign:
        previous_path = Path(previous_campaign).resolve() / "registration.json"
        previous = {"campaign": str(previous_path.parent), "registration_sha256": file_hash(previous_path),
                    "iteration": read_json(previous_path)["iteration"], "development_exposure": True}
    slots = [{"slot_id": f"{arm}:{row['task_id']}", "task_id": row["task_id"],
              "benchmark": row["benchmark"], "method": arm, "repeat": 0}
             for row in selected for arm in ARMS]
    scoring_order = sorted([row["slot_id"] for row in slots], key=lambda value: digest({"seed": 20261004, "slot": value}))
    exposed_test = partition == "exposed_test"
    instruction_only = set(counts) <= INSTRUCTION_SOURCES
    judge_policy = ("pinned author instruction checker; no model judge" if instruction_only else
                    "pinned author checker for instruction sources; same-model self-judge for other sources; diagnostic, not independent evaluation"
                    if set(counts) & INSTRUCTION_SOURCES else
                    "same-model self-judge; diagnostic, not independent evaluation")
    body = {"version": VERSION, "scope": "exposed_TEST_development_only" if exposed_test else "source_EVO_development_only", "iteration": iteration,
            "partition": partition, "original_partition": "test" if exposed_test else "evolution",
            "development_exposure": True, "formal_protocol_result": False,
            "split_path": str(Path(splits_path).resolve()), "split_sha256": file_hash(splits_path),
            "split_manifest_sha256": splits["manifest_sha256"],
            "selection_policy": "first_N_per_source_frozen_task_index_TEST_order" if exposed_test else "first_N_per_source_frozen_run0_order",
            "exposure_policy": ("User explicitly authorized on 2026-10-04: use selected TEST tasks to guide prompts and methods. Exposed tasks are contaminated development data; never formal TEST results. Formal v5 manifest remains immutable."
                                if exposed_test else "Only selected source EVO tasks are exposed; formal VAL/TEST remain unselected."),
            "contamination_list": [{"task_id": row["task_id"], "benchmark": row["benchmark"],
                                    "source_row": row["source_row"], "original_partition": "test",
                                    "question_sha256": row["question_sha256"],
                                    "exposed_material": "selected public prompt and coordinator-only checker/evaluation constraints",
                                    "exposure_use": "development inspection and future method iteration; excluded from clean confirmation"}
                                   for row in selected] if exposed_test else [],
            "selected": selected, "sources": identities, "arms": list(ARMS), "slots": slots,
            "configuration": config.model_dump(mode="json"), "code": code_identity(),
            "knowledge_mode": knowledge_mode, "protocol_deviation_from_formal_shared_evidence": knowledge_mode == "closed_book",
            "judge_policy": judge_policy,
            "generation_policy": "initial empty experience state; no updates; consume once, no replacement or resampling",
            "budget_policy": "identical per-task cap and role output limits; realized compute is unequal",
            "time_policy": "900 active seconds shared by generation and deferred judgment; exclude between-slot scheduling and request-queue idle time; each judge request is bounded by both remaining task time and its request timeout",
            "native_interface_max_steps": config.max_total_tokens if config.max_model_calls is None else config.max_model_calls,
            "scoring_policy": "all generations sealed first; method labels excluded from judge messages",
            "scoring_shuffle_seed": 20261004, "scoring_order": scoring_order, "previous_iteration": previous,
            "registered_at": utc_now()}
    body["registration_hash"] = digest(body)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError("Use a new empty campaign directory for every development iteration")
    campaign.mkdir(parents=True, exist_ok=True)
    write_json(campaign / "registration.json", body)
    TestRelease(campaign / "release", slots)
    return {"iteration": iteration, "selected": selected, "generation_slots": len(slots),
            "registration_hash": body["registration_hash"], "campaign": str(campaign)}


def assert_identity(campaign):
    registration = read_json(Path(campaign) / "registration.json")
    if (registration.get("version") != VERSION or registration.get("registration_hash") !=
            digest({key: value for key, value in registration.items() if key != "registration_hash"})):
        raise CheckpointIntegrityError("Development registration changed")
    if registration["code"] != code_identity() or file_hash(registration["split_path"]) != registration["split_sha256"]:
        raise CheckpointIntegrityError("Frozen code or split changed; register a new iteration")
    for identity in registration["sources"].values():
        if file_hash(identity["data_path"]) != identity["dataset_sha256"]:
            raise CheckpointIntegrityError("Frozen source data changed")
        for name, expected in identity.get("evidence_files", {}).items():
            if file_hash(Path(identity["evidence_path"]) / name) != expected:
                raise CheckpointIntegrityError("Frozen public evidence changed")
    for source, identity in registration["sources"].items():
        source_checker(source, identity)
    release = TestRelease(Path(campaign) / "release")
    if release.inventory["slots"] != registration["slots"]:
        raise CheckpointIntegrityError("Frozen generation inventory changed")
    return registration


class SafeModels:
    def __init__(self, config):
        self.provider = NativeModels(config)
        self.created = []

    def create(self, role, agent_id, ledger, stage):
        secret = os.environ[KEY_ENV]
        model = self.provider.create(role, agent_id, ledger, stage)
        # Reject reflected credentials before MeteredModel appends response text
        # to its calls, which upstream JIT persists even after a failed call.
        model.model = SafeTransport(model.model, secret)
        model = SafeTransport(model, secret)
        model = JudgeEnvelopeModel(model) if role == "judge" else model
        self.created.append(model)
        return model

    def close(self):
        # The native provider returns one client per created leaf. Unwrap only
        # the documented wrappers; never serialize client/credential objects.
        for model in self.created:
            while hasattr(model, "model") and not hasattr(model, "client"):
                model = model.model
            client = getattr(model, "client", None)
            if client is not None:
                client.close()


def make_environment(campaign, registration, source, output):
    config = MASConfig.model_validate(registration["configuration"])
    selected = [row for row in registration["selected"] if row["benchmark"] == source]
    dataset, tasks = source_material(source, registration["sources"][source], selected, registration["knowledge_mode"])
    checker = source_checker(source, registration["sources"][source])
    projection_hash = digest({key: dataset.tasks[key].model_dump(mode="json") for key in sorted(dataset.tasks)})
    if projection_hash != registration["sources"][source].get("source_public_projection_sha256", projection_hash):
        raise CheckpointIntegrityError("Frozen source public projection changed")
    for key, task in tasks.items():
        recorded = registration["sources"][source]["public_tasks"][key]
        if digest(task) != recorded["actor_hash"] or digest(dataset.tasks[key]) != recorded["canonical_hash"]:
            raise CheckpointIntegrityError("Frozen public task changed")
    # Explicit development-only mapping satisfies old evaluate interfaces. It
    # does not alter the immutable formal EVO/VAL/TEST membership.
    manifest = SplitManifest(test=[row["task_id"] for row in selected])
    state_path = Path(campaign) / "initial_experience.sqlite"
    if not state_path.exists():
        initial = ExperienceStore(state_path)
        initial.close()
    store = ExperienceStore(state_path, read_only=True)
    if digest(store.snapshot()) != digest(ExperienceSnapshot()):
        store.close()
        raise CheckpointIntegrityError("Initial diagnostic requires an empty experience and Agent Pool state")
    models = SafeModels(config)

    def evaluator(judge):
        spec = config.models["judge"]
        result = dataset.evaluator(judge, judge_id=spec.model, judge_api_base=spec.endpoint,
                                   judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout, checker=checker)
        if result.evaluator_version != registration["sources"][source]["evaluator_version"]:
            raise CheckpointIntegrityError("Frozen evaluator changed")
        return result

    def synth(meta):
        spec = config.models["meta"]
        return JITHarnessSynthesizer(backend="native_jit", meta_model=meta,
            meta_config={"model_id": spec.model, "api_base": spec.endpoint, "api_key": "INJECTED",
                         "max_tokens": spec.max_tokens}, candidates=config.candidates,
            max_repairs=config.max_repairs, selector_model=meta, tools={})

    pipeline = MASPipeline(config, models, evaluator, synth, tasks, dataset.private_records,
        manifest, store, output, tools={},
        knowledge_policy="model_general_knowledge_allowed" if registration["knowledge_mode"] == "closed_book" else None)
    pipeline.development_local_checker = checker is not None
    return pipeline


def unknown_budget():
    return {"model_calls": 0, "tokens": 0, "tool_calls": 0, "reserved_tokens": 0,
            "usage_unknown": True, "cost": None, "records": []}


def recover_outcome(slot, output):
    """Recover a saved answer after interruption without rerunning any actor."""
    candidates = list(Path(output).rglob("submission.json"))
    if len(candidates) != 1:
        return None
    directory = candidates[0].parent
    if not (directory / "budget.json").is_file() or not (directory / "execution.json").is_file():
        return None
    submission, execution, budget = (read_json(directory / name) for name in
                                      ("submission.json", "execution.json", "budget.json"))
    if (digest(submission.get("answer")) != submission.get("answer_hash")
            or execution.get("answer") != submission["answer"]
            or execution.get("terminated_reason") != "final_answer"):
        raise CheckpointIntegrityError("Interrupted submission differs from saved execution")
    if budget.get("usage_unknown") or budget.get("reserved_tokens", 0):
        return None
    complete = directory / "complete.json"
    if complete.exists():
        outcome = read_json(complete)
        if (outcome.get("task_id") != slot["task_id"] or outcome.get("answer_hash") != submission["answer_hash"]
                or outcome.get("status") != "submitted_unscored"):
            raise CheckpointIntegrityError("Interrupted completed artifact identity changed")
        return outcome
    return {"task_id": slot["task_id"], "status": "submitted_unscored", "evaluation": None,
            "proposals": [], "budget": budget, "answer_hash": submission["answer_hash"],
            "run_dir": str(directory.resolve()), "recovered_saved_submission": True}


def generate(pipeline, slot, output):
    task = pipeline.tasks[slot["task_id"]]
    if slot["method"] == "ours":
        return pipeline.run_task(task.task_id, pipeline.store.snapshot(), mode="evaluate", repeat=0,
                                 attribution=False, resume=False, defer_evaluation=True)
    if slot["method"] == "direct":
        return run_direct(task, {}, pipeline.models, pipeline.evaluator_factory, output / "direct", {
            "max_calls": pipeline.config.max_model_calls, "max_tokens": pipeline.config.max_total_tokens,
            "max_tool_calls": pipeline.config.max_tool_calls, "output_tokens": pipeline.config.models["exec"].max_tokens,
            "timeout_seconds": pipeline.config.task_timeout}, defer_evaluation=True)
    if slot["method"] != "native_jit":
        raise ValueError("Unknown development method")
    ledger = BudgetLedger(pipeline.config.max_model_calls, pipeline.config.max_total_tokens,
                          pipeline.config.max_tool_calls, timeout_seconds=pipeline.config.task_timeout)
    directory = output / "native"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        result = _native_jit(pipeline, task, ledger, directory)
        write_json(directory / "execution.json", result.full_dict())
        if result.terminated_reason != "final_answer" or not isinstance(result.answer, str) or not result.answer.strip():
            raise RuntimeError("Native JIT did not submit a final artifact")
        submission = {"answer": result.answer, "answer_hash": digest(result.answer), "submitted_at": utc_now()}
        write_json(directory / "submission.json", submission)
        outcome = {"task_id": task.task_id, "method": "native_jit", "status": "submitted_unscored",
                   "evaluation": None, "proposals": [], "budget": ledger.snapshot(),
                   "answer_hash": submission["answer_hash"], "run_dir": str(directory.resolve())}
        write_json(directory / "complete.json", outcome)
        return outcome
    except BaseException as exc:
        exc.jit_mas_run_failure = {"task_id": task.task_id, "budget": ledger.snapshot(),
                                   "error_type": type(exc).__name__, "error": safe_error(exc)}
        write_json(directory / "failure.json", exc.jit_mas_run_failure)
        raise
    finally:
        write_json(directory / "budget.json", ledger.snapshot())


def progress(slot, stage, status_value, budget):
    print(json.dumps({"stage": stage, "method": slot["method"], "task_id": slot["task_id"],
                      "status": status_value, "model_calls": budget.get("model_calls"),
                      "tokens": budget.get("tokens"), "usage_unknown": budget.get("usage_unknown", False)}), flush=True)


def score_saved(pipeline, record):
    """Only answer and native task/reference payload reach the judge, never arm labels."""
    submission, task_id = record["submission"], record["slot"]["task_id"]
    used = record["outcome"].get("task_generation_budget", record["outcome"]["budget"])
    elapsed = used.get("wall_seconds")
    queue_idle = used.get("request_queue_idle_seconds", 0)
    if (type(elapsed) not in (int, float) or type(queue_idle) not in (int, float)
            or elapsed < 0 or queue_idle < 0 or queue_idle > elapsed + 1e-6):
        raise CheckpointIntegrityError("Saved generation lacks valid elapsed-time accounting")
    remaining_seconds = (None if pipeline.config.task_timeout is None else
                         max(0.0, pipeline.config.task_timeout - max(0.0, elapsed - queue_idle)))
    ledger = BudgetLedger(**remaining_task_budget(pipeline.config, record["outcome"]),
                          timeout_seconds=remaining_seconds)
    try:
        local_checker = getattr(pipeline, "development_local_checker", False)
        if local_checker and ledger.remaining_seconds() == 0:
            raise TimeoutError("Shared task time exhausted before local scoring")
        judge = None if local_checker else pipeline.models.create("judge", "anonymous_submission", ledger, "evaluation")
        evaluator = pipeline.evaluator_factory(judge)
        raw = evaluator.evaluate(submission["answer"], ground_truth=task_id,
                                 private_record=pipeline.private_records[task_id])
        if local_checker and ledger.remaining_seconds() == 0:
            raise TimeoutError("Shared task time exhausted during local scoring")
        raw["submission_answer_hash"] = submission["answer_hash"]
        feedback = convert_feedback(raw)
        return {"slot": record["slot"], "official_score": feedback.score, "complete": feedback.complete,
                "evaluation": feedback.model_dump(mode="json"), "evaluation_budget": ledger.snapshot(),
                "generation_budget": record["outcome"]["budget"], "answer_hash": submission["answer_hash"],
                "generation_active_seconds": max(0.0, elapsed - queue_idle),
                "judge_time_budget_seconds": remaining_seconds,
                "evaluated_at": utc_now()}
    except BaseException as exc:
        exc.evaluation_budget = ledger.snapshot()
        raise


def _latest_budget(output):
    budgets = sorted(Path(output).rglob("budget.json"), key=lambda path: path.stat().st_mtime_ns)
    return read_json(budgets[-1]) if budgets else unknown_budget()


def run(campaign, *, unsafe_local=False, submit_only=False, environment_factory=make_environment):
    campaign = Path(campaign).resolve()
    registration = assert_identity(campaign)
    if not unsafe_local:
        raise PermissionError("Development native execution requires --unsafe-local")
    if not os.environ.get(KEY_ENV):
        raise ValueError("Credential environment variable is absent")
    if os.environ.get("JIT_MAS_MODEL_ATTEMPTS", "1") != "1":
        raise ValueError("This diagnostic consumes one transport attempt per model call")
    release = TestRelease(campaign / "release")
    if not (release.directory / "seal.json").exists():
        for slot in registration["slots"]:
            assert_identity(campaign)
            terminal = release._path(slot["slot_id"])
            if terminal.exists():
                continue
            output = campaign / "generation" / digest(slot["slot_id"])
            started = output / "started.json"
            if started.exists():
                if read_json(started)["registration_hash"] != registration["registration_hash"]:
                    raise CheckpointIntegrityError("Interrupted slot belongs to another registration")
                outcome = recover_outcome(slot, output)
                if outcome is None:
                    release.record_failure(slot["slot_id"], error_type="interrupted_generation_no_resample",
                                           budget=_latest_budget(output))
                else:
                    release.record(slot["slot_id"], outcome)
                continue
            output.mkdir(parents=True, exist_ok=True)
            write_json(started, {"registration_hash": registration["registration_hash"], "slot": slot, "started_at": utc_now()})
            pipeline = None
            try:
                pipeline = environment_factory(campaign, registration, slot["benchmark"], output)
                with redirect_stdout(sys.stderr):
                    outcome = generate(pipeline, slot, output)
                release.record(slot["slot_id"], outcome)
                progress(slot, "generation", "submitted", outcome["budget"])
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                failure = getattr(exc, "jit_mas_run_failure", {})
                budget = failure.get("budget", _latest_budget(output))
                write_json(output / "failure_summary.json", {"error_type": type(exc).__name__, "error": safe_error(exc), "budget": budget})
                release.record_failure(slot["slot_id"], error_type=type(exc).__name__, budget=budget)
                progress(slot, "generation", "failed", budget)
            finally:
                if pipeline is not None:
                    pipeline.models.close()
                    pipeline.store.close()
        assert_identity(campaign)
        release.seal()
    if not submit_only:
        by_slot = {slot["slot_id"]: slot for slot in registration["slots"]}
        for slot_id in registration["scoring_order"]:
            assert_identity(campaign)
            slot = by_slot[slot_id]
            evaluated = release._path(slot_id, "evaluations")
            if evaluated.exists():
                continue
            if release._path(slot_id, "evaluation_started").exists():
                # A crash may have consumed judge requests. Preserve that uncertainty
                # instead of asking the judge again for another result.
                write_json(evaluated, {"slot": slot, "complete": False, "official_score": None,
                    "status": "interrupted_evaluation_no_resample", "evaluation_budget": unknown_budget()})
                continue
            pipeline = None
            try:
                pipeline = environment_factory(campaign, registration, slot["benchmark"], campaign / "scoring" / digest(slot_id))
                with redirect_stdout(sys.stderr):
                    result = release.evaluate(slot_id, lambda record: score_saved(pipeline, record))
                progress(slot, "evaluation", "completed" if result["complete"] else "incomplete", result.get("evaluation_budget", {}))
            finally:
                if pipeline is not None:
                    pipeline.models.close()
                    pipeline.store.close()
    result = summary(campaign)
    write_json(campaign / "summary.json", result)
    return result


def summary(campaign):
    campaign = Path(campaign).resolve()
    registration = assert_identity(campaign)
    release = TestRelease(campaign / "release")
    sealed = (release.directory / "seal.json").exists()
    if sealed:
        release.seal()
    rows, costs = [], {arm: {"generation_model_calls": 0, "generation_tokens": 0,
                            "evaluation_model_calls": 0, "evaluation_tokens": 0,
                            "generation_active_seconds": 0, "evaluation_active_seconds": 0,
                            "request_queue_idle_seconds": 0,
                            "input_tokens": 0, "output_tokens": 0, "estimated_attempts": 0,
                            "unknown_slots": 0, "cost": None} for arm in ARMS}
    for slot in registration["slots"]:
        submission_path, evaluation_path = release._path(slot["slot_id"]), release._path(slot["slot_id"], "evaluations")
        record = read_json(submission_path) if submission_path.exists() else None
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
        if record is not None and record.get("slot") != slot:
            raise CheckpointIntegrityError("Saved generation slot identity changed")
        if evaluation is not None and (not sealed or evaluation.get("slot") != slot):
            raise CheckpointIntegrityError("Evaluation is not bound to the sealed slot")
        if evaluation is not None:
            if type(evaluation.get("complete")) is not bool:
                raise CheckpointIntegrityError("Saved evaluation completeness must be an explicit boolean")
            if evaluation["complete"] and (record is None or record.get("status") != "submitted"):
                raise CheckpointIntegrityError("A failed generation cannot have a completed observed evaluation")
        if record is not None and record["status"] == "submitted":
            saved = record["submission"]
            if digest(saved["answer"]) != saved["answer_hash"]:
                raise CheckpointIntegrityError("Sealed answer changed")
            source = Path(record["outcome"]["run_dir"]).resolve()
            if not source.is_relative_to(campaign / "generation"):
                raise CheckpointIntegrityError("Saved actor evidence escaped the campaign")
            if read_json(source / "submission.json") != saved or read_json(source / "execution.json")["answer"] != saved["answer"]:
                raise CheckpointIntegrityError("Sealed answer differs from actor evidence")
            if evaluation is not None and evaluation.get("complete"):
                feedback = evaluation.get("evaluation", {})
                if (evaluation.get("answer_hash") != saved["answer_hash"]
                        or feedback.get("raw", {}).get("submission_answer_hash") != saved["answer_hash"]
                        or feedback.get("score") != evaluation.get("official_score")
                        or feedback.get("evaluator_version") != registration["sources"][slot["benchmark"]]["evaluator_version"]):
                    raise CheckpointIntegrityError("Judge result is not bound to the saved answer and evaluator")
        complete = bool(evaluation and evaluation.get("complete"))
        score = evaluation.get("official_score") if complete else None
        native_score = score
        if complete and slot["benchmark"] == "writingbench":
            native_score = evaluation.get("evaluation", {}).get("raw", {}).get("native_mean")
            if (type(native_score) not in (int, float) or not 1 <= native_score <= 10
                    or abs((native_score - 1) / 9 - score) > 1e-12):
                raise CheckpointIntegrityError("WritingBench native and normalized scores disagree")
        if complete and slot["benchmark"] == "deepsearchqa":
            native_score = evaluation.get("evaluation", {}).get("raw", {}).get("f1")
            if native_score != score:
                raise CheckpointIntegrityError("DeepSearchQA native F1 disagrees with its saved score")
        bounds = registration["sources"][slot["benchmark"]]["public_tasks"][slot["task_id"]]["bounds"]
        normalized = normalize_score(score, bounds) if complete else None
        row = {**slot, "generation_status": record["status"] if record else "pending",
               "evaluation_status": "completed" if complete else (evaluation.get("status", "incomplete") if evaluation else "pending"),
               "score": score, "native_score": native_score, "normalized_score": normalized, "complete": complete,
               "generation_error_type": record.get("error_type") if record else None,
               "evaluation_error_type": evaluation.get("error_type") if evaluation else None}
        if slot["benchmark"] in INSTRUCTION_SOURCES:
            raw = evaluation.get("evaluation", {}).get("raw", {}) if evaluation else {}
            row.update(primary_metric=registration["sources"][slot["benchmark"]]["primary_metric"],
                       instruction_accuracy=raw.get("instruction_accuracy") if complete else None,
                       pass_count=raw.get("pass_count") if complete else None,
                       check_count=raw.get("check_count") if complete else None)
        rows.append(row)
        generation_budget = record.get("outcome", {}).get("budget", record.get("budget", {})) if record else {}
        evaluation_budget = evaluation.get("evaluation_budget", {}) if evaluation else {}
        cost = costs[slot["method"]]
        for stage, budget in (("generation", generation_budget), ("evaluation", evaluation_budget)):
            cost[stage + "_model_calls"] += budget.get("model_calls", 0)
            cost[stage + "_tokens"] += budget.get("tokens", 0)
            queue_idle = budget.get("request_queue_idle_seconds", 0)
            cost[stage + "_active_seconds"] += max(0, budget.get("wall_seconds", 0) - queue_idle)
            cost["request_queue_idle_seconds"] += queue_idle
            cost["unknown_slots"] += int(bool(budget.get("usage_unknown") or budget.get("reserved_tokens", 0)))
            for attempt in budget.get("records", []):
                if attempt.get("kind") == "model":
                    cost["input_tokens"] += attempt.get("input_tokens", 0)
                    cost["output_tokens"] += attempt.get("output_tokens", 0)
                    cost["estimated_attempts"] += int(bool(attempt.get("estimated")))
    arm_results = {}
    for arm in ARMS:
        selected_rows = [row for row in rows if row["method"] == arm]
        by_source = {}
        for source in registration["sources"]:
            selected_source = [row for row in selected_rows if row["benchmark"] == source]
            complete_source = [row for row in selected_source if row["complete"]]
            by_source[source] = {"slots": len(selected_source), "completed": len(complete_source),
                "native_mean_all_complete": sum(row["native_score"] for row in complete_source) / len(selected_source)
                    if len(complete_source) == len(selected_source) else None,
                "normalized_mean_failure_zero": sum(row["normalized_score"] for row in complete_source) / len(selected_source)}
        arm_results[arm] = {"slots": len(selected_rows), "submitted": sum(row["generation_status"] == "submitted" for row in selected_rows),
                           "completed": sum(row["complete"] for row in selected_rows), "benchmarks": by_source,
                           "macro_normalized_failure_zero": sum(row["normalized_mean_failure_zero"] for row in by_source.values()) / len(by_source)}
    complete_tasks = [row["task_id"] for row in registration["selected"]
                      if all(next(item for item in rows if item["task_id"] == row["task_id"] and item["method"] == arm)["complete"] for arm in ARMS)]
    return {"version": VERSION, "iteration": registration["iteration"], "registration_hash": registration["registration_hash"],
            "scope": registration["scope"], "sealed": sealed, "comparison_complete": all(row["complete"] for row in rows),
            "partition": registration.get("partition", "source_evo"),
            "formal_protocol_result": False, "exposure_policy": registration.get("exposure_policy"),
            "contamination_list": registration.get("contamination_list", []),
            "knowledge_mode": registration["knowledge_mode"], "judge_policy": registration["judge_policy"],
            "protocol_deviation_from_formal_shared_evidence": registration["protocol_deviation_from_formal_shared_evidence"],
            "rows": rows, "arms": arm_results, "complete_paired_task_ids": complete_tasks, "costs": costs,
            "interpretation": ("Exposed TEST development diagnostic; contaminated tasks cannot support clean TEST confirmation. "
                               if registration.get("partition") == "exposed_test" else "Source EVO development diagnostic. ")
                              + registration["judge_policy"] + "; unequal realized compute. Missing scores remain null; failure-zero values are conservative accounting, not observed scores. No formal TEST, significance, independent-judge or superiority claim."}


def key_values(values, *, integers=False):
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or key not in SOURCES or key in result:
            raise ValueError("Use unique supported SOURCE=VALUE arguments")
        result[key] = int(item) if integers else item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("register", "run", "summary", "status"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--iteration", default="v0")
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--partition", choices=PARTITIONS, default="source_evo",
                        help="Explicit exposed_test enables contaminated TEST development; default source_evo")
    parser.add_argument("--data", action="append", default=[], help="SOURCE=PINNED_FILE")
    parser.add_argument("--count", action="append", default=[], help="SOURCE=N; fixed prefix; exposed TEST maximum 2/source")
    parser.add_argument("--checker-source", action="append", default=[], help="ifeval/ifbench=PINNED_AUTHOR_CHECKER_ROOT")
    parser.add_argument("--evidence", action="append", default=[], help="SOURCE=SHARED_PACK_DIRECTORY")
    parser.add_argument("--knowledge-mode", choices=("closed_book", "frozen_evidence"), default="closed_book")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--config", help="Explicit JSON/YAML common MASConfig, containing environment names only")
    parser.add_argument("--expected-response-model")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--previous-campaign")
    parser.add_argument("--unsafe-local", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args(argv)
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        if args.mode == "register":
            if args.config:
                import yaml

                config = MASConfig.model_validate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
                if any(spec.key_env != KEY_ENV for spec in config.models.values()):
                    raise ValueError("Every role must use the designated credential environment name")
            elif args.endpoint and args.model:
                config = common_config(args.endpoint, args.model, args.expected_response_model, request_timeout=args.request_timeout)
            else:
                raise ValueError("register requires --endpoint and --model")
            result = register(args.campaign, splits_path=args.splits, data_paths=key_values(args.data),
                counts=key_values(args.count, integers=True),
                config=config,
                iteration=args.iteration, knowledge_mode=args.knowledge_mode,
                evidence_paths=key_values(args.evidence), previous_campaign=args.previous_campaign,
                partition=args.partition, checker_paths=key_values(args.checker_source))
        elif args.mode == "run":
            result = run(args.campaign, unsafe_local=args.unsafe_local, submit_only=args.submit_only)
        else:
            result = summary(args.campaign)
            if args.mode == "status":
                result = {key: result[key] for key in ("iteration", "sealed", "comparison_complete", "arms", "costs")}
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": safe_error(exc)}))
        return 1
    finally:
        logging.disable(previous_logging)


if __name__ == "__main__":
    raise SystemExit(main())
