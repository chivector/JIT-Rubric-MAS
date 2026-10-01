"""Register all frozen test conditions, submit without judging, then seal and score."""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import yaml

from jit_mas.checkpoints import CheckpointIntegrityError, snapshot_store
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_profile import validate_execution_profile
from jit_mas.pipeline import code_fingerprint, write_json
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from jit_mas.test_release import TestRelease, remaining_task_budget, score_with_pipeline
from scripts.eval.config import load_dotenv
from scripts.mas_baseline_methods import run_direct
from scripts.run_benchmark_experiment import (
    DEFAULT_SUITE, _method_config, evidence_identity, file_hash, load_registration,
)
from scripts.run_jit_mas import make_pipeline


STATIC_METHODS = {"ours_initial", "direct", "jit_matched", "rubric_fixed"}
EVOLVED_METHODS = {"ours_selected", "ours_terminal", "no_explicit_rubrics",
                   "global_only_planning", "global_only_attribution", "G", "GO"}
MAS_METHODS = {"ours_initial", "ours_selected", "ours_terminal",
               "no_explicit_rubrics", "global_only_planning", "global_only_attribution"}
CONTROL_METHODS = {"jit_matched", "rubric_fixed", "G", "GO"}
SUPPORTED_METHODS = MAS_METHODS | CONTROL_METHODS | {"direct"}
CONDITION_FIELDS = {"condition_id", "benchmark", "method", "run_id", "snapshot_path", "source_checkpoint",
                    "data", "splits", "config", "evidence_dir"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_snapshot(path, *, terminal=False):
    journal = _json(path)
    if journal.get("status") != "complete" or journal.get("selected_position") is None:
        raise ValueError("Source checkpoint selection must be completed and sealed before test registration")
    position = (max(row["position"] for row in journal["checkpoints"]) if terminal
                else journal["selected_position"])
    row = next(row for row in journal["checkpoints"] if row["position"] == position)
    snapshot = ExperienceSnapshot.model_validate(journal["validation_cache"][row["cache_key"]]["snapshot"])
    if digest(snapshot) != row["state_hash"]:
        raise ValueError("Source checkpoint snapshot hash mismatch")
    return snapshot, journal["identity"]["execution"]


def load_snapshot(condition, *, smoke=False):
    location = condition.get("snapshot_path", "initial")
    if location == "initial":
        snapshot = ExperienceSnapshot()
        source = condition.get("source_checkpoint")
    else:
        path = Path(location)
        if path.suffix in (".sqlite", ".db"):
            store = ExperienceStore(path, read_only=True)
            try:
                snapshot = store.snapshot()
            finally:
                store.close()
            source = condition.get("source_checkpoint")
        else:
            document = _json(path)
            if "validation_cache" in document:
                source = str(path)
                snapshot, _ = _checkpoint_snapshot(source, terminal=condition["method"] == "ours_terminal")
            else:
                snapshot = ExperienceSnapshot.model_validate(document)
                source = condition.get("source_checkpoint")
    provenance = {"snapshot_hash": digest(snapshot), "source_checkpoint": source}
    if source:
        expected, identity = _checkpoint_snapshot(source, terminal=condition["method"] == "ours_terminal")
        if digest(expected) != digest(snapshot):
            raise ValueError("Supplied snapshot is not the registered source checkpoint selection")
        provenance.update(source_checkpoint_sha256=file_hash(source), source_identity=identity)
        if not smoke:
            source_benchmark = ("researchrubrics" if condition["benchmark"] == "deepresearch_bench_ii"
                                else condition["benchmark"])
            source_method = condition["method"] if condition["method"] in {
                "no_explicit_rubrics", "global_only_planning", "global_only_attribution"} else "ours_selected"
            if (identity.get("software_test_only") is not False or identity.get("benchmark") != source_benchmark
                    or identity.get("method") != source_method or identity.get("run_id") != condition["run_id"]):
                raise ValueError("Source checkpoint does not match the condition's benchmark, method and run")
            if (identity.get("code_fingerprint") != code_fingerprint()
                    or identity.get("runner_sha256") != file_hash(Path(__file__).with_name("run_benchmark_experiment.py"))):
                raise ValueError("Source checkpoint was evolved under a different frozen code identity")
    if condition["method"] in STATIC_METHODS and (snapshot.version != 0 or snapshot.experiences):
        raise ValueError("Static control must use the empty initial state")
    if not smoke and condition["method"] in EVOLVED_METHODS and source is None:
        raise ValueError("Evolved test conditions require a sealed source checkpoint, not an unproven snapshot")
    if location != "initial":
        provenance["snapshot_file_sha256"] = file_hash(location)
    return snapshot, provenance


def _canonical_condition(raw, base):
    if not isinstance(raw, dict) or set(raw) - CONDITION_FIELDS:
        raise ValueError("Condition has unknown fields; keep credentials in environment variables only")
    condition = {"method": "ours_selected", "run_id": 0, "snapshot_path": "initial", **raw}
    if not isinstance(condition.get("condition_id"), str) or not condition["condition_id"].strip():
        raise ValueError("Every test condition needs a unique condition_id")
    if condition["method"] not in STATIC_METHODS | EVOLVED_METHODS or condition["run_id"] not in (0, 1, 2):
        raise ValueError("Unknown method or run ID")
    for name in ("snapshot_path", "source_checkpoint", "data", "splits", "config", "evidence_dir"):
        if name in condition and not (name == "snapshot_path" and condition[name] == "initial"):
            path = Path(condition[name])
            condition[name] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    return condition


def _condition_material(condition, suite_path, *, smoke=False):
    snapshot, provenance = load_snapshot(condition, smoke=smoke)
    if smoke:
        from jit_mas.offline import fixture_dataset
        _, _, manifest = fixture_dataset()
        config = MASConfig(backend="scripted")
        dataset = None
        data_identity = {"software_test_only": True, "manifest": manifest.model_dump(mode="json")}
    else:
        required = ("benchmark", "data", "splits", "config", "evidence_dir")
        if any(not condition.get(key) for key in required):
            raise ValueError("Every native condition requires benchmark/data/splits/config/evidence_dir")
        if (provenance.get("source_identity") and
                provenance["source_identity"].get("suite_file_sha256") != file_hash(suite_path)):
            raise ValueError("Source checkpoint belongs to a different registered suite or execution profile")
        config = MASConfig.model_validate(yaml.safe_load(os.path.expandvars(Path(condition["config"]).read_text(encoding="utf-8"))))
        if config.backend != "native_jit":
            raise ValueError("Native test campaigns cannot use synthetic transports")
        config.unsafe_local = True
        if provenance.get("source_identity"):
            source_config = dict(provenance["source_identity"].get("configuration", {}))
            target_config = config.model_dump(mode="json")
            source_config.pop("seed", None)
            target_config.pop("seed", None)
            if source_config != target_config:
                raise ValueError("Test execution configuration differs from the frozen source trajectory")
        method = condition["method"]
        if method in MAS_METHODS:
            _method_config(config, "ours_selected" if method in {"ours_initial", "ours_terminal"} else method)
        args = argparse.Namespace(suite=suite_path, **condition)
        suite, specification, dataset, split, manifest = load_registration(args, available_tools=config.available_tools)
        validate_execution_profile(config, suite)
        if method not in specification["methods"]:
            raise ValueError("Condition method is not registered on this benchmark")
        data_identity = {"dataset_sha256": dataset.dataset_sha256,
                         "split_file_sha256": file_hash(condition["splits"]),
                         "manifest_sha256": split["manifest_sha256"],
                         "configuration_file_sha256": file_hash(condition["config"]),
                         "evidence": evidence_identity(condition["evidence_dir"])}
    identity = {"software_test_only": smoke, "condition": condition, "snapshot": provenance,
                "data": data_identity, "configuration": config.model_dump(mode="json"),
                "code_fingerprint": code_fingerprint(), "test_runner_sha256": file_hash(__file__),
                "baseline_runner_sha256": file_hash(Path(__file__).with_name("mas_baseline_methods.py")),
                "suite_file_sha256": None if smoke else file_hash(suite_path)}
    identity["scorer_identity"] = digest({"code": identity["code_fingerprint"],
        "configuration": identity["configuration"], "data": data_identity,
        "runner": identity["test_runner_sha256"]})
    return snapshot, config, dataset, manifest, identity


def _check_full_campaign(conditions, suite):
    expected = {(benchmark, method, run_id)
                for benchmark, spec in suite["benchmarks"].items() for method in spec["methods"]
                for run_id in ([0] if method in STATIC_METHODS else [0, 1, 2])}
    actual = [(row["benchmark"], row["method"], row["run_id"]) for row in conditions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Formal campaign must register every suite benchmark/method/run exactly once; partial campaigns are not allowed")


def register(campaign, conditions_path=None, *, suite_path=DEFAULT_SUITE, smoke=False):
    directory = Path(campaign).resolve()
    if smoke:
        conditions = [{"condition_id": "smoke-initial", "benchmark": "synthetic-software-fixture",
                       "method": "ours_initial", "run_id": 0, "snapshot_path": "initial"}]
    else:
        if conditions_path is None:
            raise ValueError("register requires --conditions with the complete frozen condition inventory")
        raw = _json(conditions_path)
        if not isinstance(raw, list) or not raw:
            raise ValueError("Conditions file must be a nonempty JSON list")
        conditions = [_canonical_condition(row, Path(conditions_path).resolve().parent) for row in raw]
        _check_full_campaign(conditions, _json(suite_path))
    if len({row["condition_id"] for row in conditions}) != len(conditions):
        raise ValueError("Duplicate condition_id")
    if len({digest(row["condition_id"])[:12] for row in conditions}) != len(conditions):
        raise ValueError("Condition directory hash collision")
    document = {"schema_version": "registered-sealed-test-v1", "software_test_only": smoke,
                "suite_path": str(Path(suite_path).resolve()), "conditions": {}, "artifact_repeats": 1 if smoke else 3}
    inventory = []
    for condition in conditions:
        snapshot, config, dataset, manifest, identity = _condition_material(condition, suite_path, smoke=smoke)
        condition_hash = digest(identity)
        document["conditions"][condition["condition_id"]] = {"descriptor": condition, "identity": identity,
                                                             "condition_hash": condition_hash}
        for task_id in manifest.test:
            for repeat in range(document["artifact_repeats"]):
                slot = {"condition_id": condition["condition_id"], "condition_hash": condition_hash,
                        "benchmark": condition["benchmark"], "method": condition["method"], "run_id": condition["run_id"],
                        "task_id": task_id, "repeat": repeat, "experience_hash": digest(snapshot),
                        "scorer_identity": identity["scorer_identity"]}
                slot["slot_id"] = digest(slot)
                inventory.append(slot)
    if len({row["slot_id"][:12] for row in inventory}) != len(inventory):
        raise ValueError("Submission directory hash collision")
    if not smoke:
        states = {(row["descriptor"]["benchmark"], row["descriptor"]["method"], row["descriptor"]["run_id"]):
                  row["identity"]["snapshot"]["snapshot_hash"] for row in document["conditions"].values()}
        for run_id in (0, 1, 2):
            if states[("researchrubrics", "ours_selected", run_id)] != states[("deepresearch_bench_ii", "ours_selected", run_id)]:
                raise ValueError("External transfer must use the identical ResearchRubrics-selected state for each run")
            if any(states[("researchrubrics", method, run_id)] != states[("researchrubrics", "ours_selected", run_id)] for method in ("G", "GO")):
                raise ValueError("G/GO mechanism conditions must use the same selected full-method states")
    document["registration_hash"] = digest(document)
    path = directory / "conditions.json"
    if path.exists() and _json(path) != document:
        raise ValueError("Cannot change a test campaign's frozen conditions")
    if not path.exists():
        write_json(path, document)
    release = TestRelease(directory, inventory)
    return {"conditions": len(conditions), "slots": len(inventory), "inventory_hash": release.inventory["inventory_hash"],
            "registration_hash": document["registration_hash"], "software_test_only": smoke}


def _registration(campaign):
    document = _json(Path(campaign) / "conditions.json")
    if digest({key: value for key, value in document.items() if key != "registration_hash"}) != document["registration_hash"]:
        raise ValueError("Frozen test registration hash mismatch")
    return document


@dataclass
class TestEnvironment:
    pipeline: object
    snapshot: ExperienceSnapshot
    store: ExperienceStore
    condition: dict
    identity: dict
    output_dir: Path
    smoke: bool
    suite_path: Path

    __test__ = False

    def close(self):
        self.store.close()

    def assert_frozen(self):
        if (code_fingerprint() != self.identity["code_fingerprint"]
                or file_hash(__file__) != self.identity["test_runner_sha256"]
                or file_hash(Path(__file__).with_name("mas_baseline_methods.py")) != self.identity["baseline_runner_sha256"]):
            raise CheckpointIntegrityError("Code or scorer changed during a frozen test campaign")
        if self.pipeline.config.model_dump(mode="json") != self.identity["configuration"]:
            raise CheckpointIntegrityError("In-memory test configuration changed after registration")
        if not self.smoke:
            try:
                _, _, _, _, current = _condition_material(self.condition, self.suite_path, smoke=False)
            except (OSError, ValueError) as exc:
                raise CheckpointIntegrityError("Frozen test inputs became unavailable or invalid") from exc
            if digest(current) != digest(self.identity):
                raise CheckpointIntegrityError("Frozen test configuration, data, split, evidence or source state changed")


def build_environment(campaign, condition_id, *, unsafe_local=False):
    directory = Path(campaign).resolve()
    document = _registration(directory)
    if condition_id not in document["conditions"]:
        raise ValueError("Unknown registered condition")
    registered = document["conditions"][condition_id]
    condition = registered["descriptor"]
    smoke = document["software_test_only"]
    if not smoke and not unsafe_local:
        raise ValueError("Native generated-code campaign requires explicit --unsafe-local")
    snapshot, config, dataset, manifest, identity = _condition_material(condition, document["suite_path"], smoke=smoke)
    if digest(identity) != registered["condition_hash"]:
        raise ValueError("Condition or scorer identity changed after registration; do not replace or rescore test results")
    frozen = snapshot_store(snapshot, directory / "states" / f"{digest(snapshot)}.sqlite")
    try:
        # Full hashes remain in receipts; short collision-checked names avoid MAX_PATH.
        output = directory / "runs" / digest(condition_id)[:12]
        kwargs = {} if smoke else {"data": condition["data"], "splits": condition["splits"],
                                  "benchmark": condition["benchmark"], "evidence_dir": condition["evidence_dir"]}
        pipeline = make_pipeline(config, frozen, output, **kwargs)
        return TestEnvironment(pipeline, snapshot, frozen, condition, identity, output, smoke, Path(document["suite_path"]))
    except BaseException:
        frozen.close()
        raise


def submit_condition(environment, slot, method):
    """Dispatch hook: unsupported controls must never fall back to the full method."""
    pipeline, snapshot = environment.pipeline, environment.snapshot
    slot_dir = environment.output_dir / slot["slot_id"][:12]
    if method in CONTROL_METHODS:
        from jit_mas.experiment_methods import submit_method
        return submit_method(pipeline, slot["task_id"], snapshot, method=method, repeat=slot["repeat"],
                             output_dir=slot_dir, shared_dir=environment.output_dir.parents[1] / "shared_rstar")
    if method == "direct":
        config = pipeline.config
        outcome = run_direct(pipeline.tasks[slot["task_id"]], None, pipeline.models,
            pipeline.evaluator_factory, slot_dir / "direct", {
                "max_calls": config.max_model_calls, "max_tokens": config.max_total_tokens,
                "output_tokens": min(8192, config.models["exec"].max_tokens)}, defer_evaluation=True)
        outcome["experience_hash"] = digest(snapshot)
        write_json(Path(outcome["run_dir"]) / "complete.json", outcome)
        return outcome
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"No registered submitter for method {method}; cannot substitute another method")
    pipeline.output_dir = slot_dir
    return pipeline.run_task(slot["task_id"], snapshot, mode="evaluate", repeat=slot["repeat"],
                             attribution=False, defer_evaluation=True, resume=False)


def _slot_path(campaign, slot_id, kind="submissions"):
    return Path(campaign) / kind / f"{digest(slot_id)}.json"


def _recover_submission(environment, slot):
    root = environment.output_dir / slot["slot_id"][:12]
    candidates = list(root.glob("*/complete.json"))
    if len(candidates) == 1:
        outcome = _json(candidates[0])
        if outcome.get("status") == "submitted_unscored" and outcome.get("task_id") == slot["task_id"]:
            return outcome
    return None


def _semantic_execution_key(environment, slot):
    config = dict(environment.identity["configuration"])
    config.pop("seed", None)  # Source-order seed is not forwarded to serving model calls.
    method = slot["method"]
    semantic_method = "ours" if method in {"ours_initial", "ours_selected", "ours_terminal"} else method
    data = environment.identity["data"]
    return digest({"benchmark": slot["benchmark"], "task_id": slot["task_id"], "repeat": slot["repeat"],
                   "state": slot["experience_hash"], "method": semantic_method, "configuration": config,
                   "data": {key: value for key, value in data.items() if key != "configuration_file_sha256"},
                   "code": environment.identity["code_fingerprint"],
                   "runner": environment.identity["test_runner_sha256"],
                   "baseline_runner": environment.identity["baseline_runner_sha256"]})


def _zero_budget():
    return {"model_calls": 0, "tokens": 0, "reserved_tokens": 0, "tool_calls": 0,
            "wall_seconds": 0, "cost": None, "by_stage": {}, "records": []}


def _cache_generation(campaign, key, slot_id):
    record = _json(_slot_path(campaign, slot_id))
    cache = {"key": key, "source_slot_id": slot_id, "record_hash": digest(record)}
    path = Path(campaign) / "artifact_cache" / f"{key}.json"
    if path.exists():
        if _json(path) != cache:
            raise CheckpointIntegrityError("Exact-state artifact cache source changed")
    else:
        write_json(path, cache)


def _reuse_generation(campaign, release, key, slot):
    path = Path(campaign) / "artifact_cache" / f"{key}.json"
    if not path.exists():
        # Recover the tiny record/cache atomicity gap without generating a new answer.
        for prior in release.slots.values():
            if any(prior[name] != slot[name] for name in ("benchmark", "task_id", "repeat", "experience_hash")):
                continue
            prior_path = _slot_path(campaign, prior["slot_id"])
            if prior_path.exists():
                record = _json(prior_path)
                outcome = record.get("outcome", {})
                if (record["status"] == "submitted" and outcome.get("semantic_execution_key") == key
                        and not outcome.get("generation_reused")):
                    _cache_generation(campaign, key, prior["slot_id"])
                    break
    if not path.exists():
        return None
    cache = _json(path)
    record = _json(_slot_path(campaign, cache["source_slot_id"]))
    if cache["key"] != key or digest(record) != cache["record_hash"]:
        raise CheckpointIntegrityError("Exact-state artifact cache integrity failure")
    outcome = copy.deepcopy(record["outcome"])
    if outcome.get("semantic_execution_key") != key or record["status"] != "submitted":
        raise CheckpointIntegrityError("Cached artifact does not match the execution identity")
    outcome.update(reused_from={"slot_id": cache["source_slot_id"], "record_hash": cache["record_hash"]},
                   generation_reused=True,
                   task_generation_budget=outcome.get("task_generation_budget", outcome["budget"]),
                   logical_deployment_budget=outcome.get("logical_deployment_budget", outcome["budget"]),
                   budget=_zero_budget())
    return outcome


def _score_or_reuse(campaign, environment, record):
    environment.assert_frozen()
    config = dict(environment.identity["configuration"])
    config.pop("seed", None)
    outcome = record["outcome"]
    if outcome.get("reused_from") and "task_generation_budget" not in outcome:
        raise CheckpointIntegrityError("Reused artifact is missing its original per-task generation budget")
    generation = outcome.get("task_generation_budget", outcome["budget"])
    key = digest({"execution": record["outcome"]["semantic_execution_key"],
                  "answer": record["submission"]["answer_hash"], "configuration": config,
                  "task_generation_usage": {name: generation.get(name, 0)
                                            for name in ("model_calls", "tokens", "tool_calls")},
                  "remaining_task_budget": remaining_task_budget(environment.pipeline.config, outcome),
                  "code": environment.identity["code_fingerprint"]})
    cache_path = Path(campaign) / "score_cache" / f"{key}.json"
    if cache_path.exists():
        cache = _json(cache_path)
        if digest({name: value for name, value in cache.items() if name != "sha256"}) != cache["sha256"]:
            raise CheckpointIntegrityError("Cached evaluator result changed")
        result = copy.deepcopy(cache["result"])
        result.update(slot=record["slot"], generation_budget=record["outcome"]["budget"],
                      evaluation_reused_from=cache["source_slot_id"],
                      logical_evaluation_budget=result["evaluation_budget"], evaluation_budget=_zero_budget())
        return result
    result = score_with_pipeline(environment.pipeline, record)
    cache = {"key": key, "source_slot_id": record["slot"]["slot_id"], "result": result}
    cache["sha256"] = digest(cache)
    write_json(cache_path, cache)
    return result


def submit(campaign, condition_id, *, unsafe_local=False, resolve_interrupted=None):
    release = TestRelease(campaign)
    if (Path(campaign) / "seal.json").exists():
        raise ValueError("Test campaign is already sealed")
    environment = build_environment(campaign, condition_id, unsafe_local=unsafe_local)
    completed = recovered = failed = 0
    try:
        method = environment.condition["method"]
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"No registered submitter for {method}; no test slots were consumed")
        if method in CONTROL_METHODS:
            from jit_mas.experiment_methods import submit_method
            if not callable(submit_method):
                raise ValueError("Registered control submitter is unavailable")
        for slot in release.slots.values():
            if slot["condition_id"] != condition_id:
                continue
            environment.assert_frozen()
            if _slot_path(campaign, slot["slot_id"]).exists():
                completed += 1
                continue
            started = _slot_path(campaign, slot["slot_id"], "submission_started")
            execution_key = _semantic_execution_key(environment, slot)
            if started.exists():
                outcome = _recover_submission(environment, slot)
                if outcome is not None:
                    outcome["semantic_execution_key"] = execution_key
                    release.record(slot["slot_id"], outcome)
                    _cache_generation(campaign, execution_key, slot["slot_id"])
                    recovered += 1
                    continue
                if not resolve_interrupted:
                    raise RuntimeError("Interrupted submission has no completed artifact. Inspect the stopped request; "
                                       "use --resolve-interrupted REASON to record failure, never regenerate it.")
                write_json(_slot_path(campaign, slot["slot_id"], "interruption_resolution"),
                           {"reason": resolve_interrupted, "resolved_at": utc_now()})
                release.record_failure(slot["slot_id"], error_type="InterruptedSubmission", budget={"usage_unknown": True})
                failed += 1
                continue
            reused = _reuse_generation(campaign, release, execution_key, slot)
            if reused is not None:
                release.record(slot["slot_id"], reused)
                recovered += 1
                continue
            write_json(started, {"slot_hash": digest(slot), "started_at": utc_now()})
            try:
                outcome = submit_condition(environment, slot, method)
                outcome["semantic_execution_key"] = execution_key
                if digest(environment.store.snapshot()) != slot["experience_hash"]:
                    raise ValueError("Read-only test state changed during submission")
                release.record(slot["slot_id"], outcome)
                _cache_generation(campaign, execution_key, slot["slot_id"])
                completed += 1
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                budget = getattr(exc, "jit_mas_run_failure", {}).get("budget")
                if budget is None:
                    budgets = list((environment.output_dir / slot["slot_id"][:12]).glob("*/budget.json"))
                    budget = _json(budgets[0]) if len(budgets) == 1 else {"usage_unknown": True}
                release.record_failure(slot["slot_id"], error_type=type(exc).__name__, budget=budget)
                failed += 1
        return {"condition_id": condition_id, "submitted_or_preexisting": completed, "recovered": recovered, "failed": failed}
    finally:
        environment.close()


def score(campaign, condition_id, *, unsafe_local=False):
    release = TestRelease(campaign)
    if not (Path(campaign) / "seal.json").exists():
        raise ValueError("Cannot score until all registered conditions are submitted and sealed")
    environment = build_environment(campaign, condition_id, unsafe_local=unsafe_local)
    results = []
    try:
        for slot in release.slots.values():
            if slot["condition_id"] != condition_id:
                continue
            if slot["scorer_identity"] != environment.identity["scorer_identity"]:
                raise ValueError("Registered scorer identity mismatch")
            environment.assert_frozen()
            results.append(release.evaluate(slot["slot_id"], lambda record: _score_or_reuse(campaign, environment, record)))
        return {"condition_id": condition_id, "slots": len(results),
                "complete": sum(row["complete"] for row in results),
                "scores": [{"task_id": row["slot"]["task_id"], "repeat": row["slot"]["repeat"],
                            "score": row["official_score"], "complete": row["complete"]} for row in results]}
    finally:
        environment.close()


def status(campaign):
    release = TestRelease(campaign)
    generation_calls = generation_tokens = evaluation_calls = evaluation_tokens = unknown = 0
    submitted = failed = evaluated = 0
    for slot in release.slots.values():
        path = _slot_path(campaign, slot["slot_id"])
        if path.exists():
            row = _json(path)
            submitted += row["status"] == "submitted"
            failed += row["status"] == "failed"
            budget = row["outcome"]["budget"] if row["status"] == "submitted" else row.get("budget", {})
            generation_calls += budget.get("model_calls", 0)
            generation_tokens += budget.get("tokens", 0)
            unknown += bool(budget.get("usage_unknown") or not budget)
        path = _slot_path(campaign, slot["slot_id"], "evaluations")
        if path.exists():
            row = _json(path)
            evaluated += 1
            budget = row.get("evaluation_budget", {})
            evaluation_calls += budget.get("model_calls", 0)
            evaluation_tokens += budget.get("tokens", 0)
            unknown += row.get("status") == "evaluation_failed" and not budget
    shared_calls = shared_tokens = 0
    seen = set()
    for path in (Path(campaign) / "shared_rstar").glob("*.json"):
        cache = _json(path)
        if "graph" in cache and "budget" in cache and "sha256" in cache:
            if digest({key: value for key, value in cache.items() if key != "sha256"}) != cache["sha256"]:
                raise CheckpointIntegrityError("Shared rubric cache integrity failure")
            key = cache["sha256"]
        elif path.name.endswith(".failure.json") and "budget" in cache:
            key = digest(cache)
        else:
            continue
        if key not in seen:
            seen.add(key)
            shared_calls += cache["budget"].get("model_calls", 0)
            shared_tokens += cache["budget"].get("tokens", 0)
    return {"slots": len(release.slots), "submitted": submitted, "failed": failed, "evaluated": evaluated,
            "sealed": (Path(campaign) / "seal.json").exists(),
            "generation_model_calls": generation_calls, "generation_tokens": generation_tokens,
            "evaluation_model_calls": evaluation_calls, "evaluation_tokens": evaluation_tokens,
            "shared_preparation_model_calls": shared_calls, "shared_preparation_tokens": shared_tokens,
            "total_model_calls_recorded": generation_calls + evaluation_calls + shared_calls,
            "total_tokens_recorded": generation_tokens + evaluation_tokens + shared_tokens,
            "attempts_with_unknown_budget": unknown, "cost": None}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["register", "submit", "seal", "score", "status", "smoke"])
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--conditions", help="Complete suite-wide JSON condition list; relative paths resolve beside this file")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--condition", help="Explicit registered condition_id for submit or score")
    parser.add_argument("--unsafe-local", action="store_true")
    parser.add_argument("--resolve-interrupted", help="Audit reason after inspecting a stopped interrupted request; records failure without regeneration")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        with redirect_stdout(sys.stderr):
            if args.mode == "register":
                report = register(args.campaign, args.conditions, suite_path=args.suite)
            elif args.mode == "submit":
                if not args.condition:
                    raise ValueError("submit requires --condition")
                report = submit(args.campaign, args.condition, unsafe_local=args.unsafe_local,
                                resolve_interrupted=args.resolve_interrupted)
            elif args.mode == "seal":
                report = TestRelease(args.campaign).seal()
            elif args.mode == "score":
                if not args.condition:
                    raise ValueError("score requires --condition")
                report = score(args.campaign, args.condition, unsafe_local=args.unsafe_local)
            elif args.mode == "status":
                report = status(args.campaign)
            else:
                register(args.campaign, smoke=True)
                submit(args.campaign, "smoke-initial")
                TestRelease(args.campaign).seal()
                score(args.campaign, "smoke-initial")
                report = {**status(args.campaign), "software_test_only": True, "paid_requests": 0}
                if report["submitted"] != 2 or report["evaluated"] != 2 or report["failed"]:
                    raise RuntimeError("Synthetic staged test smoke did not complete both artifacts and evaluations")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.exit(2, f"Registered benchmark test: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
