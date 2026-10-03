"""Resume scoring for an already sealed RR submission inventory.

This command never generates answers and never runs EVO/VAL.  It copies the
sealed v35 inventory into a new directory, reuses complete evaluations, and
allows only previously incomplete evaluations for submitted answers to be
retried.  The six generation failures remain missing forever in the recovery
directory.  Credentials are read from the frozen config's environment names.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _copy_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _verify_runtime(runtime: Path):
    manifest = _read(runtime / "source_manifest.json")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("frozen runtime source manifest is empty")
    listed = set()
    for relative, expected in manifest.items():
        path = runtime / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen runtime file changed or missing: {relative}")
        listed.add(Path(relative).as_posix())
    for path in runtime.rglob("*"):
        if not path.is_file() or path.name == "source_manifest.json":
            continue
        relative = path.relative_to(runtime)
        if any(part in {"__pycache__", "workspaces"} for part in relative.parts):
            continue
        if relative.as_posix() not in listed:
            raise ValueError(f"unlisted frozen runtime file: {relative.as_posix()}")


def _source_paths(source: Path):
    release = source / "test_release"
    return release, release / "inventory.json", release / "seal.json"


def _inventory(release: Path):
    from jit_mas.test_release import TestRelease

    registered = TestRelease(release)
    seal = _read(release / "seal.json")
    if seal.get("inventory_hash") != registered.inventory["inventory_hash"]:
        raise ValueError("sealed inventory hash does not match inventory.json")
    for slot_id, expected in seal.get("submission_hashes", {}).items():
        record_path = release / "submissions" / (registered._path(slot_id).stem + ".json")
        if not record_path.is_file():
            raise ValueError(f"sealed submission is missing: {slot_id}")
        from jit_mas.schemas import digest
        if digest(_read(record_path)) != expected:
            raise ValueError(f"sealed submission changed: {slot_id}")
    if set(seal.get("submission_hashes", {})) != set(registered.slots):
        raise ValueError("seal does not cover exactly the registered slots")
    return registered, seal


def _classify_evaluations(release, evaluation_dir: Path):
    rows = {}
    for path in sorted(evaluation_dir.glob("*.json")):
        row = _read(path)
        slot = row.get("slot", {})
        slot_id = slot.get("slot_id")
        if slot_id not in release.slots:
            raise ValueError(f"evaluation is outside the sealed inventory: {path.name}")
        if slot_id in rows:
            raise ValueError(f"duplicate evaluation for {slot_id}")
        if slot.get("task_id") != release.slots[slot_id]["task_id"]:
            raise ValueError(f"evaluation task binding changed: {slot_id}")
        rows[slot_id] = (path, row)
    return rows


def audit_source(source: Path, runtime: Path, expected_judge_model="deepseek-v4-flash-vision"):
    release, inventory_path, seal_path = _source_paths(source)
    if not source.is_dir() or not inventory_path.is_file() or not seal_path.is_file():
        raise ValueError("source v35 output or sealed test release is missing")
    registered, seal = _inventory(release)
    submissions = release / "submissions"
    if len(registered.slots) != 33:
        raise ValueError("RR recovery requires the frozen 33-slot inventory")
    if len(list(submissions.glob("*.json"))) != 33:
        raise ValueError("sealed inventory must contain exactly 33 submission records")
    rows = _classify_evaluations(registered, release / "evaluations")
    complete, incomplete, generation_failed = [], [], []
    for slot_id, slot in registered.slots.items():
        record = _read(registered._path(slot_id))
        if record.get("status") == "failed":
            generation_failed.append(slot_id)
            continue
        if record.get("status") != "submitted":
            raise ValueError(f"unexpected submission status for {slot_id}")
        evaluation = rows.get(slot_id, (None, None))[1]
        if evaluation is None:
            raise ValueError(f"submitted slot has no historical evaluation: {slot_id}")
        if evaluation.get("complete") is True and evaluation.get("official_score") is not None:
            complete.append(slot_id)
        else:
            incomplete.append(slot_id)
    if len(complete) != 22 or len(incomplete) != 5 or len(generation_failed) != 6:
        raise ValueError(f"expected v35 inventory split 22/5/6, got {len(complete)}/{len(incomplete)}/{len(generation_failed)}")
    if not runtime.is_dir() or not (runtime / "source_manifest.json").is_file():
        raise ValueError("frozen v35 runtime source manifest is missing")
    _verify_runtime(runtime)
    evaluator_versions = set()
    for _, evaluation_pair in rows.items():
        _, evaluation = evaluation_pair
        nested = evaluation.get("evaluation") or {}
        if nested.get("evaluator_version"):
            evaluator_versions.add(nested["evaluator_version"])
        actual_model = (nested.get("raw") or {}).get("judge_model")
        if actual_model and actual_model != expected_judge_model:
            raise ValueError("historical evaluation used a different judge model")
    if len(evaluator_versions) > 1:
        raise ValueError("historical evaluations use multiple evaluator versions")
    return {"release": release, "registered": registered, "seal": seal,
            "evaluations": rows, "complete": complete, "incomplete": incomplete,
            "generation_failed": generation_failed}


def _copy_release(audit, output: Path):
    source_release = audit["release"]
    target_release = output / "test_release"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovery output: {output}")
    target_release.mkdir(parents=True)
    _copy_file(source_release / "inventory.json", target_release / "inventory.json")
    _copy_file(source_release / "seal.json", target_release / "seal.json")
    for directory in ("submissions",):
        for path in (source_release / directory).glob("*.json"):
            _copy_file(path, target_release / directory / path.name)
    archive = output / "archive" / "v35"
    for directory in ("submissions", "evaluations", "evaluation_started"):
        source_dir = source_release / directory
        if source_dir.is_dir():
            for path in source_dir.glob("*.json"):
                _copy_file(path, archive / directory / path.name)
    for slot_id, (path, row) in audit["evaluations"].items():
        if slot_id in audit["complete"] or slot_id in audit["generation_failed"]:
            _copy_file(path, target_release / "evaluations" / path.name)
    _write(output / "recovery_manifest.json", {
        "version": "rr-sealed-scoring-recovery-v1",
        "source": str(audit["release"].parent),
        "copied_complete_evaluations": audit["complete"],
        "retry_evaluations": audit["incomplete"],
        "generation_failures": audit["generation_failed"],
        "attempts": [],
        "all_task_mean": None,
    })
    return target_release


def _frozen_args(root: Path, source: Path, runtime: Path, metadata: dict):
    config_doc = metadata["config"]
    models = config_doc["models"]
    policy = metadata["structured_policy"]
    split = source / "rr_split.json"
    return SimpleNamespace(
        data=Path(metadata["data"]), joint_manifest=root / "paper/experiments/joint_task_splits_v5.json",
        split_path=str(split), exec_model=metadata["execution_model"],
        exec_endpoint=metadata["execution_endpoint"], exec_max_tokens=models["exec"]["max_tokens"],
        timeout=models["exec"]["timeout"], judge_model=metadata["judge_model"],
        judge_endpoint=metadata["judge_endpoint"], judge_max_tokens=models["judge"]["max_tokens"],
        judge_attempts=metadata["judge_attempts"], judge_parallel=metadata["judge_parallel"],
        max_inflight_requests=metadata["process_request_cap"], closed_book=True,
        structured_output=metadata["structured_output"],
        planning_string_max_length=policy["planning_string_max_length"],
        planning_communication_max_length=policy["planning_communication_max_length"],
        planning_array_max_items=policy["planning_array_max_items"],
        frozen_identity=metadata["frozen_identity"], arm="ours", evidence_dir=None,
        reuse_evaluations_from=[], root=str(root), runtime=str(runtime),
    )


def _prior_budget(row):
    budget = row.get("evaluation_budget") or {}
    if budget.get("usage_unknown") or budget.get("reserved_tokens", 0):
        raise ValueError("historical evaluation budget is unsettled or unknown")
    values = {}
    for key in ("model_calls", "tokens", "tool_calls"):
        value = budget.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid historical evaluation budget: {key}")
        values[key] = value
    return values


def _score(root: Path, source: Path, runtime: Path, output: Path, audit):
    metadata = _read(source / "pilot_metadata.json")
    launch = _read(root / "outputs/rr_deepseek_method_launch_20261002_v35/launch.json")
    if Path(launch["method_output"]).resolve() != source.resolve():
        raise ValueError("v35 launch does not bind the supplied source output")
    if Path(launch["runtime"]).resolve() != runtime.resolve():
        raise ValueError("v35 launch does not bind the supplied frozen runtime")
    manifest = _read(output / "test_release" / "inventory.json")
    args = _frozen_args(root, source, runtime, metadata)
    sys.path.insert(0, str(runtime))
    from jit_mas.experience import ExperienceStore
    from jit_mas.pipeline import code_fingerprint
    from jit_mas.schemas import digest, utc_now
    from scripts.run_rr_two_arm_pilot import _assert_identity, _config, _manifest, _pipeline
    from jit_mas.test_release import TestRelease, score_with_pipeline
    if code_fingerprint() != metadata["frozen_identity"]["code"]:
        raise ValueError("frozen runtime code fingerprint mismatch")
    config = _config(args)
    if digest(config) != metadata["frozen_identity"]["config"]:
        raise ValueError("frozen configuration mismatch")
    config.check_native()
    _assert_identity(args)
    source_store = source / "ours" / "experience.sqlite"
    if not source_store.is_file():
        raise ValueError("selected experience store is missing")
    store = ExperienceStore(source_store, read_only=True)
    try:
        pipeline = _pipeline(config, store, output / "scoring", args, _manifest(args.joint_manifest), None)
        release = TestRelease(output / "test_release")
        evaluations = _classify_evaluations(release, output / "test_release" / "evaluations")
        scores = []
        prior_accounting = {}
        stopped = None
        for slot_id in audit["incomplete"]:
            old = audit["evaluations"][slot_id][1]
            prior = _prior_budget(old)
            prior_accounting[slot_id] = prior
            record = release._read(release._path(slot_id))
            outcome = copy.deepcopy(record["outcome"])
            generation = outcome.get("budget") or {}
            if generation.get("usage_unknown") or generation.get("reserved_tokens", 0):
                raise ValueError(f"generation budget is unsettled: {slot_id}")
            total = {key: generation.get(key, 0) + prior[key] for key in prior}
            outcome["task_generation_budget"] = {**generation, **total,
                                                  "recovery_prior_evaluation": prior}
            record["outcome"] = outcome
            # TestRelease.evaluate rereads the sealed submission record, so it
            # cannot see the recovery-only budget overlay. Keep the sealed
            # record byte-for-byte unchanged and score this in-memory record,
            # then write only the new recovery evaluation artifact.
            started = release._path(slot_id, "evaluation_started")
            if started.exists():
                raise RuntimeError(f"Recovery judgment already started: {slot_id}")
            _write(started, {"record_hash": digest(record), "started_at": utc_now()})
            try:
                result = score_with_pipeline(pipeline, record)
            except (Exception, SystemExit) as exc:
                result = {"slot": record["slot"], "official_score": None,
                          "complete": False, "status": "evaluation_failed",
                          "error_type": type(exc).__name__,
                          "evaluation_budget": getattr(exc, "evaluation_budget", {})}
            _write(release._path(slot_id, "evaluations"), result)
            scores.append(result)
            _write(output / "scoring_progress.json", {"updated_at": utc_now(), "results": scores})
            if result.get("status") == "evaluation_failed":
                stopped = {"slot_id": slot_id, "error_type": result.get("error_type")}
                break
        all_rows = [_read(path) for path in (output / "test_release" / "evaluations").glob("*.json")]
        complete = [row for row in all_rows if row.get("complete") is True and row.get("official_score") is not None]
        manifest_doc = _read(output / "recovery_manifest.json")
        manifest_doc["attempts"].append({"at": utc_now(), "retried": audit["incomplete"],
                                         "completed_after": [row["slot"]["slot_id"] for row in scores if row.get("complete") is True],
                                         "prior_evaluation_accounting": prior_accounting,
                                         "stopped_after_failure": stopped})
        manifest_doc["complete_count"] = len(complete)
        manifest_doc["partial_mean"] = (math.fsum(row["official_score"] for row in complete) / len(complete)
                                         if complete else None)
        # Six v35 TEST slots are sealed generation failures.  Even when all
        # 27 submitted answers are scored, the denominator is still 33; never
        # expose the partial mean as a complete TEST mean.
        manifest_doc["all_task_mean"] = (manifest_doc["partial_mean"] if len(complete) == 33 else None)
        _write(output / "recovery_manifest.json", manifest_doc)
        return manifest_doc
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run", "status"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path(".runtime/rr_deepseek_judge_20261002_v35"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root, source, runtime, output = (p.resolve() for p in (args.root, args.source, args.runtime, args.output))
    # Resolve every jit_mas/scripts import from the frozen v35 runtime. The
    # repository checkout contains later development edits and must not enter
    # this scoring identity.
    sys.path.insert(0, str(runtime))
    if args.mode == "status":
        print(json.dumps(_read(output / "recovery_manifest.json"), ensure_ascii=True, indent=2))
        return 0
    metadata = _read(source / "pilot_metadata.json")
    audit = audit_source(source, runtime, metadata.get("judge_model", "deepseek-v4-flash-vision"))
    if args.mode == "check":
        if output.exists():
            raise SystemExit("check refuses an existing recovery output")
        print(json.dumps({"ready": True, "complete_reuse": len(audit["complete"]),
                          "retry": len(audit["incomplete"]), "generation_failures": len(audit["generation_failed"]),
                          "api_calls": 0}, ensure_ascii=True))
        return 0
    _copy_release(audit, output)
    report = _score(root, source, runtime, output, audit)
    print(json.dumps({"status": "complete" if report["all_task_mean"] is not None else "incomplete",
                      "complete": report["complete_count"], "denominator": 33,
                      "partial_mean": report["partial_mean"], "all_task_mean": report["all_task_mean"],
                      "output": str(output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
