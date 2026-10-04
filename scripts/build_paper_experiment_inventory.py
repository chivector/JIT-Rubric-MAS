"""Build a publishable inventory from experiment metadata only.

The inventory never opens actor submissions, evaluation records, prompts,
answers, or judge feedback. Body directories are inspected only by filename
count. Scores are projected only from a sealed summary's aggregate fields.
Historical campaigns remain separate and no winner is inferred automatically.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ours", "direct", "native_jit", "initial_mas", "ours_initial", "ours_selected", "jit_matched", "rubric_fixed")
SIX_SOURCES = ("researchrubrics", "deepsearchqa", "writingbench", "deepresearch_bench_ii", "ifeval", "ifbench")
SUMMARY_KEYS = (
    "model_calls", "tokens", "reserved_tokens", "tool_calls", "communication_bytes",
    "wall_seconds", "request_queue_idle_seconds", "cost", "monetary_cost", "input_tokens",
    "output_tokens", "generation_model_calls", "generation_tokens", "evaluation_model_calls",
    "evaluation_tokens", "generation_active_seconds", "evaluation_active_seconds",
    "estimated_attempts", "unknown_slots", "unknown_budgets", "unsettled_calls",
)
METADATA_FILES = frozenset({
    "registration.json", "summary.metadata.json", "summary.json", "report_summary.json",
    "pilot_metadata.json", "metadata.json", "config.json", "inventory.json", "seal.json", "joint_protocol_v5.json",
})
FORBIDDEN_DIRS = frozenset({"submissions", "evaluations", "evaluation_started", "checkpoints", "events"})
PHASES = ("evolution", "validation", "test")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.I)
_LABEL = re.compile(r"[\w. /:;+()=,\-\[\]]{1,512}\Z", re.UNICODE)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    """Read an allowlisted metadata file; reject body paths defensively."""
    resolved = path.resolve()
    if path.name not in METADATA_FILES or FORBIDDEN_DIRS.intersection(resolved.parts):
        raise ValueError("Inventory reader accepts metadata files only")
    return json.loads(path.read_text(encoding="utf-8"))


def _document(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = _read_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any) -> int | float | None:
    return value if _finite(value) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _label(value: Any) -> str | None:
    if not isinstance(value, str) or not _LABEL.fullmatch(value) or re.search(r"\bsk-[A-Za-z0-9]{8,}", value):
        return None
    return value


def _hash(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _hash_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {key: digest for key, raw in value.items() if _label(key) and (digest := _hash(raw))}


def _safe_relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "external_metadata"


def _project_budget(value: Any) -> Any:
    """Keep accounting scalars and drop request payloads or credential fields."""
    if not isinstance(value, dict):
        return None
    result = {}
    for key, item in value.items():
        if key in SUMMARY_KEYS:
            result[key] = _number(item)
        elif key in METHODS or key in {"inference", "execution", "external_evaluation", "experience_update"}:
            result[key] = _project_budget(item)
        elif key == "by_stage" and isinstance(item, dict):
            result[key] = {name: _project_budget(budget) for name, budget in item.items() if _label(name) and isinstance(budget, dict)}
    return result


def _release_dir(campaign: Path) -> tuple[str, Path | None]:
    for name in ("release", "test_release"):
        path = campaign / name
        if path.is_dir():
            return name, path
    return "", None


def _audit_release(campaign: Path, release: Path | None, release_name: str) -> dict[str, Any]:
    """Audit inventory/seal metadata and filename sets, never body contents."""
    result: dict[str, Any] = {
        "release": release_name or None, "sealed": False, "inventory_present": False,
        "seal_present": False, "inventory_hash_match": None, "seal_inventory_hash_match": None,
        "slot_count": None, "submission_count": None, "evaluation_count": None,
        "submission_file_set_complete": None, "evaluation_file_set_complete": None,
        "body_hashes_verified": False, "submission_hashes_complete": None,
        "answer_hashes_complete": None, "metadata_integrity_ok": False,
        "hash_audit_errors": 0, "errors": {},
    }
    if release is None:
        return result

    def error(code: str) -> None:
        result["hash_audit_errors"] += 1
        result["errors"][code] = result["errors"].get(code, 0) + 1

    inv_path, seal_path = release / "inventory.json", release / "seal.json"
    result["inventory_present"], result["seal_present"] = inv_path.is_file(), seal_path.is_file()
    inventory, seal = _document(inv_path), _document(seal_path)
    slots = inventory.get("slots")
    if not isinstance(slots, list):
        if inv_path.is_file():
            error("invalid_inventory_metadata")
        return result
    slot_ids = [row.get("slot_id") if isinstance(row, dict) else None for row in slots]
    valid_ids = all(isinstance(slot_id, str) and slot_id for slot_id in slot_ids)
    if not valid_ids or len(set(slot_ids)) != len(slot_ids):
        error("invalid_or_duplicate_inventory_slots")
    result["slot_count"] = len(slots)
    result["inventory_hash"] = _hash(inventory.get("inventory_hash"))
    try:
        result["inventory_hash_match"] = canonical_digest(slots) == result["inventory_hash"]
    except (ValueError, TypeError):
        result["inventory_hash_match"] = False
    if not result["inventory_hash_match"]:
        error("inventory_hash_mismatch")
    if seal_path.is_file():
        result["seal_inventory_hash_match"] = _hash(seal.get("inventory_hash")) == result["inventory_hash"] and result["inventory_hash"] is not None
        hashes = seal.get("submission_hashes")
        if not result["seal_inventory_hash_match"]:
            error("seal_inventory_hash_mismatch")
        if not isinstance(hashes, dict) or not valid_ids or set(hashes) != set(slot_ids) or any(_hash(value) is None for value in hashes.values()):
            error("seal_submission_metadata_mismatch")
    expected = {canonical_digest(slot_id) + ".json" for slot_id in slot_ids} if valid_ids else set()
    # Listing names is the only contact with these body directories.
    for directory, key in (("submissions", "submission"), ("evaluations", "evaluation")):
        names = {path.name for path in (release / directory).glob("*.json") if path.is_file()}
        result[f"{key}_count"] = len(names)
        result[f"{key}_file_set_complete"] = names == expected if valid_ids else False
        result[f"{key}_missing_files"] = len(expected - names) if valid_ids else None
        result[f"{key}_unexpected_files"] = len(names - expected) if valid_ids else None
    result["sealed"] = bool(result["inventory_present"] and result["seal_present"])
    result["metadata_integrity_ok"] = result["sealed"] and result["hash_audit_errors"] == 0
    return result


def _summary_path(campaign: Path) -> Path | None:
    for name in ("summary.metadata.json", "summary.json", "report_summary.json"):
        path = campaign / name
        if path.is_file():
            return path
    return None


def _configuration_metadata(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"sha256": canonical_digest(config) if config else None, "seed": _count(config.get("seed"))}
    result["resources"] = {name: _number(config.get(name)) for name in (
        "max_agents", "max_parallel", "max_model_calls", "max_total_tokens", "max_tool_calls",
        "max_repairs", "execution_timeout", "task_timeout", "max_inflight_requests",
    ) if name in config}
    result["public_flags"] = {name: value for name, value in config.items() if name.startswith("public_") and isinstance(value, (bool, int, float)) and (isinstance(value, bool) or _finite(value))}
    models = config.get("models")
    result["models"] = {}
    if isinstance(models, dict):
        for role in ("plan", "meta", "exec", "judge", "attribute"):
            value = models.get(role)
            if not isinstance(value, dict):
                continue
            result["models"][role] = {name: _label(value.get(name)) for name in ("model", "expected_response_model", "reasoning_effort", "thinking") if name in value}
            result["models"][role].update({name: _number(value.get(name)) for name in ("max_tokens", "temperature", "timeout", "frequency_penalty") if name in value})
            result["models"][role]["endpoint_sha256"] = canonical_digest(value["endpoint"]) if isinstance(value.get("endpoint"), str) else None
    return result


def _identity_metadata(registration: dict[str, Any], pilot: dict[str, Any], config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    frozen = pilot.get("frozen_identity") if isinstance(pilot.get("frozen_identity"), dict) else {}
    code = registration.get("code") if isinstance(registration.get("code"), dict) else frozen
    configuration = registration.get("configuration")
    if not isinstance(configuration, dict):
        configuration = pilot.get("config") if isinstance(pilot.get("config"), dict) else config
    result: dict[str, Any] = {
        "runtime_sha256": _hash(code.get("runtime", code.get("code"))),
        "runner_sha256": _hash(code.get("runner")),
        "registration_hash": _hash(registration.get("registration_hash")),
        "summary_registration_hash": _hash(summary.get("registration_hash")),
        "configuration": _configuration_metadata(configuration),
        "split_sha256": _hash(registration.get("split_sha256", pilot.get("joint_manifest_sha256"))),
        "split_manifest_sha256": _hash(registration.get("split_manifest_sha256")),
        "scoring_seed": _count(registration.get("scoring_shuffle_seed")),
        "judge_model": _label(pilot.get("judge_model")), "sources": {},
    }
    sources = registration.get("sources", config.get("sources"))
    if isinstance(sources, dict):
        for name, source in sources.items():
            if not _label(name) or not isinstance(source, dict):
                continue
            checker = source.get("checker") if isinstance(source.get("checker"), dict) else {}
            checker_identity = checker.get("identity_sha256")
            checker_identity_doc = checker.get("identity") if isinstance(checker.get("identity"), dict) else {}
            if checker_identity is None:
                checker_identity = checker_identity_doc.get("identity_sha256")
            result["sources"][name] = {
                "dataset_sha256": _hash(source.get("dataset_sha256")), "data_sha256": _hash_map(source.get("data_sha256")),
                "public_projection_sha256": _hash(source.get("source_public_projection_sha256")),
                "checker_identity_sha256": _hash(checker_identity),
                "evaluator_version": _label(source.get("evaluator_version", checker_identity_doc.get("evaluator_version"))),
                "primary_metric": _label(source.get("primary_metric", checker_identity_doc.get("metric"))),
                "metadata_sha256": _hash(source.get("metadata_sha256")), "release_inventory_hash": _hash(source.get("release_inventory_hash")),
                "release_seal_sha256": _hash(source.get("release_seal_sha256")),
            }
    if not result["sources"] and pilot:
        # RR metadata keeps one dataset identity at the campaign level rather
        # than under a per-benchmark registration.sources object.
        result["sources"]["primary"] = {
            "dataset_sha256": _hash(pilot.get("dataset_sha256")),
            "data_sha256": _hash_map(pilot.get("data_sha256")),
            "public_projection_sha256": None, "checker_identity_sha256": None,
            "evaluator_version": None, "primary_metric": None,
            "metadata_sha256": None, "release_inventory_hash": None, "release_seal_sha256": None,
        }
    judge = config.get("judge")
    if isinstance(judge, dict):
        result["external_judge"] = {name: _label(judge.get(name)) for name in ("model", "expected_response_model") if name in judge}
        result["external_judge"].update({name: _number(judge.get(name)) for name in ("transport_attempts", "parse_attempts", "max_tokens", "timeout", "task_budget_tokens") if name in judge})
        result["external_judge"]["endpoint_sha256"] = canonical_digest(judge["endpoint"]) if isinstance(judge.get("endpoint"), str) else None
    elif pilot.get("judge_model") or pilot.get("judge_endpoint"):
        result["external_judge"] = {"model": _label(pilot.get("judge_model")), "expected_response_model": None, "endpoint_sha256": canonical_digest(pilot["judge_endpoint"]) if isinstance(pilot.get("judge_endpoint"), str) else None}
    return result


def _rows_counts(rows: Any, total: int | None) -> dict[str, int | None]:
    counts: dict[str, int | None] = {name: None for name in ("generated", "generation_failed", "generation_unknown", "scored", "null_score")}
    counts["total"] = total
    if not isinstance(rows, list):
        return counts
    counts.update({name: 0 for name in counts if name != "total"})
    counts["summary_rows"] = len(rows)
    for row in rows:
        if not isinstance(row, dict):
            counts["generation_unknown"] += 1; counts["null_score"] += 1; continue
        status = row.get("generation_status")
        key = "generated" if status in {"submitted", "complete", "success"} else "generation_failed" if status == "failed" else "generation_unknown"
        counts[key] += 1
        score = row.get("native_score", row.get("score"))
        counts["scored" if _finite(score) and row.get("evaluation_status") == "completed" else "null_score"] += 1
    return counts


def _summary_scores(summary: dict[str, Any], sealed: bool) -> dict[str, Any]:
    if not sealed:
        return {}
    scores: dict[str, Any] = {}
    arms = summary.get("arms")
    if not isinstance(arms, dict):
        return scores
    for method, arm in arms.items():
        if method not in METHODS or not isinstance(arm, dict) or not isinstance(arm.get("benchmarks"), dict):
            continue
        for benchmark, values in arm["benchmarks"].items():
            if benchmark not in SIX_SOURCES or not isinstance(values, dict):
                continue
            slots, completed = _count(values.get("slots")), _count(values.get("completed"))
            counts_valid = slots is not None and completed is not None and completed <= slots
            complete = counts_valid and slots > 0 and completed == slots
            scores[f"{benchmark}:{method}"] = {
                "slots": slots, "completed": completed, "failed_or_null": slots - completed if counts_valid else None,
                "native_mean_complete": complete,
                "native_mean_all_complete": _number(values.get("native_mean_all_complete")) if complete else None,
                "normalized_mean_failure_zero": _number(values.get("normalized_mean_failure_zero")) if counts_valid and slots > 0 else None,
                "normalization_policy": "Reported theoretical-range normalization; missing native scores remain null. Failure-zero is diagnostic only.",
            }
    return scores


def _campaign(campaign: Path, root: Path, kind: str) -> dict[str, Any]:
    registration = _document(campaign / "registration.json")
    pilot = _document(campaign / "pilot_metadata.json") or _document(campaign / "metadata.json")
    config = _document(campaign / "config.json")
    summary_path = _summary_path(campaign); summary = _document(summary_path)
    release_name, release = _release_dir(campaign); audit = _audit_release(campaign, release, release_name)
    sealed = summary.get("sealed") is True and (release is None or audit["metadata_integrity_ok"])
    identity = _identity_metadata(registration, pilot, config, summary)
    reg_hash, sum_hash = identity["registration_hash"], identity["summary_registration_hash"]
    binding = reg_hash == sum_hash if reg_hash and sum_hash else None
    if binding is False:
        sealed = False
    slots = registration.get("slots")
    total = audit["slot_count"] if audit["slot_count"] is not None else len(slots) if isinstance(slots, list) else _count(summary.get("slot_count", config.get("slot_count")))
    if total is None and isinstance(summary.get("rows"), list):
        total = len(summary["rows"])
    metadata = {name: _label(summary.get(name, registration.get(name, pilot.get(name)))) for name in ("iteration", "scope", "partition", "judge_policy", "knowledge_mode", "knowledge_policy", "status")}
    # Registration is authoritative for exposure/partition binding; a summary
    # may use a narrower benchmark-local scope label.
    if _label(registration.get("partition")) is not None:
        metadata["partition"] = _label(registration.get("partition"))
    if metadata["iteration"] is None:
        match = re.search(r"(?:_|^)(v\d+)$", campaign.name); metadata["iteration"] = match.group(1) if match else None
    return {
        "path": _safe_relative(campaign, root), "kind": kind, **metadata, "sealed_summary": sealed,
        "comparison_complete": summary.get("comparison_complete") is True, "formal_protocol_result": summary.get("formal_protocol_result", registration.get("formal_protocol_result")) is True,
        "development_exposure": registration.get("development_exposure") is True, "protocol_deviation_from_formal_shared_evidence": registration.get("protocol_deviation_from_formal_shared_evidence"),
        "summary_present": summary_path is not None, "summary_path": _safe_relative(summary_path, root), "registration_binding_matches": binding,
        "identity": identity, "release_audit": audit, "counts": _rows_counts(summary.get("rows"), total),
        "complete_paired_task_count": len(summary["complete_paired_task_ids"]) if isinstance(summary.get("complete_paired_task_ids"), list) else None,
        "scores": _summary_scores(summary, sealed), "scores_source": "sealed_summary_arms" if sealed else None,
        "budgets": _project_budget(summary.get("costs", summary.get("cost"))),
    }


def _uniform_rounds(campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for campaign in campaigns:
        iteration = campaign.get("iteration")
        if campaign["kind"] == "development" and isinstance(iteration, str) and re.fullmatch(r"v\d+", iteration):
            grouped[iteration].append(campaign)
    result = []
    for iteration, records in sorted(grouped.items(), key=lambda item: int(item[0][1:])):
        identities = {(record["identity"]["runtime_sha256"], record["identity"]["runner_sha256"], record["identity"]["configuration"]["sha256"]) for record in records}
        identity_complete = len(identities) == 1 and None not in next(iter(identities))
        source_scores: dict[str, dict[str, Any]] = defaultdict(dict); duplicates = False
        for record in records:
            for key, score in record["scores"].items():
                benchmark, method = key.split(":", 1)
                if method in source_scores[benchmark]: duplicates = True
                source_scores[benchmark][method] = score
        methods = {}
        for method in ("ours", "direct", "native_jit"):
            sources = {name: source_scores.get(name, {}).get(method) for name in SIX_SOURCES}
            complete_set = all(isinstance(value, dict) for value in sources.values())
            normalized = [value.get("normalized_mean_failure_zero") for value in sources.values() if isinstance(value, dict)]
            allowed = complete_set and identity_complete and not duplicates and all(record["sealed_summary"] for record in records) and all(_finite(value) for value in normalized)
            methods[method] = {"sources_present": sum(value is not None for value in sources.values()), "sources_required": len(SIX_SOURCES), "complete_six_source_set": complete_set, "normalized_macro_failure_zero": math.fsum(normalized) / len(SIX_SOURCES) if allowed else None, "sources": sources}
        result.append({"iteration": iteration, "campaign_count": len(records), "runtime_configuration_consistent": identity_complete, "duplicate_source_method": duplicates, "formal_result": False, "interpretation": "Development diagnostic; never a formal TEST winner or cross-judge ranking.", "methods": methods})
    return result


def _formal_v5(root: Path) -> dict[str, Any]:
    path = root / "paper/experiments/joint_protocol_v5.json"; protocol = _document(path)
    workload = protocol.get("workload") if isinstance(protocol.get("workload"), dict) else {}
    required = {phase: _count(workload.get(f"{phase}_slots")) for phase in PHASES}
    results = protocol.get("results") if isinstance(protocol.get("results"), list) else []
    seen: set[tuple[str, str]] = set(); completed = dict.fromkeys(PHASES, 0); terminal = dict.fromkeys(PHASES, 0); unrecognized = 0
    for row in results:
        phase, slot_id = (row.get("phase"), row.get("slot_id")) if isinstance(row, dict) else (None, None)
        if phase not in PHASES or not isinstance(slot_id, str) or not slot_id: unrecognized += 1; continue
        if (phase, slot_id) in seen: continue
        seen.add((phase, slot_id)); status = row.get("status")
        if status in {"completed", "failed"}: terminal[phase] += 1
        if status == "completed": completed[phase] += 1
    total = _count(workload.get("total_task_slots")); overfull = any(required[p] is not None and completed[p] > required[p] for p in PHASES)
    valid = bool(protocol) and not overfull; completed_total = sum(completed.values()) if valid else None
    return {"path": _safe_relative(path, root), "present": bool(protocol), "version": _label(protocol.get("version")), "status": _label(protocol.get("status")), "protocol_sha256": _hash(protocol.get("protocol_sha256")), "split_manifest_sha256": _hash(protocol.get("split_manifest_sha256")), "required_slots_by_phase": required, "nominal_total_slots": total, "results_record_count": len(results), "unrecognized_result_records": unrecognized, "completed_slots_by_phase": completed if valid else None, "terminal_slots_by_phase": terminal if valid else None, "completed_total_slots": completed_total, "missing_completed_slots": total - completed_total if total is not None and completed_total is not None and completed_total <= total else None, "formal_run_complete": bool(total and completed_total == total and valid and unrecognized == 0), "runtime_status": _label(protocol.get("runtime_status")), "unit": "Nominal task-artifact slots, not model calls or cost."}


def build_inventory(root: Path = ROOT, outputs_dir: Path | None = None, *, recommended_version: str | None = None) -> dict[str, Any]:
    root, outputs = root.resolve(), (outputs_dir or root / "outputs").resolve()
    campaigns = []
    if outputs.is_dir():
        for path in sorted(outputs.iterdir()):
            if not path.is_dir(): continue
            kind = "development" if path.name.startswith("development_pilot_") else "rr" if path.name.startswith("rr_") else "independent_judge" if path.name.startswith("independent_") else None
            if kind and (_summary_path(path) or (path / "registration.json").is_file() or (path / "pilot_metadata.json").is_file() or (path / "metadata.json").is_file() or (path / "config.json").is_file() or _release_dir(path)[1]):
                campaigns.append(_campaign(path, root, kind))
    recommendation = None
    if recommended_version is not None:
        if not re.fullmatch(r"v\d+", recommended_version): raise ValueError("recommended_version must be a version such as v19")
        selected = [row for row in campaigns if row["kind"] == "development" and row["iteration"] == recommended_version]
        exposed = [row for row in selected if row.get("development_exposure") and row.get("partition") == "exposed_test"]
        if not selected: raise ValueError("The explicit recommended development version has no campaign metadata")
        if not exposed: raise ValueError("The explicit recommended development version has no exposed TEST campaign")
        recommendation = {"version": recommended_version, "selection": "explicit_development_choice", "campaigns": [row["path"] for row in selected], "test_exposure": True, "test_exposure_campaigns": [row["path"] for row in exposed], "formal_winner": False, "automatic_score_selection": False, "qualification": "TEST-based selection is posthoc development evidence; historical versions and judges are not concatenated."}
    result = {"schema_version": "paper-experiment-inventory-v2", "metadata_only": True, "actor_or_judge_bodies_read": False, "source_root": _safe_relative(outputs, root), "campaign_count": len(campaigns), "campaigns": campaigns, "uniform_six_source": _uniform_rounds(campaigns), "formal_v5": _formal_v5(root), "selection_note": "Exposed TEST selection is posthoc development evidence and is excluded from clean confirmation.", "recommendation": recommendation}
    result["inventory_hash"] = canonical_digest(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--recommended-version", help="Explicit development version; never infer a winner from TEST scores")
    args = parser.parse_args(argv)
    try: document = build_inventory(args.root, recommended_version=args.recommended_version)
    except ValueError as exc: parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(document, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "campaigns": document["campaign_count"], "inventory_hash": document["inventory_hash"], "recommended_version": args.recommended_version}, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
