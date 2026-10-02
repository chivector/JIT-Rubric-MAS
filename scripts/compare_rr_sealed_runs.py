"""Read-only post-hoc audit of sealed ResearchRubrics TEST runs.

No generation or judging is performed.  Running or incompatible experiments
cannot release difference conclusions or formal advantage claims.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/rr_single_agent_run0_deepseekjudge_20261002_v29"
METHOD = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v32"

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def finite_score(value):
    """Return true only for a real, finite numeric score."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))

def has_interruption(root):
    """An interrupted campaign is never eligible for sealed release."""
    return any(p.is_file() for p in Path(root).rglob("interruption.json"))

def same_present(left, right):
    """Compatibility requires both sides to record a nonempty identity."""
    return bool(left) and bool(right) and left == right

def meta(root):
    p = root / "pilot_metadata.json"
    return read(p) if p.is_file() else {}
def report(root, arm):
    for p in (root / arm / "report.json", root / "ours/report.json", root / "baseline/report.json"):
        if p.is_file(): return read(p)
    return {}
def rows(rep, section):
    out, dupes = {}, []
    for row in rep.get(section, {}).get("tasks", []) or rep.get("submitted_outcomes", []):
        slot = row.get("slot", {})
        tid = slot.get("task_id") or row.get("task_id")
        if tid:
            if tid in out: dupes.append(tid)
            out[tid] = row
    return out, dupes

def sealed_submissions(root, expected):
    hashes, errors = {}, []
    invp, sealp = root / "test_release/inventory.json", root / "test_release/seal.json"
    if not invp.is_file() or not sealp.is_file():
        return hashes, [{"error": "missing_inventory_or_seal"}]
    inv, seal = read(invp), read(sealp)
    if not isinstance(inv, dict) or not isinstance(seal, dict):
        return hashes, [{"error": "invalid_inventory_or_seal_document"}]
    if digest(inv.get("slots", [])) != inv.get("inventory_hash"):
        errors.append({"error": "inventory_hash_mismatch"})
    if seal.get("inventory_hash") != inv.get("inventory_hash"):
        errors.append({"error": "seal_inventory_hash_mismatch"})
    slots = inv.get("slots", [])
    by_slot = {s.get("slot_id"): s for s in slots}
    if len(by_slot) != len(slots):
        errors.append({"error": "duplicate_inventory_slot_id"})
    task_slots = {}
    for slot in slots:
        tid = slot.get("task_id")
        if tid in task_slots:
            errors.append({"task_id": tid, "error": "duplicate_inventory_task_id"})
        task_slots[tid] = slot
        if tid not in expected:
            errors.append({"task_id": tid, "error": "inventory_outside_test"})
    if set(task_slots) != set(expected):
        errors.append({"error": "inventory_task_set_mismatch"})
    for tid in expected:
        slotid = next((sid for sid, s in by_slot.items() if s.get("task_id") == tid), None)
        if not slotid: errors.append({"task_id": tid, "error": "missing_inventory_slot"}); continue
        p = root / "test_release/submissions" / (digest(slotid) + ".json")
        if not p.is_file(): errors.append({"task_id": tid, "error": "missing_submission"}); continue
        doc = read(p)
        if doc.get("slot") != by_slot.get(slotid) or doc.get("status") != "submitted":
            errors.append({"task_id": tid, "error": "submission_slot_or_status_mismatch"})
        if seal.get("submission_hashes", {}).get(slotid) != digest(doc):
            errors.append({"task_id": tid, "error": "submission_seal_digest_mismatch"})
        sub = doc.get("submission", {})
        ah = sub.get("answer_hash")
        if not ah: errors.append({"task_id": tid, "error": "missing_answer_hash"}); continue
        if digest(sub.get("answer")) != ah: errors.append({"task_id": tid, "error": "answer_hash_mismatch"})
        hashes[tid] = ah
    if len(hashes) != len(expected):
        errors.append({"error": "submission_hash_count_mismatch"})
    if set(seal.get("submission_hashes", {})) != set(by_slot):
        errors.append({"error": "seal_submission_slot_set_mismatch"})
    return hashes, errors

def eval_identity(rep):
    versions, models, fps = set(), set(), set()
    for section in (rep.get("results", {}), rep.get("test", {})):
        for row in section.get("tasks", []):
            e, raw = row.get("evaluation", {}), row.get("evaluation", {}).get("raw", {})
            if e.get("evaluator_version"): versions.add(e["evaluator_version"])
            if raw.get("judge_model"): models.add(raw["judge_model"])
            for rec in row.get("evaluation_budget", {}).get("records", []):
                if rec.get("request", {}).get("response_model"): fps.add(rec["request"]["response_model"])
    return sorted(versions), sorted(models), sorted(fps)

def evaluation_hash_errors(root, expected, answer_hashes):
    errors = []
    p = root / "test_release/evaluations"
    if not p.is_dir():
        return ([{"error": "missing_evaluations_directory"}]
                + [{"task_id": tid, "error": "missing_evaluation"} for tid in expected]), {}
    inventory_path = root / "test_release/inventory.json"
    inventory_slots = {slot.get("task_id"): slot
                       for slot in read(inventory_path).get("slots", [])} if inventory_path.is_file() else {}
    docs, seen = {}, set()
    for file in sorted(p.glob("*.json")):
        try:
            doc = read(file)
            if not isinstance(doc, dict):
                raise ValueError("evaluation must be an object")
        except Exception as exc:
            errors.append({"file": str(file), "error": "invalid_evaluation_json",
                           "detail": type(exc).__name__}); continue
        slot_doc = doc.get("slot")
        tid = slot_doc.get("task_id") if isinstance(slot_doc, dict) else None
        if tid not in expected:
            errors.append({"file": str(file), "error": "evaluation_outside_test"}); continue
        if tid in seen:
            errors.append({"task_id": tid, "file": str(file), "error": "duplicate_evaluation_task"}); continue
        seen.add(tid); docs[tid] = doc
        if slot_doc != inventory_slots.get(tid):
            errors.append({"task_id": tid, "file": str(file), "error": "evaluation_inventory_slot_mismatch"})
        slot_id = slot_doc.get("slot_id")
        if not slot_id or file.name != digest(slot_id) + ".json":
            errors.append({"task_id": tid, "file": str(file), "error": "evaluation_slot_filename_mismatch"})
        if doc.get("complete") is not True:
            errors.append({"task_id": tid, "file": str(file), "error": "incomplete_evaluation"})
        if not finite_score(doc.get("official_score")):
            errors.append({"task_id": tid, "file": str(file), "error": "invalid_evaluation_score"})
        evaluation = doc.get("evaluation")
        raw = evaluation.get("raw") if isinstance(evaluation, dict) else None
        if not isinstance(evaluation, dict): evaluation = {}
        if not isinstance(raw, dict): raw = {}
        if evaluation.get("complete") is not True or raw.get("complete") is not True:
            errors.append({"task_id": tid, "error": "incomplete_evaluation_feedback"})
        if evaluation.get("task_id") != tid or raw.get("task_id") != tid:
            errors.append({"task_id": tid, "error": "evaluation_task_identity_mismatch"})
        scores = [doc.get("official_score"), evaluation.get("score"),
                  raw.get("score"), raw.get("official_compliance")]
        if not all(finite_score(score) for score in scores) or len(set(scores)) != 1:
            errors.append({"task_id": tid, "error": "evaluation_score_layers_mismatch"})
        rubrics = evaluation.get("rubrics", [])
        if (not isinstance(rubrics, list) or not rubrics or any(not isinstance(row, dict) for row in rubrics)
                or len({row.get("rubric_id") for row in rubrics}) != len(rubrics)
                or any(not row.get("rubric_id") or not finite_score(row.get("score"))
                       or not finite_score(row.get("weight")) for row in rubrics)):
            errors.append({"task_id": tid, "error": "invalid_evaluation_rubrics"})
        else:
            numerator = math.fsum(row["weight"] * row["score"] for row in rubrics)
            denominator = math.fsum(row["weight"] for row in rubrics if row["weight"] > 0)
            score = numerator / denominator if denominator > 0 else 0.0
            if not finite_score(doc.get("official_score")) or not math.isclose(
                    score, doc["official_score"], rel_tol=1e-12, abs_tol=1e-12):
                errors.append({"task_id": tid, "error": "evaluation_rubric_score_mismatch"})
        response_models = [(rec.get("request") or {}).get("response_model")
                           for rec in doc.get("evaluation_budget", {}).get("records", [])
                           if rec.get("kind") == "model"]
        if (not evaluation.get("evaluator_version") or not raw.get("judge_model")
                or not response_models or not all(response_models)):
            errors.append({"task_id": tid, "error": "missing_evaluation_identity"})
        ah = doc.get("answer_hash"); raw_ah = raw.get("submission_answer_hash")
        if not ah or answer_hashes.get(tid) != ah or raw_ah != ah:
            errors.append({"task_id": tid, "file": str(file), "error": "evaluation_answer_hash_mismatch"})
    missing = set(expected) - seen
    for tid in sorted(missing):
        errors.append({"task_id": tid, "error": "missing_evaluation"})
    if len(seen) != len(expected):
        errors.append({"error": "evaluation_count_mismatch", "expected": len(expected), "found": len(seen)})
    return errors, docs

def report_evaluation_errors(rep, evaluation_docs, expected):
    """The report is a view of sealed evaluations; it cannot override them."""
    errors = []
    tasks = ((rep.get("results") or {}).get("tasks", [])
             or (rep.get("test") or {}).get("tasks", []))
    seen = set()
    for row in tasks:
        tid = row.get("slot", {}).get("task_id") or row.get("task_id")
        if tid not in expected:
            errors.append({"task_id": tid, "error": "report_outside_test"}); continue
        if tid in seen:
            errors.append({"task_id": tid, "error": "duplicate_report_task"}); continue
        seen.add(tid)
        doc = evaluation_docs.get(tid)
        if doc is None:
            continue
        if row.get("complete") is not True or not finite_score(row.get("official_score")):
            errors.append({"task_id": tid, "error": "invalid_report_score"}); continue
        if row.get("official_score") != doc.get("official_score"):
            errors.append({"task_id": tid, "error": "report_evaluation_score_mismatch"})
        for field in ("slot", "answer_hash", "evaluation", "evaluation_budget"):
            if row.get(field) != doc.get(field):
                errors.append({"task_id": tid, "field": field,
                               "error": "report_evaluation_record_mismatch"})
    if set(seen) != set(expected):
        errors.append({"error": "report_task_set_mismatch"})
    return errors

def compare_runs(baseline, method):
    b, m = Path(baseline).resolve(), Path(method).resolve()
    bm, mm = meta(b), meta(m); br, mr = report(b, "baseline"), report(m, "ours")
    expected = list(bm.get("manifest", {}).get("test", [])) or list(mm.get("manifest", {}).get("test", []))
    bs, bd = rows(br, "results"); ms, md = rows(mr, "test")
    bh, be = sealed_submissions(b, expected); mh, me = sealed_submissions(m, expected)
    be_eval, be_docs = evaluation_hash_errors(b, expected, bh)
    me_eval, me_docs = evaluation_hash_errors(m, expected, mh)
    be += be_eval + report_evaluation_errors(br, be_docs, expected)
    me += me_eval + report_evaluation_errors(mr, me_docs, expected)
    bv, bj, bf = eval_identity(br); mv, mj, mf = eval_identity(mr)
    checks = {
      "data_sha256": same_present(bm.get("data_sha256"), mm.get("data_sha256")),
      "dataset_sha256": same_present(bm.get("dataset_sha256"), mm.get("dataset_sha256")),
      "joint_manifest_sha256": same_present(bm.get("joint_manifest_sha256"), mm.get("joint_manifest_sha256")),
      "split_manifest": same_present(bm.get("manifest"), mm.get("manifest")),
      "knowledge_policy": bm.get("knowledge_policy") == mm.get("knowledge_policy") == "model_general_knowledge_allowed",
      "judge_model": same_present(bm.get("judge_model"), mm.get("judge_model")),
      "judge_endpoint": same_present(bm.get("judge_endpoint"), mm.get("judge_endpoint")),
      # Missing identity is not evidence of compatibility.
      "evaluator_version": bool(bv and mv and bv == mv),
      "returned_judge_model": bool(bj and mj and bj == mj),
      "returned_model_path": bool(bf and mf and bf == mf),
    }
    data_checks = {}
    for name, metadata in (("baseline", bm), ("method", mm)):
        data_path = Path(metadata.get("data") or "")
        data_checks[name] = bool(data_path.is_file() and sha_file(data_path) == metadata.get("data_sha256"))
    checks["actual_data_file_sha256"] = all(data_checks.values())
    pairs, diffs = [], []
    for tid in expected:
        x, y = bs.get(tid, {}), ms.get(tid, {}); xs, ys = x.get("official_score"), y.get("official_score")
        xc = x.get("complete") is True and finite_score(xs); yc = y.get("complete") is True and finite_score(ys)
        hashes_ok = (not x.get("answer_hash") or bh.get(tid) == x.get("answer_hash")) and (not y.get("answer_hash") or mh.get(tid) == y.get("answer_hash"))
        ok = xc and yc and hashes_ok; d = float(ys) - float(xs) if ok else None
        if d is not None: diffs.append(d)
        pairs.append({"task_id": tid, "baseline": xs if finite_score(xs) else None,
                      "ours": ys if finite_score(ys) else None, "baseline_complete": xc,
                      "ours_complete": yc, "answer_hash_match": hashes_ok,
                      "complete_pair": ok, "difference": d})
    done = (bm.get("status") == "completed" and mm.get("status") == "completed"
            and br.get("status") in {"submitted", "completed"}
            and mr.get("status") in {"submitted", "completed"}
            and not has_interruption(b) and not has_interruption(m))
    all_pairs = len(expected) == 33 and len(diffs) == 33
    integrity_ok = (not be and not me and not bd and not md
                    and len(expected) == len(set(expected)) == 33
                    and len(bh) == len(expected) == len(mh))
    compatible = all(checks.values())
    # Even terminal runs can be incompatible or tampered.  Do not expose a
    # numerical contrast in that case merely because both rows have scores.
    release_differences = done and compatible and integrity_ok
    valid = bool(release_differences and all_pairs)
    bscores = [p["baseline"] for p in pairs if p["baseline_complete"]]
    mscores = [p["ours"] for p in pairs if p["ours_complete"]]
    base_cap = bm.get("config", {}).get("models", {}).get("exec", {}).get("max_tokens")
    method_cap = mm.get("config", {}).get("models", {}).get("exec", {}).get("max_tokens")
    out = {
        "schema_version": "1.1",
        "status": "complete" if done else "method_running_or_incomplete",
        "baseline": str(b), "method": str(m),
        "protocol": "closed_book_exploratory_deepseek_self_judge",
        "identity_checks": checks, "identity_compatible": compatible,
        "integrity_ok": integrity_ok,
        "evaluator_versions": {"baseline": bv, "method": mv},
        "returned_judge_models": {"baseline": bj, "method": mj},
        "returned_model_paths": {"baseline": bf, "method": mf},
        "execution_response_caps": {"baseline": base_cap, "method": method_cap},
        "resource_budgets_match": base_cap == method_cap,
        "resource_note": f"Baseline exec max_tokens={base_cap}; method exec max_tokens={method_cap}. This is an exploratory comparison under the recorded resource budgets.",
        "answer_hash_audit": {
            "baseline": {"expected": len(expected), "found": len(bh), "errors": be},
            "method": {"expected": len(expected), "found": len(mh), "errors": me}},
        "row_duplicates": {"baseline": bd, "method": md},
        "baseline_status": br.get("status"), "method_status": mr.get("status"),
        "paired": pairs, "completed_pairs": len(diffs),
        "subset_mean_difference": sum(diffs) / len(diffs) if diffs and release_differences else None,
        "full_mean_difference": sum(diffs) / 33 if valid else None,
        "baseline_full_mean": sum(bscores) / 33 if len(bscores) == 33 and not be and not bd else None,
        "method_full_mean": sum(mscores) / 33 if valid else None,
        "baseline_complete_subset_mean": sum(bscores) / len(bscores) if bscores and not be and not bd else None,
        "method_complete_subset_mean": sum(mscores) / len(mscores) if mscores and not me and not md else None,
        "wins": sum(d > 1e-12 for d in diffs) if release_differences else None,
        "ties": sum(abs(d) <= 1e-12 for d in diffs) if release_differences else None,
        "losses": sum(d < -1e-12 for d in diffs) if release_differences else None,
        "comparison_valid": valid, "paired_comparison_available": valid,
        "advantage_claim_allowed": False,
        "advantage_claim_note": "Closed-book self-judge results, including differing execution budgets, do not establish the formal shared-evidence independent-judge advantage."
    }
    if not release_differences:
        for pair in out["paired"]:
            pair["difference"] = None
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--baseline", type=Path, default=BASE); ap.add_argument("--method", type=Path, default=METHOD); ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    out = compare_runs(a.baseline, a.method)
    if a.output: a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
if __name__ == "__main__": main()
