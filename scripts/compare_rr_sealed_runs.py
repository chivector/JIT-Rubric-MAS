"""Read-only post-hoc audit of sealed ResearchRubrics TEST runs.

No generation or judging is performed.  Running or incompatible experiments
cannot release difference conclusions or formal advantage claims.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/rr_single_agent_run0_deepseekjudge_20261002_v29"
METHOD = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v32"

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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
    if digest(inv.get("slots", [])) != inv.get("inventory_hash"):
        errors.append({"error": "inventory_hash_mismatch"})
    if seal.get("inventory_hash") != inv.get("inventory_hash"):
        errors.append({"error": "seal_inventory_hash_mismatch"})
    by_slot = {s.get("slot_id"): s for s in inv.get("slots", [])}
    for tid in expected:
        slotid = next((sid for sid, s in by_slot.items() if s.get("task_id") == tid), None)
        if not slotid: errors.append({"task_id": tid, "error": "missing_inventory_slot"}); continue
        p = root / "test_release/submissions" / (digest(slotid) + ".json")
        if not p.is_file(): errors.append({"task_id": tid, "error": "missing_submission"}); continue
        doc = read(p)
        if seal.get("submission_hashes", {}).get(slotid) != digest(doc):
            errors.append({"task_id": tid, "error": "submission_seal_digest_mismatch"})
        sub = doc.get("submission", {})
        ah = sub.get("answer_hash")
        if not ah: errors.append({"task_id": tid, "error": "missing_answer_hash"}); continue
        if digest(sub.get("answer")) != ah: errors.append({"task_id": tid, "error": "answer_hash_mismatch"})
        hashes[tid] = ah
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
    if not p.is_dir(): return errors
    for file in sorted(p.glob("*.json")):
        doc = read(file); tid = doc.get("slot", {}).get("task_id")
        if tid not in expected: errors.append({"file": str(file), "error": "evaluation_outside_test"}); continue
        if doc.get("complete") is not True: continue
        ah = doc.get("answer_hash"); raw_ah = doc.get("evaluation", {}).get("raw", {}).get("submission_answer_hash")
        if not ah or answer_hashes.get(tid) != ah or raw_ah != ah:
            errors.append({"task_id": tid, "file": str(file), "error": "evaluation_answer_hash_mismatch"})
    return errors

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--baseline", type=Path, default=BASE); ap.add_argument("--method", type=Path, default=METHOD); ap.add_argument("--output", type=Path)
    a = ap.parse_args(); b, m = a.baseline.resolve(), a.method.resolve(); bm, mm = meta(b), meta(m); br, mr = report(b, "baseline"), report(m, "ours")
    expected = list(bm.get("manifest", {}).get("test", [])) or list(mm.get("manifest", {}).get("test", []))
    bs, bd = rows(br, "results"); ms, md = rows(mr, "test")
    bh, be = sealed_submissions(b, expected); mh, me = sealed_submissions(m, expected)
    be += evaluation_hash_errors(b, expected, bh); me += evaluation_hash_errors(m, expected, mh)
    bv, bj, bf = eval_identity(br); mv, mj, mf = eval_identity(mr)
    checks = {
      "data_sha256": bm.get("data_sha256") == mm.get("data_sha256"),
      "dataset_sha256": bm.get("dataset_sha256") == mm.get("dataset_sha256"),
      "joint_manifest_sha256": bm.get("joint_manifest_sha256") == mm.get("joint_manifest_sha256"),
      "split_manifest": bm.get("manifest") == mm.get("manifest"),
      "knowledge_policy": bm.get("knowledge_policy") == mm.get("knowledge_policy") == "model_general_knowledge_allowed",
      "judge_model": bm.get("judge_model") == mm.get("judge_model"),
      "judge_endpoint": bm.get("judge_endpoint") == mm.get("judge_endpoint"),
      "evaluator_version": not bv or not mv or bv == mv,
      "returned_judge_model": not bj or not mj or bj == mj,
      "returned_model_path": not bf or not mf or bf == mf,
    }
    data_checks = {}
    for name, metadata in (("baseline", bm), ("method", mm)):
        data_path = Path(metadata.get("data", ""))
        data_checks[name] = bool(data_path.is_file() and sha_file(data_path) == metadata.get("data_sha256"))
    checks["actual_data_file_sha256"] = all(data_checks.values())
    pairs, diffs = [], []
    for tid in expected:
        x, y = bs.get(tid, {}), ms.get(tid, {}); xs, ys = x.get("official_score"), y.get("official_score")
        xc = x.get("complete") is True and isinstance(xs, (int, float)); yc = y.get("complete") is True and isinstance(ys, (int, float))
        hashes_ok = (not x.get("answer_hash") or bh.get(tid) == x.get("answer_hash")) and (not y.get("answer_hash") or mh.get(tid) == y.get("answer_hash"))
        ok = xc and yc and hashes_ok; d = float(ys) - float(xs) if ok else None
        if d is not None: diffs.append(d)
        pairs.append({"task_id": tid, "baseline": xs, "ours": ys, "baseline_complete": xc, "ours_complete": yc, "answer_hash_match": hashes_ok, "complete_pair": ok, "difference": d})
    done = mm.get("status") == "completed" and mr.get("status") in {"submitted", "completed"}
    all_pairs = len(expected) == 33 and len(diffs) == 33
    integrity_ok = not be and not me and not bd and not md and len(bh) == len(expected) == len(mh)
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
        "baseline_full_mean": sum(bscores) / 33 if len(bscores) == 33 else None,
        "method_full_mean": sum(mscores) / 33 if all_pairs and done else None,
        "baseline_complete_subset_mean": sum(bscores) / len(bscores) if bscores else None,
        "method_complete_subset_mean": sum(mscores) / len(mscores) if mscores else None,
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
    if a.output: a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
