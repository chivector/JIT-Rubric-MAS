import json

import pytest

from scripts import compare_rr_sealed_runs as compare


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sealed_root(tmp_path, task_ids=("t1",)):
    root = tmp_path / "run"
    slots = [{"slot_id": "baseline:" + tid, "task_id": tid} for tid in task_ids]
    _write(root / "test_release/inventory.json",
           {"slots": slots, "inventory_hash": compare.digest(slots)})
    hashes = {}
    for slot in slots:
        answer = {"answer": "answer-" + slot["task_id"],
                  "answer_hash": compare.digest("answer-" + slot["task_id"])}
        doc = {"slot": slot, "status": "submitted", "submission": answer}
        _write(root / "test_release/submissions" /
               (compare.digest(slot["slot_id"]) + ".json"), doc)
        hashes[slot["slot_id"]] = compare.digest(doc)
    _write(root / "test_release/seal.json", {
        "inventory_hash": compare.digest(slots), "submission_hashes": hashes})
    return root


def _evaluation(slot, answer_hash, score):
    return {"slot": slot, "complete": True, "official_score": score,
            "answer_hash": answer_hash,
            "evaluation": {"task_id": slot["task_id"], "complete": True,
                           "evaluator_version": "rr-fixture-v1", "score": score,
                           "rubrics": [{"rubric_id": slot["task_id"] + ":r0",
                                        "weight": 1.0, "score": score}],
                           "raw": {"task_id": slot["task_id"], "complete": True,
                                   "judge_model": "deepseek-fixture", "score": score,
                                   "official_compliance": score,
                                   "submission_answer_hash": answer_hash}},
            "evaluation_budget": {"records": [
                {"kind": "model", "request": {"response_model": "fixture/model"}}]}}


def _campaign(tmp_path, name, metadata, arm, score):
    root = _sealed_root(tmp_path / name, tuple(metadata["manifest"]["test"]))
    _write(root / "pilot_metadata.json", metadata)
    inventory = compare.read(root / "test_release/inventory.json")
    hashes, _ = compare.sealed_submissions(root, metadata["manifest"]["test"])
    tasks = []
    for slot in inventory["slots"]:
        row = _evaluation(slot, hashes[slot["task_id"]], score)
        _write(root / "test_release/evaluations" /
               (compare.digest(slot["slot_id"]) + ".json"), row)
        tasks.append(row)
    report = {"status": "completed",
              "results" if arm == "baseline" else "test": {"tasks": tasks}}
    _write(root / arm / "report.json", report)
    return root


@pytest.fixture
def complete_pair(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("dataset-fixture", encoding="utf-8")
    metadata = {"status": "completed", "data": str(data),
                "data_sha256": compare.sha_file(data),
                "dataset_sha256": compare.sha_file(data),
                "joint_manifest_sha256": "fixture-manifest",
                "manifest": {"test": ["t%02d" % index for index in range(33)]},
                "knowledge_policy": "model_general_knowledge_allowed",
                "judge_model": "deepseek-fixture", "judge_endpoint": "https://fixture.invalid/v1"}
    baseline = _campaign(tmp_path, "baseline", metadata, "baseline", 0.25)
    method = _campaign(tmp_path, "method", metadata, "ours", 0.5)
    return baseline, method


def test_missing_evaluations_are_integrity_errors(tmp_path):
    root = _sealed_root(tmp_path)
    hashes, errors = compare.sealed_submissions(root, ["t1"])
    eval_errors, docs = compare.evaluation_hash_errors(root, ["t1"], hashes)
    assert not errors
    assert docs == {}
    assert {row["error"] for row in eval_errors} >= {
        "missing_evaluations_directory", "missing_evaluation"}


def test_incomplete_nonfinite_and_duplicate_evaluations_are_rejected(tmp_path):
    root = _sealed_root(tmp_path)
    hashes, _ = compare.sealed_submissions(root, ["t1"])
    eval_dir = root / "test_release/evaluations"
    row = {"slot": {"task_id": "t1"}, "complete": False,
           "official_score": float("nan"), "answer_hash": hashes["t1"],
           "evaluation": {"raw": {"submission_answer_hash": hashes["t1"]}}}
    _write(eval_dir / "a.json", row)
    _write(eval_dir / "b.json", row)
    errors, _ = compare.evaluation_hash_errors(root, ["t1"], hashes)
    kinds = {row["error"] for row in errors}
    assert {"incomplete_evaluation", "invalid_evaluation_score",
            "duplicate_evaluation_task"} <= kinds


def test_interruption_blocks_release(tmp_path):
    root = _sealed_root(tmp_path)
    _write(root / "nested/interruption.json", {"reason": "stopped"})
    assert compare.has_interruption(root)


def test_complete_33_pair_releases_audited_difference(complete_pair):
    out = compare.compare_runs(*complete_pair)
    assert out["comparison_valid"] is True
    assert out["full_mean_difference"] == 0.25
    assert out["completed_pairs"] == 33
    assert out["advantage_claim_allowed"] is False


@pytest.mark.parametrize("mutation", ["interruption", "missing_evaluation", "report_score",
                                      "none_identity", "duplicate_evaluation", "nonfinite_score"])
def test_integrity_failures_never_release_difference(complete_pair, mutation):
    baseline, method = complete_pair
    report_path = method / "ours/report.json"
    rep = compare.read(report_path)
    eval_file = next((method / "test_release/evaluations").glob("*.json"))
    if mutation == "interruption":
        _write(method / "interruption.json", {"reason": "stopped"})
    elif mutation == "missing_evaluation":
        eval_file.unlink()
    elif mutation == "report_score":
        rep["test"]["tasks"][0]["official_score"] = 0.9
        _write(report_path, rep)
    elif mutation == "none_identity":
        for root in (baseline, method):
            meta = compare.read(root / "pilot_metadata.json")
            meta["judge_endpoint"] = None
            _write(root / "pilot_metadata.json", meta)
    elif mutation == "duplicate_evaluation":
        _write(eval_file.with_name("duplicate.json"), compare.read(eval_file))
    else:
        doc = compare.read(eval_file)
        doc["official_score"] = float("nan")
        _write(eval_file, doc)
    out = compare.compare_runs(baseline, method)
    assert out["comparison_valid"] is False
    assert out["full_mean_difference"] is None
    assert out["subset_mean_difference"] is None
    assert all(row["difference"] is None for row in out["paired"])
    json.dumps(out, allow_nan=False)
