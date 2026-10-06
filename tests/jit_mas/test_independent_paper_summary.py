import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.schemas import ExperienceSnapshot, digest
from jit_mas.token_usage import merge_usage, summarize_budget, summarize_outcome
import scripts.summarize_independent_paper as summary


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def _tiny_campaign(tmp_path, monkeypatch, *, source="static", status="submitted"):
    campaign = Path(tmp_path)
    snapshot = ExperienceSnapshot().model_dump(mode="json")
    execution = {name: name for name in ("launch_file_sha256", "configuration_file_sha256", "joint_manifest_sha256", "runner_sha256")}
    execution["initial_snapshot_hash"] = digest(snapshot)
    registration = {"protocol": {"protocol_sha256": "synthetic"}, "execution": execution}
    adapter = {"schema": "independent-test-adapter-identity-v1", "protocol_sha256": "synthetic",
               "adapter_sha256": hashlib.sha256(Path(summary.__file__).with_name("run_independent_test_release.py").read_bytes()).hexdigest(),
               **{name: execution[name] for name in ("launch_file_sha256", "configuration_file_sha256", "joint_manifest_sha256")},
               "evolution_runner_sha256": execution["runner_sha256"]}
    _write(campaign / "test_adapter_identity.json", adapter)
    run_id, method = (None, "ours_initial") if source == "static" else (0, "ours_selected")
    slots, results, receipts = [], [], []
    budget = {"records": [{"input_tokens": 10, "output_tokens": 2, "estimated": False, "stage": "execution"}]}
    for index, official in enumerate((0.1, 0.2)):
        task_id = f"q{index}"
        slot = {"slot_id": f"test:{source}:{method}:writingbench:{task_id}", "kind": "test",
                "method": method, "source": source, "run_id": run_id,
                "target": "writingbench", "task_id": task_id, "status": "pending"}
        result = {"task_id": task_id, "evaluation": None, "budget": budget, "state_hash": digest(snapshot)}
        if status == "missing":
            result = {"task_id": task_id, "evaluation": None, "budget": None,
                      "error_type": "NoEligibleSelectedState", "registration_hash": digest(registration)}
        if status == "submitted":
            submission = {"answer": f"Answer {index}", "answer_hash": digest(f"Answer {index}")}
            submission_path = campaign / "submissions" / f"{task_id}.json"
            _write(submission_path, submission)
            result.update(submission_path=str(submission_path), submission_hash=digest(submission), answer_hash=submission["answer_hash"], metadata={"recovered": True})
        row = {**slot, "status": status, "result": result, "result_hash": digest(result), "started_at": None, "finished_at": None}
        if status == "submitted":
            receipt = {"slot": row, "official_score": official, "complete": True,
                       "evaluation": {"task_id": task_id, "complete": True, "score": official,
                                      "evaluator_version": "synthetic-v1", "raw": {
                                          "evaluator_version": "synthetic-v1", "native_mean": 1 + 9 * official,
                                          "submission_answer_hash": submission["answer_hash"]}},
                       "answer_hash": submission["answer_hash"], "submission_sha256": result["submission_hash"],
                       "generation_budget": budget, "evaluation_budget": {
                           "records": [{"input_tokens": 3, "output_tokens": 1, "estimated": False, "stage": "evaluation"}]}}
            receipt["token_usage"] = summarize_outcome(receipt)
            receipt["receipt_sha256"] = digest(receipt)
            _write(campaign / "test_evaluations" / f"{digest(slot['slot_id'])}.json", receipt)
            receipts.append(receipt)
        slots.append(slot)
        results.append(row)
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        database.executescript("CREATE TABLE metadata(key TEXT PRIMARY KEY, body TEXT NOT NULL);"
                               "CREATE TABLE slots(slot_id TEXT PRIMARY KEY, ordinal INTEGER, body TEXT, status TEXT,"
                               "claim_token TEXT, owner TEXT, started_at TEXT, result TEXT, result_hash TEXT, finished_at TEXT);"
                               "CREATE TABLE selections(source TEXT, run_id INTEGER, body TEXT);"
                               "CREATE TABLE checkpoints(source TEXT, run_id INTEGER, position INTEGER, state_hash TEXT, snapshot TEXT, identity_hash TEXT);")
        database.execute("INSERT INTO metadata VALUES (?,?)", ("registration", json.dumps(registration)))
        database.execute("INSERT INTO metadata VALUES (?,?)", ("test_seal", json.dumps({
            "registration_hash": digest(registration), "required_test_slots": len(slots),
            "submission_hashes": {row["slot_id"]: row["result_hash"] for row in results}})))
        for index, row in enumerate(results):
            database.execute("INSERT INTO slots VALUES (?,?,?,?,?,?,?,?,?,?)", (
                row["slot_id"], index, json.dumps(slots[index]), status, None, None, None,
                json.dumps(row["result"]), row["result_hash"], None))
        if source != "static":
            selected = None if status == "missing" else {"position": 0, "state_hash": digest(snapshot)}
            database.execute("INSERT INTO selections VALUES (?,?,?)", (source, run_id, json.dumps({
                "source": source, "run_id": run_id, "selected": selected, "registration_hash": digest(registration)})))
            database.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?)", (
                source, run_id, 0, digest(snapshot), json.dumps(snapshot), digest(registration)))
    scores = [row["official_score"] for row in receipts]
    usage = merge_usage(row["token_usage"] for row in receipts) if receipts else merge_usage(
        summarize_budget(row["result"].get("budget")) for row in results)
    _write(campaign / "test_report.json", {
        "schema": "independent-test-report-v5", "test_feedback_released": True,
        "slots": len(slots), "evaluated": len(slots), "complete": len(receipts),
        "incomplete_or_failed": len(slots) - len(receipts), "token_usage": usage,
        "by_condition": [{"source": source, "run_id": run_id, "target": "writingbench", "method": method,
                          "scores": scores, "complete": len(receipts), "slots": len(slots),
                          "mean_official_score": sum(scores) / len(slots) if receipts else None}]})
    monkeypatch.setattr(summary, "REGISTERED_TEST_SLOTS", len(slots))
    monkeypatch.setattr(summary, "slot_registry", lambda protocol: {"artifacts": slots})
    return campaign


@pytest.mark.parametrize("gate", ["seal", "report", "release"])
def test_summary_requires_global_release_before_receipt_reads(tmp_path, monkeypatch, gate):
    campaign = _tiny_campaign(tmp_path, monkeypatch)
    if gate == "seal":
        with sqlite3.connect(campaign / "campaign.sqlite") as database:
            database.execute("DELETE FROM metadata WHERE key='test_seal'")
    elif gate == "report":
        (campaign / "test_report.json").unlink()
    else:
        report = summary.read_json(campaign / "test_report.json")
        report["test_feedback_released"] = False
        _write(campaign / "test_report.json", report)
    original = summary.read_json
    def guarded_read(path):
        assert Path(path).parent.name != "test_evaluations"
        return original(path)
    monkeypatch.setattr(summary, "read_json", guarded_read)
    with pytest.raises(CheckpointIntegrityError):
        summary.load_test_data(campaign)


@pytest.mark.parametrize("source,status", [("static", "submitted"), ("writingbench", "submitted"), ("writingbench", "missing"), ("static", "failed")])
def test_summary_loads_native_scores_and_terminal_statuses(tmp_path, monkeypatch, source, status):
    campaign = _tiny_campaign(tmp_path, monkeypatch, source=source, status=status)
    _, observations, usage = summary.load_test_data(campaign)
    if status == "submitted":
        assert [row["native_score"] for row in observations] == pytest.approx([1.9, 2.8])
        assert usage["model_calls"] == 4
        assert usage["total_tokens"] == 32
    else:
        assert all(row["native_score"] is None and row["status"] == status for row in observations)


@pytest.mark.parametrize("tamper", ["result", "receipt_hash", "slot", "submission", "submission_file", "task", "evaluator", "answer", "budget", "native", "report", "group", "adapter", "selection", "checkpoint", "state"])
def test_summary_detects_identity_receipt_and_report_tampering(tmp_path, monkeypatch, tamper):
    campaign = _tiny_campaign(tmp_path, monkeypatch, source="writingbench")
    if tamper in {"result", "selection", "checkpoint", "state"}:
        with sqlite3.connect(campaign / "campaign.sqlite") as database:
            if tamper in {"result", "state"}:
                row = database.execute("SELECT slot_id,result FROM slots LIMIT 1").fetchone()
                result = json.loads(row[1])
                result["task_id" if tamper == "result" else "state_hash"] = "changed"
                database.execute("UPDATE slots SET result=? WHERE slot_id=?", (json.dumps(result), row[0]))
                if tamper == "state":
                    database.execute("UPDATE slots SET result_hash=? WHERE slot_id=?", (digest(result), row[0]))
                    seal = json.loads(database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()[0])
                    seal["submission_hashes"][row[0]] = digest(result)
                    database.execute("UPDATE metadata SET body=? WHERE key='test_seal'", (json.dumps(seal),))
            elif tamper == "selection":
                database.execute("UPDATE selections SET body=?", (json.dumps({"registration_hash": "changed"}),))
            else:
                database.execute("UPDATE checkpoints SET identity_hash='changed'")
    else:
        path = (campaign / "test_report.json" if tamper in {"report", "group"} else
                campaign / "test_adapter_identity.json" if tamper == "adapter" else
                next((campaign / "test_evaluations").glob("*.json")))
        body = summary.read_json(path)
        if tamper == "report":
            body["token_usage"]["total_tokens"] += 1
        elif tamper == "group":
            body["by_condition"][0]["mean_official_score"] = 0.15
        elif tamper == "adapter":
            body["adapter_sha256"] = "changed"
        elif tamper == "receipt_hash":
            body["receipt_sha256"] = "changed"
        elif tamper == "native":
            body["evaluation"]["raw"]["native_mean"] = 10
        elif tamper == "submission_file":
            submission_path = Path(body["slot"]["result"]["submission_path"])
            submission = summary.read_json(submission_path)
            submission["answer"] = "changed"
            _write(submission_path, submission)
        else:
            if tamper == "slot":
                body["slot"]["task_id"] = "changed"
            elif tamper == "submission":
                body["submission_sha256"] = "changed"
            elif tamper in {"task", "evaluator"}:
                body["evaluation"]["task_id" if tamper == "task" else "evaluator_version"] = "changed"
            elif tamper == "answer":
                body["evaluation"]["raw"]["submission_answer_hash"] = "changed"
            elif tamper == "budget":
                body["generation_budget"] = {}
            body["receipt_sha256"] = digest({key: value for key, value in body.items() if key != "receipt_sha256"})
        _write(path, body)
    with pytest.raises(CheckpointIntegrityError):
        summary.load_test_data(campaign)


def _rewrite_budget(campaign, index, *, generation, evaluation=None):
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        slot_id, slot_body, status, result_body = database.execute(
            "SELECT slot_id,body,status,result FROM slots ORDER BY ordinal").fetchall()[index]
        slot = json.loads(slot_body)
        result = json.loads(result_body)
        result["budget"] = generation
        row = {**slot, "status": status, "result": result, "result_hash": digest(result),
               "started_at": None, "finished_at": None}
        database.execute("UPDATE slots SET result=?,result_hash=? WHERE slot_id=?",
                         (json.dumps(result), row["result_hash"], slot_id))
        seal = json.loads(database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()[0])
        seal["submission_hashes"][slot_id] = row["result_hash"]
        database.execute("UPDATE metadata SET body=? WHERE key='test_seal'", (json.dumps(seal),))
    receipt_path = campaign / "test_evaluations" / f"{digest(slot_id)}.json"
    if receipt_path.exists():
        receipt = summary.read_json(receipt_path)
        receipt["slot"] = row
        receipt["generation_budget"] = generation
        if evaluation is not None:
            receipt["evaluation_budget"] = evaluation
        receipt["token_usage"] = summarize_outcome(receipt)
        receipt["receipt_sha256"] = digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
        _write(receipt_path, receipt)
    report = summary.read_json(campaign / "test_report.json")
    with sqlite3.connect(campaign / "campaign.sqlite") as database:
        all_rows = database.execute("SELECT slot_id,status,result FROM slots ORDER BY ordinal").fetchall()
    usages = []
    for row_slot_id, row_status, row_result in all_rows:
        result = json.loads(row_result)
        if row_status == "submitted":
            usages.append(summary.read_json(campaign / "test_evaluations" / f"{digest(row_slot_id)}.json")["token_usage"])
        else:
            usages.append(summarize_budget(result.get("budget")))
    report["token_usage"] = merge_usage(usages)
    _write(campaign / "test_report.json", report)


def test_raw_unknown_and_reserved_budgets_are_exposed_for_failed_and_submitted_paths(tmp_path, monkeypatch):
    submitted = _tiny_campaign(tmp_path / "submitted", monkeypatch, source="writingbench")
    _rewrite_budget(submitted, 0,
                    generation={"records": [], "model_calls": 0, "tokens": 0, "usage_unknown": True},
                    evaluation={"records": [], "model_calls": 0, "tokens": 0, "reserved_tokens": 7})
    _, observations, _ = summary.load_test_data(submitted)
    submitted_audit = observations[0]["budget_audit"]
    assert submitted_audit["stages"]["generation"]["unknown"] is True
    assert submitted_audit["stages"]["evaluation"]["unsettled"] is True
    submitted_cost = summary.cost_accounting(observations)
    assert submitted_cost["unknown_budget_slots"] >= 1
    assert submitted_cost["unsettled_budget_slots"] >= 1
    assert submitted_cost["by_stage"]["evaluation"]["unsettled_budget_slots"] == 1
    assert submitted_cost["recorded_usage_lower_bound"] is True

    failed = _tiny_campaign(tmp_path / "failed", monkeypatch, source="static", status="failed")
    _rewrite_budget(failed, 0, generation=None)
    _, observations, _ = summary.load_test_data(failed)
    assert observations[0]["budget_audit"]["stages"]["generation"]["unknown"] is True
    assert summary.cost_accounting(observations)["by_stage"]["generation"]["unknown_budget_slots"] == 1


def _paired_observations():
    usage = summarize_budget({"records": [{"input_tokens": 10, "output_tokens": 2, "estimated": False}]})
    conditions = [(source, target) for source in summary.SOURCES for target in (source, "deepresearch_bench_ii", "ifeval", "ifbench")]
    observations = []
    for source, target in conditions:
        for run_id in summary.RUN_IDS:
            for task_index in (0, 1):
                observations.append({"source": source, "run_id": run_id, "target": target,
                    "method": "ours_selected", "task_id": f"{target}:q{task_index}",
                    "native_score": 10 + run_id + (1 if task_index else -1), "status": "complete", "token_usage": usage})
    for target in summary.SOURCES + ("deepresearch_bench_ii", "ifeval", "ifbench"):
        for method in summary.COMPARATORS:
            for task_index in (0, 1):
                observations.append({"source": "static", "run_id": None, "target": target,
                    "method": method, "task_id": f"{target}:q{task_index}",
                    "native_score": 10, "status": "complete", "token_usage": usage})
    return observations


def test_paired_comparisons_average_three_runs_before_task_bootstrap():
    comparisons = summary.paired_comparisons(_paired_observations(), seed=4, iterations=1000)
    row = next(row for row in comparisons if row["source"] == "researchrubrics"
               and row["target"] == "researchrubrics" and row["comparator"] == "ours_initial")
    assert [task["run_differences"] for task in row["paired_tasks"]] == [[-1, 0, 1], [1, 2, 3]]
    assert [task["mean_difference_three_runs"] for task in row["paired_tasks"]] == [0, 2]
    assert row["complete_task_clusters"] == 2
    assert row["mean_difference_all_complete"] == 1
    assert row["bootstrap_95_ci_all_complete"] == [0, 2]
    assert row["sign_flip_pvalue"] == row["holm_pvalue"] == 1


def test_missing_run_keeps_whole_task_cluster_and_inference_null():
    observations = _paired_observations()
    missing = next(row for row in observations if row["source"] == "researchrubrics"
                   and row["target"] == "ifeval" and row["run_id"] == 2 and row["task_id"] == "ifeval:q1")
    missing.update(native_score=None, status="missing")
    comparisons = summary.paired_comparisons(observations, seed=4, iterations=100)
    row = next(row for row in comparisons if row["source"] == "researchrubrics"
               and row["target"] == "ifeval" and row["comparator"] == "ours_initial")
    assert row["paired_tasks"][1]["mean_difference_three_runs"] is None
    assert row["complete_task_clusters"] == 1
    assert row["mean_difference_all_complete"] is row["bootstrap_95_ci_all_complete"] is None
    assert row["sign_flip_pvalue"] is row["holm_pvalue"] is None
    assert row["complete_pair_descriptive"]["mean_difference"] == 0
    assert row["runs"][2]["mean_difference_all_complete"] is None
    assert row["runs"][2]["mean_difference_complete_pair_descriptive"] == 1


def test_transfer_baselines_are_shared_and_physical_costs_count_once(tmp_path, monkeypatch):
    observations = _paired_observations()
    usage = merge_usage(row["token_usage"] for row in observations)
    monkeypatch.setattr(summary, "load_test_data", lambda directory: ({"protocol": {"protocol_sha256": "synthetic"}}, observations, usage))
    report = summary.summarize(tmp_path, seed=4, iterations=100)
    target_rows = [row for row in report["conditions"] if row["target"] == "deepresearch_bench_ii"]
    assert sum(row["registered_slots"] for row in target_rows) == 24
    assert merge_usage(row["token_usage"] for row in target_rows)["total_tokens"] == 24 * 12
    assert report["token_usage"]["model_calls"] == len(observations) == 108
    assert report["token_usage"]["total_tokens"] == 108 * 12
    comparisons = [row for row in report["comparisons"] if row["target"] == "deepresearch_bench_ii" and row["comparator"] == "ours_initial"]
    assert len(comparisons) == 3
    assert all([task["task_id"] for task in row["paired_tasks"]] == ["deepresearch_bench_ii:q0", "deepresearch_bench_ii:q1"] for row in comparisons)
