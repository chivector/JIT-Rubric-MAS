import importlib.util
import json
from pathlib import Path

import pytest

from jit_mas.budget import BudgetLedger
from jit_mas.schemas import digest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "rescore_sealed_experiment.py"
spec = importlib.util.spec_from_file_location("rescore_v2", MODULE_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_model_pin_is_required_and_is_exactly_registered(tmp_path):
    args = v2.parser().parse_args(["--mode", "check", "--output", str(tmp_path),
                                   "--source", "x=x", "--judge-endpoint", "https://example.invalid/v1"])
    with pytest.raises(ValueError, match="expected-response-model"):
        v2.validate_args(args)
    args = v2.parser().parse_args(["--mode", "check", "--output", str(tmp_path),
                                   "--source", "x=x", "--judge-endpoint", "https://example.invalid/v1",
                                   "--expected-response-model", "gpt-5.6-sol"])
    v2.validate_args(args)
    assert v2.judge_options(args)["expected_response_models"] == ["gpt-5.6-sol"]


def test_transport_retry_only_retries_transport_errors():
    import json as json_module

    ledger = BudgetLedger(max_calls=None, max_tokens=1000, max_tool_calls=0, timeout_seconds=60)
    calls, sleeps = [], []

    class Flaky:
        def __call__(self, *_args, **_kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise json_module.JSONDecodeError("bad", "", 0)
            return "ok"

    wrapped = v2.TransportRetryModel(Flaky(), attempts=3, retry_delay=5, ledger=ledger,
                                     sleeper=sleeps.append)
    assert wrapped([]) == "ok"
    assert len(calls) == 3 and sleeps == [5, 10]

    calls.clear()

    class BadMessage:
        def __call__(self, *_args, **_kwargs):
            calls.append(1)
            raise ValueError("malformed evaluator response")

    with pytest.raises(ValueError):
        v2.TransportRetryModel(BadMessage(), attempts=3, retry_delay=5, ledger=ledger,
                               sleeper=sleeps.append)([])
    assert len(calls) == 1


def test_create_json_is_immutable(tmp_path):
    path = tmp_path / "immutable.json"
    v2.create_json(path, {"a": 1})
    v2.create_json(path, {"a": 1})
    with pytest.raises(ValueError, match="Immutable"):
        v2.create_json(path, {"a": 2})


def _minimal_v2_output(tmp_path):
    out = tmp_path / "out"
    release = out / "test_release"
    slot = {"slot_id": "ours:synthetic", "task_id": "synthetic", "method": "ours", "repeat": 0,
            "benchmark": "researchrubrics"}
    answer = {"answer": "fixed answer", "answer_hash": digest("fixed answer"), "submitted_at": "fixed"}
    record = {"slot": slot, "status": "submitted", "submission": answer}
    inventory = {"slots": [slot], "inventory_hash": digest([slot])}
    judge = {"model": "gpt-5.6-sol", "endpoint": "https://example.invalid/v1",
             "expected_response_models": ["gpt-5.6-sol"]}
    body = {"version": v2.VERSION, "judge": judge, "sources": {}, "slot_count": 1,
            "slot_policy": {slot["slot_id"]: "first_attempt"}, "history": [],
            "history_metadata": {}, "code_sha256": {}, "registered_at": "fixed"}
    config = body | {"registration_hash": digest(body)}
    write(out / "config.json", config)
    write(release / "inventory.json", inventory)
    write(release / "submissions" / (digest(slot["slot_id"]) + ".json"), record)
    write(release / "seal.json", {"inventory_hash": inventory["inventory_hash"],
                                  "submission_hashes": {slot["slot_id"]: digest(record)}})
    row = {"version": v2.VERSION, "slot": slot, "registration_hash": config["registration_hash"],
           "record_hash": digest(record), "answer_hash": answer["answer_hash"], "judge": judge,
           "history_budget": {"tokens": 0, "model_calls": 0, "unknown_usage_count": 0, "estimated": False},
           "official_score": 1.0, "complete": True, "status": "complete",
           "evaluation_budget": {"tokens": 1, "model_calls": 1, "records": []}, "evaluated_at": "fixed"}
    row["result_hash"] = digest(row)
    evaluation = release / "evaluations" / (digest(slot["slot_id"]) + ".json")
    write(evaluation, row)
    summary = {"version": v2.VERSION, "registration_hash": config["registration_hash"],
               "slot_count": 1, "terminal_count": 1, "complete_count": 1, "statuses": {"complete": 1},
               "current_attempt_tokens": 1, "current_attempt_model_calls": 1, "history_tokens": 0,
               "cumulative_tokens": 1, "scores_exposed": False}
    write(out / "summary.json", summary)
    seal_body = {"version": v2.FINAL_VERSION, "registration_hash": config["registration_hash"],
                 "config_sha256": v2.file_hash(out / "config.json"),
                 "inventory_sha256": v2.file_hash(release / "inventory.json"),
                 "submission_seal_sha256": v2.file_hash(release / "seal.json"),
                 "summary_sha256": v2.file_hash(out / "summary.json"),
                 "evaluations": {slot["slot_id"]: v2.file_hash(evaluation)}}
    write(out / "final_seal.json", seal_body | {"final_seal_hash": digest(seal_body)})
    return out, evaluation


def test_final_seal_detects_evaluation_tamper(tmp_path):
    output, evaluation = _minimal_v2_output(tmp_path)
    assert v2.verify_final_seal(output, required=True)
    row = json.loads(evaluation.read_text(encoding="utf-8"))
    row["official_score"] = 0.0
    evaluation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Final sealed evaluation changed"):
        v2.verify_final_seal(output, required=True)


def test_registration_options_and_history_budget_are_frozen():
    config = {"version": v2.VERSION, "judge": {"expected_response_models": ["gpt-5.6-sol"],
              "transport_attempts": 3, "max_tokens": 32768}, "sources": {"x": {"seal_sha256": "abc"}},
              "slot_count": 1, "slot_policy": {}, "history": [], "history_metadata": {},
              "code_sha256": {}, "campaign_budget_tokens": 1000, "registered_at": "fixed"}
    config["registration_hash"] = digest({key: value for key, value in config.items()
                                           if key != "registration_hash"})
    v2.validate_config(config)
    config["judge"]["transport_attempts"] = 1
    with pytest.raises(ValueError, match="registration hash"):
        v2.validate_config(config)
    history = [{"tokens": 400, "model_calls": 2, "unknown_usage": True, "estimated": True},
               {"tokens": 100, "model_calls": 1, "unknown_usage": False, "estimated": False}]
    assert v2.history_cost(history) == {"tokens": 500, "model_calls": 3,
                                        "unknown_usage_count": 1, "estimated": True}


def test_transport_attempt_bound_and_endpoint_preflight(tmp_path):
    base = ["--mode", "check", "--output", str(tmp_path), "--source", "x=x",
            "--judge-endpoint", "https://example.invalid/v1", "--expected-response-model", "gpt-5.6-sol"]
    args = v2.parser().parse_args(base + ["--transport-attempts", "4"])
    with pytest.raises(ValueError, match="Transport attempts"):
        v2.validate_args(args)
    args = v2.parser().parse_args(base + ["--judge-endpoint", "https://user:secret@example.invalid/v1"])
    with pytest.raises(ValueError, match="without credentials"):
        v2.validate_args(args)


def test_source_identity_changes_when_bound_data_changes(tmp_path):
    source = tmp_path / "source"
    release = source / "release"
    release.mkdir(parents=True)
    (release / "inventory.json").write_text("{}", encoding="utf-8")
    (release / "seal.json").write_text("{}", encoding="utf-8")
    data = source / "data.jsonl"
    data.write_text("fixed\n", encoding="utf-8")
    prepared = {"x": {"source": source, "release": release,
                       "inventory": {"inventory_hash": "fixed"},
                       "info": {"kind": "rr", "metadata_hash": "meta"},
                       "data_paths": {"researchrubrics": data}}}
    first = v2.source_identity(prepared)
    data.write_text("changed\n", encoding="utf-8")
    second = v2.source_identity(prepared)
    assert first != second
