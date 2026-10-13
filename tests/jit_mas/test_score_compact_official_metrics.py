import copy
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pytest

from jit_mas.schemas import digest
from scripts import score_compact_official_metrics as scorer


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
def test_both_variants_preserve_rng_seed_and_original_record(benchmark, monkeypatch):
    factory = SimpleNamespace(seed=47)
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(DetectorFactory=factory))
    calls = []

    def strict(example, responses):
        calls.append(("strict", copy.deepcopy(example.kwargs), random.random()))
        assert factory.seed == 0
        assert responses == {"Original prompt": "Saved answer"}
        example.kwargs[0]["test"] = "mutation"
        return SimpleNamespace(follow_instruction_list=[True, False],
                               follow_all_instructions=False)

    def loose(example, responses):
        calls.append(("loose", copy.deepcopy(example.kwargs), random.random()))
        return SimpleNamespace(follow_instruction_list=[True, True],
                               follow_all_instructions=True)

    checker = SimpleNamespace(benchmark=benchmark, author=SimpleNamespace(
        InputExample=SimpleNamespace, test_instruction_following_strict=strict,
        test_instruction_following_loose=loose))
    record = {"prompt": "Original prompt", "instruction_id_list": ["one", "two"],
              "kwargs": [{"test": None, "limit": 3}, {}]}
    saved = copy.deepcopy(record)
    random_state = random.getstate()
    strict_result = scorer._check_variant(checker, "Saved answer", record, "strict")
    loose_result = scorer._check_variant(checker, "Saved answer", record, "loose")
    assert strict_result["pass_count"] == 1
    assert loose_result["pass_count"] == 2
    assert calls[0][2] == calls[1][2]
    assert calls[0][1] == calls[1][1]
    assert calls[0][1][0] == ({"limit": 3} if benchmark == "ifbench"
                               else {"test": None, "limit": 3})
    assert random.getstate() == random_state
    assert factory.seed == 47
    assert record == saved


def _fixture_report(tmp_path, monkeypatch, *, missing=False):
    records = {"first": {"prompt": "First", "instruction_id_list": ["one", "two", "three"],
                          "kwargs": [{}, {}, {}]},
               "second": {"prompt": "Second", "instruction_id_list": ["one"], "kwargs": [{}]}}
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("Synthetic dataset", encoding="utf-8")
    report_path = tmp_path / "report.json"
    answer_path = tmp_path / "first.txt"
    answer_path.write_text("Saved answer", encoding="utf-8")
    source_rows = [{"task_id": "first", "complete": True, "answer_file": answer_path.name,
                    "answer_hash": digest("Saved answer"),
                    "private_record_sha256": digest(records["first"])}]
    if not missing:
        source_rows.append({"task_id": "second", "complete": True,
                            "answer_file": answer_path.name,
                            "answer_hash": digest("Saved answer"),
                            "private_record_sha256": digest(records["second"])})
    report = {"schema": "compact-generation-run-v1", "benchmark": "ifbench",
              "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "task_ids": ["first", "second"], "records": source_rows}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dataset = SimpleNamespace(tasks={task_id: None for task_id in records}, private_records=records)
    monkeypatch.setattr(scorer, "load_benchmark", lambda *_args: dataset)
    frozen_calls = []

    def strict(example, responses):
        verdicts = [True, False, False] if example.prompt == "First" else [True]
        return SimpleNamespace(follow_instruction_list=verdicts,
                               follow_all_instructions=all(verdicts))

    def loose(example, responses):
        verdicts = [True] * len(example.instruction_id_list)
        return SimpleNamespace(follow_instruction_list=verdicts,
                               follow_all_instructions=all(verdicts))

    checker = SimpleNamespace(benchmark="ifbench", identity={"revision": "synthetic"},
        assert_frozen=lambda: frozen_calls.append(True), author=SimpleNamespace(
            InputExample=SimpleNamespace, test_instruction_following_strict=strict,
            test_instruction_following_loose=loose))
    monkeypatch.setattr(scorer, "PinnedInstructionChecker", lambda *_args: checker)
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(
        DetectorFactory=SimpleNamespace(seed=None)))
    args = SimpleNamespace(report=str(report_path), dataset=str(data_path),
                           checker_source=str(tmp_path), output=str(tmp_path / "metrics.json"))
    return args, report, frozen_calls


def test_run_preserves_source_and_reports_instruction_micro_accuracy(tmp_path, monkeypatch):
    args, report, frozen_calls = _fixture_report(tmp_path, monkeypatch)
    original_report = (tmp_path / "report.json").read_bytes()
    result = scorer.run(args)
    metrics = result["summary"]["official_metrics"]
    assert metrics == {"prompt_level_strict_accuracy": 0.5,
                       "instruction_level_strict_accuracy": 0.5,
                       "prompt_level_loose_accuracy": 1.0,
                       "instruction_level_loose_accuracy": 1.0}
    assert result["source_report_sha256"] == hashlib.sha256(original_report).hexdigest()
    assert result["task_ids"] == report["task_ids"]
    assert result["efficiency"]["api_calls"] == result["efficiency"]["tokens"] == 0
    assert len(frozen_calls) == 2
    assert (tmp_path / "report.json").read_bytes() == original_report
    with pytest.raises(FileExistsError):
        scorer.run(args)


def test_missing_request_stays_in_rows_and_failure_denominators(tmp_path, monkeypatch):
    args, _report, _calls = _fixture_report(tmp_path, monkeypatch, missing=True)
    result = scorer.run(args)
    assert result["records"][1]["status"] == "not_attempted"
    assert result["records"][1]["strict_result"] is None
    summary = result["summary"]
    assert summary["complete"] is False
    assert summary["official_metrics"]["instruction_level_strict_accuracy"] is None
    assert summary["scored_metrics"]["instruction_level_strict_accuracy"] == pytest.approx(1 / 3)
    assert summary["all_tasks_zero_for_failures"]["instruction_level_strict_accuracy"] == 0.25
    assert summary["all_tasks_zero_for_failures"]["prompt_level_loose_accuracy"] == 0.5
    assert summary["counts"]["strict"]["failed"] == 1


def test_saved_answer_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    args, _report, _calls = _fixture_report(tmp_path, monkeypatch)
    (tmp_path / "first.txt").write_text("Changed answer", encoding="utf-8")
    with pytest.raises(ValueError, match="Answer hash differs"):
        scorer.run(args)
    assert not (tmp_path / "metrics.json").exists()


def test_exception_restores_checker_randomness(monkeypatch):
    factory = SimpleNamespace(seed=23)
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(DetectorFactory=factory))

    def failing(example, responses):
        random.random()
        raise ValueError("Checker failed")

    checker = SimpleNamespace(benchmark="ifbench", author=SimpleNamespace(
        InputExample=SimpleNamespace, test_instruction_following_strict=failing))
    original_state = random.getstate()
    with pytest.raises(ValueError, match="Checker failed"):
        scorer._check_variant(checker, "Saved answer", {
            "prompt": "Prompt", "instruction_id_list": ["one"], "kwargs": [{}]}, "strict")
    assert random.getstate() == original_state
    assert factory.seed == 23
