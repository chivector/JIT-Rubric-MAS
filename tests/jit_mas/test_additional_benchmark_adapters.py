"""Synthetic checks for WritingBench and instruction-following adapters."""

import json

import pytest

from jit_mas.benchmarks import load_benchmark, normalized_question_hash


def write_rows(tmp_path, name, rows):
    path = tmp_path / (name + ".jsonl")
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_writingbench_loads_distinct_public_questions_and_private_criteria(tmp_path):
    rows = [{"index": index, "query": f"Write synthetic text {index}",
             "lang": "en", "domain1": "Synthetic", "criteria": ["PRIVATE_CRITERION"]}
            for index in (1, 2)]
    dataset = load_benchmark("writingbench", write_rows(tmp_path, "writingbench", rows))
    assert list(dataset.tasks) == ["writingbench:1", "writingbench:2"]
    for row in rows:
        task_id = f"writingbench:{row['index']}"
        assert dataset.tasks[task_id].question == row["query"]
        assert dataset.public_metadata[task_id]["question_sha256"] == normalized_question_hash(row["query"])
        assert dataset.private_records[task_id]["criteria"] == [{"criterion": "PRIVATE_CRITERION"}]
    public = json.dumps({"tasks": {task_id: task.model_dump() for task_id, task in dataset.tasks.items()},
                         "metadata": dataset.public_metadata})
    assert "PRIVATE_CRITERION" not in public
    assert len({metadata["group_id"] for metadata in dataset.public_metadata.values()}) == 2


def test_writingbench_retains_native_mean_and_reports_normalized_score(tmp_path):
    row = {"index": 1, "query": "Write synthetic text", "lang": "en", "domain1": "Synthetic",
           "criteria": ["PRIVATE_ONE", "PRIVATE_TWO"]}
    dataset = load_benchmark("writingbench", write_rows(tmp_path, "writingbench", [row]))
    calls = []

    def judge(messages, **kwargs):
        calls.append(messages)
        return {"scores": [{"criterion": "PRIVATE_ONE", "score": 1, "reason": "Synthetic"},
                           {"criterion": "PRIVATE_TWO", "score": 10, "reason": "Synthetic"}]}

    task_id = "writingbench:1"
    result = dataset.evaluator(judge).evaluate("Synthetic submission", task_id,
                                              private_record=dataset.private_records[task_id])
    assert result["complete"]
    assert result["native_mean"] == 5.5
    assert result["score"] == 0.5
    assert result["model_calls"] == len(calls) == 1


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
@pytest.mark.parametrize("verdicts,expected", [([True, False], 0), ([True, True], 1)])
def test_instruction_benchmarks_report_prompt_accuracy_and_separate_instruction_accuracy(
        tmp_path, benchmark, verdicts, expected):
    row = {"key": 5, "prompt": "Follow two synthetic constraints",
           "instruction_id_list": ["synthetic:first", "synthetic:second"], "kwargs": [{}, {}]}
    dataset = load_benchmark(benchmark, write_rows(tmp_path, benchmark, [row]))
    checker_calls = []

    def checker(prediction, instruction_ids, options):
        checker_calls.append((prediction, instruction_ids, options))
        return verdicts

    task_id = benchmark + ":5"
    result = dataset.evaluator(None, checker=checker).evaluate(
        "Synthetic submission", task_id, private_record=dataset.private_records[task_id])
    assert result["complete"]
    assert result["score"] == expected
    assert result["instruction_accuracy"] == pytest.approx(sum(verdicts) / len(verdicts))
    assert result["pass_count"] == sum(verdicts)
    assert result["model_calls"] == 0
    assert len(checker_calls) == 1
    assert result["strict"] == (benchmark == "ifeval")


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
def test_missing_checker_is_incomplete_even_with_precomputed_verdicts(tmp_path, benchmark):
    row = {"key": 5, "prompt": "Follow a synthetic constraint",
           "instruction_id_list": ["synthetic:first"], "kwargs": [{}], "checks": [True]}
    dataset = load_benchmark(benchmark, write_rows(tmp_path, benchmark, [row]))
    task_id = benchmark + ":5"
    private_record = {**dataset.private_records[task_id], "checks": [True]}
    result = dataset.evaluator(None).evaluate("An unrelated submission", task_id,
                                             private_record=private_record)
    assert not result["complete"]
    assert result["score"] is None
    assert result["model_calls"] == 0
