"""Synthetic-only data, scorer and accounting tests for the benchmark bridge."""

import csv
import json
from copy import deepcopy

import pytest

from benchmark.adapter.deepsearchqa import DeepSearchQAAdapter
from jit_mas.benchmarks import (BENCHMARK_NAMES, DeepResearchBenchIIEvaluator,
                                DeepSearchQAEvaluator, load_benchmark,
                                normalized_question_hash)
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.pipeline import convert_feedback


def dsqa_row(problem="Which synthetic places qualify?"):
    return {"problem": problem, "problem_category": "Synthetic Geography",
            "answer": "PRIVATE_ALPHA; PRIVATE_BETA", "answer_type": "Set Answer"}


def drb_row(index=1):
    return {"id": f"synthetic-{index}", "idx": index, "language": "en", "theme": "Synthetic theme",
            "prompt": f"Write synthetic report {index}. Do not cite the publicly forbidden source.",
            "content": {"task": "Private evaluation task text", "rubric": {
                "info_recall": ["PRIVATE reference coverage"], "analysis": ["PRIVATE analysis"],
                "presentation": ["PRIVATE clarity"]}, "blocked": {"title": "PRIVATE title", "urls": []}}}


def dataset_file(tmp_path, name, rows):
    path = tmp_path / (name + ".jsonl")
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class Judge:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result

    def get_token_counts(self):
        return {"input_token_count": 100, "output_token_count": 20}


def matching(details=None, extras=None):
    return {"Answer Correctness": {"Explanation": "Synthetic matching explanation",
                                  "Correctness Details": details or {"PRIVATE_ALPHA": True, "PRIVATE_BETA": False},
                                  "Excessive Answers": extras or []}}


def metered(replies):
    raw = Judge(replies)
    ledger = BudgetLedger(max_calls=20, max_tokens=200000)
    return MeteredModel(raw, ledger, "evaluation", "judge"), raw, ledger


def test_supported_benchmarks_are_explicit():
    assert BENCHMARK_NAMES == ("researchrubrics", "deepsearchqa", "deepresearch_bench_ii")
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("unverified-benchmark", "missing")


def test_deepsearchqa_public_projection_and_stable_identity(tmp_path):
    first = load_benchmark("deepsearchqa", dataset_file(tmp_path, "first", [dsqa_row()]))
    changed = dsqa_row("  WHICH synthetic places qualify?  ")
    changed.update(answer="ANOTHER_PRIVATE_ANSWER", answer_type="Single Answer")
    second = load_benchmark("deepsearchqa", dataset_file(tmp_path, "second", [changed]))
    task_id = next(iter(first.tasks))
    assert task_id == next(iter(second.tasks))
    assert task_id == "deepsearchqa:" + normalized_question_hash(dsqa_row()["problem"])
    public = json.dumps({"tasks": {k: v.model_dump() for k, v in first.tasks.items()},
                         "metadata": first.public_metadata})
    assert "PRIVATE" not in public and "answer_type" not in public and "Set Answer" not in public
    assert first.public_metadata == second.public_metadata
    assert first.dataset_sha256 != second.dataset_sha256
    assert first.lower_bounds[task_id] == 0
    assert "PRIVATE_ALPHA" not in repr(first)


def test_deepsearchqa_csv_and_directory_loading(tmp_path):
    row = dsqa_row()
    with (tmp_path / "DSQA-full.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    dataset = load_benchmark("deepsearchqa", tmp_path, available_tools=["web_search"])
    assert len(dataset.tasks) == 1
    assert next(iter(dataset.tasks.values())).tools == ["web_search"]


def test_duplicate_content_identity_is_rejected(tmp_path):
    path = dataset_file(tmp_path, "duplicate", [dsqa_row(), dsqa_row()])
    with pytest.raises(ValueError, match="row 2; private values redacted"):
        load_benchmark("deepsearchqa", path)


@pytest.mark.parametrize("field,value", [("answer", []), ("answer_type", "PRIVATE_BAD_TYPE"),
                                       ("problem_category", {}), ("problem", "")])
def test_private_row_errors_are_redacted(tmp_path, field, value):
    row = dsqa_row()
    row[field] = value
    with pytest.raises(ValueError) as error:
        load_benchmark("deepsearchqa", dataset_file(tmp_path, "invalid", [row]))
    assert "PRIVATE" not in str(error.value)
    assert error.value.__suppress_context__


def test_researchrubrics_preserves_ids_and_private_weight_bounds(tmp_path):
    row = {"sample_id": "synthetic-rr", "prompt": "Write a synthetic report", "domain": "science",
           "rubrics": [{"criterion": "PRIVATE positive", "weight": 2},
                       {"criterion": "PRIVATE negative", "weight": -3}]}
    dataset = load_benchmark("researchrubrics", dataset_file(tmp_path, "rr", [row]))
    assert list(dataset.tasks) == ["synthetic-rr"]
    assert dataset.lower_bounds["synthetic-rr"] == -1.5
    assert dataset.public_metadata["synthetic-rr"]["domain"] == "science"
    assert "PRIVATE" not in dataset.tasks["synthetic-rr"].model_dump_json()
    assert dataset.evaluator(Judge([]))._max_attempts == 1


def test_drbii_projection_retains_public_source_prohibition_not_private_task(tmp_path):
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [drb_row()]))
    task_id = "deepresearch_bench_ii:1"
    task = dataset.tasks[task_id]
    assert task.question == drb_row()["prompt"]
    assert "publicly forbidden source" in task.question
    public = task.model_dump_json() + json.dumps(dataset.public_metadata)
    assert "PRIVATE" not in public and "Private evaluation" not in public
    assert dataset.public_metadata[task_id]["source_index"] == 1
    assert dataset.lower_bounds[task_id] == 0


def test_legacy_deepsearchqa_formatter_does_not_leak_answer_type(tmp_path):
    adapter = DeepSearchQAAdapter(workspace_base=str(tmp_path))
    adapter._reset_workspace = lambda _: None
    first = dsqa_row()
    second = {**first, "answer_type": "Single Answer"}
    assert adapter.format_task(first) == adapter.format_task(second)
    assert "comprehensive answer including all" not in adapter.format_task(first)


def test_deepsearchqa_true_item_metrics_and_accounting(tmp_path):
    dataset = load_benchmark("deepsearchqa", dataset_file(tmp_path, "dsqa", [dsqa_row()]))
    task_id = next(iter(dataset.tasks))
    model, judge, ledger = metered([matching(extras=["PRIVATE_EXTRA"])])
    result = dataset.evaluator(model, judge_id="synthetic").evaluate(
        "Answer with one match and one extra", task_id, private_record=dataset.private_records[task_id])
    assert result["complete"] and result["score"] == .5
    assert result["precision"] == result["recall"] == .5
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 1, 1)
    assert len(result["feedback"]) == 3
    assert result["feedback"][-1]["weight"] == -1
    assert all(not row["is_human_authored_rubric"] for row in result["feedback"])
    assert result["prompt_status"] == "adapted_not_official_leaderboard"
    assert len(judge.calls) == result["model_calls"] == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["tokens"] == 120
    assert ledger.snapshot()["by_stage"]["evaluation"]["tokens"] == 120
    assert "PRIVATE_ALPHA" in judge.calls[0][0][1]["content"]
    feedback = convert_feedback(result)
    assert feedback.score == .5 and len(feedback.rubrics) == 3


def test_deepsearchqa_uses_notebook_f1_even_for_single_answer(tmp_path):
    row = {**dsqa_row(), "answer": "PRIVATE_ALPHA", "answer_type": "Single Answer"}
    dataset = load_benchmark("deepsearchqa", dataset_file(tmp_path, "dsqa", [row]))
    task_id = next(iter(dataset.tasks))
    result = dataset.evaluator(Judge([matching({"PRIVATE_ALPHA": True}, ["PRIVATE_EXTRA"])]))
    result = result.evaluate("both", task_id, private_record=dataset.private_records[task_id])
    assert result["score"] == pytest.approx(2 / 3)
    assert not result["fully_correct"]
    assert result["correct_with_extraneous"]


@pytest.mark.parametrize("response", ["not JSON PRIVATE_REFERENCE", {"Answer Correctness": {}},
    matching({"PRIVATE_ALPHA": 1}), matching(extras=["duplicate", "duplicate"]),
    RuntimeError("PRIVATE_REFERENCE from a transport failure"),
    '{"Answer Correctness":{"Explanation":"x","Correctness Details":{"a":true,"a":false},"Excessive Answers":[]}}'])
def test_deepsearchqa_invalid_or_failed_judge_is_incomplete_not_zero(tmp_path, response):
    dataset = load_benchmark("deepsearchqa", dataset_file(tmp_path, "dsqa", [dsqa_row()]))
    task_id = next(iter(dataset.tasks))
    model, judge, ledger = metered([response])
    result = dataset.evaluator(model).evaluate("answer", task_id, private_record=dataset.private_records[task_id])
    assert result["score"] is None and not result["complete"]
    assert len(judge.calls) == ledger.snapshot()["model_calls"] == 1
    assert "PRIVATE" not in json.dumps(result)
    assert convert_feedback(result).rubrics[0].status == "error"


def test_empty_submission_does_not_call_judge(tmp_path):
    dataset = load_benchmark("deepsearchqa", dataset_file(tmp_path, "dsqa", [dsqa_row()]))
    task_id = next(iter(dataset.tasks))
    judge = Judge([])
    result = dataset.evaluator(judge).evaluate("", task_id, private_record=dataset.private_records[task_id])
    assert not result["complete"] and result["model_calls"] == 0 and not judge.calls


def test_drbii_three_way_score_does_not_subtract_blocked(tmp_path):
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [drb_row()]))
    task_id = next(iter(dataset.tasks))
    criteria = ["PRIVATE reference coverage", "PRIVATE analysis", "PRIVATE clarity"]
    response = {"results": [{"rubric_item": item, "score": native, "reason": "Synthetic",
                             "evidence": "report sentence" if native else ""}
                            for item, native in zip(criteria, [1, -1, 0])]}
    model, judge, ledger = metered([response])
    result = dataset.evaluator(model).evaluate("report sentence", task_id,
                                               private_record=dataset.private_records[task_id])
    assert result["complete"] and result["score"] == pytest.approx(1 / 3)
    assert result["blocked_rate"] == pytest.approx(1 / 3)
    assert result["native_score_counts"] == {"-1": 1, "0": 1, "1": 1}
    assert [r["score"] for r in result["feedback"]] == [1, 0, 0]
    assert [r["native_score"] for r in result["feedback"]] == [1, -1, 0]
    assert ledger.snapshot()["tokens"] == 120
    assert convert_feedback(result).score == pytest.approx(1 / 3)


def test_drbii_batches_exactly_fifty_and_retains_duplicate_criteria(tmp_path):
    row = drb_row()
    criteria = ["Repeated item"] * 2 + [f"Criterion {i}" for i in range(49)]
    row["content"]["rubric"] = {"info_recall": criteria, "analysis": [], "presentation": []}
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [row]))
    replies = [{"results": [{"rubric_item": item, "score": 1, "reason": "ok", "evidence": "text"}
                             for item in batch]} for batch in (criteria[:50], criteria[50:])]
    model, judge, ledger = metered(replies)
    task_id = next(iter(dataset.tasks))
    result = dataset.evaluator(model).evaluate("text", task_id, private_record=dataset.private_records[task_id])
    assert result["complete"] and result["score"] == 1
    assert len(result["feedback"]) == 51
    assert result["judged_item_count"] == 51 and result["criterion_count"] == 50
    assert result["duplicate_item_count"] == 1
    assert len({row["rubric_id"] for row in result["feedback"]}) == 51
    assert len(judge.calls) == ledger.snapshot()["model_calls"] == result["model_calls"] == 2
    assert len(json.loads(judge.calls[0][0][1]["content"])["rubric_items"]) == 50


def test_drbii_repeated_criterion_uses_last_native_judgment_like_author_dictionary(tmp_path):
    row = drb_row()
    row["content"]["rubric"] = {"info_recall": ["same", "same", "other"], "analysis": [], "presentation": []}
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [row]))
    task_id = next(iter(dataset.tasks))
    reply = {"results": [{"rubric_item": item, "score": score, "reason": "r", "evidence": ""}
                         for item, score in [("same", 1), ("same", -1), ("other", 1)]]}
    result = dataset.evaluator(Judge([reply])).evaluate("text", task_id,
                                                       private_record=dataset.private_records[task_id])
    assert result["score"] == .5 and result["blocked_rate"] == .5
    assert result["dimensions"]["info_recall"]["item_count"] == 2


@pytest.mark.parametrize("bad_score", [True, 1.5, 2, "1"])
def test_drbii_rejects_invalid_native_score_without_fabricated_score(tmp_path, bad_score):
    row = drb_row()
    row["content"]["rubric"] = {"info_recall": ["item"], "analysis": [], "presentation": []}
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [row]))
    task_id = next(iter(dataset.tasks))
    reply = {"results": [{"rubric_item": "item", "score": bad_score, "reason": "r", "evidence": "e"}]}
    result = dataset.evaluator(Judge([reply])).evaluate("text", task_id,
                                                       private_record=dataset.private_records[task_id])
    assert not result["complete"] and result["score"] is None
    assert result["blocked_rate"] is None
    assert result["feedback"][0]["native_score"] is None


def test_drbii_missing_or_repeated_judge_item_rejected(tmp_path):
    dataset = load_benchmark("deepresearch_bench_ii", dataset_file(tmp_path, "drb", [drb_row()]))
    task_id = next(iter(dataset.tasks))
    reply = {"results": [{"rubric_item": "PRIVATE reference coverage", "score": 1,
                          "reason": "ok", "evidence": "text"}] * 3}
    result = dataset.evaluator(Judge([reply])).evaluate("text", task_id,
                                                       private_record=dataset.private_records[task_id])
    assert not result["complete"] and result["failed_count"] == 3


@pytest.mark.parametrize("adapter", [DeepSearchQAEvaluator, DeepResearchBenchIIEvaluator])
def test_adapters_require_injected_judge_and_identity(adapter):
    with pytest.raises(ValueError, match="injected metered judge"):
        adapter().evaluate("answer", "task", private_record={"task_id": "task"})
    with pytest.raises(ValueError, match="identity mismatch"):
        adapter(Judge([])).evaluate("answer", "different", private_record={"task_id": "task"})
    first = adapter(Judge([]), judge_id="a")
    assert first.evaluator_version != adapter(Judge([]), judge_id="b").evaluator_version


@pytest.mark.parametrize("name", ["deepsearchqa", "deepresearch_bench_ii"])
def test_synthetic_vertical_generation_attribution_direct_write_and_heldout(name, tmp_path):
    from jit_mas.bridge import JITHarnessSynthesizer
    from jit_mas.config import MASConfig
    from jit_mas.experience import ExperienceStore
    from jit_mas.offline import FixtureModels
    from jit_mas.pipeline import MASPipeline
    from jit_mas.schemas import SplitManifest, digest

    rows = ([dsqa_row("Compare synthetic source designs."), dsqa_row("Compare synthetic heldout designs.")]
            if name == "deepsearchqa" else [drb_row(1), drb_row(2)])
    dataset = load_benchmark(name, dataset_file(tmp_path, name, rows))
    first, second = dataset.tasks

    class MarkerJudge:
        def __call__(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "deepsearchqa":
                found = "Boundary conditions:" in payload["submission"]
                return matching({"PRIVATE_ALPHA": found, "PRIVATE_BETA": found})
            found = "Boundary conditions:" in payload["report"]
            return {"results": [{"rubric_item": item, "score": int(found), "reason": "Synthetic marker",
                                 "evidence": "Boundary conditions:" if found else ""}
                                for item in payload["rubric_items"]]}

        def get_token_counts(self):
            return {"input_token_count": 100, "output_token_count": 20}

    class Models(FixtureModels):
        def create(self, role, agent_id, ledger, stage):
            if role == "judge":
                return MeteredModel(MarkerJudge(), ledger, stage, agent_id)
            return super().create(role, agent_id, ledger, stage)

    provider = Models()
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = MASPipeline(MASConfig(backend="scripted"), provider,
            lambda judge: dataset.evaluator(judge, judge_id="synthetic-marker-only"),
            lambda meta: JITHarnessSynthesizer(backend="scripted", meta_model=meta),
            dataset.tasks, dataset.private_records, SplitManifest(evolution=[first], test=[second]),
            store, tmp_path / "runs")
        source = pipeline.run("evolve")[0]
        assert source["evaluation"]["complete"] and source["experience_updates"]
        assert store.snapshot().version == 1
        frozen_hash = digest(store.snapshot())
        heldout = pipeline.run("evaluate")[0]
        assert heldout["evaluation"]["complete"] and heldout["proposals"] == []
        assert digest(store.snapshot()) == frozen_hash
        assert heldout["budget"]["by_stage"]["evaluation"]["model_calls"] == 1
        for call in provider.calls:
            if call["role"] == "exec":
                assert "PRIVATE" not in json.dumps(call)
        assert source["software_test_only"] and heldout["software_test_only"]
    finally:
        store.close()
