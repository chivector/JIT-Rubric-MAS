"""Offline software tests; the tiny synthetic tasks are not benchmark results."""

import ast
import copy
import json
import re
from pathlib import Path
from threading import Barrier, Event

import pytest

from benchmark.adapter.researchrubrics import (
    ResearchRubricsAdapter,
    official_compliance_score,
    split_item,
)

FIXTURES = Path(__file__).parent / "fixtures"


def raw_task(weights=(5, -3)):
    return {
        "sample_id": "synthetic",
        "prompt": "Explain the public requirement.",
        "explicit_constraints": ["Public explicit constraint"],
        "rubrics": [
            {"criterion": f"PRIVATE_CRITERION_{i}", "weight": w, "axis": "Implicit Criteria"}
            for i, w in enumerate(weights)
        ],
        "reference_answer": "PRIVATE_REFERENCE",
    }


def verdict(score=1, quotes=None):
    return {
        "verdict": "Satisfied" if score else "Not Satisfied",
        "score": float(score),
        "confidence": 0.7,
        "reasoning": "The answer contains the requested content.",
        "evidence_quotes": quotes or [],
        "missing_elements": [],
    }


class Judge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def __call__(self, messages, **kwargs):
        self.inputs.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def get_token_counts(self):
        return {"input_token_count": 20, "output_token_count": 10}


def upstream_scores(rows):
    """Run the actual pinned main() body, stopping before its print-only output."""
    tree = ast.parse((FIXTURES / "researchrubrics_official_compliance.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    begin = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "compliance_scores")
    scope = {"sample_results": {"synthetic": rows}}
    statements = ast.Module(body=main.body[begin:begin + 2], type_ignores=[])
    exec(compile(statements, "pinned_upstream_calculation", "exec"), scope)
    return scope["compliance_scores"]["synthetic"]


@pytest.mark.parametrize("rows", [
    [{"weight": 5, "score": 1}, {"weight": -3, "score": 1}],
    [{"weight": 2, "score": 0}, {"weight": -5, "score": 1}],
    [{"weight": -5, "score": 1}, {"weight": 0, "score": 0}],
    [{"weight": 5, "score": 0, "success": False}, {"weight": 3, "score": 1}],
    [],
])
def test_aggregation_matches_pinned_official_function(rows):
    assert official_compliance_score(rows) == upstream_scores(rows)


def test_signed_score_is_never_clipped():
    assert official_compliance_score([{"weight": 1, "score": 0}, {"weight": -5, "score": 1}]) == -5


def test_public_loader_never_exports_hidden_record():
    adapter = ResearchRubricsAdapter()
    public = adapter.load_dataset(str(FIXTURES / "researchrubrics_synthetic.jsonl"))
    exported = json.dumps(public)
    for canary in ("HIDDEN_RUBRIC_CANARY", "HIDDEN_NEGATIVE_CANARY", "PRIVATE_REFERENCE_CANARY"):
        assert canary not in exported
        assert canary not in adapter.format_task(public[0])
    assert "Use the supplied evidence" in adapter.format_task(public[0])
    private = adapter.private_record(public[0]["task_id"])
    assert "HIDDEN_RUBRIC_CANARY" in json.dumps(private)
    private["rubrics"].clear()
    assert adapter.private_record(public[0]["task_id"])["rubrics"]


def test_evaluation_retains_every_official_field_and_real_evidence_offsets():
    _, private = split_item(raw_task())
    judge = Judge([verdict(1, ["evidence", "made-up quotation"]), verdict(1)])
    adapter = ResearchRubricsAdapter(judge=judge, judge_id="offline-scripted-v1")
    result = adapter.evaluate("real evidence here", "synthetic", private_record=private)
    assert result["score"] == 0.4
    assert result["complete"] is True
    assert result["model_calls"] == 2
    assert result["input_token_count"] == 40
    assert result["cost"] is None
    first = result["feedback"][0]
    assert first["criterion"] == private["rubrics"][0]["criterion"]
    assert first["raw_result"]["confidence"] == 0.7
    assert first["evidence_locations"][0] == {
        "quote": "evidence", "verified": True, "start": 5, "end": 13,
    }
    assert first["evidence_locations"][1]["verified"] is False
    assert all("evaluator_version" in row for row in result["feedback"])
    assert "PRIVATE_REFERENCE" not in json.dumps(judge.inputs)


def test_failure_stays_in_denominator_and_blocks_complete():
    _, private = split_item(raw_task((5, 3)))
    adapter = ResearchRubricsAdapter(judge=Judge([RuntimeError("offline failure"), verdict(1)]), max_attempts=1)
    result = adapter.evaluate("answer", "synthetic", private_record=private)
    assert result["score"] == 3 / 8
    assert result["denominator"] == 8
    assert result["complete"] is False
    assert result["failed_count"] == 1
    assert result["feedback"][0]["score"] == 0
    assert result["feedback"][0]["verdict"] == "Error"
    assert result["usage_known"] is False
    assert result["input_token_count"] == 20


def test_retry_protocol_is_bounded_and_counts_invalid_json_tokens(monkeypatch):
    _, private = split_item(raw_task((5,)))
    judge = Judge(["not JSON", verdict(1)])
    monkeypatch.setattr("benchmark.adapter.researchrubrics.time.sleep", lambda _: None)
    result = ResearchRubricsAdapter(judge=judge).evaluate("answer", private_record=private)
    assert result["complete"]
    assert result["model_calls"] == 2
    assert result["input_token_count"] == 40
    failed = ResearchRubricsAdapter(judge=Judge(["bad"] * 3)).evaluate("answer", private_record=private)
    assert not failed["complete"]
    assert failed["model_calls"] == 3


@pytest.mark.parametrize("broken", [
    {"score": 0.5}, {"verdict": "Not Satisfied"}, {"confidence": 2},
    {"score": float("nan")}, {"evidence_quotes": "not a list"},
])
def test_invalid_judge_output_is_incomplete(broken):
    _, private = split_item(raw_task((5,)))
    response = {**verdict(), **broken}
    result = ResearchRubricsAdapter(judge=Judge([response]), max_attempts=1).evaluate("answer", private_record=private)
    assert result["complete"] is False
    assert result["score"] == 0
    json.dumps(result, allow_nan=False)


def test_zero_denominator_preserves_upstream_zero_and_flag():
    _, private = split_item(raw_task((-5, 0)))
    result = ResearchRubricsAdapter(judge=Judge([verdict(), verdict()])).evaluate("answer", private_record=private)
    assert result["score"] == 0.0
    assert result["zero_denominator"] is True
    assert result["complete"] is True
    assert result["is_pass"] is False


def test_no_credentials_fail_explicitly_without_api_calls():
    _, private = split_item(raw_task())
    with pytest.raises(ValueError, match="explicit judge"):
        ResearchRubricsAdapter().evaluate("answer", private_record=private)


def test_too_long_document_is_neither_truncated_nor_judged():
    _, private = split_item(raw_task((5,)))
    judge = Judge([])
    result = ResearchRubricsAdapter(judge=judge, max_document_chars=2).evaluate("long", private_record=private)
    assert not result["complete"]
    assert result["model_calls"] == 0
    assert not judge.inputs


def test_public_evaluation_path_constructs_registered_adapter():
    from benchmark.registry import get
    from scripts.eval.config import load_yaml_config
    from scripts.eval.runner import make_adapter

    config = load_yaml_config(get("researchrubrics").config)
    judge = Judge([verdict(), verdict(0)])
    config["benchmark"]["adapter_kwargs"]["judge"] = judge
    adapter = make_adapter(config)
    items = adapter.load_dataset(str(FIXTURES / "researchrubrics_synthetic.jsonl"))
    result = adapter.evaluate("answer", items[0]["answer"], item=items[0])
    assert result["complete"]
    assert result["score"] == 1.0


def test_judge_never_inherits_execution_credentials():
    from scripts.eval.runner import _adapter_kwargs

    kwargs = _adapter_kwargs({
        "benchmark": {"name": "researchrubrics"},
        "model": {"model_id": "executor", "api_base": "executor-base", "api_key": "executor-key"},
    })
    assert kwargs["judge_model"] == kwargs["judge_api_base"] == kwargs["judge_api_key"] == ""


def test_duplicate_ids_or_prompts_are_rejected(tmp_path):
    path = tmp_path / "items.jsonl"
    first, second = raw_task(), raw_task()
    second["sample_id"] = "other-id"
    second["prompt"] = "  EXPLAIN the public requirement. "
    path.write_text(json.dumps(first) + "\n" + json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ResearchRubricsAdapter().load_dataset(str(path))


def test_evaluator_identity_changes_with_model_and_policy():
    assert ResearchRubricsAdapter(judge_id="a").evaluator_version != ResearchRubricsAdapter(judge_id="b").evaluator_version
    assert ResearchRubricsAdapter(max_attempts=1).evaluator_version != ResearchRubricsAdapter(max_attempts=2).evaluator_version


def test_telemetry_failure_does_not_discard_criterion_feedback():
    class BrokenTelemetry(Judge):
        def get_token_counts(self):
            raise RuntimeError("No token telemetry")

    _, private = split_item(raw_task((5,)))
    result = ResearchRubricsAdapter(judge=BrokenTelemetry([verdict()])).evaluate("answer", private_record=private)
    assert result["complete"] is True
    assert result["score"] == 1
    assert result["usage_known"] is False
    assert result["cost"] is None


class CriterionJudge:
    def __init__(self, responses, barrier=None, second_finished=None):
        self.responses = responses
        self.barrier = barrier
        self.second_finished = second_finished
        self.inputs = []
        self.counts = {}

    def __call__(self, messages, **kwargs):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        criterion_index = int(re.search(r"PRIVATE_CRITERION_(\d+)", messages[1]["content"]).group(1))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
            if criterion_index == 0:
                assert self.second_finished.wait(timeout=5)
            else:
                self.second_finished.set()
        self.counts = {
            "input_token_count": 17 + criterion_index * 12,
            "output_token_count": 3 + criterion_index * 2,
        }
        response = self.responses[criterion_index]
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)

    def get_token_counts(self):
        return dict(self.counts)


def stable_judgment(result):
    result = copy.deepcopy(result)
    result.pop("evaluator_version", None)
    result.pop("judgment_execution", None)
    for row in result["feedback"]:
        row.pop("duration", None)
        row.pop("evaluator_version", None)
    return result


@pytest.mark.parametrize("responses, expected_score, expected_complete", [
    ([verdict(0), verdict(1)], -2.5, True),
    ([RuntimeError("offline failure"), verdict(1)], -2.5, False),
    (["invalid JSON", verdict(1)], -2.5, False),
])
def test_parallel_judgments_preserve_prompt_order_usage_and_signed_scoring(responses, expected_score, expected_complete):
    _, private = split_item(raw_task((2, -5)))
    sequential_judge = CriterionJudge(responses)
    sequential = ResearchRubricsAdapter(
        judge=sequential_judge, judge_id="offline-criteria-v1", max_attempts=1,
    ).evaluate("real evidence here", private_record=private)
    barrier = Barrier(2)
    second_finished = Event()
    judges = []

    def judge_factory():
        judge = CriterionJudge(responses, barrier, second_finished)
        judges.append(judge)
        return judge

    adapter = ResearchRubricsAdapter(
        judge_factory=judge_factory, max_parallel_judgments=2,
        judge_id="offline-criteria-v1", max_attempts=1,
    )
    parallel = adapter.evaluate("real evidence here", private_record=private)
    assert stable_judgment(parallel) == stable_judgment(sequential)
    assert parallel["score"] == expected_score
    assert parallel["complete"] is expected_complete
    assert parallel["model_calls"] == 2
    assert [row["rubric_id"] for row in parallel["feedback"]] == [row["rubric_id"] for row in private["rubrics"]]
    assert parallel["input_token_count"] == (29 if isinstance(responses[0], Exception) else 46)
    assert parallel["output_token_count"] == (5 if isinstance(responses[0], Exception) else 8)
    assert len(judges) == 2
    assert [judge.inputs[0] for judge in judges] == sequential_judge.inputs
    assert parallel["judgment_execution"]["max_parallel_judgments"] == 2
    assert parallel["judgment_execution"]["judge_ownership"] == "per_rubric"
    assert parallel["evaluator_version"] != sequential["evaluator_version"]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_parallel_judgments_require_positive_integer_concurrency(workers):
    with pytest.raises(ValueError, match="positive integer"):
        ResearchRubricsAdapter(judge_factory=lambda: Judge([]), max_parallel_judgments=workers)


def test_parallel_judgments_require_independent_factory_models():
    with pytest.raises(ValueError, match="require a judge_factory"):
        ResearchRubricsAdapter(judge=Judge([]), max_parallel_judgments=2)
    with pytest.raises(ValueError, match="must be callable"):
        ResearchRubricsAdapter(judge_factory="invalid")
    _, private = split_item(raw_task())
    shared_judge = Judge([])
    adapter = ResearchRubricsAdapter(judge_factory=lambda: shared_judge, max_parallel_judgments=2)
    with pytest.raises(ValueError, match="independent judge"):
        adapter.evaluate("answer", private_record=private)
    assert not shared_judge.inputs
    adapter = ResearchRubricsAdapter(judge_factory=lambda: None, max_parallel_judgments=2)
    with pytest.raises(ValueError, match="return callable"):
        adapter.evaluate("answer", private_record=private)


def test_factory_execution_identity_records_concurrency_and_factory():
    def factory():
        return Judge([verdict()])

    def other_factory():
        return Judge([verdict()])

    versions = {
        ResearchRubricsAdapter(judge_factory=factory, max_parallel_judgments=workers).evaluator_version
        for workers in (1, 2, 3)
    }
    assert len(versions) == 3
    assert ResearchRubricsAdapter(judge_factory=other_factory, max_parallel_judgments=2).evaluator_version not in versions
    assert ResearchRubricsAdapter().evaluator_version == ResearchRubricsAdapter(max_parallel_judgments=1).evaluator_version
