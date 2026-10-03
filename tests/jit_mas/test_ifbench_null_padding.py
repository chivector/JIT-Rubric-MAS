"""Pinned author integration with schema padding, using synthetic tasks only."""

import copy
from pathlib import Path

import pytest

from jit_mas.benchmarks import IFBenchEvaluator
from jit_mas.instruction_checkers import PinnedInstructionChecker


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
GOOGLE = ROOT / "outputs/independent_v5_preparation/checkers/Google-IFEval-e49bbfe381c9c0e564b937f1c4e163a2273c65cc"


@pytest.fixture
def checker():
    return PinnedInstructionChecker(SOURCE, "ifbench")


def record(ids, options):
    return {"task_id": "synthetic-instruction-task", "prompt": "Write a synthetic completion.",
            "instruction_id_list": ids, "kwargs": options}


@pytest.mark.parametrize("ids,options,answer", [
    (["count:numbers"], [{"N": 2.0, "keyword": None, "small_n": None}], "There are 1 and 2 objects."),
    (["count:conjunctions"], [{"small_n": 2.0, "N": None, "keyword": None}], "We read and write or rest."),
    (["words:keywords_specific_position"], [{"keyword": "slowly", "n": 2, "m": 3, "N": None}],
     "First scene opens. Words move slowly through this scene."),
])
def test_official_loose_signature_accepts_only_non_null_padding(checker, ids, options, answer):
    private = record(ids, options)
    before = copy.deepcopy(private)
    original = checker.author.InputExample(key=0, instruction_id_list=ids,
                                          prompt=private["prompt"], kwargs=copy.deepcopy(options))
    with pytest.raises(TypeError):
        checker.author.test_instruction_following_loose(original, {private["prompt"]: answer})
    assert checker.check_record(answer, private) == [True]
    assert private == before
    assert checker.identity["input_kwargs_policy"]


def test_non_null_unknown_constraint_still_fails_closed(checker):
    private = record(["count:numbers"], [{"N": 2.0, "unknown_required_constraint": "required"}])
    with pytest.raises(TypeError):
        checker.check_record("There are 1 and 2 objects.", private)


def test_primary_prompt_accuracy_is_not_secondary_instruction_fraction(checker):
    private = record(["count:numbers", "count:conjunctions"],
                     [{"N": 2.0, "small_n": None}, {"small_n": 2.0, "N": None}])
    evaluator = IFBenchEvaluator(judge=None, checker=checker)
    result = evaluator.evaluate("There are 1 and 2 objects.", private["task_id"], private_record=private)
    assert result["complete"] and result["score"] == 0
    assert result["instruction_accuracy"] == 0.5
    assert result["pass_count"] == 1 and result["check_count"] == 2
    assert result["model_calls"] == 0


def test_ifeval_strict_integration_remains_unchanged():
    checker = PinnedInstructionChecker(GOOGLE, "ifeval")
    private = record(["keywords:frequency"], [{"keyword": "clock", "frequency": 2, "relation": "at least"}])
    assert "input_kwargs_policy" not in checker.identity
    assert checker.check_record("A clock is near another clock.", private) == [True]
    assert checker.check_record("A clock is alone.", private) == [False]
