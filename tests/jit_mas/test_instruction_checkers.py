"""Synthetic author-checker integration and immutable-source validation."""

import hashlib
from types import SimpleNamespace

import pytest

from jit_mas.instruction_checkers import PinnedInstructionChecker, _verify_files
from jit_mas.benchmarks import _checker_results


def test_author_checker_source_verification_rejects_modified_bytes(tmp_path):
    path = tmp_path / "author.py"
    path.write_bytes(b"synthetic pinned implementation")
    expected = {"author.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    _verify_files(tmp_path, expected)
    path.write_bytes(b"changed implementation")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_files(tmp_path, expected)


def test_record_checker_receives_prompt_without_mutating_private_record():
    record = {"prompt": "Synthetic prompt", "instruction_id_list": ["synthetic:prompt"],
              "kwargs": [{"prompt": None}]}

    class Checker:
        def check_record(self, prediction, private):
            assert private["prompt"] == "Synthetic prompt"
            private["kwargs"][0]["prompt"] = "Changed by author implementation"
            return [True]

    assert _checker_results("Synthetic answer", record, Checker()) == [True]
    assert record["kwargs"] == [{"prompt": None}]


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
def test_pinned_wrapper_calls_matching_author_metric_with_original_prompt(benchmark, monkeypatch):
    calls = []

    def evaluate(example, responses):
        calls.append((example.prompt, responses, example.kwargs))
        example.kwargs[0]["synthetic"] = "mutation"
        return SimpleNamespace(follow_instruction_list=[True])

    author = SimpleNamespace(InputExample=SimpleNamespace,
                             test_instruction_following_strict=evaluate,
                             test_instruction_following_loose=evaluate)
    checker = PinnedInstructionChecker.__new__(PinnedInstructionChecker)
    checker.author = author
    checker.benchmark = benchmark
    record = {"prompt": "Original synthetic prompt", "instruction_id_list": ["synthetic:one"],
              "kwargs": [{"synthetic": None}]}
    monkeypatch.setitem(__import__("sys").modules, "langdetect", SimpleNamespace(
        DetectorFactory=SimpleNamespace(seed=None)))
    assert checker.check_record("Synthetic response", record) == [True]
    assert calls[0][0] == "Original synthetic prompt"
    assert calls[0][1] == {"Original synthetic prompt": "Synthetic response"}
    assert record["kwargs"] == [{"synthetic": None}]
