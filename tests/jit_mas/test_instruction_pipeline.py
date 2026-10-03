"""Offline instruction benchmark entry points and scorer identity boundaries."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from jit_mas.benchmarks import ALL_BENCHMARK_NAMES, IFEvalEvaluator
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.experience import ExperienceStore
from jit_mas.instruction_checkers import PinnedInstructionChecker, _resource_hashes
from jit_mas.offline import FixtureModels
from scripts import run_jit_mas as runner


class SyntheticChecker:
    def __init__(self, revision="synthetic-v1"):
        self.identity = {"source": "synthetic-software-fixture", "revision": revision}
        self.calls = []

    def check_record(self, prediction, private):
        self.calls.append((prediction, copy.deepcopy(private)))
        return [True] * len(private["instruction_id_list"])


@pytest.mark.parametrize("benchmark", ALL_BENCHMARK_NAMES)
def test_cli_accepts_every_registered_benchmark_and_forwards_checker_source(
        tmp_path, monkeypatch, capsys, benchmark):
    captures = []
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(MASConfig, "check_native", lambda self: None)

    def make_pipeline(config, store, output, **kwargs):
        captures.append(kwargs)
        return SimpleNamespace(run=lambda *args, **options: [])

    monkeypatch.setattr(runner, "make_pipeline", make_pipeline)
    source = tmp_path / "prepared-author-source"
    arguments = ["--mode", "evaluate", "--benchmark", benchmark,
                 "--state", str(tmp_path / "state.sqlite"), "--output", str(tmp_path / "runs")]
    if benchmark in {"ifeval", "ifbench"}:
        arguments.extend(["--instruction-checker-source", str(source)])
    assert runner.main(arguments) == 0
    assert captures[0]["benchmark"] == benchmark
    assert captures[0]["checker_source_root"] == (str(source) if benchmark in {"ifeval", "ifbench"} else None)
    assert json.loads(capsys.readouterr().out)["tasks"] == 0


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
def test_pipeline_initializes_local_checker_and_keeps_private_metadata_out_of_public_tasks(
        tmp_path, monkeypatch, benchmark):
    row = {"key": 1, "prompt": "Write a synthetic response mentioning a clock.",
           "instruction_id_list": ["PRIVATE_INSTRUCTION_CANARY"],
           "kwargs": [{"private": "PRIVATE_OPTION_CANARY"}]}
    data = tmp_path / "synthetic.jsonl"
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    task_id = benchmark + ":1"
    splits.write_text(json.dumps({"test": [task_id]}), encoding="utf-8")
    checker = SyntheticChecker()
    constructions = []

    def local_checker(source, selected_benchmark):
        constructions.append((source, selected_benchmark))
        return checker

    monkeypatch.setattr("jit_mas.instruction_checkers.PinnedInstructionChecker", local_checker)
    monkeypatch.setattr(runner, "NativeModels", lambda config: SimpleNamespace())
    config = MASConfig(models={"judge": ModelConfig(model="unused-synthetic-judge")})
    source = tmp_path / "prepared-author-source"
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = runner.make_pipeline(config, store, tmp_path / "runs", data=data, splits=splits,
                                        benchmark=benchmark, checker_source_root=source)
        public = json.dumps(pipeline.tasks[task_id].model_dump())
        assert "PRIVATE_" not in public
        assert constructions == [(source, benchmark)]

        def forbidden_judge(*args, **kwargs):
            raise AssertionError("Instruction evaluation must not call a model judge")

        evaluation = pipeline.evaluator_factory(forbidden_judge).evaluate(
            "The clock keeps time.", task_id, private_record=pipeline.private_records[task_id])
        assert evaluation["complete"] is True
        assert evaluation["score"] == 1.0
        assert evaluation["model_calls"] == 0
        assert evaluation["checker_identity"] == checker.identity
        assert checker.calls[0][1]["instruction_id_list"] == ["PRIVATE_INSTRUCTION_CANARY"]
    finally:
        store.close()


@pytest.mark.parametrize("benchmark", ["ifeval", "ifbench"])
def test_missing_or_unprepared_checker_fails_before_native_model_setup(tmp_path, monkeypatch, benchmark):
    def forbidden_provider(config):
        raise AssertionError("Invalid checker resources must be rejected before model setup")

    monkeypatch.setattr(runner, "NativeModels", forbidden_provider)
    common = {"data": tmp_path / "data.jsonl", "splits": tmp_path / "splits.json", "benchmark": benchmark}
    with pytest.raises(ValueError, match="require an injected checker"):
        runner.make_pipeline(MASConfig(), None, tmp_path / "runs", **common)
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner.make_pipeline(MASConfig(), None, tmp_path / "runs", checker_source_root=tmp_path / "missing", **common)


def test_checker_identity_invalidates_pipeline_cache_without_entering_actor_context(tmp_path, monkeypatch):
    from scripts.models.openai_server import OpenAIServerModel

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Synthetic integration must not contact a model API")

    monkeypatch.setattr(OpenAIServerModel, "__call__", forbidden_network)
    store = ExperienceStore(tmp_path / "state.sqlite")
    provider = FixtureModels()
    try:
        pipeline = runner.make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs",
                                        fixture_models=provider)
        task_id = "test-poem"
        pipeline.private_records[task_id] = {"task_id": task_id, "prompt": pipeline.tasks[task_id].question,
                                            "instruction_id_list": ["PRIVATE_CANARY"], "kwargs": [{}]}
        checker = SyntheticChecker()
        pipeline.evaluator_factory = lambda judge: IFEvalEvaluator(checker=checker)
        first = pipeline.run("evaluate", [task_id])[0]
        repeated = pipeline.run("evaluate", [task_id])[0]
        assert repeated["resumed"] is True
        assert repeated["run_key"] == first["run_key"]
        checker = SyntheticChecker("synthetic-v2")
        changed = pipeline.run("evaluate", [task_id])[0]
        assert changed["run_key"] != first["run_key"]
        manifest = json.loads((Path(changed["run_dir"]) / "run_manifest.json").read_text())
        assert manifest["comparison"]["evaluator"] == changed["evaluation"]["evaluator_version"]
        assert changed["evaluation"]["raw"]["checker_identity"] == checker.identity
        assert not [call for call in provider.calls if call["role"] == "judge"]
        assert "PRIVATE_CANARY" not in json.dumps(provider.calls)
    finally:
        store.close()


def test_evaluator_rejects_changed_checker_identity_before_author_call():
    checker = SyntheticChecker()
    evaluator = IFEvalEvaluator(checker=checker)
    checker.identity["revision"] = "changed"
    result = evaluator.evaluate("Synthetic answer", "ifeval:1", private_record={
        "task_id": "ifeval:1", "instruction_id_list": ["synthetic:one"], "kwargs": [{}]})
    assert result["complete"] is False
    assert result["score"] is None
    assert checker.calls == []


def test_pinned_checker_detects_local_source_and_resource_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    author = source / "evaluation_lib.py"
    author.write_bytes(b"synthetic frozen author source")
    installed = tmp_path / "installed"
    installed.mkdir()
    package = installed / "__init__.py"
    package.write_bytes(b"synthetic frozen package")
    package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
    monkeypatch.setattr("jit_mas.instruction_checkers.PINNED_FILES", {"ifbench/__init__.py": package_hash})
    resources = ("tokenizers/punkt.zip", "tokenizers/punkt_tab.zip", "corpora/stopwords.zip",
                 "taggers/averaged_perceptron_tagger_eng.zip")
    for relative in resources:
        archive = installed / ".nltk_data" / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("synthetic.txt", b"frozen resource")
        (archive.parent / "synthetic.txt").write_bytes(b"frozen resource")
    checker = PinnedInstructionChecker.__new__(PinnedInstructionChecker)
    checker.source_root, checker.installed_root = source, installed
    checker.identity = {"code_sha256": {"evaluation_lib.py": hashlib.sha256(author.read_bytes()).hexdigest()},
                        "nltk_resources_sha256": _resource_hashes(installed), "dependencies": {}}
    checker._frozen_identity = copy.deepcopy(checker.identity)
    checker.assert_frozen()
    author.write_bytes(b"changed author source")
    with pytest.raises(ValueError, match="source hash mismatch"):
        checker.assert_frozen()
    author.write_bytes(b"synthetic frozen author source")
    archive = installed / ".nltk_data" / resources[0]
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("synthetic.txt", b"changed resource")
    with pytest.raises(ValueError, match="extracted NLTK resource"):
        checker.assert_frozen()
