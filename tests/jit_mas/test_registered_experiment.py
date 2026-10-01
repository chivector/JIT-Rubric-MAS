"""Registered CLI smoke and preflight guards; never issue real model requests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jit_mas.benchmarks import load_benchmark
from jit_mas.config import MASConfig
from jit_mas.experiment_splits import ORDER_SEEDS, build_split
from jit_mas.schemas import digest
from scripts import run_benchmark_experiment as cli


@pytest.fixture
def registration(tmp_path):
    rows = [{"problem": f"Find the date for public event {index}.",
             "problem_category": "History" if index % 2 else "Science",
             "answer_type": "Single Answer", "answer": f"PRIVATE_ANSWER_CANARY_{index}"}
            for index in range(6)]
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    dataset = load_benchmark("deepsearchqa", data)
    specification = {"expected_tasks": 6,
                     "counts": {"development": 1, "evolution": 2, "validation": 1, "test": 2},
                     "batch_size": 1, "split_seed": "synthetic-registration-software-test",
                     "official_split_policy": "Synthetic adapter test, not benchmark membership",
                     "methods": ["ours_selected", "global_only_planning"]}
    configuration = MASConfig(backend="native_jit")
    profile = {key: getattr(configuration, key) for key in (
        "max_agents", "max_parallel", "team_max_calls", "local_rounds", "candidates",
        "max_repairs", "max_model_calls", "max_total_tokens", "max_tool_calls", "execution_timeout")}
    profile.update(output_tokens={}, model="synthetic", judge_model="synthetic", temperature=0, request_timeout=120)
    suite = {"version": "jit-compose-benchmark-suite-v3", "order_seeds": list(ORDER_SEEDS),
             "benchmarks": {"deepsearchqa": specification},
             "release_hashes": {"deepsearchqa": dataset.dataset_sha256},
             "execution_profile": profile,
             "checkpoint_selection": {"minimum_complete_fraction": 0.9}}
    split = build_split(dataset, specification)
    suite_file, split_file = tmp_path / "suite.json", tmp_path / "split.json"
    suite_file.write_text(json.dumps(suite), encoding="utf-8")
    split_file.write_text(json.dumps(split), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(MASConfig(backend="native_jit").model_dump(mode="json")), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.json").write_text('{"software_test_only": true}', encoding="utf-8")
    return SimpleNamespace(data=data, dataset=dataset, suite=suite_file, splits=split_file,
                           config=config, evidence=evidence, split=split, tmp_path=tmp_path)


def arguments(registration, *, mode="evolve", extra=()):
    return cli.make_parser().parse_args([
        "--mode", mode, "--benchmark", "deepsearchqa", "--data", str(registration.data),
        "--suite", str(registration.suite), "--splits", str(registration.splits),
        "--config", str(registration.config), "--evidence-dir", str(registration.evidence),
        "--unsafe-local", "--output", str(registration.tmp_path / "run"), *extra])


def test_export_public_uses_typed_projection_without_private_fields(registration):
    args = arguments(registration, mode="export-public")
    args.output = str(registration.tmp_path / "public.json")
    report = cli.export_public(args)
    text = Path(args.output).read_text(encoding="utf-8")
    public = json.loads(text)
    assert isinstance(public, list) and len(public) == 6
    assert "PRIVATE_ANSWER_CANARY" not in text
    assert "answer_type" not in text and "problem_category" not in text
    assert all(set(row) == set(next(iter(registration.dataset.tasks.values())).model_dump()) for row in public)
    assert report["tasks"] == 6
    assert Path(report["identity"]).is_file()
    with pytest.raises(FileExistsError):
        cli.export_public(args)


def test_registration_binds_dataset_bytes_and_specification(registration):
    args = arguments(registration)
    cli.load_registration(args)
    with registration.data.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="pinned release"):
        cli.load_registration(args)


def test_registration_rejects_changed_specification(registration):
    suite = json.loads(registration.suite.read_text())
    suite["benchmarks"]["deepsearchqa"]["official_split_policy"] = "changed"
    registration.suite.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="specification"):
        cli.load_registration(arguments(registration))


@pytest.mark.parametrize("field,value,match", [
    ("unsafe_local", False, "unsafe-local"),
    ("evidence_dir", None, "evidence-dir"),
    ("config", None, "config"),
    ("run_id", 4, "run-id"),
])
def test_native_preflight_failures_before_model_construction(registration, monkeypatch, field, value, match):
    monkeypatch.setattr(cli, "make_pipeline", lambda *args, **kwargs: pytest.fail("Unexpected model construction"))
    args = arguments(registration)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        cli.build_environment(args)


def test_environment_uses_registered_source_order_and_full_validation(registration, monkeypatch):
    calls = []

    def pipeline(config, store, output, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(models=None)

    monkeypatch.setattr(cli, "make_pipeline", pipeline)
    args = arguments(registration, extra=["--run-id", "2"])
    environment = cli.build_environment(args)
    try:
        assert environment.evolution_ids == registration.split["evolution_schedule"][2]["task_ids"]
        assert environment.validation_ids == registration.split["runtime_split_manifest"]["validation"]
        assert environment.identity["order_seed"] == ORDER_SEEDS[2]
        assert environment.identity["evidence"]["files"]
        assert environment.identity["runner_sha256"]
        assert environment.identity["dataset_sha256"] == registration.dataset.dataset_sha256
        assert calls[0]["benchmark"] == "deepsearchqa"
        assert calls[0]["evidence_dir"] == str(registration.evidence)
    finally:
        environment.close()


def test_resume_rejects_changed_evidence_before_requests(registration, monkeypatch):
    monkeypatch.setattr(cli, "make_pipeline", lambda *args, **kwargs: SimpleNamespace(models=None))
    args = arguments(registration)
    first = cli.build_environment(args)
    first.close()
    (registration.evidence / "new-source.txt").write_text("changed public pack", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        cli.build_environment(args)


@pytest.mark.parametrize("target", ["data", "splits", "suite", "config", "evidence"])
def test_live_environment_rejects_input_drift_before_next_callback(registration, monkeypatch, target):
    monkeypatch.setattr(cli, "make_pipeline", lambda *args, **kwargs: SimpleNamespace(models=None))
    environment = cli.build_environment(arguments(registration))
    try:
        environment.assert_frozen()
        path = getattr(registration, target)
        if target == "evidence":
            path /= "manifest.json"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with pytest.raises(RuntimeError, match="frozen experiment"):
            environment.assert_frozen()
    finally:
        environment.close()


def test_rehashed_invalid_order_still_rejected(registration, monkeypatch):
    monkeypatch.setattr(cli, "make_pipeline", lambda *args, **kwargs: pytest.fail("Unexpected pipeline"))
    data = copy.deepcopy(registration.split)
    data["evolution_schedule"][0]["task_ids"] = [data["evolution_schedule"][0]["task_ids"][0]] * 2
    data["manifest_sha256"] = digest({key: value for key, value in data.items() if key != "manifest_sha256"})
    registration.splits.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once|source permutation"):
        cli.build_environment(arguments(registration))


def test_native_cannot_silently_use_scripted_config(registration):
    registration.config.write_text(yaml.safe_dump(MASConfig(backend="scripted").model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="native_jit"):
        cli.build_environment(arguments(registration))


def test_method_label_and_configuration_must_match(registration):
    args = arguments(registration, extra=["--method", "global_only_planning"])
    with pytest.raises(ValueError, match="local_planning"):
        cli.build_environment(args)


def test_smoke_native_input_mix_rejected(tmp_path):
    args = cli.make_parser().parse_args(["--mode", "evolve", "--smoke", "--output", str(tmp_path),
                                       "--data", "should-not-read.json"])
    with pytest.raises(ValueError, match="built-in synthetic"):
        cli.build_environment(args)


def test_smoke_real_generation_checkpoint_selection_and_status(tmp_path, monkeypatch, capsys):
    from scripts.models.openai_server import OpenAIServerModel

    monkeypatch.setattr(OpenAIServerModel, "__call__", lambda *args, **kwargs:
                        pytest.fail("Registered smoke must never issue a real API request"))
    output = tmp_path / "smoke"
    argv = ["--mode", "evolve", "--smoke", "--output", str(output)]
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["software_test_only"] is True
    assert report["paid_requests"] == 0
    assert report["status"] == "complete"
    assert [item["position"] for item in report["checkpoints"]] == [0, 1]
    assert report["accounting"]["actual_validation_slots"] == 8
    assert report["accounting"]["attempts_with_unknown_budget"] == 0
    assert cli.main(argv) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed == report
    assert cli.main(["--mode", "status", "--output", str(output)]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["source_slots"] == 1
    assert current["completed_checkpoints"] == [0, 1]
    assert current["selected_position"] == report["selected"]["position"]
