import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.prepare_joint_bundle as module
from jit_mas.schemas import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/experiments/joint_protocol_v5.json"
MANIFEST = ROOT / "paper/experiments/joint_task_splits_v5.json"


def _ids(manifest, name):
    values = manifest["memberships"][name]
    return values["evolution"] + values["validation"] + values["test"]


def test_split_projection_preserves_frozen_membership():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    projection = module._split_projection(manifest, "ifeval")
    assert projection["evolution"] == []
    assert len(projection["validation"]) == 0
    assert len(projection["test"]) == 50
    assert projection["stream"] == []
    assert SplitManifest.model_validate(projection).test == projection["test"]


def test_bundle_builder_binds_all_six_sources_without_provider_calls(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = {}
    fake_datasets = {}
    for name in module.BENCHMARKS:
        path = tmp_path / f"{name}.data"
        path.write_text("placeholder\n", encoding="utf-8")
        data[name] = path
        fake_datasets[name] = SimpleNamespace(tasks={task: object() for task in _ids(manifest, name)})
    monkeypatch.setattr(module, "load_benchmark", lambda name, path: fake_datasets[name])
    real_file_hash = module.file_hash
    expected_hashes = {name: manifest["benchmarks"][name]["dataset_sha256"] for name in module.BENCHMARKS}
    monkeypatch.setattr(module, "file_hash", lambda path: expected_hashes[Path(path).stem.split(".")[0]] if Path(path) in data.values() else real_file_hash(path))

    evidence = {}
    for name in module.SOURCE_BENCHMARKS:
        directory = tmp_path / f"{name}.evidence"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"task_ids": _ids(manifest, name)}), encoding="utf-8")
        (directory / "pack.bin").write_bytes(b"immutable")
        evidence[name] = directory
    checker = {}
    for name in module.CHECKER_BENCHMARKS:
        directory = tmp_path / f"{name}.checker"
        directory.mkdir()
        (directory / "checker.py").write_text("# pinned\n", encoding="utf-8")
        checker[name] = directory
    config = tmp_path / "config.yaml"
    config.write_text("backend: native_jit\n", encoding="utf-8")
    output = tmp_path / "bundle"
    result = module.build_bundle(protocol=PROTOCOL, manifest=MANIFEST, config=config,
                                 output_dir=output, data=data, evidence=evidence,
                                 checker=checker,
                                 preflight=None, served_models=None)
    assert result["formal_ready"] is False  # preflight defaults to synthetic/incomplete
    assert set(result["benchmarks"]) == set(module.BENCHMARKS)
    assert (output / "bundle.json").is_file()
    for name in module.BENCHMARKS:
        split = json.loads((output / f"{name}.runtime_split.json").read_text(encoding="utf-8"))
        assert split["schema_version"] == "jit-compose-joint-benchmark-split-v5"


def test_incomplete_evidence_fails_closed(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data, fake = {}, {}
    for name in module.BENCHMARKS:
        path = tmp_path / f"{name}.data"
        path.write_text("placeholder\n", encoding="utf-8")
        data[name] = path
        fake[name] = SimpleNamespace(tasks={task: object() for task in _ids(manifest, name)})
    monkeypatch.setattr(module, "load_benchmark", lambda name, path: fake[name])
    real_file_hash = module.file_hash
    expected_hashes = {name: manifest["benchmarks"][name]["dataset_sha256"] for name in module.BENCHMARKS}
    monkeypatch.setattr(module, "file_hash", lambda path: expected_hashes[Path(path).stem.split(".")[0]] if Path(path) in data.values() else real_file_hash(path))
    evidence = {}
    for name in module.SOURCE_BENCHMARKS:
        directory = tmp_path / f"{name}.evidence"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"task_ids": []}), encoding="utf-8")
        (directory / "pack.bin").write_bytes(b"immutable")
        evidence[name] = directory
    checker = {}
    for name in module.CHECKER_BENCHMARKS:
        directory = tmp_path / f"{name}.checker"
        directory.mkdir()
        (directory / "checker.py").write_text("# pinned\n", encoding="utf-8")
        checker[name] = directory
    config = tmp_path / "config.yaml"
    config.write_text("backend: native_jit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence coverage incomplete"):
        module.build_bundle(protocol=PROTOCOL, manifest=MANIFEST, config=config,
                            output_dir=tmp_path / "bundle", data=data,
                            evidence=evidence, checker=checker)


def test_unknown_assignment_is_rejected():
    with pytest.raises(ValueError, match="Unknown --data benchmark"):
        module._parse_assignments(["unknown=/tmp/file"], allowed=set(module.BENCHMARKS), label="--data")
