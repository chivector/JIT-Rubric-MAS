import json
from pathlib import Path

import pytest

from scripts.preflight_independent_experiment import run_preflight


def test_independent_preflight_expands_endpoints_and_runs_offline_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("MAS_GENERATOR_ENDPOINT", "https://generator.synthetic.invalid/v1")
    monkeypatch.setenv("MAS_JUDGE_ENDPOINT", "https://judge.synthetic.invalid/v1")
    report = run_preflight(
        Path("configs/jit_mas.independent_v5.yaml"),
        tmp_path / "preflight.json",
    )
    assert report["passed"] is True
    assert report["synthetic_only"] is True
    assert report["network_requests"] == 0
    assert report["software"]["fixture_execution_mode"] == "single_pass"
    assert report["served_models"]["meta"]["synthetic"] is True
    saved = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert saved["report_sha256"] == report["report_sha256"]


def test_independent_preflight_rejects_unexpanded_endpoint(monkeypatch, tmp_path):
    monkeypatch.delenv("MAS_GENERATOR_ENDPOINT", raising=False)
    monkeypatch.delenv("MAS_JUDGE_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="Missing expanded endpoint"):
        run_preflight(Path("configs/jit_mas.independent_v5.yaml"),
                      tmp_path / "preflight.json")
