import json
from pathlib import Path

import pytest
import yaml

from jit_mas.config import MASConfig
from jit_mas.experiment_profile import validate_execution_profile


def profile():
    root = Path(__file__).resolve().parents[2]
    suite = json.loads((root / "paper/experiments/benchmark_suite_v3.json").read_text())
    config = MASConfig.model_validate(yaml.safe_load((root / "configs/benchmark_suite_v3.yaml").read_text()))
    return config, suite


def test_registered_config_uses_same_envelope_and_large_batch_judge_cap():
    config, suite = profile()
    assert validate_execution_profile(config, suite) is config
    assert config.models["judge"].max_tokens == 16000
    assert config.max_tool_calls == 0
    assert config.unsafe_local is False


@pytest.mark.parametrize("change", ["calls", "judge_cap", "judge_model", "tools", "temperature"])
def test_profile_drift_is_rejected_before_requests(change):
    config, suite = profile()
    if change == "calls":
        config.max_model_calls += 1
    elif change == "judge_cap":
        config.models["judge"].max_tokens = 4096
    elif change == "judge_model":
        config.models["judge"].model = "silent-substitution"
    elif change == "tools":
        config.available_tools = ["web_search"]
    else:
        config.models["exec"].temperature = 0.5
    with pytest.raises(ValueError):
        validate_execution_profile(config, suite)
