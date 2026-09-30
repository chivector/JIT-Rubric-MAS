"""Public entry points must import in fresh interpreters, regardless of ordering."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "jit_mas.offline", "scripts.models", "scripts.models.openai_server",
    "scripts.kernel", "scripts.run_jit_mas", "scripts.benchmark_jit_mas_live",
])
def test_independent_entrypoint_import(module):
    code = (
        f"import importlib; importlib.import_module({module!r}); "
        "from scripts.kernel import AgentRuntime; "
        "from scripts.kernel.runtime import AgentRuntime as Native; "
        "assert AgentRuntime is Native"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=30, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr


def test_kernel_star_export_keeps_runtime():
    result = subprocess.run([sys.executable, "-c", "from scripts.kernel import *; assert AgentRuntime"],
                            capture_output=True, text=True, timeout=30,
                            cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
