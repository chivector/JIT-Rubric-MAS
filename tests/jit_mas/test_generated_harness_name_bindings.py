"""Regressions for shared native/MAS generated-code validation; no model calls."""

from types import SimpleNamespace

import pytest

from jit.meta_agent import MetaReActAgent, PROMPTS


def static_report(tmp_path, source, *, filename="action.py"):
    for name in ("action.py", "memory.py", "planning.py", "tool_policy.py"):
        (tmp_path / name).write_text(source if name == filename else "pass\n", encoding="utf-8")
    (tmp_path / "prompt.yaml").write_text(
        "system_prompt: synthetic\nplanning: {}\nsummary: {}\nfinal_answer: {}\nstep: {}\n",
        encoding="utf-8")
    # Bypass the model/client constructor. Static validation must only read code.
    return MetaReActAgent._static_harness_checks(SimpleNamespace(workspace_dir=tmp_path))


@pytest.mark.parametrize("name,expression", [
    ("LogLevel", "LogLevel.INFO"), ("Panel", "Panel('synthetic')"), ("Text", "Text('synthetic')"),
])
def test_undeclared_logging_types_are_rejected_before_any_executor_call(tmp_path, name, expression):
    source = ("class ActionStrategy:\n"
              "    def run(self, task, ctx):\n"
              f"        return ctx.logger.log({expression})\n")
    report = static_report(tmp_path, source)
    assert not report.startswith("No static errors detected")
    assert "action.py" in report and name in report


@pytest.mark.parametrize("filename", ["action.py", "memory.py", "planning.py", "tool_policy.py"])
def test_undefined_globals_in_nested_functions_are_checked_in_all_generated_modules(tmp_path, filename):
    source = ("class Strategy:\n"
              "    def run(self, task, ctx):\n"
              "        def render():\n"
              "            return missing_runtime_helper(task)\n"
              "        return render()\n")
    report = static_report(tmp_path, source, filename=filename)
    assert not report.startswith("No static errors detected")
    assert filename in report and "missing_runtime_helper" in report


def test_complete_logging_imports_are_accepted(tmp_path):
    source = ("from scripts.kernel.monitoring import LogLevel\n"
              "from rich.panel import Panel\n"
              "from rich.text import Text\n"
              "class ActionStrategy:\n"
              "    def run(self, task, ctx):\n"
              "        return ctx.logger.log(Panel(Text(task)), level=LogLevel.INFO)\n")
    assert static_report(tmp_path, source).startswith("No static errors detected")


@pytest.mark.parametrize("source", [
    "def run(LogLevel, Panel, Text):\n    return Panel(Text(str(LogLevel)))\n",
    "def run():\n    LogLevel = 'local'\n    def render():\n        return LogLevel\n    return render()\n",
    "def run():\n    return later_helper()\ndef later_helper():\n    return 'forward definition'\n",
    "def run():\n    return [str(index) for index in range(3)]\n",
    "def run():\n    try:\n        return int('synthetic')\n    except ValueError as error:\n        return str(error)\n",
    "from scripts.kernel.monitoring import LogLevel as Level\ndef run():\n    return Level.INFO\n",
    "def run():\n    return (__name__, __file__, __package__, __doc__, __spec__, __loader__, __cached__, __builtins__)\n",
    "from __future__ import annotations\ndef run(item: LaterAnnotation) -> LaterAnnotation:\n    return item\n",
])
def test_legal_scopes_forward_definitions_builtin_and_module_names_are_accepted(tmp_path, source):
    assert static_report(tmp_path, source).startswith("No static errors detected")


def test_static_validation_never_imports_or_executes_generated_source(tmp_path):
    source = "raise RuntimeError('Static validation must not execute this module')\n"
    assert static_report(tmp_path, source).startswith("No static errors detected")


@pytest.mark.parametrize("prompt_name", ["generate_system_prompt", "repair_system_prompt"])
@pytest.mark.parametrize("required_import", ["from scripts.kernel.monitoring import LogLevel",
                                             "from rich.panel import Panel", "from rich.text import Text"])
def test_shared_generation_and_repair_contracts_document_required_logging_imports(prompt_name, required_import):
    assert required_import in PROMPTS[prompt_name]
