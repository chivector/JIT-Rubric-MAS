"""Software integration checks use no live endpoint or benchmark calls."""

import ast
import copy
import json
import shutil
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from jit_mas.bridge import (JITHarnessSynthesizer, ScriptedHarnessModel,
                            _module_interface_errors, seed_response)
from jit.harness_ops import _parse_harness_response
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.execution import TeamExecutor, TeamMemory, content_hash, validate_team
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamEvent, TeamSpec
from scripts.kernel.loader import load_harness
from scripts.kernel.monitoring import AgentLogger, LogLevel
from scripts.kernel.types import RuntimeContext, StepRecord, TaskInput
from scripts.models.base import ChatMessage
from scripts.tools.base import FinalAnswerTool
from scripts.tools.registry import ToolRegistry


class ScriptedExecution:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(messages)
        return ChatMessage(role="assistant", content=value if isinstance(value, str) else json.dumps(value))

    def get_token_counts(self):
        return {"input_token_count": 5, "output_token_count": 3}


def team_fixture():
    return TeamSpec(agents=[
        AgentSpec(agent_id="collect", role="Evidence collector", capability="research", max_calls=2),
        AgentSpec(agent_id="audit", role="Comparison auditor", capability="verification", max_calls=2),
        AgentSpec(agent_id="combine", role="Synthesis", capability="synthesis",
                  depends_on=["collect", "audit"], max_calls=2)],
        synthesizer_id="combine", total_max_calls=6, max_parallel=2)


class BridgeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.task = PublicTask(task_id="public-task", question="Compare two approaches, then justify a choice.",
                               constraints=["Public constraint is visible."])
        self.graph = RubricGraph(rubrics=[])
        self.synths = []

    def tearDown(self):
        for synth in self.synths:
            for agent in synth._agents.values():
                path = agent.workspace_dir.resolve()
                self.assertEqual(path.parent.name, "workspaces")
                self.assertTrue(path.name.startswith("mas_"))
                if path.is_dir():
                    shutil.rmtree(path)

    def synthesize(self, team=None, **kwargs):
        synth = JITHarnessSynthesizer(**kwargs)
        self.synths.append(synth)
        team = team or team_fixture()
        return synth, synth.synthesize(self.task, self.graph, team)

    def test_real_generation_parser_selector_loader_and_runtime(self):
        team = team_fixture()
        synth, artifact = self.synthesize(team)
        self.assertEqual(len(artifact.meta_trajectory), 1)
        self.assertIn("<<<PYTHON_MEMORY>>>", artifact.meta_trajectory[0]["response"])
        self.assertIn("single_protocol_valid_candidate", str(artifact.selection))
        ledger = BudgetLedger()
        models = {}

        def factory(aid):
            response = {"answer": "Independent " + aid}
            if aid == "combine":
                def response(messages):
                    context = json.loads(messages[1]["content"])
                    self.assertEqual(len(context["upstream_artifacts"]), 2)
                    return {"answer": "Coherent comparison with uncertainty.",
                            "evidence_ids": [a["event_id"] for a in context["upstream_artifacts"]]}
            raw = ScriptedExecution([response])
            models[aid] = raw
            return MeteredModel(raw, ledger, "execution", aid)

        with patch("jit_mas.execution.load_harness", wraps=load_harness) as loader:
            result = TeamExecutor(factory, ledger=ledger).execute(self.task, team, artifact)
        loader.assert_called_once_with(artifact.name)
        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertEqual(len(result.sub_runs), 3)
        self.assertTrue(result.metadata["software_test_only"])
        self.assertEqual(ledger.snapshot()["model_calls"], 3)
        self.assertEqual(result.metadata["total_token_count"], 24)
        full = result.full_dict()
        self.assertIsNotNone(full["sub_runs"][0]["trajectory"][0]["model_input_messages"])
        self.assertIsNotNone(full["sub_runs"][0]["trajectory"][0]["model_output_messages"])
        self.assertNotIn("Independent audit", json.dumps(models["collect"].calls))
        self.assertNotIn("Independent collect", json.dumps(models["audit"].calls))
        self.assertIn("Public constraint is visible", json.dumps(models["collect"].calls))
        events = result.metadata["events"]
        for event in events:
            TeamEvent.model_validate(event)
        self.assertEqual(len([e for e in events if e["kind"] == "message_consumed"]), 2)
        self.assertEqual(len([e for e in events if e["kind"] == "final_answer"]), 1)

    def test_protocol_failure_uses_original_bounded_repair(self):
        class RepairModel(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(messages)
                return ChatMessage(role="assistant", content="incomplete" if len(self.calls) == 1 else seed_response())
        model = RepairModel()
        synth, artifact = self.synthesize(meta_model=model, max_repairs=1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(artifact.repair_count, 1)
        self.assertEqual(artifact.meta_trajectory[-1]["stage"], "protocol_repair")
        self.assertIn("FILE MISSING", str(model.calls[1]))

    def test_native_fails_closed_without_endpoint_and_sandbox(self):
        with self.assertRaisesRegex(ValueError, "explicit meta endpoint"):
            JITHarnessSynthesizer(backend="native_jit")
        _, artifact = self.synthesize()
        artifact.backend = "native_jit"
        with self.assertRaisesRegex(RuntimeError, "unsafe-local"):
            TeamExecutor(lambda aid: None).execute(self.task, team_fixture(), artifact)

    def test_scripted_cannot_execute_arbitrary_generated_code(self):
        _, artifact = self.synthesize()
        path = artifact.path / "action.py"
        path.write_text(path.read_text(encoding="utf-8") + "\nprint('untrusted')\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            TeamExecutor(lambda aid: None).execute(self.task, team_fixture(), artifact)

    def test_dependencies_fail_without_deadlock(self):
        team = team_fixture()
        _, artifact = self.synthesize(team)
        calls = []

        def factory(aid):
            calls.append(aid)
            return ScriptedExecution([RuntimeError("upstream unavailable") if aid == "collect" else {"answer": "ok"}])

        result = TeamExecutor(factory).execute(self.task, team, artifact)
        self.assertIsNone(result.answer)
        self.assertNotIn("combine", calls)
        self.assertEqual(result.sub_runs[-1].terminated_reason, "dependency_failed")

    def test_timeout_returns_and_cancels_followup(self):
        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general")], synthesizer_id="one")
        _, artifact = self.synthesize(team)
        finished = threading.Event()

        def slow(messages, **kwargs):
            finished.wait(1)
            return ChatMessage(role="assistant", content="late")

        started = time.monotonic()
        try:
            result = TeamExecutor(lambda aid: slow, timeout_seconds=0.02).execute(self.task, team, artifact)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertIsNone(result.answer)
            self.assertIn("TimeoutError", result.sub_runs[0].trajectory[0].observations)
        finally:
            finished.set()

    def test_role_allowlist_enforced_at_dispatch(self):
        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general")], synthesizer_id="one")
        _, artifact = self.synthesize(team)
        tool = unittest.mock.Mock()
        model = ScriptedExecution([{"tools": [{"name": "secret_tool", "arguments": {}}]}])
        result = TeamExecutor(lambda aid: model, tools={"secret_tool": tool}).execute(self.task, team, artifact)
        self.assertIsNone(result.answer)
        tool.assert_not_called()
        self.assertIn("not allowed", result.sub_runs[0].trajectory[0].observations)

    def test_checkpoint_blocks_premature_completion(self):
        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general",
                                          checkpoints=["verify_consistency"], max_calls=2)], synthesizer_id="one")
        _, artifact = self.synthesize(team)
        model = ScriptedExecution([{"answer": "premature"},
                                    {"answer": "checked", "checkpoints": {"verify_consistency": True}}])
        result = TeamExecutor(lambda aid: model).execute(self.task, team, artifact)
        self.assertEqual(result.answer, "checked")
        self.assertEqual(len(model.calls), 2)
        self.assertIn("Unconfirmed checkpoints", json.dumps(model.calls[-1]))

    def test_private_history_retains_observed_assistant_output(self):
        memory = TeamMemory()
        memory.initialize("system", TaskInput(task="question"))
        memory.update(StepRecord(model_output_messages=ChatMessage(role="assistant", content="raw output"),
                                 observations="tool result"))
        self.assertIn("raw output", json.dumps(memory.build_context().messages))
        self.assertEqual(len(memory.get_all_steps()), 1)

    def test_cyclic_and_disconnected_teams_rejected(self):
        data = team_fixture().model_dump()
        data["agents"][0]["depends_on"] = ["combine"]
        with self.assertRaisesRegex(ValueError, "DAG"):
            validate_team(data)
        data = team_fixture().model_dump()
        data["agents"][-1]["depends_on"] = ["collect"]
        with self.assertRaisesRegex(ValueError, "every worker"):
            validate_team(data)

    def test_independent_ready_roles_really_run_concurrently(self):
        team = team_fixture()
        _, artifact = self.synthesize(team)
        barrier = threading.Barrier(2, timeout=2)
        active, peak = 0, 0
        guard = threading.Lock()

        def factory(aid):
            def respond(messages):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                if aid != "combine":
                    barrier.wait()
                with guard:
                    active -= 1
                return {"answer": aid}
            return ScriptedExecution([respond])

        result = TeamExecutor(factory).execute(self.task, team, artifact)
        self.assertEqual(result.answer, "combine")
        self.assertEqual(peak, 2)

    def test_shared_budget_stops_concurrent_roles_without_double_charging(self):
        team = team_fixture()
        _, artifact = self.synthesize(team)
        ledger = BudgetLedger(max_calls=1)

        def factory(aid):
            return MeteredModel(ScriptedExecution([{"answer": aid}]), ledger, "execution", aid)

        result = TeamExecutor(factory, ledger=ledger).execute(self.task, team, artifact)
        self.assertIsNone(result.answer)
        self.assertEqual(ledger.snapshot()["model_calls"], 1)
        self.assertEqual(len([r for r in ledger.snapshot()["records"] if r["kind"] == "model"]), 1)

    def test_execution_experience_routed_to_capability(self):
        team = team_fixture()
        synth = JITHarnessSynthesizer()
        self.synths.append(synth)
        experience = {"bank": "execution", "capability": "research",
                      "instruction": "Inspect source freshness before selecting evidence."}
        artifact = synth.synthesize(self.task, self.graph, team, experiences=[experience])
        models = {}

        def factory(aid):
            models[aid] = ScriptedExecution([{"answer": aid}])
            return models[aid]

        TeamExecutor(factory).execute(self.task, team, artifact)
        instruction = experience["instruction"]
        self.assertIn(instruction, json.dumps(models["collect"].calls))
        self.assertNotIn(instruction, json.dumps(models["audit"].calls))

    def test_tool_evidence_read_and_explicit_incorporation(self):
        class SearchTool:
            name = "lookup"
            description = "Read a public fixture."
            inputs = {"query": {"type": "string", "description": "Query"}}
            def __call__(self, query):
                return "Fixture source says A is reversible."

        self.task.tools = ["lookup"]
        team = TeamSpec(agents=[
            AgentSpec(agent_id="collect", role="Collect", capability="research", tools=["lookup"], max_calls=2),
            AgentSpec(agent_id="combine", role="Combine", capability="synthesis", depends_on=["collect"], max_calls=2)],
            synthesizer_id="combine")
        _, artifact = self.synthesize(team)
        source_event = []

        def collect_finish(messages):
            observation = next(m["content"] for m in reversed(messages) if '"output": "Fixture source' in m["content"])
            source_event.append(json.loads(observation)["event_id"])
            return {"answer": "A is reversible.", "evidence_ids": source_event}

        def synth_read(messages):
            return {"tools": [{"name": "read_evidence", "arguments": {"event_id": source_event[0]}}]}

        models = {"collect": ScriptedExecution([
            {"tools": [{"name": "lookup", "arguments": {"query": "approach A"}}]}, collect_finish]),
            "combine": ScriptedExecution([synth_read, lambda messages: {
                "answer": "Choose A when reversibility matters.", "evidence_ids": source_event}])}
        result = TeamExecutor(lambda aid: models[aid], tools={"lookup": SearchTool()}).execute(self.task, team, artifact)
        kinds = [e["kind"] for e in result.metadata["events"]]
        self.assertIn("retrieved", kinds)
        self.assertIn("evidence_read", kinds)
        final_event = next(e for e in result.metadata["events"] if e["kind"] == "final_answer")
        self.assertEqual(final_event["parent_event_ids"], source_event)

    def test_native_transport_injection_runs_exception_repair_offline(self):
        class BrokenThenRepair(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(messages)
                response = seed_response()
                if len(self.calls) == 1:
                    response = response.replace("class ActionStrategy(TeamAction):\n    pass",
                        "class ActionStrategy(TeamAction):\n    def run(self, task, ctx):\n        raise RuntimeError('fixture defect')")
                return ChatMessage(role="assistant", content=response)

        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general")], synthesizer_id="one")
        model = BrokenThenRepair()
        synth, artifact = self.synthesize(team, backend="native_jit", meta_model=model,
            meta_config={"api_base": "http://127.0.0.1:1/v1", "model_id": "offline-injected-test"})
        executor = TeamExecutor(lambda aid: ScriptedExecution([{"answer": "repaired"}]), unsafe_local=True)
        result = synth.execute_with_repair(executor, self.task, team, artifact)
        self.assertEqual(result.answer, "repaired")
        self.assertEqual(result.metadata["repair_count"], 1)
        self.assertEqual(len(result.metadata["failed_execution_attempts"]), 1)
        self.assertEqual(len(model.calls), 2)

    def test_original_plan_and_execute_seed_protocol_regression(self):
        loaded = load_harness("plan_and_execute")
        registry = ToolRegistry()
        registry.register(FinalAnswerTool())
        loaded["tool_policy"].initialize(registry.get_all())
        loaded["memory"].initialize("Answer the task.", TaskInput(task="Return forty-two."))
        model = ScriptedExecution(["Return the requested answer.", {"tools": [
            {"name": "final_answer", "arguments": {"answer": "forty-two"}}]}])
        ctx = RuntimeContext(memory=loaded["memory"], planning=loaded["planning"],
            tool_policy=loaded["tool_policy"], model=model,
            execute_tool=lambda name, args: registry.get(name)(**args),
            get_tool_schemas=registry.format_schemas, logger=AgentLogger(level=LogLevel.OFF),
            prompt_templates=loaded["prompts"], max_steps=3)
        result = loaded["action"].run("Return forty-two.", ctx)
        self.assertEqual(result.answer, "forty-two")
        self.assertIsNotNone(result.full_dict()["trajectory"][0]["model_input_messages"])

    def test_original_jit_cli_help_imports_without_model_requests(self):
        completed = subprocess.run([sys.executable, "-m", "scripts.run_jit", "--help"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--meta-model", completed.stdout)
        self.assertIn("--selector", completed.stdout)
        self.assertIn("researchrubrics", completed.stdout)

    def test_exhausted_runtime_repair_preserves_observed_failure_context(self):
        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general")], synthesizer_id="one")
        synth, artifact = self.synthesize(team, max_repairs=0)
        executor = TeamExecutor(lambda aid: ScriptedExecution([RuntimeError("local model failed")]))
        with self.assertRaisesRegex(RuntimeError, "repair budget exhausted") as caught:
            synth.execute_with_repair(executor, self.task, team, artifact)
        context = caught.exception.jit_mas_failure
        self.assertEqual(context["artifact"]["name"], artifact.name)
        self.assertEqual(len(context["failed_execution_attempts"]), 1)
        failed_run = context["failed_execution_attempts"][0]["run"]
        self.assertIsNotNone(failed_run["sub_runs"][0]["trajectory"][0]["model_input_messages"])
        self.assertIn("local model failed", failed_run["sub_runs"][0]["trajectory"][0]["error"])
        self.assertEqual(len(context["meta_trajectory"]), 1)
        json.dumps(context, allow_nan=False)
        self.assertNotIn("api_key", context["artifact"])

    def test_failed_generation_retains_candidate_and_model_trace(self):
        class InvalidModel(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                return ChatMessage(role="assistant", content="not five files")

        synth = JITHarnessSynthesizer(meta_model=InvalidModel(), max_repairs=0)
        self.synths.append(synth)
        with self.assertRaisesRegex(RuntimeError, "JIT generation failed") as caught:
            synth.synthesize(self.task, self.graph, team_fixture())
        context = caught.exception.jit_mas_failure
        self.assertEqual(len(context["generated_candidates"]), 1)
        self.assertEqual(len(context["observed_meta_calls"]), 1)
        self.assertEqual(context["generated_candidates"][0]["meta_trajectory"][0]["response"], "not five files")
        json.dumps(context, allow_nan=False)

    def test_primary_and_review_assignments_reach_actual_role_contexts(self):
        team = team_fixture()
        team.coverage = {"r1": ["collect", "combine"]}
        team.primary = {"r1": "collect"}
        team.reviewers = {"r1": ["combine"]}
        self.graph = RubricGraph.model_validate({"rubrics": [
            {"rubric_id": "r1", "requirement": "Reconcile contradictory source claims"}]})
        _, artifact = self.synthesize(team)
        models = {}

        def factory(aid):
            models[aid] = ScriptedExecution([{"answer": aid}])
            return models[aid]

        result = TeamExecutor(factory).execute(self.task, team, artifact)
        primary = json.loads(models["collect"].calls[0][1]["content"])
        reviewer = json.loads(models["combine"].calls[0][1]["content"])
        unrelated = json.loads(models["audit"].calls[0][1]["content"])
        self.assertEqual(primary["primary_rubrics"], ["r1"])
        self.assertEqual(reviewer["review_rubrics"], ["r1"])
        self.assertEqual(reviewer["review_owners"], {"r1": "collect"})
        self.assertEqual(reviewer["predicted_requirements"][0]["rubric_id"], "r1")
        self.assertEqual(unrelated["review_rubrics"], [])
        self.assertIn("Independently examine", models["combine"].calls[0][0]["content"])
        self.assertEqual(result.sub_runs[-1].metadata["review_rubrics"], ["r1"])

    def test_actual_mas_contract_is_in_generation_and_repair_model_inputs(self):
        class RepairModel(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                response = seed_response()
                if len(self.calls) == 1:
                    for export in ("MemoryStrategy", "PlanningStrategy", "ActionStrategy", "ToolPolicyStrategy"):
                        response = response.replace("class " + export, "class Team" + export)
                return ChatMessage(role="assistant", content=response)

        model = RepairModel()
        synth, artifact = self.synthesize(meta_model=model, max_repairs=1)
        self.assertEqual(len(model.calls), 2)
        for messages in model.calls:
            system = json.dumps(messages[0]["content"])
            self.assertIn("JIT-MAS BINDING CONTRACT", system)
            self.assertIn("class MemoryStrategy(TeamMemory)", system)
            self.assertIn("class ActionStrategy(TeamAction)", system)
            self.assertIn("def bind_team(self, team, services)", system)
            self.assertIn("TeamServices dataclass OBJECT", system)
            self.assertIn("EXACT name", system)
        repair_input = json.dumps(model.calls[1])
        for export in ("MemoryStrategy", "PlanningStrategy", "ActionStrategy", "ToolPolicyStrategy"):
            self.assertIn("must export class " + export, repair_input)
        self.assertEqual(artifact.repair_count, 1)
        self.assertEqual(artifact.meta_trajectory[0]["model_input_messages"], model.calls[0])
        self.assertEqual(artifact.meta_trajectory[1]["model_input_messages"], model.calls[1])
        self.assertIn("JIT-MAS BINDING CONTRACT", artifact.meta_trajectory[0]["prompt"]["system_prompt"])

    def test_interface_validation_reports_all_observed_defects_before_execution(self):
        class InvalidModel(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                response = seed_response().replace(
                    "class ActionStrategy(TeamAction):\n    pass",
                    "class TeamActionStrategy(TeamAction):\n"
                    "    def run(self, task, ctx):\n"
                    "        self.services.get('model_factory')\n"
                    "        self.services['public_task']\n"
                    "        ctx.model([])\n"
                    "        return RunResult(answer='bad', metadata={'sub_runs': []})\n")
                response = response.replace("agent_prompt:", "unused_agent_prompt:")
                return ChatMessage(role="assistant", content=response)

        synth = JITHarnessSynthesizer(backend="native_jit", meta_model=InvalidModel(),
            meta_config={"model_id": "offline-only", "api_base": "http://127.0.0.1:1/v1"}, max_repairs=0)
        self.synths.append(synth)
        with self.assertRaises(RuntimeError) as caught:
            synth.synthesize(self.task, self.graph, team_fixture())
        message = str(caught.exception)
        for text in ("must export class ActionStrategy", "TeamServices is a dataclass",
                     "TeamServices is not subscriptable", "ctx.model is a coordinator guard",
                     "must populate sub_runs", "agent_prompt must be a nonempty string"):
            self.assertIn(text, message)

    def test_native_customization_remains_generated_code_without_source_rewriting(self):
        response = seed_response().replace("class ActionStrategy(TeamAction):\n    pass",
            "class ActionStrategy(TeamAction):\n"
            "    def run(self, task, ctx):\n"
            "        result = super().run(task, ctx)\n"
            "        result.metadata['generated_policy'] = 'compare_conflicting_evidence'\n"
            "        return result\n")
        model = ScriptedExecution([response])
        team = TeamSpec(agents=[AgentSpec(agent_id="one", role="One", capability="general")], synthesizer_id="one")
        synth, artifact = self.synthesize(team, backend="native_jit", meta_model=model,
            meta_config={"model_id": "offline-only", "api_base": "http://127.0.0.1:1/v1"})
        for name, emitted in _parse_harness_response(response).items():
            self.assertEqual((artifact.path / name).read_text(encoding="utf-8"), emitted)
        result = TeamExecutor(lambda aid: ScriptedExecution([{"answer": "checked"}]),
                              unsafe_local=True).execute(self.task, team, artifact)
        self.assertEqual(result.metadata["generated_policy"], "compare_conflicting_evidence")
        self.assertEqual(result.answer, "checked")

    def test_incompatible_bound_method_signatures_are_rejected_without_import(self):
        class InvalidSignatureModel(ScriptedHarnessModel):
            def __call__(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                return ChatMessage(role="assistant", content=seed_response().replace(
                    "class ActionStrategy(TeamAction):\n    pass",
                    "class ActionStrategy(TeamAction):\n"
                    "    def bind_team(self, team):\n        self.team = team\n"
                    "    def run(self, task):\n        raise RuntimeError('must never execute')\n"))

        synth = JITHarnessSynthesizer(backend="native_jit", meta_model=InvalidSignatureModel(),
            meta_config={"model_id": "offline-only", "api_base": "http://127.0.0.1:1/v1"}, max_repairs=0)
        self.synths.append(synth)
        with self.assertRaises(RuntimeError) as caught:
            synth.synthesize(self.task, self.graph, team_fixture())
        self.assertIn("incompatible ActionStrategy.bind_team", str(caught.exception))
        self.assertIn("incompatible ActionStrategy.run", str(caught.exception))

    def test_super_constructor_uses_installed_signature_without_executing_code(self):
        cases = [
            ("super().__init__(prompts=prompts, summary_interval=summary_interval)", True),
            ("super().__init__(prompts, summary_interval)", True),
            ("super().__init__(*args, summary_interval=summary_interval)", True),
            ("super().__init__(prompts=prompts)", False),
            ("super().__init__(prompts)", False),
        ]
        for call, invalid in cases:
            with self.subTest(call=call):
                source = ("from jit_mas.execution import TeamPlanning as InstalledPlanning\n"
                          "class PlanningStrategy(InstalledPlanning):\n"
                          "    def __init__(self, prompts=None, summary_interval=4, *args):\n"
                          "        raise RuntimeError('static validation must not import this')\n"
                          f"        {call}\n"
                          "        self.summary_interval = summary_interval\n")
                errors = _module_interface_errors("planning.py", ast.parse(source))
                self.assertEqual(bool(errors), invalid, errors)
                if invalid:
                    self.assertIn("incompatible PlanningStrategy super().__init__", errors[0])
                    self.assertIn("installed TeamPlanning.__init__", errors[0])

    def test_constructor_mismatch_reaches_native_bounded_repair_before_execution(self):
        invalid = seed_response().replace("class PlanningStrategy(TeamPlanning):\n    pass",
            "class PlanningStrategy(TeamPlanning):\n"
            "    def __init__(self, prompts=None, summary_interval=4):\n"
            "        super().__init__(prompts=prompts, summary_interval=summary_interval)\n"
            "        self.summary_interval = summary_interval\n")
        corrected = invalid.replace("prompts=prompts, summary_interval=summary_interval", "prompts=prompts")
        model = ScriptedExecution([invalid, corrected])
        synth, artifact = self.synthesize(backend="native_jit", meta_model=model,
            meta_config={"model_id": "offline-only", "api_base": "http://127.0.0.1:1/v1"}, max_repairs=1)
        self.assertEqual(artifact.repair_count, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("unexpected keyword argument 'summary_interval'", json.dumps(model.calls[1]))
        self.assertIn("TeamPlanning accepts prompts, not summary_interval", json.dumps(model.calls[0]))
        for filename, content in _parse_harness_response(corrected).items():
            self.assertEqual((artifact.path / filename).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
