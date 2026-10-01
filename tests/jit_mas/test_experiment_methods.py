from __future__ import annotations

import json
from pathlib import Path

import pytest

from jit_mas.config import MASConfig
from jit_mas.checkpoints import CheckpointIntegrityError
from jit_mas.experience import ExperienceStore
from jit_mas.experiment_methods import submit_method
from jit_mas.schemas import digest
from scripts.run_jit_mas import make_pipeline


@pytest.fixture
def pipeline(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    result = make_pipeline(MASConfig(backend="scripted", max_model_calls=200), store, tmp_path / "unused")
    result.evaluator_factory = lambda _: pytest.fail("Controls must not construct a judge")
    yield result
    store.close()


@pytest.mark.parametrize("method", ["jit_matched", "rubric_fixed", "G", "GO"])
def test_control_submits_real_artifact_and_never_evaluates(pipeline, tmp_path, method):
    state = pipeline.store.snapshot()
    outcome = submit_method(pipeline, "test-deployment", state, method=method, repeat=0,
                            output_dir=tmp_path / method, shared_dir=tmp_path / "shared")
    assert outcome["status"] == "submitted_unscored"
    assert outcome["software_test_only"] is True
    assert outcome["evaluation"] is None
    assert outcome["proposals"] == []
    assert outcome["experience_hash"] == digest(state)
    assert outcome["budget"]["reserved_tokens"] == 0
    assert outcome["budget"]["model_calls"] > 0
    directory = Path(outcome["run_dir"])
    submission = json.loads((directory / "submission.json").read_text())
    assert submission["answer"]
    assert digest(submission["answer"]) == outcome["answer_hash"]
    assert not (directory / "evaluation.json").exists()
    assert not any(row.get("role") == "judge" for row in pipeline.models.calls)
    assert digest(pipeline.store.snapshot()) == digest(state)
    if method == "jit_matched":
        harness = json.loads((directory / "harness.json").read_text())
        assert harness["kind"] == "upstream_MetaReActAgent"
        assert set(harness["files"]) == {"action.py", "memory.py", "planning.py", "tool_policy.py", "prompt.yaml"}
        calls = json.loads((directory / "model_calls.json").read_text())
        assert len(calls["generation"]) == 1
        assert len(calls["execution"]) >= 1
        assert "JIT-MAS BINDING CONTRACT" not in json.dumps(calls["generation"])
        assert not (directory / "planning_calls.json").exists()
    elif method == "rubric_fixed":
        harness = json.loads((directory / "harness.json").read_text())
        assert harness["meta_trajectory"] == []
        assert harness["selection"]["kind"] == "fixed_checked_in_seed"
        planning = json.loads((directory / "planning_calls.json").read_text())
        assert len(planning) == 1
        execution = json.loads((directory / "execution.json").read_text())
        assert {row["metadata"]["agent_id"] for row in execution["sub_runs"]} == {"analyst", "evidence", "writer"}
        assert all(len(row["trajectory"]) == 1 for row in execution["sub_runs"])
        assert outcome["budget"]["model_calls"] == 4


def test_guidance_pair_reuses_one_quality_only_preparation_and_freezes_boundary(pipeline, tmp_path):
    state = pipeline.store.snapshot()
    outputs = {method: submit_method(pipeline, "test-deployment", state, method=method, repeat=0,
                                    output_dir=tmp_path / method, shared_dir=tmp_path / "shared")
               for method in ("G", "GO")}
    first, second = outputs["G"]["shared_rstar"], outputs["GO"]["shared_rstar"]
    assert first["path"] == second["path"]
    assert first["sha256"] == second["sha256"]
    assert first["reused"] is False and second["reused"] is True
    shared = json.loads(Path(first["path"]).read_text())
    assert shared["sha256"] == digest({k: v for k, v in shared.items() if k != "sha256"})
    assert shared["budget"]["model_calls"] == 1
    assert set(shared["graph"]) == {"schema_version", "rubrics", "edges"}
    assert len(shared["calls"]) == 1
    assert shared["calls"][0]["phase"] == "quality_only"
    for method, outcome in outputs.items():
        directory = Path(outcome["run_dir"])
        frozen = json.loads((directory / "frozen_plan.json").read_text())
        assert bool(frozen["graph"]["rubrics"]) is (method == "GO")
        injections = json.loads((directory / "guidance_injection.json").read_text())
        assert len(injections) == 3
        for injection in injections:
            payload = json.loads(injection["messages"][1]["content"])
            assert payload["quality_guidance"] == shared["graph"]
            assert payload["quality_guidance_hash"] == shared["graph_hash"]
            assert payload["predicted_requirements"] == []
            assert injection["frozen_harness_hash"]
            assert "PRIVATE_CANARY" not in json.dumps(injection)
        assert outcome["logical_deployment_budget"]["model_calls"] == outcome["budget"]["model_calls"] + 1
        if method == "G":
            calls = json.loads((directory / "planning_calls.json").read_text())
            assert all("fixed_quality_graph" not in json.dumps(call["messages"]) for call in calls)
            harness = json.loads((directory / "harness.json").read_text())
            sidecar = json.loads((Path(harness["path"]) / "team.json").read_text())
            assert sidecar["rubrics"]["rubrics"] == []
            assert "quality_guidance" not in json.dumps(sidecar)


def test_cache_tampering_and_repeat_binding_fail_closed(pipeline, tmp_path):
    state = pipeline.store.snapshot()
    first = submit_method(pipeline, "test-deployment", state, method="G", repeat=0,
                          output_dir=tmp_path / "G", shared_dir=tmp_path / "shared")
    path = Path(first["shared_rstar"]["path"])
    row = json.loads(path.read_text())
    row["graph"]["rubrics"][0]["requirement"] = "TAMPERED_GUIDANCE"
    path.write_text(json.dumps(row))
    with pytest.raises(CheckpointIntegrityError, match="integrity"):
        submit_method(pipeline, "test-deployment", state, method="GO", repeat=0,
                      output_dir=tmp_path / "GO", shared_dir=tmp_path / "shared")
    with pytest.raises(RuntimeError, match="already started"):
        submit_method(pipeline, "test-deployment", state, method="G", repeat=0,
                      output_dir=tmp_path / "G", shared_dir=tmp_path / "shared")
    fresh = submit_method(pipeline, "test-deployment", state, method="G", repeat=1,
                          output_dir=tmp_path / "G-repeat", shared_dir=tmp_path / "shared")
    assert fresh["shared_rstar"]["path"] != str(path)
    assert fresh["shared_rstar"]["reused"] is False


def test_rstar_sentinel_never_reaches_g_construction_but_both_actor_contexts(pipeline, tmp_path, monkeypatch):
    import jit_mas.experiment_methods as methods
    from jit_mas.offline import FixtureModel
    from scripts.models.base import ChatMessage

    sentinel = "UNIQUE_RSTAR_QUALITY_REQUIREMENT_NOT_PRESENT_IN_PUBLIC_TASK"
    original_quality = methods._QualityFixture.__call__
    original_phase = FixtureModel._phase

    def quality(self, messages, **kwargs):
        response = original_quality(self, messages, **kwargs)
        graph = json.loads(response.content)
        graph["rubrics"][0]["requirement"] = sentinel
        return ChatMessage(role="assistant", content=json.dumps(graph))

    def phase(self, payload):
        response = original_phase(self, payload)
        if payload["phase"] == "predict" and "fixed_quality_graph" in payload:
            response["graph"] = payload["fixed_quality_graph"]
        return response

    monkeypatch.setattr(methods._QualityFixture, "__call__", quality)
    monkeypatch.setattr(FixtureModel, "_phase", phase)
    outcomes = {method: submit_method(pipeline, "test-deployment", pipeline.store.snapshot(),
                                     method=method, repeat=0, output_dir=tmp_path / method,
                                     shared_dir=tmp_path / "shared") for method in ("G", "GO")}
    g_path, go_path = [Path(outcomes[method]["run_dir"]) for method in ("G", "GO")]
    assert sentinel not in (g_path / "planning_calls.json").read_text()
    assert sentinel not in (g_path / "harness.json").read_text()
    assert sentinel in (go_path / "planning_calls.json").read_text()
    for directory in (g_path, go_path):
        injections = json.loads((directory / "guidance_injection.json").read_text())
        for row in injections:
            assert json.dumps(row["messages"]).count(sentinel) == 1


def test_interrupted_quality_preparation_is_not_resampled(pipeline, tmp_path, monkeypatch):
    import jit_mas.experiment_methods as methods

    calls = []

    def fail(self, messages, **kwargs):
        calls.append(messages)
        raise TimeoutError("synthetic")

    monkeypatch.setattr(methods._QualityFixture, "__call__", fail)
    with pytest.raises(TimeoutError):
        submit_method(pipeline, "test-deployment", pipeline.store.snapshot(), method="G", repeat=0,
                      output_dir=tmp_path / "G", shared_dir=tmp_path / "shared")
    with pytest.raises(RuntimeError, match="cannot be silently resampled"):
        submit_method(pipeline, "test-deployment", pipeline.store.snapshot(), method="GO", repeat=0,
                      output_dir=tmp_path / "GO", shared_dir=tmp_path / "shared")
    assert len(calls) == 1
    failure = json.loads(next((tmp_path / "shared").glob("*.failure.json")).read_text())
    assert failure["budget"]["model_calls"] == 1
    assert failure["budget"]["reserved_tokens"] == 0


def test_controls_cannot_run_on_evolution_or_validation(pipeline, tmp_path):
    for task_id in ("evolve-comparison", "validation-storage"):
        with pytest.raises(ValueError, match="restricted to test"):
            submit_method(pipeline, task_id, pipeline.store.snapshot(), method="G", repeat=0,
                          output_dir=tmp_path / task_id)
    assert not (tmp_path / "evolve-comparison").exists()


def test_rubric_fixed_rejects_insufficient_common_resource_cap(pipeline, tmp_path):
    pipeline.config.max_agents = 2
    with pytest.raises(ValueError, match="three agents"):
        submit_method(pipeline, "test-deployment", pipeline.store.snapshot(), method="rubric_fixed", repeat=0,
                      output_dir=tmp_path / "fixed")


def test_generation_policy_preserves_native_multimodal_message_blocks():
    from jit_mas.experiment_methods import _GenerationPolicy

    calls = []
    messages = [{"role": "system", "content": [{"type": "text", "text": "Original"}]},
                {"role": "user", "content": [{"type": "text", "text": "Public task"}]}]
    _GenerationPolicy(lambda messages: calls.append(messages))(messages)
    assert all(isinstance(block, dict) and block["type"] == "text" for block in calls[0][0]["content"])
    assert len(calls[0][0]["content"]) == 2
    assert len(messages[0]["content"]) == 1


def test_native_unmetered_client_mistakes_are_rejected():
    from jit_mas.experiment_methods import _native_client_policy

    assert "metered" in _native_client_policy({"action.py": "model = OpenAIServerModel()"})
    assert "forbidden" in _native_client_policy({"action.py": "import openai"})
    assert _native_client_policy({"action.py": "from scripts.kernel.types import RunResult\nresult = ctx.model(messages)"}) == ""


def test_native_final_answer_is_submission_not_external_tool_cost(pipeline, tmp_path, monkeypatch):
    from jit_mas.budget import BudgetExceeded
    from scripts.kernel.runtime import AgentRuntime

    pipeline.config.max_tool_calls = 0
    original = AgentRuntime.run

    def probe(self, *args, **kwargs):
        assert self._execute_tool("final_answer", {"answer": "submitted"}) == "submitted"
        with pytest.raises(BudgetExceeded):
            self._execute_tool("web_search", {"query": "not allowed"})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AgentRuntime, "run", probe)
    outcome = submit_method(pipeline, "test-deployment", pipeline.store.snapshot(), method="jit_matched",
                            repeat=0, output_dir=tmp_path / "native")
    assert outcome["status"] == "submitted_unscored"
    assert outcome["budget"]["tool_calls"] == 0
