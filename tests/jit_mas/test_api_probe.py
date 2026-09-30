"""Offline checks for the bounded live probe; no network requests are allowed."""

import io
import json
import logging

import pytest

from jit_mas.bridge import ScriptedHarnessModel
from jit_mas.budget import MeteredModel
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from scripts.models.base import ChatMessage
from scripts.probe_jit_mas_api import DEFAULT_STAGES, STAGES, ProbeCredentials, main, run_probe


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    from scripts.models.openai_server import OpenAIServerModel

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe unit tests must never issue real requests")

    monkeypatch.setattr(OpenAIServerModel, "__call__", forbidden)


@pytest.fixture
def credentials():
    return ProbeCredentials("https://example.invalid/v1", "offline-test", "test-secret-only")


def fixture_factory():
    provider = FixtureModels()

    class BoundedFixture(FixtureModel):
        def __call__(self, messages, **kwargs):
            if self.role == "connectivity":
                self.counts = {"input_token_count": 10, "output_token_count": 4}
                return ChatMessage(role="assistant", content='{"ok": true}')
            return super().__call__(messages, **kwargs)

        def _phase(self, payload):
            response = super()._phase(payload)
            if payload["phase"] == "reconcile":
                response["team"].update(total_max_calls=4, max_parallel=1)
            return response

    def create(ledger, stage, aid, max_tokens):
        if stage == "synthesis":
            underlying = ScriptedHarnessModel()
        else:
            role = {"execution": "exec", "evaluation": "judge",
                    "connectivity": "connectivity"}.get(stage, "global")
            underlying = BoundedFixture(provider, role, aid)
        return MeteredModel(underlying, ledger, stage, aid, max_tokens)

    return create, provider


def test_auth_failure_is_single_attempt_redacted_and_stops(tmp_path, credentials, monkeypatch):
    import scripts.probe_jit_mas_api as probe

    calls = []
    options = []
    closed = []

    class AuthenticationFailure(Exception):
        status_code = 401

    class Client:
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self

        def close(self):
            closed.append(True)

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["max_attempts"] == 1
            assert kwargs["timeout"] == 45
            assert kwargs["api_key"] == credentials.api_key
            self.client = Client()

        def __call__(self, messages, **kwargs):
            calls.append(messages)
            raise AuthenticationFailure("Invalid token: " + credentials.api_key)

    monkeypatch.setattr(probe, "OpenAIServerModel", Model)
    logging_before = logging.root.manager.disable
    report = run_probe(credentials, tmp_path / "failed", STAGES)
    assert len(calls) == 1
    assert options == [{"max_retries": 0, "timeout": 45}]
    assert closed == [True]
    assert logging.root.manager.disable == logging_before
    assert report["stages"]["connectivity"]["http_status"] == 401
    assert all(report["stages"][s]["status"] == "skipped" for s in STAGES[1:])
    assert report["budget"]["model_calls"] == 1
    assert report["budget"]["reserved_tokens"] == 0
    for path in (tmp_path / "failed").glob("*.json"):
        assert credentials.api_key not in path.read_text()


def test_defaults_run_only_connectivity_and_planning(tmp_path, credentials):
    factory, provider = fixture_factory()
    report = run_probe(credentials, tmp_path / "default", model_factory=factory)
    assert DEFAULT_STAGES == ("connectivity", "planning")
    assert report["status"] == "passed"
    assert report["budget"]["model_calls"] == 5
    assert report["stages"]["synthesis"]["status"] == "not_requested"
    assert not (tmp_path / "default" / "execution.json").exists()
    assert {c["agent_id"] for c in provider.calls} == {"global", "analyst", "integrator"}


def test_full_flow_preserves_native_static_and_seed_execution_distinction(tmp_path, credentials):
    factory, _ = fixture_factory()
    output = tmp_path / "complete"
    report = run_probe(credentials, output, STAGES, model_factory=factory)
    assert report["status"] == "passed", report
    assert report["model_source"] == "injected_offline_test"
    assert not report["native_generated_code_executed"]
    assert report["stages"]["synthesis"]["verification"] == "static_only"
    assert report["stages"]["execution"]["backend"] == "trusted_seed_live_models"
    native = json.loads((output / "native_harness.json").read_text())
    seed = json.loads((output / "trusted_seed_harness.json").read_text())
    execution = json.loads((output / "execution.json").read_text())
    assert native["backend"] == "native_jit"
    assert seed["backend"] == execution["metadata"]["backend"] == "scripted"
    assert len(execution["sub_runs"]) == 2
    assert all(r["trajectory"][0]["model_input_messages"] for r in execution["sub_runs"])
    assert report["stages"]["evaluation"]["criterion_count"] == 2
    assert report["stages"]["evaluation"]["submitted_at"] < report["stages"]["evaluation"]["evaluated_at"]
    assert report["budget"]["model_calls"] <= 20
    assert report["budget"]["tool_calls"] == report["budget"]["reserved_tokens"] == 0
    assert report["stages"]["attribution"]["staged_proposals"] == 1
    store = ExperienceStore(output / "staged_experience.sqlite", read_only=True)
    try:
        assert store.snapshot().version == 0
        assert not store.snapshot().experiences
        assert store.db.execute("SELECT COUNT(*) FROM staging").fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 0
    finally:
        store.close()


def test_missing_dependency_does_not_make_unrequested_calls(tmp_path, credentials):
    def unexpected(*args):
        raise AssertionError("No implicit prerequisite calls are allowed")

    report = run_probe(credentials, tmp_path / "missing", ["evaluation"], model_factory=unexpected)
    assert report["status"] == "incomplete"
    assert report["budget"]["model_calls"] == 0
    assert report["stages"]["evaluation"]["status"] == "skipped"


def test_reflected_key_is_blocked_before_native_source_write(tmp_path, credentials, monkeypatch):
    import jit_mas.bridge as bridge

    factory, _ = fixture_factory()
    workspaces = []
    original = bridge.JITHarnessSynthesizer._agent

    def record_workspace(self, name):
        agent = original(self, name)
        workspaces.append(agent.workspace_dir)
        return agent

    class ReflectedResponse(ScriptedHarnessModel):
        def __call__(self, messages, **kwargs):
            response = super().__call__(messages, **kwargs)
            response.content += "\n" + credentials.api_key
            return response

    def reflected_factory(ledger, stage, aid, max_tokens):
        if stage == "synthesis":
            return MeteredModel(ReflectedResponse(), ledger, stage, aid, max_tokens)
        return factory(ledger, stage, aid, max_tokens)

    monkeypatch.setattr(bridge.JITHarnessSynthesizer, "_agent", record_workspace)
    report = run_probe(credentials, tmp_path / "reflected", ["planning", "synthesis"],
                       model_factory=reflected_factory)
    assert report["stages"]["synthesis"]["status"] == "failed"
    assert "discarded before persistence" in report["stages"]["synthesis"]["error"]
    assert workspaces
    for directory in [tmp_path / "reflected", *workspaces]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json"}:
                assert credentials.api_key not in path.read_text(encoding="utf-8")


def test_cli_accepts_stdin_and_environment_without_key_argument(tmp_path, monkeypatch, capsys):
    import scripts.probe_jit_mas_api as probe

    seen = []

    def stub(credentials, output_dir, stages):
        seen.append(credentials)
        return {"status": "passed", "budget": {"model_calls": 0}}

    monkeypatch.setattr(probe, "run_probe", stub)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "endpoint": "https://example.invalid/v1", "model": "test", "api_key": "stdin-secret"})))
    assert main(["--output-dir", str(tmp_path / "stdin")]) == 0
    monkeypatch.setenv("JIT_PROBE_TEST_KEY", "env-secret")
    assert main(["--endpoint", "https://example.invalid/v1", "--model", "test",
                 "--key-env", "JIT_PROBE_TEST_KEY", "--output-dir", str(tmp_path / "env")]) == 0
    assert [c.api_key for c in seen] == ["stdin-secret", "env-secret"]
    stdout = capsys.readouterr().out
    assert "stdin-secret" not in stdout and "env-secret" not in stdout
    assert "stdin-secret" not in repr(seen[0])


def test_rejects_secret_in_endpoint_and_nonempty_output(tmp_path, credentials):
    with pytest.raises(ValueError, match="without credentials"):
        ProbeCredentials("https://user:password@example.invalid/v1", "test", "key")
    (tmp_path / "existing.json").write_text("{}")
    with pytest.raises(ValueError, match="new empty"):
        run_probe(credentials, tmp_path)
