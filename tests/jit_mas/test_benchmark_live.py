"""Offline safety and accounting checks for the opt-in native benchmark runner."""

import copy
import hashlib
import io
import json
import traceback
from types import SimpleNamespace

import pytest

from jit_mas.bridge import JITHarnessSynthesizer
from jit_mas.budget import BudgetExceeded, BudgetLedger
from jit_mas.config import MASConfig, ModelConfig, NativeModels
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels, fixture_dataset
from jit_mas.schemas import PublicTask, SplitManifest, digest
from scripts import benchmark_jit_mas_live as live
from scripts.models.base import ChatMessage
from scripts.run_jit_mas import make_pipeline


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native benchmark unit tests must not make network requests")

    monkeypatch.setattr(live.OpenAIServerModel, "__call__", forbidden)


@pytest.fixture
def credentials():
    return live.ProbeCredentials("https://example.invalid/v1", "offline-model", "offline-secret")


@pytest.mark.parametrize("approve", [False, True])
def test_cli_rejects_missing_approval_or_bad_data_before_models(
        tmp_path, monkeypatch, capsys, credentials, approve):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid local inputs must be rejected before model setup")

    monkeypatch.setattr(live, "LiveModels", forbidden)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "endpoint": credentials.endpoint, "model": credentials.model,
        "api_key": credentials.api_key})))
    data = tmp_path / "not-official.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    args = ["--data", str(data), "--splits", str(tmp_path / "missing.json"),
            "--output-dir", str(tmp_path / "run")]
    if approve:
        args.append("--unsafe-local")
    assert live.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error_type"] == ("ValueError" if approve else "PermissionError")
    assert credentials.api_key not in json.dumps(output)
    assert not (tmp_path / "run").exists()


def test_load_inputs_keeps_private_records_out_of_public_tasks(tmp_path, monkeypatch):
    data = tmp_path / "dataset.jsonl"
    data.write_bytes(b"offline adapter fixture")
    monkeypatch.setattr(live, "OFFICIAL_DATA_SHA256", hashlib.sha256(data.read_bytes()).hexdigest())
    manifest = {"evolution": ["e"], "validation": ["v1", "v2"], "test": ["t"]}
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(manifest), encoding="utf-8")

    class Adapter:
        def load_dataset(self, path):
            return [{"task_id": tid, "question": "Public question " + tid,
                     "rubrics": ["PRIVATE_CRITERION"], "reference": "PRIVATE_REFERENCE"}
                    for tid in ["e", "v1", "v2", "t", "excluded"]]

        def private_record(self, task_id):
            return {"task_id": task_id, "rubrics": ["PRIVATE_CRITERION"]}

    monkeypatch.setattr(live, "ResearchRubricsAdapter", Adapter)
    tasks, private, result = live.load_inputs(data, splits)
    assert set(tasks) == set(private) == {"e", "v1", "v2", "t"}
    assert result.evolution == ["e"]
    assert all(task.tools == [] for task in tasks.values())
    assert "PRIVATE_" not in json.dumps({key: task.model_dump() for key, task in tasks.items()})
    assert "PRIVATE_CRITERION" in json.dumps(private)


def fake_model_class(constructed, requests, closed, *, failure=None):
    class Client:
        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0, "timeout": 73}
            return self

        def close(self):
            closed.append(self)

    class Model:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.client = Client()
            constructed.append(self)

        def __call__(self, messages, **kwargs):
            requests.append((messages, kwargs))
            if failure:
                raise failure
            return ChatMessage(role="assistant", content="{\"ok\": true}")

        def get_token_counts(self):
            return {"input_token_count": 7, "output_token_count": 3}

    return Model


@pytest.mark.parametrize("non_thinking", [False, True])
def test_live_models_role_formats_and_both_ledgers_count_each_request_once(
        tmp_path, monkeypatch, credentials, non_thinking):
    constructed, requests, closed = [], [], []
    monkeypatch.setattr(live, "OpenAIServerModel", fake_model_class(constructed, requests, closed))
    session = live.SessionLedger(tmp_path / "session.json", max_calls=20, max_tokens=500_000)
    task = BudgetLedger(max_calls=20, max_tokens=500_000)
    provider = live.LiveModels(credentials, session, timeout=73, non_thinking=non_thinking)
    for role, limit in live.ROLE_TOKENS.items():
        model = provider.create(role, "agent-" + role, task, role)
        response = model([{"role": "user", "content": "Offline request"}])
        assert response.content == '{"ok": true}'
        options = constructed[-1].options
        assert options["max_attempts"] == 1
        assert options["max_tokens"] == limit
        assert options["timeout"] == 73
        assert requests[-1][1]["max_tokens"] == limit
        if non_thinking:
            assert options["extra_body"] == {"thinking": {"type": "disabled"}}
            assert options["reasoning_effort"] == "none"
        else:
            assert "extra_body" not in options
            assert "reasoning_effort" not in options
        if role in {"global", "local"}:
            assert options["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in options
    provider.close()
    assert len(constructed) == len(requests) == len(closed) == len(live.ROLE_TOKENS)
    for ledger in [session, task]:
        snapshot = ledger.snapshot()
        assert snapshot["model_calls"] == len(requests)
        assert snapshot["tokens"] == len(requests) * 10
        assert snapshot["reserved_tokens"] == 0
        assert all(not row["estimated"] for row in snapshot["records"])
        assert set(snapshot["by_stage"]) == set(live.ROLE_TOKENS)
    persisted = json.loads((tmp_path / "session.json").read_text())
    assert persisted["model_calls"] == len(requests)
    assert persisted["tokens"] == len(requests) * 10
    assert credentials.api_key not in (tmp_path / "session.json").read_text()


@pytest.mark.parametrize("thinking,effort", [
    (None, None), ("disabled", "none"), ("enabled", "high"),
    ("disabled", None), (None, "low"),
])
def test_native_models_forward_explicit_thinking_configuration_only(
        monkeypatch, credentials, thinking, effort):
    import scripts.models.openai_server as transport

    constructed, requests, closed = [], [], []
    monkeypatch.setattr(transport, "OpenAIServerModel", fake_model_class(constructed, requests, closed))
    monkeypatch.setenv("OFFLINE_BENCHMARK_TEST_KEY", credentials.api_key)
    base = ModelConfig(model=credentials.model, endpoint=credentials.endpoint,
                       key_env="OFFLINE_BENCHMARK_TEST_KEY", timeout=73)
    selected = base.model_copy(update={"thinking": thinking, "reasoning_effort": effort})
    config = MASConfig(backend="native_jit", unsafe_local=True,
                       models={role: selected.model_copy() for role in live.ROLE_TOKENS})
    provider = NativeModels(config)
    task = BudgetLedger()
    model = provider.create("global", "global", task, "planning")
    model([{"role": "user", "content": "Offline request"}])
    options = constructed[0].options
    if thinking is None:
        assert "extra_body" not in options
    else:
        assert options["extra_body"] == {"thinking": {"type": thinking}}
    if effort is None:
        assert "reasoning_effort" not in options
    else:
        assert options["reasoning_effort"] == effort
    assert options["max_attempts"] == 1
    assert options["api_key"] == credentials.api_key
    assert len(requests) == 1
    assert task.snapshot()["tokens"] == 10
    assert (digest(base) != digest(selected)) == (thinking is not None or effort is not None)


@pytest.mark.parametrize("non_thinking", [False, True])
def test_cli_non_thinking_is_opt_in_and_forwarded(monkeypatch, capsys, credentials, non_thinking):
    captured = []

    def run(supplied, data, splits, output_dir, **kwargs):
        assert supplied == credentials
        captured.append(kwargs)
        return {"status": "completed", "budget": {"model_calls": 0}}

    monkeypatch.setattr(live, "run_benchmark", run)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "endpoint": credentials.endpoint, "model": credentials.model,
        "api_key": credentials.api_key})))
    args = ["--data", "unused-data", "--splits", "unused-splits", "--output-dir", "unused-output"]
    if non_thinking:
        args.append("--non-thinking")
    assert live.main(args) == 0
    assert captured[0]["non_thinking"] is non_thinking
    assert captured[0]["unsafe_local"] is False
    assert credentials.api_key not in capsys.readouterr().out


@pytest.mark.parametrize("exhausted", ["session", "task"])
def test_either_ledger_stops_requests_before_transport(tmp_path, monkeypatch, credentials, exhausted):
    constructed, requests, closed = [], [], []
    monkeypatch.setattr(live, "OpenAIServerModel", fake_model_class(constructed, requests, closed))
    session = live.SessionLedger(tmp_path / "session.json", max_calls=0 if exhausted == "session" else 2,
                                 max_tokens=100_000)
    task = BudgetLedger(max_calls=0 if exhausted == "task" else 2, max_tokens=100_000)
    provider = live.LiveModels(credentials, session, timeout=73)
    model = provider.create("global", "global", task, "planning")
    with pytest.raises(BudgetExceeded):
        model([{"role": "user", "content": "Offline request"}])
    provider.close()
    assert requests == []
    assert task.snapshot()["reserved_tokens"] == session.snapshot()["reserved_tokens"] == 0


def test_failed_request_settles_both_ledgers_and_redacts_error(tmp_path, monkeypatch, credentials):
    constructed, requests, closed = [], [], []
    failure = ValueError("Authorization rejected: " + credentials.api_key)
    monkeypatch.setattr(live, "OpenAIServerModel", fake_model_class(
        constructed, requests, closed, failure=failure))
    session = live.SessionLedger(tmp_path / "session.json", max_calls=2, max_tokens=100_000)
    task = BudgetLedger(max_calls=2, max_tokens=100_000)
    provider = live.LiveModels(credentials, session, timeout=73)
    model = provider.create("global", "global", task, "planning")
    with pytest.raises(RuntimeError) as caught:
        model([{"role": "user", "content": "Offline request"}])
    provider.close()
    assert credentials.api_key not in str(caught.value)
    assert credentials.api_key not in "".join(traceback.format_exception(caught.value))
    assert "[REDACTED]" in str(caught.value)
    assert len(requests) == len(closed) == 1
    for ledger in [session, task]:
        assert ledger.snapshot()["model_calls"] == 1
        assert ledger.snapshot()["reserved_tokens"] == 0
        assert ledger.snapshot()["records"][0]["error"] == "RuntimeError"


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
def test_safe_transport_rejects_reflected_credentials_before_persistence(credentials, field):
    response = SimpleNamespace(content="ok", reasoning_content=None)
    setattr(response, field, "reflected " + credentials.api_key)
    model = live.SafeTransport(lambda *args, **kwargs: response, credentials.api_key)
    with pytest.raises(RuntimeError, match="discarded before persistence") as caught:
        model([])
    assert credentials.api_key not in str(caught.value)


def test_reviewed_synthesizer_rechecks_each_repaired_execution(monkeypatch):
    events = []
    artifact = SimpleNamespace(code_hash="original", repair_count=0)
    result = SimpleNamespace(sub_runs=[], answer="submitted", metadata={})
    gate = SimpleNamespace(wait=lambda item: events.append(("review", item.code_hash)))
    synthesizer = live.ReviewedSynthesizer(gate, backend="native_jit", meta_model=object(),
        meta_config={"api_base": "https://example.invalid/v1", "model_id": "offline"}, max_repairs=1)
    assert isinstance(synthesizer, JITHarnessSynthesizer)
    assert synthesizer.backend == "native_jit"

    def execute(task, team, item, **kwargs):
        events.append(("execute", item.code_hash))
        if item.repair_count == 0:
            raise RuntimeError("generated interface needs repair")
        return result

    def repair(item, error, failed_result):
        events.append(("repair", item.code_hash))
        item.code_hash = "repaired"
        item.repair_count += 1
        return item

    monkeypatch.setattr(synthesizer, "repair", repair)
    actual = synthesizer.execute_with_repair(SimpleNamespace(execute=execute), object(), object(), artifact)
    assert actual is result
    assert events == [("review", "original"), ("execute", "original"), ("repair", "original"),
                      ("review", "repaired"), ("execute", "repaired")]
    assert actual.metadata["repair_count"] == 1
    assert len(actual.metadata["failed_execution_attempts"]) == 1


@pytest.mark.parametrize("error", [PermissionError("review denied"), TimeoutError("review timeout")])
def test_review_rejection_or_timeout_never_executes_or_repairs(monkeypatch, error):
    def deny(artifact):
        raise error

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected review cannot execute or repair code")

    synthesizer = live.ReviewedSynthesizer(SimpleNamespace(wait=deny), backend="scripted")
    monkeypatch.setattr(synthesizer, "repair", forbidden)
    monkeypatch.setattr(synthesizer, "_attach_failure", lambda *args, **kwargs: None)
    with pytest.raises(type(error), match="review"):
        synthesizer.execute_with_repair(SimpleNamespace(execute=forbidden), object(), object(),
                                        SimpleNamespace(code_hash="unapproved"))


def test_failed_planning_persists_observed_response_and_budget(tmp_path):
    provider = FixtureModels()
    create = provider.create

    def corrupt_global(role, agent_id, ledger, stage):
        model = create(role, agent_id, ledger, stage)
        if role != "global":
            return model

        class InvalidJSON:
            def __call__(self, messages, **kwargs):
                response = model(messages, **kwargs)
                response.content = '{"graph":'
                return response

        return InvalidJSON()

    provider.create = corrupt_global
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs",
                                 fixture_models=provider)
        with pytest.raises(json.JSONDecodeError):
            pipeline.run("evolve")
        path = next((tmp_path / "runs").glob("*/planning_calls.json"))
        calls = json.loads(path.read_text())
        assert len(calls) == 2 and all(call["phase"] == "predict" for call in calls)
        assert all(call["response"] == '{"graph":' for call in calls)
        assert json.loads(path.with_name("budget.json").read_text())["model_calls"] == 2
        assert path.with_name("failure.json").exists()
        assert not path.with_name("submission.json").exists()
    finally:
        store.close()


@pytest.mark.parametrize("retrieved,expected", [([], False), (["old-experience"], False),
                                               (["different"], False), (["updated-experience"], True)])
def test_closed_loop_confirmation_requires_actual_updated_experience_retrieval(retrieved, expected):
    snapshot = SimpleNamespace(applied_proposals=["proposal-1"], experiences=[
        SimpleNamespace(experience_id="updated-experience"),
        SimpleNamespace(experience_id="old-experience")])
    source = {"task_id": "e", "evaluation_complete": True,
              "proposal_experiences": [{"proposal_id": "proposal-1", "experience_id": "updated-experience"}],
              "experience_updates": [{"proposal_id": "proposal-1", "source_task_id": "e",
                                      "update_rule": "direct_after_attribution"}]}
    report = {"status": "completed", "stages": {"evolution": "completed", "held_out": "completed"},
              "evolution": [source], "held_out": [{"evaluation_complete": True}],
              "held_out_retrieved_experience_ids": retrieved}
    live._confirm_closed_loop(report, snapshot)
    assert report["full_mechanism_exercised"] and report["experience_update_complete"]
    assert report["closed_loop_confirmed"] is expected
    assert report["updated_experience_reuse_demonstrated"] is expected
    assert report["held_out_retrieved_updated_experience_ids"] == (["updated-experience"] if expected else [])
    snapshot.applied_proposals.clear()
    live._confirm_closed_loop(report, snapshot)
    assert not report["closed_loop_confirmed"] and not report["full_mechanism_exercised"]


def test_no_direct_update_cannot_claim_full_loop_or_reuse():
    report = {"stages": {"evolution": "completed", "held_out": "completed"},
              "evolution": [{"task_id": "e", "evaluation_complete": True, "experience_updates": []}],
              "held_out": [{"evaluation_complete": True}], "held_out_retrieved_experience_ids": []}
    live._confirm_closed_loop(report, SimpleNamespace(experiences=[], applied_proposals=[]))
    assert not report["full_mechanism_exercised"]
    assert not report["experience_update_complete"]
    assert not report["updated_experience_reuse_demonstrated"]
    assert not report["closed_loop_confirmed"]


def test_usage_report_never_equates_reserved_attempts_or_tokens_with_physical_http():
    records = [{"kind": "model", "estimated": False, "input_tokens": 70000, "output_tokens": 1500},
               {"kind": "model", "estimated": True, "input_tokens": 8456, "output_tokens": 4096},
               {"kind": "model", "estimated": True, "input_tokens": 8456, "output_tokens": 4096}]
    budget = {"model_calls": 4, "tokens": 96604, "reserved_tokens": 9000, "records": records}
    before = copy.deepcopy(budget)
    usage = live._usage_accounting(budget)
    assert budget == before
    assert usage["model_attempts"] == 4
    assert usage["pending_model_attempts"] == 1
    assert usage["attempts_with_complete_token_counters"] == 1
    assert usage["attempts_without_complete_token_counters"] == 2
    assert usage["known_usage_tokens"] == 71500
    assert usage["estimated_reserved_tokens"] == 25104
    assert usage["known_usage_tokens"] + usage["estimated_reserved_tokens"] == budget["tokens"]
    assert usage["pending_reserved_tokens"] == 9000
    assert usage["physical_http_requests"] is None


@pytest.mark.parametrize("source,anchor", [("source-directory", None), (None, "source-hash")])
def test_submitted_resume_requires_explicit_path_and_hash_before_loading(monkeypatch, credentials, source, anchor):
    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete resume request must fail before data/model setup")

    monkeypatch.setattr(live, "load_inputs", forbidden)
    with pytest.raises(ValueError, match="resume-source"):
        live.run_benchmark(credentials, "unused", "unused", "unused", unsafe_local=True,
                           resume_source=source, resume_source_hash=anchor)


def test_cli_forwards_explicit_submitted_resume_anchor(monkeypatch, capsys, credentials):
    captured = []

    def run(*args, **kwargs):
        captured.append(kwargs)
        return {"status": "completed", "budget": {"model_calls": 0}}

    monkeypatch.setattr(live, "run_benchmark", run)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "endpoint": credentials.endpoint, "model": credentials.model, "api_key": credentials.api_key})))
    assert live.main(["--data", "unused", "--splits", "unused", "--output-dir", "unused",
                      "--resume-source", "source-directory", "--resume-source-hash", "explicit-hash",
                      "--resume-attribution", "attribution-directory",
                      "--resume-attribution-hash", "explicit-attribution-hash"]) == 0
    assert captured[0]["resume_source"] == "source-directory"
    assert captured[0]["resume_source_hash"] == "explicit-hash"
    assert captured[0]["resume_attribution"] == "attribution-directory"
    assert captured[0]["resume_attribution_hash"] == "explicit-attribution-hash"
    assert credentials.api_key not in capsys.readouterr().out


@pytest.mark.parametrize("directory,anchor,source", [
    ("frozen-attribution", None, "source"), (None, "anchor", "source"),
    ("frozen-attribution", "anchor", None)])
def test_frozen_attribution_requires_both_anchors_before_loading(monkeypatch, credentials, directory, anchor, source):
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen-attribution flags must be checked before any setup")

    monkeypatch.setattr(live, "load_inputs", forbidden)
    with pytest.raises(ValueError, match="resume-attribution"):
        live.run_benchmark(credentials, "unused", "unused", "unused", unsafe_local=True,
            resume_source=source, resume_source_hash="source-anchor" if source else None,
            resume_attribution=directory, resume_attribution_hash=anchor)


def test_runner_completion_with_zero_proposals_does_not_confirm_closed_loop(tmp_path, monkeypatch, credentials):
    manifest = SplitManifest(evolution=["e"], validation=["v1", "v2"], test=["t"])
    monkeypatch.setattr(live, "load_inputs", lambda *args: ({}, {}, manifest))
    monkeypatch.setattr(live, "LiveModels", lambda *args: SimpleNamespace(close=lambda: None))

    class Pipeline:
        def __init__(self, *args):
            self.output = args[-1]

        def run(self, mode, **kwargs):
            run_dir = self.output / "offline-fixture"
            run_dir.mkdir(parents=True)
            (run_dir / "planning_calls.json").write_text(json.dumps([
                {"phase": "predict", "messages": [{"content": json.dumps({"experiences": []})}]}]),
                encoding="utf-8")
            return [{"task_id": "e" if mode == "evolve" else "t",
                     "evaluation": {"complete": True, "score": 0.5},
                     "proposals": [], "experience_updates": [], "experience_version": 0,
                     "run_dir": str(run_dir)}]

    monkeypatch.setattr(live, "MASPipeline", Pipeline)
    report = live.run_benchmark(credentials, "unused-data", "unused-splits", tmp_path / "run",
                                unsafe_local=True)
    assert report["status"] == "completed"
    assert report["stages"] == {"evolution": "completed", "held_out": "completed"}
    assert not report["closed_loop_confirmed"]
    assert not report["full_mechanism_exercised"]
    assert not report["experience_update_complete"]
    assert not report["new_experience_applied"]
    assert report["held_out_retrieved_updated_experience_ids"] == []
    assert report["unused_validation_task_ids"] == ["v1", "v2"]
    assert not any(key.startswith("paired_validation") for key in report)
    assert report["budget"]["model_calls"] == 0
    assert report["usage_accounting"]["model_attempts"] == 0
    assert report["usage_accounting"]["physical_http_requests"] is None
    assert report["limits"]["structured_output_corrections_per_phase"] == 1
    assert report["limits"]["max_harness_repairs_per_candidate"] == 2
    assert report["limits"]["automatic_transport_retries"] == 0


def test_direct_update_live_report_with_real_offline_pipeline(tmp_path, monkeypatch, credentials):
    tasks, private, manifest = fixture_dataset()
    monkeypatch.setattr(live, "load_inputs", lambda *args: (tasks, private, manifest))
    monkeypatch.setattr(live, "LiveModels", lambda *args: SimpleNamespace(close=lambda: None))
    provider = FixtureModels()

    def pipeline(*args):
        store, output = args[-2:]
        return make_pipeline(MASConfig(backend="scripted"), store, output, fixture_models=provider)

    monkeypatch.setattr(live, "MASPipeline", pipeline)
    output = tmp_path / "run"
    report = live.run_benchmark(credentials, "unused", "unused", output, unsafe_local=True)
    assert report["status"] == "completed", report.get("error")
    assert report["closed_loop_confirmed"] and report["experience_update_complete"]
    assert report["experience_version"] == 1
    assert report["held_out_retrieved_updated_experience_ids"] == ["boundary-assumptions"]
    assert len(report["evolution"][0]["experience_updates"]) == 1
    assert not any(key.startswith("paired_validation") for key in report)
    assert not any(output.rglob("validation.json"))
    for budget_path in output.rglob("budget.json"):
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        assert "validation" not in budget["by_stage"]


@pytest.mark.parametrize("directory,anchor", [
    ("prior", None), (None, "hash"), ("prior", "hash")])
def test_historical_heldout_continuation_fails_before_loading_or_api(
        tmp_path, monkeypatch, credentials, directory, anchor):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical continuation must fail before setup")

    monkeypatch.setattr(live, "load_inputs", forbidden)
    monkeypatch.setattr(live, "LiveModels", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="historical paired-validation protocol is unsupported"):
        live.run_benchmark(credentials, "unused", "unused", output, unsafe_local=True,
            resume_heldout=directory, resume_heldout_hash=anchor)
    assert not output.exists()


def test_cli_forwards_explicit_heldout_continuation_anchor(monkeypatch, capsys, credentials):
    captured = []
    def run(*args, **kwargs):
        captured.append(kwargs)
        return {"status": "completed", "budget": {"model_calls": 0}}

    monkeypatch.setattr(live, "run_benchmark", run)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "endpoint": credentials.endpoint, "model": credentials.model, "api_key": credentials.api_key})))
    assert live.main(["--data", "unused", "--splits", "unused", "--output-dir", "unused",
        "--resume-heldout", "prior-directory", "--resume-heldout-hash", "caller-anchor",
        "--max-calls", "39", "--max-tokens", "521350"]) == 0
    assert captured[0]["resume_heldout"] == "prior-directory"
    assert captured[0]["resume_heldout_hash"] == "caller-anchor"
    assert captured[0]["max_calls"] == 39 and captured[0]["max_tokens"] == 521350
    assert credentials.api_key not in capsys.readouterr().out
