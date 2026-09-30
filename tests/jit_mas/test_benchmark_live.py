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
from jit_mas.offline import FixtureModels
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


def _scored_pair(task_id):
    side = {"comparison_fingerprint": "same-configuration",
            "evaluation": {"complete": True, "score": 0.5, "evaluator_version": "frozen-judge",
                           "rubrics": [{"rubric_id": "r1", "weight": 1, "status": "ok", "score": 1}]}}
    return {"task_id": task_id, "repeat": 0, "baseline": side, "candidate": copy.deepcopy(side)}


def test_validation_report_distinguishes_error_only_attempt_from_scored_pair():
    outcome = {"validations": [{"status": "pending", "pairs": [
        {"task_id": "v1", "repeat": 0, "error": "planning failed"}]}]}
    progress = live._validation_progress([outcome], ["v1", "v2"], 1)
    assert progress["paired_validation_attempted"]
    assert progress["paired_validation_attempted_pairs"] == 1
    assert progress["paired_validation_fully_scored_pairs"] == 0
    assert not progress["paired_validation_executed"]
    assert not progress["paired_validation_complete"]


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_validation_report_requires_whole_scored_protocol(status):
    outcome = {"validations": [{"status": status, "pairs": [_scored_pair("v1"), _scored_pair("v2")]}]}
    progress = live._validation_progress([outcome], ["v1", "v2"], 1)
    assert progress["paired_validation_fully_scored_pairs"] == 2
    assert progress["paired_validation_complete"] and progress["paired_validation_executed"]
    outcome["validations"][0]["pairs"][1]["candidate"]["evaluation"]["complete"] = False
    progress = live._validation_progress([outcome], ["v1", "v2"], 1)
    assert progress["paired_validation_fully_scored_pairs"] == 1
    assert not progress["paired_validation_complete"]


@pytest.mark.parametrize("retrieved,expected", [([], False), (["unaccepted"], False),
                                               (["different"], False), (["accepted"], True)])
def test_closed_loop_confirmation_requires_actual_accepted_retrieval(retrieved, expected):
    snapshot = SimpleNamespace(experiences=[
        SimpleNamespace(experience_id="accepted", validation_status="accepted"),
        SimpleNamespace(experience_id="unaccepted", validation_status="pending")])
    report = {"status": "completed", "stages": {"evolution": "completed", "held_out": "completed"},
              "evolution": [{"evaluation_complete": True}], "held_out": [{"evaluation_complete": True}],
              "paired_validation_complete": True, "new_experience_accepted": True,
              "held_out_retrieved_experience_ids": retrieved}
    live._confirm_closed_loop(report, snapshot)
    assert report["full_mechanism_exercised"]
    assert report["closed_loop_confirmed"] is expected
    assert report["accepted_experience_reuse_demonstrated"] is expected
    assert report["held_out_retrieved_accepted_experience_ids"] == (["accepted"] if expected else [])
    report["paired_validation_complete"] = False
    live._confirm_closed_loop(report, snapshot)
    assert not report["closed_loop_confirmed"]
    assert not report["full_mechanism_exercised"]


def test_rejected_experience_can_exercise_full_mechanism_without_reuse():
    report = {"stages": {"evolution": "completed", "held_out": "completed"},
              "evolution": [{"evaluation_complete": True}], "held_out": [{"evaluation_complete": True}],
              "paired_validation_complete": True, "new_experience_accepted": False,
              "held_out_retrieved_experience_ids": []}
    live._confirm_closed_loop(report, SimpleNamespace(experiences=[]))
    assert report["full_mechanism_exercised"]
    assert not report["accepted_experience_reuse_demonstrated"]
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
                     "proposals": [], "validations": [], "experience_version": 0,
                     "run_dir": str(run_dir)}]

    monkeypatch.setattr(live, "MASPipeline", Pipeline)
    report = live.run_benchmark(credentials, "unused-data", "unused-splits", tmp_path / "run",
                                unsafe_local=True)
    assert report["status"] == "completed"
    assert report["stages"] == {"evolution": "completed", "held_out": "completed"}
    assert not report["closed_loop_confirmed"]
    assert not report["full_mechanism_exercised"]
    assert not report["paired_validation_attempted"]
    assert not report["new_experience_accepted"]
    assert report["held_out_retrieved_accepted_experience_ids"] == []
    assert report["budget"]["model_calls"] == 0
    assert report["usage_accounting"]["model_attempts"] == 0
    assert report["usage_accounting"]["physical_http_requests"] is None
    assert report["limits"]["structured_output_corrections_per_phase"] == 1
    assert report["limits"]["max_harness_repairs_per_candidate"] == 2
    assert report["limits"]["automatic_transport_retries"] == 0


@pytest.fixture
def heldout_evidence(tmp_path, monkeypatch, credentials):
    root = tmp_path / "prior"
    root.mkdir()
    manifest = SplitManifest(evolution=["e"], validation=["v1", "v2"], test=["t"])
    tasks = {tid: PublicTask(task_id=tid, question="Public question " + tid)
             for tid in ["e", "v1", "v2", "t"]}
    private = {tid: {"rubrics": [{"rubric_id": tid + "-r", "criterion": "Official fixture criterion", "weight": 1}]}
               for tid in tasks}
    config = MASConfig(backend="native_jit", unsafe_local=True,
        models={role: ModelConfig(model=credentials.model, endpoint=credentials.endpoint,
            key_env="STDIN_ONLY_NOT_EXPORTED", max_tokens=limit, timeout=120)
            for role, limit in live.ROLE_TOKENS.items()}, max_agents=3, max_parallel=1,
        team_max_calls=6, max_model_calls=90, max_total_tokens=600_000,
        max_tool_calls=0, max_repairs=2, candidates=1, execution_timeout=750, max_validation_tasks=2)
    monkeypatch.setattr(live, "code_fingerprint", lambda: "frozen-runtime")
    monkeypatch.setattr(live, "load_inputs", lambda *args: (tasks, private, manifest))
    store = ExperienceStore(root / "experience.sqlite")
    snapshot = store.snapshot()
    empty_hash = digest(snapshot)

    def comparison(tid):
        return {"code": "frozen-runtime", "config": config.model_dump(mode="json"),
                "task": tasks[tid].model_dump(mode="json"), "manifest": digest(manifest),
                "private_hash": digest(private[tid])}

    def scored(tid, label, experience_hash):
        directory = root / "evolution" / label
        answer = "Unchanged submitted answer " + label
        answer_hash = digest(answer)
        evaluation = {"task_id": tid, "complete": True, "score": 1.0, "evaluator_version": "frozen-judge",
            "rubrics": [{**private[tid]["rubrics"][0], "score": 1.0, "status": "ok"}],
            "raw": {"submission_answer_hash": answer_hash}}
        outcome = {"task_id": tid, "evaluation": evaluation, "answer_hash": answer_hash,
            "comparison_fingerprint": digest(comparison(tid)), "experience_hash": experience_hash,
            "run_dir": str(directory.resolve())}
        for name, content in {"evaluation": evaluation, "complete": outcome,
                "submission": {"answer": answer, "answer_hash": answer_hash},
                "execution": {"answer": answer}, "run_manifest": {"comparison": comparison(tid)}}.items():
            live.write_json(directory / (name + ".json"), content)
        return outcome

    source = scored("e", "source", empty_hash)
    validation = {"status": "rejected", "reason": "No required mean task-quality improvement",
        "config_hash": digest(config.validation), "baseline_hash": empty_hash, "candidate_hash": "candidate",
        "pairs": [{"task_id": tid, "repeat": 0, "baseline": scored(tid, tid + "-base", empty_hash),
                   "candidate": scored(tid, tid + "-candidate", "candidate")} for tid in manifest.validation]}
    with store.db:
        store.db.execute("INSERT INTO validations VALUES(?,?,?)", ("validation", "p1", json.dumps(validation)))
    store.close()
    failed_dir = root / "held_out" / "failed"
    live.write_json(failed_dir / "failure.json", {"task_id": "t", "experience_hash": empty_hash})
    live.write_json(failed_dir / "run_manifest.json", {"comparison": comparison("t")})
    budget = {"model_calls": 0, "tokens": 0, "reserved_tokens": 0, "records": []}
    report = {"status": "failed", "stages": {"evolution": "completed", "held_out": "failed"},
        "accepted_experience_count": 0, "accepted_experience_version": 0,
        "manifest": manifest.model_dump(mode="json"), "dataset_revision": live.DATASET_REVISION,
        "dataset_sha256": live.OFFICIAL_DATA_SHA256, "budget": budget,
        "evolution": [{"task_id": "e", "score": 1.0, "evaluation_complete": True,
            "run_dir": source["run_dir"], "validations": [{"status": validation["status"],
                "reason": validation["reason"], "pairs": 2}]}]}
    report.update(live._validation_progress([{"validations": [validation]}], manifest.validation, 1))
    live.write_json(root / "report.json", report)
    live.write_json(root / "config.json", config)
    live.write_json(root / "session_budget.json", budget)
    return SimpleNamespace(root=root, config=config, tasks=tasks, private=private, manifest=manifest,
                           report=report, validation=validation, failed_dir=failed_dir)


def test_heldout_continuation_verifies_unchanged_complete_evidence(heldout_evidence):
    f = heldout_evidence
    before = live._heldout_evidence_files(f.root)
    old, provenance = live._load_heldout_continuation(f.root, live.heldout_continuation_digest(f.root),
        f.config, f.tasks, f.private, f.manifest)
    assert old == f.report
    assert provenance["report_sha256"] == before["report.json"]
    assert provenance["runtime_fingerprint"] == "frozen-runtime"
    assert not provenance["source_regenerated"]
    assert not provenance["attribution_regenerated"]
    assert not provenance["paired_validation_rerun"]
    assert live._heldout_evidence_files(f.root) == before


@pytest.mark.parametrize("mutation", ["anchor", "runtime", "config", "split", "submission", "pair", "committed", "report"])
def test_heldout_continuation_fails_closed_on_changed_inputs(heldout_evidence, monkeypatch, mutation):
    f = heldout_evidence
    anchor = live.heldout_continuation_digest(f.root)
    if mutation == "anchor":
        live.write_json(f.root / "extra.json", {"changed": True})
    elif mutation == "runtime":
        monkeypatch.setattr(live, "code_fingerprint", lambda: "changed-runtime")
    elif mutation == "config":
        f.config.max_repairs = 1
    elif mutation == "split":
        f.manifest.test = ["other"]
    elif mutation == "submission":
        live.write_json(f.failed_dir / "submission.json", {"answer": "already submitted"})
    elif mutation == "pair":
        path = f.root / "evolution" / "v1-base" / "evaluation.json"
        changed = json.loads(path.read_text())
        changed["score"] = 0.0
        live.write_json(path, changed)
    elif mutation == "committed":
        store = ExperienceStore(f.root / "experience.sqlite")
        with store.db:
            store.db.execute("INSERT INTO commits VALUES('p1',1,'validation')")
        store.close()
    elif mutation == "report":
        f.report["evolution"][0]["score"] = 0.25
        live.write_json(f.root / "report.json", f.report)
    if mutation != "anchor":
        anchor = live.heldout_continuation_digest(f.root)
    with pytest.raises(ValueError):
        live._load_heldout_continuation(f.root, anchor, f.config, f.tasks, f.private, f.manifest)


@pytest.mark.parametrize("directory,anchor,calls,tokens", [
    ("prior", None, 39, 521350), (None, "hash", 39, 521350),
    ("prior", "hash", 40, 521350), ("prior", "hash", 39, 521351)])
def test_heldout_flags_and_remaining_budget_rejected_before_loading(
        monkeypatch, credentials, directory, anchor, calls, tokens):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid continuation must fail before setup")

    monkeypatch.setattr(live, "load_inputs", forbidden)
    with pytest.raises(ValueError, match="[Hh]eld.?out"):
        live.run_benchmark(credentials, "unused", "unused", "unused", unsafe_local=True,
            resume_heldout=directory, resume_heldout_hash=anchor, max_calls=calls, max_tokens=tokens)


@pytest.mark.parametrize("changed_at_end", [False, True])
def test_heldout_only_continuation_never_regenerates_pairs_or_mutates_prior(
        heldout_evidence, tmp_path, monkeypatch, credentials, changed_at_end):
    f = heldout_evidence
    before = live._heldout_evidence_files(f.root)
    calls = []
    monkeypatch.setattr(live, "LiveModels", lambda *args: SimpleNamespace(close=lambda: None))

    class Pipeline:
        def __init__(self, *args):
            self.store, self.output = args[-2:]

        def run(self, mode, **kwargs):
            assert mode == "evaluate"
            assert kwargs == {"limit": 1, "resume": False}
            assert self.store.read_only
            assert self.store.path == (f.root / "experience.sqlite").resolve()
            calls.append(mode)
            run_dir = self.output / "fresh-heldout"
            live.write_json(run_dir / "planning_calls.json", [
                {"phase": "predict", "messages": [{"content": json.dumps({"experiences": []})}]}])
            return [{"task_id": "t", "evaluation": {"score": 0.5, "complete": True},
                     "experience_version": 0, "run_dir": str(run_dir)}]

    monkeypatch.setattr(live, "MASPipeline", Pipeline)
    anchor = live.heldout_continuation_digest(f.root)
    if changed_at_end:
        monkeypatch.setattr(live, "heldout_continuation_digest", lambda _: "changed-at-end")
    report = live.run_benchmark(credentials, "unused", "unused", tmp_path / "continued", unsafe_local=True,
        resume_heldout=f.root, resume_heldout_hash=anchor,
        max_calls=39, max_tokens=521350)
    assert calls == ["evaluate"]
    assert report["status"] == ("failed" if changed_at_end else "completed")
    assert report["full_mechanism_exercised"] is not changed_at_end
    if changed_at_end:
        assert "Immutable prior evidence changed" in report["error"]
    assert not report["closed_loop_confirmed"]
    assert report["evolution"] == f.report["evolution"]
    assert report["budget"]["model_calls"] == 0
    assert report["held_out_continuation"]["prior_session_usage"]["model_attempts"] == 0
    assert live._heldout_evidence_files(f.root) == before
    assert json.loads((f.root / "report.json").read_text())["status"] == "failed"
    assert not (tmp_path / "continued" / "evolution").exists()
    assert not (tmp_path / "continued" / "experience.sqlite").exists()


@pytest.mark.parametrize("nested", [False, True])
def test_heldout_output_cannot_modify_source_before_models(
        heldout_evidence, monkeypatch, credentials, nested):
    f = heldout_evidence
    before = live._heldout_evidence_files(f.root)
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid output must be rejected before model setup")

    monkeypatch.setattr(live, "LiveModels", forbidden)
    output = f.root / "nested-new-output" if nested else f.root
    with pytest.raises(ValueError, match="outside the immutable prior run"):
        live.run_benchmark(credentials, "unused", "unused", output, unsafe_local=True,
            resume_heldout=f.root, resume_heldout_hash=live.heldout_continuation_digest(f.root),
            max_calls=39, max_tokens=521350)
    assert live._heldout_evidence_files(f.root) == before
    if nested:
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
