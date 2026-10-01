"""Offline vertical integration through real JIT generation/loading/execution."""

import json
from pathlib import Path

import pytest

from jit_mas.config import MASConfig, NativeModels
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels
from jit_mas.schemas import AgentSpec, Experience, ExperienceSnapshot, TeamSpec, digest
from jit_mas.pipeline import attribution_source_digest, submitted_source_digest
from scripts.run_jit_mas import make_pipeline


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from scripts.models.openai_server import OpenAIServerModel
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline integration must not issue real model requests")
    monkeypatch.setattr(OpenAIServerModel, "__call__", forbidden)
    store = ExperienceStore(tmp_path / "state.sqlite")
    provider = FixtureModels()
    pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs", fixture_models=provider)
    yield pipeline, store, provider
    store.close()


def read_run(outcome, filename):
    return json.loads((Path(outcome["run_dir"]) / filename).read_text(encoding="utf-8"))


def test_complete_vertical_loop_and_reuse(setup):
    pipeline, store, provider = setup
    outcome = pipeline.run("evolve")[0]
    receipt = outcome["experience_updates"][0]
    assert receipt["update_rule"] == "direct_after_attribution"
    assert receipt["snapshot_hash"] == digest(store.snapshot())
    assert "validations" not in outcome and "validation" not in outcome["costs"]
    assert store.snapshot().version == 1
    assert outcome["submitted_at"] < outcome["evaluated_at"]
    assert read_run(outcome, "evaluation.json")["raw"]["submission_answer_hash"] == outcome["answer_hash"]
    frozen = read_run(outcome, "frozen_plan.json")
    assert frozen["R_global"] != frozen["R_planned"]
    assert frozen["hashes"]["TeamSpec"] == digest(frozen["TeamSpec"])
    result = read_run(outcome, "execution.json")
    assert len(result["sub_runs"]) == 3
    assert all(r["trajectory"][0]["model_input_messages"] for r in result["sub_runs"])
    kinds = {e["kind"] for e in result["metadata"]["events"]}
    assert {"artifact_published", "final_answer"} <= kinds
    assert not {"message_sent", "message_consumed", "evidence_read", "issue"}.intersection(kinds)
    before = digest(store.snapshot())
    following = pipeline.run("evaluate", limit=2)
    assert digest(store.snapshot()) == before
    next_plan = read_run(following[0], "frozen_plan.json")
    poem_plan = read_run(following[1], "frozen_plan.json")
    assert len(next_plan["TeamSpec"]["agents"]) == 3
    assert [a["agent_id"] for a in next_plan["TeamSpec"]["agents"]] == [
        a["agent_id"] for a in frozen["TeamSpec"]["agents"]]
    assert next_plan["TeamSpec"]["agents"][0]["responsibilities"] != frozen["TeamSpec"]["agents"][0]["responsibilities"]
    assert len(poem_plan["TeamSpec"]["agents"]) == 1
    assert any(r["experience_ids"] for r in next_plan["R_global"]["rubrics"])
    assert following[0]["proposals"] == []
    assert outcome["budget"]["cost"] is None
    assert {"inference", "evaluation", "update"} <= set(outcome["budget"]["by_stage"])


def test_canary_not_in_pre_execution_or_generation_workspaces(setup):
    pipeline, store, provider = setup
    outcome = pipeline.run("evolve")[0]
    for call in provider.calls:
        if call["role"] == "exec":
            assert "PRIVATE_CANARY" not in json.dumps(call)
            assert "PRIVATE_REFERENCE_CANARY" not in json.dumps(call)
            assert "Keep the response concise" in json.dumps(call)
        elif call["role"] in {"global", "local"}:
            payload = json.loads(call["messages"][-1]["content"])
            if payload["phase"] in {"predict", "local_plan", "reconcile"}:
                assert "PRIVATE_CANARY" not in json.dumps(call)
    artifact = read_run(outcome, "harness.json")
    assert "PRIVATE_CANARY" not in json.dumps(artifact["meta_trajectory"])
    for path in Path(artifact["path"]).glob("*"):
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}:
            assert "PRIVATE_CANARY" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("local_rounds", [1, 3])
def test_pre_execution_local_rounds_do_not_add_executor_passes(setup, local_rounds):
    pipeline, store, provider = setup
    pipeline.config.local_rounds = local_rounds
    outcome = pipeline.run("evaluate", limit=1)[0]
    planning = [json.loads(call["messages"][-1]["content"]) for call in provider.calls
                if call["role"] in {"global", "local"}]
    assert sum(call["phase"] == "predict" for call in planning) == 1
    assert sum(call["phase"] == "local_plan" for call in planning) == 3 * local_rounds
    assert sum(call["phase"] == "reconcile" for call in planning) == local_rounds
    execution = [call for call in provider.calls if call["role"] == "exec"]
    assert len(execution) == 3
    assert {call["agent_id"] for call in execution} == {"analyst", "evidence", "writer"}
    assert all(not any(message["role"] == "assistant" for message in call["messages"])
               for call in execution)
    writer = next(call for call in execution if call["agent_id"] == "writer")
    payload = json.loads(writer["messages"][1]["content"])
    assert "upstream_artifacts" not in payload
    shared = payload["shared_ledger"]
    assert {row["agent_id"] for row in shared["contributions"]} == {"analyst", "evidence"}
    assert shared["requirements"] and shared["outline"]
    assert {row["text"] for row in shared["evidence_spans"]} == {
        pipeline.tasks[outcome["task_id"]].question}
    assert all(row["locator"] == "public_task.question" for row in shared["source_references"])
    result = read_run(outcome, "execution.json")
    assert all(len(run["trajectory"]) == 1 for run in result["sub_runs"])
    assert result["metadata"]["model_calls_used"] == 3
    assert "shared_ledger_ready" in {event["kind"] for event in result["metadata"]["events"]}
    assert outcome["budget"]["by_stage"]["evaluation"]["model_calls"] == 1
    trace = read_run(outcome, "call_trace.json")
    assert trace["task_id"] == outcome["task_id"] and trace["run_key"] == outcome["run_key"]
    assert trace["coordination"] == "single_pass_shared_ledger"
    assert trace["shared_ledger_hash"] == result["metadata"]["shared_ledger_hash"]
    events = [event for event in result["metadata"]["events"] if event["kind"] == "agent_call"]
    assert len(trace["execution_calls"]) == 3
    assert {call["role"].casefold() for call in trace["execution_calls"]} == {"analyst", "evidence", "writer"}
    assert trace["execution_calls"] == [{**event["content"], "agent_id": event["agent_id"],
        "event_id": event["event_id"], "timestamp": event["timestamp"]} for event in events]
    evaluation_records = [record for record in outcome["budget"]["records"]
                          if record["stage"] == "evaluation" and record["kind"] == "model"]
    assert trace["evaluator_calls"] == evaluation_records
    assert len(trace["evaluator_calls"]) == 1
    assert sum(record["input_tokens"] + record["output_tokens"] for record in trace["evaluator_calls"]) == (
        outcome["budget"]["by_stage"]["evaluation"]["tokens"])
    assert not any("round" in key for key in trace)
    assert not any("round" in key for call in trace["execution_calls"] for key in call)
    assert store.snapshot().version == 0


def test_call_trace_preserves_real_evaluator_calls_when_evaluation_raises(setup):
    pipeline, store, _ = setup
    evaluator_factory = pipeline.evaluator_factory

    def failing_factory(model):
        evaluator = evaluator_factory(model)
        evaluate = evaluator.evaluate

        def fail_after_call(*args, **kwargs):
            evaluate(*args, **kwargs)
            raise RuntimeError("Synthetic failure after the evaluator call")

        evaluator.evaluate = fail_after_call
        return evaluator

    pipeline.evaluator_factory = failing_factory
    with pytest.raises(RuntimeError, match="Synthetic failure") as caught:
        pipeline.run("evaluate", limit=1)
    run_dir = Path(caught.value.jit_mas_run_failure["run_dir"])
    trace = json.loads((run_dir / "call_trace.json").read_text(encoding="utf-8"))
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))
    assert len(trace["execution_calls"]) == 3
    assert trace["evaluator_calls"] == [record for record in budget["records"]
        if record["kind"] == "model" and record["stage"] == "evaluation"]
    assert len(trace["evaluator_calls"]) == 1
    assert not (run_dir / "complete.json").exists()
    assert store.snapshot().version == 0


def test_call_trace_records_each_rubric_evaluation_call(setup):
    from benchmark.adapter.researchrubrics import split_item

    pipeline, _, provider = setup
    task_id = pipeline.manifest.test[0]
    original = pipeline.private_records[task_id]
    raw = {**original["raw_record"], "rubrics": original["raw_record"]["rubrics"] * 3}
    _, private = split_item(raw)
    private["source"] = original["source"]
    pipeline.private_records[task_id] = private
    outcome = pipeline.run("evaluate", [task_id])[0]
    trace = read_run(outcome, "call_trace.json")
    assert len(trace["evaluator_calls"]) == len(outcome["evaluation"]["rubrics"]) == 3
    assert len([call for call in provider.calls if call["role"] == "judge"]) == 3
    assert len({record["call_id"] for record in trace["evaluator_calls"]}) == 3
    assert trace["evaluator_calls"] == [record for record in outcome["budget"]["records"]
        if record["kind"] == "model" and record["stage"] == "evaluation"]


def test_resume_does_not_resubmit_or_commit_twice(setup):
    pipeline, store, provider = setup
    first = pipeline.run("evolve")[0]
    calls, version = len(provider.calls), store.snapshot().version
    again = pipeline.run("evolve")[0]
    assert again["resumed"] and first["run_key"] == again["run_key"]
    assert len(provider.calls) == calls and store.snapshot().version == version


def test_restart_after_atomic_update_restores_receipt_without_calls_or_duplicate_write(setup, monkeypatch):
    pipeline, store, provider = setup
    save = store.save_task_run

    def crash_before_completion(mode, task_id, identity, baseline, status, outcome=None):
        if status == "complete":
            raise RuntimeError("Interrupted after direct update")
        return save(mode, task_id, identity, baseline, status, outcome)

    monkeypatch.setattr(store, "save_task_run", crash_before_completion)
    with pytest.raises(RuntimeError, match="Interrupted"):
        pipeline.run("evolve")
    calls = len(provider.calls)
    assert store.snapshot().version == 1
    assert store.task_run("evolve", pipeline.manifest.evolution[0])["status"] == "submitted"
    monkeypatch.setattr(store, "save_task_run", save)
    resumed = pipeline.run("evolve")[0]
    assert len(provider.calls) == calls
    assert len(resumed["experience_updates"]) == 1
    assert resumed["experience_updates"][0]["snapshot_hash"] == digest(store.snapshot())
    assert resumed["next_experience_version"] == store.snapshot().version == 1


@pytest.mark.parametrize("obsolete", [{"validation": {}}, {"max_validation_tasks": 2}])
def test_removed_promotion_settings_fail_explicitly(obsolete):
    with pytest.raises(ValueError, match="remove obsolete"):
        MASConfig.model_validate({"backend": "scripted", **obsolete})


def test_stream_past_state_and_restart_order(setup):
    pipeline, store, provider = setup
    with pytest.raises(ValueError, match="manifest order"):
        pipeline.run("stream", ["stream-next"])
    rows = pipeline.run("stream", limit=2)
    assert rows[0]["experience_version"] == 0
    assert rows[1]["experience_version"] == 1
    calls = len(provider.calls)
    resumed = pipeline.run("stream", ["stream-first"])[0]
    assert resumed["experience_version"] == 0 and resumed["resumed"]
    assert len(provider.calls) == calls


def test_changed_policy_cannot_reuse_evolution_journal(setup):
    pipeline, store, _ = setup
    pipeline.run("evolve")
    pipeline.config.local_planning = False
    with pytest.raises(ValueError, match="changed policy"):
        pipeline.run("evolve")


def test_ablation_and_frozen_store(setup, tmp_path):
    pipeline, store, provider = setup
    pipeline.config.local_planning = False
    pipeline.config.local_attribution = False
    pipeline.config.persistent_experience = False
    pipeline.config.explicit_rubrics = False
    result = pipeline.run("evaluate")[0]
    frozen = read_run(result, "frozen_plan.json")
    assert not frozen["R_global"]["rubrics"]
    assert not frozen["TeamSpec"]["coverage"]
    assert len(frozen["TeamSpec"]["agents"]) == 3
    assert store.snapshot().version == 0
    store.freeze(tmp_path / "frozen.json")
    readonly = ExperienceStore(store.path, read_only=True)
    try:
        new_pipeline = make_pipeline(MASConfig(backend="scripted"), readonly, tmp_path / "frozen-runs")
        assert new_pipeline.run("evaluate")[0]["next_experience_version"] == 0
        with pytest.raises(PermissionError):
            readonly.rollback(0)
    finally:
        readonly.close()


def test_native_missing_endpoint_fails_without_fallback():
    with pytest.raises(ValueError, match="endpoint"):
        NativeModels(MASConfig(backend="native_jit"))


def test_snapshot_changes_cache_identity(setup):
    pipeline, store, provider = setup
    old = pipeline.run("evaluate")[0]
    pipeline.run("evolve")
    new = pipeline.run("evaluate")[0]
    assert old["run_key"] != new["run_key"]
    assert old["experience_hash"] != new["experience_hash"]


def test_attachment_contents_invalidate_cache(setup, tmp_path):
    pipeline, store, _ = setup
    attachment = tmp_path / "public.txt"
    attachment.write_text("First public evidence", encoding="utf-8")
    pipeline.tasks["test-deployment"].attachments = [str(attachment)]
    first = pipeline.run("evaluate")[0]
    attachment.write_text("Updated public evidence", encoding="utf-8")
    second = pipeline.run("evaluate")[0]
    assert first["run_key"] != second["run_key"]


def test_stage_failure_preserves_budget_and_error(setup, monkeypatch):
    pipeline, store, provider = setup
    original = provider.create
    def create(role, agent_id, ledger, stage):
        wrapped = original(role, agent_id, ledger, stage)
        if role == "global":
            def broken(messages, **kwargs):
                raise RuntimeError("synthetic transport failure")
            wrapped.model = broken
        return wrapped
    monkeypatch.setattr(provider, "create", create)
    with pytest.raises(RuntimeError, match="synthetic transport") as caught:
        pipeline.run("evaluate")
    failures = list(pipeline.output_dir.rglob("failure.json"))
    assert len(failures) == 1
    budget = json.loads(failures[0].with_name("budget.json").read_text())
    assert budget["model_calls"] == 1 and budget["tokens"] > 0
    assert budget["reserved_tokens"] == 0
    for field in ("model_calls", "tokens", "reserved_tokens", "records"):
        assert caught.value.jit_mas_run_failure["budget"][field] == budget[field]
        assert json.loads(failures[0].read_text())["budget"][field] == budget[field]
    assert store.snapshot().version == 0


def test_direct_update_never_runs_validation_tasks_or_charges_paired_calls(setup, monkeypatch):
    pipeline, store, _ = setup
    original = pipeline._run_task

    executed = []
    def forbid_validation(task_id, snapshot, ledger, audit, **kwargs):
        executed.append(task_id)
        if kwargs["mode"] == "validation" or task_id in pipeline.manifest.validation:
            raise AssertionError("Direct update must not execute promotion tasks")
        return original(task_id, snapshot, ledger, audit, **kwargs)

    monkeypatch.setattr(pipeline, "_run_task", forbid_validation)
    pipeline.manifest.validation = []
    outcome = pipeline.run("evolve")[0]
    assert executed == [outcome["task_id"]]
    assert outcome["experience_updates"][0]["version"] == store.snapshot().version == 1
    assert "validations" not in outcome and "validation" not in outcome["costs"]
    for field in ("model_calls", "tokens", "tool_calls"):
        assert sum(stage.get(field, 0) for stage in outcome["costs"].values()) == outcome["budget"][field]
    held_out = pipeline.run("evaluate")[0]
    assert held_out["experience_version"] == 1 and held_out["experience_updates"] == []
    assert store.snapshot().version == 1


def test_failed_attribution_persists_received_calls_after_submission(setup, monkeypatch):
    pipeline, store, provider = setup
    create = provider.create

    def corrupt_post(role, agent_id, ledger, stage):
        model = create(role, agent_id, ledger, stage)
        if agent_id != "global-post":
            return model

        class InvalidJSON:
            def __call__(self, messages, **kwargs):
                response = model(messages, **kwargs)
                response.content = '{"matches":'
                return response

        return InvalidJSON()

    monkeypatch.setattr(provider, "create", corrupt_post)
    with pytest.raises(json.JSONDecodeError):
        pipeline.run("evolve")
    path = next(pipeline.output_dir.glob("*/attribution.json"))
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert not recorded["complete"]
    assert len(recorded["calls"]) == 2
    assert all(call["phase"] == "align" and call["response"] == '{"matches":'
               for call in recorded["calls"])
    assert recorded["proposals"] == []
    assert path.with_name("submission.json").is_file()
    assert json.loads(path.with_name("evaluation.json").read_text())["complete"]
    assert path.with_name("failure.json").is_file()
    assert store.snapshot().version == 0


def test_explicit_submitted_resume_reuses_immutable_answer_and_eval_only(setup, tmp_path, monkeypatch):
    pipeline, _, _ = setup
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, ExperienceSnapshot(), mode="evolve", attribution=False, resume=False)
    source_dir = Path(source["run_dir"])
    manifest_path = source_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["comparison"]["code"] = "historical-source-code"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    anchor = submitted_source_digest(source_dir)
    original = {path.name: path.read_bytes() for path in source_dir.iterdir() if path.is_file()}
    provider = FixtureModels()
    state = ExperienceStore(tmp_path / "continuation.sqlite")
    continued = make_pipeline(pipeline.config, state, tmp_path / "continued", fixture_models=provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("Source continuation must not generate or execute another source harness")

    monkeypatch.setattr(continued, "synthesizer_factory", forbidden)
    try:
        outcome = continued.run_task(task_id, state.snapshot(), mode="evolve", attribution=True,
            resume_source=source_dir, resume_source_hash=anchor)
        assert provider.calls and all(call["agent_id"] in {"global-post", "analyst", "evidence", "writer"}
                                      for call in provider.calls)
        assert all(call["role"] in {"global", "local"} for call in provider.calls)
        assert set(outcome["budget"]["by_stage"]) == {"update"}
        assert outcome["answer_hash"] == source["answer_hash"]
        assert outcome["evaluation"] == source["evaluation"]
        assert outcome["comparison_fingerprint"] == digest(manifest["comparison"])
        assert outcome["resume_provenance"]["source_code_hash"] == "historical-source-code"
        assert outcome["resume_provenance"]["continuation_code_hash"] == continued.code_hash
        assert outcome["source_submission_reused"] and outcome["source_evaluation_reused"]
        assert outcome["resume_provenance"]["evaluation_has_recorded_submission_hash"]
        assert outcome["historical_source_budget"]["model_calls"] == source["budget"]["model_calls"]
        assert outcome["proposals"]
        assert submitted_source_digest(source_dir) == anchor
        assert all((source_dir / name).read_bytes() == content for name, content in original.items())
        for name in ("submission.json", "evaluation.json", "execution.json"):
            assert (Path(outcome["run_dir"]) / name).read_bytes() == original[name]
    finally:
        state.close()


@pytest.mark.parametrize("mutation", ["anchor", "task", "private", "config", "answer", "evaluation",
                                      "evaluation_answer_hash", "evaluation_null_hash", "sidecar"])
def test_submitted_resume_rejects_changed_source_before_model_calls(setup, tmp_path, mutation):
    pipeline, _, _ = setup
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, ExperienceSnapshot(), mode="evolve", attribution=False, resume=False)
    source_dir = Path(source["run_dir"])
    anchor = submitted_source_digest(source_dir)
    if mutation in {"task", "private", "config"}:
        path = source_dir / "run_manifest.json"
        manifest = json.loads(path.read_text())
        if mutation == "task":
            manifest["comparison"]["task"]["question"] = "Wrong public task"
        elif mutation == "private":
            manifest["comparison"]["private_hash"] = "wrong-private-record"
        else:
            manifest["comparison"]["config"]["max_total_tokens"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
        anchor = submitted_source_digest(source_dir)
    elif mutation in {"anchor", "answer"}:
        path = source_dir / "submission.json"
        submission = json.loads(path.read_text())
        submission["answer"] = "Changed answer"
        path.write_text(json.dumps(submission), encoding="utf-8")
        if mutation == "answer":
            anchor = submitted_source_digest(source_dir)
    elif mutation in {"evaluation", "evaluation_answer_hash", "evaluation_null_hash"}:
        path = source_dir / "evaluation.json"
        evaluation = json.loads(path.read_text())
        if mutation == "evaluation":
            evaluation["complete"] = False
        else:
            evaluation["raw"]["submission_answer_hash"] = (
                "wrong-submission-hash" if mutation == "evaluation_answer_hash" else None)
        path.write_text(json.dumps(evaluation), encoding="utf-8")
        anchor = submitted_source_digest(source_dir)
    else:
        artifact = json.loads((source_dir / "harness.json").read_text())
        path = Path(artifact["path"]) / "team.json"
        sidecar = json.loads(path.read_text())
        sidecar["task"]["question"] = "Changed sidecar task"
        path.write_text(json.dumps(sidecar), encoding="utf-8")
    provider = FixtureModels()
    state = ExperienceStore(tmp_path / "continuation.sqlite")
    continued = make_pipeline(pipeline.config, state, tmp_path / "continued", fixture_models=provider)
    try:
        with pytest.raises(ValueError):
            continued.run_task(task_id, state.snapshot(), mode="evolve", attribution=True,
                resume_source=source_dir, resume_source_hash=anchor)
        assert provider.calls == []
        assert state.snapshot().version == 0
    finally:
        state.close()


def test_legacy_submitted_resume_reports_missing_answer_binding_without_fabricating_it(setup, tmp_path):
    pipeline, _, _ = setup
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, ExperienceSnapshot(), mode="evolve", attribution=False, resume=False)
    source_dir = Path(source["run_dir"])
    path = source_dir / "evaluation.json"
    evaluation = json.loads(path.read_text())
    del evaluation["raw"]["submission_answer_hash"]
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    original_bytes = path.read_bytes()
    anchor = submitted_source_digest(source_dir)
    state = ExperienceStore(tmp_path / "continuation.sqlite")
    continued = make_pipeline(pipeline.config, state, tmp_path / "continued", fixture_models=FixtureModels())
    try:
        outcome = continued.run_task(task_id, state.snapshot(), mode="evolve", attribution=True,
            resume_source=source_dir, resume_source_hash=anchor)
        provenance = outcome["resume_provenance"]
        assert not provenance["evaluation_has_recorded_submission_hash"]
        assert provenance["evaluation_submission_binding"] == (
            "Legacy evaluation has no recorded submission hash; explicit caller anchor trusted")
        assert "submission_answer_hash" not in outcome["evaluation"]["raw"]
        assert path.read_bytes() == original_bytes
        assert (Path(outcome["run_dir"]) / "evaluation.json").read_bytes() == original_bytes
        assert submitted_source_digest(source_dir) == anchor
    finally:
        state.close()


def test_submitted_resume_updates_directly_then_runs_held_out(setup, tmp_path):
    pipeline, _, _ = setup
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, ExperienceSnapshot(), mode="evolve", attribution=False, resume=False)
    provider = FixtureModels()
    state = ExperienceStore(tmp_path / "continuation.sqlite")
    continued = make_pipeline(pipeline.config, state, tmp_path / "continued", fixture_models=provider)
    try:
        outcome = continued.run("evolve", resume=False, resume_source=source["run_dir"],
            resume_source_hash=submitted_source_digest(source["run_dir"]))[0]
        assert outcome["source_submission_reused"]
        assert outcome["experience_updates"][0]["update_rule"] == "direct_after_attribution"
        assert "validation" not in outcome["costs"]
        assert all(call["role"] not in {"exec", "judge", "meta"} for call in provider.calls)
        assert state.snapshot().version == 1
        held_out = continued.run("evaluate", resume=False)[0]
        assert held_out["experience_version"] == 1
        assert held_out["budget"]["by_stage"]["evaluation"]["model_calls"] > 0
        with pytest.raises(ValueError, match="fresh store"):
            continued.run("evolve", resume_source=source["run_dir"],
                resume_source_hash=submitted_source_digest(source["run_dir"]))
    finally:
        state.close()


def _frozen_attribution_source(setup, tmp_path):
    pipeline, _, _ = setup
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, ExperienceSnapshot(), mode="evolve", attribution=False, resume=False)
    anchor = submitted_source_digest(source["run_dir"])
    state = ExperienceStore(tmp_path / "attribution-state.sqlite")
    first = make_pipeline(pipeline.config, state, tmp_path / "first-attribution", fixture_models=FixtureModels())
    try:
        attributed = first.run_task(task_id, state.snapshot(), mode="evolve", attribution=True,
            resume_source=source["run_dir"], resume_source_hash=anchor)
    finally:
        state.close()
    return source, anchor, attributed


def test_frozen_attribution_replay_uses_exact_proposals_without_any_model_calls(setup, tmp_path, monkeypatch):
    pipeline, _, _ = setup
    source, anchor, attributed = _frozen_attribution_source(setup, tmp_path)
    location = Path(attributed["run_dir"])
    attribute_anchor = attribution_source_digest(location)
    original = {path.name: path.read_bytes() for path in location.iterdir() if path.is_file()}
    state = ExperienceStore(tmp_path / "replay-state.sqlite")
    provider = FixtureModels()
    replay = make_pipeline(pipeline.config, state, tmp_path / "replay", fixture_models=provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen attribution replay must never construct/call a model")

    monkeypatch.setattr(provider, "create", forbidden)
    try:
        outcome = replay.run_task(source["task_id"], state.snapshot(), mode="evolve", attribution=True,
            resume_source=source["run_dir"], resume_source_hash=anchor,
            resume_attribution=location / "attribution.json", resume_attribution_hash=attribute_anchor)
        assert outcome["budget"]["model_calls"] == outcome["budget"]["tokens"] == 0
        assert outcome["proposals"] == attributed["proposals"]
        assert outcome["source_attribution_reused"]
        assert "attribution_continued_at" not in outcome
        assert outcome["resume_provenance"]["attribution_code_hash"] == attributed["resume_provenance"]["continuation_code_hash"]
        assert outcome["historical_attribution_budget"]["model_calls"] == attributed["budget"]["model_calls"]
        assert set(outcome["historical_attribution_budget"]["by_stage"]) == {"update"}
        assert outcome["historical_source_budget"] == attributed["historical_source_budget"]
        assert (Path(outcome["run_dir"]) / "attribution.json").read_bytes() == original["attribution.json"]
        assert all((location / name).read_bytes() == content for name, content in original.items())
        assert submitted_source_digest(source["run_dir"]) == anchor
    finally:
        state.close()


@pytest.mark.parametrize("mutation", ["anchor", "incomplete", "proposal_mismatch", "source_hash",
                                      "source_evaluation", "finding_evidence", "proposal_evidence", "baseline"])
def test_frozen_attribution_replay_rejects_tampering_before_model_calls(setup, tmp_path, mutation):
    pipeline, _, _ = setup
    source, anchor, attributed = _frozen_attribution_source(setup, tmp_path)
    location = Path(attributed["run_dir"])
    attribute_anchor = attribution_source_digest(location)
    attribution = json.loads((location / "attribution.json").read_text())
    complete = json.loads((location / "complete.json").read_text())
    if mutation in {"anchor", "incomplete"}:
        attribution["complete"] = False
    elif mutation == "proposal_mismatch":
        complete["proposals"][0]["experience"]["instruction"] = "Different frozen proposal"
    elif mutation == "source_hash":
        provenance = json.loads((location / "resume_provenance.json").read_text())
        provenance["source_hash"] = "different-source"
        complete["resume_provenance"] = provenance
        manifest = json.loads((location / "run_manifest.json").read_text())
        manifest["continuation"] = provenance
        (location / "resume_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (location / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "source_evaluation":
        evaluation = json.loads((location / "evaluation.json").read_text())
        evaluation["score"] = 99
        (location / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    elif mutation == "finding_evidence":
        attribution["findings"][0]["supporting_evidence"] = ["invented-event"]
    elif mutation == "proposal_evidence":
        attribution["proposals"][0]["evidence"] = ["invented-event"]
        complete["proposals"] = attribution["proposals"]
    else:
        complete["experience_version"] = 1
    (location / "attribution.json").write_text(json.dumps(attribution), encoding="utf-8")
    (location / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    if mutation != "anchor":
        attribute_anchor = attribution_source_digest(location)
    state = ExperienceStore(tmp_path / "replay-state.sqlite")
    provider = FixtureModels()
    replay = make_pipeline(pipeline.config, state, tmp_path / "replay", fixture_models=provider)
    try:
        with pytest.raises(ValueError):
            replay.run_task(source["task_id"], state.snapshot(), mode="evolve", attribution=True,
                resume_source=source["run_dir"], resume_source_hash=anchor,
                resume_attribution=location, resume_attribution_hash=attribute_anchor)
        assert not provider.calls
        assert state.snapshot().version == 0
    finally:
        state.close()


def test_frozen_first_proposal_is_applied_without_calls_then_held_out(setup, tmp_path):
    pipeline, _, _ = setup
    source, anchor, attributed = _frozen_attribution_source(setup, tmp_path)
    state = ExperienceStore(tmp_path / "replay-state.sqlite")
    replay = make_pipeline(pipeline.config, state, tmp_path / "replay", fixture_models=FixtureModels())
    try:
        outcome = replay.run("evolve", resume=False,
            resume_source=source["run_dir"], resume_source_hash=anchor,
            resume_attribution=attributed["run_dir"],
            resume_attribution_hash=attribution_source_digest(attributed["run_dir"]))[0]
        assert outcome["budget"]["model_calls"] == 0
        assert outcome["proposals"] == attributed["proposals"]
        update = outcome["experience_updates"][0]
        assert update["proposal_id"] == attributed["proposals"][0]["proposal_id"]
        assert update["version"] == 1 and update["update_rule"] == "direct_after_attribution"
        assert "validations" not in outcome and "validation" not in outcome["costs"]
        held_out = replay.run("evaluate", resume=False)[0]
        assert held_out["experience_version"] == 1
        assert held_out["budget"]["by_stage"]["evaluation"]["model_calls"] > 0
    finally:
        state.close()


def test_all_three_banks_reach_their_stage_and_capability(setup):
    pipeline, store, provider = setup
    entries = [Experience(experience_id=bank, bank=bank, instruction=f"Conditional {bank} advice",
                          applicability="comparison", capability="comparison" if bank == "execution" else "",
                          source_task_ids=["prior-task"], evidence=["prior-evidence"])
               for bank in ("rubric", "organization", "execution")]
    pipeline.run_task("test-deployment", ExperienceSnapshot(version=4, experiences=entries))
    predict = [json.loads(c["messages"][-1]["content"]) for c in provider.calls
               if c["role"] == "global" and '"phase": "predict"' in c["messages"][-1]["content"]][0]
    reconcile = [json.loads(c["messages"][-1]["content"]) for c in provider.calls
                 if c["role"] == "global" and '"phase": "reconcile"' in c["messages"][-1]["content"]][0]
    assert any(e["bank"] == "rubric" for e in predict["experiences"])
    assert any(e["bank"] == "organization" for e in reconcile["experiences"])
    locals_ = {c["agent_id"]: json.loads(c["messages"][1]["content"])
               for c in provider.calls if c["role"] == "exec"}
    assert locals_["analyst"]["execution_experiences"][0]["experience_id"] == "execution"
    assert not locals_["writer"]["execution_experiences"]


def test_fixed_team_honors_same_budget_and_review_requires_primary_artifact():
    agents = [AgentSpec(agent_id="a", role="author", capability="writing", rubric_ids=["r"]),
              AgentSpec(agent_id="b", role="review", capability="review", depends_on=["a"])]
    team = TeamSpec(agents=agents, synthesizer_id="b", coverage={"r": ["a"]},
                    primary={"r": "a"}, reviewers={"r": ["b"]}, total_max_calls=6)
    with pytest.raises(ValueError, match="Fixed team"):
        MASConfig(fixed_team=team, team_max_calls=5)
    invalid = team.model_dump()
    invalid["agents"][1]["depends_on"] = []
    with pytest.raises(ValueError, match="reviewer"):
        TeamSpec.model_validate(invalid)


def test_data_preparation_preserves_whole_task_splits(tmp_path):
    from jit_mas.offline import fixture_dataset
    from scripts.prepare_researchrubrics import main
    from jit_mas.schemas import SplitManifest

    tasks, private, _ = fixture_dataset()
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(row["raw_record"]) for row in private.values()), encoding="utf-8")
    main(["--input", str(raw), "--output-dir", str(tmp_path / "prepared"),
          "--evolution-size", "1", "--validation-size", "2"])
    manifest = SplitManifest.model_validate_json((tmp_path / "prepared" / "splits.json").read_text())
    assert set(manifest.evolution + manifest.validation + manifest.test) == set(tasks)
    assert len(manifest.evolution) == 1 and len(manifest.validation) == 2
