"""Offline vertical integration through real JIT generation/loading/execution."""

import json
from pathlib import Path

import pytest

from jit_mas.config import MASConfig, NativeModels
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModels
from jit_mas.schemas import AgentSpec, Experience, ExperienceSnapshot, TeamSpec, digest
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
    assert outcome["validations"][0]["status"] == "accepted"
    assert len(outcome["validations"][0]["pairs"]) == 2
    assert store.snapshot().version == 1
    assert outcome["submitted_at"] < outcome["evaluated_at"]
    frozen = read_run(outcome, "frozen_plan.json")
    assert frozen["R_global"] != frozen["R_planned"]
    assert frozen["hashes"]["TeamSpec"] == digest(frozen["TeamSpec"])
    result = read_run(outcome, "execution.json")
    assert len(result["sub_runs"]) == 2
    assert all(r["trajectory"][0]["model_input_messages"] for r in result["sub_runs"])
    assert {"message_sent", "message_consumed", "final_answer"} <= {e["kind"] for e in result["metadata"]["events"]}
    before = digest(store.snapshot())
    following = pipeline.run("evaluate", limit=2)
    assert digest(store.snapshot()) == before
    next_plan = read_run(following[0], "frozen_plan.json")
    poem_plan = read_run(following[1], "frozen_plan.json")
    assert len(next_plan["TeamSpec"]["agents"]) == 3
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


def test_resume_does_not_resubmit_or_commit_twice(setup):
    pipeline, store, provider = setup
    first = pipeline.run("evolve")[0]
    calls, version = len(provider.calls), store.snapshot().version
    again = pipeline.run("evolve")[0]
    assert again["resumed"] and first["run_key"] == again["run_key"]
    assert len(provider.calls) == calls and store.snapshot().version == version


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
    assert len(frozen["TeamSpec"]["agents"]) == 2
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
    with pytest.raises(RuntimeError, match="synthetic transport"):
        pipeline.run("evaluate")
    failures = list(pipeline.output_dir.rglob("failure.json"))
    assert len(failures) == 1
    budget = json.loads(failures[0].with_name("budget.json").read_text())
    assert budget["model_calls"] == 1 and budget["tokens"] > 0
    assert budget["reserved_tokens"] == 0
    assert store.snapshot().version == 0


def test_all_three_banks_reach_their_stage_and_capability(setup):
    pipeline, store, provider = setup
    entries = [Experience(experience_id=bank, bank=bank, instruction=f"Conditional {bank} advice",
                          applicability="comparison", capability="comparison" if bank == "execution" else "",
                          source_task_ids=["prior-task"], evidence=["prior-evidence"],
                          validation_status="accepted") for bank in ("rubric", "organization", "execution")]
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
    assert not locals_["integrator"]["execution_experiences"]


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
