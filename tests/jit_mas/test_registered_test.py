"""Sealed multi-condition test campaigns and delayed direct baseline evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.budget import MeteredModel
from jit_mas.config import MASConfig
from jit_mas.schemas import ExperienceSnapshot, PublicTask, SplitManifest, digest
from jit_mas.test_release import TestRelease
from scripts.mas_baseline_methods import run_direct
from scripts import run_benchmark_test as cli


@pytest.fixture(autouse=True)
def forbid_paid_requests(monkeypatch):
    from scripts.models.openai_server import OpenAIServerModel
    monkeypatch.setattr(OpenAIServerModel, "__call__", lambda *args, **kwargs:
                        pytest.fail("Registered test integration must never issue paid requests"))


def test_two_fixture_artifacts_are_unscored_until_global_seal(tmp_path):
    campaign = tmp_path / "campaign"
    registration = cli.register(campaign, smoke=True)
    assert registration["slots"] == 2
    with pytest.raises(ValueError, match="Cannot score"):
        cli.score(campaign, "smoke-initial")
    result = cli.submit(campaign, "smoke-initial")
    assert result["submitted_or_preexisting"] == 2 and result["failed"] == 0
    before = cli.status(campaign)
    assert before["evaluated"] == before["evaluation_model_calls"] == 0
    release = TestRelease(campaign)
    for slot in release.slots.values():
        record = json.loads(cli._slot_path(campaign, slot["slot_id"]).read_text())
        assert record["outcome"]["evaluation"] is None
        assert record["outcome"]["proposals"] == []
        assert not (Path(record["outcome"]["run_dir"]) / "evaluation.json").exists()
        assert "evaluation" not in record["outcome"]["budget"]["by_stage"]
    release.seal()
    scored = cli.score(campaign, "smoke-initial")
    assert scored["complete"] == 2
    after = cli.status(campaign)
    assert after["generation_model_calls"] == before["generation_model_calls"]
    assert after["generation_tokens"] == before["generation_tokens"]
    assert after["evaluation_model_calls"] == 2
    assert after["attempts_with_unknown_budget"] == 0
    assert cli.score(campaign, "smoke-initial") == scored
    assert cli.status(campaign) == after


def test_entire_inventory_not_just_current_condition_must_finish(tmp_path):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    release = TestRelease(campaign)
    environment = cli.build_environment(campaign, "smoke-initial")
    try:
        slot = next(iter(release.slots.values()))
        outcome = cli.submit_condition(environment, slot, "ours_initial")
        release.record(slot["slot_id"], outcome)
    finally:
        environment.close()
    with pytest.raises(ValueError, match="Every registered"):
        release.seal()
    with pytest.raises(ValueError, match="Cannot score"):
        cli.score(campaign, "smoke-initial")


def test_completed_submissions_are_not_regenerated(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    cli.submit(campaign, "smoke-initial")
    monkeypatch.setattr(cli, "submit_condition", lambda *args: pytest.fail("Duplicate request"))
    result = cli.submit(campaign, "smoke-initial")
    assert result["submitted_or_preexisting"] == 2
    assert result["recovered"] == result["failed"] == 0


def test_interrupted_submission_blocks_replay_and_can_be_closed_without_sampling(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    original = cli.submit_condition
    called = []

    def interrupt(environment, slot, method):
        called.append(slot["slot_id"])
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "submit_condition", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.submit(campaign, "smoke-initial")
    with pytest.raises(RuntimeError, match="Interrupted submission"):
        cli.submit(campaign, "smoke-initial")
    assert len(called) == 1
    monkeypatch.setattr(cli, "submit_condition", original)
    result = cli.submit(campaign, "smoke-initial", resolve_interrupted="Stopped request inspected; no durable answer")
    assert result["failed"] == result["submitted_or_preexisting"] == 1
    TestRelease(campaign).seal()
    scores = cli.score(campaign, "smoke-initial")
    assert scores["complete"] == 1
    assert cli.status(campaign)["attempts_with_unknown_budget"] == 1


def test_crash_after_durable_artifact_recovers_exact_answer_without_resampling(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    original = cli.submit_condition

    def interrupt_after_submission(environment, slot, method):
        original(environment, slot, method)
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "submit_condition", interrupt_after_submission)
    with pytest.raises(KeyboardInterrupt):
        cli.submit(campaign, "smoke-initial")
    monkeypatch.setattr(cli, "submit_condition", original)
    result = cli.submit(campaign, "smoke-initial")
    assert result["recovered"] == result["submitted_or_preexisting"] == 1
    assert cli.status(campaign)["submitted"] == 2


def test_changed_scorer_identity_fails_before_scoring(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    cli.submit(campaign, "smoke-initial")
    TestRelease(campaign).seal()
    monkeypatch.setattr(cli, "code_fingerprint", lambda: "changed-code")
    with pytest.raises(ValueError, match="identity changed"):
        cli.score(campaign, "smoke-initial")
    assert cli.status(campaign)["evaluated"] == 0


def _smoke_with_identical_selected_state(tmp_path):
    template, campaign = tmp_path / "template", tmp_path / "campaign"
    cli.register(template, smoke=True)
    document = cli._registration(template)
    conditions = list(document["conditions"].values())
    descriptor = {**conditions[0]["descriptor"], "condition_id": "smoke-selected", "method": "ours_selected", "run_id": 1}
    _, _, _, _, identity = cli._condition_material(descriptor, document["suite_path"], smoke=True)
    condition_hash = digest(identity)
    document["conditions"][descriptor["condition_id"]] = {
        "descriptor": descriptor, "identity": identity, "condition_hash": condition_hash}
    document["registration_hash"] = digest({key: value for key, value in document.items() if key != "registration_hash"})
    campaign.mkdir()
    (campaign / "conditions.json").write_text(json.dumps(document), encoding="utf-8")
    original = list(TestRelease(template).slots.values())
    aliases = []
    for slot in original:
        alias = {key: value for key, value in slot.items() if key != "slot_id"}
        alias.update(condition_id="smoke-selected", method="ours_selected", run_id=1,
                     condition_hash=condition_hash, scorer_identity=identity["scorer_identity"])
        alias["slot_id"] = digest(alias)
        aliases.append(alias)
    TestRelease(campaign, original + aliases)
    return campaign


def test_exact_initial_selected_state_reuses_artifact_and_judgment_once(tmp_path, monkeypatch):
    campaign = _smoke_with_identical_selected_state(tmp_path)
    cli.submit(campaign, "smoke-initial")
    before = cli.status(campaign)
    monkeypatch.setattr(cli, "submit_condition", lambda *args: pytest.fail("Identical state must not resample"))
    reused = cli.submit(campaign, "smoke-selected")
    assert reused["recovered"] == 2
    assert cli.status(campaign)["generation_model_calls"] == before["generation_model_calls"]
    records = [json.loads(path.read_text()) for path in (campaign / "submissions").glob("*.json")]
    aliases = [record for record in records if record["slot"]["condition_id"] == "smoke-selected"]
    assert all(record["outcome"]["generation_reused"] for record in aliases)
    assert all(record["outcome"]["budget"]["model_calls"] == 0 for record in aliases)
    assert all(record["outcome"]["task_generation_budget"]["model_calls"] > 0 for record in aliases)
    assert all(record["outcome"]["logical_deployment_budget"]["model_calls"] > 0 for record in aliases)
    TestRelease(campaign).seal()
    cli.score(campaign, "smoke-selected")
    cli.score(campaign, "smoke-initial")
    after = cli.status(campaign)
    assert after["submitted"] == after["evaluated"] == 4
    assert after["evaluation_model_calls"] == 2
    evaluations = [json.loads(path.read_text()) for path in (campaign / "evaluations").glob("*.json")]
    assert sum("evaluation_reused_from" in result for result in evaluations) == 2


def test_score_cache_preserves_remaining_task_capacity(tmp_path, monkeypatch):
    config = MASConfig(backend="scripted", max_model_calls=10, max_total_tokens=10000)
    environment = SimpleNamespace(identity={"configuration": config.model_dump(mode="json"), "code_fingerprint": "frozen"},
                                  assert_frozen=lambda: None, pipeline=SimpleNamespace(config=config))
    calls = []

    def scorer(pipeline, record):
        calls.append(record["slot"]["slot_id"])
        return {"slot": record["slot"], "official_score": 1, "complete": True,
                "evaluation_budget": {"model_calls": 1, "tokens": 20},
                "generation_budget": record["outcome"]["budget"]}

    monkeypatch.setattr(cli, "score_with_pipeline", scorer)

    def record(slot_id, calls, *, reused=False):
        budget = {"model_calls": calls, "tokens": calls * 100, "tool_calls": 0}
        result = {"slot": {"slot_id": slot_id}, "submission": {"answer_hash": "same-answer"},
                  "outcome": {"semantic_execution_key": "same-input", "budget": budget}}
        if reused:
            result["outcome"].update(budget=cli._zero_budget(), reused_from={"slot_id": "first"},
                                     task_generation_budget=budget)
        return result

    cli._score_or_reuse(tmp_path, environment, record("first", 1))
    cli._score_or_reuse(tmp_path, environment, record("less-capacity", 2))
    cached = cli._score_or_reuse(tmp_path, environment, record("alias", 1, reused=True))
    assert calls == ["first", "less-capacity"]
    assert cached["evaluation_reused_from"] == "first"
    assert cached["evaluation_budget"]["model_calls"] == 0


def test_shared_rubric_preparation_cost_is_counted_once_and_failures_retained(tmp_path):
    campaign = tmp_path / "campaign"
    cli.register(campaign, smoke=True)
    folder = campaign / "shared_rstar"
    folder.mkdir()
    cache = {"identity": "public-only", "graph": {}, "budget": {"model_calls": 1, "tokens": 40}}
    cache["sha256"] = digest(cache)
    (folder / "one.json").write_text(json.dumps(cache), encoding="utf-8")
    (folder / "duplicate.json").write_text(json.dumps(cache), encoding="utf-8")
    (folder / "failed.failure.json").write_text(json.dumps({"budget": {"model_calls": 1, "tokens": 20}}), encoding="utf-8")
    report = cli.status(campaign)
    assert report["shared_preparation_model_calls"] == 2
    assert report["shared_preparation_tokens"] == 60
    assert report["total_model_calls_recorded"] == 2


def test_full_suite_requires_42_conditions_not_a_favorable_subset():
    suite = json.loads(cli.DEFAULT_SUITE.read_text(encoding="utf-8"))
    conditions = [{"benchmark": benchmark, "method": method, "run_id": run_id}
                  for benchmark, spec in suite["benchmarks"].items() for method in spec["methods"]
                  for run_id in ([0] if method in cli.STATIC_METHODS else [0, 1, 2])]
    assert len(conditions) == 42
    cli._check_full_campaign(conditions, suite)
    with pytest.raises(ValueError, match="every suite"):
        cli._check_full_campaign(conditions[:-1], suite)
    with pytest.raises(ValueError, match="every suite"):
        cli._check_full_campaign(conditions + conditions[:1], suite)


def test_native_registration_derives_all_three_repeats_per_test_task(tmp_path, monkeypatch):
    suite = json.loads(cli.DEFAULT_SUITE.read_text(encoding="utf-8"))
    conditions = [{"condition_id": f"{benchmark}-{method}-{run_id}", "benchmark": benchmark,
                   "method": method, "run_id": run_id, "snapshot_path": "initial"}
                  for benchmark, spec in suite["benchmarks"].items() for method in spec["methods"]
                  for run_id in ([0] if method in cli.STATIC_METHODS else [0, 1, 2])]
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps(conditions), encoding="utf-8")

    def material(condition, suite_path, *, smoke=False):
        snapshot = ExperienceSnapshot()
        identity = {"snapshot": {"snapshot_hash": digest(snapshot)}, "scorer_identity": "synthetic-scorer",
                    "condition": condition, "software_test_only": True}
        manifest = SplitManifest(test=[condition["benchmark"] + ":task-1", condition["benchmark"] + ":task-2"])
        return snapshot, MASConfig(backend="scripted"), None, manifest, identity

    monkeypatch.setattr(cli, "_condition_material", material)
    report = cli.register(tmp_path / "campaign", path)
    assert report["conditions"] == 42
    assert report["slots"] == 42 * 2 * 3
    slots = TestRelease(tmp_path / "campaign").slots.values()
    assert {slot["repeat"] for slot in slots} == {0, 1, 2}


def test_unproven_evolved_snapshot_rejected_and_static_cannot_import_memory(tmp_path):
    with pytest.raises(ValueError, match="sealed source checkpoint"):
        cli.load_snapshot({"method": "ours_selected", "snapshot_path": "initial"})
    snapshot = ExperienceSnapshot(version=2)
    path = tmp_path / "unproven.json"
    path.write_text(snapshot.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="empty initial"):
        cli.load_snapshot({"method": "ours_initial", "snapshot_path": str(path)})


def test_conditions_do_not_accept_embedded_credentials(tmp_path):
    with pytest.raises(ValueError, match="unknown fields"):
        cli._canonical_condition({"condition_id": "bad", "api_key": "DO_NOT_PERSIST"}, tmp_path)


def test_direct_deferred_baseline_is_one_call_and_never_opens_private_record(tmp_path):
    class ResponseModel:
        def __call__(self, messages, **kwargs):
            assert "PRIVATE_CANARY" not in json.dumps(messages)
            return SimpleNamespace(content="Boundary conditions: this is a synthetic answer.")

        def get_token_counts(self):
            return {"input_token_count": 20, "output_token_count": 10}

    class Provider:
        def create(self, role, agent_id, ledger, stage):
            assert role == "exec" and stage == "inference"
            return MeteredModel(ResponseModel(), ledger, stage, agent_id, 256)

    result = run_direct(PublicTask(task_id="direct-smoke", question="Write a short comparison."),
                        {"reference": "PRIVATE_CANARY"}, Provider(),
                        lambda model: pytest.fail("Evaluator constructed before test seal"),
                        tmp_path / "direct", {"max_calls": 2, "max_tokens": 10000, "output_tokens": 256},
                        defer_evaluation=True)
    assert result["status"] == "submitted_unscored"
    assert result["evaluation"] is None
    assert result["proposals"] == []
    assert result["budget"]["model_calls"] == 1
    assert result["budget"]["tokens"] == 30
    assert not (Path(result["run_dir"]) / "evaluation.json").exists()


def test_smoke_cli_produces_json_and_zero_paid_requests(tmp_path, capsys):
    assert cli.main(["--mode", "smoke", "--campaign", str(tmp_path / "smoke")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["software_test_only"] is True
    assert result["paid_requests"] == 0
    assert result["sealed"] is True
    assert result["submitted"] == result["evaluated"] == 2
    assert result["evaluation_model_calls"] == 2
