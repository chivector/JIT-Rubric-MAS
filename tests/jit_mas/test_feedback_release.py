import json

import pytest

from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import digest
from jit_mas.test_release import TestRelease, remaining_task_budget, score_with_pipeline
from scripts.run_jit_mas import make_pipeline


def test_real_offline_submission_seal_then_evaluation_without_execution_replay(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs")
        keys = pipeline.manifest.test
        inventory = [{"slot_id": key, "task_id": key, "experience_hash": digest(store.snapshot())} for key in keys]
        release = TestRelease(tmp_path / "campaign", inventory)
        for key in keys:
            outcome = pipeline.run_task(key, store.snapshot(), defer_evaluation=True)
            assert outcome["evaluation"] is None and outcome["proposals"] == []
            assert not any(row["stage"] == "evaluation" for row in outcome["budget"]["records"])
            release.record(key, outcome)
        with pytest.raises(ValueError, match="before"):
            release.evaluate(keys[0], lambda record: score_with_pipeline(pipeline, record))
        release.seal()
        before = store.snapshot()
        result = release.evaluate(keys[0], lambda record: score_with_pipeline(pipeline, record))
        assert result["complete"]
        assert result["official_score"] is not None
        assert all(row["stage"] == "evaluation" for row in result["evaluation_budget"]["records"])
        assert digest(before) == digest(store.snapshot())
        assert release.evaluate(keys[0], lambda record: pytest.fail("Must reuse frozen evaluation")) == result
        with pytest.raises(ValueError, match="sealed"):
            release.record_failure(keys[0], error_type="test", budget={})
    finally:
        store.close()


def test_partial_campaign_failed_slot_and_seal_tampering(tmp_path):
    release = TestRelease(tmp_path, [{"slot_id": "a", "task_id": "a"}, {"slot_id": "b", "task_id": "b"}])
    release.record_failure("a", error_type="timeout", budget={"spent_calls": 1})
    with pytest.raises(ValueError, match="Every registered"):
        release.seal()
    release.record_failure("b", error_type="timeout", budget={"spent_calls": 1})
    release.seal()
    assert release.evaluate("a", lambda record: pytest.fail("Failed generation must not be judged"))["official_score"] is None
    path = release._path("a")
    value = json.loads(path.read_text())
    value["error_type"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        release.evaluate("b", lambda record: {})


def test_validation_is_explicit_read_only_mode_and_cannot_be_test(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs")
        key = pipeline.manifest.validation[0]
        with pytest.raises(ValueError, match="split"):
            pipeline.run_task(key, store.snapshot(), mode="evaluate")
        with pytest.raises(ValueError, match="attribution"):
            pipeline.run_task(key, store.snapshot(), mode="validate", attribution=True)
        assert pipeline.run_task(key, store.snapshot(), mode="validate")["evaluation"]["complete"]
    finally:
        store.close()


def test_deferred_scoring_does_not_reset_budget_and_reuse_is_order_independent():
    config = MASConfig(backend="scripted", max_model_calls=10, max_total_tokens=1000, max_tool_calls=0)
    used = {"model_calls": 8, "tokens": 750, "tool_calls": 0, "reserved_tokens": 0}
    normal = remaining_task_budget(config, {"budget": used})
    reused = remaining_task_budget(config, {"budget": {"model_calls": 0, "tokens": 0, "tool_calls": 0},
        "reused_from": "previous", "task_generation_budget": used,
        "logical_deployment_budget": {"model_calls": 9, "tokens": 950, "tool_calls": 0}})
    assert normal == reused == {"max_calls": 2, "max_tokens": 250, "max_tool_calls": 0}
    assert remaining_task_budget(config, {"budget": {**used, "model_calls": 10}})["max_calls"] == 0


def test_deferred_scoring_keeps_uncapped_calls_and_remaining_token_budget():
    config = MASConfig(backend="scripted", max_model_calls=None, max_total_tokens=1000,
                       max_tool_calls=None)
    used = {"model_calls": 8, "tokens": 750, "tool_calls": 3, "reserved_tokens": 0}

    assert remaining_task_budget(config, {"budget": used}) == {
        "max_calls": None, "max_tokens": 250, "max_tool_calls": None}

    used["tokens"] = 1200
    assert remaining_task_budget(config, {"budget": used})["max_tokens"] == 0


def test_sealed_offline_scoring_runs_with_uncapped_model_and_tool_calls(tmp_path):
    config = MASConfig(backend="scripted", max_model_calls=None, max_tool_calls=None)
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(config, store, tmp_path / "runs")
        task_id = pipeline.manifest.test[0]
        outcome = pipeline.run_task(task_id, store.snapshot(), defer_evaluation=True)
        release = TestRelease(tmp_path / "campaign", [{"slot_id": task_id, "task_id": task_id}])
        release.record(task_id, outcome)
        release.seal()

        result = release.evaluate(task_id, lambda record: score_with_pipeline(pipeline, record))

        assert result["complete"]
        assert result["generation_budget"] == outcome["budget"]
        assert result["evaluation_budget"]["model_calls"] > 0
        assert result["generation_budget"]["tokens"] + result["evaluation_budget"]["tokens"] <= config.max_total_tokens
    finally:
        store.close()


def test_partial_researchrubrics_score_remains_diagnostic_not_official():
    from jit_mas.pipeline import convert_feedback

    feedback = convert_feedback({"task_id": "synthetic", "evaluator_version": "fixture",
                                 "score": 0.5, "complete": False, "feedback": []})
    assert feedback.score is None
    assert feedback.raw["score"] == 0.5
