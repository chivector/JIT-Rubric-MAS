"""Submission feedback reaches self-reflection without leaking into held-out execution."""

import json
import copy
import re
from pathlib import Path

import pytest

from jit_mas.budget import MeteredModel
from jit_mas.attribution import _compact_duplicate_event_content
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.pipeline import attribution_source_digest, submitted_source_digest
from jit_mas.schemas import digest
from scripts.run_jit_mas import make_pipeline


class SummaryCitingModel(FixtureModel):
    def _phase(self, payload):
        response = super()._phase(payload)
        if payload["phase"] == "agent_evolve":
            evidence = ["evaluation:summary", *[rubric["evidence_id"]
                        for rubric in payload["evaluation_summary"]["rubrics"]]]
            response["evidence"] = evidence
            for lesson in response["lessons"]:
                lesson["evidence"] = evidence
        return response


class SummaryCitingModels(FixtureModels):
    def create(self, role, agent_id, ledger, stage):
        if role != "local":
            return super().create(role, agent_id, ledger, stage)
        return MeteredModel(SummaryCitingModel(self, role, agent_id), ledger, stage, agent_id, 8192)


@pytest.fixture
def reflection_pipeline(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    models = SummaryCitingModels()
    pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs", fixture_models=models)
    try:
        yield pipeline, store, models
    finally:
        store.close()


def read_artifact(outcome, name):
    return json.loads((Path(outcome["run_dir"]) / name).read_text(encoding="utf-8"))


def local_payloads(models, phase):
    payloads = [json.loads(call["messages"][1]["content"]) for call in models.calls if call["role"] == "local"]
    return [payload for payload in payloads if payload["phase"] == phase]


def payload_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(payload_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(payload_keys(item) for item in value))
    return set()


def resolve_content_ref(payload, reference):
    value = payload
    for key, index in re.findall(r"([A-Za-z_]+)|\[(\d+)\]", reference["path"]):
        value = value[key] if key else value[int(index)]
    assert digest(value) == reference["content_hash"]
    return value


def test_reflection_without_individual_findings_receives_observable_submission_feedback(reflection_pipeline):
    pipeline, _, models = reflection_pipeline
    outcome = pipeline.run("evolve")[0]
    feedback = outcome["evaluation"]
    assert feedback["score"] == 0 and feedback["complete"]
    frozen = read_artifact(outcome, "frozen_plan.json")
    execution = read_artifact(outcome, "execution.json")
    reflections = local_payloads(models, "agent_evolve")
    assert len(reflections) == len(frozen["TeamSpec"]["agents"]) == 3
    for payload in reflections:
        assert payload["attribution_findings"] == []
        assert payload["team"] == frozen["TeamSpec"]
        own_events = [event for event in execution["metadata"]["events"]
                      if event.get("agent_id") == payload["agent"]["agent_id"]]
        assert own_events
        assert len(payload["own_events"]) == len(own_events)
        assert any("content_ref" in event for event in payload["own_events"])
        for supplied, original in zip(payload["own_events"], own_events):
            restored = copy.deepcopy(supplied)
            if "content_ref" in restored:
                restored["content"] = resolve_content_ref(payload, restored.pop("content_ref"))
            assert restored == original
        summary = payload["evaluation_summary"]
        assert summary["evidence_id"] == "evaluation:summary" and summary["scope"] == "submission"
        for field in ("score", "complete", "evaluator_version", "aggregation"):
            assert summary[field] == feedback[field]
        assert "individual" in summary["causal_limit"] and "causal" in summary["causal_limit"]
        assignment = payload["credit_assignment"]
        assert assignment["agent_id"] == payload["agent"]["agent_id"]
        relevant = [rubric for rubric in feedback["rubrics"]
                    if rubric["rubric_id"] in assignment["evaluated_rubric_ids"]]
        assert len(summary["rubrics"]) == len(relevant)
        for safe, rubric in zip(summary["rubrics"], relevant):
            assert set(safe) == {"evidence_id", "rubric_id", "weight", "score", "verdict", "status"}
            assert safe["evidence_id"] == "feedback:" + rubric["rubric_id"]
            for field in ("rubric_id", "weight", "score", "verdict", "status"):
                assert safe[field] == rubric[field]
        evidence = {event["event_id"]: event for event in own_events}
        evidence["planning:team"] = payload["team"]
        evidence[summary["evidence_id"]] = summary
        evidence.update({rubric["evidence_id"]: rubric for rubric in summary["rubrics"]})
        assert set(payload["valid_evidence_ids"]) == set(evidence)
        assert not {"criterion", "reason", "raw", "reference"}.intersection(payload_keys(summary))
        assert "PRIVATE_CANARY" not in json.dumps(payload)
        assert "PRIVATE_REFERENCE_CANARY" not in json.dumps(payload)


def test_summary_evidence_survives_joint_commit_and_frozen_attribution_replay(
        reflection_pipeline, tmp_path, monkeypatch):
    pipeline, store, _ = reflection_pipeline
    task_id = pipeline.manifest.evolution[0]
    source = pipeline.run_task(task_id, store.snapshot(), mode="evolve", attribution=False, resume=False)
    source_hash = submitted_source_digest(source["run_dir"])
    attributed_store = ExperienceStore(tmp_path / "attributed.sqlite")
    replay_store = ExperienceStore(tmp_path / "replay.sqlite")
    try:
        models = SummaryCitingModels()
        continued = make_pipeline(pipeline.config, attributed_store, tmp_path / "attributed-runs",
                                  fixture_models=models)
        attributed = continued.run("evolve", resume=False, resume_source=source["run_dir"],
                                  resume_source_hash=source_hash)[0]
        assert attributed["experience_updates"] and len(attributed["agent_pool_updates"]) == 3
        snapshot = attributed_store.snapshot()
        assert snapshot.version == snapshot.agent_pool.version == 1 and snapshot.experiences
        for update in attributed["agent_updates"]:
            assert "evaluation:summary" in update["evidence"]
            assignments = read_artifact(attributed, "attribution.json")["credit_assignments"]
            team = read_artifact(attributed, "frozen_plan.json")["TeamSpec"]
            agent = next(agent for agent in team["agents"] if agent["pool_agent_id"] == update["pool_agent_id"])
            assert {identifier for identifier in update["evidence"] if identifier.startswith("feedback:")} == {
                "feedback:" + rubric_id for rubric_id in assignments[agent["agent_id"]]["evaluated_rubric_ids"]}
            profile = next(member for member in snapshot.agent_pool.profiles
                           if member.pool_agent_id == update["pool_agent_id"])
            assert profile.version == 2 and "evaluation:summary" in profile.memory[-1].evidence
        replay_models = FixtureModels()
        replay = make_pipeline(pipeline.config, replay_store, tmp_path / "replay-runs",
                               fixture_models=replay_models)

        def forbidden(*args, **kwargs):
            pytest.fail("Frozen attribution must commit the recorded reflection without model calls")

        monkeypatch.setattr(replay_models, "create", forbidden)
        outcome = replay.run("evolve", resume=False,
                             resume_source=source["run_dir"], resume_source_hash=source_hash,
                             resume_attribution=attributed["run_dir"],
                             resume_attribution_hash=attribution_source_digest(attributed["run_dir"]))[0]
        assert outcome["source_attribution_reused"]
        assert outcome["budget"]["model_calls"] == outcome["budget"]["tokens"] == 0
        assert outcome["agent_updates"] == attributed["agent_updates"]
        assert outcome["proposals"] == attributed["proposals"]
        assert digest(replay_store.snapshot()) == digest(snapshot)
        assert submitted_source_digest(source["run_dir"]) == source_hash
        assert read_artifact(outcome, "agent_evolution.json") == read_artifact(attributed, "agent_evolution.json")
    finally:
        attributed_store.close()
        replay_store.close()


def test_held_out_local_planning_and_execution_receive_lessons_without_feedback_records(reflection_pipeline):
    pipeline, store, models = reflection_pipeline
    pipeline.run("evolve")
    snapshot_hash = digest(store.snapshot())
    previous_calls = len(models.calls)
    outcome = pipeline.run("evaluate", limit=1)[0]
    assert digest(store.snapshot()) == snapshot_hash and outcome["agent_updates"] == []
    held_out = models.calls[previous_calls:]
    planning = [json.loads(call["messages"][1]["content"]) for call in held_out if call["role"] == "local"]
    execution = [json.loads(call["messages"][1]["content"]) for call in held_out if call["role"] == "exec"]
    assert planning and execution and all(payload["phase"] == "local_plan" for payload in planning)
    assert any(payload["persistent_agent"]["memory"] for payload in execution)
    forbidden = {"evaluation_summary", "own_events", "local_execution", "attribution_findings",
                 "feedback", "private_records", "criterion", "reason", "raw", "reference"}
    for payload in planning + execution:
        assert not forbidden.intersection(payload_keys(payload))
        assert "PRIVATE_CANARY" not in json.dumps(payload)
        assert "PRIVATE_REFERENCE_CANARY" not in json.dumps(payload)


def test_reflection_evidence_is_normalized_from_lessons():
    from jit_mas.pipeline import MASPipeline
    from jit_mas.planning import JsonModelCalls
    from jit_mas.schemas import AgentEvolutionUpdate
    from types import SimpleNamespace

    requests = []
    response = {
        "update_id": "update", "pool_agent_id": "critic", "base_agent_version": 1, "source_task_id": "task",
        "evidence": ["event:support"],
        "lessons": [{"lesson_id": "lesson", "instruction": "Preserve conflicting observations.",
                     "applicability": "Comparisons", "source_task_ids": ["task"],
                     "evidence": ["event:support"], "counterevidence": ["event:counter"]}]}

    def model(messages):
        payload = json.loads(messages[1]["content"])
        requests.append(payload)
        return json.dumps(response)

    profile = SimpleNamespace(pool_agent_id="critic", version=1)
    result = JsonModelCalls(max_corrections=1).ask(
        model, "agent_evolve", "Distill an observed lesson.", {}, AgentEvolutionUpdate,
        validate=lambda update: MASPipeline._validate_agent_update(
            update, profile, "task", {"event:support", "event:counter"}, "update"))
    assert len(requests) == 1
    assert result.evidence == ["event:counter", "event:support"]
