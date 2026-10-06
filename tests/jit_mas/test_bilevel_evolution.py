"""End-to-end meta decisions determine one atomic bi-level evolution."""

import json
from pathlib import Path

import pytest

from jit_mas.budget import MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.offline import FixtureModel, FixtureModels
from jit_mas.pipeline import EvolutionDecision, MASPipeline, attribution_source_digest, submitted_source_digest
from jit_mas.schemas import (AgentEvolutionUpdate, AgentPoolOperation, AttributionFinding,
                              EvaluationFeedback, RubricFeedback, digest)
from jit_mas.agent_pool import seed_pool
from scripts.run_jit_mas import make_pipeline


class DecisionFixtureModel(FixtureModel):
    def _phase(self, payload):
        if payload["phase"] == "propose" and self.provider.repeated_proposal_id:
            payload = {**payload, "experiences": []}
        response = super()._phase(payload)
        phase = payload["phase"]
        if phase == "attribute_global" and self.provider.meta_assignment:
            response.update(rubric_assignments={"writer": [payload["feedback"]["rubrics"][0]["rubric_id"]]},
                            rubric_assignment_rationale={"writer": "Inspect the final synthesis for omitted boundary assumptions."})
        if phase == "agent_evolve" and self.provider.empty_reflections:
            response["lessons"] = []
        if phase == "propose" and response["proposals"]:
            experience = response["proposals"][0]["experience"]
            if self.provider.repeated_proposal_id:
                response["proposals"][0]["proposal_id"] = "proposal-1"
                experience["experience_id"] = payload["task"]["task_id"] + ":boundary-assumptions"
            experience["bank"] = self.provider.meta_bank
            if self.provider.meta_bank == "organization":
                experience["instruction"] = (
                    "For system comparisons, assign workload and recovery checks to a role before synthesis.")
            if self.provider.meta_bank == "execution":
                experience["capability"] = "comparison"
        if phase == "evolution_integrate":
            selection = self.provider.selection
            if selection == "fail":
                raise RuntimeError("Synthetic final meta decision failure")
            if selection == "writer":
                response["agent_update_ids"] = [update["update_id"]
                    for update in payload["agent_reflections"] if update["pool_agent_id"] == "writer"]
            elif selection == "reject_all":
                response.update(proposal_id=None, agent_update_ids=[])
            elif selection == "unknown_update":
                response["agent_update_ids"] = ["unknown-agent-update"]
            elif selection == "duplicate_update":
                update_id = payload["agent_reflections"][0]["update_id"]
                response["agent_update_ids"] = [update_id, update_id]
            elif selection == "unknown_proposal":
                response["proposal_id"] = "unknown-meta-proposal"
        return response


class DecisionFixtureModels(FixtureModels):
    def __init__(self, *, selection="writer", empty_reflections=False, meta_bank="rubric", meta_assignment=False,
                 repeated_proposal_id=False):
        super().__init__()
        self.selection = selection
        self.empty_reflections = empty_reflections
        self.meta_bank = meta_bank
        self.meta_assignment = meta_assignment
        self.repeated_proposal_id = repeated_proposal_id

    def create(self, role, agent_id, ledger, stage):
        if role not in {"global", "local"}:
            return super().create(role, agent_id, ledger, stage)
        return MeteredModel(DecisionFixtureModel(self, role, agent_id), ledger, stage, agent_id, 8192)


@pytest.fixture
def pipeline_factory(tmp_path):
    stores = []

    def create(**choices):
        store = ExperienceStore(tmp_path / f"state-{len(stores)}.sqlite")
        stores.append(store)
        models = DecisionFixtureModels(**choices)
        pipeline = make_pipeline(MASConfig(backend="scripted"), store,
                                 tmp_path / f"runs-{len(stores)}", fixture_models=models)
        return pipeline, store, models

    yield create
    for store in stores:
        store.close()


def read_artifact(run_dir, name):
    return json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))


def phase_calls(models):
    return [json.loads(call["messages"][-1]["content"]) for call in models.calls
            if call["role"] in {"global", "local"}]


def test_local_agent_findings_drop_evidence_outside_agent_scope():
    finding = AttributionFinding(finding_id="f1", rubric_ids=["r1"], categories=["execution"],
        hypothesis="A local process observation.", supporting_evidence=["own", "other"],
        opposing_evidence=["other"])
    localized = MASPipeline._localize_agent_findings([finding], {"own"})[0]
    assert localized.supporting_evidence == ["own"]
    assert localized.opposing_evidence == []


def test_meta_rejects_direct_update_and_specialize_overlap():
    pool = seed_pool()
    writer = next(profile for profile in pool.profiles if profile.pool_agent_id == "writer")
    operation = AgentPoolOperation(operation_id="specialize-writer", kind="specialize",
        source_task_id="task", base_pool_version=pool.version, target_agent_ids=["writer"],
        base_agent_versions={"writer": writer.version}, profiles=[writer], evidence=["evidence"],
        rationale="Improve the retained writer harness.", expected_benefit="Better local quality.",
        token_cost_tradeoff="No additional execution role.")
    update = AgentEvolutionUpdate(update_id="update-writer", pool_agent_id="writer",
        base_agent_version=writer.version, source_task_id="task", evidence=["evidence"])
    decision = EvolutionDecision(agent_update_ids=[update.update_id], pool_operations=[operation],
        rationale="Choose one atomic harness change.")
    with pytest.raises(ValueError, match="cannot both change"):
        MASPipeline._validate_evolution_decision(decision, [], [update], pool=pool,
            task_id="task", valid_ids={"evidence"})


@pytest.mark.parametrize("meta_bank", ["rubric", "organization"])
def test_final_meta_collects_every_reflection_and_commits_only_selected_harness(pipeline_factory, meta_bank):
    pipeline, store, models = pipeline_factory(meta_bank=meta_bank)
    outcome = pipeline.run("evolve")[0]
    phases = phase_calls(models)
    reflection_positions = [index for index, payload in enumerate(phases) if payload["phase"] == "agent_evolve"]
    propose_position = next(index for index, payload in enumerate(phases) if payload["phase"] == "propose")
    integrate_position = next(index for index, payload in enumerate(phases) if payload["phase"] == "evolution_integrate")
    assert len(reflection_positions) == 3
    assert max(reflection_positions) < propose_position < integrate_position

    reflections = read_artifact(outcome["run_dir"], "agent_evolution.json")
    meta = read_artifact(outcome["run_dir"], "meta_evolution.json")
    assert reflections["complete"] and meta["complete"]
    assert meta["agent_reflections"] == reflections["updates"]
    assert phases[propose_position]["agent_reflections"] == reflections["updates"]
    assert phases[integrate_position]["agent_reflections"] == reflections["updates"]
    selected = next(update for update in reflections["updates"] if update["pool_agent_id"] == "writer")
    assert meta["decision"]["agent_update_ids"] == [selected["update_id"]]
    assert meta["decision"]["proposal_id"] == meta["candidate_proposals"][0]["proposal_id"]
    assert len(meta["calls"]) == 1
    assert outcome["agent_updates"] == [selected]
    assert {update["pool_agent_id"] for update in outcome["agent_pool_updates"]} == {"writer"}
    snapshot = store.snapshot()
    assert snapshot.version == snapshot.agent_pool.version == 1
    assert {experience.bank for experience in snapshot.experiences} == {meta_bank}
    by_id = {profile.pool_agent_id: profile for profile in snapshot.agent_pool.profiles}
    assert by_id["writer"].version == 2 and by_id["writer"].memory
    assert by_id["analyst"].version == by_id["searcher"].version == 1
    assert not by_id["analyst"].memory and not by_id["searcher"].memory


def test_final_meta_can_reject_all_changes_after_collecting_empty_reflections(pipeline_factory):
    pipeline, store, models = pipeline_factory(selection="reject_all", empty_reflections=True)
    baseline = digest(store.snapshot())
    outcome = pipeline.run("evolve")[0]
    meta = read_artifact(outcome["run_dir"], "meta_evolution.json")
    assert meta["complete"] and len(meta["agent_reflections"]) == 3
    assert all(not update["lessons"] for update in meta["agent_reflections"])
    assert meta["candidate_proposals"]
    proposal = next(payload for payload in phase_calls(models) if payload["phase"] == "propose")
    assert proposal["agent_reflections"] == meta["agent_reflections"]
    assert meta["decision"]["proposal_id"] is None
    assert meta["decision"]["agent_update_ids"] == []
    assert outcome["proposals"] == outcome["agent_updates"] == outcome["experience_updates"] == []
    assert outcome.get("agent_pool_updates", []) == []
    updated = store.snapshot()
    assert digest(updated) != baseline
    assert len(updated.agent_pool.observations) == 3
    assert updated.experiences == updated.agent_pool.structural_history == []
    assert all(profile.version == 1 and not profile.memory and not profile.source_task_ids
               for profile in updated.agent_pool.profiles)


@pytest.mark.parametrize("selection", ["unknown_update", "duplicate_update", "unknown_proposal", "fail"])
def test_invalid_or_failed_final_meta_decision_never_partially_commits(pipeline_factory, selection):
    pipeline, store, models = pipeline_factory(selection=selection)
    baseline = digest(store.snapshot())
    expected_error = RuntimeError if selection == "fail" else ValueError
    expected_message = "Synthetic final meta decision failure" if selection == "fail" else "Meta evolution selected"
    with pytest.raises(expected_error, match=expected_message) as caught:
        pipeline.run("evolve")
    assert digest(store.snapshot()) == baseline
    assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0
    run_dir = caught.value.jit_mas_run_failure["run_dir"]
    meta = read_artifact(run_dir, "meta_evolution.json")
    assert not meta["complete"] and meta["decision"] is None
    assert len(meta["agent_reflections"]) == 3 and meta["candidate_proposals"]
    assert read_artifact(run_dir, "agent_evolution.json")["complete"]
    assert not read_artifact(run_dir, "attribution.json")["complete"]
    calls = [payload for payload in phase_calls(models) if payload["phase"] == "evolution_integrate"]
    assert len(calls) == (1 if selection == "fail" else 2)


def test_held_out_execution_has_no_reflection_or_final_meta_decision(pipeline_factory):
    pipeline, store, models = pipeline_factory()
    baseline = digest(store.snapshot())
    outcome = pipeline.run("evaluate", limit=1)[0]
    assert {payload["phase"] for payload in phase_calls(models)} == {"predict", "local_plan", "reconcile"}
    assert digest(store.snapshot()) == baseline
    assert not (Path(outcome["run_dir"]) / "agent_evolution.json").exists()
    assert not (Path(outcome["run_dir"]) / "meta_evolution.json").exists()


def test_full_harness_prototype_mode_rejects_meta_execution_bank(pipeline_factory):
    pipeline, store, models = pipeline_factory(meta_bank="execution")
    baseline = digest(store.snapshot())
    with pytest.raises(ValueError) as caught:
        pipeline.run("evolve")
    assert digest(store.snapshot()) == baseline
    phases = phase_calls(models)
    assert sum(payload["phase"] == "agent_evolve" for payload in phases) == 3
    assert sum(payload["phase"] == "propose" for payload in phases) == 2
    assert not any(payload["phase"] == "evolution_integrate" for payload in phases)
    assert not read_artifact(caught.value.jit_mas_run_failure["run_dir"], "attribution.json")["complete"]


def test_repeated_model_proposal_ids_commit_once_per_task(pipeline_factory, monkeypatch):
    pipeline, store, models = pipeline_factory(repeated_proposal_id=True)
    task_ids = ["evolve-comparison", "stream-first"]
    pipeline.manifest = pipeline.manifest.model_copy(update={
        "evolution": task_ids, "stream": ["stream-next"]})
    outcomes = pipeline.run("evolve", task_ids)
    expected_ids = [task_id + ":proposal-1" for task_id in task_ids]
    assert [outcome["proposals"][0]["proposal_id"] for outcome in outcomes] == expected_ids
    snapshot = store.snapshot()
    assert snapshot.version == 2
    assert snapshot.applied_proposals == expected_ids
    assert store.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 2
    assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 2
    for outcome in outcomes:
        meta = read_artifact(outcome["run_dir"], "meta_evolution.json")
        assert meta["decision"]["proposal_id"] == outcome["proposals"][0]["proposal_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed source tasks must resume without calling models")

    monkeypatch.setattr(models, "create", forbidden)
    resumed = pipeline.run("evolve", task_ids)
    assert all(outcome["resumed"] for outcome in resumed)
    assert digest(store.snapshot()) == digest(snapshot)


def frozen_decision_source(pipeline_factory, selection, *, repeated_proposal_id=False):
    source_pipeline, source_store, _ = pipeline_factory(selection=selection,
                                                       repeated_proposal_id=repeated_proposal_id)
    task_id = source_pipeline.manifest.evolution[0]
    source = source_pipeline.run_task(task_id, source_store.snapshot(), mode="evolve",
                                      attribution=False, resume=False)
    anchor = submitted_source_digest(source["run_dir"])
    continued, continued_store, _ = pipeline_factory(selection=selection, meta_assignment=True,
                                                     repeated_proposal_id=repeated_proposal_id)
    attributed = continued.run("evolve", resume_source=source["run_dir"],
                                resume_source_hash=anchor)[0]
    return source, anchor, attributed, continued_store


@pytest.mark.parametrize("selection", ["writer", "reject_all"])
@pytest.mark.parametrize("repeated_proposal_id", [False, True])
def test_frozen_final_meta_selection_replays_without_model_calls(
        pipeline_factory, monkeypatch, selection, repeated_proposal_id):
    source, source_anchor, attributed, continued_store = frozen_decision_source(
        pipeline_factory, selection, repeated_proposal_id=repeated_proposal_id)
    location = Path(attributed["run_dir"])
    attribution_anchor = attribution_source_digest(location)
    original_meta = (location / "meta_evolution.json").read_bytes()
    original_reflections = (location / "agent_evolution.json").read_bytes()
    meta = json.loads(original_meta)
    if repeated_proposal_id:
        assert meta["candidate_proposals"][0]["proposal_id"] == attributed["task_id"] + ":proposal-1"
    replay, replay_store, replay_models = pipeline_factory(selection="fail")

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen meta evolution replay must not construct or call models")

    monkeypatch.setattr(replay_models, "create", forbidden)
    outcome = replay.run("evolve", resume_source=source["run_dir"], resume_source_hash=source_anchor,
                        resume_attribution=location, resume_attribution_hash=attribution_anchor)[0]
    assert outcome["source_attribution_reused"]
    assert outcome["budget"]["model_calls"] == outcome["budget"]["tokens"] == 0
    assert not replay_models.calls
    assert outcome["proposals"] == attributed["proposals"]
    assert outcome["agent_updates"] == attributed["agent_updates"]
    assert outcome.get("agent_pool_updates", []) == attributed.get("agent_pool_updates", [])
    assert digest(replay_store.snapshot()) == digest(continued_store.snapshot())
    assert (Path(outcome["run_dir"]) / "meta_evolution.json").read_bytes() == original_meta
    assert (Path(outcome["run_dir"]) / "agent_evolution.json").read_bytes() == original_reflections
    attribution = read_artifact(location, "attribution.json")
    evaluated_id = attributed["evaluation"]["rubrics"][0]["rubric_id"]
    writer_assignment = attribution["credit_assignments"]["writer"]
    assert writer_assignment["evaluated_rubric_ids"] == [evaluated_id]
    assert writer_assignment["meta_assigned_evaluated_rubric_ids"] == [evaluated_id]
    assert writer_assignment["assignment_rationale"]
    assert (Path(outcome["run_dir"]) / "attribution.json").read_bytes() == (location / "attribution.json").read_bytes()
    assert len(meta["agent_reflections"]) == 3 and meta["candidate_proposals"]
    assert len(outcome["agent_updates"]) == (1 if selection == "writer" else 0)
    assert len(outcome["proposals"]) == (1 if selection == "writer" else 0)
    assert attribution_source_digest(location) == attribution_anchor
    assert submitted_source_digest(source["run_dir"]) == source_anchor


@pytest.mark.parametrize("mutation", ["missing_meta", "decision_selection"])
def test_frozen_final_meta_rejects_missing_or_inconsistent_decision_before_calls(
        pipeline_factory, monkeypatch, mutation):
    source, source_anchor, attributed, _ = frozen_decision_source(pipeline_factory, "writer")
    location = Path(attributed["run_dir"])
    meta_path = location / "meta_evolution.json"
    if mutation == "missing_meta":
        meta_path.unlink()
    else:
        meta = read_artifact(location, "meta_evolution.json")
        meta["decision"]["agent_update_ids"] = []
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    attribution_anchor = attribution_source_digest(location)
    replay, store, models = pipeline_factory(selection="fail")
    baseline = digest(store.snapshot())

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen meta evolution validation must precede every model call")

    monkeypatch.setattr(models, "create", forbidden)
    message = "lacks its final meta decision" if mutation == "missing_meta" else "differ from the final meta decision"
    with pytest.raises(ValueError, match=message):
        replay.run("evolve", resume_source=source["run_dir"], resume_source_hash=source_anchor,
                   resume_attribution=location, resume_attribution_hash=attribution_anchor)
    assert not models.calls
    assert digest(store.snapshot()) == baseline
    assert store.db.execute("SELECT COUNT(*) FROM evolution_commits").fetchone()[0] == 0


@pytest.mark.parametrize("assigned", [["research_quality"], []])
def test_agent_reflection_receives_only_assigned_feedback_evidence_while_meta_sees_all(assigned):
    feedback = EvaluationFeedback(task_id="routing", evaluator_version="synthetic", score=0.5, complete=True,
        rubrics=[RubricFeedback(rubric_id="research_quality", criterion="PRIVATE_RESEARCH_CRITERION",
                               weight=1, score=1, verdict="Satisfied"),
                 RubricFeedback(rubric_id="writing_quality", criterion="PRIVATE_WRITING_CRITERION",
                               weight=1, score=0, verdict="Not Satisfied")])
    events = [{"agent_id": "searcher", "event_id": "searcher-observation", "kind": "observation",
               "content": {"text": "Observed source text."}},
              {"agent_id": "writer", "event_id": "writer-private", "kind": "observation",
               "content": {"text": "PRIVATE_WRITER_HISTORY"}}]
    local = MASPipeline._agent_reflection_evidence("searcher", events, feedback,
                                                  evaluated_rubric_ids=assigned)
    meta = MASPipeline._agent_reflection_evidence("", [], feedback)
    assert [row["rubric_id"] for row in local["evaluation_summary"]["rubrics"]] == assigned
    assert local["evaluation_summary"]["score"] == meta["evaluation_summary"]["score"] == 0.5
    assert set(local["valid_evidence_ids"]) == {
        "searcher-observation", "planning:team", "evaluation:summary", *["feedback:" + rid for rid in assigned]}
    assert {row["rubric_id"] for row in meta["evaluation_summary"]["rubrics"]} == {
        "research_quality", "writing_quality"}
    assert "feedback:writing_quality" not in local["valid_evidence_ids"]
    assert "PRIVATE_" not in json.dumps(local)
    assert "PRIVATE_" not in json.dumps(meta)
