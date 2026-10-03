"""Offline fixed selection, explicit exposure, sealing and no-resampling checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import ExperienceSnapshot, digest, utc_now
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage
from scripts import run_development_pilot as pilot
from scripts.run_jit_mas import make_pipeline


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv(pilot.KEY_ENV, "SYNTHETIC_SECRET_DO_NOT_PERSIST")
    monkeypatch.setattr(pilot, "code_identity", lambda: {"runtime": "fixture-code", "runner": "fixture-runner"})
    tasks = [{"index": n, "query": f"Write a story about the fictional box {n}.",
              "criteria": [{"criterion": "PRIVATE_CANARY_WRITING_CRITERION"}]} for n in (1, 2)]
    source = tmp_path / "writing.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in tasks), encoding="utf-8")
    # A syntactically invalid unselected row must never be parsed or normalized.
    with source.open("a", encoding="utf-8") as handle:
        handle.write("\nINVALID_UNSELECTED_TEST_ROW\n")
    index = [{"task_id": f"writingbench:{n}", "benchmark": "writingbench", "partition": "evolution",
              "source_row": n, "question_sha256": pilot.normalized_question_hash(tasks[n - 1]["query"])} for n in (1, 2)]
    splits = {"version": "jit-compose-joint-subset-v5", "benchmarks": {"writingbench": {"dataset_sha256": pilot.file_hash(source)}},
              "task_index": index, "memberships": {"writingbench": {"evolution": [row["task_id"] for row in index],
              "validation": ["val-private"], "test": ["test-private"]}},
              "joint_evolution_schedule": [{"run_id": 0, "task_ids": ["writingbench:2", "writingbench:1"]}]}
    splits["manifest_sha256"] = digest(splits)
    split_path = tmp_path / "splits.json"
    pilot.write_json(split_path, splits)
    output = tmp_path / "campaign"
    config = pilot.common_config("https://fixture.invalid/v1", "synthetic-model")
    pilot.register(output, splits_path=split_path, data_paths={"writingbench": source},
                   counts={"writingbench": 2}, config=config, iteration="v0")
    return SimpleNamespace(path=output, source=source, splits=split_path, config=config)


def test_selection_uses_fixed_evo_order_without_parsing_other_rows(campaign):
    registration = pilot.assert_identity(campaign.path)
    assert [row["task_id"] for row in registration["selected"]] == ["writingbench:2", "writingbench:1"]
    assert len(registration["slots"]) == 6
    assert "PRIVATE_CANARY" not in (campaign.path / "registration.json").read_text()
    for identity in registration["sources"]["writingbench"]["public_tasks"].values():
        assert identity["canonical_hash"] != identity["actor_hash"]
        assert pilot.CLOSED_BOOK_CONSTRAINT in identity["task"]["constraints"]
    document = pilot.read_json(campaign.splits)
    document["task_index"][1]["partition"] = "test"
    document["manifest_sha256"] = digest({key: value for key, value in document.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError, match="EVO"):
        pilot.select_sources(document, {"writingbench": 1})


def functional_environment(campaign, monkeypatch, *, fail_method=None):
    events = []

    class ResponseModel:
        def __init__(self, role):
            self.role = role

        def __call__(self, messages, **kwargs):
            if self.role == "judge":
                assert (campaign.path / "release/seal.json").exists()
                assert len(list((campaign.path / "release/submissions").glob("*.json"))) == 6
                events.append("judge")
                text = json.dumps(messages)
                assert '"method"' not in text and '"slot_id"' not in text
                payload = json.loads(messages[-1]["content"])
                return ChatMessage(role="assistant", content=json.dumps({"scores": [
                    {"criterion": item["criterion"], "score": 8, "reason": "Synthetic check"}
                    for item in payload["criteria"]]}))
            assert "PRIVATE_CANARY" not in json.dumps(messages)
            events.append("actor")
            return ChatMessage(role="assistant", content="A complete fictional story.")

        def get_token_counts(self):
            return {"input_token_count": 12, "output_token_count": 9}

    class Models:
        def create(self, role, agent_id, ledger, stage):
            return MeteredModel(ResponseModel(role), ledger, stage, agent_id, 8192)

        def close(self):
            pass

    class Store:
        def snapshot(self):
            return ExperienceSnapshot()

        def close(self):
            pass

    def environment(path, registration, source, output):
        selected = [row for row in registration["selected"] if row["benchmark"] == source]
        dataset, tasks = pilot.source_material(source, registration["sources"][source], selected, "closed_book")
        spec = campaign.config.models["judge"]
        evaluator = lambda judge: dataset.evaluator(judge, judge_id=spec.model, judge_api_base=spec.endpoint,
                                                     judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout)
        result = SimpleNamespace(tasks=tasks, private_records=dataset.private_records, models=Models(),
                                 store=Store(), config=campaign.config, evaluator_factory=evaluator)

        def run_task(task_id, snapshot, **kwargs):
            assert kwargs["defer_evaluation"] and not kwargs["attribution"]
            if fail_method == "ours":
                ledger = BudgetLedger(None, 2_000_000, 0)
                result.models.create("exec", "failed", ledger, "inference")([{"role": "user", "content": "public"}])
                error = RuntimeError("Synthetic failure")
                error.jit_mas_run_failure = {"budget": ledger.snapshot()}
                raise error
            directory = Path(output) / "ours"
            directory.mkdir(parents=True)
            ledger = BudgetLedger(None, 2_000_000, 0)
            answer = result.models.create("exec", "writer", ledger, "inference")([
                {"role": "user", "content": tasks[task_id].question}]).content
            submission = {"answer": answer, "answer_hash": digest(answer), "submitted_at": utc_now()}
            pilot.write_json(directory / "submission.json", submission)
            pilot.write_json(directory / "execution.json", {"answer": answer, "terminated_reason": "final_answer"})
            outcome = {"task_id": task_id, "status": "submitted_unscored", "evaluation": None,
                       "proposals": [], "budget": ledger.snapshot(), "answer_hash": digest(answer), "run_dir": str(directory)}
            pilot.write_json(directory / "complete.json", outcome)
            pilot.write_json(directory / "budget.json", outcome["budget"])
            return outcome

        result.run_task = run_task
        return result

    def native(pipeline, task, ledger, directory):
        events.append("native-helper")
        response = pipeline.models.create("exec", "native", ledger, "inference")([
            {"role": "user", "content": task.model_dump_json()}])
        return RunResult(answer=response.content, terminated_reason="final_answer")

    monkeypatch.setattr(pilot, "_native_jit", native)
    return environment, events


def test_all_generations_seal_before_anonymous_scoring_and_resume_reuses(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    result = pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    assert result["comparison_complete"] and result["sealed"]
    assert all(row["native_score"] == 8 and row["normalized_score"] == pytest.approx(7 / 9)
               for row in result["rows"])
    assert result["arms"]["direct"]["benchmarks"]["writingbench"]["native_mean_all_complete"] == 8
    assert events.count("native-helper") == 2
    assert events.count("actor") == 6 and events.count("judge") == 6
    assert max(index for index, event in enumerate(events) if event == "actor") < events.index("judge")
    assert all(cost["generation_model_calls"] == 2 and cost["evaluation_model_calls"] == 2 for cost in result["costs"].values())
    before = list(events)
    assert pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)["comparison_complete"]
    assert events == before


def test_failed_generation_is_retained_with_cost_and_never_resampled(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch, fail_method="ours")
    result = pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    assert not result["comparison_complete"] and len(result["rows"]) == 6
    assert result["arms"]["ours"]["submitted"] == 0
    assert result["arms"]["ours"]["benchmarks"]["writingbench"]["native_mean_all_complete"] is None
    assert result["costs"]["ours"]["generation_model_calls"] == 2
    assert events.count("judge") == 4
    before = list(events)
    pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    assert events == before


def test_interrupted_generation_is_consumed_without_actor_retry(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    registration = pilot.assert_identity(campaign.path)
    slot = registration["slots"][0]
    directory = campaign.path / "generation" / digest(slot["slot_id"])
    pilot.write_json(directory / "started.json", {"registration_hash": registration["registration_hash"]})
    result = pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    row = next(row for row in result["rows"] if row["slot_id"] == slot["slot_id"])
    assert row["generation_error_type"] == "interrupted_generation_no_resample"
    assert events.count("actor") == 5
    assert result["costs"][slot["method"]]["unknown_slots"] == 1


def test_native_generation_branch_calls_real_upstream_helper(tmp_path):
    store = ExperienceStore(tmp_path / "state.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted", max_model_calls=None,
                                 max_total_tokens=2_000_000, max_tool_calls=0,
                                 execution_timeout=900, task_timeout=900), store, tmp_path / "runs")
        output = tmp_path / "native-slot"
        output.mkdir()
        outcome = pilot.generate(pipeline, {"task_id": "test-deployment", "method": "native_jit"}, output)
        assert outcome["status"] == "submitted_unscored"
        assert outcome["budget"]["model_calls"] >= 2
        harness = pilot.read_json(Path(outcome["run_dir"]) / "harness.json")
        assert harness["kind"] == "upstream_MetaReActAgent"
        assert harness["runtime_budget"]["max_steps_interface"] == 2_000_000
    finally:
        store.close()


def test_changed_source_or_sealed_answer_fails_closed(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    record_path = next((campaign.path / "release/submissions").glob("*.json"))
    record = pilot.read_json(record_path)
    record["submission"]["answer"] = "tampered"
    pilot.write_json(record_path, record)
    with pytest.raises(ValueError, match="Sealed"):
        pilot.summary(campaign.path)


def test_judge_shares_remaining_task_time_and_excludes_queue_idle(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    registration = pilot.assert_identity(campaign.path)
    slot = registration["slots"][0]
    output = campaign.path / "generation" / digest(slot["slot_id"])
    pipeline = environment(campaign.path, registration, slot["benchmark"], output)
    outcome = pilot.generate(pipeline, slot, output)
    release = pilot.TestRelease(campaign.path / "release")
    release.record(slot["slot_id"], outcome)
    record = pilot.read_json(release._path(slot["slot_id"]))
    record["outcome"]["budget"].update(wall_seconds=895, request_queue_idle_seconds=15)
    captured = []

    class Evaluator:
        def evaluate(self, prediction, ground_truth, private_record):
            return {"task_id": ground_truth, "evaluator_version": "synthetic", "complete": True,
                    "score": 0.5, "feedback": []}

    class Models:
        def create(self, role, aid, ledger, stage):
            captured.append(ledger.timeout_seconds)
            return None

    pipeline.models = Models()
    pipeline.evaluator_factory = lambda judge: Evaluator()
    result = pilot.score_saved(pipeline, record)
    assert captured == [20] and result["judge_time_budget_seconds"] == 20
    assert result["generation_active_seconds"] == 880


def test_exhausted_generation_time_keeps_answer_and_does_not_call_judge(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    registration = pilot.assert_identity(campaign.path)
    slot = registration["slots"][0]
    output = campaign.path / "generation" / digest(slot["slot_id"])
    pipeline = environment(campaign.path, registration, slot["benchmark"], output)
    outcome = pilot.generate(pipeline, slot, output)
    record = {"slot": slot, "outcome": outcome, "submission": pilot.read_json(Path(outcome["run_dir"]) / "submission.json")}
    record["outcome"]["budget"].update(wall_seconds=901, request_queue_idle_seconds=0)
    before = list(events)
    # The benchmark adapter retains an incomplete result when the ledger refuses
    # the first judge request. No model transport call or answer regeneration occurs.
    result = pilot.score_saved(pipeline, record)
    assert not result["complete"] and result["judge_time_budget_seconds"] == 0
    assert result["evaluation_budget"]["model_calls"] == 0
    assert events == before
    assert record["submission"]["answer"] == "A complete fictional story."


def test_reflected_credential_is_rejected_before_metered_call_trace(campaign, monkeypatch):
    secret = "SYNTHETIC_SECRET_DO_NOT_PERSIST"

    class ReflectedProvider:
        def __call__(self, messages, **kwargs):
            return ChatMessage(role="assistant", content=secret)

    class NativeProvider:
        def __init__(self, config):
            pass

        def create(self, role, aid, ledger, stage):
            return MeteredModel(ReflectedProvider(), ledger, stage, aid, 8192)

    monkeypatch.setattr(pilot, "NativeModels", NativeProvider)
    models = pilot.SafeModels(campaign.config)
    ledger = BudgetLedger(None, 2_000_000, 0)
    model = models.create("meta", "native-meta", ledger, "inference")
    with pytest.raises(RuntimeError):
        model([{"role": "user", "content": "Public generation request"}])
    assert model.calls == []
    assert ledger.snapshot()["model_calls"] == 1
    assert secret not in json.dumps({"calls": model.calls, "budget": ledger.snapshot()})


def test_saved_submission_recovers_without_an_actor_request(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    registration = pilot.assert_identity(campaign.path)
    slot = registration["slots"][0]
    output = campaign.path / "generation" / digest(slot["slot_id"])
    pipeline = environment(campaign.path, registration, slot["benchmark"], output)
    pilot.generate(pipeline, slot, output)
    pilot.write_json(output / "started.json", {"registration_hash": registration["registration_hash"]})
    before = events.count("actor")
    result = pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    assert result["comparison_complete"]
    assert events.count("actor") - before == 5


def test_interrupted_scoring_keeps_generation_and_unknown_cost_without_rejudging(campaign, monkeypatch):
    environment, events = functional_environment(campaign, monkeypatch)
    submitted = pilot.run(campaign.path, unsafe_local=True, submit_only=True, environment_factory=environment)
    assert submitted["sealed"] and events.count("judge") == 0
    registration = pilot.assert_identity(campaign.path)
    slot_id = registration["scoring_order"][0]
    release = pilot.TestRelease(campaign.path / "release")
    pilot.write_json(release._path(slot_id, "evaluation_started"), {"started_at": utc_now()})
    result = pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    row = next(row for row in result["rows"] if row["slot_id"] == slot_id)
    assert row["generation_status"] == "submitted"
    assert row["evaluation_status"] == "interrupted_evaluation_no_resample"
    assert events.count("actor") == 6 and events.count("judge") == 5
    assert result["costs"][row["method"]]["unknown_slots"] == 1


@pytest.fixture
def exposed_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv(pilot.KEY_ENV, "SYNTHETIC_SECRET_DO_NOT_PERSIST")
    monkeypatch.setattr(pilot, "code_identity", lambda: {"runtime": "fixture-code", "runner": "fixture-runner"})
    events = []
    rows = [{"key": n, "prompt": f"Write a story about a fictional clock {n}.",
             "instruction_id_list": ["PRIVATE_CHECK_ONE", "PRIVATE_CHECK_TWO"],
             "kwargs": [{"private": "PRIVATE_OPTION_CANARY"}, {}]} for n in (1, 2)]
    source = tmp_path / "instruction.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\nINVALID_UNSELECTED_REFERENCE\n", encoding="utf-8")
    checker_root = tmp_path / "checker"
    checker_root.mkdir()
    marker = checker_root / "revision.txt"
    marker.write_text("synthetic-checker-v1", encoding="utf-8")

    class Checker:
        def __init__(self, root, benchmark):
            self.marker = Path(root) / "revision.txt"
            self.identity = {"benchmark": benchmark, "source_hash": pilot.file_hash(self.marker)}
            self.initial_hash = self.identity["source_hash"]

        def assert_frozen(self):
            if pilot.file_hash(self.marker) != self.initial_hash:
                raise ValueError("Synthetic checker changed")

        def check_record(self, prediction, private):
            assert (output / "release/seal.json").exists()
            assert len(list((output / "release/submissions").glob("*.json"))) == 6
            assert private["instruction_id_list"] == ["PRIVATE_CHECK_ONE", "PRIVATE_CHECK_TWO"]
            events.append("checker")
            # Prompt accuracy must remain zero even when one of two checks passes.
            return [True, False]

    monkeypatch.setattr(pilot, "PinnedInstructionChecker", Checker)
    index = [{"task_id": f"ifeval:{n}", "benchmark": "ifeval", "partition": "test",
              "source_row": n, "question_sha256": pilot.normalized_question_hash(rows[n - 1]["prompt"]),
              "parent_partition": "test", "test_slice": "clean_unseen"} for n in (2, 1)]
    splits = {"version": "jit-compose-joint-subset-v5", "benchmarks": {"ifeval": {"dataset_sha256": pilot.file_hash(source)}},
              "task_index": index, "memberships": {"ifeval": {"evolution": [], "validation": [], "test": ["ifeval:1", "ifeval:2"]}},
              "joint_evolution_schedule": [{"run_id": 0, "task_ids": []}]}
    splits["manifest_sha256"] = digest(splits)
    split_path = tmp_path / "splits.json"
    pilot.write_json(split_path, splits)
    output = tmp_path / "exposed"
    config = pilot.common_config("https://fixture.invalid/v1", "synthetic-model")
    pilot.register(output, splits_path=split_path, data_paths={"ifeval": source}, counts={"ifeval": 2},
                   config=config, iteration="test-development", partition="exposed_test",
                   checker_paths={"ifeval": checker_root})
    return SimpleNamespace(path=output, source=source, splits=split_path, config=config,
                           checker_root=checker_root, events=events)


def test_test_exposure_requires_explicit_flag_and_fixed_task_index_prefix(exposed_campaign):
    document = pilot.read_json(exposed_campaign.splits)
    with pytest.raises(ValueError, match="Insufficient source EVO"):
        pilot.select_sources(document, {"ifeval": 1})
    assert [row["task_id"] for row in pilot.select_sources(document, {"ifeval": 2}, "exposed_test")] == ["ifeval:2", "ifeval:1"]
    with pytest.raises(ValueError, match="1..2"):
        pilot.select_sources(document, {"ifeval": 3}, "exposed_test")
    document["memberships"]["ifeval"]["test"].pop()
    document["manifest_sha256"] = digest({key: value for key, value in document.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError, match="TEST membership"):
        pilot.select_sources(document, {"ifeval": 1}, "exposed_test")
    registration = pilot.assert_identity(exposed_campaign.path)
    assert registration["scope"] == "exposed_TEST_development_only"
    assert not registration["formal_protocol_result"]
    assert [row["task_id"] for row in registration["contamination_list"]] == ["ifeval:2", "ifeval:1"]
    assert "PRIVATE_" not in json.dumps(registration)
    assert registration["sources"]["ifeval"]["source_public_projection_sha256"]


def test_author_checker_runs_after_all_arms_seal_without_model_judge(exposed_campaign, monkeypatch):
    environment, events = functional_environment(exposed_campaign, monkeypatch)

    def local_environment(path, registration, source, output):
        pipeline = environment(path, registration, source, output)
        selected = [row for row in registration["selected"] if row["benchmark"] == source]
        dataset, _ = pilot.source_material(source, registration["sources"][source], selected, "closed_book")
        checker = pilot.source_checker(source, registration["sources"][source])
        spec = exposed_campaign.config.models["judge"]
        pipeline.evaluator_factory = lambda judge: dataset.evaluator(
            judge, judge_id=spec.model, judge_api_base=spec.endpoint,
            judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout, checker=checker)
        pipeline.development_local_checker = True
        return pipeline

    result = pilot.run(exposed_campaign.path, unsafe_local=True, environment_factory=local_environment)
    assert result["comparison_complete"] and result["sealed"]
    assert events.count("actor") == 6 and events.count("native-helper") == 2 and events.count("judge") == 0
    assert exposed_campaign.events.count("checker") == 6
    assert all(row["native_score"] == 0 and row["instruction_accuracy"] == 0.5
               and row["primary_metric"] == "prompt_level_strict_accuracy" for row in result["rows"])
    assert all(cost["evaluation_model_calls"] == 0 and cost["evaluation_tokens"] == 0 for cost in result["costs"].values())
    before = list(exposed_campaign.events), list(events)
    pilot.run(exposed_campaign.path, unsafe_local=True, environment_factory=local_environment)
    assert (exposed_campaign.events, events) == before


def test_exposed_checker_change_fails_closed_before_actor_calls(exposed_campaign):
    (exposed_campaign.checker_root / "revision.txt").write_text("changed-checker", encoding="utf-8")
    with pytest.raises(pilot.CheckpointIntegrityError, match="checker identity changed"):
        pilot.assert_identity(exposed_campaign.path)


@pytest.mark.parametrize("endpoint", [
    "https://user:SYNTHETIC_SECRET@fixture.invalid/v1",
    "https://fixture.invalid/v1?api_key=SYNTHETIC_SECRET",
    "https://fixture.invalid/v1#SYNTHETIC_SECRET",
    "http://fixture.invalid/v1",
])
def test_registration_rejects_credential_bearing_or_insecure_config_urls(campaign, endpoint):
    config = campaign.config.model_copy(deep=True)
    config.models["exec"].endpoint = endpoint
    denied = campaign.path.parent / "denied-url"
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        pilot.register(denied, splits_path=campaign.splits, data_paths={"writingbench": campaign.source},
                       counts={"writingbench": 1}, config=config, iteration="denied")
    assert not denied.exists()


def test_registration_rejects_multiple_harness_candidates(campaign):
    config = campaign.config.model_copy(update={"candidates": 2})
    denied = campaign.path.parent / "denied-candidates"
    with pytest.raises(ValueError, match="exactly one harness candidate"):
        pilot.register(denied, splits_path=campaign.splits, data_paths={"writingbench": campaign.source},
                       counts={"writingbench": 1}, config=config, iteration="denied")
    assert not denied.exists()


def test_failed_generation_cannot_be_relabelled_as_observed_zero(campaign, monkeypatch):
    environment, _ = functional_environment(campaign, monkeypatch, fail_method="ours")
    pilot.run(campaign.path, unsafe_local=True, environment_factory=environment)
    registration = pilot.read_json(campaign.path / "registration.json")
    slot = next(x for x in registration["slots"] if x["method"] == "ours")
    release = pilot.TestRelease(campaign.path / "release")
    pilot.write_json(release._path(slot["slot_id"], "evaluations"),
                     {"slot": slot, "complete": True, "official_score": 0})
    with pytest.raises(pilot.CheckpointIntegrityError, match="failed generation cannot"):
        pilot.summary(campaign.path)
