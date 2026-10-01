"""Synthetic offline method-comparison orchestration and evidence-boundary tests."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_jit_mas_live as live
from scripts import compare_jit_mas_baselines as comparison
from scripts import compare_jit_mas_quality as quality_binding
from scripts import mas_baseline_methods as baseline_methods


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registration():
    tasks = []
    for task_id, split in (("screen-a", "screening"), ("screen-b", "screening"), ("held-a", "holdout")):
        task = live.PublicTask(task_id=task_id, question="Explain the fictional sorting box " + task_id)
        public = task.model_dump(mode="json")
        tasks.append({"public_task": public, "public_task_hash": comparison.digest(public), "split": split})
    return {"schema_version": comparison.VERSION, "arms": list(comparison.ARMS),
        "dataset_sha256": "d" * 64, "excluded_task_ids": ["previous-task"], "tasks": tasks,
        "settings": {"repeats": 1, "tools": [], "max_calls_per_arm": 120,
            "max_tokens_per_arm": 1_000_000, "timeout": 120,
            "execution": {"model": "execution-fixture", "endpoint": "https://execution.invalid/v1",
                          "non_thinking": True},
            "judge": {"model": "independent-judge-fixture", "endpoint": "https://judge.invalid/v1",
                      "non_thinking": False}}}


@pytest.fixture
def setup(tmp_path, registration):
    registration_path = tmp_path / "pre_registration.json"
    comparison.write_json(registration_path, registration)
    runtime_record = {"fingerprint": "approved", "native_code_fingerprint": "runtime-code",
                      "commit": "synthetic-commit"}
    binding = SimpleNamespace(bind_runtime=lambda *args: deepcopy(runtime_record),
                              evidence_hashes=quality_binding.evidence_hashes)
    args = SimpleNamespace(arm="direct", phase="screening", runtime_root=str(ROOT),
        registration=str(registration_path), data="unused", unsafe_local=True,
        expected_runtime_fingerprint="approved")
    selected = comparison.phase_tasks(registration, args.phase)
    tasks = {row["public_task"]["task_id"]: live.PublicTask.model_validate(row["public_task"]) for row in selected}
    private = {key: {"rubrics": [{"criterion": "Synthetic criterion", "weight": 1}]} for key in tasks}
    checks = {"runtime": deepcopy(runtime_record), "registration_sha256": comparison.file_hash(registration_path),
        "phase": args.phase, "rubric_counts": {key: 1 for key in tasks},
        "official_prompt_hashes": {"system_prompt.txt": "synthetic-system", "user_prompt.txt": "synthetic-user"}}
    clients, calls, routed = [], [], []
    control = SimpleNamespace(fail_task=None, close_error=False)
    credentials = {name: live.ProbeCredentials(identity["endpoint"], identity["model"], name + "-secret-fixture")
                   for name, identity in ((name, registration["settings"][name]) for name in ("execution", "judge"))}

    class Models:
        def __init__(self, credential, session, timeout, non_thinking):
            self.credentials, self.session = credential, session
            self.closed = False
            clients.append(self)

        def create(self, role, aid, ledger, stage):
            routed.append((self.credentials.model, role, aid, stage))
            return SimpleNamespace(role=role, model=self.credentials.model)

        def close(self):
            self.closed = True
            if control.close_error and self.credentials.model == "execution-fixture":
                raise RuntimeError("cleanup failure execution-secret-fixture judge-secret-fixture")

    def emit(task, models, evaluator, directory, *, config=None, **kwargs):
        task_id = task.task_id
        calls.append((args.arm, "evaluate", task_id))
        ticket = models.execution.session.reserve("evaluation", "judge", 1, 10)
        models.execution.session.settle(ticket, 1, 2)
        models.create("exec", "fixture", None, "inference")
        models.create("judge", "judge", None, "evaluation")
        if control.fail_task == task_id:
            raise RuntimeError("fixture failure execution-secret-fixture judge-secret-fixture")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        answer = "Synthetic submitted artifact for " + task_id
        answer_hash = comparison.digest(answer)
        feedback = {"task_id": task_id, "complete": True, "score": 1.0,
            "rubrics": [{"rubric_id": task_id + ":r0", "criterion": "Synthetic criterion", "weight": 1,
                         "score": 1, "status": "ok", "verdict": "Satisfied"}],
            "evaluator_version": evaluator(None).evaluator_version,
            "raw": {"submission_answer_hash": answer_hash}}
        outcome = {"task_id": task_id, "run_dir": str(directory.resolve()), "evaluation": feedback,
                   "answer_hash": answer_hash, "experience_version": 0, "proposals": []}
        if args.arm == "direct":
            outcome.update(method="direct_single", mode="evaluate", status="completed", experience_updates=[])
        actual_config = config or comparison.make_config(runtime, methods, registration, args.arm)
        documents = {"complete": deepcopy(outcome), "evaluation": feedback,
            "submission": {"answer": answer, "answer_hash": answer_hash},
            "execution": {"answer": answer, "metadata": {"method": args.arm}},
            "run_manifest": {"comparison": {"code": "runtime-code", "config": actual_config.model_dump(mode="json"),
                                               "task": comparison.digest(task.model_dump(mode="json"))}}}
        for name, document in documents.items():
            comparison.write_json(directory / (name + ".json"), document)
        if args.arm != "direct":
            outcome.update(experience_updates=[], next_experience_version=0)
        return outcome

    class Pipeline:
        def __init__(self, config, models, evaluator, synthesizer, tasks, private, manifest, store, output):
            self.config, self.models, self.evaluator = config, models, evaluator
            self.output = Path(output)
            assert store.read_only and not store.snapshot().experiences
            assert manifest.test == list(tasks)
            assert not manifest.evolution and not manifest.validation and not manifest.stream
            assert config.max_tool_calls == 0 and config.backend == "native_jit"

        def run(self, mode, task_ids, limit, resume):
            assert mode == "evaluate" and limit == 1 and resume is False
            return [emit(tasks[task_ids[0]], self.models, self.evaluator, self.output / task_ids[0], config=self.config)]

    runtime = SimpleNamespace(**{name: getattr(live, name) for name in (
        "PublicTask", "MASConfig", "ModelConfig", "ResearchRubricsAdapter", "SessionLedger", "ExperienceStore",
        "FileReviewGate", "ReviewedSynthesizer", "SplitManifest", "ProbeCredentials", "ROLE_TOKENS",
        "redact", "utc_now", "digest", "_usage_accounting")}, LiveModels=Models, MASPipeline=Pipeline)
    methods = SimpleNamespace(fixed_team=baseline_methods.fixed_team,
        JudgeEnvelopeModel=baseline_methods.JudgeEnvelopeModel,
        run_direct=lambda task, private, models, evaluator, outdir, limits, **kwargs:
            emit(task, models, evaluator, outdir, **kwargs))
    prepared = registration, tasks, private, checks
    return SimpleNamespace(args=args, runtime=runtime, methods=methods, binding=binding, prepared=prepared,
        credentials=credentials, clients=clients, calls=calls, routed=routed, control=control,
        runtime_record=runtime_record, study=tmp_path)


def run_arm(setup, arm):
    setup.args.arm = arm
    return comparison.run_arm(setup.args, setup.runtime, setup.methods, setup.binding,
                              setup.prepared, setup.credentials)


def run_all(setup):
    return {arm: run_arm(setup, arm) for arm in comparison.ARMS}


def test_registration_hash_and_phase_tasks_are_public_only(tmp_path, registration):
    path = tmp_path / "pre_registration.json"
    comparison.write_json(path, registration)
    loaded = comparison.load_registration(path)
    assert comparison.phase_tasks(loaded, "development") == comparison.phase_tasks(loaded, "screening")
    screening = {row["public_task"]["task_id"] for row in comparison.phase_tasks(loaded, "screening")}
    holdout = {row["public_task"]["task_id"] for row in comparison.phase_tasks(loaded, "holdout")}
    assert screening and holdout and screening.isdisjoint(holdout)


@pytest.mark.parametrize("mutation", ["question", "private_field", "duplicate", "exposed", "tools"])
def test_invalid_or_changed_registration_is_rejected(tmp_path, registration, mutation):
    row = registration["tasks"][0]
    if mutation == "question":
        row["public_task"]["question"] += " changed"
    elif mutation == "private_field":
        row["public_task"]["private_rubrics"] = ["not allowed"]
        row["public_task_hash"] = comparison.digest(row["public_task"])
    elif mutation == "duplicate":
        registration["tasks"].append(deepcopy(row))
    elif mutation == "exposed":
        registration["excluded_task_ids"].append(row["public_task"]["task_id"])
    else:
        row["public_task"]["tools"] = ["network"]
        row["public_task_hash"] = comparison.digest(row["public_task"])
    path = tmp_path / "pre_registration.json"
    comparison.write_json(path, registration)
    with pytest.raises(ValueError):
        comparison.load_registration(path)


def test_judge_has_independent_model_route_and_config(setup):
    config = comparison.make_config(setup.runtime, setup.methods, setup.prepared[0], "jit_mas")
    assert config.models["judge"].model == "independent-judge-fixture"
    assert config.models["exec"].model == "execution-fixture"
    report = run_arm(setup, "direct")
    assert report["status"] == "completed"
    assert {row[0] for row in setup.routed if row[1] == "judge"} == {"independent-judge-fixture"}
    assert {row[0] for row in setup.routed if row[1] != "judge"} == {"execution-fixture"}
    assert all(client.closed for client in setup.clients)


def test_three_arms_share_protocol_and_keep_empty_experience_store(setup):
    reports = run_all(setup)
    assert len({report["protocol_hash"] for report in reports.values()}) == 1
    assert all(report["status"] == "completed" for report in reports.values())
    assert all(report["experience_store_unchanged"] for report in reports.values())
    assert reports["fixed_team_jit"]["config"]["fixed_team"]
    assert "native JIT" in reports["fixed_team_jit"]["method_note"]
    assert reports["jit_mas"]["config"]["local_planning"]
    assert not reports["fixed_team_jit"]["config"]["local_planning"]
    summary = comparison.summarize(setup.study, "screening", setup.binding)
    assert summary["comparison_complete"]
    assert summary["mean_scores"] == {arm: 1.0 for arm in comparison.ARMS}
    assert len(setup.calls) == 6
    with pytest.raises(FileExistsError):
        run_arm(setup, "direct")


def test_failed_tasks_are_retained_without_favorable_mean_or_secret_leak(setup):
    setup.control.fail_task = "screen-a"
    reports = run_all(setup)
    for report in reports.values():
        assert report["status"] == "failed"
        assert [row["task_id"] for row in report["tasks"]] == ["screen-a", "screen-b"]
        assert report["tasks"][0]["status"] == "failed"
        assert report["tasks"][1]["status"] == "completed"
        assert "execution-secret-fixture" not in json.dumps(report)
        assert "judge-secret-fixture" not in json.dumps(report)
    summary = comparison.summarize(setup.study, "screening", setup.binding)
    assert not summary["comparison_complete"]
    assert summary["mean_scores"] == {arm: None for arm in comparison.ARMS}


def test_development_is_labeled_and_does_not_consume_holdout_selection(setup):
    setup.args.phase = "development"
    setup.prepared[-1]["phase"] = "development"
    reports = run_all(setup)
    for report in reports.values():
        assert report["phase"] == report["protocol"]["phase"] == "development"
        assert "exposed" in report["protocol"]["scope"]
        assert [row["task_id"] for row in report["tasks"]] == ["screen-a", "screen-b"]
    summary = comparison.summarize(setup.study, "development", setup.binding)
    assert "already exposed" in summary["interpretation"]
    assert not (setup.study / "holdout").exists()


def test_credentials_cannot_change_frozen_judge_model(setup):
    setup.credentials["judge"] = live.ProbeCredentials("https://judge.invalid/v1", "other-model", "unlogged")
    with pytest.raises(ValueError, match="frozen protocol"):
        run_arm(setup, "direct")
    assert setup.clients == []


def test_submission_or_evaluation_tampering_is_rejected(setup):
    run_all(setup)
    path = setup.study / "screening/jit_mas/evaluate/screen-a/submission.json"
    comparison.write_json(path, {"answer": "replacement", "answer_hash": comparison.digest("replacement")})
    with pytest.raises(ValueError, match="evidence|bound"):
        comparison.summarize(setup.study, "screening", setup.binding)


@pytest.mark.parametrize("field", ["preflight_runtime", "config", "row_status", "budget", "outcome_task_id", "answer_hash"])
def test_report_metadata_cannot_override_bound_runtime_configuration_or_completion(setup, field):
    run_all(setup)
    path = setup.study / "screening/jit_mas/report.json"
    report = comparison.read_json(path)
    if field == "preflight_runtime":
        report["preflight"]["runtime"]["native_code_fingerprint"] = "other-code"
    elif field == "config":
        report["config"]["max_agents"] = 1
    elif field == "row_status":
        report["tasks"][0]["status"] = "failed"
    elif field == "outcome_task_id":
        report["tasks"][0]["outcome"]["task_id"] = "held-a"
    elif field == "answer_hash":
        report["tasks"][0]["outcome"]["answer_hash"] = "unbound-answer"
    else:
        report["budget"]["tokens"] = report["protocol"]["settings"]["max_tokens_per_arm"] + 1
    comparison.write_json(path, report)
    with pytest.raises(ValueError):
        comparison.summarize(setup.study, "screening", setup.binding)


def test_summary_checks_execution_against_submission_not_only_file_inventory(setup):
    run_all(setup)
    arm_dir = setup.study / "screening/jit_mas"
    path = arm_dir / "evaluate/screen-a/execution.json"
    execution = comparison.read_json(path)
    execution["answer"] = "Different executed answer"
    comparison.write_json(path, execution)
    report_path = arm_dir / "report.json"
    report = comparison.read_json(report_path)
    report["evidence_sha256"] = setup.binding.evidence_hashes(arm_dir)
    comparison.write_json(report_path, report)
    with pytest.raises(ValueError):
        comparison.summarize(setup.study, "screening", setup.binding)


def test_common_protocol_binds_same_frozen_runtime_within_phase(setup):
    run_arm(setup, "direct")
    setup.runtime_record["fingerprint"] = "different-approved-runtime"
    setup.runtime_record["native_code_fingerprint"] = "other-code"
    setup.prepared[-1]["runtime"] = deepcopy(setup.runtime_record)
    setup.args.expected_runtime_fingerprint = "different-approved-runtime"
    with pytest.raises(ValueError):
        run_arm(setup, "fixed_team_jit")


def test_summary_rejects_phase_protocol_that_claims_holdout_for_screening_tasks(setup):
    run_all(setup)
    protocol_path = setup.study / "screening/protocol.json"
    protocol = comparison.read_json(protocol_path)
    protocol["phase"] = "holdout"
    comparison.write_json(protocol_path, protocol)
    for arm in comparison.ARMS:
        path = setup.study / "screening" / arm / "report.json"
        report = comparison.read_json(path)
        report["protocol"] = protocol
        report["protocol_hash"] = comparison.digest(protocol)
        comparison.write_json(path, report)
    with pytest.raises(ValueError):
        comparison.summarize(setup.study, "screening", setup.binding)


def test_cleanup_failure_is_recorded_redacted_and_closes_both_clients(setup):
    setup.control.close_error = True
    report = run_arm(setup, "direct")
    assert report["status"] == "failed"
    assert all(client.closed for client in setup.clients)
    serialized = json.dumps(comparison.read_json(setup.study / "screening/direct/report.json"))
    assert "execution-secret-fixture" not in serialized
    assert "judge-secret-fixture" not in serialized
