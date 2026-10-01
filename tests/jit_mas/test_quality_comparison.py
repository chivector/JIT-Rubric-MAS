"""Offline orchestration tests. No provider or generated code is executed."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import compare_jit_mas_quality as comparison
from scripts import benchmark_jit_mas_live as live


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def registration(monkeypatch):
    tasks = [{"task_id": "synthetic-a", "question": "Explain a fictional sorting box.", "attachments": []},
             {"task_id": "synthetic-b", "question": "Compare two fictional sorting boxes.", "attachments": []}]
    hashes = {row["task_id"]: comparison.digest(row) for row in tasks}
    monkeypatch.setattr(comparison, "TASK_HASHES", hashes)
    return {"schema_version": comparison.VERSION, "baseline_commit": "a1ee6cd",
            "dataset_sha256": comparison.DATA_SHA256, "prior_observed_official_task_ids": ["synthetic-old"],
            "protocol": {"max_calls_combined": 240},
            "tasks": [{**row, "public_task_hash": hashes[row["task_id"]]} for row in tasks]}


@pytest.fixture
def setup(tmp_path, monkeypatch, registration):
    path = tmp_path / "pre_registration.json"
    comparison.write_json(path, registration)
    binding = {"fingerprint": "approved", "native_code_fingerprint": "runtime-code", "commit": "commit"}
    checks = {"runtime": binding, "registration_sha256": comparison.file_hash(path),
              "official_prompt_hashes": {"system": "fixed"}, "rubric_counts": {key: 1 for key in comparison.TASK_HASHES}}
    tasks = {row["task_id"]: live.PublicTask(task_id=row["task_id"], question=row["question"]) for row in registration["tasks"]}
    private = {key: {"rubrics": [{"criterion": "test", "weight": 1}]} for key in tasks}
    manifest = live.SplitManifest(test=list(tasks))
    args = SimpleNamespace(arm="baseline", runtime_root=str(ROOT), registration=str(path), output_dir=str(tmp_path),
        data="unused", expected_runtime_fingerprint="approved", max_calls=120, max_tokens=1_000_000,
        request_timeout=120, non_thinking=True, unsafe_local=True)
    monkeypatch.setattr(comparison, "bind_runtime", lambda *args: deepcopy(binding))
    calls, clients = [], []

    class Models:
        def __init__(self, credentials, session, timeout, non_thinking):
            self.session = session
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    class Pipeline:
        def __init__(self, config, models, evaluator, synthesizer, tasks, private, manifest, store, output):
            self.config, self.models, self.evaluator = config, models, evaluator
            self.store, self.output = store, Path(output)
            assert store.read_only
            assert not store.snapshot().experiences
            assert manifest.test == list(comparison.TASK_HASHES)
            assert not manifest.evolution and not manifest.validation and not manifest.stream
            assert all(not task.tools for task in tasks.values())

        def run(self, mode, task_ids, limit, resume):
            calls.append((mode, task_ids, limit, resume))
            assert mode == "evaluate" and limit == 1 and resume is False
            task_id = task_ids[0]
            ticket = self.models.session.reserve("evaluation", "judge", 1, 10)
            self.models.session.settle(ticket, 1, 2)
            if runtime.fail_task == task_id:
                raise RuntimeError("fixture failure secret-fixture-value")
            directory = self.output / task_id
            directory.mkdir(parents=True)
            answer = "unchanged submitted fixture"
            answer_hash = live.digest(answer)
            evaluation = {"complete": True, "score": 0.5, "rubrics": [],
                "evaluator_version": self.evaluator(None).evaluator_version,
                "raw": {"submission_answer_hash": answer_hash}}
            outcome = {"task_id": task_id, "run_dir": str(directory), "evaluation": evaluation}
            documents = {"complete": outcome, "evaluation": evaluation,
                "submission": {"answer": answer, "answer_hash": answer_hash}, "execution": {"answer": answer},
                "run_manifest": {"comparison": {"code": "runtime-code", "config": self.config.model_dump(mode="json")}}}
            for name, document in documents.items():
                comparison.write_json(directory / (name + ".json"), document)
            outcome["experience_updates"] = []
            outcome["next_experience_version"] = 0
            return [outcome]

    runtime = SimpleNamespace(**{name: getattr(live, name) for name in (
        "MASConfig", "ModelConfig", "ResearchRubricsAdapter", "SessionLedger", "ExperienceStore",
        "FileReviewGate", "ReviewedSynthesizer", "redact", "utc_now", "digest", "_usage_accounting")},
        LiveModels=Models, MASPipeline=Pipeline, fail_task=None)
    credentials = live.ProbeCredentials("https://example.invalid/v1", "fixture-model", "secret-fixture-value")
    prepared = registration, tasks, private, manifest, checks
    return args, runtime, credentials, prepared, calls, clients


def test_frozen_public_registration_and_budget(tmp_path, registration):
    path = tmp_path / "pre_registration.json"
    comparison.write_json(path, registration)
    registration = comparison.load_registration(path)
    assert len(registration["tasks"]) == 2
    assert registration["baseline_commit"] == "a1ee6cd"
    assert registration["protocol"]["max_calls_combined"] == 240
    assert set(comparison.TASK_HASHES).isdisjoint(registration["prior_observed_official_task_ids"])


def test_registration_rejects_public_or_id_change(tmp_path, registration):
    registration["tasks"][0]["question"] += "changed"
    path = tmp_path / "registration.json"
    comparison.write_json(path, registration)
    with pytest.raises(ValueError, match="public task"):
        comparison.load_registration(path)


@pytest.mark.parametrize("field,value", [("max_calls", 121), ("max_tokens", 1000001), ("max_calls", 0), ("request_timeout", 301)])
def test_caps_rejected_before_loading_or_models(setup, field, value):
    args, runtime, _, _, _, clients = setup
    setattr(args, field, value)
    with pytest.raises(ValueError):
        comparison.preflight(args, runtime)
    assert clients == []


def test_preflight_insufficient_counts_cannot_change_tasks(setup, monkeypatch):
    args, runtime, _, prepared, _, clients = setup
    _, tasks, private, manifest, _ = prepared
    args.max_calls = 75
    monkeypatch.setattr(comparison, "load_tasks", lambda *args: (tasks, private, manifest, {key: 33 for key in tasks}))
    with pytest.raises(ValueError, match="floor is 76; tasks cannot be replaced"):
        comparison.preflight(args, runtime)
    assert clients == []


def test_preflight_reports_dynamic_counts_and_local_planning_floor(setup, monkeypatch, tmp_path):
    args, runtime, _, prepared, _, clients = setup
    _, tasks, private, manifest, _ = prepared
    args.runtime_root = str(tmp_path)
    args.max_calls = 100
    args.max_tokens = 900_000
    prompt_dir = tmp_path / "benchmark/adapter/researchrubrics_prompts"
    prompt_dir.mkdir(parents=True)
    for name in ("system_prompt.txt", "user_prompt.txt"):
        (prompt_dir / name).write_text("synthetic fixed prompt", encoding="utf-8")
    monkeypatch.setattr(comparison.subprocess, "check_output", lambda *args, **kwargs: "synthetic fixed prompt")
    monkeypatch.setattr(comparison, "load_tasks", lambda *args: (tasks, private, manifest, {key: 7 for key in tasks}))
    checks = comparison.preflight(args, runtime)[-1]
    assert checks["minimum_judge_calls"] == 14
    assert checks["remaining_nonjudge_calls"] == 86
    assert checks["theoretical_minimum_calls"] == 24
    assert "14 judge calls" in checks["budget_note"] and "86 nonjudge calls" in checks["budget_note"]
    assert "900000-token" in checks["budget_note"] and "one local planning call per task" in checks["budget_note"]
    assert clients == []


@pytest.mark.parametrize("field,value", [("unsafe_local", False), ("expected_runtime_fingerprint", None)])
def test_live_requires_explicit_review_permissions(setup, field, value):
    args, runtime, credentials, prepared, _, clients = setup
    setattr(args, field, value)
    with pytest.raises(PermissionError):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert clients == []


def test_evaluate_only_two_tasks_readonly_store_and_exact_budget(setup):
    args, runtime, credentials, prepared, calls, clients = setup
    report = comparison.run_arm(args, runtime, credentials, prepared)
    assert report["status"] == "completed"
    assert report["budget"]["model_calls"] == 2
    assert report["budget"]["tokens"] == 6
    assert report["protocol"]["max_calls_per_arm"] == 120
    assert report["protocol"]["max_tokens_per_arm"] == 1_000_000
    assert report["experience_store_unchanged"] and report["runtime_unchanged"]
    assert calls == [("evaluate", [key], 1, False) for key in comparison.TASK_HASHES]
    assert clients[0].closed
    output = Path(args.output_dir) / args.arm
    assert report["evidence_sha256"] == comparison.evidence_hashes(output)
    assert credentials.api_key not in json.dumps(report)
    with pytest.raises(FileExistsError):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert len(clients) == 1


def test_arm_protocol_mismatch_before_models(setup):
    args, runtime, credentials, prepared, _, clients = setup
    comparison.run_arm(args, runtime, credentials, prepared)
    args.arm = "candidate"
    args.non_thinking = False
    with pytest.raises(ValueError, match="identical"):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert len(clients) == 1
    assert not (Path(args.output_dir) / "candidate").exists()


def test_failed_task_retained_and_secret_redacted(setup):
    args, runtime, credentials, prepared, calls, _ = setup
    runtime.fail_task = next(iter(comparison.TASK_HASHES))
    report = comparison.run_arm(args, runtime, credentials, prepared)
    assert report["status"] == "failed" and len(report["tasks"]) == 2
    assert report["tasks"][0]["status"] == "failed"
    assert report["budget"]["model_calls"] == 2
    assert credentials.api_key not in json.dumps(report)
    assert len(calls) == 2


def run_both(setup):
    args, runtime, credentials, prepared, _, _ = setup
    comparison.run_arm(args, runtime, credentials, prepared)
    args.arm = "candidate"
    comparison.run_arm(args, runtime, credentials, prepared)
    return Path(args.output_dir)


def test_summarize_offline_only_no_new_models(setup):
    study = run_both(setup)
    summary = comparison.summarize(study)
    assert summary["comparison_complete"] and summary["mean_score_delta"] == 0
    assert summary["combined_model_calls"] == 4 and summary["combined_tokens"] == 12
    assert summary["experience_updates_enabled"] is False
    assert len(setup[-1]) == 2


def test_report_only_score_tampering_rejected(setup):
    study = run_both(setup)
    path = study / "candidate/report.json"
    report = comparison.read_json(path)
    report["tasks"][0]["outcome"]["evaluation"]["score"] = 1
    comparison.write_json(path, report)
    with pytest.raises(ValueError, match="not bound"):
        comparison.summarize(study)


def test_pipeline_post_complete_fields_cannot_claim_evolution(setup):
    study = run_both(setup)
    path = study / "candidate/report.json"
    report = comparison.read_json(path)
    report["tasks"][0]["outcome"]["next_experience_version"] = 1
    comparison.write_json(path, report)
    with pytest.raises(ValueError, match="not bound"):
        comparison.summarize(study)


@pytest.mark.parametrize("legacy_field", ["validations", "experience_updates"])
def test_frozen_comparison_requires_empty_update_bookkeeping(setup, legacy_field):
    study = run_both(setup)
    path = study / "baseline/report.json"
    report = comparison.read_json(path)
    outcome = report["tasks"][0]["outcome"]
    outcome[legacy_field] = outcome.pop("experience_updates")
    comparison.write_json(path, report)
    assert comparison.summarize(study)["comparison_complete"]
    outcome[legacy_field] = [{"proposal_id": "unexpected-update"}]
    comparison.write_json(path, report)
    with pytest.raises(ValueError, match="not bound"):
        comparison.summarize(study)


def test_legacy_evaluation_config_only_ignores_inactive_update_gate_settings():
    current = {"backend": "native_jit", "max_model_calls": 120}
    legacy = {**current, "validation": {"repeats": 1}, "max_validation_tasks": 2}
    assert comparison.evaluation_config(legacy) == current
    assert "validation" in legacy
    assert comparison.evaluation_config({**legacy, "max_model_calls": 121}) != current


@pytest.mark.parametrize("field", ["runtime_unchanged", "experience_store_unchanged"])
def test_unfrozen_arm_not_scored(setup, field):
    study = run_both(setup)
    path = study / "candidate/report.json"
    report = comparison.read_json(path)
    report[field] = False
    comparison.write_json(path, report)
    summary = comparison.summarize(study)
    assert not summary["comparison_complete"] and summary["mean_score_delta"] is None
    assert all(row["candidate_minus_baseline"] is None for row in summary["tasks"])


def test_artifact_tampering_rejected(setup):
    study = run_both(setup)
    submission = next((study / "candidate/evaluate").rglob("submission.json"))
    comparison.write_json(submission, {"answer": "replacement"})
    with pytest.raises(ValueError, match="evidence"):
        comparison.summarize(study)


def test_failure_excluded_from_macro(setup):
    args, runtime, credentials, prepared, _, _ = setup
    comparison.run_arm(args, runtime, credentials, prepared)
    args.arm = "candidate"
    runtime.fail_task = next(iter(comparison.TASK_HASHES))
    comparison.run_arm(args, runtime, credentials, prepared)
    summary = comparison.summarize(args.output_dir)
    assert summary["mean_score_delta"] is None and not summary["comparison_complete"]


def test_isolated_process_imports_selected_checkout_not_current(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    (scripts / "benchmark_jit_mas_live.py").write_text(
        "ROLE_TOKENS = " + repr(comparison.ROLE_TOKENS) + "\nMARKER = 'target-runtime'\n", encoding="utf-8")
    code = ("import importlib.util; s=importlib.util.spec_from_file_location('quality_runner', "
            + repr(str(Path(comparison.__file__).resolve())) + "); m=importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); r=m.import_runtime(" + repr(str(tmp_path)) + "); print(r.MARKER)")
    result = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "target-runtime"


def test_isolated_missing_target_does_not_fallback_to_current(tmp_path):
    code = ("import importlib.util; s=importlib.util.spec_from_file_location('quality_runner', "
            + repr(str(Path(comparison.__file__).resolve())) + "); m=importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); m.import_runtime(" + repr(str(tmp_path)) + ")")
    result = subprocess.run([sys.executable, "-I", "-c", code], text=True, capture_output=True)
    assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr


def test_baseline_requires_exact_clean_commit(monkeypatch, tmp_path):
    def fake_git(root, *args):
        return {("rev-parse", "HEAD"): "wrong", ("rev-parse", "a1ee6cd^{commit}"): "baseline",
                ("status", "--porcelain", "--untracked-files=all"): ""}[args]
    monkeypatch.setattr(comparison, "git", fake_git)
    with pytest.raises(ValueError, match="clean a1ee6cd"):
        comparison.bind_runtime(tmp_path, "baseline", SimpleNamespace())


def test_cli_help_does_not_read_credentials_or_call_models():
    result = subprocess.run([sys.executable, str(Path(comparison.__file__)), "--help"],
                            text=True, capture_output=True, input="")
    assert result.returncode == 0 and "--runtime-root" in result.stdout and "--summarize" in result.stdout


def development_setup(setup):
    args, _, _, prepared, _, _ = setup
    registration, _, _, _, checks = prepared
    registration.update(study_kind="development_regression", parent_study={
        "study_id": "sealed_original_comparison", "summary_sha256": "a" * 64})
    comparison.write_json(args.registration, registration)
    checks["registration_sha256"] = comparison.file_hash(args.registration)
    args.development_regression = True
    args.arm = "candidate"
    return setup


def test_development_regression_requires_explicit_matching_mode(setup):
    args, _, _, _, _, _ = development_setup(setup)
    with pytest.raises(ValueError, match="cannot be used"):
        comparison.load_registration(args.registration)
    registration = comparison.load_registration(args.registration,
        development_regression=True, arm="candidate")
    assert registration["parent_study"]["summary_sha256"] == "a" * 64
    registration["study_kind"] = "implementation_comparison"
    comparison.write_json(args.registration, registration)
    with pytest.raises(ValueError, match="matching registration"):
        comparison.load_registration(args.registration, development_regression=True, arm="candidate")


@pytest.mark.parametrize("parent", [
    None, "arbitrary/path", {}, {"study_id": "old"},
    {"study_id": "../../private", "summary_sha256": "a" * 64},
    {"study_id": "old", "summary_sha256": "not-a-hash"},
    {"study_id": "old", "summary_sha256": "a" * 64, "path": "private.json"},
])
def test_development_parent_anchor_structure_is_required_without_reading_parent_paths(setup, parent):
    args, _, _, prepared, _, clients = development_setup(setup)
    registration = deepcopy(prepared[0])
    registration["parent_study"] = parent
    comparison.write_json(args.registration, registration)
    with pytest.raises(ValueError, match="parent_study"):
        comparison.load_registration(args.registration, development_regression=True, arm="candidate")
    assert clients == []


def test_development_baseline_arm_is_rejected_before_loading_tasks_or_models(setup, monkeypatch):
    args, runtime, credentials, prepared, calls, clients = development_setup(setup)
    args.arm = "baseline"
    monkeypatch.setattr(comparison, "load_tasks", lambda *args: pytest.fail("must reject before private data loading"))
    with pytest.raises(ValueError, match="candidate arm only"):
        comparison.preflight(args, runtime)
    with pytest.raises(ValueError, match="candidate arm only"):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert clients == [] and calls == []
    assert not (Path(args.output_dir) / "baseline").exists()


def test_development_reports_exposure_scope_and_keeps_native_limits_and_fixed_tasks(setup):
    args, runtime, credentials, prepared, calls, clients = development_setup(setup)
    report = comparison.run_arm(args, runtime, credentials, prepared)
    study = Path(args.output_dir)
    assert report["status"] == "completed" and report["arm"] == "candidate"
    for document in (report, report["protocol"]):
        assert document["study_kind"] == "development_regression"
        assert document["parent_study"] == prepared[0]["parent_study"]
        assert "already exposed tasks" in document["scope"]
        assert "not blind testing" in document["scope"]
        assert "not an A/B comparison" in document["scope"]
        assert "not a SOTA claim" in document["scope"]
        assert "structure only" in document["parent_anchor_verification"]
    assert (study / "development_protocol.json").is_file()
    assert not (study / "comparison_protocol.json").exists()
    assert not (study / "baseline").exists()
    assert report["protocol"]["config"]["backend"] == "native_jit"
    assert report["protocol"]["config"]["unsafe_local"]
    assert report["protocol"]["max_calls_per_arm"] == 120
    assert report["protocol"]["max_tokens_per_arm"] == 1_000_000
    assert calls == [("evaluate", [key], 1, False) for key in comparison.TASK_HASHES]
    assert clients[0].closed and report["experience_updates_enabled"] is False
    with pytest.raises(ValueError, match="cannot be summarized"):
        comparison.summarize(study)
    with pytest.raises(FileExistsError):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert len(clients) == 1


def test_comparison_summarize_rejects_development_protocol_even_if_renamed(setup):
    args, _, _, prepared, _, _ = setup
    comparison.write_json(Path(args.output_dir) / "comparison_protocol.json", {
        "study_kind": "development_regression", "registration_sha256": prepared[-1]["registration_sha256"]})
    with pytest.raises(ValueError, match="Development protocol"):
        comparison.summarize(args.output_dir)


def test_development_study_cannot_reuse_original_comparison_arm_directory(setup):
    args, runtime, credentials, prepared, _, clients = setup
    comparison.run_arm(args, runtime, credentials, prepared)
    baseline = Path(args.output_dir) / "baseline/report.json"
    original = baseline.read_bytes()
    development_setup(setup)
    with pytest.raises(ValueError, match="cannot share"):
        comparison.run_arm(args, runtime, credentials, prepared)
    assert baseline.read_bytes() == original
    assert not (Path(args.output_dir) / "candidate").exists()
    assert len(clients) == 1


def test_development_flag_is_forwarded_to_isolated_worker_without_credentials_in_args(monkeypatch, tmp_path):
    seen = []

    def launch(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps({"status": "preflight_only", "preflight": {}}), returncode=0)

    monkeypatch.setattr(comparison.subprocess, "run", launch)
    status = comparison.main(["--arm", "candidate", "--runtime-root", str(tmp_path),
        "--registration", str(tmp_path / "pre_registration.json"), "--data", str(tmp_path / "data.jsonl"),
        "--output-dir", str(tmp_path), "--development-regression", "--preflight-only"])
    assert status == 0 and len(seen) == 1
    command, kwargs = seen[0]
    assert "--worker" in command and "--development-regression" in command and "-I" in command
    assert kwargs["input"] == "" and "api_key" not in " ".join(command)


def test_development_flag_cannot_enable_ab_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "summarize", lambda *args: pytest.fail("must not enter A/B summarizer"))
    assert comparison.main(["--development-regression", "--summarize", str(tmp_path)]) == 1
