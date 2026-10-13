"""Run optimized compact writing and research with native benchmark scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jit_mas.benchmarks import load_benchmark
from jit_mas.budget import BudgetLedger
from jit_mas.config import MASConfig, NativeModels
from jit_mas.evidence import load_evidence_tasks
from jit_mas.pipeline import code_fingerprint, convert_feedback, write_json
from jit_mas.schemas import digest, utc_now
from jit_mas.targeted_checklists import coverage_checklist
from jit_mas.targeted_writing import build_messages as writing_messages
from jit_mas.targeted_writing import (
    REPORT_SECTION_COUNT, REPORT_SECTION_MAX_TOKENS, generation_checks, render_report_sections, render_response,
    report_section_check, report_section_messages, report_section_revision_messages, writing_profile,
)
from scripts.env_config import resolve_env_placeholders
from scripts.mas_baseline_methods import JudgeEnvelopeModel
from scripts.run_high_score_native_experiment import _append, _failure, _redact


def _location(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_inputs(config_path, plan_path):
    config_path, plan_path = Path(config_path).resolve(), Path(plan_path).resolve()
    config = MASConfig.model_validate(resolve_env_placeholders(
        yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    materials = []
    for row in plan.get("benchmarks", []):
        name = row["name"]
        if name not in {"writingbench", "deepsearchqa", "deepresearch_bench_ii", "researchrubrics"}:
            raise ValueError("Unsupported compact benchmark: " + name)
        dataset = load_benchmark(name, _location(row["data"], plan_path.parent))
        tasks = dataset.tasks
        if row.get("evidence_dir"):
            directory = _location(row["evidence_dir"], plan_path.parent)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
            attached = load_evidence_tasks({identity: tasks[identity] for identity in manifest["tasks"]}, directory)
            tasks = {**tasks, **attached}
        selected = row["task_ids"]
        if not selected or len(set(selected)) != len(selected) or any(identity not in tasks for identity in selected):
            raise ValueError("Task inventory must be nonempty, unique and present in the dataset")
        materials.append({"benchmark": name, "dataset": dataset, "tasks": tasks, "task_ids": selected})
    if not materials:
        raise ValueError("Plan requires a nonempty benchmark inventory")
    return config, materials


def _messages(task, benchmark, private_record=None):
    if benchmark == "writingbench":
        return writing_messages(task, private_record), writing_profile(task), None
    from jit_mas.targeted_research import act_numeric_filter, build_messages, research_profile

    deterministic = act_numeric_filter(task)
    return build_messages(task, private_record), research_profile(task), deterministic


def _clients_close(models):
    errors = []
    for model in models:
        client = getattr(model, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception as error:
                errors.append(_failure(error))
    return errors


def _ledger_delta(ledger, before):
    after = ledger.snapshot()
    return {"model_calls": after["model_calls"] - before["model_calls"],
            "tokens": after["tokens"] - before["tokens"],
            "tool_calls": after.get("tool_calls", 0) - before.get("tool_calls", 0),
            "cost": (after["cost"] - before["cost"]
                     if isinstance(after.get("cost"), (int, float))
                     and isinstance(before.get("cost"), (int, float)) else None),
            "records": after.get("records", [])[before["record_count"]:]}


def _ledger_position(ledger):
    snapshot = ledger.snapshot()
    return {"model_calls": snapshot["model_calls"], "tokens": snapshot["tokens"],
            "tool_calls": snapshot.get("tool_calls", 0), "cost": snapshot.get("cost"),
            "record_count": len(snapshot.get("records", []))}


def _report_section_attempt(model, messages, ledger, directory, kind, attempts):
    write_json(directory / f"{kind}_input.json", {
        "messages": messages, "response_format": None, "messages_sha256": digest(messages),
        "max_tokens": REPORT_SECTION_MAX_TOKENS})
    attempt = {"attempt_index": len(attempts) + 1, "kind": kind, "started_at": utc_now(),
               "max_tokens": REPORT_SECTION_MAX_TOKENS, "messages_sha256": digest(messages),
               "complete": False, "failure": None}
    attempts.append(attempt)
    before = _ledger_position(ledger)
    began = time.monotonic()
    try:
        response = model(messages, max_tokens=REPORT_SECTION_MAX_TOKENS)
        content = response if isinstance(response, str) else getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty report section generation")
        (directory / f"{kind}_response.txt").write_text(content, encoding="utf-8")
        finish_reason = getattr(model, "last_request_metadata", {}).get("finish_reason")
        checks = report_section_check(content)
        attempt.update(response_file=f"{kind}_response.txt", response_sha256=digest(content),
                       finish_reason=finish_reason, checks=checks,
                       complete=checks["complete"] and finish_reason not in {
                           "length", "content_filter", "tool_calls", "function_call"})
        return content, finish_reason
    except Exception as error:
        attempt["failure"] = _failure(error)
        raise
    finally:
        attempt.update(finished_at=utc_now(), latency_seconds=time.monotonic() - began,
                       budget=_ledger_delta(ledger, before))
        write_json(directory / f"{kind}_record.json", _redact(attempt))


def _generate_report_sections(task, private_record, provider, ledger, directory, models, sections):
    contents = []
    final_reasons = []
    for index in range(REPORT_SECTION_COUNT):
        section_dir = directory / "sections" / f"section_{index + 1:02d}"
        section = {"section_index": index + 1, "started_at": utc_now(), "attempts": [],
                   "complete": False, "failure": None}
        sections.append(section)
        before = _ledger_position(ledger)
        began = time.monotonic()
        try:
            model = provider.create("exec", f"targeted-compact-section-{index + 1}", ledger, "inference")
            models.append(model)
            messages, _ = report_section_messages(task, index, private_record)
            content, reason = _report_section_attempt(model, messages, ledger, section_dir,
                                                     "generation", section["attempts"])
            if not section["attempts"][-1]["complete"]:
                messages, _ = report_section_revision_messages(task, index, content, private_record)
                content, reason = _report_section_attempt(model, messages, ledger, section_dir,
                                                         "revision", section["attempts"])
            checks = report_section_check(content)
            section.update(checks=checks, finish_reason=reason,
                           complete=section["attempts"][-1]["complete"],
                           content_sha256=digest(content), content_file="content.txt")
            (section_dir / "content.txt").write_text(content, encoding="utf-8")
            contents.append(content)
            final_reasons.append(reason)
        except Exception as error:
            section["failure"] = _failure(error)
            raise
        finally:
            section.update(finished_at=utc_now(), latency_seconds=time.monotonic() - began,
                           budget=_ledger_delta(ledger, before))
            write_json(section_dir / "record.json", _redact(section))
            write_json(directory / "section_generation.json", _redact(sections))
    reason = next((reason for reason in final_reasons if reason in {
        "length", "content_filter", "tool_calls", "function_call"}), "stop")
    return render_report_sections(contents), reason


def run(config, materials, output):
    config.check_native()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "targeted-compact-experiment-v1", "started_at": utc_now(),
                "configuration": config.model_dump(mode="json"), "code_fingerprint": code_fingerprint(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "task_inventory": [{"benchmark": item["benchmark"], "task_ids": item["task_ids"],
                                    "dataset_sha256": item["dataset"].dataset_sha256}
                                   for item in materials],
                "generation": "One call or public-evidence calculation; investment reports use five independent sections with at most one length revision per section",
                "evaluation": "Complete native benchmark evaluator with unchanged criteria",
                "optimization_checklist": {"source": "task criteria from the native benchmark dataset",
                    "task_sha256": {item["benchmark"]: {task_id: digest(coverage_checklist(
                        item["dataset"].private_records[task_id])) for task_id in item["task_ids"]}
                        for item in materials}}}
    write_json(output / "manifest.json", _redact(manifest))
    provider = NativeModels(config)
    records = []
    for material in materials:
        benchmark, dataset = material["benchmark"], material["dataset"]
        for task_id in material["task_ids"]:
            task = material["tasks"][task_id]
            directory = output / "tasks" / digest(task_id)
            ledger = BudgetLedger(max_calls=config.max_model_calls, max_tokens=config.max_total_tokens,
                                  max_tool_calls=0, timeout_seconds=config.task_timeout)
            models, durations, traces, report_sections = [], {}, [], []
            answer = evaluation = failure = finish_reason = profile = None
            checks = {"complete": False}
            status = "failed"
            stage = "generation"
            began = time.monotonic()
            stage_started = began
            write_json(directory / "started.json", {"task_id": task_id, "started_at": utc_now()})
            print(json.dumps({"benchmark": benchmark, "task_id": task_id, "status": "started"}), flush=True)
            try:
                if benchmark == "writingbench" and writing_profile(task) == "investment-report":
                    profile = "investment-report"
                    deterministic = None
                    write_json(directory / "generation_input.json", {"mode": "independent-report-sections",
                        "section_count": REPORT_SECTION_COUNT, "max_tokens_per_section": REPORT_SECTION_MAX_TOKENS,
                        "max_revisions_per_section": 1, "task_sha256": digest(task)})
                else:
                    (messages, response_format), profile, deterministic = _messages(
                        task, benchmark, dataset.private_records[task_id])
                    write_json(directory / "generation_input.json", {"messages": messages, "response_format": response_format,
                               "task_sha256": digest(task), "messages_sha256": digest(messages)})
                generation_started = time.monotonic()
                stage_started = generation_started
                if deterministic is not None:
                    answer = deterministic["answer"]
                    write_json(directory / "calculation.json", deterministic)
                    finish_reason = "deterministic"
                elif profile == "investment-report":
                    answer, finish_reason = _generate_report_sections(task, dataset.private_records[task_id],
                        provider, ledger, directory, models, report_sections)
                else:
                    model = provider.create("exec", "targeted-compact", ledger, "inference")
                    models.append(model)
                    generation_limit = config.models["exec"].max_tokens
                    kwargs = {"max_tokens": generation_limit}
                    if response_format:
                        kwargs["response_format"] = response_format
                    response = model(messages, **kwargs)
                    content = response if isinstance(response, str) else getattr(response, "content", None)
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty compact generation")
                    raw_path = directory / "raw_response.txt"
                    raw_path.write_text(content, encoding="utf-8")
                    finish_reason = getattr(model, "last_request_metadata", {}).get("finish_reason")
                    answer = render_response(profile, content)
                durations["generation"] = time.monotonic() - generation_started
                checks = generation_checks(task, answer) if benchmark == "writingbench" else {"complete": True}
                if report_sections:
                    checks["sections_complete"] = len(report_sections) == REPORT_SECTION_COUNT and all(
                        section["complete"] for section in report_sections)
                    checks["complete"] = checks["complete"] and checks["sections_complete"]
                (directory / "answer.txt").write_text(answer, encoding="utf-8")
                write_json(directory / "submission.json", {"answer": answer, "answer_hash": digest(answer),
                           "submitted_at": utc_now(), "checks": checks, "finish_reason": finish_reason})
                evaluation_started = time.monotonic()
                stage, stage_started = "evaluation", evaluation_started

                def judge_factory():
                    model = JudgeEnvelopeModel(provider.create("judge", "targeted-compact-judge", ledger, "evaluation"))
                    models.append(model)
                    return model

                judge = judge_factory()
                spec = config.models["judge"]
                evaluator = dataset.evaluator(judge, judge_id=spec.model, judge_api_base=spec.endpoint,
                    judge_max_tokens=spec.max_tokens, judge_timeout=spec.timeout,
                    max_parallel_judgments=config.judge_parallel, judge_factory=judge_factory)
                raw = evaluator.evaluate(answer, ground_truth=task_id, private_record=dataset.private_records[task_id])
                evaluation = convert_feedback(raw).model_dump(mode="json")
                write_json(directory / "evaluation.json", evaluation)
                durations["evaluation"] = time.monotonic() - evaluation_started
                complete = evaluation["complete"] and checks["complete"] and finish_reason not in {
                    "length", "content_filter", "tool_calls", "function_call"}
                status = "complete" if complete else "incomplete"
            except Exception as error:
                failure = _failure(error)
                durations[stage] = time.monotonic() - stage_started
            finally:
                latency = time.monotonic() - began
                budget = ledger.snapshot()
                budget["wall_seconds"] = latency
                for model in models:
                    traces.append({"stage": "evaluation" if isinstance(model, JudgeEnvelopeModel)
                                   else "inference", "calls": model.calls})
                close_errors = _clients_close(models)
                row = {"benchmark": benchmark, "task_id": task_id, "phase": "test", "method": "targeted-compact-v6",
                       "profile": profile, "status": status, "complete": status == "complete",
                       "answer_hash": digest(answer) if answer is not None else None,
                       "answer_file": str((directory / "answer.txt").relative_to(output)) if answer else None,
                       "score_result": {"complete": (evaluation or {}).get("complete", False),
                                        "score": (evaluation or {}).get("score"), "evaluation": evaluation},
                       "outcome": {"evaluation": evaluation} if evaluation else None,
                       "budget": budget, "latency_seconds": latency, "latency_by_stage": durations,
                       "generation_checks": checks, "generation_finish_reason": finish_reason,
                       "section_generation": report_sections,
                       "failure": failure, "failure_stage": stage if failure else None,
                       "client_close_errors": close_errors}
                write_json(directory / "record.json", _redact(row))
                write_json(directory / "call_trace.json", _redact(traces))
                _append(output / "records.jsonl", row)
                records.append(row)
                print(json.dumps({"task_id": task_id, "status": status,
                                  "score": (evaluation or {}).get("score"), "tokens": budget["tokens"],
                                  "model_calls": budget["model_calls"], "checks": checks}), flush=True)
    write_json(output / "report.json", {"finished_at": utc_now(), "records": _redact(records)})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config, materials = load_inputs(args.config, args.plan)
    return run(config, materials, args.output)


if __name__ == "__main__":
    main()
