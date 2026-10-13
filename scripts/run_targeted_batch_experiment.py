"""Run the frozen independent batch campaign with targeted prompt guidance.

The targeted layer changes only the public task wording and wraps the primary
judge response normalizer.  Dataset private records, split membership, model
metering, campaign accounting and the frozen v7 protocol remain owned by the
independent runners.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from jit_mas.independent_campaign import TERMINAL, coordinator_lock
from jit_mas.pipeline import write_json
from jit_mas.schemas import digest, utc_now
from scripts.eval.config import load_dotenv
from scripts.mas_baseline_methods import JUDGE_ENVELOPE_VERSION, JudgeEnvelopeModel
from scripts.run_benchmark_experiment import file_hash
from scripts.run_independent_batch_experiment import IndependentBatchEnvironment
from scripts.run_independent_test_release import IndependentTestReleaseRunner


TARGETED_GUIDE_VERSION = "targeted-v1-public-guide"
TARGETED_GUIDE_MARKER = f"[JIT-MAS targeted optimization guide: {TARGETED_GUIDE_VERSION}]"


def _research_guide(task, private_record):
    from jit_mas.targeted_research import PROFILE_GUIDANCE, SYSTEM_PROMPT, research_profile

    profile = research_profile(task)
    return profile, SYSTEM_PROMPT + "\n" + PROFILE_GUIDANCE.get(profile, PROFILE_GUIDANCE["ordinary"])


def _writing_guide(task, private_record):
    from jit_mas.targeted_writing import writing_profile, build_messages

    profile = writing_profile(task)
    messages, _ = build_messages(task)
    system = messages[0].get("content", "") if messages else ""
    return profile, system


def _guide_text(benchmark, task, private_record):
    if benchmark in {"deepsearchqa", "researchrubrics", "deepresearch_bench_ii"}:
        profile, guidance = _research_guide(task, private_record)
    elif benchmark == "writingbench":
        profile, guidance = _writing_guide(task, private_record)
    else:
        profile, guidance = "ordinary", "Answer every explicit public requirement in its requested format."
    from jit_mas.targeted_checklists import (CHECKLIST_GUIDANCE, coverage_checklist,
                                             targeted_high_score_directives)

    checklist = coverage_checklist(private_record)
    payload = {
        "version": TARGETED_GUIDE_VERSION,
        "benchmark": benchmark,
        "profile": profile,
        "guidance": guidance,
        "checklist_policy": CHECKLIST_GUIDANCE,
    }
    if checklist:
        payload["task_specific_coverage_checklist"] = checklist
    directives = targeted_high_score_directives(task)
    if directives:
        payload["targeted_high_score_directives"] = directives
    return TARGETED_GUIDE_MARKER + "\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _append_public_guide(question, guide):
    from jit_mas.evidence import EVIDENCE_QUESTION_HEADER

    prefix, separator, evidence = question.partition(EVIDENCE_QUESTION_HEADER)
    if separator:
        return prefix.rstrip() + "\n\n" + guide + separator + evidence
    return question.rstrip() + "\n\n" + guide


def configure_targeted_pipeline(pipeline, *, benchmark=None):
    """Apply targeted public guides and judge normalization exactly once."""
    if getattr(pipeline, "targeted_optimization_configured", False):
        return pipeline
    benchmark = benchmark or getattr(getattr(pipeline, "benchmark_dataset", None), "name", "")
    guide_hashes = {}
    tasks = dict(pipeline.tasks)
    for task_id, task in pipeline.tasks.items():
        guide = _guide_text(benchmark, task, pipeline.private_records.get(task_id))
        tasks[task_id] = task.model_copy(update={
            "question": _append_public_guide(task.question, guide),
        })
        guide_hashes[task_id] = digest(guide)
    pipeline.tasks = tasks

    original_factory = pipeline.evaluator_factory

    def targeted_evaluator_factory(judge):
        if judge is None:
            return original_factory(None)
        evaluator = original_factory(JudgeEnvelopeModel(judge))
        child_factory = getattr(evaluator, "_judge_factory", None)
        if child_factory is not None:
            evaluator._judge_factory = lambda: JudgeEnvelopeModel(child_factory())
        return evaluator

    pipeline.evaluator_factory = targeted_evaluator_factory
    pipeline.targeted_optimization_configured = True
    pipeline.targeted_optimization = {
        "version": TARGETED_GUIDE_VERSION,
        "benchmark": benchmark,
        "guide_hashes": guide_hashes,
        "public_task_count": len(guide_hashes),
        "private_records_unchanged": True,
        "judge_envelope": JUDGE_ENVELOPE_VERSION,
    }
    pipeline.targeted_optimization_hash = digest(pipeline.targeted_optimization)
    return pipeline


class TargetedTestReleaseRunner(IndependentTestReleaseRunner):
    """TEST release using the same public-guide and judge hook as EVO/VAL."""

    def _pipeline(self, slot, snapshot):
        pipeline, store, output = super()._pipeline(slot, snapshot)
        return configure_targeted_pipeline(pipeline, benchmark=slot["target"]), store, output


class TargetedBatchEnvironment(IndependentBatchEnvironment):
    """Independent v7 batch environment with targeted optimization enabled."""

    runner_path = Path(__file__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity["targeted_optimization"] = {
            "version": TARGETED_GUIDE_VERSION,
            "runner_sha256": file_hash(Path(__file__)),
            "judge_envelope": JUDGE_ENVELOPE_VERSION,
            "guide_hashes": {
                name: {
                    task_id: digest(_guide_text(name, task, dataset.private_records.get(task_id)))
                    for task_id, task in dataset.tasks.items()
                }
                for name, dataset in self.datasets.items()
            },
        }

    def _pipeline(self, source, run_id, store, output):
        pipeline = super()._pipeline(source, run_id, store, output)
        return configure_targeted_pipeline(pipeline, benchmark=source)

    def run(self, *, phase="full"):
        self.initialize()
        release = TargetedTestReleaseRunner(self.launch_path, self.output, environment=self)
        with coordinator_lock(self.output):
            self.recover()
            release._recover_interrupted_tests()
            with ThreadPoolExecutor(max_workers=self.workers) as workers:
                active = {}
                while True:
                    self.campaign.advance()
                    while len(active) < self.workers:
                        claim = self.campaign.claim("batch-worker", phase=phase)
                        if claim is None and phase == "full":
                            claim = self.campaign.claim("test-worker", kinds=("test",), prefer="test")
                        if claim is None:
                            break
                        callback = release._submit_claim if claim["kind"] == "test" else self.execute
                        active[workers.submit(callback, claim)] = claim
                    report = self.campaign.status(phase=phase)
                    report.update(active=len(active), updated_at=utc_now(), test_feedback_released=False,
                                  targeted_optimization=TARGETED_GUIDE_VERSION)
                    write_json(self.output / "status.json", report)
                    if not active:
                        break
                    finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                        active.pop(future)
            self.campaign.advance()
            relevant = [row for row in self.campaign.records() if row["kind"] != "test"]
            if phase == "first-stage":
                relevant = [row for row in relevant if row["run_id"] == 0 and row["batch_index"] == 0]
            if any(row["status"] not in TERMINAL for row in relevant):
                raise RuntimeError("No runnable work remains but the batch dependency inventory is unfinished")
            report = self.campaign.status(phase=phase)
            report.update(evolution_validation_complete=True, test_feedback_released=False,
                          targeted_optimization=TARGETED_GUIDE_VERSION)
            write_json(self.output / f"{phase}_report.json", report)
        if phase == "full":
            release.seal()
            report["test"] = release.score(workers=self.workers)
            from scripts.summarize_independent_paper import markdown, summarize

            paper = summarize(self.output)
            paper_dir = self.output.parent / f"{self.output.name}_paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            write_json(paper_dir / "paper_summary.json", paper)
            (paper_dir / "paper_summary.md").write_text(markdown(paper), encoding="utf-8")
            report["paper"] = {"directory": str(paper_dir), "registration_sha256": paper["registration_sha256"]}
            write_json(self.output / "full_report.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run", "status"), required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("first-stage", "full"), default="full")
    args = parser.parse_args(argv)
    load_dotenv()
    environment = TargetedBatchEnvironment(args.launch_config, args.output, require_preflight=args.mode == "run")
    if args.mode == "run":
        result = environment.run(phase=args.phase)
    elif args.mode == "status":
        environment.initialize()
        result = environment.campaign.status(phase=args.phase)
    else:
        workload = environment.protocol.get("workload", {})
        result = {"inputs_ready": True, "model_calls": 0,
                  "protocol_sha256": environment.protocol["protocol_sha256"],
                  "task_slots": workload.get("total_task_slots"),
                  "batch_evolution_operations": workload.get("trajectories", 0) * 8,
                  "candidate_generation_operations": workload.get("trajectories", 0) * 24,
                  "targeted_optimization": TARGETED_GUIDE_VERSION}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
