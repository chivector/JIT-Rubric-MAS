"""Paired rebuilds on independent complete validation tasks, never answer rewrites."""

from __future__ import annotations

from typing import Callable

from pydantic import Field

from .experience import candidate_snapshot
from .schemas import (ChangeProposal, ExperienceSnapshot, Record, SplitManifest,
                      EvaluationFeedback, ValidationResult, digest)


class ValidationConfig(Record):
    min_tasks: int = Field(default=2, ge=1)
    repeats: int = Field(default=1, ge=1)
    quality_tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)
    criterion_tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)
    min_mean_improvement: float = Field(default=0, ge=0, allow_inf_nan=False)


class PairedValidator:
    def __init__(self, rebuild: Callable, manifest: SplitManifest, config=None):
        self.rebuild = rebuild
        self.manifest = manifest
        self.config = config or ValidationConfig()

    def validate(self, proposal: ChangeProposal, baseline: ExperienceSnapshot, task_ids=None):
        candidate = candidate_snapshot(baseline, proposal)
        ids = list(self.manifest.validation if task_ids is None else task_ids)
        if len(ids) != len(set(ids)) or not set(ids) <= set(self.manifest.validation):
            raise ValueError("Validation requires distinct tasks from the validation split")
        if proposal.source_task_id in ids:
            raise ValueError("Cannot validate on the source task")
        hashes = {"proposal_hash": digest(proposal), "baseline_hash": digest(baseline),
                  "candidate_hash": digest(candidate), "config_hash": digest(self.config)}
        pairs = []
        status, reason = "pending", "Insufficient independent validation tasks"
        if len(ids) >= self.config.min_tasks:
            status, reason = "accepted", "Paired task quality improved without criterion regression"
            deltas, invalid, regression = [], False, False
            for task_id in ids:
                per_task = []
                for repeat in range(self.config.repeats):
                    pair = {"task_id": task_id, "repeat": repeat}
                    try:
                        old = self.rebuild(task_id, baseline.model_copy(deep=True), repeat, "baseline")
                        new = self.rebuild(task_id, candidate.model_copy(deep=True), repeat, "candidate")
                        pair.update(baseline=old, candidate=new)
                        old_eval = EvaluationFeedback.model_validate(old["evaluation"]).model_dump(mode="json")
                        new_eval = EvaluationFeedback.model_validate(new["evaluation"]).model_dump(mode="json")
                        for run, evaluation in ((old, old_eval), (new, new_eval)):
                            if run.get("task_id") != task_id or evaluation["task_id"] != task_id:
                                raise ValueError("Validation returned feedback for a different task")
                            rubric_ids = [r["rubric_id"] for r in evaluation["rubrics"]]
                            if not rubric_ids or len(set(rubric_ids)) != len(rubric_ids):
                                raise ValueError("Validation requires distinct nonempty rubric IDs")
                            if any(r["status"] != "ok" or r["score"] is None for r in evaluation["rubrics"]):
                                evaluation["complete"] = False
                        if (not old_eval["complete"] or not new_eval["complete"]
                                or old_eval["score"] is None or new_eval["score"] is None):
                            invalid = True
                        elif (old["comparison_fingerprint"] != new["comparison_fingerprint"]
                              or old_eval["evaluator_version"] != new_eval["evaluator_version"]):
                            invalid = True
                            pair["error"] = "Paired models/tools/budgets/evaluator differ"
                        else:
                            delta = new_eval["score"] - old_eval["score"]
                            per_task.append(delta)
                            if delta < -self.config.quality_tolerance:
                                regression = True
                            left = {r["rubric_id"]: r for r in old_eval["rubrics"]}
                            right = {r["rubric_id"]: r for r in new_eval["rubrics"]}
                            if left.keys() != right.keys():
                                invalid = True
                            for rid in left.keys() & right.keys():
                                a, b = left[rid], right[rid]
                                if a["weight"] != b["weight"] or a["score"] is None or b["score"] is None:
                                    invalid = True
                                    continue
                                direction = 1 if a["weight"] >= 0 else -1
                                if direction * (b["score"] - a["score"]) < -self.config.criterion_tolerance:
                                    regression = True
                    except Exception as exc:
                        invalid = True
                        pair["error"] = f"{type(exc).__name__}: {exc}"
                    pairs.append(pair)
                if per_task:
                    deltas.append(sum(per_task) / len(per_task))
            if invalid or len(deltas) < self.config.min_tasks:
                status, reason = "pending", "Incomplete or incomparable paired evidence"
            elif regression:
                status, reason = "rejected", "Task or criterion quality regressed"
            elif sum(deltas) / len(deltas) <= self.config.min_mean_improvement:
                status, reason = "rejected", "No required mean task-quality improvement"
        key = digest({**hashes, "tasks": ids, "pairs": pairs})
        return ValidationResult(proposal_id=proposal.proposal_id, base_version=baseline.version,
                                status=status, reason=reason, pairs=pairs, validation_id=key, **hashes)
