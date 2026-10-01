"""Frozen submission inventory and delayed benchmark feedback release."""

from __future__ import annotations

import json
from pathlib import Path

from .budget import BudgetLedger
from .checkpoints import CheckpointIntegrityError
from .pipeline import convert_feedback, write_json
from .schemas import digest, utc_now


class TestRelease:
    """A single-coordinator campaign; every registered slot must terminate before judging."""

    __test__ = False

    def __init__(self, directory, inventory=None):
        self.directory = Path(directory)
        path = self.directory / "inventory.json"
        if inventory is not None:
            if not inventory or any(not isinstance(row, dict) or not row.get("slot_id") for row in inventory):
                raise ValueError("Nonempty explicit submission inventory required")
            if len({row["slot_id"] for row in inventory}) != len(inventory):
                raise ValueError("Duplicate submission slot IDs")
            document = {"slots": inventory, "inventory_hash": digest(inventory)}
            if path.exists() and self._read(path) != document:
                raise ValueError("Cannot change a registered submission inventory")
            if not path.exists():
                write_json(path, document)
        self.inventory = self._read(path)
        if digest(self.inventory["slots"]) != self.inventory["inventory_hash"]:
            raise ValueError("Submission inventory hash mismatch")
        self.slots = {row["slot_id"]: row for row in self.inventory["slots"]}

    @staticmethod
    def _read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _path(self, slot_id, kind="submissions"):
        if slot_id not in self.slots:
            raise ValueError("Unregistered test slot")
        return self.directory / kind / f"{digest(slot_id)}.json"

    def record(self, slot_id, outcome):
        if (self.directory / "seal.json").exists():
            raise ValueError("Submissions are sealed")
        slot = self.slots[slot_id]
        if outcome.get("task_id") != slot["task_id"] or outcome.get("evaluation") is not None:
            raise ValueError("Expected this slot's unscored submission")
        if outcome.get("status") != "submitted_unscored" or outcome.get("proposals"):
            raise ValueError("Test submission must not evaluate or attribute")
        source = Path(outcome["run_dir"]) / "submission.json"
        submission = self._read(source)
        if digest(submission["answer"]) != submission["answer_hash"] or outcome["answer_hash"] != submission["answer_hash"]:
            raise ValueError("Submitted answer hash mismatch")
        for name in ("experience_hash", "comparison_fingerprint"):
            if name in slot and slot[name] != outcome.get(name):
                raise ValueError("Submission does not match frozen condition")
        record = {"slot": slot, "status": "submitted", "submission": submission, "outcome": outcome}
        self._record(slot_id, record)

    def record_failure(self, slot_id, *, error_type, budget):
        if (self.directory / "seal.json").exists():
            raise ValueError("Submissions are sealed")
        self._record(slot_id, {"slot": self.slots[slot_id], "status": "failed",
                              "error_type": error_type, "budget": budget})

    def _record(self, slot_id, record):
        path = self._path(slot_id)
        if path.exists():
            if self._read(path) != record:
                raise ValueError("A test slot cannot be replaced or resampled")
            return
        write_json(path, record)

    def seal(self):
        hashes = {}
        for slot_id in self.slots:
            path = self._path(slot_id)
            if not path.exists():
                raise ValueError("Every registered test slot must submit or record its failure before feedback release")
            hashes[slot_id] = digest(self._read(path))
        body = {"inventory_hash": self.inventory["inventory_hash"], "submission_hashes": hashes}
        path = self.directory / "seal.json"
        if path.exists() and self._read(path) != body:
            raise ValueError("Sealed test inventory changed")
        if not path.exists():
            write_json(path, body)
        return body

    def evaluate(self, slot_id, scorer):
        if not (self.directory / "seal.json").exists():
            raise ValueError("Cannot score before the entire test campaign is sealed")
        self.seal()
        output = self._path(slot_id, "evaluations")
        if output.exists():
            return self._read(output)
        record = self._read(self._path(slot_id))
        if record["status"] == "failed":
            result = {"slot": record["slot"], "official_score": None, "complete": False,
                      "status": "submission_failed", "generation_budget": record["budget"]}
        else:
            started = self._path(slot_id, "evaluation_started")
            if started.exists():
                raise RuntimeError("Interrupted judgment must be inspected; do not silently resample")
            write_json(started, {"record_hash": digest(record), "started_at": utc_now()})
            try:
                result = scorer(record)
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                result = {"slot": record["slot"], "official_score": None, "complete": False,
                          "status": "evaluation_failed", "error_type": type(exc).__name__,
                          "evaluation_budget": getattr(exc, "evaluation_budget", {})}
        write_json(output, result)
        return result


def remaining_task_budget(config, outcome):
    """Deferred evaluation shares the original task cap, even after artifact reuse."""
    if outcome.get("reused_from") and "task_generation_budget" not in outcome:
        raise CheckpointIntegrityError("Reused artifact lacks its original task-generation budget")
    used = outcome.get("task_generation_budget", outcome["budget"])
    if used.get("usage_unknown") or used.get("reserved_tokens", 0):
        raise CheckpointIntegrityError("Cannot reopen an unknown or unsettled task budget for scoring")
    limits = {"max_calls": (config.max_model_calls, "model_calls"),
              "max_tokens": (config.max_total_tokens, "tokens"),
              "max_tool_calls": (config.max_tool_calls, "tool_calls")}
    remaining = {}
    for name, (limit, key) in limits.items():
        value = used.get(key)
        if type(value) is not int or value < 0:
            raise CheckpointIntegrityError("Missing or invalid original task-generation accounting")
        remaining[name] = None if limit is None else max(0, limit - value)
    return remaining


def score_with_pipeline(pipeline, record):
    """Judge an already sealed answer without constructing or executing a team."""
    slot, submission = record["slot"], record["submission"]
    task_id = slot["task_id"]
    if task_id not in pipeline.manifest.test:
        raise ValueError("Saved submission is outside the target test partition")
    if digest(submission["answer"]) != submission["answer_hash"]:
        raise ValueError("Saved answer integrity failure")
    config = pipeline.config
    ledger = BudgetLedger(**remaining_task_budget(config, record["outcome"]))
    try:
        judge = pipeline.models.create("judge", "judge", ledger, "evaluation")
        evaluator = pipeline.evaluator_factory(judge)
        raw = evaluator.evaluate(str(submission["answer"]), ground_truth=task_id,
                                 private_record=pipeline.private_records[task_id])
        raw["submission_answer_hash"] = submission["answer_hash"]
        feedback = convert_feedback(raw)
    except Exception as exc:
        exc.evaluation_budget = ledger.snapshot()
        raise
    return {"slot": slot, "official_score": feedback.score, "complete": feedback.complete,
            "evaluation": feedback.model_dump(mode="json"), "evaluated_at": utc_now(),
            "evaluation_budget": ledger.snapshot(), "generation_budget": record["outcome"]["budget"],
            "answer_hash": submission["answer_hash"]}
