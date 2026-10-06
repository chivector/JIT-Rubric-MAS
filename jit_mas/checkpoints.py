"""Periodic full-validation selection, separate from direct experience updates.

Callbacks run the existing pipeline; this coordinator never changes its proposal
policy or feeds validation feedback to an evolution callback. A run directory is
owned by one coordinator process. Interrupted requests are not sampled again.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import uuid
from typing import Callable, Mapping

from .experience import ExperienceStore
from .schemas import ExperienceSnapshot, digest


class CheckpointIntegrityError(RuntimeError):
    """The frozen identity, stored state, or read-only evaluation was changed."""


class UnresolvedSourceAttempt(RuntimeError):
    """A started source request cannot be replayed safely after interruption."""


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_store(snapshot: ExperienceSnapshot, path) -> ExperienceStore:
    """Materialize one immutable historical state and return an independent RO connection."""
    path = Path(path)
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        if temporary.exists():
            temporary.unlink()
        store = ExperienceStore(temporary)
        try:
            with store.db:
                store.db.execute("DELETE FROM snapshots")
                store.db.execute("INSERT INTO snapshots VALUES(?,?)", (snapshot.version, snapshot.model_dump_json()))
                store.db.execute("UPDATE state SET value=? WHERE key='current'", (str(snapshot.version),))
        finally:
            store.close()
        temporary.replace(path)
    frozen = ExperienceStore(path, read_only=True)
    if digest(frozen.snapshot()) != digest(snapshot):
        frozen.close()
        raise CheckpointIntegrityError("Frozen snapshot database content changed")
    return frozen


def select_checkpoint(checkpoints, *, tolerance=1e-12):
    """Select against one global maximum, avoiding non-transitive pairwise ties."""
    eligible = [row for row in checkpoints if row["eligible"]]
    if not eligible:
        return None
    maximum = max(row["selection_utility"] for row in eligible)
    tied = [row for row in eligible if row["selection_utility"] >= maximum - tolerance]
    return min(tied, key=lambda row: (-row["complete_evaluations"], row["position"], row["state_hash"]))


def _outcome_dict(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise TypeError("Pipeline callback must return one outcome mapping")
    return value


def _failure(exc):
    return {"error_type": type(exc).__name__, "error": str(exc),
            **getattr(exc, "jit_mas_run_failure", {})}


class CheckpointRunner:
    """Execute source slots once and validate every cumulative scheduled state.

    ``evolve(task_id)`` must use the mutable pipeline's ``run('evolve', ...)``.
    ``evaluate(task_id, snapshot, repeat, read_only_store)`` must use a separate
    pipeline with ``run_task(..., mode='validate', attribution=False)``. Neither
    callback receives the other split's feedback. ``identity`` binds code,
    configuration, tools, evidence, model serving identity, and selector version.
    """

    def __init__(self, *, store: ExperienceStore, evolution_ids, validation_ids,
                 output_dir, identity: Mapping, lower_bounds: Mapping[str, float],
                 evolve: Callable, evaluate: Callable, batch_size=5, repeats=2,
                 minimum_completion=0.9, recover_evaluation: Callable | None = None):
        self.store = store
        self.evolution_ids, self.validation_ids = list(evolution_ids), list(validation_ids)
        if store.read_only:
            raise ValueError("Evolution requires a writable experience store")
        if not self.evolution_ids or not self.validation_ids:
            raise ValueError("Evolution and validation splits must both be nonempty")
        if len(set(self.evolution_ids)) != len(self.evolution_ids) or len(set(self.validation_ids)) != len(self.validation_ids):
            raise ValueError("Task IDs must be unique within each split")
        if set(self.evolution_ids).intersection(self.validation_ids):
            raise ValueError("Evolution and validation splits overlap")
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
                or not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1):
            raise ValueError("Batch size and repeats must be positive integers")
        if not 0 < minimum_completion <= 1:
            raise ValueError("Minimum completion fraction must be in (0, 1]")
        self.lower_bounds = {key: float(lower_bounds[key]) for key in self.validation_ids}
        if not all(math.isfinite(value) for value in self.lower_bounds.values()):
            raise ValueError("Task lower bounds must be finite")
        self.evolve, self.evaluate = evolve, evaluate
        self.recover_evaluation = recover_evaluation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.repeats = batch_size, repeats
        self.total_evaluations = len(self.validation_ids) * repeats
        self.minimum_complete = math.ceil(self.total_evaluations * minimum_completion - 1e-12)
        self.positions = sorted({0, len(self.evolution_ids), *range(batch_size, len(self.evolution_ids) + 1, batch_size)})
        self.identity = {"schema_version": "periodic-full-val-v1", "execution": dict(identity),
                         "evolution_ids": self.evolution_ids, "validation_ids": self.validation_ids,
                         "lower_bounds": self.lower_bounds, "batch_size": batch_size,
                         "repeats": repeats, "minimum_complete": self.minimum_complete,
                         "checkpoint_positions": self.positions}
        self.journal_path = self.output_dir / "checkpoint_journal.json"
        if self.journal_path.exists():
            self.journal = _read_json(self.journal_path)
            if digest(self.journal["identity"]) != digest(self.identity):
                raise CheckpointIntegrityError("Cannot resume with a changed experiment identity")
        else:
            initial = self.store.snapshot()
            self.journal = {"identity": self.identity, "initial_snapshot": initial.model_dump(mode="json"),
                            "sources": [], "checkpoints": [], "validation_cache": {},
                            "status": "running", "selected_position": None}
            self._save()
        self._check_trajectory()

    def _save(self):
        _write_json(self.journal_path, self.journal)

    def _check_trajectory(self):
        sources = self.journal["sources"]
        if sources and sources[-1]["status"] == "started":
            return
        expected = sources[-1]["after_hash"] if sources else digest(self.journal["initial_snapshot"])
        if digest(self.store.snapshot()) != expected:
            raise CheckpointIntegrityError("Mutable trajectory was changed outside the recorded source sequence")

    def _source(self, task_id, position):
        sources = self.journal["sources"]
        if len(sources) >= position:
            row = sources[position - 1]
            if row["task_id"] != task_id:
                raise CheckpointIntegrityError("Recorded source task order changed")
            if row["status"] != "started":
                return
            prior = self.store.task_run("evolve", task_id)
            if prior and (prior["baseline_hash"] != row["before_hash"]
                          or prior["baseline_version"] != row["before_version"]):
                raise CheckpointIntegrityError("Interrupted source journal baseline does not match its slot")
            if prior and prior["status"] == "complete":
                outcome = prior["outcome"]
            elif prior and prior["status"] == "submitted" and prior["outcome"]:
                # MASPipeline.run resumes its saved artifact/proposal, not a new request.
                outcome = _outcome_dict(self.evolve(task_id))
            else:
                raise UnresolvedSourceAttempt(
                    f"Source slot {position} ({task_id}) was interrupted. Resolve its persisted "
                    "pipeline outcome or call record_interrupted_source_failure; do not regenerate it.")
            if (outcome.get("next_experience_version") is not None
                    and self.store.snapshot().version != outcome["next_experience_version"]):
                raise CheckpointIntegrityError("Interrupted source outcome does not match the current store")
            row.update(status="complete", outcome=outcome, recovered=True)
        else:
            self._check_trajectory()
            before = self.store.snapshot()
            row = {"position": position, "task_id": task_id, "status": "started",
                   "before_version": before.version, "before_hash": digest(before)}
            sources.append(row)
            self._save()
            try:
                row.update(status="complete", outcome=_outcome_dict(self.evolve(task_id)))
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                row.update(status="failed", failure=_failure(exc))
        after = self.store.snapshot()
        row.update(after_version=after.version, after_hash=digest(after))
        self._save()

    def record_interrupted_source_failure(self, reason, *, budget=None):
        """Close an unresolved crashed slot without generating a replacement artifact.

        Call only after the originating process has stopped and its persisted
        output has been inspected. Committed direct updates are never rolled back.
        """
        sources = self.journal["sources"]
        if not sources or sources[-1]["status"] != "started" or not str(reason).strip():
            raise ValueError("An unresolved source and a nonempty audit reason are required")
        row = sources[-1]
        prior = self.store.task_run("evolve", row["task_id"])
        if prior and prior["status"] in ("complete", "submitted"):
            raise ValueError("Persisted outcomes must be recovered, not replaced by a failure")
        snapshot = self.store.snapshot()
        row.update(status="failed", failure={"error_type": "InterruptedSource", "error": str(reason),
                                             "budget": budget},
                   after_version=snapshot.version, after_hash=digest(snapshot))
        self._save()

    def _evaluate_slot(self, task_id, snapshot, repeat, frozen, row):
        recovered = None
        if row["status"] == "started":
            if self.recover_evaluation:
                before = digest(self.store.snapshot())
                original = digest(snapshot)
                supplied = snapshot.model_copy(deep=True)
                try:
                    recovered = self.recover_evaluation(task_id, supplied, repeat, frozen)
                finally:
                    if (digest(self.store.snapshot()) != before or digest(supplied) != original
                            or digest(frozen.snapshot()) != original):
                        raise CheckpointIntegrityError("Validation recovery mutated a frozen state or trajectory")
            if recovered is None:
                row.update(status="failed", failure={"error_type": "InterruptedValidation",
                    "error": "Started validation slot has no durable completed outcome; no resampling"})
        else:
            row["status"] = "started"
            self._save()
            before = digest(self.store.snapshot())
            original = digest(snapshot)
            supplied = snapshot.model_copy(deep=True)
            try:
                recovered = self.evaluate(task_id, supplied, repeat, frozen)
            except CheckpointIntegrityError:
                raise
            except Exception as exc:
                row.update(status="failed", failure=_failure(exc))
            finally:
                if (digest(self.store.snapshot()) != before or digest(supplied) != original
                        or digest(frozen.snapshot()) != original):
                    raise CheckpointIntegrityError("Validation mutated its state or the evolution trajectory")
        if recovered is not None:
            outcome = _outcome_dict(recovered)
            feedback = outcome.get("evaluation", outcome)
            score = feedback.get("score")
            complete = (feedback.get("complete") is True and isinstance(score, (int, float))
                        and not isinstance(score, bool) and math.isfinite(score))
            row.update(status="complete" if complete else "incomplete", outcome=outcome,
                       official_score=float(score) if complete else None,
                       complete_evaluation=complete)
        row.setdefault("official_score", None)
        row.setdefault("complete_evaluation", False)
        row["is_imputed_for_selection"] = not row["complete_evaluation"]
        row["selection_utility"] = (row["official_score"] if row["complete_evaluation"]
                                    else self.lower_bounds[task_id])
        self._save()

    def _checkpoint(self, position):
        existing = next((row for row in self.journal["checkpoints"] if row["position"] == position), None)
        if existing:
            return existing
        snapshot = self.store.snapshot()
        state_hash = digest(snapshot)
        cache_key = digest({"state": state_hash, "identity": self.identity})
        cache = self.journal["validation_cache"]
        if cache_key not in cache:
            cache[cache_key] = {"state_hash": state_hash, "first_position": position,
                                "snapshot": snapshot.model_dump(mode="json"), "slots": []}
            self._save()
        cached = cache[cache_key]
        if digest(cached["snapshot"]) != state_hash:
            raise CheckpointIntegrityError("Validation cache snapshot changed")
        frozen = snapshot_store(snapshot, self.output_dir / "states" / f"{state_hash}.sqlite")
        try:
            for task_id in self.validation_ids:
                for repeat in range(self.repeats):
                    row = next((row for row in cached["slots"] if row["task_id"] == task_id and row["repeat"] == repeat), None)
                    if row is None:
                        row = {"task_id": task_id, "repeat": repeat, "status": "pending"}
                        cached["slots"].append(row)
                    if row["status"] in ("pending", "started"):
                        self._evaluate_slot(task_id, snapshot, repeat, frozen, row)
        finally:
            frozen.close()
        count = sum(row["complete_evaluation"] for row in cached["slots"])
        checkpoint = {"position": position, "state_version": snapshot.version, "state_hash": state_hash,
                      "cache_key": cache_key, "total_evaluations": self.total_evaluations,
                      "complete_evaluations": count, "eligible": count >= self.minimum_complete,
                      "selection_utility": math.fsum(row["selection_utility"] for row in cached["slots"]) / self.total_evaluations,
                      "utility_is_conservative_surrogate": count != self.total_evaluations,
                      "reused_from": cached["first_position"] if cached["first_position"] != position else None}
        self.journal["checkpoints"].append(checkpoint)
        selected = select_checkpoint(self.journal["checkpoints"])
        self.journal["provisional_selected_position"] = selected["position"] if selected else None
        self._save()
        return checkpoint

    def accounting(self):
        """Sum actual callback attempts only, never duplicated cached checkpoints."""
        slots = list(self.journal["sources"])
        slots.extend(row for cached in self.journal["validation_cache"].values() for row in cached["slots"])
        calls = tokens = unknown = 0
        for row in slots:
            budget = row.get("outcome", {}).get("budget") or row.get("failure", {}).get("budget")
            if budget is None:
                unknown += 1
            else:
                calls += budget.get("model_calls", 0)
                tokens += budget.get("tokens", 0)
        return {"source_slots": len(self.journal["sources"]),
                "actual_validation_slots": sum(len(row["slots"]) for row in self.journal["validation_cache"].values()),
                "nominal_validation_slots": len(self.journal["checkpoints"]) * self.total_evaluations,
                "model_calls_recorded": calls, "tokens_recorded": tokens,
                "attempts_with_unknown_budget": unknown, "cost": None}

    def run(self):
        self._checkpoint(0)
        for position, task_id in enumerate(self.evolution_ids, 1):
            self._source(task_id, position)
            if position in self.positions:
                self._checkpoint(position)
        self._check_trajectory()
        selected = select_checkpoint(self.journal["checkpoints"])
        self.journal["status"] = "complete" if selected else "inconclusive"
        self.journal["selected_position"] = selected["position"] if selected else None
        self.journal["accounting"] = self.accounting()
        self._save()
        return {"status": self.journal["status"], "selected": selected,
                "checkpoints": self.journal["checkpoints"], "accounting": self.journal["accounting"],
                "journal_path": str(self.journal_path)}

    def selected_snapshot(self):
        if self.journal["status"] != "complete":
            raise ValueError("No eligible final checkpoint has been sealed")
        selected = next(row for row in self.journal["checkpoints"]
                        if row["position"] == self.journal["selected_position"])
        snapshot = ExperienceSnapshot.model_validate(self.journal["validation_cache"][selected["cache_key"]]["snapshot"])
        if digest(snapshot) != selected["state_hash"]:
            raise CheckpointIntegrityError("Sealed selected snapshot content changed")
        return snapshot
