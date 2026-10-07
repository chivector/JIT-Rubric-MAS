"""Durable scheduling for frozen five-task batches and three-way VAL selection."""

from __future__ import annotations

import json
import uuid

from .checkpoints import CheckpointIntegrityError
from .independent_campaign import IndependentCampaign, TERMINAL, _encode
from .independent_batch_protocol import (
    BATCHES_PER_TRAJECTORY, SOURCES, TARGETS, RUN_IDS,
    select_batch_candidate, validate_protocol, validation_summary,
)
from .independent_protocol import TEST_METHODS
from .schemas import ExperienceSnapshot, digest, utc_now


def _protocol_axes(protocol):
    """Return registered source/target/run axes (v6 and v7 compatible)."""
    sources = tuple(protocol.get("sources", SOURCES))
    targets = tuple(protocol.get("targets", TARGETS))
    runs = tuple(protocol.get("run_ids", sorted({row["run_id"] for row in protocol["trajectories"]})))
    methods = tuple(protocol.get("static_methods") or (TEST_METHODS if not protocol.get("test_methods") else ()))
    return sources, targets, runs, methods


def slot_registry(protocol):
    validate_protocol(protocol)
    sources, targets, runs, methods = _protocol_axes(protocol)
    entries = []
    for trajectory in protocol["trajectories"]:
        source, run_id = trajectory["source"], trajectory["run_id"]
        common = {"source": source, "run_id": run_id, "status": "pending"}
        for batch in trajectory["batches"]:
            batch_index = batch["batch_index"]
            prefix = f"{source}:run{run_id}:b{batch_index}"
            for task_id in batch["evolution_task_ids"]:
                entries.append({**common, "slot_id": f"evo:{prefix}:{task_id}",
                                "kind": "evolution", "task_id": task_id, "batch_index": batch_index})
            entries.append({**common, "slot_id": f"update:{prefix}", "kind": "batch_evolution",
                            "task_id": f"batch:{prefix}", "batch_index": batch_index})
            for candidate in batch["candidates"]:
                for task_id in trajectory["validation_task_ids"]:
                    entries.append({**common, "slot_id": f"val:{prefix}:c{candidate['candidate_index']}:{task_id}",
                                    "kind": "validation", "task_id": task_id, "batch_index": batch_index,
                                    "candidate_index": candidate["candidate_index"]})
        for target in (source, *targets):
            tasks = trajectory["test_task_ids"] if target == source else protocol["target_test_tasks"][target]
            for task_id in tasks:
                entries.append({**common, "slot_id": f"test:{source}:run{run_id}:{target}:{task_id}",
                                "kind": "test", "method": "ours_selected", "target": target, "task_id": task_id})
    # v7 registers only Ours selected-source/migration slots and therefore has
    # no static baselines.  v6 keeps its historical four static methods.
    if methods:
      for target in (*sources, *targets):
        tasks = (protocol["target_test_tasks"][target] if target in TARGETS else
                 next(row["test_task_ids"] for row in protocol["trajectories"] if row["source"] == target))
        for method in methods:
            for task_id in tasks:
                entries.append({"slot_id": f"test:static:{method}:{target}:{task_id}", "kind": "test",
                                "method": method, "source": "static", "run_id": None,
                                "target": target, "task_id": task_id, "status": "pending"})
    counts = {kind: sum(row["kind"] == kind for row in entries)
              for kind in ("evolution", "batch_evolution", "validation", "test")}
    version = "independent-batch-slot-registry-v7" if protocol.get("version", "").endswith("-v7") else "independent-batch-slot-registry-v6"
    registry = {"version": version, "artifacts": entries,
                "protocol_sha256": protocol["protocol_sha256"], "counts": {**counts, "total": len(entries)}}
    registry["registry_sha256"] = digest(registry)
    return registry


def state_sources(snapshot):
    sources = {task_id for entry in snapshot.experiences for task_id in entry.source_task_ids}
    profiles = list(snapshot.agent_pool.profiles)
    for operation in snapshot.agent_pool.structural_history:
        sources.add(operation.source_task_id)
        sources.update(operation.source_task_ids)
        profiles.extend(operation.profiles)
    for profile in profiles:
        sources.update(profile.source_task_ids)
        sources.update(task_id for lesson in profile.memory for task_id in lesson.source_task_ids)
    sources.update(observation.task_id for observation in snapshot.agent_pool.observations)
    return sources


class IndependentBatchCampaign(IndependentCampaign):
    protocol_validator = staticmethod(validate_protocol)
    registry_builder = staticmethod(slot_registry)
    registration_schema = "independent-batch-campaign-v6"

    def __init__(self, *args, **kwargs):
        # Registration identity is part of the immutable database.  Select the
        # schema before the base constructor computes that identity.
        protocol = args[1] if len(args) > 1 else kwargs.get("protocol", {})
        if protocol.get("version", "").endswith("-v7"):
            self.registration_schema = "independent-batch-campaign-v7"
        super().__init__(*args, **kwargs)
        with self._database() as database:
            database.executescript("""
                CREATE TABLE IF NOT EXISTS candidates(source TEXT,run_id INTEGER,batch_index INTEGER,
                    candidate_index INTEGER,body TEXT NOT NULL,PRIMARY KEY(source,run_id,batch_index,candidate_index));
                CREATE TABLE IF NOT EXISTS batch_selections(source TEXT,run_id INTEGER,batch_index INTEGER,
                    body TEXT NOT NULL,PRIMARY KEY(source,run_id,batch_index));
            """)

    def freeze_checkpoint(self, source, run_id, position, snapshot):
        if (source, run_id) not in self.trajectories or position not in range(0, 41, 5):
            raise ValueError("Unregistered batch winner position")
        snapshot = ExperienceSnapshot.model_validate(snapshot)
        allowed = set(self.trajectories[source, run_id]["evolution_task_ids"][:position])
        if not state_sources(snapshot) <= allowed:
            raise CheckpointIntegrityError("State contains history outside this source EVO prefix")
        state_hash = digest(snapshot)
        with self._transaction() as database:
            if position:
                selected = database.execute("SELECT body FROM batch_selections WHERE source=? AND run_id=? AND batch_index=?",
                                            (source, run_id, position // 5 - 1)).fetchone()
                if not selected or json.loads(selected[0]).get("selected", {}).get("state_hash") != state_hash:
                    raise CheckpointIntegrityError("Next batch state must be the preceding VAL winner")
            existing = database.execute("SELECT state_hash,identity_hash FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                        (source, run_id, position)).fetchone()
            if existing and tuple(existing) != (state_hash, self.registration_hash):
                raise CheckpointIntegrityError("Cannot replace an immutable batch winner")
            if not existing:
                database.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
                                 (source, run_id, position, state_hash, _encode(snapshot.model_dump(mode="json")), self.registration_hash))
        return self.checkpoint(source, run_id, position)

    def candidate(self, source, run_id, batch_index, candidate_index):
        with self._database() as database:
            row = database.execute("SELECT body FROM candidates WHERE source=? AND run_id=? AND batch_index=? AND candidate_index=?",
                                   (source, run_id, batch_index, candidate_index)).fetchone()
        if row is None:
            return None
        body = json.loads(row[0])
        if body.get("snapshot") is not None and digest(body["snapshot"]) != body["state_hash"]:
            raise CheckpointIntegrityError("Frozen candidate snapshot changed")
        return body

    def freeze_candidates(self, claim, candidates):
        if len(candidates) != 3 or {row["candidate_index"] for row in candidates} != {0, 1, 2}:
            raise ValueError("A batch must produce exactly three candidate receipts")
        source, run_id, batch_index = claim["source"], claim["run_id"], claim["batch_index"]
        allowed = set(self.trajectories[source, run_id]["evolution_task_ids"][:(batch_index + 1) * 5])
        with self._transaction() as database:
            slot = database.execute("SELECT status,claim_token FROM slots WHERE slot_id=?", (claim["slot_id"],)).fetchone()
            if claim["kind"] != "batch_evolution" or slot is None or slot["claim_token"] != claim.get("claim_token"):
                raise CheckpointIntegrityError("Candidate freezing requires its exclusive batch claim")
            for candidate in candidates:
                body = dict(candidate)
                if body.get("base_state_hash", claim["state_hash"]) != claim["state_hash"]:
                    raise CheckpointIntegrityError("Candidates must branch from the same batch input state")
                body.update(base_state_hash=claim["state_hash"], batch_index=batch_index,
                            candidate_id=f"b{batch_index}c{candidate['candidate_index']}")
                if body.get("snapshot") is not None:
                    snapshot = ExperienceSnapshot.model_validate(body["snapshot"])
                    if not state_sources(snapshot) <= allowed or digest(snapshot) != body["state_hash"]:
                        raise CheckpointIntegrityError("Candidate changed source provenance or snapshot identity")
                    body["snapshot"] = snapshot.model_dump(mode="json")
                existing = database.execute("SELECT body FROM candidates WHERE source=? AND run_id=? AND batch_index=? AND candidate_index=?",
                                            (source, run_id, batch_index, body["candidate_index"])).fetchone()
                if existing and json.loads(existing[0]) != body:
                    raise CheckpointIntegrityError("Cannot replace or resample a candidate")
                if not existing:
                    database.execute("INSERT INTO candidates VALUES(?,?,?,?,?)",
                                     (source, run_id, batch_index, body["candidate_index"], _encode(body)))
                if body.get("snapshot") is None:
                    self._mark_missing(database, source, run_id, reason="CandidateGenerationFailed",
                                       batch_index=batch_index, candidate_index=body["candidate_index"])

    def _ready(self, database, slot):
        source, run_id, kind = slot["source"], slot["run_id"], slot["kind"]
        if kind == "test":
            return super()._ready(database, slot)
        batch_index = slot["batch_index"]
        base = database.execute("SELECT state_hash FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                (source, run_id, batch_index * 5)).fetchone()
        if base is None:
            return None
        if kind == "evolution":
            return base[0]
        if kind == "batch_evolution":
            rows = database.execute("SELECT status FROM slots WHERE json_extract(body,'$.source')=? "
                "AND json_extract(body,'$.run_id')=? AND json_extract(body,'$.kind')='evolution' "
                "AND json_extract(body,'$.batch_index')=?", (source, run_id, batch_index)).fetchall()
            return base[0] if len(rows) == 5 and all(row["status"] in TERMINAL for row in rows) else None
        update = database.execute("SELECT status FROM slots WHERE slot_id=?",
                                  (f"update:{source}:run{run_id}:b{batch_index}",)).fetchone()
        if update is None or update["status"] not in TERMINAL:
            return None
        row = database.execute("SELECT json_extract(body,'$.state_hash') AS state_hash, "
                               "json_type(body,'$.snapshot') AS snapshot_type FROM candidates "
                               "WHERE source=? AND run_id=? AND batch_index=? AND candidate_index=?",
                               (source, run_id, batch_index, slot["candidate_index"])).fetchone()
        return row["state_hash"] if row and row["snapshot_type"] not in (None, "null") else None

    def claim(self, owner, *, phase="full", kinds=("evolution", "batch_evolution", "validation"), prefer="validation"):
        if phase not in {"first-stage", "full"} or not str(owner).strip():
            raise ValueError("Claim requires a valid phase and worker identity")
        if not kinds:
            return None
        with self._transaction() as database:
            placeholders = ",".join("?" for _ in kinds)
            phase_scope = (" AND json_extract(body,'$.run_id')=0 AND json_extract(body,'$.batch_index')=0"
                           if phase == "first-stage" else "")
            rows = database.execute("SELECT * FROM slots WHERE status='pending' "
                f"AND json_extract(body,'$.kind') IN ({placeholders})" + phase_scope + " ORDER BY ordinal",
                tuple(kinds)).fetchall()
            rows.sort(key=lambda row: json.loads(row["body"])["kind"] != prefer)
            for row in rows:
                slot = json.loads(row["body"])
                if slot["kind"] not in kinds:
                    continue
                if phase == "first-stage" and (slot["kind"] == "test" or slot["run_id"] != 0 or slot["batch_index"] != 0):
                    continue
                state_hash = self._ready(database, slot)
                if not state_hash:
                    continue
                token, started = uuid.uuid4().hex, utc_now()
                database.execute("UPDATE slots SET status='started',claim_token=?,owner=?,started_at=? WHERE slot_id=?",
                                 (token, owner, started, slot["slot_id"]))
                return {**slot, "status": "started", "claim_token": token, "owner": owner,
                        "started_at": started, "state_hash": state_hash, "registration_hash": self.registration_hash}
        return None

    def finish(self, claim, result, *, status="complete"):
        if claim["kind"] == "evolution" and (
                result.get("after_snapshot_hash") != claim["state_hash"] or result.get("experience_updates")):
            raise CheckpointIntegrityError("Five batch tasks must execute without persistent evolution")
        if claim["kind"] == "batch_evolution":
            with self._database() as database:
                candidates = database.execute(
                    "SELECT candidate_index,body FROM candidates WHERE source=? AND run_id=? AND batch_index=?",
                    (claim["source"], claim["run_id"], claim["batch_index"])).fetchall()
            if {row["candidate_index"] for row in candidates} != {0, 1, 2}:
                raise CheckpointIntegrityError("Batch update requires all three frozen candidates")
            if any(json.loads(row["body"]).get("base_state_hash") != claim["state_hash"]
                   for row in candidates):
                raise CheckpointIntegrityError("Batch candidates do not share the claimed input state")
        super().finish(claim, result, status=status)

    def _mark_missing(self, database, source, run_id, *, reason, batch_index=None, candidate_index=None):
        scope = (" AND json_extract(body,'$.batch_index')=? AND json_extract(body,'$.kind')='validation' "
                 "AND json_extract(body,'$.candidate_index')=?" if batch_index is not None else "")
        values = (source, run_id, batch_index, candidate_index) if batch_index is not None else (source, run_id)
        rows = database.execute("SELECT * FROM slots WHERE status='pending' "
            "AND json_extract(body,'$.source')=? AND json_extract(body,'$.run_id')=?" + scope, values).fetchall()
        for row in rows:
            slot = json.loads(row["body"])
            if slot["source"] != source or slot["run_id"] != run_id:
                continue
            if batch_index is not None and (slot.get("batch_index") != batch_index
                    or slot["kind"] != "validation" or slot.get("candidate_index") != candidate_index):
                continue
            result = {"task_id": slot["task_id"], "complete": False, "score": None,
                      "evaluation": None, "budget": None,
                      "error_type": "NoEligibleSelectedState" if slot["kind"] == "test" else reason,
                      "registration_hash": self.registration_hash}
            database.execute("UPDATE slots SET status='missing',result=?,result_hash=?,finished_at=? WHERE slot_id=?",
                             (_encode(result), digest(result), utc_now(), slot["slot_id"]))

    def advance(self):
        sources, _, runs, _ = _protocol_axes(self.protocol)
        for source in sources:
            for run_id in runs:
                final_selection = self.selection(source, run_id)
                if final_selection:
                    selected = final_selection.get("selected")
                    if selected:
                        candidate = self.candidate(source, run_id, final_selection["batch_index"], selected["candidate_index"])
                        self.freeze_checkpoint(source, run_id, selected["position"], candidate["snapshot"])
                    continue
                for batch_index in range(BATCHES_PER_TRAJECTORY):
                    with self._database() as database:
                        existing = database.execute("SELECT body FROM batch_selections WHERE source=? AND run_id=? AND batch_index=?",
                                                    (source, run_id, batch_index)).fetchone()
                    if existing:
                        selected = json.loads(existing[0]).get("selected")
                        if selected:
                            candidate = self.candidate(source, run_id, batch_index, selected["candidate_index"])
                            self.freeze_checkpoint(source, run_id, (batch_index + 1) * 5, candidate["snapshot"])
                        continue
                    validation = self.records(kind="validation", source=source, run_id=run_id, batch_index=batch_index)
                    if len(validation) != 30 or any(row["status"] not in TERMINAL for row in validation):
                        break
                    candidates = []
                    for candidate_index in range(3):
                        candidate = self.candidate(source, run_id, batch_index, candidate_index)
                        if candidate is None:
                            break
                        records = [{**row["result"], "status": row["status"]} for row in validation
                                   if row["candidate_index"] == candidate_index]
                        summary = validation_summary(records, self.trajectories[source, run_id]["validation_task_ids"], self.bounds)
                        candidates.append({"batch_index": batch_index, "candidate_index": candidate_index,
                                           "base_state_hash": candidate["base_state_hash"],
                                           "state_hash": candidate.get("state_hash") or digest({"failed_candidate": candidate["candidate_id"]}),
                                           **summary})
                    if len(candidates) != 3:
                        break
                    selected = select_batch_candidate(candidates)
                    body = {"source": source, "run_id": run_id, "batch_index": batch_index,
                            "status": "complete" if selected else "inconclusive", "selected": selected,
                            "candidates": candidates, "registration_hash": self.registration_hash}
                    final = selected is None or batch_index == BATCHES_PER_TRAJECTORY - 1
                    if selected:
                        body["selected"] = {**selected, "position": (batch_index + 1) * 5}
                    with self._transaction() as database:
                        database.execute("INSERT INTO batch_selections VALUES(?,?,?,?)", (source, run_id, batch_index, _encode(body)))
                        if final:
                            database.execute("INSERT INTO selections VALUES(?,?,?)", (source, run_id, _encode(body)))
                        if selected is None:
                            self._mark_missing(database, source, run_id, reason="NoEligibleBatchCandidate")
                    if selected:
                        candidate = self.candidate(source, run_id, batch_index, selected["candidate_index"])
                        self.freeze_checkpoint(source, run_id, (batch_index + 1) * 5, candidate["snapshot"])
                    if final:
                        break

    def select_trajectory(self, source, run_id):
        self.advance()
        selected = self.selection(source, run_id)
        if selected is None:
            raise ValueError("Final selection waits for batch eight or an inconclusive batch")
        return selected

    def status(self, *, phase="full"):
        records = self.records()
        if phase == "first-stage":
            records = [row for row in records if row["kind"] != "test" and row["run_id"] == 0 and row["batch_index"] == 0]
        counts = {}
        for row in records:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"phase": phase, "registered_slots": len(records), "statuses": counts,
                "terminal_slots": sum(count for status, count in counts.items() if status in TERMINAL),
                "registration_hash": self.registration_hash}
