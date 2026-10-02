"""Durable isolated work inventory for the independent evolution protocol."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .checkpoints import CheckpointIntegrityError
from .independent_protocol import (
    CHECKPOINT_POSITIONS, select_checkpoint, slot_registry, validate_protocol,
    validation_summary,
)
from .schemas import ExperienceSnapshot, digest, utc_now


TERMINAL = frozenset({"complete", "incomplete", "failed", "submitted", "missing"})


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


class IndependentCampaign:
    """Atomic claims, immutable checkpoint binding, and global test release."""

    def __init__(self, directory, protocol, identity, bounds):
        validate_protocol(protocol)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "campaign.sqlite"
        self.protocol = dict(protocol)
        self.identity = dict(identity)
        self.bounds = {task_id: list(values) for task_id, values in bounds.items()}
        self.trajectories = {
            (row["source"], row["run_id"]): row for row in protocol["trajectories"]
        }
        for trajectory in self.trajectories.values():
            for task_id in trajectory["validation_task_ids"]:
                if task_id not in self.bounds:
                    raise ValueError("Every registered VAL task requires frozen theoretical bounds")
        registration = {"protocol": self.protocol, "execution": self.identity,
                        "bounds": self.bounds, "schema": "independent-campaign-v5"}
        self.registration_hash = digest(registration)
        registry = slot_registry(protocol)
        with self._database() as database:
            database.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS slots(slot_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
                    body TEXT NOT NULL, status TEXT NOT NULL, claim_token TEXT, owner TEXT,
                    started_at TEXT, result TEXT, result_hash TEXT, finished_at TEXT);
                CREATE TABLE IF NOT EXISTS checkpoints(source TEXT, run_id INTEGER,
                    position INTEGER, state_hash TEXT NOT NULL, snapshot TEXT NOT NULL,
                    identity_hash TEXT NOT NULL, PRIMARY KEY(source,run_id,position));
                CREATE TABLE IF NOT EXISTS selections(source TEXT, run_id INTEGER,
                    body TEXT NOT NULL, PRIMARY KEY(source,run_id));
            """)
        with self._transaction() as database:
            existing = database.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()
            if existing:
                if json.loads(existing[0]) != registration:
                    raise CheckpointIntegrityError("Cannot resume a campaign with changed frozen inputs")
                recorded = database.execute("SELECT slot_id,body FROM slots ORDER BY ordinal").fetchall()
                if [json.loads(row[1]) for row in recorded] != registry["artifacts"]:
                    raise CheckpointIntegrityError("Registered campaign slot inventory changed")
            else:
                database.execute("INSERT INTO metadata VALUES('registration',?)", (_encode(registration),))
                database.executemany("INSERT INTO slots(slot_id,ordinal,body,status) VALUES(?,?,?,'pending')",
                    [(row["slot_id"], ordinal, _encode(row))
                     for ordinal, row in enumerate(registry["artifacts"])])

    @contextmanager
    def _database(self):
        database = sqlite3.connect(self.path, timeout=30)
        database.row_factory = sqlite3.Row
        try:
            yield database
        finally:
            database.close()

    @contextmanager
    def _transaction(self):
        with self._database() as database:
            database.execute("BEGIN IMMEDIATE")
            try:
                yield database
            except BaseException:
                database.rollback()
                raise
            else:
                database.commit()

    def freeze_checkpoint(self, source, run_id, position, snapshot):
        if (source, run_id) not in self.trajectories or position not in CHECKPOINT_POSITIONS:
            raise ValueError("Unregistered trajectory or checkpoint position")
        snapshot = ExperienceSnapshot.model_validate(snapshot)
        allowed = set(self.trajectories[source, run_id]["evolution_task_ids"][:position])
        source_ids = {task_id for item in snapshot.experiences for task_id in item.source_task_ids}
        source_ids.update(task_id for profile in snapshot.agent_pool.profiles
                          for task_id in profile.source_task_ids)
        if not source_ids <= allowed:
            raise CheckpointIntegrityError("Checkpoint includes experience from outside its source trajectory")
        state_hash = digest(snapshot)
        with self._transaction() as database:
            finished = self._evolution_rows(database, source, run_id)[:position]
            if position and (len(finished) != position or any(row["status"] not in TERMINAL for row in finished)):
                raise ValueError("Checkpoint requires all preceding EVO positions to terminate")
            existing = database.execute("SELECT * FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                        (source, run_id, position)).fetchone()
            if existing and (existing["state_hash"] != state_hash
                             or existing["identity_hash"] != self.registration_hash):
                raise CheckpointIntegrityError("Cannot replace a frozen complete checkpoint")
            if position and self._result(finished[-1]).get("after_snapshot_hash") != state_hash:
                raise CheckpointIntegrityError("Checkpoint differs from its last committed EVO outcome")
            if not existing:
                database.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
                    (source, run_id, position, state_hash,
                     _encode(snapshot.model_dump(mode="json")), self.registration_hash))
        return {"source": source, "run_id": run_id, "position": position, "state_hash": state_hash}

    def checkpoint(self, source, run_id, position):
        with self._database() as database:
            row = database.execute("SELECT * FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                   (source, run_id, position)).fetchone()
        if row is None:
            return None
        snapshot = ExperienceSnapshot.model_validate_json(row["snapshot"])
        if digest(snapshot) != row["state_hash"] or row["identity_hash"] != self.registration_hash:
            raise CheckpointIntegrityError("Frozen checkpoint identity or complete state changed")
        return {"source": source, "run_id": run_id, "position": position,
                "state_hash": row["state_hash"], "snapshot": snapshot}

    @staticmethod
    def _result(row):
        if row["result"] is None:
            return {}
        result = json.loads(row["result"])
        if digest(result) != row["result_hash"]:
            raise CheckpointIntegrityError("A terminal campaign outcome changed")
        return result

    @staticmethod
    def _evolution_rows(database, source, run_id):
        rows = database.execute("SELECT * FROM slots ORDER BY ordinal").fetchall()
        return [row for row in rows if (body := json.loads(row["body"]))["kind"] == "evolution"
                and body["source"] == source and body["run_id"] == run_id]

    def _ready(self, database, slot):
        source, run_id, kind = slot["source"], slot["run_id"], slot["kind"]
        if kind == "validation":
            row = database.execute("SELECT state_hash FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                   (source, run_id, slot["position"])).fetchone()
            return row[0] if row else None
        if kind == "evolution":
            order = self.trajectories[source, run_id]["evolution_task_ids"]
            index = order.index(slot["task_id"])
            rows = self._evolution_rows(database, source, run_id)
            if any(row["status"] not in TERMINAL for row in rows[:index]):
                return None
            preceding = index - index % 5
            checkpoint = database.execute("SELECT state_hash FROM checkpoints WHERE source=? AND run_id=? AND position=?",
                                          (source, run_id, preceding)).fetchone()
            if checkpoint is None:
                return None
            return self._result(rows[index - 1]).get("after_snapshot_hash") if index else checkpoint[0]
        if source == "static":
            return self.identity.get("initial_snapshot_hash")
        selection = database.execute("SELECT body FROM selections WHERE source=? AND run_id=?",
                                     (source, run_id)).fetchone()
        if not selection:
            return None
        return json.loads(selection[0]).get("selected", {}).get("state_hash")

    def claim(self, owner, *, phase="full", kinds=("evolution", "validation"), prefer="evolution"):
        if phase not in {"first-stage", "full"} or not str(owner).strip():
            raise ValueError("Claim requires a registered phase and nonempty worker identity")
        with self._transaction() as database:
            rows = database.execute("SELECT * FROM slots WHERE status='pending' ORDER BY ordinal").fetchall()
            rows.sort(key=lambda row: json.loads(row["body"])["kind"] != prefer)
            for row in rows:
                slot = json.loads(row["body"])
                if slot["kind"] not in kinds:
                    continue
                if phase == "first-stage":
                    if slot["kind"] == "test" or slot["run_id"] != 0:
                        continue
                    if slot["kind"] == "validation" and slot["position"] not in (0, 5):
                        continue
                    if slot["kind"] == "evolution" and slot["task_id"] not in self.trajectories[slot["source"], 0]["evolution_task_ids"][:5]:
                        continue
                state_hash = self._ready(database, slot)
                if not state_hash:
                    continue
                token, started = uuid.uuid4().hex, utc_now()
                database.execute("UPDATE slots SET status='started',claim_token=?,owner=?,started_at=? WHERE slot_id=?",
                                 (token, owner, started, slot["slot_id"]))
                return {**slot, "status": "started", "claim_token": token,
                        "owner": owner, "started_at": started, "state_hash": state_hash,
                        "registration_hash": self.registration_hash}
        return None

    def finish(self, claim, result, *, status="complete"):
        if status not in TERMINAL:
            raise ValueError("A committed result must have a terminal status")
        result = dict(result)
        slot_id = claim["slot_id"]
        with self._transaction() as database:
            row = database.execute("SELECT * FROM slots WHERE slot_id=?", (slot_id,)).fetchone()
            if row is None or row["claim_token"] != claim.get("claim_token"):
                raise ValueError("Result lacks the current exclusive slot claim")
            if row["status"] in TERMINAL:
                if row["status"] != status or self._result(row) != result:
                    raise CheckpointIntegrityError("A committed slot cannot be replaced or resampled")
                return
            if row["status"] != "started":
                raise ValueError("Only a started slot can commit a result")
            if result.get("task_id") != claim["task_id"]:
                raise ValueError("Committed result belongs to a different task")
            if result.get("state_hash") != claim["state_hash"]:
                raise CheckpointIntegrityError("Result differs from its frozen complete input state")
            if claim["kind"] == "validation" and result.get("experience_updates"):
                raise CheckpointIntegrityError("Validation must not write experience updates")
            if claim["kind"] == "test" and result.get("evaluation") is not None:
                raise CheckpointIntegrityError("TEST feedback cannot be committed before the global seal")
            if claim["kind"] == "test" and status not in {"submitted", "failed", "missing"}:
                raise ValueError("TEST generation must submit one unscored artifact or record failure")
            if claim["kind"] == "test" and status == "submitted" and not result.get("submission_path"):
                raise ValueError("Submitted TEST slot must bind its immutable submission path")
            if claim["kind"] == "evolution" and not result.get("after_snapshot_hash"):
                raise ValueError("EVO outcome must bind its committed complete state")
            if database.execute("SELECT 1 FROM metadata WHERE key='test_seal'").fetchone() and claim["kind"] == "test":
                raise ValueError("TEST submissions have been sealed")
            database.execute("UPDATE slots SET status=?,result=?,result_hash=?,finished_at=? WHERE slot_id=?",
                             (status, _encode(result), digest(result), utc_now(), slot_id))

    def submit_test(self, claim, submission_path, *, answer_hash, budget=None, metadata=None):
        """Commit one unscored TEST artifact; judging remains behind the global seal."""
        if claim.get("kind") != "test":
            raise ValueError("Only TEST claims accept submissions")
        path = Path(submission_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            submission = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("TEST submission must be a JSON artifact") from error
        if submission.get("answer_hash") != answer_hash or digest(submission.get("answer")) != answer_hash:
            raise CheckpointIntegrityError("TEST answer hash does not match its immutable submission")
        if not isinstance(submission.get("answer"), str) or not submission["answer"].strip() \
                or submission.get("evaluation") is not None:
            raise ValueError("TEST submission must contain one nonempty unscored answer")
        result = {"task_id": claim["task_id"], "state_hash": claim["state_hash"],
                  "submission_path": str(path), "submission_hash": digest(submission),
                  "answer_hash": answer_hash, "budget": budget, "evaluation": None,
                  "metadata": dict(metadata or {})}
        self.finish(claim, result, status="submitted")
        return result

    def interrupted_claims(self):
        with self._database() as database:
            rows = database.execute("SELECT * FROM slots WHERE status='started' ORDER BY ordinal").fetchall()
            claims = []
            for row in rows:
                body = json.loads(row["body"])
                state_hash = self._ready(database, body)
                claims.append({**body, "status": "started", "claim_token": row["claim_token"],
                               "owner": row["owner"], "started_at": row["started_at"],
                               "state_hash": state_hash, "registration_hash": self.registration_hash})
            return claims

    def records(self, *, kind=None, source=None, run_id=None):
        with self._database() as database:
            rows = database.execute("SELECT * FROM slots ORDER BY ordinal").fetchall()
        records = []
        for row in rows:
            body = json.loads(row["body"])
            if kind is not None and body["kind"] != kind:
                continue
            if source is not None and body["source"] != source:
                continue
            if run_id is not None and body["run_id"] != run_id:
                continue
            records.append({**body, "status": row["status"], "result": self._result(row),
                            "result_hash": row["result_hash"], "started_at": row["started_at"],
                            "finished_at": row["finished_at"]})
        return records

    def select_trajectory(self, source, run_id):
        trajectory = self.trajectories[source, run_id]
        evolution = self.records(kind="evolution", source=source, run_id=run_id)
        validation = self.records(kind="validation", source=source, run_id=run_id)
        if len(evolution) != 20 or len(validation) != 50 or any(
                row["status"] not in TERMINAL for row in evolution + validation):
            raise ValueError("Selection waits for all 20 EVO and all 50 VAL terminal slots")
        candidates = []
        for position in CHECKPOINT_POSITIONS:
            checkpoint = self.checkpoint(source, run_id, position)
            if checkpoint is None:
                raise CheckpointIntegrityError("Selection requires all five complete frozen states")
            rows = [row["result"] for row in validation if row["position"] == position]
            summary = validation_summary(rows, trajectory["validation_task_ids"], self.bounds)
            candidates.append({"position": position, "state_hash": checkpoint["state_hash"], **summary})
        selected = select_checkpoint(candidates)
        body = {"source": source, "run_id": run_id,
                "status": "complete" if selected else "inconclusive",
                "selected": selected, "candidates": candidates,
                "registration_hash": self.registration_hash}
        with self._transaction() as database:
            existing = database.execute("SELECT body FROM selections WHERE source=? AND run_id=?",
                                        (source, run_id)).fetchone()
            if existing and json.loads(existing[0]) != body:
                raise CheckpointIntegrityError("Selected source state changed after selection")
            if not existing:
                database.execute("INSERT INTO selections VALUES(?,?,?)", (source, run_id, _encode(body)))
            if selected is None:
                rows = database.execute("SELECT * FROM slots WHERE status='pending'").fetchall()
                for row in rows:
                    slot = json.loads(row["body"])
                    if slot["kind"] != "test" or slot["source"] != source or slot["run_id"] != run_id:
                        continue
                    result = {"task_id": slot["task_id"], "complete": False,
                              "error_type": "NoEligibleSelectedState", "evaluation": None,
                              "registration_hash": self.registration_hash, "budget": None}
                    database.execute("UPDATE slots SET status='missing',result=?,result_hash=?,finished_at=? WHERE slot_id=?",
                                     (_encode(result), digest(result), utc_now(), slot["slot_id"]))
        return body

    def selection(self, source, run_id):
        with self._database() as database:
            row = database.execute("SELECT body FROM selections WHERE source=? AND run_id=?",
                                   (source, run_id)).fetchone()
        return json.loads(row[0]) if row else None

    def seal_test_inventory(self):
        with self._transaction() as database:
            rows = database.execute("SELECT * FROM slots ORDER BY ordinal").fetchall()
            tests = [row for row in rows if json.loads(row["body"])["kind"] == "test"]
            if len(tests) != 2751 or any(row["status"] not in TERMINAL for row in tests):
                raise ValueError("All 2,751 TEST slots must submit or terminate before scoring")
            hashes = {}
            for row in tests:
                result = self._result(row)
                if row["status"] == "submitted":
                    submission = json.loads(Path(result["submission_path"]).read_text(encoding="utf-8"))
                    if digest(submission) != result.get("submission_hash") or digest(submission["answer"]) != submission["answer_hash"]:
                        raise CheckpointIntegrityError("TEST submission changed before sealing")
                hashes[row["slot_id"]] = row["result_hash"]
            seal = {"registration_hash": self.registration_hash, "required_test_slots": 2751,
                    "submission_hashes": hashes}
            existing = database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()
            if existing and json.loads(existing[0]) != seal:
                raise CheckpointIntegrityError("Globally sealed TEST inventory changed")
            if not existing:
                database.execute("INSERT INTO metadata VALUES('test_seal',?)", (_encode(seal),))
        return seal

    def require_test_seal(self):
        with self._database() as database:
            sealed = database.execute("SELECT body FROM metadata WHERE key='test_seal'").fetchone()
        if sealed is None:
            raise ValueError("TEST scoring waits for the global 2,751-slot submission seal")
        return self.seal_test_inventory()

    def status(self, *, phase="full"):
        records = self.records()
        if phase == "first-stage":
            records = [row for row in records if row["run_id"] == 0 and (
                row["kind"] == "validation" and row["position"] in (0, 5)
                or row["kind"] == "evolution" and row["task_id"] in
                self.trajectories[row["source"], 0]["evolution_task_ids"][:5])]
        counts = {}
        for row in records:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"phase": phase, "registered_slots": len(records), "statuses": counts,
                "terminal_slots": sum(count for status, count in counts.items() if status in TERMINAL),
                "registration_hash": self.registration_hash}


@contextmanager
def coordinator_lock(directory):
    """A crashed coordinator releases its process lock without expiring live tasks."""
    path = Path(directory) / "coordinator.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another campaign coordinator owns this directory") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
