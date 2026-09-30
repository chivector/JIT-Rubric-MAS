"""Transactional snapshots, staged proposals, and capability-based retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .schemas import (ChangeProposal, ExperienceSnapshot, PublicTask,
                      ValidationResult, digest, utc_now)


def capability_matches(scope: str, capability: str) -> bool:
    """Retrieve scoped advice across task-specific role names, not exact prose."""
    if not scope.strip():
        return True
    ignored = {"a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is",
               "it", "of", "on", "or", "the", "to", "with"}
    def terms(text):
        return set(re.findall(r"[^\W_]+", text.casefold())) - ignored
    return bool(terms(scope).intersection(terms(capability)))


def candidate_snapshot(base: ExperienceSnapshot, proposal: ChangeProposal) -> ExperienceSnapshot:
    if proposal.base_version != base.version:
        raise ValueError("Stale proposal base version")
    entries = [e.model_copy(deep=True) for e in base.experiences]
    ids = {e.experience_id for e in entries}
    if proposal.replaces_id:
        if proposal.replaces_id not in ids:
            raise ValueError("Replacement target does not exist")
        entries = [e for e in entries if e.experience_id != proposal.replaces_id]
    elif proposal.experience.experience_id in ids:
        raise ValueError("Experience ID already exists")
    if any(e.experience_id == proposal.experience.experience_id for e in entries):
        raise ValueError("Replacement collides with another experience ID")
    entry = proposal.experience.model_copy(deep=True)
    entry.validation_status = "accepted"
    entries.append(entry)
    return ExperienceSnapshot(version=base.version, experiences=entries,
                              policy_versions=base.policy_versions,
                              accepted_proposals=base.accepted_proposals + [proposal.proposal_id])


class ExperienceStore:
    def __init__(self, path, *, read_only=False):
        self.path = Path(path).resolve()
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            with self.db:
                self.db.executescript("""
                    CREATE TABLE IF NOT EXISTS snapshots(version INTEGER PRIMARY KEY, body TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS staging(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS validations(id TEXT PRIMARY KEY, proposal_id TEXT, body TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS commits(proposal_id TEXT PRIMARY KEY, version INTEGER, validation_id TEXT);
                    CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, body TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS task_runs(mode TEXT, task_id TEXT, identity TEXT,
                        baseline_version INTEGER, baseline_hash TEXT, status TEXT, body TEXT,
                        PRIMARY KEY(mode,task_id));
                """)
                self.db.execute("INSERT OR IGNORE INTO snapshots VALUES(0,?)",
                                (ExperienceSnapshot().model_dump_json(),))
                self.db.execute("INSERT OR IGNORE INTO state VALUES('current','0')")

    def close(self):
        self.db.close()

    def _write_guard(self):
        if self.read_only:
            raise PermissionError("Frozen/evaluate stores cannot update experience")

    def snapshot(self, version=None):
        if version is None:
            version = int(self.db.execute("SELECT value FROM state WHERE key='current'").fetchone()[0])
        row = self.db.execute("SELECT body FROM snapshots WHERE version=?", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown snapshot {version}")
        return ExperienceSnapshot.model_validate_json(row[0])

    def stage(self, proposal: ChangeProposal):
        self._write_guard()
        candidate_snapshot(self.snapshot(), proposal)
        with self.db:
            old = self.db.execute("SELECT body FROM staging WHERE id=?", (proposal.proposal_id,)).fetchone()
            if old and digest(json.loads(old[0])) != digest(proposal):
                raise ValueError("Proposal ID reused with different content")
            self.db.execute("INSERT OR IGNORE INTO staging VALUES(?,?)",
                            (proposal.proposal_id, proposal.model_dump_json()))

    def record_validation(self, result: ValidationResult):
        self._write_guard()
        row = self.db.execute("SELECT body FROM staging WHERE id=?", (result.proposal_id,)).fetchone()
        if not row or digest(json.loads(row[0])) != result.proposal_hash:
            raise ValueError("Validation does not identify an immutable staged proposal")
        with self.db:
            prior = self.db.execute("SELECT body FROM validations WHERE id=?", (result.validation_id,)).fetchone()
            if prior:
                old = ValidationResult.model_validate_json(prior[0])
                if old.model_dump(exclude={"created_at"}) != result.model_dump(exclude={"created_at"}):
                    raise ValueError("Validation ID reused with different content")
                return old
            self.db.execute("INSERT INTO validations VALUES(?,?,?)",
                            (result.validation_id, result.proposal_id, result.model_dump_json()))
        return result

    def commit(self, proposal: ChangeProposal, result: ValidationResult):
        self._write_guard()
        if result.status != "accepted" or not result.pairs:
            raise ValueError("Only accepted paired quality validation may be committed")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            stored = self.db.execute("SELECT body FROM validations WHERE id=?",
                                     (result.validation_id,)).fetchone()
            if not stored or digest(json.loads(stored[0])) != digest(result):
                raise ValueError("Validation must be recorded before commit")
            if result.proposal_hash != digest(proposal) or result.proposal_id != proposal.proposal_id:
                raise ValueError("Proposal content differs from validated content")
            prior = self.db.execute("SELECT version FROM commits WHERE proposal_id=?",
                                    (proposal.proposal_id,)).fetchone()
            if prior:
                return self.snapshot(prior[0])
            base = self.snapshot()
            candidate = candidate_snapshot(base, proposal)
            if (result.proposal_hash != digest(proposal) or result.baseline_hash != digest(base)
                    or result.candidate_hash != digest(candidate)):
                raise ValueError("Validation hashes do not match current and candidate states")
            version = int(self.db.execute("SELECT MAX(version) FROM snapshots").fetchone()[0]) + 1
            candidate.version = version
            self.db.execute("INSERT INTO snapshots VALUES(?,?)", (version, candidate.model_dump_json()))
            self.db.execute("UPDATE state SET value=? WHERE key='current'", (str(version),))
            self.db.execute("INSERT INTO commits VALUES(?,?,?)",
                            (proposal.proposal_id, version, result.validation_id))
            return candidate

    def rollback(self, version):
        self._write_guard()
        target = self.snapshot(version)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            before = self.snapshot().version
            self.db.execute("UPDATE state SET value=? WHERE key='current'", (str(version),))
            self.db.execute("INSERT INTO audit(body) VALUES(?)", (json.dumps({
                "event": "rollback", "before": before, "after": version, "time": utc_now()}),))
        return target

    def freeze(self, destination):
        target = Path(destination)
        if target.exists():
            raise FileExistsError("Refusing to overwrite a frozen snapshot")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.snapshot().model_dump_json(indent=2), encoding="utf-8")
        return target

    def task_run(self, mode, task_id):
        row = self.db.execute("SELECT identity,baseline_version,baseline_hash,status,body FROM task_runs "
                              "WHERE mode=? AND task_id=?", (mode, task_id)).fetchone()
        if not row:
            return None
        return {"identity": row[0], "baseline_version": row[1], "baseline_hash": row[2],
                "status": row[3], "outcome": json.loads(row[4]) if row[4] else None}

    def save_task_run(self, mode, task_id, identity, baseline, status, outcome=None):
        self._write_guard()
        with self.db:
            prior = self.task_run(mode, task_id)
            if prior and (prior["identity"] != identity or prior["baseline_hash"] != digest(baseline)):
                raise ValueError("Cannot reuse task journal with changed policy or baseline")
            self.db.execute("INSERT OR REPLACE INTO task_runs VALUES(?,?,?,?,?,?,?)",
                (mode, task_id, identity, baseline.version, digest(baseline), status,
                 json.dumps(outcome) if outcome is not None else None))


def retrieve(snapshot: ExperienceSnapshot, task: PublicTask, *, excluded_task_ids=(),
             before=None, capability=None, limit=24):
    """Small accepted bank; the model determines semantic applicability from scopes."""
    excluded = set(excluded_task_ids) | {task.task_id}
    found = []
    for entry in snapshot.experiences:
        if entry.validation_status != "accepted" or excluded.intersection(entry.source_task_ids):
            continue
        if before and entry.created_at >= before:
            continue
        if capability and entry.bank == "execution" and not capability_matches(entry.capability, capability):
            continue
        found.append(entry.model_dump(mode="json"))
    return found[-limit:]
