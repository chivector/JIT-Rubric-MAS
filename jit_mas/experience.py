"""Transactional direct experience updates and capability-based retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .schemas import ChangeProposal, ExperienceSnapshot, PublicTask, digest, utc_now


_STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "the", "to", "with", "any", "all", "that", "this", "when",
    "where", "which", "task", "tasks", "asks", "requires", "requiring", "involves",
}
_GENERAL_TERMS = {
    "article", "articles", "write", "technical", "clear", "accurate", "structured",
    "format", "content", "response", "answer", "quality", "general", "assistant",
}
_TERM_FAMILIES = (
    ("compare", "comparison", "comparisons", "comparative", "comparing", "compares",
     "contrast", "contrasting", "contrasts", "versus", "vs"),
    ("verify", "verification", "verifying", "verified", "validate", "validation"),
    ("explain", "explanation", "explanations", "explanatory", "explaining"),
    ("write", "writing", "writer", "writers", "written"),
    ("derive", "derivation", "derivations", "deriving"),
    ("prove", "proof", "proofs", "proving"),
    ("review", "reviewer", "reviewing"),
    ("synthesize", "synthesis", "synthesizer"),
    ("math", "mathematics", "mathematical"),
    ("algorithm", "algorithms"),
)
_NORMALIZED_TERMS = {word: family[0] for family in _TERM_FAMILIES for word in family}
_TASK_OPERATIONS = {"compare", "derive", "prove", "translate", "diagnose", "implement"}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", text.casefold())
    return {_NORMALIZED_TERMS.get(word, word) for word in words} - _STOP_WORDS


def _negated_operations(text: str) -> set[str]:
    words = [_NORMALIZED_TERMS.get(word, word)
             for word in re.findall(r"[^\W_]+", text.casefold())]
    negators = {"not", "never", "without", "avoid", "no", "except"}
    return {word for index, word in enumerate(words) if word in _TASK_OPERATIONS
            and negators.intersection(words[max(0, index - 3):index])}


def capability_matches(scope: str, capability: str) -> bool:
    """Match substantive capabilities; shared writing/format words are insufficient."""
    if not scope.strip():
        return True
    source, target = _terms(scope), _terms(capability)
    specific = source - _GENERAL_TERMS
    if not source.intersection(_TASK_OPERATIONS) <= target:
        return False
    if specific:
        return bool(specific.intersection(target - _GENERAL_TERMS))
    return bool(source.intersection(target, {"write"}))


def experience_applicability(entry, task: PublicTask, *, capability=None) -> dict:
    """Conservative public-signal gate, with inspectable lexical evidence, not an LLM judge.

    Signals and explicit operation boundaries are necessary public conditions.
    Unscoped legacy entries cannot widen their capability scope using generic words.
    """
    data = entry.model_dump(mode="json") if hasattr(entry, "model_dump") else entry
    task = PublicTask.model_validate(task)
    public_text = " ".join([task.question, *task.constraints, *task.capabilities])
    public_terms = _terms(public_text)
    denied_operations = _negated_operations(public_text)
    diagnostics = []
    grounded = False
    for signal in data.get("task_signals", []):
        terms = _terms(signal)
        specific = terms - _GENERAL_TERMS
        operations = specific.intersection(_TASK_OPERATIONS)
        matched_terms = terms.intersection(public_terms)
        if operations:
            matched = operations <= public_terms and not operations.intersection(denied_operations)
        else:
            matched = bool((specific or terms).intersection(public_terms))
        grounded = grounded or bool(matched and specific.intersection(public_terms))
        diagnostics.append({"signal": signal, "matched": matched,
                            "matched_terms": sorted(matched_terms),
                            "required_operations": sorted(operations)})
    # A stated operation boundary must not disappear merely because format words match.
    operations = _terms(data.get("applicability", "")).intersection(_TASK_OPERATIONS)
    boundary_matches = (not operations or operations <= public_terms) and not operations.intersection(denied_operations)
    matched = all(item["matched"] for item in diagnostics) and boundary_matches
    grounded = grounded or bool(operations and boundary_matches)
    reason = ("public_signals_match" if diagnostics else
              "applicability_operation_match" if operations else "legacy_task_scope_unknown")
    if not matched:
        reason = "public_task_scope_mismatch"
    if matched and capability is not None and data.get("bank") == "execution":
        scope = data.get("capability", "")
        strong_match = capability_matches(scope, capability)
        functional_match = bool(_terms(scope).intersection(_terms(capability), {"write"}))
        matched = strong_match or (grounded and functional_match)
        if not matched:
            reason = "capability_mismatch"
        elif not strong_match:
            reason = "public_task_and_role_function_match"
    return {"matched": matched, "reason": reason, "task_grounded": grounded,
            "signal_matches": diagnostics, "applicability_operations": sorted(operations),
            "applicability_boundary_matches": bool(boundary_matches),
            "denied_operations": sorted(denied_operations)}


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
    entries.append(entry)
    return ExperienceSnapshot(version=base.version, experiences=entries,
                              policy_versions={**base.policy_versions, "experience_update": "direct-v1"},
                              applied_proposals=base.applied_proposals + [proposal.proposal_id])


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
                    CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS commits(proposal_id TEXT PRIMARY KEY, version INTEGER);
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

    def _legacy_proposal(self, proposal_id):
        table = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='staging'").fetchone()
        if not table:
            return None
        row = self.db.execute("SELECT body FROM staging WHERE id=?", (proposal_id,)).fetchone()
        return ChangeProposal.model_validate_json(row[0]) if row else None

    def commit(self, proposal: ChangeProposal):
        """Apply one attributed update without paired promotion or quality claims."""
        self._write_guard()
        proposal = ChangeProposal.model_validate(proposal.model_dump(mode="json"))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            stored = self.db.execute("SELECT body FROM proposals WHERE id=?",
                                     (proposal.proposal_id,)).fetchone()
            previous = ChangeProposal.model_validate_json(stored[0]) if stored else self._legacy_proposal(proposal.proposal_id)
            if previous is not None and digest(previous) != digest(proposal):
                raise ValueError("Proposal ID reused with different content")
            prior = self.db.execute("SELECT version FROM commits WHERE proposal_id=?",
                                    (proposal.proposal_id,)).fetchone()
            if prior:
                if previous is None:
                    raise ValueError("Cannot verify immutable historical proposal content")
                return self.snapshot(prior[0])
            base = self.snapshot()
            candidate = candidate_snapshot(base, proposal)
            version = int(self.db.execute("SELECT MAX(version) FROM snapshots").fetchone()[0]) + 1
            candidate.version = version
            self.db.execute("INSERT INTO proposals VALUES(?,?)",
                            (proposal.proposal_id, proposal.model_dump_json()))
            self.db.execute("INSERT INTO snapshots VALUES(?,?)", (version, candidate.model_dump_json()))
            self.db.execute("UPDATE state SET value=? WHERE key='current'", (str(version),))
            self.db.execute("INSERT INTO commits(proposal_id,version) VALUES(?,?)",
                            (proposal.proposal_id, version))
            self.db.execute("INSERT INTO audit(body) VALUES(?)", (json.dumps({
                "event": "direct_update", "proposal_id": proposal.proposal_id,
                "proposal_hash": digest(proposal), "before": base.version, "after": version,
                "baseline_hash": digest(base), "updated_hash": digest(candidate), "time": utc_now()}),))
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
    """Filter public task signals before exposing a small stored bank to the model."""
    excluded = set(excluded_task_ids) | {task.task_id}
    found = []
    for entry in snapshot.experiences:
        if excluded.intersection(entry.source_task_ids):
            continue
        if before and entry.created_at >= before:
            continue
        if not experience_applicability(entry, task, capability=capability)["matched"]:
            continue
        found.append(entry.model_dump(mode="json"))
    return found[-limit:]
