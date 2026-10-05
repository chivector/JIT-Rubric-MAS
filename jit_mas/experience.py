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
        domain_terms = specific - operations
        matched_terms = terms.intersection(public_terms)
        if operations:
            matched = (operations <= public_terms
                       and domain_terms <= public_terms
                       and not operations.intersection(denied_operations))
        else:
            matched = bool((specific or terms).intersection(public_terms)) and specific <= public_terms
        grounded = grounded or bool(matched and specific.intersection(public_terms))
        diagnostics.append({"signal": signal, "matched": matched,
                            "matched_terms": sorted(matched_terms),
                            "required_operations": sorted(operations),
                            "required_domain_terms": sorted(domain_terms)})
    # A stated operation boundary must not disappear merely because format words match.
    applicability_terms = _terms(data.get("applicability", ""))
    applicability_specific = applicability_terms - _GENERAL_TERMS
    operations = applicability_specific.intersection(_TASK_OPERATIONS)
    domain_terms = applicability_specific - operations
    boundary_matches = (operations <= public_terms and domain_terms <= public_terms
                        and not operations.intersection(denied_operations))
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
    candidate = base.model_copy(deep=True)
    candidate.experiences = entries
    candidate.policy_versions = {**base.policy_versions, "experience_update": "direct-v1"}
    candidate.applied_proposals = base.applied_proposals + [proposal.proposal_id]
    return candidate


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
                    CREATE TABLE IF NOT EXISTS evolution_commits(id TEXT PRIMARY KEY, body TEXT NOT NULL,
                        version INTEGER NOT NULL);
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

    def commit_evolution(self, *, source_task_id, base_version, proposal=None, updates=(),
                         operations=(), observations=(), update_id=None):
        """Atomically persist both experience levels and dynamic Agent Pool decisions."""
        from .agent_pool import apply_evolution
        from .schemas import AgentEvolutionUpdate, AgentPoolObservation, AgentPoolOperation

        self._write_guard()
        if not isinstance(source_task_id, str) or not source_task_id.strip():
            raise ValueError("Evolution requires a nonempty source task ID")
        if type(base_version) is not int or base_version < 0:
            raise ValueError("Evolution requires a nonnegative base version")
        if proposal is not None:
            raw = proposal.model_dump(mode="json") if hasattr(proposal, "model_dump") else proposal
            proposal = ChangeProposal.model_validate(raw)
            if (proposal.source_task_id != source_task_id or proposal.base_version != base_version
                    or proposal.experience.source_task_ids != [source_task_id]):
                raise ValueError("Evolution proposal changed its source task or base version")
        normalized = []
        for update in (updates or ()):
            raw = update.model_dump(mode="json") if hasattr(update, "model_dump") else update
            normalized.append(AgentEvolutionUpdate.model_validate(raw))
        if any(update.source_task_id != source_task_id for update in normalized):
            raise ValueError("Agent update changed its source task")
        ids = [update.update_id for update in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate agent update IDs")
        normalized_operations = [AgentPoolOperation.model_validate(
            operation.model_dump(mode="json") if hasattr(operation, "model_dump") else operation)
            for operation in (operations or ())]
        normalized_observations = [AgentPoolObservation.model_validate(
            observation.model_dump(mode="json") if hasattr(observation, "model_dump") else observation)
            for observation in (observations or ())]
        if any(operation.source_task_id != source_task_id for operation in normalized_operations):
            raise ValueError("Agent Pool operation changed its source task")
        if any(observation.task_id != source_task_id for observation in normalized_observations):
            raise ValueError("Agent Pool observation changed its source task")
        if not normalized and not normalized_operations and not normalized_observations:
            if proposal is not None:
                return self.commit(proposal)
            current = self.snapshot()
            if current.version != base_version:
                raise ValueError("Stale evolution base version")
            return current
        body = {"source_task_id": source_task_id, "base_version": base_version,
                "proposal": proposal.model_dump(mode="json") if proposal is not None else None,
                "updates": [update.model_dump(mode="json") for update in normalized]}
        if normalized_operations:
            body["operations"] = [operation.model_dump(mode="json") for operation in normalized_operations]
        if normalized_observations:
            body["observations"] = [observation.model_dump(mode="json") for observation in normalized_observations]
        if update_id is None:
            update_id = digest(body)
        if not isinstance(update_id, str) or not update_id.strip():
            raise ValueError("Evolution requires a nonempty update ID")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            prior = self.db.execute("SELECT body,version FROM evolution_commits WHERE id=?",
                                    (update_id,)).fetchone()
            if prior:
                if digest(json.loads(prior[0])) != digest(body):
                    raise ValueError("Evolution update ID reused with different content")
                return self.snapshot(prior[1])
            base = self.snapshot()
            if base.version != base_version:
                raise ValueError("Stale evolution base version")
            if proposal is not None:
                stored = self.db.execute("SELECT body FROM proposals WHERE id=?",
                                         (proposal.proposal_id,)).fetchone()
                previous = (ChangeProposal.model_validate_json(stored[0]) if stored else
                            self._legacy_proposal(proposal.proposal_id))
                if previous is not None and digest(previous) != digest(proposal):
                    raise ValueError("Proposal ID reused with different content")
                committed = self.db.execute("SELECT version FROM commits WHERE proposal_id=?",
                                            (proposal.proposal_id,)).fetchone()
                if committed:
                    raise ValueError("Meta proposal already belongs to a committed update")
                candidate = candidate_snapshot(base, proposal)
            else:
                candidate = base.model_copy(deep=True)
            candidate.agent_pool = apply_evolution(base.agent_pool, normalized,
                normalized_operations, normalized_observations, source_task_id=source_task_id)
            candidate.policy_versions = {**candidate.policy_versions, "agent_pool": "dynamic-dual-evolution-v2"}
            version = int(self.db.execute("SELECT MAX(version) FROM snapshots").fetchone()[0]) + 1
            candidate.version = version
            self.db.execute("INSERT INTO snapshots VALUES(?,?)", (version, candidate.model_dump_json()))
            self.db.execute("UPDATE state SET value=? WHERE key='current'", (str(version),))
            if proposal is not None:
                self.db.execute("INSERT INTO proposals VALUES(?,?)",
                                (proposal.proposal_id, proposal.model_dump_json()))
                self.db.execute("INSERT INTO commits(proposal_id,version) VALUES(?,?)",
                                (proposal.proposal_id, version))
            self.db.execute("INSERT INTO evolution_commits VALUES(?,?,?)",
                            (update_id, json.dumps(body, allow_nan=False), version))
            self.db.execute("INSERT INTO audit(body) VALUES(?)", (json.dumps({
                "event": "dual_evolution_update", "update_id": update_id,
                "source_task_id": source_task_id, "update_hash": digest(body),
                "proposal_id": proposal.proposal_id if proposal is not None else None,
                "agent_update_ids": ids,
                "pool_operation_ids": [operation.operation_id for operation in normalized_operations],
                "pool_observation_count": len(normalized_observations),
                "before": base.version, "after": version,
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
    """Select relevant public-task advice, then preserve its chronological order.

    A recency-only cap lets new unscoped advice evict older task-grounded lessons.
    Rank eligible entries using the same public lexical evidence as the scope gate;
    scores, private rubrics and source answers do not participate in retrieval.
    """
    if limit < 0:
        raise ValueError("Experience retrieval limit must be nonnegative")
    if limit == 0:
        return []
    task = PublicTask.model_validate(task)
    excluded = set(excluded_task_ids) | {task.task_id}
    public_terms = _terms(" ".join([task.question, *task.constraints, *task.capabilities]))
    found = []
    for index, entry in enumerate(snapshot.experiences):
        if excluded.intersection(entry.source_task_ids):
            continue
        if before and entry.created_at >= before:
            continue
        diagnostic = experience_applicability(entry, task, capability=capability)
        if not diagnostic["matched"]:
            continue
        # Count distinct substantive matches, rather than repeated wording or the
        # length of an instruction. Applicable operations outrank generic formats.
        matched_terms = set().union(
            *(set(signal["matched_terms"]) for signal in diagnostic["signal_matches"])
        ) | (_terms(entry.applicability).intersection(public_terms))
        specific_matches = matched_terms - _GENERAL_TERMS
        rank = (int(diagnostic["task_grounded"]),
                len(specific_matches.intersection(_TASK_OPERATIONS)),
                len(specific_matches), index)
        found.append((rank, index, entry))
    selected = sorted(found, key=lambda item: item[0], reverse=True)[:limit]
    return [entry.model_dump(mode="json") for _, _, entry in
            sorted(selected, key=lambda item: item[1])]
