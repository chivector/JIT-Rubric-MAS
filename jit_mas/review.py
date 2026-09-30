"""Explicit, hash-bound review before unsafe-local execution, not a sandbox."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

from jit.harness_ops import SECTION_TAG_TO_FILE


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FileReviewGate:
    """Wait for an external decision bound to the exact generated artifact.

    A reviewer must inspect the source files and atomically write decision.json
    with ``decision``, ``review_key`` and the full ``binding`` from request.json.
    This gate does not make generated code safe or isolate unsafe-local code.
    """

    def __init__(self, output_dir, timeout_seconds=600, poll_seconds=0.2):
        self.output_dir = Path(output_dir).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds < 0:
            raise ValueError("review timeout must be finite and nonnegative")
        if not math.isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError("review polling interval must be finite and positive")

    @staticmethod
    def _binding(artifact):
        binding = {name: str(getattr(artifact, name)) for name in (
            "name", "backend", "code_hash", "task_hash", "team_hash", "sidecar_hash")}
        binding["path"] = str(Path(artifact.path).resolve())
        initializer = Path(artifact.path) / "__init__.py"
        # The package initializer is executable but not part of JIT's five-block hash.
        binding["package_init_hash"] = (
            hashlib.sha256(initializer.read_bytes()).hexdigest() if initializer.is_file() else None)
        return binding

    def request_path(self, artifact):
        return self.output_dir / _digest(self._binding(artifact)) / "request.json"

    def decision_path(self, artifact):
        return self.request_path(artifact).with_name("decision.json")

    @staticmethod
    def _verify(artifact):
        try:
            artifact.verify_integrity()
        except Exception as exc:
            raise PermissionError("generated artifact failed review integrity verification") from exc

    def wait(self, artifact):
        self._verify(artifact)
        binding = self._binding(artifact)
        review_key = _digest(binding)
        request_path = self.output_dir / review_key / "request.json"
        decision_path = request_path.with_name("decision.json")
        sidecar = artifact.sidecar
        allowed = {"schema_version", "task", "rubrics", "team", "experiences", "backend"}
        if not isinstance(sidecar, dict) or set(sidecar) - allowed:
            raise PermissionError("review sidecar must contain only public generation fields")
        files = {name: str((Path(artifact.path) / name).resolve())
                 for name in SECTION_TAG_TO_FILE.values()}
        initializer = Path(artifact.path) / "__init__.py"
        if initializer.is_file():
            files["__init__.py"] = str(initializer.resolve())
        _write_json(request_path, {
            "schema_version": "1.0", "review_key": review_key, "binding": binding,
            "warning": "Manual review gate only. Approval permits unsafe-local execution, not sandboxing.",
            "files": files, "public_sidecar": sidecar,
        })
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if decision_path.exists():
                try:
                    if decision_path.stat().st_size > 65536:
                        raise ValueError("review decision is too large")
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise PermissionError("invalid review decision; write decisions atomically") from exc
                if not isinstance(decision, dict):
                    raise PermissionError("review decision must be an object")
                if decision.get("review_key") != review_key or decision.get("binding") != binding:
                    raise PermissionError("review decision does not match the generated artifact hashes")
                if decision.get("decision") != "approve":
                    raise PermissionError("generated artifact was rejected or not explicitly approved")
                self._verify(artifact)
                current = self._binding(artifact)
                if current != binding:
                    raise PermissionError("generated artifact changed while awaiting review")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"review timed out waiting for {decision_path}")
            time.sleep(min(self.poll_seconds, remaining))
