"""Review decisions are explicit and cannot authorize a changed harness."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jit.harness_ops import SECTION_TAG_TO_FILE
from jit_mas.bridge import SynthesizedHarness
from jit_mas.execution import content_hash
from jit_mas.review import FileReviewGate


class FileReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "generated"
        source.mkdir()
        self.files = {name: "# Reviewed fixture source\n" for name in SECTION_TAG_TO_FILE.values()}
        for name, value in self.files.items():
            (source / name).write_text(value, encoding="utf-8")
        (source / "__init__.py").write_text("", encoding="utf-8")
        self.sidecar = {"schema_version": "1.0", "task": {"question": "Public question"},
                        "team": {}, "rubrics": {}, "experiences": [], "backend": "native_jit"}
        (source / "team.json").write_text(json.dumps(self.sidecar), encoding="utf-8")
        self.artifact = SynthesizedHarness(
            name="generated", path=source, backend="native_jit",
            team_hash=content_hash({}), task_hash=content_hash(self.sidecar["task"]),
            code_hash=content_hash(self.files), sidecar_hash=content_hash(self.sidecar),
            meta_trajectory=[{"private_marker": "not_in_review_request"}],
        )
        self.gate = FileReviewGate(self.root / "reviews", timeout_seconds=0, poll_seconds=0.001)

    def request(self, artifact=None):
        artifact = artifact or self.artifact
        with self.assertRaises(TimeoutError):
            self.gate.wait(artifact)
        return json.loads(self.gate.request_path(artifact).read_text(encoding="utf-8"))

    def decide(self, request, decision="approve", artifact=None):
        path = self.gate.decision_path(artifact or self.artifact)
        path.write_text(json.dumps({"decision": decision, "review_key": request["review_key"],
                                   "binding": request["binding"]}), encoding="utf-8")

    def test_explicit_approval_and_reuse_reverify_integrity(self):
        request = self.request()
        self.decide(request)
        with patch.object(self.artifact, "verify_integrity", wraps=self.artifact.verify_integrity) as verify:
            self.assertIsNone(self.gate.wait(self.artifact))
            self.assertIsNone(self.gate.wait(self.artifact))
        self.assertEqual(verify.call_count, 4)

    def test_request_contains_public_sidecar_and_all_executable_files(self):
        request = self.request()
        self.assertEqual(request["public_sidecar"], self.sidecar)
        self.assertEqual(set(request["files"]), set(self.files) | {"__init__.py"})
        self.assertNotIn("not_in_review_request", json.dumps(request))
        self.assertIn("not sandboxing", request["warning"])

    def test_every_binding_field_must_match(self):
        request = self.request()
        for field in request["binding"]:
            with self.subTest(field=field):
                changed = copy.deepcopy(request)
                changed["binding"][field] = "wrong"
                self.decide(changed)
                with self.assertRaisesRegex(PermissionError, "does not match"):
                    self.gate.wait(self.artifact)

    def test_review_key_must_match(self):
        request = self.request()
        request["review_key"] = "wrong"
        self.decide(request)
        with self.assertRaisesRegex(PermissionError, "does not match"):
            self.gate.wait(self.artifact)

    def test_rejection_and_unknown_decision_fail_closed(self):
        request = self.request()
        for decision in ("reject", "approved", "", None):
            with self.subTest(decision=decision):
                self.decide(request, decision)
                with self.assertRaisesRegex(PermissionError, "not explicitly approved"):
                    self.gate.wait(self.artifact)

    def test_timeout_never_creates_a_decision(self):
        self.request()
        self.assertFalse(self.gate.decision_path(self.artifact).exists())

    def test_approved_code_changed_on_disk_is_rejected(self):
        request = self.request()
        self.decide(request)
        (self.artifact.path / "action.py").write_text("# Changed code\n", encoding="utf-8")
        with self.assertRaisesRegex(PermissionError, "integrity"):
            self.gate.wait(self.artifact)

    def test_code_changed_during_review_is_rejected_after_decision(self):
        self.gate.timeout_seconds = 1

        def approve_and_change(_):
            request = json.loads(self.gate.request_path(self.artifact).read_text(encoding="utf-8"))
            self.decide(request)
            (self.artifact.path / "action.py").write_text("# Changed after request\n", encoding="utf-8")

        with patch("jit_mas.review.time.sleep", side_effect=approve_and_change):
            with self.assertRaisesRegex(PermissionError, "integrity"):
                self.gate.wait(self.artifact)

    def test_repair_requires_new_hash_bound_review(self):
        request = self.request()
        self.decide(request)
        previous_path = self.gate.request_path(self.artifact)
        self.files["action.py"] = "# Repaired fixture\n"
        (self.artifact.path / "action.py").write_text(self.files["action.py"], encoding="utf-8")
        self.artifact.code_hash = content_hash(self.files)
        self.artifact.repair_count += 1
        repaired_request = self.request()
        self.assertNotEqual(self.gate.request_path(self.artifact), previous_path)
        self.assertNotEqual(request["review_key"], repaired_request["review_key"])
        self.decide(repaired_request)
        self.gate.wait(self.artifact)

    def test_initializer_change_requires_new_review(self):
        request = self.request()
        self.decide(request)
        (self.artifact.path / "__init__.py").write_text("# New initializer\n", encoding="utf-8")
        changed = self.request()
        self.assertNotEqual(changed["review_key"], request["review_key"])

    def test_initializer_changed_during_wait_is_rejected(self):
        self.gate.timeout_seconds = 1

        def approve_and_change(_):
            request = json.loads(self.gate.request_path(self.artifact).read_text(encoding="utf-8"))
            self.decide(request)
            (self.artifact.path / "__init__.py").write_text("# Changed initializer\n", encoding="utf-8")

        with patch("jit_mas.review.time.sleep", side_effect=approve_and_change):
            with self.assertRaisesRegex(PermissionError, "changed while awaiting review"):
                self.gate.wait(self.artifact)

    def test_unexpected_private_sidecar_field_is_not_written(self):
        self.sidecar["private_evaluation"] = {"secret": "private_marker"}
        (self.artifact.path / "team.json").write_text(json.dumps(self.sidecar), encoding="utf-8")
        self.artifact.sidecar_hash = content_hash(self.sidecar)
        with self.assertRaisesRegex(PermissionError, "only public"):
            self.gate.wait(self.artifact)
        self.assertFalse(self.gate.request_path(self.artifact).exists())

    def test_malformed_decision_is_rejected(self):
        self.request()
        for content in ("{", "[]", "x" * 65537):
            with self.subTest(content=content[:20]):
                self.gate.decision_path(self.artifact).write_text(content, encoding="utf-8")
                with self.assertRaises(PermissionError):
                    self.gate.wait(self.artifact)

    def test_invalid_wait_configuration_is_rejected(self):
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                FileReviewGate(self.root, timeout_seconds=value)
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(poll=value), self.assertRaises(ValueError):
                FileReviewGate(self.root, poll_seconds=value)


if __name__ == "__main__":
    unittest.main()
