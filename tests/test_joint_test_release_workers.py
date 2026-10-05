"""Concurrency and idempotency checks for the frozen TEST submit phase."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from jit_mas.pipeline import write_json
from scripts.run_joint_test_release import JointTestReleaseRunner


class _FakeRelease:
    """Small durable TestRelease double for submit scheduling tests."""

    def __init__(self, root: Path, count: int = 6):
        self.root = root
        self.slots = {
            f"slot-{index}": {"slot_id": f"slot-{index}", "task_id": f"task-{index}",
                               "method": "ours_selected"}
            for index in range(count)
        }

    def _path(self, slot_id, kind="submissions"):
        return self.root / kind / f"{slot_id}.json"

    def record(self, slot_id, outcome):
        write_json(self._path(slot_id), {"status": "submitted", "outcome": outcome})

    def record_failure(self, slot_id, *, error_type, budget):
        write_json(self._path(slot_id), {"status": "failed", "error_type": error_type,
                                         "budget": budget})


def _runner(tmp_path, release, submit_one):
    runner = object.__new__(JointTestReleaseRunner)
    runner.campaign = tmp_path
    runner._shared_rstar_lock = threading.Lock()
    runner._check_release_inputs = lambda: None
    runner._release = lambda: release
    runner._submit_one = submit_one
    return runner


def test_submit_workers_parallelize_unique_slots(tmp_path):
    release = _FakeRelease(tmp_path)
    barrier = threading.Barrier(3)
    active = 0
    peak = 0
    guard = threading.Lock()

    def submit_one(slot, *, shared_dir):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=2)
            time.sleep(0.01)
            return {"task_id": slot["task_id"], "status": "submitted_unscored",
                    "evaluation": None, "proposals": [], "run_dir": str(tmp_path),
                    "answer_hash": slot["slot_id"], "experience_hash": "state"}
        finally:
            with guard:
                active -= 1

    runner = _runner(tmp_path, release, submit_one)
    result = runner.submit(workers=3)
    assert result["submitted_or_preexisting"] == len(release.slots)
    assert result["failed"] == 0
    assert peak >= 3


def test_default_submit_is_serial_and_reuses_durable_records(tmp_path):
    release = _FakeRelease(tmp_path, count=3)
    calls = []

    def submit_one(slot, *, shared_dir):
        calls.append(slot["slot_id"])
        return {"task_id": slot["task_id"], "status": "submitted_unscored",
                "evaluation": None, "proposals": [], "run_dir": str(tmp_path),
                "answer_hash": slot["slot_id"], "experience_hash": "state"}

    runner = _runner(tmp_path, release, submit_one)
    first = runner.submit()
    second = runner.submit()
    assert first["submitted_or_preexisting"] == second["submitted_or_preexisting"] == 3
    assert calls == ["slot-0", "slot-1", "slot-2"]


def test_unresolved_started_marker_is_checked_before_dispatch(tmp_path):
    release = _FakeRelease(tmp_path, count=3)
    started = release._path("slot-1", "submission_started")
    write_json(started, {"slot_hash": "frozen", "started_at": "now"})
    calls = []

    def submit_one(slot, *, shared_dir):
        calls.append(slot["slot_id"])
        raise AssertionError("no new paid request should be dispatched")

    runner = _runner(tmp_path, release, submit_one)
    with pytest.raises(RuntimeError, match="Interrupted TEST slot"):
        runner.submit(workers=3)
    assert calls == []


def test_resolved_started_marker_is_recorded_as_failure(tmp_path):
    release = _FakeRelease(tmp_path, count=1)
    write_json(release._path("slot-0", "submission_started"), {"slot_hash": "frozen"})
    runner = _runner(tmp_path, release, lambda *args, **kwargs: pytest.fail("must not resample"))
    result = runner.submit(workers=2, resolve_interrupted="operator inspected")
    assert result["submitted_or_preexisting"] == 0
    assert result["failed"] == 1
    assert json.loads(release._path("slot-0").read_text())["status"] == "failed"


def test_submit_workers_reject_invalid_values(tmp_path):
    release = _FakeRelease(tmp_path, count=1)
    runner = _runner(tmp_path, release, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="workers"):
        runner.submit(workers=0)
    with pytest.raises(ValueError, match="workers"):
        runner.submit(workers=17)
    with pytest.raises(ValueError, match="workers"):
        runner.submit(workers=True)
