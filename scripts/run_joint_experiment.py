"""Offline controller for the frozen joint benchmark v5 protocol.

This runner owns registration, immutable slot accounting, VAL selection and the
TEST seal barrier. Model providers are deliberately outside this module: an
executor may claim and finish one slot, but this controller never resamples an
interrupted slot and never exposes TEST feedback before every submission is
sealed. ``register``/``status``/``check`` are therefore safe to run without an
API key.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import secrets
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("researchrubrics", "deepsearchqa", "writingbench")
BENCHMARKS = SOURCES + ("deepresearch_bench_ii", "ifeval", "ifbench")
STATIC_METHODS = ("ours_initial", "direct", "jit_matched", "rubric_fixed")
RUN_IDS = (0, 1, 2)
CHECKPOINTS = (0, 15, 30, 45, 60)
TEST_METHODS = ("ours_selected",) + STATIC_METHODS
EXPECTED = {
    "evolution": 180, "validation": 450, "test": 1911, "total": 2541,
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _without_hash(document: Mapping[str, Any], *names: str) -> dict[str, Any]:
    excluded = set(names) | {"manifest_sha256", "protocol_sha256"}
    return {key: value for key, value in document.items() if key not in excluded}


def _status(status: str) -> bool:
    return status in {"complete", "failed", "submitted", "missing"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def normalize_score(score: Any, bounds: Iterable[float]) -> float:
    if not _finite(score):
        raise ValueError("A complete VAL result requires a finite native score")
    lower, upper = (float(item) for item in bounds)
    if not _finite(lower) or not _finite(upper) or upper < lower or score < lower - 1e-12 or score > upper + 1e-12:
        raise ValueError("Score is outside the pre-registered theoretical bounds")
    return 0.0 if upper == lower else max(0.0, min(1.0, (float(score) - lower) / (upper - lower)))


def _stage_order(manifest: Mapping[str, Any], run_id: int) -> list[str]:
    schedules = manifest.get("joint_evolution_schedule")
    if not isinstance(schedules, list) or run_id >= len(schedules):
        raise ValueError("v5 manifest has no frozen schedule for this run")
    stages = schedules[run_id].get("stages")
    if not isinstance(stages, list) or len(stages) != 4 or any(not isinstance(stage, list) or len(stage) != 15 for stage in stages):
        raise ValueError("Each v5 run requires four mixed 15-task EVO stages")
    order = [task_id for stage in stages for task_id in stage]
    if len(order) != 60 or len(set(order)) != 60:
        raise ValueError("Each frozen run order must contain 60 unique EVO tasks")
    return order


def _memberships(manifest: Mapping[str, Any], benchmark: str, partition: str, count: int) -> list[str]:
    values = manifest.get("memberships", {}).get(benchmark, {}).get(partition)
    if not isinstance(values, list) or len(values) != count or len(set(values)) != count:
        raise ValueError(f"Frozen {benchmark} {partition} membership must contain {count} unique IDs")
    return list(values)


def build_inventory(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Construct the complete 2,541 logical slots without generating answers."""
    slots: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        order = _stage_order(manifest, run_id)
        for ordinal, task_id in enumerate(order):
            slots.append({"slot_id": f"evo:run{run_id}:{ordinal:02d}:{task_id}", "kind": "evolution", "run_id": run_id, "ordinal": ordinal, "task_id": task_id})
        for checkpoint in CHECKPOINTS:
            for benchmark in SOURCES:
                for task_id in _memberships(manifest, benchmark, "validation", 10):
                    slots.append({"slot_id": f"val:run{run_id}:c{checkpoint}:{benchmark}:{task_id}", "kind": "validation", "run_id": run_id, "checkpoint": checkpoint, "benchmark": benchmark, "task_id": task_id})
    for method in TEST_METHODS:
        for benchmark in BENCHMARKS:
            count = int(manifest["counts"][benchmark]["test"])
            for task_id in _memberships(manifest, benchmark, "test", count):
                slots.append({"slot_id": f"test:{method}:{benchmark}:{task_id}", "kind": "test", "method": method, "benchmark": benchmark, "task_id": task_id})
    # The first pass contains one selected state plus four static methods.
    # The selected state is replaced below by the three per-run states.
    # Selected is one artifact per benchmark/task for each of the three runs.
    selected: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        for benchmark in BENCHMARKS:
            count = int(manifest["counts"][benchmark]["test"])
            for task_id in _memberships(manifest, benchmark, "test", count):
                selected.append({"slot_id": f"test:ours_selected:run{run_id}:{benchmark}:{task_id}", "kind": "test", "method": "ours_selected", "run_id": run_id, "benchmark": benchmark, "task_id": task_id})
    # Replace the single method's 273 rows with the three selected run rows.
    slots = [row for row in slots if row.get("method") != "ours_selected"] + selected
    counts = defaultdict(int)
    for row in slots:
        counts[row["kind"]] += 1
    if counts != {"evolution": 180, "validation": 450, "test": 1911}:
        raise ValueError(f"v5 slot inventory mismatch: {dict(counts)}")
    if len({row["slot_id"] for row in slots}) != EXPECTED["total"]:
        raise ValueError("v5 slot IDs are not unique")
    return slots


class JointCampaign:
    """Durable no-resampling controller for one frozen v5 registration."""

    def __init__(self, output: Path, protocol: Path, manifest: Path, *, actor_config: Path | None = None, judge_config: Path | None = None, code_paths: Iterable[Path] = (), evidence_hashes: Mapping[str, str] | None = None):
        self.output, self.protocol_path, self.manifest_path = Path(output).resolve(), Path(protocol).resolve(), Path(manifest).resolve()
        self.actor_config_path = Path(actor_config).resolve() if actor_config else None
        self.judge_config_path = Path(judge_config).resolve() if judge_config else None
        self.code_paths = [Path(path).resolve() for path in code_paths]
        self.protocol, self.manifest = read_json(self.protocol_path), read_json(self.manifest_path)
        self._validate_protocol()
        self.slots = build_inventory(self.manifest)
        self.slot_by_id = {row["slot_id"]: row for row in self.slots}
        self.registration = {
            "schema_version": "joint-experiment-registration-v5", "protocol_version": self.protocol["version"],
            "protocol_sha256": self.protocol.get("protocol_sha256"), "manifest_sha256": self.manifest.get("manifest_sha256"),
            "protocol_file_sha256": file_hash(self.protocol_path), "manifest_file_sha256": file_hash(self.manifest_path),
            "actor_config_sha256": file_hash(actor_config) if actor_config else None, "judge_config_sha256": file_hash(judge_config) if judge_config else None,
            "code_sha256": [file_hash(path) for path in code_paths], "evidence_hashes": dict(evidence_hashes or {}),
            "preflight": {"actor_config_bound": actor_config is not None, "judge_config_bound": judge_config is not None, "code_bound": bool(self.code_paths), "public_evidence_bound": set(evidence_hashes or {}) >= set(SOURCES), "provider_ready": bool(actor_config and judge_config and self.code_paths and set(evidence_hashes or {}) >= set(SOURCES))},
            "test_feedback_updates_experience": False, "test_resampling": False, "selection_uses_test": False,
            "slots": self.slots, "slot_count": EXPECTED["total"], "frozen": True,
        }
        self.registration["registration_sha256"] = digest(self.registration)

    def _validate_protocol(self) -> None:
        if self.protocol.get("version") != "jit-compose-joint-protocol-v5" or self.manifest.get("version") != "jit-compose-joint-subset-v5":
            raise ValueError("Only the frozen joint v5 protocol is accepted")
        if self.protocol.get("status") != "SUBSET_PROTOCOL_FROZEN_NOT_RUN" or self.manifest.get("status") != "SUBSET_PROTOCOL_FROZEN_NOT_RUN":
            raise ValueError("Protocol or manifest is not the frozen pre-run registration")
        if self.protocol.get("split_manifest_sha256") != self.manifest.get("manifest_sha256"):
            raise ValueError("Protocol and split manifest hashes differ")
        if digest(_without_hash(self.manifest)) != self.manifest.get("manifest_sha256"):
            raise ValueError("Joint split manifest self-hash mismatch")
        if digest(_without_hash(self.protocol)) != self.protocol.get("protocol_sha256"):
            raise ValueError("Joint protocol self-hash mismatch")
        if self.protocol.get("workload", {}).get("total_task_slots") != EXPECTED["total"]:
            raise ValueError("Formal v5 workload is not 2,541 slots")
        if self.protocol.get("workload", {}).get("test_slots") != EXPECTED["test"]:
            raise ValueError("Formal v5 TEST workload is not 1,911 slots")

    @property
    def registration_path(self) -> Path:
        return self.output / "registration.json"

    @property
    def inventory_path(self) -> Path:
        return self.output / "slot_inventory.json"

    @property
    def state_path(self) -> Path:
        return self.output / "slot_state.json"

    def register(self) -> dict[str, Any]:
        self.assert_frozen()
        self.output.mkdir(parents=True, exist_ok=True)
        if self.registration_path.exists() and read_json(self.registration_path) != self.registration:
            raise ValueError("Frozen registration already exists with different identity")
        if self.inventory_path.exists() and read_json(self.inventory_path) != {"slots": self.slots, "inventory_sha256": digest(self.slots)}:
            raise ValueError("Frozen slot inventory already exists with different membership")
        state = {row["slot_id"]: {"status": "pending", "attempts": 0, "result": None} for row in self.slots}
        if self.state_path.exists() and read_json(self.state_path) != state:
            # A resume preserves existing state; registration never resets it.
            existing = read_json(self.state_path)
            if set(existing) != set(state):
                raise ValueError("Existing slot state does not match frozen inventory")
        else:
            write_json(self.state_path, state)
        write_json(self.registration_path, self.registration)
        write_json(self.inventory_path, {"slots": self.slots, "inventory_sha256": digest(self.slots)})
        return self.status()

    def assert_frozen(self) -> None:
        """Fail closed if registered protocol/config/code identities changed."""
        if not self.registration_path.is_file():
            return
        frozen = read_json(self.registration_path)
        if self.protocol_path.is_file() and file_hash(self.protocol_path) != frozen.get("protocol_file_sha256"):
            raise ValueError("Frozen protocol file changed")
        if self.manifest_path.is_file() and file_hash(self.manifest_path) != frozen.get("manifest_file_sha256"):
            raise ValueError("Frozen split manifest changed")
        for path, expected in ((self.actor_config_path, frozen.get("actor_config_sha256")), (self.judge_config_path, frozen.get("judge_config_sha256"))):
            if path is not None and file_hash(path) != expected:
                raise ValueError("Frozen actor/judge configuration changed")
        for path, expected in zip(self.code_paths, frozen.get("code_sha256", [])):
            if file_hash(path) != expected:
                raise ValueError("Frozen runner/code identity changed")

    def _state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise ValueError("Campaign is not registered")
        state = read_json(self.state_path)
        if set(state) != set(self.slot_by_id):
            raise ValueError("Slot state does not match frozen inventory")
        return state

    def claim(self, slot_id: str, owner: str = "executor") -> dict[str, Any]:
        self.assert_frozen()
        state = self._state()
        if slot_id not in self.slot_by_id:
            raise KeyError(slot_id)
        row = state[slot_id]
        if row["status"] != "pending":
            raise ValueError("A started or terminal slot cannot be resampled")
        row.update({"status": "started", "owner": owner, "claim_token": secrets.token_hex(16), "attempts": 1})
        write_json(self.state_path, state)
        return {**self.slot_by_id[slot_id], **row}

    def finish(self, claim: Mapping[str, Any], result: Mapping[str, Any], *, status: str) -> None:
        self.assert_frozen()
        state = self._state(); slot_id = claim["slot_id"]
        if state[slot_id].get("claim_token") != claim.get("claim_token") or state[slot_id]["status"] != "started":
            raise ValueError("Only the current exclusive claim can finish")
        if not _status(status):
            raise ValueError("Result status must be terminal")
        body = dict(result)
        if body.get("slot_id", slot_id) != slot_id:
            raise ValueError("Result slot identity mismatch")
        if self.slot_by_id[slot_id]["kind"] == "test" and status == "submitted" and body.get("score") is not None:
            raise ValueError("TEST generation is unscored before the global seal")
        if self.slot_by_id[slot_id]["kind"] == "test" and status == "submitted" and not isinstance(body.get("submission_hash"), str):
            raise ValueError("Submitted TEST slot must bind an immutable submission hash")
        state[slot_id] = {"status": status, "attempts": 1, "result": body, "result_sha256": digest(body)}
        write_json(self.state_path, state)

    def validation_selection(self, run_id: int, bounds: Mapping[str, Iterable[float]]) -> dict[str, Any]:
        self.assert_frozen()
        if run_id not in RUN_IDS: raise ValueError("Unknown run")
        state = self._state(); candidates = []
        for checkpoint in CHECKPOINTS:
            by_benchmark = {}
            all_complete = True
            for benchmark in SOURCES:
                rows = [self.slot_by_id[sid] | {"state": state[sid]} for sid in state if self.slot_by_id[sid].get("kind") == "validation" and self.slot_by_id[sid].get("run_id") == run_id and self.slot_by_id[sid].get("checkpoint") == checkpoint and self.slot_by_id[sid].get("benchmark") == benchmark]
                complete = [row for row in rows if row["state"]["status"] == "complete" and isinstance(row["state"].get("result"), dict) and row["state"]["result"].get("complete") is True]
                all_complete = all_complete and len(complete) >= 9 and len(rows) == 10 and all(row["state"]["status"] in {"complete", "failed", "missing", "incomplete"} for row in rows)
                values = []
                for row in complete:
                    task_id = row["task_id"]
                    if task_id not in bounds: raise ValueError(f"Missing theoretical bound for {task_id}")
                    values.append(normalize_score(row["state"]["result"].get("score"), bounds[task_id]))
                by_benchmark[benchmark] = {"complete": len(complete), "slots": len(rows), "mean_normalized": sum(values) / 10.0 if len(rows) == 10 else None}
            utility = sum(item["mean_normalized"] for item in by_benchmark.values() if item["mean_normalized"] is not None) / 3.0 if all(item["mean_normalized"] is not None for item in by_benchmark.values()) else None
            state_hash = self._state_hash(run_id, checkpoint)
            candidates.append({"position": checkpoint, "eligible": all_complete and utility is not None and bool(state_hash), "selection_utility": utility, "by_benchmark": by_benchmark, "state_hash": state_hash})
        eligible = [row for row in candidates if row["eligible"]]
        selected = min(eligible, key=lambda row: (-row["selection_utility"], -sum(v["complete"] for v in row["by_benchmark"].values()), row["position"], row["state_hash"])) if eligible else None
        return {"run_id": run_id, "candidates": candidates, "selected": selected, "selection_uses_test": False}

    def _state_hash(self, run_id: int, checkpoint: int) -> str:
        # This is an immutable identity supplied by the EVO executor; absent
        # execution state remains explicitly unselectable.
        state = self._state(); hashes = []
        for sid, row in self.slot_by_id.items():
            if row.get("kind") == "evolution" and row.get("run_id") == run_id and row.get("ordinal", -1) < checkpoint:
                hashes.append(state[sid].get("result", {}).get("after_state_sha256") if isinstance(state[sid].get("result"), dict) else None)
        return digest({"run_id": run_id, "checkpoint": checkpoint, "after_state_hashes": hashes}) if all(hashes) or checkpoint == 0 else ""

    def seal_test(self) -> dict[str, Any]:
        self.assert_frozen()
        state = self._state(); test_rows = [(sid, row) for sid, row in self.slot_by_id.items() if row["kind"] == "test"]
        if any(state[sid]["status"] not in {"submitted", "failed", "missing"} for sid, _ in test_rows):
            raise ValueError("TEST seal waits for every 1,911 generation slot to terminate")
        if (self.output / "test_seal.json").exists():
            existing = read_json(self.output / "test_seal.json")
            if existing.get("state_sha256") != digest(state): raise ValueError("TEST seal changed")
            return existing
        seal = {"schema_version": "joint-test-seal-v5", "required_test_slots": EXPECTED["test"], "submitted": sum(state[sid]["status"] == "submitted" for sid, _ in test_rows), "failed_or_missing": sum(state[sid]["status"] in {"failed", "missing"} for sid, _ in test_rows), "state_sha256": digest(state), "feedback_released": False}
        write_json(self.output / "test_seal.json", seal)
        return seal

    def status(self) -> dict[str, Any]:
        self.assert_frozen()
        state = self._state(); counts = defaultdict(int)
        for sid, row in state.items(): counts[self.slot_by_id[sid]["kind"] + ":" + row["status"]] += 1
        return {"registered": True, "slot_count": len(state), "counts": dict(counts), "test_sealed": (self.output / "test_seal.json").is_file(), "feedback_released": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("register", "status", "check", "seal"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper/experiments/joint_protocol_v5.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper/experiments/joint_task_splits_v5.json")
    args = parser.parse_args(argv)
    campaign = JointCampaign(args.output, args.protocol, args.manifest)
    if args.mode == "register": result = campaign.register()
    elif args.mode == "status": result = campaign.status()
    elif args.mode == "seal": result = campaign.seal_test()
    else: campaign._validate_protocol(); result = {"valid": True, "slot_count": len(campaign.slots), "inventory_sha256": digest(campaign.slots)}
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
