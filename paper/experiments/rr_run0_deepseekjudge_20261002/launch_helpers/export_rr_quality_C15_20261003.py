"""Export a terminal ten-VAL engineering comparison without changing either run."""

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/rr_quality_C15_20261003"
PACKAGE = ROOT / "paper/experiments/rr_run0_deepseekjudge_20261002"
TARGET = PACKAGE / "quality_C15_20261003"
FORBIDDEN = {"api_key", "authorization", "rr_exec_api_key", "rr_judge_api_key", "password", "secret"}
ALLOWED = (None, "", "INJECTED", "REDACTED", "[REDACTED]", "***")


def validate(value):
    if isinstance(value, dict):
        for name, item in value.items():
            if str(name).lower() in FORBIDDEN and item not in ALLOWED:
                raise ValueError("Credential-bearing export field")
            validate(item)
    elif isinstance(value, list):
        for item in value:
            validate(item)


def copy(source, destination):
    content = source.read_bytes()
    if re.search(rb"\bsk-[A-Za-z0-9_-]{25,}", content):
        raise ValueError("Credential-like export text")
    if source.suffix == ".json":
        validate(json.loads(content))
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError("Refusing to change a completed artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
registration = json.loads((SOURCE / "registration.json").read_text(encoding="utf-8"))
assert summary["status"] in {"completed", "inconclusive", "interrupted"}
assert summary["test_calls"] == 0 and registration["state_updates_allowed"] is False
for artifact in sorted(SOURCE.rglob("*.json")):
    copy(artifact, TARGET / artifact.relative_to(SOURCE))
copy(ROOT / ".runtime/rr_quality_C15_20261003/source_manifest.json", TARGET / "source_manifest.json")
sys.path.insert(0, str(ROOT / ".runtime/rr_quality_C15_20261003"))
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import digest

state_path = SOURCE / "states" / (registration["state_hash"] + ".sqlite")
store = ExperienceStore(state_path, read_only=True)
try:
    snapshot = store.snapshot(registration["source_state_version"])
    assert digest(snapshot) == registration["state_hash"]
    state_content = (snapshot.model_dump_json(indent=2) + "\n").encode("utf-8")
    validate(json.loads(state_content))
    state_target = TARGET / "state_snapshot.json"
    if state_target.exists() and state_target.read_bytes() != state_content:
        raise ValueError("Refusing to change the isolated historical state")
    state_target.write_bytes(state_content)
finally:
    store.close()

historical_accounting = {name: 0 for name in ("model_calls", "tokens", "tool_calls", "wall_seconds")}
for row in registration["historical_C15"]:
    if row.get("run_dir"):
        budget = json.loads((Path(row["run_dir"]) / "budget.json").read_text(encoding="utf-8"))
        for name in historical_accounting:
            historical_accounting[name] += budget.get(name, 0)
cost_path = TARGET / "historical_accounting.json"
cost_content = (json.dumps({**historical_accounting, "cost": None}, indent=2) + "\n").encode("utf-8")
if cost_path.exists() and cost_path.read_bytes() != cost_content:
    raise ValueError("Original C15 budget changed")
cost_path.write_bytes(cost_content)
for name in ("validate_rr_quality_C15_20261003.py", "export_rr_quality_C15_20261003.py"):
    copy(ROOT / "outputs" / name, PACKAGE / "launch_helpers" / name)
manifest = {artifact.relative_to(PACKAGE).as_posix(): hashlib.sha256(artifact.read_bytes()).hexdigest()
            for artifact in sorted(PACKAGE.rglob("*"))
            if artifact.is_file() and artifact.name != "MANIFEST_SHA256.json"}
(PACKAGE / "MANIFEST_SHA256.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"exported": str(TARGET), "artifact_count": len(manifest),
                  "status": summary["status"], "valid_summary": summary["valid_summary"],
                  "selection_utility_difference": summary["selection_utility_difference"]}))
