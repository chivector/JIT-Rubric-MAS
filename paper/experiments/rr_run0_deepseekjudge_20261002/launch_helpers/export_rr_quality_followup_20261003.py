"""Export completed public VAL engineering probes and an immutable v35 progress snapshot."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "paper/experiments/rr_run0_deepseekjudge_20261002"
TARGET = PACKAGE / "quality_followup_20261003"
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
        raise ValueError("Refusing to overwrite a completed artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


for tag in ("20261003", "20261003_v2", "20261003_v4"):
    source = ROOT / ("outputs/rr_quality_preflight_" + tag)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] in {"passed", "failed"}
    assert summary["judge_requests"] == 0 and summary["formal_score"] is None
    assert summary["experience_unchanged"]
    for artifact in sorted(source.rglob("*.json")):
        copy(artifact, TARGET / tag / artifact.relative_to(source))
    copy(ROOT / (".runtime/rr_quality_preflight_" + tag) / "source_manifest.json",
         TARGET / tag / "source_manifest.json")

copy(ROOT / "outputs/rr_quality_preflight_20261003_v3.stderr.log",
     TARGET / "v3_launch_failed.stderr.log")
pair_source = ROOT / "outputs/rr_quality_pair_20261003"
pair_summary = json.loads((pair_source / "summary.json").read_text(encoding="utf-8"))
assert pair_summary["status"] in {"completed", "incomplete"}
assert pair_summary["experience_unchanged"] and pair_summary["generation_calls"] == 0
for artifact in sorted(pair_source.glob("*.json")):
    copy(artifact, TARGET / "sealed_VAL_pair" / artifact.name)
copy(ROOT / "outputs/rr_quality_pair_20261003.stderr.log",
     TARGET / "pair_initial_hash_guard_rejection.stderr.log")
for name in ("preflight_rr_quality_20261003.py", "export_rr_quality_followup_20261003.py",
             "score_rr_quality_pair_20261003.py"):
    copy(ROOT / "outputs" / name, PACKAGE / "launch_helpers" / name)

progress = json.loads((ROOT / "outputs/rr_v35_monitor_status.json").read_text(encoding="utf-8"))
validate(progress)
progress_path = TARGET / "v35_progress_snapshot.json"
if not progress_path.exists():
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
journal_path = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v35/ours/checkpoints/checkpoint_journal.json"
summary_path = TARGET / "v35_checkpoint_journal_summary.json"
if not summary_path.exists():
    content = journal_path.read_bytes()
    journal = json.loads(content)
    summary = {"status": journal["status"], "identity": journal["identity"],
        "selected_position": journal["selected_position"],
        "provisional_selected_position": journal["provisional_selected_position"],
        "checkpoints": journal["checkpoints"],
        "sources": [{name: row[name] for name in ("position", "task_id", "status", "before_version",
                                                  "before_hash", "after_version", "after_hash") if name in row}
                    for row in journal["sources"]],
        "source_journal_sha256": hashlib.sha256(content).hexdigest(),
        "source_journal_bytes": len(content), "full_source_retained_in_local_outputs": True,
        "observed_at": datetime.now(timezone.utc).isoformat()}
    validate(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
manifest = {artifact.relative_to(PACKAGE).as_posix(): hashlib.sha256(artifact.read_bytes()).hexdigest()
            for artifact in sorted(PACKAGE.rglob("*"))
            if artifact.is_file() and artifact.name != "MANIFEST_SHA256.json"}
(PACKAGE / "MANIFEST_SHA256.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"exported": str(TARGET), "artifact_count": len(manifest),
                  "exported_at": datetime.now(timezone.utc).isoformat(),
                  "formal_scores": False}))
