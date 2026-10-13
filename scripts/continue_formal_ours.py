r"""Resume the frozen formal Ours campaign with safe local credential loading.

Examples (PowerShell):
  $env:JIT_BENCHMARK_API_KEY = (Get-Content C:\secrets\jit.env.key -Raw).Trim()
  python scripts/continue_formal_ours.py --mode run
  python scripts/continue_formal_ours.py --mode status
  python scripts/continue_formal_ours.py --mode report

The script never prints credential values. It reads the campaign's frozen
transport identity from SQLite, restores missing frozen inputs from an optional
archive only after hash verification, and lets the batch runner recover durable
started slots on the next invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "tds_fix9"
DEFAULT_LAUNCH = ROOT / "outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json"
DEFAULT_ARCHIVE = ROOT.parent / "JIT-main-archive-20261011/caches/.runtime"
SECRET_NAMES = {"JIT_BENCHMARK_API_KEY", "RESCORE_JUDGE_API_KEY"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_registration(campaign: Path) -> dict:
    import sqlite3
    con = sqlite3.connect(campaign / "campaign.sqlite")
    try:
        row = con.execute("SELECT body FROM metadata WHERE key='registration'").fetchone()
    finally:
        con.close()
    if not row:
        raise RuntimeError(f"No frozen registration in {campaign}")
    return json.loads(row[0])


def load_credentials(path: Path | None) -> None:
    """Load only known key/value entries; do not echo them."""
    if path:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("credential JSON must be an object")
        else:
            values = {}
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
        for key in SECRET_NAMES:
            if isinstance(values.get(key), str) and values[key].strip():
                os.environ[key] = values[key].strip()
    if not os.environ.get("RESCORE_JUDGE_API_KEY") and os.environ.get("JIT_BENCHMARK_API_KEY"):
        os.environ["RESCORE_JUDGE_API_KEY"] = os.environ["JIT_BENCHMARK_API_KEY"]
    missing = [key for key in ("JIT_BENCHMARK_API_KEY", "RESCORE_JUDGE_API_KEY") if not os.environ.get(key)]
    if missing:
        raise RuntimeError("Missing local credentials: " + ", ".join(missing) + "; pass --credential-file or set them in the environment")


def set_frozen_transport(registration: dict) -> None:
    transport = registration.get("execution", {}).get("transport", {})
    mapping = {
        "model_attempts": "JIT_MAS_MODEL_ATTEMPTS",
        "disable_keepalive": "JIT_MAS_DISABLE_KEEPALIVE",
        "batch_meta_corrections": "JIT_MAS_BATCH_META_CORRECTIONS",
        "extra_corrections": "JIT_MAS_ALLOW_EXTRA_CORRECTIONS",
    }
    for field, env_name in mapping.items():
        if field not in transport:
            continue
        value = transport[field]
        if isinstance(value, bool):
            value = "1" if value else "0"
        os.environ[env_name] = str(value)
    retry = transport.get("connection_retry", {})
    for field, env_name in (("base_seconds", "JIT_MAS_RETRY_BASE_SECONDS"), ("max_seconds", "JIT_MAS_RETRY_MAX_SECONDS"), ("jitter_seconds", "JIT_MAS_RETRY_JITTER_SECONDS")):
        if field in retry:
            os.environ[env_name] = str(retry[field])
    os.environ["MODULAR_AGENT_API_FAILURE_ACTION"] = transport.get("api_failure_action", "exit")
    # These guards are implementation switches, not frozen transport inputs.
    # Keep them enabled whenever the formal continuation is resumed.
    os.environ["JIT_MAS_ENFORCE_STRUCTURAL_OPS"] = "1"
    os.environ["JIT_MAS_BATCH_SANITIZE"] = "1"
    os.environ["JIT_MAS_BATCH_INVALID_FALLBACK"] = "1"


def _runtime_relative(path: Path) -> Path | None:
    parts = path.resolve().parts
    try:
        index = parts.index(".runtime")
    except ValueError:
        return None
    return Path(*parts[index + 1:])


def ensure_inputs(registration: dict, archive: Path | None) -> None:
    materials = registration.get("execution", {}).get("materials", {})
    for material in materials.values():
        data = Path(material["data"])
        evidence = Path(material["evidence_dir"]) if material.get("evidence_dir") else None
        targets = [data] + ([evidence] if evidence else [])
        for target in targets:
            if target.exists():
                continue
            if archive is None:
                raise FileNotFoundError(f"Frozen input missing: {target}; pass --runtime-archive")
            relative = _runtime_relative(target)
            if relative is None:
                raise FileNotFoundError(f"Cannot infer archive-relative path for {target}")
            source = archive / relative
            if not source.exists():
                raise FileNotFoundError(f"Frozen input missing in archive: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
    # Verify all registered bytes after any restore.
    for material in materials.values():
        data = Path(material["data"])
        if sha256(data) != material["dataset_sha256"]:
            raise RuntimeError(f"Dataset hash mismatch: {data}")
        evidence = material.get("evidence")
        if evidence:
            files = evidence.get("files", {})
            root = Path(material["evidence_dir"])
            actual = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
            if actual != files or digest(actual) != evidence.get("sha256"):
                raise RuntimeError(f"Evidence hash mismatch: {root}")


def compact_status(campaign: Path) -> str:
    path = campaign / "status.json"
    if not path.exists():
        return "status=not_started"
    j = json.loads(path.read_text(encoding="utf-8")); s = j.get("statuses", {})
    keys = ("complete", "failed", "incomplete", "missing", "pending", "started", "submitted")
    return " ".join([f"{k}={s.get(k, 0)}" for k in keys]) + f" terminal={j.get('terminal_slots', 0)}"


def run(campaign: Path, launch: Path, interval: float) -> int:
    env = os.environ.copy()
    command = [sys.executable, "-u", "-m", "scripts.run_independent_batch_experiment", "--mode", "run", "--launch-config", str(launch), "--output", str(campaign), "--phase", "full"]
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        while child.poll() is None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {compact_status(campaign)}", flush=True)
            time.sleep(max(5.0, interval))
    except KeyboardInterrupt:
        print("Interrupted locally; campaign remains resumable. Re-run this command to recover durable started slots.", flush=True)
        return 130
    print(f"runner_exit={child.returncode} {compact_status(campaign)}", flush=True)
    return child.returncode or 0


def make_report(campaign: Path) -> int:
    commands = [
        [sys.executable, "scripts/build_tds8_recovery_report.py"],
        [sys.executable, "-m", "scripts.report_effective_evolution", "--campaign", str(campaign), "--output", str(ROOT / "outputs/formal_ours_20261010" / f"{campaign.name}_effective_evolution.json")],
        [sys.executable, "-m", "scripts.summarize_independent_paper", "--campaign", str(campaign), "--output-dir", str(campaign.with_name(campaign.name + "_paper"))],
    ]
    rc = 0
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT)
        rc = rc or completed.returncode
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "status", "report"), default="run")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--runtime-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    campaign, launch = args.campaign.resolve(), args.launch.resolve()
    if not campaign.exists() or not launch.exists():
        raise FileNotFoundError("campaign or launch path does not exist")
    registration = read_registration(campaign)
    # Status and report are local read-only operations and must remain usable
    # when the operator is inspecting a campaign from a machine without keys.
    # Credentials are required only when the runner is actually resumed.
    if args.mode == "run":
        load_credentials(args.credential_file.resolve() if args.credential_file else None)
    set_frozen_transport(registration)
    archive = args.runtime_archive.resolve() if args.runtime_archive and args.runtime_archive.exists() else None
    ensure_inputs(registration, archive)
    if args.mode == "status":
        print(compact_status(campaign)); return 0
    if args.mode == "report":
        return make_report(campaign)
    return run(campaign, launch, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
