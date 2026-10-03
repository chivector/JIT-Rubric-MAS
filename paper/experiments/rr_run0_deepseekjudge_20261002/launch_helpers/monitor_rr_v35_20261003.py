"""Reuse the read-only RR monitor for the isolated v35 attempt."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rr_monitor", ROOT / "outputs/monitor_rr_v34_20261002.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.METHOD = ROOT / "outputs/rr_jit_mas_run0_deepseekjudge_20261002_v35"
monitor.LAUNCH = ROOT / "outputs/rr_deepseek_method_launch_20261002_v35/launch.json"
monitor.STATUS = ROOT / "outputs/rr_v35_monitor_status.json"
monitor.PID_FILE = ROOT / "outputs/rr_v35_monitor.pid"
monitor.COMPARE = ROOT / "outputs/compare_rr_v35_v29.json"

if __name__ == "__main__":
    raise SystemExit(monitor.main())
