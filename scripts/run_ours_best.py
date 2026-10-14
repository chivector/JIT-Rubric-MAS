"""Canonical entry point for the best verified Ours compact evaluation.

The benchmark plan and data are supplied by the caller.  This thin wrapper
keeps one stable command while delegating resumability and shard-level
recovery to the tested runner.  Credentials are read only from environment
variables by the selected config; no key is stored in the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_targeted_compact_sharded import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ours config with env placeholders")
    parser.add_argument("--plan", required=True, help="Frozen public-task plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args(argv)
    run(Path(args.config), Path(args.plan), Path(args.output), shards=args.shards,
        concurrency=args.concurrency, resume=args.resume, recover_only=args.recover_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
