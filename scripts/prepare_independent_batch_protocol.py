"""Freeze the v6 independent batched protocol from its task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.independent_batch_protocol import build_protocol, validate_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("paper/experiments/joint_task_splits_v6.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper/experiments/independent_protocol_v6.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = build_protocol(manifest)
    validate_protocol(protocol, manifest)
    content = json.dumps(protocol, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            raise ValueError("Frozen v6 protocol differs from its manifest")
    elif args.output.exists():
        if args.output.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite a changed v6 protocol")
    else:
        args.output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"protocol_sha256": protocol["protocol_sha256"],
                      "workload": protocol["workload"], "status": protocol["status"]}))


if __name__ == "__main__":
    main()
