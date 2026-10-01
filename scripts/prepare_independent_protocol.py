"""Freeze the independent v5 protocol and complete logical slot registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.independent_protocol import build_protocol, slot_registry, validate_protocol


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-manifest", type=Path, default=Path("paper/experiments/joint_task_splits_v5.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--check", action="store_true", help="Validate existing independent artifacts")
    args = parser.parse_args(argv)
    protocol_path = args.output_dir / "independent_protocol_v5.json"
    registry_path = args.output_dir / "independent_slot_registry_v5.json"
    joint = _read(args.joint_manifest)
    if args.check:
        protocol = _read(protocol_path)
        validate_protocol(protocol, joint)
        registry = _read(registry_path)
        expected = slot_registry(protocol)
        if registry != expected:
            raise ValueError("Independent slot registry differs from the frozen protocol")
        print(json.dumps({"valid": True, "protocol_sha256": protocol["protocol_sha256"],
                          "registry_sha256": registry["registry_sha256"],
                          "slots": registry["counts"]}, ensure_ascii=True))
        return 0
    protocol = build_protocol(joint)
    registry = slot_registry(protocol)
    validate_protocol(protocol, joint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        protocol_path: json.dumps(protocol, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        registry_path: json.dumps(registry, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
    }
    for path, content in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite changed frozen artifact: {path}")
    for path, content in contents.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"protocol_sha256": protocol["protocol_sha256"],
                      "registry_sha256": registry["registry_sha256"],
                      "slots": registry["counts"], "model_calls": 0}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
