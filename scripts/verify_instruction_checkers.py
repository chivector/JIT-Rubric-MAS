"""Verify pinned author code and exercise only synthetic instruction examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_mas.instruction_checkers import PinnedInstructionChecker
from jit_mas.schemas import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifbench-source-root", required=True)
    parser.add_argument("--ifeval-source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    identities = {}
    for benchmark in ("ifeval", "ifbench"):
        source_root = args.ifeval_source_root if benchmark == "ifeval" else args.ifbench_source_root
        checker = PinnedInstructionChecker(source_root, benchmark)
        record = {"prompt": "Write a synthetic response containing the word cat.",
                  "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["cat"]}]}
        passed = checker.check_record("The cat is asleep.", record)
        failed = checker.check_record("The dog is asleep.", record)
        if passed != [True] or failed != [False]:
            raise ValueError("Pinned author checker synthetic smoke failed")
        identities[benchmark] = checker.identity
    report = {"version": "jit-mas-author-checker-preflight-v1", "synthetic_only": True,
              "passed": True, "checkers": identities}
    report["identity_sha256"] = digest(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("Refusing to replace pinned checker preflight identity")
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"passed": True, "synthetic_only": True, "identity_sha256": report["identity_sha256"],
                      "output": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
