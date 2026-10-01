"""Freeze public-only inventories for the v4 writing/instruction task pools.

This command never invokes a model and does not export prompts or evaluator
fields. Raw author releases stay under the ignored dataset directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from jit_mas.benchmarks import normalized_question_hash
from jit_mas.schemas import digest


RELEASES = {
    "writingbench": {
        "dataset_revision": "ae2d5176449b7b769815482641d35926f26793eb",
        "dataset_sha256": "18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe",
        "data_url": "https://raw.githubusercontent.com/X-PLUG/WritingBench/ae2d5176449b7b769815482641d35926f26793eb/benchmark_query/benchmark_all.jsonl",
        "local_path": "dataset/writingbench/benchmark_all.jsonl",
        "expected_tasks": 1000,
        "source_id_field": "index",
        "question_field": "query",
        "stratification_fields": ["lang", "domain1"],
        "license": "Apache-2.0",
        "official_split_policy": "Complete updated 1000-query author release; no official train/validation/test partition",
    },
    "ifeval": {
        "dataset_revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "dataset_sha256": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        "data_url": "https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl",
        "local_path": "dataset/ifeval/ifeval_input_data.jsonl",
        "expected_tasks": 541,
        "source_id_field": "key",
        "question_field": "prompt",
        "stratification_fields": [],
        "license": "Apache-2.0",
        "official_split_policy": "Full author evaluation pool; Hugging Face train label does not define an experimental training split",
    },
    "ifbench": {
        "dataset_revision": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
        "dataset_sha256": "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a",
        "data_url": "https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl",
        "local_path": "dataset/ifbench/IFBench_test.jsonl",
        "expected_tasks": 300,
        "source_id_field": "key",
        "question_field": "prompt",
        "stratification_fields": [],
        "license": "ODC-BY-1.0 data; Apache-2.0 code",
        "official_split_policy": "Full single-turn author test pool; excludes multi-turn and IF-RLVR training releases",
    },
}

# A PowerShell parser error exposed part of author row 1 while preparing v4.
# Preserve the fact without copying the prompt or private criteria into output.
KNOWN_EXPOSURES = {
    "writingbench": [{"source_id": 1, "source_row": 1,
                      "reason": "Partial prompt/criteria accidentally echoed by metadata parser error during v4 preparation"}],
    "ifeval": [],
    "ifbench": [],
}


def project_rows(name: str, records: list[dict]) -> list[dict]:
    """Project only IDs, public stratification, and normalized-question hashes."""
    spec = RELEASES[name]
    output = []
    seen = set()
    for position, record in enumerate(records, 1):
        source_id = record.get(spec["source_id_field"])
        if type(source_id) not in (str, int) or not str(source_id):
            raise ValueError(f"Invalid upstream task ID in {name} row {position}")
        task_id = f"{name}:{source_id}"
        if task_id in seen:
            raise ValueError(f"Duplicate upstream task ID in {name} row {position}")
        seen.add(task_id)
        question = record.get(spec["question_field"])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing public question in {name} row {position}")
        strata = [record.get(field) for field in spec["stratification_fields"]]
        if any(not isinstance(value, str) or not value.strip() for value in strata):
            raise ValueError(f"Missing public stratum in {name} row {position}")
        question_hash = normalized_question_hash(question)
        output.append({"task_id": task_id, "source_id": source_id,
                       "source_row": position, "question_sha256": question_hash,
                       "group_id": question_hash, "stratum": "|".join(strata) or "all"})
    return output


def build_inventory(name: str, path: Path | None = None) -> dict:
    """Validate a pinned release and return its metadata-only inventory."""
    spec = RELEASES[name]
    path = path or Path(spec["local_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["dataset_sha256"]:
        raise ValueError(f"Pinned release hash mismatch for {name}")
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Unexpected blank source row for {name}: {line_number}")
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid JSON for {name} at source row {line_number}") from None
        if not isinstance(record, dict):
            raise ValueError(f"Non-object row for {name}: {line_number}")
        records.append(record)
    if len(records) != spec["expected_tasks"]:
        raise ValueError(f"Pinned release task count mismatch for {name}")
    rows = project_rows(name, records)
    groups = Counter(row["group_id"] for row in rows)
    exposures = [{"task_id": f"{name}:{entry['source_id']}", **entry}
                 for entry in KNOWN_EXPOSURES[name]]
    result = {
        "version": "jit-compose-public-inventory-v4",
        "benchmark": name,
        **{field: spec[field] for field in (
            "dataset_revision", "dataset_sha256", "data_url", "expected_tasks",
            "source_id_field", "stratification_fields", "license", "official_split_policy")},
        "source_row_convention": "1-based position in pinned JSONL, before sorting or filtering",
        "normalization": "SHA256 of whitespace-collapsed casefolded public question, UTF-8",
        "scope": "Public IDs and metadata only; exact duplicates are not a semantic/source overlap guarantee",
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "exact_duplicate_groups": [
            {"group_id": group, "task_ids": [row["task_id"] for row in rows if row["group_id"] == group]}
            for group, size in sorted(groups.items()) if size > 1],
        "known_exposures": exposures,
        "exposed_task_ids": [entry["task_id"] for entry in exposures],
        "rows": rows,
    }
    result["inventory_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["all", *RELEASES], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/experiments"))
    parser.add_argument("--download", action="store_true", help="Fetch pinned raw author files to ignored dataset paths")
    args = parser.parse_args()
    names = list(RELEASES) if args.benchmark == "all" else [args.benchmark]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = RELEASES[name]
        path = Path(spec["local_path"])
        if args.download and not path.exists():
            with urlopen(spec["data_url"], timeout=120) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != spec["dataset_sha256"]:
                raise ValueError(f"Downloaded release hash mismatch for {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        document = build_inventory(name, path)
        output = args.output_dir / f"{name}_inventory_v4.json"
        output.write_text(json.dumps(document, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"benchmark": name, "tasks": len(document["rows"]),
                          "duplicate_groups": len(document["exact_duplicate_groups"]),
                          "known_exposures": len(document["known_exposures"]),
                          "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
