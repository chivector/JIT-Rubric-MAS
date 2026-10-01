"""Prepare a pinned ResearchRubrics dataset and reproducible task-level splits."""

import argparse
import json
import random
from pathlib import Path
from urllib.request import urlopen

from benchmark.adapter.researchrubrics import DATASET_REVISION, ResearchRubricsAdapter
from jit_mas.schemas import SplitManifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Existing processed_data.jsonl")
    parser.add_argument("--download", action="store_true", help="Download the pinned official public dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evolution-size", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=0,
                        help="Optional reserved development subset; not used to gate experience updates")
    parser.add_argument("--stream", action="store_true", help="Use evolution subset as an ordered online stream")
    args = parser.parse_args(argv)
    if bool(args.input) == bool(args.download):
        parser.error("Choose exactly one of --input or --download")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "processed_data.jsonl"
    manifest_path = output / "splits.json"
    if target.exists() or manifest_path.exists():
        parser.error("Output already exists; choose a fresh preparation directory")
    if args.download:
        url = f"https://huggingface.co/datasets/ScaleAI/researchrubrics/resolve/{DATASET_REVISION}/processed_data.jsonl"
        with urlopen(url, timeout=60) as response:
            content = response.read()
    else:
        content = Path(args.input).read_bytes()
    target.write_bytes(content)
    rows = ResearchRubricsAdapter().load_dataset(str(target))
    ids = [row["task_id"] for row in rows]
    if args.evolution_size < 1 or args.validation_size < 0 or args.evolution_size + args.validation_size >= len(ids):
        parser.error("Positive evolution and nonnegative reserved validation sizes must leave an untouched test split")
    random.Random(args.seed).shuffle(ids)
    a, b = args.evolution_size, args.evolution_size + args.validation_size
    manifest = SplitManifest(seed=args.seed, evolution=[] if args.stream else ids[:a],
        stream=ids[:a] if args.stream else [], validation=ids[a:b], test=ids[b:])
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"dataset_revision": DATASET_REVISION, "tasks": len(ids),
                      "manifest": str(manifest_path), "benchmark_executed": False}))


if __name__ == "__main__":
    main()
