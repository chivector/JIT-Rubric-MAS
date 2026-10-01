"""Prepare public-only, quarantined paper split candidates; never run models.

This is a draft partition, not permission to start a benchmark. Topic/source,
retrieval, runner and containment audits must precede a formal preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from jit_mas.schemas import PublicTask, SplitManifest, digest


VERSION = "jit-compose-paper-splits-v1"
SEED = "jit-compose-paper-splits-v1\n"
DATA_SHA256 = "ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb"


def rank(task_id, seed=SEED):
    return hashlib.sha256((seed + task_id).encode("utf-8")).hexdigest()


def public_projection(row):
    return PublicTask(task_id=row["sample_id"], question=row["prompt"],
        attachments=row.get("attachments", []), constraints=row.get("explicit_constraints", []))


def build_partition(public_tasks, exposure, pilot, *, evolution_count=30, validation_count=20):
    tasks = {task.task_id: task for task in public_tasks}
    if len(tasks) != len(public_tasks):
        raise ValueError("Duplicate public task IDs")
    quarantine = set(exposure["excluded_task_ids"]) | {row["public_task"]["task_id"] for row in pilot["tasks"]}
    if not quarantine <= tasks.keys():
        raise ValueError("Quarantine refers to missing tasks")
    eligible = sorted(tasks.keys() - quarantine, key=rank)
    if min(evolution_count, validation_count) < 1 or len(eligible) <= evolution_count + validation_count:
        raise ValueError("Require nonempty evolution, validation and test partitions")
    manifest = SplitManifest(evolution=eligible[:evolution_count],
        validation=eligible[evolution_count:evolution_count + validation_count],
        test=eligible[evolution_count + validation_count:])
    assignments = {task_id: split for split in ("evolution", "validation", "test")
                   for task_id in getattr(manifest, split)}
    assignments.update({task_id: "development_quarantine" for task_id in quarantine})
    question_groups, source_groups = {}, {}
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        normalized = " ".join(task.question.casefold().split())
        question_groups.setdefault(digest(normalized), []).append(task_id)
        text = task.question + "\n" + "\n".join(task.constraints)
        sources = sorted(set(url.rstrip(".,;:)\"'") for url in re.findall(r"https?://[^\s<>]+", text)))
        for source in sources:
            source_groups.setdefault(source, []).append(task_id)
        rows.append({"task_id": task_id, "partition": assignments[task_id],
            "public_task_sha256": digest(task), "public_prompt_excerpt": task.question[:180],
            "public_urls": sources, "attachment_count": len(task.attachments)})

    def overlaps(groups):
        return [{"group": group, "task_ids": ids, "partitions": sorted({assignments[i] for i in ids})}
                for group, ids in sorted(groups.items()) if len({assignments[i] for i in ids}) > 1]

    return {"version": VERSION, "status": "DRAFT_NOT_RUN", "launch_allowed": False,
        "selection": {"seed": SEED, "ranking": "Ascending SHA256(seed + task_id)",
            "unit": "whole task", "stratified": False,
            "filters": "Quarantine only; no closed-book, topic, difficulty, score or private-rubric filter."},
        "counts": {"total": len(tasks), "development_quarantine": len(quarantine),
            "evolution": len(manifest.evolution), "validation": len(manifest.validation), "test": len(manifest.test)},
        "development_quarantine": sorted(quarantine), "runtime_split_manifest": manifest.model_dump(mode="json"),
        "task_index": rows,
        "public_overlap_audit": {"exact_normalized_question_cross_partition": overlaps(question_groups),
            "exact_url_cross_partition": overlaps(source_groups),
            "topic_and_semantic_source_review": "REQUIRED_BEFORE_FORMAL_FREEZE",
            "limitations": "Exact public-text/URL checks do not establish source or topic independence; hidden sources are not inspected."},
        "interpretation": "Within-benchmark task transfer only. Not an official dataset split, topic-disjoint evaluation, or measured result."}


def validate_partition(document):
    if (document.get("version") != VERSION or document.get("status") != "DRAFT_NOT_RUN"
            or document.get("launch_allowed") is not False):
        raise ValueError("This utility validates draft partitions, not executable preregistrations")
    manifest = SplitManifest.model_validate(document["runtime_split_manifest"])
    groups = {"development_quarantine": document["development_quarantine"],
              **{name: getattr(manifest, name) for name in ("evolution", "validation", "test")}}
    all_ids = [task_id for ids in groups.values() for task_id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Development/evolution/validation/test overlap")
    if manifest.stream or len(all_ids) != document["counts"]["total"]:
        raise ValueError("Unexpected streaming or incomplete task inventory")
    expected = {task_id: name for name, ids in groups.items() for task_id in ids}
    if (len(document["task_index"]) != len(all_ids)
            or {row["task_id"]: row["partition"] for row in document["task_index"]} != expected):
        raise ValueError("Public task index does not match the split")
    if any(len(ids) != document["counts"][name] for name, ids in groups.items()):
        raise ValueError("Partition counts changed")
    eligible = sorted(set(all_ids) - set(groups["development_quarantine"]), key=rank)
    if eligible != manifest.evolution + manifest.validation + manifest.test:
        raise ValueError("Partition does not follow the declared fixed ranking")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--exposure")
    parser.add_argument("--pilot-registration")
    parser.add_argument("--output")
    parser.add_argument("--check")
    args = parser.parse_args()
    if args.check:
        document = json.loads(Path(args.check).read_text(encoding="utf-8"))
        validate_partition(document)
        print(json.dumps({"status": "valid_draft_partition", "launch_allowed": False, "counts": document["counts"]}))
        return
    if not all((args.data, args.exposure, args.pilot_registration, args.output)):
        parser.error("Preparation needs data, exposure, pilot-registration and output")
    data = Path(args.data).read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset does not match the pinned release")
    exposure = json.loads(Path(args.exposure).read_text(encoding="utf-8"))
    pilot = json.loads(Path(args.pilot_registration).read_text(encoding="utf-8"))
    if exposure["dataset_sha256"] != DATA_SHA256 or pilot["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Exposure records describe a different dataset")
    # Project immediately; never copy private criteria, weights, or reference answers.
    tasks = [public_projection(json.loads(line)) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    document = build_partition(tasks, exposure, pilot)
    document["provenance"] = {"dataset_sha256": DATA_SHA256, "dataset_revision": exposure["dataset_revision"],
        "exposure_record_sha256": hashlib.sha256(Path(args.exposure).read_bytes()).hexdigest(),
        "pilot_registration_sha256": hashlib.sha256(Path(args.pilot_registration).read_bytes()).hexdigest()}
    validate_partition(document)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({"status": document["status"], "launch_allowed": False, "counts": document["counts"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never echo a malformed private dataset row in diagnostics.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
