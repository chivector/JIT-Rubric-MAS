"""Build a provenance-preserving synthesis repair from an EVO candidate.

The repair is explicit engineering, not another model-proposed evolution round.
It uses the existing five-task reflections and host validation; the source
receipt remains unchanged and the new decision/snapshot are saved separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from jit_mas.batch_evolution import BatchEvolutionDecision, candidate_batch_snapshot
from jit_mas.agent_pool import effective_pool
from jit_mas.pipeline import write_json
from jit_mas.schemas import ChangeProposal, ExperienceSnapshot, digest, utc_now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def repair(campaign, candidate_path, replay_manifest, output):
    source = read(candidate_path)
    manifest = read(replay_manifest)
    with sqlite3.connect(Path(campaign) / "campaign.sqlite") as database:
        base = ExperienceSnapshot.model_validate_json(database.execute(
            "SELECT snapshot FROM checkpoints WHERE source=? AND position=0",
            ("researchrubrics",)).fetchone()[0])
        slots = [json.loads(row[0]) for row in database.execute(
            "SELECT body FROM slots WHERE status='complete'")]
    slot = next(row for row in slots if row["source"] == "researchrubrics"
                and row["kind"] == "batch_evolution" and row["batch_index"] == 0)
    records = [read(path) for path in sorted((Path(campaign) / "slots" / digest(slot["slot_id"])
                                            / "reflections").glob("*/reflection.json"))]
    proposals = [ChangeProposal.model_validate(item) for row in records for item in row.get("proposals", [])]
    # Existing replay decision uses namespaced experience IDs. Preserve them.
    proposals = [item.model_copy(update={"experience": item.experience.model_copy(update={
        "experience_id": item.experience.experience_id if item.experience.experience_id.startswith(item.source_task_id + ":")
        else item.source_task_id + ":" + item.experience.experience_id})}) for item in proposals]
    decision = source["decision"]
    writer = next(item for item in decision["agent_updates"] if item["pool_agent_id"] == "writer")
    # Keep the writer's retained memory and broad skills, then add the evidenced
    # synthesis rule to its existing direct update.
    quantitative = next(item for item in writer["lessons"] if "quantitative detail" in item["instruction"])
    base_writer = next(item for item in effective_pool(base.agent_pool).profiles if item.pool_agent_id == "writer")
    writer["skills"] = dict(base_writer.skills)
    writer["skills"]["synthesis"] += ("\nWhen synthesizing multi-entity factual or quantitative material, "
        "retain every supported entity and its requested dimensions in a compact coverage matrix. "
        "Preserve upstream quantitative details and source identities; recompute printed arithmetic "
        "before using it to rank alternatives. Do not drop a member or citation during prose compression.")
    decision = BatchEvolutionDecision.model_validate(decision)
    snapshot = candidate_batch_snapshot(base, decision, source_task_ids=manifest["source_task_ids"],
                                       proposals=proposals)
    receipt = {"status": "complete", "snapshot": snapshot.model_dump(mode="json"),
        "state_hash": digest(snapshot), "base_state_hash": digest(base),
        "decision": decision.model_dump(mode="json"), "created_at": utc_now(),
        "repair_metadata": {"kind": "explicit engineering repair of synthesis coverage",
            "source_candidate": str(candidate_path), "source_receipt_hash": digest(source),
            "source_evo_lesson": quantitative["lesson_id"],
            "source_task_ids": writer["source_task_ids"], "evidence": quantitative["evidence"],
            "note": "Not a new model-proposed evolution round; all baseline and failed candidates retained."}}
    write_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair(args.campaign, args.candidate, args.replay_manifest, args.output)


if __name__ == "__main__":
    main()
