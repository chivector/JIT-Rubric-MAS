"""Build an immutable NBA source supplement from the verified fact receipt."""

from __future__ import annotations

import json
from pathlib import Path

from jit_mas.pipeline import write_json
from jit_mas.schemas import digest, utc_now
from jit_mas.targeted_nba import build_host_calculations


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "outputs/evolution_mechanism_audit_20261010/nba_verified_fact_matrix_v3.json"
FETCH = ROOT / "outputs/evolution_mechanism_audit_20261010/nba_fetch_raw/fetch_manifest.json"
TASK_ID = "6847465956a0f6376a605378"


def build(output: Path) -> dict:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    fetch = json.loads(FETCH.read_text(encoding="utf-8"))
    by_id = {}
    for source in matrix["sources"]:
        by_id.setdefault(source["source_id"], source)
    sources = []
    for source in by_id.values():
        sources.append({
            "source_id": source["source_id"],
            "title": source.get("title") or source["source_id"],
            "date": source.get("scope"),
            "url": source.get("final_url") or source.get("requested_url"),
            "final_url": source.get("final_url") or source.get("requested_url"),
            "status": "ok",
            "excerpt": source["excerpt"],
            "raw_text_path": source.get("raw_text_path"),
            "http_status": source.get("http_status"),
            "content_hash_sha256": source.get("content_hash_sha256"),
        })
    successful = [row for row in fetch if row.get("http_status") == 200]
    failed = [row for row in fetch if row.get("http_status") != 200]
    facts = []
    for fact in matrix["facts"]:
        facts.append({key: fact[key] for key in ("entity", "dimension", "claim", "source_ids")})
    payload = {
        "schema_version": "nba-source-supplement-v3",
        "created_at": utc_now(),
        "matrix_path": str(MATRIX),
        "matrix_sha256": digest(matrix),
        "tasks": {TASK_ID: {
            "sources": sources,
            "derived_calculations": {
                "source_matrix_version": matrix["schema_version"],
                "facts": facts,
                "policy": [
                    "Use the host calculation table below as canonical inputs and totals.",
                    "Keep exactly the six listed players and include two concrete basketball-specific facts per player.",
                    "Define NBA as National Basketball Association at first use.",
                    "Cite each factual claim with its supplied source ID; never use blanket [GK].",
                    "Do not attribute Kobe Bryant's Mamba Mentality to Michael Jordan.",
                    "Keep observations within the requested up-to-2025 scope; label the Curry current profile as 2026-only if mentioned.",
                ],
                "comparison_rows": [
                    {"player": row["player"], "championships": row["championships"],
                     "mvps": row["mvps"], "source_ids": row["source_ids"]}
                    for row in build_host_calculations()["rows"]
                ],
                "host_calculations": build_host_calculations(),
            },
        }},
        "preparation": {
            "matrix_source_count": len(matrix["sources"]),
            "unique_source_count": len(sources),
            "successful_http_requests": len(successful),
            "failed_http_requests": len(failed),
            "http_requests": len(fetch),
            "failed_statuses": {str(status): sum(row.get("http_status") == status for row in failed)
                                for status in sorted({row.get("http_status") for row in failed})},
            "bytes": sum(row.get("bytes", 0) for row in fetch),
            "fetch_duration_seconds": sum(row.get("fetch_duration_seconds", 0) for row in fetch),
            "raw_scope": "HTTP source preparation only; not model API usage or cost",
        },
        "provenance": {
            "failed_fetches_retained_in_manifest": True,
            "matrix_facts_only_from_http_200_sources": True,
            "no_model_api_calls": True,
        },
    }
    write_json(output, payload)
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("C:/J/nba_source_supplement_v3.json"))
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"output": str(args.output), "sources": len(result["tasks"][TASK_ID]["sources"]),
                      "http_requests": result["preparation"]["http_requests"]}, ensure_ascii=True))
