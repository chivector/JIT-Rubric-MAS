"""Archive compact run receipts and answers without source archives or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUN_NAMES = (
    "compact_ifbench_mixed_v1_20261009", "compact_ifbench_mixed_v2_20261009",
    "compact_ifbench_mixed_v3_20261009", "compact_ifbench_mixed_v4_20261009",
    "compact_ifbench_mixed_v5_20261009", "compact_ifeval_new12_v2_20261009",
    "compact_ifeval_new12_v3_20261009", "compact_ifeval_new12_v4_20261009",
    "compact_ifeval_new12_v5_20261009", "compact_ifbench_new12_v5_20261009",
    "targeted_rr_esg_coverage_20261009", "targeted_rr_esg_compact_v3_20261009",
    "targeted_rr_esg_compact_v4_20261009", "targeted_rr_esg_compact_v5_20261009",
    "targeted_rr_esg_arithmetic_v6_20261009", "targeted_rr_esg_evidence_v7_20261009",
)
FILES = {"manifest.json", "report.json", "full_report.json", "submission.json", "answer.md",
         "registration.json", "quality_diagnostics.json", "source_verification.json",
    "result_summary.md", "source_additions.json", "patch_receipt.json",
    "full_scoring_registration.json", "source_manifest.json", "public_source_additions.json",
    "calculation_checks.json", "answer.diff"}


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def archive(output: Path) -> dict:
    if output.exists():
        raise ValueError("Choose a new archive output directory")
    candidates = []
    summaries = []
    for name in RUN_NAMES:
        directory = ROOT / "outputs" / name
        report_path = directory / "report.json"
        if not report_path.exists():
            report_path = directory / "full_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        full_report_path = directory / "full_report.json"
        if full_report_path.exists():
            report = json.loads(full_report_path.read_text(encoding="utf-8"))
        summary = report.get("summary")
        if summary is None:
            summary = {key: report.get(key) for key in
                       ("complete", "score", "criterion_subset_score", "full_benchmark_score",
                        "new_generation_model_calls", "new_generation_tokens")}
            summary["generation_tokens"] = report.get("generation_budget", {}).get("tokens")
            summary["evaluation_tokens"] = report.get("evaluation_budget", {}).get("tokens")
        elif name == "compact_ifbench_mixed_v1_20261009":
            scores = [row["score_result"]["score"] for row in report["records"]
                      if row["score_result"].get("complete")]
            summary = {**summary, "mean_all_tasks": sum(scores) / len(report["task_ids"]),
                       "original_mean_denominator": "scored tasks only"}
        summaries.append({"run": name, "report_sha256": sha256(report_path.read_bytes()),
                          "full_report_sha256": (sha256(full_report_path.read_bytes())
                                                 if full_report_path.exists() else None),
                          "summary": summary})
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            if relative.parts[0] in {"answers", "records", "raw_responses"} or path.name in FILES:
                candidates.append((name, relative, path.read_bytes()))
    secrets = [value.encode() for name, value in os.environ.items() if len(value) >= 8
               and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    for name, relative, body in candidates:
        if any(secret in body for secret in secrets):
            raise ValueError(f"Credential value found in archive candidate {name}/{relative.as_posix()}")
    output.mkdir(parents=True)
    archive_path = output / "receipts_and_answers.zip"
    inventory = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, relative, body in candidates:
            filename = f"{name}/{relative.as_posix()}"
            info = zipfile.ZipInfo(filename, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, body)
            inventory.append({"file": filename, "bytes": len(body), "sha256": sha256(body)})
    manifest = {"archive": archive_path.name, "archive_sha256": sha256(archive_path.read_bytes()),
                "credential_scan": "exact environment credential values checked before writing",
                "runs": summaries, "files": inventory}
    (output / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    return {"runs": len(summaries), "files": len(inventory),
            "archive_bytes": archive_path.stat().st_size, "archive_sha256": manifest["archive_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(archive(Path(args.output).resolve())))


if __name__ == "__main__":
    main()
