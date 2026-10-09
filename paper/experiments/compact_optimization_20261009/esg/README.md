# Targeted ESG repair

The experiment starts from the preserved compact v5 answer, freezes its disclosed subjective component scores, and repairs arithmetic locally. The v6 report corrects category roll-ups, contributions, sensitivity and ranking and fixes source transcriptions. The v7 patch adds facts already present in the public source archive. No v6/v7 generation or network retrieval calls occur. Inputs, component judgments and weights remain visible; arithmetic reproducibility does not establish the judgments as empirical measurements.

`render_arithmetic_v6.py` reproduces the frozen calculation table and ordering without dependencies beyond Python's standard library:

```powershell
python paper/experiments/compact_optimization_20261009/esg/render_arithmetic_v6.py --input-answer outputs/targeted_rr_esg_compact_v5_20261009/answer.md --output outputs/esg_arithmetic_replay
```

`repair_evidence_v7.py` applies the exact local enrichment to the archived v6 answer and verifies public source text hashes before taking 0-based, end-exclusive source slices:

```powershell
python paper/experiments/compact_optimization_20261009/esg/repair_evidence_v7.py --input-answer outputs/targeted_rr_esg_arithmetic_v6_20261009/answer.md --source-archive outputs/repair_subset_20261009/public_source_archive --output outputs/esg_evidence_replay
```

The output includes the complete answer, unified diff, source URLs/archive locators, original text hashes, exact slice hashes, calculation checks and zero-generation-call accounting. Output directories are immutable. The source PDFs are not required in the Git archive; replay requires locally retained JSON text receipts whose hashes match the manifest.

For a Git artifact replay, extract the packaged experiment-results ZIP, then use the contained `targeted_rr_esg_arithmetic_v6_20261009/answer.md` as `--input-answer`. It must have normalized-text SHA256 `4a30cfc25d40e44455210217ee3b5df102ddff7141ad7ffa3df55dffa8346f69`. Source receipt filenames are listed by `source_manifest.json` and `SOURCE_SLICES`; point `--source-archive` to the local public-source archive containing those receipts. If only packaged selected-source excerpts are available, they support auditing the claims and hashes but do not reconstruct full source text for the replay command.

Evaluation reports are separate artifacts. Seven-criterion subset results do not represent a full ResearchRubrics score. Full evaluations cover all 28 original native criteria of the single ESG task, with unchanged evaluator inputs. The frozen scoring order is H&M, Patagonia, Fast Retailing, independent of any evaluator-specific preferred order.
