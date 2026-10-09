# Saved independent TEST subset rescoring

`scripts/rescore_saved_test_subset.py` scores existing sealed independent-campaign answers in a new versioned directory. It never constructs a team, executes an actor, updates experience, or replaces the source campaign's submissions and receipts.

```powershell
$env:RESCORE_JUDGE_API_KEY = '<set locally>'
.venv\Scripts\python.exe -m scripts.rescore_saved_test_subset `
  --campaign C:\J\r17_rr_recovery_v4 `
  --output outputs\rr_scoring_recovery_v1 `
  --target researchrubrics --count 1 --recover-prejudge-path-error
```

The judge credential variable comes from the original registered judge model's `key_env`; load the appropriate local environment before launching. IFEval and IFBench use their pinned author checker and need no model request. Model-scored benchmarks use the repository's benchmark evaluator adapters; this is not a claim of official leaderboard reproduction.

`--count` selects the first submitted target slots in the original immutable SQLite ordinal order. Generation failures are retained in the source inventory and skipped because they contain no answer. The command verifies the complete source seal, slot result and submission hashes, selected source checkpoint, source EVO provenance, pinned dataset bytes, and any original evaluation receipt. The runtime judge context follows `recovery_runtime_provenance.json` when present. Registration records the current scorer code, evaluator identity, effective configuration, transport environment, source identities, original budgets, and answer bytes before any judge call.

Previous known judge usage is added to the original generation usage, so the new evaluation receives only the remaining original call/token/tool allowance. Unknown or unsettled previous evaluation costs fail closed. `--recover-prejudge-path-error` is an explicit diagnostic exception for the confirmed legacy bug: a string `submission_path` passed to `_read`, which attempted `path.read_text()` before constructing a judge. Only an intact `evaluation_failed` / `AttributeError` receipt with exactly `{"usage_unknown": true}` qualifies. The new provenance records the inferred zero judge calls; the original unknown budget remains unchanged. This option does not repair the historical accounting record or authorize recovery of another exception.

The output directory must not already exist and must be outside the source campaign. Each attempt writes a started marker before evaluation and an immutable receipt afterward. Completed or incomplete output cannot be reused by this command; inspect it and explicitly register a further version rather than silently rerunning. An unresolved original started judgment without a receipt is rejected. Keep every prior attempt and its cost when planning a subsequent recovery.

Outputs are `registration.json`, `started/`, `evaluations/`, and `summary.json`. Subset scores are diagnostic and must not be reported as complete benchmark means. A new recovery is required to finish the remaining registered inventory.
