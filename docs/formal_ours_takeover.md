# Ours formal experiment takeover

The resumable campaign is `tds_fix9`; its frozen launch is
`outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json`.
The source `tds_fix5` is kept unchanged. The current continuation reuses completed
receipts and reopens only blocked or missing scopes.

## Start or resume

Use a local credential file or environment variables. The file may be `.env`
style or JSON and must contain `JIT_BENCHMARK_API_KEY` and
`RESCORE_JUDGE_API_KEY`; values are never printed.

```powershell
$env:JIT_BENCHMARK_API_KEY = (Get-Content C:\secrets\jit.env.key -Raw).Trim()
$env:RESCORE_JUDGE_API_KEY = $env:JIT_BENCHMARK_API_KEY
python scripts/continue_formal_ours.py --mode run --interval 30
```

With a local file instead:

```powershell
python scripts/continue_formal_ours.py --mode run `
  --credential-file C:\secrets\jit.env `
  --runtime-archive C:\Users\chizh\Desktop\JIT-main-archive-20261011\caches\.runtime
```

The script reads the frozen transport settings from `tds_fix9/campaign.sqlite`,
verifies every registered dataset/evidence hash, restores missing `.runtime`
inputs from the archive when needed, and prints compact progress lines. If the
process is interrupted, rerun the same command; durable started slots are
recovered by the batch runner.

## Inspect and report

```powershell
python scripts/continue_formal_ours.py --mode status
python scripts/continue_formal_ours.py --mode report
```

`status` and `report` are local read-only commands and do not require API
credentials. Only `--mode run` loads and validates the local credential file.

The report command writes the provenance-aware recovery report under
`outputs/formal_ours_20261010/tds_fix9_recovery_report/` and the effective
evolution CSV/JSON under the same output tree. `tds_fix5_final_report/` remains
the authoritative baseline for utility, Agent Pool and structural-operation
figures.

Do not edit `tds_fix5`, the frozen launch/configuration, or registered input
files in place. A changed frozen input must be restored from an archive with a
matching SHA256 or placed in a new campaign with a new registration hash.

## DeepResearch Bench II continuation

The current tds_fix9 run intentionally preserves the DRBII inconclusive root
batch and does not reopen its downstream slots. After tds_fix9 is terminal,
create a separate continuation so its database snapshot stays auditable:

```powershell
python -m scripts.migrate_tds5_reuse `
  --source tds_fix9 `
  --output tds_fix10 `
  --sources deepresearch_bench_ii `
  --launch outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json
python scripts/continue_formal_ours.py --mode run `
  --campaign tds_fix10 `
  --launch outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json `
  --credential-file C:\secrets\jit.env
```

When the b0 small-judge repair has already selected a candidate, resume from
b1 with a fresh continuation that rebuilds b1 candidates. The
`--inclusive-first` flag is required for this case; it reopens the first
requested batch and discards stale no-op candidates while preserving b0
checkpoints and receipts:

```powershell
$env:JIT_BENCHMARK_API_KEY = (Get-Content C:\secrets\jit.env.key -Raw).Trim()
$env:RESCORE_JUDGE_API_KEY = $env:JIT_BENCHMARK_API_KEY
$env:JIT_MAS_MODEL_ATTEMPTS = '5'
$env:JIT_MAS_DISABLE_KEEPALIVE = '1'
$env:JIT_MAS_BATCH_META_CORRECTIONS = '2'
$env:JIT_MAS_ALLOW_EXTRA_CORRECTIONS = '1'
$env:JIT_MAS_BATCH_META_MAX_TOKENS = '4096'
$env:JIT_MAS_RETRY_BASE_SECONDS = '5'
$env:JIT_MAS_RETRY_MAX_SECONDS = '30'
$env:JIT_MAS_RETRY_JITTER_SECONDS = '1'
$env:MODULAR_AGENT_API_FAILURE_ACTION = 'exit'
python -m scripts.migrate_tds5_reuse `
  --source tds_fix12 --output tds_fix25 `
  --sources deepresearch_bench_ii --first-batch 1 --inclusive-first `
  --launch outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json
python -m scripts.run_drbii_repair `
  --launch outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json `
  --output tds_fix25 --batch-size 10 --model-attempts 5
```

This branch is isolated from the formal tds_fix5/tds_fix9 snapshots. It can be
resumed by rerunning the same repair command after a transient endpoint outage.
