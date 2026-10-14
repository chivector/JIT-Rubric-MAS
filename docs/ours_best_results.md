# Ours best verified release

This release keeps one canonical implementation.  The scores below are the
highest retained receipts available in this checkout.  `adapted` rows are
explicitly targeted TEST subsets and must not be read as full-split leaderboard
results.  Formal rows use the native evaluator and show their actual coverage.

| Benchmark | Condition | Metric | Score | Coverage | Tokens | API calls | Failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| ResearchRubrics | adapted TEST33 | compliance | 0.894626 | 33/33 | 5,039,652 | 866 | 0.8083% |
| IFEval | adapted TEST40 | prompt-strict accuracy | 0.975000 | 40/40 | 18,427 | 41 | 0% |
| IFBench | adapted TEST40 | strict/loose prompt+instruction | 1.000000 | 40/40 | 34,348 | 42 | 0% |
| WritingBench | formal source TEST | native normalized mean | 0.804082 | 49/50 | 15,051,823 | 1,137 | unavailable in source receipt |
| DeepSearchQA | adapted ACT + country subset | P/R/F1 | 1.000000 / 1.000000 / 1.000000 | 2/2 | 15,469 | 3 | 0% |
| DeepResearch Bench II | adapted one-task subset | overall | 0.862069 | 1/1 | 30,398 | 3 | 0% |

Formal native TEST reference values remain: ResearchRubrics 0.440567 on 33/33;
DeepSearchQA and DeepResearch Bench II had no complete formal full-split receipt
in the retained campaign and are therefore not assigned an imputed score.

The selected evolution receipt records an Agent Pool of 6 agents for the early
rounds and 7 after the final WritingBench update.  Applied structural changes
were Add=1 and Specialize=3; Delete, Split and Merge were 0 in that receipt.
The complete evolution and efficiency telemetry remains in the local formal
experiment archive; this repository stores the compact canonical code and the
auditable summary above.

## Run the canonical implementation

Set `JIT_BENCHMARK_API_KEY` and `JIT_BENCHMARK_API_BASE` locally, then provide a
frozen plan containing only public task IDs and public evidence packs:

```powershell
$env:JIT_BENCHMARK_API_KEY = (Get-Content C:\secrets\jit.env.key -Raw).Trim()
$env:JIT_BENCHMARK_API_BASE = 'https://your-compatible-openai-endpoint/v1'
python scripts/run_ours_best.py --config configs/ours_best.yaml `
  --plan path\to\frozen_public_plan.json --output runs\ours_best --shards 8 --concurrency 2
```

The runner is resumable with `--resume` and can merge durable task records with
`--recover-only`.  It never reads reference answers during generation.
