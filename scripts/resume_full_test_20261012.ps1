# Resume the two benchmark TEST runs after setting JIT_BENCHMARK_API_KEY.
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($env:JIT_BENCHMARK_API_KEY)) {
    throw "Set JIT_BENCHMARK_API_KEY in this PowerShell session before running this script."
}
$python = ".\.venv\Scripts\python.exe"
$config = "outputs/full_test_results_20261012/composure_full_test_config.yaml"
& $python scripts/run_targeted_compact_sharded.py --config $config --plan outputs/targeted_optimization_20261010/full_split_adapted_20261012/deepsearchqa_full_test_plan.json --output outputs/full_test_runs_20261012/dsqa_rerun --shards 8 --concurrency 2 --resume
& $python scripts/run_targeted_compact_sharded.py --config $config --plan outputs/targeted_optimization_20261010/full_split_adapted_20261012/deepresearch_bench_ii_full_test_plan.json --output tdrbii_full_test_20261012_rerun --shards 8 --concurrency 8 --resume
& $python scripts/build_full_test_result_table.py
