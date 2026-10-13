# 高分版本实验与完整记录

本轮把完整 JIT-MAS 的研究/写作实验与 IFEval、IFBench 的 compact 生成放在同一个实验目录中。原始逐题记录保留预算、评分、失败和状态变化；汇总脚本输出官方指标、效率指标和可直接画图的 evolution curve。脚本均从仓库根目录执行，Python 环境为 `.venv`。

## 完整 JIT-MAS：进化与 TEST

入口为 `scripts/run_high_score_native_experiment.py`，参数如下。

| 参数 | 含义 |
|---|---|
| `--config` | JSON/YAML 的 `MASConfig`；要求 `backend: native_jit`、完整的 meta/global/local/exec/judge 模型配置与对应环境变量 |
| `--plan` | JSON 任务计划；相对路径以计划文件所在目录为基准 |
| `--output` | 新实验目录；运行时必填 |
| `--check` | 只校验配置、任务 ID、数据、证据、初始状态和 pinned checker，不请求模型、不创建实验输出 |
| `--resume` | 以完全相同的身份恢复现有输出，跳过已有终结记录 |

模型密钥只从配置指定的 `key_env` 读取。原生执行还要求配置显式允许 `unsafe_local`，其含义见 `MASConfig.check_native()`。`--check` 不检查密钥值或 endpoint 连通性。

计划中每个 benchmark 支持 `name`、`data`、`evolution_ids`、`test_ids`，可选 `evidence_dir`、`checker_source` 和 `initial_snapshot`。IFEval、IFBench 必须指定固定 checker 源码目录。`initial_snapshot` 必须是完整的 `ExperienceSnapshot` JSON，只在新建该 benchmark 的状态库时播种。

下面是存放在 `outputs/example_experiment/plan.json` 的完整 40 轮 WritingBench 进化、3 题 TEST 样例。任务 ID 来自本轮实际计划。

```json
{
  "benchmarks": [
    {
      "name": "writingbench",
      "data": "../../dataset/writingbench/benchmark_all.jsonl",
      "evolution_ids": [
        "writingbench:1", "writingbench:2", "writingbench:3", "writingbench:4",
        "writingbench:5", "writingbench:6", "writingbench:7", "writingbench:8",
        "writingbench:9", "writingbench:10", "writingbench:11", "writingbench:12",
        "writingbench:13", "writingbench:14", "writingbench:15", "writingbench:16",
        "writingbench:17", "writingbench:19", "writingbench:20", "writingbench:21",
        "writingbench:22", "writingbench:23", "writingbench:24", "writingbench:25",
        "writingbench:26", "writingbench:27", "writingbench:28", "writingbench:29",
        "writingbench:30", "writingbench:31", "writingbench:32", "writingbench:33",
        "writingbench:34", "writingbench:35", "writingbench:36", "writingbench:37",
        "writingbench:38", "writingbench:39", "writingbench:40", "writingbench:41"
      ],
      "test_ids": ["writingbench:18", "writingbench:104", "writingbench:203"]
    }
  ]
}
```

两组 ID 必须互不重叠且属于所选数据集。指定证据目录时，目录中的 `manifest.json` 必须覆盖所有选中题，其清单必须属于同一数据集。runner 用原证据 manifest 的全部任务调用证据加载器，再把本次选中任务替换进 pipeline；无需裁剪或改写证据清单。

```powershell
.\.venv\Scripts\python.exe scripts/run_high_score_native_experiment.py `
  --config outputs/high_score_experiment_20261009/native_config.json `
  --plan outputs/high_score_experiment_20261009/evolution_plan.json --check

.\.venv\Scripts\python.exe scripts/run_high_score_native_experiment.py `
  --config outputs/high_score_experiment_20261009/native_config.json `
  --plan outputs/high_score_experiment_20261009/evolution_plan.json `
  --output outputs/high_score_experiment_20261009/native_evolution
```

进化按计划顺序调用 `pipeline.run("evolve", [task_id])`，评分、归因和校验通过后才能提交可复用经验与 agent 更新。TEST 在该 benchmark 的最终状态上调用 `run_task(..., mode="evaluate", attribution=False)`。每个 benchmark 有独立状态库，某题失败后继续下一题，失败调用和预算也保留。40 轮表示尝试了 40 个任务，不表示成功提交了 40 个新状态版本。

## 原生记录、checkpoint 与日志

原生输出布局为：

```text
native_evolution/
  manifest.json
  records.jsonl
  report.json
  slots/<slot_id>/started.json
  slots/<slot_id>/record.json
  slots/<slot_id>/stdout_stderr.log
  writingbench/runtime_split.json
  writingbench/experience.sqlite
  writingbench/runs/<run_key>/
```

`manifest.json` 保存配置、数据/证据/checker/初始 snapshot 身份，代码与 runner hash，以及重试和 judge envelope 版本。`slot_id` 绑定 benchmark、phase、轮次和 task ID。`records.jsonl` 是逐题汇总的唯一原生输入，完成一题即追加并 flush/fsync；同一题的 `record.json` 提供独立副本。

| 逐题字段 | 含义 |
|---|---|
| `benchmark`, `phase`, `round_index`, `task_id`, `status` | 任务身份、阶段、1 开始的轮次、终结状态；原生状态包括 complete/incomplete/failed/interrupted |
| `started_at`, `finished_at`, `latency_seconds` | 开始结束时间与任务墙钟耗时；正常记录含生成、评分、归因和提交，client 关闭耗时不计入 |
| `outcome.evaluation` | 完整评分，包括 rubric verdict 与 `raw` 中的 benchmark 原始指标 |
| `budget` | `tokens`、`model_calls`、`tool_calls`、`cost`、`by_stage` 和逐次 `records` |
| `failure` | 异常类型、message、traceback 与 `run_failure`/`provider_failure`，包含失败 run 路径和可用预算 |
| `snapshot_before_hash`, `snapshot_after_hash` | 本轮开始结束的完整可复用状态 hash |
| `experience_version_before`, `experience_version_after` | 持久状态版本；未提交时通常不变 |
| `pool_before`, `pool_after` | 原始 Pool snapshot，包括 profile、观察、更新和结构历史 |
| `agent_pool_size_before`, `agent_pool_size_after` | 实際有效 Pool 数量；尚未初始化的原始空池仍按有效 seed agents 计数 |
| `operations_requested`, `operations_applied` | 通过验证的请求操作与真实提交操作；applied 只来自结构历史中新出现的 operation ID |
| `harness_before`, `harness_after`, `harness_modification_count` | agent 的 prompt/skills/memory/tools/策略/通信/harness 前后；仅 version 增加不计修改，新增/删除 agent 单独记录 |
| `harness_artifact`, `harness_repair_count`, `harness_meta_trajectory`, `harness_failed_attempts` | 本题运行 harness、实际构造/repair 过程及失败构造尝试；与持久 agent 更新次数分开 |

Add/Delete/Split/Merge 之外，还保留 prune/specialize/reorganize。空列表表示明确没有操作，`null` 表示该测量不可得。harness 修改包括持久 agent memory 或其他上下文内容变化，不仅是生成代码改写。

汇总脚本另外读取每轮 `meta_evolution.json` 的调用审计。`evolution_operation_attempts` 保存每次整合尝试的编号、验证结果与错误、提案 operation ID/类型/目标 agent ID/profile ID/内容 hash。`operation_counts_proposed` 统计所有尝试中的操作，`operation_counts_proposed_unique` 在同一任务内按 operation ID 去重；两次失败修正使用相同 ID 时只算一个独立提案。`operation_counts_validated` 对应原 `operations_requested`，`operation_counts_applied` 对应真实提交。验证失败的提案不会计入 validated 或 applied，各类操作的逐轮与累计列分别保留。

每个成功提交的完整状态保存在 `experience.sqlite` 的 `snapshots` 表中，`evolution_commits` 和 `task_runs` 记录提交及任务状态。轮次与 snapshot version 可能不同；用该轮记录的 `experience_version_after` 读取 checkpoint：

```python
from jit_mas.experience import ExperienceStore

store = ExperienceStore("outputs/example_experiment/writingbench/experience.sqlite", read_only=True)
try:
    checkpoint = store.snapshot(version=1)
    checkpoint_json = checkpoint.model_dump_json(indent=2)
finally:
    store.close()
```

本入口没有自动生成每 5 轮的独立 checkpoint JSON，完整历史版本在 SQLite 中。逐题 `runs/<run_key>/` 保留生成/执行/评分/归因的文件，例如 `run_manifest.json`、`frozen_plan.json`、`harness.json`、`execution.json`、`call_trace.json`、`submission.json`、`evaluation.json`、`attribution.json`、`agent_evolution.json`、`meta_evolution.json`、`complete.json`、`failure.json`、`budget.json`；阶段未走到时对应文件可能不存在。

## 失败、恢复和单独诊断

生成和评分可能成功、随后归因或提交失败，此时逐题 `status` 仍为 failed，`outcome` 可能为空，评分已经存在于 `failure.run_failure.run_dir/evaluation.json`。汇总脚本读取该文件恢复已发生的评分，并标记 `evaluation_recovered_from`；失败状态、预算和未提交操作都保留。查看曲线时同时使用 `evaluation_complete` 与 `complete`，区分已评分和完整进化成功。

恢复命令只增加 `--resume`：

```powershell
.\.venv\Scripts\python.exe scripts/run_high_score_native_experiment.py `
  --config outputs/high_score_experiment_20261009/native_config.json `
  --plan outputs/high_score_experiment_20261009/evolution_plan.json `
  --output outputs/high_score_experiment_20261009/native_evolution --resume
```

配置、计划、数据、代码、runner 和重试身份必须与原 manifest 相同。已出现在 JSONL 或 `record.json` 中的任务不会再次请求模型。只有 `started.json` 的任务被标为 interrupted，从 task journal 或匹配 run 的 complete/failure/budget 文件恢复已有结果与成本记录；不自动重跑或补交该轮。更改配置或代码、重新尝试终结题应使用新实验目录。

同题另跑诊断时保存独立 `run_id` 和 `phase: diagnostic`。通用 API 可在对应 split 允许的 mode 下调用 `run_task(task_id, frozen_snapshot, mode="evolve", attribution=False)`，并验证运行前后 `digest(store.snapshot())` 相同；它只执行和评分，不提交进化。diagnostic 属于记录阶段标签，当前原生 CLI 只直接提供 evolution/test。给 CLI 的 `evolution_ids` 会进入真实 evolution，不会自动变成诊断。本轮 RR 同题诊断以单独输出目录记录。

恢复时如果 `records.jsonl` 的最后一行因中断只写入一部分，runner 将该尾部保存在 `records.jsonl.partial-<hash>`，并从 slot 副本恢复记录；中间损坏或已换行的损坏记录仍会报错。

## Compact：IFEval 与 IFBench

入口为 `scripts/run_compact_benchmark_subset.py`，使用单独的 exec 配置。每题最多一次生成请求，禁用 SDK/provider 重试，之后调用 pinned checker。它不更新 Agent Pool；对应曲线的 Pool 或结构操作为 `null`。

```powershell
.\.venv\Scripts\python.exe scripts/run_compact_benchmark_subset.py `
  --config configs/compact_generation.example.yaml `
  --benchmark ifbench --dataset dataset/ifbench/IFBench_test.jsonl `
  --checker-source outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d `
  --task-id ifbench:0 --task-id ifbench:17 --task-id ifbench:10 `
  --literal-construction --numeric-construction --numeric-layout template --positional-construction `
  --output outputs/example_compact_ifbench
```

`--config`、`--benchmark`、`--dataset`、`--checker-source`、`--output` 必填；`--benchmark` 可选 ifeval/ifbench。题目使用可重复的 `--task-id` 和/或 `--count N`，后者取数据加载顺序中的前 N 题，两者合并去重。`--numeric-layout` 为 array/template，默认 array。三个 construction 开关按公开约束选用支持的结构。输出目录必须不存在，当前 compact CLI 不提供 resume。

输出 `manifest.json`、`report.json`、`records/<task_hash>.json`、`answers/<task_hash>.txt`、`raw_responses/<task_hash>.txt`；report schema 为 `compact-generation-run-v1`。逐题记录包括配置/题目/答案 hash、完整预算、生成完成状态、latency 与 `score_result`。细节见 [compact_benchmark_generation.md](compact_benchmark_generation.md)。

## 官方 strict/loose 补分

入口 `scripts/score_compact_official_metrics.py` 对已保存的 compact 答案运行官方 strict/loose 两种 checker，不请求模型，也不修改答案。`--report`、`--dataset`、`--checker-source`、`--output` 都必填；输出必须是区别于原 report 的新文件。

```powershell
.\.venv\Scripts\python.exe scripts/score_compact_official_metrics.py `
  --report outputs/example_compact_ifbench/report.json `
  --dataset dataset/ifbench/IFBench_test.jsonl `
  --checker-source outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d `
  --output outputs/example_compact_ifbench/official_metrics.json
```

脚本核对 dataset、private record、答案和 report hash，保存 checker identity、逐条 instruction verdict。schema 为 `compact-official-metrics-v1`，四项指标是 Prompt/Instruction level × strict/loose Accuracy。`summary.official_metrics` 仅在相应评分全部完成时给值；`scored_metrics` 是已评分项的均值，`all_tasks_zero_for_failures` 使用完整任务/指令分母。补分本身 API calls、tokens、cost 为 0，checker latency 单独保留。

统一汇总保留 compact report 的完整 `task_ids`：缺少逐题回执的任务标记为 `not_attempted`，仍进入完整任务分母；有补分时保留对应的指令分母。缺失回执的效率测量保持 `null`。

## WritingBench 固定清单方法

本轮另将此前的 `writingbench-checklist-v2` 方法用于同三题重新生成和 native rubric 评分，输出在 `outputs/high_score_experiment_20261009/compact_writingbench`，逐题标签为 `phase: test`、`method: compact-writing`。保持原 system prompt、每题固定内容清单和模型参数，每题一次生成调用，后接 native judge；完整 provider 重试次数也保存。它不更新经验或 Agent Pool。

目录中的 `frozen_method/` 保留原方法源码副本、原 prompt/每题 guidance，`manifest.json` 绑定源代码、prompt、数据、runner 与模型配置指纹。`answers/` 和 `raw_responses/` 保存完整原文，`records/<task_hash>/` 保存 started、record、call_trace 和日志，`records.jsonl` 用于统一汇总。逐题 `budget` 已合并生成和评测所有调用；`budgets.generation`、`budgets.evaluation` 保留原阶段账本，`latency_by_stage` 保留各阶段时间，`generation_finish_reason` 与 `status` 保留截断或其他不完整状态。

`comparison.json`/`comparison.md` 对照此前通用 one-pass、此前 checklist-v2 和本轮 fresh checklist-v2，包含各题得分、总生成+评测 token、调用和可得 latency；完整 JIT-MAS 的三题 TEST 结束后可补齐同题对照。统一汇总按 method/run/phase 分组。

## 统一汇总、费用与 evolution curve

```powershell
.\.venv\Scripts\python.exe scripts/summarize_high_score_experiment.py `
  outputs/high_score_experiment_20261009
```

位置参数是实验根目录，`--output` 可指定汇总目录，默认 `<root>/summary`；可选 `--rates rates.json`。脚本递归读取原生 `records.jsonl` 与 compact report，避免重复计入逐题副本；命名为 `official_metrics.json` 的 strict/loose 补分会按 source report 与 task ID 合并。可以在实验进行中重复运行以刷新已落盘题，尚未结束的当前题不计入此次汇总。

| 文件 | 内容 |
|---|---|
| `efficiency_summary.json` | 总量、benchmark 与 run/method/phase 条件分组：tokens、model calls、API attempts/retries/failures、工具调用、latency sum/mean/p50/p95、任务失败率、阶段预算、费用 |
| `benchmark_metrics.json` | 各官方指标及观察/完整任务分母、完成数、评分数；TEST/evolution/diagnostic 分组独立 |
| `evolution_operations.json` | 各 run/method/phase 的全部提案尝试、任务内独立提案、通过验证请求和真实提交操作数量 |
| `run_summary.md` | 可直接查看的指标、效率摘要与曲线路径 |
| `evolution_curve/<benchmark>/<run_id>_<method>.jsonl` | 完整逐轮结构化数据 |
| 同路径 `.csv` | 扁平列用于绘图，嵌套对象/数组保存为 JSON 字符串 |
| `evolution_curve/<benchmark>/<run_id>_<method>_<phase>.jsonl` 与 `.csv` | 单独阶段的曲线，累计预算和操作数量在该阶段内重新计算 |

各 benchmark 汇总指标：DRBII 的 Overall、InformationRecall、Analysis、Presentation；DSQA 的 Precision、Recall、F1；RR 的 compliance；WritingBench 的 nativeMean 和 normalized_score；IFEval/IFBench 的四项 strict/loose Accuracy。WritingBench 原始均分范围为 1–10，归一化分数是 `(nativeMean - 1) / 9`。DSQA 的 fully-correct 和其他 evaluator 原始字段仍在逐题 `outcome.evaluation.raw` 中。

逐轮曲线包括 `score_per_round`、`performance_*`、`tokens`/`cumulative_tokens`、calls/attempts/retries、latency、阶段预算、Pool 数量、proposed/proposed_unique/validated/applied 各类操作计数、harness hash/修改/repair 和成功/失败状态。`round_index` 在不同 phase 内从 1 重新开始，`sample_index` 是排序后的整体序号。本轮 WritingBench 的 `_evolution` 文件只有 40 行，`_test` 文件只有 3 行；原 combined 文件保留 43 行以兼容原路径。每个 run/method/phase 文件各自累计，不将 TEST 预算加到纯进化曲线中，也不把同题 diagnostic 合并成原 TEST 成绩。缺少测量保持 `null`，明确的零保留 0。

预算中的 `model_calls` 是 ledger 记录的模型调用数，`api_attempts` 来自 transport attempt 审计，可能因重试更多；失败率分别记录任务失败与 API attempt 失败。tokens 按每次预算记录中的 input/output 合计，provider usage 不可得时保留估算及其标记，应读取逐次记录判断实际 usage 与估算。未知 attempts 或不完整 usage 不补造为零。

当前 ledger 的 `cost` 未提供真实账单金额，因此 actual_cost 保持 `null`。需要估价时提供实际费率 JSON，例如下面的数值仅演示文件格式：

```json
{
  "currency": "USD",
  "models": {
    "provider-returned-model-id": {
      "input_per_million": 0.10,
      "output_per_million": 0.20
    }
  }
}
```

`models` 使用逐次 request 的 response_model，可添加 `default` 对象作为未匹配模型的费率。`--rates` 只产生 estimated_cost，不替代 actual_cost；缺少任一调用费率或 token 记录时对应估价保持 `null`。由于失败调用可能使用预算估算，estimated_cost 也不等同于 API 账单。

## 离线验证

新增记录与补分能力可用 focused tests 验证，无需 API：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_run_high_score_native_experiment.py `
  tests/jit_mas/test_high_score_summary.py `
  tests/jit_mas/test_score_compact_official_metrics.py -q
```
