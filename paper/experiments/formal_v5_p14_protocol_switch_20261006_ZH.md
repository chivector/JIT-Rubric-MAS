# v5 p14 协议切换记录（2026-10-06）

## 结论

p14 是按旧的三 benchmark 联合进化协议启动的运行。现根据实验计划改为“ResearchRubrics、DeepSearchQA、WritingBench 分别从空状态独立进化，再迁移到 TEST”的协议，因此 p14 在完成新的正式实验边界前停止。p14 的产物和成本完整保留，但不作为论文最终成绩、正式 checkpoint 选版或 superiority 证据。

本次停机是协议切换，不是对已有题目或答案的质量筛选，也没有重采样、补题、修改原 journal、receipt、registration、membership 或配置。停止后没有继续发起 API 请求。

## p14 停止时状态

运行目录为 `outputs/formal_v5_evo_val_20261005_p14`，原协议登记见 [formal_v5_p14_run_20261005_ZH.md](formal_v5_p14_run_20261005_ZH.md)。journal 的终态汇总如下：

| 阶段 | complete | failed | started | pending | 总槽位 |
|---|---:|---:|---:|---:|---:|
| EVO | 4 | 8 | 1 | 167 | 180 |
| VAL | 27 | 3 | 0 | 420 | 450 |

journal 记录 **819 calls / 11,673,572 tokens**，其中 816 calls / 11,600,101 tokens 为 provider 实际回报，3 calls / 73,471 tokens 为估算值。该数字是 p14 journal 的混合记录，不是新独立实验的预算。

唯一未封账的 started EVO 槽位为：

- slot：`evo:run0:12:6847465956a0f6376a6053aa`
- receipt：`outputs/formal_v5_evo_val_20261005_p14/run0/researchrubrics/6b5100967e16a745585da4c1bdefa24ee121de9e90379a57dadba0cfa6c55f82`
- receipt 已有 `run_manifest.json`、`call_trace.json`、`evaluation.json`、`execution.json` 和 `planning_calls.json`；没有 `complete.json`、`failure.json` 或独立 durable `budget.json`。
- `run_manifest.json` 显示 `experience_version=4`、`experience_hash=41f8a4060a16bf1d69c8943e9df6f0aa3924ae2936241730efacb5191cf08288`，没有 submitted/evaluated 时间。

该 started receipt 对应的服务端消耗属于**可恢复但尚未完整封账的 unknown server cost**。不能按零成本、失败零调用或已成功进化处理；也不能重新生成该槽位来替换它。p14 的其他失败和已记录成本同样只作为历史运行开销保留。

## 状态库边界

p14 停止时 live store 只读核对结果为：

- experience store version：`4`
- state hash：`41f8a4060a16bf1d69c8943e9df6f0aa3924ae2936241730efacb5191cf08288`
- 经验条目：3
- Agent Pool version：4
- profile 数量：6
- 已记录结构操作：0
- snapshots：5
- 上述 started EVO 没有对应的新 evolution commit，task run 仍为 `started`，baseline version 为 4。

旧 joint p14 state、Pool、checkpoint、经验条目和临时 harness 不得迁移到新的独立 benchmark 实验。否则会把联合协议的历史经验混入独立 source × run 轨迹，破坏从零初始化和来源隔离。

## 新正式实验范围

新实验采用 [experiment_plan_v5_independent_ZH.md](experiment_plan_v5_independent_ZH.md) 和 [independent_protocol_v5.json](independent_protocol_v5.json)：

1. ResearchRubrics、DeepSearchQA、WritingBench 各自建立独立的空 experience store 和 Agent Pool；三个来源之间不共享、不合并、不接续经验。
2. 每个来源运行 3 条固定 run，分别完成 20 EVO、5 个 checkpoint 和每个 checkpoint 10 道 VAL；VAL 只读，不向经验库回流。
3. 每个来源独立完成选版后，将 3 个来源的 selected 状态迁移到 TEST；DeepResearch Bench II、IFEval、IFBench 只作为迁移目标，不参与 source 进化。
4. 构建 MAS 时只预测 rubric、选择/适配 full agent harness prototype，并记录临时创建；不做双层归因、不更新长期 Pool。取得完整 judge 反馈后，才进行全局与局部双层归因及进化，并在该阶段执行 Pool 的 Add、Delete/Prune、Split、Merge、Specialize、Reorganize。
5. 新实验必须使用新的 registration、独立状态库和完整身份 fingerprint；不得用 p14 的 checkpoint、selection 或 partial score 拼接正式结果。

## 安全导出清单

为便于论文复核，可导出以下最小证据集：

- registration、protocol/config 和代码 fingerprint；
- EVO/VAL journal、每个 checkpoint 的文件及 state hash；
- 每个 task receipt 的 `run_manifest.json`、`budget.json`、`failure.json` 或 `complete.json`；
- 选版记录、迁移矩阵、最终评分汇总和成本审计表。

导出包应排除答案正文、judge reasoning、私有 rubric 内容和所有 API 凭据。p14 原始目录仍作为协议切换诊断保留，并与新独立实验目录分开归档。
