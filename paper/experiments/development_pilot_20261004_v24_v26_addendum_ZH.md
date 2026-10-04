# v24–v26 运行审计与版本选择（2026-10-04）

本附录记录 v19 之后为提高终稿保真度而进行的配置验证。三轮均使用固定 source-EVO 任务：ResearchRubrics 1、DeepSearchQA 2、WritingBench 2，每轮 15 个三臂槽；不使用正式 VAL/TEST 结果，也不跨轮拼接分数。

## v24

- 配置：新增长文终稿保真提示；`max_inflight_requests=1`、`max_parallel=1`。
- 结果：15 槽封存；Direct 5/5、Ours 3/5、Native JIT 4/5 完成评分。
- 主要失败：`max_parallel=1` 与迭代团队的 `total_max_calls=null` 约束不兼容；另有 API connection error 和 Native JIT 未提交终稿。
- Ours 保守宏均值：`0.3901758119`；该轮不用于版本选择。

## v25

- 修正：恢复 `max_parallel=2`，保留 `max_inflight_requests=1`。
- 结果：15 槽封存；Direct 5/5、Ours 4/5、Native JIT 4/5 完成评分。
- 主要失败：配置仍遗漏 `team_max_calls=null` / `max_model_calls=null` / `max_tool_calls=null`，导致一项 Ours 团队在本地资源校验阶段失败；另有一项 Native JIT 消耗上限后未提交终稿。
- Ours 保守宏均值：`0.3901758119`；不替换 v19。

## v26（最终配置验证）

- 修正：补齐三个迭代共享账本的无限上限字段，同时保留 `max_parallel=2`、`max_inflight_requests=1`。
- 结果：15/15 槽生成和评分全部完成，三臂均为 5/5；没有生成失败或评分未完成。
- benchmark 均值（归一化）：ResearchRubrics Ours `0.500000`，DeepSearchQA Ours `0.562500`，WritingBench Ours `0.800000`。
- 全源开发宏均值：Direct `0.6554537624`，Ours `0.6208333333`，Native JIT `0.6357019811`。
- 该轮证明运行器配置和单并发请求门控可稳定完成，但没有超过 v19 的 Ours 宏均值，因此不替换推荐版本。

## 推荐版本与独立评分

保留 `outputs/development_pilot_20261004_exposed_v19` 及对应 v19 source-EVO campaign 作为当前开发推荐版本：Ours `0.6255243272`，Direct `0.5546392824`，Native JIT `0.5349074817`；29/30 槽完成评分，Ours 10/10。`exposed_v19` 中的 4 个 TEST 题已污染开发流程，只能作为 prompt 迭代诊断，不能写成 clean formal TEST。

独立 GPT-5.6-sol 重评分目录为 `outputs/independent_gpt56_dev_v19_20261004`。最终完成 18/30 槽：IFEval/IFBench 12/12、DeepSearchQA 3/6、WritingBench 3/6；ResearchRubrics 与 DeepResearchBench II 因网关超时未完成。已完成槽的独立结果只能作为部分审计，不能替代完整六源比较，也不应把未完成槽按零分解释。

独立评分已完成槽的逐项记录（归一化分数）如下；缺失项保持未完成状态：

| 来源 | 任务 | Direct | Ours | Native JIT |
|---|---|---:|---:|---:|
| IFEval | 1203 | 1.000000 | 1.000000 | 0.000000 |
| IFEval | 1246 | 1.000000 | 1.000000 | 1.000000 |
| IFBench | 13 | 0.000000 | 0.000000 | 1.000000 |
| IFBench | 22 | 0.000000 | 1.000000 | 0.000000 |
| DeepSearchQA | d042… | 1.000000 | 0.857143 | 未完成 |
| DeepSearchQA | ba5c… | 0.000000 | 未完成 | 未完成 |
| WritingBench | 335 | 未完成 | 未完成 | 0.755556 |
| WritingBench | 433 | 0.733333 | 0.177778 | 未完成 |

其中 IFEval/IFBench 使用 pinned author checker；其他来源由指定的 GPT-5.6-sol gateway 评分。网关超时、`evaluation_started` 无结果和不完整槽均保留在评分目录中，没有复采或补零。

所有 v23–v26 失败、耗用和配置哈希均保留在各自 `outputs/` 目录；正式论文主表只保留完整、预先固定且可解释的单一推荐版本。源码改动包含 `jit_mas/execution.py` 的终稿质量审计提示，聚焦保留长文的标题、段落、列表、表格、引用和章节顺序；测试 `tests/jit_mas/test_execution_quality.py` 与 `tests/jit_mas/test_iterative_execution.py` 共 142 项通过。
