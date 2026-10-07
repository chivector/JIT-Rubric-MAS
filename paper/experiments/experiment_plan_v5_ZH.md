# 六 benchmark 评测：三个源 benchmark 分批独立进化 v6

2026-10-06 修正：**ResearchRubrics、DeepSearchQA、WritingBench 各自从零进化；每条轨迹 40 道 EVO，分为 8 批，每批 5 题之间不做进化。每批结束后产生 3 个候选，由固定 VAL 选 1 个进入下一批。** DeepResearch Bench II、IFEval、IFBench 保留迁移 TEST。完整规则见 [独立进化方案](experiment_plan_v5_independent_ZH.md)。本文件保留原路径，内容已切换为 v6；[旧 v5 机器协议](independent_protocol_v5.json) 与其运行记录退役并保留历史，不适用于新实验。

## 状态与方法边界

每个 `benchmark × run` 使用独立空经验库和独立 Agent Pool 状态，共九条轨迹。初始持久快照为空；首次构建时各轨迹使用相同的六个未进化角色原型（writer、searcher、critic、planner、analyst、generalist），其 memory、source_task_ids 和 evidence 均为空。“从零”指没有学得经验或历史，不表示没有初始 harness。不同来源、不同 run 不共享、不合并、不接续学得的 meta-agent 经验或 full agent harness prototypes；模型底座、初始化规则、执行机制与资源上限匹配。

**构建 MAS 时不做双层归因，也不更新长期 Pool。** Meta 根据当前公开 query 预测 rubrics，选择和适配现有完整 harness；没有合适原型时临时创建，记录原因、完整 profile、来源和使用情况。执行时不合并或删除 Pool 成员。

**批次执行结束后才双层归因与进化。** 5 题均使用同一冻结经验与 Pool 状态，记录执行、judge 反馈及临时 harness，不在题间提交更新。Meta 根据有完整反馈的题目及其全部 rubrics 做全局归因，再依据 RubricGraph、职责分配及评测对齐，把相关 rubrics 分配给各 agent。Agent 反思自身 context、tools、memory、skills 与 harness 策略；Meta 汇总 5 题信息，从同一批输入状态生成 3 个候选更新，包括 Pool 的 Add、Delete/Prune、Split、Merge、Specialize、Reorganize，以及是否保留临时 harness。候选互不接续；Pool 可扁平或层次化，不预设固定结构。

## EVO 与选版

| 独立来源 | 每条轨迹 EVO | 每个候选 VAL | 本源 TEST |
|---|---:|---:|---:|
| ResearchRubrics | 40 | 10 | 33 |
| DeepSearchQA | 40 | 10 | 50 |
| WritingBench | 40 | 10 | 50 |

- 每个来源运行三个固定题序：run 0/1/2，对应 `20261001/20261002/20261003`；每条轨迹从空库开始。
- 每批 5 道 EVO 使用同一冻结状态，题间不更新经验或 Pool；批次结束后，从该状态及本批记录生成 3 个候选。失败仍消耗位置，不补题、不重采样。
- 每个候选只评本来源固定 10 道 VAL。全部 30 个候选 VAL 槽位终态后选 1 个完整状态，再执行下一批；VAL 不写经验，反馈只用于选择。
- C0 为初态，不额外做 VAL；C5/C10/C15/C20/C25/C30/C35/C40 记录 8 个批次 winner。每个候选至少 9/10 完整评分才有资格；没有合格候选则报告该轨迹不确定，不借用其他来源或 run。
- 最终 Selected 固定为第 8 批 winner（C40），不再从历批 winner 中重选。当前 VAL/TEST 成员不变；RR 允许将父 VAL 未被当前 VAL/TEST 使用的 10 题转入 EVO，新增 EVO 成员和题序单独冻结登记。

## 迁移 TEST 与论文数据

本源 TEST 使用对应来源、对应 run 的 Selected。三个迁移 benchmark 均测试全部 `3 来源 × 3 run`，不在目标上进化或重新选版。Initial、Direct、matched JIT、固定 rubric-MAS 每题各一份静态产物，跨来源和 run 共享。

完整库存为 **360 EVO + 2,160 VAL + 2,751 TEST = 5,271 槽位**；另计 **216 次候选生成**及其成本。任务内允许多次模型调用和协作；统一 token/时间预算、工具权限与并发设置在执行前冻结。静态或已选定状态的 TEST 可按依赖就绪生成，但全部 TEST 产物及失败/缺失统一封存后才评分。

保留九条进化轨迹、每批输入状态、三个候选及 winner、完整 checkpoint、逐题产物和评分、失败/缺失、实际与估算 token、成本来源、选版记录、迁移矩阵、配对 bootstrap 与预注册 Holm 校正。三个来源分开报告，不计算跨量纲总排行榜。

[原联合方案](experiment_plan_v5_joint_ZH.md)、逐题进化的独立 v5 协议及其原始运行记录均保留为历史；旧 Pool、checkpoint 和分数不接续或混入本次分批进化结果。
