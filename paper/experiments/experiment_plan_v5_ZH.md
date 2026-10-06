# 六 benchmark 评测：三个源 benchmark 独立进化 v5

2026-10-06 确认：**ResearchRubrics、DeepSearchQA、WritingBench 各自从零进化；DeepResearch Bench II、IFEval、IFBench 保留迁移 TEST。** 完整规则见 [独立进化方案](experiment_plan_v5_independent_ZH.md)，机器协议见 [independent_protocol_v5.json](independent_protocol_v5.json)。

## 状态与方法边界

每个 `benchmark × run` 使用独立空经验库和独立 Agent Pool 状态，共九条轨迹。初始持久快照为空；首次构建时各轨迹使用相同的六个未进化角色原型（writer、searcher、critic、planner、analyst、generalist），其 memory、source_task_ids 和 evidence 均为空。“从零”指没有学得经验或历史，不表示没有初始 harness。不同来源、不同 run 不共享、不合并、不接续学得的 meta-agent 经验或 full agent harness prototypes；模型底座、初始化规则、执行机制与资源上限匹配。

**构建 MAS 时不做双层归因，也不更新长期 Pool。** Meta 根据当前公开 query 预测 rubrics，选择和适配现有完整 harness；没有合适原型时临时创建，记录原因、完整 profile、来源和使用情况。执行时不合并或删除 Pool 成员。

**取得完整 judge 反馈后才双层归因与进化。** Meta 根据全部 rubrics 做全局归因，再依据 RubricGraph、职责分配及评测对齐，把相关 rubrics 分配给各 agent。Agent 反思自身 context、tools、memory、skills 与 harness 策略；Meta 汇总后确定两层更新及 Pool 的 Add、Delete/Prune、Split、Merge、Specialize、Reorganize，并决定是否保留本题临时 harness。Pool 可以扁平或层次化，不预设固定结构。

## EVO 与选版

| 独立来源 | 每条轨迹 EVO | 每个 checkpoint VAL | 本源 TEST |
|---|---:|---:|---:|
| ResearchRubrics | 20 | 10 | 33 |
| DeepSearchQA | 20 | 10 | 50 |
| WritingBench | 20 | 10 | 50 |

- 每个来源运行三个固定题序：run 0/1/2，对应 `20261001/20261002/20261003`；每条轨迹从空库开始。
- 仅本来源 EVO 可以更新经验；每 5 道题保存 C0/C5/C10/C15/C20 完整状态。失败仍消耗位置，不补题、不回滚、不早停。
- 每个 checkpoint 只评本来源固定 10 道 VAL；VAL 使用只读快照，不向 EVO 回流评分，也不写长期经验。
- 全部 20 EVO 和 50 VAL 终态后，以本来源 VAL 选择一个完整状态；至少 9/10 完整评分才有资格。没有合格状态则报告不确定，不借用其他来源或 run。

## 迁移 TEST 与论文数据

本源 TEST 使用对应来源、对应 run 的 Selected。三个迁移 benchmark 均测试全部 `3 来源 × 3 run`，不在目标上进化或重新选版。Initial、Direct、matched JIT、固定 rubric-MAS 每题各一份静态产物，跨来源和 run 共享。

完整库存为 **180 EVO + 450 VAL + 2,751 TEST = 3,381 槽位**。任务内允许多次模型调用和协作；统一 token/时间预算、工具权限与并发设置在执行前冻结。静态或已选定状态的 TEST 可按依赖就绪生成，但全部 TEST 产物及失败/缺失统一封存后才评分。

保留九条进化轨迹、完整 checkpoint、逐题产物和评分、失败/缺失、实际与估算 token、成本来源、选版记录、迁移矩阵、配对 bootstrap 与预注册 Holm 校正。三个来源分开报告，不计算跨量纲总排行榜。

[原联合方案](experiment_plan_v5_joint_ZH.md) 与其原始运行记录保留为历史；联合 Pool、checkpoint 和分数不混入独立进化正式结果。
