# 六个 Benchmark 的联合进化协议（v4）

本文件用于和合作者对齐实验。**主方法是在不同 benchmark 的题目上共同进化一个 MAS generator，再冻结同一版本用于所有 benchmark，而不是每个 benchmark 各选一个版本。** 这是实验设置，不是已完成实验或性能结果；旧 v3 运行器及其测试不能作为新增 benchmark 和联合运行已完成的证明。

## 题目用途与数量

取消独立 Dev 集。每个 benchmark 的题目归属以 [joint_task_splits_v4.json](joint_task_splits_v4.json) 为准，便于阅读的全部具体题号见 [task_assignments_v4.md](task_assignments_v4.md)，整体设置见 [joint_protocol_v4.json](joint_protocol_v4.json)。题号绑定固定数据版本、作者 ID／原始行号和题目哈希，不按每次运行重新抽题。

| Benchmark | EVO：更新经验 | VAL：选择版本 | TEST：最终评测 |
|---|---:|---:|---:|
| ResearchRubrics | 30 | 20 | 51：clean 33 + exposed 18 |
| DeepSearchQA | 150 | 100 | 650 |
| WritingBench（1,000 题版） | 150 | 100 | 750 |
| DeepResearch Bench II | 0 | 0 | 132 |
| IFEval | 0 | 0 | 541 |
| IFBench（单轮 300 题版） | 0 | 0 | 300 |
| **合计** | **330** | **220** | **2,424** |

- RR、DSQA 已有 EVO/VAL 题号不变；原 Dev 并入 TEST，不重新抽取有利题目。
- RR 原 18 题含历史暴露和 pilot。移入 TEST 不会消除暴露，保留 `test_slice=historically_exposed_or_reserved` 标记：主表独立测试效应只用 clean 33 题，另外报告 exposed 18 题，不混称 51 题全部未见。
- WritingBench 使用固定的新版 1,000 题。按公开语言／领域和固定哈希划分，避免答案、私有 rubric 或得分参与分题。清单准备时作者 index 1 的私有行意外回显，已登记暴露并强制纳入 EVO，不得进入 VAL/TEST。
- IFEval、IFBench 和 DR Bench II 仅作外部测试，不参与进化、VAL、来源选择或目标 benchmark 调参。IFEval 的托管 split 名称不代表本实验把它当训练集。
- 所有划分均为项目自定义实验划分，不宣称作者提供了这些 train/val/test。正式执行前审计跨数据集重复与已知同源题；冲突按版本化、与结果无关的修订处理。

## 如何联合进化

三个 source benchmark 共用**一份** rubric / organization / execution 经验库，从空库出发；不是先分开训练再挑一个。

共 6 个阶段，每阶段混合 **5 RR + 25 DSQA + 25 WritingBench = 55 题**。阶段内按冻结顺序交错运行，每道 EVO 题只处理一次。保存位置为：

`C0, C55, C110, C165, C220, C275, C330`

位置按已尝试的题数计，不按成功写入次数计。失败、无合法更新或评价不完整也消耗原定题位，不补新题，不通过质量重试挑答案。合法且有证据支持的归因经验直接写入，每题至多一次，**不恢复 accept / hold / reject**。后续始终从最新经验状态继续，不回滚到临时 VAL 最优状态，不早停，不多跑一遍。

运行 3 个预定顺序种子：`20261001 / 20261002 / 20261003`。每个种子对应一条完整联合轨迹；顺序改变，但 EVO/VAL/TEST 题目不变。

## VAL 到底评哪些题、怎么选

**每个 checkpoint 都评全部固定 VAL：20 RR + 100 DSQA + 100 WritingBench = 220 题，每题独立生成 2 次，共 440 个 task–artifact slots。** 不是最近一批、不是累计最近题，也不是同一个答案重复打分。

VAL 使用只读快照，不归因、不更新经验，内容和反馈不进入下一题 EVO。完全相同状态与执行身份可复用已完成 VAL，但必须记录来源，不能额外抽样。

入选资格：每个 benchmark 分别至少 90% 完整评价，即 RR 至少 `36/40`、DSQA `180/200`、WritingBench `180/200`。不能以全局平均完成率掩盖某个 benchmark 失败。

唯一的联合选择分数为：**先将每题分数按预先定义的理论上下界归一化，再在每个 benchmark 内平均，最后三个 benchmark 等权平均。**

- RR：`(score - L_t) / (H_t - L_t)`；正权重总和非零时，`L_t = 负权重之和 / 正权重之和`，`H_t = 1`；退化范围 `[0,0]` 按 0 计。
- DSQA：原生 F1，本身为 `[0,1]`。
- WritingBench：固定原生 `[1,10]` 量表，使用 `(score - 1) / 9`。
- 不完整评价只在 checkpoint 选择时以归一化 0 保守代入；官方观测分数保持 `null`，不伪造零分。归一化范围不能根据结果重新拟合。

进化全部结束后，从全部 7 个候选（含 C0）中选唯一全局最优。`1e-12` 容差内并列时，依次选完整评价更多、位置更早、状态哈希字典序更小者。没有合格候选则该轨迹结论不充分，不补一个更有利的种子。

## 最终保留什么版本

每条联合轨迹选出的状态在**所有六个 benchmark 上保持同一个 state hash**，不按目标 benchmark 重新选 checkpoint 或经验库。

可发布的通用 meta-agent 包包含：模型 serving identity、固定 generator 代码与 prompts、配置、工具／检索策略、选中的共享 `ExperienceSnapshot`，以及代码／数据／协议哈希。**这是经验增强的 generator 版本，不是重新训练的模型权重，也不是一套固定团队。** 新题仍由 JIT 根据任务和适用经验生成 MAS；task 内部保留 single-pass shared ledger，不引入循环协商。

论文报告全部 3 条轨迹。为了交付一个默认版本，事先指定 **run 0 的联合 VAL winner** 为默认发布包；不能看 TEST 后再选最好的 run。分 benchmark 独立进化可另列机制对照，但不是主方法。

## 最终 TEST 与对照

每个入选状态及每条轨迹的末态 C330 均在全部六个 benchmark 的每道 TEST 题上生成 3 个答案，三条轨迹全部报告。静态 baseline 每题只生成 3 个答案总计，供三条轨迹共享比较，不复制成 9 个独立观测。核心对照为初始版、联合进化选中版、末态版、Direct、同底座原生 JIT、rubric-guided 固定 single-pass MAS；其余消融须在执行前另行固定完整条件清单。

同题的公开材料、底座、资源上限保持一致。**全部条件、全部 benchmark 的 TEST 答案先封存，再调用测试评分器、释放反馈。** TEST 不更新、不归因、不换题、不选新版本。严格记录真实调用和复用，分别报告 token、调用数、失败率与完成率。

分别报告 RR 原生 rubric 得分、DSQA F1、WritingBench 写作分、DR Bench II 原生满足率及 blocked rate。IFEval 主指标为作者 prompt-level strict accuracy，IFBench 为 prompt-level loose accuracy，其他 strict/loose 和 instruction-level 指标作辅助。不把六种原始分数平均成“总排行榜分”。归一化 macro 仅用于联合 VAL 选择。

以整道题作配对统计单位，汇总重复与三条轨迹，报告置信区间；不是把 rubric 或 API 调用当独立样本。“选中版对初始版”分为四个研究／写作 benchmark 的主检验族与两个指令遵循 benchmark 的诊断族，分别做 Holm 校正。对原生 JIT、固定 MAS 的比较分别构成含 `4×2=8` 个检验的主要对照族和含 `2×2=4` 个检验的诊断对照族；RR 机制对照另成 4 个检验的族。RR 主效应仅使用 clean 33 题，exposed 18 题另外描述。缺失保持缺失并报告可行区间，不只挑完成题宣称更优。

## 工作量与边界

3 条联合轨迹共 `990` 个 EVO slots、`9,240` 个名义 VAL slots；选中版本最终 TEST 共 `21,816` 个名义 slots（含 exposed RR），均不是 API 调用数或保证成功数。相同状态复用可减少实际工作；静态 baseline、额外消融、证据准备、裁判调用另计。

本次对齐协议及题目清单不自动启动全量实验，也不声明旧 v3 driver 已具备六套 adapter 与联合选择执行能力。正式结果必须来自支持本协议、身份冻结、完整记录和先封存后评分的真实运行；历史 API 探针、离线 smoke 与旧协议结果不能填入新主表。

**给合作者的一句话：330 道跨 benchmark 混合题共同进化一份经验增强的 JIT generator，每 55 题用完整固定 220 题 VAL 选整体版本；最终冻结同一版本在六个 benchmark 测试，不针对目标 benchmark 再选择或适配。**
