# 六 benchmark 联合进化：小子集协议 v5

历史方案：2026-10-06 用户明确切换为三个源 benchmark 各自从空库独立进化。当前方案见 [experiment_plan_v5_ZH.md](experiment_plan_v5_ZH.md)，本文件及联合协议保留用于审计旧运行。

目的：在可承担的工作量内，验证跨 benchmark 联合进化后，**同一份冻结的 meta-agent 经验与 full agent harness prototype 状态** 能否改善不同任务。保留 v4 的题目与评价协议，只缩减题数、checkpoint 数与重复生成次数；不运行全量 benchmark。

## 方法与阶段边界

**构建 MAS 时不做双层归因，也不修改 Agent Pool；执行后进化时才做双层归因并统一更新两层状态。** 新 query 到达后，meta-agent 根据公开 query 预测 rubrics，显式识别任务关键点，并以此设计角色、职责、拓扑与协作。Meta 读取当前冻结的 Agent Pool 快照，选择和组合合适的 full agent harness prototypes 并作任务适配；若没有合适的 harness，Meta 可为当前任务临时创建完整 agent harness，并记录创建原因、完整 profile、来源及使用情况。构建时的 rubric 职责分配是设计依据，不是对尚未发生的执行结果归因；临时创建只影响当前 MAS，Pool 的合并、删除及其他结构操作均留到反思进化阶段。

执行结束并获得 judge 反馈后，meta-agent 根据全部 rubrics、当前 MAS 结构和执行证据进行全局归因；再按当前 MAS 的 rubric 职责及评测对齐关系，把每个 agent 相关/负责的 rubrics 分配给它。各 agent 基于自身轨迹进行局部归因与 harness 反思，meta-agent 汇总这些内容，确定本次进化更新。RubricGraph、职责映射和语义对齐记录提供可校验的归因结构，具体机制及实现边界见 [双层进化实现](dual_evolution_agent_pool_ZH.md)。

- **meta-level** 累积预测 rubrics 的经验，以及依据这些 rubrics 设计 MAS 结构的经验；下一道题仍根据其 query 生成任务条件化 MAS。
- **MAS-level** 累积角色的 context、tools、memory、skills、策略与 harness policy 等经验，维护可复用的 **full agent harness prototypes**。Agent Pool 的成员、能力分工及层次关系均可演化；初始经典角色是种子，不是固定角色全集。Meta 根据历史任务、失败模式、可观察质量收益及实际 Token 成本，在每轮进化中选择 Add、Delete/Prune、Split、Merge、Specialize、Reorganize 等结构操作，并决定是否保留本轮实际执行且有反馈的临时 harness。Pool 可形成树状或层次化森林，也可保持扁平；不预设必须采用的根、深度或角色划分。`agent_pool` 是代码中的存储名称，不是仅缓存角色 prompt 的方法定义。

两层状态及 Pool 的完整成员、harness 与层次关系共同构成 checkpoint。EVO 中，Meta 汇总全局和局部反思后，确定经验、角色 harness 和 Pool 结构的本轮更新；通过来源、证据、身份、版本及结构检查后，在同一事务中原子写入。VAL/TEST 只利用冻结状态设计和执行 MAS，可记录仅供当前任务使用的临时 harness，但不开展执行后双层归因、反思、长期 Pool 更新或基于反馈跨题积累。本节明确方法边界，不改变已注册的题目成员、题序、评分和选版规则。

## 题目与用途

| Benchmark | EVO：更新经验 | VAL：选版本 | TEST：最终评价 |
|---|---:|---:|---:|
| ResearchRubrics | 20 | 10 | 33 |
| DeepSearchQA | 20 | 10 | 50 |
| WritingBench | 20 | 10 | 50 |
| DeepResearch Bench II | 0 | 0 | 40（20 英文、20 中文） |
| IFEval | 0 | 0 | 50 |
| IFBench | 0 | 0 | 50 |
| **合计** | **60** | **30** | **273** |

共 **363 道不同题目**，没有 Dev。其余 2,611 道是 unused/reserve，不用于本次调试或默认追加测试。

- 具体原始题号见 [task_assignments_v5.md](task_assignments_v5.md)；稳定 task ID、原始记录号、父分区与三个 run 的题序见 [joint_task_splits_v5.json](joint_task_splits_v5.json)。题号按固定原始文件从 1 开始，CSV 不计表头；不能重排文件后重新编号。
- 只在 v4 对应父分区内，使用公开元数据分层和固定哈希抽样；EVO/VAL/TEST 不跨区，不依据答案、私有 rubric 或实验结果挑题。所有 run 使用相同成员，只改变进化顺序。
- RR 保留全部 33 道 clean test；原 18 道历史曝光/隔离题不进入此次选中测试，v4 记录不删除。WritingBench 原始第 1 题保留曝光标记，只能进入 EVO 或 unused，不强制必选。
- v1-v4 文件保留为历史记录，v5 的抽样不能重新解释历史运行结果。未来使用 reserve 必须另行冻结注册，并审计是否已被研究过程暴露。

## 进化与 VAL

1. 三个源 benchmark 在每个 run 内共用一份空库起步的经验，不分别培养三个生成器。三个固定题序种子为 `20261001/20261002/20261003`。
2. 每条轨迹一次遍历 60 道 EVO，共四个混合阶段，每阶段 `5 RR + 5 DSQA + 5 WritingBench = 15 题`；保存 `C0、C15、C30、C45、C60` 五个候选。失败或无有效提案仍消耗该题位置，不补题、不回滚、不早停。
3. **每个 checkpoint 都评全部固定 30 道 VAL，每题生成一次。** 不是最近一批，也不是累计变化的 VAL。VAL 不写经验，不把反馈交给源题执行 agent。
4. 每个 benchmark 至少 `9/10` 完整评分才有筛选资格。按固定理论范围归一化（RR 保留有符号权重范围；DSQA `[0,1]`；WritingBench `[1,10]`），先在 benchmark 内平均，再三者等权平均。缺失原生得分为 null，仅筛选用标注的下界；不能用观察到的最高/最低分归一化。
5. 全部四阶段结束后，从五个候选中选一个完整状态；最大值相对容差 `1e-12` 内，依次按完整评价数量、较早位置、状态哈希打破平局。没有合格候选则报告该 run 不确定，不追加有利 run。
6. 经验仍直接写入，不恢复 accept/hold/reject。MAS 构建与执行后双层归因/进化分为独立阶段；每个 checkpoint 同时保存 meta-level 经验、完整角色 harness 原型及动态 Pool 结构。每轮允许结构提案，但是否操作及采用哪种组织由 Meta 根据证据决定，不强制每轮增删成员。评价和执行器单次 shared-ledger 通信等协议不变。


## Budget-aware MAS 生成

MAS 生成 Meta-Agent 必须将 token 开销作为团队设计依据，而不仅在执行时被动接受预算截断。全局分析和团队协调读取冻结的任务总 token 预算、单次输出上限，以及实时账本中的已用、在途预留和剩余 token。在满足重要任务要求的前提下，Meta 选择够用的 Agent 数量、职责分工、依赖拓扑与通信方式，并为最终答案和必要检查保留输出空间。额外角色、重复推理、长上下文与多轮协作需要有预期质量收益；简单任务可以直接使用一个适合的成熟 Agent。

团队的 `budget_plan` 记录预计输入/输出 token、调用数、工具数、通信量、后续阶段预留及质量—成本权衡；计划与预留必须符合实时余量。预计调用数是估计，不额外引入固定协作轮数限制。Agent 在已有 skill、memory 和 harness 基础上选择符合任务预算的局部策略；EVO 将角色实际成本与质量反馈共同用于两层成本经验。预计成本不等于实际账单或质量保证；生成、局部规划、通信、执行及重试计入真实推理成本，评分、归因和长期更新成本另列，未知价格保持 null。

本要求不改变题目成员、库存或已注册的 VAL 质量选版及平局规则。各方法在匹配的资源上限下报告原生质量、真实 token、失败率和质量—成本关系；若要按成本惩罚重新选版或增加不同预算下的实验，须另行注册。

## TEST 与基线

**一个 run 只选一个通用版本，用于全部六个 benchmark，不按 benchmark 另选 checkpoint。** 三个 run 都报告；默认发布包预先指定为 run 0 的联合 VAL winner，不看 TEST 挑版本。包内冻结生成模型身份、代码、prompt、配置、工具/检索策略，以及 meta-level 经验和 full agent harness prototypes、成员层次与结构更新记录的完整联合快照；这些是经验与原型状态，不是新训练的权重。遇到新题仍预测 rubrics、复用或临时创建角色 harness 并生成任务条件化 MAS，冻结的长期 Pool 保持不变。

五个核心方法为 **Initial、Selected、Direct、原生 JIT、固定 rubric-MAS**。三个 Selected 状态各对每道 TEST 生成一份；四种静态方法每题各生成一份，跨三个 run 共享，不冒充三份独立样本。底座、允许的公开输入、证据材料和资源上限匹配。全部答案先按冻结库存提交并封存，再评分；TEST 无经验更新、补题或按质量重试。

Terminal 对照、G/GO、额外进化消融和外部 MAS 基线不在本次必跑库存内；需要时单独注册，不自动启动。采用代理底座的原生 JIT 不冒称原始 JIT-27B 复现。

## 工作量与报告

| 阶段 | 名义任务单元 |
|---|---:|
| EVO | `3 × 60 = 180` |
| VAL | `3 × 5 × 30 × 1 = 450` |
| TEST | `273 × (3 Selected + 4 static) × 1 = 1,911` |
| **合计** | **2,541** |

较 v4 按同样五个核心方法计算的 61,134 个任务单元，减少约 **95.8%**。这是逻辑任务/产物槽位，不是 API 调用次数或 token 上限；单题内可能多次调用。相同完整状态的合法缓存复用记录来源并扣除实际成本，证据准备、评分调用和人工审计另外计量。

四个研究/写作 benchmark 是主对照族，两个指令遵循 benchmark 是独立诊断族，各自做 Holm 校正；与 matched JIT、固定 rubric-MAS 的次要对比分别组成 8 项主 benchmark 和 4 项诊断比较族。各数据集保留原生指标，不平均成跨量纲排行榜。报告全部 run、逐题配对差值、整题 bootstrap 区间、失败/缺失和成本；缺失不得静默删除或作为正常得分。

这是 **subset track**：小样本区间可能较宽；单状态单题一份产物不能估计该状态生成方差；三个种子只控制进化题序，不保证服务端采样种子。不能将结果冒称官方全量、全面优于其他方法或 SOTA。

机器可读注册见 [joint_protocol_v5.json](joint_protocol_v5.json)。本次文档和清单修改没有发起真实 API 实验；旧 v3 三 benchmark 运行入口不能直接冒充 v5 六 benchmark 联合执行。正式结果只由与 v5 冻结身份匹配的真实运行记录产生。
