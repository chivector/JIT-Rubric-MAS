# 六 benchmark 评测、四 benchmark 独立进化：分批候选协议 v7

本文件是 [主实验方案](experiment_plan_v5_ZH.md) 的详细执行约束。目标是分别验证 ResearchRubrics、DeepSearchQA、WritingBench、DeepResearch Bench II 从零进化的 Ours MAS，并在 IFEval、IFBench 上评价迁移表现。每个进化 benchmark 只运行一次；旧 v5/v6 注册、运行记录、题号和经验不接续。

## 题目与用途

| Benchmark | EVO：更新经验 | VAL：选版本 | TEST（本源/迁移） |
|---|---:|---:|---:|
| ResearchRubrics | 40 | 10 | 33 |
| DeepSearchQA | 40 | 10 | 50 |
| WritingBench | 40 | 10 | 50 |
| DeepResearch Bench II | 40 | 10 | 40（20 英文、20 中文） |
| IFEval | 0 | 0 | 50 |
| IFBench | 0 | 0 | 50 |
| **合计** | **160** | **40 个固定 VAL 题** | **273 个 TEST 题** |

VAL 是每个 benchmark 固定的 10 道题，但在 8 个批次、3 个候选中重复评估，形成 960 个 VAL 评估槽位。四个进化 benchmark 各有一条独立轨迹，共四条轨迹；不设置三个 run，也不把单次运行拆成重复样本。

## 分层划分

EVO、VAL、TEST 在每个 benchmark 内分别按完整可用库存的领域比例分层，并尽量使三个子集接近总体分布，避免训练集中单一领域而验证/测试集中换成另一领域。优先使用公开字段：ResearchRubrics 的 `domain`、DeepSearchQA 的 `problem_category`、WritingBench 的一级领域与语言、DeepResearch Bench II 的公开主题与语言。DeepResearch Bench II 的 TEST 固定保留 20 道英文和 20 道中文，并在各语言内尽量保持主题比例。

领域库存不足、子集容量过小、固定成员、重复组或曝光隔离导致无法接近总体比例时，采用确定性的最小配额偏差规则。注册中必须冻结稳定 task ID、原始记录号、子集、领域/语言标签、题序和抽样种子，并报告每个子集的领域计数、理想配额偏差、覆盖率及原因。重复组保持完整，曝光边界优先于均衡目标。

IFEval 和 IFBench 没有可信的统一领域字段：优先使用可用公开类别；若无法可靠分层，则使用固定哈希抽样并明确披露不能保证领域均衡。已登记开发曝光题不得进入正式 TEST；历史未知曝光保留审计限制。

## 分批进化与选版

1. 每个 benchmark 从独立空经验库和空 Agent Pool 起步。四条轨迹使用相同的六个初始角色原型（writer、searcher、critic、planner、analyst、generalist），不共享经验、Pool、harness、checkpoint 或测试反馈。模型、prompt、工具清单、token/时间预算和并发限制在运行前冻结。
2. 每条轨迹的 40 道 EVO 按固定顺序分为 8 批，每批 5 题。一批内所有题读取同一个冻结状态，题间不做归因、经验提交或 Pool 更新；失败和不完整反馈仍消耗题位，不补题或重采样。
3. 一批 5 题全部终态后，基于完整执行记录、judge 反馈和全部 rubrics 做全局/局部归因与角色反思，从同一批输入状态生成 3 个互不接续的候选。候选可 Add、Delete/Prune、Split、Merge、Specialize、Reorganize Pool，或新建、修改、保留临时 harness。
4. 每个候选只在本 benchmark 固定的 10 道 VAL 上执行一条完整轨迹；3 个候选共 30 个 VAL 槽位。VAL 使用不可变快照，不写经验；反馈只进入选版服务。候选至少 9/10 道完整评分才有资格。
5. 本批全部 VAL 槽位终态后，按预注册的官方指标归一化和固定平局规则选择一个 winner，作为下一批唯一输入。如果 3 个候选都未达到 9/10 完整评分，则该 benchmark 轨迹标记为不确定并停止；不借用其他来源、补题、重抽或挑选有利候选，对应 C40、本源 TEST 和迁移 TEST 标记缺失/不确定。C0 是共同初始化下的空经验初态，不额外执行 C0 VAL；保存 C5/C10/C15/C20/C25/C30/C35/C40 及所有候选，最终 Selected 固定为 C40。

## Ours-only TEST 与官方指标

正式 TEST 只运行 Ours 的最终 Selected，不运行 Initial、Direct、matched JIT、固定 rubric-MAS、Terminal、G/GO 或其他外部方法。四个进化 benchmark 使用各自 C40 做本源 TEST；IFEval 和 IFBench 分别测试四个来源的 C40，形成 `4 来源 × 2 目标` 迁移矩阵。TEST 不参与进化、选版、调参或按质量重试；全部产物、失败和缺失统一封存后评分。

评分器版本、代码哈希、配置和输入快照在 TEST 前冻结。各 benchmark 保留官方原生指标：

- ResearchRubrics：官方加权任务分数；
- DeepSearchQA：任务级 F1，并单独记录 precision、recall 和 fully-correct；
- WritingBench：官方 author checklist 的 1–10 分及任务宏平均；
- DeepResearch Bench II：**Overall、InformationRecall、Analysis、Presentation**，另报 blocked-source 和缺失情况；
- IFEval：官方 strict accuracy，必要时附 loose/instruction-level；
- IFBench：官方 loose accuracy，必要时附 strict/instruction-level。

DeepResearch Bench II 的 EVO/VAL 只使用运行前冻结的公开 rubric 与 scorer/judge 合约；TEST 使用冻结的官方评分器和独立输入快照。TEST 的答案、rubric、官方分数和反馈不会进入 EVO/VAL，也不会用于调整候选或选版。

不把不同 benchmark 的分数合并成总排行榜。单次运行不报告 run 间方差；可对逐题结果报告任务级 bootstrap 区间，并把 C40 的结果、失败/缺失和迁移来源分开呈现。

## 资源、延迟与进化曲线

每个 EVO、候选 VAL、候选生成和 TEST 槽位都记录：输入/输出/总 token、模型与工具调用数、重试、队列和限流等待、执行 latency、端到端 wall-clock、并发度、超时、失败、缓存/恢复来源、实际成本和成本来源。候选生成、judge、归因、工具调用和协作通信单独计账，未知价格保留 null。

每个 C0/C5…C40、每批候选和 winner 保留可直接画曲线的事件：Agent Pool 总数及各角色数、Add/Delete/Prune/Split/Merge/Specialize/Reorganize 次数、harness 新建/修改/删除/保留次数、memory/rubric/evidence 条目数、RubricGraph 节点/边（可用时）、VAL 分数、完成率、token、latency、cost、失败率、状态哈希和版本号。

## 工程修复与结果审计

运行监控若触发预注册工程阈值（崩溃、格式失败率、token 超预算、latency/限流、服务故障、数据泄漏或分层约束无法执行），立即记录事件并允许版本化修复。修复须记录触发条件、原因、代码/prompt/config diff、影响阶段和新版本哈希；受影响阶段按同一冻结 split 重跑，保留原始轨迹和修订轨迹，并在报告/附录中同时披露。

TEST 产生任何官方分数后，禁止根据分数修改实现、题目、split、scorer、选版规则或静默删除失败；不得挑选最有利 checkpoint。必要的协议修订须在重新运行前冻结，并预先指定主结果和修订前后比较方式。

## 工作量与封存

| 阶段 | 名义槽位 |
|---|---:|
| EVO | 160 |
| VAL | 960（4 benchmark × 8 批 × 3 候选 × 10） |
| Ours TEST | 573（173 本源 + 400 迁移） |
| **合计** | **1,693** |

另有 `4 benchmark × 8 批 × 3 候选 = 96` 次候选生成。逻辑槽位不是 API 调用次数或 token 上限。全部槽位按冻结身份原子封存，记录失败/缺失、实际成本、状态哈希、评分器版本和恢复来源；正式结果只来自身份匹配的真实运行记录。

现有 v6 机器协议和执行器仍按旧的三来源、多 run 拓扑解释；正式运行本协议前，必须另行生成四来源、单次运行、Ours-only 的 v7 manifest、协议哈希和执行身份，不能直接复用旧注册。
