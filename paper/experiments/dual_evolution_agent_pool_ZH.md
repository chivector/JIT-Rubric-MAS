# Meta-Agent 与 Full Agent Harness Prototype 的双层进化实现

本文说明当前代码的双层进化机制及其与实验状态的关系。它不修改已冻结的 v4/v5 题目成员、题序或评分协议，也不报告正式实验结果。

## 两层分别学习什么

本方法是两层进化循环：**meta-level** 的 meta-agent 学习根据公开 query 预测 rubrics 的经验，以及据此选择团队、分配职责、设计依赖拓扑和信息流的经验；**MAS-level** 的各 agent 学习自身角色的 harness，由 meta-agent 维护可复用的 **full agent harness prototypes** 及动态 Agent Pool。原型具有可追踪身份和版本，覆盖角色 context/prompt、命名 skill 指令、条件化 memory、reasoning/planning strategy、工具偏好、通信习惯及已安装的 harness policy，不只是 prompt 层面的 role cache。`agent_pool`、`AgentProfile` 和 `AgentPoolSnapshot` 是代码中的存储与类型名称。默认 Writer、Searcher、Critic、Planner、Analyst 和 Generalist 只是初始种子，不是固定角色全集；角色、能力标签、成员数量和层次组织可随 EVO 演化。种子没有已学习的历史，完成任务计数也不等于质量认证。

**构建 MAS 不做双层归因，也不修改长期 Agent Pool。** 新任务到达时，Meta 根据公开 query 预测 rubrics，显式思考任务关键点，然后以此设计 MAS。Meta 读取包含身份、能力、版本、历史计数、层次关系和已保留 harness 摘要的 Pool 快照，选择和组合适合的原型并设定目标与协作关系；入选 Agent 的长期状态提供已保留的 skill、方法、工具策略与 harness，只对当前任务作局部适配。若没有合适的原型，Meta 可临时创建完整 `AgentProfile`，包含角色、能力、context/prompt、skills、memory、工具偏好、策略、通信与 harness policy；`AgentSpec.temporary_profile` 和 `creation_rationale` 显式记录原型及创建理由。临时原型与所选长期原型一起冻结到当前任务计划，用于执行、归因和恢复；`temporary_agents.json` 记录预测阶段创建的全部临时候选及是否被最终团队选中。临时原型只存在于本轮任务，不自动成为长期 Pool 成员。构建阶段只选择、组合或临时创建，Merge、Delete 等结构操作均发生在执行后的反思进化阶段。构建阶段不读取尚未产生的 judge 反馈，不属于双层归因，也不写入长期状态。`task_prompt` 等当前任务适配不直接改写长期 profile；`selected_skills: null` 使用原型的 skill library，`[]` 表示本任务不用这些 skill。

池化路径冻结选中成员的 profile 和计划，复用仓库内的 MAS scaffold，不再为每道题重新生成全部 harness Python。当前 skill 是可复用指令，harness 进化是已安装 `rubric_mas` 的 full/recent memory、窗口大小和工具排序策略，尚不支持任意新工具代码或新 Python harness 的自动安装。工具权限、预算和执行模式仍由任务配置约束。

## 动态 Agent Pool

Pool 可以保持扁平，也可以通过 `AgentProfile.parent_agent_id` 的可校验父子关系形成树状或层次化森林；不存在预先固定的根、角色树、深度或分化路线。通用原型可因多次出现的需求或失败模式分化成专业子原型；稀少、冗余或收益不足的原型也可被删除、合并或重组。Meta 在每轮 EVO 的 `evolution_integrate` 阶段提出并选择 typed `AgentPoolOperation`，通过 `EvolutionDecision.pool_operations` 显式保存下列操作，未选中操作不写入：

| 操作 | 本轮进化中的含义 |
|---|---|
| Add | 增加新的完整 harness 原型，包括将本轮临时创建、被最终团队选中并实际执行且有反馈的原型纳入 Pool。 |
| Delete/Prune | 移除无效、冗余或成本收益不足的原型，并显式处理受影响的层次关系。 |
| Split | 将原型的职责分化为多个完整子原型，显式决定原成员及子成员的保留和关系。 |
| Merge | 将重叠原型整合为一个完整原型，并处理被整合成员及其关系。 |
| Specialize | 调整角色、能力与完整 harness，使原型适配有证据支持的专门任务。 |
| Reorganize | 调整父子层次或成员组织，保持身份、引用和无环约束一致。 |

结构决策读取当前快照、本轮临时创建记录、各 Agent 反思、全局归因及 `AgentPoolObservation` 保留的相关历史任务、失败模式、可观察质量反馈和实际 Token 成本。每个操作保存动机、目标、基准版本、来源任务与证据；操作支持不代表必须每轮改变结构，提交也不代表已通过独立实验认证。仅提出但未执行的临时候选缺少本轮执行反馈，不能据此宣称已学习或成熟。父子关系描述可复用原型的组织，不自动约束任务 MAS 的拓扑；Meta 仍针对 query 选择和组合成员。删除持久原型不会改变已冻结任务中的 profile 或历史轨迹。

## 基于 rubric 的设计与双层归因

RubricGraph 是显式的结构化合约：节点 `PredictedRubric` 保存 requirement、来源、importance、confidence、预期证据与适用条件；边 `RubricEdge` 保存 prerequisite、support、overlap 或 tradeoff 关系及不确定性。图的校验拒绝重复 rubric ID 和不存在的边端点。构建结束后冻结预测图与 `TeamSpec`，其中 `coverage`、`primary`、`reviewers` 和各 agent 的 `rubric_ids` 记录当前 MAS 的职责；这些是执行前设计，不是执行后信用判断。

形式化地，令预测图为 \(G=(R,E)\)，当前 MAS 的 agents 为 \(A\)，职责关系为 \(C\subseteq R\times A\)，由 `AgentSpec.rubric_ids`、`coverage`、`primary` 和 `reviewers` 中的执行/审核职责合并得到。judge 给出的评测 rubric 集合为 \(J\)，语义对齐关系为 \(M\subseteq J\times R\)。Meta 全局分析还可以根据已执行角色或协作关系，显式指定该 agent 需要反思的补充评测 rubrics \(D_a\subseteq J\)，逐角色记录非空理由。最终局部评测范围为 \(J_a=\{j\in J\mid\exists r\in R:(j,r)\in M\land(r,a)\in C\}\cup D_a\)。预测 ID 与评测 ID 不要求相同，支持一对多、多对一以及未匹配项；结构职责、语义对齐和 Meta 的有理由分配共同决定局部归因范围，而不是只在 prompt 中要求“根据 rubric 归因”。每个 scope 还保留与其预测 rubric 相连的图边，用于解释跨 rubric 关系；图边不自动扩大其评测职责集合。团队拓扑、事件和证据用于解释跨角色的影响，结构连接本身不证明某个 agent 的因果贡献。

归因范围以 typed `RubricCreditAssignment` 显式保存到 `attribution.json`；`build_credit_assignments` 依据冻结图、当前团队、语义对齐与 Meta 全局分配构造各角色的 scope。`global_outline.rubric_assignments` 和 `rubric_assignment_rationale` 保存补充分配及理由，校验拒绝未知 agent、未知评测 rubric 或缺少理由；scope 中保留 `meta_assigned_evaluated_rubric_ids` 和 `assignment_rationale`，恢复时重新构造并核对相同范围。局部归因和后续 harness 反思复用该 scope，反思 payload 同时包含 `credit_assignment` 与 `assigned_predicted_rubrics`，`evaluation_summary` 的逐项 rubrics 只保留该角色的相关数值反馈。

**双层归因只发生在执行结束、获得 judge 反馈之后。** Meta 先查看全部 rubrics、两份冻结预测、完整团队结构、提交答案和共享执行证据，形成关于 rubric 预测、角色分工、拓扑和交接的全局归因假说；随后按当前 MAS 的职责、语义对齐和显式补充分配向各 agent 分配相关 rubrics。每个 agent 查看相关反馈、自身完整轨迹与关联交接证据，进行局部归因并提出 context、tools、memory、skills 等 harness 的改进。最后 Meta 收集局部内容，保留分歧、反证和不确定性，确定本次两层进化内容。未预测或没有原执行/审核归属的 criterion 失败仍应分析 prediction/organization 缺口；执行后将它分配给某 agent 反思，并不反向赋予该 agent 执行责任或构成 execution 失败证据。缺少局部证据的 rubric 由全局分析保留，不强制归咎于任一 agent。

## Budget-aware 团队组织

生成 MAS 的 Meta-Agent 必须兼顾质量与 token 开销。全局规划读取任务总预算、单次输出上限及实时账本中的已用、在途预留和剩余 token；Meta 结合任务要求和池内能力，选择够用的成员、职责、依赖和通信，避免重复工作。团队的 `budget_plan` 记录预计输入/输出 token、模型/工具调用、通信量、后续阶段预留及质量—成本权衡，计划与预留应符合实时余量。额外角色或协作应有预期收益，不能把硬上限当作必须花完的目标；预计调用数也不作为新的固定轮数限制。

预算意识属于双层进化：Meta 学习不同团队与协作方式的成本条件，并在 Pool 增删、分化、合并和重组时比较质量覆盖与 Token 成本；Agent 在反思中读取自身实际成本，学习有效的上下文使用、skill 选择、证据交接和终止策略。Agent 自主选择局部方法，同时遵守任务预算与已安装 harness；节省 token 不能替代覆盖重要任务要求。只有 EVO 可将实际成本与质量反馈一起用于经验更新，VAL/TEST 成本只用于报告。规划估计与实测分开记录，未知价格保持 null；冻结的 checkpoint 和发布包绑定两层状态、Pool 完整结构、预算策略、prompt 与执行配置。

## 反馈、写入与复用

只有 EVO/stream 完成提交、外部评分和归因后，入选 Agent 才反思自己的执行轨迹。反思输入含自身实际事件、团队计划、相关归因假说、显式 rubric scope、分配的预测 rubrics 和相关分项数值反馈；全局总分仅作为提交级背景。反馈摘要不包含私有 criterion、reference answer 或裁判推理；整体得分不能证明某个 Agent 的独立贡献或策略的因果效果。持久 memory 应是可引用过程证据的条件化经验，不应保存任务答案或隐藏评分要求。

各 Agent 返回候选 `AgentEvolutionUpdate`，不会自行写入长期 prototype；本轮被最终团队选中并实际执行的临时 Agent 同样参与基于自身证据的局部反思。full agent harness prototype 路径的 meta-level 提案只写 `rubric` 或 `organization` 经验库，角色专属执行经验写入相应的 harness prototype。最后的 `evolution_integrate` 阶段由 Meta 查看全部候选 Agent 更新、局部归因、临时创建记录和 meta-level 提案，同时审视 Pool 历史使用、失败模式、质量反馈及 Token 成本，选择本次实际应用的 Agent update IDs、至多一个经验提案及 Pool 结构操作，并说明是否保留每个已执行临时原型及取舍。`meta_evolution.json` 保存候选与最终决策；只有该决策选中的经验、harness 和 Pool 结构更新进入原子提交。Meta 的协调决定不是对更新质量的实验认证，也不恢复 accept/hold/reject 或新增独立任务验证门槛。

Meta experience、Meta 选中的 Agent updates 和 Pool 结构操作在同一 SQLite 事务中写入同一个版本化 `ExperienceSnapshot`；身份、版本、来源、证据引用、父子引用、无环约束或任一必要反思失败时不发生部分写入。重复任务和恢复使用已登记提交、冻结临时原型及更新，不重复学习。freeze 导出两层状态和 Pool 完整结构；rollback 同时恢复两层和成员组织。VAL/TEST 使用冻结长期状态，可为当前 query 临时创建并记录 harness，但不执行归因与反思，不保留临时原型，不把评测反馈写回任一层或用于后续 held-out 任务。当前任务和 held-out 来源的 memory 在检索前被排除，并按任务与角色筛选。

## 执行与实验范围

双层进化与执行调用次数是两个独立配置。默认 `single_pass` 每个执行角色一次模型调用；可选 `iterative_shared_ledger` 将工具结果和更新后的账本交付给仍在执行的角色，支持继续调用和 Writer 修订。当前 DAG 调度对每个角色只派发一次，不会因收到消息而重新唤醒已经结束的上游角色。失败草稿不能成为最终答案或下游完成产物。移除可选调用数上限仍保留 token 和 timeout 停止条件。

[`experiment_plan_v5_ZH.md`](experiment_plan_v5_ZH.md) 保留联合进化方案和父级题目来源；[`experiment_plan_v5_independent_ZH.md`](experiment_plan_v5_independent_ZH.md) 注册分源独立进化及迁移评价。正式 checkpoint、选版和测试产物必须绑定完整 Meta + full agent harness prototype 状态及其哈希，不能只导出 Meta advice。独立版的实际运行状态仍是 `PROTOCOL_FROZEN_NOT_RUN`，完整启动条件见 [`run_launch_plan_v5_independent_ZH.md`](run_launch_plan_v5_independent_ZH.md)。当前离线 smoke 和回归测试验证软件连接、原子更新、角色复用与 held-out 隔离，不构成模型性能或角色成熟度的实验结论。

实现入口：`jit_mas/agent_pool.py`、`planning.py`、`bridge.py`、`execution.py`、`pipeline.py`、`experience.py`。运行和限制详见 [`docs/jit_mas.md`](../../docs/jit_mas.md)。
