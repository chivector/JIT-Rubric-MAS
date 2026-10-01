# Meta-Agent 与 Agent Pool 的双层进化实现

本文说明当前代码的双层进化机制及其与实验状态的关系。它不修改已冻结的 v4/v5 题目成员、题序或评分协议，也不报告正式实验结果。

## 两层分别学习什么

Meta-Agent 学习任务需求预测、团队选择、职责分配、依赖拓扑和信息流设计。Agent Pool 中的成员保留稳定身份和版本，学习如何完成自身角色：角色 prompt、命名 skill 指令、条件化 memory、reasoning/planning strategy、工具偏好、通信习惯及已安装的 harness policy。默认种子包括 Writer、Searcher、Critic、Planner、Analyst 和 Generalist；种子没有已学习的历史，完成任务计数也不等于质量认证。

新任务到达时，Meta 只读取包含身份、能力、版本和历史计数的 catalogue，选择成员并设定目标与协作关系。每个入选 Agent 读取自己的长期状态，根据公开任务选择 skill、方法、工具与局部适配。协调阶段保留它的身份和内部选择，不由 Meta 任意替换。`task_prompt` 等当前任务适配不直接改写长期 profile；`selected_skills: null` 使用保留的 skill library，`[]` 表示本任务不用这些 skill。

池化路径冻结选中成员的 profile 和计划，复用仓库内的 MAS scaffold，不再为每道题重新生成全部 harness Python。当前 skill 是可复用指令，harness 进化是已安装 `rubric_mas` 的 full/recent memory、窗口大小和工具排序策略，尚不支持任意新工具代码或新 Python harness 的自动安装。工具权限、预算和执行模式仍由任务配置约束。

## Budget-aware 团队组织

生成 MAS 的 Meta-Agent 必须兼顾质量与 token 开销。全局规划读取任务总预算、单次输出上限及实时账本中的已用、在途预留和剩余 token；Meta 结合任务要求和池内能力，选择够用的成员、职责、依赖和通信，避免重复工作。团队的 `budget_plan` 记录预计输入/输出 token、模型/工具调用、通信量、后续阶段预留及质量—成本权衡，计划与预留应符合实时余量。额外角色或协作应有预期收益，不能把硬上限当作必须花完的目标；预计调用数也不作为新的固定轮数限制。

预算意识属于双层进化：Meta 学习不同团队与协作方式的成本条件，Agent 在反思中读取自身实际成本，学习有效的上下文使用、skill 选择、证据交接和终止策略。Agent 自主选择局部方法，同时遵守任务预算与已安装 harness；节省 token 不能替代覆盖重要任务要求。只有 EVO 可将实际成本与质量反馈一起用于经验更新，VAL/TEST 成本只用于报告。规划估计与实测分开记录，未知价格保持 null；冻结的 checkpoint 和发布包绑定两层状态、预算策略、prompt 与执行配置。

## 反馈、写入与复用

只有 EVO/stream 完成提交、外部评分和归因后，入选 Agent 才反思自己的执行轨迹。反思输入含自身实际事件、团队计划、相关归因假说和提交级数值反馈。反馈摘要不包含私有 criterion、reference answer 或裁判推理；整体得分不能证明某个 Agent 的独立贡献或策略的因果效果。持久 memory 应是可引用过程证据的条件化经验，不应保存任务答案或隐藏评分要求。

Meta experience 和全部 Agent update 在同一 SQLite 事务中写入同一个版本化 `ExperienceSnapshot`；身份、来源、证据引用或任一反思失败时不发生部分写入。重复任务和恢复使用已登记提交及更新，不重复学习。freeze 导出两层状态；rollback 同时恢复两层。VAL/TEST 使用冻结状态，不执行反思，也不把评测反馈写回任一层。当前任务和 held-out 来源的 memory 在检索前被排除，并按任务与角色筛选。

## 执行与实验范围

双层进化与执行调用次数是两个独立配置。默认 `single_pass` 每个执行角色一次模型调用；可选 `iterative_shared_ledger` 将工具结果和更新后的账本交付给仍在执行的角色，支持继续调用和 Writer 修订。当前 DAG 调度对每个角色只派发一次，不会因收到消息而重新唤醒已经结束的上游角色。失败草稿不能成为最终答案或下游完成产物。移除可选调用数上限仍保留 token 和 timeout 停止条件。

[`experiment_plan_v5_ZH.md`](experiment_plan_v5_ZH.md) 保留联合进化方案和父级题目来源；[`experiment_plan_v5_independent_ZH.md`](experiment_plan_v5_independent_ZH.md) 注册分源独立进化及迁移评价。正式 checkpoint、选版和测试产物必须绑定完整 Meta + Agent Pool 状态及其哈希，不能只导出 Meta advice。独立版的实际运行状态仍是 `PROTOCOL_FROZEN_NOT_RUN`，完整启动条件见 [`run_launch_plan_v5_independent_ZH.md`](run_launch_plan_v5_independent_ZH.md)。当前离线 smoke 和回归测试验证软件连接、原子更新、角色复用与 held-out 隔离，不构成模型性能或角色成熟度的实验结论。

实现入口：`jit_mas/agent_pool.py`、`planning.py`、`bridge.py`、`execution.py`、`pipeline.py`、`experience.py`。运行和限制详见 [`docs/jit_mas.md`](../../docs/jit_mas.md)。
