# Codex 实施任务：将 JIT 扩展为 Rubric-Driven、全局—局部协同的自进化 MAS

你是一名负责实现研究原型的资深 Agent Systems 工程师。请在当前 JIT 仓库内直接完成增量开发，而不只是给出设计建议。项目暂称 `JIT-MAS`，名称不是论文创新性声明。

本任务的目标不是把一个 Agent 复制成若干个固定角色，也不是在 JIT 外面再套一个完全独立的多智能体框架。我们需要保留 JIT 的任务条件化、可执行 harness 生成能力，将它扩展为能够根据预测质量要求动态组队、执行开放式生成，并从全局—局部归因中积累可追溯、可复用经验的系统。

## 1. 不可改变的研究目标

给定一道开放式任务，系统在作答前预测关键质量要求及其重要性，并据此组织当前任务的团队。全局分析器先依据问题和历史 rubric 经验，预测要求、分析关联、提出候选角色与初步分工；候选 agent 再从局部视角提出可承担要求、所需信息、能力边界和协作依赖；全局分析器协调重叠、遗漏和预算，形成当前任务专用的 MAS。

任务完成并提交答案后，独立 evaluator 给出逐条标准及反馈。系统对照事前预测与事后反馈，先由全局分析器梳理整体问题和 rubric 关联，再由各 agent 结合自己的完整可观测执行轨迹进行局部分析，最后全局整合，区分要求预测、团队组织和具体执行的问题，提出有证据的修改。

归因后的首个结构合法修改直接进入版本化经验，不再经过 accept / hold / reject 质量筛选。保留来源、适用范围、不确定性、原子写入与回滚；直接入库不代表质量改善。下一道任务利用这些经验重新预测、重新组队。

完整闭环必须是：

`预测要求 → 全局—局部协同组队 → JIT 生成可执行 MAS harness → 执行并提交 → 独立逐条评价 → 全局—局部协同归因 → 结构/来源检查 → 直接版本化更新经验 → 下一任务重新组队`

持续进化的是“理解任务质量要求，并将其转化为分工与协作的能力”，不是一支永远固定的团队。

本轮默认采用冻结模型参数的经验驱动实现。不要自动启动 SFT、DPO、RL 或大模型训练，也不要把记忆更新宣传成参数训练。保留后续训练所需的结构化轨迹和偏好数据接口即可。Evaluator 本轮保持冻结、独立和可配置；不要顺带实现 evaluator 自进化，使研究目标失焦。

当前执行契约采用 single-pass shared-ledger：每个执行角色最多调用模型一次，贡献者写入结构化共享账本，Writer 一次读取完成快照并提交。任务前全局—局部规划、任务后归因仍然保留；经验改为归因后直接更新，不再运行跨任务成对接纳验证；它们不是任务执行期的通信轮次。详见 `docs/jit_mas_single_pass.md`。本仓库未发现 SA-RQ、probe miner 或 reward-kernel 实现，不能声称相关 pipeline 已实现或测试。

## 2. 先审计真实代码，再确定改动位置

首先读取当前仓库的 `AGENTS.md`、README、依赖和 Git 状态，记录分支与 commit。保留用户未提交的改动；不要擅自 reset、删除文件、提交或推送。

JIT 上游入口：
- 仓库：https://github.com/bingreeky/JIT
- 论文：https://arxiv.org/abs/2608.25593

重点核对以下真实接口，具体以当前 checkout 为准：
- `jit/meta_agent.py`、`jit/harness_ops.py`、`jit/schemas.py`、`jit/prompt.yaml`、`jit/selector.py`。
- `scripts/kernel/protocols.py`、`scripts/kernel/types.py`、`scripts/kernel/runtime.py`。
- `scripts/models/`、`scripts/tools/`、`scripts/eval/`、`scripts/run_jit.py`。
- `harness_factory/harnesses/roma/`、`aggagent/` 等实际存在的分层、聚合实现，以及对应 descriptions。
- `benchmark/adapter/base.py`、`benchmark/registry.py`、`benchmark/config/`。

需要特别确认：
1. `BaseAction.run(task, ctx)` 在现有协议中拥有整个执行循环，不要假设它只是一个单步 action。
2. `RunResult` 已有 `sub_runs` 和 `metadata`；优先复用它们保存团队与局部轨迹，不要另造不兼容的结果体系。核对 `full_dict()` 与精简 `dict()` 的差异，归因所需的模型输入输出不能在序列化时被丢弃；也要检查具体 Memory 是否真正实现了完整轨迹读取。
3. 生成协议是四个 Python 模块加 `prompt.yaml` 的五个 tagged blocks；不要未经适配就在原解析器中增加第六个 block。
4. JIT 现有流程包含生成、选择、执行与异常修复。区分 selector/review 和任务 evaluator；前者不能读取当前任务的隐藏评分答案。
5. 论文描述、上游 README 和本地实现可能不完全一致。逐项记录差异，不要假设论文中的训练、streaming bank 或所有 seed 已经在当前代码中实现。

产出 `docs/jit_mas_codebase_audit.md`：用“现有路径/类/函数 → 复用方式 → 必须新增的能力 → 兼容性影响”描述改造，不要凭空编造已存在的接口。审计后继续实现，非阻塞性的设计选择自行作出并记录。

## 3. 必须参考的开放式 MAS 工作

研究以下一手论文与官方代码，聚焦能直接指导实现的机制。不要把阅读相关工作变成无限扩展范围的调研，也不要以代码可见等同于允许复制；借用源码前核对许可。

### 3.1 Meta-Team：分布式经验与协同归因
- 论文：https://arxiv.org/abs/2605.29790
- 仓库：https://github.com/zz-haooo/Meta-Team
- 重点检查 `core/reflection_runner.py`、`core/message_store.py`、`tools/reflection.py`、ResearchRubrics 的 evaluation 与实验入口。
- 借鉴 agent-local context 保留、事后跨 agent 证据交换，以及 agent/interaction/team 层面的经验组织。
- 不直接照搬“进化一支持久团队”的研究对象：本项目进化的是组队规则与经验，部署时仍然为每题重新生成 MAS。

### 3.2 Co-STORM：开放式报告生成中的多视角协作
- 论文：https://arxiv.org/abs/2408.15232
- 官方仓库：https://github.com/stanford-oval/storm
- 重点检查 `knowledge_storm/collaborative_storm/` 中专家、讨论管理、知识组织与报告生成实现。
- 借鉴多视角发现遗漏、证据组织、协调重复讨论和最终综合；不要强制所有任务采用 Wikipedia 文体、固定专家数量或人类参与流程。

### 3.3 ResearchRubrics：真实开放式任务与逐条评价接口
- 论文：https://arxiv.org/abs/2511.07685
- 官方仓库：https://github.com/scaleapi/researchrubrics
- 数据集：https://huggingface.co/datasets/ScaleAI/researchrubrics
- 检查数据字段、逐条 rubric evaluator、原始 judge prompt、评分聚合和异常处理；它是 benchmark，不是 MAS 方法。
- 当前官方实现的聚合为 `sum(weight * score) / sum(positive weights)`，负权重保留在分子中；分母为零时的上游行为也需要保留并标记。以实际固定版本验证这些语义，不能改成普通平均或悄悄裁剪分数。

在 `docs/jit_mas_related_work.md` 中记录每项参考的具体机制、文件、版本与本项目差异。把“源论文已有机制”“本项目提出的组合/扩展”“仅为工程简化的假设”分开。无法联网时记录未核验项，继续使用已可访问的本地资料，不得伪造调研结果。

## 4. 架构边界：JIT 是生成与执行基础，不是装饰

新增一个薄的任务级协调层，串联 rubric 预测、事前协同组队、JIT 生成、单次执行、评价和经验更新。继续复用 JIT 的模型客户端、工具注册、harness 加载、结果序列化、基准适配和执行前生成/修复逻辑。不得另引一整套 MAS 框架来替换 JIT。

建议分离三个职责：
- `GlobalAnalyzer`：事前预测与协调，事后全局归因与修改提案。事前和事后使用不同上下文，不能发生未来反馈泄漏。
- `JITHarnessSynthesizer`：接收任务、预测 rubrics、最终 TeamSpec、可用能力和已提交经验，通过真实 JIT 生成路径产生可执行 harness。
- `TeamExecutor`：运行生成的 harness，管理独立 agent 上下文、结构化共享账本、工具、预算和单次最终综合，不负责给自己打正式分数。

模块映射建议：
- Memory：私有局部上下文、共享结构化贡献/证据、不可变事件日志和只读完成快照。
- Planning：将冻结的 TeamSpec 转化为局部指令及真实前向数据依赖，不做任务执行期重规划。
- ToolPolicy：贡献者可在其唯一调用中请求一批获准外部工具；Writer 只做终端完成/提交，不开放外部工具或 agent 间通信操作。
- Action：按实际协议执行贡献者、确定性合并账本、让 Writer 单读单写、终止及收集局部结果。

TeamSpec 必须影响实际执行的角色、工具、依赖、检查点和预算，而不是只写进日志。通过类型化配置/sidecar 传入，具体与现有 loader 兼容；不要依赖跨任务全局变量。

主路径必须支持：`TeamSpec → JIT 五文件生成 → 校验/选择 → 原生加载执行 → 分 agent 轨迹`。生成器可以复用新增的稳定 MAS primitives。单次账本协议是共享执行边界，不是固定领域团队；领域能力、rubrics、贡献结构和必要的前向依赖仍由当前任务决定。

允许提供 `template/mock` 后端完成离线测试，但必须在结果里标明，不得把模板运行或普通模型替代 JIT checkpoint 的结果声称为原版 JIT 复现。真实 `native_jit` 路径必须接通，缺少 endpoint 时明确失败，不能静默回退后报告成功。

## 5. 最小结构化数据契约

优先使用项目现有的数据建模方式。字段需要类型校验、稳定 ID、schema_version 和可序列化输出；名字可随真实架构调整，但以下语义不能缺失。

### 5.1 PublicTask / PrivateEvaluationRecord
`PublicTask` 只含执行允许看到的问题、附件、显式约束和可用工具。隐藏 rubrics、评分权重、参考答案、未来反馈属于独立的 `PrivateEvaluationRecord`。

### 5.2 PredictedRubric / RubricGraph
每条预测包含：`rubric_id`、可操作的质量要求、来源（显式/推断/历史经验）、非负 importance、独立的 confidence、期望证据、适用条件、是否为禁止项及相关经验 ID。

重要性不是置信度，更不是 evaluator 的真实权重。图边支持 prerequisite、support、overlap、tradeoff 等关系，并记录证据/理由与不确定性。Rubric 关系图不必是 DAG；不要把关联自动解释成因果。

### 5.3 LocalPlan
包含候选 agent 的能力签名、可承担 rubric、建议补充/细化的要求、所需输入、预期输出、协作对象/依赖、工具需求、预算、不能覆盖的部分和风险。候选 agent 可以质疑全局初稿，不能只能“确认接受”。

### 5.4 TeamSpec
包含团队角色、职责、rubric-to-agent 多对多覆盖映射、主责与前向复核安排、产物依赖、工具权限、预算、最终综合职责、终止条件及选择理由。已有 `max_calls` 与团队预算保留为资源上界，但每执行角色的实际模型调用上限为一次；未分配的预算不能形成第二轮执行。

不要“一条 rubric 一个 agent”，不要固定永远是 researcher/writer/critic，也不要要求每个任务必须使用多个执行 agent。简单任务允许退化为单执行 agent；复杂测试必须能实际调用两个以上具有独立上下文的 agent。

### 5.5 EvidenceRef / TeamEvent
保存 `run_id`、`agent_id`、`event_id`、时间、来源工具/贡献、产物版本、内容 hash、可定位的片段和父事件关联。区分“检索到”“发布到账本”“被下游消费”“进入最终答案”；历史消息记录与事后归因证据访问继续保留原有语义。

当前共享账本确定性合并 `requirements`、`outline`、`evidence_spans`、`source_references`、`contributions` 与原始 `tool_evidence`。它不是调用/token 计费账本，也不是额外的 LLM 角色。历史消息事件保持为历史证据，不能据此重新开放任务执行期的消息队列。

完整轨迹指实际可观测的模型输入输出、工具交互、消息、决策记录与产物，不要求获取模型供应方未暴露的内部思维过程。

### 5.6 EvaluationFeedback / RubricAlignment
反馈保留官方 criterion、原始 weight、score/verdict、评价理由、可提供的答案证据定位、evaluator 版本及缺失/错误状态。

Alignment 支持同义、部分覆盖、合并/拆分、多对多、漏预测、未匹配预测和重要性差异。禁止仅依靠字符串完全相等。未匹配的预测不自动等于错误，因为评价标准也可能不穷尽全部质量维度。匹配不确定性应保留。

### 5.7 AttributionFinding / ChangeProposal
归因记录相关 rubrics、agent/交互/组织组件、支持及反对证据、可能的替代解释和不确定性。

修改记录目标经验/策略、旧版本、具体 diff、产生原因、证据链、预期收益、风险和适用条件。归因后的首个修改通过结构与来源契约检查后直接写入，记录 update receipt；不再包含验证计划、接纳状态或验证结果。

## 6. 单任务完整数据流

### A. 读取经验并预测要求

从当前已提交状态快照读取相关 rubric 经验、组织策略和局部执行经验。全局分析器只依据 PublicTask、公开能力说明和合法历史经验，生成 `R_global` 与初始角色提案；此时不要调用 evaluator，也不要开始产出最终答案。

预测应包含问题特有的隐含要求与重要性，不是把固定的“准确性、完整性、流畅性”清单套在每题上。固定类别只能作为先验。

### B. 执行前全局—局部协同规划

让候选 agent 并行输出 LocalPlan。全局分析器合并局部新发现，处理重复职责、遗漏要求、跨 rubric 依赖、冲突和预算，生成 `R_planned` 与 TeamSpec。

保留“全局初稿 → 局部反馈 → 全局整合”的执行前规划。`local_rounds` 与 `local_planning` 只作用于这一事前阶段，不是执行 agent 的通信、澄清或回写轮次，不新增任务执行期协商配置。

在正式执行前分别冻结 `R_global`、`R_planned` 和 TeamSpec，记录 hash、时间及经验版本。后续不能用 evaluator 的标准覆盖这些文件；这样才能分别衡量全局预测和局部规划的贡献。

### C. 调用 JIT 生成并验证 MAS harness

将上述结构作为生成上下文接入 JIT。校验生成代码的协议兼容性及 TeamSpec 一致性，调用已有 selector 的合法路径选择候选。选择时不得使用当前任务的隐藏 rubric 分数。

执行前错误修复与质量进化必须分开：在任何执行角色的模型调用开始前，编译、接口等异常允许有界 repair；一旦角色调用开始，执行失败必须保留，禁止通过 repair 后整队重跑。当前答案仅仅得分较低，也不能触发重做同一测试题并覆盖原结果。

### D. 单次共享账本执行

执行 agent 具有私有上下文，每角色最多一次模型调用。默认 Analyst 与 Evidence 独立、可并行：前者写任务要求与提纲，后者写证据片段与来源。两者都向结构化共享账本贡献，不向其他 agent 发消息，不广播完整聊天历史。较小任务可合并职责；真正必要的前向数据依赖 DAG 保持兼容。

贡献者可在唯一调用中请求一批合法外部工具；结果自动进入账本，不再次召回贡献者。调度尊重前向依赖、并发限制及总预算，拒绝依赖环；不能把 rubric 关联图直接当执行 DAG。不存在 `send_message`、`read_evidence`、`raise_issue` 或 agent 消息队列。缺少信息应写为 limitation，不触发 clarification。

账本由运行时确定性合并，不调用 LLM 求共识。Writer 一次读取完整只读快照，一次综合最终产物；只使用 `complete`/`final_answer` 等角色允许的终端协议，不调用外部工具、不重新请求贡献者。综合须处理证据冲突、信息缺口和未满足要求，保留来源与不确定性，不能仅拼接文本，也不能强制创作任务使用无关引用或百科体裁。自报检查不是独立验证。

Writer 的执行提示必须包含：

> Read the structured shared ledger once, synthesize the final response from the analyst requirements and evidence spans, and do not initiate additional inter-agent communication.

任务执行期不做迭代协商、澄清或反馈回写；任务前动态规划、任务后有界归因追问仍保留；经验直接更新，不再运行跨任务成对质量筛选。

### E. 冻结提交并独立评价

保存最终答案及 hash 后，才能对正式提交调用 evaluator。Evaluator 输入应为任务、正式标准、答案和确有必要的参考材料；不默认输入本系统预测的 rubrics、组织设计或 agent 自我评价，避免用自己的标准循环证明自己优秀。

评分之后才把逐条反馈释放给归因阶段。公开任务约束可以提前看；隐藏评价标准不可以。权限边界必须由数据接口与执行环境落实，不能只写一句提示词禁止读取。

### F. 全局—局部协同归因

1. 对齐两版事前预测与评价标准，记录遗漏、粒度差异、重要性偏差与未匹配项。
2. 全局分析器读取逐条反馈、TeamSpec、最终产物、事件索引及共享摘要，识别整体得失、跨 rubric 关联和待解释问题。不要默认把全体长轨迹拼进一个上下文。
3. 将相关评价维度分配给相应 agent 的局部分析器。局部分析器继承该 agent 的局部执行上下文，并可按索引读取完整原始轨迹和相关上下游证据；不能只看最终答案或截断摘要。
4. 各局部分析器提交成功/失败原因、证据引用、替代解释和修改建议。必要时进行一次有界的跨 agent 追问，不假定局部自述天然可信。
5. 全局分析器整合为以下类别，可多标签，不能强行把每条失败唯一归给某一个 agent：
   - prediction：要求被遗漏、误解、重要性估计不当或关系判断错误；
   - organization：已有要求没有落实责任/协作/预算，或交接、复核、综合机制不合理；
   - execution：组织安排合理，但具体检索、推理、生成或核验没有做到；
   - external_or_uncertain：工具/环境问题、反馈冲突、证据不足或原因未知。
6. 没有任何 agent 负责的遗漏项优先审查预测/组织，不能随便责怪某个局部 agent。成功经验也需要保留适用条件与证据。

初版把结论称为“证据支持的归因假设”。可提供少量局部重放/替换干预来验证关键假设，但不强制计算全量 Shapley，也不能把 LLM 反思文字当成已经识别的真实因果效应。

## 7. 经验更新与保留准则

维护三个逻辑经验库，可共享一个轻量持久化后端：
- RubricExperience：哪些任务信号意味着哪些质量要求，常见遗漏、重要性规律、依赖/权衡及反例。
- OrganizationExperience：什么要求组合适合什么角色、职责边界、交接与复核、资源分配方式。
- ExecutionExperience：面向能力签名/角色功能的局部检索、综合、证据核验和生成经验。

局部经验按能力或职责签名检索，不能只绑定一次性的 `agent_3`。经验必须携带来源任务、证据、适用范围、提交版本及反证；不要无限追加未经压缩的反思全文，更不要把题目专属答案作为泛化能力沉淀。

持久状态可表示为 `S_n=(rubric_bank, organization_bank, execution_bank, policy_versions)`，并与 JIT 的 harness references/archive 关联但不混为一谈。每道新题执行 `BuildMAS(task, S_n)`；freeze 的是经验和策略快照，不是固定 TeamSpec。

### 7.1 归因后的直接更新

保持当前每个来源任务至多处理首个 reconciled proposal 的简单顺序策略，不按验证分数选择候选。首个提案经过结构、允许目标、来源、基础版本和幂等检查后，直接原子写入经验；没有提案则不更新。

删除 staging → paired validation → accept/hold/reject 的质量选择流程，不再要求独立验证题、接纳阈值、最小样本数或 pending 状态。契约错误仍然明确失败，不应伪装为质量拒绝；自报 confidence 也不代表修改有效。

直接写入减少额外调用，但不能保证经验有益或单调改善。正式效果应通过冻结状态与独立测试任务衡量。可选 validation 数据仅用于外部分析；如果人工根据分析调参，应记录为开发暴露，不能继续称为 untouched test。

### 7.2 提交与回滚

持久化保留 proposal ID、来源任务和证据、适用范围、不确定性、版本、update receipt 与可回滚快照。重复运行不能重复写入，旧基础版本不能覆盖当前状态。evaluate/freeze 模式保持只读，测试反馈不能改变后续测试题的经验。

执行 agent 不得直接编辑 evaluator、评分标准、split manifest、预算守卫、运行内核或经验库。进化只通过受控提案与直接提交接口写入允许目标；移除质量筛选不等于移除安全边界。

## 8. Benchmark 与防泄漏协议

先实现 ResearchRubrics adapter，同时提供少量不依赖网络的合成 fixtures；fixtures 只用于软件正确性，不能冒充正式 benchmark 结果。

适配时同时返回正式分数与逐条反馈。保留原始评价字段、带符号权重、缺失项和失败状态。不应因某条 rubric 的 API 失败而从分母删除它以提高得分；标明评价不完整并按固定重试/失败协议处理。

不要把原始带隐藏 rubrics 的数据项直接传入 JIT 生成器、planner、executor、工具工作目录或可检索记忆。

至少支持：
- `evolve`：仅在 evolution split 上归因并生成修改，将首个结构合法提案直接写入版本化经验；不运行 validation 任务作为接纳门槛。
- `evaluate`：对冻结状态进行 held-out 评测；当前任务仍动态组队，但禁止更新持久经验，不把上一道 test 的反馈传给下一道 test。
- `stream`：明确标识的在线设置；第 t 个任务的最终提交发生在其反馈释放之前，反馈只影响后续任务。固定顺序、记录每步状态版本，不与 frozen held-out 指标混报。

保存显式 split manifest、种子、任务 ID 和去重规则。进化/可选外部分析/测试以完整 task 为单位划分，不把同一题的 rubrics 拆到不同集合。检索时排除当前任务、受限集合及未来记录；可选 validation 仅供外部分析，不参与在线写入判断，其原文反馈不应成为普通执行记忆。

无隐藏 rubric 可用的真实部署任务，允许独立 evaluator 依据任务与外部质量政策生成评价标准，但必须记录其来源为 model-generated，不能伪称人工 ground truth，也不能使用预测 rubrics 替代独立评价。

## 9. 运行安全、预算和可复现性

JIT 工作目录隔离不自动等于安全沙箱。核对当前实现；生成代码和工具执行不能访问 evaluator 私有数据、宿主凭据和无关文件。优先复用可用沙箱/能力代理；没有可信隔离时，不得声称隐藏数据已被安全隔离，真实不可信代码路径应 fail closed，或仅在用户显式启用的 unsafe-local 模式运行并清楚标记。

复用现有客户端，分别配置 meta/global/local/exec/judge 的模型、endpoint、key 环境变量、超时和 token 限额。允许同型号复用，但各角色上下文独立。不要硬编码模型名、真实 key、价格或假定 meta 与 exec 应有相同输出上限。

建议低成本默认值：最多 4 个执行角色、最多 2 个并行执行、1 轮局部规划、1 轮局部归因、最多 2 次异常修复、1 个 harness 候选；这些均可配置，meta 生成上限仍须足够输出合法五文件。

实现团队级、线程安全的预算计量：全局分析、局部规划、JIT 生成/选择/修复、执行、综合、评价及归因都不能遗漏；已删除的成对验证不应继续计入调用或成本，也不能因 sub_runs 被重复计费。每个 agent 的预算是团队预算的分配，不是每个 agent 都独占一份完整预算。

分别报告单题推理成本、外部评价成本和经验更新成本；额外分析另行计量，不虚增已删除的验证成本；同时保存 token、模型调用、工具调用、通信量、墙钟时间及失败原因。没有可信价格表时费用为 unknown，不能写成零。

对缓存、resume、状态快照使用包含 task、模型/提示词、代码、经验版本、evaluator 版本和输入产物 hash 的标识。不同策略的运行不得错误命中同一旧缓存。首次版本优先采用串行跨任务经验更新；任务内部可并行。

## 10. 建议组织与必须提供的接口

以下是新增结构建议，不是对上游已有文件的声明；依据审计结果调整，避免无谓重构：

- `jit_mas/`：schemas、global/local planning、JIT bridge、single-pass team execution/shared ledger、rubric alignment、attribution、experience、budget。
- `harness_factory/harnesses/rubric_mas/`：合法的五文件 MAS 参考实现，供原生 runtime 与 JIT 参考生成使用。
- `harness_factory/descriptions/rubric_mas.md`：对应的协议内设计描述。
- `benchmark/adapter/researchrubrics.py` 与匹配的 config/registry 接入。
- `scripts/run_jit_mas.py`：统一 CLI，提供 evolve/evaluate/stream、单任务 smoke、freeze/rollback 的实际入口。
- `tests/jit_mas/`、`docs/`、轻量 fixtures、`.env.example` 配置示例。

阶段间必须有可测试的函数/服务边界，包括 predict、local_plan、reconcile、synthesize、execute、evaluate、align、attribute、propose、commit。不需要为每个名字单独建文件，也不要引入庞大微服务设施。

## 11. 消融与观察指标

配置层至少能关闭局部规划、局部归因和持久经验，避免以后做消融时重新改主执行路径。为下列比较预留可运行配置，初次不要求执行昂贵全量实验：
- 原版 JIT；不要未经检查将它命名为“纯 single-agent”，因为其 harness 可以包含递归/聚合结构。
- 固定 MAS。
- 依据任务动态生成 MAS，但不显式预测 rubrics。
- 完整方法但无局部规划；完整方法但仅全局归因；完整方法但不更新经验。

主指标使用官方 task quality；诊断指标包括事前预测的加权覆盖、语义匹配精度/不确定性、规划责任覆盖、遗漏来源、跨 agent 交接与证据传播、直接更新数及跨题收益。规划覆盖不等于最终满足，局部自报 confidence 不等于已校准概率。

可另设 oracle-rubric upper-bound 实验，但必须显式标注 privileged information，不能混进主结果或默认配置。不要强制假设任何消融或完整方法必然获胜。

## 12. 必须通过的验收测试

1. 原版 JIT 入口和至少一个原有 seed 的离线回归保持兼容；新增 adapter 在正确注册后可被公共评估路径加载。
2. 原生 JIT bridge 的集成测试实际经过五文件解析、加载和协议执行，不是绕开 JIT 的模拟函数。可用 scripted model 返回固定合法代码，但需明确这是软件测试。
3. 两种不同任务通过 planner 产生不同的职责/依赖/预算或人数；不能只换角色名称。复杂 fixture 实际存在独立 agent 会话及可追踪的交接。
4. 局部规划能改变全局初稿；同时保存 R_global、R_planned 与 TeamSpec。最终提交之前没有正式 evaluator 调用。
5. 隐藏 rubric/参考答案的 canary 不出现在执行模型输入、生成 prompt、共享工作区或 agent 可访问路径中；同时测试公开显式约束仍可正常使用。
6. Rubric alignment 支持同义改写、部分匹配、拆分/合并、遗漏和不确定匹配，不改写原始预测。
7. 用带已知事件证据的 fixtures 分别验证：未预测要求、预测了但无人负责/交接失败、安排合理但执行失败，以及证据不足不强行归因。成功案例也能生成有条件的经验提案。
8. 经验更新直接应用归因后的首个合法提案，不再生成 accept/hold/reject 结果或调用成对验证；来源、目标、版本、幂等与只读守卫仍有效。持久化后必须能影响另一道任务，而非只改日志。
9. ResearchRubrics 聚合与固定版本官方函数一致，覆盖正负权重、零分母、失败项；严格区分正式指标与我们新增的诊断指标。
10. 总预算在并发情况下正确扣除；每执行角色至多一次模型调用，子运行不重复计费；超时/依赖失败不会死锁，也不会通过 repair 后整队重跑。并行贡献确定性进入账本，Writer 只读一次；消息类操作与 Writer 外部工具必须被拒绝。
11. freeze/evaluate 模式不能写经验；stream 只读取过去版本；resume 不重复提交或误用旧状态缓存；rollback 恢复正确的已提交快照。
12. 运行没有凭据的 smoke 不会发出真实付费 API 请求；缺 endpoint 的 native 模式给出明确错误，不会伪造真实结果。

合成测试可以使用技术调研、实验方案和创意写作等结构不同的任务。示例和预设输出只用于验收，不得把任务 ID/关键词的 if-else 规则写进真实 planner，制造“动态组队”的假象。

## 13. 实施次序与最终交付

按可运行的纵向切片开发，每一步执行对应测试：
1. 审计与数据契约、隐私边界和预算接口。
2. 全局—局部事前规划 → TeamSpec → 原生 JIT MAS 生成与执行。
3. ResearchRubrics 接入 → 事后全局—局部归因。
4. 归因修改 → 直接版本化经验持久化、freeze、rollback 与下一任务复用。
5. 补齐消融开关、CLI、日志、回归与端到端测试。

不要只完成组队就宣称闭环完成，也不要把直接更新与复用写成 TODO。无凭据时完成可执行代码、scripted integration tests 和明确的真实调用入口；将未运行的真实 benchmark 诚实列为未验证。

最终交付：实际修改后的代码、简明架构说明、审计/相关工作文档、配置示例、数据准备步骤、精确到命令的离线 smoke 和少样本真实运行说明、测试结果及尚存风险。

在最终报告中说明：改动了哪些真实路径，复用了 JIT 什么机制，如何保证 MAS 不是固定模板，三类经验怎样影响下一题，哪些测试实际执行，哪些需要凭据/数据/沙箱，以及是否发生任何与任务目标不一致的工程降级。

不要自动跑全量 benchmark 或训练，不要捏造论文指标，不要承诺性能提升或论文新颖性。先给出简短实施计划，然后继续修改并测试，直到完成可运行闭环或遇到明确的外部阻塞。
