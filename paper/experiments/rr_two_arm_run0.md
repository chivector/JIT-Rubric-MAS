# ResearchRubrics run0 状态与失败记录（无有效两臂比较）

本入口只运行 v5 冻结成员中的 ResearchRubrics，执行 run0 一次；这是 RR 单 benchmark 对比，不是六 benchmark 联合进化实验的完整复现。

- 原计划为两臂同时启动：Direct Single-Agent 直接生成全部 33 道 TEST；JIT-MAS 从空经验开始。本轮实际先完成了 baseline v15，JIT-MAS v16 独立运行，不能按并行初始运行报告。
- JIT-MAS 按 run0 中 RR 的相对顺序遍历 20 道 EVO，保存 C0、C5、C10、C15、C20。
- 每个候选评估同一固定 10 道 VAL，每题一次。完整评分至少 9/10 才可选；使用每题固定理论范围归一化，缺失原生分数为 null、仅选择效用计 0；平局按完整数、较早位置、状态哈希选择。
- 完整状态相同的候选允许复用同身份 VAL 结果并记录来源。无合格候选则不确定，不补跑有利轨迹。
- 原计划两臂使用相同 DeepSeek 执行模型、公开题面和冻结证据；本轮实际为闭卷探索，未使用共享公开证据目录，GPT judge 独立评分。TEST 不更新经验、不按得分重采样。
- 66 个 TEST 槽位全部提交或登记失败后封存，再释放评分。完整题均值缺失时保持 null；另列完整子集均值、完整配对数、失败和 token 开销。

正式运行前必须准备覆盖全部 63 道选中题的公开证据目录，含可验证的 `manifest.json` 与证据包。`--closed-book` 只用于另行明确同意的协议变更；本轮因证据目录尚未就绪，仅作为闭卷探索，不能支持正式优势结论。

执行入口：

```powershell
.venv\Scripts\python.exe -m scripts.run_rr_two_arm_pilot `
  --data outputs/researchrubrics_live_20260930/processed_data.jsonl `
  --joint-manifest paper/experiments/joint_task_splits_v5.json `
  --evidence-dir <verified-evidence-directory> `
  --output <new-output-directory>
```

凭据只从进程环境 `RR_EXEC_API_KEY` 与 `RR_JUDGE_API_KEY` 读取，不写入实验配置。默认模型为 `deepseek-v4-flash-vision` 和 `gpt-5.6-sol`。`--check-only` 验证固定数据 SHA256、成员与证据包，不调用模型，也不要求凭据。预检目录与正式运行目录分别使用新路径。

用户允许通用知识后，探索版可使用 `--closed-book`。`--arm baseline` 单独运行固定 33 道 TEST；`--arm ours` 单独进化、选版后测试；默认 `--arm both` 同时启动两臂。

连接故障后可用 `--arm baseline --closed-book --reuse-baseline-from <source-output-directory>` 在新目录保留已有未评分生成记录，只生成缺失题。来源数据、TEST 成员、执行配置与答案哈希均须一致；复制的生成预算和来源路径写入产物。新评分必须统一 judge 身份，不能混用旧 judge 的分数。此类恢复包含此前失败请求的重试，须披露，不能表述为无故障的初始一次运行。

`--judge-model`、`--judge-max-tokens`、`--judge-attempts`、`--judge-parallel` 与 `--max-inflight-requests` 均在新运行前固定。`JIT_MAS_MODEL_ATTEMPTS` 控制传输重试；`JIT_MAS_DISABLE_KEEPALIVE=1` 禁用连接复用。默认启用 TLS 校验；临时 `JIT_MAS_TLS_VERIFY=0` 可配合 `JIT_MAS_TLS_ENDPOINT` 仅限制指定执行端点，评分端点保留校验。传输故障的 token 用量可能仅为估计，无法据此可靠估算费用。

`--structured-output json_schema` 是默认的 JIT 输出约束：规划和归因使用请求中原有的 Pydantic JSON Schema，执行角色使用与运行时协议对应的 JSON Schema。规划的自由字符串默认最多 2048 字符、communication 最多 1024 字符、数组最多 64 项，分别通过 `--planning-string-max-length`、`--planning-communication-max-length`、`--planning-array-max-items` 固定，避免网关重复生成耗尽输出预算。执行 checkpoint 键精确绑定当前角色的检查项，evidence ID 绑定实际观测事件，source ID/ref 要求单一引用；运行时仍校验引用是否实际存在。v16 的答案和证据正文未加 decoder 长度限制；修复版 v20 在保留多行文本的前提下按输出预算限制长度，参数见末尾修复记录。参数写入运行元数据和冻结身份；Single-Agent 的直接答案仍为自然文本，judge 的原始 rubric 提示和评分规则保持原有实现。网关不支持 schema 时可在另一个新目录显式选择 `json_object` 或 `none`，不得在冻结运行中切换。格式错误仍登记失败，不修补已有答案或评分。

默认 `--judge-parallel 2` 在每个 rubric 使用独立 judge 实例并共享当前题的计量账本；整个进程的请求并发仍最多 2。两臂使用同一评分并发、原始 rubric 顺序、提示和有符号权重聚合。可显式冻结为 `--judge-parallel 1` 使用顺序评分。

产物包括 `pilot_metadata.json`、两臂 `report.json`、checkpoint journal、逐题提交和预算、`test_release/seal.json`、逐题评分及 `comparison.json`。运行期间冻结代码、配置、数据、清单与证据哈希；工程修复须保留旧记录并另建新版本，不能边跑边修改冻结身份。

此前 `outputs/rr_two_arm_pilot_20261001` 两臂因 split 路径错误在实验 API 请求前退出；该目录保留为失败预检，不计入真实 benchmark 成绩。

2026-10-02（北京时间）已对 `https://hk.xty.app/v1` 的 `gpt-5.6-sol` 完成一次独立 JSON 评分可用性探测，网关返回 `model=gpt-5.6-sol`、`finish_reason=stop`。该标识不能独立证明上游模型快照；探测不计入 RR 成绩。此前 Single-Agent v7 的 33/33 成绩 `0.4275497708174814` 来自 `gpt-5.4`，只能作为旧 judge 结果；新对比须复用封存答案并统一重评分。JIT-MAS v9/v10 为无有效 TEST 成绩的失败/中断目录；v10 的首两次失败为多余 JSON 数据和 LocalPlan 多余字段，输出未达到提高后的 token 上限。

Single-Agent 的统一 `gpt-5.6-sol` 重评分已在 `outputs/rr_single_agent_run0_gpt56sol_20261002_v15` 完成：33/33 完整，全量均值 `0.46926705010467296`；33 份原封存答案全部复用，27 份同 judge 身份的完整评分从 v8 复用，其余重新评分以恢复连接失败。实际响应 model 全为 `gpt-5.6-sol`。`summary.json` 审计显示 826 个 rubric、32 道含负权重题、0 个 incomplete，33/33 答案哈希一致；`valid_two_arm_comparison=false`。这是闭卷探索结果，仍不构成有效两臂比较。JIT-MAS 当前独立目录为 `outputs/rr_jit_mas_run0_gpt56sol_20261002_v16`，执行代码冻结于 `.runtime/rr_gpt56sol_repair_20261002`；此前 v11/v12/v13/v14 均为失败或中断的工程恢复尝试，须保留并披露，不能合并成一次无故障轨迹。

v16 已于 `2026-10-02T06:03:57Z` 结束，终态为 `inconclusive`：20/20 道 EVO 已处理，2 道运行完成、18 道失败，仅一次完整经验更新到 v1；最后一题仍有一个 rubric 评分失败，运行完成不等于完整评分。C0/C5 为 4/10（C5 复用 C0），C10/C15/C20 为 0/10（C15/C20 复用 C10），五个候选全部不合格；未选择 checkpoint，未生成 TEST 答案，TEST 均值保持 null。运行耗时 3922.75 秒，记录 594 次模型调用、6,192,041 tokens，费用未知。

只读诊断发现闭卷适配存在职责冲突：seed pool 的 Searcher 保留检索与来源核验策略，规划即使把工具清单修正为 `[]`，仍分派搜索论文或核验 URL 的职责。一些角色反复声明即将检索、返回 `continue=true`，没有实际工具调用，最终耗尽计划中的 `max_calls`。其他独立工程失败包括 proposal ID 跨来源重复、归因引用了不存在的 agent/rubric、JSON 截断和连接故障。上述记录属于执行可靠性诊断，不能据此声称多智能体优势。

修复版显式传递并冻结 `model_general_knowledge_allowed`，贯穿 planning、execution 和状态身份；检索角色在闭卷上下文中整理模型一般知识，保留 Agent Pool 的原始身份、动态图组队和经验进化。归因 schema 绑定实际 agent、rubric、evidence ID；proposal ID 带来源任务与状态版本；最终 synthesizer 即使为 Critic 角色，也须返回完整任务产物。执行 schema 根据每次请求的 token 上限限制 answer 字符数为其两倍，checkpoint reason 最多 512 字符、ledger 单项最多 2048 字符，并保留多行文本。原始评分规则与旧答案不修改。

不评分的公开 VAL 输入预检 v17 因最后调用超过任务预算失败，v18 因长 Critic 评审导致 JSON 截断失败；两次均不计分且记录保留。v19 预检通过，约 38 秒提交终稿，judge 请求 0。v20 在 C0 阶段发现角色擅自添加有限调用上限、一般知识引用不合法来源和连接失败，已中断，未产生 EVO 或 TEST。新冻结版 v21 把未配置的调用上限绑定为 null，并禁止闭卷生成来源记录，从空经验启动同一计划；目录为 `outputs/rr_jit_mas_run0_gpt56sol_20261002_v21`，代码冻结于 `.runtime/rr_closed_book_repair_20261002_v21`。当前工作区另外加强了终态提交必须携带 checkpoint 的 schema 约束，该改动未注入正在运行的 v21。

v21 已于北京时间 2026-10-02 15:34 停止：7 个 EVO 失败、1 个已启动但中断、12 个未开始，经验仍为 v0；C0/C5 均 7/10（C5 复用 C0），不满足 9/10。没有 TEST 答案或评分，均值仍为 null。原 journal 的 running 和 report 的 evolving 保留，另写 interruption.json 标识人工停止；不可当作完整运行。已落盘记录为 511 次模型调用、5,102,238 tokens，不包含第 8 题尚未落盘的请求，费用未知。

EVO2 的 Analyst 连续 60 次返回相同 canonical JSON，continue=true、只有标题、无工具或 peer 请求；后续输入只有新增 assistant 历史，输入从 4,699 增至 50,422 tokens，最终耗尽预算。其余已定位失败包括三次 lesson evidence/counterevidence 未包含在顶层 update.evidence、一次重复绑定 pool member、一次 reviewer 未依赖 primary owner，以及一次 API 连接失败。没有自动补证据或替换失败答案。

修复版增加：无新外部输入的 continue 回合追加明确 user 指令，相同响应重复后给予一次纠正，再次重复则登记失败；有效内容修订、peer 输入和工具观察保留正常迭代，最近记忆保留待修订草稿。`explicit-user-no-progress-v1` 写入身份和执行元数据。反思纠正列出缺失的顶层证据 ID，pool/reviewer 校验列出具体冲突；原拒绝规则、评分和 9/10 阈值不变。v22 不评分预检发现执行网关拒绝 schema 的 if/then（HTTP 400），judge 请求 0，6 次模型调用、78,613 tokens；没有正式 EVO/VAL/TEST 成绩。v23 冻结于 `.runtime/rr_closed_book_repair_20261002_v23`，以支持的 anyOf 表达终稿/checkpoint 或继续/tool 请求契约（rr-execution-v4），原 runtime 强校验保留。先做不评分公开 VAL 输入预检，通过后再从空经验运行原 20 EVO/五次 10 VAL/33 TEST 计划。旧实验各自独立保存，后续尝试不能报告为无故障的一次初始运行。

v23 不评分预检在执行阶段拒绝了 checkpoint 说明字符串（原契约要求 boolean 或结构化 status/reason），7 次调用、77,215 tokens，judge 请求 0；v24 在规划阶段拒绝 synthesizer 漏掉 critic 依赖的 DAG，6 次调用、69,332 tokens，judge 请求 0。v25 冻结于 `.runtime/rr_closed_book_repair_20261002_v25`，使用完整对象 anyOf 分支（rr-execution-v5），避免 decoder 不继承父节点 properties；解析合并 observation/ledger 消息中的全部可见 evidence ID。`explicit-user-no-progress-checkpoint-v2` 保留无进展保护，并仅为 checkpoint 格式错误提供一次角色内纠正，错误轨迹和预算保留；未观察证据、非法工具/peer、缺失答案仍失败。真实 checkpoint 示例与纠正指令保持诚实限制，不修改返回内容或评分。

离线全量回归在修复最近记忆前为 936 passed、1 failed（旧测试期待最后一条 assistant）；修正后相关测试分组通过，最后 planning/pool/reflection/runner 119 passed，execution/runner/agent_pool 114 passed，追加 checkpoint 边界测试 34 passed；diff check 通过。v25 无评分预检执行协议通过：92.187 秒、8 次调用、73,685 tokens，judge 请求 0，无经验更新，不计正式成绩；答案虽为 9,264 字符，但后续人工审计发现风险段落未写完，不能称内容完整或质量通过。随后从空经验启动 `outputs/rr_jit_mas_run0_gpt56sol_20261002_v25` 完整计划，baseline v15 保留；当前仍无方法组 TEST 结果，不能报告差值或优势。

当前导出交接包为 `paper/experiments/rr_run0_gpt56sol_20261002`，包含 v15 完整评分、v16 终态失败记录及工程预检/中断摘要；v21 按 interruption.json 标为中断。所有旧轨迹独立保留，不合并为一次成功实验。

v25 于北京时间 16:40 封存 C0：9/10 完整、eligible=true，normalized selection utility=0.14616904918346751。该值有理论范围归一化偏移，不能与 Single-Agent TEST 官方均值比较。EVO1 已开始。C0 有多份 title-only 原始 model_output，被原样封存并评为低分；有一题由 no-progress guard 阻断。没有修改已封存答案、重采样该 VAL 或注入工作区改动。

v25 已于北京时间 2026-10-02 17:01 退出（原报告 `failed/SystemExit=1`，pilot `incomplete`）。终端日志报告 Judge HTTP 401/额度耗尽；持久文件仅确认 C5 最后一题的 16 条 `AuthenticationError` 和 12 条 `SystemExit`，未保存 HTTP 状态或余额正文。`ours_v25/interruption.json` 记录此证据边界，原 journal/report 不改写。EVO 实际处理 5/20（2 完成、3 失败），经验 v2；C5 2 完整、2 失败、1 个 started 后评分中断，余下 5 VAL 及 C10/C15/C20 未执行。C0 虽合格但仅 provisional position=0，未最终选版。TEST 33 个 `NoSelectedCheckpoint` 占位项没有真实生成尝试或答案，均值/配对差值仍为 null。525 次记录调用、3,411,927 tokens 中 28 次失败调用用量为估计，费用未知。交接包已刷新最终输出、提交答案/预算/失败证据及 SHA256。后续修复和新实验必须使用新冻结代码与新目录；不能合并旧失败作为成功运行。

用户暂时授权 DeepSeek V4 Flash 作 Judge 后，Single-Agent v29 于北京时间 2026-10-02 18:38:42 完成：33/33 TEST 完整评分，全量均值 `0.47553977079687665`，评分耗时 `2095.015` 秒。33 份封存答案均复用，GPT Judge 分数未混用；新 DeepSeek 评分为 827 次记录调用。执行与 Judge 请求模型均为 `deepseek-v4-flash-vision`，返回 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`。这是闭卷探索、自评条件，不能替代原共享证据、独立 GPT Judge 条件。完整逐题评分及答案哈希引用见 `rr_run0_deepseekjudge_20261002`。

DeepSeek v26/v27/v28/v29 工程预检分别遇到 unsupported `uniqueItems`、连接错误、连接错误和标题终稿拒绝；所有历史独立保存。v30 改用 `json_object` 后无评分预检通过，从空经验正式启动。C0/C5 均为 8/10，不满足 9/10；前 11 EVO 全部失败，经验仍 v0，错误包括 ledger 形状、JSON 截断、归因 agent ID 与进化证据引用。北京时间约 18:59 停止 v30，另写 `interruption.json`，原 journal/report 保留；没有 TEST 或有效配对差值。

新 v31 冻结于 `.runtime/rr_deepseek_judge_20261002_v31`，使用 `--structured-output json_schema_planning`：规划/归因保持 JSON Schema，执行使用 JSON Object。执行角色可自主进行一次 JSON/checkpoint/ledger 形状纠正；原始错误、模型输出和预算均保留，非法工具、peer 与未观察证据仍立即拒绝。相关回归 `142 passed`。v31 无评分预检通过后从空经验启动原 20 EVO、五次 10 VAL、选版后 33 TEST；输出目录为 `outputs/rr_jit_mas_run0_deepseekjudge_20261002_v31`。终态为 `failed/SystemExit=1`：C0 完成 10/10 且合格，但 EVO 仅处理 5/20（2 完成、3 失败），在 C5 的 Judge 评分阶段连续 5 次 APIConnectionError 触发模型客户端熔断，未完成全部 EVO，因而没有最终 checkpoint，也没有真实 TEST 生成或评分；33 个 TEST 槽位仅登记 `submission_failed`，方法均值和配对差值均为 `null`。该故障是网关连接中断，日志没有 HTTP 状态，不能据此归因于余额或 schema。单代理沿用 v29 完整 DeepSeek 评分，两臂 Judge 身份一致；这属于工程恢复后的闭卷探索，当前无有效两臂比较，不能表述为方法优于基线或一次无故障初始并行运行。

### v32/v33 工程恢复附录（动态状态）

以下状态以 `2026-10-02T14:47:00Z` 的落盘快照为准；运行中状态不作为 TEST 结论，也不产生方法优势声明。

v32 使用冻结代码完成了 C0 的 10 个 VAL 槽位（9/10 完整，`eligible=true`，selection utility `0.448814031420831`），随后处理 5 个 EVO 槽位，其中 1 个完成、4 个失败；C5 第一题失败。实验记录随后写入 `interruption.json`，终止时共 16 个 EVO/VAL 槽位已达到终态、6 个失败；没有选择最终 checkpoint，没有 TEST 答案或 TEST 均值。v32 运行与 v33 轨迹保持独立，不能合并经验或成绩。

v33 于 `2026-10-02T14:32:39Z` 使用新的冻结代码（启动提交 `edba32635e97bafef5e8879288a640167f3a0567`）启动；无评分 preflight 已通过，仍从空经验执行相同的 20 EVO、五个 10 VAL checkpoint 和 33 TEST 协议。当前快照显示 C0 已完成 4 个、失败 1 个、另有 1 个 started；v33 尚未到达最终 checkpoint 或 TEST 阶段，不能计算或报告两臂差值。v33 的恢复策略包括新的 soft policy、精确去重上下文（119836→94476）和 Writer 反枚举修复；这些是工程实现变化，需在最终结果中单独披露。
