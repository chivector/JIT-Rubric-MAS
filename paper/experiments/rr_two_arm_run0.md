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

当前导出交接包为 `paper/experiments/rr_run0_gpt56sol_20261002`，包含 v15 完整评分、v16 终态失败记录及工程预检/中断摘要；v21 运行信息属于导出时的快照，须以原输出目录的终态为准。所有旧轨迹独立保留，不合并为一次成功实验。
