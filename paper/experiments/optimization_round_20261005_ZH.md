# Ours 优化轮次记录（2026-10-05）

本轮目标是提高 Ours 在冻结 v5 实验协议下的最终指标，并保留所有能够复核的中间结果。本文档只记录已经落盘、可由哈希和日志复核的事实；未完成的槽位不计入正式均值。

## 代码与运行身份

- 远端：`https://github.com/chivector/JIT-Rubric-MAS.git`，分支 `main`。
- 历史 p8 代码提交：`92159fb`（在 `68adf5e` 的不可变 evidence pack 缓存、`ece1d16` 的 prevalidated renderer 和 `9866b34` 的 rubric baseline 修复基础上继续优化）；随后登记了新的 prevalidated evidence bundle。
- 历史 p8 候选 bundle：`.runtime/formal_v5_assets_20261004/bundle_p8_val3_v1/bundle.json`，`formal_ready=true`；该 bundle 使用重新抓取并完成结构预验证的公开 evidence、actor timeout 60 s、judge timeout 120 s、judge 并发上限 8。旧 `joint_run_*` 输出不能与新 bundle 混用；正式顺序 VAL 使用 `--val-workers 1`。
- 冻结协议：`paper/experiments/joint_protocol_v5.json`；六个 benchmark，EVO=60/run，VAL=30/run，TEST=273/run，三次 run，checkpoint 为 C0/C15/C30/C45/C60。
- 形式化槽位总数：EVO/VAL 630；TEST release 1,911。TEST 必须在 EVO/VAL 形成完整 `evo_val_report.json` 后执行。

## 已采用的优化

1. 去掉 provider 不接受的 `thinking` 参数，改用 `reasoning_effort: none`；actor/judge 的请求均通过冻结配置发送。
2. 将 judge 输出上限降至 1,024 tokens，避免 16,000-token 配置在中转网关上长时间无响应；ResearchRubrics 的 judge 请求增加一次瞬时失败重试（`max_attempts=2`）。
3. 在规划和 public refinement 中加入“证据包只有 retrieval-failure 时”的受控策略：保留有用的通用知识，但逐项标记为 GK/待验证，不伪造数字、引文、日期、URL 或来源归属。
4. 对市场进入类任务加入覆盖清单：技术/产品护城河、竞品与合作伙伴、监管工具、组织角色、专利/商业秘密、供应链取舍、退出选项、alternative-protein 方向和验证来源计划。
5. 保留完整的冻结输入、provider preflight、checker、checkpoint snapshot、状态哈希和 TestRelease 接线，禁止在已登记运行中替换输入或重采样。
6. 将 judge 传输层改为显式 `httpx.Client(trust_env=...)`，避免 Windows 系统代理被隐式注入；在同一进程内把 judge rubric 并发限制为 2，并为 ResearchRubrics 保留一次瞬时失败重试。
7. 强化 EVO/VAL resume：VAL 只读取登记时的不可变 checkpoint，live state 与 durable prefix 不一致时 fail closed；无 eligible checkpoint 时记录 inconclusive 而不是伪造选择。
8. 针对正式运行中观察到的 pooled candidate identity/version 漂移，在结构化输出的唯一 correction turn 中加入 catalogue 原样复制和 immutable identity 约束；该修复已通过 agent-pool 回归测试。
9. 新增 deterministic public evidence fallback：从公开 Bing RSS 发现页面、用 Jina Reader 固定抓取、写入现有不可变 pack 格式，不调用额外生成模型。新 evidence 覆盖 RR 63/63、DSQA 80/80、DRBII 40/40 冻结任务；成功 source 分别为 146、99、8 个，任务级有 source 的数量分别为 54、44、8。
10. 将已由 canonical renderer 生成的 pack 在冻结 manifest 中登记为 `prevalidated-structure-v1`：运行时仍检查 manifest/file/pack/task/source hash、JSON body、每个 source slice、span 不重叠、truncation 标志和 tokenizer token count，但不重复执行二次方窗口排序。RR 63 pack 冷加载约 2.4 s（此前单 pack 约 73.9 s），warm cache 约 0.23 s；证据内容和 pack 哈希保持冻结。
11. 增加可选只读 VAL 并行（`--val-workers`），journal identity 绑定 worker 数且 EVO 仍严格串行；真实 p3 探测在中转网关并发排队后产生 2 个 timeout、无 complete 槽，因此正式配置退回顺序 VAL，避免以吞吐换取失败率。

## 可复核结果

| 实验 | judge | 状态 | 得分 | 说明 |
|---|---|---:|---:|---|
| 历史 single-agent Direct（ResearchRubrics 33 TEST） | GPT-5.6-Sol | complete | **0.469267** | `paper/experiments/rr_run0_gpt56sol_20261002/baseline_v15/summary.json`；旧 exploratory protocol |
| 历史 Ours（DeepSeek self-judge，33 TEST） | DeepSeek | complete | **0.462092** | `paper/experiments/rr_run0_deepseekjudge_20261002/ours_v34/report_summary.json`；不能替代本轮 GPT-5.6-Sol 正式结果 |
| Ours 单任务保守答案诊断 | GPT-5.6-Sol | complete | **0.447059** | 25 rubrics；用于定位“缺少可执行细节”的失败原因 |
| Ours 单任务增强答案诊断 | GPT-5.6-Sol | complete | **0.717647** | 25 rubrics，61/85 加权得分；加入市场/监管/组织/IP/退出/替代蛋白等具体覆盖 |
| v5 `joint_run_prevalidated_v1` 已完成 VAL C0 RR 槽 | GPT-5.6-Sol | partial | **0.552941 / 0.845070** | 新 evidence、p2 judge 并发；两个槽完整落盘，仍不足以形成 VAL checkpoint |
| v5 `joint_run_p8_probe` 首个 VAL C0 RR 槽 | GPT-5.6-Sol | complete | **0.494118** | p8 judge 并发吞吐探测；仅用于选择传输配置，不与正式均值混合 |
| v5 `joint_run_best` EVO/VAL | GPT-5.6-Sol | incomplete | — | C0 首槽曾被中断，恢复后标为 `InterruptedWithoutDurableOutcome`；第二槽在 judge 超时后仍无结果 |
| v5 `joint_run_stable` 已完成槽 | GPT-5.6-Sol | partial | **0.517647 / 0.873239 / 0.188235** | 3 个 ResearchRubrics VAL 槽完整落盘；另有 2 个 failed、1 个 started、624 个 pending，未形成 checkpoint 选择或正式均值 |
| v5 TEST Release | GPT-5.6-Sol | not started | — | 由于没有完整 `evo_val_report.json` 和可选 checkpoint，按协议禁止启动 |

历史分数和单任务诊断仅用于优化方向与回归对照，不能宣称为本轮六 benchmark 的正式 TEST 均值，也不能据此宣称 Ours 已全面超过 baseline。

## 当前正式日志状态

`joint_run_best/evo_val_journal.json` 的最后可复核计数为：`pending=628, failed=1, started=1`。更新后的 `joint_run_stable/evo_val_journal.json` 计数为：`complete=3, failed=2, pending=624, started=1`；三个完整槽的原始 score 依次为 `0.5176470588`、`0.8732394366`、`0.1882352941`。其中第三槽之后出现 pooled candidate 缺少精确 persistent identity/version 的契约错误，第四个正在规划的槽因 actor 网关连接错误中断。上述槽位只作为 partial formal audit，不构成 benchmark 均值、checkpoint 选择或 TEST 结果。
新 evidence bundle 的 `joint_run_evidence_v1` 和 `joint_run_evidence_direct_v1` 均在第一个 actor 槽形成 durable journal 之前因 provider 请求长时间无响应而停止；它们没有新增 complete 槽，也没有改变上述正式计数。稳定传输配置仍是当前可复核的候选配置。
`joint_run_prevalidated_v1` 最终落盘 2 个 complete 槽（RR C0，原始 `0.5529411765`、`0.8450704225`），以及 1 个未完成 started 槽，journal 为 `complete=2, started=1, pending=627`；`joint_run_p8_probe` 落盘 1 个 complete（`0.4941176471`）和 1 个 started；`joint_run_formal_p8_v1` 落盘 1 个 complete（`0.4470588235`）和 1 个 started。新代码身份的 `joint_run_val3_v1` 在并行探测中留下 2 failed、3 started、625 pending；`joint_run_final_candidate_v1` 顺序探测在首槽长等待后停止并留下 1 started、629 pending。所有这些输出均没有 checkpoint 选择或正式均值。

## 验证

- 聚焦回归：原有 `358 passed`；本轮相关 evidence pack、joint executor、bundle 和 fixed-team 测试合并运行 `63 passed`（其中包含 `29 passed` evidence、`22 passed` executor/journal 并发与冻结测试、`14 passed` fixed-team 定向结果的重叠子集），新增 fast validator 没有改变原 pack 内容。
- 之前完整测试：`2647 passed, 1 warning, 60 subtests passed`。
- `scripts/run_joint_test_release.py` 已通过 `py_compile` 和 CLI/协议测试，但尚未执行 TEST，因为正式 EVO/VAL 前置条件未满足。

## 失败原因与下一步

本轮主要失败不是 Ours 生成器无法输出，而是 actor/judge 中转服务出现长时间连接等待；历史 GPT 运行还记录过 quota/authentication failure。历史 formal p8 输出已停止，保留 journal 的 durable 状态；若 provider 再次无法收敛，只保留 consumed failure，不把 partial 槽位写成正式均值，也不以 incomplete EVO/VAL 启动 TEST。

## 2026-10-05 软 budget-aware 收尾更新

用户明确要求不把简单/中等/复杂任务映射成固定角色上限。最新实现删除了这套难度分档、角色数/调用数建议和对应 Prompt；正式 p12 配置中的 `adaptive_budget_enforcement: false` 仅作为旧配置兼容字段。运行保留 configured `max_agents`、共享 ledger 的 token/调用预算、未来阶段预留和真实余额检查，规划器根据任务要求、独立检查的预期收益、上下文不确定性与实时剩余预算决定角色数和协作深度；预算耗尽时 fail closed。

p12 冻结 bundle：`.runtime/formal_v5_assets_20261004/bundle_p12_soft_budget/bundle.json`，registration/protocol 哈希已写入 bundle，provider preflight 为非 synthetic。代码、配置和 evidence 在登记后保持不变。p11 的硬上限运行已停止并保留为失败审计；p12 是当前正式候选。

### 正式运行调度审计

- `joint_run_p12_soft_budget`：顺序 VAL 探测落盘 `complete=5, failed=1, started=1, pending=623`，累计 token `1,746,442`（含超时请求估算 `60,021`）；其中一个失败是中转请求超时，未被计入成绩均值。该输出未继续使用，因为单路吞吐过低。
- `joint_run_p12_parallel`：尝试 8 路只读 VAL；中转服务出现请求容量排队超时，已停止并保留失败诊断，未与正式结果混合。
- `joint_run_p12_parallel2`：使用同一冻结 bundle、2 路只读 VAL，不改变模型或方法；已由操作方中止，最终计数与成本见下文；槽位结果和 token usage 保留在 `evo_val_journal.json`。并行仅改变执行调度，不能改变 checkpoint、候选、评分或答案。

### Token accounting

所有 EVO/VAL/TEST 槽位现在写入 `token_usage`，区分 provider usage、estimated usage、unknown usage，并按 inference/evaluation/execution stage 汇总；失败请求不会复用前一请求的计数。TEST 槽位的 `complete.json`、submit/score journal 和最终 report 均保留相同字段。专项 token/budget/OpenAI 回归为 29 passed，联合 executor/bundle/adaptive/pooled/baseline 回归为 65 passed。

p12 顺序运行的前几个完整槽已观察到真实 provider token；例如首槽 34 次模型调用、总计 248,617 token，其中 Ours inference 171,373 token、judge evaluation 77,244 token。这个数值用于最终 performance-token trade-off 表，不会把估算值冒充 provider usage。

在 p12 EVO/VAL 形成完整 `evo_val_report.json` 之前，不启动 TEST；完成后严格执行 register → submit → seal → score，并保留所有失败槽位、原始答案、judge rubric、token usage、哈希和 provider identity。


## 角色数规则纠偏（当前实现）

本轮删除了从题目长度、关键词和 simple/moderate/complex 标签推导角色数/调用数的函数、规划 payload 和 Prompt 规则。角色数与协作深度由规划器依据任务需求、有用的独立检查、边际质量收益和实时预算决定。保留原有 configured `max_agents`、`TeamBudgetPlan`、未来阶段预留、每阶段预算刷新及运行时 ledger 校验。旧配置 `adaptive_budget_enforcement: false` 继续可读；`true` 明确拒绝，防止重新启用已移除策略。

拟定的 p13（固定三次团队调用、关闭 refinement、缩小总预算）只做过配置登记，没有执行；已通过 `.runtime/formal_v5_assets_20261004/p13_abandoned.json` 标记撤销。纠偏后的下一轮沿用 p12 的 `team_max_calls=null`、`max_model_calls=null`、`max_total_tokens=2,000,000` 和原有 public refinement 配置，不以另一种固定调用数代替角色数硬限制。新代码必须重新登记运行，不能拼接到旧 journal。

### p12 最终中断审计

| 输出目录后缀 | complete | failed | started | pending | 已落盘总 token | 其中估算 token |
|---|---:|---:|---:|---:|---:|---:|
| `joint_run_p12_soft_budget` | 5 | 1 | 1 | 623 | 1,746,442 | 60,021 |
| `joint_run_p12_parallel` | 0 | 3 | 8 | 619 | 432,512 | 346,123 |
| `joint_run_p12_parallel2` | 22 | 2 | 2 | 604 | 5,621,177 | 107,494 |

各目录新增 `operator_interruption.json`；保留原 journal、答案、失败记录和 token usage。以上成本是终态槽位已落盘值，尚未覆盖中断中的 started 槽位，不应写成整个 campaign 的精确账单。parallel2 的 inference 为 4,595,606 provider tokens；evaluation 为 1,025,571 tokens（含估算 107,494）。报告分别列 actor、judge、合计，估算量独立列示。

parallel2 的两个终态失败均为 DeepSearchQA 规划输出的 coverage/primary/agent.rubric_ids 不一致，并非 provider 超时；不通过静默重分配责任来掩盖该错误。ResearchRubrics 公寓题的 evidence 包存在明显跑题网页，需要进一步修复检索相关性；哈希验证通过并不证明内容相关。中断和重启同样属于本轮开发成本，全部保留，不挑选高分槽位拼成正式结果。

历史 pilot 与当前 p12 的题目和 evidence 条件不同，不能通过直接比较均值宣称方法优劣或成本比。当前没有完整 checkpoint 选择、TEST release 或六 benchmark 正式结果，也没有证据证明 Ours 的 token 已介于 Direct 和 Native JIT 之间。

本次纠偏回归：182 passed，覆盖短题允许三角色、长题允许单角色、协作执行、真实预算不足拒绝、预算估算不作为硬迭代上限，以及 token accounting。TEST submit 并行器 `cf4656b` 另有 20 passed；评分阶段保持串行。

## 2026-10-05 新证据与预算配置审计

旧的 `evidence_bing_*_merged_p11` 包虽然结构哈希完整，但 RR 公寓题曾返回英语语法页面，不能作为正式输入。提交 `b57cedf`/`a32af39`/`bc0a554` 增加住房领域查询、强相关词门控、过滤结果审计、地点特定公共来源和年份弱词过滤。新包为：

- `evidence_bing_rr_v5_relevance`：63/63 选中任务；
- `evidence_bing_dsqa_v5_relevance`：80/80 选中任务；
- `evidence_bing_drbii_v5_relevance`：40/40 选中任务。

三包均绑定到 `bundle_p14_relevance`/后续候选 bundle，`formal_ready=true`，并保留每个 pack、manifest、source hash、filtered_results 和 retrieval-failure 记录。公寓题抽查未再出现 `many` 语法页；对 RSS/Jina 无法获得相关页面的任务，包中明确记录空检索，不补造事实。

为检验性能—token 权衡，登记了以下新配置和独立输出；它们不能互相拼接，也不能冒充完整 v5 结果：

| 配置/输出 | 关键设置 | 已完成 VAL 槽 | 已知分数 | 已落盘 token | 状态 |
|---|---|---:|---|---:|---|
| `p14_relevance` | iterative shared ledger，完整 refinement | 2 | 0.3765、0.8028 | 293,163、266,494 | operator 中止，成本/质量诊断 |
| `p15_single_pass` | single pass，完整 refinement | 3 | 0.5412、0.6338、另 1 个继续恢复 | 362,443、201,532、其余见 journal | 当前正式候选，按 journal 继续 |
| `p17_patch_refine` | single pass，patch refinement | 2 complete、1 failed | 0.7606、0.6000；1 个 JSONDecodeError | 322,864、298,354、失败 111,484 | 撤销，未进入选版 |

`p17` 的失败来自 refinement JSON 解码契约，保留为失败实验；这说明 patch 模式在当前中转服务上不能直接替代完整 refinement。p15 的每槽 token 由 `token_usage.by_stage` 分列 inference/evaluation，失败槽也计入估算或 provider usage，不能从正式均值中静默删除。完整 `evo_val_report.json` 和 TEST release 仍未形成，当前任何分数都不是论文最终均值。
