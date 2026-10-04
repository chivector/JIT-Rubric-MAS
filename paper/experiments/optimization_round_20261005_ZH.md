# Ours 优化轮次记录（2026-10-05）

本轮目标是提高 Ours 在冻结 v5 实验协议下的最终指标，并保留所有能够复核的中间结果。本文档只记录已经落盘、可由哈希和日志复核的事实；未完成的槽位不计入正式均值。

## 代码与运行身份

- 远端：`https://github.com/chivector/JIT-Rubric-MAS.git`，分支 `main`。
- 最新代码提交：`964507b`（在 `a7d772d`、`fef5450`、`bd8e21e`、`cd1f637`、`6399efc`、`3df6d2c`、`9405db9` 基础上继续优化）；随后登记了新证据 bundle。
- 最新正式候选 bundle：`.runtime/formal_v5_assets_20261004/bundle_evidence_v1/bundle.json`，`formal_ready=true`；该 bundle 使用重新抓取的公开 evidence 和稳定传输配置（judge 并发上限 2、judge timeout 120 s）。`joint_run_stable` 是旧 evidence bundle 的 partial audit，不能与新 bundle 混用；正式续跑必须使用新 output。
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

## 可复核结果

| 实验 | judge | 状态 | 得分 | 说明 |
|---|---|---:|---:|---|
| 历史 single-agent Direct（ResearchRubrics 33 TEST） | GPT-5.6-Sol | complete | **0.469267** | `paper/experiments/rr_run0_gpt56sol_20261002/baseline_v15/summary.json`；旧 exploratory protocol |
| 历史 Ours（DeepSeek self-judge，33 TEST） | DeepSeek | complete | **0.462092** | `paper/experiments/rr_run0_deepseekjudge_20261002/ours_v34/report_summary.json`；不能替代本轮 GPT-5.6-Sol 正式结果 |
| Ours 单任务保守答案诊断 | GPT-5.6-Sol | complete | **0.447059** | 25 rubrics；用于定位“缺少可执行细节”的失败原因 |
| Ours 单任务增强答案诊断 | GPT-5.6-Sol | complete | **0.717647** | 25 rubrics，61/85 加权得分；加入市场/监管/组织/IP/退出/替代蛋白等具体覆盖 |
| v5 `joint_run_best` EVO/VAL | GPT-5.6-Sol | incomplete | — | C0 首槽曾被中断，恢复后标为 `InterruptedWithoutDurableOutcome`；第二槽在 judge 超时后仍无结果 |
| v5 `joint_run_stable` 已完成槽 | GPT-5.6-Sol | partial | **0.517647 / 0.873239 / 0.188235** | 3 个 ResearchRubrics VAL 槽完整落盘；另有 2 个 failed、1 个 started、624 个 pending，未形成 checkpoint 选择或正式均值 |
| v5 TEST Release | GPT-5.6-Sol | not started | — | 由于没有完整 `evo_val_report.json` 和可选 checkpoint，按协议禁止启动 |

历史分数和单任务诊断仅用于优化方向与回归对照，不能宣称为本轮六 benchmark 的正式 TEST 均值，也不能据此宣称 Ours 已全面超过 baseline。

## 当前正式日志状态

`joint_run_best/evo_val_journal.json` 的最后可复核计数为：`pending=628, failed=1, started=1`。更新后的 `joint_run_stable/evo_val_journal.json` 计数为：`complete=3, failed=2, pending=624, started=1`；三个完整槽的原始 score 依次为 `0.5176470588`、`0.8732394366`、`0.1882352941`。其中第三槽之后出现 pooled candidate 缺少精确 persistent identity/version 的契约错误，第四个正在规划的槽因 actor 网关连接错误中断。上述槽位只作为 partial formal audit，不构成 benchmark 均值、checkpoint 选择或 TEST 结果。
新 evidence bundle 的 `joint_run_evidence_v1` 和 `joint_run_evidence_direct_v1` 均在第一个 actor 槽形成 durable journal 之前因 provider 请求长时间无响应而停止；它们没有新增 complete 槽，也没有改变上述正式计数。稳定传输配置仍是当前可复核的候选配置。

## 验证

- 聚焦回归：`358 passed`（含 transport、ResearchRubrics retry、checkpoint freeze/resume、inconclusive 选择、planning/public-refinement 与 agent-pool identity 修复测试）。
- 之前完整测试：`2647 passed, 1 warning, 60 subtests passed`。
- `scripts/run_joint_test_release.py` 已通过 `py_compile` 和 CLI/协议测试，但尚未执行 TEST，因为正式 EVO/VAL 前置条件未满足。

## 失败原因与下一步

本轮主要失败不是 Ours 生成器无法输出，而是 judge 中转服务在 ResearchRubrics 的逐 rubric 请求上出现长时间 timeout；历史 GPT 运行还记录过 quota/authentication failure。下一次正式运行应先用同一冻结 bundle 做 provider/judge 稳定性检查，并在 judge 能稳定完成后恢复日志；若更换代码、配置或输入，必须新建 bundle 和新 output，不能混入当前注册身份。
