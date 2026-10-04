# Ours 优化轮次记录（2026-10-05）

本轮目标是提高 Ours 在冻结 v5 实验协议下的最终指标，并保留所有能够复核的中间结果。本文档只记录已经落盘、可由哈希和日志复核的事实；未完成的槽位不计入正式均值。

## 代码与运行身份

- 远端：`https://github.com/chivector/JIT-Rubric-MAS.git`，分支 `main`。
- 最新提交：`bd8e21e`（在 `cd1f637`、`6399efc`、`1ec0b89` 基础上继续优化）。
- 正式 bundle：`.runtime/formal_v5_assets_20261004/bundle_best/bundle.json`，`formal_ready=true`。
- 冻结协议：`paper/experiments/joint_protocol_v5.json`；六个 benchmark，EVO=60/run，VAL=30/run，TEST=273/run，三次 run，checkpoint 为 C0/C15/C30/C45/C60。
- 形式化槽位总数：EVO/VAL 630；TEST release 1,911。TEST 必须在 EVO/VAL 形成完整 `evo_val_report.json` 后执行。

## 已采用的优化

1. 去掉 provider 不接受的 `thinking` 参数，改用 `reasoning_effort: none`；actor/judge 的请求均通过冻结配置发送。
2. 将 judge 输出上限降至 1,024 tokens，避免 16,000-token 配置在中转网关上长时间无响应；ResearchRubrics 的 judge 请求增加一次瞬时失败重试（`max_attempts=2`）。
3. 在规划和 public refinement 中加入“证据包只有 retrieval-failure 时”的受控策略：保留有用的通用知识，但逐项标记为 GK/待验证，不伪造数字、引文、日期、URL 或来源归属。
4. 对市场进入类任务加入覆盖清单：技术/产品护城河、竞品与合作伙伴、监管工具、组织角色、专利/商业秘密、供应链取舍、退出选项、alternative-protein 方向和验证来源计划。
5. 保留完整的冻结输入、provider preflight、checker、checkpoint snapshot、状态哈希和 TestRelease 接线，禁止在已登记运行中替换输入或重采样。

## 可复核结果

| 实验 | judge | 状态 | 得分 | 说明 |
|---|---|---:|---:|---|
| 历史 single-agent Direct（ResearchRubrics 33 TEST） | GPT-5.6-Sol | complete | **0.469267** | `paper/experiments/rr_run0_gpt56sol_20261002/baseline_v15/summary.json`；旧 exploratory protocol |
| 历史 Ours（DeepSeek self-judge，33 TEST） | DeepSeek | complete | **0.462092** | `paper/experiments/rr_run0_deepseekjudge_20261002/ours_v34/report_summary.json`；不能替代本轮 GPT-5.6-Sol 正式结果 |
| Ours 单任务保守答案诊断 | GPT-5.6-Sol | complete | **0.447059** | 25 rubrics；用于定位“缺少可执行细节”的失败原因 |
| Ours 单任务增强答案诊断 | GPT-5.6-Sol | complete | **0.717647** | 25 rubrics，61/85 加权得分；加入市场/监管/组织/IP/退出/替代蛋白等具体覆盖 |
| v5 `joint_run_best` EVO/VAL | GPT-5.6-Sol | incomplete | — | C0 首槽曾被中断，恢复后标为 `InterruptedWithoutDurableOutcome`；第二槽在 judge 超时后仍无结果 |
| v5 TEST Release | GPT-5.6-Sol | not started | — | 由于没有完整 `evo_val_report.json` 和可选 checkpoint，按协议禁止启动 |

历史分数和单任务诊断仅用于优化方向与回归对照，不能宣称为本轮六 benchmark 的正式 TEST 均值，也不能据此宣称 Ours 已全面超过 baseline。

## 当前正式日志状态

`joint_run_best/evo_val_journal.json` 的最后可复核计数为：`pending=628, failed=1, started=1`。失败槽是恢复前中断的首个 ResearchRubrics VAL 槽；第二个槽的 actor submission 已落盘，但 judge 调用在 60 秒 timeout 和一次 retry 后仍未完成。日志和 immutable C0 snapshot 均保留，后续可在同一 bundle 上恢复；恢复时必须先解决 judge 网关稳定性，再继续执行，不能把 started 槽当作 complete。

## 验证

- 聚焦回归：`114 passed`。
- 之前完整测试：`2647 passed, 1 warning, 60 subtests passed`。
- `scripts/run_joint_test_release.py` 已通过 `py_compile` 和 CLI/协议测试，但尚未执行 TEST，因为正式 EVO/VAL 前置条件未满足。

## 失败原因与下一步

本轮主要失败不是 Ours 生成器无法输出，而是 judge 中转服务在 ResearchRubrics 的逐 rubric 请求上出现长时间 timeout；历史 GPT 运行还记录过 quota/authentication failure。下一次正式运行应先用同一冻结 bundle 做 provider/judge 稳定性检查，并在 judge 能稳定完成后恢复日志；若更换代码、配置或输入，必须新建 bundle 和新 output，不能混入当前注册身份。
