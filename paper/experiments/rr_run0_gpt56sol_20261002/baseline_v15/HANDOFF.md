# ResearchRubrics Single-Agent run0：已完成

2026-10-02，闭卷探索实验。固定 TEST 33 题，每题一份 Single-Agent 答案，执行模型为 `deepseek-v4-flash-vision`，judge 为 `gpt-5.6-sol`。全量 **33/33 完整评分，均值 0.46926705010467296**。

## 恢复过程

- 全部 33 份答案复用自 `outputs/rr_single_agent_run0_worldknowledge_20261002_v7`，答案文本与 `answer_hash` 均逐题一致，没有重新生成或按评分挑选答案。
- 27 份已完整的 GPT-5.6-Sol 评价复用自 `outputs/rr_single_agent_run0_gpt56sol_20261002_v8`；逐份核对完整状态、评价内容、分数、答案身份一致。HTTP 连接故障之后补完其余 6 题。
- 原始 HTTP 失败记录保留。评价预算中 8 条没有 `response_model` 的历史失败尝试均为 `APIConnectionError`；它们不构成有效判分。当前 33 题最终评价均完整，不含 null 或不完整分数。

## 独立核验

- 原始数据 SHA256：`ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`，与冻结元数据一致。
- 826 个 rubric 的权重序列与原始数据逐题一致；32 题包含负权重。每题按官方 `sum(weight * rubric_score) / sum(positive weights)` 重新计算，与所有保存分数一致，再对固定 33 题取算术均值。
- 33 份 sealed submission 与 v7 的答案文本及哈希全部一致，两边 seal 的记录哈希均通过核验。整个 submission 记录哈希因复用路径和来源字段改变，与 v7 不同；答案未变。
- 33 题 evaluator 身份一致：`researchrubrics:2dc80e2d4c38ddd80439517c259d93c6954b193f:d74ab5448f3a49fcbe74ab48f6e42f11bd12442aad3687684a70f7abfbf82140`。
- 828 条具有返回身份的 judge 调用记录全部为 `gpt-5.6-sol`。这是网关报告的别名，尚无独立核验的不可变模型快照；不可将别名当作供应商版本证明。

## 使用范围

这是允许模型一般知识的闭卷探索结果，不是原计划完整共享检索证据条件下的正式复现。JIT-MAS 仍没有完整冻结 TEST 成绩，因此目前没有有效两臂对比，不能据此声称方法优于 baseline。后续比较须保持同一 TEST membership、judge 身份与评分协议，并单独保留失败运行，不将失败或不完整结果视作有效成绩。

`comparison.json` 包含逐题分数；`pilot_metadata.json` 保存配置和冻结身份；`test_release/submissions`、`test_release/seal.json`、`test_release/evaluations` 保存答案、封存记录和完整评分；`summary.json` 为机器可读摘要。合作者可直接复用这批 sealed 答案及完整评分，继续完成 JIT-MAS 的进化、checkpoint 选择和冻结 TEST。
