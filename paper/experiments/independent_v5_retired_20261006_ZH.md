# 逐题独立进化 v5 退役记录（2026-10-06）

原 campaign `C:\JITv5` 采用逐题进化，每 5 题保存一个 checkpoint，最终从 C0/C5/C10/C15/C20 中选择。2026-10-06 方案修正为：40 道 EVO 分 8 批，每批 5 题使用同一冻结状态，批后生成 3 个候选，固定 VAL 选 1 个进入下一批。旧 campaign 因协议改变退役，原始记录与成本保留；不将旧状态、Pool、checkpoint 或部分评分接续或拼接到 v6。

以下是北京时间 **2026-10-06 14:12:41**（UTC `2026-10-06T06:12:41.060701+00:00`）的只读快照。通过 SQLite `mode=ro` 读取注册、库存、结果预算和 checkpoint 元数据；校验结果及原始 checkpoint JSON 的哈希，未查看答案或 VAL/TEST 分数，未修改旧数据库、槽位或原始产物。当时没有 Python 实验进程；数据库仍保留 2 个 started 槽位，不伪装为已结算。

## 冻结身份

| 项目 | 值 |
|---|---|
| Campaign | `C:\JITv5` |
| Registration schema | `independent-campaign-v5` |
| Registration SHA256 | `1676a76c39442e72beca4b2bf9c34a681ff13808a4f43fc1f3160bf6d57e9957` |
| Protocol SHA256 | `19e4181960b63e26e7d12f4cb67b1be64e5e08a540fb739104a97a4c5f50b31b` |
| Code fingerprint | `c18db930cc9aead8013ac81e7248a97462ecde68ed27e1ae60158346fd6c77b3` |

## 库存与中断

| 阶段 | complete | incomplete | failed | started | pending | 总数 |
|---|---:|---:|---:|---:|---:|---:|
| EVO | 33 | 0 | 40 | 1 | 106 | 180 |
| VAL | 66 | 2 | 4 | 1 | 377 | 450 |
| TEST | 0 | 0 | 0 | 0 | 2,751 | 2,751 |

EVO/VAL 共 **145/630** 槽位终态，99 个 complete；complete 不等同于有效进化提交。没有 Selected 记录。TEST 全部 pending，没有全局封存或 `test_report.json`，尚未释放 TEST 评分。

EVO 技术失败为：27 个 `ValueError`、4 个 `JSONDecodeError`、5 个 `ContextLimitExceeded`、2 个 `RuntimeError`、2 个 `APIConnectionError`；VAL 失败为 1 个 `RuntimeError` 和 3 个 `APIConnectionError`。失败保留原槽位和成本，不因结果重新抽题。

仍为 started 的槽位：

- `val:researchrubrics:run1:c10:684397d188c1deceb49af31d`，开始于 UTC `2026-10-06T05:58:08.029692+00:00`。
- `evo:deepsearchqa:run0:deepsearchqa:28bc43e923c9e6440f9a4bf1f33ae22aaf996bd4614a2d46cdfc5c9f82b75165`，开始于 UTC `2026-10-06T05:58:22.019763+00:00`。

这两个槽位目录没有 durable `budget.json`、`complete.json` 或 `failure.json`；其服务端实际消耗未知，不计为零，也不重新生成来覆盖原槽位。

## 已记录成本

终态槽位预算共记录 **4,987 次模型调用、64,932,713 tokens**，其中 4,932 次有 provider 用量，55 次为估算；估算 token 为 **1,904,346**。因此已记录 provider token 下界为 **63,028,367**；加上两个未封账 started 槽位后，全 campaign 的精确总 token 仍为 unknown。这里不推算货币费用。

| 阶段 | 记录 calls | Provider calls | 估算 calls | 记录 tokens | 估算 tokens |
|---|---:|---:|---:|---:|---:|
| inference | 1,142 | 1,136 | 6 | 28,204,356 | 447,758 |
| evaluation | 3,285 | 3,236 | 49 | 12,157,670 | 1,456,588 |
| update | 560 | 560 | 0 | 24,570,687 | 0 |
| **合计** | **4,987** | **4,932** | **55** | **64,932,713** | **1,904,346** |

终态预算未发现未知预算或未结算 reservation；这不消除 started 槽位的未知费用。以上成本只属于短根 v5 campaign，不包含此前长路径尝试，也不作为 v6 成本或最终性能。

## 状态保留

数据库保留 23 个 checkpoint。九条轨迹 C0 均为 version 0、0 条经验、Pool version 0。已保存的非初态如下；数值只描述旧协议进度，不是 v6 初始经验。

| 来源 | Run | 已保存位置 | 最后保存 version | 最后保存经验数 | Pool version / profiles |
|---|---:|---|---:|---:|---|
| ResearchRubrics | 0 | C5/C10/C15/C20 | 8 | 8 | 8 / 6 |
| ResearchRubrics | 1 | C5/C10/C15/C20 | 10 | 10 | 10 / 6 |
| ResearchRubrics | 2 | C5/C10/C15/C20 | 4 | 4 | 4 / 6 |
| DeepSearchQA | 0 | C5/C10 | 7 | 6 | 7 / 6 |
| DeepSearchQA | 1/2 | 仅 C0 | 0 | 0 | 0 / 0 |
| WritingBench | 0/1/2 | 仅 C0 | 0 | 0 | 0 / 0 |

旧 [运行登记](independent_v5_execution_20261006_ZH.md) 与 [机器协议](independent_protocol_v5.json) 继续解释其原始运行身份；当前分批设计见 [实验计划](experiment_plan_v5_independent_ZH.md)。
