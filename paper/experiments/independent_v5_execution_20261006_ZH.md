# 独立进化 v5 运行登记（2026-10-06）

本文件登记独立进化 v5 的运行身份、技术失败和成本，不是最终成绩报告。依据 [experiment_plan_v5_independent_ZH.md](experiment_plan_v5_independent_ZH.md) 与 [independent_protocol_v5.json](independent_protocol_v5.json)，ResearchRubrics、DeepSearchQA、WritingBench 分别进化，每个来源运行三个固定题序；DeepResearch Bench II、IFEval、IFBench 仅用于迁移 TEST。

当前运行根为 **`C:\JITv5`**。此前三个长路径 campaign 保留为技术诊断，不将其旧槽位、部分评分、经验状态或成本拼接进新短根 campaign。本文只读核对 SQLite 和预算元数据，未调用模型、未修改实验产物、未复制或重新归档当前 campaign，未读取 TEST 答案或评估反馈。

## 当前短根的冻结身份

| 项目 | 登记值 |
|---|---|
| Campaign 根 | `C:\JITv5` |
| 首个槽位开始时间 | `2026-10-05T18:24:39.138663+00:00` |
| Protocol SHA256 | `19e4181960b63e26e7d12f4cb67b1be64e5e08a540fb739104a97a4c5f50b31b` |
| Registration hash | `1676a76c39442e72beca4b2bf9c34a681ff13808a4f43fc1f3160bf6d57e9957` |
| Code fingerprint | `c18db930cc9aead8013ac81e7248a97462ecde68ed27e1ae60158346fd6c77b3` |
| EVO/VAL runner SHA256 | `7b3d2055b658c88857980282a59d59e2511b2710d322db59c9fbe6c0eeb42530` |
| Configuration file SHA256 | `c19fd7dece89c47cdf334b13656f8ccb55094f15fce2bbc9d141247d705a31d5` |
| Launch file SHA256 | `da37d1edfa4e72bf46bf78fdce6204bbfb895f09f8067b7d0a751dba00cf03af` |
| 初始持久状态 hash | `c52e772d2a6d6f191fc17d6004ae1ad2e61bfa514e7f2a6e70ace72d2c9f665d` |

短根使用与旧长路径 `independent_run_v5_20261006_retry3` 相同的 retry3 代码、配置和输入身份，因而 registration hash 相同；两者仍是不同的运行尝试，必须同时以 **campaign 根和冻结身份** 区分。短根不把旧长路径失败槽重新置为 pending，不接续旧尝试的已学习状态。

使用 `native_jit` / `iterative_shared_ledger`。生成、规划、执行和反思请求 `deepseek-v4-flash-vision`，预检返回 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`；judge 请求并预检返回 `gpt-5.6-sol`。逐调用服务端身份以预算回单为准。单题上限为 2,000,000 tokens / 900 秒；meta/global/local/judge 单次输出上限为 16,000，exec 为 8,192。模型与工具调用次数没有固定硬上限。凭据不写入本文。

完整库存为 **180 EVO + 450 VAL + 2,751 TEST = 3,381 槽位**。每条 `source × run` 轨迹只使用自己的 20 道 EVO，保存 C0/C5/C10/C15/C20，并在每个 checkpoint 上评本来源固定 10 道 VAL。全部候选完成终态后才按本来源 VAL 选版；没有合格 Selected 时保留不确定，不借用其他来源或 run。

## “从零”与六个 seed prototypes

**from0 指没有已学习的任务经验**，并非取消通用角色初态。九条轨迹的持久 C0 均为 store version 0、0 条学习经验、Pool version 0；未继承旧 joint 或旧独立 campaign 的经验、harness 记忆、结构操作或进化提交。

持久 C0 的 Pool 尚未初始化，profiles 为 0。构建阶段通过 `effective_pool()` 为这种初态提供相同的六个 `seed_pool()` harness prototypes：`writer`、`searcher`、`critic`、`planner`、`analyst`、`generalist`。这些原型包含预先定义的角色 prompt、skills、推理/规划/通信策略及 harness 默认策略，没有历史任务学习记录；不得称为进化获得的成熟原型。

因此须同时记录“持久初态没有学习经验”与“按初始化规则获得相同 seed prototypes”。SQLite C0 中 profiles 为 0，不意味着 Meta 在运行时没有可选原型；六个默认原型也不意味着使用了历史进化经验。

构建 MAS 时根据公开 query 预测 rubrics、选择和适配原型；缺少合适原型时可临时创建并记录完整 harness。构建与执行阶段保持长期 Pool 冻结，不做双层归因，不合并或删除长期成员。取得完整 judge 反馈后才做 RubricGraph 支持的全局与局部归因，由 Meta 汇总确定两层进化及 Pool 结构操作。

## 三次旧长路径尝试

以下名称均相对于 `.runtime/formal_v5_assets_20261004`。这些尝试不是当前短根的正式数据，不按分数高低筛选或删除。

| 旧 campaign | Registration hash | Code fingerprint |
|---|---|---|
| `independent_run_v5_20261006` | `b9c67134413733478458b0053aae19018129fd0bcd8f12f9be29f0b497bfd22a` | `3661e3a95bce36541e9a2dccb5ffbb5fe9baf105a9619a2b7bc6bf48621d5899` |
| `independent_run_v5_20261006_retry1` | `0e4f9148486ade580a756c6690683215f23b41689ede5545640388f2cd09da18` | `5b35020cd646cb3b7033fc6307ab412235b57c8b8961a7817cfdbf2bf682e3c2` |
| `independent_run_v5_20261006_retry3` | `1676a76c39442e72beca4b2bf9c34a681ff13808a4f43fc1f3160bf6d57e9957` | `c18db930cc9aead8013ac81e7248a97462ecde68ed27e1ae60158346fd6c77b3` |

### 槽位与技术失败

| 旧 campaign / 阶段 | complete | incomplete | failed | started | pending |
|---|---:|---:|---:|---:|---:|
| 初次 / EVO | 0 | 0 | 2 | 1 | 177 |
| 初次 / VAL | 2 | 0 | 0 | 0 | 448 |
| retry1 / EVO | 0 | 0 | 3 | 1 | 176 |
| retry1 / VAL | 0 | 0 | 3 | 0 | 447 |
| 长路径 retry3 / EVO | 0 | 0 | 133 | 0 | 47 |
| 长路径 retry3 / VAL | 0 | 0 | 133 | 0 | 317 |

三次旧尝试的 TEST 均为 **2,751 pending**，没有生成、封存或释放 TEST 评分。只读检查时，三次旧尝试所有九条轨迹的 store version 均为 0，没有 `evolution_commits`；技术失败没有产生可接续的已学习状态。

初次两个 EVO `ValueError` 分别为重复修改被 specialize 的同一 Pool 成员、引用不可用的过程证据。retry1 的六个失败均为 `FileNotFoundError`，保存 `planning_calls.json.tmp` 时失败。旧长路径 retry3 的 266 个失败也均为 `FileNotFoundError`，没有持久预算回单；这些是实现/产物路径诊断，不能解释成任务质量零分或 API 全面不可用。

### 已记录成本与 unknown

| 旧 campaign | Provider 实际 calls | Provider 实际 tokens | 估算 calls / tokens | 有预算槽 | 无预算或未封账槽 |
|---|---:|---:|---:|---:|---:|
| 初次 | 156 | 1,743,395 | 0 / 0 | 4 | 1 started |
| retry1 | 27 | 565,425 | 0 / 0 | 6 | 1 started |
| 长路径 retry3 | 0（无记录） | 0（无记录） | 0 / 0（无记录） | 0 | 266 failed |

前两次已知 provider 成本合计为 **183 calls / 2,308,820 tokens**，只是已记录部分，不能视为包含全部中断消耗的精确账单。长路径 retry3 的“无记录”不是零服务端费用的证据；266 个失败槽预算缺失须单列 unknown accounting，不能替代成精确零值。

前两次尚未封账的 started 槽为：

- 初次：`evo:researchrubrics:run0:6847465956a0f6376a60543e`
- retry1：`evo:researchrubrics:run0:6847465956a0f6376a6053f3`

上述两槽没有 durable `budget.json`，服务端消耗未知。保留原 slot、receipt 和 started 记录，不以重新生成覆盖，不把费用归入新短根任务预算。旧尝试的 known、estimated 和 unknown 成本在总研究开销中另表报告。

## 当前短根的工程快照

只读快照时间：`2026-10-05T18:39:20.487974+00:00`。运行仍在进行，下表不表示最终完成率或最终性能。

| 阶段 | complete | incomplete | failed | started | pending |
|---|---:|---:|---:|---:|---:|
| EVO | 1 | 0 | 2 | 1 | 176 |
| VAL | 1 | 1 | 0 | 1 | 447 |
| TEST | 0 | 0 | 0 | 0 | 2,751 |

当时没有 Selected 记录。EVO 的 `complete` 仅表示执行回调产生终结记录，不保证 judge 反馈完整或经验已更新：已完成的首个 EVO，其反馈 `complete=false`，`experience_updates=[]`，store 仍为 version 0，没有进化提交。归因/提案校验失败也不能绕过验证写入经验。

**EVO 缺少完整 judge 反馈时不更新经验或 Pool。** 题目仍消耗原 EVO 位置；不补题、不重采样、不把不完整反馈当作完整反馈。VAL 的不完整或失败原生分保持 `null`；选版用的保守替代值与官方观测分分开，不伪造零分。

### 当时已记录预算

| 阶段 | Provider 实际 calls | Provider 实际 input tokens | Provider 实际 output tokens | 估算 calls | 估算 tokens |
|---|---:|---:|---:|---:|---:|
| inference | 40 | 665,798 | 64,766 | 0 | 0 |
| evaluation | 111 | 307,196 | 40,699 | 34 | 1,024,300 |
| update | 21 | 649,060 | 46,402 | 0 | 0 |
| **合计** | **172** | **1,622,054** | **151,867** | **34** | **1,024,300** |

已记录混合预算为 **206 calls / 2,798,221 tokens**：provider 实际 1,773,921 tokens，估算 1,024,300 tokens。估算来自 ledger 中明确标记 `estimated` 的记录，不能称为 provider 精确账单。另有两个 started 槽尚无 durable budget，未完整封账的服务端成本保持 unknown，不能按零处理。

预算按阶段分开，避免将 judge 和进化费用误当作纯回答成本。预算上限不代表实际费用；没有价格时货币费用保持未知。后续以同一短根 campaign 的只读汇总更新进度，不重新归档、替换或回写原始记录。

## TEST 与论文报告边界

截至上述快照，TEST 全部 pending，尚未生成、封存或评分；本次登记没有读取 TEST 答案或反馈。Selected 由本来源全部 EVO/VAL 终态后的固定规则确定；三个迁移目标不参与进化或来源筛选。全部必跑 TEST 的产物、失败或缺失统一封存后才评分。

本文不报告最终 benchmark 指标、不宣称提升，不根据部分 VAL 得分调整正在运行的代码或预算。最终论文表须保留逐来源、逐 run、逐 checkpoint 的完整率、Selected 身份、原生分、失败/缺失和实际/估算/unknown 成本；旧技术诊断与新正式运行分别报告。

只读汇总工具为 `.runtime/summarize_independent_v5_20261006.py`。它只读打开 `campaign.sqlite`，验证结果与 checkpoint hash，输出 JSON 和 Markdown 到 campaign 外部目录，不读取 TEST evaluation 文件。本登记位于 `paper/experiments`，不参与运行 code fingerprint；未变更模型调用、prompt、配置、题目成员或评分器。

### 封存与释放后的论文统计

新增 `scripts/summarize_independent_paper.py`，仅在全局 TEST 封存且 `test_report.json` 已完整释放后读取评分回单。它校验注册库存、执行适配器、Selected checkpoint、提交产物和评分回单身份；输出原生分、逐 run/逐题结果、完整率、缺失和实际/估算/unknown token 成本。

```powershell
& .\.venv\Scripts\python.exe -m scripts.summarize_independent_paper --campaign C:\JITv5 --output-dir .runtime\independent_v5_final_paper_20261006\paired --seed 0 --iterations 10000
```

每个来源的三个 run 先在同题内平均配对差值，再以题为单位 bootstrap 和双侧 sign-flip；固定 Holm 校正族为 6/6/12/12。静态对照在来源和 run 间共享，每个物理槽位的费用只计一次。缺失保持 null；任一注册配对缺失时，完整均值、主置信区间和推断检验保持 null，完整配对的观测均值只作为描述结果。该工具不调用 API，不回写 campaign，不据结果修改或重采样任务。该脚本的成本汇总只覆盖 TEST 生成与延期评分；EVO/VAL 的 generation/judge/update 成本仍由既有独立运行汇总单独报告，不能把 TEST token 误称为全实验成本。
