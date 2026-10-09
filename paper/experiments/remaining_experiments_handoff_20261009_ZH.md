# 剩余实验交接与分工

更新时间：2026-10-09，上海时间。A/B/C/D 为待分配的负责人代号。本轮已授权同步已有结果；当前 RR TEST 持续跟踪至进程结束。交接文档本身不启动新的模型实验。

## 当前采用的方案

以 [详细独立进化计划](experiment_plan_v5_independent_ZH.md)、[v7 Ours-only 协议](independent_protocol_v7_ours_only.json) 和 [v7 任务成员](task_assignments_v7_ours_only.md) 为准：RR、DeepSearchQA、WritingBench、DeepResearch Bench II 各一条 40 EVO 轨迹，8 批 × 5 题；每批 3 个同输入状态候选，各评固定 10 VAL；最终 Selected 固定为第八批 winner C40。候选至少完成 9/10 VAL 才有资格。

本源 TEST 为 RR33、DSQA50、WB50、DRBII40；四个 C40 各迁移到 IFEval50 与 IFBench50，共 573 TEST。不增加 baseline、run1/run2，不给 IFEval/IFBench 做 EVO/VAL。`experiment_plan_v5_ZH.md` 仍有旧三来源、多 run 描述，不能据它启动旧协议。

## 负责人和交付

| 负责人 | 工作包 | 已有状态 | 剩余工作与验收 |
|---|---|---|---|
| A：RR / 当前协调者 | `C:\J\r17_rr_recovery_v4` | 40 EVO 槽位已终态（39 complete、1 failed），8 批完成；240 VAL 为 235 complete、5 failed；C40 candidate 1 已选中 | 等当前 133 TEST 全部终态，封存、评分、生成最终报告；保留失败和恢复差异；上传最终增量包 |
| B：WritingBench | `C:\J\r15` 中 WB 来源 | 有效 C40；150 TEST 生成终态为 142 submitted、8 failed；142 条评分全部 `AttributeError` | 诊断评分错误，建立单独的评分恢复版本，仅评原 142 份答案；保留原失败 receipts 和 8 个生成失败；交付 WB/IFEval/IFBench 三组指标 |
| C：DeepSearchQA | `C:\J\r15` 中 DSQA 来源 | 最后有效 C30；第七批三候选均 `ContextLimitExceeded`，最终 inconclusive；150 TEST missing | 用合成任务验证上下文修复，冻结新版本/恢复规则与主结果身份；获得合法 C40 后完成 50+50+50 TEST、封存和评分 |
| D：DRBII | `C:\J\r15` 中 DRBII 来源 | 最后有效 C5；第二批三候选均因 identity/source tasks/local evidence 校验失败，140 TEST missing | 验证候选更新修复并冻结新版本/恢复规则；获得合法 C40 后完成本源40（20英文+20中文）及两迁移各50，封存和评分 |

可以并行做本地诊断和合成预检。真实模型阶段先协调共享服务的吞吐；当前冻结 generator 的请求并发为 1，不能通过多进程绕过限流。每个输出目录只允许一个 coordinator，来源之间不共享 experience、Agent Pool 或 checkpoint。

## A：接管当前 RR

唯一活动输出是 `C:\J\r17_rr_recovery_v4`。2026-10-09 启动的实际 Python worker PID 为 `36612`（PID 仅供本机核对，重启后会变）。运行代码工作树为 `C:\J\r17-worktree`，启动文件是其 `.runtime/continue_rr_v4_test.py`；该本机文件含凭据，不能传给他人或原样提交。

当前脚本按 `submit(workers=4)` → `seal()` → `score(workers=4)` → paper summary 顺序执行。不要再启动同目录 TEST runner。主仓库的只读监控文件是 `.runtime/rr_test_followup_20261009/status.json` 与 `summary.md`；运行输出内旧 `status.json` 停在更早阶段，应以 campaign SQLite 与监控快照为准。

选中状态必须绑定：

```text
source = researchrubrics
run_id = 0
position = 40
candidate_index = 1
state_hash = a782b141a78516681fe6d1b0665a98bd7ea6a85e8a5e9efb073f138fc7a9bfed
registration_sha256 = 6b3013d96f761135343378382947aafcc8c0acb5b481b246edde2e63662015e4
```

RR 本源生成已完成：32 submitted、1 `ContextLimitExceeded`。迁移 TEST 还在运行，IFEval 已出现连接、超时和 JSON 错误，具体分组见 [最新库存](results_20261009/campaign_catalog.json)。生成失败仍消耗原槽位，不能按完成率或分数补跑。

恢复上下文与原配置的差异已登记在 `recovery_runtime_provenance.json`：所有 model spec 实际使用 65,536 context、2,048 margin、`oldest_turns`，launch identity 为 131,072。原 `r17_rr_v3` 第八批候选完成数为 2/10、0/10、7/10，无有效 C40，继续保留 inconclusive，不能覆盖。

验收文件：`test_continuation_submit.json`、`test_continuation_seal.json`、`test_report.json`、`test_continuation_finished.json`、`full_report.json`，以及同级 `r17_rr_recovery_v4_paper/paper_summary.{json,md}`。若进程退出且没有最终报告，先查 `test_continuation_error.json` 和 `logs/test_continuation.*.log`，记录停止点；核对 claim、已落盘 submission/complete 与评分 started 标记后，按同注册身份恢复。不能删除失败 receipt、seal 或 started 标记来强行重试。

## B：WritingBench 只恢复评分

C40 candidate 1 的 hash 为 `d50fc57f79c78bed52c0d2b0ac24db3987e16e90dc5244b77880433e346a53a2`。已提交的原答案分布为 WB49、IFEval49、IFBench44；对应 142 个 `test_evaluations/*.json` 都是 `evaluation_failed / AttributeError`。其余 8 槽是生成失败，必须保留。

先在副本上复现调用/对象接口错误并写清根因，做合成回归；新建版本化 scoring recovery 输出及清单，逐项绑定原 slot_id、submission_hash、answer_hash、state_hash、原失败 receipt 和评分修复版本。只评原答案，禁止重新生成。原 `r15` 目录保持完整。

不能直接重复 `--mode score`：现有 `_score_one()` 会读取原失败 receipt；已存在的 `test_report.json` 改变也会触发完整性错误。需要显式恢复实现和独立报告，而不是清空原目录。验收时 142 个评分恢复记录必须逐项对应原答案，8 个生成失败继续在固定分母中体现。

## C / D：新版本演化与 TEST

DSQA 原第七批失败信息为 `Required initial or current messages exceed frozen context window`。DRBII 原第二批失败信息为 `Batch agent update changed its identity, source tasks or local evidence`。先用不含正式 TEST 的合成任务验证修复，不根据 TEST 分数选择实现。

为各来源单独冻结：代码提交、manifest/split、数据与证据 SHA256、generator/judge 真实返回身份、上下文窗口及裁剪策略、transport/retry、预算与并发、工程恢复点、原/新结果的报告关系。新运行使用新输出目录。若修复会影响早期演化输入或选择，默认从 C0 重跑；仅在有明确版本化规则与可验证冻结状态时恢复受影响阶段，不能把 C30 或 C5 冒充最终 Selected。

启动模板只在负责人生成并预检自己的新 launch 后使用：

```powershell
python -m scripts.run_independent_batch_experiment --mode check --launch-config <new-launch.json> --output <new-output>
python -m scripts.run_independent_batch_experiment --mode run --launch-config <new-launch.json> --output <new-output> --phase first-stage
python -m scripts.run_independent_batch_experiment --mode run --launch-config <new-launch.json> --output <new-output> --phase full
```

`--phase first-stage` 只完成第一批，不能据它的报告声称完成 40 EVO。`full` 会按依赖继续演化、TEST、封存和评分。不能把四来源旧 launch 指向单来源输出，也不能让 TEST CLI 默认的 `IndependentEnvironment` 错配 batch 注册；恢复 batch TEST 时需传入原 `IndependentBatchEnvironment`。

DSQA 交付原生 task-level F1（附 precision/recall/fully-correct）以及两组迁移；DRBII 交付 Overall、InformationRecall、Analysis、Presentation 与 blocked-source/missing，同时核对语言分层。原来的 inconclusive、失败预算和所有恢复差异都随结果提交。

## 历史数据与统一汇总

旧 `.r/dsv12` 虽然有 RR/DSQA/WB C40、DRBII C35，但 `.runtime/dsv12_code_identity_audit.json` 记录了临时替换 registered fingerprint，28 个槽位需独立有效性审查（DSQA5、DRBII23）。不能把它们直接拼进 r15/r17 的主结果。dsv7–14、r14、r16 和原 r17 都归档为独立历史尝试，停止的旧 coordinator 不重新启动。

接收人从 [结果 Release](https://github.com/chivector/JIT-Rubric-MAS/releases/tag/experiment-results-20261009) 下载所需资产，并按 `archive_index.json` / `manifest.json` 验 SHA256。凭据通过本机环境变量配置，不放入 launch、代码或日志。归档中的 SQLite 是备份快照，不是在其他机器上直接接管活动 coordinator 的许可或步骤。

恢复 r17 的代码依赖可在另一个 checkout 中执行（不操作当前运行工作树）：

```powershell
git fetch origin main
git bundle verify <download-dir>/rr-runtime.bundle
git fetch <download-dir>/rr-runtime.bundle refs/heads/rr-r17-recovery
git checkout --detach FETCH_HEAD
```

随后核对归档中实际 split 与 launch/config/preflight。更换本机路径会影响冻结文件身份，迁移应先明确路径映射/新注册规则，不能通过篡改数据库里的哈希绕过检查。

每个负责人提交：运行目录与代码版本、注册/协议/状态哈希、原始产物及校验清单、全部成功/失败/缺失槽位、逐题评分、原生指标、token/latency/cost 来源和恢复差异。协调者最后核对四来源各自 C40 与 573 TEST 的完整矩阵；缺失保持 `null`，不得跨 benchmark 合并成总排行榜。
