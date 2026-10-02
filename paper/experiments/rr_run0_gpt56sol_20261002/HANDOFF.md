# ResearchRubrics run0 交接

Single-Agent 已完成闭卷探索实验：33/33 TEST 完整，官方均值 `0.46926705010467296`；执行模型为 `deepseek-v4-flash-vision`，Judge 网关响应别名为 `gpt-5.6-sol`。`baseline_v15` 保留封存答案和逐 rubric 评分。33 份答案全部复用；27 份同 Judge 完整评分复用，6 题补评恢复连接故障。该网关别名没有独立验证的不可变上游快照。

方法组最新 v25 于北京时间 2026-10-02 17:01 退出。原 `ours/report.json` 为 `failed/SystemExit=1`，`pilot_metadata.json` 为 `incomplete`；原 journal 的 `running` 字段原样保留，终态应以 `ours_v25/interruption.json` 为准。此前终端日志报告 Judge HTTP 401/额度耗尽；落盘 `budget.json` 和 `call_trace.json` 只独立确认 C5 最后一题评分的 16 条 `AuthenticationError` 和 12 条 `SystemExit`，没有保存 HTTP 状态或余额正文。

- C0：9/10 完整，合格；选择效用 `0.14616904918346751`。这是固定理论范围归一化效用，不能与 baseline TEST 官方均值直接比较。
- EVO：5/20 已处理，2 完成、3 失败，经验到 v2；余下 15 题未开始。
- C5：2 完整、2 失败、1 个 started 后评分中断；未封存完整 checkpoint。C10/C15/C20 未到达，未最终选版。
- TEST：33 个槽位登记为 `submission_failed/NoSelectedCheckpoint` 占位，真实生成尝试、答案和有效评分均为 0。方法 TEST 均值和配对差值保持 `null`，没有有效两臂比较。
- 开销：525 次记录调用、3,411,927 tokens；28 次失败调用的 token 为估计，金额未知。

`ours_v25` 包含最终 report、metadata、journal、scoring progress、封存 TEST 占位记录，以及实际提交答案、预算和失败证据。大型重复执行轨迹与 SQLite 经验库保留在原 `outputs/rr_jit_mas_run0_gpt56sol_20261002_v25`。本包 SHA256 manifest 覆盖导出文件；所有原评分和失败答案保留，没有替换低分或拼接旧运行。

v16、v20、v21 及各次无评分预检独立保存，不能合并为一次成功实验。`frozen_code_v25` 是 v25 的实际执行源码，工作区后续修复没有注入旧冻结运行。短/空答案源自模型原始输出，并非解析丢失。已定位的工程失败包括 reconciliation 新增 prediction 中不存在的角色、pool identity 校验、自审以及 JSON 截断。

后续工作区修复将有池 reconciliation 限于原候选身份，并保留 self-review/DAG 严格检查；终稿内容检查只基于公开任务，拒绝明显标题/完成声明冒充正文。旧分数不重写。Judge 额度恢复后应重新预检，在全新冻结版本和输出目录从空经验执行原 20 EVO、五次 10 VAL、选版后 33 TEST；保留 baseline v15。不得复用 v25 的失败轨迹作为新成功运行。

本轮允许模型一般知识，为闭卷探索协议；原计划共享公开证据条件尚未复现，因此尚不能支持正式方法优势结论。执行命令和历史记录见上级 `rr_two_arm_run0.md`。凭据仅通过本机环境变量传递，不进入交接包或 Git。
