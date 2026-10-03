# ResearchRubrics run0: DeepSeek Judge 交接

## 最新完成结果与继续入口（2026-10-03，北京时间）

v34 已完成 33/33 TEST，严格封存比较通过：方法均值 `0.4620917542763036`，Single-Agent v29 为 `0.47553977079687665`，配对差 `-0.013448016520573093`，18 胜、15 负。完整封存提交与逐 rubric 评分见 `ours_v34/test_release/`，摘要见 `ours_v34/report_summary.json`，比较见 `COMPARISON_v34_v29.json`。v34 选择 C20，空经验启动，20 EVO 中 18 项完整评分。TEST 没有失败或截断；唯一 `length` 出现在 C20 VAL，后续纠正完成。

v35 修复终端 Critic 仍提交 review-only 内容的职责冲突，同时纳入通用交付物与事实核验提示、reviewer 拓扑纠正。修复提交 `7f01bcf`，130 项相关回归通过；公开 VAL 无评分预检已通过（0 Judge 请求、无经验更新），正式运行已启动。它仍从空经验执行 20 EVO、五次固定 10 VAL、选版后 33 TEST，v29 baseline 保持封存。查看本机 `outputs/rr_deepseek_method_launch_20261002_v35/launch.json` 与 `outputs/rr_v35_monitor_status.json`；终态后用 `scripts.compare_rr_sealed_runs` 对 v29/v35 运行严格审计。

最新提质记录见 [QUALITY_FOLLOWUP_20261003.md](QUALITY_FOLLOWUP_20261003.md)：工作区增加诚实 checkpoint 纠正、具体拓扑字段诊断、净收益与敏感性重算、解决实质审阅缺陷、案例范围和引用检查，冻结 v35 没有被修改。公开 VAL 单题两份已封存回答的诊断分数为 `0.4298245614 → 0.5175438596`（`+0.0877192982`），不是完整 TEST 增益。`2026-10-03T06:45:19Z` 的 v35 快照为 17/20 EVO 达终态，C15 10/10 完整、utility `0.5366494910`；尚无 v35 TEST。新目录 `quality_followup_20261003/` 保留失败与通过预检、两份评分、源码 manifest 和带原始文件 hash 的进度摘要。

这是闭卷、DeepSeek V4 Flash 自评且输出预算不同（方法 12,288 vs baseline 8,192）的探索实验，尚无正式方案优势。v35 是观察 v34 TEST 后的工程修复尝试，应同时呈现 v34 和 v35，不能当作新的未接触 TEST 确认性结果。凭据仍只注入进程环境。下文保留 v30/v31 的历史交接，不代表最新进度。

## 最新状态更新（2026-10-03T07:38Z 进度快照）

固定 10 VAL 的 C15 质量诊断最终仅有 8/10 题达到可用完整状态，结果为 failed/inconclusive，不能给出有效整体提升结论。原 C15 selection utility 为 `0.5366494909781279`；诊断过程中出现的 `0.42758920` 仅是把缺失题按零计入的无效统计，不得与原 utility 或完整 VAL 均值比较，也不得冒充整体改善。

v35 已选择 C15 snapshot 13（选中经验版本 13，轨迹最终经验版本为 18）。TEST 当前为 12 题已提交但未评分、4 题失败，尚无 TEST 分数或有效两臂差值。冻结源码 manifest 未改变。最新 EVO20/C20 Writer 的 length 截断经过纠正后完成提交，但恢复成功不证明数学内容语义无损，数学质量仍需独立检查。

工作区正在增强 planning `assignment_audit`，把实际 owner/reviewer/dependency 冲突同时纳入诊断和 DAG 覆盖；截至本更新尚无真实新模型验证。该更新只记录状态，不改写历史封存结果、评分身份、预算或冻结运行。

Single-Agent v29 已完成 33/33 TEST 评分，均值 `0.47553977079687665`。33 份答案均复用原封存文本，原文本由 `baseline_v29/sealed_answer_references.json` 引用；本包保留全部逐题 DeepSeek 原始评分、证据、预算及封存清单。

方法 v30 已人工中断；终态以 `ours_v30/interruption.json` 为准，原 report/pilot/journal 状态保留。已登记 EVO 12/20，状态计数为 `{'failed': 11, 'started': 1}`，没有最终 checkpoint 或 TEST 答案。v30 与 v31 轨迹独立，不能合并成成功实验。

v30 的前 12 个 EVO 槽位中 11 个失败、1 个 started 后中断；C0/C5/C10 均为 8/10，未达到 9/10 门槛。

方法 v31 的当前状态为 `failed`，pilot 为 `incomplete`；此导出以已落盘终态为准。已登记 EVO 5/20，最终 checkpoint 为 `None`。C0/C5/C10/C15/C20 每个仍要求固定 10 VAL 中至少 9 个完整；TEST 保持 33 题各一次。当前有效两臂比较为 `False`，配对差值为 `None`。

v31 实际处理 5 个 EVO（2 完成、3 失败），C0 为 10/10；C5 有 6 个完成、2 个生成/校验失败、1 个评分中断、1 个未开始。失败证据见 `diagnostics_v31/`：启动 stderr 记录连续 5 次连接错误后客户端 `SystemExit=1`，并保留 C5 三个失败槽位的 `failure.json`/`call_trace.json`。TEST 仅登记 33 个 `submission_failed` 占位项，没有真实生成或评分尝试。

v26 baseline 的六题局部评分、v26–v29 失败预检及 v30/v31 成功预检分别保留；不可拼成一次无故障运行。v26 的 uniqueItems 语法拒绝、v27/v28 的连接错误、v29 的标题终稿拒绝，以及 v30 的 json_object、v31 的 json_schema_planning 通过记录见 summary。预检不评分、不更新经验；v31 正式方法从空经验开始。

v31 的终止原因是 C5 Judge 阶段连续 APIConnectionError：客户端每次请求重试 5 次，连续失败计数达到 5 后抛出 `SystemExit(1)`。该日志没有 HTTP response/status，不能归因于余额、HTTP 401 或 schema。C5 的另外两项失败独立记录为 reconciliation 输出超过角色 token ceiling，以及响应 JSON 截断。证据路径见本地运行输出 `outputs/rr_deepseek_method_launch_20261002_v31/ours.stderr.log` 与 v31 C5 run artifacts；导出摘要保留其 SHA256。

请求执行和 Judge 模型均为 `deepseek-v4-flash-vision`，服务返回 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`。这是用户授权的闭卷探索、自评结果，尚未复现原计划的共享公开证据条件。不能据此宣称正式方法优势。

`frozen_code_v28` 和 `frozen_code_v31` 含各次实际相关源码和冻结 manifest；启动辅助脚本只从 stdin/环境读取凭据。当前 journal/report 采用紧凑摘要并记录原始文件 SHA256；完整原始输出和 SQLite 留在本机 outputs。运行期间再次执行导出脚本可原子刷新本包。`MANIFEST_SHA256.json` 覆盖本包所有文件；旧 GPT Judge 交接包未改写。
