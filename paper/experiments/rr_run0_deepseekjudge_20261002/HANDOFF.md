# ResearchRubrics run0: DeepSeek Judge 交接

Single-Agent v29 已完成 33/33 TEST 评分，均值 `0.47553977079687665`。33 份答案均复用原封存文本，原文本由 `baseline_v29/sealed_answer_references.json` 引用；本包保留全部逐题 DeepSeek 原始评分、证据、预算及封存清单。

方法 v30 已人工中断；终态以 `ours_v30/interruption.json` 为准，原 report/pilot/journal 状态保留。已登记 EVO 12/20，状态计数为 `{'failed': 11, 'started': 1}`，没有最终 checkpoint 或 TEST 答案。v30 与 v31 轨迹独立，不能合并成成功实验。

方法 v31 的当前状态为 `failed`，pilot 为 `incomplete`；此导出以已落盘终态为准。已登记 EVO 5/20，最终 checkpoint 为 `None`。C0/C5/C10/C15/C20 每个仍要求固定 10 VAL 中至少 9 个完整；TEST 保持 33 题各一次。当前有效两臂比较为 `False`，配对差值为 `None`。

v26 baseline 的六题局部评分、v26–v29 失败预检及 v30/v31 成功预检分别保留；不可拼成一次无故障运行。v26 的 uniqueItems 语法拒绝、v27/v28 的连接错误、v29 的标题终稿拒绝，以及 v30 的 json_object、v31 的 json_schema_planning 通过记录见 summary。预检不评分、不更新经验；v31 正式方法从空经验开始。

v31 的终止原因是 C5 Judge 阶段连续 APIConnectionError：客户端每次请求重试 5 次，连续失败计数达到 5 后抛出 `SystemExit(1)`。该日志没有 HTTP response/status，不能归因于余额、HTTP 401 或 schema。C5 的另外两项失败独立记录为 reconciliation 输出超过角色 token ceiling，以及响应 JSON 截断。证据路径见本地运行输出 `outputs/rr_deepseek_method_launch_20261002_v31/ours.stderr.log` 与 v31 C5 run artifacts；导出摘要保留其 SHA256。

请求执行和 Judge 模型均为 `deepseek-v4-flash-vision`，服务返回 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`。这是用户授权的闭卷探索、自评结果，尚未复现原计划的共享公开证据条件。不能据此宣称正式方法优势。

`frozen_code_v28` 和 `frozen_code_v31` 含各次实际相关源码和冻结 manifest；启动辅助脚本只从 stdin/环境读取凭据。当前 journal/report 采用紧凑摘要并记录原始文件 SHA256；完整原始输出和 SQLite 留在本机 outputs。运行期间再次执行导出脚本可原子刷新本包。`MANIFEST_SHA256.json` 覆盖本包所有文件；旧 GPT Judge 交接包未改写。
