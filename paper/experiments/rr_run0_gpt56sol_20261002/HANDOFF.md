# ResearchRubrics run0 交接

当前可交付的 Single-Agent 为闭卷探索：33/33 完整，均值 `0.46926705010467296`，执行模型 `deepseek-v4-flash-vision`，judge 网关报告 `gpt-5.6-sol`。这里保存 sealed 答案、完整逐 rubric 评分及 SHA256 清单；复用了 33 份答案和 27 份完整评分，补评 6 题以恢复连接故障。

JIT-MAS v16 终态为 `inconclusive`，20 个 EVO 槽位处理完毕但只有一次经验更新；全部 checkpoint 不满足 9/10 完整阈值，没有 TEST 答案或均值。`ours_v16` 保存原始报告和 checkpoint journal，不能把这些失败记录与后续运行合并成一次成功实验。

新实现修复闭卷职责冲突、归因 ID 引用及 proposal ID 重复，并要求最终 synthesizer 提交完整任务产物。规划保留动态图组队和 Agent Pool 进化，judge 的原始 rubric 与有符号权重聚合不变。输出 schema 为角色文本预留 JSON 闭合空间，保留多行答案。v17/v18 不评分执行预检失败摘要随包提供；v19 同一公开 VAL 输入预检通过、judge 请求 0，预检不计入正式成绩。v20 在 C0 中断，未进入 EVO/TEST；v21 在新冻结代码目录从空经验执行完整 20 EVO、五次 10 VAL、选版后 33 TEST。v21 导出元数据只是当前运行快照。

v21 的冻结 runner 单独保存在 `frozen_code_v21`；工作区之后补充的“终态 answer 必须携带 checkpoint”约束尚未注入 v21，不能把新工作区代码身份误作 v21 执行身份。

正式共享公开证据条件尚未复现。当前结果没有有效两臂比较，差值保持 null；不得据此声称方法优势。模型别名未由独立供应商快照验证。

v21 已于北京时间 2026-10-02 15:34 人工停止：7 个 EVO 失败、1 个未落盘执行的中断题、12 个未开始；经验 v0，C0/C5 均 7/10，不合格，没有 TEST。`ours_v21/interruption.json` 是权威中断说明；原 journal/report 的 running/evolving 保留，不能据此误认为仍在运行。已落盘 511 次模型调用、5,102,238 tokens 是下界。

修复版增加明确 user continuation 与无进展保护，重复完整 JSON 且没有新外部输入时纠正一次，再重复则失败；不会把标题草稿当终稿。反思/pool/reviewer 纠正提示列出具体 ID，原校验规则和评分阈值保留。v22 schema 网关不兼容、v23 checkpoint 格式错误、v24 DAG 错误均为无评分预检失败；v25 预检通过（92 秒、8 次调用、73,685 tokens、judge=0），源码保存在 `frozen_code_v25`，随后从空经验启动完整方法组计划。当前 `ours_v25` 元数据只是运行快照，正式终态需读取原输出；不得把旧失败成功化。详见上级 `rr_two_arm_run0.md`。

本包 `MANIFEST_SHA256.json` 覆盖导出的原始文件。所有凭据只通过本机环境变量传递；运行命令和实现见上级 `rr_two_arm_run0.md`。合作者可检查封存答案与评分、延续修复后的新运行，但应保留旧失败轨迹并披露闭卷协议与恢复过程。
