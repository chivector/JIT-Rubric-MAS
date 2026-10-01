# RR 两臂运行交接

这是一份 **未完成的运行记录**，不是有效 benchmark 对比结果。

- 固定输入：ResearchRubrics 官方数据 SHA256 `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`，v5 RR 成员 `20 EVO / 10 VAL / 33 TEST`。
- 编排：两臂同时提交；Single-Agent 直接 TEST；JIT-MAS 按 `C0/C5/C10/C15/C20` 进化和 VAL 选版；TEST 先封存后评分。
- 本次快速协议允许模型通用知识，不使用外部检索；该偏离已写入 `pilot_metadata.json` 的 `knowledge_policy`。
- 两臂在首次 DeepSeek 生成请求返回 HTTP 401 `Invalid token` 后终止；有效提交数为 0，judge 没有产生有效评分。
- `comparison.json`、两臂 `report.json`、预算和封存槽位保留了失败证据；请勿把其中的 null 分数当作模型质量结果，也不要从本目录恢复重采样。
- GPT judge 端点的独立连通性检查为 HTTP 200；阻断点是 DeepSeek 执行凭据/网关认证。

修复 DeepSeek 凭据后，应使用新输出目录重新运行，不要覆盖此目录。正式入口和协议说明见 `paper/experiments/rr_two_arm_run0.md` 与 `scripts/run_rr_two_arm_pilot.py`。
