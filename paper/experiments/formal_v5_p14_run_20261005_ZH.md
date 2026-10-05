# v5 联合实验 p14 运行登记（2026-10-05）

本次从空经验库执行 v5 联合实验。三个 run 使用固定 EVO 题序与相同题目成员，保留全部失败和成本；本文件是运行登记，不是最终成绩报告。

## 实现与输入身份

- 源码提交：`3deda0c`，已推送 `chivector/JIT-Rubric-MAS` 的 `main`。
- 独立源码副本：`.runtime/formal_v5_p14_code_20261005`，由该提交的核心源码归档提取；与工作树的运行 fingerprint 一致：`f117788641b34cc31331964b531752722e532798c424c8af16a3aa6ad14b1dcf`。
- 源码 tar SHA256：`2c5119c0ec364776c2ea54cefa323698a093a8d3fe4b09697834f951e1312373`。
- Bundle：`.runtime/formal_v5_assets_20261004/bundle_p14_resume_20261005/bundle.json`；绑定 p14 配置与 `*_v5_relevance` 固定证据包。
- Registration SHA256：`04b5640cbf11c30ec20381276848e52e146b9817d75b29f7452b584a06fef2bf`。
- 构建只读取 Pool 并记录临时 harness；完整 judge 反馈后才归因与进化。VAL/TEST 不写长期经验。

生成模型为 `deepseek-v4-flash-vision`，预检返回 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`；judge 请求 `gpt-5.6-sol`，预检返回 `gpt-5.6-sol-2026-07-09`。身份来自网关回报，实际每次调用身份保存在预算记录中。五个角色真实预检全部通过，共 5 calls / 1,054 provider-reported tokens；预检成本单列，不计入任务成绩。凭据仅注入进程环境。

使用 `iterative_shared_ledger`；每题 2,000,000 token 上限，调用数无固定上限；meta/global/local 单次输出上限 12,000，exec 8,192，judge 1,024。VAL 与 TEST 提交各使用 2 workers。指标和选版规则沿用冻结协议。

## 执行与数据位置

1. EVO/VAL：`outputs/formal_v5_evo_val_20261005_p14`，共 180 EVO + 450 VAL 槽位，保存 C0/C15/C30/C45/C60 完整状态。
2. 三个 run 都有合格 checkpoint 才启动 TEST；无合格候选则保留 inconclusive，不补题。
3. TEST：`outputs/formal_v5_test_20261005_p14`，按 `register → submit 全部 1,911 槽 → seal → score` 自动衔接。
4. 阶段状态：`.runtime/formal_v5_assets_20261004/p14_formal_stage_status_20261005.json`；日志：同目录 `p14_formal_runner_20261005.log`。

论文数据以完整 registration、journal、checkpoint SQLite、逐题提交/评分及 budget receipt 为依据。中断槽不重采样，未知成本不能按零报告；旧 p20/p12 产物与本次分开保留，见 [中断审计](formal_v5_interruption_audit_20261005.md)。最终报告完成前，不将部分评分填入正式成绩表。

## 启动前验证

Pipeline、双层进化、动态 Pool 与经验校验回归共 **116 passed**；joint bundle、executor 与 TEST worker 回归 **33 passed**。严格经验适用性门限保留，仅修正不满足既有 scope 的合成 fixture；新负例仍验证不相关任务不能检索该经验。提交失败的已结算 token 现在可进入失败槽成本统计，跨任务重复短 proposal ID 与冻结重放均有回归覆盖。
