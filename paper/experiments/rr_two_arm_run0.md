# ResearchRubrics 两臂单次运行

本入口只运行 v5 冻结成员中的 ResearchRubrics，执行 run0 一次；这是 RR 单 benchmark 对比，不是六 benchmark 联合进化实验的完整复现。

- 两臂同时启动：Direct Single-Agent 直接生成全部 33 道 TEST；JIT-MAS 从空经验开始。
- JIT-MAS 按 run0 中 RR 的相对顺序遍历 20 道 EVO，保存 C0、C5、C10、C15、C20。
- 每个候选评估同一固定 10 道 VAL，每题一次。完整评分至少 9/10 才可选；使用每题固定理论范围归一化，缺失原生分数为 null、仅选择效用计 0；平局按完整数、较早位置、状态哈希选择。
- 完整状态相同的候选允许复用同身份 VAL 结果并记录来源。无合格候选则不确定，不补跑有利轨迹。
- 两臂使用相同 DeepSeek 执行模型、公开题面和冻结证据；GPT judge 独立评分。TEST 不更新经验、不按得分重采样。
- 66 个 TEST 槽位全部提交或登记失败后封存，再释放评分。完整题均值缺失时保持 null；另列完整子集均值、完整配对数、失败和 token 开销。

正式运行前必须准备覆盖全部 63 道选中题的公开证据目录，含可验证的 `manifest.json` 与证据包。`--closed-book` 只用于另行明确同意的协议变更；当前用户要求按原计划准备共享证据。

执行入口：

```powershell
.venv\Scripts\python.exe -m scripts.run_rr_two_arm_pilot `
  --data outputs/researchrubrics_live_20260930/processed_data.jsonl `
  --joint-manifest paper/experiments/joint_task_splits_v5.json `
  --evidence-dir <verified-evidence-directory> `
  --output <new-output-directory>
```

凭据只从进程环境 `RR_EXEC_API_KEY` 与 `RR_JUDGE_API_KEY` 读取，不写入实验配置。默认模型为 `deepseek-v4-flash-vision` 和 `gpt-5.6-sol`。`--check-only` 验证固定数据 SHA256、成员与证据包，不调用模型，也不要求凭据。预检目录与正式运行目录分别使用新路径。

用户允许通用知识后，探索版可使用 `--closed-book`。`--arm baseline` 单独运行固定 33 道 TEST；`--arm ours` 单独进化、选版后测试；默认 `--arm both` 同时启动两臂。

连接故障后可用 `--arm baseline --closed-book --reuse-baseline-from <source-output-directory>` 在新目录保留已有未评分生成记录，只生成缺失题。来源数据、TEST 成员、执行配置与答案哈希均须一致；复制的生成预算和来源路径写入产物。新评分必须统一 judge 身份，不能混用旧 judge 的分数。此类恢复包含此前失败请求的重试，须披露，不能表述为无故障的初始一次运行。

`--judge-model`、`--judge-max-tokens`、`--judge-attempts`、`--judge-parallel` 与 `--max-inflight-requests` 均在新运行前固定。`JIT_MAS_MODEL_ATTEMPTS` 控制传输重试；`JIT_MAS_DISABLE_KEEPALIVE=1` 禁用连接复用。默认启用 TLS 校验；临时 `JIT_MAS_TLS_VERIFY=0` 可配合 `JIT_MAS_TLS_ENDPOINT` 仅限制指定执行端点，评分端点保留校验。传输故障的 token 用量可能仅为估计，无法据此可靠估算费用。

默认 `--judge-parallel 2` 在每个 rubric 使用独立 judge 实例并共享当前题的计量账本；整个进程的请求并发仍最多 2。两臂使用同一评分并发、原始 rubric 顺序、提示和有符号权重聚合。可显式冻结为 `--judge-parallel 1` 使用顺序评分。

产物包括 `pilot_metadata.json`、两臂 `report.json`、checkpoint journal、逐题提交和预算、`test_release/seal.json`、逐题评分及 `comparison.json`。运行期间冻结代码、配置、数据、清单与证据哈希；工程修复须保留旧记录并另建新版本，不能边跑边修改冻结身份。

此前 `outputs/rr_two_arm_pilot_20261001` 两臂因 split 路径错误在实验 API 请求前退出；该目录保留为失败预检，不计入真实 benchmark 成绩。
