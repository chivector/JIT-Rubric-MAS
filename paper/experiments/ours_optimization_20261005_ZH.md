# Ours 持续优化记录：2026-10-05

## 执行约定

现有 v5 产物作为中间 checkpoint。后续仅运行 Ours，已完成的 Direct/Baseline 和 Native JIT 产物及分数只读复用，不重新生成或评分。每轮沿同一条路线解决当前最主要的瓶颈，只改一个因素，立即用固定少量样本验证；有效则保留，无效则回退。收敛前不启动完整 benchmark，也不同时维护多个候选方案。

局部结果登记为开发诊断，不改写原 v5 协议、封存结果或已有失败。公开输入、证据、配置、代码和经验快照均绑定身份；所有 Ours 产物先封存，再并发评分。每个槽只消费一次，失败保留，不用重采样覆盖。源 EVO 是默认开发范围；如另行显式选用 VAL/TEST，结果仍属于开发诊断，须记录曝光范围，不能当作干净的最终确认结果。

并发分三层记录：样本 workers、评分 workers、进程级 max_inflight_requests。i01 使用 16；后续 i02 将三层上限提升到 64，并保持每题的 agent 数量、任务内部并行结构和预算不变。多个样本和 benchmark 可并行，但源内经验更新仍遵守依赖顺序。任务总 token 预算、真实使用量、缺失分数和失败率均保留。

## 当前诊断

- `.runtime/formal_v5_assets_20261004/joint_run_p20_max_concurrency.err.log` 记录 `ModuleNotFoundError: jit_mas`，该轮没有模型执行结果。Python 入口应以 `python -m scripts.<module>` 启动。
- p18/p19 的规划截断集中在 `AgentSpec.communication`。该字段出现数万字符的重复职责和禁止事项，单次耗尽 12,000 输出 token，唯一纠正轮仍可再次截断。
- 首个单点改动仅将结构化规划的 `communication` 注解限制为 1,024 字符，并要求简述交接对象、内容和格式。原任务、公开证据、质量要求、agent 数量、预算及执行路线保持原配置。
- DSQA 是优先优化对象。旧 v19 ACT 样本误纳入不满足 composite 阈值的 Tennessee，并遗漏 Utah；另一道多条件国家筛选题出现 Unknown/Fails 混淆与重复内部评审文字。这些问题留作通信截断修复验证后的下一步诊断，不在同一轮修改。

## 固定局部验证样本

| Benchmark | 原始行 / task ID | 固定 v19 Direct / Ours / JIT |
|---|---|---:|
| ResearchRubrics | 60 / `6847465956a0f6376a6054a7` | 0.450 / 0.575 / 0.450 |
| DeepSearchQA | 129 / `deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c` | 0.933333 / 0.857143 / 1.000000 |
| DeepSearchQA | 170 / `deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c` | 0.000000 / 0.000000 / 0.666667 |

参照来自 `outputs/development_pilot_20261004_rr_evidence_v19/summary.json` 和 `outputs/development_pilot_20261004_dsqa_evidence_v19/summary.json`。这些旧分数继续保留其原 judge/model、证据和配置身份；新轮次与旧参照的差值是开发观察，不能把全部变化归因于通信字段限制。

本轮复用 `outputs/development_rr_evidence_20261004` 与 `outputs/development_dsqa_shared_evidence_20261004` 的原有固定公开证据包，避免最新自动搜索包中的无关字典页或空检索影响 MAS 验证。DSQA 证据存在人工整理和检索预算偏离，登记为开发证据轨道；不能混入原正式共享证据结果。

## 局部运行入口

`scripts/run_ours_iteration.py` 只生成 Ours，支持显式跨 benchmark 任务清单、只读经验快照和历史结果参照。每轮使用新的输出目录；代码或配置改变后不能继续旧登记。

```powershell
$iterationArgs = @(
  '--bundle', '.runtime/formal_v5_assets_20261004/bundle_p20_max_concurrency/bundle.json',
  '--output', 'outputs/ours_iteration_20261005_i01',
  '--benchmark', 'researchrubrics', '--benchmark', 'deepsearchqa',
  '--task-id', 'researchrubrics=6847465956a0f6376a6054a7',
  '--task-id', 'deepsearchqa=deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c',
  '--task-id', 'deepsearchqa=deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c',
  '--evidence', 'researchrubrics=outputs/development_rr_evidence_20261004',
  '--evidence', 'deepsearchqa=outputs/development_dsqa_shared_evidence_20261004',
  '--workers', '16', '--judge-workers', '16', '--max-inflight', '16',
  '--reference', 'outputs/development_pilot_20261004_rr_evidence_v19/summary.json',
  '--reference', 'outputs/development_pilot_20261004_dsqa_evidence_v19/summary.json'
)
.venv/Scripts/python.exe -m scripts.run_ours_iteration --mode register @iterationArgs
.venv/Scripts/python.exe -m scripts.run_ours_iteration --mode run @iterationArgs
```

i02 只改变并发配置，使用同一批三个固定样本和只读参照；i01 保持冻结。登记参数为：`--output outputs/ours_iteration_20261005_i02 --workers 64 --judge-workers 64 --max-inflight 64 --judge-parallel 64`，其余参数与上例相同。并发提升属于执行吞吐调整，不改变 prompt、MAS 结构、预算或样本选择；若服务端出现限流或超时，保留 i01 并将下一轮降回可观测稳定值。

`register` 仅检查材料和登记；`run` 才调用 API。未提供 `--state` 时从空经验起步；有可复用的冻结 checkpoint 时通过 `--state <snapshot.sqlite>` 显式指定，不能把任意状态称为 Selected。当前 p18/p19 仅保存 C0，尚无完整封存的联合 VAL winner。

## 本轮状态

首轮已登记至 `outputs/ours_iteration_20261005_i01/registration.json`：固定 3 个 Ours 槽、原有公开输入及证据哈希与参照匹配，注册哈希 `e92e4e3070f0f2e74dbb19e76dc7b086073ceb2871c45f30fd0774a714f85927`。并发迭代 i02 已登记至 `outputs/ours_iteration_20261005_i02/registration.json`，注册哈希 `c46a913830a73df76807884ddbf412caca4c9f62b321696e25a0a2632d1eed4d`，三层并发均为 64。两轮全部槽仍为 pending，模型调用和 token 均为 0。旧参照与新配置不相同，匹配标记已记录，不能据其差值单独证明通信字段改动或并发调整的因果作用。

相关规划与局部运行器验证 **82 passed**，`git diff --check` 通过。当前会话未提供 `JIT_BENCHMARK_API_KEY` 和 `RESCORE_JUDGE_API_KEY`，两个现有 endpoint 的匿名 `/models` 请求均返回 401；真实 Ours 验证等待有效凭据。不能据离线测试或登记宣称分数提升，也不启动 Baseline/JIT 或完整评测补偿这一缺口。

## 动态 Pool 检查与 GitHub 发布

用户授权：若真实指标偏低，可以调整实现方案；验证并确定方案后，将代码提交并推送至 `https://github.com/chivector/JIT-Rubric-MAS`。当前本地 `main` 与远端 `origin/main` 对齐，已确认目标仓库可访问且当前认证具备推送权限。

已准备本地小样本启动器 `outputs/dynamic_pool_benchmark_20261005/run.ps1`，使用上表三个 source EVO 任务，共四次生成：

1. DSQA 行 170 从初始 C0 冻结快照执行，只评分，不归因或进化。
2. RR 行 60 执行并进行 judge 后双层归因、反思和动态 Pool 更新。
3. DSQA 行 129 复用上一题的长期状态，再执行并进化。
4. 冻结更新后的 C2 状态，再执行 DSQA 行 170，只评分，不更新经验。

两次 probe 保持模型、judge、预算与公开证据一致；同时检查 `native_score`、完整率、实际 Token/调用成本、临时 harness 是否审议入池、结构操作及下一题是否复用更新后的原型。构建和执行阶段保持长期 Pool 不变，只有执行后进化阶段可增删、拆分、合并、专化或重组。Pool 操作可以为空，不以强制改变结构作为成功条件。

如果观察到低分，先区分执行失败、证据不足、规划/分工、输出与 harness 缺陷，再选择一个有过程证据支持的实现因素修改；另行登记新的代码和配置身份，用固定少量 EVO 样本复查，有效则保留，无效则回退。随后在未参与该轮调优的 EVO 任务上补充检查，保留失败和成本，不能用同题反复调优的分数宣称泛化提升，也不能将答案或私有评分标准写入设计策略或 harness。

本轮动态 Pool、双层进化、RubricGraph 职责分配、规划 transport 和开发运行器相关回归 **219 passed**，核心实现审查未发现必须修正的行为缺陷。启动器静态检查通过；凭据预检仍缺 `JIT_BENCHMARK_API_KEY` 和 `RESCORE_JUDGE_API_KEY`，四次生成均未启动，模型调用和 Token 均为 0。尚无新真实分数，因此尚未作性能结论或推送最终版本。

最终发布纳入确定的 `jit_mas` 实现、开发评测脚本、相关测试和方法/运行文档。上述本地启动器依赖被忽略的 `.runtime` bundle、数据和公开证据缓存，不作为干净 clone 可直接启动的公开示例；凭据、运行状态及本地生成物保留在已有忽略目录。公开运行入口继续使用 `scripts/run_jit_mas.py`、`configs/jit_mas.native.example.yaml` 和 `docs/jit_mas.md`，所需数据与证据按文档单独准备。
