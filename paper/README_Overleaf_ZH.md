# JIT-Compose Overleaf 工程

**JIT-Compose: Evolving Multi-Agent Workflow Synthesis for Open-Ended Generation**

本目录是论文源文件。主文档为 `main.tex`；`appendix.tex`、`overview.tex` 和 `experiment_protocol.tex` 由主文档引入，不是独立编译入口。实验结果表保持空白，不以软件测试、API 探针或历史 pilot 代替真实正式结果。

## 编译

将本目录作为独立 Overleaf 工程导入，主文档设为 `main.tex`，编译器使用 `pdfLaTeX`。不要与仓库根目录的历史稿件副本混合。无需 Python、外部字体或 shell-escape。

```text
main.tex                         主文档
appendix.tex                     补充分析与实验协议入口
overview.tex                     TikZ 架构图
experiment_protocol.tex          当前六 benchmark 联合进化协议附录
experiments/                     注册、公开题目清单与中文实验计划
references.bib                   参考文献
author_notes/results_template.json 空结果模板
MANIFEST_SHA256.json             本目录完整性清单
```

本地有 TeX Live 时：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

没有 latexmk 时依次运行 `pdflatex main`、`bibtex main`、`pdflatex main`、`pdflatex main`，或使用本目录 `compile.sh` / `compile.ps1`。旧 `revision_notes/build_validation.json` 只代表当时版本；本轮未声称已完成新的 PDF 编译或排版检查。

## 方法边界

论文方法名为 **JIT-Compose**；实现仍使用 `jit_mas/` 和 `scripts.run_jit_mas`。流程保留原生 JIT 的任务条件化代码生成：预测需求与协同规划 → 原生 harness → single-pass shared ledger → 提交产物 → 独立评分 → 协同归因 → 直接经验更新。

逐条经验的 accept/hold/reject 质量门已删除。结构、来源、目标、版本和幂等检查仍执行；每道源题最多直接提交一个提案。周期 val 比较整个 checkpoint，不决定单条经验是否写入，也不让进化轨迹回退到当前 val 最优状态。

## 当前实验注册

活动协议为 [`experiments/joint_protocol_v5.json`](experiments/joint_protocol_v5.json)，精确任务身份与联合顺序见 [`experiments/joint_task_splits_v5.json`](experiments/joint_task_splits_v5.json)，合作者可直接查阅[简明实验设置](experiments/experiment_plan_v5_ZH.md)与[完整题号清单](experiments/task_assignments_v5.md)。v1-v4 注册和划分文件作为历史版本保留，不改写。v5 缩减题数与重复次数，保留跨 benchmark 联合进化和一个通用生成器版本。

| Benchmark | Evolution | Val | Test | 角色 |
|---|---:|---:|---:|---|
| ResearchRubrics | 20 | 10 | 33 | 联合源；保留全部 33 道 clean test，历史 18 题不纳入 |
| DeepSearchQA | 20 | 10 | 50 | 联合源，事实检索与完整性 |
| WritingBench | 20 | 10 | 50 | 联合源，从固定 1,000 题版本抽取 |
| DeepResearch Bench II | 0 | 0 | 40 | 冻结跨 benchmark 测试，20 英文 + 20 中文 |
| IFEval | 0 | 0 | 50 | 冻结指令遵循诊断 |
| IFBench | 0 | 0 | 50 | 冻结单轮新约束诊断 |
| 合计 | 60 | 30 | 273 | 363 道不同题目 |

**没有 Dev。** 只在 v4 对应父分区内，按公开元数据分层和固定哈希抽子集：EVO 不移到 VAL/TEST，VAL 不移到 TEST，不按结果挑题。未选中的 2,611 题为 unused/reserve，不是 Dev，也不默认跑；追加实验须另行注册。RR 的历史 18 题仍在 v4 保留曝光记录，不进入本轮。WritingBench 原始第 1 条保留已知曝光，只能进入 EVO 或 unused，不强制必选。题号是固定数据版本中从 1 开始的数据记录序号，CSV 不计 header；稳定 task ID 与文件哈希共同确定身份。答案、隐藏 rubric 和得分不用于抽样。

三个源 benchmark 在**每个 run 内共享同一个经验库**，不是各自进化后选一个。固定题序种子为 20261001/20261002/20261003；三个 run 各从空库开始。每条轨迹共 60 道源题，一次遍历、四个混合阶段，每阶段 5 RR + 5 DSQA + 5 WritingBench，共 15 题。预存 C0、C15、C30、C45、C60 五个候选。

每个候选都评**固定全部 30 道 VAL，每题一份产物**，不是最近的源题批次，也不累计更换 VAL。三个 benchmark 分别至少 9/10 完整评价。按固定理论得分范围归一化，再先对每个 benchmark 求均值、最后三个 benchmark 等权平均；不能用已观察到的 val/test 最小最大值归一化。缺失官方分数保留 null，筛选使用明确标记的可行下界。最终从五个候选中按预定规则选一个完整状态；失败和无提案仍消耗源题位置，不补抽、回滚或早停，VAL 不写经验。

**每个 run 只选一个通用版本，原样用于全部六个 benchmark。** 最终 meta-agent 发布包包含固定生成模型身份、代码、prompts、配置、工具/检索策略和联合经验快照，不是训练后的模型权重，也不是一支固定生成 MAS。面对新任务仍按任务生成 MAS。默认发布版本提前指定为 run 0 的 VAL winner，不能按 TEST 结果挑最好 run；论文报告全部三个 run。

五个核心方法是 Initial、Selected、Direct、原生 JIT、固定 rubric-MAS。三个 Selected 状态各对每道 TEST 生成 1 份；四种静态控制每题各 1 份，跨运行共享，不能扩大成 3 个独立样本。全部测试条件先登记并封存答案，再评分与释放反馈；底座、公开输入和资源上限相同。Terminal、G/GO 与额外消融不在本次必跑清单内，须单独注册，不自动开启。

RR、DSQA、WritingBench、DRBII 是四个主 benchmark；IFEval/IFBench 是独立诊断组，分别进行 Holm 校正。各 benchmark 原生指标和区间分列，不平均成混合排行榜。小样本区间可能较宽；每状态每题只有一份产物，不估计同状态生成方差，三个种子只控制进化题序。本协议是 subset track，不能冒称官方全量或 SOTA。

核心工作量为 `3×60 = 180` 个 EVO、`3×5×30 = 450` 个 VAL、`273×(3+4) = 1,911` 个 TEST，共 **2,541 个名义任务单元**，较 v4 同五方法口径的 61,134 减少约 95.8%。这不是 API 次数、token 上限或已完成调用数；证据预处理、评分调用、人工审计和额外实验另计，实际成本扣除可审计的同状态复用。

## 实现与结果是两件事

已有三 benchmark 的数据与评分适配、稳定公开 ID、split 冻结、固定证据包、checkpoint 日志/快照/全量 val selector、真实产物延迟评分入口，离线测试覆盖相应闭环。**本次 v5 协议与清单不代表现有 v3 入口已经支持六 benchmark 联合执行，更不代表已完成正式实验。** 只有真实执行、冻结身份和相应 condition 审计齐全的 artifact 才能填入结果模板。密钥仅从环境变量读取。

已有三 benchmark 基础设施的命令入口，不作为 v5 六 benchmark 启动指令：

```powershell
.venv\Scripts\python.exe -m scripts.prepare_benchmark_suite --help
.venv\Scripts\python.exe -m scripts.run_benchmark_experiment --help
.venv\Scripts\python.exe -m scripts.prepare_benchmark_evidence --help
.venv\Scripts\python.exe -m scripts.run_benchmark_test --help
```

执行一个无需 API 的 checkpoint smoke：

```powershell
.venv\Scripts\python.exe -m scripts.run_benchmark_experiment --mode evolve --smoke --output outputs/checkpoint_smoke_v3
.venv\Scripts\python.exe -m scripts.run_benchmark_test --mode smoke --campaign outputs/test_release_smoke_v3
```

原生生成代码必须显式启用 `--unsafe-local`，这不是 OS sandbox；当前题证据工具和数据边界不等于能阻止任意恶意 Python 访问宿主文件。论文不得宣称已经实现强系统隔离。真实进化、证据请求和正式测试均需指定输入与输出，不由文档更新自动启动。

既有 `run_benchmark_test` 采用 register → submit → seal → score，先登记库存、逐条件提交、整个库存封存后才评分。v5 必须以其六 benchmark 联合状态和测试清单建立新的运行库存，不能沿用 v3 的 42 个 condition 与 23,247 个 test slots。状态来源和原始数据/配置/证据哈希进入注册；任意未封存的经验库不能冒充 VAL winner。两个 smoke 命令只覆盖既有 checkpoint 与延迟评分路径，不验证全部 v5 条件，也不是正式 benchmark 结果。

本套件是 **controlled shared-evidence track**，不是官方不受限 leaderboard。DSQA 与 DRBII 使用作者 schema/聚合规则的明确计量适配，prompt 与共享代理 judge 不宣称官方完全相同。DSQA 的 F1 按作者 notebook 的实际 TP/FP/FN 公式；DRBII 的 -1 是 blocked 标签，不是负权重。DRBII 发布文件含 9,415 条原始 rubric，其中 6 条维度内重复，按作者字典覆盖后为 9,409 个唯一评分条目。

## 稿件与数据完整性

`author_notes/` 为内部编辑材料，匿名投稿时另行筛选。保留数据许可与来源，不把参考报告、私有答案、评测 rubrics 或 API 凭据并入公开 manifest。历史修改记录不代表本轮性能或编译验证。

原 `acl.sty`、`acl_natbib.bst` 保持不变及原许可证。Git blob SHA-1：

- `acl.sty`: `d9b74d0e6d1ea7a41929ff30111a112e1d23f959`
- `acl_natbib.bst`: `086cb0bc745cf1120aef9c99aa295ccdba3e736c`
