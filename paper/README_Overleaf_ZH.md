# JIT-Compose Overleaf 工程

**JIT-Compose: Evolving Multi-Agent Workflow Synthesis for Open-Ended Generation**

本目录是论文源文件。主文档为 `main.tex`；`appendix.tex`、`overview.tex` 和 `experiment_protocol.tex` 由主文档引入，不是独立编译入口。实验结果表保持空白，不以软件测试、API 探针或历史 pilot 代替真实正式结果。

## 编译

将本目录作为独立 Overleaf 工程导入，主文档设为 `main.tex`，编译器使用 `pdfLaTeX`。不要与仓库根目录的历史稿件副本混合。无需 Python、外部字体或 shell-escape。

```text
main.tex                         主文档
appendix.tex                     补充分析与实验协议入口
overview.tex                     TikZ 架构图
experiment_protocol.tex          当前三 benchmark 协议附录
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

活动注册为 [`experiments/benchmark_suite_v3.json`](experiments/benchmark_suite_v3.json)，中文详细计划为 [`experiments/benchmark_plan_v3_ZH.md`](experiments/benchmark_plan_v3_ZH.md)。`protocol_v2.json` 保留 RR 细则；v1/v2 原始注册和划分文件不改写。不要把旧协议中的每条更新成对验证当作当前方法。

| Benchmark | Development | Evolution | Val | Test | 角色 |
|---|---:|---:|---:|---:|---|
| ResearchRubrics | 18 | 30 | 20 | 33 | 主方法、baseline、三个进化消融及 G/GO |
| DeepSearchQA | 50 | 150 | 100 | 600 | 独立进化与六个核心条件 |
| DeepResearch Bench II | 0 | 0 | 0 | 132 | RR 选中状态的冻结外部迁移 |

RR 保留旧成员，DSQA 只按公开 `problem_category` 分层，不能用私有答案或 `answer_type` 分层或提示 executor。两个源 benchmark 各有 3 次独立运行，从独立空经验库开始，固定题序种子 20261001/20261002/20261003，不跨 benchmark 混合训练经验。DRBII 不使用目标 val，也不在 RR/DSQA 两种迁移来源中事后挑选。

精确成员保存在 `experiments/researchrubrics_splits_v3.json`、`deepsearchqa_splits_v3.json`、`deepresearch_bench_ii_splits_v3.json`。正式 profile 是仓库 `configs/benchmark_suite_v3.yaml`；密钥只取环境变量。v3 judge 输出上限为 16,000，并固定每请求一次尝试，明确覆盖旧 v2 的 4,096 上限及重试选项。

RR 每处理 5 个源题位置、DSQA 每 25 个位置做一次 val；分别保存 C0 至 C30/C150 的 7 个 checkpoint。每次都使用**固定全部 20/100 道 val，每题两份产物**，不是最近源题批次。历史结果缓存，相同完整状态复用结果。最终在全部七个 checkpoint 中筛选，完整评价门槛 36/40 或 180/200；缺失的官方分数保持 null，筛选使用明确标记的保守下界。失败和无提案也消耗源题位置，不补抽或早停。

Selected/terminal 每个状态、每道 test 生成 3 份；静态控制每题总共 3 份，跨运行共享，不能扩大成 9 个独立样本。先登记全套 test 库存并封存所有答案，再评分与释放反馈。DRBII 只读接收三个 RR winner，任何目标评分都不改变或选择源状态。

主计划为 **31,617 个名义任务执行单元**：RR 6,195、DSQA 22,650、DRBII 2,772。它不是 API 调用数，不是已经完成的实验数量；证据预处理、G/GO 共享 rubrics、登记重试和人工审计另计。各 benchmark 独立指标和区间分列，不平均成一个混合“总分”。

## 实现与结果是两件事

数据与评分适配、稳定公开 ID、分层 split 冻结、固定证据包、checkpoint 日志/快照/全量 val selector、真实产物延迟评分都有实现入口。离线测试覆盖公开/私有分离、计量、失效处理及生成到经验写入的闭环；正式状态仍是 **`REGISTERED_NOT_RUN`**。只有真实执行、冻结身份和相应 condition 审计齐全的 artifact 才能填入结果模板。

仓库命令入口：

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

正式测试使用 `run_benchmark_test` 的 register → submit → seal → score：先注册全部 42 个 condition 和 23,247 个名义 test slots，逐条件提交答案，整个库存封存后才评分。状态来源和原始数据/配置/证据哈希进入注册；任意未封存的经验库不能冒充 val 选中状态。两个 smoke 命令分别覆盖 checkpoint 与延迟评分路径，不是全量正式 benchmark。

本套件是 **controlled shared-evidence track**，不是官方不受限 leaderboard。DSQA 与 DRBII 使用作者 schema/聚合规则的明确计量适配，prompt 与共享代理 judge 不宣称官方完全相同。DSQA 的 F1 按作者 notebook 的实际 TP/FP/FN 公式；DRBII 的 -1 是 blocked 标签，不是负权重。DRBII 发布文件含 9,415 条原始 rubric，其中 6 条维度内重复，按作者字典覆盖后为 9,409 个唯一评分条目。

## 稿件与数据完整性

`author_notes/` 为内部编辑材料，匿名投稿时另行筛选。保留数据许可与来源，不把参考报告、私有答案、评测 rubrics 或 API 凭据并入公开 manifest。历史修改记录不代表本轮性能或编译验证。

原 `acl.sty`、`acl_natbib.bst` 保持不变及原许可证。Git blob SHA-1：

- `acl.sty`: `d9b74d0e6d1ea7a41929ff30111a112e1d23f959`
- `acl_natbib.bst`: `086cb0bc745cf1120aef9c99aa295ccdba3e736c`
