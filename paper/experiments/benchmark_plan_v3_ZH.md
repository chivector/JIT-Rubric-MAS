# 论文三 Benchmark 实验计划与执行协议 v3

当前注册入口是 `benchmark_suite_v3.json`。本文件说明要检验的主张、各 benchmark 的题目用途、筛选频率和执行顺序；`protocol_v2.json` 保留为 ResearchRubrics 细则，旧文件不改写。正式结果状态是 `REGISTERED_NOT_RUN`：代码实现和离线软件验证不等于真实模型实验结果，也不等于性能达到 SOTA。

## 1. 为什么选择这三个 Benchmark

| Benchmark | 检验的问题 | 必做范围 | 不作出的主张 |
|---|---|---|---|
| ResearchRubrics | 预测需求、协同规划和归因是否帮助开放式任务质量与经验复用 | 主方法、三个 baseline、三个进化消融、G/GO 机制对照 | 33 道自定义 test 的结果不等于完整官方 leaderboard |
| DeepSearchQA | 方法能否改善多步事实检索的答案集合完整性，同时减少额外错误答案 | 独立进化；主方法初始/选中/终态及三个 baseline | 从作者 eval 中再划分的任务不能称为作者官方训练集 |
| DeepResearch Bench II | ResearchRubrics 上获得的经验能否迁移到另一套中英文长报告任务 | 132 题全量外部留出，只读迁移及四个静态对照 | 不用目标题进化、选 checkpoint 或挑选最好的源运行 |

不把所有 JIT 环境强行纳入本论文。OfficeBench、OdysseyBench、DeepPlanning 等包含交互环境、办公产物或特定约束求解，需要独立的环境控制与产物评分；它们不是本次“固定公开证据上的研究产物生成”主张的替代检验。Skill-MAS、Meta-Team 是比较方法，不是数据集；只有固定作者实现或明确报告适配版本后才增加相应行，不能把普通固定团队换个名字当作复现。

## 2. 每个 Benchmark 的固定划分

| Benchmark | 总题数 | Development | Evolution | Validation | Test | 每多少源题做一次 val |
|---|---:|---:|---:|---:|---:|---:|
| ResearchRubrics | 101 | 18 | 30 | 20 | 33 | 5 |
| DeepSearchQA | 900 | 50 | 150 | 100 | 600 | 25 |
| DeepResearch Bench II | 132 | 0 | 0 | 0 | 132 | 不做目标 val |

这里的 Evolution 是经验库进化，不训练模型参数。Development 只用于实现调试和协议校准，不能进入正式经验库或最终分数。Val 只选整个经验状态，不进入逐题归因，也不供人类看完结果后修改实现。Test 只产生最终评价，不写经验。

### ResearchRubrics

- 完整保留 `splits_v2.json` 中 18/30/20/33 的成员与三份源题顺序，不因为增加 benchmark 重新抽取“有利”的题目。
- 18 道 development 保留全部历史暴露、保守隔离和先前四题 pilot。历史 API 调试结果不能填写新的论文主表。
- 这是项目自定义任务级划分，不声称 JIT 或 ResearchRubrics 官方发布了该划分。任务不重叠不代表主题不重叠。

### DeepSearchQA

- 固定作者 revision `b2623f8653065c2672de6d941fc5434cd652376c` 的 900 道题；作者发布的是单个 `eval` split。
- 使用公开 `problem_category` 的 17 个类别进行分层；**不能使用私有 `answer_type`、答案数量、答案内容或模型分数进行分层**。
- 首先将已有暴露任务及其重复组放入 development。随后依次分配 development 剩余额度、evolution、validation，test 使用其余全部任务。
- 每一步在保持重复组完整及总题数精确的约束下，最小化各公开类别相对剩余总体的比例配额绝对偏差；类别、任务和同分分配采用固定排序与 SHA256 排序。该算法是 `jit_mas/experiment_splits.py` 的确定性实现，不依赖数据行顺序。
- 固定成员种子为 `jit-compose-deepsearchqa-v3`。少数类别不保证在四个 subset 都出现，manifest 记录实际分层；不能为了各列“好看”而事后换题。
- 任务 ID 来自标准化公开问题的完整 SHA256，命名空间为 `deepsearchqa:`。答案改动不改变题目 ID，但改变原始数据文件哈希，导致旧注册不能继续使用。
- 归一化重复题不能跨 split。跨 benchmark 的完全重复公开问题必须在正式运行前解决，并留下版本化协议变更；不能在看过分数后静默删除。

### DeepResearch Bench II

- 固定 revision `b38f360603db9531b102aef8c166cedb8509b6f6` 的 132 题：英文 66、中文 66、22 个主题。
- 不另划目标训练或验证集。三个 ResearchRubrics 运行各自通过源 val 选出的状态，原样冻结后全部迁移；不同时试 DSQA 状态再挑赢的来源。
- 不能假定 66+66 是 66 对翻译题，公开记录不足以支持这个配对。语言和主题结果仅作预先登记的描述性分项；主统计单位仍是实际任务。
- 推理使用原始顶层 `prompt`，保留其中公开的禁止引用来源说明；不能错误地用 evaluator 内部 `content.task` 替代题面。私有 rubric 和 blocked 结构只进入提交后的 evaluator。

## 3. 进化次数与 Val 筛选

两个进化 benchmark **各自有 3 次独立运行、各自从空经验库开始**。不先在 RR 进化，再把状态当作 DSQA 的初始状态；跨 benchmark 迁移只在单独登记的 DRBII 外部测试中发生。

| 项目 | ResearchRubrics | DeepSearchQA |
|---|---|---|
| 每次运行处理源题 | 30 道，各一次 | 150 道，各一次 |
| 保存与筛选位置 | C0/C5/C10/C15/C20/C25/C30 | C0/C25/C50/C75/C100/C125/C150 |
| 每个 checkpoint 使用的 val 范围 | 全部固定 20 题 | 全部固定 100 题 |
| 每道 val 的独立产物数 | 2 | 2 |
| 每次 val 分母 | 40 | 200 |
| 参与筛选的最低完整评价数 | 36 | 180 |

三份题序种子固定为 20261001、20261002、20261003。它们只控制源题顺序，不代表 API 模型采样受相同 seed 控制。RR 沿用已冻结顺序；DSQA 使用登记的哈希顺序。题目成员只划分一次，不能每个 seed 重新划分 val/test。

**每次都在全量固定 val 上比较所有已出现的 checkpoint，不是只验证最近 5/25 道源题，也不是只看上轮胜者。** 历史 checkpoint 的评价缓存；完全相同状态与执行身份复用旧结果，并标记 `reused_from`，不给未改变的状态反复抽样挑高分。

经验仍按逐题归因后的首个合规提案直接写入，每个源题至多一次。失败、评价不完整、无提案和结构检查不通过仍消耗一个登记源题位置；不能补抽到成功写满 30/150 次。Val 不恢复 accept/hold/reject，不决定单条经验是否落库。

进化始终从最新累积状态继续，不回滚到 val 最优状态、不早停。全部源题位置处理后，每次运行独立封存一个 winner，三次运行全部报告，不能再挑一个最好的 run。

筛选效用采用固定分母：完整任务评价使用真实分数，缺失评价使用任务可行下界。RR 下界是负权重之和除以正权重之和；正分母为零时保留原适配器的零与标志。DSQA 下界为 0。缺失官方字段仍为 `null`，不能把填充下界叫成真实分数。先计算最大效用，再将距最大值不超过 `1e-12` 的状态列为同分，依次选择完整评价更多、源题位置更早、状态哈希字典序更前的 checkpoint。没有满足完整度门槛的状态时该运行是 inconclusive，不用 test 挑替代状态。

## 4. 比较矩阵与 Test 重复次数

| 方法 | RR test 33 | DSQA test 600 | DRBII test 132 |
|---|---|---|---|
| Ours initial | 空经验状态 | 独立空经验状态 | 空经验状态 |
| Ours selected | 本 benchmark 的 3 个 val winner | 本 benchmark 的 3 个 val winner | **RR 的 3 个 winner** |
| Ours terminal | 3 个 C30 | 3 个 C150 | 不加这一行 |
| Direct one-call | 做 | 做 | 做 |
| Matched-backbone JIT | 做 | 做 | 做 |
| Rubric-guided fixed organization | 做 | 做 | 做 |
| No explicit rubrics | 独立进化与选中状态 | 不做 | 不做 |
| Global-only planning | 独立进化与选中状态 | 不做 | 不做 |
| Global-only attribution | 独立进化与选中状态 | 不做 | 不做 |
| G / GO | 同一批选中状态的两种构建条件 | 不做 | 不做 |

每个选中或终态状态，每道 test 生成 3 份独立产物，即每道题有 3 runs × 3 repeats。初始状态与每个静态 baseline 每道题总共只生成 3 份，在三个进化运行比较中共享，不能复制成 9 个独立样本。Selected=terminal/initial 时，只有完整执行身份相同才可复用产物，须显式记录，不虚增调用或样本数。

Matched JIT 采用相同代理生成与执行模型的原生 JIT 路径，保留其自身执行组织方式，在相同整体预算下比较；不是复现原作者 JIT-27B 权重。Direct 是质量/成本参考。固定团队 baseline 保留 task-specific rubrics、固定 Analyst/Evidence/Writer roster 和固定单次 ledger harness，不用未经审计的普通 fixed-team pilot 代替。

三项消融分别在 RR 上从空状态独立进化，完整使用相同任务、题序、全量 val 和筛选机会。Global-only attribution 仍能访问同样的合法证据，不通过故意截断 baseline 的上下文制造优势。

G/GO 固定相同的质量 rubric `R*`、公开背景、经验状态和整体预算。G 的 constructor 看不到 `R*`，只有在团队和代码结构冻结后才能向 executor 注入；GO 在构建时也能看到。普通 `explicit_rubrics=false` 开关不是这个对照。每项 condition 的有效产物必须包含相应冻结及注入审计；未产生审计记录的 condition 不可伪装成完成的对照实验。

## 5. 证据、Evaluator 与安全边界

三者统一采用 **controlled shared-evidence track**：仅根据每道题的公开信息预先构建一份固定证据包，各方法、checkpoint 和重复共享。证据准备最多 1 次 query-planning 调用、4 个搜索 query、每个 query 前 5 个结果、按固定顺序去重后至多 8 页，加公开附件。提取与轮转截断固定到 32,768 `cl100k_base` tokens，记录 tokenizer 版本、URL、定位、时间、内容哈希及失败。不能在看到成绩后换证据。预处理计入单独成本，也报告部署等价成本。

任务工具接口只提供当前题的冻结 pack，不新增实时检索。**这不是 OS 级 sandbox 声明**：本机生成 Python 的 `--unsafe-local` 仍拥有宿主进程权限，必须标识。接口隔离测试证明公开/私有数据流与工具边界，不证明任意恶意生成代码无法读宿主文件；论文不能把二者混为一谈。

| Benchmark | 主指标与保留信息 | 与作者评分的关系 |
|---|---|---|
| RR | 原 signed weighted compliance；负权重、逐条标准、失败保留 | 固定作者 prompt/公式与明确 judge 身份 |
| DSQA | 每题 TP/FP/FN 计算 F1，再按任务取均值；另报 P/R、fully correct、额外答案率 | 适配作者 notebook v1 的 item-matching schema 和公式；prompt 非逐字复用、judge 非作者固定 Gemini 时不宣称官方等价 |
| DRBII | 每题 rubric 原生分数等于 1 的比例；-1 的 blocked rate 单报；维度与语言分项 | 适配固定作者三路评分、每批 50 条；prompt/重试/严格格式验证差异全部留在结果 provenance |

DSQA 的私有 `answer_type` 不进入 inference、split metadata 或任务 ID。当前作者 notebook 对 Single Answer 和 Set Answer 都通过 TP/FP/FN 聚合；与论文中单答案二值表述存在差别，按固定 notebook 的实际代码报告，严格 fully-correct 率另外列出。

DRBII 固定发布文件包含 9,415 条原始 rubric，非论文摘要中的 9,430；其中 6 条同维度重复文本。作者输出按维度和文本为键覆盖，因此最终评分分母对应 9,409 个唯一条目。实现保留所有原始判分反馈，但用同维度最后一次条目结果计算聚合，区分 `judged_item_count` 和 `criterion_count`，不能将 -1 当负权重直接求均值。

所有 judge 走现有 `MeteredModel`，不另起未计量 client。格式错误或调用失败不伪造零分；完整标志为 false，分数为 null。已落地的严格适配器每个 criterion/batch 只调用一次；任何增加的传输重试政策必须在正式执行身份中固定，并对所有 condition 一致，不重跑低分产物。

生成、规划与执行模型按注册配置固定 `deepseek-v4-flash-vision`，共同 judge 为 `claude-sonnet-4-5-20250929`，温度 0。代理 model ID 不等于已经核实服务端权重版本；每轮须记录实际 endpoint/model 配置和可取得的服务版本。该统一代理 judge 设置不是 DSQA 作者 Gemini 2.5 Flash 或 DRBII 作者 GPT-5.5 的 leaderboard 复现。

执行配置为仓库 `configs/benchmark_suite_v3.yaml`，正式 driver 在调用前验证注册 profile：meta/global/local/judge 输出上限均为 16,000，executor 为 8,192；每题最多 200 次模型尝试和 200 万账本 tokens，超时 180 秒/请求与 600 秒/执行。三角色、并行上限 2、每角色单次调用、一次局部规划、一个 JIT candidate、至多两次角色执行前的接口修复。v3 将 judge 上限从旧 v2 的 4,096 提高到 16,000，以适应 DSQA 大答案集和 DRBII 每批 50 条；同时固定 judge/生成/执行每请求一次尝试，不继承 v2 的可选传输重试。旧 v2 JSON 原样保存，生效值以 v3 profile 为准。

## 6. 正式执行顺序与已实现入口

1. 在 development 上完成实现调试，冻结代码、配置、作者数据 revision 与原始文件哈希。
2. 用 `scripts.prepare_benchmark_suite` 固化公开 manifest，核对各分区总数、同题组、跨 benchmark 同题与历史暴露记录。仅公开 metadata 进入 manifest。
3. 用 `scripts.run_benchmark_experiment --mode export-public` 导出 `PublicTask`，再用 `scripts.prepare_benchmark_evidence` 显式准备冻结证据。两者都不向 query planner 传入私有 evaluator 数据。
4. 用 `scripts.run_benchmark_experiment --mode evolve` 运行登记题序。checkpoint coordinator 负责逐源题日志、不可变快照、历史只读 val、固定全量 selector、缓存复用和选中状态封存。一次命令对应一个已登记 method/run，不自动启动整套矩阵。
5. 先注册全部 test condition/task/state/repeat 库存，完成所有答案提交并封存后，才统一评分。延迟评分入口不会在生成答案过程中调用 evaluator；RR winner 在 DRBII 上只读，不允许 target attribution。
6. 汇总全部运行、缺失和成本，再做已登记统计与盲审；不能因为某一比较不显著而调实现并继续沿用同一 test 声称确认性结果。

以下命令只启动无需 API 的软件 smoke：

```powershell
.venv\Scripts\python.exe -m scripts.run_benchmark_experiment --mode evolve --smoke --output outputs/checkpoint_smoke_v3
.venv\Scripts\python.exe -m scripts.run_benchmark_experiment --mode status --output outputs/checkpoint_smoke_v3
.venv\Scripts\python.exe -m scripts.run_benchmark_test --mode smoke --campaign outputs/test_release_smoke_v3
```

真实进化入口需要显式 `--benchmark`、`--data`、`--splits`、`--config`、`--evidence-dir`、`--run-id`、`--method`、`--output` 和 `--unsafe-local`。证据准备另需 `--allow-network`、公开 task 文件和环境变量中的凭据。不要把 API key 写入论文、配置样例或提交记录。

正式配置使用 `--config configs/benchmark_suite_v3.yaml`；三个已冻结 manifest 分别为 `paper/experiments/researchrubrics_splits_v3.json`、`deepsearchqa_splits_v3.json`、`deepresearch_bench_ii_splits_v3.json`。模型凭据取环境变量 `JIT_MAS_API_BASE` / `JIT_MAS_API_KEY` 与 `JIT_MAS_JUDGE_API_BASE` / `JIT_MAS_JUDGE_API_KEY`，不将密钥写入配置。RR 消融只更改注册的相应布尔开关，不能改模型或资源上限获得额外优势。

正式 test coordinator 是 `scripts.run_benchmark_test`，分为 `register`、`submit`、`seal`、`score`、`status` 五个阶段。`register --conditions <JSON路径> --campaign <目录>` 要求完整套件库存：RR 25 个 condition、DSQA 10 个、DRBII 7 个，共 42 个；每个条件明确 `condition_id`、`benchmark`、`method`、`run_id`、`snapshot_path`、`data`、`splits`、`config`、`evidence_dir`。进化状态还须绑定封存后的 `source_checkpoint`，不能拿任意经验库伪装成 val winner；静态条件使用 `snapshot_path: "initial"`。相对路径相对于 conditions 文件解析，凭据只从环境读取。

逐个执行 `--mode submit --condition <condition_id> --unsafe-local` 后，`--mode seal` 才能成功；任何缺失槽位都阻止提前评分。随后逐条件执行 `--mode score --condition <condition_id> --unsafe-local`，失败记录保留，不为刷分重新生成答案。上述命令均带同一 `--campaign <目录>`；score 仅评价已提交产物，但当前 native campaign 环境初始化仍要求显式的本机执行标记。完整库存在精确状态复用前为 23,247 个 test slots。离线 `--mode smoke` 只使用两个标记为合成软件测试的题目，验证真实提交/封存/评分状态机，不运行这 42 个正式条件。

## 7. 统计、成本与论文表格

各 benchmark 独立报告自己的指标，**不直接平均 RR compliance、DSQA F1 和 DRBII satisfaction 得出一个“综合分数”**。主差值是每题三个 selected states × 三次产物均值减去 initial 三次产物均值。RR/DSQA/DRBII 的任务数分别为 33/600/132；配对 bootstrap 重采样整题 10,000 次，seed 20261001，保留同题下所有 method/run/repeat 关联。重复调用和 rubric 条数不增加独立任务样本数。

三个 benchmark 的 selected-initial 对照逐项报告效应与区间；如声称“整体均有效”，对三项配对 sign-flip 检验做 Holm 校正，不只报告显著的数据集。每个 benchmark 的两个次要 baseline 对照（matched JIT、rubric-fixed）分别标识，并对该六项跨数据集家族做 Holm；RR 的三个消融加 G/GO 单列四项机制家族。逐题 sign-flip 为双侧 100,000 draws、seed 20261001，使用 `(1+extreme)/100001`。同时报告三个进化运行差异与各自选中位置。

Test 不完整时报告完成率、失败类别、分数及配对差值可行区间，不只平均成功题宣称优于 baseline。RR 保留预先固定十题的人工审计：selected run0/repeat0、initial repeat0、matched JIT repeat0 共 30 份匿名产物，两位独立评审及分歧裁决；没有声称已完成人类研究。

| 部分 | Evolution | Val | Test/机制 | 合计名义任务执行单元 |
|---|---:|---:|---:|---:|
| RR 全计划 | 包含于原 v2 细账 | 包含于原 v2 细账 | 六个核心条件、三个消融、G/GO | 6,195 |
| DSQA 六个核心条件 | 150×3=450 | 7×100×2×3=4,200 | selected 5,400 + terminal 5,400 + 四个静态条件 7,200 | 22,650 |
| DRBII 五个条件 | 0 | 0 | RR selected 1,188 + 四个静态条件 1,584 | 2,772 |
| **全部必做** | | | | **31,617** |

这里的单位不是 API 调用数，不保证每个执行成功；一个任务可能包含多次规划、生成、执行和 rubric 判分调用。精确状态复用降低实际成本；证据预处理、共享 G/GO rubric、任何登记重试与人工审计另计。报告 stage 分项 ledger，不重复累加序列化子运行。

封存后可使用 `python -m scripts.summarize_benchmark_suite --campaign <目录> --output <新报告.json>` 汇总。该命令不调用模型，使用注册槽位的固定分母，报告任务级配对 bootstrap、sign-flip 与 Holm 校正，保留未完成评价的可行区间。延迟评分只使用原任务生成后剩余的调用和 token 额度；产物复用不重置这份逻辑额度，共享 G/GO 准备成本另计。

建议论文主表并列三个 benchmark 的独立质量和置信区间，另表报告每 benchmark 的任务失败、实际 tokens、工具调用、总进化/val 成本与每题部署成本。附录给出固定 manifest、三个 checkpoint 曲线、完整缺失记录、语言/主题描述性分项，以及 prompt/judge 适配差异。

## 8. 一手资料与数据使用

- ResearchRubrics 数据：[固定 revision](https://huggingface.co/datasets/ScaleAI/researchrubrics/tree/85de3115053d1453ed612caacf4a405edc1ad756)；评分实现：[固定作者 commit](https://github.com/scaleapi/researchrubrics/tree/2dc80e2d4c38ddd80439517c259d93c6954b193f)。
- DeepSearchQA：[作者 README](https://huggingface.co/datasets/google/deepsearchqa/blob/b2623f8653065c2672de6d941fc5434cd652376c/README.md)、[作者评分 notebook](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code)、[技术报告](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)。Notebook v1 内容 SHA256 为 `aa1ace0a1e0023a5cad5e662968bcf18c6b4f5c0666cbad78a46ba5f813559b2`。
- DeepResearch Bench II：[固定作者仓库](https://github.com/imlrz/DeepResearch-Bench-II/tree/b38f360603db9531b102aef8c166cedb8509b6f6)、[评分调用](https://github.com/imlrz/DeepResearch-Bench-II/blob/b38f360603db9531b102aef8c166cedb8509b6f6/run_evaluation.py)、[聚合实现](https://github.com/imlrz/DeepResearch-Bench-II/blob/b38f360603db9531b102aef8c166cedb8509b6f6/aggregate_scores.py)。保留逐行 license 与来源；两条任务含非商业许可（idx 26/110），idx119 为 CC0，其余129条 CC BY 4.0。只提交公开 ID/哈希清单，不随论文仓库再分发参考报告或私有评价数据。

只有完整原生模型产物、账本和上述协议匹配时，才能往正式结果表填数；离线 fixtures、API 探针、历史 gated-update 结果和未封存 pilot 均不属于新主表。
