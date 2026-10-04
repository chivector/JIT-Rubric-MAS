# 2026-10-04 三方法开发诊断与显式 TEST 曝光记录

本文保留从 v0 到后续轮次的全部注册、固定库存、真实失败与成本。初期任务来自固定源 EVO，后续经用户明确授权扩展为台账中的少量 `exposed_TEST_development_only`；没有把曝光结果称为 formal TEST、干净确认或独立 judge 结果。各 campaign 在自身全部生成终态并封存后评分，中断和未启动注册分别记录。

历史章节保留当时的中途快照；其中“当前 / 等待”按本节所述版本与记录时点解读，后续同轮封存段补终态，旧 source hash / 测试成绩不能套到新版。以下最新统一版本表只用本版本结果，不跨轮择优。

曝光审计边界：本文台账中的 **5 个固定 selected TEST ID**仅表示正式选用的开发任务，**不代表整个开发过程总共只见过 5 个 ID 或其他历史 private evaluation 从未曝光**。v9 冻结后的历史评价搜索事件另见文末与 supplemental exposure ledger；已有一个额外 RR task_id 被确认可见，其余实际可见范围因原输出截断而为 `unknown`。未来干净确认必须排除已知 ID 并审计未知范围，不能据本轮固定选题范围作全局未曝光声明。

## 当前完整统一轮次：v20，三十槽全部封存、25槽完整评分

五个campaign先统一注册，固定30槽全部终结封存；**25完整评分、2生成失败、0项已记录评价异常类型、3评分未完成且无异常类型**。Direct 10/10完成评分、Initial-MAS 7/10完成评分、Native JIT 8/10完成评分。下表只用本轮全部固定来源，失败保持缺失值，旧轮较高成绩不替入。

| Benchmark 与输入 | 槽 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|---:|
| IFEval strict prompt | 6 | 1.000000 | 1.000000 | 1.000000 |
| IFBench loose prompt | 6 | 0.000000 | `null` | 0.500000 |
| DSQA F1 / shared evidence | 6 | 0.200000 | `null` | `null` |
| RR native weighted / shared evidence | 3 | 0.412500 | `null` | 0.387500 |
| DRB aggregate / shared evidence | 3 | 0.551724 | 0.500000 | 0.500000 |
| WritingBench mean / 10 | 6 | 8.8 | 8.3 | 8.7 |

六来源equal normalized failure-zero的**事后描述性macro**：Direct **0.5119798628**、Initial-MAS **0.4685185185**、Native JIT **0.5476313523**。本轮Initial-MAS宏均值仍未超过两个baseline。它不是新增正式主指标，不证明所有benchmark都更好；具体胜负、缺失与真实观察0见完整库存。Baseline失败影响failure-zero宏均值，不能据此宣称已评分任务质量全面提高。

IFEval / IFBench采用pinned author checker；其余四来源仍为同serving Exp模型诊断性judge，不是独立judge或official leaderboard。10道固定开发题包括已授权曝光的TEST题，temperature0.6单轮差值不能证明稳定提升或单项改动因果。相同上限下三方法实际算量不同。

已记录费用 **191模型调用 / 2,069,992tokens**，含生成 **118 / 1,810,825** 与评价 **73 / 259,167**；unknown usage slots=1；中断槽位的未知消耗不冒充0，上数是已持久化下界，美元费用`null`。新版实际唯一full suite **2,467 passed + 60 subtests**，12 changed Python source/test compile通过。独立synthetic预检及身份请求13调用 / 61,938tokens另计；源、证据、参数与checker身份绑定见v20附录。

## 历史统一轮次：v19，三十槽全部封存、29槽完整评分

五个campaign先统一注册，固定30槽全部终结封存；**29完整评分、1生成失败、0评价失败**。Direct 10/10完成、Initial-MAS 10/10完成、Native JIT 9/10完成。下表只用本轮全部固定来源，失败保持缺失值，旧轮较高成绩不替入。

| Benchmark 与输入 | 槽 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|---:|
| IFEval strict prompt | 6 | 1.000000 | 1.000000 | 0.500000 |
| IFBench loose prompt | 6 | 0.000000 | 0.500000 | 0.500000 |
| DSQA F1 / shared evidence | 6 | 0.466667 | 0.428571 | 0.833333 |
| RR native weighted / shared evidence | 3 | 0.450000 | 0.575000 | 0.450000 |
| DRB aggregate / shared evidence | 3 | 0.517241 | 0.431034 | 0.465517 |
| WritingBench mean / 10 | 6 | 8.7 | 8.1 | `null` |

六来源equal normalized failure-zero的**事后描述性macro**：Direct **0.5546392824**、Initial-MAS **0.6255243272**、Native JIT **0.5349074817**。本轮Initial-MAS宏均值高于两个baseline。它不是新增正式主指标，不证明所有benchmark都更好；具体胜负、缺失与真实观察0见完整库存。Baseline失败影响failure-zero宏均值，不能据此宣称已评分任务质量全面提高。

IFEval / IFBench采用pinned author checker；其余四来源仍为同serving Exp模型诊断性judge，不是独立judge或official leaderboard。10道固定开发题包括已授权曝光的TEST题，temperature0.6单轮差值不能证明稳定提升或单项改动因果。相同上限下三方法实际算量不同。

完整费用 **218模型调用 / 1,916,153tokens**，含生成 **117 / 1,580,546** 与评价 **101 / 335,607**；失败与组件调用均计费，美元费用`null`。新版实际唯一full suite **2,282 passed + 60 subtests / 452.21秒**，4 changed Python source/test compile通过。独立synthetic预检4调用 / 17,807tokens另计；源、证据、参数与checker身份绑定见v19附录。

## 历史统一轮次：v18，三十槽全部封存、26槽完整评分

五个campaign先统一注册，固定30槽全部终结封存；**26完整评分、4生成失败、0评价失败**。Direct 10/10完成、Initial-MAS 10/10完成、Native JIT 6/10完成。下表只用本轮全部固定来源，失败保持缺失值，旧轮较高成绩不替入。

| Benchmark 与输入 | 槽 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|---:|
| IFEval strict prompt | 6 | 1.000000 | 1.000000 | 1.000000 |
| IFBench loose prompt | 6 | 0.500000 | 1.000000 | `null` |
| DSQA F1 / shared evidence | 6 | 0.466667 | 0.466667 | `null` |
| RR native weighted / shared evidence | 3 | 0.312500 | 0.287500 | `null` |
| DRB aggregate / shared evidence | 3 | 0.517241 | 0.448276 | 0.517241 |
| WritingBench mean / 10 | 6 | 8.5 | 8.6 | 8.7 |

六来源equal normalized failure-zero的**事后描述性macro**：Direct **0.6129510826**、Initial-MAS **0.6827660459**、Native JIT **0.3954661558**。本轮Initial-MAS宏均值高于两个baseline。它不是新增正式主指标，不证明所有benchmark都更好；具体胜负、缺失与真实观察0见完整库存。Baseline失败影响failure-zero宏均值，不能据此宣称已评分任务质量全面提高。

IFEval / IFBench采用pinned author checker；其余四来源仍为同serving Exp模型诊断性judge，不是独立judge或official leaderboard。10道固定开发题包括已授权曝光的TEST题，temperature0.6单轮差值不能证明稳定提升或单项改动因果。相同上限下三方法实际算量不同。

完整费用 **394模型调用 / 5,397,201tokens**，含生成 **322 / 5,200,435** 与评价 **72 / 196,766**；失败与组件调用均计费，美元费用`null`。新版实际唯一full suite **2,215 passed + 60 subtests / 448.16秒**，187 source compile通过。独立synthetic预检7调用 / 74,505tokens另计；源、证据、参数与checker身份绑定见v18附录。

## 历史统一轮次：v17，三十槽全部封存、二十九槽完整评分

同source / config的30槽先统一注册，五个campaign已全部保存sealed summaries、controller exit0；**29完整评分、1生成失败（Native JIT WritingBench433）、0评价失败**。Direct与Initial-MAS均10/10完成，Native JIT为9/10。下表只用v17本轮全部固定来源，旧版本的较高行不替入本轮，失败保持`null`和完整成本。

| Benchmark 与输入 | generation slots | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|---|
| IFEval，exposed TEST / closed-book，strict prompt | 6 | 1.000000 | 0.500000 | 1.000000 |
| IFBench，exposed TEST / closed-book，loose prompt | 6 | 0.000000 | 1.000000 | 0.500000 |
| DSQA，source EVO / 两题shared evidence，F1 | 6 | 0.307692 | 0.466667 | 0.833333 |
| RR，source EVO / shared evidence，native weighted score（官方加权公式） | 3 | 0.425000 | 0.000000（实际评分） | 0.450000 |
| DRB，exposed TEST / shared evidence，聚合分 | 3 | 0.517241（30/58） | 0.482759（28/58） | 0.465517（27/58） |
| WritingBench，source EVO / closed-book，mean / 10 | 6 | 8.8 | 8.3 | `null`（normalized failure-zero 0.422222） |

六来源equal normalized failure-zero的**事后描述性macro**为Direct **0.5261194388**、Initial-MAS **0.5550506401**、Native JIT **0.6182408150**；不是新增正式主指标或clean TEST确认。Initial-MAS的整体指标高于Direct、低于Native JIT，本轮IFBench高于两个baseline；但IFEval低于两个baseline、DSQA低于Native JIT、RR实际0且低于两条baseline、DRB仍低于Direct、WritingBench完整均值低于Direct。不能只摘IFBench或恢复提交成功的部分声称总体超过两baseline。

RR raw 0是已提交已评分的观察值，按注册native bounds对应normalized **0.06976744186046512**，没有改成缺失值置零。DRB的initial保留来自运行前固定的公开结构guard，其他九个Initial-MAS任务选择revision；没有按Judge分数选择初稿或修改稿。工程eligible只证明初始执行终态合法，不证明语义或所有公开约束满足。公开actor计数还显示RR标题坍缩漏过v1阈值、IFEval文字频次退化未被v1覆盖，完整负向结果与诊断见本轮附录。

全账费用 **243模型调用 / 2,177,861tokens**，含生成 **142 / 1,903,482**、评价 **101 / 274,379**；unknown / estimated / queue idle为0、美元费用`null`。唯一generation失败的全部原始调用与计费保留。新版实际一次完整验证 **2,092 passed + 60 subtests、467.30秒**，185 Git source compile / diff check通过；同五注册source、模型、23份数据/证据/正式文件hash及14份checker依赖身份均匹配。

任务、manifest及config seed固定；生成temperature0.6存在随机性，源码与config两项变更共同存在，single-run差值不能直接当作稳定提升或单项修复因果。RR / DSQA / DRB / WritingBench仍使用同serving Exp模型作诊断性self-judge，非独立Judge或official leaderboard协议。example仅表示本轮已验证的开发候选，不能解释为胜出推荐；v17独立synthetic预检11调用 / 130,620tokens不计入以上30槽。

## 历史统一轮次：v16，三十槽全部封存、二十六槽完整评分

同 source / config 的30槽先统一注册，五个campaign已全部保存sealed summaries、controller exit0；**26完整评分、4生成失败（Initial-MAS3 / Native JIT1）、0评价失败**。该表只用v16全部固定来源和原始库存；v14、v15及更早结果保留在历史段，不将较高旧行拼入本版本。

| Benchmark 与输入 | generation slots | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|---|
| IFEval，exposed TEST / closed-book，strict prompt | 6 | 1.000000 | `null`（failure-zero 0.500000） | 0.500000 |
| IFBench，exposed TEST / closed-book，loose prompt | 6 | 0.000000 | `null`（failure-zero 0.500000） | `null`（failure-zero 0.500000） |
| DSQA，source EVO / 两题 shared evidence，F1 | 6 | 0.500000 | 0.500000 | 0.833333 |
| RR，source EVO / shared evidence，native weighted score（官方加权公式） | 3 | 0.450000 | 0.562500 | 0.437500 |
| DRB，exposed TEST / shared evidence，聚合分 | 3 | 0.500000（29/58） | 0.000000（0/58，实际评分） | 0.517241（30/58） |
| WritingBench，source EVO / closed-book，mean / 10 | 6 | 8.5 | `null`（normalized failure-zero 0.400000） | 8.5 |

六来源equal normalized failure-zero的**事后描述性macro**为Direct **0.5536175711**、Initial-MAS **0.4155038760**、Native JIT **0.6101087053**；不是新增正式主指标或clean TEST确认。Initial-MAS在该固定RR开发样本的0.5625高于两基线，WB335的8.2也高于两基线8.0，但其整体低于两条基线、完成率7/10、DRB已提交却实际得0，不能只摘成功行声称总体优势。任务、manifest及config seed固定一致；生成温度0.6仍有随机性，各轮single-run差值不能直接当作稳定提升或单项修复的因果证据。

RR的native weighted score仅沿用官方加权公式；当前criteria判断仍由同serving Exp模型作诊断性self-judge，不是official leaderboard评测协议或独立Judge。历史RR段中的同类分数亦按这一边界解读。

全部费用 **208模型调用 / 1,755,858tokens**，含生成 **107 / 1,461,504**、评价 **101 / 294,354**；unknown / estimated为0、美元费用`null`。新版全测 **2,048 passed + 60 subtests、468.10秒**，183 Git source compile / diff check通过；身份、完整三臂费用及失败阶段见文末v16封存总账。示例配置属于这一验证开发候选，不能解释为胜出配置推荐。

## 历史统一轮次：v14，三十槽全部封存、二十四槽完整评分

同 source / config 的30槽先统一注册；五个 campaign 已全部保存 sealed summaries、controller exit0，共30槽24完整评分、6生成失败。下表只用同一v14身份的全部来源，不以v9或未来v15的较高行替换，失败保留`null`与成本。

| Benchmark 与输入 | generation slots | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|---|
| IFEval，exposed TEST / closed-book，strict prompt | 6 | 1.000000 | 1.000000 | 1.000000 |
| IFBench，exposed TEST / closed-book，loose prompt | 6 | 0.000000 | `null`（failure-zero 0.500000） | 0.500000 |
| DSQA，source EVO / 两题 shared evidence，F1 | 6 | 0.500000 | `null`（failure-zero 0.500000） | 0.833333 |
| RR，source EVO / shared evidence，native weighted score（官方加权公式） | 3 | 0.312500 | `null`（failure-zero 0） | 0.425000 |
| DRB，exposed TEST / shared evidence，聚合分 | 3 | 0.500000 | 0.258621 | `null`（failure-zero 0） |
| WritingBench，source EVO / closed-book，mean / 10 | 6 | 8.8 | `null`（normalized failure-zero 0.388889） | `null`（normalized failure-zero 0.433333） |

`null`仍是未观测分数。DRB 的 Native JIT 失败不能解释为本方法实质击败它；Initial-MAS 的 DRB 分数也低于本轮 Direct，且从v9的25/58降至15/58。六来源equal normalized failure-zero的**事后描述性macro**为Direct **0.537855**、Initial-MAS **0.441252**、Native JIT **0.538630**，不是新增正式主指标或clean TEST确认。完整调用成本与失败阶段见文末；本轮不能声称Initial-MAS整体领先。

## 历史统一轮次：v9 六 benchmark，全三十槽已封存

以下为同一冻结 v9 版本五个 campaign 的完整注册范围，现已全部封存评分。单元格为完整原生均值；`null` 表示该来源有生成失败，括号中的 failure-zero 仅是 normalized 保守汇总，不能解释为观测零分。源码 / config / data / evidence 身份绑定原 registration，所有三十槽和失败费用保留。

| Benchmark 与输入 | 已注册 generation slots | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|---|
| IFEval，exposed TEST / closed-book，strict prompt | 6 | 0.500000 | 1.000000 | 1.000000 |
| IFBench，exposed TEST / closed-book，loose prompt | 6 | 0.000000 | `null`（failure-zero 0） | 0.500000 |
| DSQA，source EVO / 两题 shared evidence，F1 | 6 | 0.437500 | 0.466667 | 0.833333 |
| RR，source EVO / shared evidence，native weighted score（官方加权公式） | 3 | 0.200000 | 0.487500 | 0.387500 |
| DRB，exposed TEST / shared evidence，聚合分 | 3 | 0.396552 | 0.431034 | 0.465517 |
| WritingBench，source EVO / closed-book，mean / 10 | 6 | 8.8 | `null`（normalized failure-zero 0.444444） | 8.8 |

已注册共 **30 固定槽、6 benchmarks、5 个 campaign**，每个 campaign 独立注册、全生成封存后评分；主指标、失败 `null` 和全部成本一起填入，不跨输入版本择优。v8 的 39 槽已全部结束，其结果与全部更早历史保留如下。

## 历史两类指令轮次：v10，十二槽封存、十一槽完成评分

v10 仅覆盖此前已曝光的两题 IFEval 与两题 IFBench，三臂共 12 固定槽；11 槽完整评分、1 槽 Initial-MAS 生成失败保持 `null`。下表与 v9 分开，不拼成同版本六 benchmark 结果。

| 主指标 / 全固定库存 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| IFEval strict prompt，2题 | 1.000000 | 0.500000 | 0.500000 |
| IFBench loose prompt，2题 | 0.000000 | `null`（failure-zero 0.500000） | 1.000000 |
| 两来源 normalized failure-zero macro | 0.500000 | 0.500000 | 0.750000 |

数字约束任务的真实成功改善了一个 Initial-MAS 提交，未形成整体领先：另一个 IFBench 提交失败，IFEval 一个任务退步，宏平均与 Direct 持平、低于 Native JIT。详细 12 槽及全部成本见文末 v10 封存结果，代码验证仍按各轮实际报告。

历史 IFBench-only v12 的六槽已封存：Direct 两题主分 **0 / 0**，Initial-MAS **`null` / 1**，Native JIT **`null` / 0**。两题 normalized failure-zero 分别为 **0 / 0.5 / 0**，两臂原生完整均值仍为 `null`；v11 的两个 Initial-MAS 失败与全部历史保留，不把各轮较高行拼成同版全 benchmark 成绩。

初始 v0 背景（保留原始记录）：

本轮用于检验真实 API 下的执行可靠性、输出质量和方法细节，为后续开发迭代提供证据。任务仅来自固定源 EVO；没有运行正式 VAL/TEST，没有训练经验轨迹，也没有据此认定本方法优于两个基线。所有生成槽先终结并封存，随后统一评分。

## v0 注册与固定选题

- 轮次：`v0`，注册时间为北京时间 **2026-10-04 00:35:16**。
- 目录：`outputs/development_pilot_20261004_v0`。
- 注册哈希：`0fea0a710c7ae25c1b04ba0894b026e615c6c50ec3719ee9f84afd51f7cbe603`。
- 来源：`paper/experiments/joint_task_splits_v5.json` 的 `run_id=0` 固定源序列。
- manifest 内容哈希：`e4c2c6735b3625b01335f3942bc1ac256b34ea564eb777950157a347dc4261cd`。
- split 文件字节哈希：`2839518b037e81b7f6f8765bda40ce5f7d42ffa3b4fe2b04d76442efe0f15cce`。
- 选题规则：按原 run0 顺序，每个指定源取前 N 题。本轮 ResearchRubrics 取 1 题，WritingBench 取 2 题；没有根据输出、评分或人工偏好挑题。源行号为原文件中从 1 开始的数据记录号。

| 执行顺序 | Benchmark | 源行号 | 固定 task_id | 原正式分区 |
|---|---|---:|---|---|
| 1 | ResearchRubrics | 60 | `6847465956a0f6376a6054a7` | EVO |
| 2 | WritingBench | 335 | `writingbench:335` | EVO |
| 3 | WritingBench | 433 | `writingbench:433` | EVO |

每题依次执行 `direct`、`ours`、`native_jit`，共 **9 个生成槽**，每槽一次采样。只规范化选中的 EVO 原始行；JSONL 中未选记录不解析，评分参考与公开任务分别保存于协调器对象。数据完整文件仅以流式字节哈希核验版本。

ResearchRubrics 固定文件哈希为 `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`；WritingBench 为 `18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe`。

## 三个实际方法入口

| 方法 | 真实代码入口 | 本轮行为 |
|---|---|---|
| Single-agent / Direct | `scripts.mas_baseline_methods.run_direct(..., defer_evaluation=True)` | 一次模型调用，直接提交最终成品。 |
| Initial-MAS / ours | `MASPipeline.run_task(..., mode="evaluate", attribution=False, defer_evaluation=True)` | 当前完整任务条件规划、公开任务质量要求推断、初始 Agent Pool 的能力和技能、原生 JIT harness 生成、共享账本执行；提交后停止。 |
| Native JIT | `jit_mas.experiment_methods._native_jit` → `MetaReActAgent.run(generate_only=True)` → `load_harness` → `AgentRuntime.run` | 使用上游原生任务条件 harness 生成与运行接口；不替换成固定三角色团队。 |

Initial-MAS 从空 `ExperienceSnapshot` 开始，使用当前代码中的初始 Agent Pool seed，没有历史 EVO 经验或更新后的角色记忆。配置保留 `persistent_experience=True`、`local_attribution=True` 和 `evolving_agent_pool=True`；实际调用显式设为 `attribution=False`，不写入经验、不执行反思或归因更新。

现有延迟评分接口要求任务位于运行 manifest 的 `test` 字段。Runner 对已验证的源 EVO 任务建立局部、明确标记为 `source_EVO_development_only` 的接口映射；正式 manifest 的成员关系、VAL/TEST 集与 formal protocol 均不修改。

本轮 Native JIT 使用同一个 DeepSeek API 模型生成 harness。未使用原论文训练的 27B JIT 模型或原论文完整数据、工具与配置，因此该控制不构成 trained 27B JIT 的完整复现。

## 模型、共同输入与资源预算

执行与评分各角色均请求 `deepseek-v4-flash-vision`，使用用户提供并在 registration 冻结的兼容 API 服务地址；公开文档不发布个人临时 gateway，示例配置使用 credential-free placeholder。预检返回模型身份为 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`，配置通过 `expected_response_model` 固定并逐请求核验。`temperature=0`；`thinking` 与 `reasoning_effort` 使用 endpoint 默认值，没有单独禁用推理。

本轮是**闭卷开发诊断**，采用 `model_general_knowledge_allowed`。三个方法收到同一公开题目和同一附加约束：允许一般模型知识，无浏览或外部工具，不声称打开了不可访问附件、进行了检索或测量，不伪造不可验证来源，对重要事实不确定性作说明。原始公开任务哈希与附加约束后的 actor 任务哈希分别冻结，评分仍使用原始任务与对应私人评价记录。

闭卷输入与正式共享证据轨道存在明确差异，注册记录 `protocol_deviation_from_formal_shared_evidence=true`。缺少证据的任务应解释为闭卷条件下的输出表现；不能把该结果视为正式共享证据实验结论。三个方法的外部工具允许列表均为空。

| 资源项 | 所有方法的共同设置 |
|---|---|
| 最终执行角色输出上限 | 8,192 tokens |
| Meta / Global / Local 输出上限 | 每调用 16,000 tokens |
| Judge 输出上限 | 每调用 8,192 tokens |
| 整题 token 上限 | 2,000,000；生成与延迟评分共享剩余额度 |
| 整题活动时间 | 900 秒；生成与评分共享 |
| 单请求 timeout | 180 秒，并受整题剩余时间约束 |
| 模型调用 / 团队调用上限 | `null`；没有新增独立调用次数上限 |
| 最大 Agents / 并行 Agents | 3 / 2 |
| 进程请求并发上限 | 2 |
| 候选数 / 结构修复上限 | 1 / 2；不允许依据质量反馈重跑 |
| 上下文 | 131,072 tokens，保留 2,048-token margin；actor 为 `oldest_turns`，judge 为 `reject` |

900 秒定义为每槽生成账本的 `wall_seconds - request_queue_idle_seconds` 加评分阶段同样计算的活动时间。槽间调度等待、生成完成后等待整体封存的时间及请求队列空闲等待均不扣减该额度。评分账本从生成剩余时间开始，原生请求策略同时限制 judge 单次请求 timeout。生成耗尽时间时保留已提交答案，评分记为未完成，不重新调用 actor 或 judge。

Native JIT 的上游 `ctx.max_steps` 接口需要整数。共同调用上限为 `null` 时，接口整数由既有 2,000,000-token envelope 推导；实际资源守卫仍是共同 token 和时间账本，`independent_step_cap=false`。该兼容值与来源写入 harness 证据。

共同上限不代表实际耗费相等。分别报告各方法真实生成、评分 token/call 数与活动时间，provider 缺失用量时保留估计标记；无法确定的中断费用明确记为 unknown。推理 tokens 属于 provider completion 用量，不将可见回答长度冒充总推理开销。没有已知价格时美元费用保持 `null`。

## 封存、评分与恢复边界

九个注册生成槽必须全部提交或记失败，才能生成 `release/seal.json` 并开始任何评分。错误、超时和未产生答案的槽均保留，不补题、不删除、不通过反复采样追求更高分。

评分使用同一 DeepSeek Exp 模型和各 benchmark 当前评价器，是**同模型自评开发诊断**，不具有独立 judge 的证据强度。固定 shuffle seed 为 `20261004`，顺序在注册时通过 hash 排列冻结；judge 请求仅含回答、原始任务与私人评价材料，不含方法名、槽 ID 或方法成本。固定评分顺序为：

1. `ours:writingbench:433`
2. `native_jit:writingbench:433`
3. `direct:writingbench:433`
4. `direct:6847465956a0f6376a6054a7`
5. `direct:writingbench:335`
6. `native_jit:6847465956a0f6376a6054a7`
7. `ours:6847465956a0f6376a6054a7`
8. `ours:writingbench:335`
9. `native_jit:writingbench:335`

每次模型调用只有一次 transport attempt，SDK 不作隐藏重试。复用同轮终态结果不会产生额外 actor/judge 请求。发生中断时，只恢复已落盘并通过答案、执行与预算一致性验证的提交；无法恢复的生成槽记为失败，未确认完成的评分记为未完成及用量 unknown，不重新生成整题或重新抽取 judge 结果。

私人评价标准、参考答案和评价反馈不进入任何 actor 上下文，也不在本轮反馈给同一生成过程。封存并评分后，人可以在**已曝光 EVO 开发任务**范围内分析反馈；任何进一步代码或提示修改应另建 iteration 和 registration，并标明开发曝光，不能覆盖 v0 或称其为独立确认。

WritingBench 同时报告原生 checklist mean `[1,10]` 与归一化值 `(native_mean-1)/9`；ResearchRubrics 保留原生分数，并按每题冻结的理论上下界另外归一化。缺失真实分数保留 `null`。`normalized_mean_failure_zero` 仅用于保守的全槽计账，不能当作已观测的 judge 分数；只有该源全部槽完成时才给出 `native_mean_all_complete`。

## 可复现命令与证据身份

生产运行入口为 `scripts/run_development_pilot.py`。本轮配置文件为 `.runtime/development_pilot_20261004.config.json`，配置仅保存 `key_env=JIT_BENCHMARK_API_KEY`；密钥通过该环境变量提供，不写入命令、配置、日志或实验文档。

```powershell
.venv\Scripts\python.exe -m scripts.run_development_pilot --mode register --campaign outputs/development_pilot_20261004_v0 --iteration v0 --config .runtime/development_pilot_20261004.config.json --count researchrubrics=1 --count writingbench=2 --data researchrubrics=outputs/researchrubrics_live_20260930/processed_data.jsonl --data writingbench=dataset/writingbench/benchmark_all.jsonl

.venv\Scripts\python.exe -m scripts.run_development_pilot --mode run --campaign outputs/development_pilot_20261004_v0 --unsafe-local

.venv\Scripts\python.exe -m scripts.run_development_pilot --mode status --campaign outputs/development_pilot_20261004_v0

.venv\Scripts\python.exe -m scripts.run_development_pilot --mode summary --campaign outputs/development_pilot_20261004_v0
```

`register` 要求新的空目录；已经注册的本轮无需再次执行第一条命令。生成 Python 采用已有 `unsafe_local` 执行机制，不是操作系统沙箱。

v0 冻结 runtime fingerprint：`719b0b57d06f1d74bf5d1d6e8d8dab8d6b3deb7895eb5d9c8cfe6887900548ad`；runner 文件 SHA256：`07ca00e99ffd7e38d87a3891cc50f51273912b5c43a276ee4aeff1e93b80b7c0`。运行过程中逐槽核验代码、注册、配置、数据、split 与选中公开证据身份；身份变更要求另建开发轮次。凭据反射在模型响应进入 metered call trace 之前拒绝，失败调用仍计入成本。

离线功能校验已完成：runner 11 项及基线资源兼容 9 项，共 **20 项通过**。包括真实上游 Native JIT helper 的离线调用分支、全部生成封存后才评分、方法标签不进入 judge、失败成本保留、任务与评分中断不复采、共享活动时间、WB 原生与归一化指标分离以及凭据反射防落盘。随后完整离线回归为 **1,155 项通过，另有 60 项 subtests 通过，耗时 119.73 秒**；日志为 `.runtime/development_pilot_20261004.tests.log`。这些是软件检查，不能当作 benchmark 提分证据。

## v0 真实结果与全库存失败

v0 已终结全部 9 个生成槽并完成封存与延迟评分，`sealed=true`、`comparison_complete=false`。只有 Direct 的两项 WritingBench 提交成功并获得完整评分；另 7 个生成槽失败，均保留在全库存中。没有三方法均成功的配对任务，`complete_paired_task_ids=[]`。结果来源为 `outputs/development_pilot_20261004_v0/summary.json` 与对应封存记录，没有重采样、补题或覆盖失败。

| 方法 | 提交槽 / 3 | 完成评分 / 3 | RR 原生分数 | WB 原生均分 | 生成 tokens | 评分 tokens | 生成活动时间 / 秒 | 评分活动时间 / 秒 | 生成失败 / unknown |
|---|---|---|---|---|---|---|---|---|---|
| Direct | 2/3 | 2/3 | `null` | 8.9 | 18,925 | 14,374 | 53.298 | 19.798 | 1 / 0 |
| Initial-MAS | 0/3 | 0/3 | `null` | `null` | 217,083 | 0 | 278.860 | 0 | 3 / 0 |
| Native JIT | 0/3 | 0/3 | `null` | `null` | 94,011 | 0 | 143.327 | 0 | 3 / 0 |

| 固定任务 | 方法 | 生成 / 评分状态 | 原生分数 | 归一化分数 | 失败类型与观察 |
|---|---|---|---:|---:|---|
| RR60 | Direct | failed / submission_failed | `null` | `null` | `ValueError`：未返回非空、无工具请求的最终成品。 |
| RR60 | Initial-MAS | failed / submission_failed | `null` | `null` | `RuntimeError`：`APIConnectionError`，连接错误。 |
| RR60 | Native JIT | failed / submission_failed | `null` | `null` | `TypeError`：harness 响应解析收到 `NoneType`。 |
| WB335 | Direct | submitted / completed | 7.8 | 0.755556 | 无。 |
| WB335 | Initial-MAS | failed / submission_failed | `null` | `null` | `RuntimeError`：`APIConnectionError`，连接错误。 |
| WB335 | Native JIT | failed / submission_failed | `null` | `null` | `TypeError`：harness 响应解析收到 `NoneType`。 |
| WB433 | Direct | submitted / completed | 10.0 | 1.000000 | 无。 |
| WB433 | Initial-MAS | failed / submission_failed | `null` | `null` | `RuntimeError`：没有完整 JSON 对象，响应可能截断；`writer_1` 的 `AgentSpec.max_calls` 耗尽。 |
| WB433 | Native JIT | failed / submission_failed | `null` | `null` | `TypeError`：harness 响应解析收到 `NoneType`。 |

`writer_1` 的耗尽信息来自生成的具体 AgentSpec；共同整题与团队调用上限仍为 `null`。该失败提示需要进一步检查规划是否为角色分配了过小的局部资源，不能把它解释为共同整题 cap 已耗尽。

| 方法 | 生成调用 | 评分调用 | 全部 input tokens | 全部 output tokens | 估计用量调用 | unknown 槽 | 美元成本 |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct | 3 | 2 | 13,666 | 19,633 | 0 | 0 | `null` |
| Initial-MAS | 8 | 0 | 124,134 | 92,949 | 2 | 0 | `null` |
| Native JIT | 3 | 0 | 46,011 | 48,000 | 0 | 0 | `null` |

Initial-MAS 的两次连接失败费用以账本保守估计保留，不能视为 provider 返回的精确用量。三个方法该轮的 request queue idle 记录均为 0。全槽保守计账的 source macro normalized failure-zero 为 Direct **0.438889**、Initial-MAS **0**、Native JIT **0**；其中失败槽的真实评分仍为 `null`，这些零值不是 judge 观测分数。

v0 暴露的是可用性和接口可靠性问题。只有 Direct 完成的两项任务不能构成对 Initial-MAS 或 Native JIT 输出质量的完整配对比较，也不能据此宣称本方法优势。失败、耗费及现有输出均保留，后续先处理有证据的执行阻断。

## 合成 API 参数探测

参数探测仅使用合成短请求，不接触任何 benchmark 私人标准或参考答案。原始结构化结果位于 `.runtime/benchmark_api_probe_20261004.json` 和 `.runtime/benchmark_thinking_probe_20261004.json`。

基础连接和 JSON-object 合成请求均返回 HTTP 200，实际模型身份均为 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`。基础请求记录 prompt/completion 为 **88/30 tokens**，其中 reasoning tokens 为 **27**；JSON-object 请求为 **92/36 tokens**，其中 reasoning tokens 为 **29**，返回合法 JSON 对象。该证据说明服务可达且支持该短请求的结构化输出，不能保证长规划或长 harness 请求不会截断。

另一个独立合成句子任务要求比较 rolling deployment 与 blue-green deployment；`temperature=0`、`max_tokens=256`、`max_attempts=1`，仅比较下列两种请求参数：

| 参数方案 | 状态 | 正文字符数 | reasoning 字符数 | input / output tokens | 用时 / 秒 |
|---|---|---:|---:|---:|---:|
| `thinking={type:disabled}` | ok | 224 | 86 | 103 / 51 | 1.688 |
| 同上并加 `reasoning_effort=none` | ok | 277 | 0 | 24 / 50 | 0.984 |

该合成观察支持尝试两参数同时设置，以减少服务把有限输出额度消耗在 reasoning 上的风险；不证明其对 benchmark 质量一定有益，也不证明 v0 所有失败均由 thinking 导致。

## v1 开发轮次与解释边界

v1 注册时间为北京时间 **2026-10-04 00:40:53**，目录为 `outputs/development_pilot_20261004_v1`，注册哈希为 `1efadc1af27e4c889fb15718986a2402491ec7f7fa922dd70ae62fb59c2f22f4`。配置文件为 `.runtime/development_pilot_20261004_v1.config.json`，关联 v0 为已曝光的前一开发轮次。

v1 的生产代码 fingerprint 和 runner SHA256 与 v0 相同；任务、固定顺序、方法入口、闭卷政策与所有预算保持原注册设置。唯一参数调整是 **meta/global/local/exec/judge 五个角色统一设置 `thinking="disabled"` 与 `reasoning_effort="none"`**。这包括 judge，所有三种方法均采用同一参数政策，没有根据单方法结果选择性调整。v1 另注册新的九个槽，保留 v0 的原始封存结果；它不是 v0 失败槽的原地补跑，也不是未曝光任务的确认实验。

```powershell
.venv\Scripts\python.exe -m scripts.run_development_pilot --mode register --campaign outputs/development_pilot_20261004_v1 --iteration v1 --config .runtime/development_pilot_20261004_v1.config.json --previous-campaign outputs/development_pilot_20261004_v0 --count researchrubrics=1 --count writingbench=2 --data researchrubrics=outputs/researchrubrics_live_20260930/processed_data.jsonl --data writingbench=dataset/writingbench/benchmark_all.jsonl

.venv\Scripts\python.exe -m scripts.run_development_pilot --mode run --campaign outputs/development_pilot_20261004_v1 --unsafe-local
```

由于 actor **和 judge** 参数同时变化，v0 与 v1 的分数差不能单独归因于 actor 的生成质量改进，也不能当作固定 evaluator 条件下的方法提分。可以分别描述提交率、错误模式、资源耗费和同轮共同 judge 下的已完成配对结果；不跨轮把评分差解释为 actor 参数的因果收益。

## v1 真实结果与全库存失败

v1 已终结并封存全部 9 个生成槽，`sealed=true`、`comparison_complete=false`。Direct 的三题与 Initial-MAS 的 RR60 完成评分，其余 5 槽生成失败，没有三方法均成功的完整配对任务。以下数字来自 `outputs/development_pilot_20261004_v1/summary.json`；v0、v1 的全部终态独立保留。

| 方法 | 提交槽 / 3 | 完成评分 / 3 | RR 原生分数 | WB 原生均分 | 生成 tokens | 评分 tokens | 生成活动时间 / 秒 | 评分活动时间 / 秒 | 生成失败 / unknown |
|---|---|---|---|---|---|---|---|---|---|
| Direct | 3/3 | 3/3 | 0.6375 | 8.5 | 9,825 | 119,411 | 26.079 | 58.781 | 0 / 0 |
| Initial-MAS | 1/3 | 1/3 | 0.5375 | `null` | 134,521 | 89,244 | 82.204 | 52.828 | 2 / 0 |
| Native JIT | 0/3 | 0/3 | `null` | `null` | 73,171 | 0 | 70.187 | 0 | 3 / 0 |

| 固定任务 | 方法 | 生成 / 评分状态 | 原生分数 | 归一化分数 | 失败类型与观察 |
|---|---|---|---:|---:|---|
| RR60 | Direct | submitted / completed | 0.6375 | 0.662791 | 无。 |
| RR60 | Initial-MAS | submitted / completed | 0.5375 | 0.569767 | 无。 |
| RR60 | Native JIT | failed / submission_failed | `null` | `null` | `NameError`：生成代码使用了未定义的 `LogLevel`。 |
| WB335 | Direct | submitted / completed | 8.0 | 0.777778 | 无。 |
| WB335 | Initial-MAS | failed / submission_failed | `null` | `null` | `JSONDecodeError`：成品协议 JSON 缺少分隔符。 |
| WB335 | Native JIT | failed / submission_failed | `null` | `null` | `NameError`：生成代码使用了未定义的 `LogLevel`。 |
| WB433 | Direct | submitted / completed | 9.0 | 0.888889 | 无。 |
| WB433 | Initial-MAS | failed / submission_failed | `null` | `null` | `RuntimeError`：没有完整 JSON 对象，响应可能截断；`writer_main` 的 `AgentSpec.max_calls` 耗尽。 |
| WB433 | Native JIT | failed / submission_failed | `null` | `null` | `NameError`：生成代码使用了未定义的 `json`。 |

| 方法 | 生成调用 | 评分调用 | 全部 input tokens | 全部 output tokens | 估计用量调用 | unknown 槽 | 美元成本 |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct | 3 | 30 | 111,212 | 18,024 | 0 | 0 | `null` |
| Initial-MAS | 12 | 28 | 190,989 | 32,776 | 0 | 0 | `null` |
| Native JIT | 5 | 0 | 52,191 | 20,980 | 0 | 0 | `null` |

该轮完整 RR 评价含 28 个 rubric 级 judge 调用；Direct 另有两项 WritingBench 调用。request queue idle 记录均为 0。全槽保守 source macro normalized failure-zero 为 Direct **0.748062**、Initial-MAS **0.284884**、Native JIT **0**；失败真实分数仍为 `null`。

Native JIT 的静态导入问题和 Initial-MAS 的 JSON 协议问题作为独立工程诊断处理。尚无完整的三方法质量比较；当前唯一完成的 Direct/Initial-MAS 同轮配对 RR60 中，Initial-MAS 原生分数低 **0.1000**，不能宣称已经获得方法优势。

## RR60 配对差异与可泛化开发建议

这里只分析已曝光的源 EVO 输出与评分，不读取 VAL/TEST。Private criterion 原文、特定答案实体和数值不写入生产提示；生产代码未因本分析修改。

只读证据位置：

- Direct 封存回答：`outputs/development_pilot_20261004_v1/release/submissions/c296811038cdb53eae40312066a9662d44c04bd645c604f27e1490372f9f0cc5.json`。
- Initial-MAS 封存回答：`outputs/development_pilot_20261004_v1/release/submissions/93963a2b3e373d3e1c12fb2778e29d3f2d390ea9fa66a0d1115ffaee02119192.json`。
- 两个回答对应的 EVO 评价位于 `release/evaluations` 中同名文件。
- Initial-MAS 过程：`generation/93963a2b3e373d3e1c12fb2778e29d3f2d390ea9fa66a0d1115ffaee02119192/c184f4be30d4ee4714db71a6ed155663e7df0c8757313fd10892c8ca975103ff/execution.json`，该路径以 v1 目录为根；公共质量规划见同目录 `frozen_plan.json`。

两份评价只有 4 项 verdict 发生变化：Initial-MAS 在完整引言覆盖上获益，其余差异涉及来源身份与具体主张的对应，以及量化长期情景的出处和假设。加权净差为 -8，正权重分母为 80，得到原生差 **-0.1000**。该观察不是对那些被 judge 接受的来源或数字进行外部真实性核验；Direct 的回忆型引文与宏观估计是否确实受到指定文献支持仍未验证。

1. **保留主张到来源的对应，避免合成时只剩作者姓名。** Analyst 的 `model_output` 事件 `e4` 中，`ledger.outline` 含三组作者、年份、出版物完整题名和来源类型；writer 实际收到的 `sub_runs[writer_1].trajectory[0].model_input_messages` 中三组完整题名均存在，而最终 `answer` 中均消失。`source_references`、`evidence_spans`、`evidence_ids` 为空，相关信息只埋在长 outline 内。该轮来源信息没有被上下文交接删除，损失发生在最终合成。通用候选改进是保留可定位的来源身份与其支持的主张，逐处标注是否已核验，把“未联网核验”和“没有任何可识别出版物”分开；缺乏可靠身份的来源不得补造。本轮观察不能证明这些模型回忆的出版物身份正确，也不能保证恢复题名即可得到有效来源分。

2. **公开任务要求数据和长期分析时，分开覆盖实测结果与有条件的预测证据。** 公共规划明确提出研究数据、专家分析与长期后果，但 analyst 的长期部分和 writer 成品都主要给出定性理论，缺少可定位研究支持的量化情景、融资或适用条件及不确定性口径。这项缺口在上游已出现，不能仅靠把 outline 传给 writer 解决。通用候选改进是规划时分别列出“实测结论”和“模型或情景结论”的证据需求、条件和缺口；已有可靠材料才填写量化预测，没有材料就保留缺口，不能为追逐评分发明估计、出处或区间。共享证据轨道可以解决本轮闭卷限制，应在相同公开材料下同时评估三个方法。

3. **终稿检查要定位成品证据，不能只依据贡献者自报 completed。** Analyst 的 checkpoints 全部报 completed，但仅一轮 analyst 加一轮 writer 即终结，规划未设置 reviewer。若公共任务要求比较研究或方案，终稿应保留共同设计条件、衡量口径及主要数量的来源和适用条件；已有信息应形成能直接比较的成品结构。该例的设计资料在 analyst outline 和最终各案例正文中均存在，但最终汇总表主要比较结果，设计口径仍散落于正文；数量的总额、人数、支付频率与周期之间也缺少清晰口径说明。通用候选改进是令最终检查逐项引用成品段落或表格，并检查关键数值与口径的内部一致性，必要时由既有团队预算内的检查角色指出具体缺失再修订；不把 private criterion 或任务专名写成跨题硬编码规则。

以上为三个需要新轮次验证的通用候选，没有将不完整的自评结果转化为已证实的质量增益。

## v2 注册与组合改动

v2 注册时间为北京时间 **2026-10-04 00:53:36**，目录为 `outputs/development_pilot_20261004_v2`；注册哈希为 `8b6f83823cd3619b459ae6c782ca2cef3790ca7ba0081b066b4465c2e42bc766`。与 v1 的配置对象和选题完全一致：五角色继续统一 `thinking="disabled"`、`reasoning_effort="none"`，actor、judge 的模型、预算及闭卷知识政策相同。该轮明确关联已曝光的 v1，没有写入新经验或替换固定任务。

v2 runtime fingerprint 为 `fd63c076b4c4fdbd9efd555a47002d16462bf926b0adb3c5b28dc767b52d1595`；runner SHA256 仍为 `07ca00e99ffd7e38d87a3891cc50f51273912b5c43a276ee4aeff1e93b80b7c0`。本轮登记后冻结生产代码。

针对 v1 的实际阻断及 RR60 的过程证据，v2 组合验证以下改动：

- 共用 JIT 生成器补充必要的 imports，并使用 `symtable` 检查生成代码的静态 name 绑定，旨在执行前发现未定义符号；原生 JIT 与 Initial-MAS 都通过共用生成器受益。
- Global/Local 规划及严格 MAS 执行协议使用 JSON-object 响应格式，旨在减少长成品协议的 JSON 解析失败。
- 最终交付检查保留 public brief 的语言、长度、受众或 persona 与事实要求，检查成品而非仅凭 completed 自报；它不能替代真实来源或事实核验。
- 通用 final submission gate 要求保留来源的题名、作者、日期及其支持的结论、单位和假设，并注明 remembered provenance；区分实测研究与有条件预测。该改动依据已存在于 analyst 和 writer 输入、最终合成却遗漏的出版信息，不采用 private criterion 文本、固定任务专名或答案数值。

v2 已终结全部生成槽并完成封存评分，完整结果见下节。模型 self-judge 未改变，不构成独立事实核验；同三题多轮曝光及多个改动一起生效，使该轮只能支持开发诊断，不能把任一分差直接归因为其中单一改动、形成未曝光测试优势或认定正式 benchmark 提分。v0、v1、v2 的全部记录均保留。

## v2 真实结果与全库存失败

v2 的 9 槽全部终结并封存，7 槽成功提交且完整评分，`sealed=true`、`comparison_complete=false`。Direct 和 Initial-MAS 均为 3/3；Native JIT 只完成 RR60，两项 WritingBench 未提交最终成品。唯一三方法完整配对题为 RR60。数字来源为 `outputs/development_pilot_20261004_v2/summary.json`，失败内容另核验对应 `generation/*/failure_summary.json`；没有重采样或替换失败。

| 方法 | 提交槽 / 3 | 完成评分 / 3 | RR 原生分数 | WB 原生均分 | 生成 tokens | 评分 tokens | 生成活动时间 / 秒 | 评分活动时间 / 秒 | 生成失败 / unknown |
|---|---|---|---|---|---|---|---|---|---|
| Direct | 3/3 | 3/3 | 0.7125 | 8.6 | 9,706 | 112,709 | 24.907 | 54.860 | 0 / 0 |
| Initial-MAS | 3/3 | 3/3 | 0.4625 | 8.2 | 187,612 | 78,196 | 108.673 | 50.453 | 0 / 0 |
| Native JIT | 1/3 | 1/3 | 0.5625 | `null` | 96,288 | 95,814 | 102.952 | 47.110 | 2 / 0 |

| 固定任务 | 方法 | 生成 / 评分状态 | 原生分数 | 归一化分数 | 失败类型与观察 |
|---|---|---|---:|---:|---|
| RR60 | Direct | submitted / completed | 0.7125 | 0.732558 | 无。 |
| RR60 | Initial-MAS | submitted / completed | 0.4625 | 0.500000 | 无。 |
| RR60 | Native JIT | submitted / completed | 0.5625 | 0.593023 | 无。 |
| WB335 | Direct | submitted / completed | 8.2 | 0.800000 | 无。 |
| WB335 | Initial-MAS | submitted / completed | 7.6 | 0.733333 | 无。 |
| WB335 | Native JIT | failed / submission_failed | `null` | `null` | `RuntimeError`：`Native JIT did not submit a final artifact`。 |
| WB433 | Direct | submitted / completed | 9.0 | 0.888889 | 无。 |
| WB433 | Initial-MAS | submitted / completed | 8.8 | 0.866667 | 无。 |
| WB433 | Native JIT | failed / submission_failed | `null` | `null` | `RuntimeError`：`Native JIT did not submit a final artifact`。 |

| 方法 | 生成调用 | 评分调用 | 全部 input tokens | 全部 output tokens | 估计用量调用 | unknown 槽 | 美元成本 |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct | 3 | 30 | 104,857 | 17,558 | 0 | 0 | `null` |
| Initial-MAS | 16 | 30 | 225,350 | 40,458 | 0 | 0 | `null` |
| Native JIT | 12 | 28 | 152,793 | 39,309 | 0 | 0 | `null` |

该轮 request queue idle 记录均为 0，provider 用量均有明确记录。全槽保守 source macro normalized failure-zero 为 Direct **0.788501**、Initial-MAS **0.650000**、Native JIT **0.296512**；Native JIT 的失败真实分数仍为 `null`，其 WritingBench 原生均分不报告。

本轮 Initial-MAS 的提交可靠性达到 3/3，仍未取得质量优势：三题原生分数均低于同轮 Direct，RR60 还低于 Native JIT。RR60 的 Initial-MAS 与 Direct 差为 **-0.2500**，与 Native JIT 差为 **-0.1000**；两项 WritingBench 平均较 Direct 低 **0.4**。Initial-MAS 生成耗费也高于 Direct。静态名字检查和结构化协议的组合变更没有充分解决 Native JIT 的成品交付问题；已有结果不支持宣布方法胜出。后续应以公开任务与过程证据定位缺口，在共同材料和固定 evaluator 条件下另注册验证，继续保留失败和实际耗费。

## v2 WB433 已完成评分的公共质量检查

WB433 的两份封存文案均为**中文**，Direct 原生均分 **9.0**，Initial-MAS 为 **8.8**。公开 query 使用英文并要求 Xiaohongshu 风格，但没有明确的“用英语输出”句子；不能把新增的语言检查提示当成成品已改为英文的证据，也不能以此题声称英语输出问题已经解决。

五个评价维度中，Initial-MAS 的一个月改善描写为 8，其余四项为 9；Direct 五项均为 9。评价对该处的建议是更具体地呈现变化。该 0.2 分差不要求或授权补造检测数据；当前文本已具备生活场景、体感、吸收过程和成分介绍，改进应来自公开 brief 和真实成品检查。

本题公开资料给出的使用形态是清洁后均匀涂抹、等待后**洗掉**，而两个回答都写成薄膜布贴片、揭下后吸收余液。两个回答还加入了公开资料不支持的特定技术或因果功效：Direct 写微囊缓释及深层吸收；Initial-MAS 写植物萃取能最大保留活性并改善吸收，两者均将电脑辐射与皮肤问题或防护效果建立明确关联。这些内容获得较高 self-judge 分数，也没有变成有依据的产品事实。

最多两个可泛化候选改进：

1. **先确认产品形态、用法和技术资料，再写体验。** 把涂抹、冲洗及后续护理写成连贯流程，避免把其他常见产品的膜材或留置用法迁移进来。成分与功能可依照给定介绍；不存在的专利、载体、吸收深度、防辐射功效或实测效能不补造。若技术参数没有提供，可自然介绍已给出的植物提取物和配方取舍，不将“未说明工艺”扩写成更强的黑科技事实。
2. **在同一生活情境中写具体的主观前后变化和过程节奏。** 公开要求个人经历、一个月时程、办公室场景和朋友式语气。候选写法应将现有的紧绷、妆面卡粉或肤感变化放进同一可辨认场景，交代使用习惯以及涂抹、等待、冲洗后的感受，少用笼统的“明显变好”。没有真实记录时，不新增改善百分比、皮肤仪读数、精确吸收率、所谓实验结论或虚构使用日记；主观脚本也不能被包装成作者真实完成的试验。

证据为 v2 `release/submissions/930d18d6024efa4c9897c81004fab6c4e949b906e3eca02558e587398b39d44f.json`（Initial-MAS）及 `release/submissions/4a93709310f472fb353e194aeb25f31033f12e9fcd39a2f7b6f3bf1dfbf3c2b6.json`（Direct），对应 `release/evaluations` 的同名文件，以及注册中 WB433 的公开 `task.question`。该检查仅针对已封存的曝光源 EVO，生产代码和 v2 终态均不修改，不在同一配置下重采样追求更高分。

## DSQA v2 固定源 EVO 的实际结果

`dsqa_v2` 注册于北京时间 **2026-10-04 01:02:50**，目录为 `outputs/development_pilot_20261004_dsqa_v2`，注册哈希为 `9eac091d9c6e080ead8c485080c56ef7173e906309e744bc5a68bdcf6b7a9595`。沿用 v2 冻结代码、五角色参数与共同预算，关联前轮 v2；输入仍为闭卷一般知识。按原 run0 源 EVO 顺序固定取前两题，为 DSQA 源行 **129**、**170**。数据文件为 `dataset/deepsearchqa/DSQA-full.csv`，SHA256 为 `25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396`。

该轮 6 槽均终结并封存，5 槽成功提交且完整评分；`sealed=true`、`comparison_complete=false`。评分使用已说明的同模型适配 F1 evaluator，保留 private reference 隔离；没有因低分追加回答或重评。

| 固定任务 | 方法 | 生成 / 评分状态 | 原生 F1 | 失败 |
|---|---|---|---:|---|
| DSQA129 | Direct | submitted / completed | 0.461538 | 无。 |
| DSQA129 | Initial-MAS | submitted / completed | 0.285714 | 无。 |
| DSQA129 | Native JIT | submitted / completed | 0.625000 | 无。 |
| DSQA170 | Direct | submitted / completed | 0.800000 | 无。 |
| DSQA170 | Initial-MAS | submitted / completed | 0.500000 | 无。 |
| DSQA170 | Native JIT | failed / submission_failed | `null` | `RuntimeError`：未提交最终成品。 |

| 方法 | 提交与完成评分 / 2 | 完整原生均分 | 全槽 failure-zero | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---:|---:|---|---|---|---|
| Direct | 2/2 | 0.630769 | 0.630769 | 2 / 2 | 2,037 / 2,519 | 6.813 / 3.281 | 2,928 / 1,628 |
| Initial-MAS | 2/2 | 0.392857 | 0.392857 | 10 / 2 | 101,201 / 1,816 | 56.203 / 2.875 | 87,824 / 15,193 |
| Native JIT | 1/2 | `null` | 0.312500 | 6 / 1 | 48,456 / 642 | 53.812 / 1.563 | 33,047 / 16,051 |

各方法 estimated attempts、unknown slots 与 queue idle 均为 0，美元费用均为 `null`。DSQA129 是唯一三方法完整配对题，完整 task_id 为 `deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c`；DSQA170 为 `deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c`。数字来自该目录 `summary.json` 与失败证据。该闭卷开发轮中 Initial-MAS 两题均低于 Direct，已完成的三方配对也低于 Native JIT；不能宣称提分。正式共享证据能力和跨任务泛化还未由该轮验证。

## 显式 TEST 曝光开发入口

用户于 **2026-10-04** 明确允许参考测试题指导提示和方法。依据该授权，runner 新增显式 `--partition exposed_test`；默认仍为 `source_evo`，没有任何隐式回退或根据评分改选题。这个模式属于 `exposed_TEST_development_only`，`formal_protocol_result=false`，不修改不可变 v5 manifest，也不把曝光结果称为正式 TEST 或干净确认实验。注册保存 original partition、污染清单、用户授权来源及 exposure policy；后续涉及这些任务的开发结果都须保留曝光标记。

选题政策为 **`first_N_per_source_frozen_task_index_TEST_order`**：遍历 v5 `task_index` 原始顺序，仅筛选指定源的 `partition=test`，同时核验该源完整 TEST 集合与 `memberships[source].test` 一致；首批每源最多 2 题。该顺序没有按任务难度、检查内容或输出质量调整。此次开发准备已只读下列四条的公开 prompt 与 checker 约束，未解析整份 TEST 内容或参考答案；这四题从准备阶段起已曝光，不抹除已有检查。

| 源 | 固定 task_id | 源行 | 曝光内容 | 评分主指标 |
|---|---|---:|---|---|
| IFEval | `ifeval:1203` | 37 | 公开问题与词频约束 | 官方 pinned prompt strict accuracy |
| IFEval | `ifeval:1246` | 47 | 公开问题与 postscript 约束 | 官方 pinned prompt strict accuracy |
| IFBench | `ifbench:13` | 14 | 公开问题与句内词位约束 | 官方 pinned prompt loose accuracy |
| IFBench | `ifbench:22` | 23 | 公开问题与数字、不同连词计数约束 | 官方 pinned prompt loose accuracy |

数据路径为 `dataset/ifeval/ifeval_input_data.jsonl` 和 `dataset/ifbench/IFBench_test.jsonl`，字节 SHA256 分别为 `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`、`d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`，均与 v5 一致。注册另外冻结选中原始公开任务投影的 SHA256；checker IDs 和 kwargs 保持协调器私有，不进入 actor 公开投影。

IFEval 使用 `outputs/independent_v5_preparation/checkers/Google-IFEval-e49bbfe381c9c0e564b937f1c4e163a2273c65cc` 的官方严格 checker；IFBench 使用同目录下 `IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d` 的官方 loose checker。注册冻结源代码、依赖和资源身份。**每个 prompt 的全部指令通过才计主分 1，否则为 0**；`instruction_accuracy`、`pass_count`、`check_count` 另报，不能替换主指标。不调用模型 judge，不基于 checker 反馈修复或重采当前封存答案；全部三臂生成终态后才评分，失败与实际成本规则保持一致。

示例注册命令如下；配置只写环境变量名 `JIT_BENCHMARK_API_KEY`。

```powershell
.venv\Scripts\python.exe -m scripts.run_development_pilot --mode register --campaign outputs/development_pilot_20261004_exposed_if_v0 --iteration exposed_if_v0 --partition exposed_test --config .runtime/development_pilot_20261004_v2.config.json --count ifeval=2 --count ifbench=2 --data ifeval=dataset/ifeval/ifeval_input_data.jsonl --data ifbench=dataset/ifbench/IFBench_test.jsonl --checker-source ifeval=outputs/independent_v5_preparation/checkers/Google-IFEval-e49bbfe381c9c0e564b937f1c4e163a2273c65cc --checker-source ifbench=outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d

.venv\Scripts\python.exe -m scripts.run_development_pilot --mode run --campaign outputs/development_pilot_20261004_exposed_if_v0 --unsafe-local
```

新 runner 离线测试 **14 项通过**；真实本地 pinned 数据与 checker 的临时注册预检通过，四题共 12 槽，网络请求为 0。测试验证默认 EVO 拒绝 TEST、显式曝光固定顺序与 membership 核验、污染记录与公私隔离、三臂封存后才 checker、prompt 主分 0 与 instruction accuracy 0.5 的分离、零模型评分调用、恢复不重采及 checker 变更拒绝。这里未填入尚未运行的新 benchmark 分数。

## 两篇原论文支持的公开成品审查与约束修订候选

[Self-Refine: Iterative Refinement with Self-Feedback（Madaan 等，2023）](https://arxiv.org/html/2303.17651v2) 的 §2 将原输入与实际输出交给同一模型生成反馈，再将输入、实际稿与反馈交给模型修订；反馈需要指出具体位置和可执行修改，§4 的消融支持这种反馈比笼统“改进答案”更有用。可操作适配为：critic 只看原始 public task（含共同公开 source packet）与实际完整稿，逐项检查公开要求，给出“缺陷位置—公开依据—具体修改”，同时列出应保留的已满足要求；revision 必须返回完整 artifact，只修改解决真实缺陷所需的内容，避免改掉已满足的语言、数量、位置、格式或结尾约束，并在修改后重新检查全部公开要求。不给 critic evaluation、私有参考答案或评分权重。原文也观察到错误反馈导致失败；同模型自审不是独立事实核验。固定一轮 review 加一轮 revision、最小必要编辑和约束保留是本项目待验证的工程适配，原论文没有保证这些措施对当前模型或题目必然提分；调用、tokens 和失败均须计入实际预算，保持三臂成品全部终态后再评分。

[Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining（Akinfaderin 等，2025）](https://arxiv.org/html/2511.01807v1) 的 §3 方法比先写大纲更具体：先生成逐词编号的草稿，再保持词数重排为连贯正文，最后抽取干净成品。可借鉴其长度控制思路：先从公开指令构造简短交付结构，明确长度单位和计算范围，为必需内容分配配额并预留必要尾段，再检查实际 decoded artifact；计数或结构草稿不进入最终交付，未执行计数时不宣称精确核验。公开词位、句数、关键词或尾段要求应在修订时保留，不能为了润色整体重写而破坏；跨约束最小编辑与段落配额仍是工程推断，不是该论文直接验证的组合方法。§6 与 Appendix B 显示效果随模型、长度和任务变化，较长输出计数更不稳定，创作任务没有一致获益；主要实验是英文单文档摘要，质量评估也使用 LLM judge。因此不把论文收益外推为中文约1500字、任意多重约束或当前 benchmark 已获证实的优势，也不把逐词编号机械推广到所有写作任务。

## exposed v3 指令任务：原始封存结果与成本

`exposed_v3` 注册于北京时间 **2026-10-04 01:17:38**，目录为 `outputs/development_pilot_20261004_exposed_v3`，注册哈希为 `1201b11a5a0f63e312691781ab576ee84c7df268368701fe8ffe4ae9a4bd78ff`。选题仍为上述固定四条 TEST 曝光任务，顺序与污染标记不变。runtime fingerprint 为 `5eae8840c1c49b9fb9c313dbace556c5bcfe6e1090b37fdf6a45837c23d1f68d`，runner SHA256 为 `d4743693820d0cf7cf7cc6e8689c51e5026d6cc0636c4a46815390116a8d4675`。配置启用 `public_refinement=true`，Initial-MAS 在同一生成账本内加入固定的一次公开审查与一次完整修订；revision 是唯一提交稿，不依据评分在草稿与修订稿之间择优。

原始 12 槽全部终结并封存，9 槽提交成功、3 槽生成失败。原始评价仅有四项 IFEval 完整完成，五项有效 IFBench 提交均因 checker `TypeError` 未完成评分；`comparison_complete=false`。IFEval 的 Direct 主指标为 **1/2**，Initial-MAS 为 **2/2**，Native JIT 两槽生成失败。该两题开发观察不能推出整体 benchmark 优势，也不能用较高的逐指令比例代替 prompt 主指标。

| 方法 | 提交 / 4 | 原始完成评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 原始评分活动秒 | 全部 input / output tokens |
|---|---|---|---:|---:|---:|---:|---|
| Direct | 4/4 | 2/4 | 4 | 2,498 | 12.233 | 1.047 | 901 / 1,597 |
| Initial-MAS | 4/4 | 2/4 | 26 | 184,947 | 125.781 | 1.030 | 160,595 / 24,352 |
| Native JIT | 1/4 | 0/4 | 12 | 97,258 | 96.968 | 0.234 | 71,506 / 25,752 |

三方法的评分模型调用和评分 tokens 均为 0；指令评估使用本地官方 checker。estimated attempts、unknown slots、queue idle 均为 0，美元费用为 `null`。共同预算相同，Initial-MAS 的实际生成耗费明显高于 Direct；公开 review/revision 的调用已计入前述生成账本，不另设免费预算。原始全槽 failure-zero source macro 为 Direct **0.25**、Initial-MAS **0.50**、Native JIT **0**，其中 IFBench 的真实原始分数均为 `null`，这些会计零值不是“checker 判错”。原 `summary.json`、封存 inventory 和全部失败均保留。

## IFBench TypeError 的适配修正与一次事后诊断

只读定位发现，pinned IFBench 的 strict 实现会移除数据 schema 中不适用于当前指令的 `None` 参数，而其 loose 实现直接把整条 kwargs 传给各指令的 `build_description`；数据中的空字段因此触发签名 `TypeError`。旧包装器照原始 schema 透传，导致五项提交全部 evaluator-incomplete。这是 checker 集成错误，不是模型违反了全部指令。

修正仅在 IFBench 适配边界删除值为 `None` 的 padding，保留每个非空参数、原 private record 与原记录的随机 seed 输入；非空未知参数仍报错，不靠抛弃约束取得通过。Google IFEval strict 路径不变。新的 IFBench checker identity 登记该输入投影政策；没有修改 pinned 作者代码或评分主指标。修正在 v3 完成后落地，不能重写 v3 的注册身份或原始未完成评价。

首先在 `.runtime` 准备候选模块与 patch，真实官方 checker 的六项合成回归通过。随后另写一次 `.runtime/exposed_v3_corrected_checker_analysis.json`：仅检查九个已封存 final 与四个 Initial-MAS draft，共 **13 次本地 checker 调用**，其 evaluator 活动时间合计 **6.946 秒**，**0 API、0 actor 重新生成**；这项事后检查耗费单列，不改原 v3 成本账本。分析前后核验原注册、inventory、seal、submissions、evaluations、summary 与当时生产 checker 文件的哈希一致；原 v3 记录未被修改。下表的 IFBench 数字属于**修正评分器后的事后开发诊断**，原始评价依旧是 `null`。

| 固定任务 | 方法 | 原始状态 / 主分 | 一次修正诊断的 prompt 主分 | 通过指令 / 总指令 | 修正诊断 schema error |
|---|---|---|---:|---|---|
| IFEval1203 | Direct | completed / 0 | 0 | 1/2 | 无。 |
| IFEval1203 | Initial-MAS | completed / 1 | 1 | 2/2 | 无。 |
| IFEval1203 | Native JIT | 生成失败 / `null` | `null` | `null` | 不适用；生成 `RuntimeError`。 |
| IFEval1246 | Direct | completed / 1 | 1 | 1/1 | 无。 |
| IFEval1246 | Initial-MAS | completed / 1 | 1 | 1/1 | 无。 |
| IFEval1246 | Native JIT | 生成失败 / `null` | `null` | `null` | 不适用；生成 `RuntimeError`。 |
| IFBench13 | Direct | incomplete / `null` | 0 | 0/1 | 无。 |
| IFBench13 | Initial-MAS | incomplete / `null` | 0 | 0/1 | 无。 |
| IFBench13 | Native JIT | 生成失败 / `null` | `null` | `null` | 不适用；生成 `AttributeError`。 |
| IFBench22 | Direct | incomplete / `null` | 1 | 2/2 | 无。 |
| IFBench22 | Initial-MAS | incomplete / `null` | 0 | 1/2 | 无。 |
| IFBench22 | Native JIT | incomplete / `null` | 1 | 2/2 | 无。 |

修正诊断的 IFBench Direct 主指标为 **1/2**、Initial-MAS 为 **0/2**；Native JIT 的唯一有效提交通过，另一槽生成失败，因此其完整源均分仍为 `null`，全槽 failure-zero 为 0.5。这些区别不能被合并成“所有失败都得 0 分”。有效封存答案的 evaluator schema error 均已消失，但 Initial-MAS 在 IFBench 没有取得优势；结合 IFEval/IFBench，两源保守 macro 为 Direct 0.5、Initial-MAS 0.5，仅是已曝光四题的开发计账。

## 固定 draft 与 final 的事后约束检查

四份 Initial-MAS 草稿都来自各 run 的 `execution_draft.json`，通过 `public_refinement.json` 的 draft hash、revision hash 和最终封存 answer hash 核验。final 检查复用上节那次结果，各 draft 仅额外检查一次。没有选更高分的版本、回改提交、重新调用 critic/revision 或因评分改写答案。

| 固定任务 | draft prompt pass | sealed final prompt pass | draft 通过 / 总数 | final 通过 / 总数 |
|---|---|---|---|---|
| IFEval1203 | 否 | 是 | 1/2 | 2/2 |
| IFEval1246 | 是 | 是 | 1/1 | 1/1 |
| IFBench13 | 否 | 否 | 0/1 | 0/1 |
| IFBench22 | 否 | 否 | 1/2 | 1/2 |

这项事后消融诊断显示一题的公开约束在 final 中得到修复，三题的通过状态未改变；不能把一次已曝光任务的变化当作独立因果实验，不能认定同模型自审一定提升质量。指令主分来自确定性的官方 checker，不是 LLM judge；公开 critic 仍与 actor 同模型，也不是事实独立核验者。今后的机械约束改进应重新注册，并保留各版本失败和额外成本。

## RR frozen evidence v3：共享材料后的执行失败

`rr_evidence_v3` 注册于北京时间 **2026-10-04 01:18:32**，目录为 `outputs/development_pilot_20261004_rr_evidence_v3`；注册哈希为 `267005628b6403f86f9b3f428048dafd8ac70350af249658587e3387e7aef572`，代码身份与上述 exposed v3 相同。固定源 EVO 仍仅 RR60，三个方法使用相同冻结公开材料与共同预算，`knowledge_mode=frozen_evidence`，没有引入 private rubric/reference。材料目录为 `outputs/development_rr_evidence_20261004`，注册中的唯一 evidence 文件 SHA256 为 `865710a85821e802ce665d4f9a30b2ff9b1f7e740587737b08de5835fc8594f3`。该输入消除了本轮闭卷相对于共享材料的偏离，但仍属于曝光源 EVO 开发，不是正式 TEST。

全部三槽终结并封存，只有 Direct 完成提交与同模型评分，原生分为 **0.4125**，归一化为 **0.453488**。Initial-MAS 因贡献者 ledger 出现未允许字段及 evidence span/source-ref 结构不合约而失败；Native JIT 因 schema 字符串与预期 mapping 接口不一致产生 `'str' object has no attribute 'items'`。Initial-MAS 与 Native JIT 的真实分数均为 `null`，没有完整配对任务，原 `summary.json` 不覆盖。

| 方法 | 提交 / 评分完成 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---|---|---|---|
| Direct | 1 / 1 | 1 / 28 | 5,889 / 78,257 | 9.125 / 57.016 | 71,785 / 12,361 |
| Initial-MAS | 0 / 0 | 7 / 0 | 137,175 / 0 | 95.594 / 0 | 105,164 / 32,011 |
| Native JIT | 0 / 0 | 1 / 0 | 23,312 / 0 | 18.187 / 0 | 17,867 / 5,445 |

estimated attempts、unknown slots、queue idle 均为 0，美元费用均为 `null`。失败已经耗费的调用和 tokens 全部报告；资源未耗尽不代表协议执行成功。RR 仍使用同一 Exp 模型的 private evaluator，封存后才评分，不能当作独立 judge。由于两个方法未交付，且公开输入由闭卷变为共享材料，此轮不提供本方法数值提升或相对于 Direct/JIT 的质量优势证据，跨轮分差也不能归因为单一提示或 review/revision。

## v3 后的注册与计分守卫

两个 v3 campaign 均结束后，生产适配才加入上述 IFBench 空字段修正；旧记录及单独事后分析全部保留。Runner 另将配置路径中的各角色 endpoint 限定为无 userinfo、query、fragment 的 HTTPS URL，拒绝 `candidates>1`，并在 summary 中拒绝为失败生成槽标记 complete 的评价记录。这些守卫防止凭据进入注册或不实的 observed score，不改变任何既有原始分数。

该次相关回归只运行一次，四个测试文件共 **43 项通过，耗时 5.77 秒**；包括官方 checker 三类合成签名回归、非空约束拒绝丢弃、主分与逐指令比例分离、URL/candidate 注册拒绝，以及失败生成槽不能伪装为已观测零分。没有重跑全套测试或调用 API。更早的 v3 冻结前完整离线检查为 **1,216 tests 与 60 subtests 通过，耗时 131.96 秒**，日志为 `.runtime/development_pilot_20261004_v3_full_tests.log`；该完整检查发生于本节适配修正之前，不代表之后的新完整回归。

## exposed v4：中断、未封存与未知耗费

`exposed_v4` 于北京时间 **2026-10-04 01:30:39** 注册，目录为 `outputs/development_pilot_20261004_exposed_v4`，注册哈希为 `adc9a2244411c380469000625382166d119163b057c45149a3a2d096a2ab89ca`。仍注册上述固定四题、三方法共 12 槽，保留 `exposed_TEST_development_only` 与全部污染记录。该轮于北京时间 **01:37:47** 主动中断；`interruption.json` 记录的原因是已确认的生成 parser 类型错误：默认 `repair_json` 返回 JSON 字符串，而使用路径仅处理 dict/list 分支，导致工具动作未被执行。该判断来自程序与合成检查，不是依据题目评分选择中断。

中断时 **3 槽已开始、2 槽已有 final submission**；运行中的槽为 Native JIT / IFEval1203。`sealed=false`、`evaluations_invoked=false`、`comparison_complete=false`。该轮**已观测评分数为 0**，所有分数均未观测，不能填成 0 分、按成功的两个稿计算优势，或与后轮混合计算成完整实验。全部原始文件保留；旧槽不恢复、不在原注册下重采，后续代码变体使用新注册。

| 已开始的槽 | 生成状态 | 已记录生成调用 | 已记录生成 tokens | 已记录生成活动秒 | 评分 |
|---|---|---:|---:|---:|---|
| Direct / IFEval1203 | final 已保存，未封存 | 1 | 858 | 4.797 | 未调用，`null`。 |
| Initial-MAS / IFEval1203 | final 已保存，未封存 | 8 | 107,434 | 56.203 | 未调用，`null`。 |
| Native JIT / IFEval1203 | 运行中被中断 | unknown | unknown | unknown | 未调用，`null`。 |

两个已终结生成槽合计为 **9 调用、108,292 tokens、61.000 活动秒**，仅是有记录部分。Native JIT 当时的活动调用与用量尚在进程内存中，不能由磁盘文件恢复完整账单；`interruption.json` 明确保存 unknown，不能伪造零耗费。其余 9 个注册槽尚未开始；本轮没有完整成本总额或完整成对分数。中断记录、未封存提交与未知项均不删除。

## exposed v5：全库存结果与实际成本

`exposed_v5` 于北京时间 **2026-10-04 01:43:36** 注册，目录为 `outputs/development_pilot_20261004_exposed_v5`，注册哈希为 `f5555da90076a6e7efe3d4b405c1def772965b3e31a5055a441ed5d2a4264dab`。它使用新的代码身份，runtime fingerprint 为 `30c0935e797e02afcad3f63fae8ad6e8dca30aeaa2489938374b88f12fc837f2`，runner SHA256 为 `2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`；不是恢复 v4 的未封存运行。固定 task_index 顺序、四题与 TEST 曝光政策不变，输入仍为共同闭卷任务。各方法保持共同 2,000,000 tokens / 900 活动秒预算及已登记角色输出上限，实际耗费不同。

12 槽全部生成终结后封存，再统一调用 pinned author checker；**10 槽有效提交并完整评分、2 槽生成失败**，`sealed=true`、`comparison_complete=false`。IFEval 主指标为 prompt strict accuracy，IFBench 为 prompt loose accuracy；下表同时列出次级指令计数，但不以其替换 prompt 主分。没有生成后按 checker 结果修补、重采或选择 draft/final 中较高分者。

| 固定任务 | 方法 | 生成 / 评分状态 | prompt 主分 | 通过指令 / 总数 | 失败 |
|---|---|---|---:|---|---|
| IFEval1203 | Direct | submitted / completed | 1 | 2/2 | 无。 |
| IFEval1203 | Initial-MAS | failed / submission_failed | `null` | `null` | `ValueError`：重复 JSON 对象键，未交付有效提交。 |
| IFEval1203 | Native JIT | submitted / completed | 1 | 2/2 | 无。 |
| IFEval1246 | Direct | submitted / completed | 1 | 1/1 | 无。 |
| IFEval1246 | Initial-MAS | submitted / completed | 1 | 1/1 | 无。 |
| IFEval1246 | Native JIT | failed / submission_failed | `null` | `null` | `KeyError`：生成执行失败。 |
| IFBench13 | Direct | submitted / completed | 0 | 0/1 | 无生成或评分错误；checker 未通过。 |
| IFBench13 | Initial-MAS | submitted / completed | 0 | 0/1 | 无生成或评分错误；checker 未通过。 |
| IFBench13 | Native JIT | submitted / completed | 0 | 0/1 | 无生成或评分错误；checker 未通过。 |
| IFBench22 | Direct | submitted / completed | 0 | 1/2 | 无生成或评分错误；全部指令未同时通过。 |
| IFBench22 | Initial-MAS | submitted / completed | 1 | 2/2 | 无。 |
| IFBench22 | Native JIT | submitted / completed | 1 | 2/2 | 无。 |

IFEval 的 Direct 为 **2/2**；Initial-MAS 与 Native JIT 均是**一题有效且通过、另一题生成失败**，完整源原生均分为 `null`，不能以 1/1 的有效稿表现代替全源成绩。其 failure-zero 为 0.5，仅是将缺失保守计零的账目。IFBench 两题的六槽均完成本地评分，Initial-MAS 和 Native JIT 各为 **1/2**，Direct 为 **0/2**；三方法完整配对题仅 IFBench13、IFBench22。

| 方法 | 提交 / 完整评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 本地评分活动秒 | 全部 input / output tokens | request queue idle 秒 |
|---|---|---:|---:|---:|---:|---|---:|
| Direct | 4 / 4 | 4 | 3,084 | 13.625 | 1.048 | 901 / 2,183 | 0 |
| Initial-MAS | 3 / 3 | 25 | 231,824 | 123.078 | 0.860 | 199,659 / 32,165 | 0.016 |
| Native JIT | 3 / 3 | 11 | 89,218 | 88.016 | 0.750 | 64,065 / 25,153 | 0 |

三方法评价模型调用及评价 tokens 均为 **0**；评分由官方 pinned checker 在本地执行，未使用模型 judge。全部生成合计 **40 调用、324,126 tokens、224.719 活动秒**，本地评分合计 **2.658 秒**。失败槽已消耗的调用与 tokens 均在上述成本中；各方法 estimated attempts、unknown slots 均为 0，美元费用均为 `null`，不把未知价格写为免费。公开 review/revision 仍与 actor 同一模型并在生成账本内计费，不能当作独立事实审查。

两源全槽保守 macro normalized failure-zero 三方法均为 **0.5**。本轮在 IFBench 的已曝光两题上，Initial-MAS 与 Native JIT 多通过一题；IFEval 则只有 Direct 两题均成功提交并通过，Initial-MAS 仍存在 JSON 协议失败且实际生成成本较高。因此不宣称总体质量、可靠性或成本优势，也不将不同代码版本和中断轮的变化归因为单个组件。数字来自原始 `summary.json`、封存 inventory 与失败记录，原失败真实分数保持 `null`；四题均已污染，不是正式 TEST、独立确认或显著性证据。

## DSQA129：ACT 2023 官方全州共享证据准备

仅针对已使用的固定源 EVO DSQA129，另准备 `outputs/development_dsqa129_act2023_evidence_20261004`。准备过程只读取该题已登记的公开任务投影和 ACT 官方原始资料，未读取 DSQA 参考答案、未按题目条件筛选行、未计算答案成员，也未扩大到 DSQA170。这里记录证据准备，不报告尚未发生的 evidence campaign 分数。

主来源为 [ACT《Average ACT Scores by State — Graduating Class of 2023》](https://www.act.org/content/dam/act/unsecured/documents/2023-Average-ACT-Scores-by-State.pdf) 的两页完整表，辅助来源为 [ACT《2023 National Graduating Class Profile Report》](https://www.act.org/content/dam/act/unsecured/documents/2023-National-ACT-Profile-Report.pdf) 的队列及 benchmark 定义页（报告印刷页 3–4，PDF 页 5–6）。州表请求重定向到 ACT 同域 `secured/documents` 版本，两份下载均为 HTTP 200。来源保留原 PDF 字节、URL、publisher/cohort、页码与文件 hash；没有依赖后来的 2024 表替代 2023 数据。

原表 **52 行**完整转录，包含 **50 州、District of Columbia、National**，保持出版者的原行序和原显示精度。列序为 geography、estimated graduates tested %、average composite score、English readiness %、**Reading readiness %、Math readiness %**、Science readiness %，Reading 在 Math 前，不能按题目提及顺序错换这两列。Composite 使用 ACT score points；参与率分母为预计毕业生总人数，学科达标率分母为报告中的已测试毕业生队列。原表注释将毕业生总量预测来源标为 WICHE 2020 年 12 月版本；English / Reading / Math / Science 的 benchmark score cutoffs 分别为 18 / 22 / 22 / 23，与达标学生比例的百分数单位不同。队列和单位说明随同完整表进入共享材料，不把 DC 或 National 当作州。

| 冻结材料 | SHA256 |
|---|---|
| 官方州表 PDF | `59af7cdd36ed8eea4cc4f5aa1860845b49cfe5b29be6298f2f2310f269a65a2b` |
| 官方全国报告 PDF | `6d0a8da18138ec90ada3f0825006c68461d75aa2e47d4008945ec15f232f2f83` |
| 完整原值 CSV | `ad7b1071f6333c48211c5eaff64e248b52f1850b5f2a86a5a124b6b00edc7a7e` |
| 主 evidence pack 文件 | `4fa09242670d8f1e3eb344390e9f5cb841e0bac51c55c68d300a4c903d0c66f3` |
| Evidence manifest | `8adc50cdd4297015477e5fcb2686b2f8382fa78dbb97ec87c7c58b72a2593d82` |

`prepare_pack.py` 使用隔离安装于 `.runtime` 的 `pypdf==5.9.0`，逐页解析并验证原表学科列序、52 个唯一 geography、50 州完整覆盖与显示数值范围；两处跨行地理名称完整还原。保留原始数字字符串及 PDF 页/行 provenance，生成 `act2023_full_state_table.csv`、`state_table_extraction.json`、冻结 source cache、共享 pack 与 manifest。公开源定义另以摘要随包提供，所有方法将收到同份完整表，不提供筛选后的候选答案。

共享 pack 为 **1,709 cl100k tokens**，两来源完整、未截断；canonical public task identity、`load_evidence_pack`、manifest 与 `load_evidence_tasks` / `apply_evidence_pack` 一致性检查通过。离线重放验证 **7 个冻结 artifact 的 SHA256 均未变化**。准备脚本没有网络或模型请求；后续 actor 读取该材料产生的输入 tokens 应由新 campaign 正常计账，不能当作免费输入。

公开检索实际使用 3 次 web tool invocation，含 2 个搜索 query、2 份官方文件 open 和 2 页官方 PDF screenshot；两次 PDF 下载分别为 **160,474 / 828,292 bytes**，耗时 **2.194 / 1.869 秒**，来源失败为 0。准备中的 post-build verifier 曾两次错误假设 renderer 字段为 `url/text` 而触发 `KeyError`；两份未验证初始 pack 原样归档，改用 `locator/segments` 后才生成通过验证的主 pack，未隐瞒失败或覆盖注册中的材料。

`offline_preparation_audit.json` 合并这三个本地 build attempt：确定性 local planner callable 3 次、缓存 replay 工具 9 次、本地 ledger estimated tokens **5,862**、ledger 活动秒 **0.345**；这些不是远程模型调用或计费 tokens。成功准备尝试的整体本地脚本耗时为 **0.316 秒**，与 ledger 活动时间属于不同测量范围，不能相加为总墙钟时间。**Actor / judge / remote planner API 为 0**；网页检索墙钟总时间、重定向 hop 数及人工整理时间没有完整记录，美元费用为 `null`。下载、验证失败、成功 build、离线重放与本地 ledger 语义分别保留在 `download_log.json`、`preparation_report.json` 和 `offline_preparation_audit.json`，未改生产依赖、正式 evidence pack 或 immutable v5 manifest。

## 未开始的三个 v5 注册：生成前替代

在采用新的 v6 代码/配置注册前，以下三目录虽然已有 registration 和 slot inventory，但没有开始任何生成。核验时每个目录**仅存在 `registration.json` 和 `release/inventory.json`**，没有 generation 目录、started marker、提交、评价或 runtime database；inventory 的 task/method slots 与原注册一致。确认这些前置条件后，才分别写入 `superseded_before_generation.json`，保留原 registration / inventory 文件 SHA256 及新注册引用。

| 原注册目录 | 原 registration hash | 注册槽 / 已开始槽 | 替代注册 |
|---|---|---|---|
| `outputs/development_pilot_20261004_v5` | `4d706b5263e8889bbe9f0b195f6c1f1880251736d96b88d69831706187a8d3e5` | 15 / 0 | `source_v6` |
| `outputs/development_pilot_20261004_dsqa_act_v5` | `5ff7cd176e05d8e876a1d1274b50de175aef2b9635ceb79cb313010d42b9aadb` | 3 / 0 | `dsqa_act_v6` |
| `outputs/development_pilot_20261004_drb_v5` | `a811825f4b7b1b0b57b8ad64d0ea67453548690da92e08add33a6deb4bf830a0` | 3 / 0 | `drb_v6` |

三目录共 **21 个未开始槽、0 generation attempts、0 API、0 评分观察**，不作为 21 项失败或 0 分结果，不恢复这些原注册。所有原始注册、inventory 和外部日志保留；标记前后原文件 hash 均不变。这里的 `v5` 指上述未生成目录，与已经运行且封存的 `exposed_v5` 不同，也不改变其原始分数与成本。

## v6 已登记身份、曝光台账与冻结验证

v6 配置登记 `public_refinement_response_format=json_schema` 和 `public_positional_construction=true`；这些是新变体的配置事实，是否提高有效交付和质量仍需看封存后的全库存结果。下列四个新注册均已存在，本节只记录身份，不填入运行中的评分或尚未完成的任务结果。

| 新注册目录 | 注册时间（北京时间 2026-10-04） | registration hash | 范围 |
|---|---|---|---|
| `outputs/development_pilot_20261004_exposed_v6` | 02:00:55 | `f98271c912344292fb483403b0fab3b1163312417dcc23d04d5994670a78862a` | 原四条 IFEval / IFBench 曝光 TEST，12 槽。 |
| `outputs/development_pilot_20261004_v6` | 02:01:24 | `95dbb2ab16a8eb529bca232c9a860d097e9730ceb61e204bfca5d48e2bd1bcd3` | 固定源 EVO：RR1 / WB2 / DSQA2，15 槽。 |
| `outputs/development_pilot_20261004_dsqa_act_v6` | 02:01:24 | `9e5203ad62965012a9b74acb6eb32cea5e787ab8ab1ee3a781f1112ef5d2900d` | 固定源 EVO DSQA129，同份 ACT 完整表，3 槽。 |
| `outputs/development_pilot_20261004_drb_v6` | 02:01:24 | `0e0c8f6b41918b55c367a63b46809578ff067c7ea20898b2cb494ea5b19f7e8f` | 原 DeepResearch Bench II 第 1 条曝光 TEST，3 槽。 |

`paper/experiments/test_exposure_20261004.json` 仍保留原有 **5 个 task_id**，仅追加 `exposed_v6` 和 `drb_v6` 的 registration refs。每个新引用均核验 task_id、benchmark、source row、原 partition 和 question hash 与既有台账一致；未新增 ID、读取新题或改写既有曝光记录。源 EVO 注册不混入 TEST 台账；immutable formal v5 manifest 不改。

v6 最终完整离线检查为 **1,329 tests 与 60 subtests 通过，耗时 154.39 秒**；compileall 与 diffcheck 均 exit 0。较早 v5 检查中的一项过时测试断言已在测试侧修正，这里记录的是修正后最终完整验证结果，不沿用较早检查计数。该验证不调用模型 API，也不证明 benchmark 数值优势；真实结果须待各轮全部生成终态、封存及评分后单独登记。

另提供 `configs/development_pilot.example.json` 与同名 `.md` 说明，保留代表性五角色结构和共同预算，使用 `https://api.example.com/v1`、`YOUR_MODEL_ALIAS`、`expected_response_model=null`；凭据仅通过 `JIT_BENCHMARK_API_KEY` 环境变量导入。模板会随开发配置演进，不是上述 v6 注册快照。模板没有本地数据路径或实际凭据，用户必须设置提供方 endpoint、模型身份及兼容选项，再注册自己的最终配置，不能把模板视为本轮完整复现身份。

## exposed v6：封存结果、失败位置与全成本

上述 `exposed_v6` 原注册的实际运行已结束，进程 exit 0，`summary.json` 已保存。四题三方法共 **12 槽全部终结并封存**，其中 **10 槽有效提交且完整评分、2 槽生成失败**；`sealed=true`、`comparison_complete=false`。仍先封存三臂再使用相同 pinned author checker：IFEval 报 prompt strict accuracy，IFBench 报 prompt loose accuracy，评分模型调用与评分 tokens 为 0。下表保留原始失败 `null` 与次级指令计数，不覆盖任何旧轮记录。

| 固定任务 | 方法 | 生成 / 评分状态 | prompt 主分 | 通过指令 / 总数 | 失败 |
|---|---|---|---:|---|---|
| IFEval1203 | Direct | submitted / completed | 0 | 1/2 | checker 未通过全部指令。 |
| IFEval1203 | Initial-MAS | submitted / completed | 1 | 2/2 | 无。 |
| IFEval1203 | Native JIT | submitted / completed | 1 | 2/2 | 无。 |
| IFEval1246 | Direct | submitted / completed | 1 | 1/1 | 无。 |
| IFEval1246 | Initial-MAS | submitted / completed | 1 | 1/1 | 无。 |
| IFEval1246 | Native JIT | submitted / completed | 1 | 1/1 | 无。 |
| IFBench13 | Direct | submitted / completed | 0 | 0/1 | checker 未通过。 |
| IFBench13 | Initial-MAS | failed / submission_failed | `null` | `null` | `RuntimeError`：初始执行未交付有效稿。 |
| IFBench13 | Native JIT | failed / submission_failed | `null` | `null` | `AttributeError`：生成代码使用不存在的 logging enum。 |
| IFBench22 | Direct | submitted / completed | 0 | 1/2 | 全部指令未同时通过。 |
| IFBench22 | Initial-MAS | submitted / completed | 1 | 2/2 | 无。 |
| IFBench22 | Native JIT | submitted / completed | 1 | 2/2 | 无。 |

IFEval 原生均分为 Initial-MAS **1.0（2/2）**、Native JIT **1.0（2/2）**、Direct **0.5（1/2）**。IFBench 的 Initial-MAS 与 Native JIT 均是一题有效且通过、一题生成失败，完整源原生均分仍为 `null`，failure-zero 为 0.5；Direct 两题完整评分均未通过，原生均分和 failure-zero 均为 0。完整三方法配对任务为 IFEval1203、IFEval1246、IFBench22，IFBench13 不计入完整配对质量均分。

| 方法 | 提交 / 完整评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 本地评分活动秒 | 全部 input / output tokens |
|---|---|---:|---:|---:|---:|---|
| Direct | 4 / 4 | 4 | 2,347 | 10.468 | 1.062 | 901 / 1,446 |
| Initial-MAS | 3 / 3 | 25 | 253,375 | 145.578 | 0.781 | 212,180 / 41,195 |
| Native JIT | 3 / 3 | 12 | 104,541 | 96.485 | 0.749 | 76,290 / 28,251 |

全部生成合计 **41 调用、360,263 tokens、252.531 活动秒**，本地评分合计 **2.592 秒**。三方法 request queue idle、estimated attempts、unknown slots 均为 0，美元费用均为 `null`。Initial-MAS 的 IFBench13 失败槽已耗费 **5 调用、62,531 tokens、61.047 秒**；Native JIT 对应失败槽已耗费 **4 调用、40,520 tokens、37.938 秒**，均完整包含于总成本，没有因为未评分删去失败账单。

Initial-MAS 的 IFBench13 失败发生在 **public refinement 之前的初始 actor 执行**：原 `failure.json` 所保存的执行事件仅有两次 model output，预算中两次对应 `output_tokens=8192`、`request.finish_reason=length`。两稿出现大量重复散文，输出已达每次执行上限；没有 `execution_draft.json` 或 `public_refinement.json`，故此槽没有进入后续 `json_schema` 审查/修订分支。不能把此失败解释为 schema 修订已运行但无效，也不能删除两次已有耗费后重新生成。证据为 generation 目录 `c6c539f6824ea16878d967def2775a6d28b4028cef12396ea679656974904c32` 的 `failure_summary.json`，及其 run 子目录的 `failure.json` / `budget.json`。

Native JIT 的 IFBench13 原始 `failure.json` 确认为 `type object 'LogLevel' has no attribute 'WARNING'`。所选生成 `action.py` 第 127 行在 draft 验证未通过后的分支调用 `ctx.logger.log(..., level=LogLevel.WARNING)`，但实际 `scripts/kernel/monitoring.py` 的枚举只有 OFF、ERROR、INFO、DEBUG。因此可确认的是生成代码引用了不支持的 runtime logging API；该异常遮蔽了验证失败后的处理，不能据此认定未提交稿本来能够通过 checker。其生成 harness 身份与调用、失败记录保留在 generation 目录 `e50c309839916bcf312c0e5354c1c763022698fd0875433c58fbd22c5811f8ce/native`。该只读定位没有改 runtime、生成文件或封存记录。

两源全槽 conservative macro failure-zero 为 Initial-MAS **0.75**、Native JIT **0.75**、Direct **0.25**。这是已曝光四题的一轮开发观察：Initial-MAS 与 Native JIT 的质量会计分相同，Initial-MAS 实际调用和 tokens 更多；Direct 的提交可靠性仍为 4/4，另两方法各为 3/4。结果支持继续定位重复输出和 runtime API 兼容缺口，不支持宣布 Initial-MAS 已优于 Native JIT，或把这四题的差值当作整体 benchmark / 干净 TEST 优势。跨 v5/v6 同时存在代码、组织提示、结构化输出与公开约束施工的变化，不能将分差归因为单一组件。公开审查仍使用 actor 同模型，虽本节分数来自官方 checker，也不能将该审查称为独立事实 judge。

## DSQA ACT v6：共同官方证据后的实际结果

上述 `dsqa_act_v6` 注册 `9e5203ad62965012a9b74acb6eb32cea5e787ab8ab1ee3a781f1112ef5d2900d` 已完成。固定源 EVO 仅 DSQA129，三方法收到同份官方 ACT 2023 完整 52 行表与定义说明；`knowledge_mode=frozen_evidence`。**3/3 槽均有效提交、封存后完整评分**，`sealed=true`、`comparison_complete=true`，唯一完整配对为已登记的该任务。下列原生 F1 来自同 Exp 模型的适配 evaluator，private reference 隔离且不回流给 actor；不复印答案成员或参考答案。

| 方法 | 生成 / 评分状态 | 原生 F1 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---:|---|---|---|---|
| Direct | submitted / completed | 0.400000 | 1 / 1 | 2,173 / 443 | 1.656 / 1.296 | 2,485 / 131 |
| Initial-MAS | submitted / completed | 0.875000 | 9 / 1 | 97,287 / 443 | 30.406 / 1.000 | 89,390 / 8,340 |
| Native JIT | submitted / completed | 1.000000 | 3 / 1 | 30,919 / 419 | 20.234 / 1.047 | 25,172 / 6,166 |

全部生成为 **13 调用、130,379 tokens、52.296 活动秒**；评分为 **3 模型调用、1,305 tokens、3.343 活动秒**，合计 **16 调用、131,684 tokens**。三方法 request queue idle、estimated attempts、unknown slots 均为 0，美元费用均为 `null`。每槽仍使用相同 2,000,000 tokens / 900 活动秒共同预算，judge 时间从生成后剩余活动时间内扣除。公开材料输入和公开审查/修订耗费已在生成账本计入；此前离线 ACT evidence 准备成本另在准备审计记录中保留，不再次算作模型 API。

对照此前 `dsqa_v2` 同一公开源题的闭卷开发观察如下；这是输入和代码均已变化的历史对照，不能作为仅变更 prompt 的收益实验。

| 方法 | dsqa_v2 闭卷 F1 | dsqa_act_v6 共享官方证据 F1 |
|---|---:|---:|
| Direct | 0.461538 | 0.400000 |
| Initial-MAS | 0.285714 | 0.875000 |
| Native JIT | 0.625000 | 1.000000 |

闭卷依赖模型一般知识，新轮明确提供官方完整表、单位和分母；同时跨轮代码、组织提示与公开 refinement 配置发生变化，无法隔离知识材料、提示或组件各自贡献，也不能假定 Direct 必然因材料更完整而上升。本轮 Initial-MAS 高于 Direct **0.475**，仍低于 Native JIT **0.125**，且实际生成调用与 tokens 更高；一题已经曝光的源 EVO 不支持整体 benchmark、方法优于 JIT、统计显著或干净确认的结论。评分使用同模型 evaluator，并非独立 judge；本节数字只以原始 `summary.json` 的全库存、全部成本和封存结果为准。

## source v6：RR / WritingBench / DSQA 的闭卷全库存

`outputs/development_pilot_20261004_v6` 的 `source_v6` 原注册已结束，注册哈希仍为 `95dbb2ab16a8eb529bca232c9a860d097e9730ceb61e204bfca5d48e2bd1bcd3`。固定源 EVO 选题为 RR60、WB335、WB433、DSQA129、DSQA170，三方法共 **15 槽**，不改选题顺序。输入为共同闭卷一般知识，保留相对于正式共享证据主轨道的偏离标记；没有把刚准备的 ACT 或 DRB 材料注入本轮。全槽终结并封存，**14 槽有效提交且完整评分、1 槽生成失败**，`sealed=true`、`comparison_complete=false`。

| 固定任务 | Direct 原生分数 | Initial-MAS 原生分数 | Native JIT 原生分数 | 状态 |
|---|---:|---:|---:|---|
| RR60 | 0.512500 | 0.375000 | 0.537500 | 三方法 submitted / completed。 |
| WB335 | 7.800000 | `null` | 8.800000 | Initial-MAS `JSONDecodeError`，未交付有效稿；另两臂完整评分。 |
| WB433 | 9.000000 | 9.000000 | 8.800000 | 三方法 submitted / completed。 |
| DSQA129 | 0.307692 | 0.500000 | 0.444444 | 三方法 submitted / completed。 |
| DSQA170 | 0.666667 | 1.000000 | 1.000000 | 三方法 submitted / completed。 |

WritingBench 数字是原 `raw.native_mean`，不是 0–1 归一化分；失败槽真实分数为 `null`。完整源均分分别为：DSQA 的 Initial-MAS **0.750000**、Native JIT **0.722222**、Direct **0.487179**；WritingBench 的 Direct **8.4**、Native JIT **8.8**，Initial-MAS 因一题失败不报完整原生均分。完整三方法配对仅 RR60、WB433 和两项 DSQA，WB335 不能加入完整配对质量均分。

| 方法 | 提交 / 完整评分 / 5 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---|---|---|---|
| Direct | 5 / 5 | 5 / 32 | 11,025 / 88,892 | 28.062 / 50.093 | 82,080 / 17,837 |
| Initial-MAS | 4 / 4 | 38 / 31 | 497,478 / 74,021 | 234.218 / 45.156 | 495,737 / 75,762 |
| Native JIT | 5 / 5 | 15 / 32 | 139,591 / 101,032 | 131.329 / 54.656 | 190,448 / 50,175 |

全部生成为 **58 调用、648,094 tokens、393.609 活动秒**；评分为 **95 模型调用、263,945 tokens、149.905 活动秒**，合计 **153 调用、912,039 tokens**。失败 WB335 已耗费的调用和 tokens 全部计入 Initial-MAS，不按未评分删去。各方法 request queue idle、estimated attempts、unknown slots 均为 0，美元费用均为 `null`。RR / WB / DSQA 的 evaluator 均为同 Exp 模型的适配诊断评分，封存后才使用 private criteria/reference，不回流到本轮 actor；这不是独立 judge。

按 `summary.json` 的各源归一化再做全槽保守 macro，Direct 为 **0.618638**、Initial-MAS 为 **0.537683**、Native JIT 为 **0.719552**。Initial-MAS 在两道 DSQA 的本轮 F1 均值较高，但 RR 低于另两臂，WB335 生成失败，WB433 只与 Direct 持平；成本也更高。因此不能将两道 DSQA 的差值解释为整体优势，或略去失败后宣布胜出。样本均已用于开发，只有单一模型与两题 DSQA，也没有显著性或泛化确认。

同一 v6 代码/配置下，DSQA129 的闭卷 F1 为 Direct **0.307692**、Initial-MAS **0.500000**、Native JIT **0.444444**；另注册的 ACT 共享材料轮分别为 **0.400000 / 0.875000 / 1.000000**。这补充了上一节的历史对照：输入知识状态确实变化，不能将材料轮分差说成仅由 prompt 改进造成，也不能将一次共享材料对照当作独立因果实验。

## DRB v6：有效提交被修订成标题后的零分

`outputs/development_pilot_20261004_drb_v6` 的原注册 `0e0c8f6b41918b55c367a63b46809578ff067c7ea20898b2cb494ea5b19f7e8f` 已结束。仍是既有台账中的单条 DRB1 曝光 TEST，输入为共同闭卷任务；本节不复制题干、private criteria 或参考答案。**3/3 槽均 submitted / completed**，全部封存，`comparison_complete=true`。每臂 evaluator 均完成完整 **58 项检查**，各用两次同模型评分调用，没有 incomplete 评价。

| 方法 | 完成检查 | 原生聚合分 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---:|---|---|---|---|
| Direct | 31/58 | 0.534483 | 1 / 2 | 4,040 / 15,725 | 14.281 / 16.532 | 10,679 / 9,086 |
| Initial-MAS | 0/58 | 0.000000 | 11 / 2 | 175,983 / 6,275 | 87.172 / 9.562 | 154,572 / 27,686 |
| Native JIT | 30/58 | 0.517241 | 4 / 2 | 48,724 / 14,807 | 52.922 / 16.484 | 43,128 / 20,403 |

全部生成为 **16 调用、228,747 tokens、154.375 活动秒**；评分为 **6 模型调用、36,807 tokens、42.578 活动秒**，合计 **22 调用、265,554 tokens**。queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`。Initial-MAS 的 0 是已观测的完整评分，不是生成失败、缺失分数或 failure-zero 替代值；不能改成 `null` 或删除这项负结果。

原 Initial-MAS `public_refinement.json` 保存 **4,038 字符**的有效 draft，随后 `public-revision` 将其变成 **30 字符、单个非空行的标题**。revision 调用只输出 26 tokens、`finish_reason=stop`，审查调用也为 stop；不是空白答案、输出上限截断或 API 失败。审查/修订记录 status 为 completed，本题没有激活 public positional construction。v6 普通长文修订采用 strict `json_schema`，格式有效并没有保住公开任务所需的完整成文，最终封存的 revision 才是被评分对象。

该次 draft / revision 身份和 hash 保留在 `generation/852816289aee579bf7e94186c64db778e63e003e23c63a47fa52ecdf38c41f25/10e5abeb40cc2ba2344bf5c0c66b286ceb91648fdfad1ffe5ee56fca5888c664/public_refinement.json`，原 seal、submission、evaluation、summary 不改。没有给 draft 补评分、换回可能更高分的 draft、在同注册下重采或用 judge 反馈回改。严格结构化输出对该实际长稿造成成品覆盖的诊断值得后续另注册验证，但单轮、多组件变化与同模型审查不能确认单一因果机制。此轮 Initial-MAS 明显低于 Direct 和 Native JIT，且耗费更高，原始零分完整保留；曝光 TEST 和同模型 private evaluator 也不构成干净确认或独立 judge。

## DRB1 五份官方共享材料准备

在 DRB v6 闭卷运行之外，另冻结 `outputs/development_drb1_primary_evidence_20261004`，供后续另注册、三方法同份输入的开发对照。只针对原有已曝光 DRB1 公共任务，不读取 private criterion/reference、不生成研究报告或答案、不新增题目。不访问或使用公开任务明令排除的文章及其镜像；五个来源均为政府官方站点，来源检索只沿此前已经核验的固定 URL。

| 固定来源 | 保留的公开事实与时间身份 |
|---|---|
| [财政部国库司《2021年财政收支情况》](https://www.mof.gov.cn/jrttts/202201/t20220129_3785852.htm) | 2022年1月发布的2021全年收支情况，亿元单位；全国/中央/地方本级一般公共预算、政府性基金预算分开。 |
| [财政部预算司《2021年全国政府性基金收入决算表》](https://yss.mof.gov.cn/2021zyjs/202207/t20220728_3830522.htm) | 2022年7月发布的2021决算，表内预算数、决算数及两项百分比列身份保留。 |
| [深圳政府公报转载国务院分税制决定](https://www.sz.gov.cn/zfgb/1994/gb49/content/post_10090769.html) | 文件文尾1993年12月，改革自1994年1月；75:25是当时增值税分享规定，非后来每年的通用比例。 |
| [上海住建委汇编的1998年住房改革通知](https://zjw.sh.gov.cn/wjhb/20180912/0011-30557.html) | 1998年下半年开始停止实物分配、逐步货币化，地方确定具体步骤；页面标题与文尾日期差异原样标注。 |
| [北京人大转载2021年房地产税试点授权决定](https://www.bjrd.gov.cn/xwzx/qgrd/202110/t20211025_2519396.html) | 2021年10月23日通过的部分地区试点授权，与全国统一税法及各地实际启动区别；转载日期另列。 |

年度收支情况中的土地收入 **87,051 亿元**和后来决算表中的 **84,977.85 亿元**各自保留版本，不混用或编造差异原因；一般公共预算与政府性基金预算的范围、单位和地方本级含义同时注明。材料没有已核验的 **1998–2008 土地收入增长序列**，明确不得由2021点值补造增长倍数或因果估计。该包是五份公开原始文件要点的人工摘要，不是完整任务答案或覆盖全部论点的文献综述。

Packet 为 **2,982 cl100k tokens**，五来源均完整保留摘要、未截断，`manifest`、canonical public task identity 与 shared loaders 检查通过；离线重放后 **5 个冻结 artifact 的 hash 不变**。主 pack 文件 SHA256 为 `981bfa35d1d61b5997899ef5ae27a9dcec41076a17a3bdb3b2f88fa3af4b2a9f`，manifest 为 `137944ddd067bef93d46b40bcad9d19b080621621eea1850fe7dc338a80875de`。四份原 HTML 字节及提取文本保存来源 hash；深圳页面的原 HTML 下载失败，另保存实际官方 web.open 返回的完整工具文本与页面行提取 hash，raw HTML hash 明确为 `null`，不伪造原字节身份。

准备保留此前协调器提供的 **3 次 web calls / 9 queries**出处线索；本次没有新增搜索 query，实际为 **3 次 web calls / 8 document opens（5 个唯一官方 URL）**。另外直接 HTTP 抓取共 **5 attempts、4 成功、1 `URLError` 失败**，并行抓取墙钟 **0.451 秒**；失败原记录保留，随后以官方页面 browser extraction 补齐深圳来源，不删除失败后宣称首次全成功。准备与本地缓存 builder replay 均为 **0 actor/judge/remote planner API**；builder 的 mandatory planner callable 与缓存 search/crawl 是确定性本地执行，估计 tokens 不当作计费模型用量。网页工具总墙钟、人工整理时间与美元费用未完整记录，未知项不填 0；实际来源抓取、hash、缓存 replay 使用账本与覆盖限制详见 `download_log.json`、`browser_fallback_log.json` 和 `preparation_report.json`。这批材料没有参与上述已结束的闭卷 DRB v6 分数，后续使用必须新注册并标注知识输入变化。

## exposed v7：全 actor penalty 变体的不利结果

`exposed_v7` 于北京时间 **2026-10-04 02:23:29** 注册，目录为 `outputs/development_pilot_20261004_exposed_v7`，注册哈希为 `e584db4752b6360526dbbc0ef57fc9fa99db7924fac213e24f8d537c3e43e16e`。它已实际运行并封存，不能与下节从未开始的 v7 注册混同。选题仍为既有四条曝光指令任务，12 槽，formal TEST 与独立确认排除标记不变。

该轮配置 `public_refinement_response_format=json_schema_review`：公开 review 和支持的词槽结构修订采用 strict schema，普通长文 revision 采用 JSON object，原 `json_schema` / `json_object` 模式仍保留。路由由 public task 的可支持约束决定，不按 benchmark ID 或私有评分结果选择。同时 meta/global/local/exec 四个角色的 `frequency_penalty` 均登记为 **0.25**，public-review / revision 随其使用的模型角色继承该设置；judge 未配置该 penalty。这是本轮实际注册事实，并非仅有后续计划。

**12 槽全部终结并封存，只有 Direct 四题有效提交且完整评分；Initial-MAS 和 Native JIT 各四题均生成失败**，`comparison_complete=false`。初始结构化规划出现字段校验错误，Native 生成也出现缺少必需 harness 文件（包括已观察的 `prompt.yaml`）等格式问题；每槽按原异常类型保留，不假定全部槽的具体根因一致。它们没有有效稿可交给 checker，也没有证据表明公开 refinement 已修复这些早期失败。

| 固定任务 | Direct prompt 主分（通过指令 / 总数） | Initial-MAS 状态 / 主分 | Native JIT 状态 / 主分 |
|---|---|---|---|
| IFEval1203 | 0（1/2） | `ValidationError` / `null` | `ValueError` / `null` |
| IFEval1246 | 1（1/1） | `ValidationError` / `null` | `ValueError` / `null` |
| IFBench13 | 0（0/1） | `ValidationError` / `null` | `ValueError` / `null` |
| IFBench22 | 0（1/2） | `ValidationError` / `null` | `ValueError` / `null` |

| 方法 | 提交 / 完整评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 本地评分活动秒 | 全部 input / output tokens |
|---|---|---:|---:|---:|---:|---|
| Direct | 4 / 4 | 4 | 1,747 | 8.124 | 1.218 | 901 / 846 |
| Initial-MAS | 0 / 0 | 10 | 92,452 | 45.718 | 0 | 80,565 / 11,887 |
| Native JIT | 0 / 0 | 12 | 163,460 | 102.047 | 0 | 133,345 / 30,115 |

生成合计 **26 调用、257,659 tokens、155.889 活动秒**，本地 checker 为 **1.218 秒、0 模型调用 / tokens**。三方法 queue idle、estimated attempts、unknown slots 为 0，美元费用为 `null`；八项失败已经消耗的 22 次调用和 255,912 tokens 均在总账保留。Direct 的 IFEval 为 1/2、IFBench 为 0/2，macro failure-zero 为 **0.25**；Initial-MAS / Native JIT 的保守 macro 为 **0**，但八个真实分数均为 `null`，不是八个已观测 checker 零分。

该轮是明确的不利开发结果：修改 review/revision response mode 没有取得已观测的有效交付提升，而向全部结构化 planner / generator 同时施加 penalty 的变体未交付两臂的任何最终稿。由于多个配置一同改变，不能把 frequency penalty 单独当作已隔离因果结论；原 sealed inventory、失败调用、分数缺失与 summary 全部保留，不删除这轮或在同注册下重采。后续窄化设置必须重新注册；未发生的新轮不填分数，也不选择较早或更高分 draft 替代原提交。

## 四个未开始的 v7 注册：生成前替代

在上面的 `exposed_v7` 实际运行之外，下列四个 v7 目录只完成注册，分别核验仅有 `registration.json` 和 `release/inventory.json`、inventory 与原注册一致、没有 generation 目录 / started marker / submission / evaluation / runtime database 后，才写 `superseded_before_generation.json`。已开始并封存的 `exposed_v7` 完全排除在该操作外。

| 原目录 | 原 registration hash | 注册槽 / 已开始槽 |
|---|---|---|
| `outputs/development_pilot_20261004_v7` | `911a9b3e6e4aed0ea42e6787e499947b4bd72a02c7f7d77ad6c45bba8e69154b` | 15 / 0 |
| `outputs/development_pilot_20261004_dsqa_act_v7` | `00186916d9232c3585d4f41ac63cc546dd0a2ce340fc04da94c43f93b34abdd8` | 3 / 0 |
| `outputs/development_pilot_20261004_rr_evidence_v7` | `af156c7cf439e02cc227e70a43c4f7983682b1fde891f299c8605ac3f920da48` | 3 / 0 |
| `outputs/development_pilot_20261004_drb_evidence_v7` | `bc144ef23700496d8ab876a458efa2e881d1dfc9d103298a9f9c1de7a1ab281c` | 3 / 0 |

共 **24 个未开始槽、0 generation attempts / API / 评分观察**，不报告为 24 个失败或 24 个零分。原 registration / inventory SHA256 前后不变，外部日志保留。标记仅记录将被新 v8 变体替代的政策，replacement registration hash 为 `null`，没有虚构尚未绑定的身份或把未来共享 DSQA 两题与原 ACT 单题当作同一任务范围。此处不新增 API、修改 runtime 或恢复原注册。

## v8 新注册：窄化 revision penalty 与共享 DSQA 两题范围

v7 不利结果保留后，v8 使用新的冻结注册，保留 `json_schema_review` 和公开词槽施工，五个 model role 的 `frequency_penalty` 均为 **`null`**，不向结构化 planner / generator 统一传该选项。另登记 **`MASConfig.public_revision_frequency_penalty=0.25`**，仅在 public revision 的实际调用 kwargs 中应用并审计；public review 与其他调用不随之更改。该设置是待本轮实际结果检验的方法组件，不视为已证实收益或依据分数的回退。

| 新目录 | 注册时间（北京时间 2026-10-04） | registration hash | 固定槽 |
|---|---|---|---:|
| `outputs/development_pilot_20261004_exposed_v8` | 02:28:16 | `f6c9c55828aac58b58d1850ebb594c8d4029492e56f24c914726f048683cff1b` | 12 |
| `outputs/development_pilot_20261004_v8` | 02:29:01 | `1ea022e846d216ed094489b35e716619e52aa5350dc84a6ca13daf366f8e2d39` | 15 |
| `outputs/development_pilot_20261004_dsqa_evidence_v8` | 02:30:14 | `588f25dec3e70b316dce54052058873d7c367f9839d808c2aa5d7efa2b995154` | 6 |
| `outputs/development_pilot_20261004_rr_evidence_v8` | 02:29:01 | `eed3475758b64cf9d84cf54cb63e45d25d53c165e2c0bfcfb096594904f2bd85` | 3 |
| `outputs/development_pilot_20261004_drb_evidence_v8` | 02:29:02 | `72f9e2cae8a859600bbb90dcec0ed3ec6218865a99f6855d105e25752d8a87ac` | 3 |

五个注册合计 **39 个固定 generation slots**，选题范围在评分前确定，各 campaign 仍须全槽生成终态后封存再评分。共享 DSQA 新范围为固定前两道源 EVO、三方法 **6 槽**，包含 ACT129 与源 DSQA170 各自的完整共享包；这比旧 ACT 单题、3 槽扩大了任务范围，不能将两题均分与旧单题分数当作同一实验。源 EVO 闭卷、RR / DRB 共享材料与已曝光指令轮分别登记，不混合知识输入或偷偷改选题。

该轮固定 selected 台账只追加已存在的 `exposed_v8` 和 `drb_evidence_v8` 注册引用，仍为原有 **5 个正式选用 TEST ID**，所有 task_id、source row、original partition、question hash 不变；这里只说明本轮注册不增加 selected ID，不能代替整个开发过程的曝光审计。代表性公开 example 配置已同步为窄化设置，endpoint / model 仍为占位符、key 仅取环境变量。此节记录注册及范围，不填写尚未封存的 v8 分数、成本或待完成的 full-test 结果，也不预先宣称修复已提升 benchmark。

## exposed v8：全提交后的配对持平与具体约束差异

`outputs/development_pilot_20261004_exposed_v8/summary.json` 已封存，注册身份为 `f6c9c55828aac58b58d1850ebb594c8d4029492e56f24c914726f048683cff1b`。原定 **12/12 槽全部 submitted / completed**，`comparison_complete=true`，没有生成失败、未完成 checker 或漏报槽。评分对象仍为已封存最终提交；IFEval 主指标是原生 **prompt-level strict accuracy**，IFBench 主指标是原生 **prompt-level loose accuracy**，下表括号仅另列指令通过数，不用它替换 prompt 主指标。

| 固定任务 | Direct 主分（通过指令 / 总数） | Initial-MAS 主分（通过指令 / 总数） | Native JIT 主分（通过指令 / 总数） |
|---|---|---|---|
| IFEval1203 | 1（2/2） | 1（2/2） | 1（2/2） |
| IFEval1246 | 1（1/1） | 1（1/1） | 1（1/1） |
| IFBench13 | 0（0/1） | 1（1/1） | 0（0/1） |
| IFBench22 | 1（2/2） | 0（1/2） | 1（2/2） |

三方法 IFEval 均为 **2/2**、IFBench 均为 **1/2**，全部四题主指标均为 **3/4**，按两个来源等权的 macro normalized failure-zero 均为 **0.75**；此轮没有缺失分数，保守聚合与实际完整主指标相同。Initial-MAS 的公开词槽施工通过 IFBench13，但在 IFBench22 的另一项约束上失分，Direct / Native JIT 的得失位置相反。不能只报道词槽题或拿其中一题宣称总体超过 Native JIT。相比上一轮全 planner/generator penalty 的八个生成失败，本轮恢复了所有交付；多组件与随机调用差异仍不足以单独证明某个配置的因果收益。

| 方法 | 提交 / 完整评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 本地评分活动秒 | 全部 input / output tokens | request queue idle 秒 |
|---|---|---:|---:|---:|---:|---|---:|
| Direct | 4 / 4 | 4 | 2,204 | 10.016 | 1.062 | 901 / 1,303 | 0.016 |
| Initial-MAS | 4 / 4 | 24 | 226,856 | 101.296 | 1.078 | 203,897 / 22,959 | 0 |
| Native JIT | 4 / 4 | 12 | 89,363 | 83.329 | 1.048 | 65,275 / 24,088 | 0 |

全库存生成合计 **40 调用、318,423 tokens、194.641 活动秒**；原生本地 checker 合计 **3.188 秒、0 模型调用 / tokens**。三方法 estimated attempts / unknown slots 均为 0，美元费用均为 `null`；不能把 token 消耗当作已知美元账单。Initial-MAS 在同主指标下用了最多调用和 tokens。已曝光 TEST 的开发属性保持不变，不能称为 formal TEST、干净确认或 trained 27B JIT 复现；本轮评分为冻结原生 checker，不存在独立模型 judge。

## DSQA shared evidence v8：两题全库存与生成失败

`outputs/development_pilot_20261004_dsqa_evidence_v8/summary.json` 已封存，注册身份为 `588f25dec3e70b316dce54052058873d7c367f9839d808c2aa5d7efa2b995154`。固定两个来源 EVO 任务，各三方法、**6 个预注册槽**；各任务完整 frozen packet 对三臂相同。**5 槽 submitted / completed、1 槽 Initial-MAS 在 DSQA170 生成失败**，`comparison_complete=false`。不删掉失败后只报 ACT 单题或成功槽均值，也不把未评分当作观察到的 F1 零分。

| 固定任务 | Direct F1 | Initial-MAS F1 / 状态 | Native JIT F1 |
|---|---:|---|---:|
| DSQA129（ACT 全州表） | 0.400000 | 1.000000 / completed | 1.000000 |
| DSQA170（另一份完整 packet） | 0.000000 | `null` / `RuntimeError`, submission_failed | 0.666667 |
| 两题完整原生均值 | 0.200000 | `null` | 0.833333 |
| 两题 normalized failure-zero 均值 | 0.200000 | 0.500000 | 0.833333 |

ACT129 的 Initial-MAS 从 ACT v6 的 **0.875** 到此次 **1.0**，Direct 仍为 **0.4**、Native JIT 仍为 **1.0**；这只是一题多组件开发改动后的配对观察，不隔离 prompt、审查模式或采样调用的单独作用。共享证据两题范围比旧 ACT 单题更大，不能把此次两题宏均值与旧单题分数作同口径比较；与闭卷轮的差异还同时包含知识输入变化。保留 DSQA170 的完整失败，当前两题全库存结果没有证明 Initial-MAS 超过 Native JIT。

| 方法 | 提交 / 完整评分 / 2 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---|---|---|---|
| Direct | 2 / 2 | 2 / 2 | 12,664 / 2,698 | 7.657 / 2.531 | 13,477 / 1,885 |
| Initial-MAS | 1 / 1 | 16 / 1 | 268,000 / 402 | 109.235 / 0.875 | 233,871 / 34,531 |
| Native JIT | 2 / 2 | 6 / 2 | 98,430 / 1,377 | 49.875 / 2.157 | 84,957 / 14,850 |

生成合计 **24 调用、379,094 tokens、166.767 活动秒**；评价合计 **5 调用、4,477 tokens、5.563 活动秒**；合计 **29 调用、383,571 tokens**。Initial-MAS 的失败本身消耗 **8 调用、186,228 tokens、78.875 活动秒**，已经包含在其 16 次生成调用的总账中；没有重采、替换失败槽或以较早 draft 代替封存提交。request queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`。DSQA 的 private evaluator 仍为同 endpoint / 同模型的 self-judge，评分不回流 actor；该两题开发观察不是独立 judge 的确认结果，也不是 formal TEST。

## v8 冻结版本验证与后续候选边界

协调器完成的 v8 full suite 为 **1,385 passed + 60 subtests，148.77 秒**。按 Git 源文件清单的 **171 个 Python 源文件 compile 检查通过**；另一次全目录扫描发现 **2 个 ignored 输出目录中的历史模型生成文件有 syntax error**，这两份原失败 artifact 保留，不修改、删除或当作发布源码。这些验证结果针对冻结 v8 源版本，不提前承诺尚未注册、运行的候选版本质量。

本文此时只补已封存的 exposed v8 / DSQA shared evidence v8，主来源闭卷、RR / DRB shared evidence 等其各自完整封存后再追加。后续公开数值施工或取消 revision penalty 的候选需新代码 / 配置身份与注册，保留全部旧库存；同配置不按 checker / judge 结果重采，不挑选较高 draft，也不把未完成轮次的个别行当作整体比较。

## source v8：闭卷五题全库存与不利结果

`outputs/development_pilot_20261004_v8/summary.json` 已完整封存，注册身份为 `1ea022e846d216ed094489b35e716619e52aa5350dc84a6ca13daf366f8e2d39`。仍为固定 EVO RR60、WB335 / WB433、DSQA129 / DSQA170，三臂 **15 个预注册槽**，共同 closed-book / general-knowledge 输入。**14 槽 submitted / completed，1 槽 Initial-MAS 在 RR60 生成失败**，`comparison_complete=false`；原始失败及已评分零分都保留。

| 固定任务与原生尺度 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| RR60，native weighted score（官方加权公式） | 0.637500 | `null` / `ValidationError`, submission_failed | 0.537500 |
| WB335，原生 mean / 10 | 8.6 | 7.4 | 8.2 |
| WB433，原生 mean / 10 | 9.0 | 9.0 | 9.0 |
| DSQA129，F1 | 0.307692 | 0.428571 | 0.428571 |
| DSQA170，F1 | 0.666667 | 0.000000 | 1.000000 |

WritingBench 两题**原生均值**为 Direct **8.8**、Initial-MAS **8.2**、Native JIT **8.6**，相应 normalized 均值另为 **0.866667 / 0.800000 / 0.844444**，不把归一化值称为原生分。DSQA 两题完整 F1 均值为 **0.487179 / 0.214286 / 0.714286**；RR 的 Initial-MAS 原生分保持 `null`。按 RR / WritingBench / DSQA 三来源等权且缺失仅在保守聚合中置零，macro normalized failure-zero 为 **0.672212 / 0.338095 / 0.709499**。Initial-MAS 的 DSQA170 是完成 evaluator 后观察到的 **真实 0 分**，与 RR60 未生成成功的 `null` 不同，不能改成失败或删掉。

| 方法 | 提交 / 完整评分 / 5 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens | request queue idle 秒 |
|---|---|---|---|---|---|---:|
| Direct | 5 / 5 | 5 / 32 | 11,590 / 112,539 | 30.390 / 51.922 | 105,127 / 19,002 | 0 |
| Initial-MAS | 4 / 4 | 36 / 4 | 487,922 / 11,202 | 221.859 / 6.812 | 438,561 / 60,563 | 0 |
| Native JIT | 5 / 5 | 14 / 32 | 145,342 / 121,431 | 148.733 / 53.845 | 211,330 / 55,443 | 0.015 |

全库存生成为 **55 调用、644,854 tokens、400.982 活动秒**，评价为 **68 调用、245,172 tokens、112.579 活动秒**，合计 **123 调用、890,026 tokens**。Initial-MAS 的 RR60 失败耗费 **8 调用、125,173 tokens、59.250 活动秒**，已经包含在生成总账中，不丢弃失败费用。它没有可评分提交，因此没有该槽 private evaluator 调用，不能用较少评价调用代表方法更节省。estimated attempts / unknown slots 均为 0，美元费用为 `null`。

本轮 Initial-MAS 没有形成总体数值优势：它在 WB433 与另两臂持平，在 DSQA129 与 Native JIT 持平且高于 Direct，但 WB335 较低、DSQA170 完整零分、RR60 无有效提交，且生成耗费最高。所有五行同时报告，不拿局部持平或较高的一行覆盖其余结果，也不跨轮选用较高答案代替 v8 固定提交。该结果仍为少量、已开发曝光的 EVO 任务，评价模型与 actor 同 endpoint / 同模型，不构成独立 judge、formal TEST 或充分的方案优势确认。公开通用能力的后续改动须新注册并重新完整评估，当前原封存记录不修改、不重采。

## RR shared evidence v8：共同材料下的完整三臂结果

`outputs/development_pilot_20261004_rr_evidence_v8/summary.json` 已封存，注册身份为 `eed3475758b64cf9d84cf54cb63e45d25d53c165e2c0bfcfb096594904f2bd85`。固定来源 EVO RR60、三臂同一 frozen evidence packet，原定 **3/3 槽 submitted / completed**，`comparison_complete=true`；没有生成失败或不完整评价。这里只报告已经封存的原生 native weighted score（官方加权公式） 及成本，不复制题干、private criteria、参考答案或权重。

| 方法 | 原生 native weighted score（官方加权公式） | normalized score | 提交 / 完整评分 / 1 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---:|---:|---|---|---|---|---|
| Direct | 0.362500 | 0.406977 | 1 / 1 | 1 / 28 | 5,680 / 72,713 | 8.140 / 44.828 | 65,933 / 12,460 |
| Initial-MAS | 0.500000 | 0.534884 | 1 / 1 | 8 / 28 | 143,922 / 69,551 | 49.765 / 43.515 | 189,897 / 23,576 |
| Native JIT | 0.487500 | 0.523256 | 1 / 1 | 3 / 28 | 45,312 / 100,151 | 35.188 / 46.625 | 124,331 / 21,132 |

全库存生成合计 **12 调用、194,914 tokens、93.093 活动秒**；评价合计 **84 调用、242,415 tokens、134.968 活动秒**；合计 **96 调用、437,329 tokens**。三臂 request queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`。三臂评价全部完成，各 28 次调用均计入成本；Initial-MAS 生成阶段的调用与 tokens 仍显著高于另外两臂。

同 frozen packet 的这一个已曝光 EVO 任务上，Initial-MAS 原生分高于 Direct **0.1375**、高于 Native JIT **0.0125**，可作为局部开发观察，不能推成六 benchmark 或总体优势。在同版本闭卷 source v8 中，Direct / Initial-MAS / Native JIT 分别为 **0.6375 / failed-null / 0.5375**；共享材料轮的两条 baseline 分数反而较低，Initial-MAS 则成功交付。两种输入的差异包含知识材料变化和具体调用差异，不能把此次交付恢复或单题分差归因于某一 prompt，也不能只保留较高输入版本。旧闭卷失败、旧 shared 运行失败和此次完整三臂结果各自保留。

评价仍由相同 endpoint / 模型执行 private self-judge，反馈不回流 actor，不是独立 judge 或 formal TEST 的确认。此时 DRB shared evidence v8 尚在运行，其结果等全槽终态、封存并完成评价后再写；尚未注册运行的后续候选只作计划，不填写未来分数或按此次评价选择替代提交。

## DRB shared evidence v8：全库存的不利结果

`outputs/development_pilot_20261004_drb_evidence_v8/summary.json` 已封存，注册身份为 `72f9e2cae8a859600bbb90dcec0ed3ec6218865a99f6855d105e25752d8a87ac`。仍为曝光台账中原有 DRB1，三臂同一份五来源官方 frozen evidence，**3/3 槽 submitted / completed**，`comparison_complete=true`；没有生成失败、分数缺失或不完整评价。共有 58 项评价的原生聚合结果如下，不复印 private criteria / 权重或原题。

| 方法 | 完成评价结果 | 原生聚合分 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---:|---|---|---|---|
| Direct | 30/58 | 0.517241 | 1 / 2 | 5,644 / 13,045 | 11.312 / 14.359 | 11,356 / 7,333 |
| Initial-MAS | 16/58 | 0.275862 | 8 / 2 | 152,888 / 12,348 | 80.922 / 12.094 | 138,264 / 26,972 |
| Native JIT | 3/58 | 0.051724 | 3 / 2 | 50,087 / 9,008 | 49.812 / 10.766 | 40,443 / 18,652 |

全部生成合计 **12 调用、208,619 tokens、142.046 活动秒**；评价合计 **6 调用、34,401 tokens、37.219 活动秒**；合计 **18 调用、243,020 tokens**。三臂 request queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`。三项分数均是完成 evaluator 后的实际观测，不是失败置零；原提交、两次 judge 调用和完整库存保持不变。

Initial-MAS 在此单题低于 Direct、高于 Native JIT，生成耗费最高，不构成稳定超过两个基线的证据。旧 closed-book DRB v6 的 Direct / Initial-MAS / Native JIT 分别为 **31/58、0/58、30/58**；本轮同时改了审查/修订模式、revision penalty 和知识输入，不能把 Initial-MAS 的恢复或 Native JIT 的下降单独归因于 evidence packet、某段 prompt 或 response format。官方 packet 的覆盖限制仍保留，不能为了得分补造缺失年份序列、因果估计或使用公共任务禁止的文章。

至此，v8 的五个 campaign 均已结束：**39 个固定槽、37 个完整评分、2 个生成失败保持 `null`**。两个失败分别是闭卷 RR60 和共享材料 DSQA170 的 Initial-MAS，不从库存删除；已评分的真实零分另行保留。各 campaign 先封存自己的全部生成再评分，不冒称跨五 campaign 的所有生成在第一次评分前已统一封存。全部结果共同提供开发诊断；显式 TEST 曝光与同模型 private self-judge 的局限仍然存在，不作为干净 formal TEST / 独立确认，也没有选择其中较高输入版本覆盖较低版本。


## v8 五 campaign 的完整算量汇总

下表只汇总上面已封存的五个 v8 campaign，所有 **39 个原注册槽**都计入，包括两个 Initial-MAS 生成失败的实际开销。各输入模式与 benchmark 原生评分尺度不同，**不合并其 raw scores**，也不从 shared / closed-book 中挑较高版本；算量可以在保留原库存后汇总。

| 方法 | 完整评分 / 注册槽 | 生成 / 评分调用 | 生成 / 评分 tokens | 全部模型调用 / tokens | 生成 / 评分活动秒 |
|---|---|---|---|---|---|
| Direct | 13 / 13 | 13 / 64 | 37,782 / 200,995 | 77 / 238,777 | 67.515 / 114.702 |
| Initial-MAS | 11 / 13 | 92 / 35 | 1,279,588 / 93,503 | 127 / 1,373,091 | 563.077 / 64.374 |
| Native JIT | 13 / 13 | 38 / 64 | 428,534 / 231,967 | 102 / 660,501 | 366.937 / 114.441 |

五轮合计 **143 generation calls / 1,745,904 tokens、163 evaluation calls / 526,465 tokens**，共 **306 模型调用、2,272,369 tokens**。生成与评价活动秒分别求和为 **997.529 / 293.517**；多槽并行可能重叠，不把其和宣称为协调器总 elapsed。另有已逐轮列明的本地 checker 耗时、共 **0.031 秒 request queue idle**，美元费用为 `null`；evidence 准备与 v9 合成 transport probes 在各自章节另账，不混入这 39 个 v8 slots。完整算量显示 Initial-MAS 开销最高、交付率也非最高，不能只报道某个高分槽掩盖资源与失败。

## v9 已集成候选：公开数值施工、引用去重与完整内容保留

v8 全部结束后，协调器已将以下通用候选集成至生产源码，配置候选为 `.runtime/public_refinement_v9.config.json`。本节记录代码集成与注册阶段的实际配置；当时 targeted checks、两次 provider probe 和五个 campaign 的冻结注册已完成，尚无封存分数。随后已完成的 campaign 结果在首表与后续章节逐次填写，验证状态也按实际 full suite 记录，不提前认定方案优势。

- 新增 `public_numeric_construction` 开关与 `jit_mas/public_numeric_slots.py`。仅从 public question / constraints 识别受支持的、无歧义的英文 exact 数值个数及可选连词要求，建立严格槽位 schema，再确定性渲染；不依赖 benchmark ID、checker、参考答案、私有评分或 evidence metadata。保证限于隔开的 ASCII 整数字面量个数，以及支持的连词词表出现，不保证一般语义数值计数、连词语法、事实、genre 或其他约束。引用、例子、范围限定、unsupported / mixed 条件保守拒绝进入该施工分支，不默默放宽公共约束。
- 数值与位置规则混合时的共用 conflict guard 使两种施工都不激活，由普通公开 review / revision 处理完整请求，不先施工其中一种而丢掉另一种。保留 `json_schema_review`：review 和已支持施工的 revision 用 strict schema，普通 prose revision 用 JSON object；不是按题号或私有评分路由。
- Contributor / ledger 修复提示要求每个实际稳定 `source_id` 只保留一个 source reference，多个已观测 URL / locator 在同一 locator 字符串中合并，多条 evidence span 可复用该 `source_ref`。这是保持所有支持内容的公开协议提示，不以重命名 fixed-pack ID、删除证据或虚构来源满足 schema；没有观测来源时仍只在 outline 保留可支持的记忆材料。
- 公开 review / revision 将 supplied template 作为组织形式，适配当前主体与用途，不挪用不相关流程、费用、机构名或占位日期；先覆盖明确需要的内容，修订后对照原请求和 draft 保留已满足的实例、步骤、成员、来源与差异，避免可选套话挤掉必要 deliverable。critic 仍可出错，不把其建议当作事实权威，也不拿较高 draft 与 final 择优。
- 五个 ModelConfig 的 `frequency_penalty` 与 `public_revision_frequency_penalty` 全部为 `null`，不再向公开 prose revision 单独加 0.25。该更改与数值施工、引用协议和内容保留同时发生，只能在新注册下验证组合行为，不能提前声称已修复 v8 的失分或分离各组件因果。

已注册覆盖 IFEval / IFBench 各两题、DSQA shared 两题、RR shared 一题、DRB shared 一题、WritingBench closed-book 两题，共 **30 generation slots、6 benchmarks、5 个独立 campaign**。共同模型与预算沿用；每个 campaign 都需在评分前注册固定全库存、全部生成终态并封存，失败和成本全报，不按旧分数挑较高 row、同配置重采或使用未封存结果。实际 registration hashes 和已有测试/probe结论另列如下；逐题分数和完整成本待各 campaign 封存后填入。

## v9 transport 合成探测：单列成本与首次记录缺口

数值施工整合后，协调器另做了两次真实 endpoint 的**合成 public transport probe**，不来自 benchmark 题目，也没有 official evaluator 或分数。它们不计入拟定 30 benchmark generation slots，不是对同 benchmark 低分或失败的重采样。两次均为 **2048 output-token cap、无 frequency penalty**，返回模型身份均匹配 `/mnt/data/datas/models/DeepSeek-V4-Flash-Vision-Exp`。

| 独立 probe artifact | 状态 / finish reason | 模型调用 | input / output / total tokens | API 活动秒 | 脚本 elapsed 秒 | ledger wall 秒 |
|---|---|---:|---|---:|---:|---:|
| `.runtime/numeric_slots_probe_v9_20261004.json` | `JSONDecodeError`（extra data） / `length` | 1 | 373 / 2,048 / 2,421 | 6.625 | 7.250 | 7.265 |
| `.runtime/numeric_slots_probe_v9b_20261004.json` | completed / `stop` | 1 | 836 / 249 / 1,085 | 1.750 | 2.266 | 2.266 |

首次 probe 脚本没有将 schema 文本放入 system prompt，这与已经存在的真实 refiner 行为不同；虽请求了严格 response format，返回仍触顶并在本地 JSON 解码出现 extra data。随后另存的 v9b probe 改为与生产一致地显式附上 schema 文本，同一 cap / 无 penalty 下通过 schema 与确定性渲染，得到合成任务要求的 **3 个整数词槽、2 个词表连词**。这是该有限 lexical construction 和 transport 的一次可行性检查，不是 benchmark 质量得分，也不确认 schema 文本单独导致成功或一般语法、事实和写作质量已得到保证。

两次合计 **2 调用、3,506 tokens（input 1,209 / output 2,297）、8.375 API 活动秒**，脚本 elapsed 合计 **9.516 秒**；queue idle / estimated usage 均为 0，美元费用为 `null`。API 时间、脚本 elapsed、ledger wall 各有计时边界，不相加为同一墙钟时长。原失败 artifact 不覆盖：首次只保留 error、实际账单、format / model identity 与 input hash，**没有保存失败 response 原文**，因此无法完整逐字复核其生成内容或做响应 replay；该审计缺口明确保留。v9b 则保存完整 response、calls 和 hash，不复印其正文到发布文档。

更早一次本地 `-m .runtime` 调用因模块入口写法不合法而退出，随后以 `runpy` 执行修正；这次本地启动错误为 **0 API**，不计入两个模型 probe，但不将它虚构成模型调用成功。上述诊断与 benchmark 全库存分开记录，未来 v9 campaign 必须各自冻结注册、封存和完整评分，不能把合成通过当作预先完成的效果验证。

## v9 五个冻结注册与已有 targeted 验证

五个 registration 在同一份源码 / runner identity 下冻结，全部在 benchmark 评分前登记。协调器按 **instructions → DSQA shared → RR shared → DRB shared → WritingBench closed-book** 顺序执行；顺序不依分数改变，各自全生成终态、封存后才评分。此处只读取 registration 的 metadata / hash，不读取运行中生成、题干或私有内容。

| campaign 目录 | 注册时间（北京时间 2026-10-04） | registration hash | 固定槽 |
|---|---|---|---:|
| `outputs/development_pilot_20261004_exposed_v9` | 02:57:49 | `3a3d0b6088764be6f3db985a5d1f5ab38040d33209012feaf6c489983cd52ede` | 12 |
| `outputs/development_pilot_20261004_dsqa_evidence_v9` | 02:57:50 | `f4324d1b65e942ae832cdae1b53a8121f11458c95b8131d483ffb7990e6ee356` | 6 |
| `outputs/development_pilot_20261004_rr_evidence_v9` | 02:57:51 | `72072910d2da242f7863a09c8687f4dba953c20d49b2c436c93eeed2430aae41` | 3 |
| `outputs/development_pilot_20261004_drb_evidence_v9` | 02:57:51 | `fd3ab67ccf1384bae7ccfc6573059f07b6a2526a17a295141bc8f5a9d03e4c16` | 3 |
| `outputs/development_pilot_20261004_writing_v9` | 02:57:52 | `d6cc137c8b8f8fd92297fcbabfb62ddfb0e7b5876b885a5546ac1fa0b0f0bcba` | 6 |

共同 runtime fingerprint 为 `e4cf46d9570aa978424070d8aa8c36b6dd9e374bde5c1c812fcafab8505caacf`，runner SHA256 为 `2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`；config 中 public numeric / positional construction 均启用，五个模型角色及 revision penalty 全部 `null`。配套 source / evidence identities 以各自原 registration 为准，不改 immutable v5 manifest、这些注册不新增正式 selected TEST ID，也不把 shared 与 closed-book 结果混成同一种输入；另外的 incidental legacy 搜索曝光在下节单独审计。

协调器报告已有针对性检查：**numeric slots 211 passed、schema 84 passed、refinement integration 26 passed、public pilot 458 passed**。各检查集合可能重叠，不累加成一个总 suite 数。全套 v9 测试先出现一个 fixture 预算失败，修正后的最终通过结果在文末完整登记；合成 provider probe 的一次失败和一次成功在上节单列成本与审计限制，不替代 benchmark 评分。新增模块与三个新测试已纳入发布候选文件清单，最终 credential / 原题 / private reference 扫描等全部结果文档完成后进行，当前不声称新候选已经扫描通过。

## v9 冻结后的 incidental legacy evaluation 搜索曝光

在查找 formal manifest SHA 的只读 review 中，`/root/baseline_runtime` 的检索范围错误地把 `paper/experiments` 下的历史 JSON 评价也纳入，看到了一份 `rr_run0_deepseekjudge_20261002/ours_v34/test_release/evaluations/9cf8e23c5b2c28701dda165674b89370e35347b0f5556f7d966fefa51abd63e2.json` 中的 private rubric 片段。已确认可见的 metadata task_id 为 **`6847465956a0f6376a605476`**，与当前固定 EVO RR60 的 **`6847465956a0f6376a6054a7`**不同；不复制片段、私有权重、原题或参考答案。

该检索的原工具输出报告 **6,387,091 tokens、heavy truncated**；这是工具报告的原输出规模，不是已证明模型实际读到的全部量。后续**只按文件名**重建有 **289 个匹配路径**，不能把它说成 289 道已读取题目，也不能反推出被截掉的文本哪些真正可见。完整已见 task-ID 范围仍为 **`unknown`**，不填 0、不给出只曝光这一额外 ID 的完整性承诺。metadata-only audit 为 `.runtime/final_v9_incidental_exposure_baseline_runtime.json`，路径清单 SHA256 为 `28ae636fbed1f1d5e0e965a6016e46a3c9356482524fab48d7209e2f2ce4f33b`；audit 与本文均不包含 rubric 正文。

事件发生时 v9 source / config 已冻结；该 agent 报告未把片段用于修改 v9 prompt、方法源码、选题、评分或答案选择，协调器没有审阅具体 rubric，当前实际 actor 输入仍不含 private evaluator / rubric。冻结身份与 30 个注册槽不变，不以新看到的旧分数更换答案或输入版本。以上限制描述本轮处理方式，不能抹去开发者已看到 private 片段的事实，也不证明过去其他历史 TEST 都从未曝光。

`paper/experiments/test_exposure_20261004.json` 的原 **5 个 selected task object 和 9 个 campaign refs 原样保留**，另加 `supplemental_exposure_events` 的 `incidental_legacy_evaluation_search`，含原搜索 command、已知 metadata / file、截断与 unknown 范围、audit hash 和未来排除政策；不将 incident 伪装成当前六 benchmark 的选题或第五个之外的正式运行槽。未来 formal / 干净确认至少须排除已知 `6847465956a0f6376a605476`，并完成 unknown 范围审计才能声称剩余集合未曝光；immutable v5 manifest 本轮不修改。后续查找已知 formal artifact 身份只读其 byte hash，不再用正文搜索穿过历史 private evaluation。此次 metadata 审计为 **0 benchmark / actor / judge API**。

## exposed v9：IFEval 完成交付、IFBench 两项生成失败

`outputs/development_pilot_20261004_exposed_v9/summary.json` 已封存，注册身份为 `3a3d0b6088764be6f3db985a5d1f5ab38040d33209012feaf6c489983cd52ede`。原定 **12 个固定槽**全部终态，**10 槽 submitted / completed、2 槽 Initial-MAS 生成失败**，`comparison_complete=false`。IFEval 仍用原生 prompt-level strict，IFBench 仍用原生 prompt-level loose；指令通过数只作 secondary，不能替换主指标。

| 固定任务 | Direct 主分（通过指令 / 总数） | Initial-MAS 主分 / 状态 | Native JIT 主分（通过指令 / 总数） |
|---|---|---|---|
| IFEval1203 | 0（1/2） | 1（2/2） | 1（2/2） |
| IFEval1246 | 1（1/1） | 1（1/1） | 1（1/1） |
| IFBench13 | 0（0/1） | `null` / `RuntimeError`, submission_failed | 1（1/1） |
| IFBench22 | 0（1/2） | `null` / `JSONDecodeError`, submission_failed | 0（1/2） |

IFEval 两题 prompt 主分为 Direct **1/2**、Initial-MAS **2/2**、Native JIT **2/2**。IFBench 完整原生均值为 **0 / null / 0.5**，Initial-MAS 两个未交付真实分数始终为 `null`，仅在保守 failure-zero 聚合中计 0。四题 / 两来源 macro normalized failure-zero 为 **0.25 / 0.50 / 0.75**。Native JIT 此轮通过 IFBench13；Initial-MAS 没有 IFBench22 有效提交，**不能据已通过的合成 probe 宣称 v9 数值施工已在该 benchmark 成功**。不能删掉失败后只报 Initial-MAS 两道成功 IFEval 的均值，也不把其 IFBench 缺失当作观察到的 checker 零分。

| 方法 | 提交 / 完整评分 / 4 | 生成调用 | 生成 tokens | 生成活动秒 | 本地评分活动秒 | 全部 input / output tokens |
|---|---|---:|---:|---:|---:|---|
| Direct | 4 / 4 | 4 | 2,829 | 12.749 | 1.063 | 901 / 1,928 |
| Initial-MAS | 2 / 2 | 30 | 317,862 | 168.500 | 0.516 | 267,947 / 49,915 |
| Native JIT | 4 / 4 | 14 | 99,399 | 98.094 | 1.015 | 71,026 / 28,373 |

生成合计 **48 调用、420,090 tokens、279.343 活动秒**，原生本地 checker **2.594 秒、0 模型调用 / tokens**。Initial-MAS 的 IFBench13 失败为 **5 调用、59,891 tokens、56.468 活动秒**，IFBench22 失败为 **9 调用、95,784 tokens、39.219 活动秒**；两项合计 **14 调用、155,675 tokens**均已包含在生成账单中。queue idle、estimated attempts、unknown slots 为 0，美元费用为 `null`。

本轮保留 v9 原有 freeze：失败后可以只读公开日志作诊断，但不改变正在执行的其余四个 campaign 的源代码、配置或固定输入，不按 official checker 重采、选更高 draft 或覆盖提交。该指令小样本仍是已曝光 TEST 开发，不是 clean formal TEST；额外 incidental legacy evaluation 搜索的已知 / unknown 污染范围也按上节披露，不由当前固定 5 个 selected ID 掩盖。

## v9 full suite 首次运行：一个离线 fixture 预算失败，待重跑

首次 full suite 报告 **1,705 passed + 60 subtests、1 failed**；失败来自 `tests/jit_mas/test_iterative_execution.py` 的离线 fixture 预算不足，而非新的真实 API benchmark 评分。该 fixture 上限为 **50,000 tokens**，保守 byte estimator 记录的两次 mock 调用总额为 **51,068**，超过了 fixture envelope。负责测试的 agent 将 envelope 改为共同的 **2,000,000**、保留全部语义断言，并安排 targeted 后最终 full suite 重跑。

不能把这次有失败的首次 suite 改写成 all-pass；随后重跑的最终计数、时长与源码 compile 结果另节记录，两个运行事实同时保留。这个 test-only 预算修正不改变被冻结的 v9 生产源码或真实 campaign 配置。发布候选清单应包含该 tracked 测试文件，后续最终扫描覆盖全部 **39 个候选文件**；合成 probe 的真实调用成本另账，不混进离线测试或本节的 12 benchmark slots。

## DSQA shared evidence v9：六项完整评分与两题均值

`outputs/development_pilot_20261004_dsqa_evidence_v9/summary.json` 已封存，注册身份为 `f4324d1b65e942ae832cdae1b53a8121f11458c95b8131d483ffb7990e6ee356`。固定 ACT129 / DSQA170 两个 source EVO，三臂同各自完整 frozen packet，**6/6 槽 submitted / completed**，`comparison_complete=true`；没有失败、分数缺失或不完整评价。

| 固定任务 | Direct F1 | Initial-MAS F1 | Native JIT F1 |
|---|---:|---:|---:|
| DSQA129（ACT 全州表） | 0.875000 | 0.933333 | 1.000000 |
| DSQA170（另一份完整 packet） | 0.000000 | 0.000000 | 0.666667 |
| 两题完整均值 | 0.437500 | 0.466667 | 0.833333 |

两道 DSQA170 的 0 均是**完成 evaluator 后的真实观测零分**，不改成 `null`，也不以 ACT 较高分遮盖。Initial-MAS 交付恢复后的两题均值略高于 Direct，仍明显低于 Native JIT。相比 v8，Initial-MAS 的 DSQA170 从生成失败变为完整零分，ACT 从 1.0 变为 0.933333；v8 保守两题均值为 0.5，此轮完整均值为 0.466667，不能用成功交付称作整体指标提升。Direct ACT 从 0.4 到 0.875、Native JIT 两题不变；这组单轮、多组件及调用差异不分离 prompt 的因果效果，更不能与旧 ACT 单题 / 闭卷版本择高拼接。

| 方法 | 提交 / 完整评分 / 2 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---|---|---|---|
| Direct | 2 / 2 | 2 / 2 | 12,182 / 2,138 | 6.110 / 2.359 | 12,950 / 1,370 |
| Initial-MAS | 2 / 2 | 16 / 2 | 296,712 / 3,584 | 93.890 / 2.359 | 273,457 / 26,839 |
| Native JIT | 2 / 2 | 6 / 2 | 97,191 / 1,570 | 47.500 / 2.140 | 84,778 / 13,983 |

全库存生成 **24 调用、406,085 tokens、147.500 活动秒**，评价 **6 调用、7,292 tokens、6.858 活动秒**，合计 **30 调用、413,377 tokens**。queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`。同 endpoint / 同模型 private self-judge 只在全生成封存后评分，反馈不回流；两个已开发曝光 source EVO 任务不构成独立 judge 的优势确认，所有原提交保留。

## RR shared evidence v9：三个有效提交与未裁剪原生分

`outputs/development_pilot_20261004_rr_evidence_v9/summary.json` 已封存，注册身份为 `72072910d2da242f7863a09c8687f4dba953c20d49b2c436c93eeed2430aae41`。固定当前 source EVO RR60，三臂同一 frozen packet，**3/3 槽 submitted / completed**，`comparison_complete=true`。下表保留 evaluator 的原生加权 native weighted score（官方加权公式），**不 clip 原生分、不用 normalized 值替换原生尺度**；不发布私有权重或 rubric 文本。

| 方法 | 原生 native weighted score（官方加权公式） | normalized score | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---:|---:|---|---|---|---|
| Direct | 0.200000 | 0.255814 | 1 / 28 | 5,043 / 53,857 | 5.407 / 40.265 | 48,097 / 10,803 |
| Initial-MAS | 0.487500 | 0.523256 | 8 / 28 | 170,972 / 91,742 | 63.219 / 47.094 | 233,704 / 29,010 |
| Native JIT | 0.387500 | 0.430233 | 3 / 28 | 45,804 / 85,229 | 34.187 / 45.047 | 110,268 / 20,765 |

全库存生成 **12 调用、221,819 tokens、102.813 活动秒**，评价 **84 调用、230,828 tokens、132.406 活动秒**，合计 **96 调用、452,647 tokens**。三臂 queue idle、estimated attempts、unknown slots 均为 0，美元费用为 `null`；全部 28 次评价调用/臂计入，生成没有失败，不把较高分答案拿去替换其他轮次。

这个已开发曝光的单题上 Initial-MAS 高于 Direct **0.2875**、高于 Native JIT **0.1**，但自身原生分比 v8 同 packet 的 **0.5**略低；另外两臂也从 v8 的 **0.3625 / 0.4875**到此轮 **0.2 / 0.3875**。这里只陈述各原注册结果，不能从旧 Direct / Native JIT 较高或较低分中自由选择对照，也不把单题优势推成六 benchmark、SOTA、trained 27B JIT 复现或独立 judge 结论。当前任务 ID 与已披露 incidental legacy TEST ID 不同；这个区别不能消除整体开发未知曝光范围，未来干净确认仍遵循 supplemental ledger 的排除 / 审计政策。

## v9 最终完整验证：保留首次失败，重跑已通过

test-only fixture envelope 修正后，最终 full suite 实际结果为 **1,706 passed + 60 subtests，141.74 秒**；**175 个 Git source Python compile 通过，diff check 通过**。原先 **1,705 passed + 60 subtests / 1 failed** 的记录保留，不将第一次运行改写成成功。修正只将离线 mock fixture 的 50,000-token envelope 改为共同 2,000,000，原语义断言不变，生产 source / config 以及注册的 30 个真实槽保持冻结。

上述验证证明冻结版本通过当前代码检查，不单独保证任何 benchmark 的分数，也不以较早轮次较高结果拼成一个版本。发布候选初扫描只对明确 allowlist 的 **39 个 code / test / doc / example config 文件**执行，不穿越 historical evaluation 或 dataset；后续新文件与最终结果文档仍需再次刷新和扫描。

## DRB shared evidence v9：有效三臂、尚未超过 Native JIT

`outputs/development_pilot_20261004_drb_evidence_v9/summary.json` 已封存，注册身份为 `fd3ab67ccf1384bae7ccfc6573059f07b6a2526a17a295141bc8f5a9d03e4c16`。原定 **3/3 槽 submitted / completed**，`comparison_complete=true`，三臂同五份官方 frozen packet，各自完整 58 项评价，无生成失败或评价缺失。

| 方法 | 完成评价结果 | 原生聚合分 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---:|---|---|---|---|
| Direct | 23/58 | 0.396552 | 1 / 2 | 6,447 / 14,556 | 14.375 / 14.453 | 12,900 / 8,103 |
| Initial-MAS | 25/58 | 0.431034 | 8 / 2 | 145,174 / 11,892 | 67.859 / 13.594 | 132,433 / 24,633 |
| Native JIT | 27/58 | 0.465517 | 3 / 2 | 41,337 / 14,176 | 35.094 / 15.984 | 39,988 / 15,525 |

生成合计 **12 调用、192,958 tokens、117.328 活动秒**；评价 **6 调用、40,624 tokens、44.031 活动秒**；合计 **18 调用、233,582 tokens**。queue idle、estimated attempts、unknown slots 为 0，美元费用为 `null`。所有原生分为实际完整评价，不用失败置零替代或改写。

Initial-MAS 比 Direct 多两项通过、比 Native JIT 少两项通过，不能声称领先两个 baseline。对照 v8 同 packet 的 **30/58、16/58、3/58**，三臂变化方向与幅度不同；review/revision 内容保留、penalty 与调用均变化，单题 self-judge 观察不能隔离因果。官方 packet 对未核验的早期增长序列仍明确留空，不为补分编造统计、因果关系或使用被公开任务禁止的文章。显式 TEST 曝光与 incidental unknown 范围照常披露，仍不是 clean formal TEST / 独立 judge 结论。

## WritingBench closed-book v9：完整两题库存与规划阶段失败

`outputs/development_pilot_20261004_writing_v9/summary.json` 已封存，注册身份为 `d6cc137c8b8f8fd92297fcbabfb62ddfb0e7b5876b885a5546ac1fa0b0f0bcba`。原定 **6 槽、5 槽 submitted / completed**；Initial-MAS WB335 在生成阶段 `ValueError`，`comparison_complete=false`，失败保留 `null` 和费用，不用 WB433 的有效分作两题平均。

| 固定任务，原生 mean / 10 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| WB335 | 8.6 | `null` / `ValueError`, submission_failed | 8.6 |
| WB433 | 9.0 | 9.0 | 9.0 |
| 两题完整原生均值 | 8.8 | `null` | 8.8 |
| 两题 normalized failure-zero | 0.866667 | 0.444444 | 0.866667 |

| 方法 | 提交 / 完整评分 / 2 | 生成 / 评分调用 | 生成 / 评分 tokens | 生成 / 评分活动秒 | 全部 input / output tokens |
|---|---|---|---|---|---|
| Direct | 2 / 2 | 2 / 2 | 7,023 / 10,455 | 13.453 / 5.235 | 13,616 / 3,862 |
| Initial-MAS | 1 / 1 | 12 / 1 | 156,002 / 2,900 | 70.704 / 2.110 | 140,008 / 18,894 |
| Native JIT | 2 / 2 | 6 / 2 | 64,852 / 9,652 | 55.141 / 4.890 | 58,239 / 16,265 |

生成合计 **20 调用、227,877 tokens、139.298 活动秒**；评价 **5 调用、23,007 tokens、12.235 活动秒**；合计 **25 调用、250,884 tokens**。queue idle、estimated attempts、unknown slots 为 0，美元费用为 `null`。WB335 的失败本身耗费 **6 调用、99,980 tokens、44.219 活动秒**，已计入总账；没有有效稿可评分，因此少一次评价调用不代表方法更节省。

只读封存失败的 model-call / planning metadata，调用 phase 为 `predict`、三次 `local_plan`、两次 `reconcile`（attempt 0 / 1），全部 `finish_reason=stop`。两次 reconcile 的 JSON 根键均为 `schema_version / graph / team`，各有一个 validation error，最终包含 review-cycle / terminal-reviewer 相关验证标记；这是**规划 reconcile 的失败**，没有进入 Writer execution 或 public refinement，不能归因为 Writer 的 8192 截断或把 local-plan agent 名当作已经执行的成稿。证据保留在生成目录的 `planning_calls.json` / `failure.json`，不打印生成 rubric、题干或稿件正文。两题全报显示 Initial-MAS 没有 WritingBench 两题完整交付优势。

## v9 指令失败的公开机械诊断与 fail-closed 边界

协调器与诊断 agent 对已封存 public-only 日志的核对表明：IFBench13 是 **Writer execution 连续两次各 8192 output tokens、finish length 的重复 prose**，不是 planning 阶段；没有把此失败伪装成 checker 零分。IFBench22 的 numeric revision 返回第一 JSON root 仅 **8 个 slots，而公共请求 / 动态 schema 要求 15 个**，第八个还缺 `number / after`，随后追加了独立 bare `number_slots` fragments，本地报 `JSONDecodeError`。服务没有兑现 requested strict schema 的完整要求，synthetic 三槽 probe 成功不能外推到实际更长约束。

原 response / 账单与 failure artifacts 保留，不从首 root 擅自提取较像答案的片段、不手补 slots / 数字、不按私有分数选择 draft，也不覆盖缺失提交。IFBench22 的失败 locator 为 `generation/167cb165e856c0301562bdd9a2ad605a318f9047127f9928d864901285b2eec2/f124a7a7d76e8715b9ffc05d581a46c44899348db64e504e107dc190c6624625/failure.json`；IFBench13 为 `generation/c6c539f6824ea16878d967def2775a6d28b4028cef12396ea679656974904c32/376ada7749ce1e21428c8dcb600080b31978a8dd0e8f9efd397320511a6dc733/failure.json`，二者均相对于 `outputs/development_pilot_20261004_exposed_v9`。格式计数与公开约束是诊断材料，不复制 private checker keys、rubric 文本或参考答案到生产提示。

## v9 六 benchmark、三十槽的完整汇总

五个 campaign 已全部终结并分别封存，共 **30 固定槽、27 个完整评分、3 个 Initial-MAS 生成失败保持 `null`**：IFBench13、IFBench22、WB335。Direct / Native JIT 各 10/10 完整评分，Initial-MAS 7/10。每个来源的主指标与完整原生均值见首表及逐轮表，WritingBench / IFBench 缺失不以较高成功稿代替；没有挑掉失败后声称全 benchmark 优势。

| 方法 | 完整评分 / 注册槽 | 生成 / 评分调用 | 生成 / 评分 tokens | 全部模型调用 / tokens | 生成 / 评分活动秒 |
|---|---|---|---|---|---|
| Direct | 10 / 10 | 10 / 34 | 33,524 / 81,006 | 44 / 114,530 | 52.094 / 63.375 |
| Initial-MAS | 7 / 10 | 74 / 33 | 1,086,722 / 110,118 | 107 / 1,196,840 | 464.172 / 65.673 |
| Native JIT | 10 / 10 | 32 / 34 | 348,583 / 110,627 | 66 / 459,210 | 270.016 / 69.076 |

生成总计 **116 调用、1,468,829 tokens、786.282 活动秒**；评价 **101 调用、301,751 tokens、198.124 活动秒**；合计 **217 模型调用、1,770,580 tokens**。三个生成失败共 **20 调用、255,655 tokens**全部计入 Initial-MAS 总账；本地 checker 耗时亦在各轮计入评价活动秒。活动时间可能并行重叠，不充作协调器 elapsed；queue idle / estimated attempts / unknown slots 为 0，美元费用仍为 `null`，两个 standalone transport probes 的 **2 调用 / 3,506 tokens**另账。

若对上述六来源的 normalized failure-zero 做**事后描述性等权宏平均**，Direct 为 **0.409422**、Initial-MAS 为 **0.477567**、Native JIT 为 **0.682625**。这是保留全部预注册槽后的派生汇总，不是新增正式 protocol 或 SOTA 主指标，也不能抹去 Initial-MAS 在 Native JIT 后面的源结果、三个失败与更高费用。RR 单题高于两条 baseline、IFEval 与 Native JIT 持平，DSQA / DRB 尚未超过 Native JIT，WritingBench 两题不完整，IFBench 两题均失败；不能得出稳定全面优于 single-agent 与 Native JIT 的结论。

后续约束专属源码修改如需验证，必须另冻结 identity / registration 并单独报完整固定槽；未注册、未生成的候选不预写结果。即使新指令轮完成，也不能把其分数与 v9 普通任务较高行拼成一个版本覆盖六 benchmark 的优势声明。所有 v9 原 submissions、failures、seal、evaluation 与 summary 保留。

## 发布候选初扫描：限定 39 文件，后续仍须刷新

`.runtime/publication_audit_v9_initial_20261004.json` 记录对明确 allowlist 的 **39 个文件**所做初扫描：真实 DPAPI key 的 UTF-8 / UTF-16 / URL-encoded / base64 形式命中 **0**；reference-answer / private-rubric 长字面量候选 **0**；仅用既有 selected ledger 的 **5 个 question SHA256 元数据**对候选解码字符串作精确复制比较，命中 **0**。不打开历史 evaluation、dataset、原题或 private reference answer，不在 `paper/experiments` 递归正文搜索。

三个 credentialed-URL 候选与一个长凭据候选分布于 `tests/jit_mas/test_api_probe.py` / `test_development_pilot.py`，对应已审查且文件 hash 不变的合成拒绝 / 反射 fixtures，真实 key 均未命中，未解决候选数 **0**。该检查不证明不存在局部题干复制或未知历史 private 内容；它没有加载这些正文作全量比较，也不能消除 supplemental incidental exposure。历史源输出、private dataset、`.runtime` 和机器 helper `scripts/wait_v37_then_launch_v38.py` 明确不发布。

这只是当时 39 文件的初快照，文档完成与后续新增模块 / 测试后必须重新生成 allowlist、复查当前内容并做最终扫描；不把旧 file hashes 当作最终工作区证明。本文的追加不改变 runtime，扫描 **0 model API、0 stage / commit / push**；发布由协调器按经审查的具体 allowlist 进行。

## v10 独立合成兼容性检查：三个失败与一个成功均保留

在任何 v10 benchmark 注册 / 生成前，协调器对完整 public review + revision 做了四个有实际方法差异的 synthetic probe。它们使用合成公开约束，不做 official checker / rubric 评分；不是固定 benchmark 槽的重采样，也不能把最后一次成功当作 benchmark 改善。

| 独立 artifact（均在 `.runtime/`） | 完整调用 / tokens | input / output tokens | API 活动秒合计 / ledger 活动秒 | 实际结果 |
|---|---|---|---|---|
| `full_public_refinement_probe_v10_20261004.json` | 2 / 7,047 | 6,716 / 331 | 3.203 / 3.875 | `ValidationError`；named numeric schema 的 before / after 为空串 |
| `full_public_refinement_probe_v10b_20261004.json` | 2 / 7,129 | 6,807 / 322 | 3.109 / 3.797 | `ValidationError`；增加通用 nonblank clause 提示后仍为空串 |
| `full_public_refinement_probe_v10c_20261004.json` | 2 / 7,204 | 6,805 / 399 | 3.391 / 3.969 | `ValidationError`；construction 改 JSON object transport，完整 prose 的 opening / context 重复数字，严格本地校验拒绝 |
| `full_public_refinement_probe_v10d_20261004.json` | 2 / 6,692 | 6,414 / 278 | 3.141 / 5.219 | completed；全文 alphabetic marker template 与 typed 数字 mapping 通过本地校验，revision 57 output tokens |

四次合计 **8 模型调用、28,072 tokens（26,742 input / 1,330 output）、12.844 API 活动秒、16.860 ledger 活动秒**；estimated / unknown 为 0，request queue idle 为 0，美元费用为 `null`。四个 artifact 均保留完整 raw responses、audit 和费用，所有响应为 finish stop；前三次的失败账单未删除或转移到 benchmark 总账。v9 的两个 transport probes **2 调用 / 3,506 tokens**与这里四次亦分别保留。

v10d 使用自然全文中的唯一 alphabetic markers 和 typed 数字映射，render 仅替换 marker，不追加词句；minimum FANBOYS 要求用词法计数验证。JSON object transport 与严格本地 schema / marker / 数量检查共同工作，不靠提取第一 root、补缺失数字、删除额外内容或 fallback 接受失败响应。a / b / c / d 的 prompt、布局或 transport 有真实变更，不能用四次顺序断言某一因素的确定因果，也不能从 synthetic 成功预写真实 IFBench 结果。

## v10 仅指令任务：注册与冻结方法（原注册记录）

`outputs/development_pilot_20261004_exposed_v10/registration.json` 已存在，注册身份为 **`ffebb51b4d70513cb25763e8efca0dbb302f56b901be16991e8008d660ac2270`**，runtime fingerprint 为 **`db23cfede5116c2ea0c2c1086f4c256a91c0344838ebefd466f5b4c13a301140`**，runner SHA 为 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。此轮仅固定此前已曝光的 IFEval 两题与 IFBench 两题，每题三臂，共 **12 固定槽**，按 v5 task_index 原序过滤 TEST，不增题、不依分数换序；selected ledger 仍为原五 ID，本轮仅追加 registration reference，incidental legacy unknown 范围仍保留。

已注册配置启用 `public_positional_draft_guidance=true`，只在公共约束可机械编译的位置分支提供内部 draft guidance；numeric construction 使用 `public_numeric_construction_layout=template`，采用完整自然 prose、唯一 alphabetic markers 与 typed 数字 mapping，array / named 形式仍是可选通用接口。construction transport 为 `public_construction_response_format=json_object`，本地严格校验不变；review mode 仍为 `json_schema_review`。所有角色与 revision 的 frequency penalty 均为 `None`；普通非可编译任务的 draft / review / revision 组织未以 benchmark ID 路由。

三臂同 8192 execution / judge output cap、其他组织角色 16000、2,000,000 task tokens、900 active seconds、无 model-call cap、`candidates=1`；900 秒由生成与其 deferred evaluation 共享，不计跨槽调度和请求 queue idle。Initial-MAS 从空经验状态开始、不写回；全部 12 次生成终态先封存，再调用 pinned author instruction checker，失败保留完整费用与 `null`，不重采或择优。此指令轮无模型 judge，来源仍明确为 **exposed TEST development only**，不修改 immutable formal v5 manifest。

协调器报告 public 15 modules 的 targeted validation 为 **716 passed**；v10 full suite 尚在执行，在获得实际结果前不沿用 v9 的 1,706 passed 作为新版本验证。此处不读取 live 生成或预填分数。v10 的 12 指令槽若完成，将单独报告全库存和费用，不能与 v9 普通任务的较高行拼成同一版本覆盖六 benchmark 的结论。

## 发布候选文件名刷新：44 文件，尚未最终扫描

只依据 Git changed / untracked 文件名刷新 allowlist，实际为 **44 文件：13 tracked changed、31 untracked candidates**。相较 39 文件快照，新增 `jit_mas/public_numeric_template.py` 与 `test_public_numeric_named_objects.py`、`test_public_positional_draft_guidance.py`、`test_public_numeric_template.py`、`test_public_construction_transport.py`；数组 / named / template 与 transport 的测试均纳入候选。

候选名单位于 `.runtime/publication_allowlist_20261004.txt`，metadata 位于 `.runtime/publication_allowlist_candidate_20261004.json`，当前状态为 filename snapshot only / pending final scan。没有扫描正在变化的内容，不把 39 文件的旧 hashes 当作新 44 文件的证明；最终代码、测试与结果文档完整后再复核。机器 helper `scripts/wait_v37_then_launch_v38.py`、`.runtime`、`outputs`、dataset 与 private benchmark 内容仍明确排除；本次刷新 **0 model API、0 stage / commit / push**。

## 44 文件发布隐私扫描：当前快照已审查，最终文档仍需重扫

`.runtime/publication_audit_v10_preseal_20261004.json` 保留自动扫描原始候选，`.runtime/publication_audit_v10_preseal_reviewed_20261004.json` 保留人工归类后结果。只读取 allowlist 的 44 个候选文件与已知五题 SHA 元数据，真实 DPAPI key 仅通过临时进程环境导入：原文 / UTF-16 / URL-encoded / base64 命中 **0**，已知完整 selected question 的精确 hash 命中 **0**，个人绝对路径命中 **0**；没有展开 historical evaluations / dataset 或加载原题、private reference answers。

自动候选仍如实保留：长凭据字面量 **1**、credentialed URL **3**，均为 `test_development_pilot.py` / `test_api_probe.py` 已审查且 hash 不变的人工合成反射 / 拒绝 fixtures。新候选 **2**分别为 `jit_mas/experiment_methods.py` 的一个通用 loopback 接口地址，以及 `tests/jit_mas/test_public_positional_draft_guidance.py` 的一个 reference-answer 负向 distractor；后者作者明确确认是人工合成 public-only 边界测试，不来自 benchmark 参考答案、private rubric、评分记录或 incidental archived 内容。未输出匹配值，未移除有效测试，归类后 publication findings remaining 为 **0**。

所有 44 个 hash 在人工审查时复核一致，因此该快照未发现需阻止发布的凭据或完整已知题目复制。此检查不排除局部题干或未知历史私有材料，也不消除曝光 ledger 的 unknown 范围；补结果文档或后续源码变更后必须再扫描。此轮 **0 model API、0 source changes、0 stage / commit / push**，不会把初扫描状态当作最终发布凭证。

## v10 已封存生成的公开机械诊断：失败从 Writer 移至 refinement

协调器确认 v10 的 **12 个固定生成槽均已终态且 release seal 存在**，author grading 仍在执行，尚无最终 summary；这里不预填 official 分数。Initial-MAS IFBench13 的原生成失败仍为 `ValidationError / null`，耗费 **6 模型调用、64,557 tokens**，不重采或择优。

公共 generation audit 显示，此题的 conditional positional draft guidance 已使 Writer **一次调用、847 output tokens、finish stop**，没有复现 v9 的两次 8192-token 重复截断。失败发生在 public refinement：review **1,266 output tokens**、revision **945 output tokens**，两者 finish stop，JSON 完整解析、词槽数量 / 前后位置结构通过。本地校验留下 **4 项**句子 / 词边界问题：一个 dotted title / internal period，两个 prefix 与一个 suffix 的 apostrophe possessive；这不是 JSON root 不完整、截断或 checker 零分。

上述诊断只说明一个已封存任务的故障阶段变化，不能当作 IFBench 成功率提升、完整方法优势或 v10 official score。后续对 contraction / possessive / hyphen / title abbreviation 的通用提示修改如要验证，必须另冻结和注册固定槽，保留 v10 原失败和账单；当前 v10 源码不据正在评分的结果改变。正式分数与全成本仍待完整 sealed summary，结果轮次分别报告。

## v10 最终封存结果：一个 numeric 成功，仍未整体领先

`outputs/development_pilot_20261004_exposed_v10/summary.json` 已封存，协调器 exit 0，注册身份为 `ffebb51b4d70513cb25763e8efca0dbb302f56b901be16991e8008d660ac2270`。原 **12 固定槽、11 submitted / completed、1 Initial-MAS 生成失败**，`comparison_complete=false`；未提交失败的 score / native_score / normalized_score 全部为 `null`，不是 observed 0。此前“评分仍在执行”的段落是当时诊断快照，此处补录最终结果，原 submissions / failures / seal / scoring artifacts 不变。

| 固定任务及主指标 | Direct：主分 / instruction counts | Initial-MAS：主分 / instruction counts | Native JIT：主分 / instruction counts |
|---|---|---|---|
| IFEval1203，strict prompt | 1 / 2/2 | 0 / 1/2 | 0 / 1/2 |
| IFEval1246，strict prompt | 1 / 1/1 | 1 / 1/1 | 1 / 1/1 |
| IFBench13，loose prompt | 0 / 0/1 | `null`，`ValidationError / submission_failed`，无 checker score | 1 / 1/1 |
| IFBench22，loose prompt | 0 / 0/2 | 1 / 2/2 | 1 / 2/2 |

Instruction counts 是 secondary 诊断，不能替换 strict / loose prompt 主指标。IFEval 两题均值 Direct **1.0**、Initial-MAS / Native JIT 各 **0.5**；IFBench 完整均值 Direct **0**、Native JIT **1**、Initial-MAS **`null`**。把未提交槽保守置零的 IFBench normalized 均值是 **0.5**，两来源 macro Direct / Initial-MAS / Native JIT 为 **0.5 / 0.5 / 0.75**，不把失败置零改写为评分结果。

| 方法 | 提交 / 完整评分 / 4 | 生成调用 / tokens | 本地 checker 活动秒 | 生成活动秒 | input / output tokens |
|---|---|---|---:|---:|---|
| Direct | 4 / 4 | 4 / 2,801 | 4.063 | 14.578 | 901 / 1,900 |
| Initial-MAS | 3 / 3 | 29 / 299,377 | 2.812 | 134.282 | 267,692 / 31,685 |
| Native JIT | 4 / 4 | 12 / 93,875 | 3.814 | 95.468 | 67,277 / 26,598 |

生成合计 **45 模型调用、396,053 tokens（335,870 input / 60,183 output）、244.328 活动秒**；author checker 无模型调用 / tokens，本地评价活动秒合计 **10.689**。Initial-MAS IFBench13 的 **6 调用、64,557 tokens**失败费用已计入总账；全部三臂 queue idle、estimated attempts、unknown slots 为 0，美元费用为 `null`，活动秒不是并行协调器 elapsed。四个 v10 synthetic probes **8 调用 / 28,072 tokens**单独另账，未放入这些 benchmark 槽。

IFBench22 从 v9 的生成失败变成真实完整通过，是该轮一个 numeric 工程与约束交付改善；不能据此忽略 IFEval1203 的实际 0、IFBench13 的缺失或更高成本。v10 相比 v9 的指令 macro，Initial-MAS 仍 **0.5**、Native JIT 仍 **0.75**；不同轮 prompt / layout / transport 有变动，单次固定开发小样本不能隔离因果。当前结果不足以认定 Initial-MAS 全面优于 Direct / Native JIT，不与 v9 的普通任务较高行拼成同版六 benchmark 优势。

v10 full suite 仍待实际完成报告，当前不把 targeted 716 passed 或 v9 1,706 passed 当作新全测试通过。任何下一轮通用词槽提示修改只能在 v10 封存后另注册、报告固定完整库存，原失败保留；本次文档追加与后续隐私扫描均不调用模型 API、不改生产、不 stage / commit / push。

## v10 最终完整验证：实际 1,964 passed，不沿用旧版本

`.runtime/final_v10_validation_baseline_runtime.json` 已记录最终 full suite **1,964 passed + 60 subtests，457.68 秒**，**180 个 Git source Python compile 通过，diff check 通过**。该验证只运行一次 full suite，registered code / model metadata / dataset identity 匹配、formal v5 字节不变；原 incidental exposure 与 ignored generated-code compile 诊断照常保留。验证本身 **0 model API、0 production changes**。

这补录了上文曾“仍待完成”的真实终态，验证的是已注册、已封存 v10 快照，不能自动当作后续 prompt 修改的 full-suite 证明，也不能把代码测试通过等同 benchmark 全面领先。v10 固定12槽仍保持 11 graded / 1 generation failed、三臂 macro **0.5 / 0.5 / 0.75**，原失败费用不变。

## v11 词槽通用提示开发：两个独立合成 probe，首失败保留

v10 封存后，协调器只加强公共词槽 construction prompt：明确 apostrophe / possessive / contraction / hyphen / dotted-title 禁止要求与动态自查，不改 schema、validator 或其他任务分支。首版 synthetic 完整 review + revision 仍复制 dotted title，`ValidationError`，原 response、audit 与账单保留；随后实际 prompt 增加正向 `Doctor / Professor / of` 表达，并声明 construction 词法要求优先于 draft / critic 的 typography，第二次完整 synthetic 通过。

| 独立 artifact（均在 `.runtime/`） | 结果 | 模型调用 / tokens | input / output tokens | API 活动秒 / ledger 活动秒 |
|---|---|---|---|---|
| `full_public_refinement_probe_v11_pos_20261004.json` | `ValidationError`；dotted title 未去除 | 2 / 8,972 | 7,960 / 1,012 | 5.828 / 7.860 |
| `full_public_refinement_probe_v11b_pos_20261004.json` | completed；新正向措辞与 construction precedence | 2 / 9,255 | 8,099 / 1,156 | 6.313 / 8.422 |

两个合计 **4 模型调用、18,227 tokens（16,059 input / 2,168 output）、12.141 API 活动秒、16.282 ledger 活动秒**；全部响应 finish stop，queue idle / estimated / unknown 为 0，美元费用 `null`。这两次是不同实际 prompt 的合成兼容性检查，无 official checker / model judge 评分，不是 benchmark 槽重采样；没有将首失败答案选作后一次提交。v10 的四个 probes **8 调用 / 28,072 tokens**、v9 两个 probes **2 调用 / 3,506 tokens**继续分账，不漏开发成本。

当前 `jit_mas/public_word_slots.py` 字节 SHA256 为 **`2a0c9e91a4d47ccd628b64d986e57080a1dd32828c1bc9e5d2b9b451605fd366`**。此前旧 prompt 的 101 个相关 tests 属于旧 `0b90…` hash；当前 101-test 重跑尚待真实结果，不把旧测试套到这个版本，也不重复声称新 full suite 已通过。真实 benchmark v11 的 registration / 生成 / sealed summary 尚未在此补录，后续只能针对同两题 IFBench 的固定六槽分别报告；不会多运行 IFEval 或拼成同版六 benchmark 结论。最终结果文档完整后按44文件 allowlist 再做隐私审查，不读取额外问题 / private评分、不改源、不调用模型、不 stage / commit / push。

## v11 IFBench 固定六槽注册：等待终态与封存评分

`outputs/development_pilot_20261004_ifbench_v11/registration.json` 随后注册成功，身份为 **`281eeff32ffb41bcd77e6ab11a3b98ecce7815315883480c37f2c65daa2b1b8f`**，runtime fingerprint **`778540c6d6ee21d10e6034e448385631276a37fb4acbfa5ea02774b5898e8274`**，runner SHA 仍为 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。固定此前 IFBench13 / 22 两题各三臂，共六槽，仍为 exposed TEST development only，按原 membership / task_index 顺序，不据 v10 成败换题。

五个 selected ID 不变，exposure ledger 仅追加本 campaign registration reference，现 **11 个 references**；supplemental incidental legacy unknown 范围照常保留。第二个 synthetic probe 的 parsed / render 检查成功仅说明本地结构兼容，未做 official checker；不能当作这六槽的预先成绩。真实生成与评价待全库存终态、seal、summary 后分别补录，不把 v10 的 12 槽或 v9 六 benchmark 表替换成跨版本择优数据。注册前的本地 CLI import 启动失败未调用 API、未耗用任何固定槽，不计为 benchmark 方法失败。

## v11b 当前源码验证：101 个相关 tests，180 个 compile

`.runtime/final_v11b_targeted_validation_baseline_runtime.json` / `final_v11b_registration_identity_baseline_runtime.json` 记录最终 prompt 的相关 **101 tests passed、2.89 秒、exit 0**，**180 个 source Python compile 通过**。runtime **`778540c6d6ee21d10e6034e448385631276a37fb4acbfa5ea02774b5898e8274`**、runner **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**与 v11 registration 一致，word slots SHA 为上文 **`2a0c9e91…`**。

旧版 `0b90…` hash 的 101 tests / 2.65 秒不沿用。当前没有重复 full suite：v10 的 **1,964 passed + 60 subtests**仅属于旧完整验证快照；上述 targeted / compile 才对应当前词槽 prompt。源码测试通过仍未保证固定 benchmark 交付或主指标通过。

## v11 最终 IFBench 六槽：两个 Initial-MAS 生成失败均保留

`outputs/development_pilot_20261004_ifbench_v11/summary.json` 已封存、协调器 exit 0，固定六槽 **4 submitted / completed、2 Initial-MAS generation failed**，`comparison_complete=false`。失败字段全部 `null`，没有 official checker score，更不能从两次 synthetic 的完成状态给失败虚构评分。

| 固定任务，loose prompt 主分 / secondary counts | Direct | Initial-MAS | Native JIT |
|---|---|---|---|
| IFBench13 | 0 / 0/1 | `null`，`JSONDecodeError / submission_failed` | 1 / 1/1 |
| IFBench22 | 0 / 1/2 | `null`，`ValidationError / submission_failed` | 1 / 2/2 |
| 两题完整主指标均值 | 0.0 | `null` | 1.0 |
| 全固定库存 normalized failure-zero | 0.0 | 0.0 | 1.0 |

| 方法 | 提交 / 完整评分 / 2 | 生成调用 / tokens | 生成活动秒 / 本地 checker 活动秒 | input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 2 / 770 | 5.203 / 1.858 | 483 / 287 |
| Initial-MAS | 0 / 0 | 13 / 145,421 | 99.719 / 0 | 116,520 / 28,901 |
| Native JIT | 2 / 2 | 6 / 52,516 | 55.890 / 1.688 | 36,120 / 16,396 |

合计 **21 生成模型调用、198,707 tokens（153,123 input / 45,584 output）、160.812 生成活动秒**，author checker **0 模型调用 / tokens、3.546 本地活动秒**。两个 Initial-MAS 失败共 **13 调用、145,421 tokens**已全部计入：IFBench13 **7 / 91,356**、IFBench22 **6 / 54,065**。queue idle / estimated attempts / unknown slots 为 0，美元费用 `null`。没有提交使 checker 用时为0，不代表生成更经济。

公开失败诊断显示，IFBench13 Writer 首次 **8192 output / finish length**后，任务内固定修正调用 **1502 output / finish stop**；最后 revision 又达 **8192 output / finish length**、JSON 未闭合，本地 `JSONDecodeError`，没有走到 lexical validator。这是一个注册任务内的执行 / 修正流程，所有调用计费、最后固定提交失败保留，没有另起任务重采。IFBench22 仅先记录实际 `ValidationError` 与完整失败费用，阶段尚待只读公共日志核实，不擅自把它归为词槽或 numeric validator 问题。

本轮没有 Initial-MAS 整体领先：两个槽均不可评分，而 Direct / Native JIT 各两槽完整；v10 numeric 成功仍保留在原版本，不能拿来替换此轮 IFBench22。词槽通用提示的 synthetic 成功没有外推成真实任务成功，也没有选择较高旧 round 作为最终优势证据。

## v12 配置变体：源码不变、两次独立合成成功、六槽已注册待封存

v11 全部终结后，协调器只改变两项配置：`public_construction_response_format` 从 **`json_object`**改为 **`json_schema`**，`public_revision_frequency_penalty` 从 **`None`**改为 **`0.25`**；其他模型角色仍全部 `None`。word-slot prompt、schema / validator 和其余 runtime source 不变，source fingerprint 仍 **`778540c6…`**、word slots SHA 仍 **`2a0c9e91…`**，当前 source 对应的 v11b 101-targeted / 180-compile 证明保持适用，不能据此宣称新 full suite 已跑或隔离某个配置因素的效果。

| 独立 synthetic artifact（`.runtime/`） | 结果 | 调用 / tokens | input / output tokens | API 活动秒 / ledger 活动秒 |
|---|---|---|---|---|
| `full_public_refinement_probe_v12_pos_20261004.json` | completed | 2 / 9,013 | 8,020 / 993 | 6.142 / 8.406 |
| `full_public_refinement_probe_v12_numeric_20261004.json` | completed | 2 / 6,747 | 6,434 / 313 | 3.343 / 5.109 |

两次全流程合成检查共 **4 调用、15,760 tokens（14,454 input / 1,306 output）、9.485 API 活动秒、13.515 ledger 活动秒**，estimated / unknown / queue idle 为0、费用 `null`；没有模型或 official checker 评分，单独计入开发账，不是 benchmark 重采。全部原 artifacts 保留，不能用它们预写真实六槽成绩。

`outputs/development_pilot_20261004_ifbench_v12/registration.json` 已注册，身份 **`a4fb54ecdbeb165315d5022360d93eb785f289063eca9303f5302a323c6e78e6`**，previous iteration 绑定 v11。同 IFBench13 / 22 各三臂、共六固定槽，source / runner 身份不变、配置身份另冻结；全部生成终态后封存再用 pinned author checker，不替换 v11 的两个失败、不加 IFEval / 普通源。曝光 ledger 仅新增这个 registration reference，仍为原 **5 selected IDs、12 campaign references**，supplemental unknown 曝光照常披露。待六槽 sealed summary 才补实际分数和全成本，最终结果文档后按精确44文件 allowlist 直接重扫；当前 **0 model API、0 source changes、0 stage / commit / push**。

## v11 只读诊断补录：JSON 截断与重复 template marker 分别失败

随后公共 generation audit 明确了 v11 IFBench22 的阶段：三次 planning、Writer **639 output**、review **267 output**均成功，最后 template revision **479 output / 1,780 chars / finish stop**、完整 JSON。15 个 numeric values 的 closed keys 与 canonical 数字全部合法，template 无额外 Unicode decimal number / unknown marker，FANBOYS 类别满足；唯一重复 marker 出现两次，其他14个各一次，合计16处而非15处，本地 strict uniqueness 拒绝。因此实际是 **final template revision 的 `ValidationError`**，不是 planning、数字解析或 missing typed keys；没有删掉重复 marker、放宽检查或手补结果。

v11 IFBench13 的未闭合 revision 则在第24个 preceding string 重复成 **34,839 chars**，prefix / suffix 未输出。前23个 preceding chunks 没有缩写点，仅能描述已收到部分；不能因此声称 apostrophe / possessive / dotted-title 的目标问题已全面消除，后续字段根本未到达。本地在 JSON parse 阶段失败，没有进入完整 lexical validator。上述具体机械诊断不改变两项原失败 `null`、费用或 official 无提交状态。

## v12 最终 IFBench 六槽：numeric 通过与两臂失败一起报告

`outputs/development_pilot_20261004_ifbench_v12/summary.json` 已封存、协调器 exit 0，原六槽 **4 submitted / completed、2 generation failed**，`comparison_complete=false`。Initial-MAS / Native JIT 的 IFBench13 均为 `RuntimeError / submission_failed`，没有观测分数，仍保留 `null`。

| 固定任务，loose prompt 主分 / secondary counts | Direct | Initial-MAS | Native JIT |
|---|---|---|---|
| IFBench13 | 0 / 0/1 | `null`，`RuntimeError` | `null`，`RuntimeError` |
| IFBench22 | 0 / 1/2 | 1 / 2/2 | 0 / 1/2 |
| 两题完整原生均值 | 0.0 | `null` | `null` |
| 两题 normalized failure-zero | 0.0 | 0.5 | 0.0 |

| 方法 | 提交 / 完整评分 / 2 | 生成调用 / tokens | 生成活动秒 / 本地 checker 活动秒 | input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 2 / 8,995 | 26.922 / 1.922 | 483 / 8,512 |
| Initial-MAS | 1 / 1 | 11 / 113,299 | 86.562 / 0.797 | 88,491 / 24,808 |
| Native JIT | 1 / 1 | 7 / 80,508 | 92.468 / 0.812 | 49,903 / 30,605 |

全库存生成 **20 调用、202,802 tokens（138,877 input / 63,925 output）、205.952 活动秒**；本地 author checker **0 模型调用 / tokens、3.531 活动秒**。两个失败已全部计入各臂预算记录，queue idle / estimated attempts / unknown slots 为0、美元费用 `null`。此前两个配置 preflight synthetic **4 调用 / 15,760 tokens**另账保留，不当作 benchmark 提交。

Initial-MAS IFBench13 公开阶段为三次 planning finish stop 后，Writer 两次各 **8192 output / finish length**，未进入 public refinement；不能将这项 RuntimeError 归因于 strict construction transport 或 revision `0.25` penalty，也不能从未调用的 refinement 解释 Writer 重复。Numeric 任务的真实1分使 Initial-MAS 两题 failure-zero 高于本轮另两臂，但两题完整均值缺失、词槽任务不可评分、样本仅两个已曝光任务，不能称作整体方法稳定领先或用 Native JIT 的新失败替换其旧完整通过来造优势。

v12 两项配置同时变化，single fixed development round 没有隔离单一因素；source fingerprint **`778540c6…`**与 word-slot SHA **`2a0c9e91…`**仍未变化。v11b **101 targeted passed / 180 compile**继续对应此 source，v10 full suite **1,964 + 60**仅属于其旧快照。全部 v11 / v12 原失败和更早结果保留，各轮报告而不择优拼成一个版本。

## 后续温度迁移假设：来源属于 R1，当前模型并非 R1

[DeepSeek-R1 官方 README 的 Usage Recommendations](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md#usage-recommendations) 对 **R1 系列**建议温度 **0.5–0.7、推荐0.6**，目的是减少重复或不连贯输出。协调器拟将五个角色统一 temperature **0.6**、三臂同配置，其他项与 v12 一致，作为最后固定 IFBench 六槽的配置变体。这只是从 R1 向当前 **DeepSeek-V4-Flash-Vision-Exp**迁移的开发假设，不是当前模型的官方配置规范、因果证明或预期得分保证；没有因此引入按 private 分数择优的重采样。

实际端到端 MAS synthetic pipeline preflight 正在执行，覆盖不同公开合成小说约束与初稿 Writer，而非只测试 refinement。首 helper 因 evaluator constructor factory 的本地接口错误在 model call 前失败，artifact 保留、**0 模型调用 / 0 benchmark 槽耗用**；修正 helper 只让实际 evaluate 调用才拒绝评分，没有 grader 调用。新的 preflight 结果、费用和任何后续 registration / sealed benchmark 分数尚未在本文补录，不能预先称成功；待真实终态后分别计账。本文这一阶段只读结果、写文档、**0 model API / 0 source changes / 0 staging**。

## v13 温度端到端合成 preflight：初稿失败，没有 benchmark 注册

`.runtime/full_mas_temperature_v13b_20261004/preflight.json` 随后实际终态为 **failed / `RuntimeError`**。五角色配置均为 temperature **0.6**，仅执行 Initial-MAS 的合成公开小说任务，覆盖 planning 与初稿 Actor；不是三臂 benchmark，也没有任何 v13 六槽 registration 或分数。

已计账 **4 模型调用、43,296 tokens（31,658 input / 11,638 output）**，API 记录用时 **38.498 秒**，budget 的 `wall_seconds` 字段 **40.594 秒**、request queue idle **0.015 秒**，estimated attempts 为0、美元费用 `null`。三次 planning finish stop，初稿 Writer **8192 output / finish length**；其已生成 `AgentSpec.max_calls=1`严格耗尽，因此没有追加 Actor 调用，没有进入 public refiner、没有 grader API。这个组织内 AgentSpec cap 是实际生成 harness 的限制，独立于实验共同 model-call-cap `null`，不隐瞒两者区别。

此前 evaluator constructor factory 的本地 `AssertionError` artifact 原样保留，**0 模型调用 / 0 benchmark 槽耗用**；修 helper 后的模型调用不是据低 official 分数重采，也没有覆盖原失败账。完整 pipeline 结果说明 R1 温度迁移假设与 refinement-only synthetic 成功都未解决这个初稿交付问题，不能称作当前 V4 Exp 模型的实证改善或稳定优势。

下一版 public positional draft projection 与 literal-frequency 预分配仍是待冻结、待新注册验证的开发变更。正式 source / configuration / campaign identity 与测试尚未在此补录，不预写下一版成功、hash 或完整六 benchmark 得分，也不按各历史轮较高行拼接结果；所有原 round 保留，最终发布 allowlist 与隐私扫描等待实际新文件和最后 sealed summaries。

## v14 统一开发版本：源码已冻结，三十槽先注册后运行

v14 source fingerprint 为 **`9999e59640fb1551d5803343120bf8811592087580ca86904ff02c41667f94f6`**，runner SHA 仍为 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。当前相关16模块实际 **456 passed + 38 subtests、76.39 秒**，**181 个 Git Python compile / diff check 通过**；新增54行为 tests 的 `test_public_positional_draft_projection.py`。新版 full suite 已独立启动，尚待实际结果，不能沿用 v10 的 1,964 passed 或旧 v11b 的101 tests 当作这个 source 的完整验证。

已冻结配置为四个生成角色 **meta / global / local / exec temperature 0.6**，**judge 0.0**；role frequency penalties 全 `None`，只有最终 public revision 的 component penalty **0.25**。Construction transport **`json_schema`**、numeric template、positional construction / guidance / projection 启用。v13 的 unused judge 0.6从未实际评分，v14独立保留 judge 0，避免把 evaluator 的解码变化与 generation 变体混在一起。此 source 还增加通用 literal-frequency 缺口预分配提示：minimum 允许小幅余量，exact / upper bound 保持上界；不硬码 benchmark 词或数量，不改变调用、parser、validator 或私有评分。

Public positional draft projection 只在 execution Actor 的公共任务 deep copy 中推迟完整、独立、正向且可编译的位置指令；planning、原任务、final refinement 和评分保留完整原输入。已识别的 quoted / negative / conditional / scope 歧义触发 audit 记录的原子保守 decline，整个 Actor 任务维持原样；其他约束、证据与 public rubrics 不删除，上游贡献 / rubric 仍可能重引位置，不能声称彻底隔离了所有精确位置负担。库默认关闭，必须同时启用 refinement / positional construction / positional draft guidance 三个前置 flag；这是方法候选，不是缺少评分约束的捷径。当前v14已知未加冒号的“逐字引用下一段”与 lazy blockquote scope guard 缺口，尚未改源，不能称所有引文都被拒绝；后续修补必须另记录 source 身份与验证，不把 v14 分数当作补丁后的结果。

五个 campaign 已全部注册，再按顺序运行，**10 固定任务 × 三臂 = 30 固定槽**，同 source / config，不用较早轮次的高分拼成新版本。下列当前只填身份、范围和输入；真实分数、失败与费用在各 campaign 全生成终态、seal、saved summary 后补录。

| campaign 目录后缀（均为 `outputs/development_pilot_20261004_…`） | 来源 / 分区 / 输入 | generation slots | registration hash |
|---|---|---:|---|
| `exposed_v14` | IFEval2 / IFBench2；exposed TEST；closed-book | 12 | `31f64200b1768de4edb73d77ccd189c994f6399d55ec39ef1ce2221ca19b4038` |
| `dsqa_evidence_v14` | DSQA2；source EVO；同两份 frozen packs | 6 | `2447fd52b234c8ab0eac88afab99ac599fbca768b298a6e48f679d8d07badcaf` |
| `rr_evidence_v14` | RR1；source EVO；同 frozen packet | 3 | `2ac1248589b46b30b150f8609db81a1c8894265c43373da7524320a06439cbcb` |
| `drb_evidence_v14` | DRB1；exposed TEST；同五份官方文档 | 3 | `a4b918c52f0f404359d0269eadaca5b42e7cf86c863a9234336e6ac146666d8c` |
| `writing_v14` | WritingBench2；source EVO；closed-book | 6 | `1c0fe32f0e13028e476b5c7325e1358730ae7e2137d9f32647b5898a868bf2ca` |

Parent iteration 绑定指令的 `exposed_v10`与其余各自`*_v9`，不以中间单源最高行替换。Exposure ledger 为该统一 batch 追加五个 registration references，现 **17 references**；selected TEST 原五ID不变，EVO campaign references 只记录共同 batch 身份，不新增 TEST 选题，supplemental incidental unknown 范围照常披露。正式 v5 immutable manifest 不变，初始经验仍为空、不更新，主表最终只呈现统一 v14 的全部六来源，原历史保留。

## v14 端到端合成 preflight：完成交付，不是 official 分数

`.runtime/full_mas_temperature_v14_20261004/preflight.json` 实际 completed、outcome **`submitted_unscored`**，**6 模型调用、74,096 tokens（66,713 input / 7,383 output）**；API 记录用时 **34.640 秒**，budget `wall_seconds` **37.625 秒**、queue idle / estimated 为0、美元费用 `null`。所有响应 finish stop，Actor Writer **1629 output**、public review **1493 output**、revision **817 output**；没有 evaluator / grader 调用，model identity 核验匹配。

Execution metadata 的 projection active 为 True，原任务与最终完整约束保留，revision public diagnostics 的合成位置计数通过。此任务使用不同公开小说约束，非benchmark题、未调用 author checker；因此只证明该 preflight 的完整工程路径交付，不将结果算作三十槽成绩或认定温度 / projection 的独立因果作用。v13 的4调用失败与0调用 helper 错误全部保留，没有用此成功覆盖原账单。

公开 `configs/development_pilot.example.json` / `.md`已同步上述代表性组合并通过 MASConfig 本地验证：四个生成角色0.6 / judge0、projectionTrue / strict construction / revision-only0.25，仍只含 `api.example.com`、`YOUR_MODEL_ALIAS`、`JIT_BENCHMARK_API_KEY`、`expected_response_model=null`，没有个人 endpoint / key / 本机路径。文档保留库默认关闭、unsupported参数 fail-closed / no fallback 边界，不据组合配置声明跨 benchmark 改善。

只按实际 Git 文件名刷新 publication allowlist，现 **45 文件：13 tracked changed、32 untracked candidates**，比44文件只新增 `tests/jit_mas/test_public_positional_draft_projection.py`。Machine helper、outputs、dataset、`.runtime`仍排除；当前只是文件名快照，最终内容隐私审查等待全source与三十槽结果文档完整。本阶段 **0 model API、0 Python production / test edits、0 stage / commit / push**。

## v14 完整验证：2,018 passed，已知 scope guard 缺口单独保留

`.runtime/final_v14_validation_baseline_runtime.json` 记录一次新版 full suite 实际 **2,018 passed + 60 subtests，pytest 514.40 秒 / wrapper 516.547 秒、exit 0**；**181 Git Python compile / diff check PASS**。Frozen source **`9999e596…`** / runner **`2a9e9e26…`**与五个 registration 的30槽一致，正式 v5 文件、data / evidence / split 字节不变；日志 SHA256 为 **`2ef9124c84db78d22180d446af50a797ff2eaba4718c895fb8b973f1301c329f`**。该验证不混用旧 v10 full suite，也不是 benchmark 分数。

测试通过不表示所有语言 scope 都覆盖：已发现未加冒号的“逐字引用下一段”与 lazy blockquote 两个已知 projection guard 缺口；另已确认部分 quoted / negative 位置指令被 projection 拒绝后，final compiler 仍会接受，两个阶段的scope边界不一致。当前没有修改冻结 source。后续共享 scope guard与ordinary revision提示修正将用独立 source 身份与验证记录，只承诺支持的有限syntax，不能把本轮30槽结果称作补丁后的 benchmark 表现或自然语言全面安全；原 scope 问题与所有失败 artifacts 保留。

## v14 指令十二槽：两类指令一起报告，词槽生成仍失败

`outputs/development_pilot_20261004_exposed_v14/summary.json` 已封存，注册为上文 **`31f64200…`**，固定12槽 **11 submitted / completed、1 Initial-MAS `RuntimeError / submission_failed`**，`comparison_complete=false`。

| 固定任务，主分 / secondary instruction counts | Direct | Initial-MAS | Native JIT |
|---|---|---|---|
| IFEval1203，strict prompt | 1 / 2/2 | 1 / 2/2 | 1 / 2/2 |
| IFEval1246，strict prompt | 1 / 1/1 | 1 / 1/1 | 1 / 1/1 |
| IFBench13，loose prompt | 0 / 0/1 | `null`，`RuntimeError`，无 checker score | 1 / 1/1 |
| IFBench22，loose prompt | 0 / 1/2 | 1 / 2/2 | 0 / 1/2 |

IFEval 两题完整均值三臂均 **1**；IFBench Direct **0**、Native JIT **0.5**、Initial-MAS **`null`**。IFBench 的 Initial-MAS failure-zero为 **0.5**，两来源 macro Direct / Initial-MAS / Native JIT **0.5 / 0.75 / 0.75**；这个保守汇总不把缺失提交改为观测0，也不把 secondary counts 当主指标。

| 方法 | 生成调用 / tokens | 生成 / 本地 checker 活动秒 | input / output tokens |
|---|---|---|---|
| Direct | 4 / 1,985 | 10.782 / 4.390 | 901 / 1,084 |
| Initial-MAS | 27 / 275,328 | 136.984 / 3.140 | 239,584 / 35,744 |
| Native JIT | 12 / 95,589 | 99.984 / 4.437 | 67,124 / 28,465 |

全生成 **43 模型调用、372,902 tokens、247.750 活动秒**，author checker **0模型API、11.967 本地活动秒**，queue idle / estimated / unknown为0，美元费用`null`。失败IFBench13自身 **4调用 / 47,099tokens**已计入：planning三次finish stop，首Writer **8192 output / finish length**；projection audit active True、guidance active、原task保留，但未进入public refiner。不能说投影已消除上游所有位置要求或解决重复；不重采该槽、不提交旧稿fallback。

## v14 DSQA 两题 shared evidence：ACT 全通过，另一题生成失败

`outputs/development_pilot_20261004_dsqa_evidence_v14/summary.json` 已封存，注册 **`2447fd52…`**，固定6槽 **5 submitted / completed、1 Initial-MAS `RuntimeError`**，同两份原 frozen packs，`comparison_complete=false`。

| 固定任务，F1 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| ACT129 | 1.000000 | 1.000000 | 1.000000 |
| DSQA170 | 0.000000 | `null` / `RuntimeError` | 0.666667 |
| 两题完整均值 | 0.500000 | `null` | 0.833333 |
| 两题 normalized failure-zero | 0.500000 | 0.500000 | 0.833333 |

| 方法 | 生成 / judge 调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部 input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 12,915 / 2,861 | 10.437 / 2.734 | 13,682 / 2,094 |
| Initial-MAS | 15 / 1 | 231,104 / 402 | 97.812 / 0.953 | 202,375 / 29,131 |
| Native JIT | 6 / 2 | 103,778 / 1,712 | 59.484 / 2.859 | 87,624 / 17,866 |

生成 **23调用 / 347,797tokens / 167.733活动秒**，judge **5调用 / 4,975tokens / 6.546活动秒**，合计 **28模型调用 / 352,772tokens**；estimated / unknown / queue idle为0、美元费用`null`。Ours DSQA170本任务 **7调用 / 149,605tokens**失败已计入campaign **15调用 / 231,104生成tokens**，两者不能混同。其global + 三local + reconcile五个planning调用均stop，非终端analyst随后 **2×4096 output / finish length**失败，没有最终Writer / public refinement / grader；不归因于planning schema或未经证实的source ID错误。ACT的成功不替代另一题失败。

## v14 RR shared evidence：ordinary revision 额外键失败

`outputs/development_pilot_20261004_rr_evidence_v14/summary.json` 已封存，注册 **`2ac12485…`**，固定3槽 **2 submitted / completed、1 Initial-MAS `ValidationError`**，`comparison_complete=false`。

| 指标 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| 完整 native raw weighted score（官方加权公式） | 0.312500 | `null` | 0.425000 |
| normalized failure-zero | 0.360465 | 0.000000 | 0.465116 |

| 方法 | 生成 / judge 调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部 input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 28 | 5,561 / 68,657 | 8.656 / 42.469 | 62,601 / 11,617 |
| Initial-MAS | 9 / 0 | 182,776 / 0 | 70.469 / 0 | 163,755 / 19,021 |
| Native JIT | 3 / 28 | 43,949 / 81,356 | 32.234 / 47.297 | 105,333 / 19,972 |

生成 **13调用 / 232,286tokens / 111.359活动秒**，judge **56调用 / 150,013tokens / 89.766活动秒**，合计 **69模型调用 / 382,299tokens**，unknown / estimated / queue idle为0、美元费用`null`。Raw native weighted score（官方加权公式）保持原生加权范围，不clip；初始方法本轮不可评分，不能拿 v9 的0.4875填入。

Initial-MAS 九次调用全部finish stop、analyst / Writer已执行，public review **1209 output**、ordinary revision **1959 output**。最终 JSON额外顶层 `additionalProperties` 被 PublicRevision strict extra-forbidden拒绝；它属于schema metadata，不能擅自删掉后当有效提交，也没有fallback原稿、模型评分或输出private criterion input values。故障在final public revision，不是截断或planning失败；后续schema-instance/answer-only通用提示如修正须另注册，不能原地改此失败。

## v14 DRB shared evidence：低于 Direct，Native JIT 失败非观测0

`outputs/development_pilot_20261004_drb_evidence_v14/summary.json` 已封存，注册 **`a4b918c5…`**，固定3槽 **2 submitted / completed、1 Native JIT `AttributeError / submission_failed`**，`comparison_complete=false`。

| 主指标 | Direct | Initial-MAS | Native JIT |
|---|---|---|---|
| 完整原生评价 | 29/58 = 0.500000 | 15/58 = 0.258621 | `null` / `AttributeError` |
| normalized failure-zero | 0.500000 | 0.258621 | 0.000000 |

| 方法 | 生成 / judge 调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部 input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 2 | 7,451 / 16,838 | 20.031 / 15.562 | 14,964 / 9,325 |
| Initial-MAS | 10 / 2 | 191,837 / 14,621 | 88.453 / 13.734 | 177,406 / 29,052 |
| Native JIT | 1 / 0 | 22,747 / 0 | 15.032 / 0 | 18,006 / 4,741 |

生成 **12调用 / 222,035tokens / 123.516活动秒**，judge **4调用 / 31,459tokens / 29.296活动秒**，合计 **16模型调用 / 253,494tokens**。Native JIT的失败调用 / 22,747tokens全计入，没有完整58项评价，不能称观察低分或本方法实质击败JIT。unknown / estimated / queue idle为0，美元费用`null`。

Initial-MAS实际15/58低于本轮Direct29/58，且较v9同packet的25/58退步；不能用漂亮旧DRB或RR行替代v14。五份原官方packet未改变、禁止文章没有被使用、未知早期增长统计没有补造；模型self-judge仍为开发诊断、不是独立judge或clean formal TEST结论。

## v14 四组 sealed 中途快照：完整三十槽账另见后文

在以上四组刚封存的中途快照，24固定槽有20完整评分、4生成失败；每臂8槽，WritingBench当时未完成。下表保留这部分已封存账，不能称v14完整30槽费用；其后WritingBench终态与完整六来源总账已在下文补录，不再使用pending替代最终结果。

| 方法 | 完整评分 / 已封存槽 | 生成 / 评价调用 | 生成 / 评价tokens | 全部模型调用 / tokens |
|---|---|---|---|---|
| Direct | 8 / 8 | 8 / 32 | 27,912 / 88,356 | 40 / 116,268 |
| Initial-MAS | 5 / 8 | 61 / 3 | 881,045 / 15,023 | 64 / 896,068 |
| Native JIT | 7 / 8 | 22 / 30 | 266,063 / 83,068 | 52 / 349,131 |

四组生成 **91调用 / 1,175,020tokens / 650.358活动秒**，评价 **65模型调用 / 186,447tokens / 137.575活动秒**（指令本地checker耗时亦包含），合计 **156调用 / 1,361,467tokens**。活动秒不作为并行协调器elapsed；未知价格仍`null`。v14端到端synthetic **6调用 / 74,096tokens**与所有更早synthetic/probe失败均单独另账，没有作为official分数或benchmark槽混入。

## v15 待冻结候选：两种防护与 strict ordinary revision，暂无结果

当前生产仍冻结v14身份。计划在原30槽全终态并封存后，才应用quotation / lazy-blockquote及final compiler共享scope guard与schema提示的泛化修正；预计source涉及word slots与execution wrapper，未在此提前写入完成身份。共同消息明确输出符合schema的response instance而非schema本体，ordinary revision要求answer-only，typed construction保留其所需结构、不误强制answer-only。另配置 `public_refinement_response_format=json_schema`，使ordinary revision也请求strict schema；其余v14配置不变。没有删额外键、拿旧稿fallback或假设服务一定兑现远程schema。

准备的后续范围为RR1 + WritingBench2、两个campaign同source/config共 **9固定三臂槽**，先同时注册再逐组全生成封存评分；不重跑未改动的IFBench / DSQA失败，不将未来RR / WB较高值替换v14旧行。源码、注册、验证和真实成绩均待实际发生后另补，不能声称某个修正的单项因果或新版已成功。

公开example JSON / MD已先同步这一 **v15 development candidate / results pending**：仅将review-mode从`json_schema_review`改为`json_schema`，其余四actor0.6 / judge0、projectionTrue、constructionstrict、revision-only0.25与v14相同；仍为通用安全placeholder，没有真实endpoint/key/localpath，并通过MASConfig本地验证。Quote guard只描述已识别的保守decline边界，不声称所有引文都覆盖。最终45文件隐私扫描等待v14及后续两个campaign saved summaries、新source/验证和最后文档完整；本阶段 **0 model API、0 Python生产/测试修改、0 staging**。

## v14 WritingBench 完整六槽：两个失败与全部预算一起保留

`outputs/development_pilot_20261004_writing_v14/summary.json` 已封存，注册 **`1c0fe32f…`**，原六槽 **4 submitted / completed、2 generation failed**，`comparison_complete=false`。这补齐前文WritingBench的中途快照，所有原失败不改成观测0。

| 固定任务，原生mean / 10 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|
| WB335 | 8.6 | `null` / `ValidationError` | `null` / `TimeoutError` |
| WB433 | 9.0 | 8.0 | 8.8 |
| 两题完整原生均值 | 8.8 | `null` | `null` |
| 两题normalized failure-zero | 0.866667 | 0.388889 | 0.433333 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 6,641 / 9,925 | 14.297 / 4.171 | 13,193 / 3,373 |
| Initial-MAS | 17 / 1 | 256,533 / 2,779 | 118.827 / 1.890 | 227,007 / 32,305 |
| Native JIT | 39 / 1 | 1,776,373 / 2,924 | 926.687 / 2.015 | 1,677,888 / 101,409 |

生成 **58调用 / 2,039,547tokens / 1,059.811活动秒**，评价 **4调用 / 15,628tokens / 8.076活动秒**，合计 **62模型调用 / 2,055,175tokens**。Unknown / estimated / queue idle为0、美元费用`null`。Native生成926.687活动秒是两题合计，不当作单题共同900秒额度改变；WB335在共同时间额度内自然终态超时，之后另一题独立预算正常运行，未提前结束或临时放宽其资源。

Initial-MAS WB335失败自身 **11调用 / 202,325tokens**、全部finish stop，public review **1805 output** / revision **3237 output**；ordinary PublicRevision额外`additionalProperties`触发strict extra-forbidden，与RR的schema metadata回显同类，不是Writer截断或planning失败。Native WB335 **36调用 / 1,750,050 known tokens**、`TimeoutError`，usage unknown为False，完整失败成本保留。没有删额外字段、改选旧稿、fallback提交、替换失败或依据评分重采。

## v14 最终完整三十槽账：二十四评分、六失败、没有整体优势

五个campaign全部sealed、controller exit0；30固定槽 **24完整评分、6 generation failed保持null**。Direct **10/10**，Initial-MAS **6/10**，Native JIT **8/10**；Initial-MAS失败IFBench13 / DSQA170 / RR / WB335，Native JIT失败DRB / WB335。最新六来源主表仅属于此统一source/config，不把v9漂亮行或未来v15 RR/WB替入。

| 方法 | 完整评分 / 注册槽 | 生成 / 评价调用 | 生成 / 评价tokens | 全部模型调用 / tokens | 生成 / 评价活动秒 |
|---|---|---|---|---|---|
| Direct | 10 / 10 | 10 / 34 | 34,553 / 98,281 | 44 / 132,834 | 64.203 / 69.326 |
| Initial-MAS | 6 / 10 | 78 / 4 | 1,137,578 / 17,802 | 82 / 1,155,380 | 512.545 / 19.717 |
| Native JIT | 8 / 10 | 61 / 31 | 2,042,436 / 85,992 | 92 / 2,128,428 | 1,133.421 / 56.608 |

生成总 **149调用 / 3,214,567tokens / 1,710.169活动秒**，评价 **69模型调用 / 202,075tokens / 145.651活动秒**（含本地checker耗时），全部 **218模型调用 / 3,416,642tokens（3,071,443 input / 345,199 output）**。六项失败均计入，unknown / estimated / queue idle为0、美元费用`null`；活动时间可重叠，不当作controllerelapsed。v14端到端synthetic的 **6调用 / 74,096tokens**另账，未混入benchmark分数或此总账。

按六个saved arm的normalized failure-zero原数作事后等权macro，Direct **0.5378552972**、Initial-MAS **0.4412515964**、Native JIT **0.5386304910**。这保留全部库存的描述性汇总不能替代来源主指标、缺失分数或正式protocol；Initial-MAS低于两个baseline且交付率低、费用高于Direct，不能声称全面胜出。部分指令与ACT任务通过未抵消DRB退步、普通revision失败、非终端analyst截断或JIT自身超时；新版本修改需另注册而非挑高分旧行。

## v15 新source已冻结，两campaign九槽注册：结果仍待封存

v14全部终态后已实际应用共享scope guard与ordinary schema response修正，新的runtime **`d8156aa31e5312a3f938d1284619918b104889bca154318b6f4800f866d50abb`**、runner仍 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。生产涉及`execution.py / public_word_slots.py / public_refinement.py`：projection与final positional compiler共享有限syntax scope guard，普通revision保留answer-only、typed construction保持结构，共用措辞明确输出response instance符合schema而非schema本体。配置ordinary revision也改请求strict`json_schema`，其余v14actor/judge/penalty/budget/evidence不变。

实际相关8模块 **297 tests passed / 24.41秒**，**182 Git Python compile / diff check通过**，工件`.runtime/final_v15_targeted_validation_baseline_runtime.json`。原完整schema断言保留；新增projection4例与finalscope4例，已有2个capture fixture按新的schema提取调整。新版full suite已单独运行，尚待实际count，不用v14的2018+60或旧source成绩代替。

| 新campaign | 固定范围 / 输入 | generation slots | registration hash |
|---|---|---:|---|
| `outputs/development_pilot_20261004_rr_evidence_v15` | RR1、source EVO、同shared packet | 3 | `d17cf2652f572aaace7a7b8232ed0d8c27d8c72e6b5ea1e39ecef571db420849` |
| `outputs/development_pilot_20261004_writing_v15` | WB2、source EVO、closed-book | 6 | `936cbf6b80c9647fdc692d05bd28fc0b439b26e95e1d787d43778c7cee548d8b` |

两个campaign先注册再顺序全生成封存评分，共9固定槽；两份registration实际绑定同新source与strict配置。Ledger仅新增两个references，仍 **5 selected TEST IDs / 19 campaign references**，supplemental unknown曝光范围不消除。没有重跑无关IFBench/DSQA失败、删v14额外字段或选旧稿fallback；也不能把后续RR/WB成绩替进v14统一六来源表。当前不预填v15分数或单项因果。

实际Git文件名刷新候选为 **46文件：13 tracked changed / 33 untracked candidates**，新增`tests/jit_mas/test_public_word_position_scope.py`，不继续盲称45；helper与ignored roots仍排除。Final privacy audit等待两saved summaries及新版full suite终态、最后文档完整；本次 **0 model API、0 Python生产/测试修改、0 staging**。

## v15 RR 完整三槽：新失败发生在非终端交接，schema阶段未运行

`outputs/development_pilot_20261004_rr_evidence_v15/summary.json` 已sealed，注册 **`d17cf265…`**，三槽 **2 submitted / completed、1 Initial-MAS RuntimeError**，`comparison_complete=false`。

| 指标 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| 完整native raw weighted score（官方加权公式） | 0.375000 | `null` / `RuntimeError` | 0.450000 |
| normalized failure-zero | 0.418605 | 0.000000 | 0.488372 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 28 | 5,375 / 63,271 | 8.484 / 41.000 | 57,393 / 11,253 |
| Initial-MAS | 6 / 0 | 93,412 / 0 | 49.516 / 0 | 79,644 / 13,768 |
| Native JIT | 5 / 28 | 81,351 / 100,213 | 52.344 / 52.063 | 155,800 / 25,764 |

生成 **12调用 / 180,138tokens / 110.344活动秒**，judge **56调用 / 163,484tokens / 93.063活动秒**，合计 **68模型调用 / 343,622tokens**。Unknown / estimated / queue idle为0、美元费用`null`，失败6调用 / 93,412tokens全部计入。规划五次stop之后，非终端analyst **4096 output / finish length**、其max_calls1耗尽，没有Writer / public refiner / grader；不能说新的schema-instance提示失败，也不能说其已真正修复此前RR的finalrevision问题，实际没有到那个阶段。

## v15 WritingBench 完整六槽：规划与模板失败，普通revision成功不等整体领先

`outputs/development_pilot_20261004_writing_v15/summary.json` 已sealed，注册 **`936cbf6b…`**，六槽 **4 submitted / completed、2 generation failed**，`comparison_complete=false`。

| 固定任务，原生mean / 10 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|
| WB335 | 8.6 | `null` / `ValueError` | 8.6 |
| WB433 | 8.8 | 8.6 | `null` / `TemplateSyntaxError` |
| 两题完整原生均值 | 8.7 | `null` | `null` |
| 两题normalized failure-zero | 0.855556 | 0.422222 | 0.422222 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 6,651 / 9,963 | 19.390 / 4.407 | 13,174 / 3,440 |
| Initial-MAS | 13 / 1 | 160,495 / 2,934 | 75.530 / 2.204 | 142,424 / 21,005 |
| Native JIT | 5 / 1 | 76,086 / 6,686 | 58.032 / 2.218 | 64,618 / 18,154 |

生成 **20调用 / 243,232tokens / 152.952活动秒**，judge **4调用 / 19,583tokens / 8.829活动秒**，合计 **24模型调用 / 262,815tokens**，unknown / estimated / queue idle为0、美元费用`null`。Initial-MAS WB335失败自身 **6调用 / 95,985tokens**均为planning stop：第一轮reconcile的coverage与agent.rubric_ids不一致先被拒绝，第二轮修正rubrics / 清空reviewers后仍将a0 synthesizer置于上游（depends_on=[]，a1依赖a0），违反final sink依赖方向。Generic异常虽含cycle措辞，但没有证据证明真实dependency cycle；没有Writer或public refiner，不能将它解释为strict ordinary schema修正失败。只读证据为`.runtime/v15_wb335_planning_diagnosis_baseline_runtime.json`，不复制题干或私有错误值。

Initial-MAS WB433 ordinary revision已实际成功、完整评分8.6，低于同轮Direct8.8；另一题失败未被该单题通过替换。Native JIT WB433 TemplateSyntaxError仍null、全部失败费用计入，不从旧round选择更有利对照。

## v15 九固定槽全库存与独立完整验证

两campaign已全部sealed、controller exit0，共 **9固定槽、6完整评分、3 generation failed保持null**。Direct **3/3**，Initial-MAS **1/3**，Native JIT **2/3**；本轮只覆盖RR / WritingBench，不把这些分数替入v14统一六来源表。

| 方法 | 完整评分 / 槽 | 生成 / judge调用 | 生成 / judge tokens | 全部模型调用 / tokens | 生成 / 评价活动秒 |
|---|---|---|---|---|---|
| Direct | 3 / 3 | 3 / 30 | 12,026 / 73,234 | 33 / 85,260 | 27.874 / 45.407 |
| Initial-MAS | 1 / 3 | 19 / 1 | 253,907 / 2,934 | 20 / 256,841 | 125.046 / 2.204 |
| Native JIT | 2 / 3 | 10 / 29 | 157,437 / 106,899 | 39 / 264,336 | 110.376 / 54.281 |

生成总 **32调用 / 423,370tokens / 263.296活动秒**，judge **60调用 / 183,067tokens / 101.892活动秒**，全部 **92模型调用 / 606,437tokens（513,053 input / 93,384 output）**。Unknown / estimated / queue idle为0、美元费用`null`，活动时间不当作并行controllerelapsed。三个原失败保留，没有补采、删字段、首JSONroot截取或旧稿fallback。

`.runtime/final_v15_validation_baseline_runtime.json` 实际完整验证为 **2,026 passed + 60 subtests、pytest499.50秒 / wrapper501.704秒、exit0**，**182 source Python compile / diff check通过**。Source **`d8156aa31…`** / runner **`2a9e9e26…`**保持注册冻结身份，日志SHA256 **`d904d5a7f93f41cba84af6ada2cbcd966b4d6d71e88b7c34e768f7b17308c214`**；不是套用v14成绩，也不能据代码测试通过声称九槽全部交付或方法领先。

## v16 新修复开发状态：真实范围与验证仍待下一次冻结

v15全部终态后，执行预算内的atomic fact handoff与planning / pipeline public-only阶段提示已作为新source修复应用，针对非终端交接篇幅过长、已有final public review时重复安排terminal审阅及final sink方向 / coverage不一致；不把WB335的generic cycle错误文本当作已证明的循环。支持位置规则的规划提示减少初稿role prompt再施加精确位置负担。预算、penalties、模型调用cap、评分不因此放宽；default refinement off的兼容行为保留。此前交接公开诊断中outline占59%、12项达到9,719chars，不能把真实内容篇幅错称纯重复。

新增`tests/jit_mas/test_public_planning_stages.py`含22cases，按实际Git文件名刷新候选为 **47文件：13 tracked changed / 34 untracked candidates**。这只是文件名快照；当前新targeted / fullsuite、runtime/config/data/evidence身份与全部新注册尚待实际冻结信号，本文不预填v16分数、sourcehash或成功，最终47-file privacy audit与staging继续等待正式结束通知。

公开example保留已用v15的安全通用参数与有限语法边界，历史九槽完整失败 / partial delivery已披露；新source须独立注册、验证。本文所有追加 **0 model API、0 Python生产/测试修改、0 staging**。

## v16 实际冻结与五个新注册：统一六来源仍待真实封存

v16 runtime实际冻结为 **`8a7bf3f657d72d2e34c310cccb8511c39f53be508f15861cd7daade9bf4048ef`**，runner仍 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。相关20模块 **536 passed + 38 subtests，pytest105.48秒 / wrapper107.172秒**，**183 Git source Python compile / diff check通过**，正式v5八文件byte hash不变。新版fullsuite已单独启动，尚无结果，不套用v15的2026+60。

参数与v15一致：四生成角色0.6 / judge0、strict ordinary与construction、revision-only0.25、role penalties None，共同2m tokens / 900 active seconds / null model-call cap。源修复针对预算内atomic fact handoff与public-only阶段安排、依赖方向、terminal sink、coverage一致性；default refinement off兼容保留，未按private评分增加调用或放宽公共checker。效果仍待全库存，不因修复工程通过先称benchmark改善。

| campaign（目录均为`outputs/development_pilot_20261004_…`） | 原固定来源 / 输入 | generation slots | registration hash |
|---|---|---:|---|
| `exposed_v16` | IFEval2 / IFBench2、exposed TEST、closed-book | 12 | `790f23fccda97ba1f05b76b837113acdc33497458a16b7ea413bd98b7c80c34f` |
| `dsqa_evidence_v16` | DSQA2、source EVO、同两frozen packs | 6 | `96d7ad328a0651c7d6912fb501dfe7e8b0b68f32e6d0e68d2f5792beb968d810` |
| `rr_evidence_v16` | RR1、source EVO、同shared packet | 3 | `550e64d9f19eb94afafd9c286f9dc864ca50423ab0c03c86ebb60fe5e9a1d789` |
| `drb_evidence_v16` | DRB1、exposed TEST、同五官方文档 | 3 | `11aab29df404853dc340d54820f813b9fc284e8f652a776c56ccb42d404c45de` |
| `writing_v16` | WB2、source EVO、closed-book | 6 | `44c8edecbfd9cedb8ffa4c8282393c3926c0522b64f306b435b28e1874acfc8f` |

五个registrations已全部落盘，同source/config共 **30固定槽 / 10任务 / 6benchmarks**。Exposure ledger只新增五个references，仍 **5 selected TEST IDs / 24 campaign references / 1 incidental legacy事件**；EVO只记录batch身份，不增TEST选题，unknown曝光范围仍披露。API现已开始但本文只读registration metadata，不读live生成或private评价。

此处没有v16成绩、成本或完整验证count。待各campaign全部生成终态封存、全部saved summaries与新版全测实际结果后，最新统一六来源表只用v16全部固定槽、失败null与成本；v14与v15完整历史继续保留，不能把其较高行拼进v16。实际47候选文件名已刷新，final privacy scan与staging继续等待明确结束通知。本阶段 **0 model API、0 Python生产/测试修改、0 staging**。

## v16 新版完整验证已结束，benchmark仍按各组封存记录

`.runtime/final_v16_validation_baseline_runtime.json` 的实际完整验证为 **2,048 passed + 60 subtests、pytest468.10秒 / wrapper470.328秒、exit0**；**183 Git source Python compile / diff check通过**。runtime **`8a7bf3f657d72d2e34c310cccb8511c39f53be508f15861cd7daade9bf4048ef`** / runner **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`** 与五个注册、30固定槽匹配，data23 / checker14 / formal8冻结文件身份一致。验证日志SHA256 **`eafe21466c058ac7735ecc0dd225dd0da8b5e0f7a9ed2f2c24029d95e7cbc1d5`**。这属于v16独立全测，不能用来改写v15结果，也不证明方法已达到benchmark优势。

下文最先保留指令与DSQA两组刚sealed时的中途记录；随后RR / DRB / WritingBench的终态及全部三十槽总账已补齐，顶部现已统一更新为完整v16。没有以中途局部结果或旧版本高分拼成完整v16。

## v16 指令组十二固定槽：九评分、三生成失败、三臂macro相同

`outputs/development_pilot_20261004_exposed_v16/summary.json` 已sealed，注册 **`790f23fc…`**；原12槽 **9 submitted / completed、3 generation failed**，`comparison_complete=false`。IFEval主指标仍为author checker的 **prompt-level strict accuracy**；IFBench沿用已注册pinned checker的 **prompt-level loose accuracy**，instruction-level通过数量只作secondary，不替换主指标。

| 固定任务、主prompt accuracy | Direct | Initial-MAS | Native JIT |
|---|---:|---|---|
| IFEval1203 | 1 | `null` / `JSONDecodeError` | 0 |
| IFEval1246 | 1 | 1 | 1 |
| IFBench13 | 0 | `null` / `RuntimeError` | 1 |
| IFBench22 | 0 | 1 | `null` / `UnboundLocalError` |
| IFEval完整两题均值 | 1.0 | `null` | 0.5 |
| IFBench完整两题均值 | 0.0 | `null` | `null` |
| IFEval / IFBench normalized failure-zero | 1.0 / 0.0 | 0.5 / 0.5 | 0.5 / 0.5 |
| 两来源等权failure-zero macro | 0.5 | 0.5 | 0.5 |

IFEval1203的Direct / Native secondary分别 **2/2、1/2**，Initial-MAS缺失；1246三臂均 **1/1**。IFBench13的Direct / Native分别 **0/1、1/1**，Initial-MAS缺失；22的Direct / Initial-MAS分别 **1/2、2/2**，Native缺失。这保留了有效提交的部分指令满足率与真正生成失败的区别；没有把三个失败写成observed0。

| 方法 | 完整评分 / 槽 | 生成调用 / tokens | 生成 / checker活动秒 | input / output tokens |
|---|---|---|---|---|
| Direct | 4 / 4 | 4 / 2,126 | 11.250 / 4.047 | 901 / 1,225 |
| Initial-MAS | 2 / 4 | 24 / 255,015 | 157.467 / 1.797 | 215,453 / 39,562 |
| Native JIT | 3 / 4 | 12 / 102,048 | 93.546 / 2.686 | 75,494 / 26,554 |

该组生成 **40模型调用 / 359,189tokens / 262.263活动秒**，本地checker **0模型API / 0tokens / 8.530活动秒**；全部 **291,848 input / 67,341 output tokens**。失败开销均计入；unknown / estimated / queue idle为0、美元费用`null`。三臂macro相同，且Initial-MAS仅2/4完成，不构成其整体领先的证据。

只读public actor metadata定位Initial-MAS IFEval1203为 **8调用 / 104,960tokens**：前六调用及初稿有效，review已验证，最后revision **8192 output / finish length、49,099chars**且高重复，未闭合JSON导致失败；没有回退提交旧稿。证据为`.runtime/v16_ifeval1203_*`诊断工件，仅记录阶段和计量，不复制题目或私有参考。

Initial-MAS IFBench13为 **4调用 / 46,481tokens**：三次planning均stop且有效、positional stage提示已active；初始Writer单次 **8192 output / finish length**，规划自行给出的`AgentSpec.max_calls=1`及`team.total_max_calls=1`耗尽，尚未进入public refiner。证据为`.runtime/v16_ifbench13_initial_execution_failure_baseline_runtime.json`。这不是共同token/time预算被另行缩小，也不能据提示已生效声称writer重复已解决。Native IFBench22保留`UnboundLocalError`；当前记录不推断尚未核实的更细阶段。

## v16 DSQA两共享证据任务：六槽全部评分，交付修复没有带来第二题F1提升

`outputs/development_pilot_20261004_dsqa_evidence_v16/summary.json` 已sealed，注册 **`96d7ad32…`**；原6槽 **6 submitted / completed、0 generation failed**，`comparison_complete=true`，仍使用同两份frozen primary-source packets。

| 固定任务、native F1 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| DSQA129 / ACT官方原始表 | 1.0 | 1.0 | 1.0 |
| DSQA170 / 同另一公开证据包 | 0.0 | 0.0 | 0.666667 |
| 完整两题均值 | 0.5 | 0.5 | 0.833333 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 12,515 / 2,460 | 8.407 / 2.610 | 13,267 / 1,708 |
| Initial-MAS | 17 / 2 | 310,429 / 2,143 | 102.234 / 2.500 | 285,378 / 27,194 |
| Native JIT | 6 / 2 | 97,527 / 1,461 | 51.610 / 2.062 | 84,484 / 14,504 |

生成 **25调用 / 420,471tokens / 162.251活动秒**，评价 **6调用 / 6,064tokens / 7.172活动秒**，合计 **31模型调用 / 426,535tokens（383,129 input / 43,406 output）**。unknown / estimated / queue idle为0、美元费用`null`；同一Exp模型self-judge并非独立Judge，只有两题，不支持广泛泛化结论。

DSQA170在v14的Initial-MAS非终端交接截断曾导致generation failure；v16预算内紧凑handoff后该题交付并完成评分，是工程交付层面的改善。其实际内容F1仍 **0**，不能写成内容质量提升；本轮完整均值仍低于Native JIT。原v14失败与其费用继续保留，未被v16有效提交覆盖。

## v16 RR完整三槽：这一固定开发任务真实高于两baseline

`outputs/development_pilot_20261004_rr_evidence_v16/summary.json` 已sealed，注册 **`550e64d9…`**；三槽全部submitted / completed，`comparison_complete=true`。

| 指标 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| native raw weighted score（官方加权公式） | 0.450000 | 0.562500 | 0.437500 |
| normalized score | 0.488372 | 0.593023 | 0.476744 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 28 | 6,038 / 83,282 | 10.329 / 53.046 | 75,957 / 13,363 |
| Initial-MAS | 8 / 28 | 149,883 / 65,005 | 52.735 / 41.672 | 191,515 / 23,373 |
| Native JIT | 3 / 28 | 44,558 / 77,769 | 33.406 / 47.188 | 101,678 / 20,649 |

生成 **12调用 / 200,479tokens / 96.470活动秒**，评价 **84调用 / 226,056tokens / 141.906活动秒**，合计 **96模型调用 / 426,535tokens（369,150 input / 57,385 output）**。unknown / estimated / queue idle为0、美元费用`null`。

Initial-MAS在该固定EVO开发任务的真实分数高于本轮两baseline，也高于v9同任务的0.4875。选题、immutable manifest与config seed没有因此改变；各轮源码/config差异、生成随机性和同一Exp模型self-judge使单次差值不能证明某项修复的因果效应或稳定跨benchmark优势。共享来源packet没有改变，不依此局部成功替代其他来源的失败和低分。

## v16 DRB完整三槽：Initial-MAS提交只有标题，真实得零分

`outputs/development_pilot_20261004_drb_evidence_v16/summary.json` 已sealed，注册 **`11aab29d…`**；三槽全部submitted / completed，`comparison_complete=true`。Direct **29/58 = 0.500000**、Initial-MAS **0/58 = 0.000000**、Native JIT **30/58 = 0.517241**。Initial-MAS是有效提交后的实际观察0分，和generation failure的`null`不同。

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 2 | 6,651 / 14,695 | 16.860 / 13.922 | 13,272 / 8,074 |
| Initial-MAS | 8 / 2 | 155,758 / 6,043 | 68.203 / 8.828 | 141,093 / 20,708 |
| Native JIT | 3 / 2 | 47,083 / 15,666 | 47.625 / 14.109 | 44,233 / 18,516 |

生成 **12调用 / 209,492tokens / 132.688活动秒**，评价 **6调用 / 36,404tokens / 36.859活动秒**，合计 **18模型调用 / 245,896tokens（198,598 input / 47,298 output）**。unknown / estimated / queue idle为0、美元费用`null`；同五份官方文档、禁用文章边界与原始任务hash未改变。

`.runtime/v16_drb_public_refinement_title_collapse_baseline_runtime.json` 的只读public-output性质诊断表明，Initial-MAS初稿 **6,141chars / 147行 / 68段 / 12标题 / 102句末标点 / 4,714 CJK字符**，最终仅 **29chars / 1行 / 1标题 / 0句末标点 / 19 CJK字符**，恰为初稿第一行Markdown标题、长度仅原稿 **0.472%**。八次生成全finish stop，ordinary answer-only revision输出 **23tokens**、完整通过本地schema；numeric / positional / projection三路由均inactive。review有8项issue，机械长度/完整性关键词计数0不代表语义上没有相关要求。

该提交不是placeholder、拒绝、任务确认、schema本体或已证缺资料，也不是截断导致的生成异常；只根据上述metadata不进一步猜测因果。没有恢复初稿fallback或选择更有利旧版本。v16 DRB低于两baseline，并从v14的15/58、v9的25/58退步，必须与RR成功一起保留。

## v16 WritingBench完整六槽：一题小幅高分与另一题失败一起报告

`outputs/development_pilot_20261004_writing_v16/summary.json` 已sealed，注册 **`44c8edec…`**；六槽 **5 submitted / completed、1 Initial-MAS generation failed**，`comparison_complete=false`。

| 固定任务、native mean / 10 | Direct | Initial-MAS | Native JIT |
|---|---:|---|---:|
| WB335 | 8.0 | 8.2 | 8.0 |
| WB433 | 9.0 | `null` / `JSONDecodeError` | 9.0 |
| 完整两题原生均值 | 8.5 | `null` | 8.5 |
| normalized failure-zero | 0.833333 | 0.400000 | 0.833333 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 6,603 / 9,958 | 13.423 / 4.313 | 13,200 / 3,361 |
| Initial-MAS | 10 / 1 | 197,993 / 6,266 | 139.390 / 2.031 | 156,995 / 47,264 |
| Native JIT | 6 / 2 | 67,277 / 9,606 | 59.938 / 4.140 | 58,958 / 17,925 |

生成 **18调用 / 271,873tokens / 212.751活动秒**，评价 **5调用 / 25,830tokens / 10.484活动秒**，合计 **23模型调用 / 297,703tokens（229,153 input / 68,550 output）**。unknown / estimated / queue idle为0、美元费用`null`。WB335的8.2较两baseline8.0高，但WB433失败保留缺失及费用，不据前者声称WritingBench整体领先。

`.runtime/v16_writingbench433_planning_failure_baseline_runtime.json` 的只读metadata进一步定位Initial-MAS WB433为 **initial global.predict**：attempt0与原有唯一contract-correction attempt1均 **JSONDecodeError / Unterminated string、16,000 output / finish length**。该题实际 **2生成调用 / 63,559tokens / 约82.690活动秒**，共同2m / 900秒 / null call与tool cap没有耗尽；尚无local plan、reconcile、frozen plan、初执行或public refiner，不能写成末端revision失败。两次规划response均77,777chars、UTF8 SHA相同，只记录观察，不推断传输或缓存原因；没有新增质量重采样或读取评分reference。

## v16最终三十槽总账、冻结身份与发布边界

五个campaign的全部saved summaries已sealed、controller exit0：**30固定槽 / 10任务 / 6benchmarks、26完整评分 / 4生成失败 / 0评价失败**。Direct **10/10**、Initial-MAS **7/10**、Native JIT **9/10**；Initial-MAS失败IFEval1203 / IFBench13 / WB433，Native JIT失败IFBench22。每组均在全部生成终态封存之后评分；无失败替换、quality resampling、旧稿fallback或跨版本择优。

| 方法 | 完整评分 / 注册槽 | 生成 / 评价调用 | 生成 / 评价tokens | 全部模型调用 / tokens | 生成 / 评价活动秒 |
|---|---|---|---|---|---|
| Direct | 10 / 10 | 10 / 34 | 33,933 / 110,395 | 44 / 144,328 | 60.269 / 77.938 |
| Initial-MAS | 7 / 10 | 67 / 33 | 1,069,078 / 79,457 | 100 / 1,148,535 | 520.029 / 56.828 |
| Native JIT | 9 / 10 | 30 / 34 | 358,493 / 104,502 | 64 / 462,995 | 286.125 / 70.185 |

生成总 **107调用 / 1,461,504tokens / 866.423活动秒**，评价 **101模型调用 / 294,354tokens / 204.951活动秒**（包括0 API的本地instruction checker耗时），全部 **208模型调用 / 1,755,858tokens（1,471,878 input / 283,980 output）**。失败开销全计入，unknown / estimated / queue idle为0、美元费用`null`；活动秒求和不作为并行controller elapsed。v14端到端synthetic **6调用 / 74,096tokens**与所有更早probe另账，不混入本轮benchmark。

由六个saved source arm的normalized failure-zero原数计算等来源描述性macro：Direct **0.5536175710594315**、Initial-MAS **0.41550387596899224**、Native JIT **0.6101087053372538**。Initial-MAS整体低于两baseline、成本高于Direct，并存在ordinary revision语法有效但正文被压成标题的实际质量退步；没有以单次RR及WB335成功抵消这些限制。该开发诊断不能冒称最高benchmark成绩、clean formal TEST、独立Judge或训练版27B JIT复现。

runtime **`8a7bf3f657d72d2e34c310cccb8511c39f53be508f15861cd7daade9bf4048ef`** / runner **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**在五组期间不变，新版实际完整验证仍为 **2,048 passed + 60 subtests / 468.10秒、183 Git source compile / diff check通过**，不是沿用v15成绩。Exposure ledger保留 **5 selected TEST IDs / 24 campaign references / 1 supplemental incidental legacy事件**；已知额外历史ID与unknown可见范围不抹除，正式v5不可变文件未改。

公开example标为 **v16 validated development candidate**，其有效参数与实际五registration逐叶一致；仅五角色的endpoint / model / expected response-model pin为通用占位，共15字段。其library defaults / unsupported parameter fail-closed / 有限scope语法 / 最后revision-only penalty边界继续披露，不推荐为胜出配置。发布候选按实际Git文件名固定47项；只公开代码、合成测试、说明文档与hash元数据，不加入outputs / .runtime / dataset或本机进程helper。文档完成后执行精确47-file隐私审查，本文工作 **0 model API、0生产/测试修改、0 staging**。

## v17已冻结开发候选：structured planning与公开候选guard，完整结果尚待

v16代码已提交为父HEAD **`17019c4c04edecba53101176e4fd2a9957921b2b`**。新v17 production / tests已冻结，runtime **`fbb842503a6b0a07ba5728f28d277a38defbe7661f98c68a257a09a946ecd146`**、runner仍 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**；`.runtime/public_refinement_v17.config.json`字节SHA256 **`f83030961c1f9fe5bdda9da0775a72afeaee1f8444c5e0dea36a8f0c0394de48`**。

配置仅新增 **`planning_response_format=json_schema`** 与 **`public_refinement_guard=true`**；四生成角色temperature0.6 / judge0、全部role frequency penalty None、最后public revision-only0.25、2m token / 900 active seconds及所有其他预算、采样和旧flags均与v16相同。公开example已同步这两个新flag，仍只用通用endpoint / model / response-pin占位；它现在是 **v17 frozen development candidate / benchmark results pending**，不能套用v16全测成绩或声称胜出。

### ordinary revision guard：工程有效性与公开结构保护，不是语义正确证明

guard在评分前按已冻结规则，从唯一initial artifact与其ordinary revision确定提交项，没有Judge、reference、private rubric、分数择优或额外采样。Initial eligible仅表示执行已经`final_answer`且答案为非空文本；不能据此认定全部公开约束或语义正确。若initial仍处于active projection，或存在active positional / numeric construction，则不eligible，不以尚未最终构造的草稿作fallback。

只在eligible普通revision的 **本地response JSON / schema验证失败**，或 **极端正文丢失** 时保留initial。极端结构规则在源码中提前固定：initial至少 **1,000非空白字符、3个punctuation-or-line chunks**；revision至多 **100非空白字符、1个chunk**，且revision / initial非空白字符比值不超过 **0.1**。这些是通用结构门槛，不是benchmark权重或题目答案，也不能发现所有内容退步。Review失败、provider / 认证错误、budget或deadline耗尽、typed construction失败继续向上报错，guard不吞。

这些结构计数不是语义或显式约束验证。若公开任务实际要求简短答案，guard可能保留较长旧稿、拒绝本来正确的短revision；反之 **101chars或2个chunk** 的坍缩可能漏过。1,000 / 100两阈值已数学蕴含比例不超过0.1，该ratio不是独立保障。Public diagnostics的列表 / 位置可能截断，不能当作author checker；缺失或未报告项不应被补成失败0，也不表示全部约束通过。实际v17保持这一有限heuristic源码，不据运行结果调整阈值。

审计保留initial / revision与选中answer的hash、initial eligibility、通用阈值、selection reason、raw组件失败和所有调用 / token / 活动时间。原本可恢复的ordinary revision本地失败可记为completed-with-component-failure并选择旧initial；其组件失败账不删除。没有有效initial的planning或execution失败仍保持整槽failure-null。新guard的库默认关闭且要求public refinement启用；没有将“eligible”表述为author checker已通过。

### strict planning：限生成注解，保留完整原任务与真正共同预算

predict / local-plan / reconcile请求strict JSON schema，库默认仍为`json_object`。长度限制只针对生成的规划注解，例如rationale **1024chars**与task_prompt **4096chars**，不截断原始public task、evidence、要求的final artifact或其必需事实。Uncapped iterative mode把生成计划中的`max_calls` / `total_max_calls`限定为`null`，expected-call估计不再暗设组织内有限call ceiling；共同token / deadline及停止条件继续有效，single-pass兼容不改变。

Provider未兑现requested strict schema时仍执行本地schema / annotation与graph / coverage验证。若finish reason为`length`，完整raw response与计费留在audit；原有唯一contract correction继续使用完整original inputs及失败metadata / raw hash，不把截断长尾重放进纠正请求。没有新增纠正次数、execution质量采样、格式或模型fallback。规划同时强调只以公开条件作硬筛选、检查源事实和已知 / 反证 / 未知状态；不把缺失国籍、额外日期证明或实时库存保证写成新的公开资格条件。

### 注册与完整验证已完成，benchmark结果待真实封存

v17实际targeted为 **580 passed + 38 subtests / 100.063秒**，新版仅一次full suite最终 **2,092 passed + 60 subtests、pytest467.30秒 / wrapper469.438秒、exit0**；**185 Git source Python compile / diff check通过**。验证工件`.runtime/final_v17_validation_baseline_runtime.json`及`.runtime/final_v17_test_metadata_summary_baseline_runtime.json`记录源码与五registration同一冻结身份，日志SHA256 **`e3acb494aa47aa3214aa61e3a4e36a6dfa8ea8167febb54e3aacba4af21570ec`**、source-set SHA256 **`deb1f972c50eda9b47904e09f12a6f4f16cfae76c8fd9524a03d5c095083f27a`**。正式v5八文件、23 unique data / evidence / formal bytehash、14 unique checker / NLTK bytehash与distribution version匹配；30槽选取 / sources / splits / models / 2m900与v16一致，normalized配置仅有两个新flag变化。

真实full-MAS synthetic预检已completed，`.runtime/full_mas_structured_guard_v17_20261004/preflight.json`记录 **11调用 / 130,620tokens / 61.422秒 / 美元费用null**，单独计费而非benchmark槽或author score。三阶段共5次planning调用（2 local、reconcile原有唯一contract correction1），全部finish stop、requested json_schema且实际response model pin匹配，frozen team各max_calls为null；typed public revision选择revision，预算与initial / final hash均保存。这个合成执行成功及完整代码测试通过均不证明真实benchmark质量。

五个campaign已全部真实注册、同十任务 **30固定槽**，之后才按既定顺序开始benchmark生成；每组先全生成终态封存再评分。五个registration的parsed配置canonical SHA256同为 **`e828bc174b3b7f1bcf3ddc2aafe2b35b58ab98bb0b09e2dec2dcc4534a85c709`**。下表登记实际hash，逐组结果只在sealed后追加，不读取live答案或private评分。

| v17 campaign | 固定公开来源与输入 | generation slots | registration hash（逐组结果见后文） |
|---|---|---:|---|
| `exposed_v17` | IFEval2 / IFBench2、exposed TEST、原checker、closed-book | 12 | `66c51c30e98868ac237e98940489d9500452990f0bdc9bc3ab963a1bb1396e80` |
| `dsqa_evidence_v17` | DSQA2、source EVO、原两frozen packs | 6 | `2c954915dab810c9673549728c3898aff98e9916ecdf41254abe44b9c800ef3c` |
| `rr_evidence_v17` | RR1、source EVO、原shared packet | 3 | `94ef8885d35e5954b94734f52f86b182ad274365914b6ee075689ce7c7491822` |
| `drb_evidence_v17` | DRB1、exposed TEST、原五官方文档 | 3 | `b8d398e80d5db801e792e8ea0f52e74af11e2599fc5f6a5c6f8dd1faf950b45b` |
| `writing_v17` | WB2、source EVO、closed-book | 6 | `769dfbe84368dfc405c21438ecb741fca4dfc89325fd0403f6325a5cbeee3602` |

各previous campaign对应v16同组；固定task / pack / checker / v5 selection顺序和scoring均不改，三真实入口与common预算一致。Exposure ledger暂仍 **5 selected TEST IDs / 24已写campaign references / 1 supplemental legacy事件**；本轮五个真实registration已知，待所有sealed后与结果同步补references至29，不增题、不虚构hash。最新完整六表仍是v16，v17需全部新槽一起报告null / observed0 / fallback及费用，不替入旧版较高行。

v16已发布47-file审查只适用于上一个commit，不可复用为这次新候选的审查。待五组sealed、新full suite与最后文档齐全，再按实际Git候选文件名执行新一轮精确publication audit；`scripts/wait_v37_then_launch_v38.py`及outputs / .runtime / dataset仍永不加入发布。此阶段文档工作 **0 API、0生产/测试/runtime helper修改、0 staging**。

## v17指令组十二槽全部封存：IFBench本两题领先，IFEval较两baseline低

`outputs/development_pilot_20261004_exposed_v17/summary.json` 已sealed，注册 **`66c51c30…`**；原12槽 **12 submitted / completed、0 generation / evaluation failures**，`comparison_complete=true`。IFEval仍为author checker的strict prompt-level accuracy，IFBench仍为pinned checker的loose prompt-level accuracy；所有0均是有效提交的实际观测0分。

| 固定任务、主prompt accuracy | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| IFEval1203 | 1 | 0 | 1 |
| IFEval1246 | 1 | 1 | 1 |
| IFBench13 | 0 | 1 | 0 |
| IFBench22 | 0 | 1 | 1 |
| IFEval完整两题均值 | 1.0 | 0.5 | 1.0 |
| IFBench完整两题均值 | 0.0 | 1.0 | 0.5 |
| 两来源等权normalized macro | 0.5 | 0.75 | 0.75 |

IFEval1203的instruction-level secondary为Direct **2/2**、Initial-MAS **1/2**、Native **2/2**；1246三臂均 **1/1**。IFBench13为Direct **0/1**、Initial-MAS **1/1**、Native **0/1**；22为Direct **1/2**、Initial-MAS **2/2**、Native **2/2**。不以secondary替换prompt主指标；不能将Initial-MAS的IFEval0.5写成胜出。

| 方法 | 完整评分 / 槽 | 生成调用 / tokens | 生成 / checker活动秒 | input / output tokens |
|---|---|---|---|---|
| Direct | 4 / 4 | 4 / 2,698 | 13.720 / 3.579 | 901 / 1,797 |
| Initial-MAS | 4 / 4 | 28 / 332,182 | 205.158 / 3.562 | 278,676 / 53,506 |
| Native JIT | 4 / 4 | 37 / 363,076 | 229.828 / 3.250 | 295,987 / 67,089 |

生成 **69模型调用 / 697,956tokens / 448.706活动秒**，本地checker **0 API / 0tokens / 10.391活动秒**；总 **575,564 input / 122,392 output tokens**。unknown / estimated / queue idle为0、美元费用`null`。这是第一组完整账，不当作30槽总账；其余四组仍待真实封存，最新完整六表暂保留v16。

`.runtime/v17_ours_public_receipts_baseline_runtime.json` 的公共receipt表明，Initial-MAS四题规划生成的AgentSpec / Team call caps均为null，guard全部 **selected revision**，未走retain-initial或body-loss分支。IFBench13为 **6调用 / 67,639tokens**：3 planning stop、1 writer stop、2 refiner stop，实际Protocol correction **0次**，不能称它被纠错救回。IFBench22为 **7调用 / 72,522tokens**：writer length后使用原有same-role唯一protocol correction至stop，null call cap允许这次现有执行纠正；IFEval1203为 **9调用 / 145,147tokens**，实际local plan length后hash-only planning唯一纠正至stop；IFEval1246为 **6调用 / 46,874tokens**。这些预算已包含在组总账，不另叠加。

selected / submission / execution hash一致；本两题IFBench实际高于两baseline，而两来源macro与Native JIT持平、IFEval低于两baseline。采样temperature0.6仍有随机性，两项源码 / config变化共同存在，不能把single-run变化归因于guard fallback（本组未触发）、某一纠正策略或稳定全benchmark优势。

IFEval1203的公开文字约束计数诊断显示，初稿满足两项最低频次，revision丢失了至少一项。匿名要求下限为 **8 / 10**；第一项initial的exact-strict / casefold-strict / broad计数为 **9 / 9 / 11**，final为 **4 / 4 / 13**，宽泛统计存在歧义，归为 **UNKNOWN**；第二项initial为 **12 / 12 / 12**，final为 **4 / 4 / 4**，归为 **FAIL**。这里没有公开原文字面量，也没有把UNKNOWN算作FAIL。答案由 **7,976chars膨胀到26,120chars**，不是v1 body-loss heuristic检查的极短输出，说明工程eligible与JSON/schema正确仍不能保证已满足的公开文字约束得到保留。

## v17 DSQA六槽完整封存：均值高于Direct，仍低于Native JIT

`outputs/development_pilot_20261004_dsqa_evidence_v17/summary.json` 已sealed，注册 **`2c954915…`**。原6槽 **6 submitted / completed，0 generation / evaluation failures**，`comparison_complete=true`；使用与v16相同的两份frozen primary-source packets。

| 固定任务native F1 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| DSQA129 / ACT官方原始表 | 0.615385 | 0.933333 | 1.000000 |
| DSQA170 / 同一完整证据包 | 0.000000 | 0.000000 | 0.666667 |
| 两题完整均值 | 0.307692 | 0.466667 | 0.833333 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 12,456 / 2,441 | 8.203 / 2.500 | 13,205 / 1,692 |
| Initial-MAS | 17 / 2 | 332,915 / 4,624 | 150.953 / 2.797 | 292,850 / 44,689 |
| Native JIT | 7 / 2 | 114,913 / 1,548 | 50.891 / 2.249 | 102,042 / 14,419 |

生成 **26调用 / 460,284tokens / 210.047活动秒**；评价 **6调用 / 8,613tokens / 7.546活动秒**；合计 **32模型调用 / 468,897tokens（408,097 input / 60,800 output）**。unknown / estimated / queue idle为0，美元费用`null`。Initial-MAS本组均值高于Direct，但低于Native JIT，DSQA170仍为真实已评分0。只有两题、同一serving Exp模型self-judge，不能据此宣称DSQA稳定领先或独立Judge验证。

## v17 RR完整封存：真实观察0分，公开actor工件显示标题坍缩

`outputs/development_pilot_20261004_rr_evidence_v17/summary.json` 已sealed，注册 **`94ef8885…`**。三槽全部submitted / completed，**0 generation / evaluation failures**，`comparison_complete=true`；使用同一固定EVO任务、共享packet与评分公式。

| 指标 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| native weighted score（官方加权公式） | 0.425000 | 0.000000 | 0.450000 |
| normalized score | 0.465116 | 0.069767 | 0.488372 |

Initial-MAS的raw 0是有效提交的实际观察分数，不是generation failure或缺失值置零。按注册native bounds归一化后为 **0.06976744186046512**，不能擅自把normalized值改为0；本组低于两baseline，也低于v16同题的0.5625，负向结果完整保留。

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 28 | 5,931 / 79,822 | 9.859 / 46.875 | 72,961 / 12,792 |
| Initial-MAS | 8 / 28 | 171,553 / 23,228 | 53.938 / 31.609 | 175,401 / 19,380 |
| Native JIT | 3 / 28 | 45,737 / 93,384 | 36.625 / 46.469 | 117,697 / 21,424 |

生成 **12调用 / 223,221tokens / 100.422活动秒**；评价 **84调用 / 196,434tokens / 124.953活动秒**；合计 **96模型调用 / 419,655tokens（366,059 input / 53,596 output）**。unknown / estimated / queue idle为0，美元费用`null`，Judge仍为同一serving模型而非独立Judge。

只读公开actor receipt的诊断工件`.runtime/v17_rr60_public_actor_diagnosis_runner_implementation.json`记录：initial为 **14,597chars / 12,612非空白字符 / 1,948空白分词 / 84非空行 / 39段 / 18个Markdown标题 / 160个punctuation-or-line chunks**；final为 **229chars / 197非空白字符 / 33空白分词 / 1行 / 1个Markdown标题 / 1个chunk / 0句末标点 / 0正文字符**。最终仅标题，不是acknowledgement、拒绝或placeholder；它扩展了初稿首标题，首标题101chars被保留为前缀，但没有保留正文。非空白字符比例为 **1.562%**。initial SHA256为`024d2bfde9853ac568841ba327b28e5f644ca936b271f6b364c74d44da16857f`，final为`cfef40a9dfdac9771f3cc7305049285cbcacf19a4c41f0f58ef67e432f8b889e`。

8次模型调用全部finish stop，包含4 planning、2 execution、1 public review、1 public revision；review输出 **1,969tokens**，revision仅 **51tokens**。JSON/schema验证通过，draft与public input相同，validated revision / execution / submission字符串及hash一致，没有旧稿fallback。两typed路由inactive，initial工程eligible；guard选择`revision`，reason为`validated_revision_without_catastrophic_body_loss`。**197非空白字符超过v1的100上限**，所以本轮既定heuristic没有拦截这次标题坍缩。

原公共题面与附加证据分开后为300chars；有限匿名grammar探测中，显式minimum / range word count、section count、heading结构计数均为0。零匹配不能证明所有语义要求不存在，也不能用实际0分的私有criteria反向设计规则。这个诊断只读取已封存的公开生成工件与summary成绩 / 成本元数据，没有读取任何evaluation body、private rubric、reference answer或CSV，也没有进行API调用、重采样或源码改动。

## v17 DRB完整封存：保留initial得到有效提交，低于Direct、高于Native JIT

`outputs/development_pilot_20261004_drb_evidence_v17/summary.json` 已sealed，注册 **`b8d398e8…`**。三槽全部submitted / completed，**0 generation / evaluation failures**，`comparison_complete=true`；同一固定已曝光任务与五份官方公开证据不变。

| native satisfaction | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| 通过项 / 完整58项 | 30 / 58 | 28 / 58 | 27 / 58 |
| 完整得分 | 0.517241 | 0.482759 | 0.465517 |

Initial-MAS由本轮预先登记的v1 body-loss guard保留initial，获得完整提交与评分；其得分高于Native JIT，仍低于Direct，不能写成高于两个baseline。retain-initial的工程资格不证明语义或公开内容全部正确，也没有通过Judge择优或删除组件原始调用。

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 1 / 2 | 6,896 / 15,669 | 17.234 / 14.984 | 13,874 / 8,691 |
| Initial-MAS | 8 / 2 | 153,415 / 13,647 | 62.250 / 14.765 | 144,269 / 22,793 |
| Native JIT | 3 / 2 | 44,981 / 13,971 | 40.235 / 14.359 | 41,984 / 16,968 |

生成 **12调用 / 205,292tokens / 119.719活动秒**；评价 **6调用 / 43,287tokens / 44.108活动秒**；合计 **18模型调用 / 248,579tokens（200,127 input / 48,452 output）**。unknown / estimated / queue idle为0，美元费用`null`。五份公开证据与禁止文章边界不变，actual source hash和原task身份保持注册值。

## v17 WritingBench完整封存：本方法两题完成，但均值仍低于Direct

`outputs/development_pilot_20261004_writing_v17/summary.json` 已sealed，注册 **`769dfbe8…`**。原6槽 **5 submitted / completed，1 Native JIT generation failed，0 evaluation failures**，`comparison_complete=false`；失败的score / native score / normalized score保持`null`。

| 固定任务native mean / 10 | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---|
| WB335 | 8.6 | 7.6 | 8.6 |
| WB433 | 9.0 | 9.0 | `null` / `TemplateRuntimeError` |
| 两题完整原生均值 | 8.8 | 8.3 | `null` |
| normalized failure-zero | 0.866667 | 0.811111 | 0.422222 |

| 方法 | 生成 / judge调用 | 生成 / judge tokens | 生成 / 评价活动秒 | 全部input / output tokens |
|---|---|---|---|---|
| Direct | 2 / 2 | 6,783 / 10,038 | 14.703 / 4.124 | 13,348 / 3,473 |
| Initial-MAS | 17 / 2 | 250,156 / 9,406 | 134.282 / 4.155 | 218,095 / 41,467 |
| Native JIT | 4 / 1 | 59,790 / 6,601 | 46.250 / 1.828 | 52,313 / 14,078 |

生成 **23调用 / 316,729tokens / 195.235活动秒**；评价 **5调用 / 26,045tokens / 10.107活动秒**；合计 **28模型调用 / 342,774tokens（283,756 input / 59,018 output）**。unknown / estimated / queue idle为0，美元费用`null`；未把Native JIT的失败改为观察0分或通过其他轮次有效提交替换。Initial-MAS两题全部完成，其原生均值低于Direct，不能用Native JIT缺失完整均值声称实质击败它。

## v17统一三十槽总账、公开receipt与发布候选边界

五个campaign全部saved summaries已sealed、controller exit0：**30固定槽 / 10任务 / 6 benchmarks，29完整评分 / 1生成失败 / 0评价失败**。Direct与Initial-MAS各 **10/10**，Native JIT **9/10**；唯一生成失败为Native JIT WB433 `TemplateRuntimeError`。每组在自身全部生成终态封存后才评分，没有quality resampling、选分数、替换失败、跨版本挑行或修改固定源 / 证据。

| 方法 | 完整提交 / 注册槽 | 生成 / 评价调用 | 生成 / 评价tokens | 全部模型调用 / tokens | 生成 / 评价活动秒 |
|---|---|---|---|---|---|
| Direct | 10 / 10 | 10 / 34 | 34,764 / 107,970 | 44 / 142,734 | 63.719 / 72.062 |
| Initial-MAS | 10 / 10 | 78 / 34 | 1,240,221 / 50,905 | 112 / 1,291,126 | 606.581 / 56.888 |
| Native JIT | 9 / 10 | 54 / 33 | 628,497 / 115,504 | 87 / 744,001 | 403.829 / 68.155 |

生成合计 **142调用 / 1,903,482tokens / 1,074.129活动秒**；评价 **101模型调用 / 274,379tokens / 197.105活动秒**，活动秒包含不调用模型的pinned instruction checkers。全账 **243模型调用 / 2,177,861tokens（1,833,603 input / 344,258 output）**，所有失败调用仍在账内；unknown / estimated / queue idle为0，美元费用`null`。11调用 / 130,620tokens的v17端到端synthetic probe单独记录，不混入benchmark总数。

按六个sealed source arms的normalized原值计算equal-weight描述性macro：Direct **0.5261194387898478**、Initial-MAS **0.5550506400546497**、Native JIT **0.6182408149930203**。Initial-MAS整体高于Direct、低于Native JIT。统一表完整保留IFEval、RR、DSQA和WritingBench不领先的结果；没有用IFBench成功或DRB恢复覆盖它们，亦不把Native JIT缺失槽当观察0分。

`.runtime/v17_ours_public_receipts_final_baseline_runtime.json`（SHA256 `989b0d414b675bb0bec671600ae8f553015b8a0abfaf4d9923452efe57f2c9ab`）只读10个公开actor receipts，记录 **40 planning / 18 execution / 20 refiner调用**，共78调用 / 1,240,221tokens。全部frozen Team / AgentSpec call caps为null，selected / execution / submission hash一致。只有DRB1因`catastrophic_body_loss`保留initial，其他九案选择revision，没有JSON解析失败触发的fallback。Hash-only planning correction实际出现于IFEval1203 local、DSQA170 predict和WB433 local；IFBench22实际使用一次已有execution protocol correction，IFBench13第一writer正常stop、0次纠错。WB335 **8调用 / 139,120tokens**，WB433 **9调用 / 111,036tokens**，最终规划均stop。这些公开机制记录不是成绩差异的单项因果证明。

本轮runtime **`fbb842503a6b0a07ba5728f28d277a38defbe7661f98c68a257a09a946ecd146`**、runner **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**在全部五注册与30槽期间保持冻结。最终实际一次full suite为 **2,092 passed + 60 subtests / 467.30秒（wrapper469.438秒）**，185 source compile / diff check通过；正式v5八文件、23 unique data / evidence / formal与14 unique checker / NLTK身份匹配。Exposure ledger更新为 **5 selected TEST IDs / 29 campaign references / 1 incidental legacy事件**，不增题，也不抹去已知额外历史ID和unknown可见范围。

example仅代表v17已验证开发候选，实际差异仍只有五角色endpoint / model / response-pin的15个占位字段；library defaults、有限语法scope、unsupported参数fail-closed与revision-only penalty边界不变。此前v16的47-file发布审计属于已提交的旧身份，本轮须另按真实Git候选逐文件审查；`.runtime`、outputs、dataset及本机敏感helper永不加入发布。本轮文档与公开metadata工作为 **0 model API / 0生产和测试编辑 / 0 staging**，本节不对后续尚未登记或尚未封存的版本给出结果或final-clean结论。

## v18 main源候选已转入：公开最低频次与单ATX标题保护，完整库存待封存

v17已独立commit并push，父HEAD为 **`575a7cb8cc3a8172b4b0050cca069f0fb3dcc94d`**。v18隔离实现通过独立只读review及 **256 targeted / 12.79秒**，随后三文件按审阅SHA转入main，记录`.runtime/v18_main_transfer_receipt.json`。当前runtime为 **`47ead26692c3a264eed3794685b9c06d7a6536f7c56ee340ea0ea2f3951fd2b7`**，runner仍为 **`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`**。main实际一次完整测试与真实synthetic预检随后完成，五组固定30槽注册已落盘；本轮benchmark数值仍待各组sealed，不把旧v17完整测试或结果当成v18验证。

三文件为`jit_mas/public_refinement.py`（SHA256 `11a2bf72eaced5d87b094de9b40a59e2c944d270b02fb5dffa010e5d6d707067`）、新`jit_mas/public_literal_constraints.py`（`cc9cd64c56b32e22b03e6188f90230ef38b9fa0ed0f1c3263ffe1008ed4ebbf6`）和新`tests/jit_mas/test_public_literal_constraints.py`（`1412e12b38db231dabe3c213902b789cb04bdba9af37d7747bf5a0918f4270a5`）。变化只在guard的公开可判定保护、配套公开提示和合成行为测试，没有添加benchmark ID路由、私有criterion或参考答案。

`.runtime/public_refinement_v18.config.json`字节SHA仍为 **`f83030961c1f9fe5bdda9da0775a72afeaee1f8444c5e0dea36a8f0c0394de48`**，与v17配置完全相同：没有新flag、frequency变化、温度变化、预算扩大或证据包修改。四生成角色0.6 / judge0、role frequency None、revision-only0.25、planning / public construction strict schema、2m token / 900 active seconds、null call caps、固定任务 / 证据 / checker规则保持原方案。example JSON也保留原有字节，只更新配套说明区分新源候选；它不是已完成的本轮注册或胜出配置推荐。

### Guard v2的有限公开ASCII单词最低频次保护

新compiler只接受原public question / constraints中有限、独立、正向的英文Use / Include指令：单个ASCII字母词、有显式数字最低出现次数，支持相同句中用and连接多项。指令词参数可以有引号；引文示例、否定、条件、部分计数scope、大小写限定及未支持的frequency family保守拒绝，**任一已识别但未支持的frequency family使整个frequency plan为UNKNOWN**，而非只保留容易的部分。这个scope探测是有限文本约定，不能保证覆盖全部自然语言歧义；不重写原任务，也不凭frequency未知状态选择candidate。

计数遍历**完整decoded artifact**，不依赖有截断的public diagnostics词表。Original-case standalone exact-strict出现次数达到公开最低值才为 **PASS**；即使更宽的casefolded overlapping substring计数仍低于最低值才为 **FAIL**；其余为 **UNKNOWN**。Casefold strict计数仅作diagnostic，不独立决定PASS。词法单位包括Unicode字母数字 / underscore / combining marks及内部apostrophe / dash连接，不能靠在复合词或组合单位中抽取ASCII前缀证明strict pass。完整文本计数仍不区分语义与过程注释，不能证明稿件完成公开请求或整体内容正确。

只有initial仍工程eligible、**全部compiled initial rules均PASS**、revision至少一项明确FAIL时，才保留唯一initial并记`explicit_public_literal_minimum_regression`；revision UNKNOWN不是FAIL，initial任一UNKNOWN也不能用于该选择。Audit保存原公共span、匿名可发布计数对应的完整内部规则、两候选hash和选择reason，private evaluator不进入编译或选择。文档只发布匿名counts / 类别 / hash，不复制已选benchmark的原文字面量或完整题面。

### 新单ATX标题分支保持独立，未泛化扩大普通短文阈值

新分支只识别**一个严格Markdown ATX标题、没有正文**的revision，并要求initial至少 **1,000非空白字符 / 3个punctuation-or-line chunks**、revision / initial非空白字符比例不超过 **0.1**。这是与既有短输出heuristic并列的分支，没有把原普通正文的100字符上限整体提高，也不接受任意短回复、普通标题或两行摘要的语义分类。

有限、独立、正向的title / headline-only任务被识别后会抑制结构保留；未支持的only-output scope为UNKNOWN时只抑制新的单标题分支，既有v1短输出heuristic仍按原边界运行。两者都是heuristic而非author checker，有限scope可能漏识别，正确短revision可能被拒绝，101字符普通短文或两chunk正文也可能漏过旧分支。保留initial不证明所有显式或语义约束满足，尤其不能仅因过了工程执行schema就认证其内容正确。

已有JSON / schema local失败保留、provider / review / budget / typed失败继续失败、projected或尚未完成typed construction初稿ineligible等边界不变。Public review与revision仍只有既定 **2模型调用**；编译和本地计数没有额外模型API、Judge调用、质量重采样或分数择优。所有raw失败、各组件调用及成本、两候选hash与最终选择仍完整保存。

顶部最新完整统一表仍是v17，不预填本轮未封存的benchmark数值。Exposure ledger仍为 **5 selected TEST IDs / 29 campaign references / 1 legacy incident**；五个真实新注册的metadata已知，全部运行封存后再同步引用至34，不凭计划增加题目。新的publication审查按实际dirty路径准备为6项：三份source / test文件、实验文档、example MD与最终ledger；example JSON没有参数变化，字节保持已发布v17值，作为未改的配套配置核对，不能为计数而强行修改。当前不执行final privacy结论，不能复用v17的10-file发布审计。

### v18 main单次完整测试、独立真实预检与五组注册

`.runtime/final_v18_validation_baseline_runtime.json`与`.runtime/final_v18_test_metadata_summary_baseline_runtime.json`记录main **唯一一轮full suite：2,215 passed + 60 subtests / pytest448.16秒（wrapper450.157秒）、exit0**；**187 first-party Git source Python compile / diff check通过**。正式v5八文件、23份metadata / data / evidence、14份checker / NLTK及依赖版本身份一致，完整source集合before / after SHA256为 **`4fb7e92bd952b199e816b4b07b18d0ee38bea06bdfb270e3ddf31142e88e507c`**，87份测试文件集SHA256为 **`8e1c91c6f1b4bf06721eb6fca3ff1edb4bb2251277d5ac4883c6b1b91c779f23`**。UTF8测试日志SHA256 **`55ec1eb95158ed562841ebea5a3a8881d814d159562bf17af92fd81e586788d8`**。主工作区runtime / runner与上列冻结身份一致；这些是软件验证，不是benchmark分数。

真实full-MAS synthetic预检`.runtime/full_mas_literal_guard_v18_20261004/preflight.json`已completed，`.runtime/v18_preflight_metadata.json`只读元数据记录 **7模型调用 / 74,505tokens / 31.172秒 / 美元费用null**。4次planning全部finish stop、requested strict schema、实际model pin一致，reconcile使用原有唯一contract correction一次，frozen team call caps为null。Guard为v2，两个匿名最低词频规则下限为 **4 / 5**，initial与revision的exact-strict / casefold-strict / broad计数均分别 **8 / 8 / 8**与 **7 / 7 / 7**，全部PASS，选择revision；没有触发保留initial。Draft / revision / selected hashes和原始调用审计保留，没有author checker、私有评分或benchmark成绩。这个synthetic只证明该路径运行成功，**7调用 / 74,505tokens独立计费，不加入后续30 benchmark槽**，也不证明本轮任务全部满足。

五个真实registration metadata已核对共 **30 fixed generation slots**，全部recorded source为runtime **`47ead266…`** / runner **`2a9e9e26…`**，parsed配置canonical SHA256一致为 **`e828bc174b3b7f1bcf3ddc2aafe2b35b58ab98bb0b09e2dec2dcc4534a85c709`**，与v17相同。Controller执行流程为先注册全部五组，再按既定顺序运行；每组仍在三臂全部生成终态封存后评分。表内只记录已存在的注册身份，结果等封存后追加。

| v18 campaign | 固定来源 / knowledge mode | generation slots | registration hash（结果不含在本表） |
|---|---|---:|---|
| `exposed_v18` | IFEval2 / IFBench2，exposed TEST / closed-book | 12 | `30cc38f8b759ae434c3efe77cab5c835818422b48874812965a6180d54dd954b` |
| `dsqa_evidence_v18` | DSQA2，source EVO / 同两份frozen evidence | 6 | `41ef677d3e6d9129895537554e02529303227594e5284b52ff06160961aebd12` |
| `rr_evidence_v18` | RR1，source EVO / 原shared evidence | 3 | `c4a68e2aac5a8cfcee765b91a2ff211d2d0ca430f51086b26ec33f14ace70d56` |
| `drb_evidence_v18` | DRB1，exposed TEST / 原五份官方证据 | 3 | `dcb9d35f05ad0d4605cef5ae75fcf09884425829a308d9d54b66a3bda21a3318` |
| `writing_v18` | WB2，source EVO / closed-book | 6 | `fd71ece821a79976ef3f0a76a7b2ca06316cf090362ebe9aad071c788a19809a` |

固定源 / task顺序、原pack、模型设置、2m900预算与scoring规则仍同v17，差别是本节明确的v2源实现；parent campaign为同来源v17。当前各benchmark未按本节更新数值或完整macro，全部失败、真实观察0、guard保留与所有成本须随实际sealed summaries一并报出，不能只摘保留成功案例。

`.runtime/final_v18_registration_binding_baseline_runtime.json`（SHA256 `daa82ada781f44aa68e0426b6e434bd054a6b6ddb83310b7f7471c5873341efa`）随后核对五组30槽与parent v17的selected / source / pack / checker / model parameters / budgets / config / scoring一致，顶层差别为code与新iteration登记metadata。完整测试报告补入该binding链接，没有重跑测试；main runtime、runner、raw config及模型配置身份保持上述值。这是注册和软件身份核对，不是已封存答案或成绩证明。

## v18首组指令实验已封存：Initial-MAS四题满分，完整六源仍待后续四组

`outputs/development_pilot_20261004_exposed_v18/summary.json` 已sealed，注册 **`30cc38f8…`**。原12槽 **11 submitted / completed，1 Native JIT generation failed，0 evaluation failures**，`comparison_complete=false`。Initial-MAS四题生成与评分全部成功；IFEval使用pinned author strict prompt-level accuracy，IFBench使用pinned loose prompt-level accuracy，secondary指标不替换主指标。

| 固定任务prompt accuracy | Direct | Initial-MAS | Native JIT |
|---|---:|---:|---|
| IFEval1203 | 1 | 1 | 1 |
| IFEval1246 | 1 | 1 | 1 |
| IFBench13 | 0 | 1 | 0 |
| IFBench22 | 1 | 1 | `null` / `AttributeError` |
| IFEval两题完整均值 | 1.0 | 1.0 | 1.0 |
| IFBench两题完整均值 | 0.5 | 1.0 | `null` |
| IFBench normalized failure-zero | 0.5 | 1.0 | 0.0 |
| 本组两来源等权normalized macro | 0.75 | 1.0 | 0.5 |

IFEval1203三臂instruction secondary均为 **2/2**，1246均 **1/1**；IFBench13为Direct **0/1**、Initial-MAS **1/1**、Native **0/1**，22为Direct与Initial-MAS各 **2/2**，Native失败时secondary保持`null`。Native IFBench13的0是完整提交的观察0分，22的失败仍是缺失分数；failure-zero统计没有把失败冒称真实评分0，也没有给Native计算一个完整IFBench均值。

| 方法 | 完整提交 / 槽 | 生成调用 / tokens | 生成 / checker活动秒 | input / output tokens |
|---|---|---|---|---|
| Direct | 4 / 4 | 4 / 2,741 | 14.281 / 3.578 | 901 / 1,840 |
| Initial-MAS | 4 / 4 | 28 / 282,949 | 148.125 / 3.814 | 242,787 / 40,162 |
| Native JIT | 3 / 4 | 11 / 104,687 | 97.844 / 2.469 | 76,848 / 27,839 |

生成共 **43模型调用 / 390,377tokens / 260.250活动秒**；pinned checkers共 **0模型API / 0tokens / 9.861活动秒**；全部 **320,536 input / 69,841 output tokens**，失败调用已计入，unknown / estimated / queue idle为0，美元费用`null`。同配置独立synthetic预检7调用 / 74,505tokens未混入此账。

Initial-MAS IFEval完整均值从v17的0.5变为本轮1.0，IFBench仍为1.0；本组macro高于Direct与Native failure-zero统计，IFEval则与两个baseline持平。这个局部观察不能提前变成完整六源结论，也不能仅凭分数推断v2保留分支被触发或单项保护的因果；具体候选选择、匿名三值计数和hash关系等整批sealed后的只读公开receipt汇总。当前顶部仍保留完整v17，ledger仍29，后续四组结果继续按完整固定库存报出。

Native IFBench22的公开生成代码随后做了只读AST接口核查，工件`.runtime/v18_native_ifb22_interface_diagnostic_baseline_runtime.json`（SHA256 `41a7997faffdcb916be8419966dbf5c4d5684b491882062f2df1927e847a105d`）。绑定harness的四份生成Python文件hash均一致；生成action在`select_tools()`返回的框架`ToolSelection`上调用了不存在的`get_skills_prompt()`，而该方法属于tool policy，生成policy自身已经实现。公开框架类型与loader注入没有显示adapter错误。这属于Native生成代码接口误用，保留为方法生成/执行失败；没有给返回类型补方法或重试此槽。失败预算实记 **1模型调用 / 20,249tokens / 0执行模型调用 / 0工具调用**，已包含于上表Native全部费用，不另计为隐藏重试。核查没有import或execute生成代码、调用API、读取评分正文或改动source与tests。

## v18统一三十槽最终封存：完整任务、费用与公开候选审计

本节由五份sealed summary、registration metadata与公开actor receipt投影生成，没有读取评分正文、参考答案或private rubric。所有30固定槽均保留，失败不替换；末次统一运行controller的exit状态另由运行日志核对。

| Benchmark | task ID | 方法 | native score | normalized score | generation / evaluation |
|---|---|---|---:|---:|---|
| ifeval | ifeval:1203 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:13 | Direct | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:13 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:13 | Native JIT | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:22 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:22 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:22 | Native JIT | `null` | `null` | failed / submission_failed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Direct | 0.933333 | 0.933333 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Initial-MAS | 0.933333 | 0.933333 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Native JIT | `null` | `null` | failed / submission_failed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Direct | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Initial-MAS | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Native JIT | `null` | `null` | failed / submission_failed |
| researchrubrics | 6847465956a0f6376a6054a7 | Direct | 0.312500 | 0.360465 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Initial-MAS | 0.287500 | 0.337209 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Native JIT | `null` | `null` | failed / submission_failed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Direct | 0.517241 | 0.517241 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Initial-MAS | 0.448276 | 0.448276 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Native JIT | 0.517241 | 0.517241 | submitted / completed |
| writingbench | writingbench:335 | Direct | 8.000000 | 0.777778 | submitted / completed |
| writingbench | writingbench:335 | Initial-MAS | 8.200000 | 0.800000 | submitted / completed |
| writingbench | writingbench:335 | Native JIT | 8.600000 | 0.844444 | submitted / completed |
| writingbench | writingbench:433 | Direct | 9.000000 | 0.888889 | submitted / completed |
| writingbench | writingbench:433 | Initial-MAS | 9.000000 | 0.888889 | submitted / completed |
| writingbench | writingbench:433 | Native JIT | 8.800000 | 0.866667 | submitted / completed |

| 方法 | 生成 calls / tokens | 评价 calls / tokens | 活动秒 generation / evaluation | input / output tokens |
|---|---|---|---|---|
| Direct | 10 / 34,611 | 34 / 82,409 | 63.781 / 72.983 | 88,657 / 28,363 |
| Initial-MAS | 77 / 1,181,727 | 34 / 90,043 | 552.875 / 71.813 | 1,085,726 / 186,044 |
| Native JIT | 235 / 3,984,097 | 4 / 24,314 | 826.157 / 21.984 | 3,789,677 / 218,734 |

全账 **394模型调用 / 5,397,201tokens**；生成 **322 / 5,200,435**、评价 **72 / 196,766**。unknown slots=0，estimated attempts=0，queue idle=0.000秒，美元费用保持`null`。独立synthetic和软件测试不计入30槽。

失败库存：

- ifbench / ifbench:22 / Native JIT：generation=AttributeError，evaluation=none。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- deepsearchqa / deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c / Native JIT：generation=RuntimeError，evaluation=none。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- deepsearchqa / deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c / Native JIT：generation=RuntimeError，evaluation=none。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- researchrubrics / 6847465956a0f6376a6054a7 / Native JIT：generation=AttributeError，evaluation=none。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。

| Benchmark | Direct normalized failure-zero | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| IFEval strict prompt | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| IFBench loose prompt | 0.5000000000 | 1.0000000000 | 0.0000000000 |
| DSQA F1 / shared evidence | 0.4666666667 | 0.4666666667 | 0.0000000000 |
| RR native weighted / shared evidence | 0.3604651163 | 0.3372093023 | 0.0000000000 |
| DRB aggregate / shared evidence | 0.5172413793 | 0.4482758621 | 0.5172413793 |
| WritingBench mean / 10 | 0.8333333333 | 0.8444444444 | 0.8555555556 |

这些normalized值保持各注册bounds；RR真实观察0与缺失分数不同，负分不裁剪。完整均值遇到任一未评分槽为`null`。六来源macro为事后描述，不是事前新增的正式主指标；两个instruction来源可以分别看各原主指标。

| 同题两方法都完整评分的比较 | ours wins | ties | losses | 未成对评分（不计为胜出） |
|---|---:|---:|---:|---:|
| Initial-MAS vs Direct | 2 | 6 | 2 | 0 |
| Initial-MAS vs Native JIT | 2 | 2 | 2 | 4 |

上表只比较同题已评分normalized值，以10道原任务为固定库存；缺失pair单独记录，不当作质量胜出，不替代原benchmark指标或六来源macro。

公开10项actor投影`.runtime/v18_ours_public_receipts_baseline_runtime.json`，SHA256 `4653987ba13fe448ac5362b289107e24f469967e160b1c057280e7d4b0817815`。读取的是公开执行与refinement审计，题面、原答案、literal词与span不在本节复制。

| 公开候选审计指标 | 全部10槽实际汇总 |
|---|---|
| all_10_ours_terminal | `true` |
| all_code_matches_registration | `true` |
| all_saved_teams_uncapped | `true` |
| selected_candidate_counts | `{"initial_draft": 2, "revision": 8}` |
| selection_reason_counts | `{"catastrophic_revision_body_loss": 1, "markdown_heading_only_large_draft_body_loss": 1, "validated_revision_without_catastrophic_body_loss": 8}` |
| refinement_status_counts | `{"completed": 10}` |
| guard_version_counts | `{"public-artifact-regression-guard-v2": 10}` |
| all_selected_submission_execution_and_audit_hashes_match | `true` |
| total_generation_tokens | `1181727` |
| total_generation_model_calls | `77` |
| total_planning_calls | `40` |
| total_execution_model_calls | `17` |
| total_refinement_calls | `20` |
| literal_regression_selections | `0` |
| heading_only_selections | `1` |
| local_invalid_revision_selections | `0` |

| task ID | refinement status | selected | selection reason |
|---|---|---|---|
| ifeval:1203 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifeval:1246 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:13 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:22 | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | completed | revision | validated_revision_without_catastrophic_body_loss |
| 6847465956a0f6376a6054a7 | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepresearch_bench_ii:1 | completed | initial_draft | markdown_heading_only_large_draft_body_loss |
| writingbench:335 | completed | initial_draft | catastrophic_revision_body_loss |
| writingbench:433 | completed | revision | validated_revision_without_catastrophic_body_loss |

候选选择在评分前完成，不根据judge分数选稿；计数或结构guard只是有限的公开工程保护，不是语义正确证明。实际未触发的分支不能据分数改善解释为已发生救回。若存在component failure，原response、异常与预算仍保留，不能声称全部组件调用成功。

曝光台账同步为 **5 selected TEST IDs / 34 campaign references / 1 legacy incident**，只新增本轮五个实际注册引用；已知额外历史RR ID和unknown可见范围仍保留，不作全局只有五题曝光或剩余历史TEST干净声明。正式v5八文件、原共同证据和模型配置不修改。

后续独立版本的公开诊断设计尚未实现、未登记、没有成绩：筛选任务应把原题hard conditions与inferred rubric建议分离，并用原始观测、单位、scope重算数值比较，未知证据保持UNKNOWN；写作任务可对已验证空issues且纯CR/LF删除造成的段落退化保留initial，但单行/代码等公开布局转换必须排除，近似字数不变成隐含精确CJK阈值。这两项不能引用为当前v18已经实现或评分提升原因。

## v19公开观测候选：原题引用、完整字数与最小修改，结果尚未登记

v18已独立commit与push，父HEAD为`b30414fbe79b062c0a8adc5dfc2e985f354e7770`。本候选runtime冻结为`85477ec5e928af0ad7bd853d513a560bfff4d843cfd16d35789c770333cc197f`，runner仍为`2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`。配置字节仍为`f83030961c1f9fe5bdda9da0775a72afeaee1f8444c5e0dea36a8f0c0394de48`；没有模型、温度、frequency、预算、源包或选题变化，没有添加API调用或质量重采样。顶部仍为完整v18，曝光台账仍34个campaign references。

Guard v3在现有两次review / revision调用中加入公开诊断：完整decoded artifact的Unicode已赋名Han ideograph字符、非空白字符、非空行、空行分段及换行计数；Han字符不是words / tokens或语言判定，没有隐含正文限定，也不把近似长度改成精确阈值。另用有限Decimal语法检查公开draft / contributor材料及已验证review中打印的数字比较，最多128项且明确truncated；不支持的数值格式保持未观测，不能从科学计数法、千分位、分数或算式中截取ASCII尾部来证明比较。程序仅验证打印的算式，不验证数值来源、字段、单位、日期、candidate membership或引用上下文，正确引用一个FALSE条件也不是自动答案缺陷。

Review使用原有PublicIssue三字段，在public_basis字符串中可用`[TASK_QUOTE]...[/TASK_QUOTE]`给原题引用；本地只确认8至512字符引文是否为原question / constraints精确substring，未标注或未验证状态保留，不筛掉合法的source-backed事实修正。精确substring不证明引文是正向指令、适用scope或能推出critique。Inferred rubrics及上游PASS声明仍是fallible建议，不能变成新硬排除。**没有实现完整的typed membership contract**；仅凭这些诊断不能保证最终自由文本服从全部条件。

Ordinary revision的附加提示要求已验证`issues=[]`时逐字符保留draft（JSON换行转义仍保留段落），有issues时做有依据的局部修正；active positional / numeric construction仍按原schema生成。新增窄选择分支仅在initial原工程eligible、空issues、至少两个空行分隔段落、revision为单行且恰好等于initial只删除CR / LF时保留initial，reason为`empty_review_line_break_only_regression`。其他空白或内容变化不触发；有限布局转换词、单段/单行及代码/表格等请求将scope降为UNKNOWN并抑制该分支。该heuristic可能漏掉未支持的合法转换，不认证initial语义更好。两次原调用、revision原文、异常、hash与全部成本仍保留；guard默认off行为与原JSON schema不变。

开发中先后保留了实际检查：新的功能/边界模块最后为**67 passed / 1.28秒**，更早相关六模块为258 passed / 10.97秒（当时尚未加入末尾逗号边界用例）。首次完整检查误用不限定目录的pytest命令，把outputs及paper中的历史source snapshots也收集进来，产生**91 collection errors / 85.83秒（wrapper87.672秒）**，没有完成测试断言，不能报告passed；保留`.runtime/v19_initial_collection_error_validation.json`与日志SHA`ab05456e9c3010af1288044db13b35ea977e141db34f56fba4e3c5e6fb629108`。随后将完整收集范围限定为当前`tests/`，新版唯一正确scope完整验证仍运行中。排查只查看错误路径/类型，不为方法设计读取历史评分正文。

两个独立synthetic真实API预检不含benchmark / evaluator：第一source候选预检completed，2调用 / 8,592tokens / 5.719秒；因末尾逗号导致初稿FALSE算式观测数为0，保留此缺口后修正有限语法。最终冻结source第二预检completed，**2调用 / 9,215tokens / 5.406秒**，初稿FALSE算式观测1、review issue1、原题引文exact substring1、选择revision。人读虚构记录显示eligibility结论得到修正，但修订仍打印了未显式标FALSE的原比较条件；不能称全部数学表达已修复或有独立质量认证。两预检合计**4调用 / 17,807tokens / 11.125秒**独立计费，不混入随后30槽；第二预检工件`.runtime/public_guarded_review_v19b_20261004/preflight.json`。这些属于开发hypothesis与软件路径检查，不是benchmark分数。

## v19正确scope完整验证与五组注册均已通过

当前tests/完整检查为 **2,282 passed + 60 subtests / pytest452.21秒（wrapper454.219秒）、exit0**。四份变更Python source / test编译通过，文件before / after与runtime身份一致；UTF8完整测试日志SHA256 `3192a571096ee6bd025522ec0643863300e6d508f2b8b88a688cc4c8c8be28e3`。首次历史snapshot误收集的91错误另列，不作为完整测试passed，也未覆盖其日志。

| 当前变更Python文件 | full validation冻结SHA256 |
|---|---|
| `jit_mas/public_review_observations.py` | `a3769147468ce1e1a6db8e5c82a6f58b9baf35a79956b15ce40a7fe1c967122c` |
| `jit_mas/public_refinement.py` | `131997129e8f06db0a245d0486421bbd8014fc6d62057b2fd73804ba305ef4fe` |
| `tests/jit_mas/test_public_review_observations.py` | `45f42bb824491a48e24e5df1a1e7a8852ffa9e3b50448066f37608887ab1b4be` |
| `tests/jit_mas/test_public_literal_constraints.py` | `af0aef66faf1b7c8be9702a11dc0038b79eea07d1f9b590b21843bd7a02048d0` |

五组固定30槽登记已完成；生成前统一binding核验通过。原检查helper把父registration内容hash误当成parent登记使用的文件字节SHA，造成controller在生成前退出；修正为检查真实`registration_sha256`后，沿同五份registration继续，没有替换或重登记，没有已开始的generation被重试。这个bookkeeping错误不算模型/任务失败，也不隐藏模型成本。

| v19 campaign | 槽 | 实际registration hash |
|---|---:|---|
| `exposed_v19` | 12 | `a2a0723b5bfaae25665d6d36c822ba6d128b945ff8b273fd917420136374d7a0` |
| `dsqa_evidence_v19` | 6 | `5397777cb8af9be89b524534fcfc668b4b20a1aeff326946308c1ff5f3534b52` |
| `rr_evidence_v19` | 3 | `189d3fe12b7770cf5db73ccc570e34371466e6281e20c4cbb9bce8ea2c29a5e8` |
| `drb_evidence_v19` | 3 | `e45ed441e388a4464a45531673cb5cc34bb3071f8970ef37ad2b0159c62beec1` |
| `writing_v19` | 6 | `d937aa9bb8601698a08d565236a7ae2a8adb67d8eac387d7d39b69059ff9fb8d` |

完整binding工件`.runtime/final_v19_registration_binding_root.json`，SHA256 `36029ecbd649b250fbe4bf7c9c81f19e7db491126c2d066da1e7067156052456`。相对各source v18父轮，task / pack / checker / model parameters / config / budgets / slot order / scoring保持完全相同，仅code与新iteration登记metadata变化。全部登记完成且核验通过后才开始生成，source此后冻结至全部30槽终结。

顶部仍为已封存v18；v19数值、实际调用、组件失败、完整库存和曝光引用待全组sealed后更新。

## v19统一三十槽最终封存：完整任务、费用与公开候选审计

本节由五份sealed summary、registration metadata与公开actor receipt投影生成，没有读取评分正文、参考答案或private rubric。所有30固定槽均保留，失败不替换；末次统一运行controller的exit状态另由运行日志核对。

| Benchmark | task ID | 方法 | native score | normalized score | generation / evaluation |
|---|---|---|---:|---:|---|
| ifeval | ifeval:1203 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Native JIT | 0.000000 | 0.000000 | submitted / completed |
| ifeval | ifeval:1246 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:13 | Direct | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:13 | Initial-MAS | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:13 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:22 | Direct | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:22 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:22 | Native JIT | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Direct | 0.933333 | 0.933333 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Initial-MAS | 0.857143 | 0.857143 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Direct | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Initial-MAS | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Native JIT | 0.666667 | 0.666667 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Direct | 0.450000 | 0.488372 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Initial-MAS | 0.575000 | 0.604651 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Native JIT | 0.450000 | 0.488372 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Direct | 0.517241 | 0.517241 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Initial-MAS | 0.431034 | 0.431034 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Native JIT | 0.465517 | 0.465517 | submitted / completed |
| writingbench | writingbench:335 | Direct | 8.400000 | 0.822222 | submitted / completed |
| writingbench | writingbench:335 | Initial-MAS | 7.600000 | 0.733333 | submitted / completed |
| writingbench | writingbench:335 | Native JIT | 8.600000 | 0.844444 | submitted / completed |
| writingbench | writingbench:433 | Direct | 9.000000 | 0.888889 | submitted / completed |
| writingbench | writingbench:433 | Initial-MAS | 8.600000 | 0.844444 | submitted / completed |
| writingbench | writingbench:433 | Native JIT | `null` | `null` | failed / submission_failed |

| 方法 | 生成 calls / tokens | 评价 calls / tokens | 活动秒 generation / evaluation | input / output tokens |
|---|---|---|---|---|
| Direct | 10 / 35,128 | 34 / 97,354 | 65.845 / 73.937 | 103,763 / 28,719 |
| Initial-MAS | 75 / 1,174,247 | 34 / 127,835 | 512.048 / 74.487 | 1,154,564 / 147,518 |
| Native JIT | 32 / 371,171 | 33 / 110,418 | 295.142 / 71.985 | 382,524 / 99,065 |

全账 **218模型调用 / 1,916,153tokens**；生成 **117 / 1,580,546**、评价 **101 / 335,607**。unknown slots=0，estimated attempts=0，queue idle=0.016秒，美元费用保持`null`。独立synthetic和软件测试不计入30槽。

失败库存：

- writingbench / writingbench:433 / Native JIT：generation=AttributeError，evaluation=none。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。

| Benchmark | Direct normalized failure-zero | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| IFEval strict prompt | 1.0000000000 | 1.0000000000 | 0.5000000000 |
| IFBench loose prompt | 0.0000000000 | 0.5000000000 | 0.5000000000 |
| DSQA F1 / shared evidence | 0.4666666667 | 0.4285714286 | 0.8333333333 |
| RR native weighted / shared evidence | 0.4883720930 | 0.6046511628 | 0.4883720930 |
| DRB aggregate / shared evidence | 0.5172413793 | 0.4310344828 | 0.4655172414 |
| WritingBench mean / 10 | 0.8555555556 | 0.7888888889 | 0.4222222222 |

这些normalized值保持各注册bounds；RR真实观察0与缺失分数不同，负分不裁剪。完整均值遇到任一未评分槽为`null`。六来源macro为事后描述，不是事前新增的正式主指标；两个instruction来源可以分别看各原主指标。

| 同题两方法都完整评分的比较 | ours wins | ties | losses | 未成对评分（不计为胜出） |
|---|---:|---:|---:|---:|
| Initial-MAS vs Direct | 2 | 4 | 4 | 0 |
| Initial-MAS vs Native JIT | 3 | 1 | 5 | 1 |

上表只比较同题已评分normalized值，以10道原任务为固定库存；缺失pair单独记录，不当作质量胜出，不替代原benchmark指标或六来源macro。

公开10项actor投影`.runtime/v19_ours_public_receipts_root.json`，SHA256 `11fa45801e49c65bfc015974e0c9e5684bade8e8663a40f060cfabcea47dc8d2`。读取的是公开执行与refinement审计，题面、原答案、literal词与span不在本节复制。

| 公开候选审计指标 | 全部10槽实际汇总 |
|---|---|
| all_10_ours_terminal | `true` |
| all_code_matches_registration | `true` |
| all_saved_teams_uncapped | `true` |
| selected_candidate_counts | `{"initial_draft": 2, "revision": 8}` |
| selection_reason_counts | `{"invalid_local_revision_response": 1, "markdown_heading_only_large_draft_body_loss": 1, "validated_revision_without_catastrophic_body_loss": 8}` |
| refinement_status_counts | `{"completed": 9, "completed_with_component_failure": 1}` |
| guard_version_counts | `{"public-artifact-regression-guard-v3": 10}` |
| all_selected_submission_execution_and_audit_hashes_match | `true` |
| total_generation_tokens | `1174247` |
| total_generation_model_calls | `75` |
| total_planning_calls | `39` |
| total_execution_model_calls | `16` |
| total_refinement_calls | `20` |
| literal_regression_selections | `0` |
| empty_review_line_break_selections | `0` |
| heading_only_selections | `1` |
| local_invalid_revision_selections | `1` |

| task ID | refinement status | selected | selection reason |
|---|---|---|---|
| ifeval:1203 | completed_with_component_failure | initial_draft | invalid_local_revision_response |
| ifeval:1246 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:13 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:22 | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | completed | revision | validated_revision_without_catastrophic_body_loss |
| 6847465956a0f6376a6054a7 | completed | initial_draft | markdown_heading_only_large_draft_body_loss |
| deepresearch_bench_ii:1 | completed | revision | validated_revision_without_catastrophic_body_loss |
| writingbench:335 | completed | revision | validated_revision_without_catastrophic_body_loss |
| writingbench:433 | completed | revision | validated_revision_without_catastrophic_body_loss |

候选选择在评分前完成，不根据judge分数选稿；计数或结构guard只是有限的公开工程保护，不是语义正确证明。实际未触发的分支不能据分数改善解释为已发生救回。若存在component failure，原response、异常与预算仍保留，不能声称全部组件调用成功。

曝光台账同步为 **5 selected TEST IDs / 39 campaign references / 1 legacy incident**，只新增本轮五个实际注册引用；已知额外历史RR ID和unknown可见范围仍保留，不作全局只有五题曝光或剩余历史TEST干净声明。正式v5八文件、原共同证据和模型配置不修改。

完整typed membership contract仍未实现：筛选任务应把原题hard conditions与inferred rubric建议分离，并用原始观测、单位、scope重算数值比较，未知证据保持UNKNOWN；本轮写作保护已对已验证空issues且纯CR/LF删除造成的段落退化实现有限initial保留，但单行/代码等公开布局转换必须排除，近似字数不变成隐含精确CJK阈值。完整predicate合同仍未实现，不能将有限算式/substring观测称为membership已由程序验证。

## v19已封存后的公开观察边界与回退结果

完整投影显示，十个初稿及其公开材料、十份 review 均没有命中有限的显式数值关系表达式；因此本轮不能宣称程序实际纠正了 benchmark 中的数值筛选。Synthetic 预检中的算式观察与真实批次须分开。34 个 review issue 的引文观察分别为 `exact_public_substring=12`、`unknown=20`、`unverified=2`；每条 issue 的 exact 状态只表示至少一个受支持引文命中，不表示全部引文命中，也不证明该建议的适用范围或推论。有限数值 parser 最多观察128个匹配子式，对空格千位分隔等未完整支持的表达可能截取局部关系，不能称为整条算式、实体资格或语义验证。

本轮空 review 纯换行分支触发 **0** 次。WB433 的公开初稿为802字符，选定修订为73字符；变化包含正文变化，不是只删除CR/LF，因此新分支不适用；既有正文损失规则的1000非空白字符初稿门槛也未满足。空issues复制提示不能保证模型照做，未触发分支不能称为已经救回正文。布局词表有残余漏判，例如有限语法外的单行输出表述可能造成误保留；Unicode-name Han计数、`splitlines()`行计数与归一LF事件各有约定，不能混称自然语言字数或全部Unicode换行验证。

IFB13 的只读诊断仅访问已封存公开 actor 成品与 pinned checker 的通用 tokenizer 实现，没有读取实际 grader task arguments、评分正文、私有 rubric 或数据CSV。Typed构造有24个前置句槽、35个前缀词槽和32个后缀词槽；渲染成品与保存的3960字符答案及其哈希完全一致。结构句数与通用NLTK句分割均为25，两个ASCII字母词槽却各被通用tokenizer拆为两个词，使目标句68个结构词槽变成70词，指定词从预期第36词移到第37词。这个可观察差异解释位置偏移，但不证明整体0分只有一个原因。诊断工件 `.runtime/v19_ifb13_tokenization_diagnostic_baseline_runtime.json`，SHA256 `0124ae48c6b1cb69f8acd3406ec55fd2d23a140de0b716d9e478476ec810d5dc`。后续应明确定义公开token convention并泛化校验每个词槽在该通用tokenizer下为一词；不能按某个题ID或本例单词特判，修复必须绑定新源码身份、新测试与新实验，不回写v19答案或重算旧成绩。

v18与v19都保留为各自完整固定批次：Initial-MAS的IFEval为1.0→1.0，IFBench为1.0→0.5，DSQA为0.466667→0.428571，RR为0.2875→0.575，DRB为0.448276→0.431034，WritingBench为8.6→8.1。六来源事后macro为0.6827660459→0.6255243272。v19不是所有来源都改善的候选；temperature0.6单次运行与同时改动的提示使这些差值不能作为单一改动的因果证据，baseline自身表现及失败也变化。当前不能宣称已稳定或全面超越baseline，不能拼接不同版本的最高行当作一次方法结果。

## v20候选：原题数值观测与局部修订

用户进一步要求尽可能在所有benchmark领先。本轮在相同固定10题、三个方法和原共同预算上检验三项通用方法改动；原始v19结果不回写，不按评估分数选稿。

- 词槽协议更新为 `public-word-position-slots-v2`：两个数组使用同一有限正向trie regex，排除整个受支持通用Treebank多词compound family及其大小写，不针对题ID或本例词特判。关键词落入该不支持family时compiler保守decline，不替换语义、不引入runtime NLTK/download或新重试。实际own模块129 tests、10相关模块437 tests通过，离线580个通用tokenizer比较一致；这不是全部语言句界或官方任务判定认证。
- 新 `public_membership_observations=true` 默认关闭，向既有contributor/writer和公开review/revision传入原题、constraints、条件span/hash、canonical fixed public pack中的完整表格与有限Decimal检查。明确字段、单位、原始cell/entity和declared year足够时才计算；其余UNKNOWN。上游PASS单独保留为声明，整体membership始终UNKNOWN、independently_verified=False，不生成隐藏正确名单、别名等价判断或语义资格证明。日期仅依given source.date与原题唯一year声明绑定，没有通用正文cohort矛盾解析或独立历史验证；结果只在记录的declaredscope条件下成立。
- 新 `public_revision_mode=patch` 默认仍full，普通修订改为既有第二次调用输出 `edits`：每项绑定已验证issue索引，old_text必须在同一原始初稿出现恰好一次，程序拒绝重叠/链式/不存在的匹配，最后同时应用。原文可有重叠出现，不能以str.count非重叠计数代替唯一性检查。空issues只允许空edits，在本地保留完整原稿；missed defect也会被保留，不称正确性认证。Active positional/numeric保留原构造schema，provider/review/budget/typed失败处理不放宽，guard默认关闭；已开启guard的普通patch应用错误可以保留eligible初稿并记录component failure、原response/hash/调用成本。整篇也可能被声明为编辑区，所以仅保证区间外原文保真，最小修改与全部布局保留仍是提示要求。候选guard版本为v4。

实际公开DSQA数值题预检得到5个threshold、1张表52行、2个binding与3个UNKNOWN，共104个cell checks（59PASS/45FAIL/0numericUNKNOWN），其中composite阈值发现初稿literal mention的一项明确FAIL；literal mention不是语义成员判断。另一个历史/语义筛选题没有被转换成数值资格判断。完整数值题sidecar紧凑JSON68,225字符，会增加既有调用input tokens，必须按实际计费，不能称免费或等实际算量。

新版源runtime SHA256 `33f814e6d036ae2b0758d84a13607de21f26954381da3fa877e5d2c58d59c7ce`，runner `2a9e9e26f5987b45062f3b970542880c7c58e8de6e557d17f7a289cf571b3afc`。模型角色参数、温度、penalty、工具和共同预算保留v19；只新增上述两个opt-in设置，private config文件SHA256 `6bd458cdec819c2f3499d916e45fab131b07639bfbf26d1f0677cb6e2ceaaf3c`。正式v5八文件不改变。

三个独立synthetic真实API预检全部completed，固定各2次调用，合计 **6 calls / 31,097tokens**，不含benchmark或evaluator。空review场景在本地逐字符保留了两段文本；数值场景应用两项精确edit并明确标FALSE/TRUE，未修改中间换行；词槽场景经过新schema及renderer完成，但没有独立质量评分。六次检查前后source身份均相同。另一次独立tiny模型身份请求 **1 call / 10tokens** 确认response-model名称与原pin相同，GET /models不计模型调用；模型名称不能证明隐藏weights完全未变。合计 **7 calls / 31,107tokens** 单独计费，不能混入随后固定30槽。此时全套检查正在运行，尚无v20 benchmark结果，不提前宣称全面提升。

## v20复合单位补正后的最终冻结

原候选runtime `33f814e6d036ae2b0758d84a13607de21f26954381da3fa877e5d2c58d59c7ce` 的初次完整检查在结束前由root主动终止，尚未注册或生成v20 benchmark。原因是同行最后找到recognized unit 后截断 `kg/km`、`points per student` 等分母的明确correctness缺口；未把中断测试称为通过。部分日志SHA256 `5aa215af1e8ea7657cefb05b22ecea140b3effb8734b7bf36785deb42da8fd71`，归档 `.runtime/v20_initial_interrupted_validation.json`。源代码补正后只把有限recognized-unit后的分母/指数/乘积/连字符复合后缀设UNKNOWN，不扩date或语义能力；14个新增synthetic unit cases通过，模块累计 **77 targeted tests**，同行只读14negative与2positive control复核通过。Actual公开数值题104个已绑定cell检查及其59PASS/45FAIL观察不变，另三字段仍UNKNOWN。

最终module SHA256 `3133876f736bd65a7a336ec81c0712e6c601c2d6de903a682996e860e3ce6f43`，模块测试 `30b9b03c0c5ac5f895bdeb057e823aead32ee77eee1fa8e8e5889c3129bbfd66`；最终runtime `d2e59c721302cfcba5b00602dd9a344b477ca07f50b1d8b7e5565b5ddc934227`，runner及private config仍是前节值。

最终源码身份又完成三项固定各2调用的synthetic真实API预检，全部completed且before/after identity相同，合计 **6 calls / 30,831tokens**，无benchmark或judge；原候选6调用/31,097tokens和独立模型身份1调用/10tokens仍保留。最终两项ordinary场景分别通过本地空edits完全拷贝和精确数值修复，word-slot场景通过新schema和renderer；这不等于独立质量或官方任务check。全部独立预检及身份检查共 **13 calls / 61,938tokens**，不能计入随后30槽。最终preflight工件 `.runtime/public_patches_v20b_20261004/preflight.json`，SHA256 `8614c66bcf601fe9b9035b77e59485baa3d50dd48e9e2d521038b94d080922d3`。最终源码完整验证已经完成：2,467 passed、1 个预期 warning、60 subtests passed，pytest 用时 382.77s；12 个变更 Python 源码及测试文件的 AST/compile 验证通过，全部文件 SHA256 及 runtime 在验证前后保持一致。完整日志 SHA256 8004440146324dc2748f309bd60aa40a60e426e35b30783a41e25211158bf150。五组共 30 个生成槽位在生成前完成注册，绑定检查确认相对 v19 只新增 public_revision_mode=patch 与 public_membership_observations=true 配置；题目、共享材料、checker、模型参数、预算和评分顺序均保持一致。随后五组均已封存，完整库存及失败分析见后节。

## v20统一三十槽最终封存：完整任务、费用与公开候选审计

本节由五份sealed summary、registration metadata与公开actor receipt投影生成，没有读取评分正文、参考答案或private rubric。所有30固定槽均保留，失败不替换；末次统一运行controller的exit状态另由运行日志核对。

| Benchmark | task ID | 方法 | native score | normalized score | generation / evaluation |
|---|---|---|---:|---:|---|
| ifeval | ifeval:1203 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1203 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Direct | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifeval | ifeval:1246 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:13 | Direct | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:13 | Initial-MAS | 1.000000 | 1.000000 | submitted / completed |
| ifbench | ifbench:13 | Native JIT | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:22 | Direct | 0.000000 | 0.000000 | submitted / completed |
| ifbench | ifbench:22 | Initial-MAS | `null` | `null` | failed / submission_failed |
| ifbench | ifbench:22 | Native JIT | 1.000000 | 1.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Direct | 0.400000 | 0.400000 | submitted / completed |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Initial-MAS | `null` | `null` | submitted / incomplete |
| deepsearchqa | deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | Native JIT | `null` | `null` | submitted / incomplete |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Direct | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Initial-MAS | 0.000000 | 0.000000 | submitted / completed |
| deepsearchqa | deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | Native JIT | `null` | `null` | failed / submission_failed |
| researchrubrics | 6847465956a0f6376a6054a7 | Direct | 0.412500 | 0.453488 | submitted / completed |
| researchrubrics | 6847465956a0f6376a6054a7 | Initial-MAS | `null` | `null` | submitted / interrupted_evaluation_no_resample |
| researchrubrics | 6847465956a0f6376a6054a7 | Native JIT | 0.387500 | 0.430233 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Direct | 0.551724 | 0.551724 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Initial-MAS | 0.500000 | 0.500000 | submitted / completed |
| deepresearch_bench_ii | deepresearch_bench_ii:1 | Native JIT | 0.500000 | 0.500000 | submitted / completed |
| writingbench | writingbench:335 | Direct | 8.600000 | 0.844444 | submitted / completed |
| writingbench | writingbench:335 | Initial-MAS | 7.600000 | 0.733333 | submitted / completed |
| writingbench | writingbench:335 | Native JIT | 8.800000 | 0.866667 | submitted / completed |
| writingbench | writingbench:433 | Direct | 9.000000 | 0.888889 | submitted / completed |
| writingbench | writingbench:433 | Initial-MAS | 9.000000 | 0.888889 | submitted / completed |
| writingbench | writingbench:433 | Native JIT | 8.600000 | 0.844444 | submitted / completed |

| 方法 | 生成 calls / tokens | 评价 calls / tokens | 活动秒 generation / evaluation | input / output tokens |
|---|---|---|---|---|
| Direct | 10 / 33,099 | 34 / 106,788 | 56.062 / 73.562 | 112,839 / 27,048 |
| Initial-MAS | 77 / 1,346,030 | 6 / 33,716 | 492.578 / 28.797 | 1,245,393 / 134,353 |
| Native JIT | 31 / 431,696 | 33 / 118,663 | 492.297 / 79.451 | 427,720 / 122,639 |

已记录账目 **191模型调用 / 2,069,992tokens**；生成 **118 / 1,810,825**、评价 **73 / 259,167**。unknown slots=1，estimated attempts=3，queue idle=0.000秒，未知消耗保持缺失，此处调用/tokens仅是已持久化下界。美元费用保持`null`。独立synthetic和软件测试不计入30槽。

失败库存：

- ifbench / ifbench:22 / Initial-MAS：generation=ValidationError，evaluation=none，evaluation_status=submission_failed。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- deepsearchqa / deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c / Native JIT：generation=RuntimeError，evaluation=none，evaluation_status=submission_failed。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- deepsearchqa / deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c / Initial-MAS：generation=none，evaluation=none，evaluation_status=incomplete。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- deepsearchqa / deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c / Native JIT：generation=none，evaluation=none，evaluation_status=incomplete。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。
- researchrubrics / 6847465956a0f6376a6054a7 / Initial-MAS：generation=none，evaluation=none，evaluation_status=interrupted_evaluation_no_resample。native分数缺失，failure-zero仅是保守汇总规则，不冒称已评分0。

| Benchmark | Direct normalized failure-zero | Initial-MAS | Native JIT |
|---|---:|---:|---:|
| IFEval strict prompt | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| IFBench loose prompt | 0.0000000000 | 0.5000000000 | 0.5000000000 |
| DSQA F1 / shared evidence | 0.2000000000 | 0.0000000000 | 0.0000000000 |
| RR native weighted / shared evidence | 0.4534883721 | 0.0000000000 | 0.4302325581 |
| DRB aggregate / shared evidence | 0.5517241379 | 0.5000000000 | 0.5000000000 |
| WritingBench mean / 10 | 0.8666666667 | 0.8111111111 | 0.8555555556 |

这些normalized值保持各注册bounds；RR真实观察0与缺失分数不同，负分不裁剪。完整均值遇到任一未评分槽为`null`。六来源macro为事后描述，不是事前新增的正式主指标；两个instruction来源可以分别看各原主指标。

| 同题两方法都完整评分的比较 | ours wins | ties | losses | 未成对评分（不计为胜出） |
|---|---:|---:|---:|---:|
| Initial-MAS vs Direct | 1 | 4 | 2 | 3 |
| Initial-MAS vs Native JIT | 2 | 3 | 1 | 4 |

上表只比较同题已评分normalized值，以10道原任务为固定库存；缺失pair单独记录，不当作质量胜出，不替代原benchmark指标或六来源macro。

公开10项actor投影`.runtime/v20_ours_public_receipts_root.json`，SHA256 `4b308ae404a8934f9b0000a65e0d01baada63b4a956b3acf92280e919a916ee5`。读取的是公开执行与refinement审计，题面、原答案、literal词与span不在本节复制。

| 公开候选审计指标 | 全部10槽实际汇总 |
|---|---|
| all_10_ours_terminal | `true` |
| all_code_matches_registration | `true` |
| all_saved_teams_uncapped | `true` |
| selected_candidate_counts | `{"None": 1, "initial_draft": 1, "revision": 8}` |
| selection_reason_counts | `{"None": 1, "invalid_public_patch_application": 1, "validated_revision_without_catastrophic_body_loss": 8}` |
| refinement_status_counts | `{"completed": 8, "completed_with_component_failure": 1, "failed": 1}` |
| guard_version_counts | `{"public-artifact-regression-guard-v4": 10}` |
| all_selected_submission_execution_and_audit_hashes_match | `true` |
| selected_candidate_hash_check_count | `9` |
| all_saved_refinement_audit_hashes_match | `true` |
| total_generation_tokens | `1346030` |
| total_generation_model_calls | `77` |
| total_planning_calls | `39` |
| total_execution_model_calls | `18` |
| total_refinement_calls | `20` |
| literal_regression_selections | `0` |
| empty_review_line_break_selections | `0` |
| heading_only_selections | `0` |
| local_invalid_revision_selections | `0` |
| patch_status_counts | `{"applied": 7, "failed": 1}` |
| applied_patch_edit_count | `20` |
| applied_zero_edit_patches | `4` |
| patch_application_initial_selections | `1` |
| membership_observation_slots | `10` |

| task ID | refinement status | selected | selection reason |
|---|---|---|---|
| ifeval:1203 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifeval:1246 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:13 | completed | revision | validated_revision_without_catastrophic_body_loss |
| ifbench:22 | failed | None | None |
| deepsearchqa:d042556cd6083779d4fe4afa21924d414f4966a16a0f271c9ae30a2bc8c3a60c | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepsearchqa:ba5cecc13c11812a2ed051df55fef96311cec89e54def887ab0d760620fec02c | completed | revision | validated_revision_without_catastrophic_body_loss |
| 6847465956a0f6376a6054a7 | completed | revision | validated_revision_without_catastrophic_body_loss |
| deepresearch_bench_ii:1 | completed_with_component_failure | initial_draft | invalid_public_patch_application |
| writingbench:335 | completed | revision | validated_revision_without_catastrophic_body_loss |
| writingbench:433 | completed | revision | validated_revision_without_catastrophic_body_loss |

候选选择在评分前完成，不根据judge分数选稿；计数或结构guard只是有限的公开工程保护，不是语义正确证明。实际未触发的分支不能据分数改善解释为已发生救回。若存在component failure，原response、异常与预算仍保留，不能声称全部组件调用成功。

曝光台账同步为 **5 selected TEST IDs / 44 campaign references / 1 legacy incident**，只新增本轮五个实际注册引用；已知额外历史RR ID和unknown可见范围仍保留，不作全局只有五题曝光或剩余历史TEST干净声明。正式v5八文件、原共同证据和模型配置不修改。

全条件typed membership及最终集合认证仍未实现：筛选任务应把原题hard conditions与inferred rubric建议分离，并用原始观测、单位、scope重算数值比较，未知证据保持UNKNOWN；本轮注册patch修订使已验证空issues只允许空edits，本地保留完整原稿；漏审缺陷也会保留。非空issues的合法patch只保证编辑区间外原文保真，不能认证最小修改或全部布局保留；旧full模式的纯CR/LF窄guard仍保留，近似字数不变成隐含精确CJK阈值。只实现了有限原题span/table observations sidecar，完整语义predicate合同仍未实现，不能将有限算式/substring观测称为membership已由程序验证。

## v20失败分析与恢复边界

本轮并未实现全面领先，也未提高已记录的描述性宏均值：Direct / Initial-MAS / Native JIT 分别为 0.5119798628 / 0.4685185185 / 0.5476313523。IFEval 三方均为 1.0；DRB 为 0.551724 / 0.500000 / 0.500000，WritingBench 为 8.8 / 8.3 / 8.7。IFBench、DSQA 与 RR 的 Initial-MAS 均有未评分槽，完整来源均值保持 null，不能用旧轮较高结果补入。Initial-MAS 实际生成 9/10 提交，完整评分 7/10；其余为一项生成失败、两项评分缺失。下面只定位公开工程证据，不归因私有分数或认证语义正确性。

首个 controller 在 RR 最后一项评价期间 exit=1；三份生成提交已封存，已有两项评价保留。日志没有可确认的异常类型，因此根本原因仍未知。运行器现有恢复流程跳过既有生成与评价，把已开始但无结果的评价标记 interrupted_evaluation_no_resample，再完成 DRB / WritingBench；恢复 controller exit=0。没有重采生成或中断评价。该 RR 槽已消耗的全部请求量未持久化，usage_unknown 保留，不能冒充 0；30 槽的已记录下界为 **191 calls / 2,069,992 tokens**，另有 1 个 unknown usage slot。完整独立预检仍另计 13 calls / 61,938 tokens。恢复审计 `.runtime/v20_controller_interruption_root.json`，SHA256 `d735fd45f57189a8e4aa7ebf64f2c039f170e93097f312c86cac07827288772d`。

IFBench 数值构建失败：JSON object 及 transported schema 的字段/类型均合法，15 个 numeric_values 均存在且为 canonical integers，但正文模板只使用了 12 个不同必需 marker，漏 3 个；after-model 跨字段校验因此失败。初稿没有有效最终构建凭据，不能直接保留。另一位置构建输出通过相同 v2 schema、renderer、hash 对齐以及有限通用 tokenizer 核验：25 句，目标 38 个槽分别形成一个非标点 token，句/词位置吻合；该槽官方 checker 的本轮实际分数为 1.0。这只说明该输出未重现旧 compound 分词偏差，不能单独归因其本轮 1.0、认证所有条件或宣称稳定提升。先前将本轮 IFBench 来源均值 null 错误解读成该槽 0 分的口径已纠正；null 来自另一槽生成失败。匿名工件 `.runtime/v20_ifbench_public_technical_diagnostic_baseline_runtime.json`，SHA256 `222ed8dfd455aacf15816c4f45b56764a45dd8ce03726a390ee2aad37b080683`。

八项 ordinary 修订有七项 patch applied、一项 applicability 失败并保留初稿；两项 typed 构建保留原有协议。Applied patch 共 20 个 edit 条目、其中四项空 edit 列表；条目数不等于实际修复次数。DSQA 数值集合项的一条 edit 原文与替换文相同，最终完全未改，公开 7 项的两个有限数值比较全部 PASS，但其他条件及全集仍 UNKNOWN。语义集合项采用 11 候选 × 6 条件表，24 YES、6 NO、33 UNKNOWN、3 AMBIGUOUS，明确提交“固定包下无可确认合格成员”；原材料自身声明历史和覆盖缺口，不能简单归因提示词，也不能强行把 UNKNOWN 转为 PASS。数值侧车的另三列 UNKNOWN 来自有限 header binder，不能直接解释为原始数据缺失。匿名结构/充分性工件 `.runtime/v20_public_dsqa_set_diagnosis_writing_quality.json`，SHA256 `43931450aed46f47d60522dafdcd289352b76ab8cad509d6b6cdddc69ade6096`。

WritingBench 两项初稿分别 2,460 / 857 字符，提交为 2,978 / 857 字符；第二项通过空 edits 保留完整初稿，本轮没有复现 v19 的 802→73 字符压缩。但本轮均值仍低于 Direct，不能把结构保留等同质量提升。DRB 的无效 patch 保留 3,516 字符初稿并记录组件失败，实际得分为 0.5；没有把失败修订当作成功修复。下一候选采用一次明确预算上限的 typed 本地校验反馈，以及完整观察审计关联的 compact 输入；它们尚需独立注册重跑，不能据软件测试宣称 benchmark 提升。
