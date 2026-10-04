# 2026-10-04 优化轮候选与预检记录

本文档记录下一轮开发优化的冻结候选、公开预检和正式协议边界。它不添加正式
VAL/TEST 结果，也不替换 v20 的完整 benchmark 版本。任何尚未有 sealed summary
的数值都保持为空；失败请求的实际计费记录保留，不能用估计 token 或
failure-zero 值冒充完成分数。

## 优化目标与候选顺序

当前源码候选（v26 current）优先处理两个已观察到的工程瓶颈：普通长文本 revision
使用严格 schema 时容易出现 schema 回显、过短标题或 JSON 截断；patch revision
在合法初稿上可能因为编辑定位失败而退回初稿。候选 C3 是一个组合候选，用于
验证完整的可运行路径，不把任一改变单独归因于分数提升：

| 项目 | C3 候选设置 |
|---|---|
| 普通 public revision | `public_revision_mode=full` |
| review / 普通 revision transport | `public_refinement_response_format=json_schema_review`；review 仍用 strict schema，普通 prose revision 使用 `json_object` |
| typed positional/numeric construction | `public_construction_response_format=json_schema`，保留本地 schema 与 renderer |
| revision repetition penalty | `public_revision_frequency_penalty=null` |
| planning / execution | 与当前 v26 源码候选一致，不增加 role、model-call cap 或质量重采样 |
| 公共资源 | 仍只使用公开 task 与允许的共享材料；不向 actor 传 evaluator、private rubric、reference answer 或 score |

C3 不能同时证明 full、transport 和 penalty 三项分别有效。因此后续必须按单变量
顺序继续：先固定 C3 的成功路径和失败成本，再只切换
`public_refinement_response_format`，然后只切换 `public_revision_mode`，最后才
比较 penalty；每个变体必须有新的 registration、源码/config hash 和完整 sealed
摘要。不得根据某一题的 judge 分数选择 revision 或重新采样失败槽。

这一设计借鉴公开方法的可检验部分而不外推其收益。Self-Refine 的基本流程是同一
模型生成初稿、反馈和修订，[论文摘要](https://arxiv.org/abs/2303.17651) 报告其在
若干任务上的平均改进，但这不保证当前模型或 benchmark 提升；其公开实现也明确
区分 Init、Feedback、Iterate 阶段（[Self-Refine repository](https://github.com/madaan/self-refine)）。
GEPA 将完整执行轨迹和错误作为可操作反馈，并以 Pareto 候选搜索 prompt 或程序
（[GEPA repository](https://github.com/gepa-ai/gepa)）；本项目只采用“记录失败原因、
逐变体比较”的思想，不把 private score 回流给 actor。关于内在自我纠错可能退化的
公开反例也需保留在解释中（[Huang et al., 2024](https://arxiv.org/abs/2310.01798)）。

## 公开服务预检

`.runtime/service_probe_20261004.json` 的两端合成预检各只有一次调用：actor usage
为 **116 tokens**，judge usage 为 **15 tokens**，两端均返回 stop。该预检只验证
transport、response model 和 usage 记录，不是 benchmark 分数，也不证明长文本
或 structured construction 成功。

候选 C/C2 的 key 截断认证失败已单独保留。其 token 是估计值，不是真实 provider
usage；失败 attempts 不得写成成功调用、完成答案或质量结果。C3 注册后使用新的
registration identity：

`69e0405db82ccca398eea7627e814ad104d63ac62e1cb2c2cbf171864010fb62`

注册目标是 source EVO 的 **2 个 ResearchRubrics、2 个 DeepSearchQA、2 个
WritingBench**，每题三臂共 **18 个 generation slots**，先只生成并提交，随后再以
独立步骤进行 GPT judge。当前 C3 的 18 个 generation slots 已全部消费并封存，但只有 1 个 direct 槽形成有效提交，其余为连接失败或无有效提交；judge 调用为 0，因而没有可比较的 score、均值、胜负或完成率。公开 progress 可保留
`run.log` 的 stage、method、task-id、status、calls、tokens 和 error type；不要复制
题干、答案、rubric 或 provider 原文。

## 与 v5 正式协议的差异

v5 protocol 文件仍声明 `SUBSET_PROTOCOL_FROZEN_NOT_RUN` / `PROTOCOL_FROZEN_NOT_RUN`：
它冻结了任务 split、资源记录和比较规则，但不是已经执行的正式结果（见
[`joint_protocol_v5.json`](joint_protocol_v5.json) 与
[`independent_protocol_v5.json`](independent_protocol_v5.json)）。v5 规划要求匹配的
方法资源、共同公开输入、证据材料、真实 token/失败率和封存后评分；成本、评分和
证据准备也要分开计量（见 [`experiment_plan_v5_ZH.md`](experiment_plan_v5_ZH.md)）。

C3 当前 registration 的 `knowledge_mode=closed_book`，且
`protocol_deviation_from_formal_shared_evidence=true`。因此它是 source EVO
development-only diagnostic：闭卷运行不能声称已经复现 v5 的 shared-evidence
正式条件，也不能把 C3 的同模型 self-judge 当成独立 evaluator。若要进入正式
shared-evidence 轨道，必须另建 versioned runtime amendment，固定公共 evidence
pack 的字节/hash、来源日期与 scope，并为三臂注册相同的 evidence 输入；不能在本轮
运行中临时把 evidence 注入某个方法或把旧闭卷结果拼入新结果。

共同资源仍应显式记录为每槽 2,000,000 task tokens、900 active seconds、生成与
评价共享；`max_parallel`、`max_inflight_requests`、模型输出上限、失败调用和
queue-idle 分开写入 ledger。C3 当前使用 `max_parallel=2`、
`max_inflight_requests=1`；这改变吞吐/排队条件，必须与 v20/v26 的 `2` 版本化
区分，不能宣称完全同资源。任何比较应先报告实际 calls、input/output tokens、
active seconds、unknown/estimated slots 和失败类型，再报告分数。

## 预先冻结的比较范围

1. **开发范围**：只使用 source EVO 选中的 2 RR、2 DSQA、2 WB；不读取或重选
   TEST 题，不把历史 exposed 题当作 clean formal TEST。
2. **方法范围**：Direct、当前 Ours、Native JIT 三臂；三臂共享 endpoint/model
   identity、预算、公开输入和 judge policy。Ours 的 public review/revision 只见
   public task、实际 draft、公开 contributor material 和允许的 knowledge policy。
3. **评分范围**：生成先完成并 sealed，之后独立 GPT judge；未完成 evaluation、
   provider connection error、submission failure 和 unknown usage 分别保存，不能
   以 failure-zero 代替观测分数。C3 当前没有最终 score。
4. **因果范围**：C3 只回答“该组合是否能稳定走完并留下可审计产物”。要回答
   full、schema transport、patch/full 或 penalty 哪一项有效，必须用新 registration
   做单变量变体；不得以局部题的成功或失败挑选配置。
5. **安全范围**：API key 只来自环境变量；配置和文档不保存 credential。metadata
   helper 只能读取 registration、seal、budget、hash 和匿名错误类型，不读取 private
   evaluator、reference answer、rubric body 或原始 actor response。

## 原始公开来源

代码与 benchmark adapter 以仓库中固定 revision 为准，不用运行时最新分支替换：

- [JIT-Agent source repository](https://github.com/bingreeky/JIT)
- [WritingBench source repository](https://github.com/X-PLUG/WritingBench)
- [IFBench source repository](https://github.com/allenai/IFBench)
- [GEPA source repository](https://github.com/gepa-ai/gepa)
- [Self-Refine source repository](https://github.com/madaan/self-refine)
- [Reflexion source repository](https://github.com/noahshinn/reflexion)
- [Process-level evaluation of Deep Research Agents (arXiv:2606.09748)](https://arxiv.org/abs/2606.09748)
- [Multi-Turn Evaluation of Deep Research Agents](https://github.com/sabharwalrishabh/Multi-Turn-Evaluation-of-DRAs)

这些链接只支持方法背景、代码与公开 benchmark 规范的可追溯性，不构成当前
provider、模型或本轮候选配置的质量保证。最终结果必须以本地 registration、sealed
summary、ledger 和 evaluator identity 为准。

## 当前状态

截至本记录写入时，C3 的全部 18 个 generation slots 已封存，但没有独立 judge summary，也没有可比较的最终 score。C/C2 的认证失败、C3 的 connection failure、估计 token 与真实 usage 的区别都保留为失败/成本记录；不补采、不重试、不填写最终分数。若 shared evidence 或资源设置要改变，先建立新的 versioned registration，再开始下一轮。
