> 历史编辑记录：以下 v3 说明对应旧的成对验证与 accept/hold/reject 方法。2026-10-01 起，当前方法已改为归因后直接版本化更新；请以当前 `main.tex`、`appendix.tex` 与 `overview.tex` 为准，旧记录不作为现行协议。

# JIT-Compose v3：全文改稿说明

## 正式名称与标题

方法名：**JIT-Compose**。展开为 **Just-in-Time Composition of Agent Teams and Workflows**。

标题：**JIT-Compose: Evolving Multi-Agent Workflow Synthesis for Open-Ended Generation**。

本次将上一轮批准的名称、标题和摘要落实到了完整论文，而非仅提供替换片段。论文改名不要求把实现中的 `jit_mas/` 或 `scripts.run_jit_mas` 重命名。

## 统一后的研究主线

**为当前问题生成写作过程，为未来问题保留经过验证的生成经验。**

这里的 writing workflow 是可执行的研究与写作组织：参与角色、职责、信息依赖、工具调用、交接、核验与最终综合。它不是仅调整文风、输出大纲或文章模板，也不新增此前没有描述的风格生成器。

本次区分了两个时间尺度：每个任务都根据公开问题及当前经验生成团队与工作流；跨任务更新的是 rubric、组织和局部执行经验，而不是令所有任务套用上一次成功的工作流。模型参数与 evaluator 保持冻结。

## 实质性改写

**摘要。** 使用已经确认的新版摘要，开头直接提出“写什么”和“怎样协作产出”的区别，中段明确现场合成的对象，结尾落到跨任务保留生成经验。没有填入猜测的性能数字。

**引言。** 以写作过程的组织决策开场，保留“两个方法的局部总结都正确，但最终比较缺乏可比性”的解释性例子。随后用逐题合成与跨题经验进化串联全局—局部协商、执行、逐项归因和独立验证。这个例子仍标注为说明，不作为实测案例。

**问题定义。** 用 `ω_t = (T_t, h_t)` 表示某题的团队说明及其可执行 harness；用 `S_t` 表示跨题的经验与版本状态。这只是对已有 TeamSpec 与五文件 harness 的合并记号，不是增加图学习模块、优化器或训练程序。类似任务允许生成相似结构，不能用代码 hash 的不同证明适应性。

**方法。** 五个阶段统一为：预测任务质量规范；共同设计团队与协作；现场合成并执行写作工作流；把反馈追溯到设计与执行；进化未来合成所用的经验。保留原有的不确定性、评分隔离、负权重聚合、拒绝或待验证更新等边界。

**框架图。** 上半部分对应单任务即时合成，下半部分对应跨任务经验更新。明确每题重新生成的是角色、协作协议与可执行工作流，虚线回路承载的是已接受经验。没有加入未经验证的收益曲线或伪造生成案例。

**实验与结论。** 保留相同 rubric、不同构建可见性的 G/GO 控制，且 rubric-blind 必须覆盖 JIT 的完整生成/选择/修复路径，不只是 planner。跨题测试需要在初始/进化状态下分别生成工作流，不能固定原团队只重写答案。总结以 workflow synthesis 为对象，但不把生成成功、记录了责任或经验入库等同于质量提升。

**附录。** 补充 workflow synthesis 的明确含义与审计边界，统一术语、交叉引用和元数据；原有诊断、统计与复现细节保留。代码包路径保持原名，以避免把编辑改名误当作代码已改。

## 继承关系与证据边界

即时生成 executable harness、五文件协议和运行异常修复继续明确归于已有 JIT 底座。本项目的叙述集中于质量要求怎样条件化协作结构，以及逐条反馈怎样修改下一次生成的经验。不会因改名把底座贡献重新声明为创新。

本次修改依据已提供的 v2 修正版源码、v3 标题摘要说明和对话中的方法设计。没有新增外部文献调研，没有改写既有 BibTeX 文献数据，也没有运行模型或 benchmark。源码实现、实际配置及逐任务结果仍未随本轮材料提供，故结果表继续明确留空；这不等于断言作者没有运行实验。

论文中的 working-draft 提示应在真实结果、实现对齐和证据核查完成之后再处理，不能只删除提示便视作实证终稿。未经核查的模型型号、任务规模划分、收益或统计显著性均未补写。

## 版面与编译

由已修复的 Overleaf v2 工程增量修改，主文档和直接依赖保持位于 ZIP 根目录。使用原 ACL 样式和 BibTeX 样式，未缩小页边距、字体或使用负间距挤正文。

最终页数、命令、参考文献条数、编译警告检查以及干净目录验证见 `revision_notes/build_validation.json`。编译在本地 TeX Live 完成，不声称登录或测试了用户的 Overleaf 项目。

`revision_notes/changes_from_v2_fixed.patch` 记录相对上一份修正版的正文、附录和框架图变更；历史 diff 中出现旧方法名不表示正文残留旧名。`author_notes/` 是作者内部材料，不参与论文编译，不宜原样随匿名审稿附件提交。

## AI 使用范围草稿

We used ChatGPT to assist with literature organization, manuscript drafting, and editing, and Codex to assist with implementation and software testing. Human authors are responsible for checking the implementation, references, experimental design, and all reported results. AI tools are not listed as authors.

以上披露沿用前稿的范围说明，需作者据实际使用及最终核查情况确认；本轮不声称已经完成未提供的人工审计。
