# JIT-Compose v3 — Overleaf 工程

**JIT-Compose: Evolving Multi-Agent Workflow Synthesis for Open-Ended Generation**

本包是完整论文源码，不是标题或摘要的替换片段。由已修复的 v2 Overleaf 工程修改；正文、摘要、图注、算法、附录及 PDF 元数据已统一。实际实验结果仍保留明确的待填状态。

## 导入与编译

直接将 ZIP 作为新项目上传。主文档设为根目录 `main.tex`，编译器设为 `pdfLaTeX`。使用项目支持的完整 TeX Live 环境；本包不要求 Python、外部字体、shell-escape 或预生成的 `.bbl`。

不要将它与旧项目的 `paper/` 子目录混合。`appendix.tex`、`overview.tex` 不是独立主文档。已在原项目替换文件且仍有旧错误时，可清理编译缓存后重新构建。

```text
main.tex                         唯一主文档
appendix.tex                     补充材料
overview.tex                     可编辑 TikZ 框架图
references.bib                   参考文献数据库
acl.sty                         原 ACL 论文样式
acl_natbib.bst                   原 ACL BibTeX 样式
compile.sh / compile.ps1         可选本地构建脚本
author_notes/                   编辑说明、摘要、空结果模板（不编译）
revision_notes/                 修改 diff 与构建验证（不编译）
MANIFEST_SHA256.json             文件完整性清单
```

参考文献使用正常的 BibTeX 流程：修改 `references.bib` 后重新编译。`acl.sty` 已选择 `acl_natbib` 样式，不需在主文档重复设置。

## 本地命令

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

没有 latexmk 时：

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

也可运行 `bash compile.sh` 或 PowerShell 下的 `./compile.ps1`。

## 稿件与实现命名

论文方法名为 **JIT-Compose**。实现路径 `jit_mas/` 与 `scripts.run_jit_mas` 保持已提供设计中的名称；当前文稿与本地代码的直接更新方法同步。writing workflow 指可执行的团队与协作过程，不是新增的文风/文体生成器。

## 验证与证据

### 2026-10-01 方法修订与实验状态

当前方法已移除经验更新的 accept / hold / reject 质量筛选：任务提交与逐条评价后，全局—局部归因给出首个提案，经结构、来源、允许目标、基础版本和幂等检查后直接原子写入版本化经验。保留来源证据、适用范围、不确定性、提交回执、回滚和冻结测试只读。移除筛选不保证经验更好，正式效果仍须独立测量。

当前主流程为：预测要求与协同规划 → 原生 JIT 生成 → single-pass shared ledger 执行 → 提交 → 独立评价 → 协同归因 → 直接经验更新。Figure 1、算法、公式、成本描述和结果模板已同步；不再包含在线成对验证调用。

`paper/experiments/protocol_v1.json` 和 `splits_v1.json`（在独立论文目录中为 `experiments/`）保留为**未运行的旧方案审计记录**。18/30/20/33 分配、验证轮换和接纳阈值均已暂停，标记为 `OBSOLETE_ON_HOLD_METHOD_CHANGE`，并保持 `DRAFT_NOT_RUN`、`launch_allowed=false`。本次没有采用新的题目分配，也没有启动正式实验。可选 validation 集只供外部分析，不再决定是否入库。

此前四题 baseline pilot 仍全部属于 development；已有暴露记录不能因方法修改而清除。旧 smoke 与历史 API 结果仍只证明各自记录的代码版本，不能填入新方法论文主表。正式运行前必须重新固定任务来源与划分、进化顺序、重复次数、检索快照、baseline 对齐、模型版本、隔离和独立评价协议。

`revision_notes/changes_from_v2_fixed.patch`、`author_notes/Revision_v3_ZH.md` 和旧 `build_validation.json` 是历史记录，其成对验证/接纳措辞不是现行方法。当前修改尚未重新编译或进行 PDF 排版检查；不能引用旧构建记录声称本次已编译通过。

最终构建记录见 `revision_notes/build_validation.json`。验证在本地 TeX Live 的干净目录完成，并另外使用 `output` jobname 检查引用流程；这不等于已登录用户的 Overleaf 项目。Underfull hbox 是排版提示，需与致命错误区分。

源码中没有编造实验数据。软件测试通过、代码生成成功和历史 LaTeX 编译通过，均不能当作论文方法的性能证据。`author_notes/` 为作者内部编辑材料，不参与编译，不应未经筛选作为匿名投稿附件。

## 原样保留的样式文件

原包中的 `acl.sty` 和 `acl_natbib.bst` 均未修改，保留其许可声明。Git blob SHA-1：

- `acl.sty`: `d9b74d0e6d1ea7a41929ff30111a112e1d23f959`
- `acl_natbib.bst`: `086cb0bc745cf1120aef9c99aa295ccdba3e736c`

本轮未重新抓取或更换模板版本。
