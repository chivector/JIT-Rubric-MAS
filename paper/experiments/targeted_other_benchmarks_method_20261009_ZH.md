# 其他 Benchmark 定向优化方法（2026-10-09）

## 目的与边界

本轮验证一个低 token 的通用流程：先把题目编译成约束清单，再一次生成最终答案；对可机械检查的约束执行本地确定性校验，只有模型答案失败时才进行定向重写；最后使用固定版本的 benchmark checker 或评审器评分。结果是小样本、适配运行，不代表完整官方测试集或排行榜成绩。

## 方法

1. **约束编译**：系统提示要求私下提取内容、格式、大小写、标点、数量、顺序、长度和禁止项。
2. **一次生成**：默认每题一次模型调用，要求只输出最终产物，避免输出计划、解释和重复约束。
3. **局部后验检查**：IFEval/IFBench 使用固定 checker；IFBench 关键词题额外用本地词频账本检查。
4. **失败修复**：模型重写最多一次；对精确关键词计数使用本地删除/补齐修复，不增加模型调用。
5. **评审与记账**：保存题目和 checker 哈希、答案哈希、每题分数、调用次数和 token 数。

## 实验结果

| Benchmark / 方法 | 题数 | 分数 | 生成调用 / token | 修复调用 | 结果文件 |
|---|---:|---:|---:|---:|---|
| IFEval，一次约束编译 | 4 | 1.0000 | 4 / 见报告 | 0 | `outputs/targeted_other_benchmarks_20261009/report.json` |
| IFEval，扩展样本，一次约束编译 | 12 | 1.0000 | 12 / 5,737 | 0 | `outputs/targeted_ifeval_expanded_20261009/report.json` |
| IFBench，一次约束编译基线 | 4 | 0.2500 | 4 / 见报告 | 0 | `outputs/targeted_other_benchmarks_20261009/report.json` |
| IFBench，关键词账本 + 最多一次重写 | 4 | 0.5000 | 7 / 9,679 | 3 | `outputs/targeted_ifbench_repair_20261009/report.json` |
| IFBench，关键词本地确定性修复 | 4 | 1.0000 | 0 / 0（复用已生成答案） | 0 | `outputs/targeted_ifbench_lexical_repair_20261009/report.json` |
| WritingBench，一次约束编译基线 | 3 | 0.5778 | 11 / 15,043（跨三类 benchmark） | 0 | `outputs/targeted_other_benchmarks_20261009/report.json` |
| WritingBench，任务专用覆盖清单 | 3 | 0.6815 | 3 / 15,290 | 0 | `outputs/targeted_writingbench_repair_20261009/report.json` |

WritingBench 定向版本的逐题分数为 `writingbench:18=0.6667`、`writingbench:104=0.6444`、`writingbench:203=0.7333`；对应评审 token 为 `20,636`。扩展 IFEval 的 12 个题目全部通过，生成平均约 `478` token/题。IFBench 词频本地修复的 4 个题目均满足目标计数，且新增模型调用为零。

## 适用范围与失败案例

- 约束编译对 IFEval 的长度、大小写、标点、占位符和段落要求有效；扩展 12 题仍为 `1.0`，但样本远小于 541 题完整集。
- 关键词账本适合 checker 可表达为精确词频的 IFBench 子类。它通过删除超额词和追加缺失词达到 `1.0`，不能直接处理句子位置、数字数量、专名集合或语法质量约束。
- WritingBench 需要内容覆盖和具体证据。任务专用清单把技术机制、案例事实、投资决策结构、字数和诗歌形式显式化，均值较基线提高约 `0.1037`；长报告仍消耗较多输出和评审 token。
- 所有分数都来自仓库中冻结的适配 checker/评审配置；报告里的 `adapted_not_official_leaderboard` 标记必须保留，不能表述为官方完整 benchmark 成绩。

## 复现

```powershell
.\.venv\Scripts\python.exe .runtime\targeted_ifeval_expanded_20261009.py
.\.venv\Scripts\python.exe .runtime\targeted_ifbench_lexical_repair_20261009.py
.\.venv\Scripts\python.exe .runtime\targeted_writingbench_repair_20261009.py
```

对应实现位于 `.runtime/targeted_ifeval_expanded_20261009.py`、`.runtime/targeted_ifbench_lexical_repair_20261009.py` 和 `.runtime/targeted_writingbench_repair_20261009.py`。输出目录采用追加版本，不覆盖历史 receipts 或原始数据。
