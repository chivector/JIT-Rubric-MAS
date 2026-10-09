# 低 token 的约束生成入口

`scripts/run_compact_benchmark_subset.py` 为 IFEval 和 IFBench 提供每题一次模型调用的入口。模型直接读取公开题目和约束，省去规划、组队和全篇审查；支持的词频规则会编译为紧凑计数清单。精确词频、数字数量与句内位置约束可选择严格 JSON schema，再由 renderer 输出正文。

## 运行

设置 `MAS_EXEC_MODEL`、`MAS_EXEC_ENDPOINT` 和 `MAS_EXEC_API_KEY`，然后使用固定版本的 checker 源码目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_compact_benchmark_subset.py `
  --config configs/compact_generation.example.yaml `
  --benchmark ifbench `
  --dataset dataset/ifbench/IFBench_test.jsonl `
  --checker-source outputs/independent_v5_preparation/checkers/IFBench-1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d `
  --task-id ifbench:0 --task-id ifbench:17 --task-id ifbench:10 `
  --literal-construction --numeric-construction --numeric-layout template --positional-construction `
  --output outputs/compact_ifbench_example
```

`--task-id` 可重复；`--count N` 取数据加载顺序中的前 N 题。IFEval 使用 `--benchmark ifeval`、对应数据文件和固定 IFEval checker 目录。输出目录必须尚不存在。

本轮最终版本在 IFBench 17 道迭代题通过 15 道，使用 13,691 token；另 12 道新题通过 8 道，使用 7,055 token。IFEval 12 道新题通过 11 道，使用 4,178 token。逐版本结果和失败项见 `paper/experiments/compact_optimization_20261009_ZH.md`。

三个 construction 开关可以同时启用。编译器根据每道题选择一种 schema；未支持或有冲突的规则在调用前退回普通生成。数字构造默认使用 `array`，`--numeric-layout template` 可以保留全文自然布局，以字母占位符承载数字。精确词频构造把模型创作的文本间隙与固定次数的关键词交织；间隙不能另含目标词，重叠或冲突的目标词会拒绝编译。严格 schema 不保证语义内容完整，普通生成也可能违反词频；实际结果以调用之后的固定 checker 为准。

## 结果与成本

每次运行保存 `manifest.json`、`report.json`、逐题记录和答案。记录绑定配置、数据、checker、代码及实际消息 hash，并保存 generation token、调用次数、结束原因和完整率。截断回答保留原文且标为 incomplete；评测异常保留已成功生成的答案。报告同时提供成功评分均值和将未完成题计为零的均值，避免只看成功题。

该入口关闭 SDK 与 provider 重试，每题最多一次调用，不自动修复答案。修改提示或参数后应选择新输出目录。

完整 JIT-MAS 管线仍可使用 `public_refinement: true`。另加 `public_skip_empty_revision: true` 后，通过有效空审查、原稿可保留且没有已编译词频失败的普通文本，会直接保留原稿，节省一次 revision。数字/位置构造和投影稿仍走原有构造流程。空审查只表示审查没有提出问题。
