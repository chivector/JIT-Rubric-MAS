# 2026-10-09 TEST 评分工程修复与固定两题验证

原 `C:\J\r15` 的 142 个已提交 TEST 与 `C:\J\r17_rr_recovery_v4` 的 87 个已提交 TEST 均在评分入口记录 `evaluation_failed / AttributeError`。精确离线复现表明：`scripts/run_independent_test_release.py` 的 `_score_one` 将 SQLite 保存的字符串 `submission_path` 传给 `_read`，而 `_read` 直接调用 `.read_text()`，抛出 `AttributeError: 'str' object has no attribute 'read_text'`。异常发生在读取原答案、构造 pipeline 和 judge 之前。修复统一使用 `Path(path).read_text()`；这解释评分工程失败，不能解释生成质量或证明整库存质量提高。

固定采用原登记顺序中的一个 WritingBench 本源已提交答案和一个最新 RR 本源已提交答案，不按分数选题，不重生成答案。通过原模型配置、`NativeModels` 和对应 benchmark evaluator 仅调用 judge，逐项核对原封存、slot/result/submission/answer/state hash 与生成预算。两次运行的原 submission 和原失败评分 receipt 的字节 SHA256 在前后完全一致。

| 来源与题目 | 完整评分 | 原生结果 | Judge 调用 | Judge tokens | Actor 调用 |
| --- | --- | --- | ---: | ---: | ---: |
| r15 WritingBench `writingbench:569` | 是 | native mean 8.0；归一化 0.7777777778 | 1 | 6,063 | 0 |
| r17 recovery RR `6847465956a0f6376a605407` | 是 | 0.5862068966 | 25 | 84,586 | 0 |

共 26 次 judge 请求、90,649 tokens；没有 actor 请求、状态进化或原回执覆盖。RR 沿用本次 recovery 已公开登记的实际 64K 上下文与 2,048 margin，冻结启动身份为 128K 的差异保留；WritingBench 保持原注册配置。这里的两个分数只验证评分通路和固定答案的诊断结果，不能作为完整 TEST 平均数或泛化改善。

原验证产物在 `outputs/repair_subset_20261009/scoring_verification`，包含 summary、每题 identity、started 和 evaluation。原历史 `usage_unknown=true` 的评分失败账本保持原样；零 judge 调用是结合已证明异常位置作出的工程诊断，不把历史未知账重新宣布为官方完整账。

可复用恢复入口为 `python -m scripts.rescore_saved_test_subset --campaign <原目录> --output <新目录> --target <benchmark> --count <固定前缀题数>`。针对已证实路径错误需显式 `--recover-prejudge-path-error`，使用条件和单次请求记录见 `docs/saved_test_rescoring.md`。后续需按固定登记槽位做独立版本化评分恢复，保留原 142 / 87 个失败回执、生成失败、成本及上下文差异，不重新抽题或根据 TEST 反馈调参。
