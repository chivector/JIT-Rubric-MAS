# 实验结果归档：2026-10-09

本目录保存本机已有实验的状态、逐槽结果元数据、候选选择、checkpoint 摘要与评分完成情况。失败、中断、missing 和恢复版本分别保留；`full_report.json` 存在不表示全部评估成功。

- [完整 campaign 库存](campaign_catalog.json)：快照时间与各来源 EVO/VAL/TEST 状态，以这里的实际记录为准。
- [逐文件校验清单](compact_manifest.json)：本目录 compact JSON 的 SHA256。
- `campaigns/<目录别名>/summary.json`：注册身份、模型、批次选版、checkpoint、错误分类与已有 TEST 报告。
- `campaigns/<目录别名>/slots.json`：每个注册槽位的身份、终态、分数、用量摘要、提交哈希；不把 pending 当作失败或完成。
- `campaigns/<目录别名>/evaluations.json`：已有评分 receipt 的完成状态和工程错误类型。
- [剩余实验交接](../remaining_experiments_handoff_20261009_ZH.md)：四人分工、停止条件、验收要求及恢复边界。

原始结果压缩包与文件级清单上传至 [GitHub Release](https://github.com/chivector/JIT-Rubric-MAS/releases/tag/experiment-results-20261009)。发布资产索引保存在 `release_assets.json`，完整归档范围与排除项目见 Release 的 `archive_index.json`。归档包括 `outputs/`、`.r/`、`C:\J` 的实验结果目录及相关 `.runtime` 实验材料；已有 `paper/experiments` 历史结果继续保存在 Git 中。

SQLite 使用只读连接产生一致副本，包含已提交 WAL 内容，不直接复制运行中的主文件。文件级快照并非整个目录的同一事务；运行中的 r17 还会追加结果，最终报告及最终增量包将再次上传。原始 launch/config/split 和快照保留各自身份，不替换运行树的冻结文件。缓存、锁、WAL/SHM、依赖仓库、凭据文件与启动脚本不作为实验结果上传；出现凭据的文字副本会标记脱敏并记录原件与归档字节哈希。

`rr-runtime.bundle` 保留 r17 所依赖、尚未进入 `main` 的四个代码提交。它的前置提交是 `2f03166e464b5c8222ddb788125d5fce99945180`，终点是 `a7a0df262a6f9fd35a33b20782d58225f126737d`。实际运行使用的未提交 RR recovery split 随 `r17_runtime_metadata` 资产保存，不能用 bundle 中的旧 split 替代。

最新 RR recovery v4 已选出 C40，state hash 为 `a782b141a78516681fe6d1b0665a98bd7ea6a85e8a5e9efb073f138fc7a9bfed`。原 r17 v3 最终选版仍为 inconclusive。恢复版运行时上下文为 65,536、margin 2,048、`oldest_turns`，冻结 launch 声明 131,072；这是带工程差异的恢复结果，须连同 provenance 报告。

TEST 全部终态并封存后才评分。分母使用注册槽位数，缺失评分时完整均值保持 `null`；完整样本的描述性均值不能替代完整均值。不同 benchmark 的原生指标分别报告，历史和恢复版本不拼成一条新轨迹。
