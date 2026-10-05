# v5 联合实验中断审计（2026-10-05）

本记录只读核对下列运行目录的 registration、journal、receipt 与 budget；未调用 API，未修改原始实验产物。两次尝试均未完成注册的三条轨迹及 TEST，属于中断与实现诊断，不能作为论文最终成绩或正式 superiority 证据。

## 冻结身份

| 项目 | p20：`outputs/formal_v5_evo_val_20261005` | p12：`outputs/formal_v5_evo_val_20261005_p12_current` |
|---|---|---|
| Bundle SHA256 | `5b59d73ab6837936bcbfa2b4d412bd357a927ba96aa888eb47d4f73faa088810` | `57203efe3d0400badd221bb951122955c783496a0ed1c2d2fabff0ef8a57647a` |
| Config SHA256 | `8e195bf48cc8ad626d1c75dcef50a907ae9ef9f048fda172a0def7518e21557f` | `e1e6a2b73901263f12c18159100b0aa14f7b9ec0ce3bdac99fb6875f5805106a` |
| Code fingerprint | `5ba5142a64f5d6046972841c70df3b764a497d94f2b64efde454c8c940721c03` | `b88bc5b88221dd343e5cca9d113cedca8688b3ac7938349d5bbcda0a64601606` |
| Runner SHA256 | `985fcafb653412701255f6cfeff767f1442ab16fbb052508029198b12fa4b95b` | `985fcafb653412701255f6cfeff767f1442ab16fbb052508029198b12fa4b95b` |
| VAL workers / maximum inflight requests | `8 / 16` | `2 / 2` |

两次尝试均为 `deepseek-v4-flash-vision` 生成、`gpt-5.6-sol` 评分，每题 token 上限为 2,000,000，judge 输出上限为 1,024。协议和题目成员沿用 v5 冻结清单；不同代码或配置身份的结果不可拼接成同一次完整实验。

## 槽位与失败

| 尝试 / 阶段 | complete | failed | started | pending |
|---|---:|---:|---:|---:|
| p20 EVO | 5 | 7 | 1 | 167 |
| p20 VAL | 19 | 11 | 0 | 420 |
| p12 EVO | 0 | 0 | 0 | 180 |
| p12 VAL | 0 | 0 | 2 | 448 |

两个 journal 均仍标记 `running`，只保存 `run0:c0`，尚无选中版本。p20 的 C0 完整评分为 RR `2/10`、DSQA `7/10`、WritingBench `10/10`，前两项不满足注册的 `9/10` checkpoint 资格。

p20 的 18 个失败槽位包括：8 个 `InterruptedWithoutDurableOutcome`、7 个 `ValueError`（4 个进化提交、1 个 Pool 结构验证、2 个 reconciliation 验证）、1 个 `RuntimeError`（WritingBench 执行桥接）、1 个 `JSONDecodeError`（DSQA 公开反思解析）、1 个 `APITimeoutError`（DSQA 规划请求）。Provider 失败已确认一次超时，其消耗为估算；其他失败不能一概归为 API 不可用。

p20 未完成的 EVO 是 run 0 ordinal 12、task `6847465956a0f6376a6053aa`。p12 两个已开始 VAL 是 RR task `6847465956a0f6376a605391` 与 `6847465956a0f6376a60535d`，仅有 `run_manifest.json`，没有 durable budget、complete 或 failure receipt。

## 已知成本与遗漏

p20 journal 仅报告 **431 calls / 7,157,154 tokens**。四个 EVO 在生成、评分、归因完成并保存 `complete.json` 后，`commit_evolution` 失败；该异常未携带预算，journal 将其记为 unknown 且数值为零：

| run 0 EVO ordinal | calls | tokens |
|---|---:|---:|
| 1（DSQA） | 19 | 711,075 |
| 6（RR） | 36 | 934,212 |
| 7（DSQA） | 19 | 254,881 |
| 8（WritingBench） | 18 | 607,222 |
| **journal 漏计合计** | **92** | **2,507,390** |

p20 磁盘 34 份 `budget.json` 去重后的已记录合计为 **523 calls / 9,664,544 tokens**：其中 provider 实际回报 **522 calls / 9,589,658 tokens**，一次超时估算 **1 call / 74,886 tokens**。实际回报部分是可证实的消耗下界；含估算的总数不能称为精确账单。另有中断无预算的未知消耗，仍需单列 unknown，不能按零处理。

p12 journal 的 `0 calls / 0 tokens` 只表示尚无已完成预算记录；两个 started 请求的服务端消耗未知，不能宣称未产生调用或零费用。上述两次尝试的成本应在总实验开销中保留，并与新尝试分开报告。

## 恢复与后续边界

重复 proposal ID 导致的跨任务提交冲突根因已修复；提交异常预算保留补丁在本次审计时仍待 commit。初次只读核对时当前代码与 p12 fingerprint 一致；随后预算补丁改变 fingerprint，当前值为 `4038d2b82e7857b8b550544dee4f4f360ed5e180a01c18417be04f53fd88ce3b`，因此当前工作树已不满足 p12 的冻结身份。不得改 registration 或 journal 哈希来绕过检查。

使用 p12 原冻结代码与配置恢复时，已 started 的两槽应消耗为中断失败并保留 unknown 成本，不能再次生成；pending 槽才可继续。p20 除代码身份不匹配外还存在 started EVO，现有 `_assert_live_prefix` 会拒绝未解决的 started 进化位置。使用修复后实现应另建完整身份绑定的新尝试，保留两次旧尝试全部产物。

新尝试（如 p14）监控可只读汇总 journal，不读取答案或裁判正文：

```powershell
$journalPath = 'outputs/formal_v5_evo_val_20261005_p14/evo_val_journal.json'
$journal = Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
$slots = $journal.slots.PSObject.Properties.Value
$slots | Group-Object kind,status | Select-Object Name,Count
$journal.token_usage | Select-Object model_calls,actual_model_calls,estimated_model_calls,total_tokens,estimated_tokens,usage_unknown
$slots | Where-Object { $_.status -eq 'started' } | Select-Object slot_id,started_at,attempts
$slots | Where-Object { $_.status -eq 'failed' } | ForEach-Object { [pscustomobject]@{ slot_id = $_.slot_id; error_type = $_.result.error_type; tokens = $_.result.token_usage.total_tokens; unknown = $_.result.token_usage.usage_unknown } }
```

监控路径须替换为实际新尝试目录；状态汇总不能替代磁盘 receipt 的成本审计，也不能触发 started 槽位重采样。
