# Live quality check: 2026-09-30

Historical report: the results below describe the recorded 2026-09-30 implementation,
not the later direct-update method. Historical acceptance terminology does not
specify the current online update contract.

## Conclusion

**These results do not establish SOTA or a general quality improvement.** The
initial frozen implementation comparison was incomplete. After narrow protocol
fixes, both exposed tasks completed in a separately registered development run.
Successful execution is not the same as improved answer quality.

| ResearchRubrics task | Baseline `a1ee6cd` | Initial candidate | Development candidate |
| --- | --- | --- | --- |
| Neural-network learning, `6847465956a0f6376a605414` | Planning failed | Execution failed; repair connection failed | 54/85 = **63.53%**, 23 criteria |
| LiDAR/camera comparison, `6847465956a0f6376a605434` | 51/120 = **42.50%**, 33 criteria | Local planning failed | 50/120 = **41.67%**, 33 criteria |

Failures are retained as failures, not zero scores or excluded observations.
The original paired mean difference is **null**. The development column is not
blind, paired, held-out, or a SOTA estimate. Its LiDAR score does not demonstrate
an improvement over the baseline; the neural-network task has no scored baseline.
All scores above are the original unmodified evaluator outputs, including the
judge inconsistencies described below.

## Protocol and provenance

- Public task selection was frozen at `2026-09-30T12:02:17.626810+00:00`, before
  inspecting rubric counts. Each implementation had one attempt at each task,
  with bounded native repair rather than task resampling or best-of selection.
- Dataset: `ScaleAI/researchrubrics`, revision
  `85de3115053d1453ed612caacf4a405edc1ad756`; local SHA-256
  `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`.
- All roles and the separately invoked judge used the supplied endpoint's
  `deepseek-v4-flash-vision` model identifier, temperature 0, non-thinking mode,
  and zero transport retries. The endpoint listed only that model. Its identity
  behind the service is not independently verified; the judge is not a different
  model family. Output caps were 16000 for planning/generation, 8192 for execution,
  and 4096 for each rubric judgment.
- Both arms used the same closed-book configuration: no search or other task
  tools, at most three dynamically assigned roles, six team calls, native JIT
  generation, and an empty read-only experience snapshot at version 0.
- Generated source was read in full and exact-hash approved before host execution.
  `--unsafe-local` was explicitly authorized; review is not a security sandbox.
- The originally proposed 70-call/500000-token limits were amended before the
  first calls, after the user's authorization, to 120 calls/1000000 ledger tokens
  per arm. Both initial arms used identical limits; no limit was auto-expanded.
- Official ResearchRubrics prompt text and signed-weight scoring were unchanged.
  Private criteria were available to evaluation, not execution-role prompts.
  `quality_audit_mode` remained `off`; the review below was post-run inspection.

The initial run began at `2026-09-30T12:36:33Z`. The development run began at
`2026-09-30T12:50:29.953207+00:00` and finished at
`2026-09-30T12:58:35.696593+00:00`. Development registration explicitly records
the prior exposure and anchors the original summary SHA-256:
`8c07e939313e1b92196076dcda8421d2e486d0dae1ee7040cf4c90ba8c36594c`.

| Frozen runtime | Fingerprint |
| --- | --- |
| Baseline | `4f7187ce05f1f06e76315f6ff38d6c6bb2c560e9086628e98c24361217ccbbde` |
| Initial candidate | `50cedb24af297faa0e0b24050924b9268a438bb0933d54f74beb3144e03ffbcf` |
| Development candidate | `4474f8989df4eb5c475d3c039694b12428e2cc21b6fc6a35b0ae237a62f62bec` |

Each runtime and experience store remained unchanged during its run. This was
`evaluate` mode, not live attribution, paired experience acceptance, or cross-task
reuse of an accepted update. No full benchmark or training was run. Offline fixture
coverage of the full loop must not be presented as real experience-learning success.

## Observed failures and fixes

The baseline neural-network plan requested unavailable tools. The initial
candidate's neural-network writer supplied a reasoned checkpoint with status
`completed`, which the executor rejected. Its subsequent native repair call had
an `APIConnectionError`. The initial LiDAR local planner copied a role name into
the distinct capability field and repeated the mismatch after generic feedback.

The follow-up kept the overall JIT-MAS design and strict resource policies:

- Preserve `completed` as a **neutral self-report**, never convert it to `passed`.
  Reasons and observed evidence references remain mandatory and validated.
- Give local planning the exact immutable agent ID/capability and expected/actual
  mismatch diagnostics. No silent identity rewriting or unavailable-tool grants.
- Align the actual native seed description and bridge contract with worker
  completion envelopes, honest checkpoint statuses and zero-tool operation.
- Keep all published DAG ancestor artifacts available to synthesis while keeping
  private role histories separate; this earlier candidate change remains active.
- Isolate exposed-task development runs in a separate protocol. The comparison
  summarizer rejects development evidence rather than treating it as fresh A/B data.

The development run generated and approved two native packages and needed no
additional JIT repair package. It completed all 56 rubric judgments. This supports
the protocol fixes on these tasks, not general reliability or quality superiority.

## Answer and judge audit

An independent review read each final answer before inspecting its evaluation.
No answer, verdict, weight or score was changed after this inspection.

- **Neural-network answer:** forward values, gradients and the numerical update
  are consistent. However, two ASCII diagrams depict an inverted V while claiming
  a U-shaped MSE curve. The explanation overgeneralizes a one-step fit that depends
  on the chosen learning rate. Preprocessing, train/validation/test separation and
  overfitting are missing, weakening the distinction between fitting and learning.
- **LiDAR answer:** latency, energy and computational scale lack meaningful
  engineering analysis. It lists architectures but provides no controlled,
  same-task empirical comparison or reproducible experiment protocol. These are
  substantive gaps, not merely formatting preferences.
- **Judge consistency:** development neural-network criterion `r16` and LiDAR
  criteria `r28`/`r29` describe a penalized omission as present but return
  `NotSatisfied` with no negative-weight deduction. The LiDAR `r32` reason claims
  Tesla is absent although the answer explicitly discusses Tesla and camera-only
  cost. The neural-network acronym judgment also grants credit while admitting an
  unexpanded acronym. These contradictions make the scores less trustworthy.
- Exact quotation checks: baseline LiDAR 91/117, development neural-network 58/89,
  development LiDAR 66/90. Non-exact quotations are not automatically fabricated;
  ellipses, formatting and paraphrases can also cause mismatches.

## Why this is not a SOTA comparison

The ResearchRubrics paper reports full-benchmark product evaluations on 101 tasks.
Its Table 5 gives binary scores of 0.615 for Gemini Deep Research, 0.597 for OpenAI
Deep Research and 0.487 for Perplexity Deep Research. These are published paper
results, not a verified current leaderboard. Our two exposed closed-book tasks and
same-model judge are not comparable to those product evaluations. A single task
scoring above a published aggregate does not establish superiority.
[Official paper and Table 5](https://arxiv.org/html/2511.07685v1#S4.T5).

The official evaluator's model configuration, long-document handling and retry
settings also differ from this bounded test. Comparable evaluation would require
matching the task set, information access and judge protocol, plus repeated runs
and uncertainty estimates rather than selecting successful examples.
[Pinned official evaluator](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/evaluate_rubrics/evaluate_single_report.py).

Meta-Team reports 55.8/100 with 20 evolution tasks and 81 held-out tasks, repeated
evaluation and a different backbone/tool configuration. That number is likewise
not a threshold this two-task run can claim to beat.
[Meta-Team paper](https://arxiv.org/html/2605.29790v1).

## Verification and local evidence

Final offline verification: **451 tests and 27 subtests passed** via
`python -m pytest tests/jit_mas -q`; `compileall` and `git diff --check` passed.
Final artifact audit verified all 60 sealed evidence-file hashes and the three
completed answers' submission/evaluation/runtime/configuration bindings. No
outstanding ledger reservation or running API worker remained.

| Run | Model attempts | Ledger tokens |
| --- | ---: | ---: |
| Baseline | 46 | 168168 |
| Initial candidate | 12 | 159495 |
| Development candidate | 74 | 350086 |
| Total | **132** | **677749** |

There were 89 completed rubric judgments. One failed repair attempt accounts for
83083 conservatively estimated tokens; the remaining 594666 have adapter counters,
which are not independently verified provider billing. Model attempts are ledger
reservations, not an instrumented physical HTTP count. Two additional authenticated
model-list requests are metadata calls, not model attempts. Credentials were passed
through child-process stdin, not committed or included in reports.

Evidence remains in local ignored output directories, not necessarily on GitHub:

- [Original registration](../outputs/quality_comparison_20260930/pre_registration.json)
  and [unchanged comparison summary](../outputs/quality_comparison_20260930/comparison_summary.json).
- [Development registration](../outputs/quality_development_20260930/pre_registration.json)
  and [complete development report](../outputs/quality_development_20260930/candidate/report.json).
- [Final integrity audit](../outputs/quality_final_audit_20260930.json).
- [Development neural-network answer](../outputs/quality_development_20260930/candidate/evaluate/2d72e922d583fad739a691ab1e3edff79592bccf1c3ab5ee958465a03bfdf84f/submission.json).
- [Development LiDAR answer](../outputs/quality_development_20260930/candidate/evaluate/1cda0cfcb14825d8b3a799c82da21e4ff970b4cc49e10f1b9a066077cad1ebfa/submission.json).

The next quality study needs genuinely unexposed tasks, a reliable independent
judge, and an evidence-access protocol appropriate for research reports. Increasing
API volume alone does not resolve the observed content or evaluation problems.
