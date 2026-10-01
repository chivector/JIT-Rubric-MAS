# Quality-focused implementation update

This document records the quality iteration before the single-pass executor
migration. Its experiment results are historical, not measurements of the current
executor. See [single-pass task execution](jit_mas_single_pass.md) for the current
call, ledger, failure and tool contracts; earlier references to in-role correction
budgets no longer authorize additional task-execution calls.

This is an incremental update to the existing JIT-MAS loop, not a replacement MAS
framework or a fixed team. The reference implementation is commit `a1ee6cd`.
The native five-file JIT generation, selection, loading and repair path remains intact.

## Problems addressed

- A synthesizer could receive only its direct reviewer's short response, losing the
  original draft. Each role now receives the published artifacts of all its DAG
  ancestors, deduplicated, with recorded handoffs and communication accounting.
  Other agents' private conversations remain inaccessible.
- A checkpoint previously needed `true` to finish, encouraging unsupported success
  claims. A structured `failed`, `unverified` or `not_applicable` report with a reason
  can now complete a local assignment honestly. Reports are explicitly self-reported,
  never independently verified, and travel with the artifact. Missing checks and
  unexplained `false` values still require correction within the allocated budget.
  The later `completed` status is also a neutral completion self-report, not `passed`
  and not evidence of independent verification.
- Completion evidence must have appeared in that role's observed context. Merely
  guessing a valid global event ID is insufficient. Tool results become citable only
  after observation. Completion references and final-submission permissions are
  checked before tool side effects.
- Planning and execution prompts treat predicted rubrics and generated hints as
  fallible hypotheses. Technical reviews must inspect assumptions and inferences;
  synthesis must inspect original contributions and resolve material disagreements.
  These instructions do not prescribe subject-specific answers or force mathematics
  onto unrelated tasks.
- Local planning checks exact tool and dependency IDs, reporting all detected
  violations to the existing bounded correction path. Prompts make actual role call
  allocations explicit and disallow reliance on future feedback in a single DAG pass.
  Role IDs and capability text are immutable local-plan identities; correction prompts
  provide their exact original values rather than allowing paraphrased identities.
- The actual `rubric_mas` seed prompt contract now matches the executor's supported
  completion protocol. This changes generation guidance, not generated answers or
  frozen artifacts, and retains the native JIT generation and repair path.
- Experience retrieval now considers public task operations and applicable role
  functions. Generic writing vocabulary is insufficient to transfer comparison advice
  to a non-comparison task. Selection diagnostics are recorded for execution roles.
- Attribution preserves judge reasons and optional risk diagnostics, distinguishes
  self-reports from observed checks, and challenges source-specific private content
  thresholds before turning them into transferable advice.

## Evaluation remains independent

The official ResearchRubrics prompt, signed-weight calculation and verdicts are unchanged. The adapter's optional `quality_audit_mode="risk_only"`
annotates negative-weight rows and unverified quotations without changing scores,
calling another model, or deciding semantic consistency. Its default is `off`.
Risk flags are not proven evaluator errors and do not constitute quality certification.

An explained failed checkpoint means a role reported its limitation, not that its
answer is good. A `completed` checkpoint likewise does not certify correctness. Only
independent evaluation and separately reported downstream comparisons provide quality
evidence. Since 2026-10-01, experience is updated directly after attribution, without
a paired-validation promotion stage.

## Historical verification and limits (2026-09-30)

This section records the old gated-update implementation. Preserve its measurements;
its acceptance and paired-validation statements do not describe the current direct-update path.

Final local verification on 2026-09-30: **451 tests and 27 subtests passed**.
`compileall` and `git diff --check` also passed. The real API studies described below
are separate from these offline regression tests.

The offline regression suite exercises the installed executor, native JIT bridge with
scripted transport, planning corrections, evidence guards, scope filtering and official
score invariance. The CLI smoke also exercises generation, execution, per-rubric
evaluation, attribution, paired validation, accepted updates and cross-task reuse:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/jit_mas -q
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode smoke --state outputs/quality_improvements_final_20260930/experience.sqlite --output outputs/quality_improvements_final_20260930/runs
```

The 2026-09-30 smoke completed seven harness builds, accepted fixture snapshot version
1 and ran two following fixture tasks, with zero paid requests. Every fixture result
is marked `software_test_only`. This establishes wiring, not benchmark improvement.

The subsequent real API studies used **132 model attempts and 677749 ledger tokens**
in total. No full benchmark or training was run. See the
[live quality report](jit_mas_live_quality_20260930.md) for the independent audit and
artifact references. These results do not establish accepted experience reuse or a
positive overall quality gain.

`scripts/compare_jit_mas_quality.py` supports that bounded implementation comparison,
not experience acceptance. It launches one arm in a separate process importing only
its selected checkout, requires a clean `a1ee6cd` baseline, freezes separate runtime
fingerprints, and binds a common model/configuration/official judge protocol. It uses
an empty read-only experience store and `evaluate`, with exact-hash generated-code
review before execution. Credentials enter only through stdin. Existing arm outputs
cannot be overwritten or silently rerun; failed or incomplete arms cannot produce a
normal mean score difference. `--summarize` checks stored evidence without model calls.

The initial preregistration is
`outputs/quality_comparison_20260930/pre_registration.json`. It froze two previously
untested public tasks (neural-network learning and autonomous-driving sensor comparisons)
before inspecting their rubric counts. Before the first live call, the user's latest
authorization replaced the earlier 70-attempt/500000-token limit with **120 attempts
and 1000000 ledger tokens per arm**, identically for both implementations. There was
no automatic budget expansion. The preflight lower bound is **66 calls**, comprising
56 official rubric judgments plus at least five planning/generation/execution calls
for each of two tasks. This is not a completion guarantee for dynamic teams, retries
or repairs. Explicit preflight fingerprint approval and exact-hash generated-code
review remain required; tasks and teams must not be changed simply to fit the budget.

## Real study outcomes

The initial frozen A/B consumed **58 attempts and 327663 ledger tokens**. Its evidence
is sealed, including failures:

| Arm | Neural-network task | Sensor-comparison task |
| --- | --- | --- |
| Baseline | Planning failed; no scored final answer | Score 0.425 |
| Candidate | Checkpoint failure followed by repair `APIConnectionError`; no scored final answer | Local-plan identity failure; no scored final answer |

The paired mean difference is `null`. No failed task was removed or replaced, and the
candidate produced no complete scored deliverable in that initial comparison.

A subsequent, explicitly registered **candidate-only development regression** reused
the same two now-exposed tasks after limited protocol fixes. It consumed **74 attempts
and 350086 ledger tokens**, completed both tasks, and needed no additional JIT harness
repair:

| Task | Signed score | Normalized score |
| --- | --- | --- |
| Neural-network learning | 54/85 | 0.635294 |
| Sensor comparison | 50/120 | 0.416667 |

This establishes successful execution of these two development cases, not a new blind
test, an A/B improvement, or a SOTA claim. The read-only experience bank stayed empty
at version 0. These real runs used `evaluate`, so they did not validate experience
acceptance or the full learning loop demonstrated only by the offline fixture smoke.

The runner's explicit `--development-regression` mode permits only a candidate arm.
It requires `study_kind="development_regression"` and a structured `parent_study`
anchor containing the original study ID and summary hash. The anchor is structurally
checked, not dereferenced as an arbitrary path. Its separate `development_protocol.json`
and reports label the exposed-task scope; it cannot share an original A/B arm directory
or overwrite consumed arms. The normal `--summarize` entry point rejects development
protocols and cannot average these scores into the sealed A/B comparison. Credentials,
native execution, manual code review and the explicit budget ceilings are unchanged.

The scope gate is an inspectable lexical heuristic, not a semantic retriever; it may
miss paraphrases. The source-count check is likewise a limited quantity/category
heuristic, not proof of private-rubric leakage. Prompt guidance cannot guarantee factual
correctness. Unsafe local execution remains host execution, even after exact-hash review,
not a security sandbox.
The original implementation verification document records an earlier offline-only
stage and should not be read as a current benchmark report.
