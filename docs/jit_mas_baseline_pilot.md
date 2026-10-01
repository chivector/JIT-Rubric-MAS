# Small Method Baseline Pilot

This study compares three implementations without replacing the JIT runtime:

| Arm | Answer generation |
| --- | --- |
| `direct` | One strong, public-task-only model response. |
| `fixed_team_jit` | Fixed Analyst/Evidence/Writer roster, but a real task-conditioned native JIT harness. This is not a fixed-harness baseline. |
| `jit_mas` | Predicted rubrics, global/local pre-execution planning, and a real native JIT harness. |

Both team arms use single-pass shared-ledger execution, isolated role contexts,
and no task-level negotiation or clarification. All arms use the same answer
model, no tools, one attempt per selected task, and the same final-output cap.
They do **not** use equal compute: team planning, harness generation, and worker
calls are charged and reported in addition to writer and evaluator calls.

The independent judge uses a different endpoint and model identifier. Those
identifiers do not independently establish the underlying provider/model's
identity. The pinned official ResearchRubrics prompts, signed criterion weights,
and score calculation are unchanged. A transport wrapper removes only one exact
Markdown fence containing a complete JSON object, preserving the raw response
and all judgment bytes. It neither repairs judgments nor makes another call.

## Protocol

`scripts/compare_jit_mas_baselines.py` reads an immutable `pre_registration.json`.
Task selection uses public questions and a recorded local exposure exclusion
list. Selection records also disclose any accidental private-data exposure;
affected tasks are excluded. Two screening tasks precede two holdout tasks.

All three methods finish screening before an implementation revision is made.
At most one evidence-led revision and one explicitly exposed-task development
phase are planned. The final runtime is frozen before private holdout preflight;
all holdout methods run under that same runtime. No task replacement or automatic
rerun is permitted. Failures remain in the report, and incomplete comparisons
do not receive favorable aggregate means.

Workers import a separate frozen checkout. The common phase protocol binds the
runtime fingerprint, launcher, per-arm configurations, task identities, dataset,
judge, prompts, and role limits. Generated Python requires a manual review
decision bound to exact source and initializer hashes before unsafe-local
execution. This review is **not** a sandbox.

Credentials enter worker processes through stdin, never configuration files or
command-line arguments. Each worker saves submitted answers before evaluation,
per-rubric feedback, model-call ledgers, raw judge envelopes, and artifact hashes.
Summary validation cross-checks submissions, evaluations, execution artifacts,
completion records, native run manifests, budgets, and phase provenance.

## Interpretation

This is a small, single-run, closed-book diagnostic pilot, not a representative
benchmark, SOTA claim, significance test, or replication of published competing
MAS frameworks. Empty read-only experience banks isolate fresh-task behavior;
this study does not measure evaluator evolution, experience updates, or cross-task
experience-learning gains. Provider tokens and conservative ledger estimates
must be reported separately. Monetary cost requires verified pricing and is
not inferred from token counts.

The live registration and results are kept under
`outputs/baseline_pilot_20260930/`. Outputs are intentionally not source-controlled.
