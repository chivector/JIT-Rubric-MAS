# Implementation verification

Environment: Windows PowerShell, repository-local `.venv`, Python 3.12.
The checkout has no `.git`, so local branch, commit and pre-existing Git diffs cannot be reported.
No repository or ancestor `AGENTS.md` was found. No reset, commit or push was performed.

Executed on 2026-09-30:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/jit_mas -q
.\.venv\Scripts\python.exe -m compileall -q jit_mas scripts/run_jit_mas.py scripts/prepare_researchrubrics.py benchmark/adapter/researchrubrics.py harness_factory/harnesses/rubric_mas
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode smoke --state outputs/jit_mas_final/experience.sqlite --output outputs/jit_mas_final/runs
```

The suite contains 110 passing offline checks. These cover the original CLI/seed, real JIT
generation/parser/selector/loader/runtime and exception repair with scripted transport,
independent role contexts, actual concurrency and handoffs, tool permissions and evidence
propagation, canary separation, semantic alignment, attributed omissions and uncertain causes,
paired old/new rebuilding, signed scoring and failed criteria, transaction/rollback/freeze,
stream ordering and resume, budget reservations, cache identity and persisted failure records.

Smoke generated and executed seven harnesses: one source, baseline/candidate on two validation
tasks, and two following tasks. A proposal passed the synthetic quality gate, produced accepted
snapshot version 1, and changed the next comparison team's requirements and roster. All these
outputs are marked `software_test_only`; this is not a benchmark performance result.

Also checked the CLI freeze path and the native no-configuration error path. The latter reports
`native_jit requires explicit meta model and endpoint` before any request. Native transport is
patched to fail if invoked in the offline integration tests.

Not executed: live JIT checkpoint generation, live global/local/execution/judge requests,
real ResearchRubrics quality evaluation, full benchmark runs, any training, or secure sandbox
deployment. Native untrusted Python requires a trusted sandbox that is not supplied here, or
explicit unsafe-local execution with its documented host-access limitations. HTTP timing and
live evidence variation are not eliminated by the paired validation harness.

Pinned first-party source research and licenses are recorded in
[related work](jit_mas_related_work.md), [ResearchRubrics](jit_mas_researchrubrics.md), and
[runtime audit](jit_mas_runtime_audit.md). No performance gain or novelty claim is made.
