# JIT-MAS implementation and run guide

This incremental research prototype keeps JIT's task-conditioned executable harness generation.
It adds a task coordinator around the existing five-block parser, selector, loader, model
client, tool registry, Action protocol and RunResult. It does not introduce another MAS framework.

## Closed loop

`PublicTask -> predict -> independent local plans -> reconcile -> frozen plans -> JIT generation
-> native harness execution -> frozen submission -> independent ResearchRubrics evaluation
-> semantic alignment -> global/local/global attribution -> staged proposal -> paired validation
-> atomic accepted snapshot -> newly built team on the next task`.

- `schemas.py` validates public/evaluation, rubric, team, evidence, feedback and proposal contracts.
- `planning.py` obtains task-specific requirements, responsibilities, dependencies and budgets
  from JSON model outputs. Local agents can add requirements or challenge the initial allocation.
  No task-ID or keyword roster rules exist in the native planner.
- `bridge.py` invokes `MetaReActAgent.run(generate_only=True)`, parses the original four Python
  blocks plus YAML, checks the contract, selects a candidate using the original selector, and
  supports bounded exception-only repair. A typed, hashed `team.json` sidecar is separate from
  the five generated blocks. A single valid candidate uses the existing deterministic `_pick`;
  multiple candidates use the existing public-task `_judge_case` selector, not the task evaluator.
- `execution.py` binds the sidecar to the generated Action instance, loads it with `load_harness`,
  and calls `AgentRuntime.run`. Each selected role has its own messages, model instance and full
  trajectory. The actual roster, tool permissions, DAG, checkpoints and per-role call allocation
  come from TeamSpec. A one-agent task is valid. Synthesis is a model call with incoming artifacts,
  conflict/gap instructions and evidence references, not string concatenation.
- `attribution.py` retains the initial and reconciled predictions separately. Global analysis
  sees an event index and shared artifacts; each local analysis sees its own complete observed
  inputs/outputs and connected evidence, with one bounded indexed evidence request. Findings
  remain hypotheses with alternatives and uncertainty, not identified causal effects.
- `experience.py` stores three logical banks in SQLite. Rubric advice enters prediction;
  organization advice enters reconciliation; execution advice enters matching capability
  contexts. Retrieval excludes the current task and validation/test origins. Each task builds
  a new team; the store never substitutes a permanent roster.
- `validation.py` rebuilds baseline and candidate from independent validation questions.
  It compares full tasks, requires complete identical-version evaluation, checks task quality
  and signed criterion direction, and records accepted/rejected/pending. Costs cannot compensate
  for quality regressions. The default two-task threshold is an engineering rule, not statistical
  evidence of generalization. Repeated use of validation can overfit; untouched test stays separate.

## Install and offline verification

Use Python 3.11 or later. The session used a local `.venv` with Python 3.12. The smaller dependency
set below is sufficient for the provided offline tests; the original requirements include extra
benchmark, document and retrieval packages.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pydantic pyyaml jinja2 openai requests tiktoken json_repair tenacity tqdm rich loguru psutil numpy pillow pytest
.\.venv\Scripts\python.exe -m pytest tests/jit_mas -q
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode smoke --state outputs/smoke/experience.sqlite --output outputs/smoke/runs
```

Smoke makes no paid requests. `offline.py` is an explicitly labelled synthetic model fixture;
the fixture's preset answers and task branches never enter the native provider. Smoke runs one
source task, four paired validation builds and two following tasks through real JIT generation,
parsing, selection, loading and execution. It exercises acceptance and cross-task reuse, not
scientific performance. `software_test_only: true` is persisted throughout. Tests also inject
scripted responses into the native bridge, including a broken Action followed by real JIT repair.
This verifies software wiring, not a JIT checkpoint's ability to write useful code.

## Data preparation

ResearchRubrics is pinned to code `2dc80e2d4c38ddd80439517c259d93c6954b193f` and dataset
`85de3115053d1453ed612caacf4a405edc1ad756`. See [source audit](jit_mas_researchrubrics.md).
The following explicitly downloads only the small public dataset and prepares a seeded,
task-level split; it does not run evaluation or training. A local file can replace `--download`
with `--input path/to/processed_data.jsonl`.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_researchrubrics --download --output-dir dataset/researchrubrics-local --seed 0 --evolution-size 5 --validation-size 4
```

IDs and normalized question text must be unique across splits. Rubrics of one task are never
split apart. Raw records remain evaluator inputs, not execution workspace files. A stream
manifest is prepared with `--stream` in a separate preparation directory and uses a separate
state database. Do not combine stream results with frozen held-out results.

## Small native runs

Configure all five `MAS_*` roles in the root `.env` using the appended `.env.example` entries.
`configs/jit_mas.native.example.yaml` has independent model, endpoint, key variable, timeout and
token limits. No model IDs, prices or real credentials are hardcoded. A normal hosted model used
as the generator is not a reproduction of the JIT checkpoint. Missing settings fail explicitly.

No trusted generated-code sandbox is included in this checkout. Native execution therefore
fails closed by default. The commands below deliberately require `--unsafe-local`: generated
Python then has host access, including files and environment, and hidden data is NOT securely
isolated. Use an isolated disposable environment without sensitive credentials or documents
for such runs. Static protocol checks and a separate working directory are not security sandboxes.
No native calls were executed during implementation.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode evolve --config configs/jit_mas.native.example.yaml --data dataset/researchrubrics-local/processed_data.jsonl --splits dataset/researchrubrics-local/splits.json --state outputs/native/experience.sqlite --output outputs/native/runs --limit 1 --unsafe-local
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode evaluate --config configs/jit_mas.native.example.yaml --data dataset/researchrubrics-local/processed_data.jsonl --splits dataset/researchrubrics-local/splits.json --state outputs/native/experience.sqlite --output outputs/native/test --limit 1 --unsafe-local
```

The first command can run at most one source task and two validation tasks under each of two
states by default. Each has its own whole-pipeline budget; validation costs are recorded
separately. It does not automatically run the full benchmark. `stream` takes the same arguments
with a stream manifest, preserves manifest order and records the state before each submission.

The native CLI currently exposes registered `web_search`, `crawl_page`, `wiki_search` and
`final_answer`. Crawl uses the existing non-model extraction fallback; its normal hidden model
summarizer is not used. Other tools need metered capability adapters before this CLI accepts
them. This bounds unaccounted tool-internal model calls; it does not confine arbitrary Python
in unsafe-local. Existing JIT entry points and their tools remain available.

## State, resume and artifacts

```powershell
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode freeze --state outputs/native/experience.sqlite --output outputs/native/frozen-v1.json
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode rollback --state outputs/native/experience.sqlite --version 0
```

Freeze exports the accepted snapshot; evaluate opens the store read-only and fixes the same
snapshot for all test tasks. Rollback selects a preserved version and writes an audit event.
Evaluation feedback never becomes the next test task's memory. Repeating a submitted evolve or
stream task resumes its persistent journal without submitting or committing again. Changed
policy requires a fresh state database for that task. `--no-resume` is for fresh execution jobs,
not for silently overwriting an already journaled submission.

Run directories contain frozen plan hashes, planning calls, generated harness identity,
full execution/sub-run traces, immutable submission hash/time, raw per-criterion evaluation,
alignment and attribution calls, staged proposals, complete result and budget records.
Failure paths retain budget and error records; bridge failures also attach attempted code/trace
metadata. Pair records retain both rebuilds and can be consumed as future preference data;
there is no parameter-training job in this implementation.

Cache identity includes public task, local attachment contents, model and policy config,
prompt/code/tool/description hashes, whole experience snapshot, evaluator version, private
record hash and split manifest. Unsnapshotted live-tool or remote-attachment native executions
do not reuse the task-result cache. Persisted submissions still resume as immutable submissions.
Provider sampling and live search results cannot be made identical by a seed; these differences
are explicitly recorded. Validation should use controlled evidence snapshots for research claims.

Token reservations and call/tool counters share a lock across agents. Parents never re-charge
serialized child trajectories. Reports separate inference, external evaluation, update and
validation usage. Monetary cost is `null` (unknown), not zero. Network timeouts bound waiting;
Python cannot kill a running local thread, so a timed-out operation can finish later. New agent
dispatch stops, pending reservations stay visible, and this limitation is recorded in the runtime
audit. Hard process termination requires an external sandbox/proxy.

## Ablations and limits

Configuration supports `local_planning: false`, `local_attribution: false`,
`persistent_experience: false`, and `explicit_rubrics: false`. The last still builds task-specific
responsibilities and dependencies. `fixed_team` accepts a validated task-independent TeamSpec
with empty rubric maps for a fixed-MAS control. The original `scripts.run_jit` remains the
original-JIT control; its recursive and aggregate seeds mean it is not necessarily single-agent.
None of these configurations implies a performance ordering.

Frozen official criteria are required by this version's ResearchRubrics CLI. It does not add a
model-generated-criteria deployment evaluator or an evaluator-learning loop. It does not implement
Shapley causal attribution, model parameter training, a learned semantic memory retriever, or a
production sandbox. The small accepted bank is scope-tagged and capped at retrieval; model prompts
decide applicability, while execution advice is filtered by capability. Large-scale memory
consolidation remains outside this first vertical implementation.

See [codebase audit](jit_mas_codebase_audit.md), [runtime audit](jit_mas_runtime_audit.md), and
[related work](jit_mas_related_work.md) for the verified source mechanisms and implementation choices.
