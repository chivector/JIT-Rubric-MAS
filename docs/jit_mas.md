# JIT-MAS implementation and run guide

This research prototype uses two evolution levels. At the meta-level, the Meta-Agent evolves
rubric prediction and rubric-guided MAS design experience. At the MAS-level, participating
agents evolve reusable **full agent harness prototypes**: role context/prompt, skills, memory,
tool preferences, reasoning/planning strategies, communication and installed harness policy.
The Agent Pool also evolves its membership, role specialization and optional hierarchy.
`agent_pool`, `AgentProfile` and `AgentPoolSnapshot` remain the code/storage names for those
prototypes. The task coordinator uses JIT's loader, model client, tool registry, Action protocol
and RunResult. The default pooled path reuses the installed MAS scaffold; the legacy
task-specific five-block generation path remains available with `evolving_agent_pool: false`.

## Closed loop

**MAS construction performs no bi-level attribution and does not modify the persistent Agent Pool.** Before execution, Meta predicts
rubrics from the public query, then uses them to design responsibilities and topology,
select and combine retained harness prototypes, and adapt the selected roles to this query.
When no retained harness fits, Meta can create a complete task-local profile and record
its creation rationale and use. Local
planning is task adaptation; rubric ownership is a design contract, not execution credit.
Task-local creation only changes the current MAS. Merge, Delete and other Pool structural
operations belong to post-execution reflection and evolution. Construction neither reads
future judge feedback nor evolves persistent profiles or their hierarchy.

`PublicTask + frozen Agent Pool -> Meta rubric prediction and MAS design
-> select retained or create recorded task-local full harness prototypes
-> agent-owned task adaptation -> reconcile
-> frozen plans and profiles -> reusable MAS scaffold
-> role execution under the selected execution mode -> shared ledger -> Writer submission
-> frozen submission -> independent ResearchRubrics evaluation
-> semantic alignment -> Meta global attribution -> typed per-agent rubric assignment
-> agent local attribution and harness reflection -> Meta evolution integration
-> select harness updates and Pool Add/Delete/Prune/Split/Merge/Specialize/Reorganize operations
-> structural/provenance checks -> atomic Meta, harness-prototype and Pool structure update
-> reuse evolved members on the next task`.

Only evolution/stream tasks enter the post-evaluation attribution and evolution stages.
Validation/test tasks construct and execute a MAS from frozen state, including recorded
task-local creation when needed; their feedback and temporary profiles do not enter either
evolution level or accumulate across held-out tasks.

- `schemas.py` validates public/evaluation, rubric, team, evidence, feedback and proposal contracts.
  `RubricGraph` contains predicted requirement nodes and typed prerequisite/support/overlap/
  tradeoff edges. `TeamSpec.coverage`, `primary`, `reviewers` and agent `rubric_ids` record the
  current MAS's rubric responsibilities. Prediction graph IDs and edge endpoints are validated;
  responsibilities are checked against the frozen graph and team.
- `planning.py` obtains task-specific requirements, responsibilities, dependencies and budgets
  from JSON model outputs. Local agents can add requirements or challenge the initial allocation.
  Meta sees a prototype catalogue containing traceable identities, capabilities, versions,
  task counts, hierarchy and retained harness summaries. It can propose a complete task-local
  profile when the catalogue lacks a suitable harness; `AgentSpec.temporary_profile` and
  `creation_rationale` freeze that creation. `temporary_agents.json` records all temporary
  prediction candidates, including whether the final team selected them.
  Each selected agent sees its own scoped profile and chooses retained skills, tools, reasoning,
  memory and harness policy. Reconciliation preserves its identity and internal choices.
- `agent_pool.py` initializes Writer, Searcher, Critic, Planner, Analyst and Generalist as seeds,
  rather than a fixed role universe. Membership, roles, capabilities and optional parent-child
  organization can change during evolution. `AgentProfile.parent_agent_id` permits a flat
  Pool or an acyclic forest; no fixed root, hierarchy, depth or specialization route is prescribed.
  Seed profiles have no learned task history; maturity counts reflect completed source tasks,
  not a quality guarantee. Profiles contain role prompts, named skill instructions, conditional
  evidence-linked memory, preferred tools, reasoning/planning methods, communication habits and
  a typed harness policy. Custom profiles can be supplied in the versioned snapshot.
- `bridge.py` freezes selected profiles in the hashed public sidecar and installs the checked-in
  MAS scaffold without a harness-generation model call. Agent prompts and policies come from
  retained or frozen task-local profiles. With the pool disabled, it invokes `MetaReActAgent.run(generate_only=True)`, parses the original four Python
  blocks plus YAML, checks the contract, selects a candidate using the original selector, and
  supports bounded pre-execution interface repair. A typed, hashed `team.json` sidecar is separate from
  the five generated blocks. A single valid candidate uses the existing deterministic `_pick`;
  multiple candidates use the existing public-task `_judge_case` selector, not the task evaluator.
- `execution.py` binds the sidecar to the generated Action instance, loads it with `load_harness`,
  and calls `AgentRuntime.run`. Each selected role has its own messages, model instance and full
  observable trace. In the default `single_pass` mode, each role has at most one
  execution-model call. Task-conditioned Analyst and Evidence
  functions are independent and may run in parallel; their requirements, outline, evidence spans,
  source references, contributions and raw tool evidence are merged into a structured ledger.
  The Writer reads the complete snapshot once and produces the final artifact without external
  tools or inter-agent requests. Contributors may request one permitted external-tool batch;
  results enter the ledger without recalling the contributor. Genuine forward dependencies and
  small combined-role tasks remain supported. Missing information becomes an explicit limitation.
  In `iterative_shared_ledger` mode, an active role can receive tool results and updated ledger
  observations on later calls; the Writer can use permitted tools and revise before submitting.
  The cooperative scheduler dispatches initial DAG-ready roles, then may reactivate completed
  peers when public messages or revised artifacts require clarification. Each role keeps its
  private history while only public ledger events cross role boundaries. Finite call ceilings,
  token budgets, and timeouts remain binding; no fixed round count is imposed. Only a successful
  terminal contribution enters the downstream artifact set; a failed role's draft is not a final answer.
- `attribution.py` retains the initial and reconciled predictions separately. Attribution starts
  after immutable submission and judge feedback. Meta global analysis considers all rubrics
  and the current MAS before each agent analyzes its assigned subset. Global analysis
  sees an event index and shared artifacts; each local analysis sees its own complete observed
  inputs/outputs and connected evidence, with one bounded indexed evidence request. Findings
  remain hypotheses with alternatives and uncertainty, not identified causal effects.
- `RubricCreditAssignment` makes local rubric scope an explicit validated record. Semantic
  alignment links evaluated criteria to predicted requirements; frozen execution/review
  responsibilities determine the per-agent assignment. Let `C` combine `rubric_ids`,
  `coverage`, `primary` and `reviewers`, relating predicted rubrics to agents,
  and let `M` relate evaluated to predicted rubrics;
  the semantic subset for agent `a` contains evaluated `j` when some predicted `r` satisfies
  both `(j, r) in M` and `(r, a) in C`. Meta may add an explicit subset `D_a` of related
  evaluated criteria after inspecting the completed roles and collaboration. The final
  scope is that semantic subset union `D_a`; each supplementary assignment requires a
  nonempty rationale and known agent/criterion IDs. These assignments and rationales are
  preserved in `global_outline` and rebuilt when replaying frozen attribution. Assigning
  a missed/unowned criterion for reflection does not retroactively make it an execution
  responsibility: its prediction/organization gap remains part of global analysis.
  Each assignment preserves adjacent rubric graph edges without automatically assigning
  neighboring evaluated criteria. Assignments and their responsibility bases are saved in
  `attribution.json` rather than inferred only from prompt wording.
- `experience.py` retains three logical banks in SQLite for compatibility. In the full agent
  harness prototype path, new meta-level proposals use only rubric or organization banks:
  rubric advice enters prediction and organization advice enters reconciliation. Role-specific
  execution practices evolve in the corresponding harness prototype; the legacy execution
  bank remains available to the historical path. Retrieval excludes the current task and
  validation/test origins. Teams vary by task; retained prototypes and their evolution history
  persist across tasks, while explicit Pool operations can change membership and capabilities.
- `experience.py` writes the Meta-selected proposal directly after evolution integration, preserving
  schema/target/provenance checks, base-version consistency, duplicate protection and rollback.
  There is no paired rebuild or accept/hold/reject quality decision. The snapshot records
  `applied_proposals`; `experience_updates` contain write receipts, not quality verdicts.
  A valid write is not evidence of improved task quality. Optional validation datasets are
  reserved for external analysis and cannot gate the online update path.
- After complete evolution/stream feedback and attribution, each participating retained or task-local agent
  reflects using its typed rubric assignment, assigned predicted requirements, local trace,
  observed event records, numerical submission feedback and only its relevant rubric scores, plus
  supported findings. The feedback summary excludes private criteria, references and judge
  reasoning; a team score does not establish an individual agent's causal contribution.
  Each agent proposes conditional process lessons
  and changes to its own skills, prompt, strategies or harness. Meta then reads all candidate
  agent updates, temporary creation records and experience proposals in `evolution_integrate`.
  It selects this task's updates, decides whether to retain executed new harnesses, and proposes explicit
  Add, Delete/Prune, Split, Merge, Specialize or Reorganize operations from task history, failure
  modes, observed quality and actual token cost, preserved as `AgentPoolObservation` records.
  `EvolutionDecision.pool_operations` saves typed `AgentPoolOperation` decisions.
  Operation support does not require a structural
  change on every round. Its decision and reasons are recorded in `meta_evolution.json`.
  Only this final selection can commit; candidate reflections and task-local creation are not
  direct writes. Meta experience, selected agent updates and Pool operations commit together
  in one SQLite transaction, with identity/version, provenance, parent-reference and cycle checks.
  An incomplete reflection or invalid operation cannot partially update one layer.
  Held-out validation/test tasks neither reflect nor update the pool.

Pool structural operations carry complete target profiles where a new or specialized harness
is required, explicit source identities and base versions, a rationale and evidence references.
Add may retain a task-local creation selected by the final team, executed and supported by feedback;
Delete/Prune removes a member and handles affected hierarchy;
Split creates specialized prototypes; Merge combines overlapping prototypes; Specialize changes
role, capabilities and harness; Reorganize changes the optional hierarchy. Meta chooses the
organization from evidence rather than following a predefined role tree. Pool hierarchy is
prototype organization, not the current task's execution topology. Removing a persistent member
does not change profiles already frozen in historical task artifacts.

`evolving_agent_pool` defaults to `true`. Set it to `false` for the historical per-task
generation control. `persistent_experience: false` disables learning in both layers;
`fixed_team` retains its configured control path. `local_planning: false` uses retained agent
policies without local method adaptation. A local `selected_skills: null` retains the skill
library; `selected_skills: []` deliberately selects no retained skills. Task adaptation belongs to `AgentSpec.task_prompt`
and does not rewrite a profile. Named skills are reusable instructions rather than installed
third-party tools. Harness evolution currently selects the installed `rubric_mas` harness,
full/recent conversation memory and allowed/preferred-first tool ordering; it does not generate
arbitrary new Python for each agent. Tool permissions, topology, execution mode and budgets
remain enforced by the current task.

The MAS generator and Pool evolution consider quality and token cost together. Each global/local planning
request receives the live shared budget, including consumed and reserved tokens. Reconciliation
records a typed `TeamSpec.budget_plan` with per-role input/output token, model/tool call and
communication estimates, later-stage reserves, a quality-cost rationale and a stopping policy.
Estimates must fit the remaining shared resources and role response ceilings; they do not
create extra call or round caps. Actors receive current resources and their role estimate.
Observed role costs enter evidence-linked evolution reflection, with unknown usage flagged
and monetary prices left null. Estimates guide design; actual ledger guards enforce limits.

`execution_mode` defaults to `single_pass`. See the [single-pass executor contract](jit_mas_single_pass.md)
for its unidirectional architecture and migration boundaries. For `iterative_shared_ledger`,
agent, team, task-model and tool-call ceilings remain configurable; setting an optional call
ceiling to `null` removes that ceiling. Token budgets, request timeouts and execution timeouts
still apply. This runtime option alone does not satisfy every prerequisite of the independent
v5 formal campaign, including its registration, concurrency and context-window contracts.
`local_rounds`/`local_planning` refer only
to pre-execution planning; bounded attribution follow-ups remain separate. Experience
updates are direct and do not create extra execution or validation calls. Single-pass call
ceilings do not permit a second execution call. Native JIT interface repair is allowed before role execution begins, but
an error after any role model call starts cannot repair and replay the whole team.
Earlier smoke and live reports are historical evidence for their recorded implementations;
old paired-validation counts are not measurements of the direct-update path.

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
source evolution task and one frozen held-out fixture task through scaffold reuse, loading and
execution. It exercises atomic dual evolution and cross-task profile reuse, not
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
.\.venv\Scripts\python.exe -m scripts.prepare_researchrubrics --download --output-dir dataset/researchrubrics-local --seed 0 --evolution-size 5 --validation-size 0
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
Historical native pilots are documented separately and do not establish current method quality.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode evolve --config configs/jit_mas.native.example.yaml --data dataset/researchrubrics-local/processed_data.jsonl --splits dataset/researchrubrics-local/splits.json --state outputs/native/experience.sqlite --output outputs/native/runs --limit 1 --unsafe-local
.\.venv\Scripts\python.exe -m scripts.run_jit_mas --mode evaluate --config configs/jit_mas.native.example.yaml --data dataset/researchrubrics-local/processed_data.jsonl --splits dataset/researchrubrics-local/splits.json --state outputs/native/experience.sqlite --output outputs/native/test --limit 1 --unsafe-local
```

The first command runs at most one evolution source task, then directly persists the final
Meta-selected experience proposal, selected harness updates and Pool structural operations if present. It does not execute validation tasks or automatically run the
full benchmark. `stream` takes the same arguments
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

Freeze exports Meta experience, complete full agent harness prototypes and Pool membership/hierarchy in the same committed snapshot;
evaluate opens the store read-only and fixes both layers for all test tasks. Rollback restores
both layers and Pool structure to a preserved version and writes an audit event.
Evaluation feedback never becomes the next test task's memory. Repeating a submitted evolve or
stream task resumes its persistent journal without submitting or committing again. Changed
policy requires a fresh state database for that task. `--no-resume` is for fresh execution jobs,
not for silently overwriting an already journaled submission.

Run directories contain frozen plan hashes, planning calls, generated harness identity,
full execution/sub-run traces, immutable submission hash/time, raw per-criterion evaluation,
alignment, typed rubric credit assignments and attribution calls, agent reflection calls,
task-local harness creation records, Meta evolution integration candidates/decisions,
Pool structural operations, proposals, dual update receipts,
complete result and budget records.
Failure paths retain budget and error records; bridge failures also attach attempted code/trace
metadata. Historical pair records remain auditable but are not produced by the active update
path; there is no parameter-training job in this implementation.

Cache identity includes public task, local attachment contents, model and policy config,
prompt/code/tool/description hashes, whole experience snapshot, evaluator version, private
record hash and split manifest. Unsnapshotted live-tool or remote-attachment native executions
do not reuse the task-result cache. Persisted submissions still resume as immutable submissions.
Provider sampling and live search results cannot be made identical by a seed; these differences
are explicitly recorded. External comparative analyses should use controlled evidence snapshots
for research claims; such analyses are not online update gates.

Token reservations and call/tool counters share a lock across agents. Parents never re-charge
serialized child trajectories. Reports separate inference, external evaluation and update
usage; removed paired-validation calls are not charged. Monetary cost is `null` (unknown), not zero. Network timeouts bound waiting;
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
production sandbox. The small versioned bank is scope-tagged and capped at retrieval; public
task operations and role capability filter applicable advice before model prompting.
When the bank exceeds the cap, task-grounded lessons and distinct substantive public
matches take priority, with recency breaking ties. The selected lessons retain their
chronological order; private scores and rubric text do not rank them. This
lexical gate is not a learned semantic retriever. Large-scale memory
consolidation remains outside this first vertical implementation.

See the [quality-focused update](jit_mas_quality.md) for the latest implementation changes,
offline verification and the distinction between execution success and answer quality.

See [codebase audit](jit_mas_codebase_audit.md), [runtime audit](jit_mas_runtime_audit.md), and
[related work](jit_mas_related_work.md) for the verified source mechanisms and implementation choices.
