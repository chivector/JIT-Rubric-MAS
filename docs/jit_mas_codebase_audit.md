# JIT-MAS Codebase Audit

Audit date: 2026-09-30. The input was the repository-root
`JIT_to_Rubric_MAS_Codex_Prompt.md`. This audit records inspected implementation
contracts, not assumed paper features.
The checkout observations below describe the initial audit. Current executor
contracts are being migrated to [single-pass shared-ledger coordination](jit_mas_single_pass.md);
verification of this migration is pending, and historical results are unchanged.

## Checkout and Instructions

- At the initial audit, the workspace was `C:\Users\chizh\Desktop\JIT-main`.
  `git status --short` reported that it was not a Git repository. There was no
  local `.git`, so branch, commit, tracked/untracked state and a reliable baseline
  diff were unavailable. No reset, checkout, commit or push was performed by that audit.
- No `AGENTS.md` was found in the repository or the inspected ancestor paths.
  The root implementation prompt and session instructions govern this change.
- Read the root README, `requirements.txt`, the JIT meta-agent/selector/parser,
  kernel types/protocols/loader/runtime, model and tool interfaces, benchmark
  adapter/registry, and ROMA/AggAgent implementations and descriptions.
- The runtime audit resolved official upstream HEAD to
  `ababa06c2f54d799fd9fbc356e5368f61a452260`. A selected-file comparison found
  the README, meta-agent, schemas, selector, and four kernel files identical
  after newline normalization. This is useful provenance, not evidence that
  this unpacked directory has that exact commit or no preexisting user edits.

## Reuse and Additions

| Existing Path / Interface | Reused Behavior | Added Capability | Compatibility |
| --- | --- | --- | --- |
| `jit/meta_agent.py:MetaReActAgent` | Task-conditioned generation, static checks, review, bounded repair | `jit_mas/bridge.py:JITHarnessSynthesizer` supplies public task, frozen rubric graph, TeamSpec and committed experience as generation context | Original entry point remains available; MAS uses `generate_only` before independent evaluation |
| `jit/harness_ops.py:_parse_harness_response` and `jit/prompt.yaml` | Parses the last complete occurrence of each of five tagged blocks | Typed `team.json` sidecar binds configuration and hashes to generated code | No sixth tagged block or incompatible parser protocol |
| `jit/selector.py:_pick`, `_judge_case` | Completeness-gated choice and public-task judge selection | Bridge selects generated MAS candidates without hidden task criteria; original JIT also retains its logprob path | Harness selection is distinct from benchmark evaluation |
| `scripts/kernel/protocols.py:BaseAction.run` | Action owns the entire run loop | `TeamAction` schedules one call per execution role, merges a structured ledger and runs one final Writer read/call | No task-execution dialogue loop or replacement MAS framework |
| `scripts/kernel/loader.py:load_harness` | Imports four strategies and YAML prompts; supports legacy module aliases and reload after repair | Generated strategies can reuse stable MAS primitives and receive `bind_team` services | Original class exports and protocol types remain intact |
| `scripts/kernel/runtime.py:AgentRuntime.run` | Initializes memory, forms RuntimeContext, dispatches Action and saves results | TeamExecutor injects configured role clients and the allowed tool registry before using the native lifecycle | Bypasses network-creating constructor only, not Action dispatch or native loading |
| `scripts/kernel/types.py:RunResult` / `StepRecord` | Hierarchical `sub_runs`, `metadata`, observable model/tool I/O | Independent agent results, event index, evidence lineage, team/hash metadata | No replacement result hierarchy; existing consumers can use compact output |
| `scripts/models/OpenAIServerModel` and `scripts/tools/registry.py` | Existing model callable and tools | Independently configured meta/global/local/exec/judge endpoints, role tool permissions and shared metering | No provider SDK or MAS stack replacement |
| `harness_factory/harnesses/roma` | Recursive decomposition, dependency artifacts and aggregation | Its design informs task dependencies and final synthesis | Original JIT is not characterized as inherently single-agent |
| `harness_factory/harnesses/aggagent` | Multiple rollouts, full-history sub-contexts and aggregation | Reuses the established hierarchical trace convention | New team roles arise from task requirements, not fixed rollout copies |
| `benchmark/adapter/base.py`, `benchmark/registry.py`, `benchmark/config/` | Common task formatting, tool and evaluation boundary | ResearchRubrics adapter, public/private split and criterion feedback | Registered in the common loading path; original adapters remain accessible |

## Trace Semantics

`StepRecord.dict()` omits `model_input_messages` and
`model_output_messages`; `full_dict()` preserves them. `RunResult.full_dict()`
recursively preserves child runs. Consequently MAS persistence and attribution
use `full_dict()`, including every observable local message/tool interaction.
The default `BaseMemory.get_all_steps()` returns an empty list, so inheriting the
base API alone does not guarantee a usable trace. ROMA and AggAgent provide
concrete accessors, and `TeamMemory` implements its own full step storage.

Task execution contributes public `requirements`, `outline`, `evidence_spans`,
`source_references`, per-role `contributions` and raw `tool_evidence` to a
deterministically merged ledger. Analyst and Evidence functions are independent
by default and may run in parallel. A contributor may request a single permitted
external-tool batch; results enter the ledger without another contributor model
call. The Writer reads the completed snapshot once and uses only terminal
completion/submission, not external tools or agent messaging. Missing information
is retained as a limitation. A real forward dependency is allowed, but not a
second turn or backward conversation. Small tasks can combine functions; domain
capabilities and rubrics remain task-conditioned.

Global attribution receives a shared event index and artifacts, then local
analysts receive complete local traces. Selected indexed cross-agent evidence
can be requested in one bounded follow-up. Provider-internal reasoning is neither
required nor reconstructed. Findings are hypotheses with checked evidence IDs,
counterevidence, alternatives and uncertainty, not experimentally established
causal effects.

## Public and Private Boundaries

`PublicTask` contains only the question, public attachments, explicit constraints,
available tools and capabilities. Extra fields are rejected. The adapter produces
a separate private evaluation record. Generation receives a public adapter and
public-only item; executor and its sidecar do not receive the raw benchmark row.
The coordinator saves the final answer and hash before the official evaluator
opens the private record. Pre-execution plan snapshots are never overwritten by
official criteria. Quality feedback triggers later proposals, not hidden retries
that replace the original submission.

Native JIT interface repair remains available before execution-role model calls
begin. Once any role call has started, failure preserves that attempt; it must
not repair the harness and replay the entire team against the same task.

The source runtime's `set_tool_workspace` asks tools to use a directory. It does
not restrict Python imports, filesystem access, environment variables, network,
or native execution. Dynamic loading executes Python in the host process.
Therefore **workspace isolation is not a security sandbox**. The native path
fails closed when a trusted sandbox is unavailable, unless the user explicitly
selects `unsafe-local`. That mode is labeled as host-access execution and cannot
promise protection of host credentials, unrelated files, or hidden evaluation
records. Scripted integration checks demonstrate data-interface non-disclosure,
not security against malicious generated Python. A deployment of untrusted
harnesses requires a real external sandbox/capability boundary.

## Paper, README and Implementation Differences

- The inspected JIT paper v2 (`2608.25593`, 2026-09-03) describes 13 seeds;
  this source directory originally provides 11, without `react` or `aorchestra`.
  The new `rubric_mas` seed is an implementation extension, not an original seed.
- Root README describes a trainable harness generator and archive improvement.
  This checkout includes inference, generation, selection and repair; it does
  not supply the paper's full model training or streaming harness bank. The MAS
  implementation freezes model parameters and stores versioned experience. It
  does not implement SFT, DPO or RL, nor claim their effects.
- A native path using an arbitrary compatible meta model does not reproduce
  JIT-27B. A proper checkpoint endpoint and appropriate selector/tokenizer are
  necessary for such a comparison. Scripted fixtures are labeled software tests.
- Core JIT result dataclasses remain unchanged. New public/evaluation/experience
  contracts use Pydantic v2, already present in the declared dependencies, for
  boundary validation and JSON serialization.

## State, Direct Updates and Budget

`GlobalAnalyzer` independently predicts, collects local challenges and reconciles
the plan. Rubric, organization and execution experience affect requirements,
team choices and capability-specific behavior; committed state does not fix a
team for subsequent tasks. After attribution, `ExperienceStore.commit(proposal)`
directly applies the first reconciled proposal through structural/provenance,
base-version and duplicate guards. Snapshots record `applied_proposals` and the
pipeline returns `experience_updates` receipts. There is no paired quality gate,
accept/hold/reject state, or automatic validation-task execution.

`local_rounds` and `local_planning` apply only to pre-execution planning, not to
executor conversations. Bounded attribution evidence follow-up remains separate.
Direct persistence does not add model calls; it preserves atomic snapshots,
rollback and read-only evaluation.

`BudgetLedger` reserves before calls under a lock, settles provider usage or a
conservative bound, and records stages. Unknown monetary cost stays unknown.
Serializing sub-runs never charges the ledger again. This resource ledger is not
the structured knowledge ledger. Existing execution allocations share the team
ceilings, but the effective model-call limit is one per execution role; unused
team budget does not create a second turn. Evaluate mode uses a frozen snapshot; stream mode records order
and versions. Cache identities include task, configuration, code, evaluator,
manifest and experience state.

Structural validity does not guarantee useful experience. Direct updates can
propagate mistaken advice, so final claims require untouched frozen tests.
Optional validation analyses are external to the write path and any human tuning
based on them must be disclosed as development exposure. No full benchmark or
training run was started for this implementation.

No SA-RQ, probe miner or reward-kernel implementation was found in the inspected
code. `scripts/probe_jit_mas_api.py` probes API/software compatibility; it is not
an evaluator-co-evolution or probe-mining pipeline. No such pipeline or performance
claim is established by this executor migration.

See [runtime audit](jit_mas_runtime_audit.md) for bridge, loading and repair
details; [primary-source review](jit_mas_related_work.md) for mechanism and
license provenance; and [ResearchRubrics audit](jit_mas_researchrubrics.md) for
the pinned evaluator semantics and official-versus-diagnostic score boundary.
