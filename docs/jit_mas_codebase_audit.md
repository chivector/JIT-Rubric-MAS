# JIT-MAS Codebase Audit

Audit date: 2026-09-30. The input was the repository-root
`JIT_to_Rubric_MAS_Codex_Prompt.md`. This audit records inspected implementation
contracts, not assumed paper features.

## Checkout and Instructions

- The workspace is `C:\Users\chizh\Desktop\JIT-main`. `git status --short`
  reports that it is not a Git repository. There is no local `.git`, so branch,
  commit, tracked/untracked state, and a reliable baseline diff are unavailable.
  No reset, checkout, commit, or push was performed.
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
| `jit/meta_agent.py:MetaReActAgent` | Task-conditioned generation, static checks, review, bounded repair | `jit_mas/bridge.py:JITHarnessSynthesizer` supplies public task, frozen rubric graph, TeamSpec and accepted experience as generation context | Original entry point remains available; MAS uses `generate_only` before independent evaluation |
| `jit/harness_ops.py:_parse_harness_response` and `jit/prompt.yaml` | Parses the last complete occurrence of each of five tagged blocks | Typed `team.json` sidecar binds configuration and hashes to generated code | No sixth tagged block or incompatible parser protocol |
| `jit/selector.py:_pick`, `_judge_case` | Completeness-gated choice and public-task judge selection | Bridge selects generated MAS candidates without hidden task criteria; original JIT also retains its logprob path | Harness selection is distinct from benchmark evaluation |
| `scripts/kernel/protocols.py:BaseAction.run` | Action owns the entire run loop | `TeamAction` implements scheduling, communication, local execution and final synthesis | No mistaken single-action interpretation or new MAS framework |
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
  implementation freezes model parameters and stores validated experience. It
  does not implement SFT, DPO or RL, nor claim their effects.
- A native path using an arbitrary compatible meta model does not reproduce
  JIT-27B. A proper checkpoint endpoint and appropriate selector/tokenizer are
  necessary for such a comparison. Scripted fixtures are labeled software tests.
- Core JIT result dataclasses remain unchanged. New public/evaluation/experience
  contracts use Pydantic v2, already present in the declared dependencies, for
  boundary validation and JSON serialization.

## State, Validation and Budget

`GlobalAnalyzer` independently predicts, collects local challenges and reconciles
the plan. Rubric, organization and execution experience affect requirements,
team choices and capability-specific behavior; the accepted state does not fix
a team for subsequent tasks. `ExperienceStore` separates staging, validation
and accepted snapshots. `PairedValidator` calls the complete pipeline on old and
candidate states for independent validation tasks, so each side rebuilds the
team and harness. Schema validity alone cannot authorize an update.

`BudgetLedger` reserves before calls under a lock, settles provider usage or a
conservative bound, and records stages. Unknown monetary cost stays unknown.
Serializing sub-runs never charges the ledger again. Execution allocations share
the team limits. Evaluate mode uses a frozen snapshot; stream mode records order
and versions. Cache identities include task, configuration, code, evaluator,
manifest and experience state.

The acceptance criteria and finite validation budget are engineering choices,
not theoretical guarantees. Repeated adaptive reuse of validation tasks can
overfit; final claims require untouched held-out tasks. No full benchmark or
training run was started for this implementation.

See [runtime audit](jit_mas_runtime_audit.md) for bridge, loading and repair
details; [primary-source review](jit_mas_related_work.md) for mechanism and
license provenance; and [ResearchRubrics audit](jit_mas_researchrubrics.md) for
the pinned evaluator semantics and official-versus-diagnostic score boundary.
