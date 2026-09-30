# JIT Runtime And Generation Audit

Inspected on 2026-09-30. The supplied directory has no `.git`, so its branch,
commit, and pre-existing tracked changes cannot be determined. No reset, commit,
push, model training, live model request, or full benchmark was performed.

## Upstream Evidence And Local Differences

The upstream HEAD resolved through `git ls-remote` to
`ababa06c2f54d799fd9fbc356e5368f61a452260`. Direct reads of that immutable revision
matched the local `README.md`, `jit/meta_agent.py`, `jit/schemas.py`,
`jit/selector.py`, and `scripts/kernel/{loader,runtime,types,protocols}.py`, after
normalizing CRLF and trailing whitespace. This verifies those files, not the
provenance of the entire exported directory.

Primary references:

- [JIT paper v2, September 3, 2026](https://arxiv.org/html/2608.25593v2).
- [Pinned upstream README](https://github.com/bingreeky/JIT/blob/ababa06c2f54d799fd9fbc356e5368f61a452260/README.md).
- [Pinned meta-agent](https://github.com/bingreeky/JIT/blob/ababa06c2f54d799fd9fbc356e5368f61a452260/jit/meta_agent.py).

| Subject | Primary description | Observed local implementation |
| --- | --- | --- |
| Harness inventory | Paper Table 1 lists 13 seeds. | The original directory contains 11: no separate ReAct or AOrchestra seed. `rubric_mas` is a new twelfth directory. |
| Action semantics | The paper formalizes per-transition actions. | `BaseAction.run(task, ctx)` owns the entire run. The extension implements that actual protocol. |
| Training | Paper section 4.1 describes SFT, preference learning, repair supervision and Evo-GDPO. | The inspected runners are inference clients; no matching training pipeline is supplied. This extension changes experience, not model weights. |
| Reference retrieval | The paper describes an evolving harness bank. | The generation code defaults to eleven static prose descriptions; code mode randomly samples seeds. No quality-gated cross-task streaming bank is implemented in the existing runner. |
| Streaming | Paper section 4.2 retains feedback for later tasks. | Existing generate/select/execute and resume files are not an accepted experience bank. New experience persistence is separate from harness references. |
| Selection | The README documents log-probability and judge selection. | `jit.selector` supports both. The new one-candidate default reuses its completeness-gated `_pick`; multiple candidates use its public-prompt `_judge_case` and `_pick`. No hidden evaluator criterion enters selection. |

## Reused Interfaces

| Existing interface | Reuse | Added behavior | Compatibility |
| --- | --- | --- | --- |
| `MetaReActAgent.run` | `generate_only=True`, public adapter, independent run workspace | The task, predicted graph, negotiated TeamSpec and retrieved experiences become generation context. | Does not execute the legacy benchmark evaluator during generation. |
| `_parse_harness_response` | Original last-tagged-block parser, used by generation and repair | Strict static protocol checks; trusted-seed check for scripted tests | Still exactly four Python modules and `prompt.yaml`. |
| `MetaReActAgent._repair_harness` | Original error-conditioned repair prompt and file updates | Bounded protocol/runtime exception repair with failed attempts retained | Score-only quality changes cannot trigger repair. |
| `load_harness` | Original workspace discovery, import invalidation, class exports and prompt injection | Instance-bound `Action.bind_team(team, services)` | No kernel protocol edits or process-wide task variables. |
| `AgentRuntime.run` | Original initialization, tool catalog, `RuntimeContext`, Action dispatch and `RunResult` | Dependency scheduler lives in the generated Action's inherited primitives | Constructor-only dependency injection avoids creating unused API clients; the run lifecycle is unchanged. |
| `RunResult.sub_runs`, `.metadata`, `.full_dict()` | Independent role results and recursive complete observed I/O | Immutable event records, artifact hashes, explicit handoffs, failure records | Compact `.dict()` intentionally omits raw model messages; saved attribution input must use `.full_dict()`. |
| `ToolRegistry` | Original schema formatting and execution lookup | Public-task and per-role allowlists checked before actual dispatch | A role cannot invoke another role's tool through the cooperative primitives. |
| `roma` | Inspected context isolation, artifact storage and dependency execution | DAG validation and bounded scheduling, without a fixed decomposition template | The original seed is unchanged. |
| `aggagent` | Inspected independent rollouts and selective trajectory access | Local histories remain separate; only directed messages/artifacts are shared | The original seed is unchanged. |

`BaseMemory.get_all_steps()` defaults to an empty list. Both inspected ROMA and
AggAgent memories override it. Their existence does not establish that every
generated Memory preserves raw I/O. The new `TeamMemory` explicitly retains
observed assistant responses as well as tool observations, and its raw steps are
returned in each role's `RunResult`. No provider-private reasoning is requested.

## Binding And Execution

`JITHarnessSynthesizer.synthesize(task, rubrics, team, experiences)` writes
`team.json` beside the five generated files. This sidecar is a typed public
configuration artifact; it is not a sixth generation block. Code, task, TeamSpec,
rubrics, and experience contents are checked before execution. The Action receives
the corresponding per-run services directly on its instance.

`TeamExecutor.execute` creates separate memory and model instances through the
provided role model factory. All workers must have a path to the designated final
synthesizer. Ready workers run with `max_parallel`; failed dependencies propagate
without waiting indefinitely. Team call reservations are atomic, individual
role calls are bounded, and the shared metered-model ledger charges calls once.
Tools and communication use the same ledger. Serialization never charges costs.

The runtime puts each role's responsibilities, predicted requirements, tool
allowlist, checkpoints and capability-matched execution experiences into its own
context. Communication exposes versioned artifacts and directed messages, not
other roles' private histories. Tools can retrieve public evidence; roles can
explicitly read shared evidence, raise issues, and request input. There is no
unbounded role spawning, discussion, or task-local replanning loop.

Primary ownership and review assignments are separately injected into the actual
role context, with the owner's identity and the relevant requirements. Reviewers
are instructed to examine primary artifacts and identify unresolved conflicts.
Final bridge failures retain complete failed attempts and observed meta calls on
`exception.jit_mas_failure`, allowing the trusted coordinator to persist them even
when exception repair is exhausted.

Events distinguish `retrieved`, `evidence_read`, `message_sent`,
`message_consumed`, `artifact_published`, and `final_answer`. A final event's parent
IDs come from the synthesizer's explicit `evidence_ids`. This records a model's
claim of incorporation, not a guarantee that its interpretation is correct.
Checkpoint completion likewise requires explicit reports; quality verification
still belongs to the independent evaluator. Local `subtask_complete` cannot be
misreported as the team's `final_answer`.

The generated implementation can customize JIT modules and prompts while using
these primitives. The scripted backend emits the checked-in reference modules
and is only a software fixture; task-specific topology still comes from TeamSpec.
It is never evidence of original JIT checkpoint quality or learned adaptation.

## Safety And Remaining Limits

The existing loader imports Python in the host process. Workspace directories,
AST checks, allowlists and timeouts do not create a sandbox. There is no validated
isolation service in this checkout. Consequently `native_jit` requires an explicit
meta endpoint and refuses local execution unless `unsafe_local=True` was selected.
It never falls back to scripted generation. `scripted` accepts only exact trusted
seed source, even when a caller injects its own fake model.

Unsafe local generated Python can bypass cooperative tool guards or inspect host
files and credentials. It must not be described as protecting a private benchmark
from adversarial code. A deployment with such requirements needs a separate
process/container capability boundary and evaluator-owned private storage.

Per-call timeouts bound waiting through daemon threads; they cannot forcibly stop
a provider request or arbitrary local tool already in progress. A timeout cancels
further team scheduling. Underlying provider usage may arrive later and remains
in the shared ledger; arbitrary late tool side effects are not prevented. Client
transport timeouts and a genuine isolated execution worker remain necessary for
strong termination guarantees. Unknown monetary prices stay unknown.

## Executed Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/jit_mas -p test_bridge_execution.py -v
```

The 20 offline tests passed, covering real JIT generation/parser/selection/load/run,
original `plan_and_execute` execution, full role traces, actual concurrent roles,
shared-budget exhaustion, dependency failure, cycle rejection, checkpoint gating,
tool permissions, evidence propagation, capability-based experience routing,
timeout return, native endpoint/sandbox gates, code integrity, and actual native
exception repair with an injected scripted transport. They also cover original
`scripts.run_jit --help` imports, failure-context retention after exhausted repair
or invalid generation, and primary/reviewer assignment in actual model inputs.

No hosted model, trained JIT checkpoint, real ResearchRubrics judge, or real task
quality benchmark was called. Native transport injection tests validate software
wiring only. They do not establish model quality, causal attribution accuracy,
secure isolation, or cross-task performance gains.
