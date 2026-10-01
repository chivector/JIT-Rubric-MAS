# JIT Runtime And Generation Audit

Initial inspection: 2026-09-30. At that inspection the supplied directory had no
`.git`, so its branch, commit and pre-existing tracked changes could not be
determined. No reset, commit, push, model training, live model request or full
benchmark was performed by that audit. The provenance and verification sections
below retain those historical observations. Execution contracts are now being
migrated to [single-pass shared-ledger coordination](jit_mas_single_pass.md);
verification of that migration is pending, not covered by the initial test count.

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
| Streaming | Paper section 4.2 retains feedback for later tasks. | Existing generate/select/execute and resume files are not a versioned experience bank. New experience persistence is separate from harness references. |
| Selection | The README documents log-probability and judge selection. | `jit.selector` supports both. The new one-candidate default reuses its completeness-gated `_pick`; multiple candidates use its public-prompt `_judge_case` and `_pick`. No hidden evaluator criterion enters selection. |

## Reused Interfaces

| Existing interface | Reuse | Added behavior | Compatibility |
| --- | --- | --- | --- |
| `MetaReActAgent.run` | `generate_only=True`, public adapter, independent run workspace | The task, predicted graph, frozen pre-execution TeamSpec and retrieved experiences become generation context. | Does not execute the legacy benchmark evaluator during generation. |
| `_parse_harness_response` | Original last-tagged-block parser, used by generation and repair | Strict static protocol checks; trusted-seed check for scripted tests | Still exactly four Python modules and `prompt.yaml`. |
| `MetaReActAgent._repair_harness` | Original error-conditioned repair prompt and file updates | Bounded pre-execution interface repair with failed attempts retained | No repair-and-team-replay after any execution-role model call begins; score-only changes never trigger repair. |
| `load_harness` | Original workspace discovery, import invalidation, class exports and prompt injection | Instance-bound `Action.bind_team(team, services)` | No kernel protocol edits or process-wide task variables. |
| `AgentRuntime.run` | Original initialization, tool catalog, `RuntimeContext`, Action dispatch and `RunResult` | Dependency scheduler lives in the generated Action's inherited primitives | Constructor-only dependency injection avoids creating unused API clients; the run lifecycle is unchanged. |
| `RunResult.sub_runs`, `.metadata`, `.full_dict()` | Independent role results and recursive complete observed I/O | Immutable event records, artifact hashes, explicit handoffs, failure records | Compact `.dict()` intentionally omits raw model messages; saved attribution input must use `.full_dict()`. |
| `ToolRegistry` | Original schema formatting and execution lookup | Public-task and per-role allowlists checked before actual dispatch | A role cannot invoke another role's tool through the cooperative primitives. |
| `roma` | Inspected context isolation, artifact storage and dependency execution | DAG validation and bounded scheduling, without a fixed decomposition template | The original seed is unchanged. |
| `aggagent` | Inspected independent rollouts and selective trajectory access | Local histories remain separate; structured contributions and raw tool evidence enter a shared ledger | The original seed is unchanged. |

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
without waiting indefinitely. Each execution role is invoked at most once. The
default Analyst and Evidence functions can run independently in parallel; genuine
forward data dependencies and smaller combined-role plans remain compatible.
Team call reservations are atomic. Existing role and team ceilings do not permit
extra execution turns. `BudgetLedger` meters usage separately from the structured
knowledge ledger; serialization never charges costs again.

The runtime puts each role's responsibilities, predicted requirements, tool
allowlist, checkpoints and capability-matched execution experiences into its own
context. The structured shared ledger deterministically merges `requirements`,
`outline`, `evidence_spans`, `source_references`, `contributions` and raw
`tool_evidence`, never other roles' private histories. A contributor can request
one batch of permitted external tools. Results enter the ledger automatically,
without recalling the role. The Writer reads the completed snapshot once and
submits the final artifact without external tools or additional agent turns.
Missing information is an explicit limitation, not a request for clarification.
There are no task-execution message queues, `send_message`, `read_evidence`, or
`raise_issue` operations.

Primary ownership and relevant requirements remain part of role context. A genuine
forward review dependency may inspect an earlier contribution once; it cannot
send the contributor back through another model turn. The Writer resolves or
reports material conflicts from the completed ledger. Final bridge failures
retain complete failed attempts and observed meta calls on
`exception.jit_mas_failure`, allowing the trusted coordinator to persist them even
when pre-execution repair is exhausted. Once a role model call starts, execution
failure cannot trigger a repair followed by whole-team replay.

Audit events preserve retrieval, published contributions, tool results, ledger
consumption and final-submission lineage. Historical `message_sent`,
`message_consumed`, and `evidence_read` records remain valid records of earlier
runs; they do not authorize message operations in the new executor. A final
event's parent IDs come from the synthesizer's explicit `evidence_ids`. This
records a model's claim of incorporation, not a guarantee of correct interpretation.
Checkpoint completion likewise requires explicit reports; quality verification
still belongs to the independent evaluator. Local `subtask_complete` cannot be
misreported as the team's `final_answer`.

The generated implementation can customize JIT modules and prompts while using
these primitives. The scripted backend emits the checked-in reference modules
and is only a software fixture; task-specific topology still comes from TeamSpec.
It is never evidence of original JIT checkpoint quality or learned adaptation.
Pre-execution global/local planning, post-submission attribution's bounded evidence
follow-up remain outer phases. As of 2026-10-01, the experience update is direct:
no paired-validation calls or quality-promotion decisions follow attribution.
No SA-RQ, probe-mining or reward-kernel implementation was found; this executor
change must not be presented as verification of such a pipeline.

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

## Historical Executed Verification

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

That initial audit called no hosted model, trained JIT checkpoint, real
ResearchRubrics judge, or real task quality benchmark. Native transport injection tests validate software
wiring only. They do not establish model quality, causal attribution accuracy,
secure isolation, or cross-task performance gains.

Subsequent historical live studies are recorded separately in
[the live quality report](jit_mas_live_quality_20260930.md). Neither those studies
nor the initial audit establishes the pending single-pass migration's test results.
