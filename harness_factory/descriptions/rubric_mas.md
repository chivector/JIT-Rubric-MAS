### Harness: rubric_mas

This reference implements the original JIT four-module protocol and prompt.yaml.
It contains no fixed team. A validated task-specific TeamSpec is bound to the
Action instance through bind_team(team, services), without process-wide task state.

- Memory retains each role's own complete observable inputs, outputs, tool results,
  and model-visible messages. Shared artifacts contain no private conversations.
- Planning deterministically serializes the already-frozen TeamSpec, including
  its responsibilities and execution dependency DAG. It makes no model calls.
  Rubric relations are not interpreted as execution dependencies.
- ToolPolicy exposes the existing JIT registry; execution checks each actual tool
  request against the current role's allowlist and the public task permissions.
- Action schedules forward dependencies within the concurrency limit. Each role
  model runs once, publishes a structured contribution or a raw external-tool
  batch, and the designated Writer reads a frozen shared ledger and writes once.
  Local completion cannot submit the team's final answer.
- Prompt templates guide task-appropriate evidence handling and synthesis. Creative
  tasks need not cite unrelated sources or adopt an encyclopedia article format.

The loader requires the exact exported class names below. The installed bases
already implement the original JIT abstract protocols:

| File | Export | Installed Base |
| --- | --- | --- |
| `memory.py` | `MemoryStrategy` | `jit_mas.execution.TeamMemory` |
| `planning.py` | `PlanningStrategy` | `jit_mas.execution.TeamPlanning` |
| `action.py` | `ActionStrategy` | `jit_mas.execution.TeamAction` |
| `tool_policy.py` | `ToolPolicyStrategy` | `jit_mas.execution.TeamToolPolicy` |

For example, `class ActionStrategy(TeamAction)` inherits a working
`bind_team(self, team, services)` and `run(self, task, ctx)`. An override can call
`super().run(task, ctx)` and preserve the resulting `RunResult`. The team is a
plain validated dictionary; services is a `TeamServices` object with attributes,
including `model_factory`, `public_task`, `rubrics`, and `experiences`. It is not a
dictionary. `ctx.model` is an intentionally blocked coordinator callable; role
calls must use the independently metered models obtained through the services.
The `model` parameter passed to `Planning.init_plan` and `Planning.update_plan`
is this same blocked guard, even when renamed or assigned to another variable.
The generic JIT model-based planning/replanning lifecycle does not apply here:
inherit `TeamPlanning`, or use a deterministic override that reads the frozen
TeamSpec. Forwarding the unused guard to `super().init_plan(...)` is valid;
calling it is not. Keep `should_replan` false and do not run model-based summaries.

Customize task-relevant memory, planning, prompts and control flow without
renaming these exports or inventing unprovided methods. Each worker needs its own
Memory instance; changing a shared memory's `current_role` does not satisfy
isolation. `ctx.get_tool_schemas(...)` returns a JSON string, not a tool catalog.
The outer result must use the actual `RunResult.sub_runs` field with `RunResult`
children, never only a `metadata['sub_runs']` list of dictionaries. The inherited
team executor already supplies these contracts and runs the final synthesizer
exactly once after all contributions. The task normally selects independent
Analyst/Evidence capabilities followed by a Writer; these are protocol roles, not
hard-coded agent IDs or fixed domain responsibilities. Genuine forward data
dependencies are allowed. There is no task-internal negotiation, clarification,
peer messaging, or draft-review-rewrite loop. A larger `max_calls` never permits a
second role-model call. The evaluator remains outside the execution team and
receives the final submission only afterward.

`prompt.yaml` must contain nonempty `system_prompt` and `agent_prompt` strings,
plus `planning`, `summary`, `final_answer`, and `step` mappings. `system_prompt`
can render `tools` and `skills_prompt`; `agent_prompt` is plain task-specific text.
The runtime separately supplies each role's task, requirements, responsibility,
tools, checkpoints and `shared_ledger` as structured JSON; `upstream_artifacts` is
no longer a runtime field. The Writer's shared ledger contains `requirements`,
`outline`, `evidence_spans`, `source_references`, `contributions` (published
artifact fields without duplicated ledger bodies), and `tool_evidence` (full
retrieved events). Aggregated ledger entries retain `agent_id` attribution. It aggregates
all contributors. Other roles see only published forward dependencies, never
peers' private histories. The coordinator freezes the ledger and emits
`shared_ledger_ready` and `shared_ledger_read` events rather than sending messages.
Contributor claims and predicted rubrics remain fallible; the Writer should
resolve material disagreements without starting another round.

The Writer finishes with `{"answer":"...","evidence_ids":[],"checkpoints":{}}`.
A non-Writer completion additionally requires
`"ledger":{"requirements":[],"outline":[],"evidence_spans":[],"source_references":[]}`.
Requirements and outline are lists of strings. Evidence spans have `{text,
source_ref}` and source references have `{source_id, locator}`; every span's
`source_ref` must match a source ID in that same contribution. All four fields may
be empty, but missing provenance must be described as a limitation, not invented.
The contributor's `answer` is a brief analysis/evidence summary, not a competing
full report. A non-synthesizer can also use `complete` with these same arguments,
never `final_answer`; only the designated
synthesizer may use `final_answer` to submit the final deliverable. Generated
`agent_prompt` must not force generic `think`/`tools`-only envelopes or a tool call
on every turn. Allow direct answer/checkpoints completion without a tool call.
Contributors may request one independent external tool batch within the role's
allowlist and remaining shared tool budget. Only tool producers may omit `answer`:
the runtime publishes raw tool results into the ledger without fabricating an
analysis or calling the producer again. Tool results are not evidence already
observed by the producing model. The Writer may not call tools other than terminal
`complete`/`final_answer`. `send_message`, `read_evidence`, and `raise_issue` are
removed even when tool budget remains. With zero tool budget, do not request tools;
use available artifacts and explain limitations in the completion instead.

Every assigned checkpoint uses its exact name, with `true` only for an actual
self-check, or a structured `{status, reason, evidence_ids}` report. Allowed
statuses are `completed`, `passed`, `failed`, `unverified`, and `not_applicable`.
`completed` neutrally reports that the check was performed; it does not mean
`passed` or independently verified. Structured reports require a nonempty reason
and a list of exact evidence event IDs already observed by that role; an empty list
is not verification. Honest explained limitations are allowed. Missing checks,
unexplained `false`, or unobserved evidence do not satisfy the protocol. Never
require unconditional `true` values. All checkpoint reports remain self-reports,
not substitutes for external evaluation or a team final submission.

All runtime calls must preserve the bound TeamSpec,
independent model factory, tool guards, budget, and complete sub_runs. The checked-in
seed is also used by the explicitly labeled scripted software-test backend.

Generation and repair receive a system-message supplement with actual installed
method signatures and the short TeamAction and TeamPlanning implementations. This
supplements the generic JIT references while retaining native MetaReActAgent generation, repair,
five-block parsing, selection and loading. Model outputs are validated as emitted;
the bridge does not rename classes or rewrite generated source to make it pass.
Only code/interface failures before any role-model invocation may trigger bounded
native repair. Once a role invocation starts, a response, protocol, or runtime
failure is preserved as a failed single-pass attempt; no whole-team replay or
post-submission quality repair is allowed.
Static interface checks identify common direct coordinator calls and simple
aliases before execution; they are not a sandbox or a proof of arbitrary code safety.

This is a cooperative execution primitive, not a sandbox. Dynamic native code is
disabled without an external isolation boundary or explicit unsafe-local mode.
