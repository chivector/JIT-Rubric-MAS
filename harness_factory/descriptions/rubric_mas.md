### Harness: rubric_mas

This reference implements the original JIT four-module protocol and prompt.yaml.
It contains no fixed team. A validated task-specific TeamSpec is bound to the
Action instance through bind_team(team, services), without process-wide task state.

- Memory retains each role's own complete observable inputs, outputs, tool results,
  and model-visible messages. Shared artifacts contain no private conversations.
- Planning translates the frozen TeamSpec into responsibilities and an execution
  dependency DAG. Rubric relations are not interpreted as execution dependencies.
- ToolPolicy exposes the existing JIT registry; execution checks each actual tool
  request against the current role's allowlist and the public task permissions.
- Action schedules ready roles within the concurrency limit, hands off versioned
  artifacts with event parents, and executes the designated synthesis role after
  its dependencies. Local completion cannot submit the team's final answer.
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

Customize task-relevant memory, planning, prompts and control flow without
renaming these exports or inventing unprovided methods. Each worker needs its own
Memory instance; changing a shared memory's `current_role` does not satisfy
isolation. `ctx.get_tool_schemas(...)` returns a JSON string, not a tool catalog.
The outer result must use the actual `RunResult.sub_runs` field with `RunResult`
children, never only a `metadata['sub_runs']` list of dictionaries. The inherited
team executor already supplies these contracts and runs the final synthesizer
exactly once after its dependencies.

`prompt.yaml` must contain nonempty `system_prompt` and `agent_prompt` strings,
plus `planning`, `summary`, `final_answer`, and `step` mappings. `system_prompt`
can render `tools` and `skills_prompt`; `agent_prompt` is plain task-specific text.
The runtime separately supplies each role's task, requirements, responsibility,
tools, checkpoints and upstream artifacts as structured JSON. All roles finish
their assignment with `{"answer":"...","evidence_ids":[],"checkpoints":{}}`.
Each assigned checkpoint needs its exact name mapped to boolean `true` after the
check has been performed. Local completion is not a team final submission.

All runtime calls must preserve the bound TeamSpec,
independent model factory, tool guards, budget, and complete sub_runs. The checked-in
seed is also used by the explicitly labeled scripted software-test backend.

Generation and repair receive a system-message supplement with actual installed
method signatures and the short TeamAction implementation. This supplements the
generic JIT references while retaining native MetaReActAgent generation, repair,
five-block parsing, selection and loading. Model outputs are validated as emitted;
the bridge does not rename classes or rewrite generated source to make it pass.

This is a cooperative execution primitive, not a sandbox. Dynamic native code is
disabled without an external isolation boundary or explicit unsafe-local mode.
