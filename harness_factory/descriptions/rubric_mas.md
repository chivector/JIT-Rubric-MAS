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

Generated modules can subclass TeamMemory, TeamPlanning, TeamToolPolicy, and
TeamAction from jit_mas.execution, customizing task-relevant memory, planning,
prompts, and control flow. All runtime calls must preserve the bound TeamSpec,
independent model factory, tool guards, budget, and complete sub_runs. The checked-in
seed is also used by the explicitly labeled scripted software-test backend.

This is a cooperative execution primitive, not a sandbox. Dynamic native code is
disabled without an external isolation boundary or explicit unsafe-local mode.
