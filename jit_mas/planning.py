"""Task-conditioned global/local planning over JIT's model callable protocol."""

from __future__ import annotations

import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel

from .schemas import AgentSpec, LocalPlan, PlannedTeam, Prediction, PublicTask

T = TypeVar("T", bound=BaseModel)


def as_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value


class JsonModelCalls:
    """Each call is a fresh context; records contain only observable I/O."""

    def __init__(self):
        self.call_records: list[dict[str, Any]] = []
        self._records_lock = threading.Lock()

    def ask(self, model: Callable, phase: str, instructions: str, payload: dict,
            schema: type[T], *, agent_id: str = "global") -> T:
        messages = [
            {"role": "system", "content": instructions + "\nReturn only one JSON object "
             "conforming to this JSON Schema:\n" + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": json.dumps(
                {"phase": phase, "agent_id": agent_id, **as_json(payload)}, ensure_ascii=False)},
        ]
        response = model(copy.deepcopy(messages))
        content = response if isinstance(response, str) else getattr(response, "content", None)
        if not isinstance(content, str):
            raise ValueError(f"{phase}: model response must contain JSON text")
        with self._records_lock:
            self.call_records.append({"phase": phase, "agent_id": agent_id,
                                      "messages": messages, "response": content})
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1])
        return schema.model_validate(json.loads(stripped))


PREDICT_PROMPT = """You organize an executable team for this public task before any answer
or evaluation exists. Infer task-specific, observable quality requirements, including implicit
requirements and prohibitions when justified. Importance is nonnegative planning priority;
confidence is uncertainty in the prediction, never a hidden evaluator weight. Distinguish
explicit, inferred, and experience-derived requirements. Link experience IDs when used.
Propose capabilities and concrete responsibilities, not a permanent cast of named roles.
One agent may cover several requirements, and several agents may contribute to one.
Use one agent when sufficient. Explain evidence needs and rubric relationships; a rubric
relationship is not automatically an execution dependency or a causal claim. Accepted
historical experience is conditional advice, not a source of task answers. Do not answer
the task. Choose only public tools and remain within the supplied limits."""

LOCAL_PLAN_PROMPT = """You are an independent candidate agent planning from your assigned
capability, the public task, and the initial quality graph. Inspect the draft critically.
Identify requirements you can own, needed inputs, concrete outputs, collaboration
dependencies, tools, resource needs, uncovered requirements, and risks. You may challenge
the global draft, add missed requirements, refine ambiguous ones, or recommend merging
redundant responsibilities. New requirements need new stable rubric IDs. Do not merely
confirm acceptance, impersonate other agents, or produce the final answer. Preserve your
assigned agent_id and capability. Use only public tools. Treat experience as conditional."""

RECONCILE_PROMPT = """Reconcile the global draft with independent local plans for this task.
Consider every local addition, challenge, gap and resource request. Incorporate useful
discoveries, merge redundant responsibilities, resolve conflicting dependencies, and
record the selection rationale, including reasons for rejecting material local suggestions.
Generate the revised graph and an executable TeamSpec. Preserve stable rubric IDs for
unchanged requirements. Coverage is many-to-many; assign a primary owner and appropriate
review arrangements. Agents' rubric_ids must agree with coverage. All selected tools must
be public. Execution dependencies must be acyclic; rubric relationships need not be.
Ensure the final synthesizer depends transitively on all contributing agents and has enough
budget for synthesis and checks. Its job is to reconcile conflicts and gaps in the requested
output genre, not just concatenate contributions. Agent allocations share the team budget.
The roster, dependencies, checkpoints and budgets should follow this task's requirements,
not a fixed workflow. Include the supplied local plans unchanged in local_plans. Return no
answer to the task and do not invent evaluation feedback."""


class GlobalAnalyzer(JsonModelCalls):
    def __init__(self, global_model: Callable,
                 local_model_factory: Callable[[str], Callable] | None = None, *,
                 max_agents: int = 4, max_parallel: int = 2, local_rounds: int = 1,
                 total_max_calls: int = 16, explicit_rubrics: bool = True):
        super().__init__()
        if not 1 <= local_rounds <= 3 or max_agents < 1 or max_parallel < 1:
            raise ValueError("Planning requires positive limits and one to three local rounds")
        self.global_model = global_model
        self.local_model_factory = local_model_factory or (lambda _agent_id: global_model)
        self.max_agents = max_agents
        self.max_parallel = max_parallel
        self.local_rounds = local_rounds
        self.total_max_calls = total_max_calls
        self.explicit_rubrics = explicit_rubrics
        self.last_prediction: Prediction | None = None

    def _limits(self) -> dict:
        return {"max_agents": self.max_agents, "max_parallel": self.max_parallel,
                "total_max_calls": self.total_max_calls}

    def _prompt(self, prompt: str) -> str:
        if self.explicit_rubrics:
            return prompt
        return prompt + "\nABLATION: Do not explicitly predict rubrics. Return graph rubrics=[] " \
            "and edges=[], all rubric_ids=[] and additions=[], and coverage/primary/reviewers={}. " \
            "Still derive concrete responsibilities, dependencies, tools, checks and budgets " \
            "from the public task and applicable experience."

    def predict(self, task: PublicTask, experiences: Sequence = ()) -> Prediction:
        task = PublicTask.model_validate(task)
        prediction = self.ask(self.global_model, "predict", self._prompt(PREDICT_PROMPT),
                              {"task": task, "experiences": experiences,
                               "limits": self._limits()}, Prediction)
        if not self.explicit_rubrics and (prediction.graph.rubrics or prediction.graph.edges):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        ids = [agent.agent_id for agent in prediction.candidates]
        if len(ids) > self.max_agents or len(ids) != len(set(ids)):
            raise ValueError("Candidate count exceeds limit or contains duplicate IDs")
        rubric_ids = {r.rubric_id for r in prediction.graph.rubrics}
        for agent in prediction.candidates:
            self._check_agent(task, agent, rubric_ids)
        self.last_prediction = prediction.model_copy(deep=True)
        return prediction

    @staticmethod
    def _check_agent(task: PublicTask, agent: AgentSpec, rubric_ids: set[str]):
        if not set(agent.tools) <= set(task.tools):
            raise ValueError(f"Agent {agent.agent_id} requested unavailable tools")
        if not set(agent.rubric_ids) <= rubric_ids:
            raise ValueError(f"Agent {agent.agent_id} references unknown rubrics")

    def local_plan(self, task: PublicTask, prediction: Prediction, candidate: AgentSpec,
                   experiences: Sequence = ()) -> LocalPlan:
        capability = set(re.findall(r"\w+", candidate.capability.casefold()))
        local_experiences = []
        for experience in as_json(experiences):
            signature = set(re.findall(r"\w+", experience.get("capability", "").casefold()))
            if experience.get("bank") != "execution" or capability.intersection(signature):
                local_experiences.append(experience)
        plan = self.ask(self.local_model_factory(candidate.agent_id), "local_plan",
                        self._prompt(LOCAL_PLAN_PROMPT),
                        {"task": task, "prediction": prediction, "candidate": candidate,
                         "experiences": local_experiences, "limits": self._limits()},
                        LocalPlan, agent_id=candidate.agent_id)
        if plan.agent_id != candidate.agent_id or plan.capability != candidate.capability:
            raise ValueError("Local plan changed its agent identity or capability")
        if not set(plan.tools) <= set(task.tools):
            raise ValueError("Local plan requested unavailable tools")
        if not self.explicit_rubrics and (plan.rubric_ids or plan.additions):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        known = {r.rubric_id for r in prediction.graph.rubrics}
        added = [r.rubric_id for r in plan.additions]
        if len(added) != len(set(added)) or known.intersection(added):
            raise ValueError("Local additions must use unique, new rubric IDs")
        if not set(plan.rubric_ids) <= known | set(added):
            raise ValueError("Local plan references unknown rubrics")
        return plan

    def reconcile(self, task: PublicTask, prediction: Prediction,
                  plans: Sequence[LocalPlan], experiences: Sequence = ()) -> PlannedTeam:
        result = self.ask(self.global_model, "reconcile", self._prompt(RECONCILE_PROMPT),
                          {"task": task, "prediction": prediction, "local_plans": plans,
                           "experiences": experiences, "limits": self._limits()}, PlannedTeam)
        # Local testimony belongs to its author, not the reconciler.
        result.local_plans = [plan.model_copy(deep=True) for plan in plans]
        team = result.team
        if not self.explicit_rubrics and (result.graph.rubrics or result.graph.edges
                                         or team.coverage or team.primary or team.reviewers):
            raise ValueError("Explicit rubrics are disabled for this ablation")
        if (len(team.agents) > self.max_agents or team.max_parallel > self.max_parallel
                or team.total_max_calls > self.total_max_calls):
            raise ValueError("Reconciled team exceeds configured resource limits")
        known = {r.rubric_id for r in result.graph.rubrics}
        if not set(team.coverage) <= known or not set(team.reviewers) <= known:
            raise ValueError("Team coverage references unknown rubrics")
        for agent in team.agents:
            self._check_agent(task, agent, known)
            assigned = {rid for rid, owners in team.coverage.items() if agent.agent_id in owners}
            if assigned != set(agent.rubric_ids):
                raise ValueError("Agent assignments disagree with rubric coverage")
        if any(not team.coverage.get(rid) or rid not in team.primary for rid in known):
            raise ValueError("Every planned rubric requires coverage and a primary owner")
        dependencies = {agent.agent_id: set(agent.depends_on) for agent in team.agents}
        ancestors = set(dependencies[team.synthesizer_id])
        previous: set[str] = set()
        while ancestors != previous:
            previous = set(ancestors)
            ancestors.update(dep for aid in previous for dep in dependencies[aid])
        if ancestors != set(dependencies) - {team.synthesizer_id}:
            raise ValueError("Final synthesizer must depend on every contributing agent")
        return result

    def build(self, task: PublicTask, experiences: Sequence = (), *,
              local_planning: bool = True) -> PlannedTeam:
        prediction = self.predict(task, experiences)
        initial = prediction.model_copy(deep=True)
        result = None
        for _round in range(self.local_rounds if local_planning else 1):
            plans = []
            if local_planning:
                with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                    futures = [pool.submit(self.local_plan, task, prediction, candidate,
                                           experiences) for candidate in prediction.candidates]
                    plans = [future.result() for future in futures]
            result = self.reconcile(task, prediction, plans, experiences)
            prediction = Prediction(graph=result.graph, candidates=result.team.agents)
        self.last_prediction = initial
        return result
