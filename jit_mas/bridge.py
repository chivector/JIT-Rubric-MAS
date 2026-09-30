"""Task-conditioned JIT generation, five-block parsing, selection and repair."""

from __future__ import annotations

import ast
import copy
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jit.harness_ops import SECTION_TAG_TO_FILE, WORKSPACE_DIR, _parse_harness_response
from jit.meta_agent import MetaReActAgent
from jit.schemas import MetaAgentRequest
from jit.selector import _judge_case, _pick
from scripts.models.base import ChatMessage

from .execution import _data, content_hash, validate_team
from .schemas import PublicTask, RubricGraph, TeamSpec

SEED_DIR = Path(__file__).resolve().parents[1] / "harness_factory" / "harnesses" / "rubric_mas"


def seed_response() -> str:
    """Trusted fixture response, still sent through JIT's real generation parser."""
    return "\n\n".join(f"<<<{tag}>>>\n{(SEED_DIR / name).read_text(encoding='utf-8')}"
                         f"\n<<<END_{tag}>>>" for tag, name in SECTION_TAG_TO_FILE.items())


class ScriptedHarnessModel:
    """Offline software fixture, never represented as a trained JIT checkpoint."""

    model_id = "scripted-harness-software-test"
    kwargs = {"max_tokens": 64000}

    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatMessage(role="assistant", content=seed_response())

    def get_token_counts(self):
        return {"input_token_count": 0, "output_token_count": 0}


@dataclass
class SynthesizedHarness:
    name: str
    path: Path
    backend: str
    team_hash: str
    task_hash: str
    code_hash: str
    sidecar_hash: str
    meta_trajectory: list = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    repair_count: int = 0

    @property
    def sidecar(self):
        return json.loads((self.path / "team.json").read_text(encoding="utf-8"))

    def verify_integrity(self):
        files = {name: (self.path / name).read_text(encoding="utf-8")
                 for name in SECTION_TAG_TO_FILE.values()}
        if content_hash(files) != self.code_hash:
            raise ValueError("generated harness changed after validation")
        if content_hash(self.sidecar) != self.sidecar_hash:
            raise ValueError("TeamSpec sidecar changed after generation")
        if self.backend == "scripted":
            expected = _parse_harness_response(seed_response())
            if files != expected:
                raise ValueError("scripted backend only executes the trusted checked-in seed")

    def to_dict(self):
        return {"name": self.name, "path": str(self.path), "backend": self.backend,
                "team_hash": self.team_hash, "task_hash": self.task_hash,
                "code_hash": self.code_hash, "sidecar_hash": self.sidecar_hash,
                "meta_trajectory": self.meta_trajectory, "selection": self.selection,
                "repair_count": self.repair_count}


class _PublicGenerationAdapter:
    def __init__(self, description, tools):
        self.description = description
        self.tools = tools

    def format_task(self, item):
        return self.description

    def get_tools(self):
        return []

    def get_task_tools(self, item):
        return self.tools


class JITHarnessSynthesizer:
    def __init__(self, backend="scripted", meta_model=None, meta_config=None, *,
                 candidates=1, max_repairs=2, selector_model=None, tools=None):
        if backend not in {"scripted", "native_jit"}:
            raise ValueError("backend must be scripted or native_jit")
        self.backend = backend
        self.meta_config = dict(meta_config or {})
        if backend == "native_jit":
            if not self.meta_config.get("api_base"):
                raise ValueError("native_jit requires an explicit meta endpoint (api_base)")
            if not self.meta_config.get("model_id"):
                raise ValueError("native_jit requires an explicit meta model_id")
        if not 1 <= candidates <= 8 or not 0 <= max_repairs <= 8:
            raise ValueError("candidates or max_repairs outside bounded limits")
        self.meta_model = meta_model
        if backend == "scripted" and meta_model is None:
            self.meta_model = ScriptedHarnessModel()
        self.candidates = candidates
        self.max_repairs = max_repairs
        self.selector_model = selector_model
        self.tools = tools or {}
        self._agents = {}
        self._generation_trajectories = {}

    def _attach_failure(self, exception, *, artifact=None, failed_attempts=(), names=None,
                        first_model_call=0):
        candidates = []
        for name in (self._agents if names is None else names):
            agent = self._agents[name]
            files = {}
            for filename in SECTION_TAG_TO_FILE.values():
                path = agent.workspace_dir / filename
                if path.is_file():
                    files[filename] = content_hash(path.read_text(encoding="utf-8"))
            candidates.append({"name": name, "path": str(agent.workspace_dir),
                               "file_hashes": files,
                               "meta_trajectory": copy.deepcopy(self._generation_trajectories.get(name, []))})
        calls = getattr(self.meta_model, "calls", [])
        # Store only public generation inputs and observed outputs, never client
        # configuration, credentials, or arbitrary exception object internals.
        exception.jit_mas_failure = {
            "artifact": artifact.to_dict() if artifact is not None else None,
            "failed_execution_attempts": copy.deepcopy(list(failed_attempts)),
            "meta_trajectory": copy.deepcopy(artifact.meta_trajectory) if artifact is not None else [],
            "observed_meta_calls": copy.deepcopy(calls[first_model_call:]) if isinstance(calls, list) else [],
            "generated_candidates": candidates,
        }

    def _agent(self, name):
        config = dict(self.meta_config)
        if self.meta_model is not None:
            # Constructing JIT's normal client does not issue requests. Replace it
            # with the injected metered/scripted transport before generation.
            config.setdefault("model_id", getattr(self.meta_model, "model_id", "injected-meta-model"))
            config.setdefault("api_base", "http://127.0.0.1:1/v1")
            config.setdefault("api_key", "EMPTY")
        agent = MetaReActAgent(config, {"meta_references": {"mode": "desc"},
                                       "meta_review": {"enabled": False}}, workspace_name=name)
        if self.meta_model is not None:
            agent.model = self.meta_model
        self._agents[name] = agent
        return agent

    @staticmethod
    def _description(sidecar):
        reference = (SEED_DIR.parent.parent / "descriptions" / "rubric_mas.md").read_text(encoding="utf-8")
        return (
            "Generate a task-conditioned JIT MAS harness. These are PUBLIC inputs only:\n"
            + json.dumps(sidecar, ensure_ascii=False, indent=2)
            + "\n\nMAS extension contract (preserve the original five tagged blocks):\n"
            + reference
            + "\nThe runtime binds Action.bind_team(team, services) after native loading. "
            "Reuse jit_mas.execution.TeamAction, TeamMemory, TeamPlanning, TeamToolPolicy "
            "as protocol-compatible bases, or implement the same bind_team and independent "
            "services.model_factory interface. TeamSpec must determine actual roles, tools, "
            "dependencies, checkpoints and budgets. Do not hard-code another team. "
            "Do not load team.json at module scope or write any process-wide task state. "
            "Use services.public_task/rubrics/experiences at run time. Each action must return "
            "RunResult with each role's full observed I/O in sub_runs and immutable event metadata. "
            "Runtime errors may be repaired; no evaluator feedback is available. "
            "Keep system_prompt, agent_prompt, planning, summary, final_answer, step in prompt.yaml."
        )

    def _validate(self, agent):
        error = agent._static_harness_checks()
        if not error.startswith("No static errors detected"):
            raise ValueError(error)
        files = {name: (agent.workspace_dir / name).read_text(encoding="utf-8")
                 for name in SECTION_TAG_TO_FILE.values()}
        for filename, class_name in [("memory.py", "MemoryStrategy"),
                                     ("planning.py", "PlanningStrategy"),
                                     ("action.py", "ActionStrategy"),
                                     ("tool_policy.py", "ToolPolicyStrategy")]:
            tree = ast.parse(files[filename], filename)
            if not any(isinstance(n, ast.ClassDef) and n.name == class_name for n in tree.body):
                raise ValueError(f"{filename} must export class {class_name}")
        action_tree = ast.parse(files["action.py"])
        action_class = next(n for n in action_tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionStrategy")
        inherits_team = any(isinstance(b, ast.Name) and b.id == "TeamAction" for b in action_class.bases)
        bound_method = any(isinstance(n, ast.FunctionDef) and n.name == "bind_team" for n in action_class.body)
        if not inherits_team and not bound_method:
            raise ValueError("ActionStrategy must inherit TeamAction or implement bind_team")
        if self.backend == "scripted" and files != _parse_harness_response(seed_response()):
            raise ValueError("scripted backend emitted untrusted code; use native_jit with isolation")
        return files

    def synthesize(self, task, rubrics, team, experiences=()):
        prior_names = set(self._agents)
        calls = getattr(self.meta_model, "calls", [])
        first_call = len(calls) if isinstance(calls, list) else 0
        try:
            return self._synthesize(task, rubrics, team, experiences)
        except BaseException as exc:
            self._attach_failure(exc, names=[name for name in self._agents if name not in prior_names],
                                 first_model_call=first_call)
            raise

    def _synthesize(self, task, rubrics, team, experiences=()):
        task = PublicTask.model_validate(_data(task))
        rubrics = RubricGraph.model_validate(_data(rubrics))
        team = TeamSpec.model_validate(_data(team))
        team_data = validate_team(team, task)
        sidecar = {"schema_version": "1.0", "task": _data(task), "rubrics": _data(rubrics),
                   "team": team_data, "experiences": [_data(e) for e in experiences],
                   "backend": self.backend}
        description = self._description(sidecar)
        public_tools = {name: tool for name, tool in self.tools.items() if name in _data(task)["tools"]}
        adapter = _PublicGenerationAdapter(description, public_tools)
        artifacts, rows, failures = [], [], []
        for rollout in range(self.candidates):
            name = "mas_" + uuid.uuid4().hex
            agent = self._agent(name)
            result = agent.run(MetaAgentRequest(benchmark_adapter=adapter,
                item={"id": _data(task)["task_id"]}, tools=[], generate_only=True,
                max_repairs=0, repair_only_on_error=True))
            trajectory = list(result.meta_agent_trajectory)
            self._generation_trajectories[name] = trajectory
            repairs = 0
            while True:
                try:
                    files = self._validate(agent)
                    break
                except (ValueError, SyntaxError, FileNotFoundError, yaml.YAMLError) as exc:
                    if repairs >= self.max_repairs:
                        failures.append(f"candidate {rollout}: {exc}")
                        files = None
                        break
                    event = agent._repair_harness(str(exc), [{"error": str(exc)}],
                                                  evaluation_result={}, validation_error=str(exc))
                    event["stage"] = "protocol_repair"
                    trajectory.append(event)
                    repairs += 1
            if files is None:
                continue
            sidecar_path = agent.workspace_dir / "team.json"
            sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
            artifact = SynthesizedHarness(name, agent.workspace_dir, self.backend,
                content_hash(team_data), content_hash(_data(task)), content_hash(files),
                content_hash(sidecar), trajectory, repair_count=repairs)
            artifacts.append(artifact)
            completion = "\n\n".join(f"<<<{tag}>>>\n{files[filename]}<<<END_{tag}>>>"
                                       for tag, filename in SECTION_TAG_TO_FILE.items())
            rows.append({"case": _data(task)["task_id"], "rollout": len(artifacts) - 1,
                         "n_sub": len(files), "protocol_score": 1.0, "user": description,
                         "completion": completion})
        if not artifacts:
            raise RuntimeError("JIT generation failed: " + "; ".join(failures))
        if len(rows) == 1:
            selection = _pick(rows, "protocol_score", "single_protocol_valid_candidate")
        else:
            model = self.selector_model or self.meta_model
            if model is None:
                raise ValueError("multiple candidates require an explicitly metered selector_model")
            verdict = _judge_case(_data(task)["task_id"], rows, model, 24000)
            if verdict["choice"] is None:
                raise ValueError("selector did not return a valid choice")
            for i, row in enumerate(rows):
                row["judge_score"] = float(i == verdict["choice"])
            selection = _pick(rows, "judge_score", "judge_pick")
            selection["judge_verdict"] = verdict
        selected = artifacts[selection["selected"][_data(task)["task_id"]]]
        selected.selection = selection
        selected.verify_integrity()
        return selected

    def repair(self, artifact, failure, failed_run=None):
        """Bounded exception-only repair. Task quality or evaluator scores are forbidden."""
        if not isinstance(failure, Exception):
            raise TypeError("repair requires an execution/interface exception, never a score")
        if artifact.repair_count >= self.max_repairs:
            raise RuntimeError("JIT exception repair budget exhausted")
        artifact.verify_integrity()
        agent = self._agents.get(artifact.name) or self._agent(artifact.name)
        agent._last_task_description = self._description(artifact.sidecar)
        failed_cases = [{"error": str(failure)}]
        if failed_run is not None:
            failed_cases[0]["trajectory"] = [s.full_dict() for r in failed_run.sub_runs for s in r.trajectory]
        event = agent._repair_harness(str(failure), failed_cases, evaluation_result={},
                                     validation_error=str(failure))
        event["stage"] = "execution_exception_repair"
        artifact.meta_trajectory.append(event)
        artifact.repair_count += 1
        artifact.code_hash = content_hash(self._validate(agent))
        artifact.verify_integrity()
        return artifact

    def execute_with_repair(self, executor, task, team, artifact, rubrics=None, experiences=()):
        failed_attempts = []
        while True:
            result = None
            try:
                result = executor.execute(task, team, artifact, rubrics=rubrics, experiences=experiences)
                errors = [str(step.error) for run in result.sub_runs for step in run.trajectory if step.error]
                errors += [run.metadata["error"] for run in result.sub_runs if run.metadata.get("error")]
                if errors and result.answer is None:
                    raise RuntimeError("; ".join(errors))
                result.metadata["failed_execution_attempts"] = failed_attempts
                result.metadata["repair_count"] = artifact.repair_count
                return result
            except BaseException as exc:
                failed_attempts.append({"error": str(exc), "run": result.full_dict() if result else None,
                                        "harness_hash": artifact.code_hash})
                # Safety gates and exhaustion are not repairable harness defects.
                if (not isinstance(exc, Exception) or isinstance(exc, (PermissionError, TimeoutError)) or
                        any(term in str(exc).lower() for term in ("unsafe-local", "sandbox", "budget", "timeout",
                                                                "exhausted", "sidecar", "changed after"))):
                    self._attach_failure(exc, artifact=artifact, failed_attempts=failed_attempts)
                    raise
                try:
                    self.repair(artifact, exc, result)
                except BaseException as repair_error:
                    self._attach_failure(repair_error, artifact=artifact, failed_attempts=failed_attempts)
                    raise
