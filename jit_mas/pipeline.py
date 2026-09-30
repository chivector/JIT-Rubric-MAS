"""The task-level closed loop. Private records enter only after submission."""

from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path

from .attribution import RubricAttributor
from .budget import BudgetLedger
from .config import MASConfig
from .experience import ExperienceStore, retrieve
from .planning import GlobalAnalyzer
from .schemas import (EvaluationFeedback, ExperienceSnapshot, PlannedTeam, Prediction,
                      PublicTask, RubricFeedback, RubricGraph, SplitManifest, digest, utc_now)
from .validation import PairedValidator


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ("jit_mas", "jit", "scripts/kernel", "scripts/models", "scripts/tools",
                      "benchmark/adapter", "harness_factory/descriptions", "harness_factory/harnesses/rubric_mas"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix in (".py", ".yaml", ".txt", ".md"):
                files.append((path.relative_to(root).as_posix(), digest(path.read_text(encoding="utf-8"))))
    return digest(files)


def input_fingerprints(task):
    fingerprints = []
    for attachment in task.attachments:
        path = Path(attachment)
        if "://" not in attachment and path.is_file():
            content = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content.update(chunk)
            fingerprints.append({"path": attachment, "hash": content.hexdigest()})
        else:
            fingerprints.append({"path": attachment, "unsnapshotted": True})
    return fingerprints


def convert_feedback(raw):
    return EvaluationFeedback(
        task_id=raw["task_id"], evaluator_version=raw["evaluator_version"],
        score=raw["score"], complete=raw["complete"],
        zero_denominator=raw.get("zero_denominator", False),
        source=raw.get("evaluator_source", "benchmark"), raw=raw,
        rubrics=[RubricFeedback(rubric_id=row["rubric_id"], criterion=row["criterion"],
                               weight=row["weight"], score=row["score"], verdict=row["verdict"],
                               reason=row.get("reasoning", ""), status=row["status"],
                               evidence=row.get("evidence_quotes", []), raw=row)
                 for row in raw["feedback"]])


class MASPipeline:
    def __init__(self, config: MASConfig, model_provider, evaluator_factory, synthesizer_factory,
                 tasks: dict[str, PublicTask], private_records: dict, manifest: SplitManifest,
                 store: ExperienceStore, output_dir, tools=None):
        self.config, self.models = config, model_provider
        self.evaluator_factory, self.synthesizer_factory = evaluator_factory, synthesizer_factory
        self.tasks, self.private_records, self.manifest = tasks, private_records, manifest
        self.store, self.output_dir, self.tools = store, Path(output_dir), tools or {}
        self.code_hash = code_fingerprint()
        all_ids = set(manifest.evolution + manifest.validation + manifest.test + manifest.stream)
        if not all_ids <= tasks.keys() or not all_ids <= private_records.keys():
            raise ValueError("Every split task needs a public task and separate evaluation record")
        questions = [" ".join(tasks[key].question.split()).casefold() for key in all_ids]
        if len(questions) != len(set(questions)):
            raise ValueError("Normalized duplicate tasks across splits")

    def run_task(self, task_id, snapshot: ExperienceSnapshot, *, mode="evaluate", repeat=0,
                 attribution=False, resume=True):
        ledger = BudgetLedger(self.config.max_model_calls, self.config.max_total_tokens,
                              self.config.max_tool_calls)
        audit = {}
        try:
            return self._run_task(task_id, snapshot, ledger, audit, mode=mode, repeat=repeat,
                                  attribution=attribution, resume=resume)
        except BaseException as exc:
            run_dir = audit.get("run_dir", self.output_dir / "failures" / uuid.uuid4().hex)
            failure = {"task_id": task_id, "experience_hash": digest(snapshot), "mode": mode,
                       "error_type": type(exc).__name__, "error": str(exc), "time": utc_now()}
            for name in ("failed_attempts", "meta_trajectory", "artifact"):
                if hasattr(exc, name):
                    failure[name] = getattr(exc, name)
            if hasattr(exc, "jit_mas_failure"):
                failure["jit_mas_failure"] = exc.jit_mas_failure
            write_json(run_dir / "failure.json", failure)
            audit["run_dir"] = run_dir
            raise
        finally:
            if audit.get("run_dir") and not audit.get("cached"):
                write_json(audit["run_dir"] / "budget.json", ledger.snapshot())

    def _run_task(self, task_id, snapshot, ledger, audit, *, mode, repeat, attribution, resume):
        from .execution import TeamExecutor

        task = self.tasks[task_id]
        def model(role, agent_id, stage):
            return self.models.create(role, agent_id, ledger, stage)

        evaluator = self.evaluator_factory(model("judge", "judge", "evaluation"))
        comparison = {"config": self.config.model_dump(mode="json"), "code": self.code_hash,
                      "evaluator": evaluator.evaluator_version, "task": task.model_dump(mode="json"),
                      "private_hash": digest(self.private_records[task_id]), "repeat": repeat,
                      "manifest": digest(self.manifest), "tools": sorted(self.tools),
                      "attachments": input_fingerprints(task)}
        comparison_hash = digest(comparison)
        run_key = digest({"comparison": comparison_hash, "snapshot": digest(snapshot),
                          "mode": mode, "attribution": attribution})
        cacheable = (self.config.backend == "scripted" or
                     (not task.tools and not any(a.get("unsnapshotted") for a in comparison["attachments"])))
        if not cacheable or not resume:
            run_key = digest({"identity": run_key, "attempt": uuid.uuid4().hex})
        run_dir = self.output_dir / run_key
        audit["run_dir"] = run_dir
        complete_path = run_dir / "complete.json"
        if resume and cacheable and complete_path.is_file():
            cached = json.loads(complete_path.read_text(encoding="utf-8"))
            if cached["run_key"] != run_key:
                raise ValueError("Cached run identity mismatch")
            cached["resumed"] = True
            audit["cached"] = True
            return cached
        if run_dir.exists():
            run_key = digest({"identity": run_key, "retry": uuid.uuid4().hex})
            run_dir = self.output_dir / run_key
            audit["run_dir"] = run_dir
            complete_path = run_dir / "complete.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_manifest.json", {"comparison": comparison,
            "experience_version": snapshot.version, "experience_hash": digest(snapshot),
            "backend": self.config.backend, "mode": mode, "run_key": run_key})
        experience = retrieve(snapshot, task, excluded_task_ids=self.manifest.validation + self.manifest.test)
        if not self.config.persistent_experience:
            experience = []
        analyzer = GlobalAnalyzer(model("global", "global", "inference"),
                                  lambda aid: model("local", aid, "inference"),
                                  max_agents=self.config.max_agents, max_parallel=self.config.max_parallel,
                                  local_rounds=self.config.local_rounds, total_max_calls=self.config.team_max_calls,
                                  explicit_rubrics=self.config.explicit_rubrics)
        if self.config.fixed_team is None:
            planned = analyzer.build(task, experience, local_planning=self.config.local_planning)
            prediction = analyzer.last_prediction
        else:
            if self.config.fixed_team.coverage or any(a.rubric_ids for a in self.config.fixed_team.agents):
                raise ValueError("Fixed MAS ablation uses explicit task-independent responsibilities without rubrics")
            prediction = Prediction(graph=RubricGraph(rubrics=[]), candidates=self.config.fixed_team.agents)
            planned = PlannedTeam(graph=prediction.graph, team=self.config.fixed_team)
        frozen = {"created_at": utc_now(), "experience_version": snapshot.version,
                  "experience_hash": digest(snapshot), "R_global": prediction.graph.model_dump(mode="json"),
                  "R_planned": planned.graph.model_dump(mode="json"), "TeamSpec": planned.team.model_dump(mode="json")}
        frozen["hashes"] = {key: digest(frozen[key]) for key in ("R_global", "R_planned", "TeamSpec")}
        write_json(run_dir / "frozen_plan.json", frozen)
        write_json(run_dir / "planning_calls.json", analyzer.call_records)
        synth = self.synthesizer_factory(model("meta", "meta", "inference"))
        artifact = synth.synthesize(task, planned.graph, planned.team, experiences=experience)
        write_json(run_dir / "harness.json", artifact.to_dict())
        executor = TeamExecutor(lambda aid: model("exec", aid, "inference"), tools=self.tools,
                                ledger=ledger, timeout_seconds=self.config.execution_timeout,
                                unsafe_local=self.config.unsafe_local)
        # JIT's bounded repair consumes runtime exceptions, never quality feedback.
        result = synth.execute_with_repair(executor, task, planned.team, artifact,
                                           rubrics=planned.graph, experiences=experience)
        write_json(run_dir / "harness.json", artifact.to_dict())
        write_json(run_dir / "execution.json", result.full_dict())
        if result.terminated_reason != "final_answer" or result.answer is None:
            write_json(run_dir / "budget.json", ledger.snapshot())
            raise RuntimeError("Team did not submit a final answer; official evaluation was not invoked")
        submission = {"answer": result.answer, "answer_hash": digest(result.answer), "submitted_at": utc_now()}
        write_json(run_dir / "submission.json", submission)
        # Sole transition at which the trusted coordinator opens private evaluation data.
        feedback = convert_feedback(evaluator.evaluate(str(result.answer), ground_truth=task_id,
                                                       private_record=self.private_records[task_id]))
        write_json(run_dir / "evaluation.json", feedback)
        proposals = []
        if attribution and feedback.complete:
            attributor = RubricAttributor(model("global", "global-post", "update"),
                                          lambda aid: model("local", aid, "update"),
                                          max_parallel=self.config.max_parallel,
                                          local_attribution=self.config.local_attribution)
            findings = attributor.attribute(task, prediction.graph, planned.graph, planned.team, result, feedback)
            proposals = attributor.propose(task, findings, snapshot.version, snapshot.experiences)
            write_json(run_dir / "attribution.json", {
                "findings": [f.model_dump(mode="json") for f in findings],
                "alignments": {k: v.model_dump(mode="json") for k, v in attributor.last_alignments.items()},
                "calls": attributor.call_records, "proposals": [p.model_dump(mode="json") for p in proposals]})
        outcome = {"run_key": run_key, "task_id": task_id, "mode": mode, "backend": self.config.backend,
                   "software_test_only": self.config.backend == "scripted", "resumed": False,
                   "experience_version": snapshot.version, "experience_hash": digest(snapshot),
                   "comparison_fingerprint": comparison_hash, "answer_hash": submission["answer_hash"],
                   "submitted_at": submission["submitted_at"], "evaluated_at": utc_now(),
                   "plan_hashes": frozen["hashes"], "evaluation": feedback.model_dump(mode="json"),
                   "budget": ledger.snapshot(), "proposals": [p.model_dump(mode="json") for p in proposals],
                   "run_dir": str(run_dir), "uncontrolled_variation":
                   [] if self.config.backend == "scripted" else ["Provider sampling and live tool evidence are not snapshotted"]}
        stages = outcome["budget"]["by_stage"]
        outcome["costs"] = {"inference": stages.get("inference", {}),
                            "external_evaluation": stages.get("evaluation", {}),
                            "experience_update": stages.get("update", {}),
                            "validation": {"model_calls": 0, "tokens": 0, "tool_calls": 0, "cost": None}}
        write_json(complete_path, outcome)
        return outcome

    def run(self, mode, task_ids=None, *, limit=1, resume=True):
        from .schemas import ChangeProposal

        if mode not in ("evolve", "evaluate", "stream"):
            raise ValueError("Mode must be evolve, evaluate or stream")
        self.code_hash = code_fingerprint()
        allowed = getattr(self.manifest, {"evolve": "evolution", "evaluate": "test", "stream": "stream"}[mode])
        selected = list(allowed[:limit] if task_ids is None else task_ids)
        if not selected or not set(selected) <= set(allowed):
            raise ValueError("Tasks do not belong to the selected mode's split")
        if mode == "stream" and selected != [item for item in allowed if item in selected]:
            raise ValueError("Stream order must match the manifest")
        frozen = self.store.snapshot()
        outcomes = []
        for task_id in selected:
            state = frozen if mode == "evaluate" else self.store.snapshot()
            update = mode != "evaluate" and self.config.persistent_experience
            identity = digest({"task": self.tasks[task_id], "private": self.private_records[task_id],
                               "config": self.config.model_dump(mode="json"), "code": self.code_hash,
                               "manifest": self.manifest.model_dump(mode="json"),
                               "attachments": input_fingerprints(self.tasks[task_id])})
            journal = self.store.task_run(mode, task_id) if update else None
            if journal:
                if not resume or journal["identity"] != identity:
                    raise ValueError("Task was already submitted under this store; use a fresh store for changed policy")
                if journal["status"] == "complete":
                    outcome = journal["outcome"]
                    outcome["resumed"] = True
                    outcomes.append(outcome)
                    continue
                state = self.store.snapshot(journal["baseline_version"])
                if digest(state) != journal["baseline_hash"]:
                    raise ValueError("Task journal baseline changed")
            if update and mode == "stream":
                earlier = allowed[:allowed.index(task_id)]
                if any(not self.store.task_run(mode, key) or
                       self.store.task_run(mode, key)["status"] != "complete" for key in earlier):
                    raise ValueError("Stream tasks must be submitted in manifest order")
            if update and not journal:
                self.store.save_task_run(mode, task_id, identity, state, "started")
            outcome = (journal["outcome"] if journal and journal["outcome"] else
                       self.run_task(task_id, state, mode=mode, attribution=update, resume=resume))
            if update:
                self.store.save_task_run(mode, task_id, identity, state, "submitted", outcome)
            outcome["validations"] = []
            if update:
                # One independently interpretable proposal is validated per task.
                for raw in outcome["proposals"][:1]:
                    proposal = ChangeProposal.model_validate(raw)
                    if proposal.proposal_id in self.store.snapshot().accepted_proposals:
                        break  # Recovery after atomic commit but before the journal completion.
                    self.store.stage(proposal)
                    def rebuild(tid, snapshot, repeat, label):
                        return self.run_task(tid, snapshot, mode="validation", repeat=repeat,
                                             attribution=False, resume=resume)
                    validator = PairedValidator(rebuild, self.manifest, self.config.validation)
                    validation = validator.validate(proposal, state,
                        self.manifest.validation[:self.config.max_validation_tasks])
                    validation = self.store.record_validation(validation)
                    if validation.status == "accepted":
                        self.store.commit(proposal, validation)
                    outcome["validations"].append(validation.model_dump(mode="json"))
                    for pair in validation.pairs:
                        for side in ("baseline", "candidate"):
                            budget = pair.get(side, {}).get("budget", {})
                            for field in ("model_calls", "tokens", "tool_calls"):
                                outcome["costs"]["validation"][field] += budget.get(field, 0)
            outcome["next_experience_version"] = self.store.snapshot().version
            if update:
                self.store.save_task_run(mode, task_id, identity, state, "complete", outcome)
            outcomes.append(outcome)
        write_json(self.output_dir / f"{mode}_report.json", outcomes)
        return outcomes
