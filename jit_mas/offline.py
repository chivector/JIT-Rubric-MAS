"""Explicit synthetic software fixtures. Never imported by the native model provider."""

from __future__ import annotations

import json
import threading

from scripts.models.base import ChatMessage

from .budget import MeteredModel
from .schemas import PublicTask, SplitManifest


FIXTURES = {
    "evolve-comparison": "Compare two message queue designs for a small service.",
    "validation-storage": "Compare two storage designs for an archival service.",
    "validation-recovery": "Compare recovery designs for a batch processor.",
    "test-deployment": "Compare deployment designs for a reporting service.",
    "test-poem": "Write a four-line poem about a clock, using plain language.",
    "stream-first": "Compare caching designs for a catalogue service.",
    "stream-next": "Compare logging designs for a telemetry service.",
}


def fixture_dataset():
    from benchmark.adapter.researchrubrics import split_item

    tasks, private = {}, {}
    for task_id, question in FIXTURES.items():
        public, record = split_item({"sample_id": task_id, "prompt": question,
            "explicit_constraints": ["Keep the response concise."],
            "rubrics": [{"criterion": "PRIVATE_CANARY: state relevant boundary conditions", "weight": 2,
                         "axis": "completeness"}], "reference": "PRIVATE_REFERENCE_CANARY"})
        record["source"] = "synthetic-software-fixture"
        tasks[task_id] = PublicTask(task_id=task_id, question=public["question"],
                                   constraints=public["explicit_constraints"])
        private[task_id] = record
    manifest = SplitManifest(evolution=["evolve-comparison"],
        validation=["validation-storage", "validation-recovery"],
        test=["test-deployment", "test-poem"], stream=["stream-first", "stream-next"])
    return tasks, private, manifest


def _rubric(rid, text, source="inferred", experience_ids=None):
    return {"rubric_id": rid, "requirement": text, "source": source, "importance": 1,
            "confidence": 0.7, "experience_ids": experience_ids or []}


def _agent(aid, rubrics, depends=(), capability="comparison"):
    return {"agent_id": aid, "role": aid, "capability": capability,
            "rubric_ids": rubrics, "responsibilities": ["Develop a concise, task-specific contribution"],
            "depends_on": list(depends), "max_calls": 2, "max_tokens": 1024}


class FixtureModel:
    def __init__(self, provider, role, agent_id):
        self.provider, self.role, self.agent_id = provider, role, agent_id
        self.model_id = "scripted-software-fixture"
        self.counts = {}

    def get_token_counts(self):
        return self.counts

    def __call__(self, messages, **kwargs):
        self.counts = {"input_token_count": len(json.dumps(messages)) // 4, "output_token_count": 80}
        with self.provider.lock:
            self.provider.calls.append({"role": self.role, "agent_id": self.agent_id, "messages": messages})
        if self.role == "judge":
            text = messages[-1]["content"]
            answer = text.split("## Document Content\n", 1)[1].split("\n\n## Rubric Criterion", 1)[0]
            score = float("Boundary conditions:" in answer)
            response = {"score": score, "verdict": "Satisfied" if score else "Not Satisfied",
                        "confidence": 1.0, "reasoning": "Synthetic exact-marker software assertion",
                        "evidence_quotes": ["Boundary conditions:"] if score else [], "missing_elements": []}
        elif self.role == "exec":
            payload = json.loads(messages[1]["content"])
            has_boundary = any(r["rubric_id"] == "r3" for r in payload["predicted_requirements"])
            inherited = payload.get("upstream_artifacts", [])
            has_boundary = has_boundary or "Boundary conditions:" in json.dumps(inherited)
            response = {"answer": "A concise comparison of reliability and operating cost."
                        + (" Boundary conditions: validate workload and recovery assumptions." if has_boundary else ""),
                        "evidence_ids": [row["event_id"] for row in inherited], "checkpoints": {}}
        else:
            payload = json.loads(messages[-1]["content"])
            payload["offline_no_rubrics"] = "ABLATION:" in messages[0]["content"]
            response = self._phase(payload)
        return ChatMessage(role="assistant", content=json.dumps(response))

    def _phase(self, p):
        phase = p["phase"]
        if phase == "predict":
            poem = p["task"]["task_id"] == "test-poem"
            experience = [e for e in p.get("experiences", []) if e["bank"] == "rubric"]
            graph = {"rubrics": [_rubric("r1", "Use the requested form" if poem else "Compare relevant tradeoffs")], "edges": []}
            if experience:
                graph["rubrics"].append(_rubric("r3", "State boundary conditions", "experience",
                                               [experience[0]["experience_id"]]))
            ids = [r["rubric_id"] for r in graph["rubrics"]]
            agents = ([_agent("composer", ids, capability="creative-writing")] if poem else
                      [_agent("analyst", ids), _agent("integrator", ids, ["analyst"], "synthesis")])
            if p.get("offline_no_rubrics"):
                graph = {"rubrics": [], "edges": []}
                for a in agents:
                    a["rubric_ids"] = []
            return {"graph": graph, "candidates": agents}
        if phase == "local_plan":
            a = p["candidate"]
            adds = []
            if a["agent_id"] == "analyst" and p["prediction"]["graph"]["rubrics"]:
                adds = [_rubric("r2", "Discuss failure recovery")]
            return {"agent_id": a["agent_id"], "capability": a["capability"],
                    "rubric_ids": a["rubric_ids"], "additions": adds,
                    "challenge": "Include recovery behavior before synthesis", "max_calls": 2}
        if phase == "reconcile":
            graph = p["prediction"]["graph"]
            for plan in p["local_plans"]:
                graph["rubrics"].extend(plan["additions"])
            ids = [r["rubric_id"] for r in graph["rubrics"]]
            agents = p["prediction"]["candidates"]
            for a in agents:
                a["rubric_ids"] = ids
            if "r3" in ids and len(agents) > 1:
                agents.insert(1, _agent("boundary-check", ["r3"], ["analyst"], "verification"))
                agents[-1]["depends_on"] = ["boundary-check"]
            coverage = {rid: [a["agent_id"] for a in agents if rid in a["rubric_ids"]] for rid in ids}
            return {"graph": graph, "team": {"agents": agents, "synthesizer_id": agents[-1]["agent_id"],
                "coverage": coverage, "primary": {rid: owners[0] for rid, owners in coverage.items()},
                "total_max_calls": 16, "max_parallel": 2, "selection_rationale": "Explicit synthetic fixture allocation"},
                "local_plans": p["local_plans"]}
        if phase == "align":
            matches = []
            if any(r["rubric_id"] == "r3" for r in p["prediction"]["rubrics"]):
                matches = [{"predicted_ids": ["r3"], "evaluated_ids": [p["feedback"]["rubrics"][0]["rubric_id"]],
                            "relation": "equivalent", "confidence": 0.9, "rationale": "Same boundary requirement"}]
            return {"matches": matches}
        if phase in ("attribute_global", "attribute_integrate"):
            rubric = p["feedback"]["rubrics"][0]
            finding = {"finding_id": "boundary-gap", "rubric_ids": [rubric["rubric_id"]],
                       "categories": ["prediction"], "hypothesis": "Boundary assumptions were omitted",
                       "supporting_evidence": ["feedback:" + rubric["rubric_id"], "planning:global"],
                       "alternatives": ["The small fixture is not evidence of real performance"], "uncertainty": 0.4}
            return {"findings": [finding], **({"questions": {}} if phase == "attribute_global" else {})}
        if phase.startswith("attribute_local"):
            return {"findings": []}
        if phase == "propose":
            task_id, version = p["task"]["task_id"], p["base_version"]
            if p["experiences"]:
                return {"proposals": []}
            evidence = p["findings"][0]["supporting_evidence"]
            experience = {"experience_id": "boundary-assumptions", "bank": "rubric",
                          "instruction": "For system comparisons, predict explicit workload and recovery boundary conditions.",
                          "applicability": "Technical comparison tasks", "source_task_ids": [task_id],
                          "evidence": evidence, "task_signals": ["system comparison"]}
            return {"proposals": [{"proposal_id": f"{task_id}-boundary-v{version}", "source_task_id": task_id,
                "base_version": version, "experience": experience, "diff": "Add conditional boundary prediction advice",
                "rationale": "Observed omission in a synthetic source task", "evidence": evidence,
                "expected_benefit": "More explicit assumptions", "risks": ["Overgeneralization"],
                "validation_plan": "Rebuild old/new states on two independent validation tasks"}]}
        raise ValueError(f"Unsupported synthetic phase {phase}")


class FixtureModels:
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def create(self, role, agent_id, ledger, stage):
        if role == "meta":
            from .bridge import ScriptedHarnessModel
            underlying = ScriptedHarnessModel()
        else:
            underlying = FixtureModel(self, role, agent_id)
        return MeteredModel(underlying, ledger, stage, agent_id, 64000 if role == "meta" else 8192)
