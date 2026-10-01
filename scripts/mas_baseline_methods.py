"""Small comparison helpers; no model construction, hidden retries, or training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time

from jit_mas.budget import BudgetLedger
from jit_mas.pipeline import convert_feedback, write_json
from jit_mas.schemas import AgentSpec, PublicTask, TeamSpec, digest, utc_now
from scripts.kernel.types import RunResult, StepRecord


JUDGE_ENVELOPE_VERSION = "strict-single-json-fence-v1"
_PUBLIC_FIELDS = ("task_id", "question", "attachments", "constraints", "tools", "capabilities")
_JSON_FENCE = re.compile(r"\s*```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*\Z")


def fixed_team() -> TeamSpec:
    """Task-independent team only; the comparison still uses native JIT generation."""
    return TeamSpec(agents=[
        AgentSpec(agent_id="analyst", role="Analyst", capability="requirements and reasoning",
            responsibilities=["Identify public-task requirements, assumptions and uncertainties.",
                              "Publish a concise summary and ledger requirements/outline; do not draft the final answer."],
            max_calls=1, max_tokens=8192),
        AgentSpec(agent_id="evidence", role="Evidence Contributor", capability="evidence and provenance",
            responsibilities=["Identify relevant evidence from supplied public information and state source limitations.",
                              "Publish ledger evidence_spans/source_references; never invent observations or sources."],
            max_calls=1, max_tokens=8192),
        AgentSpec(agent_id="writer", role="Writer", capability="final synthesis",
            responsibilities=["Read the frozen shared ledger and reconcile supported contributions with the public task.",
                              "Write the complete final deliverable once, retaining important uncertainty."],
            depends_on=["analyst", "evidence"], max_calls=1, max_tokens=8192),
    ], synthesizer_id="writer", total_max_calls=3, max_parallel=2,
        selection_rationale="Fixed task-independent Analyst/Evidence/Writer baseline with native task-conditioned JIT harness generation.")


def _public_task(task):
    raw = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    return PublicTask.model_validate({key: raw[key] for key in _PUBLIC_FIELDS if key in raw})


def direct_messages(task):
    """Do not serialize arbitrary dataset/runner fields into the direct model."""
    public = _public_task(task)
    return [{"role": "system", "content": (
        "Complete the supplied public task accurately and fully in one response. Respect its format, audience "
        "and constraints. Check reasoning, calculations and internal consistency before answering, and state "
        "important uncertainty or missing information honestly. No browsing or tools are available: do not "
        "pretend to have searched, opened attachments, performed measurements, or verified unavailable sources. "
        "Return only the final requested artifact, without planning notes, internal checks, an editing preface, "
        "or a request for another turn.")},
        {"role": "user", "content": json.dumps(public.model_dump(mode="json", include=set(_PUBLIC_FIELDS)),
                                                   ensure_ascii=False)}]


def _reject_constant(value):
    raise ValueError("Non-JSON constant: " + value)


class JudgeEnvelopeModel:
    """Remove only a complete JSON-object fence; preserve all judgment bytes inside."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, messages, **kwargs):
        response = self.model(messages, **kwargs)
        original = response if isinstance(response, (str, dict)) else getattr(response, "content", None)
        normalized = original
        if isinstance(original, str):
            match = _JSON_FENCE.fullmatch(original)
            if match:
                try:
                    parsed = json.loads(match["body"], parse_constant=_reject_constant)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    normalized = match["body"]
        changed = normalized != original
        self.calls.append({"version": JUDGE_ENVELOPE_VERSION, "original_content": copy.deepcopy(original),
                           "normalized_content": copy.deepcopy(normalized), "normalized": changed})
        if not changed:
            return response
        if isinstance(response, str):
            return normalized
        result = copy.copy(response)
        result.content = normalized
        return result

    def __getattr__(self, name):
        return getattr(self.model, name)


def _response_fields(response):
    if isinstance(response, str):
        return response
    # The raw transport/client object is deliberately excluded from saved traces.
    result = {key: getattr(response, key) for key in ("role", "content", "reasoning_content")
              if getattr(response, key, None) is not None}
    if getattr(response, "tool_calls", None):
        result["tool_calls"] = [{"id": getattr(call, "id", None), "type": getattr(call, "type", None),
            "function": {"name": getattr(getattr(call, "function", None), "name", None),
                         "arguments": getattr(getattr(call, "function", None), "arguments", None)}}
                                for call in response.tool_calls]
    return result


def run_direct(task, private, models, evaluator_factory, outdir, limits, *, defer_evaluation=False):
    """One metered answer, immutable submission, then the shared official evaluator.

    limits contains max_calls/max_tokens and optional output_tokens (default 8192).
    The provider must return metered, credential-safe models, as LiveModels does.
    """
    public = _public_task(task)
    max_calls, max_tokens = limits["max_calls"], limits["max_tokens"]
    output_tokens = limits.get("output_tokens", 8192)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (max_calls, max_tokens, output_tokens)) or output_tokens > 8192:
        raise ValueError("Positive integer limits are required; direct output is capped at 8192 tokens")
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Direct comparison requires an empty output directory; no reruns")
    ledger = BudgetLedger(max_calls=max_calls, max_tokens=max_tokens, max_tool_calls=0)
    step = StepRecord(step_number=1, model_input_messages=direct_messages(public), start_time=time.time())
    execution = RunResult(trajectory=[step], terminated_reason="error", metadata={"method": "direct_single"})
    trace = {"task_id": public.task_id, "method": "direct_single", "execution_calls": [], "evaluator_calls": []}
    outcome = None
    execution_saved = False
    stage = "inference"
    try:
        model = models.create("exec", "direct", ledger, "inference")
        response = model(copy.deepcopy(step.model_input_messages), max_tokens=output_tokens)
        step.model_output_messages = _response_fields(response)
        answer = response if isinstance(response, str) else getattr(response, "content", None)
        if not isinstance(answer, str) or not answer.strip() or getattr(response, "tool_calls", None):
            raise ValueError("Direct model must return a nonempty final artifact without tool requests")
        records = [record for record in ledger.snapshot()["records"] if record.get("stage") == "inference"]
        if len(records) != 1 or records[0].get("kind") != "model":
            raise RuntimeError("Direct execution requires exactly one metered model call")
        step.input_token_count, step.output_token_count = records[0]["input_tokens"], records[0]["output_tokens"]
        step.total_token_count = step.input_token_count + step.output_token_count
        step.action_output, execution.answer = answer, answer
        execution.terminated_reason = "final_answer"
        step.end_time = time.time()
        step.duration = step.end_time - step.start_time
        write_json(output / "execution.json", execution.full_dict())
        execution_saved = True
        submission = {"answer": answer, "answer_hash": digest(answer), "submitted_at": utc_now()}
        write_json(output / "submission.json", submission)
        if defer_evaluation:
            outcome = {"task_id": public.task_id, "method": "direct_single", "mode": "evaluate",
                "status": "submitted_unscored", "run_dir": str(output.resolve()),
                "answer_hash": submission["answer_hash"], "submitted_at": submission["submitted_at"],
                "evaluation": None, "experience_version": 0, "proposals": [], "experience_updates": []}
            return outcome
        stage = "evaluation"
        judge = models.create("judge", "judge", ledger, "evaluation")
        evaluator = evaluator_factory(judge)
        raw = evaluator.evaluate(answer, ground_truth=public.task_id, private_record=private)
        raw["submission_answer_hash"] = submission["answer_hash"]
        feedback = convert_feedback(raw)
        write_json(output / "evaluation.json", feedback)
        outcome = {"task_id": public.task_id, "method": "direct_single", "mode": "evaluate",
            "status": "completed" if feedback.complete else "incomplete", "run_dir": str(output.resolve()),
            "answer_hash": submission["answer_hash"], "submitted_at": submission["submitted_at"],
            "evaluated_at": utc_now(), "evaluation": feedback.model_dump(mode="json"),
            "experience_version": 0, "proposals": [], "experience_updates": []}
    except BaseException as exc:
        if not execution_saved:
            step.error = exc
        write_json(output / "failure.json", {"task_id": public.task_id, "method": "direct_single",
            "stage": stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if not execution_saved:
            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            write_json(output / "execution.json", execution.full_dict())
        budget = ledger.snapshot()
        trace["execution_calls"] = [row for row in budget["records"] if row.get("kind") == "model" and row["stage"] == "inference"]
        trace["evaluator_calls"] = [row for row in budget["records"] if row.get("kind") == "model" and row["stage"] == "evaluation"]
        write_json(output / "call_trace.json", trace)
        write_json(output / "budget.json", budget)
        if outcome is not None:
            outcome["budget"] = budget
            write_json(output / "complete.json", outcome)
    return outcome
