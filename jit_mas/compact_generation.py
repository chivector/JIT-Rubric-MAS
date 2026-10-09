"""One-call public-task generation with optional strict construction schemas.

This module deliberately has no baseline or refinement integration.  It accepts
only the public task projection, generates once, and leaves scoring to the
fixed benchmark checker owned by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .public_literal_constraints import public_literal_plan
from .public_literal_slots import literal_slots_plan, render as render_literal
from .public_numeric_slots import (numeric_plan, public_construction_conflict,
                                   render as render_numeric)
from .public_numeric_template import template_plan, render as render_numeric_template
from .public_word_slots import position_plan, render as render_position
from .schemas import PublicTask, digest


PROMPT_VERSION = "compact-public-one-call-v5"
SYSTEM_PROMPT = (
    "Complete the public task in one response. Privately extract every explicit "
    "constraint and check the final artifact before emitting it. Output only the "
    "requested artifact, with no checklist, process notes, or editing preface. "
    "Prefer the shortest answer that satisfies all requirements. Each sentence must "
    "directly advance the requested content. Avoid repetition, filler and invented "
    "personal background, qualifications or achievements. Letter limits count every "
    "character in the complete answer, including titles: use short alternative words "
    "that avoid restricted letters. For numbered sections, use the requested marker "
    "with Arabic indices unless the task explicitly requests another numbering style."
)


def safe_error(exc: Exception) -> dict[str, str]:
    """Remove credential environment values from durable provider error text."""
    message = str(exc)
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(marker in name.upper()
                                            for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    return {"type": type(exc).__name__, "message": message}


def task_public_payload(task: PublicTask) -> dict[str, Any]:
    """Return the only task fields that may enter the generation prompt."""
    if not isinstance(task, PublicTask):
        raise TypeError("compact generation requires a PublicTask")
    return {"question": task.question, "constraints": list(task.constraints)}


def literal_ledger(task: PublicTask) -> list[dict[str, Any]]:
    """Expose only finite literal operator/count data to the model prompt."""
    plan = public_literal_plan(task)
    if plan is None:
        return []
    return [{"literal": rule.literal, "operator": rule.operator,
             "count": rule.minimum} for rule in plan.rules]


def literal_task_payload(task: PublicTask, plan) -> dict[str, Any]:
    """Replace compiled directives with their already supplied slot protocol."""
    payload = task_public_payload(task)
    sources = {"question": payload["question"],
               **{f"constraints[{index}]": value
                  for index, value in enumerate(payload["constraints"])}}
    for source, text in sources.items():
        spans = [span for span in plan.source_plan.instruction_spans if span.source == source]
        for span in sorted(spans, key=lambda value: value.start, reverse=True):
            text = text[:span.start] + "Use the supplied word insertion protocol" + text[span.end:]
        if source == "question":
            payload["question"] = text
        else:
            payload["constraints"][int(source.removeprefix("constraints[").removesuffix("]"))] = text
    return payload


def build_messages(task: PublicTask, *, construction: str = "ordinary", plan=None):
    """Build a compact public-only request and optional construction instruction."""
    if construction not in {"ordinary", "numeric", "numeric_template", "position", "literal"}:
        raise ValueError("unknown compact construction")
    system = SYSTEM_PROMPT
    response_format = None
    if construction == "numeric":
        from .public_numeric_slots import prepare_prompt_instruction
        system += "\n" + prepare_prompt_instruction(plan)
        system += "\nReturn one complete JSON object without process notes, then stop."
        response_format = plan.response_format()
    elif construction == "numeric_template":
        markers = [f"<{key}>" for key in plan.marker_keys]
        system += (
            "\nReturn exactly one JSON object with answer_template and numeric_values. "
            "answer_template is the complete final artifact, with every numeric literal "
            "replaced by one of these placeholders: " + ", ".join(markers) + ". "
            "Use each listed placeholder exactly once; do not include decimal digits "
            "anywhere in answer_template. numeric_values contains exactly the schema's "
            "keys, each mapped to one canonical ASCII integer string, without leading "
            "zeros. The renderer replaces each placeholder with that integer. Use "
            "necessary dates, quantities or list labels; the integers must make sense "
            "in the requested content. Keep the prose concise and do not repeat content "
            "to fill placeholders. Stop immediately after the complete root object."
        )
        if plan.conjunction_count is not None:
            system += (f" The artifact must use at least {plan.conjunction_count} distinct "
                       "coordinating conjunctions from and, but, or, so, yet, for, nor, "
                       "in grammatical clauses.")
        response_format = plan.response_format()
    elif construction == "position":
        from .public_word_slots import prepare_prompt_instruction
        system += "\n" + prepare_prompt_instruction(plan)
        system += ("\nKeep each preceding sentence short. following_sentences must be [] "
                   "unless additional sentences are necessary to complete the public task.")
        response_format = plan.response_format()
    elif construction == "literal":
        from .public_literal_slots import prepare_prompt_instruction
        system += "\n" + prepare_prompt_instruction(plan)
        response_format = plan.response_format()
    ledger = literal_ledger(task)
    if ledger and construction != "literal":
        system += ("\nPrivate final check: satisfy this literal ledger over the complete "
                   "answer; it is a lower-bound or exact requirement as labeled. "
                   + json.dumps(ledger, ensure_ascii=False, separators=(",", ":")))
    payload = literal_task_payload(task, plan) if construction == "literal" else task_public_payload(task)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload,
                                                          ensure_ascii=False,
                                                          separators=(",", ":"))}]
    return messages, response_format


def choose_construction(task: PublicTask, *, numeric: bool = False,
                        positional: bool = False, literal: bool = False,
                        numeric_layout: str = "array"):
    """Choose at most one strict public schema; unsupported rules fall back."""
    if numeric_layout not in {"array", "template"}:
        raise ValueError("numeric layout must be array or template")
    if (numeric or positional) and public_construction_conflict(task):
        return "ordinary", None
    if numeric:
        plan = template_plan(task) if numeric_layout == "template" else numeric_plan(task)
        if plan is not None:
            return ("numeric_template" if numeric_layout == "template" else "numeric"), plan
    if positional:
        plan = position_plan(task)
        if plan is not None:
            return "position", plan
    if literal:
        plan = literal_slots_plan(task)
        if plan is not None:
            return "literal", plan
    return "ordinary", None


def _json_content(response):
    content = response if isinstance(response, str) else getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty model response")
    return content


def generate_one(task: PublicTask, model, ledger: BudgetLedger, *, numeric=False,
                 positional=False, literal=False, numeric_layout="array",
                 max_tokens=4096) -> dict[str, Any]:
    """Generate one artifact, with no retry or local answer mutation."""
    construction, plan = choose_construction(task, numeric=numeric, positional=positional,
                                             literal=literal, numeric_layout=numeric_layout)
    messages, response_format = build_messages(task, construction=construction, plan=plan)
    answer = None
    raw_response = None
    error = None
    try:
        kwargs = {"max_tokens": max_tokens}
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = model(messages, **kwargs)
        raw_response = response if isinstance(response, str) else getattr(response, "content", None)
        if construction == "numeric":
            answer = render_numeric(plan.model.model_validate_json(_json_content(response)))
        elif construction == "numeric_template":
            answer = render_numeric_template(plan.model.model_validate_json(_json_content(response)))
        elif construction == "position":
            answer = render_position(plan.model.model_validate_json(_json_content(response)))
        elif construction == "literal":
            answer = render_literal(plan.model.model_validate_json(_json_content(response)))
        else:
            answer = _json_content(response)
    except Exception as exc:
        error = safe_error(exc)
    request_metadata = getattr(model, "last_request_metadata", {})
    if request_metadata.get("finish_reason") == "length":
        error = {"type": "IncompleteGeneration",
                 "message": "Provider truncated the response at its output limit"}
    elif request_metadata.get("finish_reason") in {"content_filter", "tool_calls", "function_call"}:
        error = {"type": "IncompleteGeneration",
                 "message": "Provider did not finish a final text artifact"}
    return {"construction": construction, "answer": answer,
            "raw_response": raw_response,
            "raw_response_hash": digest(raw_response) if raw_response is not None else None,
            "answer_hash": digest(answer) if answer is not None else None,
            "error": error, "budget": ledger.snapshot(),
            "complete": answer is not None and error is None,
            "finish_reason": request_metadata.get("finish_reason"),
            "prompt_hash": digest(messages), "prompt_version": PROMPT_VERSION}


def code_identity(path: Path | None = None) -> str:
    """Hash this module (or an explicitly supplied runner) for run receipts."""
    location = Path(path) if path is not None else Path(__file__)
    return hashlib.sha256(location.read_bytes()).hexdigest()


def frozen_identity(*, config_bytes: bytes, dataset_sha256: str, checker_identity: Any,
                    prompt_hash: str, runner_sha256: str | None = None) -> dict[str, Any]:
    return {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "dataset_sha256": dataset_sha256,
            "checker_identity": checker_identity,
            "prompt_hash": prompt_hash,
            "runner_sha256": runner_sha256 or code_identity(),
            "prompt_version": PROMPT_VERSION}


def make_ledger(max_tokens: int, *, timeout: float | None = 900) -> BudgetLedger:
    return BudgetLedger(max_calls=1, max_tokens=max_tokens, max_tool_calls=0,
                        timeout_seconds=timeout)
