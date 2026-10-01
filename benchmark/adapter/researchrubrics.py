"""ResearchRubrics adapter with public tasks and evaluator-owned records.

Scoring and judge prompts are pinned to the official MIT-licensed release.
See docs/jit_mas_researchrubrics.md for provenance and deliberate differences.
This separation is a data contract, not an OS sandbox for generated Python.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .base import BenchmarkAdapter

UPSTREAM_COMMIT = "2dc80e2d4c38ddd80439517c259d93c6954b193f"
DATASET_REVISION = "85de3115053d1453ed612caacf4a405edc1ad756"
PROMPT_DIR = Path(__file__).with_name("researchrubrics_prompts")
SCHEMA_VERSION = "1.0"
QUALITY_AUDIT_VERSION = "researchrubrics-risk-only-v1"


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def split_item(raw: dict) -> tuple[dict, dict]:
    """Allowlist public fields; never pass the raw dataset row to an agent."""
    sample_id = raw.get("sample_id")
    question = raw.get("prompt")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("ResearchRubrics rows require a nonempty sample_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"ResearchRubrics {sample_id} requires a nonempty prompt")
    raw_rubrics = raw.get("rubrics")
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        raise ValueError(f"ResearchRubrics {sample_id} requires nonempty rubrics")
    rubrics = []
    for index, rubric in enumerate(raw_rubrics):
        if not isinstance(rubric, dict) or not isinstance(rubric.get("criterion"), str):
            raise ValueError(f"Invalid criterion at {sample_id}:{index}")
        if not rubric["criterion"].strip():
            raise ValueError(f"Empty criterion at {sample_id}:{index}")
        _finite_number(rubric.get("weight"), "weight")
        entry = copy.deepcopy(rubric)
        # Index is part of identity: identical wording can carry distinct axes.
        digest = hashlib.sha256(rubric["criterion"].encode("utf-8")).hexdigest()[:12]
        entry["rubric_id"] = f"{sample_id}:r{index}:{digest}"
        rubrics.append(entry)
    public = {
        "schema_version": SCHEMA_VERSION,
        "task_id": sample_id,
        "sample_id": sample_id,
        "question_id": sample_id,
        "question": question,
        # Legacy JIT expects an answer slot; this is only an opaque lookup ID.
        "answer": sample_id,
        "domain": str(raw.get("domain", "")),
    }
    for name in ("attachments", "explicit_constraints", "available_tools"):
        if name in raw:
            public[name] = copy.deepcopy(raw[name])
    private = {
        "schema_version": SCHEMA_VERSION,
        "task_id": sample_id,
        "sample_id": sample_id,
        "rubrics": rubrics,
        "source": "human-authored:ScaleAI/researchrubrics",
        "dataset_revision": DATASET_REVISION,
        "raw_record": copy.deepcopy(raw),
    }
    return public, private


def official_compliance_score(rows: list[dict]) -> float:
    """Signed numerator / positive denominator; never clip, including at zero."""
    scores = [_finite_number(row["score"], "score") for row in rows]
    weights = [_finite_number(row["weight"], "weight") for row in rows]
    numerator = sum(score * weight for score, weight in zip(scores, weights))
    denominator = sum(weight for weight in weights if weight > 0)
    return numerator / denominator if denominator > 0 else 0.0


def _judgment_hash(row: dict) -> str:
    fields = ("rubric_id", "criterion", "weight", "verdict", "score", "reasoning",
              "evidence_quotes", "missing_elements", "status", "success", "evaluator_version")
    payload = {key: row.get(key) for key in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _quality_risks(row: dict, answer_hash: str) -> dict:
    """Report observable risks, not semantic judgments or score corrections."""
    count = len(row["evidence_quotes"])
    verified = sum(location.get("verified") is True for location in row["evidence_locations"])
    flags = []
    if row["weight"] < 0:
        flags.append("negative_weight_polarity_requires_review")
    if count > verified:
        flags.append("unverified_evidence_quote")
    if verified == 0:
        flags.append("no_verified_evidence_quote")
    return {
        "version": QUALITY_AUDIT_VERSION, "mode": "risk_only",
        "semantic_consistency": "unassessed", "risk_flags": flags,
        "negative_weight": row["weight"] < 0,
        "quote_count": count, "verified_quote_count": verified,
        "unverified_quote_count": count - verified,
        "answer_sha256": answer_hash, "judgment_sha256": _judgment_hash(row),
        "model_calls": 0, "score_modified": False,
    }


class ResearchRubricsAdapter(BenchmarkAdapter):
    """Frozen independent binary judge, injectable for offline software tests.

    ``judge`` follows JIT's model interface: judge(messages, **kwargs) returns
    a ChatMessage, a JSON string, or a dict. An injected metered model owns its
    own accounting; this adapter reports usage but never debits that ledger.
    """

    def __init__(
        self,
        judge_model: str = "",
        judge_api_base: str = "",
        judge_api_key: str = "",
        judge_max_tokens: int = 4096,
        judge_timeout: float = 120,
        max_attempts: int = 3,
        max_document_chars: int = 400000,
        judge: Callable | None = None,
        judge_id: str = "",
        quality_audit_mode: str = "off",
    ):
        if quality_audit_mode not in {"off", "risk_only"}:
            raise ValueError("quality_audit_mode must be off or risk_only")
        if not 1 <= int(max_attempts) <= 3:
            raise ValueError("ResearchRubrics max_attempts must be between 1 and 3")
        if int(judge_max_tokens) <= 0 or float(judge_timeout) <= 0:
            raise ValueError("Judge token limit and timeout must be positive")
        if int(max_document_chars) <= 0:
            raise ValueError("max_document_chars must be positive")
        self._judge_model = judge_model
        self._judge_api_base = judge_api_base
        self._judge_api_key = judge_api_key
        self._max_tokens = int(judge_max_tokens)
        self._timeout = float(judge_timeout)
        self._max_attempts = int(max_attempts)
        self._max_document_chars = int(max_document_chars)
        self._judge = judge
        self._quality_audit_mode = quality_audit_mode
        self._private_records: dict[str, dict] = {}
        self._system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
        self._user_prompt = (PROMPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
        self._judge_id = judge_id or judge_model or (
            f"injected:{type(judge).__module__}.{type(judge).__qualname__}" if judge else "unconfigured"
        )
        identity = {
            "upstream": UPSTREAM_COMMIT,
            "judge": self._judge_id,
            "endpoint": judge_api_base,
            "prompts": [self._system_prompt, self._user_prompt],
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
            "attempts": self._max_attempts,
            "max_document_chars": self._max_document_chars,
            "adapter": "jit-researchrubrics-v1",
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.evaluator_version = f"researchrubrics:{UPSTREAM_COMMIT}:{digest}"

    def load_dataset(self, path: str) -> list[dict]:
        location = Path(path)
        if location.is_dir():
            location /= "processed_data.jsonl"
        public_rows = []
        private_records = {}
        prompts = set()
        with location.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    public, private = split_item(json.loads(line))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(f"Invalid ResearchRubrics row {line_number}: {exc}") from exc
                sample_id = public["sample_id"]
                prompt_key = " ".join(public["question"].split()).casefold()
                if sample_id in private_records or prompt_key in prompts:
                    raise ValueError(f"Duplicate ResearchRubrics task: {sample_id}")
                prompts.add(prompt_key)
                private_records[sample_id] = private
                public_rows.append(public)
        self._private_records = private_records
        return public_rows

    def private_record(self, task_id: str) -> dict:
        """Coordinator/evaluator-only accessor; never export into agent config."""
        return copy.deepcopy(self._private_records[task_id])

    def format_task(self, item: dict) -> str:
        question = item.get("question", "")
        constraints = item.get("explicit_constraints", [])
        if constraints:
            question += "\n\nExplicit constraints:\n" + json.dumps(constraints, ensure_ascii=False)
        return question

    def get_tools(self) -> list[str]:
        return ["web_search", "crawl_page", "final_answer"]

    def _get_judge(self) -> Callable:
        if self._judge is None:
            if not all((self._judge_model, self._judge_api_base, self._judge_api_key)):
                raise ValueError("ResearchRubrics requires an explicit judge model, endpoint and API key")
            from scripts.models import OpenAIServerModel

            model = OpenAIServerModel(
                model_id=self._judge_model,
                api_base=self._judge_api_base,
                api_key=self._judge_api_key,
                max_attempts=1,
            )
            # The adapter owns retries, including malformed JSON; no nested SDK retries.
            model.client = model.client.with_options(max_retries=0, timeout=self._timeout)
            self._judge = model
        return self._judge

    @staticmethod
    def _response_data(response: Any) -> tuple[dict, str]:
        if isinstance(response, dict):
            data = copy.deepcopy(response)
            return data, json.dumps(data, ensure_ascii=False, allow_nan=False)
        text = response if isinstance(response, str) else response.content
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Judge response must be a JSON object")
        # Retain invalid responses as text, never place NaN/Infinity in artifacts.
        json.dumps(data, allow_nan=False)
        return data, text

    @staticmethod
    def _validate_verdict(data: dict) -> None:
        expected = {"Satisfied": 1.0, "Not Satisfied": 0.0}
        verdict = data.get("verdict")
        if verdict not in expected or _finite_number(data.get("score"), "score") != expected[verdict]:
            raise ValueError("Judge verdict and binary score must agree")
        confidence = _finite_number(data.get("confidence"), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("Judge confidence must be between zero and one")
        if not isinstance(data.get("reasoning"), str):
            raise ValueError("Judge reasoning is required")
        for field in ("evidence_quotes", "missing_elements"):
            entries = data.get(field, [])
            if not isinstance(entries, list) or any(not isinstance(x, str) for x in entries):
                raise ValueError(f"Judge {field} must be a list of strings")

    def _evaluate_rubric(self, prediction: str, rubric: dict, sample_id: str) -> dict:
        started = time.perf_counter()
        row = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "rubric_id": rubric["rubric_id"],
            "rubric_title": rubric["criterion"],
            "criterion": rubric["criterion"],
            "axis": rubric.get("axis", ""),
            "weight": rubric["weight"],
            "verdict": "Error",
            "score": 0.0,
            "confidence": 0.0,
            "reasoning": "",
            "success": False,
            "status": "error",
            "error": None,
            "evidence_quotes": [],
            "evidence_locations": [],
            "missing_elements": [],
            "raw_response": None,
            "raw_result": None,
            "attempts": 0,
            "input_token_count": 0,
            "output_token_count": 0,
            "tokens_used": 0,
            "usage_known": True,
            "cost": None,
            "evaluator_version": self.evaluator_version,
        }
        if len(prediction) > self._max_document_chars:
            row["error"] = "Document exceeds configured whole-document judge limit; not truncated"
            row["duration"] = time.perf_counter() - started
            return row
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._user_prompt.format(
                document_content=prediction,
                rubric_title=rubric["criterion"],
                rubric_category=rubric.get("axis", ""),
                rubric_weight=rubric["weight"],
            )},
        ]
        judge = self._get_judge()
        for attempt in range(self._max_attempts):
            row["attempts"] += 1
            response = None
            try:
                response = judge(
                    copy.deepcopy(messages), max_tokens=self._max_tokens,
                    timeout=self._timeout, response_format={"type": "json_object"},
                )
                data, raw_response = self._response_data(response)
                row["raw_result"], row["raw_response"] = data, raw_response
                self._validate_verdict(data)
                row.update({key: data[key] for key in ("verdict", "score", "confidence", "reasoning")})
                row["evidence_quotes"] = data.get("evidence_quotes", [])
                row["missing_elements"] = data.get("missing_elements", [])
                for quote in row["evidence_quotes"]:
                    start = prediction.find(quote) if quote else -1
                    row["evidence_locations"].append({
                        "quote": quote, "verified": start >= 0,
                        "start": start if start >= 0 else None,
                        "end": start + len(quote) if start >= 0 else None,
                    })
                row.update(success=True, status="ok", error=None)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["reasoning"] = f"Evaluation failed: {row['error']}"
                if response is not None and row["raw_response"] is None:
                    row["raw_response"] = getattr(response, "content", str(response))
            finally:
                # Token counters are per call. Failed calls cannot reuse stale counters.
                try:
                    counts = judge.get_token_counts() if response is not None and hasattr(judge, "get_token_counts") else {}
                    if not isinstance(counts, dict):
                        counts = {}
                except Exception:
                    counts = {}
                for key in ("input_token_count", "output_token_count"):
                    value = counts.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        row["usage_known"] = False
                    else:
                        row[key] += value
            if row["success"]:
                break
            if attempt + 1 < self._max_attempts:
                time.sleep(2 ** attempt)
        row["tokens_used"] = row["input_token_count"] + row["output_token_count"]
        row["duration"] = time.perf_counter() - started
        return row

    def evaluate(self, prediction: str, ground_truth: str = "", **kwargs) -> dict:
        """Return official compliance plus unabridged per-criterion feedback."""
        private = kwargs.get("private_record")
        if private is None:
            sample_id = ground_truth or (kwargs.get("item") or {}).get("sample_id", "")
            private = self.private_record(str(sample_id))
        else:
            private = copy.deepcopy(private)
        sample_id = str(private["sample_id"])
        if ground_truth and str(ground_truth) != sample_id:
            raise ValueError("Private evaluation record does not match task ID")
        if not private.get("rubrics"):
            raise ValueError("Private evaluation record requires nonempty rubrics")
        # Missing credentials are a configuration error, never a fabricated zero result.
        self._get_judge()
        feedback = [self._evaluate_rubric(prediction, rubric, sample_id) for rubric in private["rubrics"]]
        denominator = sum(row["weight"] for row in feedback if row["weight"] > 0)
        numerator = sum(row["weight"] * row["score"] for row in feedback)
        score = official_compliance_score(feedback)
        complete = all(row["success"] for row in feedback)
        result = {
            "schema_version": SCHEMA_VERSION,
            "task_id": sample_id,
            "sample_id": sample_id,
            "score": score,
            "official_compliance": score,
            "metric": "researchrubrics_signed_compliance",
            "numerator": numerator,
            "denominator": denominator,
            "zero_denominator": denominator == 0,
            "complete": complete,
            "status": "complete" if complete else "incomplete",
            "failed_count": sum(not row["success"] for row in feedback),
            # This compatibility flag is a diagnostic, not another official metric.
            "is_pass": complete and denominator > 0 and score >= 1.0,
            "feedback": feedback,
            "evaluator_version": self.evaluator_version,
            "evaluator_source": private.get("source", "human-authored:ScaleAI/researchrubrics"),
            "upstream_commit": UPSTREAM_COMMIT,
            "judge_model": self._judge_id,
            "model_calls": sum(row["attempts"] for row in feedback),
            "input_token_count": sum(row["input_token_count"] for row in feedback),
            "output_token_count": sum(row["output_token_count"] for row in feedback),
            "usage_known": all(row["usage_known"] for row in feedback),
            "cost": None,
        }
        if self._quality_audit_mode == "risk_only":
            answer_hash = hashlib.sha256(prediction.encode("utf-8")).hexdigest()
            for row in feedback:
                row["quality_audit"] = _quality_risks(row, answer_hash)
            audits = [row["quality_audit"] for row in feedback]
            counts = Counter(flag for audit in audits for flag in audit["risk_flags"])
            result["quality_audit"] = {
                "version": QUALITY_AUDIT_VERSION, "mode": "risk_only",
                "semantic_consistency": "unassessed", "screened_rows": len(audits),
                "flagged_rows": sum(bool(audit["risk_flags"]) for audit in audits),
                "risk_counts": dict(sorted(counts.items())),
                "negative_weight_rows": sum(audit["negative_weight"] for audit in audits),
                **{key: sum(audit[key] for audit in audits) for key in
                   ("quote_count", "verified_quote_count", "unverified_quote_count")},
                "answer_sha256": answer_hash, "official_evaluator_version": self.evaluator_version,
                "model_calls": 0, "score_modified": False,
                "note": "Risk flags are not proven contradictions, fabricated evidence, or a new score. "
                        "Semantic consistency remains unassessed; flags alone must not change validation decisions.",
            }
        return result
