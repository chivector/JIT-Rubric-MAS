"""Offline baseline and judge-envelope tests; no network calls."""

import copy
import json

import pytest

from benchmark.adapter.researchrubrics import ResearchRubricsAdapter
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.schemas import PublicTask, digest
from scripts.mas_baseline_methods import JudgeEnvelopeModel, direct_messages, fixed_team, run_direct
from scripts.models.base import ChatMessage


def verdict():
    return {"verdict": "Satisfied", "score": 1, "confidence": 0.8,
            "reasoning": "Preserved exactly.", "evidence_quotes": ["Final artifact."], "missing_elements": []}


class Model:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply

    def get_token_counts(self):
        return {"input_token_count": 5, "output_token_count": 3}


def test_fixed_team_is_task_independent_parallel_and_one_call():
    team = fixed_team()
    assert [a.agent_id for a in team.agents] == ["analyst", "evidence", "writer"]
    assert [a.depends_on for a in team.agents] == [[], [], ["analyst", "evidence"]]
    assert team.max_parallel == 2 and team.total_max_calls == 3
    assert team.synthesizer_id == "writer"
    assert all(a.max_calls == 1 and a.max_tokens == 8192 for a in team.agents)
    assert not team.coverage and not team.primary and not team.reviewers
    assert all(not a.rubric_ids and not a.tools and not a.checkpoints for a in team.agents)
    assert fixed_team() == team and fixed_team() is not team


def test_direct_messages_are_public_whitelist_only():
    task = {"task_id": "synthetic", "question": "Explain a fictional box.", "constraints": ["Be concise"],
            "private_record": {"rubrics": "hidden-canary"}, "rubrics": ["hidden-canary"],
            "api_key": "credential-canary", "answer": "gold-canary", "attachments": []}
    messages = direct_messages(task)
    assert len(messages) == 2
    payload = json.loads(messages[-1]["content"])
    assert set(payload) == {"task_id", "question", "constraints", "attachments", "tools", "capabilities"}
    assert all(canary not in json.dumps(messages) for canary in ("hidden-canary", "credential-canary", "gold-canary"))
    assert "one response" in messages[0]["content"] and "No browsing" in messages[0]["content"]


def test_strict_fence_preserves_inner_bytes_raw_message_and_usage():
    body = '{ "score": 1, "reasoning": "unchanged", "nested": {"k": 2} }'
    original = ChatMessage(role="assistant", content=" \n```json\r\n" + body + "\r\n```\n",
                           reasoning_content="opaque", raw=object())
    raw = Model(original)
    ledger = BudgetLedger()
    wrapper = JudgeEnvelopeModel(MeteredModel(raw, ledger, "evaluation", "judge", 4096))
    result = wrapper([{"role": "user", "content": "judge"}])
    assert result.content == body
    assert result is not original and original.content.startswith(" \n```json")
    assert result.reasoning_content == original.reasoning_content
    assert wrapper.calls[0]["original_content"] == original.content
    assert wrapper.calls[0]["normalized_content"] == body and wrapper.calls[0]["normalized"]
    assert wrapper.get_token_counts() == raw.get_token_counts()
    assert ledger.snapshot()["model_calls"] == len(raw.calls) == 1
    assert ledger.snapshot()["tokens"] == 8
    assert "raw" not in wrapper.calls[0]


@pytest.mark.parametrize("content", [
    '```json\n{"score":1}\n```\nExplanation',
    'Explanation\n```json\n{"score":1}\n```',
    '```json\n[1,2]\n```', '```json\n"text"\n```',
    '```json\n{"unfinished":\n```', '```json\n{"score":NaN}\n```',
    '```json\n{"score":1}\n```\n```json\n{}\n```',
    '```\n{"score":1}\n```', '```JSON\n{"score":1}\n```',
    '{"score":1}', 'plain response',
])
def test_unapproved_or_invalid_envelopes_are_passed_unchanged_without_guessing(content):
    response = ChatMessage(role="assistant", content=content)
    raw = Model(response)
    wrapper = JudgeEnvelopeModel(raw)
    assert wrapper([]) is response
    assert wrapper.calls == [{"version": "strict-single-json-fence-v1", "original_content": content,
                              "normalized_content": content, "normalized": False}]
    assert len(raw.calls) == 1


def test_string_fence_supported_but_invalid_still_rejected_by_official_adapter():
    private = {"task_id": "t", "sample_id": "t", "rubrics": [
        {"rubric_id": "r", "criterion": "Be accurate", "weight": 1}]}
    good = Model("```json\n" + json.dumps(verdict()) + "\n```")
    raw = ResearchRubricsAdapter(judge=JudgeEnvelopeModel(good), judge_id="fixture", max_attempts=1).evaluate(
        "Final artifact.", ground_truth="t", private_record=private)
    assert raw["complete"] and raw["score"] == 1
    invalid = Model("```json\n" + json.dumps(verdict()) + "\n```\nExplanation")
    raw = ResearchRubricsAdapter(judge=JudgeEnvelopeModel(invalid), judge_id="fixture", max_attempts=1).evaluate(
        "Final artifact.", ground_truth="t", private_record=private)
    assert not raw["complete"] and len(invalid.calls) == 1


@pytest.fixture
def direct_case(tmp_path):
    task = PublicTask(task_id="synthetic", question="Explain the fictional box.", constraints=["Be concise"])
    private = {"task_id": task.task_id, "sample_id": task.task_id, "rubrics": [
        {"rubric_id": "hidden-id", "criterion": "hidden-criterion-canary", "weight": 1}]}
    session = BudgetLedger(max_calls=20, max_tokens=100000)

    class Provider:
        def __init__(self):
            self.exec = Model(ChatMessage(role="assistant", content="Final artifact.", raw=object()))
            self.judge = Model(ChatMessage(role="assistant", content=json.dumps(verdict())))
            self.created = []

        def create(self, role, agent_id, ledger, stage):
            self.created.append((role, agent_id, stage))
            if role == "judge":
                assert (tmp_path / "execution.json").is_file()
                submitted = json.loads((tmp_path / "submission.json").read_text())
                assert submitted["answer"] == "Final artifact."
                assert submitted["answer_hash"] == digest(submitted["answer"])
            model = self.exec if role == "exec" else self.judge
            return MeteredModel(MeteredModel(model, session, stage, agent_id, 8192), ledger, stage, agent_id, 8192)

    provider = Provider()

    def evaluator(judge):
        assert (tmp_path / "submission.json").is_file()
        return ResearchRubricsAdapter(judge=JudgeEnvelopeModel(judge), judge_id="fixture", max_attempts=1)

    return task, private, provider, evaluator, tmp_path, {"max_calls": 20, "max_tokens": 100000}, session


def test_direct_one_call_then_evaluation_after_immutable_submission(direct_case):
    *args, session = direct_case
    result = run_direct(*args)
    provider, output = args[2], args[4]
    assert result["status"] == "completed" and result["evaluation"]["complete"]
    assert len(provider.exec.calls) == len(provider.judge.calls) == 1
    assert "hidden-criterion-canary" not in json.dumps(provider.exec.calls)
    assert "hidden-criterion-canary" in json.dumps(provider.judge.calls)
    assert result["budget"]["model_calls"] == session.snapshot()["model_calls"] == 2
    assert result["budget"]["tokens"] == session.snapshot()["tokens"] == 16
    execution = json.loads((output / "execution.json").read_text())
    assert execution["answer"] == "Final artifact."
    assert len(execution["trajectory"]) == 1 and not execution["sub_runs"]
    assert "raw" not in execution["trajectory"][0]["model_output_messages"]
    trace = json.loads((output / "call_trace.json").read_text())
    assert len(trace["execution_calls"]) == len(trace["evaluator_calls"]) == 1
    assert json.loads((output / "evaluation.json").read_text())["raw"]["submission_answer_hash"] == result["answer_hash"]
    with pytest.raises(ValueError, match="no reruns"):
        run_direct(*args)
    assert len(provider.exec.calls) == 1


def test_direct_model_failure_persists_budget_and_never_creates_judge(direct_case):
    *args, _ = direct_case
    provider, output = args[2], args[4]
    provider.exec.reply = RuntimeError("safe transport failure")
    with pytest.raises(RuntimeError, match="safe transport"):
        run_direct(*args)
    assert len(provider.exec.calls) == 1 and not provider.judge.calls
    assert provider.created == [("exec", "direct", "inference")]
    assert not (output / "submission.json").exists()
    assert json.loads((output / "budget.json").read_text())["model_calls"] == 1
    assert json.loads((output / "failure.json").read_text())["stage"] == "inference"
    assert (output / "call_trace.json").is_file() and (output / "execution.json").is_file()


def test_direct_empty_answer_not_repaired_or_judged(direct_case):
    *args, _ = direct_case
    args[2].exec.reply = ChatMessage(role="assistant", content="", reasoning_content="not a final answer")
    with pytest.raises(ValueError, match="nonempty final artifact"):
        run_direct(*args)
    assert len(args[2].exec.calls) == 1 and not args[2].judge.calls


def test_evaluator_failure_keeps_original_submission_and_execution(direct_case):
    *args, _ = direct_case
    output = args[4]

    def fail(_judge):
        assert (output / "submission.json").is_file()
        raise RuntimeError("evaluator setup failure")

    args[3] = fail
    with pytest.raises(RuntimeError, match="evaluator setup"):
        run_direct(*args)
    execution = json.loads((output / "execution.json").read_text())
    assert execution["terminated_reason"] == "final_answer" and execution["trajectory"][0]["error"] is None
    assert json.loads((output / "submission.json").read_text())["answer"] == "Final artifact."
    assert json.loads((output / "failure.json").read_text())["stage"] == "evaluation"
    assert json.loads((output / "budget.json").read_text())["model_calls"] == 1


def test_judge_transport_failure_remains_incomplete_with_its_charge(direct_case):
    *args, session = direct_case
    args[2].judge.reply = RuntimeError("safe judge failure")
    result = run_direct(*args)
    assert result["status"] == "incomplete" and not result["evaluation"]["complete"]
    assert result["budget"]["model_calls"] == session.snapshot()["model_calls"] == 2
    assert result["budget"]["reserved_tokens"] == 0
    assert len(args[2].exec.calls) == len(args[2].judge.calls) == 1
