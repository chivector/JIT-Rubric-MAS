import json
from types import SimpleNamespace

import scripts.run_targeted_compact_experiment as runner
from jit_mas.schemas import PublicTask, digest


class DummyLedger:
    def __init__(self, *args, **kwargs):
        self.records = []

    def snapshot(self):
        return {"model_calls": len(self.records), "tokens": sum(row["tokens"] for row in self.records),
                "tool_calls": 0, "cost": None, "records": self.records, "by_stage": {}}


def config():
    model = SimpleNamespace(max_tokens=128, model="model", endpoint="https://example.test", timeout=20)
    return SimpleNamespace(models={"exec": model, "judge": model}, max_model_calls=10,
        max_total_tokens=1000, task_timeout=30, judge_parallel=1, check_native=lambda: None,
        model_dump=lambda **kwargs: {})


def setup(monkeypatch, *, fail_first=False, fail_evaluation=False, finish_reason="stop"):
    created = []

    class Model:
        def __init__(self, role, ledger, stage):
            self.role, self.ledger, self.stage = role, ledger, stage
            self.calls = []
            self.last_request_metadata = {"finish_reason": finish_reason}
            self.client = SimpleNamespace(close=lambda: None)

        def __call__(self, messages, **kwargs):
            self.ledger.records.append({"stage": self.stage, "kind": "model", "tokens": 20})
            self.calls.append(messages)
            if fail_first and self.role == "exec" and len(created) == 1:
                raise RuntimeError("synthetic provider failure")
            return "artifact"

    class Provider:
        def __init__(self, value):
            pass

        def create(self, role, agent_id, ledger, stage):
            created.append((role, stage))
            return Model(role, ledger, stage)

    class Evaluator:
        def __init__(self, judge):
            self.judge = judge

        def evaluate(self, answer, ground_truth, private_record):
            self.judge([{"role": "user", "content": answer}])
            if fail_evaluation:
                raise RuntimeError("synthetic evaluator failure")
            return {"task_id": ground_truth, "complete": True, "score": 1,
                    "evaluator_version": "fixture", "feedback": [{"rubric_id": "requirement",
                        "criterion": "Complete artifact", "weight": 1, "score": 1, "status": "ok",
                        "verdict": "Satisfied", "reasoning": "Artifact satisfies the requirement."}]}

    tasks = {identity: PublicTask(task_id=identity, question="Write a concise answer.")
             for identity in ["task:1", "task:2"]}
    dataset = SimpleNamespace(dataset_sha256="dataset", private_records={identity: {} for identity in tasks},
                              evaluator=lambda judge, **kwargs: Evaluator(judge))
    monkeypatch.setattr(runner, "NativeModels", Provider)
    monkeypatch.setattr(runner, "BudgetLedger", DummyLedger)
    return created, {"benchmark": "deepsearchqa", "dataset": dataset,
                     "tasks": tasks, "task_ids": list(tasks)}


def test_deterministic_answer_skips_generation_and_retains_native_judging(monkeypatch, tmp_path):
    created, material = setup(monkeypatch)
    material["task_ids"] = ["task:1"]
    monkeypatch.setattr(runner, "_messages", lambda task, benchmark, private_record=None: (([], None), "act_numeric", {"answer": "computed"}))
    rows = runner.run(config(), [material], tmp_path / "out")
    assert created == [("judge", "evaluation")]
    assert rows[0]["status"] == "complete" and rows[0]["budget"]["model_calls"] == 1
    assert rows[0]["budget"]["tokens"] == 20 and rows[0]["score_result"]["score"] == 1


def test_generation_failure_keeps_budget_and_continues_next_task(monkeypatch, tmp_path):
    created, material = setup(monkeypatch, fail_first=True)
    monkeypatch.setattr(runner, "_messages", lambda task, benchmark, private_record=None: (([], None), "ordinary", None))
    rows = runner.run(config(), [material], tmp_path / "out")
    assert rows[0]["status"] == "failed" and rows[0]["budget"]["tokens"] == 20
    assert rows[0]["latency_by_stage"]["generation"] >= 0
    assert rows[0]["failure_stage"] == "generation"
    assert rows[1]["status"] == "complete" and rows[1]["budget"]["tokens"] == 40
    assert created == [("exec", "inference"), ("exec", "inference"), ("judge", "evaluation")]


def test_truncated_generation_is_incomplete_with_score_preserved(monkeypatch, tmp_path):
    _, material = setup(monkeypatch, finish_reason="length")
    material["task_ids"] = ["task:1"]
    monkeypatch.setattr(runner, "_messages", lambda task, benchmark, private_record=None: (([], None), "ordinary", None))
    rows = runner.run(config(), [material], tmp_path / "out")
    assert rows[0]["status"] == "incomplete" and rows[0]["score_result"]["score"] == 1
    trace = json.loads((tmp_path / "out/tasks" / digest("task:1") / "generation_input.json").read_text())
    assert trace["messages_sha256"] == digest([])


def test_evaluation_failure_keeps_answer_and_both_stage_budgets(monkeypatch, tmp_path):
    _, material = setup(monkeypatch, fail_evaluation=True)
    material["task_ids"] = ["task:1"]
    monkeypatch.setattr(runner, "_messages", lambda task, benchmark, private_record=None: (([], None), "ordinary", None))
    rows = runner.run(config(), [material], tmp_path / "out")
    row = rows[0]
    assert row["failure_stage"] == "evaluation" and row["budget"]["model_calls"] == 2
    assert row["budget"]["tokens"] == 40 and row["answer_file"]
    assert set(row["latency_by_stage"]) == {"generation", "evaluation"}


def setup_report(monkeypatch, responses):
    _, material = setup(monkeypatch)
    material["benchmark"] = "writingbench"
    material["task_ids"] = ["task:1"]
    material["tasks"]["task:1"] = PublicTask(task_id="task:1", question="Community group buying")
    response_iterator = iter(responses)
    requests = []

    class Provider:
        def __init__(self, config):
            pass

        def create(self, role, agent_id, ledger, stage):
            class Model:
                def __init__(self):
                    self.calls = []
                    self.last_request_metadata = {"finish_reason": "stop"}

                def __call__(self, messages, **kwargs):
                    ledger.records.append({"stage": stage, "kind": "model", "tokens": 20})
                    requests.append({"role": role, "agent_id": agent_id, "messages": messages,
                                     "kwargs": kwargs})
                    self.calls.append(messages)
                    response = next(response_iterator) if role == "exec" else "judgment"
                    if isinstance(response, Exception):
                        raise response
                    if isinstance(response, tuple):
                        response, reason = response
                        self.last_request_metadata = {"finish_reason": reason}
                    return response

            return Model()

    monkeypatch.setattr(runner, "NativeModels", Provider)
    return material, requests


def test_report_sections_are_independent_plain_requests_and_fully_preserved(monkeypatch, tmp_path):
    contents = [f"opening{index} " + "word " * 1048 + f"closing{index}" for index in range(5)]
    material, requests = setup_report(monkeypatch, contents)
    rows = runner.run(config(), [material], tmp_path / "out")
    row = rows[0]
    assert row["complete"] and row["budget"]["model_calls"] == 6
    assert row["budget"]["tokens"] == 120
    assert len(row["section_generation"]) == 5
    assert all(section["complete"] for section in row["section_generation"])
    assert all(section["budget"]["tokens"] == 20 for section in row["section_generation"])
    directory = tmp_path / "out/tasks" / digest("task:1")
    answer = (directory / "answer.txt").read_text(encoding="utf-8")
    assert answer.count("## ") == 5
    for index, content in enumerate(contents):
        assert content in answer
        section_dir = directory / "sections" / f"section_{index + 1:02d}"
        assert (section_dir / "generation_response.txt").read_text(encoding="utf-8") == content
        assert (section_dir / "content.txt").read_text(encoding="utf-8") == content
        assert not (section_dir / "revision_input.json").exists()
    generated = [request for request in requests if request["role"] == "exec"]
    assert len(generated) == 5
    for request in generated:
        assert request["kwargs"] == {"max_tokens": 2000}
        assert len(request["messages"]) == 2
        payload = json.loads(request["messages"][1]["content"])
        assert "current_section" not in payload
        assert not any(content in request["messages"][1]["content"] for content in contents)


def test_report_section_revision_is_metered_and_preserves_original_generation(monkeypatch, tmp_path):
    first = "word " * 200
    revision = "word " * 1050
    material, requests = setup_report(monkeypatch, [first, revision] + [revision] * 4)
    rows = runner.run(config(), [material], tmp_path / "out")
    row = rows[0]
    assert row["complete"] and row["budget"]["model_calls"] == 7
    assert row["budget"]["tokens"] == 140
    section = row["section_generation"][0]
    assert section["budget"]["model_calls"] == 2 and section["budget"]["tokens"] == 40
    assert [attempt["kind"] for attempt in section["attempts"]] == ["generation", "revision"]
    assert [attempt["budget"]["tokens"] for attempt in section["attempts"]] == [20, 20]
    assert section["attempts"][0]["complete"] is False and section["attempts"][1]["complete"] is True
    assert all(attempt["latency_seconds"] >= 0 for attempt in section["attempts"])
    directory = tmp_path / "out/tasks" / digest("task:1")
    section_dir = directory / "sections/section_01"
    assert (section_dir / "generation_response.txt").read_text(encoding="utf-8") == first
    assert (section_dir / "revision_response.txt").read_text(encoding="utf-8") == revision
    assert (section_dir / "content.txt").read_text(encoding="utf-8") == revision
    payload = json.loads(requests[1]["messages"][1]["content"])
    assert payload["current_section"] == first and payload["current_word_count"] == 200


def test_report_section_api_failure_keeps_partial_sections_and_continues(monkeypatch, tmp_path):
    first = "word " * 1050
    material, _ = setup_report(monkeypatch, [first, RuntimeError("synthetic section failure"), "artifact"])
    material["task_ids"] = ["task:1", "task:2"]
    rows = runner.run(config(), [material], tmp_path / "out")
    failed, completed = rows
    assert failed["failure_stage"] == "generation" and failed["status"] == "failed"
    assert failed["budget"]["model_calls"] == 2 and failed["budget"]["tokens"] == 40
    assert len(failed["section_generation"]) == 2
    section = failed["section_generation"][1]
    assert section["failure"]["type"] == "RuntimeError"
    assert section["attempts"][0]["failure"]["type"] == "RuntimeError"
    assert section["budget"]["tokens"] == 20 and section["latency_seconds"] >= 0
    directory = tmp_path / "out/tasks" / digest("task:1")
    assert (directory / "sections/section_01/content.txt").read_text(encoding="utf-8") == first
    saved = json.loads((directory / "sections/section_02/record.json").read_text(encoding="utf-8"))
    assert saved["attempts"][0]["budget"]["tokens"] == 20
    assert not (directory / "submission.json").exists()
    assert completed["complete"] and completed["budget"]["model_calls"] == 2


def test_report_section_still_overlong_is_incomplete_without_trimming(monkeypatch, tmp_path):
    overlong = "opening " + "word " * 1198 + "closing"
    valid = "word " * 1050
    material, _ = setup_report(monkeypatch, [overlong, overlong] + [valid] * 4)
    rows = runner.run(config(), [material], tmp_path / "out")
    row = rows[0]
    assert row["status"] == "incomplete" and row["score_result"]["score"] == 1
    assert row["generation_checks"]["word_range_satisfied"] is True
    assert row["generation_checks"]["sections_complete"] is False
    assert row["section_generation"][0]["checks"]["word_count"] == 1200
    directory = tmp_path / "out/tasks" / digest("task:1")
    assert overlong in (directory / "answer.txt").read_text(encoding="utf-8")
    assert "condensed" not in (directory / "answer.txt").read_text(encoding="utf-8")


def test_report_section_truncation_gets_one_revision_and_remains_incomplete(monkeypatch, tmp_path):
    valid = "word " * 1050
    material, _ = setup_report(monkeypatch, [(valid, "length"), (valid, "length")] + [valid] * 4)
    rows = runner.run(config(), [material], tmp_path / "out")
    row = rows[0]
    assert row["status"] == "incomplete" and row["generation_finish_reason"] == "length"
    assert row["score_result"]["score"] == 1
    assert len(row["section_generation"][0]["attempts"]) == 2
    assert row["section_generation"][0]["checks"]["complete"] is True
    assert row["section_generation"][0]["complete"] is False
    assert row["budget"]["model_calls"] == 7
