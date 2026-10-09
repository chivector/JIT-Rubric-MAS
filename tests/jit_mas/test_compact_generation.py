import json
from types import SimpleNamespace

import pytest

from jit_mas.compact_generation import (build_messages, choose_construction,
                                        generate_one, literal_ledger,
                                        task_public_payload)
from jit_mas.schemas import PublicTask
from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.benchmarks import BenchmarkDataset
from jit_mas.schemas import digest
from scripts import run_compact_benchmark_subset as runner


def task(question, constraints=None):
    return PublicTask(task_id="private-id", question=question,
                      constraints=constraints or [])


def test_prompt_contains_only_public_projection_and_compact_literal_ledger():
    value = task("Use word shore at least 2 times.")
    assert task_public_payload(value) == {"question": value.question,
                                          "constraints": value.constraints}
    messages, response_format = build_messages(value)
    assert response_format is None
    assert "private-id" not in json.dumps(messages)
    assert literal_ledger(value) == [{"literal": "shore", "operator": ">=", "count": 2}]


def test_unsupported_construction_falls_back_without_changing_ordinary_generation():
    mode, plan = choose_construction(task("Use word shore exactly 2 times."), numeric=True)
    assert mode == "ordinary" and plan is None


def test_literal_prompt_compiles_directives_without_repeating_raw_count_instructions():
    public = task("Write a poem. Include keyword shore once in your response.",
                  ["Use word tide twice. Keep it brief."])
    construction, plan = choose_construction(public, literal=True)
    messages, _ = build_messages(public, construction=construction, plan=plan)
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == "Write a poem. Use the supplied word insertion protocol."
    assert payload["constraints"] == ["Use the supplied word insertion protocol. Keep it brief."]
    assert "'shore'" in messages[0]["content"] and "'tide'" in messages[0]["content"]
    assert "literal ledger" not in messages[0]["content"]
    assert public.question.endswith("Include keyword shore once in your response.")


def test_numeric_construction_uses_one_strict_schema_render_and_one_call():
    value = task("Include exactly 1 numeric literals in the response.")
    mode, plan = choose_construction(value, numeric=True)
    assert mode == "numeric" and plan is not None

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return type("Response", (), {"content": json.dumps({
                "opening": "A result",
                "number_slots": [{"before": "It contains", "number": "7",
                                  "after": "items."}],
                "closing": ""})})()

    model = FakeModel()
    result = generate_one(value, model, BudgetLedger(max_calls=1, max_tokens=10000),
                          numeric=True)
    assert result["error"] is None
    assert result["answer"] == "A result It contains 7 items."
    assert len(model.calls) == 1
    assert model.calls[0][1]["response_format"]["type"] == "json_schema"


def test_schema_flags_can_select_one_plan_and_conflicting_public_families_fall_back():
    mode, plan = choose_construction(task("Include exactly 2 numbers in the response."),
                                     numeric=True, positional=True)
    assert mode == "numeric" and plan is not None
    mode, plan = choose_construction(task("Include exactly 2 numbers in the response. "
                                        "Use word shore as the 2nd word in the 1st sentence."),
                                     numeric=True, positional=True)
    assert mode == "ordinary" and plan is None


def test_failed_structured_answer_keeps_consumed_call_budget_without_retry():
    class InvalidResponse:
        calls = 0

        def __call__(self, messages, **kwargs):
            self.calls += 1
            return "{\"opening\":\"missing slots\"}"

        def get_token_counts(self):
            return {"input_token_count": 12, "output_token_count": 8}

    ledger = BudgetLedger(max_calls=1, max_tokens=10000)
    provider = InvalidResponse()
    model = MeteredModel(provider, ledger, "inference", max_tokens=1000)
    result = generate_one(task("Include exactly 2 numbers in the response."),
                          model, ledger, numeric=True)
    assert result["answer"] is None and result["answer_hash"] is None
    assert result["error"]["type"] == "ValidationError"
    assert result["raw_response"] == "{\"opening\":\"missing slots\"}"
    assert result["raw_response_hash"] is not None
    assert provider.calls == result["budget"]["model_calls"] == 1
    assert result["budget"]["tokens"] == 20


def test_length_finish_reason_preserves_artifact_but_marks_generation_incomplete():
    class TruncatedModel:
        last_request_metadata = {"finish_reason": "length"}

        def __call__(self, messages, **kwargs):
            return "A partially generated artifact"

    result = generate_one(task("Write an artifact."), TruncatedModel(),
                          BudgetLedger(max_calls=1, max_tokens=10000))
    assert result["answer"] == "A partially generated artifact"
    assert result["answer_hash"] is not None
    assert result["complete"] is False
    assert result["error"]["type"] == "IncompleteGeneration"


def test_provider_failure_error_redacts_credential_environment_values(monkeypatch):
    monkeypatch.setenv("COMPACT_TEST_KEY", "secret-example-key")

    class FailedModel:
        def __call__(self, messages, **kwargs):
            raise RuntimeError("request rejected secret-example-key")

    result = generate_one(task("Write an artifact."), FailedModel(),
                          BudgetLedger(max_calls=1, max_tokens=10000))
    assert result["error"]["message"] == "request rejected [REDACTED]"


@pytest.mark.parametrize("finish_reason", ["content_filter", "tool_calls", "function_call"])
def test_nonfinal_provider_response_preserves_text_but_is_incomplete(finish_reason):
    class NonfinalModel:
        last_request_metadata = {"finish_reason": finish_reason}

        def __call__(self, messages, **kwargs):
            return "Partial artifact"

    result = generate_one(task("Write an artifact."), NonfinalModel(),
                          BudgetLedger(max_calls=1, max_tokens=10000))
    assert result["answer"] == "Partial artifact"
    assert result["complete"] is False
    assert result["finish_reason"] == finish_reason


def test_literal_construction_generates_named_gaps_and_renders_public_counts_once():
    class LiteralModel:
        calls = 0

        def __call__(self, messages, **kwargs):
            self.calls += 1
            assert kwargs["response_format"]["json_schema"]["name"] == "PublicLiteralSlots"
            return json.dumps({"gaps": {"gap_000": "A", "gap_001": "stretches along the",
                                       "gap_002": "; the", "gap_003": "rises."}})

    model = LiteralModel()
    result = generate_one(task("Use word shore once, word tide twice."), model,
                          BudgetLedger(max_calls=1, max_tokens=10000), literal=True)
    assert result["construction"] == "literal"
    assert result["answer"] == "A shore stretches along the tide ; the tide rises."
    assert result["complete"] is True and model.calls == 1


def test_numeric_template_layout_can_run_once_with_a_complete_prose_template():
    class TemplateModel:
        def __call__(self, messages, **kwargs):
            assert kwargs["response_format"]["json_schema"]["name"] == "PublicNumericTemplate"
            return json.dumps({"answer_template": "The group has <NUM_A> members and <NUM_B> leaders.",
                               "numeric_values": {"NUM_A": "7", "NUM_B": "2"}})

    result = generate_one(task("Include exactly 2 numbers in the response."), TemplateModel(),
                          BudgetLedger(max_calls=1, max_tokens=10000), numeric=True,
                          numeric_layout="template")
    assert result["construction"] == "numeric_template"
    assert result["answer"] == "The group has  7  members and  2  leaders."
    assert result["complete"] is True


def test_truncated_structured_output_is_preserved_without_a_rendered_answer():
    class TruncatedModel:
        last_request_metadata = {"finish_reason": "length"}

        def __call__(self, messages, **kwargs):
            return '{"gaps":{"gap_000":"A'

    result = generate_one(task("Use word shore once."), TruncatedModel(),
                          BudgetLedger(max_calls=1, max_tokens=10000), literal=True)
    assert result["raw_response"] == '{"gaps":{"gap_000":"A'
    assert result["answer"] is None and result["complete"] is False
    assert result["error"]["type"] == "IncompleteGeneration"


def test_runner_preserves_generated_artifact_when_fixed_checker_raises(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models:\n  exec:\n    model: fixture\n    endpoint: https://example.invalid/v1\n    key_env: COMPACT_TEST_KEY\n", encoding="utf-8")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    public = task("Write an artifact.")
    dataset = BenchmarkDataset("ifbench", {public.task_id: public},
        {public.task_id: {"private": "do not generate from this"}}, {}, {}, "dataset")
    monkeypatch.setenv("COMPACT_TEST_KEY", "fixture")
    monkeypatch.setattr(runner, "load_benchmark", lambda *args: dataset)

    class Checker:
        identity = {"fixture": "frozen"}

        def __init__(self, *args):
            pass

        def assert_frozen(self):
            pass

    class Evaluator:
        def evaluate(self, *args, **kwargs):
            raise ValueError("checker unavailable")

    class Model:
        def __init__(self):
            self.model = SimpleNamespace(client=SimpleNamespace(close=lambda: None))

        def __call__(self, messages, **kwargs):
            assert "do not generate from this" not in json.dumps(messages)
            return "The generated artifact"

    monkeypatch.setattr(runner, "PinnedInstructionChecker", Checker)
    monkeypatch.setattr(dataset, "evaluator", lambda *args, **kwargs: Evaluator())
    monkeypatch.setattr(runner, "_model", lambda *args: Model())
    output = tmp_path / "unique-output"
    args = SimpleNamespace(benchmark="ifbench", config=str(config_path),
        dataset=str(data_path), checker_source=str(tmp_path), output=str(output),
        task_id=[public.task_id], count=None,
        numeric_construction=False, positional_construction=False,
        literal_construction=False, numeric_layout="array")
    report = runner.run(args)
    row = report["records"][0]
    assert row["answer_hash"] == digest("The generated artifact")
    assert row["score_result"]["status"] == "evaluation_failed"
    assert (output / row["answer_file"]).read_text(encoding="utf-8") == "The generated artifact"
    assert (output / row["raw_response_file"]).read_text(encoding="utf-8") == "The generated artifact"
    assert report["summary"]["requested"] == report["summary"]["attempted"] == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["completion_rate"] == 0
    assert report["summary"]["mean_all_tasks"] == 0
    with pytest.raises(FileExistsError):
        runner.run(args)
