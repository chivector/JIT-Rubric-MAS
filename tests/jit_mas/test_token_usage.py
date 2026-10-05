from jit_mas.token_usage import (
    merge_usage,
    summarize_budget,
    summarize_outcome,
    usage_totals_from_slots,
)


def test_summarize_budget_preserves_provider_and_estimated_usage_by_stage():
    usage = summarize_budget({
        "records": [
            {"kind": "model", "stage": "inference", "input_tokens": 11,
             "output_tokens": 7, "estimated": False},
            {"kind": "model", "stage": "evaluation", "input_tokens": 5,
             "output_tokens": 3, "estimated": True},
            {"kind": "tool", "stage": "inference", "bytes": 10},
        ],
    })
    assert usage["model_calls"] == 2
    assert usage["actual_model_calls"] == 1
    assert usage["estimated_model_calls"] == 1
    assert usage["input_tokens"] == 16 and usage["output_tokens"] == 10
    assert usage["total_tokens"] == 26 and usage["estimated_tokens"] == 8
    assert usage["source"] == "mixed"
    assert usage["by_stage"]["inference"]["total_tokens"] == 18
    assert usage["by_stage"]["evaluation"]["estimated_tokens"] == 8


def test_legacy_aggregate_budget_is_retained_as_unattributed_estimate():
    usage = summarize_budget({"model_calls": 2, "tokens": 123, "records": []})
    assert usage["model_calls"] == 2
    assert usage["total_tokens"] == 123
    assert usage["unattributed_tokens"] == 123
    assert usage["estimated_tokens"] == 123
    assert usage["usage_unknown"] is True
    assert usage["source"] == "unknown"


def test_summarize_outcome_combines_generation_and_evaluation_budgets():
    usage = summarize_outcome({
        "generation_budget": {"records": [{"kind": "model", "stage": "inference",
                                             "input_tokens": 4, "output_tokens": 6,
                                             "estimated": False}]},
        "evaluation_budget": {"records": [{"kind": "model", "stage": "evaluation",
                                              "input_tokens": 3, "output_tokens": 2,
                                              "estimated": False}]},
    })
    assert usage["total_tokens"] == 15
    assert usage["actual_model_calls"] == 2
    assert set(usage["by_stage"]) == {"inference", "evaluation"}


def test_slot_aggregate_ignores_pending_and_sums_terminal_rows():
    rows = [
        {"status": "pending", "result": {"token_usage": {"total_tokens": 999}}},
        {"status": "complete", "result": {"token_usage": summarize_budget({
            "records": [{"kind": "model", "stage": "inference", "input_tokens": 2,
                         "output_tokens": 3, "estimated": False}],
        })}},
        {"status": "failed", "result": {"token_usage": summarize_budget({
            "records": [{"kind": "model", "stage": "inference", "input_tokens": 5,
                         "output_tokens": 7, "estimated": True}],
        })}},
    ]
    usage = usage_totals_from_slots(rows)
    assert usage["model_calls"] == 2
    assert usage["total_tokens"] == 17
    assert usage["estimated_tokens"] == 12
    assert usage["source"] == "mixed"
