import json

import pytest

from jit_mas.schemas import PublicTask
from jit_mas.targeted_writing import (
    REPORT_SECTION_COUNT, Poems, build_messages, generation_checks, render_poems,
    render_report_sections, render_response, report_length_feedback, report_section_check,
    report_section_messages, report_section_revision_messages, writing_profile,
)


def test_poems_renderer_enforces_lines_and_character_palindrome():
    task = PublicTask(task_id="unrelated-id", question="Write modern poems; one poem must be a palindrome.")
    parsed = Poems.model_validate({"poems": [{"title": "乡村", "lines": ["这是新乡村"] * 8}] * 3})
    answer = render_poems(parsed)
    checks = generation_checks(task, answer)
    assert checks["poem_count"] == 4
    assert checks["body_line_counts"] == [8, 8, 8, 9]
    assert checks["exact_palindrome"] is True and checks["complete"] is True
    assert build_messages(task)[1]["json_schema"]["strict"] is True


def test_long_report_checks_actual_word_count_and_public_only_messages():
    task = PublicTask(task_id="private-identity", question="Community group buying: write 5000-6000 words.")
    messages, schema = build_messages(task)
    assert schema is None and "private-identity" not in str(messages)
    assert "322.8/1.5371" in messages[0]["content"]
    assert generation_checks(task, "word " * 5200)["complete"] is True
    assert generation_checks(task, "word " * 3000)["complete"] is False
    assert writing_profile(task) == "investment-report"


def test_writing_messages_include_task_specific_coverage_checklist():
    task = PublicTask(task_id="private-identity", question="Write an introduction.")
    messages, _ = build_messages(task, {"rubrics": [{"criterion": "Use a heading.",
                                                       "weight": 2, "axis": "Quality"}]})
    assert "Use a heading." in str(messages)


def test_report_renderer_preserves_every_section_and_measures_final_length():
    contents = [f"opening{index} " + "evidence " * 1048 + f"closing{index}"
                for index in range(REPORT_SECTION_COUNT)]
    answer = render_report_sections(contents)
    task = PublicTask(task_id="report", question="Community group buying: write 5000-6000 words.")
    assert answer.count("## ") == REPORT_SECTION_COUNT
    for content in contents:
        assert content in answer
    assert "condensed" not in answer
    assert generation_checks(task, answer)["complete"] is True
    assert str(generation_checks(task, answer)["word_count"]) in report_length_feedback(answer)


def test_report_response_does_not_trim_overlong_content():
    content = "opening " + "evidence " * 6700 + "last substantive conclusion"
    assert render_response("investment-report", content) == content
    task = PublicTask(task_id="report", question="Community group buying: write 5000-6000 words.")
    assert generation_checks(task, content)["complete"] is False


def test_report_sections_are_independent_plain_requests_with_shared_facts_and_checklist():
    task = PublicTask(task_id="private-identity", question="Community group buying 322.8 billion yuan.",
                      constraints=["For corporate investors"])
    private = {"criteria": [{"name": "Data", "criteria_description": "Explain the data.",
                              "9-10": "Accurate causal interpretation."}]}
    for index in range(REPORT_SECTION_COUNT):
        messages, schema = report_section_messages(task, index, private)
        payload = json.loads(messages[1]["content"])
        assert schema is None and len(messages) == 2
        assert "plain English prose" in messages[0]["content"]
        assert "1000-1150 words" in messages[0]["content"]
        assert "nine paragraphs of 115-120 words" in messages[0]["content"]
        assert "Hard stop at 1,100 words" in messages[0]["content"]
        assert payload["public_question_and_supplied_facts"] == task.question
        assert payload["public_constraints"] == task.constraints
        assert payload["section_index"] == index + 1
        assert payload["target_word_range"] == [1000, 1150]
        assert "322.8/1.5371" in payload["shared_analysis_rules"]
        assert payload["task_specific_coverage_checklist"][0]["high_score_expectation"] == "Accurate causal interpretation."
        assert "private-identity" not in str(messages)
        assert "current_section" not in payload


def test_report_section_revision_contains_only_one_complete_section():
    task = PublicTask(task_id="report", question="Community group buying.")
    original = "evidence " * 1200
    messages, schema = report_section_revision_messages(task, 3, original)
    payload = json.loads(messages[1]["content"])
    assert schema is None
    assert payload["current_section"] == original
    assert payload["current_word_count"] == 1200
    assert "do not output JSON" in messages[0]["content"]
    assert "Do not cut" in payload["revision_instruction"]
    assert "Hard stop at 1,100 words" in payload["revision_instruction"]
    assert report_section_check(original)["complete"] is False
    assert report_section_check("evidence " * 1000)["complete"] is True
    assert report_section_check("evidence " * 1150)["complete"] is True
    assert report_section_check("evidence " * 1151)["complete"] is False
    assert report_section_check("evidence " * 999)["complete"] is False


def test_report_helpers_reject_missing_sections_and_invalid_indices():
    task = PublicTask(task_id="report", question="Community group buying.")
    with pytest.raises(ValueError, match="exactly 5"):
        render_report_sections(["one section"])
    with pytest.raises(ValueError, match="nonempty"):
        render_report_sections(["valid"] * 4 + [""])
    for index in (-1, 5, True, "0"):
        with pytest.raises(ValueError, match="index"):
            report_section_messages(task, index)
    with pytest.raises(ValueError, match="nonempty"):
        report_section_revision_messages(task, 0, "")
