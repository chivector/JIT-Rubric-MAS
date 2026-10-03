"""Public relevance must survive the memory cap without weakening source isolation."""

import copy

import pytest

from jit_mas.experience import retrieve
from jit_mas.schemas import Experience, ExperienceSnapshot, PublicTask


def lesson(name, **changes):
    values = dict(experience_id=name, bank="organization",
                  instruction="Assign responsibilities within the available budget.",
                  applicability="", source_task_ids=["source-" + name],
                  evidence=["submission:" + name])
    values.update(changes)
    return Experience(**values)


def test_older_grounded_lesson_survives_many_recent_unscoped_entries():
    relevant = lesson("comparison", applicability="Comparative research",
                      task_signals=["compare", "research"])
    bank = ExperienceSnapshot(experiences=[relevant, *[
        lesson(str(index)) for index in range(30)]])
    original = copy.deepcopy(bank.model_dump(mode="json"))
    task = PublicTask(task_id="target", question="Compare two research methods.")
    found = retrieve(bank, task)
    assert len(found) == 24
    assert found[0]["experience_id"] == "comparison"
    assert [entry["experience_id"] for entry in found[1:]] == [str(i) for i in range(7, 30)]
    assert bank.model_dump(mode="json") == original


def test_specific_public_matches_beat_recency_and_repeated_words():
    entries = [lesson("specific", task_signals=["compare", "storage", "security"]),
               lesson("generic", task_signals=["compare", "article writing"]),
               lesson("repetition", task_signals=["compare compare comparison comparing"])]
    task = PublicTask(task_id="target", question="Write an article comparing storage security.")
    assert retrieve(ExperienceSnapshot(experiences=entries), task, limit=1)[0]["experience_id"] == "specific"


def test_operation_boundary_and_role_scope_remain_binding_before_ranking():
    entries = [lesson("forbidden", applicability="Comparative research", task_signals=["research"]),
               lesson("wrong-role", bank="execution", applicability="Research",
                      task_signals=["research"], capability="creative storytelling"),
               lesson("valid", task_signals=["research"])]
    task = PublicTask(task_id="target", question="Explain this research without comparison.")
    found = retrieve(ExperienceSnapshot(experiences=entries), task,
                     capability="research explanation", limit=1)
    assert [entry["experience_id"] for entry in found] == ["valid"]


def test_higher_relevance_never_overrides_current_restricted_or_future_source():
    grounded = dict(task_signals=["compare", "storage"])
    entries = [lesson("current", source_task_ids=["target"], **grounded),
               lesson("restricted", source_task_ids=["validation"], **grounded),
               lesson("future", created_at="2026-10-04T00:00:00+00:00", **grounded),
               lesson("past", created_at="2026-10-01T00:00:00+00:00")]
    task = PublicTask(task_id="target", question="Compare storage systems.")
    found = retrieve(ExperienceSnapshot(experiences=entries), task, limit=1,
                     excluded_task_ids=["validation"], before="2026-10-03T00:00:00+00:00")
    assert [entry["experience_id"] for entry in found] == ["past"]


def test_equal_public_relevance_prefers_recent_but_returns_chronological_order():
    entries = [lesson(str(index), task_signals=["compare"]) for index in range(4)]
    task = PublicTask(task_id="target", question="Compare systems.")
    found = retrieve(ExperienceSnapshot(experiences=entries), task, limit=2)
    assert [entry["experience_id"] for entry in found] == ["2", "3"]
    found[0]["source_task_ids"].append("mutated")
    assert entries[2].source_task_ids == ["source-2"]


def test_retrieval_limit_zero_returns_no_advice_and_negative_is_invalid():
    snapshot = ExperienceSnapshot(experiences=[lesson("one")])
    task = PublicTask(task_id="target", question="Explain a concept.")
    assert retrieve(snapshot, task, limit=0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        retrieve(snapshot, task, limit=-1)
