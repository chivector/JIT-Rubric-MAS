import copy
import json

import pytest

from jit_mas.attribution import (
    _compact_duplicate_event_content, build_credit_assignments, compact_local_execution_task,
)
from jit_mas.schemas import EvaluationFeedback, RubricAlignment, RubricGraph, TeamSpec, digest


def test_global_artifacts_keep_unique_evidence_and_reuse_exact_submission():
    answer = "Complete submitted artifact" * 200
    submission = {"evidence_id": "submission:original", "answer": answer}
    events = [{"event_id": "writer:final", "agent_id": "writer", "kind": "final_answer", "content": answer},
              {"event_id": "writer:draft", "agent_id": "writer", "kind": "artifact", "content": answer + " draft"},
              {"event_id": "writer:usage", "kind": "resource_usage", "content": {"output_tokens": 800}}]
    original = copy.deepcopy(events)

    compacted = _compact_duplicate_event_content(events, None, submission=submission)

    assert events == original
    assert compacted[0] == {key: value for key, value in original[0].items() if key != "content"} | {
        "content_ref": {"source": "submission", "path": "submission.answer", "content_hash": digest(answer)}}
    assert compacted[1:] == original[1:]
    assert compacted[0]["event_id"] == "writer:final"


def test_local_trace_reference_takes_priority_and_preserves_unique_submission():
    local = {"answer": "Same completed artifact"}
    submission = {"answer": "Same completed artifact"}
    events = [{"event_id": "writer:final", "content": local["answer"]}]

    compacted = _compact_duplicate_event_content(events, local, submission=submission)

    assert compacted[0]["content_ref"]["source"] == "local_execution"
    assert compacted[0]["content_ref"]["path"] == "local_execution.answer"


def test_task_compaction_keeps_criteria_agent_policies_and_all_unique_trace_content():
    task = {"task_id": "writingbench:32", "question": "论文要求" * 800}
    nested_input = {"public_task": task, "agent": {"agent_id": "writer", "prompt": "Original policy"},
                    "rubrics": [{"criterion": "Complete all requirements"}],
                    "shared_ledger": {"outline": ["Unique upstream contribution"]}}
    local = {"answer": "Original final answer", "metadata": {"agent_id": "writer"},
             "trajectory": [{"model_input_messages": [{"role": "system", "content": "Complete role context"},
                 {"role": "user", "content": json.dumps(nested_input, ensure_ascii=False)}],
                 "model_output_messages": [{"role": "assistant", "content": "Complete original output"}],
                 "action_output": "Original final answer", "start_time": 1}], "sub_runs": []}
    original = copy.deepcopy(local)

    compacted = compact_local_execution_task(local, task)

    assert local == original
    encoded = json.loads(compacted["trajectory"][0]["model_input_messages"][1]["content"])
    assert encoded["public_task"] == {"content_ref": {
        "source": "task", "path": "task", "content_hash": digest(task)}}
    encoded["public_task"] = task
    assert encoded == nested_input
    compacted["trajectory"][0]["model_input_messages"][1]["content"] = original["trajectory"][0]["model_input_messages"][1]["content"]
    assert compacted == original


def test_task_compaction_keeps_distinct_tasks_invalid_json_and_system_input():
    task = {"task_id": "task", "question": "Complete public task"}
    local = {"trajectory": [{"model_input_messages": [
        {"role": "user", "content": json.dumps({"task": {**task, "question": "Distinct observed task"}})},
        {"role": "user", "content": "Unstructured observed input"},
        {"role": "system", "content": json.dumps({"public_task": task})}]}]}

    assert compact_local_execution_task(local, task) == local


def test_answer_compaction_references_exact_complete_action_output_without_mutating_trace():
    answer = "Complete original writer answer with unique factual details. " * 100
    task = {"task_id": "task", "question": "Complete public task"}
    local = {"answer": answer, "metadata": {"agent_id": "writer"}, "trajectory": [
        {"action_output": "Distinct prior draft", "model_output_messages": [
            {"role": "assistant", "content": "Distinct prior protocol output"}]},
        {"action_output": answer, "model_output_messages": [
            {"role": "assistant", "content": json.dumps({"answer": answer})}]}]}
    original = copy.deepcopy(local)

    compacted = compact_local_execution_task(local, task)

    assert local == original
    assert compacted["trajectory"] == original["trajectory"]
    reference = compacted["answer"]["content_ref"]
    assert reference == {"source": "local_execution", "path": "local_execution.trajectory[1].action_output",
                         "content_hash": digest(answer)}
    restored = copy.deepcopy(compacted)
    restored["answer"] = restored["trajectory"][1]["action_output"]
    assert restored == original
    assert len(json.dumps(compacted)) < len(json.dumps(original))
    event = {"event_id": "writer:original-answer", "content": answer}
    supplied = _compact_duplicate_event_content([event], local)[0]
    assert supplied["content_ref"] == reference


@pytest.mark.parametrize("answer,action_output", [
    ("Unique final answer" * 100, "Different observed action output" * 100),
    ("Short exact answer", "Short exact answer"),
    ({"answer": "structured answer"}, {"answer": "structured answer"}),
])
def test_answer_compaction_preserves_unique_short_and_structured_answers(answer, action_output):
    local = {"answer": answer, "trajectory": [{"action_output": action_output}]}

    assert compact_local_execution_task(local, {"task_id": "task", "question": "Original question"}) == local


def test_meta_rationale_error_names_exact_missing_and_extra_agent_keys():
    graph = RubricGraph(rubrics=[{"rubric_id": "requirement", "requirement": "Complete the task"}])
    team = TeamSpec(agents=[{"agent_id": "writer", "role": "Writer", "capability": "writing",
                            "rubric_ids": ["requirement"]}], synthesizer_id="writer",
                    coverage={"requirement": ["writer"]}, primary={"requirement": "writer"})
    feedback = EvaluationFeedback(task_id="task", evaluator_version="fixture", complete=True, score=0,
        rubrics=[{"rubric_id": "criterion", "criterion": "Complete the task", "weight": 1, "score": 0}])
    alignment = RubricAlignment(matches=[{"predicted_ids": ["requirement"], "evaluated_ids": ["criterion"],
                                         "confidence": 1, "relation": "equivalent", "rationale": "Same requirement"}])

    with pytest.raises(ValueError, match="missing_agent_ids=\\['writer'\\]; extra_agent_ids=\\['stale'\\]"):
        build_credit_assignments(graph, team, alignment, feedback,
                                 meta_assignments={"writer": ["criterion"]},
                                 meta_assignment_rationales={"stale": "Wrong key"})
