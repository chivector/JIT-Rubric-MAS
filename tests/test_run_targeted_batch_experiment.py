import copy
import json
from types import SimpleNamespace

from jit_mas.evidence import EVIDENCE_QUESTION_HEADER
from jit_mas.schemas import PublicTask
from scripts.mas_baseline_methods import JudgeEnvelopeModel
from scripts.run_targeted_batch_experiment import configure_targeted_pipeline


def test_targeted_hook_preserves_evidence_private_records_and_identity():
    evidence = json.dumps({"sources": [{"id": "source-one", "text": "A verified measurement."}]})
    task = PublicTask(task_id="research-one", question="Write a research report." + EVIDENCE_QUESTION_HEADER + evidence)
    private = {"research-one": {"rubrics": [{"rubric_id": 1, "criterion": "Cite measurements", "weight": 2}]}}
    original = copy.deepcopy(private)
    passed_judges = []

    def evaluator_factory(judge):
        passed_judges.append(judge)
        return SimpleNamespace(_judge_factory=lambda: "child", identity="unchanged")

    pipeline = SimpleNamespace(tasks={task.task_id: task}, private_records=private,
                               evaluator_factory=evaluator_factory)
    configure_targeted_pipeline(pipeline, benchmark="researchrubrics")
    question = pipeline.tasks[task.task_id].question
    prefix, marker, preserved_evidence = question.partition(EVIDENCE_QUESTION_HEADER)
    assert "Cite measurements" in prefix
    assert marker == EVIDENCE_QUESTION_HEADER
    assert preserved_evidence == evidence
    assert pipeline.private_records == original
    assert pipeline.evaluator_factory(None).identity == "unchanged"
    evaluator = pipeline.evaluator_factory("primary")
    assert isinstance(passed_judges[-1], JudgeEnvelopeModel)
    assert isinstance(evaluator._judge_factory(), JudgeEnvelopeModel)
    configure_targeted_pipeline(pipeline, benchmark="researchrubrics")
    assert pipeline.tasks[task.task_id].question == question
