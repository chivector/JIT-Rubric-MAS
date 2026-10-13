from jit_mas.evidence import EVIDENCE_QUESTION_HEADER
from jit_mas.schemas import PublicTask
from jit_mas.targeted_checklists import targeted_high_score_directives


def test_nba_guide_requires_coverage_bounded_arithmetic_and_specific_sources():
    task = PublicTask(task_id="nba", question="Which NBA player has been most influential?")
    guide = " ".join(targeted_high_score_directives(task))
    assert "at least two" in guide and "argmax" in guide
    assert "0-10" in guide and "source IDs" in guide


def test_source_text_cannot_trigger_unrelated_task_guidance():
    task = PublicTask(task_id="other", question="Write an essay." + EVIDENCE_QUESTION_HEADER
                      + '{"sources":[{"text":"Which NBA player is influential?"}]}')
    assert targeted_high_score_directives(task) == []


def test_sea_guide_requires_country_coverage_and_approval_gates():
    task = PublicTask(task_id="sea", question="Advise a plant-based meat startup on the Southeast Asian market.")
    guide = " ".join(targeted_high_score_directives(task))
    assert "Singapore/Thailand/Indonesia" in guide
    assert "approvals before launch" in guide and "contingency" in guide
