from types import SimpleNamespace
import json

import pytest

from jit_mas.evidence import (EVIDENCE_QUESTION_HEADER, EVIDENCE_TASK_CONSTRAINT,
                             public_instruction_question)
from jit_mas.schemas import PublicTask
from scripts.run_evolution_mechanism_pilot import configure_supplemental_sources


def test_supplement_replaces_failed_pack_and_binds_actual_sources_and_calculations(tmp_path):
    task = PublicTask(task_id="nba", question="Compare NBA influence." + EVIDENCE_QUESTION_HEADER
                      + '{"sources":[{"status":"failed"}]}')
    pipeline = SimpleNamespace(tasks={"nba": task})
    payload = {"tasks": {"nba": {"sources": [{"source_id": "nba-legends", "status": "ok",
        "title": "NBA legends", "url": "https://www.nba.com/", "excerpt": "Observed source text."}],
        "derived_calculations": {"formula": "1+2", "result": 3}}}, "preparation": {"http_requests": 1}}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    configure_supplemental_sources(pipeline, path)
    question = pipeline.tasks["nba"].question
    assert "Observed source text" in question and '"result": 3' in question
    assert '"status": "failed"' not in question
    assert task.question != question
    assert pipeline.source_supplement["preparation"]["http_requests"] == 1


def test_supplement_rejects_failed_source(tmp_path):
    pipeline = SimpleNamespace(tasks={"nba": PublicTask(task_id="nba", question="Compare NBA influence.")})
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"tasks": {"nba": {"sources": [{"status": "failed"}]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="successfully fetched"):
        configure_supplemental_sources(pipeline, path)


def test_canonical_supplement_keeps_evidence_out_of_instruction_routing(tmp_path):
    question = "Compare NBA influence."
    task = PublicTask(task_id="nba", question=question,
                      constraints=[EVIDENCE_TASK_CONSTRAINT])
    pipeline = SimpleNamespace(tasks={"nba": task})
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"tasks": {"nba": {"sources": [{
        "source_id": "source", "status": "ok", "title": "Title",
        "url": "https://www.nba.com/", "excerpt": "Hidden lexical hint: exact word count."}]}}}),
        encoding="utf-8")
    configure_supplemental_sources(pipeline, path)
    augmented = pipeline.tasks["nba"]
    assert "Hidden lexical hint" in augmented.question
    assert public_instruction_question(augmented.model_dump(mode="json")) == question
