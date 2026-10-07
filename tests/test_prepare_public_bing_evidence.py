import json
from types import SimpleNamespace

import scripts.prepare_public_bing_evidence as module
from jit_mas.schemas import PublicTask


def _task(question: str) -> PublicTask:
    return PublicTask.model_validate({
        "schema_version": "1.0",
        "task_id": "test-task",
        "question": question,
        "attachments": [],
        "constraints": [],
        "tools": [],
        "capabilities": [],
    })


def test_housing_planner_removes_request_scaffolding_and_keeps_facets():
    task = _task(
        "Help me compile a list of apartments in San Francisco (looking for at least "
        "2 bedroom). I am moving for the summer, have a cat, need roommates and parking, "
        "and my budget is $2200 per month."
    )
    response = module._Planner()([
        {"role": "system", "content": "planner"},
        {"role": "user", "content": json.dumps({"public_task": task.model_dump(mode="json")})},
    ])
    queries = json.loads(response.content)["queries"]
    assert len(queries) == 3
    joined = " ".join(queries).casefold()
    assert "san francisco" in joined
    assert "apartments" in joined
    assert "parking" in joined and "roommates" in joined
    assert "2200" in joined
    assert "compile" not in joined
    assert "looking" not in joined
    assert "least" not in joined


def test_planner_adds_primary_source_hints_for_named_data_providers():
    task = _task(
        "According to the U.S. Department of Energy and National Safety Council, "
        "which states meet the electric vehicle and traffic-death criteria in 2021 and 2022?"
    )
    response = module._Planner()([
        {"role": "system", "content": "planner"},
        {"role": "user", "content": json.dumps({"public_task": task.model_dump(mode="json")})},
    ])
    queries = json.loads(response.content)["queries"]
    joined = " ".join(queries)
    assert "site:energy.gov" in joined
    assert "site:nsc.org" in joined


def test_search_seeds_named_primary_data_sources(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel></channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search(
        "Department Energy National Safety Council electric vehicle registrations site:energy.gov"
    )
    urls = [row["url"] for row in result["results"]]
    assert "https://afdc.energy.gov/data/10963" in urls
    assert "https://injuryfacts.nsc.org/motor-vehicle/road-users/motor-vehicle-deaths-by-state/" in urls


def test_relevance_gate_drops_dictionary_page_but_keeps_named_school_result(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item><title>Number - English Grammar Today</title>
        <link>https://dictionary.example/grammar/number</link></item>
      <item><title>TDSB secondary student enrolment report</title>
        <link>https://www.tdsb.example/reports/secondary-students</link></item>
    </channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search("number secondary students Toronto District School")
    urls = [row["url"] for row in result["results"]]
    assert "https://dictionary.example/grammar/number" not in urls
    assert "https://www.tdsb.example/reports/secondary-students" in urls
    assert any(row["title"] == "Number - English Grammar Today"
               for row in result["filtered_results"])


def test_relevance_gate_does_not_treat_a_year_as_a_domain_anchor(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item><title>2016 - Wikipedia</title>
        <link>https://en.wikipedia.org/wiki/2016</link></item>
      <item><title>Toronto District School Board annual report</title>
        <link>https://www.tdsb.example/reports/2016</link></item>
    </channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search("2016 Toronto District School Board annual report")
    urls = [row["url"] for row in result["results"]]
    assert "https://en.wikipedia.org/wiki/2016" not in urls
    assert "https://www.tdsb.example/reports/2016" in urls


def test_relevance_gate_keeps_housing_listing_with_multiple_domain_anchors(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item><title>San Francisco apartments and rentals</title>
        <link>https://housing.example/san-francisco-apartments</link></item>
      <item><title>Number and grammar terms</title>
        <link>https://dictionary.example/number</link></item>
    </channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search("San Francisco apartments rent 2200 parking")
    urls = [row["url"] for row in result["results"]]
    assert "https://housing.example/san-francisco-apartments" in urls
    assert "https://dictionary.example/number" not in urls


def test_relevance_gate_rejects_dictionary_sense_for_named_research_question(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item><title>DEPARTMENT | English dictionary definition</title>
        <link>https://dictionary.cambridge.org/ja/dictionary/english/department</link></item>
      <item><title>Electric vehicle registrations by state 2021 2022</title>
        <link>https://www.energy.gov/data/electric-vehicle-registrations</link></item>
    </channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search(
        "Department Energy National Safety Council states electric vehicle registrations 2021 2022"
    )
    urls = [row["url"] for row in result["results"]]
    assert "https://dictionary.cambridge.org/ja/dictionary/english/department" not in urls
    assert "https://www.energy.gov/data/electric-vehicle-registrations" in urls
    assert any(row["reason"] == "obvious-lexical-page" for row in result["filtered_results"])


def test_relevance_gate_rejects_single_anchor_wrong_topic_page(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item><title>What is HUD? Heads-up display explained</title>
        <link>https://example.org/hud-explainer</link></item>
      <item><title>HUD housing assistance data and eligibility</title>
        <link>https://www.hud.gov/data</link></item>
    </channel></rss>"""

    def fake_get(*_args, **_kwargs):
        return SimpleNamespace(text=rss, raise_for_status=lambda: None)

    monkeypatch.setattr(module.requests, "get", fake_get)
    result = module._search("HUD housing assistance eligibility data")
    urls = [row["url"] for row in result["results"]]
    assert "https://example.org/hud-explainer" not in urls
    assert "https://www.hud.gov/data" in urls
    assert any(row["reason"] == "insufficient-informative-query-overlap"
               for row in result["filtered_results"])


def test_pack_quality_gate_rejects_only_retrieval_notices():
    pack = {
        "queries": ["Department Energy electric vehicle data"],
        "sources": [{"source_id": "search-0", "status": "empty"}],
        "archive": [{"kind": "web_search", "status": "empty", "query_index": 0}],
    }
    try:
        module._validate_pack_quality(pack)
    except ValueError as exc:
        assert "no relevant" in str(exc)
    else:  # pragma: no cover - defensive assertion for a fail-closed gate
        raise AssertionError("empty retrieval notice must not be accepted as evidence")
