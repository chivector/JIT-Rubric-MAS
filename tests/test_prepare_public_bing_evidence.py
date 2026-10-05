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
