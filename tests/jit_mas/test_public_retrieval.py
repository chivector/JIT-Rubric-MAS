"""Public retrieval fallback checks using synthetic responses and no network."""

import json
import urllib.parse

import pytest

from jit_mas.public_retrieval import PublicRetriever, RetrievalConfig
from scripts import prepare_public_evidence


class Response:
    def __init__(self, body, content_type):
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.body[:limit]


@pytest.mark.parametrize("prefix", ["//duckduckgo.com", "https://duckduckgo.com"])
def test_duckduckgo_fallback_unwraps_redirect_before_url_exclusions(prefix):
    excluded = "https://private.example/synthetic-reference"
    allowed = "https://public.example/synthetic-source"
    calls = []

    def opener(request, *, timeout):
        calls.append(request.full_url)
        if "bing.com" in request.full_url:
            return Response(b"<rss><channel/></rss>", "application/rss+xml")
        links = [f'{prefix}/l/?uddg={urllib.parse.quote(url, safe="")}&amp;rut=synthetic'
                 for url in (excluded, allowed)]
        body = "".join(f'<a class="result__a" href="{link}">Synthetic title</a>' for link in links)
        return Response(body.encode(), "text/html")

    retriever = PublicRetriever(opener=opener, excluded_urls=[excluded])
    result = retriever.search("Synthetic question")
    assert [row["url"] for row in result["results"]] == [allowed]
    assert [row["backend"] for row in result["attempts"]] == ["bing_rss", "duckduckgo_html"]
    assert [row["status"] for row in result["attempts"]] == ["empty", "ok"]
    assert len(calls) == 2


def test_bing_success_does_not_invoke_fallback():
    calls = []

    def opener(request, *, timeout):
        calls.append(request.full_url)
        return Response(b"<rss><channel><item><link>https://public.example/source</link>"
                        b"<title>Synthetic</title></item></channel></rss>", "application/rss+xml")

    result = PublicRetriever(opener=opener).search("Synthetic question")
    assert result["results"][0]["source"] == "bing_rss"
    assert len(calls) == 1


def test_retrieval_byte_limit_fails_without_echoing_page_content():
    def opener(request, *, timeout):
        return Response(b"PRIVATE_CANARY" * 1000, "text/html")

    retriever = PublicRetriever(RetrievalConfig(max_bytes=1024), opener=opener)
    result = retriever.search("Synthetic question")
    assert result["results"] == []
    assert all(attempt["error"] == "ValueError" for attempt in result["attempts"])
    assert "PRIVATE_CANARY" not in json.dumps(result)


def test_fetch_records_final_redirect_url():
    requested = "https://public.example/redirect"
    final = "https://public.example/source"
    response = Response(b"Synthetic content", "text/plain")
    response.geturl = lambda: final
    retriever = PublicRetriever(opener=lambda *args, **kwargs: response)
    result = retriever.fetch(requested)
    assert result["url"] == final
    assert result["requested_url"] == requested


def test_fetch_rejects_excluded_redirect_before_reading_response_body():
    excluded = "https://private.example/reference"
    response = Response(b"PRIVATE_CANARY", "text/plain")
    response.geturl = lambda: excluded
    reads = []
    response.read = lambda limit: reads.append(limit) or response.body
    retriever = PublicRetriever(opener=lambda *args, **kwargs: response, excluded_urls=[excluded])
    with pytest.raises(ValueError, match="excluded source"):
        retriever.fetch("https://public.example/redirect")
    assert reads == []


def test_prepare_public_evidence_honors_max_pages(tmp_path, monkeypatch, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps([{"task_id": "synthetic-public", "question": "Synthetic question"}]),
                          encoding="utf-8")
    observed = []

    class Retriever:
        identity = {"implementation": "synthetic"}

        def __init__(self, config):
            pass

        def search(self, query):
            return []

        def fetch(self, url):
            return {"text": "Synthetic content"}

    class Builder:
        def __init__(self, *args, config, **kwargs):
            observed.append(config.max_pages)

        def build(self, task):
            return {"task_id": task.task_id}

    monkeypatch.setattr(prepare_public_evidence, "PublicRetriever", Retriever)
    monkeypatch.setattr(prepare_public_evidence, "EvidencePackBuilder", Builder)
    monkeypatch.setattr(prepare_public_evidence, "save_evidence_pack", lambda *args: None)
    monkeypatch.setattr(prepare_public_evidence, "build_evidence_manifest",
                        lambda *args: {"manifest_sha256": "synthetic"})
    assert prepare_public_evidence.main(["--tasks", str(tasks_path), "--output", str(tmp_path / "packs"),
                                         "--max-pages", "2"]) == 0
    assert observed == [2]
    assert json.loads(capsys.readouterr().out)["created"] == 1
