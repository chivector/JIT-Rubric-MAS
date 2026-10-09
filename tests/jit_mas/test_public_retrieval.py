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
        body = "".join(f'<a class="result__a" href="{link}">Synthetic question title</a>' for link in links)
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

    result = PublicRetriever(opener=opener).search("Synthetic")
    assert result["results"][0]["source"] == "bing_rss"
    assert len(calls) == 1


@pytest.mark.parametrize("query,title,link", [
    ("Universal basic income Finland Canada trial", "Universal dictionary",
     "https://dictionary.example/universal"),
    ("Patagonia Uniqlo H&M sustainability", "Japanese lottery results",
     "https://lottery.example/results"),
])
def test_irrelevant_nonempty_bing_results_trigger_fallback(query, title, link):
    calls = []

    def opener(request, *, timeout):
        calls.append(request.full_url)
        if "bing.com" in request.full_url:
            return Response((f"<rss><channel><item><link>{link}</link>"
                             f"<title>{title}</title></item></channel></rss>").encode(),
                            "application/rss+xml")
        return Response(f'<a class="result__a" href="https://public.example/source">{query}</a>'.encode(),
                        "text/html")

    result = PublicRetriever(opener=opener).search(query)
    assert result["results"][0]["source"] == "duckduckgo_html"
    assert result["attempts"][0]["status"] == "empty"
    assert result["attempts"][0]["filtered_results"][0]["url"] == link
    assert len(calls) == 2


def test_relevance_uses_snippet_without_counting_years_as_domain_matches():
    def opener(request, *, timeout):
        return Response(b"<rss><channel><item><link>https://public.example/2025</link>"
                        b"<title>2025 report</title><description>Finland basic income trial results</description>"
                        b"</item><item><link>https://dictionary.example/2025</link>"
                        b"<title>2025 definition</title></item></channel></rss>", "application/rss+xml")

    result = PublicRetriever(opener=opener).search("Finland basic income trial 2025 report")
    assert [row["url"] for row in result["results"]] == ["https://public.example/2025"]
    assert len(result["attempts"][0]["filtered_results"]) == 1


def test_fetch_extracts_readable_html_without_scripts_navigation_or_hidden_content():
    content = b"""<!doctype html><html><head><title>Research report</title>
        <script>private_tracking_payload</script><style>.hidden { color:red }</style></head>
        <body><nav><div>Unrelated menu</div></nav><input hidden><main><h1>Trial findings</h1>
        <p>Finland <strong>basic income</strong> &amp; employment.</p>
        <table><tr><th>Year</th><th>Result</th></tr><tr><td>2025</td><td>12</td></tr></table>
        <pre>def useful():\n    return 12</pre><div hidden>Hidden text</div>
        <footer>Cookie links</footer></main></body></html>"""
    retriever = PublicRetriever(opener=lambda *args, **kwargs: Response(content, "text/html; charset=utf-8"))
    result = retriever.fetch("https://public.example/report")
    assert "Finland basic income & employment." in result["text"]
    assert "Year\tResult" in result["text"]
    assert "def useful():\n    return 12" in result["text"]
    assert all(value not in result["text"] for value in
               ("private_tracking_payload", "Unrelated menu", "Cookie links", "Hidden text", "<main>"))
    assert result["sha256"] != result["text_sha256"]
    assert result["text_extraction"] == "visible-html-text-v1"


def test_fetch_respects_plain_text_charset():
    retriever = PublicRetriever(opener=lambda *args, **kwargs:
                               Response("caf\u00e9".encode("iso-8859-1"), "text/plain; charset=iso-8859-1"))
    result = retriever.fetch("https://public.example/report")
    assert result["text"] == "caf\u00e9"
    assert result["text_extraction"] == "plain-text-v1"


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


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_planner_config_accepts_only_explicit_meta_credentials(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv("SYNTHETIC_PLANNER_KEY", "synthetic-value")
    config = {"backend": "native_jit", "models": {"meta": {
        "model": "synthetic-model", "endpoint": "https://synthetic.example/v1",
        "key_env": "SYNTHETIC_PLANNER_KEY"}}}
    path = tmp_path / f"planner{suffix}"
    path.write_text(json.dumps(config) if suffix == ".json" else prepare_public_evidence.yaml.safe_dump(config),
                    encoding="utf-8")
    parsed = prepare_public_evidence.read_planner_config(path)
    assert parsed.models["meta"].model == "synthetic-model"
    monkeypatch.delenv("SYNTHETIC_PLANNER_KEY")
    with pytest.raises(ValueError, match="Missing API credential"):
        prepare_public_evidence.read_planner_config(path)


def test_native_query_planner_is_metered_once_and_receives_public_task_only(monkeypatch):
    from jit_mas.budget import BudgetLedger
    from jit_mas.config import MASConfig
    from jit_mas.evidence import EvidencePackBuilder
    from jit_mas.schemas import PublicTask
    from scripts.models import openai_server

    monkeypatch.setenv("SYNTHETIC_PLANNER_KEY", "synthetic-value")
    observed = {}

    class Model:
        def __init__(self, **kwargs):
            observed["options"] = kwargs
            self.client = self

        def with_options(self, **kwargs):
            return self

        def __call__(self, messages, **kwargs):
            observed["messages"] = messages
            return {"queries": ["Finland basic income trial", "Canada guaranteed income evaluation"]}

    monkeypatch.setattr(openai_server, "OpenAIServerModel", Model)
    config = MASConfig(models={"meta": {"model": "synthetic-model", "endpoint": "https://synthetic.example/v1",
                                        "key_env": "SYNTHETIC_PLANNER_KEY", "max_tokens": 512}})
    ledger = BudgetLedger(max_calls=1, max_tokens=10000, max_tool_calls=12)
    planner = prepare_public_evidence.query_planner(config, ledger, 512)
    builder = EvidencePackBuilder(planner, lambda query: [], lambda url: "", ledger=ledger)
    task = PublicTask(task_id="synthetic-public", question="Compare Finland and Canada basic income trials")
    pack = builder.build(task)
    assert pack["queries"] == ["Finland basic income trial", "Canada guaranteed income evaluation"]
    assert pack["usage"]["model_calls"] == 1
    assert json.loads(observed["messages"][1]["content"])["public_task"] == task.model_dump(mode="json")
    assert observed["options"]["max_attempts"] == 1


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
def test_query_planner_discards_reflected_credentials_before_archiving(monkeypatch, field):
    from types import SimpleNamespace

    from jit_mas.budget import BudgetLedger
    from jit_mas.config import MASConfig
    from jit_mas.evidence import EvidencePackBuilder
    from jit_mas.schemas import PublicTask
    from scripts.models import openai_server

    secret = "synthetic-query-credential-canary"
    monkeypatch.setenv("SYNTHETIC_PLANNER_KEY", secret)

    class Model:
        def __init__(self, **kwargs):
            self.client = self

        def with_options(self, **kwargs):
            return self

        def __call__(self, messages, **kwargs):
            return SimpleNamespace(**{field: secret})

    monkeypatch.setattr(openai_server, "OpenAIServerModel", Model)
    config = MASConfig(models={"meta": {"model": "synthetic", "endpoint": "https://synthetic.example/v1",
                                        "key_env": "SYNTHETIC_PLANNER_KEY", "max_tokens": 512}})
    ledger = BudgetLedger(max_calls=1, max_tokens=10000, max_tool_calls=12)
    planner = prepare_public_evidence.query_planner(config, ledger, 512)
    builder = EvidencePackBuilder(planner, lambda query: [], lambda url: "", ledger=ledger)
    pack = builder.build(PublicTask(task_id="synthetic-public", question="Finland income trials"))

    assert pack["status"] == "failed"
    assert pack["usage"]["model_calls"] == 1
    assert secret not in json.dumps(pack)
