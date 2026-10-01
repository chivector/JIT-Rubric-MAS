"""Public-only search and page retrieval for fixed evidence preparation.

The module uses unauthenticated Bing RSS first and DuckDuckGo HTML second.
It records every endpoint and excludes coordinator-supplied private locators.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .evidence import canonical_url


@dataclass(frozen=True)
class RetrievalConfig:
    timeout: float = 30.0
    user_agent: str = "jit-mas-public-retrieval/1.0"
    max_results: int = 5
    max_bytes: int = 2_000_000
    bing_rss_url: str = "https://www.bing.com/search?format=rss&q={query}"
    duckduckgo_url: str = "https://html.duckduckgo.com/html/?q={query}"

    def __post_init__(self):
        if self.timeout <= 0 or self.max_results < 1 or self.max_bytes < 1024:
            raise ValueError("Invalid public retrieval limits")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _fetch_bytes(url, config, opener, *, excluded_urls=()):
    request = urllib.request.Request(url, headers={"User-Agent": config.user_agent})
    with opener(request, timeout=config.timeout) as response:
        final_url = canonical_url(response.geturl() if hasattr(response, "geturl") else url)
        if final_url in excluded_urls:
            raise ValueError("Public retrieval redirected to an excluded source")
        body = response.read(config.max_bytes + 1)
        if len(body) > config.max_bytes:
            raise ValueError("Public page exceeds configured byte limit")
        return body, response.headers.get("Content-Type", ""), final_url


def _rss_results(body, limit):
    root = ElementTree.fromstring(body)
    results = []
    for item in root.findall(".//item")[:limit]:
        link = item.findtext("link", "").strip()
        if not link:
            continue
        try:
            link = canonical_url(link)
        except ValueError:
            continue
        results.append({"url": link, "title": item.findtext("title", "").strip(),
                        "snippet": item.findtext("description", "").strip(), "source": "bing_rss"})
    return results


def _ddg_results(body, limit):
    text = body.decode("utf-8", errors="replace")
    results = []
    for match in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text, re.I | re.S):
        raw_url = html.unescape(match.group(1))
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
            raw_url = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        try:
            url = canonical_url(raw_url)
        except ValueError:
            continue
        title = re.sub(r"<[^>]+>", "", html.unescape(match.group(2))).strip()
        results.append({"url": url, "title": title, "snippet": "", "source": "duckduckgo_html"})
        if len(results) >= limit:
            break
    return results


class PublicRetriever:
    def __init__(self, config=None, *, opener=None, excluded_urls=()):
        self.config = config or RetrievalConfig()
        self.opener = opener or urllib.request.urlopen
        self.excluded_urls = frozenset(canonical_url(url) for url in excluded_urls)
        self.identity = {"implementation": "jit_mas.public_retrieval",
                         "version": "public-retrieval-v1",
                         "config": asdict(self.config),
                         "excluded_urls_sha256": hashlib.sha256(
                             "\n".join(sorted(self.excluded_urls)).encode()).hexdigest()}

    def _excluded(self, url):
        return canonical_url(url) in self.excluded_urls

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be nonempty")
        attempts = []
        endpoints = [("bing_rss", self.config.bing_rss_url), ("duckduckgo_html", self.config.duckduckgo_url)]
        for backend, template in endpoints:
            endpoint = template.format(query=urllib.parse.quote_plus(query.strip()))
            record = {"backend": backend, "endpoint": endpoint, "query": query.strip(),
                      "retrieved_at": _utc_now(), "status": "error"}
            try:
                body, content_type, final_url = _fetch_bytes(endpoint, self.config, self.opener,
                                                           excluded_urls=self.excluded_urls)
                rows = _rss_results(body, self.config.max_results) if backend == "bing_rss" else _ddg_results(body, self.config.max_results)
                rows = [row for row in rows if not self._excluded(row["url"])]
                record.update(status="ok" if rows else "empty", content_type=content_type, final_url=final_url,
                              result_count=len(rows), results=rows)
                attempts.append(record)
                if rows:
                    return {"results": rows, "attempts": attempts, "identity": self.identity}
            except Exception as exc:
                record["error"] = type(exc).__name__
                attempts.append(record)
        return {"results": [], "attempts": attempts, "identity": self.identity, "error": "all public search backends failed"}

    def fetch(self, url):
        canonical = canonical_url(url)
        if self._excluded(canonical):
            return {"url": canonical, "status": "excluded", "text": ""}
        body, content_type, final_url = _fetch_bytes(canonical, self.config, self.opener,
                                                   excluded_urls=self.excluded_urls)
        text = body.decode("utf-8", errors="replace")
        return {"url": final_url, "requested_url": canonical, "status": "ok", "content_type": content_type,
                "retrieved_at": _utc_now(), "text": text,
                "sha256": hashlib.sha256(body).hexdigest()}


def validate_public_retrieval(*, retriever=None):
    """Run deterministic endpoint and exclusion checks for a synthetic query."""
    client = retriever or PublicRetriever()
    result = client.search("NIST atomic clock synthetic validation")
    return {"identity": client.identity, "query": "NIST atomic clock synthetic validation",
            "result_count": len(result.get("results", [])),
            "backend_attempts": [{"backend": row.get("backend"), "status": row.get("status")}
                                 for row in result.get("attempts", [])],
            "public_only": all(row.get("url") not in client.excluded_urls for row in result.get("results", [])),
            "network": True}
