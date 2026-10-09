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
from html.parser import HTMLParser

from .evidence import canonical_url


RETRIEVAL_STOPWORDS = frozenset({
    "about", "after", "also", "and", "are", "based", "been", "before", "between",
    "can", "compare", "could", "describe", "does", "each", "explain", "find", "for",
    "from", "give", "have", "how", "include", "into", "its", "list", "make", "more",
    "most", "not", "other", "our", "please", "provide", "research", "should", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "through", "under", "use", "using", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "write", "you", "your",
    "analysis", "annual", "data", "detailed", "guide", "information", "number", "official",
    "overview", "published", "report", "reports", "statistics", "year", "years",
})


def _has_minimum_relevance(query, row):
    public_query = re.sub(r"\bsite:\S+", "", query, flags=re.I).casefold()
    terms = set(re.findall(r"[^\W_]+", public_query))
    terms = {term for term in terms if len(term) >= 3 and not term.isdecimal()
             and term not in RETRIEVAL_STOPWORDS}
    if not terms:
        return True
    text = html.unescape(" ".join(str(row.get(key, "")) for key in ("title", "url", "snippet")))
    text = urllib.parse.unquote(text).casefold()
    matches = sum(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) is not None
                  for term in terms)
    return matches >= min(2, len(terms))


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


class _VisibleHTMLText(HTMLParser):
    ignored = frozenset({"script", "style", "nav", "footer", "aside", "template", "noscript", "svg"})
    voids = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
                       "meta", "param", "source", "track", "wbr"})
    blocks = frozenset({"article", "main", "section", "div", "p", "br", "h1", "h2", "h3",
                        "h4", "h5", "h6", "li", "tr", "pre", "blockquote"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_tags = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        hidden = (tag in self.ignored or "hidden" in attributes
                  or attributes.get("aria-hidden", "").lower() == "true")
        if self.hidden_tags:
            if tag not in self.voids:
                self.hidden_tags.append(tag)
        elif hidden:
            if tag not in self.voids:
                self.hidden_tags.append(tag)
        elif tag in self.blocks:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if self.hidden_tags:
            if tag in self.hidden_tags:
                reverse_index = self.hidden_tags[::-1].index(tag)
                del self.hidden_tags[len(self.hidden_tags) - reverse_index - 1:]
        elif tag in self.blocks:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden_tags:
            self.parts.append(data)

    def text(self):
        lines = [line.rstrip() for line in "".join(self.parts).splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _page_text(body, content_type):
    charset = re.search(r"charset\s*=\s*[\"']?([^\s;\"']+)", content_type, re.I)
    encoding = charset.group(1) if charset else "utf-8"
    text = body.decode(encoding, errors="replace")
    if "html" not in content_type.lower() and not re.match(r"\s*(?:<!doctype\s+html|<html\b)", text, re.I):
        return text, "plain-text-v1"
    parser = _VisibleHTMLText()
    parser.feed(text)
    parser.close()
    return parser.text(), "visible-html-text-v1"


class PublicRetriever:
    def __init__(self, config=None, *, opener=None, excluded_urls=()):
        self.config = config or RetrievalConfig()
        self.opener = opener or urllib.request.urlopen
        self.excluded_urls = frozenset(canonical_url(url) for url in excluded_urls)
        self.identity = {"implementation": "jit_mas.public_retrieval",
                         "version": "public-retrieval-v2",
                         "html_extraction": "visible-html-text-v1",
                         "relevance": "two-public-query-term-title-url-snippet-v1",
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
                filtered = [row for row in rows if not _has_minimum_relevance(query, row)]
                rows = [row for row in rows if _has_minimum_relevance(query, row)]
                record.update(status="ok" if rows else "empty", content_type=content_type, final_url=final_url,
                              result_count=len(rows), results=rows, filtered_results=filtered)
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
        text, extraction = _page_text(body, content_type)
        return {"url": final_url, "requested_url": canonical, "status": "ok", "content_type": content_type,
                "retrieved_at": _utc_now(), "text": text,
                "text_extraction": extraction, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
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
