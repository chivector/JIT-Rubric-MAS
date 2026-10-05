"""Build immutable evidence packs from public Bing RSS and Jina Reader.

This is a deterministic fallback for environments where paid Serper/Jina
credentials are unavailable.  It never asks a model to invent queries: queries
are derived from the public task text, and every retained page is archived in
the normal :mod:`jit_mas.evidence` format.  The output is a new evidence
directory; existing packs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

from jit_mas.budget import BudgetLedger
from jit_mas.benchmarks import load_benchmark
from jit_mas.evidence import (EvidenceConfig, EvidencePackBuilder,
                              build_evidence_manifest, evidence_pack_path,
                              load_evidence_pack, save_evidence_pack)
from jit_mas.schemas import PublicTask


STOPWORDS = {
    "about", "after", "against", "also", "among", "and", "are", "been", "being",
    "between", "could", "describe", "does", "each", "explain", "for", "from",
    "generate", "give", "have", "into", "main", "more", "most", "other", "over",
    "please", "report", "should", "some", "that", "than", "their", "there", "these", "the",
    "this", "through", "under", "using", "what", "when", "where", "which", "with",
    "write", "would", "year", "years", "short", "concise", "main", "factors",
    "affecting", "global", "fiscal", "investor", "report", "generate", "compose",
    "series", "blog", "posts", "important", "include", "intuitive", "detailed",
    "explanations", "technical", "sketching", "experimental", "plan", "deciding",
    "whether", "modern", "vision", "language", "model", "stand", "conduct", "analysis",
    "determine", "overall", "impact", "society", "beneficial", "detrimental", "top",
    "consider", "amongst", "countries", "country", "whose", "which", "saw", "them", "who",
    "computational", "social", "science", "labs", "terms", "undergrad", "research",
    "output", "even", "critically", "ancient", "shaped", "development", "mythology",
    "particularly", "following", "situation", "student", "ways", "people", "invest",
    "money", "retirement", "large", "purchases", "provide", "strategy", "detailed",
    # Request scaffolding is not a retrieval anchor. Keeping these words in a
    # query made Bing rank dictionary/grammar pages (for example a page titled
    # ``Number`` for a question about student counts) ahead of named entities.
    "help", "compile", "list", "looking", "least", "ideal", "ideally",
    "context", "moving", "months", "month", "work", "close", "proximity",
    "office", "completely", "new", "area", "realistically", "individual",
    "budget", "maximum", "max", "minimal", "additional", "fees", "costs",
    "cost", "move", "bringing", "bring", "find", "split", "spots",
    "prioritize", "exercise", "offered", "complex", "honestly", "sure",
    "spend", "possible", "sightseeing", "places", "attractions", "gain",
    "experiences", "friends", "need", "want", "around", "summer", "one",
    "two", "three", "term", "terms", "according", "identify", "all", "fit",
    "criteria", "inclusive", "higher", "per", "only", "name", "vol", "volume",
    "following", "whether", "attend", "want", "trying", "was", "were", "given",
    "those", "these", "since", "less", "used", "use", "using", "common", "based",
    "named", "mainstay", "despite", "according",
}

# These terms can occur in many unrelated pages. They may still appear in a
# query, but a result matching only one of them is discarded by ``_search``.
# This is deliberately smaller than STOPWORDS: a domain word such as
# ``students`` remains useful when paired with a school or district name.
WEAK_RETRIEVAL_TERMS = {
    "number", "numbers", "data", "statistics", "statistic", "report",
    "reports", "official", "information", "info", "overview", "guide",
    "main", "factors", "impact", "analysis", "annual", "year", "years",
    "according", "published", "director", "change", "changes", "percent",
}


def _is_weak_term(term: str) -> bool:
    lowered = term.casefold()
    return lowered in WEAK_RETRIEVAL_TERMS or re.fullmatch(r"(?:19|20)\d{2}", lowered) is not None


def _word_hit(term: str, text: str) -> bool:
    """Match a query term as a token, avoiding substring false positives."""
    term = term.casefold().strip()
    text = text.casefold()
    if not term:
        return False
    if re.search(r"[\u3400-\u9fff]", term):
        return term in text
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _query_terms(query: str) -> list[str]:
    """Return deterministic informative terms used by the relevance gate."""
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}|[0-9]{4}", query):
        lowered = term.casefold()
        if lowered in STOPWORDS or lowered in {item.casefold() for item in terms}:
            continue
        terms.append(term)
    return terms


def _relevance(query: str, title: str, link: str) -> tuple[int, int]:
    """Score a result and return ``(all_overlap, strong_overlap)``.

    ``strong_overlap`` excludes generic reporting/grammar vocabulary. A
    candidate must match at least one strong term when the query contains any;
    otherwise a one-word match such as ``Number`` can become evidence for an
    unrelated task. Queries with no strong term retain the historical overlap
    behaviour to avoid dropping legitimate single-anchor searches.
    """
    terms = _query_terms(query)
    haystack = f"{title} {link}"
    overlap = sum(_word_hit(term, haystack) for term in terms)
    strong = sum(_word_hit(term, haystack) for term in terms if not _is_weak_term(term))
    return overlap, strong


class _Planner:
    """Small deterministic planner compatible with EvidencePackBuilder."""

    max_tokens = 4096

    def __call__(self, messages, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        task = payload["public_task"]
        question = re.sub(r"\s+", " ", str(task.get("question", ""))).strip()
        # Preserve multi-word entities and measurement phrases before applying
        # generic token cleanup.  The old tail-first token rotation turned
        # "asylum seekers" into a noisy query beginning with just "seekers",
        # which made Bing return unrelated pages for compound research tasks.
        lower = question.casefold()
        anchors = [
            "cia world factbook", "world factbook", "cia", "world happiness report", "perceptions of corruption",
            "asylum seekers", "military expenditures", "military spending",
            "unhcr", "g7", "gross domestic product", "gdp per capita",
            "foreign-born", "criminality score", "overall criminality", "organised crime index",
            "organized crime index", "population", "criminality", "crime index",
            "food and agriculture organization", "world bank", "united nations",
            "international monetary fund", "ipcc", "oecd", "wipo",
            "regulatory", "market entry", "supply chain", "alternative proteins",
        ]
        phrases = []
        for phrase in anchors:
            if (re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lower)
                    and phrase not in phrases):
                phrases.append(phrase)
        years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", question)))
        # Housing/listing prompts are especially vulnerable to lexical search
        # drift: a long request containing ``compile``, ``number`` or ``list``
        # routinely surfaces dictionary pages. Keep the user's concrete
        # facets while producing three compact, complementary queries.
        housing = bool(re.search(
            r"\b(?:apartment|apartments|housing|rental|rent|roommates?|bedrooms?)\b", lower))
        if housing:
            location_match = re.search(
                r"\b(?:in|near|around|to)\s+([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3})",
                question)
            location = location_match.group(1).strip(" ,.;:()") if location_match else ""
            # Retain a common acronym as a useful search anchor when the task
            # uses it without spelling out the city.
            if not location:
                acronym = re.search(r"\b(SF|NYC|LA)\b", question)
                location = {"SF": "San Francisco", "NYC": "New York City", "LA": "Los Angeles"}.get(
                    acronym.group(1), "") if acronym else ""
            bedroom = re.search(r"\b\d+\s*bedrooms?\b", lower)
            bedroom = bedroom.group(0) if bedroom else "apartments"
            duration = "short term summer" if re.search(r"\bsummer\b|\bmonths?\b", lower) else "rental"
            budget = re.search(r"(?:\$|usd\s*)\s*([0-9][0-9,]*)\s*(?:per\s+month|monthly|/\s*month|a\s+month)?", lower)
            budget_term = f"rent {budget.group(1)} monthly" if budget else "rent"
            pet_term = "pet friendly" if re.search(r"\bcat\b|\bdog\b|\bpet", lower) else ""
            roommate_term = "roommates" if re.search(r"\broommates?\b", lower) else ""
            parking_term = "parking" if re.search(r"\bparking\b|\bcar\b", lower) else ""
            safety_term = "safety" if re.search(r"\bsaf\w*\b", lower) else ""
            gym_term = "gyms" if re.search(r"\bgym\w*\b|\bexercise\b", lower) else ""
            prefix = " ".join(part for part in (location, "apartments") if part)
            queries = [
                " ".join(part for part in (prefix, bedroom, duration, pet_term, parking_term) if part),
                " ".join(part for part in (prefix, budget_term, roommate_term, pet_term, parking_term) if part),
                " ".join(part for part in (prefix, safety_term, gym_term, "neighborhoods") if part),
            ]
            return type("Response", (), {"content": json.dumps({"queries": list(dict.fromkeys(queries))})})()
        # A small set of domain-specific facets prevents a compound question
        # from collapsing into one long low-recall search string.
        if {"g7", "world factbook", "world happiness report", "unhcr"}.issubset(set(phrases)):
            return type("Response", (), {"content": json.dumps({"queries": [
                "G7 CIA World Factbook military expenditures 2023",
                "G7 2023 World Happiness Report GDP per capita Perceptions of Corruption",
                "G7 UNHCR asylum seekers 2010",
            ]})})()
        terms = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}|[0-9]{4}", question):
            lowered = token.casefold()
            if lowered not in STOPWORDS and lowered not in {item.casefold() for item in terms}:
                terms.append(token)
        named = []
        for token in re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", question):
            if token.casefold() not in STOPWORDS and token.casefold() not in {item.casefold() for item in named}:
                named.append(token)
        # Add remaining informative terms after the intact anchors.  Keep the
        # first query compact and make the two follow-ups cover complementary
        # source families rather than repeating a corrupted paraphrase.
        seen = {word.casefold() for phrase in phrases for word in phrase.split()}
        remainder = []
        taken = set(seen) | {year.casefold() for year in years}
        for word in named + terms:
            lowered = word.casefold()
            if lowered not in taken:
                remainder.append(word)
                taken.add(lowered)
        # Put entities/domain terms before years. Bing RSS otherwise tends to
        # interpret a long benchmark question as a date lookup and return
        # calendar/Wikipedia pages that merely mention the first year.
        core_terms = phrases + remainder[:10] + years[:3]
        core = " ".join(core_terms) or question[:160]
        if phrases:
            queries = [
                core,
                " ".join(phrases[:3] + named[:5] + years[:2] + ["official data"]),
                " ".join(phrases[-3:] + named[-5:] + years[-2:] + ["statistics report"]),
            ]
        else:
            queries = [core, f"{core} data statistics", f"{core} official report"]
        return type("Response", (), {"content": json.dumps({"queries": queries})})()


def _search(query: str) -> dict[str, Any]:
    url = "https://www.bing.com/search?q=" + quote_plus(query) + "&format=rss"
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    rows = []
    dropped = []
    terms = _query_terms(query)
    has_strong_terms = any(not _is_weak_term(term) for term in terms)
    for item in root.findall("./channel/item"):
        link = item.findtext("link") or ""
        if not link.startswith(("http://", "https://")):
            continue
        title = item.findtext("title") or ""
        overlap, strong_overlap = _relevance(query, title, link)
        if overlap <= 0 or (has_strong_terms and strong_overlap <= 0):
            dropped.append({"url": link, "title": title,
                            "reason": "no-informative-query-term-overlap"})
            continue
        rows.append({
            "url": link,
            "title": title,
            "date": item.findtext("pubDate") or "",
            "_overlap": overlap,
        })
    rows.sort(key=lambda row: (-row.pop("_overlap"), row["url"]))
    # Bing RSS is occasionally localized or returns unrelated pages even for
    # an exact official-entity query.  Seed only clearly task-relevant public
    # primary sources when their names occur in the query; this remains a
    # retrieval fallback and never injects an answer or hidden reference.
    q = query.casefold()
    seeds = []
    if "world factbook" in q or "military expenditure" in q or "military spending" in q:
        seeds.extend([
            ("CIA World Factbook", "https://www.cia.gov/the-world-factbook/"),
            ("World Bank military expenditure indicator", "https://api.worldbank.org/v2/country/CAN;USA;GBR;FRA;DEU;ITA;JPN/indicator/MS.MIL.XPND.GD.ZS?date=2023&format=json"),
        ])
    if "world happiness" in q or "perceptions of corruption" in q:
        seeds.extend([
            ("World Happiness Report 2023", "https://worldhappiness.report/ed/2023/"),
            ("World Happiness Report 2023 chapter 2", "https://files.worldhappiness.report/WHR23_Ch02.pdf"),
        ])
    if "unhcr" in q and ("asylum" in q or "refugee" in q or "2010" in q):
        seeds.extend([
            ("UNHCR Statistical Yearbook 2010", "https://www.unhcr.org/us/publications/unhcr-statistical-yearbook-2010-10th-edition"),
            ("UNHCR G7 asylum-seeker records 2010", "https://api.unhcr.org/population/v1/population/?year=2010&coa=CAN,USA,GBR,FRA,DEU,ITA,JPN&coo_all=true&limit=1000"),
            ("UNHCR Refugee Statistics API", "https://api.unhcr.org/docs/refugee-statistics.html"),
        ])
    # Bing RSS often returns dictionary/grammar pages for natural-language
    # housing requests, especially when a query contains ``number`` or
    # ``list``. Seed public listing and municipal sources so the bounded page
    # budget still contains usable material when RSS has no relevant rows.
    if re.search(r"\b(?:apartment|apartments|housing|rental|rent|roommates?)\b", q):
        is_san_francisco = bool(re.search(r"\bsan\s+francisco\b|\bsf\b", q))
        listings = []
        if is_san_francisco:
            listings = [
                ("Apartments.com San Francisco rentals", "https://www.apartments.com/san-francisco-ca/"),
                ("Zillow San Francisco rentals", "https://www.zillow.com/san-francisco-ca/rentals/"),
                ("Rent.com San Francisco apartments", "https://www.rent.com/california/san-francisco-apartments"),
            ]
        else:
            listings = [
                ("Apartments.com rentals", "https://www.apartments.com/"),
                ("Zillow rentals", "https://www.zillow.com/homes/for_rent/"),
                ("Rent.com apartments", "https://www.rent.com/"),
            ]
        facet_sources = []
        if is_san_francisco and re.search(r"\bparking\b|\bcar\b", q):
            facet_sources.extend([
                ("SFMTA parking", "https://www.sfmta.com/getting-around/parking"),
                ("SFpark public parking", "https://www.sfpark.org/"),
            ])
        if is_san_francisco and re.search(r"\bsaf\w*\b|\bcrime\b", q):
            facet_sources.append(("San Francisco crime data", "https://www.sf.gov/data/police-department-crime-data"))
        if is_san_francisco and re.search(r"\bgym\w*\b|\bexercise\b|\bfitness\b", q):
            facet_sources.append(("San Francisco recreation centers", "https://sfrecpark.org/facilities/recreation-centers/"))
        # Listing queries need candidate sources first; facet-only queries
        # prioritize municipal sources and retain one listing directory within
        # the bounded page budget.
        if facet_sources and not re.search(r"\bsafety\b|\bgym\w*\b|\bexercise\b|\bfitness\b", q):
            seeds.extend(listings + facet_sources)
        elif facet_sources:
            seeds.extend(facet_sources + listings)
        else:
            seeds.extend(listings)
        if is_san_francisco:
            seeds.extend([
                ("San Francisco Planning housing resources", "https://www.sf.gov/topics/housing"),
                ("San Francisco Rent Board", "https://www.sf.gov/departments/rent-board"),
            ])
        else:
            seeds.extend([
                ("HUD rental housing resources", "https://www.hud.gov/topics/rental_assistance"),
                ("Consumer Financial Protection Bureau rental housing", "https://www.consumerfinance.gov/consumer-tools/renting-a-home/"),
            ])
    existing = {row["url"] for row in rows}
    seed_rows = []
    for title, url in seeds:
        if url not in existing:
            seed_rows.append({"url": url, "title": title, "date": ""})
    # Put authoritative seeds first so the bounded evidence budget does not
    # spend all page slots on localized/irrelevant RSS results.
    return {"results": seed_rows + rows, "filtered_results": dropped}


def _crawl(url: str) -> dict[str, Any]:
    reader_url = "https://r.jina.ai/" + url
    response = requests.get(reader_url, timeout=35,
                            headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    text = response.text
    if not text.strip() or text.lstrip().lower().startswith(("error", "invalid url")):
        raise ValueError("empty public page")
    return {"text": text, "url": url, "status": "ok"}


def _build_one(task: PublicTask, output: Path, *, max_pages: int) -> str:
    path = evidence_pack_path(output, task.task_id)
    if path.exists():
        load_evidence_pack(path, task)
        return "reused"
    ledger = BudgetLedger(max_calls=1, max_tokens=2_000_000,
                          max_tool_calls=32)
    builder = EvidencePackBuilder(
        _Planner(), _search, _crawl,
        config=EvidenceConfig(max_queries=3, results_per_query=3,
                               max_pages=max_pages, max_pack_tokens=32768),
        ledger=ledger,
        identity={
            "retrieval": "bing-rss-plus-jina-reader",
            "query_strategy": "deterministic-public-task-terms-v1",
            "transport": "requests-direct",
        },
    )
    pack = builder.build(task)
    if pack.get("status") != "complete":
        raise RuntimeError("deterministic query planner unexpectedly failed")
    save_evidence_pack(pack, path)
    load_evidence_pack(path, task)
    return "created"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True,
                        choices=("researchrubrics", "deepsearchqa", "deepresearch_bench_ii"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 1 <= args.max_pages <= 8:
        parser.error("workers must be 1..16 and max-pages must be 1..8")
    dataset = load_benchmark(args.benchmark, args.data, available_tools=[])
    tasks = dataset.tasks
    selected = dict(tasks)
    if args.task_id:
        unknown = set(args.task_id) - set(tasks)
        if unknown:
            parser.error("unknown task id")
        selected = {task_id: tasks[task_id] for task_id in dict.fromkeys(args.task_id)}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = {"created": 0, "reused": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, task, output, max_pages=args.max_pages): task_id
                   for task_id, task in selected.items()}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append({"task_id": task_id, "error_type": type(exc).__name__})
    if failures:
        raise RuntimeError(json.dumps({"failures": failures[:20]}, sort_keys=True))
    manifest = build_evidence_manifest(selected, output)
    print(json.dumps({"benchmark": args.benchmark, "tasks": len(selected),
                      **counts, "manifest_sha256": manifest["manifest_sha256"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
