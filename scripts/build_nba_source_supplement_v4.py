"""Build the immutable, source-bound NBA supplement (v4).

This receipt is deliberately independent of the v3 receipt.  It merges all
windows belonging to one source id, takes titles from the fetch manifest, and
keeps date scope tied to text actually present in the locally archived page.
No model or provider is contacted by this script.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from jit_mas.pipeline import write_json
from jit_mas.schemas import digest, utc_now


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "outputs/evolution_mechanism_audit_20261010/nba_verified_fact_matrix_v3.json"
FETCH = ROOT / "outputs/evolution_mechanism_audit_20261010/nba_fetch_raw/fetch_manifest.json"
RAW = ROOT / "outputs/evolution_mechanism_audit_20261010/nba_fetch_raw"
TASK_ID = "6847465956a0f6376a605378"

# These two requests were made after the 44-request archived manifest.  They
# are retained as failed preparation attempts, never as evidence.
ADDITIONAL_TARGETED_403 = [
    {
        "requested_url": "https://www.nba.com/news/stephen-curry-nba-finals-mvp-2022",
        "final_url": "https://www.nba.com/news/stephen-curry-nba-finals-mvp-2022",
        "http_status": 403,
        "title": "Access Denied",
        "fetch_duration_seconds": 0.5746737000299618,
        "evidence_eligible": False,
    },
    {
        "requested_url": "https://www.nba.com/news/stephen-curry-first-unanimous-mvp-kia-most-valuable-player",
        "final_url": "https://www.nba.com/news/stephen-curry-first-unanimous-mvp-kia-most-valuable-player",
        "http_status": 403,
        "title": "Access Denied",
        "fetch_duration_seconds": 0.6376394000835717,
        "evidence_eligible": False,
    },
]

# Other exploratory links discovered in already archived HTML.  They are
# reported separately so the 44-request manifest and the two targeted Curry
# attempts remain auditable without counting these follow-ups as evidence.
EXPLORATORY_FOLLOWUPS = [
    {
        "requested_url": "https://www.nba.com/news/history-season-recap-2015-16",
        "final_url": "https://www.nba.com/news/history-season-recap-2015-16",
        "http_status": 404,
        "title": "Content Unavailable",
        "fetch_duration_seconds": 1.1820745000150055,
        "evidence_eligible": False,
    },
    {
        "requested_url": "https://www.nba.com/news/history-season-recap-2021-22",
        "final_url": "https://www.nba.com/news/history-season-recap-2021-22",
        "http_status": 404,
        "title": "Content Unavailable",
        "fetch_duration_seconds": 1.1400156000163406,
        "evidence_eligible": False,
    },
    {
        "requested_url": "https://www.nba.com/news/history-mvp-award-winners",
        "final_url": "https://www.nba.com/news/history-mvp-award-winners",
        "http_status": 403,
        "title": "Access Denied",
        "fetch_duration_seconds": 0.7105019999435171,
        "evidence_eligible": False,
    },
    {
        "requested_url": "https://www.nba.com/news/history-finals-mvp-winners",
        "final_url": "https://www.nba.com/news/history-finals-mvp-winners",
        "http_status": 403,
        "title": "Access Denied",
        "fetch_duration_seconds": 0.5687966999830678,
        "evidence_eligible": False,
    },
]


def _raw_text(source: dict) -> str:
    path = source.get("raw_text_path")
    if isinstance(path, str):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace")
    sid = source.get("source_id", "")
    fallback = RAW / f"{sid}.txt"
    return fallback.read_text(encoding="utf-8", errors="replace") if fallback.exists() else ""


def _body_date(text: str) -> str | None:
    """Return a date string only when it appears in the archived body."""
    patterns = (
        r"(?:Updated on|Published|Archive|From NBA\.com Staff|Shaun Powell)\s*[^\n]{0,80}?"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})",
        r"\b(\d{2}\.\d{2}\.\d{2,4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    )
    clean = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " "))
    for pattern in patterns:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _merge_sources(matrix: dict, fetch_rows: list[dict]) -> list[dict]:
    manifest = {row.get("source_id"): row for row in fetch_rows}
    merged: dict[str, dict] = {}
    for row in matrix.get("sources", []):
        sid = row["source_id"]
        fetched = manifest.get(sid, {})
        text = _raw_text(row)
        item = merged.setdefault(sid, {
            "source_id": sid,
            "title": fetched.get("title") or sid,
            "date": _body_date(text),
            "url": fetched.get("final_url") or row.get("final_url") or row.get("requested_url"),
            "final_url": fetched.get("final_url") or row.get("final_url") or row.get("requested_url"),
            "status": "ok",
            "http_status": fetched.get("http_status", row.get("http_status")),
            "raw_text_path": row.get("raw_text_path"),
            "content_hash_sha256": fetched.get("content_hash_sha256") or row.get("content_hash_sha256"),
            "fetch_duration_seconds": fetched.get("fetch_duration_seconds"),
            "excerpt_windows": [],
        })
        excerpt = row.get("excerpt")
        if isinstance(excerpt, str) and excerpt.strip() and excerpt not in [w["text"] for w in item["excerpt_windows"]]:
            item["excerpt_windows"].append({"text": excerpt, "source_matrix": "nba-verified-fact-matrix-v3"})
        if not item["date"]:
            item["date"] = _body_date(text)
    return list(merged.values())


def _append_raw_window(sources: list[dict], sid: str, start: str, end: str | None = None,
                       *, scope: str | None = None) -> None:
    """Append the exact normalized body span; fail if its anchor is absent."""
    source = next(row for row in sources if row["source_id"] == sid)
    raw = _raw_text(source).replace("\r", "")
    offset = raw.find(start)
    if offset < 0:
        raise ValueError(f"Source {sid} has no exact raw anchor {start!r}")
    finish = raw.find(end, offset) if end is not None else -1
    if end is not None and finish < 0:
        raise ValueError(f"Source {sid} has no exact raw end anchor {end!r}")
    text = raw[offset:finish] if finish >= 0 else raw[offset:]
    window = {"text": re.sub(r"\s+", " ", text).strip(),
              "raw_anchor": start, "extraction": "exact archived body span; whitespace normalized"}
    if scope:
        window["scope"] = scope
    if window["text"] not in [row["text"] for row in source["excerpt_windows"]]:
        source["excerpt_windows"].append(window)


def _append_curry_historical_bio(sources: list[dict]) -> dict:
    from bs4 import BeautifulSoup

    html_path = RAW / "nba_curry_profile.html"
    html_bytes = html_path.read_bytes()
    source = next(row for row in sources if row["source_id"] == "nba_curry_profile")
    actual_sha = hashlib.sha256(html_bytes).hexdigest()
    if actual_sha != source["content_hash_sha256"]:
        raise ValueError("Curry HTML differs from the archived fetch SHA256")
    soup = BeautifulSoup(html_bytes.decode("utf-8"), "html.parser")
    node = soup.find("script", id="__NEXT_DATA__")
    if node is None or not node.string:
        raise ValueError("Curry archived HTML has no NEXT_DATA JSON")
    data = json.loads(node.string)
    bio_html = data["props"]["pageProps"]["player"]["cmsBio"]
    bio = BeautifulSoup(bio_html, "html.parser")
    career = next((container for container in bio.select(".cplayer-bio__container")
                   if container.select_one(".cplayer-bio__title") is not None
                   and container.select_one(".cplayer-bio__title").get_text(strip=True) == "PROFESSIONAL CAREER"), None)
    if career is None:
        raise ValueError("Curry archived bio has no professional career section")
    text = career.select_one(".cplayer-bio__content").get_text(" ", strip=True)
    if ("Four-time NBA Champion (2015, 2017, 2018, 2022)" not in text
            or "Two-time NBA Most Valuable Player (2014-15, 2015-16)" not in text):
        raise ValueError("Curry historical bio lacks expected dated award statements")
    source["raw_html_path"] = html_path.relative_to(ROOT).as_posix()
    source["historical_bio_scope"] = "Historical NBA CMS biography includes achievements through the 2023-24 entry point; current-page 2026 panels are separate."
    source["excerpt_windows"].append({
        "text": text,
        "json_path": "$.props.pageProps.player.cmsBio",
        "html_section": "PROFESSIONAL CAREER",
        "source_html_sha256": actual_sha,
        "cms_bio_sha256": hashlib.sha256(bio_html.encode("utf-8")).hexdigest(),
        "extraction": "HTML decoded from exact archived NEXT_DATA JSON",
        "scope": "Four championships in 2015/2017/2018/2022 and MVP seasons 2014-15/2015-16 precede 2025. Do not use current-page panels as the date support.",
    })
    return {"entity": "Stephen Curry", "dimension": "historical_dated_career_achievements",
            "claim": "NBA historical CMS bio explicitly records four championships (2015, 2017, 2018, 2022) and two NBA Most Valuable Player seasons (2014-15, 2015-16).",
            "source_ids": ["nba_curry_profile"], "source_html_sha256": actual_sha,
            "evidence_json_path": "$.props.pageProps.player.cmsBio"}


def _facts(matrix: dict) -> list[dict]:
    return [{key: fact[key] for key in ("entity", "dimension", "claim", "source_ids")}
            for fact in matrix.get("facts", [])]


def build(output: Path) -> dict:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    fetch_rows = json.loads(FETCH.read_text(encoding="utf-8"))
    sources = _merge_sources(matrix, fetch_rows)
    _append_raw_window(sources, "kobe_hof", "By The\nNumbers", "Related Hall of Famers",
                       scope="Career achievements: five NBA titles; 2008 NBA MVP; two Finals MVPs")
    _append_raw_window(sources, "shaq_hof", "By The\nNumbers", "Related Hall of Famers",
                       scope="Career achievements: four NBA titles; 2000 NBA MVP; three Finals MVPs")
    _append_raw_window(sources, "nba_shaq_archive", "Don\u2019t let the size fool you.", "For three straight years",
                       scope="Distinct off-court entertainment identity; not just nickname/dance-move description")
    _append_raw_window(sources, "nba_jordan_legend", "But trouble was brewing.", "Emotionally drained",
                       scope="Contemporary gambling scrutiny; no conviction or adjudicated legal finding stated")
    _append_raw_window(sources, "nba_jordan_legend", "He was also known to be ultra-demanding", "But winning was the soothing elixir.",
                       scope="Concrete negative interpersonal effect: teammates' feathers ruffled by critiques")
    # Kobe's archive is absent from the matrix because the earlier fact-list
    # deduplicated to HoF; retain its real source and explicit Mamba philosophy.
    if not any(row["source_id"] == "nba_kobe_archive" for row in sources):
        fetched = next(row for row in fetch_rows if row["source_id"] == "nba_kobe_archive")
        sources.append({"source_id": "nba_kobe_archive", "title": fetched["title"],
                        "date": _body_date((RAW / "nba_kobe_archive.txt").read_text(encoding="utf-8")),
                        "url": fetched["final_url"], "final_url": fetched["final_url"], "status": "ok",
                        "http_status": fetched["http_status"], "raw_text_path": (RAW / "nba_kobe_archive.txt").relative_to(ROOT).as_posix(),
                        "content_hash_sha256": fetched["content_hash_sha256"],
                        "fetch_duration_seconds": fetched["fetch_duration_seconds"], "excerpt_windows": []})
    _append_raw_window(sources, "nba_kobe_archive", "Get into the mind of Kobe", "Related",
                       scope="Mamba mentality is Kobe's leadership/winning philosophy")
    dated_curry = _append_curry_historical_bio(sources)
    facts = _facts(matrix)
    facts.append(dated_curry)
    derived = {
        "source_matrix_version": matrix.get("schema_version"),
        "facts": facts,
        "arithmetic_template": {
            "status": "template_only; host computes totals at runtime",
            "bounded_range": [0, 10],
            "inputs": ["championships", "regular_season_mvps", "audience", "current_generation", "next_generation", "social_cultural"],
            "normalization": {
                "championships": "championships / observed_max_championships * 5",
                "regular_season_mvps": "regular_season_mvps / observed_max_regular_season_mvps * 5",
                "court_stat_score_0_10": "normalized_championships + normalized_mvps",
            },
            "weighted_formula": "0.30*court_stat_score + 0.20*audience + 0.15*current_generation + 0.15*next_generation + 0.20*social_cultural",
            "winner_rule": "host recomputes every displayed operand and applies argmax; this receipt does not provide canonical totals",
        },
        "comparison_rows": [
            {"player": player, "source_ids": source_ids,
             "claim_scope": "Use source-backed championship/MVP facts; host supplies numeric values and totals."}
            for player, source_ids in (
                ("Michael Jordan", ["nba_jordan_legend"]),
                ("LeBron James", ["nba_lebron_archive"]),
                ("Magic Johnson", ["nba_magic_legend"]),
                ("Kobe Bryant", ["kobe_hof", "nba_kobe_archive"]),
                ("Stephen Curry", ["nba_curry_profile", "nba_curry_archive", "nba_champs_history", "ua_curry_tour"]),
                ("Shaquille O'Neal", ["shaq_hof", "nba_shaq_archive"]),
            )
        ],
        "policy": [
            "Keep exactly the six listed players and preserve a coverage matrix: two concrete basketball facts plus one distinct on-court and one off-court trait per player.",
            "Use only claim-specific source IDs; never use blanket [GK] or a future verification list as a citation.",
            "Keep observations within the requested up-to-2025 scope. Curry's NBA profile bio is a historical bio embedded in the archived page; current news/awards panels remain 2026-only.",
            "Do not call the score objective, unbiased, definitive, or debate-ending. The winner is conditional on the declared weights, bounded qualitative inputs and source scope; retain material uncertainty and alternative weighting outcomes.",
            "Include at least one concrete negative or controversial social-impact example with careful attribution and no unsupported legal conclusion. Jordan gambling is described only as contemporary scrutiny in the supplied NBA profile; do not state a conviction or adjudicated finding.",
            "Kobe's Mamba mentality belongs to Kobe; Shaq's low-post dominance and entertainer role belong to Shaq. Do not transfer traits between players.",
        ],
    }
    successful = [row for row in fetch_rows if row.get("http_status") == 200]
    failed = [row for row in fetch_rows if row.get("http_status") != 200]
    preparation = {
        "raw_manifest_path": str(FETCH),
        "raw_manifest_sha256": digest(fetch_rows),
        "raw_44_request_count": len(fetch_rows),
        "successful_http_requests": len(successful),
        "failed_http_requests": len(failed),
        "failed_statuses": {str(status): sum(row.get("http_status") == status for row in failed)
                            for status in sorted({row.get("http_status") for row in failed})},
        "additional_targeted_fetches": ADDITIONAL_TARGETED_403,
        "additional_targeted_403_count": len(ADDITIONAL_TARGETED_403),
        "exploratory_followups": EXPLORATORY_FOLLOWUPS,
        "exploratory_followup_count": len(EXPLORATORY_FOLLOWUPS),
        "bytes": sum(row.get("bytes", 0) for row in fetch_rows),
        "fetch_duration_seconds": sum(row.get("fetch_duration_seconds", 0) for row in fetch_rows),
        "raw_scope": "HTTP source preparation only; not model API usage or cost",
    }
    payload = {
        "schema_version": "nba-source-supplement-v4",
        "created_at": utc_now(),
        "matrix_path": str(MATRIX),
        "matrix_sha256": digest(matrix),
        "tasks": {TASK_ID: {"sources": sources, "derived_calculations": derived}},
        "preparation": preparation,
        "provenance": {
            "v3_receipt_unchanged": True,
            "failed_fetches_retained_separately": True,
            "matrix_facts_only_from_http_200_sources": True,
            "no_model_api_calls": True,
            "canonical_numeric_totals": "not included; host computes from source-bound rows",
        },
    }
    write_json(output, payload)
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("C:/J/nba_source_supplement_v4.json"))
    args = parser.parse_args()
    result = build(args.output)
    task = result["tasks"][TASK_ID]
    print(json.dumps({"output": str(args.output), "sources": len(task["sources"]),
                      "raw_44_requests": result["preparation"]["raw_44_request_count"],
                      "additional_403": result["preparation"]["additional_targeted_403_count"]}, ensure_ascii=True))
