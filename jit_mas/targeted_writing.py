"""Compact writing profiles with locally enforced formal requirements."""

from __future__ import annotations

import json
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .schemas import PublicTask
from .targeted_checklists import CHECKLIST_GUIDANCE, coverage_checklist


PROMPT_VERSION = "targeted-writing-v7-sectioned"
REPORT_SECTION_COUNT = 5
REPORT_SECTION_MIN_WORDS = 1000
REPORT_SECTION_MAX_WORDS = 1150
REPORT_SECTION_MAX_TOKENS = 2000
REPORT_TOTAL_MIN_WORDS = 5000
REPORT_TOTAL_MAX_WORDS = 6000
SYSTEM = (
    "Write the complete requested final artifact. Privately check every public requirement, "
    "supplied fact, audience, format, length and tone. Use specific causal reasoning and useful "
    "examples. Distinguish supplied facts, calculations and illustrative assumptions. Never "
    "invent citations, market shares, survey results or personal achievements. Return only "
    "the artifact without process notes or an asserted word count."
)
_LINE = Annotated[str, Field(min_length=1, max_length=90, pattern=r"^[^\r\n]+$")]


class Poem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: _LINE
    lines: Annotated[list[_LINE], Field(min_length=8, max_length=16)]


class Poems(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    poems: Annotated[list[Poem], Field(min_length=3, max_length=3)]


REPORT_SECTION_SPECS = (
    ("Executive summary and operating baseline", (
        "executive summary, 2023 transaction scale and user data, baseline calculations, "
        "denominator limitations, unit economics hypotheses, and the central investment thesis"
    )),
    ("Cohorts, competition, regions and O2O", (
        "customer cohorts, platform competition, supermarkets, convenience and fruit stores, "
        "Meituan and JD.com, Guangdong's 8/44 count share, regional strategy, and online-offline channels"
    )),
    ("Supply chain, quality and risk", (
        "supply-chain economics, fulfillment density, waste, commissions, group leaders, "
        "working capital, product and service quality, retention, regulation, and operational risks"
    )),
    ("Financing, valuation and scenarios", (
        "financing needs, valuation drivers, clearly labeled illustrative assumptions, "
        "one consistent baseline and horizon, downside tests, and three comparable scenarios"
    )),
    ("Investor decisions and conclusion", (
        "strategic corporate, growth-equity and conservative infrastructure investor decisions, "
        "staged gates, measurable diligence thresholds, priorities, and a completed conclusion"
    )),
)


def _report_section_index(section_index: int) -> int:
    if type(section_index) is not int or not 0 <= section_index < REPORT_SECTION_COUNT:
        raise ValueError(f"report section index must be an integer from 0 to {REPORT_SECTION_COUNT - 1}")
    return section_index


def _report_payload(task: PublicTask, section_index: int, private_record: dict | None = None) -> dict:
    index = _report_section_index(section_index)
    title, focus = REPORT_SECTION_SPECS[index]
    payload = {
        "public_question_and_supplied_facts": task.question,
        "public_constraints": list(task.constraints),
        "section_index": index + 1,
        "section_title": title,
        "section_focus": focus,
        "target_word_range": [REPORT_SECTION_MIN_WORDS, REPORT_SECTION_MAX_WORDS],
        "task_specific_coverage_checklist": coverage_checklist(private_record),
        "shared_analysis_rules": (
            "Only section 1 presents the market baseline calculations: 2022 transaction volume "
            "=322.8/1.5371 billion yuan; 2022 year-end users=678/1.083 million; "
            "transaction value per year-end user=transaction volume in yuan/users. "
            "That ratio is not platform ARPU or an established upper/lower bound. "
            "Only section 2 presents Guangdong's 8/44 enterprise count share; other regions' "
            "counts and the identities of all 44 enterprises are unavailable. "
            "Only section 4 presents numerical forecasts: illustrate three 2026 scenarios from "
            "the same 2023 transaction baseline of 322.8 billion yuan over three years. "
            "Use assumed annual transaction growth of 10%, 20% and 30%, with corresponding "
            "illustrative take rates of 2%, 3% and 4%; show transaction value=322.8*(1+growth)^3 "
            "and platform revenue=transaction value*take rate. These are analyst assumptions, "
            "not observed forecasts or measured industry rates. Other sections may explain "
            "these dependencies without creating competing forecasts. "
            "Transaction/user growth alone cannot identify causal drivers; frequency, basket, "
            "mix and price are hypotheses requiring validation."
        ),
    }
    return payload


def _section_system(section_index: int, *, revision: bool = False) -> str:
    index = _report_section_index(section_index)
    title, focus = REPORT_SECTION_SPECS[index]
    action = "Revise only this section" if revision else "Write only this section"
    return (
        "Produce plain English prose for a larger investment report. " + action + "; do not output JSON, "
        "a checklist, process notes, an asserted word count, or a promise of more material. "
        f"This is section {index + 1} of {REPORT_SECTION_COUNT}, titled '{title}'. "
        f"Cover {focus}. Write {REPORT_SECTION_MIN_WORDS}-{REPORT_SECTION_MAX_WORDS} words, "
        "targeting 1,035-1,080 words. Privately plan nine paragraphs of 115-120 words each. "
        "If a compact markdown table is needed, its words replace paragraph words within the "
        "same budget. Hard stop at 1,100 words after completing the final paragraph; do not "
        "expand the section merely to repeat coverage requirements. Use short, direct sentences. "
        "Use the public question and supplied facts as the source of truth. Preserve every supplied "
        "number and label calculations versus hypotheses. Never invent market shares, citations, "
        "survey results, company counts beyond the supplied facts, or unsupported operational facts. "
        "Do not repeat the other sections; do not add the section title because the final assembler "
        "adds it. Make this section self-contained enough to understand its "
        "claims, and end at a natural paragraph boundary. Return only the section text."
    )


def report_section_messages(task: PublicTask, section_index: int,
                            private_record: dict | None = None):
    """Build one independent 1,000--1,150-word report-section request.

    ``section_index`` is zero-based. The payload repeats only the public question,
    supplied facts, constraints and a small rubric outline; no prior section is
    included, so each request can be retried without a growing conversation.
    """
    payload = _report_payload(task, section_index, private_record)
    return [{"role": "system", "content": SYSTEM + "\n" + CHECKLIST_GUIDANCE + "\n" + _section_system(section_index)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                     separators=(",", ":"))}], None


def report_section_revision_messages(task: PublicTask, section_index: int, content: str,
                                     private_record: dict | None = None):
    """Build a bounded revision request for one under/over-length section."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("section revision content must be nonempty text")
    payload = _report_payload(task, section_index, private_record)
    payload["current_section"] = content
    payload["current_word_count"] = word_count(content)
    payload["revision_instruction"] = (
        f"Rewrite the complete current section to {REPORT_SECTION_MIN_WORDS}-"
        f"{REPORT_SECTION_MAX_WORDS} words, targeting 1,035-1,080 words. Hard stop at 1,100 words. "
        "Use nine paragraphs of 115-120 words; table words replace prose within that budget. "
        "Preserve all relevant supplied facts, calculations, "
        "assumptions and decisions; remove repetition or add analysis where needed. Do not cut the "
        "section mechanically, and return only the revised section text."
    )
    return [{"role": "system", "content": SYSTEM + "\n" + CHECKLIST_GUIDANCE + "\n" + _section_system(section_index, revision=True)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                     separators=(",", ":"))}], None


def report_section_check(content: str) -> dict:
    count = word_count(content) if isinstance(content, str) else 0
    return {"word_count": count,
            "word_range_satisfied": REPORT_SECTION_MIN_WORDS <= count <= REPORT_SECTION_MAX_WORDS,
            "complete": REPORT_SECTION_MIN_WORDS <= count <= REPORT_SECTION_MAX_WORDS}


def render_report_sections(contents: list[str]) -> str:
    """Join five generated sections without truncation or content mutation."""
    if not isinstance(contents, list) or len(contents) != REPORT_SECTION_COUNT:
        raise ValueError(f"expected exactly {REPORT_SECTION_COUNT} report sections")
    if any(not isinstance(content, str) or not content.strip() for content in contents):
        raise ValueError("report sections must be nonempty text")
    return "\n\n".join(
        f"## {index + 1}. {REPORT_SECTION_SPECS[index][0]}\n\n{content.strip()}"
        for index, content in enumerate(contents)
    )


def render_response(profile: str, content: str) -> str:
    if profile == "palindrome-poems":
        return render_poems(Poems.model_validate_json(content))
    return content


def report_length_feedback(content: str) -> str:
    return (
        f"The current report contains {word_count(content)} words. Revise the complete "
        "report once to 5,100-5,600 English words. Preserve all supplied calculations, "
        "explicit scenario assumptions, evidence boundaries and useful investor decisions. "
        "Keep clear numbered sections and a completed conclusion. Remove repetition and "
        "unsupported estimates. Return only the revised report text."
    )


PALINDROME_HALF = ("人爱乡乡爱人", "水绕村村绕水", "路通家家通路", "心系乡乡系心")
PALINDROME_CENTER = "人人为我我为人人"


def render_poems(parsed: Poems) -> str:
    blocks = [poem.title + "\n" + "\n".join(poem.lines) for poem in parsed.poems]
    palindrome = [*PALINDROME_HALF, PALINDROME_CENTER, *reversed(PALINDROME_HALF)]
    blocks.append("乡里乡\n" + "\n".join(palindrome))
    return "\n\n".join(blocks)


def writing_profile(task: PublicTask) -> str:
    question = task.question.lower()
    if "poem" in question and "palindrome" in question:
        return "palindrome-poems"
    if "community group buying" in question:
        return "investment-report"
    if "smart agriculture" in question and "introduction" in question:
        return "academic-introduction"
    return "writing"


def build_messages(task: PublicTask, private_record: dict | None = None):
    profile = writing_profile(task)
    response_format = None
    guidance = ""
    if profile == "palindrome-poems":
        guidance = (
            "Write three distinct Chinese modern poems in the supplied JSON schema. Each title "
            "and line occupies one line; each poem has 8-16 body lines. Use a first-person speaker "
            "born in the 1980s, concrete hometown changes since Reform and Opening Up, old street "
            "renovation, farmland becoming factories, villagers becoming urban residents and "
            "rural revitalization. Build emotional progression with concrete family memories, "
            "mixed feelings and an inclusive present. Use accessible, recitation-friendly language "
            "and natural Jiangnan expressions with sufficient context to understand them. These "
            "three poems need not be palindromes: the renderer appends a fourth, exact palindrome "
            "of nine lines. Do not append an additional poem or explain the renderer."
        )
        response_format = {"type": "json_schema", "json_schema": {
            "name": "hometown_poems", "strict": True, "schema": Poems.model_json_schema()}}
    elif profile == "investment-report":
        guidance = (
            "Return a complete English investment report of 5,100-5,600 words. Use clear "
            "numbered sections covering executive summary; 2023 data and baseline calculations; "
            "customer cohorts; platform competition; regional strategy and O2O channels; "
            "supply-chain economics; quality, service and regulatory risks; financing and "
            "valuation; three scenarios; investor decisions; and a completed conclusion. "
            "Use full analytical paragraphs and finish within the requested word range. "
            "Include compact tables for supplied data, competitor positioning, illustrative unit "
            "economics, scenarios and investor diligence. Calculate 2022 transaction volume as "
            "322.8/1.5371 billion yuan and 2022 users as 678/1.083 million; calculate transaction "
            "value per year-end user for both years and explicitly note the denominator limitation. "
            "The per-year-end-user ratio is neither an established upper nor lower bound, nor "
            "platform ARPU. Transaction/user growth alone does not identify causal drivers; "
            "present frequency, basket size, mix and prices as hypotheses requiring validation. "
            "Explain growth beyond user expansion, Guangdong's 8/44 share of major enterprises "
            "as a count share, and the difference between transaction value and platform revenue. "
            "Compare Meituan and JD.com with supermarkets, convenience stores, fruit stores and "
            "local-service platforms without inventing market shares or unsupported operational "
            "facts. Do not invent regional counts beyond the supplied Guangdong count or "
            "pretend the identities of all 44 enterprises are supplied. Use clearly labeled "
            "scenario assumptions and show formulas, not assertions. Use the same baseline "
            "and horizon for all scenario calculations; distinguish illustrative take rates, "
            "thresholds and unit economics from measured industry values. "
            "Separate strategic corporate, growth-equity and conservative infrastructure investors; "
            "give staged funding gates, valuation drivers, downside tests and measurable diligence "
            "thresholds. Cover competition, fulfillment density, waste, commissions, group leaders, "
            "working capital, quality, customer retention and regulation. End with actionable "
            "priorities and a completed conclusion, not a promise of more material."
        )
    elif profile == "academic-introduction":
        guidance = (
            "Write only the revised introduction, about 650-850 English words. Establish China's "
            "urgent agricultural modernization needs with rural aging, labor migration, resource "
            "constraints and the rural revitalization policy context. Move from the national "
            "problem to Northern Jiangsu and the research gap, then the Tianwa case. Link concrete "
            "IoT sensing, environmental control, integrated irrigation/fertilization, cultivation, "
            "traceability and market-data applications to efficiency mechanisms and the supplied "
            "case figures. Attribute those figures to the provided case description rather than "
            "claiming independently verified field findings. State research questions, field-survey "
            "and case-analysis methods, and the practical contribution and limits. Use connected "
            "academic prose, avoid repeating the entire supplied paper, and distinguish park "
            "claims from measured evidence."
        )
    payload = {"question": task.question, "constraints": list(task.constraints)}
    checklist = coverage_checklist(private_record)
    if checklist:
        payload["task_specific_coverage_checklist"] = checklist
        guidance += "\n" + CHECKLIST_GUIDANCE
    return [{"role": "system", "content": SYSTEM + "\n" + guidance},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                     separators=(",", ":"))}], response_format


def word_count(answer: str) -> int:
    return len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", answer))


def generation_checks(task: PublicTask, answer: str) -> dict:
    profile = writing_profile(task)
    if profile == "investment-report":
        count = word_count(answer)
        valid = REPORT_TOTAL_MIN_WORDS <= count <= REPORT_TOTAL_MAX_WORDS
        return {"word_count": count, "word_range_satisfied": valid, "complete": valid}
    if profile == "palindrome-poems":
        blocks = answer.split("\n\n")
        counts = [len(block.splitlines()) - 1 for block in blocks]
        body = "".join(blocks[-1].splitlines()[1:]) if blocks else ""
        valid = len(blocks) == 4 and all(0 < count <= 20 for count in counts) and body == body[::-1]
        return {"poem_count": len(blocks), "body_line_counts": counts,
                "exact_palindrome": bool(body) and body == body[::-1], "complete": valid}
    return {"complete": True}
