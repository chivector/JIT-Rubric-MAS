"""Apply a local source-enrichment patch to the frozen v6 ESG report."""

from __future__ import annotations

import argparse
from decimal import Decimal
import difflib
import hashlib
import json
from pathlib import Path
import re

EXPECTED_INPUT_SHA256 = "4a30cfc25d40e44455210217ee3b5df102ddff7141ad7ffa3df55dffa8346f69"

SOURCE_SLICES = [
    ("hm2024", "23d99aa73f3bcb101dc0300d3d6786734bf6ad9b2443b5b801d95787908d6846.json", "H&M Group Annual and Sustainability Report 2024", "March 2025; FY2024", [(5632, 6208), (62350, 62900), (77930, 79170), (252648, 255130), (653450, 654060), (220400, 223450), (261650, 263660), (342208, 344030), (355186, 357105), (366430, 367545)]),
    ("patagonia2025", "3e95fb3494d634be2c9a858e9d0b01642c5e682aede949a1bfaaf2fcc40ab2da.json", "Patagonia Work in Progress Report 2025", "November 2025; FY25", [(11378, 12180), (27600, 31400), (98850, 99500), (100230, 100850), (118820, 120130), (138100, 139100), (158050, 159150), (143900, 145200), (146300, 149300)]),
    ("frfinance2024", "375298a22439c0479b2f2125ed265a1f22f8a2bad2de7f5e12f8d47e2103080e.json", "Fast Retailing Integrated Report 2024", "February 2025 publication; FY2024", [(2450, 2950), (16640, 17120), (142550, 142980), (171940, 173140), (128728, 129010), (84800, 85800)]),
    ("fr2024", "241320156d3a67a8e635a0b260cffad2ca9da496d02a54266de86a23149ec2c3.json", "Fast Retailing Integrated Report 2024 sustainability chapter", "FY2024", [(4700, 5430)]),
    ("frclimate2025", "26519701c145f49104dc3433f5518b684d155c94e852642298a498856d831145.json", "Fast Retailing Response to Climate Change", "19 March 2026 update; FY2022-FY2025", [(9992, 10920), (13320, 17810)]),
    ("frenergy", "4ac9d9689e29fea23bfb495250b57c8c25b6a8e44d2ea90093302e0f9cc26543.json", "Fast Retailing Improved Energy Efficiency", "19 March 2026 update; FY2022-FY2025", [(11027, 11915)]),
    ("frratings", "71637e19c67292d22720051907709e0ed687bbd4c31986a85b8ccdfe24244538.json", "Fast Retailing Evaluation by Society", "3 September 2026 update", [(3977, 5500), (5800, 6400), (7483, 8190)]),
    ("hmslavery", "78ce926c323555f2c78b2a0d48248ffe99d6395d97b221724612471de2d67d15.json", "H&M Group Modern Slavery Statement 2024", "2024 reporting year; 2025 publication", [(13300, 14770), (20285, 23200)]),
    ("fla2025", "7c68fd3f39323cadb1e9ed3ff9a84506a5ca50c7234730d95a17b222a2bb1061.json", "Patagonia Post-Accreditation Report 2025", "September 2025", [(6970, 7570)]),
    ("bcorp", "631760007224248c14d38b5cb61529a9dddb7f34c0ae90bd0b2143824d0811c1.json", "B Lab Patagonia Certified B Corporation assessment", "Current archived page; links to 2024 disclosures", [(2700, 4677)]),
    ("blabdisclosure", "e78510eba19c411c3543c33d5d92f394c16efbec507a6a498decb80ef0b128bc.reextracted.json", "Patagonia Works Disclosure Report", "Submitted December 2023; B Lab", [(0, 700), (4850, 6080), (10390, 11300)]),
    ("hmwff", "8ad4dcdcc5779ad831daa37c87f247778031afd2ab8a6e42f31a653caccb46d2.json", "H&M ranks 1st in Fashion Revolution's What Fuels Fashion? Report", "25 September 2025", [(0, 1967)]),
    ("fti2023", "a0812aacf1ea74115a5bc7112df71d2e9c122fbbb7c8938f3e7844c4d6d0c1fa.json", "Fashion Transparency Index 2023 findings", "2023", [(2280, 3260), (4700, 5500)]),
]

LEGAL = """## Legal entities and reporting boundaries

H & M Hennes & Mauritz AB is the legal entity described as H&M Group/H&M. Ramsbury Invest AB is formally its parent; Stefan Persson and family through Ramsbury/private holdings held 81.9% of votes and 61.8% of shares at 30 November 2024, excluding treasury shares. [hm2024]

Patagonia, Inc. sits within Patagonia Works, the holding company for Patagonia, Inc. and GPIW. In 2022 ownership transferred to Patagonia Purpose Trust (all voting stock, 2% of total) and Holdfast Collective (all non-voting stock, 98%). The Trust protects the company's purpose and controls governance. FY25 reporting excludes Patagonia Provisions and Fletcher Chouinard Designs. [patagonia2025]

UNIQLO Co., Ltd. is named in the report; Fast Retailing Co., Ltd. is the holding/parent company of the reporting group, whose brands include UNIQLO, GU and Theory. The excerpts support group affiliation but do not establish the exact subsidiary ownership percentage. [frfinance2024]
"""

ADDITIONS = """## Additional archived evidence

### Common reporting year and stronger climate boundaries

Scope1 means direct operations emissions; Scope2 purchased electricity/heat/steam (market and location accounting differ); Scope3 other value-chain emissions. FY2024 is the common label below, not identical calendar periods. Patagonia ends April, FR August and H&M November. Amounts are tCO2e. [hm2024; patagonia2025; frclimate2025]

| FY2024 | Scope1 | Scope2 location | Scope2 market | Scope3 |
|---|---:|---:|---:|---|
| H&M | 15,102 | 364,693 | 26,553 | 8,729,000 including use-phase; SBT exclusion 6,955,000 |
| Patagonia | 1,363 | 3,711 | 117 | 176,955 |
| FR Group | 8,760 | 297,360 | 43,154 | 5,198,890 calculated sum of disclosed categories |

FR FY24 Scope3 sum = 3,630,293+6,392+644,578+87,429+14,680+54,031+475+759,664+1,348. Categories marked not relevant or '-' contribute no reported quantity; the UNIQLO/GU target subset 3,389,624 is already contained in Category1 and is not added again. The sum is analyst-calculated from all disclosed categories, not a claimed source-published total. FY25 Scope1/location-Scope2/market-Scope2 are 9,301/301,434/20,906. FY25 categories marked with a star are assured by KPMG AZSA Sustainability Co., Ltd. Revised factors from FY22 and FY25 leased-asset/factor changes affect trend comparability. [frclimate2025]

Matched reported net revenue: H&M FY22/23/24 = 223,553/236,035/234,478 MSEK; Patagonia FY23/24/25 = 1,568/1,500/1,470 million USD (rounded); FR FY22/23/24 = 2,301,122/2,766,557/3,103,836 million JPY. No FX or inflation conversion. Patagonia FY24/25 Scope1 intensities .908667/.908163, market-Scope2 .078000/.063265, and Scope3 117.970000/123.277551 tCO2e/million USD: emissions growth concentrates in Scope3, which rises 4.50%. FY23 emissions remain unavailable; no continuous three-year series is claimed. [hm2024; patagonia2025; frfinance2024]

FR FY22/23/24 separate Scope1 intensities .004232/.003455/.002822, market-Scope2 .069117/.030906/.013903, and calculated all-disclosed-Scope3 2.494814/1.957650/1.674989 tCO2e/million JPY. Own-operation Scope1 and market-Scope2 decline; value-chain intensity also declines, subject to factor/boundary caveats. [frclimate2025; frfinance2024]

### Resources, materials and waste

Renewable share FY22/23/24/25: FR electricity 42.4/67.6/84.7/93.5%; yearly changes +25.2/+17.1/+8.8 percentage-points. H&M FY22/23/24 electricity 92/94/96% (+2/+2 points); Patagonia FY23/24/25 total-energy renewable 38/60/59% (+22/-1 points). Electricity and total energy differ, so the figures are not a shared cross-company measure. [frenergy; hm2024; patagonia2025]

H&M uses WWF's Water Risk Filter, contextual basin targets, water meters and functional effluent-treatment plants; withdrawal totals, water-stress volume shares and per-unit intensities remain unavailable. Patagonia/FR quantitative water footprints and comparable risk scores remain unavailable. H&M's recycled-water shares +3.8/+3.6 points FY22-FY24 show reuse progress, but a percentage alone cannot establish absolute watershed impact. [hm2024]

H&M defines recycled/sustainably sourced materials as meeting its sourcing requirements and aligning with Textile Exchange's preferred-fiber definition: consistently reduced impacts/increased benefits for climate, nature and people versus conventional equivalents. Reported commercial material percentages apply to shell fabric, excluding fillings, linings and trims; waste-weight-factor updates affect historical comparability. Total share FY22/23/24 is 82/83/89%; recycled 24/27/29.5%; recycled polyester 75/79/94%. Certifications include organic/recycled/Better Cotton, RMS, RWS, GCS and FSC/PEFC. Complete linked sourcing rules are not in these excerpts. [hm2024]

Patagonia's official preferred definition adds a holistic transformation of production systems to the Textile Exchange reduced-impact/increased-benefit standard, with most preferred materials backed by rigorous third-party social/animal/environmental certifications. Its chart labels 2023/24/25 preferred materials 81.4/83.2/84.1%, including fabrics and trims/BOM weight; yearly improvement is +1.8/+0.9 points. These boundaries differ from H&M shell-only reporting. [patagonia2025]

FR defines lower-impact classification through qualitative/quantitative GHG, water, biodiversity, human-rights and animal-welfare standards and aims to identify the materials best meeting them; the complete finalized taxonomy is unavailable. The 2024 report says 18.2% overall/47.4% polyester, while the current climate page says FY24 15.9/41.5 and FY25 19.4/46.4%, using previous Fall/Winter/current Spring/Summer collections. Preserve the different vintages/definitions instead of silently joining them. Three comparable annual FR material observations remain unavailable. [fr2024; frclimate2025]

H&M own-operation packaging reuse/recycling FY22/23/24 is 92/90/93%, disposal 8/10/7%. FY24 product waste is .7% of commercial-product weight, 95% reused/recycled, but prior two years are N/A. Packaging includes distribution-centre transport, return, product and pallet materials. Comparable three-year Patagonia/FR waste-diversion data are unavailable; Scope3 Category5 tCO2e is not waste tonnage. [hm2024]

### Wages, forced labor, grievances and disclosed cases

Patagonia's ten-year roadmap aimed at living wages in every factory by 2025; the report expects not to reach 100%. The 2024 39% attainment figure concerns in-scope factories, not all 65,000+ workers individually; Tier1/Tier2 worker splits and verified wage-program coverage counts are unavailable. H&M seeks normal-hour wages meeting basic needs/discretionary income via wage management, collective bargaining and responsible purchasing; a comparable time-bound all-factory attainment target/verified tier worker counts are unavailable. FR commits to living wages in its supply chain, but deadline and verified tier worker coverage are unavailable. [patagonia2025; hm2024; frfinance2024]

H&M prohibits cotton from Turkmenistan and Syria; this is a raw-material restriction, not a whole-country ban on every sourcing category. Comparable Patagonia/FR country-wide blacklist disclosures are unavailable. H&M disclosed four forced-labor cases FY24 (three forced overtime, one uniform deposit requiring one-year service; Tier1 three/Tier2 one), versus three FY23/one FY22. Suppliers refunded deposits, made overtime voluntary with individual consent and strengthened grievance systems; H&M reports resolution and ongoing follow-up. This differs from its NMC channel's zero forced-labor complaints and cannot be generalized to no forced labor. [hmslavery]

Patagonia describes recruitment-fee audits and leading fabric suppliers eliminating new fees by 2020, saving approximately 3,000 workers $1.7m annually; a 2022 Thailand/Taiwan survey of over 1,600 workers found 3% still owed fees. FY24 workweek averaged 54 hours with nine noncompliance instances (five excessive monthly overtime, one missing weekly rest, three inadequate breaks); each had remediation/long-term solutions. These are corporate reports, not independent findings. FR FY24 six factories had zero-tolerance violations, three confirmed improvements; issue categories include forced/child labor, without claiming all six involved forced labor. [patagonia2025; frfinance2024]

H&M Speak Up allows anonymity subject to local law and prohibits retaliation; FSLM-scope grievance mechanisms FY24 cover 100% Tier1/99.61% Tier2, not all-factory coverage. NMC outcomes: seven unresolved from FY23, fourteen new, twenty resolved, one carried forward; resolving 20/21 available cases is 95.2381%, not 20/14 new cases. Multilingual/no-cost access, substantiation share and closure days are unavailable. Patagonia has multiple channels including confidentiality, but FLA recommends follow-up timelines; tier coverage/accessibility/outcomes are unavailable. FR accessible channel operation, tier coverage and outcome counts remain unavailable. [hm2024; fla2025]

B Lab's December 2023 company-disclosure report describes Patagonia's California wage/hour class-action allegation, disputed by the company and settled to avoid litigation costs; settlement was less than 1% of revenue and court approval was pending at publication. This is B Lab-published company disclosure, not an independent investigation or an exact cash fine. Reported timekeeping review is the response. Comparable qualifying external H&M/FR controversy financial consequences and source-backed stock/sales attribution are unavailable. [blabdisclosure]

### External ratings and triangulation

FR's updated company page reports February2025 Sustainalytics LowRisk (numeric0-100 score unavailable), 2025 CDP climate/water A and forest B, and ISS Prime since2024; MSCI index inclusion is not an MSCI alphabetical ESG rating. H&M's25September2025 company announcement reports What Fuels Fashion2025 rank1/200 and71%, explicitly a decarbonization-transparency score, not FTI or overall ethical performance. B Lab's Patagonia profile reports166.0 overall with governance17.8/workers23.7/community79.2/environment40.4/customers4.7; 2020score151.4 is historical. Its current assessment issue month is unavailable. [frratings; hmwff; bcorp]

FTI2023's archived methodology evaluates public disclosure; it is older than the requested current/previous-year vintage and supplies no full three-brand/five-category scores here. For all brands, current MSCI grade/date, complete Sustainalytics score/category and current FTI overall/five-section scores remain incomplete. Ratings measure different constructs/vintages: BCorp system impact, FLA labor management, Sustainalytics unmanaged financial risk, CDP environmental disclosure, and WFF decarbonization transparency. They do not justify pooling point scores. No frozen analyst score is recalibrated from these additions; rating divergence remains partly untestable rather than invented. [fti2023; frratings; hmwff; bcorp; fla2025]
"""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_additions(archive):
    sources = []
    for source_id, filename, title, date, spans in SOURCE_SLICES:
        receipt = read_json(archive / filename)
        source_text = receipt["text"]
        observed_hash = sha256(source_text)
        if receipt.get("text_sha256") and observed_hash != receipt["text_sha256"]:
            raise ValueError("Archived source hash mismatch: " + filename)
        if any(not 0 <= start < end <= len(source_text) for start, end in spans):
            raise ValueError("Invalid archived source slice: " + filename)
        sources.append({"source_id": source_id, "title": title, "date": date,
            "url": receipt["url"], "archive_locator": filename,
            "source_text_sha256": observed_hash,
            "sections": [{"span_start": start, "span_end": end,
                          "text": source_text[start:end], "sha256": sha256(source_text[start:end])}
                         for start, end in spans]})
    return sources


def calculate_checks():
    revenue = [Decimal("2301122"), Decimal("2766557"), Decimal("3103836")]
    scope1 = [9738, 9558, 8760]
    scope2 = [159047, 85502, 43154]
    scope3 = [[4243676, 24815, 552711, 83335, 14822, 54554, 0, 764228, 2731],
              [3977760, 15536, 503393, 97879, 14891, 54809, 0, 750291, 1391],
              [3630293, 6392, 644578, 87429, 14680, 54031, 475, 759664, 1348]]
    return {"fr_scope3_totals_FY22_23_24": [sum(row) for row in scope3],
        "fr_scope1_intensity": [str(Decimal(value) / denominator) for value, denominator in zip(scope1, revenue)],
        "fr_market_scope2_intensity": [str(Decimal(value) / denominator) for value, denominator in zip(scope2, revenue)],
        "fr_all_disclosed_scope3_intensity": [str(Decimal(sum(row)) / denominator) for row, denominator in zip(scope3, revenue)]}


def repair_answer(original):
    if sha256(original) != EXPECTED_INPUT_SHA256:
        raise ValueError("This patch requires the frozen v6 answer hash")
    revised = original.replace("MSCI, Sustainalytics and Fashion Transparency Index scores, external investigations and company responses to controversies are unavailable in these excerpts.",
        "External rating and company-disclosure supplements appear below; important rating and investigation gaps remain.")
    revised = revised.replace("Separate scope-specific annual revenue denominators are not supplied. FR supplies combined Scope1+2, not separate scopes; production Scope3 covers raw materials, fabric and garment manufacturing for UNIQLO/GU, whereas revenue is consolidated group revenue.",
        "This compact baseline omitted revenue and disaggregated FR scopes now supplied in the archive supplement; its earlier production-only Scope3 series remains separately labeled.")
    revised = revised.replace("FR grievance-channel detail is unavailable; FLA confirms Patagonia's confidential channel.",
        "The archive supplement adds H&M grievance coverage/outcomes; FR detailed outcomes remain unavailable.")
    revised = revised.replace("Audit findings above are disclosed issues; external controversy histories and responses remain unavailable.",
        "Further disclosed cases and partial ratings follow; external investigation gaps remain.")
    revised = revised.replace("## Weighted assessment and calculation", LEGAL + "\n## Weighted assessment and calculation", 1)
    revised = revised.replace("## Sources", ADDITIONS + "\n## Sources", 1)
    return revised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-answer", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    original = args.input_answer.read_text(encoding="utf-8")
    if sha256(original) != EXPECTED_INPUT_SHA256:
        raise ValueError("This patch requires the frozen v6 answer hash")
    additions = source_additions(args.source_archive)
    revised = repair_answer(original)
    existing_source_ids = {"hm2024", "patagonia2025", "frfinance2024", "fr2024", "frclimate2025", "fla2025"}
    new_references = "\n".join(f"- [{source['source_id']}] {source['title']}. {source['date']}. {source['url']}"
                              for source in additions if source["source_id"] not in existing_source_ids)
    revised = revised.replace("## Small chart", new_references + "\n\n## Small chart", 1)
    count = len(re.findall(r"\S+", revised))
    if count > 3500:
        raise ValueError("Report exceeds 3500 whitespace words: " + str(count))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "answer.md").write_text(revised, encoding="utf-8")
    source_identity = [{"source_id": source["source_id"], "url": source["url"],
        "source_text_sha256": source["source_text_sha256"], "archive_locator": source["archive_locator"],
        "sections": [{key: section[key] for key in ("span_start", "span_end", "sha256")}
                     for section in source["sections"]]} for source in additions]
    manifest = {"source_additions": source_identity, "source_additions_hash": sha256(json.dumps(source_identity, sort_keys=True)),
        "input_answer_hash": sha256(original), "answer_hash": sha256(revised),
        "new_generation_model_calls": 0, "new_generation_tokens": 0,
        "basis": "frozen v6 arithmetic report derived from v5, with targeted public archive enrichment",
        "word_count_whitespace": count, "truncated": False,
        "frozen_scoring_input_policy": "No component inputs, weights, category totals or ranks are changed"}
    for filename, value in [("public_source_additions.json", additions), ("source_manifest.json", manifest),
                            ("calculation_checks.json", calculate_checks()),
                            ("submission.json", {"answer": revised, **manifest})]:
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), revised.splitlines(keepends=True),
                                      fromfile="v6/answer.md", tofile="v7/answer.md"))
    (args.output / "answer.diff").write_text(diff, encoding="utf-8")
    print(json.dumps({"words": count, "source_additions_hash": manifest["source_additions_hash"],
                      "sources": len(additions), "new_generation_calls": 0}))


if __name__ == "__main__":
    main()
