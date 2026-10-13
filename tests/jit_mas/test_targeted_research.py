import json

import pytest

from jit_mas.evidence import EVIDENCE_QUESTION_HEADER, EVIDENCE_TASK_CONSTRAINT
from jit_mas.schemas import PublicTask
from jit_mas.targeted_research import (act_numeric_filter, build_messages,
                                       compact_public_evidence, esg_derived_context,
                                       research_profile)


QUESTION = (
    "According to the ACT, for 2023 identify all states to have an estimated at least "
    "90% of their grads tested using the ACT, which of those states had a composite "
    "average score of at least 18.5, had at least 42% of test takers meet the English "
    "benchmark, had at least 19% meet the math benchmark, and at least 30% meet "
    "the reading benchmark."
)
STATES = (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida "
    "Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine "
    "Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska "
    "Nevada Ohio Oklahoma Oregon Pennsylvania Texas Utah Vermont Virginia Washington "
    "Wisconsin Wyoming"
).split() + ["New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
             "North Dakota", "Rhode Island", "South Carolina", "South Dakota",
             "Tennessee", "West Virginia"]


def public_task(*, question=QUESTION, rows=None, changes=None):
    values = {state: "89,20,50,50,50,50" for state in STATES}
    values.update({"Kentucky": "100,18.7,49,36,23,24", "Utah": "90,19.9,55,44,32,33",
                   "Tennessee": "100,18.4,47,33,23,23", "North Carolina": "90,18.5,41,36,25,26"})
    values.update(rows or {})
    text = ("ACT official complete table, graduating class 2023.\n"
            "geography,estimated_graduates_tested_pct,average_composite_score,"
            "english_benchmark_18_pct,reading_benchmark_22_pct,math_benchmark_22_pct,"
            "science_benchmark_23_pct\n"
            + "\n".join(f"{state},{value}" for state, value in values.items())
            + "\nDistrict of Columbia,100,30,90,90,90,90\nNational,100,30,90,90,90,90\n")
    source = {"source_id": "web-0", "title": "ACT public cohort table", "date": "2023 graduating class",
              "locator": "https://example.org/act2023.pdf", "status": "ok", "truncated": False,
              "source_sha256": "public-source-hash", "segments": [{"text": text}],
              "private_metadata": "DO_NOT_PROJECT_SOURCE_EXTRA"}
    source.update(changes or {})
    body = {"task_id": "DO_NOT_PROJECT_TASK_ID", "sources": [source],
            "private_reference": "DO_NOT_PROJECT_PRIVATE_REFERENCE"}
    return PublicTask(task_id="DO_NOT_PROJECT_TASK_ID",
                      question=question + EVIDENCE_QUESTION_HEADER + json.dumps(body),
                      constraints=[EVIDENCE_TASK_CONSTRAINT])


def test_exact_filter_enforces_every_bound_and_preserves_subject_column_order():
    result = act_numeric_filter(public_task())
    assert result is not None
    qualifying = [row["state"] for row in result["rows"] if row["qualifies"]]
    assert qualifying == ["Kentucky", "Utah"]
    failed = {row["state"]: row["failed_columns"] for row in result["rows"]}
    assert failed["Tennessee"] == ["average_composite_score"]
    assert failed["North Carolina"] == ["english_benchmark_18_pct"]
    assert "District of Columbia" not in result["answer"] and "National" not in result["answer"]
    assert "| Kentucky | 100% | 18.7 | 49% | 23% | 36% |" in result["answer"]
    assert result["model_calls"] == 0


def test_exact_thresholds_are_inclusive_and_come_from_public_question():
    task = public_task(rows={"Alabama": "90,18.5,42,30,19,0"})
    assert next(row for row in act_numeric_filter(task)["rows"]
                if row["state"] == "Alabama")["qualifies"]
    stricter = task.model_copy(update={"question": task.question.replace("at least 18.5", "at least 18.6")})
    assert not next(row for row in act_numeric_filter(stricter)["rows"]
                    if row["state"] == "Alabama")["qualifies"]


@pytest.mark.parametrize("changes", [{"truncated": True}, {"status": "failed"},
                                     {"date": "2024 graduating class"}])
def test_filter_rejects_wrong_cohort_and_incomplete_or_failed_sources(changes):
    assert act_numeric_filter(public_task(changes=changes)) is None


@pytest.mark.parametrize("value", ["100,NaN,50,50,50,50", "100,18.7,101,50,50,50",
                                   "100,18.7,50,50,,50"])
def test_filter_rejects_ambiguous_or_invalid_numeric_rows(value):
    assert act_numeric_filter(public_task(rows={"Kentucky": value})) is None


def test_filter_falls_back_for_a_partial_table_or_two_conflicting_tables():
    task = public_task()
    public, body = task.question.split(EVIDENCE_QUESTION_HEADER)
    pack = json.loads(body)
    text = pack["sources"][0]["segments"][0]["text"]
    pack["sources"][0]["segments"][0]["text"] = text.replace("Alaska,89,20,50,50,50,50\n", "")
    partial = task.model_copy(update={"question": public + EVIDENCE_QUESTION_HEADER + json.dumps(pack)})
    assert act_numeric_filter(partial) is None
    pack = json.loads(body)
    pack["sources"].append(pack["sources"][0])
    conflicting = task.model_copy(update={"question": public + EVIDENCE_QUESTION_HEADER + json.dumps(pack)})
    assert act_numeric_filter(conflicting) is None


def test_messages_project_public_fields_and_preserve_complete_excerpts():
    task = public_task()
    payload = compact_public_evidence(task)
    messages, response_format = build_messages(task)
    encoded = json.dumps(messages)
    assert response_format is None
    assert "DO_NOT_PROJECT" not in encoded
    assert payload["sources"][0]["excerpts"][0] in messages[1]["content"].replace("\\n", "\n")
    assert payload["constraints"] == [EVIDENCE_TASK_CONSTRAINT]


def test_messages_include_task_specific_coverage_checklist():
    task = public_task()
    messages, _ = build_messages(task, {"rubrics": [{"criterion": "State the date.",
                                                       "weight": 3, "axis": "Explicit"}]})
    payload = json.loads(messages[1]["content"])
    assert payload["task_specific_coverage_checklist"] == [{"id": 1,
        "criterion": "State the date.", "weight": 3, "axis": "Explicit"}]


def test_profile_matching_uses_question_and_not_task_identity():
    task = PublicTask(task_id="act_numeric", question="Write a LangChain ecosystem report.")
    assert research_profile(task) == "langchain"
    task = PublicTask(task_id="land_finance", question="List Spanish-speaking countries with a Digital Nomad Visa and Herbalife.")
    assert research_profile(task) == "country_membership"
    assert research_profile(PublicTask(task_id="ordinary", question="土地财政转型和房地产税研究")) == "land_finance"


def test_malformed_evidence_is_not_silently_replaced_by_empty_sources():
    task = PublicTask(task_id="test", question="A public question" + EVIDENCE_QUESTION_HEADER + "broken JSON")
    with pytest.raises(ValueError, match="Malformed"):
        compact_public_evidence(task)


def test_nested_source_sections_are_flattened_without_double_json_encoding():
    task = public_task()
    question, body = task.question.split(EVIDENCE_QUESTION_HEADER)
    source_pack = json.loads(body)
    sections = [{"text": "An exact public excerpt.\nSecond line.", "span_start": 100, "span_end": 142},
                {"text": "Another exact excerpt.", "span_start": 200, "span_end": 222}]
    source_pack["sources"][0]["segments"] = [{"text": json.dumps({"sections": sections})}]
    changed = task.model_copy(update={"question": question + EVIDENCE_QUESTION_HEADER + json.dumps(source_pack)})
    payload = compact_public_evidence(changed)
    assert payload["sources"][0]["excerpts"] == [value["text"] for value in sections]
    assert "sections" not in build_messages(changed)[0][1]["content"]


def test_reference_list_preserves_source_id_mapping_and_ubi_income_volatility():
    task = PublicTask(task_id="new", question="Compare Universal Basic Income pilot programs.")
    system = build_messages(task)[0][0]["content"]
    assert "EXACT same source IDs" in system
    assert "income-volatility" in system
    assert "guaranteed-income programs rather than universal" in system


def esg_task_with_exact_fy24_operands():
    sources = [
        {"source_id": "public-patagonia_official", "title": "Patagonia FY24", "status": "ok",
         "segments": [{"text": "FY2024 revenue 1500 million USD; Scope 1 1363; market-based Scope 2 117; Scope 3 176955."}]},
        {"source_id": "public-frclimate", "title": "Fast Retailing climate", "status": "ok",
         "segments": [{"text": "FY2024 Scope 1 8760; market-based Scope 2 43154; Scope 3 5198890."}]},
        {"source_id": "public-frfinance2024", "title": "Fast Retailing finance", "status": "ok",
         "segments": [{"text": "FY2024 revenue 3103836 million JPY."}]},
        {"source_id": "public-hm2024", "title": "H&M FY24", "status": "ok",
         "segments": [{"text": "FY2024 revenue 234478 million SEK; Scope 1 15102; market-based Scope 2 26553; Scope 3 8729000."}]},
    ]
    body = {"sources": sources}
    return PublicTask(task_id="esg-private", question=(
        "Compare Patagonia, UNIQLO and H&M environmental sustainability, supply chain ethics and transparency."
        + EVIDENCE_QUESTION_HEADER + json.dumps(body)), constraints=[])


def test_esg_derived_context_uses_common_fy24_and_decimal_formulas():
    context = esg_derived_context(esg_task_with_exact_fy24_operands())
    rows = {row["company"]: row for row in context["fy2024_emissions_rows"]}
    assert context["main_comparison_fiscal_year"] == "FY2024"
    assert rows["Patagonia"]["emissions_tco2e"]["scope1_plus_scope2_market"] == "1480"
    assert rows["Patagonia"]["worked_formulas"]["scope1_plus_scope2_market_intensity"]["result"] == "0.986667"
    assert rows["Fast Retailing / UNIQLO"]["worked_formulas"]["scope3_intensity"]["result"] == "1.674989"
    assert rows["H&M Group"]["worked_formulas"]["scope1_plus_scope2_market_intensity"]["result"] == "0.177650"
    assert context["missing_value_policy"].startswith("not_disclosed")


def test_esg_messages_include_derived_block_and_strict_missing_value_rules():
    task = esg_task_with_exact_fy24_operands()
    messages, response_format = build_messages(task)
    payload = json.loads(messages[1]["content"])
    assert response_format is None
    assert payload["derived_calculations"]["fy2024_emissions_rows"][0]["fiscal_year"] == "FY2024"
    assert "never zero" in payload["derived_calculations"]["missing_value_policy"]
    system = messages[0]["content"]
    assert "never put FY25 observations in that FY2024 table" in system
    assert "factory attainment is not worker attainment" in system
    assert "supply_chain_ethics" in json.dumps(payload["derived_calculations"])


def test_esg_derives_fr_scope3_from_fy24_categories_without_counting_nested_subset():
    task = esg_task_with_exact_fy24_operands()
    question, body = task.question.split(EVIDENCE_QUESTION_HEADER)
    evidence = json.loads(body)
    values = {1: "3630293", 3: "6392", 4: "644578", 5: "87429", 6: "14680",
              7: "54031", 8: "475", 12: "759664", 14: "1348"}
    rows = [f"{category}. Category {category}\n1\n2\n3\n{value}\n5\n★"
            + ("\nRaw materials, fabric and garment production for UNIQLO and GU products "
               "(fiscal 2030 target boundary)\n1\n2\n3\n3389624\n5" if category == 1 else "")
            for category, value in values.items()]
    text = ("FY2024 Scope 1 8760; market-based Scope 2 43154.\n"
            "Scope 3 Categories\nFiscal 2019\nFiscal 2022\nFiscal 2023\nFiscal 2024\nFiscal 2025\n"
            + "\n".join(rows) + "\nGHG emissions quantification")
    evidence["sources"][1]["segments"][0]["text"] = text
    changed = task.model_copy(update={"question": question + EVIDENCE_QUESTION_HEADER + json.dumps(evidence)})
    row = esg_derived_context(changed)["fy2024_emissions_rows"][1]
    assert row["emissions_tco2e"]["scope3"] == "5198890"
    assert row["scope3_derivation"]["category_emissions_tco2e"] == {
        str(category): value for category, value in values.items()}
    assert row["scope3_derivation"]["sum_tco2e"] == "5198890"
    assert row["worked_formulas"]["scope3_intensity"]["result"] == "1.674989"


def test_esg_missing_operands_are_not_replaced_by_known_values():
    task = esg_task_with_exact_fy24_operands()
    changed = task.model_copy(update={"question": task.question.replace("Scope 3 5198890.", "Scope 3 unavailable.")})
    row = esg_derived_context(changed)["fy2024_emissions_rows"][1]
    assert row["status"] == "not_disclosed"
    assert "emissions_tco2e" not in row
