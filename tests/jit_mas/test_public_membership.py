"""Source-bound membership observations, never private answer-set selection."""

import hashlib
import json
import copy

import pytest

from jit_mas.evidence import EVIDENCE_QUESTION_HEADER, EVIDENCE_SCOPE, EVIDENCE_TASK_CONSTRAINT
from jit_mas.public_membership import build_public_membership_observations, public_membership_model_input
from jit_mas.schemas import PublicTask, digest


def public_task(question="For 2023, list entities whose score is at least 2.8 points.",
                text="score uses points.\nentity,score\nBirch,2.7\nCedar,2.8\nFir,2.9",
                *, date="2023 cohort", constraints=None, source_changes=None):
    public = PublicTask(task_id="PRIVATE_TASK_ID_CANARY", question=question, constraints=constraints or [])
    if text is None:
        return public
    sha = hashlib.sha256(text.encode()).hexdigest()
    source = {"source_id": "web0", "source_sha256": sha, "date": date, "status": "ok",
              "kind": "provided_table", "title": "Supplied cohort table", "locator": "https://example.org/table",
              "retrieved_at": "2099-12-01", "truncated": False,
              "segments": [{"text": text, "source_sha256": sha, "span_start": 0, "span_end": len(text)}]}
    source.update(source_changes or {})
    body = {"task_id": public.task_id, "scope": EVIDENCE_SCOPE, "sources": [source]}
    public.question += EVIDENCE_QUESTION_HEADER + json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                            separators=(",", ":"), allow_nan=False)
    public.constraints.append(EVIDENCE_TASK_CONSTRAINT)
    return public


def observe(*args, **kwargs):
    return build_public_membership_observations(public_task(*args, **kwargs))


def test_supplied_cells_overrule_a_pass_claim_without_certifying_membership():
    public = public_task()
    materials = {"contributions": [{"agent_id": "PUBLIC_ANALYST", "ledger": {
        "outline": ["Birch: score 2.7 >= 2.8; PASS"],
        "source_references": [{"source_id": "web0", "locator": "https://example.org/table"}],
        "evidence_spans": [{"text": "Birch,2.7", "source_ref": "web0"}]}}],
        "private_evaluation": "PRIVATE_SCORE_CANARY"}
    observed = build_public_membership_observations(public, materials, "Birch and Cedar qualify.")
    table, = observed["table_observations"]
    assert [row["checks"][0]["arithmetic_truth"] for row in table["rows"]] == [False, True, True]
    assert [row["checks"][0]["status"] for row in table["rows"]] == ["fail", "pass", "pass"]
    assert all(row["membership_status"] == "unknown" for row in table["rows"])
    assert observed["upstream_claims"][0]["labels"] == ["PASS"]
    assert observed["upstream_claims"][0]["status"] == "unknown"
    assert table["rows"][0]["entity_literal_present_in_draft"] is True
    assert observed["draft_hash"] == digest("Birch and Cedar qualify.")
    assert observed["membership_status"] == "unknown" and observed["independently_verified"] is False
    serialized = json.dumps(observed)
    assert "PRIVATE_TASK_ID_CANARY" not in serialized and "PRIVATE_SCORE_CANARY" not in serialized
    assert "semantic entailment" in observed["limitations"] and not observed["truncated"]


@pytest.mark.parametrize("operator,expected", [
    ("at least", [False, True, True]), (">=", [False, True, True]),
    ("more than", [False, False, True]), (">", [False, False, True]),
    ("at most", [True, True, False]), ("<=", [True, True, False]),
    ("less than", [True, False, False]), ("<", [True, False, False]),
    ("=", [False, True, False]),
])
def test_inclusive_and_strict_comparisons_use_the_original_operator(operator, expected):
    observed = observe(question=f"For 2023, list entities whose score is {operator} 2.8 points.")
    assert [row["checks"][0]["arithmetic_truth"] for row in observed["table_observations"][0]["rows"]] == expected


def test_decimal_comparisons_keep_digits_beyond_float_precision():
    observed = observe(question="For 2023, list entities whose score is more than 0.1 points.",
                       text="score uses points.\nentity,score\nBirch,0.10000000000000000000000000001")
    assert observed["table_observations"][0]["rows"][0]["checks"][0]["arithmetic_truth"] is True


@pytest.mark.parametrize("text,changes", [
    (None, None), ("entity,score\nBirch,2.7", None),
    ("score uses points.\nentity,score\nBirch,2.7", {"truncated": True}),
    ("score uses points.\nentity,score\nBirch,2.7", {"status": "failed"}),
    ("score uses points.\nentity,score\nBirch,2.7", {"segments": []}),
    ("score uses points.\nentity,score\nBirch,2.7", {"source_id": None}),
    ("score uses points.\nentity,score\nBirch,2.7", {"source_sha256": "not-a-source-hash"}),
])
def test_missing_failed_or_partial_evidence_never_becomes_a_verified_pass(text, changes):
    observed = observe(text=text, source_changes=changes)
    assert observed["membership_status"] == "unknown"
    assert not any(row["checks"] for table in observed["table_observations"] for row in table["rows"])


@pytest.mark.parametrize("date,reason", [
    (None, "missing_declared_source_scope"), ("", "missing_declared_source_scope"),
    ("2024 cohort", "source_requested_year_mismatch"),
    ("2022-2023 pooled cohorts", "source_requested_year_mismatch"),
])
def test_observation_year_does_not_fall_back_to_the_retrieval_date(date, reason):
    observed = observe(date=date)
    binding = observed["table_observations"][0]["bindings"][0]
    assert binding["status"] == "unknown" and binding["reason"] == reason
    assert observed["table_observations"][0]["rows"][0]["checks"] == []


@pytest.mark.parametrize("question,reason", [
    ("As of June 2023, list entities whose score is at least 2.8 points.", "unsupported_finer_date_scope"),
    ("For 2022 and 2023, list entities whose score is at least 2.8 points.", "ambiguous_requested_year_scope"),
    ("As of 2023-04-02, list entities whose score is at least 2.8 points.", "unsupported_finer_date_scope"),
])
def test_unsupported_or_multiple_requested_dates_remain_unknown(question, reason):
    table, = observe(question=question)["table_observations"]
    assert table["bindings"][0]["reason"] == reason and table["rows"][0]["checks"] == []


@pytest.mark.parametrize("text,reason", [
    ("score uses kg.\nentity,score\nBirch,2.7", "requested_source_unit_mismatch"),
    ("score uses points. score uses percent.\nentity,score\nBirch,2.7", "missing_or_conflicting_explicit_source_unit"),
    ("score uses points.\nentity,score_per_person\nBirch,2.7", "ambiguous_or_missing_full_header_binding"),
    ("score uses points.\nentity,score,score_points\nBirch,2.7,2.7", "ambiguous_or_missing_full_header_binding"),
    ("score uses points.\nentity,score_2\nBirch,2.7", "ambiguous_or_missing_full_header_binding"),
])
def test_field_unit_and_qualified_header_ambiguity_prevent_label_only_binding(text, reason):
    table, = observe(text=text)["table_observations"]
    assert table["bindings"][0]["reason"] == reason and table["rows"][0]["checks"] == []


def test_numeric_bound_digits_are_not_a_header_qualifier():
    observed = observe(question="For 2023, list entities whose score is at least 2.8 points.",
                       text="score_2 uses points.\nentity,score_2\nBirch,2.7")
    assert observed["table_observations"][0]["bindings"][0]["status"] == "unknown"


@pytest.mark.parametrize("unit", [
    "kg/km", "kg / km", "kg^2", "kg**2", "kg\u00b2", "kg\u00b3", "kg\u00b7m",
    "kg\u00d7m", "kg-force", "kg\u2011force", "kg squared", "kg cubed",
    "kg per entity", "points per supplier",
])
def test_a_recognized_unit_prefix_cannot_drop_a_compound_or_denominator(unit):
    source_unit = "points" if unit.startswith("points") else "kg"
    observed = observe(question=f"For 2023, list entities whose score is at least 2.8 {unit}.",
                       text=f"score uses {source_unit}.\nentity,score\nBirch,2.9")
    threshold, = observed["numeric_thresholds"]
    assert threshold["supported"] is False and threshold["status"] == "unknown"
    table, = observed["table_observations"]
    assert table["bindings"][0]["status"] == "unknown" and table["rows"][0]["checks"] == []


@pytest.mark.parametrize("question", [
    "Do not include entities whose score is at least 2.8 points.",
    "If useful, include entities whose score is at least 2.8 points.",
    'An example is "include entities whose score is at least 2.8 points."',
    'Read this: "list entities, whose score is at least 2.8 points."',
    "List entities whose score is at least 2.8 points or whose income is high.",
    "For 2023, list entities whose score is at least 2e3 points.",
    "For 2023, list entities whose score is at least 2,000 points.",
    "For 2023, list entities whose score is at least 2/3 points.",
    "For 2023, list entities whose score is != 2.8 points.",
    "For 2023, list entities whose score is => 2.8 points.",
    "For 2023, list entities whose score is never less than 2.8 points.",
    "For 2023, list entities whose score is at least 2.8 percentiles.",
    "For 2023, list entities whose score is at least 2.8 km.",
    "For 2023, list entities whose score is at least " + "8" * 40 + " points.",
    "For 2023, list entities whose score is at least 2.8 and at most 3.0 points.",
])
def test_unsupported_or_nonpositive_thresholds_do_not_create_verified_filters(question):
    observed = observe(question=question)
    assert not any(row["checks"] for table in observed["table_observations"] for row in table["rows"])


@pytest.mark.parametrize("value", ["NaN", "Infinity", "2e3", "2,700", "2/3", "8" * 40, ""])
def test_unsupported_source_values_are_unknown_not_fail(value):
    cell = '"' + value + '"' if "," in value else value
    observed = observe(text="score uses points.\nentity,score\nBirch," + cell)
    check, = observed["table_observations"][0]["rows"][0]["checks"]
    assert check["arithmetic_truth"] is None and check["status"] == "unknown"


@pytest.mark.parametrize("delimiter", ["\n", "\\n"])
def test_complete_composite_and_percentage_tables_have_raw_source_spans(delimiter):
    text = delimiter.join([
        "All _pct columns use percent units; average_composite_score uses example score points.",
        "entity,estimated_graduates_tested_pct,average_composite_score,subject_benchmark_7_pct",
        "Birch,92,2.7,75", "Cedar,95,2.8,80"])
    question = ("For 2023, identify entities with an estimated at least 90% of their grads tested, "
                "which had a composite average score of at least 2.8, "
                "and had at least 70% meet the subject benchmark.")
    observed = observe(question=question, text=text)
    table, = observed["table_observations"]
    assert [binding["status"] for binding in table["bindings"]] == ["bound", "bound", "unknown"]
    assert [check["arithmetic_truth"] for check in table["rows"][0]["checks"]] == [True, False]
    for row in table["rows"]:
        span = row["row_span"]
        assert text[span["start"]:span["end"]] == span["text"]
        for check in row["checks"]:
            span = check["cell_span"]
            assert text[span["start"]:span["end"]] == span["text"]
            assert span["text_hash"] == digest(span["text"])
    question_span = observed["numeric_thresholds"][1]["original_span"]
    assert question[question_span["start"]:question_span["end"]] == question_span["text"]


def test_markdown_and_quoted_csv_cells_keep_entity_and_cell_offsets():
    text = "score uses points.\n| entity | score |\n| --- | ---: |\n| Birch | 2.7 |"
    table, = observe(text=text)["table_observations"]
    assert table["format"] == "markdown" and table["rows"][0]["checks"][0]["status"] == "fail"
    text = 'score uses points.\nentity,score\n"Birch, western",2.7'
    table, = observe(text=text)["table_observations"]
    row, = table["rows"]
    assert row["entity"] == "Birch, western"
    assert row["entity_cell_span"]["text"] == '"Birch, western"'
    assert row["checks"][0]["cell_span"]["text"] == "2.7"


@pytest.mark.parametrize("malformed", ["Cedar,2.8,9", '"Cedar,2.8'])
def test_malformed_ragged_table_prevents_partial_rows_from_becoming_verified(malformed):
    table, = observe(text="score uses points.\nentity,score\nBirch,2.7\n" + malformed)["table_observations"]
    assert table["incomplete"] is True and table["bindings"][0]["reason"] == "incomplete_table_rows"
    assert table["rows"][0]["checks"] == []


def test_bounded_rows_conditions_and_claims_disclose_truncation():
    text = "score uses points.\nentity,score\n" + "\n".join(f"Item{i},2.7" for i in range(129))
    observed = observe(text=text)
    assert observed["truncated"] is True and len(observed["table_observations"][0]["rows"]) == 128
    public = public_task(text=None, question="List entities.", constraints=[f"Condition {i}." for i in range(35)])
    observed = build_public_membership_observations(public, {"contributions": [{"ledger": {"outline": ["PASS"] * 65}}]})
    assert observed["truncated"] is True and len(observed["hard_conditions"]) == 32
    assert len(observed["upstream_claims"]) == 64


def test_semantic_conditions_remain_original_without_new_personal_or_stock_guarantees():
    question = ("As of June 2024, identify every location meeting these requirements:\n"
                "1. The main language is Spanish.\n2. It lies on the Pacific coast.\n"
                "3. It has a coast on another major body of water.\n"
                "4. It has a dedicated digital nomad visa.\n5. It recognizes same-sex marriage.\n"
                "6. Both specified products can be purchased locally.")
    public = public_task(question=question, text=None)
    observed = build_public_membership_observations(public, {"contributions": [{"ledger": {
        "requirements": ["All nationalities must be eligible and both products always in stock."],
        "outline": ["Nothing is verified; UNKNOWN"]}}]})
    assert observed["original_request"]["text"] == question
    assert len(observed["hard_conditions"]) == 7
    assert all(row["status"] == "unknown" for row in observed["hard_conditions"])
    assert observed["numeric_thresholds"] == [] and observed["table_observations"] == []
    assert not any("nationalities" in row["original_span"]["text"] or "stock" in row["original_span"]["text"]
                   for row in observed["hard_conditions"])


def test_noncanonical_evidence_and_arbitrary_metadata_cannot_create_a_source():
    public = public_task(text=None)
    public.question += EVIDENCE_QUESTION_HEADER + '{"not_a_pack":"score uses points."}'
    payload = public.model_dump(mode="json") | {"evaluation": "PRIVATE_BODY_CANARY", "reference": "PRIVATE_REFERENCE_CANARY"}
    observed = build_public_membership_observations(payload, {"tool_observations": [{"content": {
        "output": "score uses points.\nentity,score\nBirch,2.7"}}],
        "private_metadata": "PRIVATE_TOOL_CANARY"})
    assert observed["fixed_pack_status"] == "unknown" and observed["table_observations"] == []
    encoded = json.dumps(observed)
    assert "PRIVATE_BODY_CANARY" not in encoded and "PRIVATE_REFERENCE_CANARY" not in encoded
    assert "PRIVATE_TOOL_CANARY" not in encoded


def test_constraints_only_numeric_requirement_has_its_original_index_and_offsets():
    observed = observe(question="For 2023, list entities.", constraints=["Each entity has score at least 2.8 points."])
    threshold, = observed["numeric_thresholds"]
    assert threshold["original_span"]["source"] == "constraints[0]"
    assert observed["table_observations"][0]["rows"][0]["checks"][0]["status"] == "fail"


def test_invalid_public_input_is_rejected_without_reading_other_fields():
    with pytest.raises(TypeError):
        build_public_membership_observations(object())
    with pytest.raises(TypeError):
        build_public_membership_observations({"question": 2, "private": "DO_NOT_READ"})


def matrix_rows(table):
    return [dict(zip(table["row_fields"], row)) for row in table["rows"]]


def test_default_and_explicit_full_model_input_are_unchanged_independent_snapshots():
    full = observe()
    original = copy.deepcopy(full)
    default = public_membership_model_input(full)
    explicit = public_membership_model_input(full, "full")
    assert default == explicit == original and default is not full
    default["table_observations"][0]["rows"][0]["checks"][0]["observed_value"] = "999"
    explicit["hard_conditions"][0]["original_span"]["text"] = "Different request"
    assert full == original
    assert "full_observation_hash" not in public_membership_model_input(full)


def test_compact_preserves_all_rows_rules_truth_and_source_scope_with_an_audit_hash():
    full = observe()
    compact = public_membership_model_input(full, "compact")
    assert compact["full_observation_hash"] == digest(full)
    assert compact["observation_version"] == full["version"]
    table, = compact["table_observations"]
    rows = matrix_rows(table)
    assert [row["entity"] for row in rows] == [row["entity"] for row in full["table_observations"][0]["rows"]]
    assert [row["arithmetic_truth"] for row in rows] == [[False], [True], [True]]
    assert [row["observed_values"] for row in rows] == [["2.7"], ["2.8"], ["2.9"]]
    assert len({row["row_id"] for row in rows}) == 3
    assert table["threshold_ids"] == [threshold["threshold_id"] for threshold in full["numeric_thresholds"]]
    assert table["scope"] == full["table_observations"][0]["scope"]
    for key in ("source_id", "declared_source_sha256", "declared_date", "segment_hash", "headers", "truncated", "incomplete"):
        assert table[key] == full["table_observations"][0][key]
    assert table["coverage"] == {"observed_rows": 3, "observed_checks": 3, "all_observed_rows_retained": True}
    assert all(row["source_span"] == [original["row_span"]["start"], original["row_span"]["end"]]
               for row, original in zip(rows, full["table_observations"][0]["rows"]))
    assert compact["membership_status"] == "unknown" and compact["independently_verified"] is False
    assert "not complete eligibility" in compact["matrix_convention"]
    assert "not an affirmative inclusion" in compact["matrix_convention"]


def test_compact_rule_and_claim_anchors_keep_original_offsets_without_duplicate_text():
    public = public_task()
    materials = {"contributions": [{"ledger": {"outline": ["Birch: score 2.7; PASS"],
        "source_references": [{"source_id": "web0", "locator": "https://example.org/table"}]}}]}
    full = build_public_membership_observations(public, materials)
    compact = public_membership_model_input(full, "compact")
    for name in ("hard_conditions", "numeric_thresholds"):
        for original, projected in zip(full[name], compact[name]):
            assert projected["original_span"] == {key: value for key, value in original["original_span"].items() if key != "text"}
            assert projected["condition_id"] == original["condition_id"]
    claim, = compact["upstream_claims"]
    assert "text" not in claim["claim_span"]
    assert claim["labels"] == ["PASS"] and claim["status"] == "unknown"
    assert claim["declared_source_references"] == full["upstream_claims"][0]["declared_source_references"]
    assert "condition_text" not in compact["numeric_thresholds"][0]
    assert "text" not in compact["original_request"]


def test_compact_preserves_unbound_rules_and_unknown_source_cells_without_false_failures():
    question = ("For 2023, list entities with score at least 2.8 points, "
                "and had subject benchmark at least 70%.")
    text = "score uses points. All _pct columns use percent units.\nentity,score,subject_benchmark_7_pct\nBirch,NaN,80"
    full = observe(question=question, text=text)
    compact = public_membership_model_input(full, "compact")
    table, = compact["table_observations"]
    row, = matrix_rows(table)
    assert len(table["threshold_ids"]) == 2
    assert row["arithmetic_truth"] == [None, None] and row["observed_values"] == ["NaN", None]
    assert row["numeric_status"] == "unknown"
    assert [binding["status"] for binding in table["bindings"]] == ["bound", "unknown"]
    assert table["bindings"][1]["reason"] == full["table_observations"][0]["bindings"][1]["reason"]
    unknown, = table["unknown_cells"]
    assert unknown["row_id"] == row["row_id"] and unknown["threshold_id"] == table["threshold_ids"][0]
    assert unknown["reason"] == "unsupported_source_cell_decimal"
    assert unknown["cell_span"]["text_hash"] == full["table_observations"][0]["rows"][0]["checks"][0]["cell_span"]["text_hash"]


@pytest.mark.parametrize("changes", [{"date": "2024 cohort"}, {"question": "As of June 2023, list entities whose score is at least 2.8 points."},
                                     {"text": "score uses kg.\nentity,score\nBirch,2.7"}])
def test_compact_does_not_upgrade_unknown_date_or_unit_bindings(changes):
    full = observe(**changes)
    compact = public_membership_model_input(full, "compact")
    table, = compact["table_observations"]
    assert table["bindings"][0]["status"] == "unknown"
    assert table["bindings"][0]["reason"] == full["table_observations"][0]["bindings"][0]["reason"]
    assert all(row["arithmetic_truth"] == [None] and row["observed_values"] == [None] for row in matrix_rows(table))


def test_compact_retains_header_unit_fragments_that_have_no_offset_fields():
    compact = public_membership_model_input(observe(text="entity,score (points)\nBirch,2.7"), "compact")
    binding, = compact["table_observations"][0]["bindings"]
    assert binding["status"] == "bound"
    assert binding["unit_evidence"] == [{"source": "header", "text": "(points)"}]


def test_compact_preserves_duplicate_entities_and_mentions_in_excluded_text_without_deciding_membership():
    full = build_public_membership_observations(public_task(text="score uses points.\nentity,score\nBirch,2.7\nBirch,2.9"),
                                                draft="Excluded entity: Birch. Its name is mentioned, not included.")
    compact = public_membership_model_input(full, "compact")
    table, = compact["table_observations"]
    rows = matrix_rows(table)
    assert len(rows) == 2 and rows[0]["entity"] == rows[1]["entity"] == "Birch"
    assert rows[0]["row_id"] != rows[1]["row_id"]
    assert [row["arithmetic_truth"] for row in rows] == [[False], [True]]
    assert all(row["entity_literal_present_in_draft"] for row in rows)
    assert compact["membership_status"] == "unknown" and table["membership_status"] == "unknown"
    assert not any("eligible" in field or "included" in field for field in table["row_fields"])


def test_compact_preserves_existing_truncation_limits_and_is_an_independent_snapshot():
    text = "score uses points.\nentity,score\n" + "\n".join(f"Item{i},2.7" for i in range(129))
    full = observe(text=text)
    original = copy.deepcopy(full)
    compact = public_membership_model_input(full, "compact")
    assert compact["truncated"] is True and compact["limits"] == full["limits"]
    assert len(compact["table_observations"][0]["rows"]) == 128
    compact["table_observations"][0]["rows"][0][3][0] = "999"
    compact["table_observations"][0]["scope"]["requested_years"].append("2099")
    assert full == original


def test_compact_preserves_semantic_only_conditions_without_new_requirements_or_evidence():
    full = observe(question="Find locations.\n1. Main language is Spanish.\n2. Both specified products can be purchased locally.", text=None)
    compact = public_membership_model_input(full, "compact")
    assert compact["table_observations"] == [] and compact["numeric_thresholds"] == []
    assert compact["fixed_pack_status"] == "unknown"
    assert [condition["condition_id"] for condition in compact["hard_conditions"]] == [condition["condition_id"] for condition in full["hard_conditions"]]
    assert all(condition["status"] == "unknown" for condition in compact["hard_conditions"])
    assert compact["limitations"] == full["limitations"]


@pytest.mark.parametrize("bad_format", ["short", "", None, True])
def test_model_input_rejects_unknown_formats_without_changing_the_snapshot(bad_format):
    full = observe()
    original = copy.deepcopy(full)
    with pytest.raises(ValueError):
        public_membership_model_input(full, bad_format)
    assert full == original


def test_compact_rejects_unknown_versions_extra_body_fields_and_non_object_inputs():
    full = observe()
    with pytest.raises(ValueError):
        public_membership_model_input(full | {"version": "future-unknown-schema"}, "compact")

    class UnreadBody:
        def __deepcopy__(self, memo):
            raise AssertionError("Unexpected body was read")

        def __repr__(self):
            raise AssertionError("Unexpected body was displayed")

    with pytest.raises(ValueError):
        public_membership_model_input(full | {"unexpected_body": UnreadBody()}, "compact")
    with pytest.raises(TypeError):
        public_membership_model_input("raw metadata", "compact")


@pytest.mark.parametrize("corruption", ["duplicate_rule", "duplicate_check", "unknown_check", "inconsistent_truth", "verified_row"])
def test_compact_never_silently_drops_or_reinterprets_corrupted_observation_ids(corruption):
    full = observe()
    row = full["table_observations"][0]["rows"][0]
    if corruption == "duplicate_rule":
        full["numeric_thresholds"].append(copy.deepcopy(full["numeric_thresholds"][0]))
    elif corruption == "duplicate_check":
        row["checks"].append(copy.deepcopy(row["checks"][0]))
    elif corruption == "unknown_check":
        row["checks"][0]["threshold_id"] = "MISSING.N1"
    elif corruption == "inconsistent_truth":
        row["checks"][0]["arithmetic_truth"] = True
    else:
        row["membership_status"] = "verified"
    with pytest.raises(ValueError):
        public_membership_model_input(full, "compact")


def test_large_rectangular_compaction_reduces_repeated_audit_fields_without_pruning_rows():
    question = ("For 2023, identify entities with an estimated at least 90% of their grads tested, "
                "which had a composite average score of at least 2.8, and had at least 70% meet the subject benchmark.")
    text = ("All _pct columns use percent units; average_composite_score uses example score points.\n"
            "entity,estimated_graduates_tested_pct,average_composite_score,subject_benchmark_7_pct\n"
            + "\n".join(f"Item{i},92,2.7,75" for i in range(52)))
    full = observe(question=question, text=text)
    compact = public_membership_model_input(full, "compact")
    size = lambda value: len(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    assert size(compact) < size(full) * 0.35
    table, = compact["table_observations"]
    rows = matrix_rows(table)
    assert len(rows) == 52 and table["coverage"]["observed_checks"] == 104
    assert all(row["arithmetic_truth"] == [True, False, None] for row in rows)
    assert table["bindings"][2]["status"] == "unknown"
    assert compact["full_observation_hash"] == digest(full)
