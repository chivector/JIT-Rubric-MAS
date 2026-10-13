import copy
from decimal import Decimal, ROUND_HALF_UP

import pytest

from jit_mas.targeted_nba import (
    build_host_calculations,
    build_measured_host_calculations,
    build_weight_sensitivity,
    validate_host_calculations,
)


def _score_table(calculations, *, places=3, reverse=False, bold=False, curly=False):
    headers = ["Player", "Championships", "MVPs", "Court Stat Score", "Audience",
               "Current Gen", "Next Gen", "Social/Cultural", "Total Score"]
    fields = ["player", "championships", "mvps", "court_stat_score_0_10", "audience",
              "current_generation", "next_generation", "social_cultural", "total_0_10"]
    # Grow the table alongside new formula dimensions instead of hard-coding
    # the current host receipt into validation tests.
    for field in calculations["weights"]:
        if field not in fields:
            headers.insert(-1, field.replace("_", " ").title())
            fields.insert(-1, field)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in reversed(calculations["rows"]) if reverse else calculations["rows"]:
        values = []
        for field in fields:
            value = str(row[field])
            if field == "player":
                value = value.replace("'", "\u2019") if curly else value
            elif field in ("court_stat_score_0_10", "total_0_10"):
                value = str(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
            values.append("**" + value + "**" if bold else value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def test_host_nba_matrix_has_six_rows_and_recomputed_argmax():
    result = build_host_calculations()
    assert result["players"] == [
        "Michael Jordan", "LeBron James", "Magic Johnson", "Kobe Bryant",
        "Stephen Curry", "Shaquille O'Neal",
    ]
    assert all(row["championships"] >= 1 and row["mvps"] >= 1 for row in result["rows"])
    assert result["winner"] == "Michael Jordan"
    assert result["rows"][0]["total_0_10"] == "9.800"


def test_host_audit_is_separate_from_official_score():
    result = build_host_calculations()
    audit = validate_host_calculations(
        "Michael Jordan LeBron James Magic Johnson Kobe Bryant Stephen Curry Shaquille O'Neal",
        result,
    )
    assert audit["winner_mentioned"] and audit["all_fixed_players_mentioned"]
    assert len(audit["host_total_table"]) == 6
    assert not audit["numeric_table_verified"]
    assert not audit["valid"]
    assert audit["errors"] == ["No Markdown score table with a Player column was found."]
    assert "official" in audit["scope"]


@pytest.mark.parametrize("places", [2, 3])
def test_host_audit_checks_real_numeric_cells_with_display_variants(places):
    calculations = build_host_calculations()
    audit = validate_host_calculations(
        _score_table(calculations, places=places, reverse=True, bold=True, curly=True), calculations)
    assert audit["numeric_table_verified"] and audit["valid"]
    assert audit["all_fixed_players_mentioned"]
    assert audit["errors"] == []
    assert len(audit["checked_rows"]) == 6
    assert all(cell["matches"] for row in audit["checked_rows"] for cell in row["cells"].values())


@pytest.mark.parametrize("field", ["championships", "audience", "total_0_10"])
def test_host_audit_rejects_changed_input_or_total(field):
    canonical = build_host_calculations()
    rendered = copy.deepcopy(canonical)
    rendered["rows"][0][field] = Decimal(str(rendered["rows"][0][field])) - Decimal("0.1")
    audit = validate_host_calculations(_score_table(rendered), canonical)
    assert not audit["numeric_table_verified"]
    assert any(f"Michael Jordan.{field}" in error for error in audit["errors"])


def test_host_audit_rejects_missing_row_or_column():
    canonical = build_host_calculations()
    rendered = copy.deepcopy(canonical)
    rendered["rows"] = rendered["rows"][:-1]
    audit = validate_host_calculations(_score_table(rendered), canonical)
    assert not audit["numeric_table_verified"]
    assert any("Missing player row: Shaquille O'Neal" in error for error in audit["errors"])
    table = _score_table(canonical).replace("Court Stat Score", "Unsupported Stat")
    audit = validate_host_calculations(table, canonical)
    assert not audit["numeric_table_verified"]
    assert any("court_stat_score_0_10" in error for error in audit["errors"])


def test_host_audit_does_not_accept_numbers_hidden_in_prose_cells():
    canonical = build_host_calculations()
    table = _score_table(canonical).replace("| 6 | 5 |", "| 6 titles and 9 claims | 5 |", 1)
    audit = validate_host_calculations(table, canonical)
    assert not audit["numeric_table_verified"]
    assert any("Michael Jordan.championships" in error for error in audit["errors"])


def test_host_audit_requires_new_weighted_dimensions_when_formula_expands():
    canonical = build_host_calculations()
    canonical["weights"]["economic_impact"] = "0.1"
    for row in canonical["rows"]:
        row["economic_impact"] = "8.000"
    assert validate_host_calculations(_score_table(canonical), canonical)["numeric_table_verified"]
    omitted = _score_table(canonical).replace("Economic Impact", "Unknown")
    assert not validate_host_calculations(omitted, canonical)["numeric_table_verified"]


def test_measured_host_calculations_requires_source_backed_economy_and_all_dimensions():
    required = ["court_stat_score_0_10", "audience", "current_generation",
                "next_generation", "economy", "social_cultural"]
    observations = []
    for player, value in (("Michael Jordan", "10"), ("LeBron James", "9")):
        row = {"player": player}
        for field in required:
            row[field] = {"value": value, "upper_bound": "10", "source_ids": [f"source_{field}"],
                          "measure_id": field, "unit": "count", "as_of": "2025-12-31"}
        observations.append(row)
    result = build_measured_host_calculations(observations)
    assert result["version"] == "nba-host-calculation-measured-v1"
    assert result["winner"] == "Michael Jordan"
    assert "economy" in result["dimensions"]
    missing_source = copy.deepcopy(observations)
    missing_source[0]["economy"]["source_ids"] = []
    with pytest.raises(ValueError, match="source_ids"):
        build_measured_host_calculations(missing_source)
    changed_scope = copy.deepcopy(observations)
    changed_scope[1]["economy"]["as_of"] = "2024-12-31"
    with pytest.raises(ValueError, match="incompatible"):
        build_measured_host_calculations(changed_scope)
    changed_units = copy.deepcopy(observations)
    changed_units[1]["economy"]["unit"] = "USD million"
    with pytest.raises(ValueError, match="incompatible"):
        build_measured_host_calculations(changed_units)


def test_measured_host_calculations_normalizes_and_clips_with_same_bound():
    required = ["court_stat_score_0_10", "audience", "current_generation",
                "next_generation", "economy", "social_cultural"]
    rows = []
    for player, value in (("Player A", "5"), ("Player B", "30")):
        rows.append({"player": player, **{
            field: {"value": value, "upper_bound": "10", "source_ids": ["source"],
                    "measure_id": field, "unit": "count", "as_of": "2025-12-31"}
            for field in required
        }})
    result = build_measured_host_calculations(rows)
    assert result["rows"][0]["total_0_10"] == "5.000"
    assert result["rows"][1]["total_0_10"] == "10.000"
    assert result["normalization"]["economy"]["formula"] == "10 * min(value / upper_bound, 1)"


def test_weight_sensitivity_is_explicitly_conditional_and_recomputable():
    result = build_weight_sensitivity(build_host_calculations(), {
        "court_heavy": {
            "court_stat_score_0_10": "0.60", "audience": "0.10",
            "current_generation": "0.10", "next_generation": "0.10",
            "social_cultural": "0.10",
        }
    })
    assert result["version"] == "nba-weight-sensitivity-v1"
    assert {row["scenario"] for row in result["scenarios"]} == {"base", "court_heavy"}
    assert result["scenarios"][0]["winner"] == "Michael Jordan"
    assert "Conditional" in result["scope"]


@pytest.mark.parametrize("weights", [
    {"court_stat_score_0_10": "1"},
    {"court_stat_score_0_10": "NaN", "audience": "0.2", "current_generation": "0.15",
     "next_generation": "0.15", "social_cultural": "0.2"},
    {"court_stat_score_0_10": "0.3", "audience": "0.2", "current_generation": "0.15",
     "next_generation": "0.15", "social_cultural": "0.2", "extra": "0"},
])
def test_weight_sensitivity_rejects_undefined_or_nonfinite_scenarios(weights):
    with pytest.raises(ValueError):
        build_weight_sensitivity(build_host_calculations(), {"invalid": weights})


def test_weight_sensitivity_preserves_ties_and_rejects_base_override():
    calculations = build_host_calculations()
    equal = {field: "0" for field in calculations["weights"]}
    equal["current_generation"] = "1"
    result = build_weight_sensitivity(calculations, {"current_only": equal})
    assert result["scenarios"][1]["co_winners"] == ["Michael Jordan", "LeBron James", "Stephen Curry"]
    assert result["scenarios"][1]["margin"] == "0.000"
    with pytest.raises(ValueError, match="reserved"):
        build_weight_sensitivity(calculations, {"base": calculations["weights"]})
