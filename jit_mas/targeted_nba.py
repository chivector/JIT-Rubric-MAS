"""Deterministic, source-bound NBA comparison calculations.

The model may write the article, but the comparison inputs, normalization and
argmax are calculated by the host.  Keeping this small calculation outside the
prompt makes it possible to audit the exact numbers shown to the model and to
check a submitted answer without trusting prose arithmetic.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any


PLAYERS = (
    "Michael Jordan", "LeBron James", "Magic Johnson", "Kobe Bryant",
    "Stephen Curry", "Shaquille O'Neal",
)

# These are facts in nba_verified_fact_matrix_v3.json.  Every row has two
# basketball-specific counters so rubric checks do not depend on prose.
FACTS = {
    "Michael Jordan": {"championships": 6, "mvps": 5, "source_ids": ["nba_jordan_legend"]},
    "LeBron James": {"championships": 4, "mvps": 4, "source_ids": ["nba_lebron_archive"]},
    "Magic Johnson": {"championships": 5, "mvps": 3, "source_ids": ["nba_magic_legend"]},
    "Kobe Bryant": {"championships": 5, "mvps": 1, "source_ids": ["kobe_hof"]},
    "Stephen Curry": {"championships": 4, "mvps": 2, "source_ids": ["ua_curry_tour", "nba_curry_profile"]},
    "Shaquille O'Neal": {"championships": 4, "mvps": 1, "source_ids": ["shaq_hof"]},
}

# The historical pilot deliberately uses ``build_host_calculations`` below and
# its receipt must remain byte-for-byte stable.  The helpers added after it are
# opt-in tools for future source-complete runs; they do not alter the pilot
# inputs or scores.
MEASURED_DIMENSIONS = (
    "audience", "current_generation", "next_generation", "economy",
    "social_cultural",
)
MEASURED_DEFAULT_WEIGHTS = {
    "court_stat_score_0_10": Decimal("0.30"),
    "audience": Decimal("0.20"),
    "current_generation": Decimal("0.15"),
    "next_generation": Decimal("0.15"),
    "economy": Decimal("0.10"),
    "social_cultural": Decimal("0.10"),
}


def _q(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def _parse_nonnegative_decimal(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{label} is not a decimal") from None
    if not number.is_finite() or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def build_host_calculations() -> dict[str, Any]:
    """Return a reproducible bounded matrix and weighted totals.

    The two count-based dimensions are normalized by the same fixed observed
    maxima.  The remaining four dimensions are intentionally transparent,
    bounded analyst inputs supplied as hypotheses; they cannot silently alter
    the count normalization.  The final score is a weighted sum and is always
    recomputed from the displayed operands.
    """
    max_champ = Decimal(max(row["championships"] for row in FACTS.values()))
    max_mvp = Decimal(max(row["mvps"] for row in FACTS.values()))
    # Stable qualitative inputs are bounded and explicitly labelled as
    # analysis inputs. They are not presented as measured statistics.
    qualitative = {
        "Michael Jordan": (9, 10, 10, 10),
        "LeBron James": (9, 10, 10, 9),
        "Magic Johnson": (8, 9, 9, 10),
        "Kobe Bryant": (8, 9, 9, 8),
        "Stephen Curry": (9, 10, 9, 9),
        "Shaquille O'Neal": (8, 8, 8, 8),
    }
    rows = []
    for player in PLAYERS:
        fact = FACTS[player]
        audience, current, next_gen, social = qualitative[player]
        court = _q(Decimal(fact["championships"]) / max_champ * Decimal(5)
                    + Decimal(fact["mvps"]) / max_mvp * Decimal(5))
        total = _q(Decimal(court) * Decimal("0.30")
                   + Decimal(audience) * Decimal("0.20")
                   + Decimal(current) * Decimal("0.15")
                   + Decimal(next_gen) * Decimal("0.15")
                   + Decimal(social) * Decimal("0.20"))
        rows.append({"player": player, **fact, "audience": audience,
                     "current_generation": current, "next_generation": next_gen,
                     "social_cultural": social, "court_stat_score_0_10": str(court),
                     "total_0_10": str(total)})
    winner = max(rows, key=lambda row: Decimal(row["total_0_10"]))["player"]
    return {
        "version": "nba-host-calculation-v1",
        "scope": "facts available by 2025; Curry title count corroborated by a 2025 source; current-page scope is recorded separately",
        "players": list(PLAYERS),
        "normalization": {
            "championships": {"formula": "championships / 6 * 5", "max_observed": 6},
            "mvps": {"formula": "regular_season_mvps / 5 * 5", "max_observed": 5},
            "bounded_range": [0, 10],
        },
        "weights": {"court_stat_score_0_10": "0.30", "audience": "0.20",
                    "current_generation": "0.15", "next_generation": "0.15",
                    "social_cultural": "0.20"},
        "rows": rows,
        "formula": "0.30*court_stat_score + 0.20*audience + 0.15*current_generation + 0.15*next_generation + 0.20*social_cultural",
        "winner": winner,
        "winner_rule": "argmax(total_0_10) after host recomputation",
    }


def build_measured_host_calculations(
    observations: list[dict[str, Any]],
    *,
    weights: dict[str, str | Decimal] | None = None,
) -> dict[str, Any]:
    """Build an opt-in score from source-backed, comparable measurements.

    ``observations`` is intentionally strict.  Each row must contain all six
    dimensions and each dimension must be a mapping with numeric ``value``
    and ``upper_bound``, comparable ``measure_id``, ``unit`` and ``as_of``
    metadata, and at least one ``source_ids`` entry. The host applies the
    same declared bounded normalization to every player in a dimension. This
    makes an economic dimension explicit without inventing revenue or
    audience numbers when the evidence pack does not contain them.  The
    function is not called by the v1 pilot; it is a guardrail for a future
    source-complete experiment.
    """
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")
    selected = dict(MEASURED_DEFAULT_WEIGHTS)
    if weights is not None:
        selected = {}
        for key, value in weights.items():
            try:
                selected[key] = Decimal(str(value))
            except (InvalidOperation, ValueError):
                raise ValueError(f"invalid weight for {key!r}") from None
    required = ("court_stat_score_0_10", *MEASURED_DIMENSIONS)
    if set(selected) != set(required):
        raise ValueError("weights must contain exactly: " + ", ".join(required))
    if (any(not value.is_finite() or value < 0 for value in selected.values())
            or sum(selected.values()) != Decimal("1")):
        raise ValueError("weights must be non-negative and sum exactly to 1")

    players: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    common_measures: dict[str, dict[str, str]] = {}
    for observation in observations:
        if not isinstance(observation, dict) or not isinstance(observation.get("player"), str):
            raise ValueError("every observation needs a player")
        player = observation["player"]
        if player in players:
            raise ValueError(f"duplicate player: {player}")
        players.append(player)
        for field in required:
            item = observation.get(field)
            if not isinstance(item, dict):
                raise ValueError(f"{player}.{field} needs value metadata")
            source_ids = item.get("source_ids")
            if not isinstance(source_ids, list) or not source_ids or not all(isinstance(s, str) and s for s in source_ids):
                raise ValueError(f"{player}.{field} needs non-empty source_ids")
            if not isinstance(item.get("measure_id"), str) or not item["measure_id"].strip():
                raise ValueError(f"{player}.{field} needs measure_id")
            if not isinstance(item.get("unit"), str) or not item["unit"].strip():
                raise ValueError(f"{player}.{field} needs unit")
            if not isinstance(item.get("as_of"), str) or not item["as_of"].strip():
                raise ValueError(f"{player}.{field} needs as_of")
            try:
                number = Decimal(str(item.get("value")))
                upper = Decimal(str(item.get("upper_bound")))
            except (InvalidOperation, ValueError):
                raise ValueError(f"{player}.{field} has invalid value or upper_bound") from None
            if not number.is_finite() or number < 0 or not upper.is_finite() or upper <= 0:
                raise ValueError(f"{player}.{field} needs non-negative value and positive upper_bound")
            measure = {"measure_id": item["measure_id"], "unit": item["unit"],
                       "as_of": item["as_of"], "upper_bound": str(upper),
                       "formula": "10 * min(value / upper_bound, 1)"}
            previous = common_measures.setdefault(field, measure)
            if measure != previous:
                raise ValueError(f"{player}.{field} has incompatible measurement metadata")
        values = {field: _q(Decimal("10") * min(
            Decimal(str(observation[field]["value"])) / Decimal(str(observation[field]["upper_bound"])),
            Decimal("1"))) for field in required}
        total = _q(sum(selected[field] * values[field] for field in required))
        normalized_rows.append({
            "player": player,
            **{field: str(values[field].quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)) for field in required},
            "total_0_10": str(total),
            "source_ids": {field: list(observation[field]["source_ids"]) for field in required},
            "raw_values": {field: str(observation[field]["value"]) for field in required},
            "measure_ids": {field: observation[field]["measure_id"] for field in required},
            "units": {field: observation[field]["unit"] for field in required},
            "as_of": {field: observation[field]["as_of"] for field in required},
        })
    winner = max(normalized_rows, key=lambda row: Decimal(row["total_0_10"]))["player"]
    return {
        "version": "nba-host-calculation-measured-v1",
        "dimensions": list(required),
        "weights": {field: str(selected[field]) for field in required},
        "normalization": common_measures,
        "rows": normalized_rows,
        "formula": " + ".join(f"{selected[field]}*{field}" for field in required),
        "winner": winner,
        "winner_rule": "argmax(total_0_10) after host recomputation",
        "scope": "Opt-in measured observations with declared comparable proxies; not used by historical v1 receipts or a certificate of unbiased influence.",
    }


def build_weight_sensitivity(
    calculations: dict[str, Any] | None = None,
    scenarios: dict[str, dict[str, str | Decimal]] | None = None,
) -> dict[str, Any]:
    """Recompute winners under declared weight scenarios.

    This reports conditional robustness and is deliberately separate from the
    official score.  It can consume either the historical v1 receipt or the
    measured receipt above, provided every weighted field is present in each
    row.  No scenario is silently chosen by the host.
    """
    calculations = calculations or build_host_calculations()
    weights = calculations.get("weights")
    rows = calculations.get("rows")
    if not isinstance(weights, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("calculations lacks weights or rows")
    fields = list(weights)
    base = {field: _parse_nonnegative_decimal(value, f"base weight {field}")
            for field, value in weights.items()}
    named = {"base": base}
    for name, values in (scenarios or {}).items():
        if name == "base":
            raise ValueError("scenario 'base' is reserved for the supplied receipt")
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError(f"scenario {name!r} must specify every weighted field")
        parsed = {field: _parse_nonnegative_decimal(values[field], f"{name} weight {field}") for field in fields}
        named[name] = parsed
    result = []
    for name, scenario_weights in named.items():
        if sum(scenario_weights.values()) != Decimal("1"):
            raise ValueError(f"scenario {name!r} weights must be non-negative and sum exactly to 1")
        scored = []
        for row in rows:
            try:
                values = {field: _parse_nonnegative_decimal(row[field], f"row value {field}") for field in fields}
                if any(value > 10 for value in values.values()):
                    raise ValueError("row value exceeds 10")
                total = _q(sum(scenario_weights[field] * values[field] for field in fields))
            except (KeyError, InvalidOperation, ValueError):
                raise ValueError(f"row missing weighted field for scenario {name!r}") from None
            scored.append({"player": row["player"], "total_0_10": str(total)})
        scored.sort(key=lambda row: Decimal(row["total_0_10"]), reverse=True)
        result.append({"scenario": name, "weights": {field: str(value) for field, value in scenario_weights.items()},
                       "winner": scored[0]["player"], "rows": scored,
                       "co_winners": [row["player"] for row in scored if row["total_0_10"] == scored[0]["total_0_10"]],
                       "margin": str(_q(Decimal(scored[0]["total_0_10"]) - Decimal(scored[1]["total_0_10"]))) if len(scored) > 1 else None})
    return {"version": "nba-weight-sensitivity-v1", "fields": fields, "scenarios": result,
            "scope": "Conditional recomputation; does not establish an objective or unbiased ranking."}


def _plain_markdown(value: str) -> str:
    # Formatting and curly apostrophes are display variants, not different
    # players or values. Do not strip arbitrary prose from a numeric cell.
    return re.sub(r"[*_`]", "", value).strip().replace("\u2019", "'").replace("\u2018", "'")


def _header_key(value: str) -> str:
    value = _plain_markdown(value.replace("_", " ")).casefold()
    value = re.sub(r"\b0\s*(?:[-–/]\s*)?10\b", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


_COLUMN_ALIASES = {
    "player": {"player", "nba player", "candidate", "name"},
    "championships": {"championships", "championship", "titles", "nba titles"},
    "mvps": {"mvps", "mvp", "regular season mvps", "regular season mvp"},
    "court_stat_score_0_10": {"court stat score", "court score", "on court score", "court statistical score"},
    "audience": {"audience", "audience score", "audience impact"},
    "current_generation": {"current gen", "current generation", "current generation score"},
    "next_generation": {"next gen", "next generation", "next generation score"},
    "social_cultural": {"social cultural", "social cultural score", "social culture", "social"},
    "economy": {"economy", "economy score", "economic", "economic score", "economic impact"},
    "economic": {"economy", "economy score", "economic", "economic score", "economic impact"},
    "total_0_10": {"total", "total score", "total influence score", "overall score"},
}


def _column_names(field: str) -> set[str]:
    names = set(_COLUMN_ALIASES.get(field, set()))
    # Newly weighted dimensions work when the renderer names a header after
    # its host row field, without changing this audit when the formula grows.
    base = _header_key(field)
    names.update({base, base + " score"})
    return names


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    def cells(line):
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    lines = text.splitlines()
    tables = []
    for index in range(len(lines) - 1):
        header, divider = lines[index], lines[index + 1]
        if "|" not in header or "|" not in divider:
            continue
        columns, separators = cells(header), cells(divider)
        if (len(columns) < 2 or len(columns) != len(separators)
                or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)):
            continue
        body = []
        for line in lines[index + 2:]:
            if not line.strip() or "|" not in line:
                break
            body.append(cells(line))
        tables.append((columns, body))
    return tables


def _numeric_cell(value: str) -> Decimal | None:
    plain = _plain_markdown(value).replace("\u2212", "-")
    # A trailing source ID is citation metadata, while additional numeric/prose
    # operands must not be mistaken for the cell's actual score.
    plain = re.sub(r"\s*(?:\[[^\]\n]+\])+\s*$", "", plain).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", plain):
        return None
    try:
        number = Decimal(plain)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _matches_displayed_number(actual: Decimal, expected: Decimal) -> bool:
    # Accept the canonical value or its explicitly rounded two/three-place
    # display. A broad tolerance could hide changed input data or wrong totals.
    return actual in {expected, expected.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                      expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)}


def validate_host_calculations(answer: str, calculations: dict[str, Any]) -> dict[str, Any]:
    """Compare a real Markdown score table against every canonical host cell.

    Mention coverage remains a separate diagnostic. Numeric verification does
    not replace the official rubric score or certify the article's semantics.
    """
    text = answer if isinstance(answer, str) else ""
    rows = calculations.get("rows", [])
    plain_text = _plain_markdown(text)
    mentioned = {row["player"]: bool(re.search(re.escape(row["player"]), plain_text, re.I)) for row in rows}
    winner = calculations.get("winner")
    fields = list(dict.fromkeys(["championships", "mvps", *calculations.get("weights", {}), "total_0_10"]))
    expected_by_player = {_plain_markdown(row["player"]).casefold(): row for row in rows}
    candidates = []
    for headers, body in _markdown_tables(text):
        mapping = {}
        for field in ["player", *fields]:
            matches = [index for index, header in enumerate(headers) if _header_key(header) in _column_names(field)]
            if len(matches) == 1:
                mapping[field] = matches[0]
        if "player" not in mapping:
            continue
        coverage = sum(1 for cells in body if len(cells) > mapping["player"]
                       and _plain_markdown(cells[mapping["player"]]).casefold() in expected_by_player)
        candidates.append((len(mapping), coverage, headers, body, mapping))
    errors = []
    checked = []
    selected_headers = None
    if not rows:
        errors.append("Host calculation receipt has no fixed player rows.")
    if not candidates:
        errors.append("No Markdown score table with a Player column was found.")
    else:
        # Prefer the complete score table over fact/source tables elsewhere in
        # the article, then prefer the table containing the fixed player set.
        _, _, selected_headers, body, mapping = max(candidates, key=lambda item: item[:2])
        missing_columns = [field for field in ["player", *fields] if field not in mapping]
        if missing_columns:
            errors.append("Score table missing or duplicating columns: " + ", ".join(missing_columns))
        observed = {}
        for cells in body:
            if len(cells) <= mapping["player"]:
                continue
            key = _plain_markdown(cells[mapping["player"]]).casefold()
            if key not in expected_by_player:
                continue
            if key in observed:
                errors.append("Duplicate player row: " + expected_by_player[key]["player"])
            observed[key] = cells
        for key, row in expected_by_player.items():
            player = row["player"]
            if key not in observed:
                errors.append("Missing player row: " + player)
                continue
            cells = observed[key]
            values = {}
            for field in fields:
                if field not in mapping:
                    continue
                cell_index = mapping[field]
                raw = cells[cell_index] if cell_index < len(cells) else ""
                actual = _numeric_cell(raw)
                try:
                    expected = Decimal(str(row[field]))
                except (KeyError, InvalidOperation):
                    errors.append(f"Host value unavailable: {player}.{field}")
                    continue
                valid = actual is not None and expected.is_finite() and _matches_displayed_number(actual, expected)
                values[field] = {"cell": raw, "observed": str(actual) if actual is not None else None,
                                 "expected": str(expected), "matches": valid}
                if not valid:
                    errors.append(f"Score cell mismatch: {player}.{field}: expected {expected}, got {raw!r}")
            checked.append({"player": player, "cells": values})
    verified = bool(rows) and not errors
    return {
        "version": "nba-host-calculation-audit-v2",
        "host_winner": winner,
        "winner_mentioned": bool(winner and mentioned.get(winner)),
        "all_fixed_players_mentioned": bool(rows) and all(mentioned.values()),
        "player_mentions": mentioned,
        "host_total_table": [{"player": row["player"], "total_0_10": row["total_0_10"]} for row in rows],
        "numeric_table_verified": verified,
        "valid": verified,
        "errors": errors,
        "required_numeric_columns": fields,
        "observed_table_headers": selected_headers,
        "checked_rows": checked,
        "scope": "Markdown numeric cells compared to canonical host inputs and totals; independent of official rubric evaluation and prose quality.",
    }
