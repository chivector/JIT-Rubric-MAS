"""Repair frozen ESG component arithmetic without model or network calls."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

BRANDS = ("Patagonia", "Fast Retailing", "H&M")
METRIC_NAMES = {
    "E": ("Scope1+2 trend judgment", "Scope3 trend judgment", "Renewable electricity judgment", "Materials judgment"),
    "S": ("Audit-evidence judgment", "Living-wage judgment", "Grievance judgment", "Forced-labor judgment"),
    "T": ("Supplier-disclosure judgment", "Data-completeness judgment", "Third-party-verification judgment"),
}
METRIC_WEIGHTS = {"E": (".4", ".3", ".2", ".1"),
                  "S": (".4", ".3", ".2", ".1"), "T": (".4", ".3", ".3")}
COMPONENTS = {
    "Patagonia": {"E": ("6", "4", "9.8", "8.4"), "S": ("5", "5", "10", "10"), "T": ("0", "7", "10")},
    "Fast Retailing": {"E": ("8", "6", "9.4", "1.8"), "S": ("7", "0", "5", "10"), "T": ("0", "5", "5")},
    "H&M": {"E": ("7", "5", "9.6", "8.9"), "S": ("6.4", "0", "10", "5"), "T": ("9.9", "10", "5")},
}
CATEGORY_WEIGHTS = {"E": Decimal(".4"), "S": Decimal(".35"), "T": Decimal(".25")}
ALTERNATIVE_WEIGHTS = {"E": Decimal(".5"), "S": Decimal(".3"), "T": Decimal(".2")}


def calculate():
    categories = {brand: {category: sum(Decimal(value) * Decimal(weight)
        for value, weight in zip(COMPONENTS[brand][category], METRIC_WEIGHTS[category]))
        for category in METRIC_NAMES} for brand in BRANDS}
    totals = {brand: sum(categories[brand][category] * weight
        for category, weight in CATEGORY_WEIGHTS.items()) for brand in BRANDS}
    alternatives = {brand: sum(categories[brand][category] * weight
        for category, weight in ALTERNATIVE_WEIGHTS.items()) for brand in BRANDS}
    ranks = sorted(BRANDS, key=lambda brand: totals[brand], reverse=True)
    return categories, totals, alternatives, ranks


def render():
    categories, totals, alternatives, ranks = calculate()
    rows = ["# Frozen component arithmetic", "",
        "All component scores are subjective analyst inputs from compact v5, not source-calibrated measurements. No threshold adjustment, reweighting or predetermined rank is applied.", "",
        "Category = sum(component score * within-category weight). Composite = .40*E + .35*S + .25*T. Alternative = .50*E + .30*S + .20*T.", "",
        "| Category | Metric | Weight | Patagonia | Fast Retailing | H&M |", "|---|---|---|---|---|---|"]
    for category, names in METRIC_NAMES.items():
        for index, name in enumerate(names):
            values = [category, name, METRIC_WEIGHTS[category][index],
                      *[COMPONENTS[brand][category][index] for brand in BRANDS]]
            rows.append("| " + " | ".join(values) + " |")
    rows.extend(["", "| Category | Patagonia | Fast Retailing | H&M |", "|---|---|---|---|"])
    for category in METRIC_NAMES:
        values = [category, *[" + ".join(f"{value}*{weight}" for value, weight
            in zip(COMPONENTS[brand][category], METRIC_WEIGHTS[category]))
            + " = " + str(categories[brand][category]) for brand in BRANDS]]
        rows.append("| " + " | ".join(values) + " |")
    rows.extend(["", "| Rank | Brand | E*.40 | S*.35 | T*.25 | Total | Alternative |",
                 "|---|---|---|---|---|---|---|"])
    for rank, brand in enumerate(ranks, 1):
        values = [str(rank), brand, *[str(categories[brand][category] * CATEGORY_WEIGHTS[category])
            for category in METRIC_NAMES], str(totals[brand]), str(alternatives[brand])]
        rows.append("| " + " | ".join(values) + " |")
    calculations = {"components": COMPONENTS, "metric_weights": METRIC_WEIGHTS,
        "categories": {brand: {category: str(value) for category, value in values.items()}
                       for brand, values in categories.items()},
        "totals": {brand: str(value) for brand, value in totals.items()},
        "alternative_totals": {brand: str(value) for brand, value in alternatives.items()}, "ranking": ranks}
    return "\n".join(rows) + "\n", calculations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-answer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body, calculations = render()
    if args.input_answer:
        original = args.input_answer.read_text(encoding="utf-8")
        calculations["source_answer_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arithmetic.md").write_text(body, encoding="utf-8")
    (args.output / "calculations.json").write_text(json.dumps(calculations, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
