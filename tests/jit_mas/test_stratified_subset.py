"""Public-only checks for the domain-balanced independent batch subset."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

from jit_mas.schemas import digest
from jit_mas.joint_protocol import BENCHMARKS, PARTITIONS, SOURCES
from jit_mas.independent_batch_protocol import build_protocol, validate_protocol
from jit_mas.stratified_subset import (
    COUNTS, RR_DOMAINS_SHA256, build_stratified_subset, render_assignments,
    select_equal_strata, validate_stratified_subset,
)
from scripts import prepare_stratified_batch_subset


DIRECTORY = Path(__file__).resolve().parents[2] / "paper" / "experiments"


@pytest.fixture(scope="module")
def registration():
    def read(name):
        return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))

    return {
        "parent": read("joint_task_splits_v4.json"),
        "previous": read("joint_task_splits_v5.json"),
        "exposures": read("test_exposure_20261004.json"),
        "frozen": read("joint_task_splits_v6_stratified.json"),
    }


def _stratified_rows(populations):
    rows = []
    for stratum, population in populations.items():
        for index in range(population):
            task_id = f"{stratum}:{index}"
            rows.append({"task_id": task_id, "group_id": task_id,
                         "selection_stratum": stratum})
    return rows


def _selected_counts(rows, selected):
    index = {row["task_id"]: row["selection_stratum"] for row in rows}
    return Counter(index[task_id] for task_id in selected)


def test_equal_domain_sampling_does_not_repeat_population_imbalance():
    rows = _stratified_rows({"large": 24, "small": 6})
    eligible = {row["task_id"] for row in rows}
    selected = select_equal_strata(rows, eligible, 10, "fixed")
    assert len(set(selected)) == 10
    assert _selected_counts(rows, selected) == {"large": 5, "small": 5}
    assert selected == select_equal_strata(list(reversed(rows)), eligible, 10, "fixed")


def test_scarce_domain_is_exhausted_and_remaining_quota_is_balanced():
    rows = _stratified_rows({"large": 60, "medium": 5, "rare": 2})
    selected = select_equal_strata(rows, {row["task_id"] for row in rows}, 12, "fixed")
    assert _selected_counts(rows, selected) == {"large": 5, "medium": 5, "rare": 2}


def test_small_validation_subset_covers_as_many_domains_as_possible():
    rows = _stratified_rows({f"domain-{index}": 5 for index in range(12)})
    selected = select_equal_strata(rows, {row["task_id"] for row in rows}, 10, "fixed")
    assert len(_selected_counts(rows, selected)) == 10
    assert set(_selected_counts(rows, selected).values()) == {1}


def test_equal_sampling_balances_the_complete_subset_including_fixed_tasks():
    rows = _stratified_rows({"first": 24, "second": 24})
    eligible = {row["task_id"] for row in rows}
    fixed = {f"first:{index}" for index in range(20)}
    selected = select_equal_strata(rows, eligible, 40, "fixed", fixed_ids=fixed)
    assert fixed <= set(selected)
    assert _selected_counts(rows, selected) == {"first": 20, "second": 20}


def test_unavoidably_overrepresented_fixed_domain_leaves_other_domains_balanced():
    rows = _stratified_rows({"first": 24, "second": 24, "third": 24})
    fixed = {f"first:{index}" for index in range(20)}
    selected = select_equal_strata(rows, {row["task_id"] for row in rows}, 40,
                                  "fixed", fixed_ids=fixed)
    assert _selected_counts(rows, selected) == {"first": 20, "second": 10, "third": 10}


def test_equal_sampling_preserves_duplicate_groups_and_rejects_impossible_quota():
    rows = [
        {"task_id": f"{stratum}:{index}", "group_id": stratum,
         "selection_stratum": stratum}
        for stratum in ("first", "second") for index in range(2)
    ]
    eligible = {row["task_id"] for row in rows}
    selected = select_equal_strata(rows, eligible, 2, "fixed")
    assert set(selected) in ({"first:0", "first:1"}, {"second:0", "second:1"})
    with pytest.raises(ValueError, match="Exact quotas"):
        select_equal_strata(rows, eligible, 3, "fixed")
    with pytest.raises(ValueError, match="duplicate group"):
        select_equal_strata(rows, {"first:0"}, 1, "fixed")


def test_equal_sampling_rejects_unknown_task_and_handles_empty_partition():
    rows = _stratified_rows({"domain": 3})
    assert select_equal_strata(rows, set(), 0, "fixed") == []
    with pytest.raises(ValueError, match="unknown"):
        select_equal_strata(rows, {"absent"}, 1, "fixed")


def test_frozen_stratified_registration_rebuilds_from_public_metadata(registration):
    frozen = registration["frozen"]
    domains = frozen["researchrubrics_public_domains"]
    assert digest(domains) == RR_DOMAINS_SHA256
    assert all(set(row) == {"task_id", "domain"} for row in domains)
    assert frozen == build_stratified_subset(
        registration["parent"], registration["previous"],
        list(reversed(domains)), registration["exposures"])
    assert validate_stratified_subset(
        frozen, registration["parent"], registration["previous"],
        registration["exposures"]) == frozen
    assert (DIRECTORY / "task_assignments_v6_stratified.md").read_text(
        encoding="utf-8") == render_assignments(frozen)
    protocol = json.loads((DIRECTORY / "independent_protocol_v6_stratified.json").read_text(
        encoding="utf-8"))
    assert protocol == build_protocol(frozen)
    assert validate_protocol(protocol, frozen) == protocol


def test_stratified_memberships_keep_exact_quotas_groups_and_exposure_isolation(registration):
    frozen = registration["frozen"]
    index = {row["task_id"]: row for row in frozen["task_index"]}
    seen = set()
    groups = {}
    for name in BENCHMARKS:
        for part, count in zip(PARTITIONS, COUNTS[name]):
            tasks = frozen["memberships"][name][part]
            assert len(tasks) == count
            assert not seen.intersection(tasks)
            seen.update(tasks)
            assert all(index[task_id]["partition"] == part for task_id in tasks)
    assert len(seen) == sum(sum(counts) for counts in COUNTS.values())
    for row in index.values():
        groups.setdefault((row["benchmark"], row["group_id"]), set()).add(row["partition"])
        if row["benchmark"] == "researchrubrics" and row["former_development"]:
            assert row["partition"] == "unused"
    assert all(len(parts) == 1 for parts in groups.values())
    excluded = {row["task_id"] for row in registration["exposures"]["tasks"]}
    assert all(index[task_id]["partition"] not in ("validation", "test")
               for task_id in excluded)
    supplemental = {task_id for event in registration["exposures"].get(
        "supplemental_exposure_events", [])
        for task_id in event.get("known_additional_legacy_task_ids", [])}
    assert set(frozen["supplemental_known_exposed_task_ids"]) == supplemental
    assert "6847465956a0f6376a605476" in supplemental
    assert index["6847465956a0f6376a605476"]["partition"] == "test"
    assert index["6847465956a0f6376a605476"]["known_supplemental_exposure"]
    assert index["writingbench:1"]["partition"] == "evolution"
    assert set(frozen["memberships"]["researchrubrics"]["test"]) == (
        set(registration["previous"]["memberships"]["researchrubrics"]["test"]) - excluded)
    for schedule in frozen["joint_evolution_schedule"]:
        assert len(schedule["task_ids"]) == len(set(schedule["task_ids"])) == 120
        for source in SOURCES:
            assert {task_id for task_id in schedule["task_ids"]
                    if index[task_id]["benchmark"] == source} == set(
                        frozen["memberships"][source]["evolution"])


def test_writingbench_covers_six_domains_and_balances_languages_in_each_subset(registration):
    audits = registration["frozen"]["stratification_audit"]["writingbench"]
    for part, language_count in (("evolution", 20), ("validation", 5), ("test", 25)):
        audit = audits[part]
        counts = audit["domain_counts"]
        assert len(counts) == 6
        assert max(counts.values()) - min(counts.values()) <= 1
        assert audit["language_counts"] == {"en": language_count, "zh": language_count}
        for domain in counts:
            language_counts = audit["domain_language_counts"]
            assert abs(language_counts.get(f"en|{domain}", 0)
                       - language_counts.get(f"zh|{domain}", 0)) <= 1


def test_rr_validation_covers_available_domains_and_records_missing_parent_domain(registration):
    audit = registration["frozen"]["stratification_audit"]["researchrubrics"]["validation"]
    assert len(audit["domain_counts"]) == 9
    assert sorted(audit["domain_counts"].values()) == [1] * 8 + [2]
    assert not audit["unrepresented_available_domains"]
    assert "Hypotheticals & Philosophy" in audit["unavailable_domains"]


def test_dsqa_redistributes_scarce_domain_capacity_without_large_domain_dominance(registration):
    audits = registration["frozen"]["stratification_audit"]["deepsearchqa"]
    for part in PARTITIONS:
        audit = audits[part]
        counts = audit["domain_counts"]
        capacities = audit["available_domain_counts"]
        unsaturated = [counts.get(domain, 0) for domain, capacity in capacities.items()
                       if counts.get(domain, 0) < capacity]
        assert len(counts) >= min(audit["count"], len(capacities))
        assert not unsaturated or max(unsaturated) - min(unsaturated) <= 1
        assert all(count <= capacities[domain] for domain, count in counts.items())


def test_every_distribution_audit_matches_selected_public_rows(registration):
    frozen = registration["frozen"]
    for name in BENCHMARKS:
        rows = [row for row in frozen["task_index"] if row["benchmark"] == name]
        domains = {row["domain"] for row in rows}
        for part in PARTITIONS:
            audit = frozen["stratification_audit"][name][part]
            selected = [row for row in rows if row["partition"] == part]
            assert audit["count"] == len(selected)
            assert audit["domain_counts"] == Counter(row["domain"] for row in selected)
            assert audit["language_counts"] == Counter(
                row["language"] for row in selected if row["language"])
            available = set(audit["available_domain_counts"])
            assert set(audit["unrepresented_available_domains"]) == (
                available - set(audit["domain_counts"]))
            assert set(audit["unavailable_domains"]) == domains - available
            targets = audit["equal_target_domain_counts"]
            deviations = {domain: audit["domain_counts"].get(domain, 0) - target
                          for domain, target in targets.items()}
            assert audit["domain_quota_deviations"] == deviations
            assert audit["quota_absolute_deviation"] == sum(map(abs, deviations.values()))
            ideal = len(selected) / len(domains)
            assert audit["uniform_domain_ideal_count"] == pytest.approx(ideal)
            assert audit["uniform_domain_absolute_deviation"] == pytest.approx(
                sum(abs(audit["domain_counts"].get(domain, 0) - ideal) for domain in domains))
            assert audit["supplemental_exposed_task_ids"] == sorted(
                row["task_id"] for row in selected if row["known_supplemental_exposure"])
    drbii = frozen["stratification_audit"]["deepresearch_bench_ii"]["test"]
    assert drbii["language_counts"] == {"en": 20, "zh": 20}
    for name in ("ifeval", "ifbench"):
        audit = frozen["stratification_audit"][name]["test"]
        assert "no public domain labels" in audit["selection_policy"]


@pytest.mark.parametrize("location", ["public_domains", "task_domain", "audit"])
def test_self_rehashed_domain_or_audit_tampering_is_rejected(registration, location):
    changed = copy.deepcopy(registration["frozen"])
    if location == "public_domains":
        changed["researchrubrics_public_domains"][0]["domain"] = "forged"
        changed["researchrubrics_public_domains_sha256"] = digest(
            changed["researchrubrics_public_domains"])
    elif location == "task_domain":
        changed["task_index"][0]["domain"] = "forged"
    else:
        changed["stratification_audit"]["writingbench"]["validation"]["domain_counts"] = {
            "forged": 10}
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="pinned"):
        validate_stratified_subset(changed, registration["parent"],
                                   registration["previous"], registration["exposures"])


def test_cli_creates_checks_registration_and_preserves_old_v6_bytes(
        registration, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare_stratified_batch_subset, "public_rr_domains",
                        lambda *_: registration["frozen"]["researchrubrics_public_domains"])
    old_paths = [DIRECTORY / name for name in (
        "joint_task_splits_v6.json", "independent_protocol_v6.json", "task_assignments_v6.md")]
    old_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths}
    args = ["--directory", str(DIRECTORY), "--output-dir", str(tmp_path)]
    assert prepare_stratified_batch_subset.main(args) == 0
    assert prepare_stratified_batch_subset.main(args) == 0
    assert prepare_stratified_batch_subset.main(
        args + ["--check", "--rr-data", "does-not-exist.jsonl"]) == 0
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(receipts) == 3
    assert all(row["valid"] and row["model_calls"] == 0 for row in receipts)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths} == old_hashes
    protocol_path = tmp_path / "independent_protocol_v6_stratified.json"
    original_protocol = protocol_path.read_text(encoding="utf-8")
    protocol_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reconstruction"):
        prepare_stratified_batch_subset.main(args + ["--check"])
    assert protocol_path.read_text(encoding="utf-8") == "{}"
    protocol_path.write_text(original_protocol, encoding="utf-8")
    changed = tmp_path / "task_assignments_v6_stratified.md"
    changed.write_text("forged", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reconstruction"):
        prepare_stratified_batch_subset.main(args)
    assert changed.read_text(encoding="utf-8") == "forged"


def test_public_domain_loader_rejects_unpinned_raw_bytes_before_parsing(registration, tmp_path):
    raw = tmp_path / "processed_data.jsonl"
    raw.write_text("not the pinned dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="raw bytes differ"):
        prepare_stratified_batch_subset.public_rr_domains(registration["parent"], raw)
