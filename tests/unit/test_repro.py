from copy import deepcopy

import pytest

from unlimited_ocr_2_rocm.repro import compare_reproduction


def successful():
    return {
        "gates": {"A": "PASS", "B": "PASS", "C": "PASS", "D": "PASS"},
        "environment_id": "first",
        "fresh_environment": True,
        "machine_identity": {"uuid_sha256": "device1", "build": "26200", "arch": "gfx1151"},
        "artifact_hashes": {"model": "modelhash", "fixtures": "fixturehash", "wrapper": "codehash"},
        "package_versions": {"torch": "2.12.0+rocm7.14.1"},
        "required_cases_passed": True,
        "producer_evidence_valid": True,
    }


def test_gate_e_requires_a_successful_distinct_clean_baseline_pair():
    first = successful()
    second = deepcopy(first)
    second["environment_id"] = "second"
    assert compare_reproduction(first, second)["status"] == "PASS"
    assert compare_reproduction(None, second)["status"] == "NOT PROVEN"
    assert compare_reproduction(first, first)["status"] == "NOT PROVEN"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("fresh_environment", False),
        ("machine_identity", {"uuid_sha256": "other_device"}),
        ("machine_identity", {}),
        ("artifact_hashes", {"model": "changed"}),
        ("package_versions", {"torch": "other"}),
        ("required_cases_passed", False),
        ("gates", {"A": "PASS", "B": "PASS", "C": "FAIL", "D": "NOT TESTED"}),
    ],
)
def test_gate_e_refuses_incomparable_or_failed_runs(field, bad):
    first = successful()
    second = deepcopy(first)
    second["environment_id"] = "second"
    second[field] = bad
    assert compare_reproduction(first, second)["status"] == "NOT PROVEN"


def test_failed_baseline_does_not_count():
    first = successful()
    first["gates"]["C"] = "FAIL"
    second = successful()
    second["environment_id"] = "second"
    assert compare_reproduction(first, second)["status"] == "NOT PROVEN"


def test_provenance_is_bound_to_producers_not_reporting_interpreter(monkeypatch):
    import sys

    from unlimited_ocr_2_rocm.repro import bind_provenance

    workflow = {"clean_environment": False, "run_id": "run1"}
    producers = [{"environment_id": "original", "isolated_environment": True}] * 2
    monkeypatch.setattr(sys, "prefix", "some unrelated environment")
    bound = bind_provenance(workflow, producers, None, False)
    assert bound["environment_id"] == "original"
    assert bound["fresh_environment"] is False
    assert bound["producer_evidence_valid"] is True
    assert bind_provenance(workflow, producers, None, True)["producer_evidence_valid"] is False


def test_freshness_requires_matching_creation_witness():
    from unlimited_ocr_2_rocm.repro import bind_provenance

    workflow = {"clean_environment": True, "run_id": "run2"}
    producers = [{"environment_id": "second", "isolated_environment": True}] * 2
    witness = {"environment_id": "second", "run_id": "run2", "created_new": True}
    assert bind_provenance(workflow, producers, witness, True)["fresh_environment"] is True
    for bad in (None, {}, dict(witness, run_id="different"), dict(witness, created_new=False)):
        assert bind_provenance(workflow, producers, bad, True)["producer_evidence_valid"] is False
    other = [*producers, {"environment_id": "first", "isolated_environment": True}]
    assert bind_provenance(workflow, other, witness, True)["producer_evidence_valid"] is False
