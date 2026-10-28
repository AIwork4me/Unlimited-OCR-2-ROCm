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
