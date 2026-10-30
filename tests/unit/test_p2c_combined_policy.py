"""P2-C combined (suppression × raised budget) policy gates (CPU only)."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.inference import validate_ngram

P2C_POLICY = Path(__file__).parents[2] / "eval/experiments/p2c-combined/policy.json"
P2A_POLICY = Path(__file__).parents[2] / "eval/experiments/p2a-token-budget/policy.json"


def load_module(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p2c_policy_authorizes_the_official_suppression():
    assert validate_ngram(35, 1024, P2C_POLICY) is None


def test_p2c_policy_is_pinned_and_unique():
    from unlimited_ocr_2_rocm.inference import (
        _AUTHORIZED_NGRAM_POLICIES,  # noqa: PLC2701 - pin-uniqueness test
        EXPECTED_NGRAM_POLICY_SHA256,
        EXPECTED_P2C_POLICY_SHA256,
    )

    assert EXPECTED_NGRAM_POLICY_SHA256 != EXPECTED_P2C_POLICY_SHA256
    assert len(_AUTHORIZED_NGRAM_POLICIES) == 2


def test_p2c_forged_policy_is_rejected_by_pinned_digest(tmp_path):
    forged = tmp_path / "forged-p2c.json"
    payload = json.loads(P2C_POLICY.read_text("utf-8"))
    payload["budgets_new_tokens"] = [12288, 16384, 39321]
    forged.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="not one of the pinned frozen"):
        validate_ngram(35, 1024, forged)


def test_p2c_policy_does_not_authorize_39321_or_other_sizes():
    policy = json.loads(P2C_POLICY.read_text("utf-8"))
    assert 39321 not in policy["budgets_new_tokens"]
    assert policy["budgets_new_tokens"] == [12288, 16384]
    assert policy["ngram_sizes"] == [35]
    assert policy["ngram_window"] == 1024
    with pytest.raises(ValueError, match="not authorized"):
        validate_ngram(35, 1023, P2C_POLICY)


def test_driver_prepare_requires_budget_in_both_policies(tmp_path):
    driver = load_module("run_p2c_combined", "scripts/run_p2c_combined.py")
    base = dict(
        policy=P2C_POLICY,
        budget_policy=P2A_POLICY,
        ngram_size=35,
        ground_truth=tmp_path / "gt.json",
        dataset_root=tmp_path,
        lock=tmp_path / "lock.json",
        output=tmp_path / "out",
    )
    for bad_budget in (8192, 39321):
        args = argparse.Namespace(**base, budget=bad_budget)
        with pytest.raises(ValueError, match="not authorized"):
            driver.prepare_command(args)
    # 39321 must fail even if a forged P2-C policy listed it: the P2-A budget
    # policy is a separate, jointly required gate.
    forged = tmp_path / "forged-p2c.json"
    payload = json.loads(P2C_POLICY.read_text("utf-8"))
    payload["budgets_new_tokens"] = [39321]
    forged.write_text(json.dumps(payload), "utf-8")
    forged_base = dict(base, policy=forged)
    args = argparse.Namespace(**forged_base, budget=39321)
    with pytest.raises(ValueError, match="not one of the pinned frozen|not authorized"):
        driver.prepare_command(args)


def test_driver_policy_binding_and_admission(tmp_path):
    driver = load_module("run_p2c_combined", "scripts/run_p2c_combined.py")
    policy = driver.load_policy(P2C_POLICY)
    assert policy["experiment"] == "P2C_COMBINED_NGRAM_BUDGET"
    for budget in (12288, 16384):
        admitted = driver.admission_report(policy, pages=5, budget=budget)
        assert admitted["admitted"] is True
        assert admitted["total_upper_bound_tokens"] == 5 * 300 + budget + 4096
    tight = dict(policy["admission"])
    tight["context_cap_tokens"] = 16000
    refused = driver.admission_report({"admission": tight}, pages=5, budget=16384)
    assert refused["admitted"] is False
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": 1, "experiment": "P2B_NGRAM_SUPPRESSION_RCA"}), "utf-8")
    with pytest.raises(ValueError, match="Not the frozen P2-C"):
        driver.load_policy(wrong)
    with pytest.raises(ValueError, match="Not the frozen P2-A"):
        driver.load_budget_policy(wrong)


def _report(gpu_bytes: int, rss_bytes: int, mismatches: list | None = None) -> dict:
    return {
        "memory_after": {
            "gpu_peak_allocated_bytes": gpu_bytes,
            "process_peak_rss_bytes": rss_bytes,
        },
        "experiment": {"page_guard": {"forced_mismatches": mismatches or []}},
    }


def test_stage_gates_admit_a_clean_stage():
    driver = load_module("run_p2c_combined", "scripts/run_p2c_combined.py")
    policy = driver.load_policy(P2C_POLICY)
    documents = [{"doc_id": doc, "state": "TRUNCATED"} for doc in policy["documents"]]
    reports = [_report(9_500_000_000, 3_000_000_000) for _ in policy["documents"]]
    verdict = driver.evaluate_stage_gates(policy, documents, reports)
    assert verdict["next_budget_admitted"] is True
    assert verdict["failures"] == []
    assert verdict["resource_peaks"]["gpu_peak_allocated_gib"] == pytest.approx(8.848, abs=0.01)


def test_stage_gates_fail_on_every_dimension():
    driver = load_module("run_p2c_combined", "scripts/run_p2c_combined.py")
    policy = driver.load_policy(P2C_POLICY)
    documents = [
        {"doc_id": policy["documents"][0], "state": "TIMEOUT"},
        {"doc_id": policy["documents"][1], "state": "TRUNCATED"},
        {"doc_id": policy["documents"][2], "state": "TRUNCATED"},
        # documents[3] missing entirely (NOT_RUN)
    ]
    reports = [
        _report(11_000_000_000, 3_000_000_000),
        _report(9_000_000_000, 5_000_000_000, mismatches=[{"step": 42}]),
    ]
    verdict = driver.evaluate_stage_gates(policy, documents, reports)
    assert verdict["next_budget_admitted"] is False
    joined = " ".join(verdict["failures"])
    assert "not completed" in joined
    assert "supervision failures" in joined
    assert "GPU peak" in joined
    assert "host RSS peak" in joined
    assert "forced-marker mismatches" in joined
