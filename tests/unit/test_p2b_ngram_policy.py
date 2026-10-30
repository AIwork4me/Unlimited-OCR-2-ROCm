"""P2-B opt-in n-gram policy gates and replay diagnostics (CPU only)."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.inference import validate_ngram

POLICY_PATH = Path(__file__).parents[2] / "eval/experiments/p2b-ngram-suppression/policy.json"


def load_module(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suppression_off_needs_no_policy():
    assert validate_ngram(0, 0, None) is None


def test_suppression_requires_policy_file(tmp_path):
    with pytest.raises(ValueError, match="frozen experimental"):
        validate_ngram(35, 1024, None)


def test_frozen_policy_authorizes_official_configuration():
    assert validate_ngram(35, 1024, POLICY_PATH) is None


def test_forged_policy_is_rejected_by_pinned_digest(tmp_path):
    forged = tmp_path / "forged-policy.json"
    payload = json.loads(POLICY_PATH.read_text("utf-8"))
    payload["ngram_sizes"] = [3, 35]
    forged.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        validate_ngram(35, 1024, forged)


def test_unlisted_ngram_size_is_refused():
    with pytest.raises(ValueError, match="not authorized"):
        validate_ngram(36, 1024, POLICY_PATH)


def test_wrong_window_is_refused_including_the_model_default_trap():
    with pytest.raises(ValueError, match="not authorized"):
        validate_ngram(35, 100, POLICY_PATH)


def test_nonpositive_window_is_refused_before_the_model_coerces_it():
    with pytest.raises(ValueError, match="explicit positive"):
        validate_ngram(35, 0, POLICY_PATH)
    with pytest.raises(ValueError, match="explicit positive"):
        validate_ngram(35, -1, POLICY_PATH)


def test_malformed_or_missing_policy_is_refused(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", "utf-8")
    with pytest.raises(ValueError, match="Unreadable"):
        validate_ngram(35, 1024, broken)
    with pytest.raises(ValueError, match="Unreadable"):
        validate_ngram(35, 1024, tmp_path / "absent.json")


def test_driver_policy_binding_and_admission(tmp_path):
    driver = load_module("run_p2b_ngram", "scripts/run_p2b_ngram.py")
    policy = driver.load_policy(POLICY_PATH)
    assert policy["experiment"] == "P2B_NGRAM_SUPPRESSION_RCA"
    admitted = driver.admission_report(policy, pages=5, budget=8192)
    assert admitted["admitted"] is True
    assert admitted["total_upper_bound_tokens"] == 5 * 300 + 8192 + 4096
    tight = dict(policy["admission"])
    tight["context_cap_tokens"] = 12000
    refused = driver.admission_report({"admission": tight}, pages=5, budget=8192)
    assert refused["admitted"] is False
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": 1, "experiment": "SOMETHING_ELSE"}), "utf-8")
    with pytest.raises(ValueError, match="Not the frozen P2-B"):
        driver.load_policy(wrong)


def test_driver_prepare_validates_the_user_ngram_size_not_the_policys(tmp_path):
    """Independent-review finding 1: prepare must gate the --ngram-size value."""
    driver = load_module("run_p2b_ngram", "scripts/run_p2b_ngram.py")
    args = argparse.Namespace(
        policy=POLICY_PATH,
        ngram_size=36,
        budget=8192,
        ground_truth=tmp_path / "gt.json",
        dataset_root=tmp_path,
        lock=tmp_path / "lock.json",
        output=tmp_path / "out",
    )
    # Refused at the gate before any dataset file is touched.
    with pytest.raises(ValueError, match="not authorized"):
        driver.prepare_command(args)


def test_replay_ban_algorithm_matches_the_frozen_model_semantics():
    analyzer = load_module("analyze_ngram_evidence", "scripts/analyze_ngram_evidence.py")
    # prefix (2,) recurs at index 1 only -> successor 3 is banned.
    assert analyzer.banned_tokens([1, 2, 3, 1, 2], 2, 10) == {3}
    # trailing window drops the older occurrence at index 0/1.
    assert analyzer.banned_tokens([1, 2, 1, 2, 1, 2, 1, 2], 2, 3) == {1}


def test_replay_detects_forced_marker_corner():
    analyzer = load_module("analyze_ngram_evidence", "scripts/analyze_ngram_evidence.py")
    # Per-page cap 3 forces <PAGE> at step 3; emitting 7 instead is the
    # audited native corner (all--inf row, arbitrary argmax tie-break).
    report = analyzer.replay(
        [5, 5, 5, 7, 100, 1],
        ngram_size=35,
        window=1024,
        expected_pages=1,
        page_token_id=100,
        eos_id=1,
        max_page_tokens=3,
    )
    assert report["guard"]["forced_steps"] >= 1
    assert report["guard"]["forced_mismatches"], "the corner must be flagged"
    assert report["verdict"]["forced_marker_corner_detected"] is True


def test_replay_flags_marker_overshoot_and_clean_force():
    analyzer = load_module("analyze_ngram_evidence", "scripts/analyze_ngram_evidence.py")
    report = analyzer.replay(
        [100, 100, 1],
        ngram_size=35,
        window=1024,
        expected_pages=1,
        page_token_id=100,
        eos_id=1,
        max_page_tokens=20000,
    )
    assert report["guard"]["marker_overshoot"] == 1
    # the overflow EOS force at step 2 must match the emitted EOS
    assert report["guard"]["forced_mismatches"] == []


def test_replay_repeated_tail_cycle():
    analyzer = load_module("analyze_ngram_evidence", "scripts/analyze_ngram_evidence.py")
    tail = analyzer.repeated_tail([9, 7, 8, 7, 8, 7, 8])
    assert tail["cycle_len"] == 2
    assert tail["count"] == 3
    assert analyzer.repeated_tail([1, 2, 3, 4, 5]) == {}
