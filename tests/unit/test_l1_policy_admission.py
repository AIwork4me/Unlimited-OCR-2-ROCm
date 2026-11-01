"""L1 policy admission (Task C): the third pinned-digest gate, fail-closed.

Required matrix: correct digest accepted; incorrect digest rejected; wrong
page count rejected; wrong token budget rejected; wrong n-gram size/window
rejected; missing document identity rejected; missing page hashes rejected;
missing safety qualification rejected; missing resource admission rejected;
unapproved experiment name rejected.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.config import file_sha256
from unlimited_ocr_2_rocm.inference import (
    AUTHORIZED_BUDGET_POLICIES,
    EXPECTED_L1_POLICY_SHA256,
    validate_budget,
    validate_l1_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "eval/experiments/l1-fifteen-page/policy.json"
SELECTION_PATH = REPO_ROOT / "eval/experiments/l1-fifteen-page/selection.json"


def test_l1_policy_digest_is_pinned_in_the_budget_gate() -> None:
    """Regression for the independent release review P1: the merged digest-pinned
    validate_budget must admit the L1/L1.1-B policies at 39321 — otherwise the
    reviewed drivers physically refuse their own authorized policies."""
    from unlimited_ocr_2_rocm.inference import EXPECTED_L11_B_POLICY_SHA256

    assert validate_budget(39321, POLICY_PATH) is None
    l11 = REPO_ROOT / "eval/experiments/l11-fifteen-page-native-b/policy.json"
    assert validate_budget(39321, l11) is None
    assert AUTHORIZED_BUDGET_POLICIES[EXPECTED_L1_POLICY_SHA256]["budgets"] == frozenset({39321})
    assert AUTHORIZED_BUDGET_POLICIES[EXPECTED_L11_B_POLICY_SHA256]["budgets"] == frozenset({39321})
    # The budget gate still binds experiment identity and refuses cross-use:
    # the P2-A policy must NOT admit 39321, and the L1 policy must NOT admit
    # a budget outside its frozen list.
    p2a = REPO_ROOT / "eval/experiments/p2a-token-budget/policy.json"
    with pytest.raises(ValueError, match="not authorized"):
        validate_budget(39321, p2a)
    with pytest.raises(ValueError, match="not authorized"):
        validate_budget(16384, POLICY_PATH)


def _accept(policy_path: Path = POLICY_PATH) -> dict:
    return validate_l1_policy(
        policy_path,
        ngram_size=35,
        ngram_window=1024,
        max_new_tokens=39321,
        expected_pages=15,
        repo_root=REPO_ROOT,
    )


def _rejects(tampered: Path, digest: str, **overrides: object) -> None:
    kwargs: dict = {
        "ngram_size": 35,
        "ngram_window": 1024,
        "max_new_tokens": 39321,
        "expected_pages": 15,
        "repo_root": REPO_ROOT,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        validate_l1_policy(tampered, expected_digest=digest, **kwargs)


def test_correct_digest_accepted() -> None:
    policy = _accept()
    assert policy["experiment"] == "L1_FIFTEEN_PAGE_NATIVE"
    assert file_sha256(POLICY_PATH) == EXPECTED_L1_POLICY_SHA256
    assert file_sha256(SELECTION_PATH) == policy["selection_receipt_sha256"]


def test_incorrect_digest_rejected(tmp_path: Path) -> None:
    tampered = tmp_path / "policy.json"
    tampered.write_text('{"schema": 1, "experiment": "L1_FIFTEEN_PAGE_NATIVE"}', encoding="utf-8")
    with pytest.raises(ValueError, match="pinned frozen identity"):
        validate_l1_policy(
            tampered,
            ngram_size=35,
            ngram_window=1024,
            max_new_tokens=39321,
            expected_pages=15,
            repo_root=REPO_ROOT,
        )


def _tampered_copy(tmp_path: Path, mutate) -> tuple[Path, str]:
    policy = copy.deepcopy(json.loads(POLICY_PATH.read_text(encoding="utf-8")))
    mutate(policy)
    path = tmp_path / "tampered-policy.json"
    path.write_text(json.dumps(policy, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path, file_sha256(path)


def test_wrong_page_count_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pages must be"):
        validate_l1_policy(
            POLICY_PATH,
            ngram_size=35,
            ngram_window=1024,
            max_new_tokens=39321,
            expected_pages=14,
            repo_root=REPO_ROOT,
        )


def test_wrong_token_budget_rejected(tmp_path: Path) -> None:
    _rejects(
        *_tampered_copy(tmp_path, lambda p: p.update(budgets_new_tokens=[39320])),
        max_new_tokens=39320,
    )
    with pytest.raises(ValueError, match="not authorized"):
        validate_l1_policy(
            POLICY_PATH,
            ngram_size=35,
            ngram_window=1024,
            max_new_tokens=39320,
            expected_pages=15,
            repo_root=REPO_ROOT,
        )


def test_wrong_ngram_size_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="N-gram"):
        validate_l1_policy(
            POLICY_PATH,
            ngram_size=30,
            ngram_window=1024,
            max_new_tokens=39321,
            expected_pages=15,
            repo_root=REPO_ROOT,
        )


def test_wrong_ngram_window_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="N-gram"):
        validate_l1_policy(
            POLICY_PATH,
            ngram_size=35,
            ngram_window=100,
            max_new_tokens=39321,
            expected_pages=15,
            repo_root=REPO_ROOT,
        )


def test_missing_document_identity_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["documents"] = []

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_missing_page_hashes_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["page_hashes"] = policy["page_hashes"][:14]

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_missing_safety_qualification_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy.pop("guard_collision_safety")

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_safety_qualification_digest_mismatch_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["guard_collision_safety"]["qualification_sha256"] = "0" * 64

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_missing_resource_admission_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy.pop("resource_admission")

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_resource_hard_limit_change_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["resource_admission"]["gpu_allocated_hard_limit_gib"] = 24

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_unapproved_experiment_name_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["experiment"] = "P2C_COMBINED_NGRAM_BUDGET"

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_selection_receipt_disagreement_rejected(tmp_path: Path) -> None:
    def mutate_hashes(policy: dict) -> None:
        policy["page_hashes"] = ["0" * 64] * 15

    _rejects(*_tampered_copy(tmp_path, mutate_hashes))

    def mutate_doc(policy: dict) -> None:
        policy["documents"] = ["book_eng_600ba37c"]

    _rejects(*_tampered_copy(tmp_path, mutate_doc))


def test_missing_selection_receipt_rejected(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["selection_receipt_sha256"] = "1" * 64

    _rejects(*_tampered_copy(tmp_path, mutate))


def test_context_admission_arithmetic_enforced(tmp_path: Path) -> None:
    def mutate(policy: dict) -> None:
        policy["budgets_new_tokens"] = [131072]

    _rejects(*_tampered_copy(tmp_path, mutate))
