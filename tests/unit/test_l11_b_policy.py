"""The L1.1-B policy must pass the frozen L1 admission gate and nothing else."""

from pathlib import Path

from unlimited_ocr_2_rocm.config import file_sha256
from unlimited_ocr_2_rocm.inference import EXPECTED_L11_B_POLICY_SHA256, validate_l1_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "eval/experiments/l11-fifteen-page-native-b/policy.json"
RECEIPT = REPO_ROOT / "eval/experiments/l11-fifteen-page-native-b/selection_receipt.json"


def test_l11_b_policy_admits_through_the_frozen_gate() -> None:
    assert file_sha256(POLICY) == EXPECTED_L11_B_POLICY_SHA256
    policy = validate_l1_policy(
        POLICY,
        ngram_size=35,
        ngram_window=1024,
        max_new_tokens=39321,
        expected_pages=15,
        repo_root=REPO_ROOT,
    )
    assert policy["experiment"] == "L11_FIFTEEN_PAGE_NATIVE_B"
    assert policy["documents"] == ["paper_eng_64b5e1f4"]


def test_l11_b_receipt_is_the_reviewed_packet_bytes() -> None:
    assert (
        file_sha256(RECEIPT) == "8d80ae7bb720df2908d34de301a8ac12fd7490f6d3bb27460fd475bfc428b9ec"
    )


def test_tampered_l11_b_policy_is_refused(tmp_path: Path) -> None:
    tampered = tmp_path / "policy.json"
    tampered.write_text(POLICY.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        validate_l1_policy(
            tampered,
            ngram_size=35,
            ngram_window=1024,
            max_new_tokens=39321,
            expected_pages=15,
            repo_root=REPO_ROOT,
        )
    except ValueError as error:
        assert "not the pinned frozen identity" in str(error)
    else:
        raise AssertionError("tampered policy must be refused")
