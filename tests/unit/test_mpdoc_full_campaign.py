"""Full MPDocBench campaign runner: scope classification, admission arithmetic,
authorization gating and deterministic full-scope manifests.

CPU-only tests over synthetic manifests; the classification rules encoded here
are the ones the real 420-document dry-run must reproduce exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/run_mpdoc_full.py"
REPO_ROOT = Path(__file__).parents[2]

from unlimited_ocr_2_rocm.benchmark.campaign import (  # noqa: E402
    CAMPAIGN_BUDGET,
    CAMPAIGN_POLICY_SHA256,
    classify_document,
    execution_plan,
    production_budget,
)


def test_production_budget_is_the_frozen_default() -> None:
    assert production_budget() == 8192


def test_classification_boundaries() -> None:
    # 1..5 pages: production path, no policy, 8192 cap.
    for pages in (1, 3, 5):
        row = classify_document(pages)
        assert row["category"] == "PRODUCTION_ADMITTED"
        assert row["budget_new_tokens"] == 8192
        assert row["admitted"] is True
    # 6..15 pages: single-request native is validated only behind a campaign
    # policy at the official-class budget; L1.1-B validated exactly 15 pages.
    for pages in (6, 10, 15):
        row = classify_document(pages)
        assert row["category"] == "CAMPAIGN_POLICY_REQUIRED"
        assert row["budget_new_tokens"] == CAMPAIGN_BUDGET == 39321
    # >15 pages: native single-request has NO validated identity; the official
    # windowed protocol belongs to SGLang, not this runner. Never silently run.
    for pages in (16, 40, 200):
        row = classify_document(pages)
        assert row["category"] == "NATIVE_UNSUPPORTED_OVER_15P"
        assert row["admitted"] is False
    with pytest.raises(ValueError):
        classify_document(0)


def test_admission_arithmetic_matches_frozen_bounds() -> None:
    # 300 prompt tokens/page upper bound (P2A frozen admission rule),
    # 4096 margin, 131072 context cap.
    row = classify_document(15)
    assert row["prompt_upper_bound_tokens"] == 15 * 300
    assert row["total_upper_bound_tokens"] == 15 * 300 + 39321 + 4096
    assert row["admitted"] is True
    # Even a 292-page document would pass the context arithmetic — the binding
    # constraint is the page-count identity, not the context window. Make sure
    # the arithmetic never masquerades as a page-count approval.
    row = classify_document(292)
    assert row["total_upper_bound_tokens"] <= 131072
    assert row["category"] == "NATIVE_UNSUPPORTED_OVER_15P"


def test_execution_plan_refuses_campaign_docs_without_authorization() -> None:
    manifest_docs = [
        {"id": "prod_a", "pages": [{"path": f"p{i}.png"} for i in range(3)]},
        {"id": "camp_b", "pages": [{"path": f"p{i}.png"} for i in range(8)]},
        {"id": "huge_c", "pages": [{"path": f"p{i}.png"} for i in range(16)]},
    ]
    plan = execution_plan(manifest_docs, config={"schema": 1, "label": "FULL_SCOPE_CAMPAIGN"})
    assert plan["executable"] == ["prod_a"]
    assert plan["blocked"] == {
        "camp_b": "CAMPAIGN_POLICY_REQUIRED_NO_AUTHORIZATION",
        "huge_c": "NATIVE_UNSUPPORTED_OVER_15P",
    }
    # An authorization entry flips campaign-class docs into the executable set,
    # but NEVER the unsupported ones — and ONLY when it names the pinned
    # fifth campaign-policy digest, not an arbitrary string.
    authorized = {
        "schema": 1,
        "campaign_policy_sha256": CAMPAIGN_POLICY_SHA256,
        "authorized_by": "owner",
    }
    plan = execution_plan(manifest_docs, config={"schema": 1, "authorization": authorized})
    assert plan["executable"] == ["camp_b", "prod_a"]  # sorted by id
    assert plan["blocked"] == {"huge_c": "NATIVE_UNSUPPORTED_OVER_15P"}


def test_execution_plan_refuses_campaign_docs_under_a_foreign_digest() -> None:
    manifest_docs = [
        {"id": "prod_a", "pages": [{"path": f"p{i}.png"} for i in range(3)]},
        {"id": "camp_b", "pages": [{"path": f"p{i}.png"} for i in range(8)]},
    ]
    authorized = {
        "schema": 1,
        "campaign_policy_sha256": "d" * 64,
        "authorized_by": "owner",
    }
    plan = execution_plan(manifest_docs, config={"schema": 1, "authorization": authorized})
    assert plan["executable"] == ["prod_a"]
    assert plan["blocked"]["camp_b"] == "CAMPAIGN_POLICY_DIGEST_MISMATCH"


def test_execution_plan_is_deterministic() -> None:
    docs = [
        {"id": f"doc_{i:03d}", "pages": [{"path": f"p{j}.png"} for j in range((i % 20) + 1)]}
        for i in range(30)
    ]
    one = execution_plan(docs, config={})
    two = execution_plan(list(reversed(docs)), config={})
    assert one == two


def test_dryrun_cli_is_deterministic_and_excludes_nothing(tmp_path: Path) -> None:
    gt = _synthetic_gt(tmp_path, page_counts=(1, 5, 6, 15, 16))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"datasets": {"MPDocBench": {"revision": "r1"}}}), "utf-8")
    index = _synthetic_page_index(tmp_path, gt)
    out1 = tmp_path / "dryrun-1.json"
    out2 = tmp_path / "dryrun-2.json"
    for output in (out1, out2):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "dryrun",
                "--ground-truth",
                str(gt),
                "--dataset-root",
                str(tmp_path),
                "--lock",
                str(lock),
                "--page-index",
                str(index),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    one = json.loads(out1.read_text("utf-8"))
    two = json.loads(out2.read_text("utf-8"))
    assert one == two  # deterministic: same inputs -> byte-identical report
    assert one["expected_documents"] == 5
    assert one["zero_exclusion"] is True
    assert one["classified_documents"] == 5
    counts = one["category_counts"]
    assert counts["PRODUCTION_ADMITTED"] == 2
    assert counts["CAMPAIGN_POLICY_REQUIRED"] == 2
    assert counts["NATIVE_UNSUPPORTED_OVER_15P"] == 1
    # Every document appears exactly once with a category and a page hash verdict.
    assert len(one["documents"]) == 5
    assert all(doc["page_hashes"] == "VERIFIED" for doc in one["documents"])
    assert "full_scope_manifest_sha256" in one


def test_dryrun_flags_hash_mismatch_instead_of_skipping(tmp_path: Path) -> None:
    gt = _synthetic_gt(tmp_path, page_counts=(2,))
    index = _synthetic_page_index(tmp_path, gt)
    # Corrupt one page after the frozen index was written from the real bytes.
    victim = next((tmp_path / "images").rglob("*.png"))
    victim.write_bytes(b"tampered")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"datasets": {"MPDocBench": {"revision": "r1"}}}), "utf-8")
    out = tmp_path / "dryrun.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "dryrun",
            "--ground-truth",
            str(gt),
            "--dataset-root",
            str(tmp_path),
            "--lock",
            str(lock),
            "--page-index",
            str(index),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text("utf-8"))
    row = report["documents"][0]
    assert row["page_hashes"] == "MISMATCH"
    assert report["page_hash_failures"] == 1
    assert report["zero_exclusion"] is True  # counted and reported, never dropped


CAMPAIGN_POLICY = REPO_ROOT / "eval/experiments/mpdoc-full-campaign/policy.json"


def test_fifth_campaign_policy_authorizes_the_official_class_gates() -> None:
    """The FIFTH pinned digest admits budget 39321 + suppression 35/1024 for
    the MPDOC_FULL_CAMPAIGN experiment — and nothing else."""
    from unlimited_ocr_2_rocm.inference import validate_budget, validate_ngram

    assert CAMPAIGN_POLICY.is_file(), "the frozen fifth campaign policy must exist in the repo"
    # Exactly the official-class budget; nothing above, nothing below.
    validate_budget(39321, policy_path=CAMPAIGN_POLICY)
    with pytest.raises(ValueError):
        validate_budget(39322, policy_path=CAMPAIGN_POLICY)
    with pytest.raises(ValueError):
        validate_budget(16384, policy_path=CAMPAIGN_POLICY)
    # Official suppression identity, generated ids only (window 1024, not the
    # frozen model's coerced 100).
    validate_ngram(35, 1024, policy_path=CAMPAIGN_POLICY)
    with pytest.raises(ValueError):
        validate_ngram(35, 100, policy_path=CAMPAIGN_POLICY)
    # (0, 0) is the documented suppression-off production default and never
    # reaches the policy gate — covered by the n-gram gate's own tests.


def test_fifth_campaign_policy_refuses_a_forged_copy() -> None:
    from unlimited_ocr_2_rocm.inference import validate_budget

    forged = json.loads(CAMPAIGN_POLICY.read_text("utf-8"))
    forged["budgets_new_tokens"] = [39321, 65536]
    forged_file = CAMPAIGN_POLICY.parent / "forged.policy.json"
    forged_file.write_text(json.dumps(forged, indent=1), "utf-8")
    try:
        with pytest.raises(ValueError, match="not pinned by this gate"):
            validate_budget(65536, policy_path=forged_file)
        # Even the legitimate budget is refused through the forged file.
        with pytest.raises(ValueError, match="not pinned by this gate"):
            validate_budget(39321, policy_path=forged_file)
    finally:
        forged_file.unlink()


def test_campaign_policy_identity_is_bound_across_gates() -> None:
    """campaign.py's plan gate and inference.py's budget/n-gram gates must pin
    the SAME digest as the frozen policy file on disk — one identity, three
    enforcement points, zero drift."""
    from unlimited_ocr_2_rocm.config import file_sha256
    from unlimited_ocr_2_rocm.inference import (
        _AUTHORIZED_NGRAM_POLICIES,
        AUTHORIZED_BUDGET_POLICIES,
        EXPECTED_CAMPAIGN_POLICY_SHA256,
    )

    on_disk = file_sha256(CAMPAIGN_POLICY)
    assert CAMPAIGN_POLICY_SHA256 == on_disk
    assert EXPECTED_CAMPAIGN_POLICY_SHA256 == on_disk
    assert AUTHORIZED_BUDGET_POLICIES[on_disk]["experiment"] == "MPDOC_FULL_CAMPAIGN"
    assert AUTHORIZED_BUDGET_POLICIES[on_disk]["budgets"] == frozenset({39321})
    assert _AUTHORIZED_NGRAM_POLICIES[on_disk] == "MPDOC_FULL_CAMPAIGN"
    policy = json.loads(CAMPAIGN_POLICY.read_text("utf-8"))
    assert policy["experiment"] == "MPDOC_FULL_CAMPAIGN"
    assert policy["schema"] == 1
    assert policy["budgets_new_tokens"] == [39321]
    assert policy["ngram_sizes"] == [35] and policy["ngram_window"] == 1024
    # The policy binds the frozen full-scope manifest, not a document list.
    assert policy["documents"]["expected_count"] == 221
    assert policy["ledger"]["dataset_documents"] == 420


def _synthetic_gt(tmp_path: Path, page_counts: tuple[int, ...]) -> Path:
    docs = []
    for number, count in enumerate(page_counts):
        name = f"doc{number:02d}"
        pages = []
        for index in range(1, count + 1):
            rel = f"images/{name}/page_{index}.png"
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"{name}:{index}".encode())
            pages.append(rel)
        docs.append({"page_info": {"image_path": f"{name}.pdf", "images_list": pages}})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(docs), encoding="utf-8")
    return gt


def _synthetic_page_index(tmp_path: Path, gt: Path) -> Path:
    """Frozen page index over the synthetic bytes (D0-equivalent, tiny)."""
    from unlimited_ocr_2_rocm.config import file_sha256

    rows = json.loads(gt.read_text("utf-8"))
    pages = []
    for doc in rows:
        for rel in doc["page_info"]["images_list"]:
            pages.append({"path": rel, "sha256": file_sha256(tmp_path / rel)})
    index = tmp_path / "page-index.json"
    index.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return index
