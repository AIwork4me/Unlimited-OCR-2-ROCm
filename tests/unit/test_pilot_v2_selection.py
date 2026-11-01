"""Pilot-v2 frozen selection binding: the predeclared 12-document selection is
pinned by digest and every document must be campaign-class in the frozen
classification with the policy binding recorded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SELECTION = REPO_ROOT / "eval/experiments/mpdoc-full-campaign/PILOT_V2_FROZEN_SELECTION.json"
CAMPAIGN_POLICY = REPO_ROOT / "eval/experiments/mpdoc-full-campaign/policy.json"

#: Frozen digest of PILOT_V2_FROZEN_SELECTION.json (pinned like every other
#: execution identity; changing the selection requires a new reviewed commit
#: AND a new owner authorization, never an in-place edit).
EXPECTED_SELECTION_SHA256 = "a9cf629e8ee2491a1c5c19fefea303e3cd6bb541fe0fa87ca94fbc7c0a8cb85d"


@pytest.mark.skipif(
    not SELECTION.is_file(), reason="frozen selection not yet committed to this tree"
)
def test_frozen_selection_digest_and_binding() -> None:
    from unlimited_ocr_2_rocm.config import file_sha256

    assert file_sha256(SELECTION) == EXPECTED_SELECTION_SHA256
    sel = json.loads(SELECTION.read_text("utf-8"))
    assert sel["schema"] == 1 and sel["label"] == "PILOT_V2_FROZEN_SELECTION"
    assert len(sel["documents"]) == 12
    # every selected document is campaign-class (6..15 pages)
    from unlimited_ocr_2_rocm.benchmark.campaign import classify_document

    for doc in sel["documents"]:
        row = classify_document(doc["pages"])
        assert row["category"] == "CAMPAIGN_POLICY_REQUIRED"
        assert doc["doc_id"] == row.get("doc_id", doc["doc_id"])  # identity echo
        assert doc["budget_new_tokens"] == row["budget_new_tokens"] == 39321
        assert doc["timeout_seconds"] == 7200
    # population + rule binding
    assert sel["population"]["count"] == 221
    frozen_manifest = (
        "frozen full-scope manifest "
        "f59337e42eb387b81baffe4bd37923e4ad32e95d65f90c45acb9c7e43446d5c0"
    )
    assert sel["population"]["source"] == frozen_manifest
    # policy binding matches the pinned fifth digest
    binding = sel["policy_binding"]
    from unlimited_ocr_2_rocm.inference import EXPECTED_CAMPAIGN_POLICY_SHA256

    assert binding["campaign_policy_sha256"] == EXPECTED_CAMPAIGN_POLICY_SHA256
    assert file_sha256(CAMPAIGN_POLICY) == EXPECTED_CAMPAIGN_POLICY_SHA256
    assert binding["retries"] == 0 and binding["watchdog_gib"] == 16
    assert binding["ngram"] == "35/1024 generated-ids-only"
    ids = [doc["doc_id"] for doc in sel["documents"]]
    assert len(set(ids)) == 12
    order = {d["doc_id"]: d["order"] for d in sel["documents"]}
    assert ids == sorted(ids, key=lambda i: order[i])
