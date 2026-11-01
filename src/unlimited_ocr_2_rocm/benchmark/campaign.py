"""Full-campaign scope classification and admission arithmetic.

The frozen L1/L1.1-B policies authorize exactly ONE predeclared 15-page
document each — they are NOT a blanket authorization for a 420-document
campaign. This module therefore classifies every document of the frozen
manifest into an execution identity BEFORE any GPU second is spent:

- ``PRODUCTION_ADMITTED``      pages <= 5, budget 8192, no policy needed.
- ``CAMPAIGN_POLICY_REQUIRED`` 6..15 pages, official-class budget 39321,
  executable only behind the reviewed FIFTH pinned campaign policy digest
  (``CAMPAIGN_POLICY_SHA256``, eval/experiments/mpdoc-full-campaign/policy.json)
  AND an explicit owner authorization recorded in the run config naming that
  digest. Either condition alone is refused.
- ``NATIVE_UNSUPPORTED_OVER_15P`` pages > 15: the native single-request path
  has no validated identity there (the official windowed protocol is an
  SGLang identity, not a native one). These documents are never executed and
  never silently dropped — they stay NOT_RUN/BLOCKED in the ledger.

Admission arithmetic reuses the frozen P2-A bounds: 300 prompt tokens per
page (upper bound), 4096 margin, 131072 context cap. Context admission is
necessary but never sufficient: budget sufficiency is a runtime outcome
(TRUNCATED accounting), and the page-count identity is the binding gate.
"""

from __future__ import annotations

from typing import Any

PRODUCTION_BUDGET = 8192
CAMPAIGN_BUDGET = 39321
PROMPT_UPPER_BOUND_TOKENS_PER_PAGE = 300
REQUIRED_MARGIN_TOKENS = 4096
CONTEXT_CAP_TOKENS = 131072

PRODUCTION_MAX_PAGES = 5
CAMPAIGN_MAX_PAGES = 15

#: Pinned digest of the frozen fifth campaign policy
#: (eval/experiments/mpdoc-full-campaign/policy.json). Must equal
#: ``inference.EXPECTED_CAMPAIGN_POLICY_SHA256`` (the budget/n-gram gate's
#: pin of the same file) — the equality is guarded by
#: tests/unit/test_mpdoc_full_campaign.py::
#: test_campaign_policy_identity_is_bound_across_gates. This module keeps its
#: own constant so the CPU-only plan/dryrun path never imports the torch-side
#: inference stack.
CAMPAIGN_POLICY_SHA256 = "47a494a985bf0058396c3dd762f83593217072a9708ecf3e08d6291fe2677edc"

CATEGORIES = ("PRODUCTION_ADMITTED", "CAMPAIGN_POLICY_REQUIRED", "NATIVE_UNSUPPORTED_OVER_15P")


def production_budget() -> int:
    return PRODUCTION_BUDGET


def classify_document(pages: int) -> dict[str, Any]:
    """Deterministic per-document execution identity; no I/O, no policy I/O."""
    if pages <= 0:
        raise ValueError(f"A document must have at least one page, got {pages}")
    if pages <= PRODUCTION_MAX_PAGES:
        category = "PRODUCTION_ADMITTED"
        budget = PRODUCTION_BUDGET
    elif pages <= CAMPAIGN_MAX_PAGES:
        category = "CAMPAIGN_POLICY_REQUIRED"
        budget = CAMPAIGN_BUDGET
    else:
        category = "NATIVE_UNSUPPORTED_OVER_15P"
        budget = CAMPAIGN_BUDGET
    prompt_bound = pages * PROMPT_UPPER_BOUND_TOKENS_PER_PAGE
    total = prompt_bound + budget + REQUIRED_MARGIN_TOKENS
    return {
        "category": category,
        "pages": pages,
        "budget_new_tokens": budget,
        "prompt_upper_bound_tokens": prompt_bound,
        "required_margin_tokens": REQUIRED_MARGIN_TOKENS,
        "context_cap_tokens": CONTEXT_CAP_TOKENS,
        "total_upper_bound_tokens": total,
        "admitted": total <= CONTEXT_CAP_TOKENS and category != "NATIVE_UNSUPPORTED_OVER_15P",
    }


def execution_plan(manifest_docs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Split the frozen manifest into the executable set and the explicit
    blocked set. Deterministic under any input ordering.

    Campaign-class documents enter the executable set only when the run
    config carries an authorization block naming the PINNED fifth campaign
    policy digest (``CAMPAIGN_POLICY_SHA256``); a foreign digest is refused
    with ``CAMPAIGN_POLICY_DIGEST_MISMATCH``. Unsupported documents NEVER do.
    Even with a matching authorization, execution stays fail-closed end to
    end: the worker refuses campaign-class documents without a frozen budget
    policy in the run config, and the pinned budget gate inside
    ``inference.validate_budget`` admits 39321 only for the same digest.
    """
    executable: list[str] = []
    blocked: dict[str, str] = {}
    authorization = config.get("authorization")
    authorized_digest = (
        authorization.get("campaign_policy_sha256") if isinstance(authorization, dict) else None
    )
    for doc in sorted(manifest_docs, key=lambda row: row["id"]):
        row = classify_document(len(doc["pages"]))
        if row["category"] == "PRODUCTION_ADMITTED":
            executable.append(doc["id"])
        elif row["category"] == "CAMPAIGN_POLICY_REQUIRED":
            if authorized_digest == CAMPAIGN_POLICY_SHA256:
                executable.append(doc["id"])
            elif authorized_digest:
                blocked[doc["id"]] = "CAMPAIGN_POLICY_DIGEST_MISMATCH"
            else:
                blocked[doc["id"]] = "CAMPAIGN_POLICY_REQUIRED_NO_AUTHORIZATION"
        else:
            blocked[doc["id"]] = "NATIVE_UNSUPPORTED_OVER_15P"
    return {"executable": executable, "blocked": blocked}
