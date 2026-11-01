"""Public benchmark report builder: honest label derivation, no hand-entered
Radeon scores, strict denominator and metric coverage checks.

The builder turns (ledger summary, official-scorer output, locked official
reference, comparability audit) into public-results.json + Markdown. Every
FULL-style label is DERIVED from the ledger accounting; a mismatch between
the claimed scope and the accounting must be impossible to publish.
"""

from __future__ import annotations

import pytest

from unlimited_ocr_2_rocm.benchmark.public_report import (
    build_public_report,
    derive_result_status,
    eight_metrics_present,
)


def _ledger_summary(
    passed=420, failed=0, truncated=0, not_run=0, expected_docs=420, expected_pages=3135
):
    counts = {
        "PASS": passed,
        "FAIL": failed,
        "TIMEOUT": 0,
        "OOM": 0,
        "TRUNCATED": truncated,
        "NOT_RUN": not_run,
    }
    return {
        "dataset": "MPDocBench",
        "status": "COMPLETE" if counts["PASS"] == expected_docs else "PARTIAL",
        "expected_documents": expected_docs,
        "expected_pages": expected_pages,
        "counts": counts,
    }


def _scorer_result(overall=84.1, **overrides):
    metrics = {
        "TextEdit": 99.0,
        "T-TextEdit": 99.5,
        "TableTEDS": 87.0,
        "T-TableTEDS": 75.0,
        "FormulaCDM": 99.0,
        "ReadOrderEdit": 100.0,
        "FigureF1": 100.0,
        "HeadingTEDS": 99.0,
    }
    metrics.update(overrides.pop("metrics", {}))
    result = {"metrics": metrics, "overall": overall, "scorer_revision": "scorer-qwen@pristine"}
    result.update(overrides)
    return result


def _reference():
    return {
        "official_overall": 84.6,
        "source": "Baidu Unlimited-OCR upstream README (locked, auditable)",
        "locked_by": "owner",
    }


def _audit(status="EQUIVALENT", reasons=()):
    return {"status": status, "reasons": list(reasons)}


def test_full_evaluated_report_derives_labels_and_delta() -> None:
    report = build_public_report(
        dataset="MPDocBench",
        run_summary=_ledger_summary(),
        scorer_result=_scorer_result(),
        official_reference=_reference(),
        comparability=_audit(),
        run_id="r" * 64,
        git_sha="a" * 40,
        evidence_sha256="e" * 64,
    )
    assert report["result_status"] == "FULL_EVALUATED"
    assert report["comparability_status"] == "EQUIVALENT"
    assert report["overall"] == 84.1
    assert report["official_overall"] == 84.6
    assert report["delta"] == pytest.approx(-0.5)
    assert report["absolute_delta"] == pytest.approx(0.5)
    assert report["denominator"]["documents"] == 420
    assert report["denominator"]["pages"] == 3135


def test_label_is_full_with_failures_when_attempted_but_failed() -> None:
    summary = _ledger_summary(passed=418, failed=2)
    assert derive_result_status(summary, eight_metrics_complete=True) == "FULL_WITH_FAILURES"


def test_attempted_but_unscored_documents_block_full_label() -> None:
    summary = _ledger_summary(passed=418, failed=2, not_run=0)
    # All documents attempted, but the scorer population must match: a
    # missing metric makes the run NOT full-evaluated.
    assert derive_result_status(summary, eight_metrics_complete=False) == "FULL_SCOPE_ATTEMPTED"
    summary_partial = _ledger_summary(passed=400, not_run=20)
    assert derive_result_status(summary_partial, eight_metrics_complete=True) == "PARTIAL"


def test_wrong_denominator_is_refused() -> None:
    with pytest.raises(ValueError, match="denominator"):
        build_public_report(
            dataset="MPDocBench",
            run_summary=_ledger_summary(expected_docs=419),
            scorer_result=_scorer_result(),
            official_reference=_reference(),
            comparability=_audit(),
            run_id="r" * 64,
            git_sha="a" * 40,
            evidence_sha256="e" * 64,
        )


def test_missing_metric_is_refused_from_full() -> None:
    scorer = _scorer_result(metrics={"TableTEDS": None})
    report = build_public_report(
        dataset="MPDocBench",
        run_summary=_ledger_summary(),
        scorer_result=scorer,
        official_reference=_reference(),
        comparability=_audit(),
        run_id="r" * 64,
        git_sha="a" * 40,
        evidence_sha256="e" * 64,
    )
    assert report["result_status"] == "FULL_SCOPE_ATTEMPTED"
    # Denominator metric missing -> Overall must be null by the official rule.
    assert report["overall"] is None
    assert report["delta"] is None


def test_non_equivalent_protocol_is_labeled_and_delta_not_official() -> None:
    report = build_public_report(
        dataset="MPDocBench",
        run_summary=_ledger_summary(),
        scorer_result=_scorer_result(),
        official_reference=_reference(),
        comparability=_audit(status="NON_EQUIVALENT", reasons=["budget differs from official"]),
        run_id="r" * 64,
        git_sha="a" * 40,
        evidence_sha256="e" * 64,
    )
    assert report["comparability_status"] == "NON_EQUIVALENT"
    assert "Reference Comparison / Non-equivalent Protocol" in report["claim_limit"]


def test_no_completed_run_reports_honest_empty_state() -> None:
    report = build_public_report(
        dataset="MPDocBench",
        run_summary=_ledger_summary(passed=0, not_run=420),
        scorer_result=None,
        official_reference=_reference(),
        comparability=_audit(status="UNVERIFIED"),
        run_id=None,
        git_sha="a" * 40,
        evidence_sha256=None,
    )
    assert report["result_status"] == "NOT_RUN"
    assert report["overall"] is None
    assert report["delta"] is None


def test_eight_metrics_present_rejects_subset() -> None:
    metrics = {
        m: 1.0
        for m in (
            "TextEdit",
            "T-TextEdit",
            "TableTEDS",
            "T-TableTEDS",
            "FormulaCDM",
            "ReadOrderEdit",
            "FigureF1",
            "HeadingTEDS",
        )
    }
    assert eight_metrics_present(metrics) is True
    del metrics["FigureF1"]
    assert eight_metrics_present(metrics) is False
