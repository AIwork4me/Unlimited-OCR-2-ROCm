"""Public benchmark report builder (mission workflow E).

Turns frozen evidence — the ledger run summary, the official-scorer output,
the locked official reference and the protocol comparability audit — into the
publishable ``public-results.json`` and the README scorecard rows.

Hard rules encoded here:
- Radeon scores are NEVER hand-entered: they come only from the scorer result
  object, whose identity hashes are part of the report.
- FULL-style labels are DERIVED from the ledger accounting; the builder
  refuses any configuration where the accounting does not support the scope
  (wrong denominator, unattempted documents, missing metrics).
- A null denominator metric forces Overall = null (official aggregation rule).
- Non-equivalent protocols are labeled and can never claim an official
  reproduction; their delta is reported as Reference Comparison only.
"""

from __future__ import annotations

from typing import Any

EIGHT_METRICS = (
    "TextEdit",
    "T-TextEdit",
    "TableTEDS",
    "T-TableTEDS",
    "FormulaCDM",
    "ReadOrderEdit",
    "FigureF1",
    "HeadingTEDS",
)

DENOMINATORS = {"MPDocBench": (420, 3135), "MarathonDocBench": (100, 18272)}

RESULT_STATES = ("PASS", "FAIL", "TIMEOUT", "OOM", "TRUNCATED", "NOT_RUN")

#: Metrics whose population size defines the official Overall denominator.
DENOMINATOR_METRICS = (
    "TextEdit",
    "TableTEDS",
    "FormulaCDM",
    "ReadOrderEdit",
    "FigureF1",
    "HeadingTEDS",
)


def eight_metrics_present(metrics: dict[str, Any]) -> bool:
    return all(name in metrics for name in EIGHT_METRICS)


def derive_result_status(run_summary: dict[str, Any], *, eight_metrics_complete: bool) -> str:
    """FULL label hierarchy from the mission contract, derived — never set.

    FULL_EVALUATED       every document attempted AND all 8 metrics present.
    FULL_WITH_FAILURES   every document attempted, failures exist, metrics complete.
    FULL_SCOPE_ATTEMPTED every document attempted, but metric coverage incomplete.
    PARTIAL              some documents never attempted.
    NOT_RUN              nothing attempted.
    """
    counts = run_summary["counts"]
    attempted = sum(counts.get(state, 0) for state in RESULT_STATES) - counts.get("NOT_RUN", 0)
    expected = run_summary["expected_documents"]
    if attempted == 0:
        return "NOT_RUN"
    if attempted < expected:
        return "PARTIAL"
    if not eight_metrics_complete:
        return "FULL_SCOPE_ATTEMPTED"
    if counts.get("PASS", 0) == expected:
        return "FULL_EVALUATED"
    return "FULL_WITH_FAILURES"


def build_public_report(
    *,
    dataset: str,
    run_summary: dict[str, Any] | None,
    scorer_result: dict[str, Any] | None,
    official_reference: dict[str, Any],
    comparability: dict[str, Any],
    run_id: str | None,
    git_sha: str,
    evidence_sha256: str | None,
) -> dict[str, Any]:
    expected_docs, expected_pages = DENOMINATORS[dataset]
    if run_summary is not None:
        if (
            run_summary.get("dataset") != dataset
            or run_summary.get("expected_documents") != expected_docs
            or run_summary.get("expected_pages") != expected_pages
        ):
            raise ValueError(
                "denominator mismatch: the ledger run does not cover the "
                f"frozen {dataset} scope ({expected_docs} docs / {expected_pages} pages)"
            )

    metrics: dict[str, Any] = {name: None for name in EIGHT_METRICS}
    overall = None
    if scorer_result is not None:
        presented = scorer_result.get("metrics", {})
        missing = [name for name in EIGHT_METRICS if name not in presented]
        if missing:
            raise ValueError(
                f"scorer result does not carry the official eight metrics; missing: {missing}"
            )
        metrics = {name: presented[name] for name in EIGHT_METRICS}
        # Official aggregation rule: a denominator metric without samples (null)
        # forces Overall to null — never silently averaged away.
        if any(metrics[name] is None for name in DENOMINATOR_METRICS):
            overall = None
        else:
            overall = scorer_result.get("overall")

    if run_summary is None:
        status = "NOT_RUN"
    else:
        # Metric coverage that counts toward a FULL label: every denominator
        # metric present AND non-null (null = unscored population by rule).
        scoreable = all(metrics[name] is not None for name in DENOMINATOR_METRICS)
        if scorer_result is None:
            scoreable = False
        status = derive_result_status(run_summary, eight_metrics_complete=scoreable)

    official_overall = official_reference.get("official_overall")
    comparability_status = comparability.get("status", "UNVERIFIED")
    if overall is None or comparability_status != "EQUIVALENT":
        delta = None
        absolute_delta = None
    else:
        delta = round(overall - official_overall, 2)
        absolute_delta = round(abs(overall - official_overall), 2)

    claim_limit = (
        "Official Benchmark Reproduction (EQUIVALENT protocol)"
        if comparability_status == "EQUIVALENT"
        and status in ("FULL_EVALUATED", "FULL_WITH_FAILURES")
        else "Reference Comparison / Non-equivalent Protocol"
        if status.startswith("FULL")
        else "NOT a benchmark result: scope incomplete or unattempted"
    )

    return {
        "schema": 1,
        "dataset": dataset,
        "dataset_denominator": {"documents": expected_docs, "pages": expected_pages},
        "run_id": run_id,
        "git_sha": git_sha,
        "evidence_sha256": evidence_sha256,
        "platform": "AMD Radeon 8060S (gfx1151)",
        "operating_system": "Ubuntu 24.04.4",
        "backend": "native-transformers",
        "counts": (run_summary or {}).get("counts"),
        "denominator": {"documents": expected_docs, "pages": expected_pages},
        "metrics": metrics,
        "overall": overall,
        "official_overall": official_overall,
        "delta": delta,
        "absolute_delta": absolute_delta,
        "comparability_status": comparability_status,
        "comparability_reasons": comparability.get("reasons", []),
        "official_reference_source": official_reference.get("source"),
        "result_status": status,
        "claim_limit": claim_limit,
    }
