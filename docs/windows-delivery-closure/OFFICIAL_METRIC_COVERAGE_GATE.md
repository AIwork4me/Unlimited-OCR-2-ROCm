# Official Eight-Metric Coverage Gate (Phase 4.6)

Acceptance rules for recording official benchmark results, derived from the
actual `scorer.METRICS` contract and enforced presentation logic. These rules
bind every future Windows campaign report; violating any of them invalidates
the report.

## Recording rules

| Situation | Required record |
| --- | --- |
| Valid measurement | the scorer's percentage for that metric only |
| Missing prediction (failed inference, empty export) | document counted in the failure catalog; metric values still come from the official scorer run over the full inventory — never hand-entered |
| Partial document (some pages not processed) | counts as its exported artifact (empty or strict-mode content); never dropped from the denominator |
| Truncated output (`finish_reason=length` / budget hit) | TRUNCATED state in the ledger; strict export = empty file unless the reviewed diagnostic mode explicitly includes truncations, and that mode never feeds the official accounting |
| Failed inference (FAIL/TIMEOUT/OOM/INTERRUPTED) | explicit empty prediction file + failure state; stays in denominator |
| Not-run documents | NOT_RUN in the ledger; campaign completeness field shows n/420 (or n/100) — a partial campaign is labeled partial, never aggregated into an official Overall |
| Non-applicable metric (task absent for a document/sample) | reflected in the official scorer's per-task sample coverage (`documents_without_samples`); absence is never silently treated as not-applicable for the aggregate |
| Overall availability | mean of all eight percentages; if any of the eight is null/NaN/inf/out-of-range, Overall is **null/BLOCKED** — never the average of seven, never zero |
| Official delta | Windows Overall minus paper Overall (84.59 / 77.06), full precision, only when a complete campaign's Overall exists |

## Hard gates (fail closed)

1. Prediction inventory must equal the full GT inventory before scoring.
2. GT bytes must match the frozen lock hash on the scoring host.
3. Scorer source and dependency lock must pass per-file/per-pin hash checks
   in a torch-free environment.
4. CDM tools must be verified working (real render) before FormulaCDM may be
   reported; otherwise FormulaCDM = BLOCKED and Overall = null.
5. Denominators 420/3135 and 100/18272 are immutable.
6. No metric value may be invented, interpolated, carried over from another
   campaign, or substituted with CER/synthetic results.
7. Comparability with the official protocol must be established and stated
   (native ≠ official SGLang protocol); until then any Overall is explicitly
   labeled with the runtime protocol that produced it.
