# T-TableTEDS eight-metric coverage — separate one-document experiment

Date: 2026-10-30 (Asia/Shanghai). Branch `feat/t-tableteds-coverage`
(PR #19), stacked on the P1.5 runner hardening (PR #18). This experiment
closes the one metric the P1/P1.5 8-document subset could not exercise:
**T-TableTEDS (merged_table TEDS)**. It is a NEW, independent experiment
identity — nothing is retroactively added to the previous 8-document
experiment and no Overall value is invented anywhere.

## 1. Predeclared selection (frozen before inference)

`eval/results/Linux gfx1151 verified/mpdoc-ttableted/selection.json`,
SHA256 `0ce25dafb463e891…`, committed before any inference. Ground-truth
scan of the frozen authorized MPDocBench corpus found **45 of 420**
documents carrying genuine merged-table annotations (`merged_html` fields
on `table` category blocks). Selection criteria (structural only, no model
output consulted): smallest page count, a true cross-page truncated-table
pair, minimal non-table content, ≤5 pages under unchanged production
limits.

Selected: **`handwritten_chn_375d58b7`** — 2 pages (1.95/2.13 Mpx, both in
the frozen D0 page index), exactly 2 table blocks: anno 1 (page 1,
`fenduan=上`/upper) and anno 2 (page 2, `fenduan=下`/lower), each carrying
`merged_html` (891 chars) — one cross-page merged-table pair. No text
blocks. Not present in the P1 8-document selection.

## 2. Independent experiment identity

Ledger run `e438812a4b4e91f2fe7591aad291e77633b9caf0b1a330bd3b73911107cd73a6`
(full 420-doc frozen manifest; 1-doc selection; unchanged limits: 5 pages,
8192 tokens, 16 Mpx/page; P1.5 supervised runner). Run executed only after
the runner's fault accounting was validated (12/12 CPU failure-injection
tests plus the real GPU resume evidence in the P1.5 report).

## 3. Result

| Item | Value |
| --- | --- |
| Attempt | `handwritten_chn_375d58b7/000001` |
| State | **PASS** — finish `stop`, complete page structure, sealed prediction |
| Tokens | 430 (well under the unchanged 8192 budget; 43.7 s wall) |
| Official scorer selftest gate | PASS (8/8 known-answer checks incl. T-TableTEDS=50 fixture) |
| **merged_table samples** | **1** (previously 0 across the whole 8-doc subset) |
| **T-TableTEDS** | **74.99** — computed by the original official scorer |
| TableTEDS | 74.99 (same table pair, single-table view) |
| HeadingTEDS | 100.0 |
| TextEdit / T-TextEdit / ReadOrderEdit / FigureF1 / FormulaCDM | null — this document contains no text-block, merged-text, reading-order, figure or formula ground-truth samples |
| Overall | **null by rule** (five metrics have no samples on this document) |

The null metrics are correct behavior, not failures: coverage.json records
zero sample populations for those tasks on a pure-table document. The
T-TableTEDS **computation path** — GT `merged_html` → cross-page merge
matching → TEDS over matched merged-table pairs — is now proven live on
real data with the original scorer, closing the eight-metric coverage gap
identified in the PR #16 review (§2 of the review: known-answer 8/8 vs
real-data 7/8).

## 4. What this does NOT claim

- No retroactive change to the P1/P1.5 8-document experiment (its
  T-TableTEDS remains null; its Overall remains null).
- No full-corpus T-TableTEDS estimate: one document, one merged pair.
- No leaderboard comparison of any kind (REAL_DATA_INTEGRATION_ONLY).

Raw evidence: ledger run under `.b/mpdoc-runs/MPDocBench/e438812a…`,
scorer run under `.b/scorer-runs/ttableted` (untracked). Sanitized
receipts: `eval/results/Linux gfx1151 verified/mpdoc-ttableted/`.
