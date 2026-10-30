# P2-B n-gram suppression RCA — physical evidence on the four frozen documents

Date: 2026-10-30. Ledger run `6e1ae22ef79370b6ad75c965bacc476e649c669d97228fabc80e69257d90471b`,
driven by `scripts/run_p2b_ngram.py` under the frozen policy
(`eval/experiments/p2b-ngram-suppression/policy.json`, SHA-256
`0095b0c47da9a693d6bef70305a4385b986ea79aac129d628a500dac0ef8e1f5`, pinned in
`inference.EXPECTED_NGRAM_POLICY_SHA256`). Everything except the suppression
itself is byte-frozen against the P2-A/P1.5 controls: same model revision
(9cf077c, loader-hash-verified), same pages (frozen index gate), same prompt,
crop, vision MATH, DASH decoder, native page guard, greedy sampling, 8192
budget, 7200 s supervision. Controls were NOT re-run.

## Headline results

| Document | ngram=0 control @8192 | ngram=35 experiment | Repetition outcome |
|---|---|---|---|
| newspaper_chn_eb87830d | TRUNCATED, 1/5 pages, `，德不治` loop = 86.7%+ of output | TRUNCATED, **2/5 pages** | loop broken at the exact first-ban step; no cycle remains up to length 1024 |
| newspaper_eng_ed12985c | TRUNCATED, 2/4 pages, `Member IREPS…` ×552 | TRUNCATED, **3/4 pages** | loop gone; only a nascent 9-token cycle in the final 36 tokens |
| paper_chn_3567e55d | TRUNCATED, 5/5 pages (stops at 8,245 given 12288) | TRUNCATED, 5/5 pages | 1 ban in 8192 steps |
| handwritten_chn_1ae487be | TRUNCATED @8192 (stops at 10,021 given 12288) | TRUNCATED, 4/5 pages | 2 bans in 8192 steps |

All four attempts: `finish_reason=length` at 8192 tokens; walls 556–572 s;
decode 14.8–15.1 tok/s; prefill 4.1–4.7 s; GPU peak allocated 8.55 GiB /
reserved 8.77 GiB; host peak RSS ≤ 2.7 GiB — **the identical envelope to the
ngram=0 controls: suppression adds no measurable resource cost.**

## Causal attribution (single-variable proof)

The experiment outputs are **byte-identical to the ngram=0 controls until
exactly the first replayed ban step**, then diverge — for every document:

| Document | First ban step (replay) | First diverging char vs control @8192 |
|---|---|---|
| newspaper_chn | 711 | 1192 (the `德不治` run) |
| newspaper_eng | 6324 | 24653 |
| handwritten_chn | 5421 | 11488 |
| paper_chn | 6434 | 14024 |

At newspaper_chn's divergence point the control continues the loop while the
experiment emits a new `<BLOCK>` with different content — the ban demonstrably
broke the loop at the replayed step. Replay statistics (post-hoc over persisted
`token_ids`, the review-approved deterministic replay): bans engaged in 11 /
15 / 2 / 1 steps of 8192 (max ban size 2); **forced-marker mismatches: 0 in
all four documents** — the audited native §7 corner did not occur and every
guard-forced step was honored.

## Official scorer comparison (same 4-doc GT slice, diagnostic exports)

Per-metric document populations verified identical between the two passes
(coverage receipts committed); **Overall is null by rule** (T-TableTEDS has no
sample in this subset; metrics from different subsets are never aggregated).

| Metric | ngram=0 @8192 | ngram=35 @8192 | Δ |
|---|---|---|---|
| TextEdit | 53.94 | **59.85** | +5.91 |
| T-TextEdit | 45.30 | **50.92** | +5.62 |
| TableTEDS | 46.97 | 46.97 | 0 |
| T-TableTEDS | null | null | — (no merged tables) |
| FormulaCDM | 90.64 | **91.35** | +0.71 |
| ReadOrderEdit | 51.38 | **59.17** | +7.79 |
| FigureF1 | 58.33 | **67.60** | +9.27 |
| HeadingTEDS | 29.07 | **31.65** | +2.58 |

Every non-null metric improved or held; no quality regression at this budget.
These remain truncation-inclusive diagnostics on a four-document subset — NOT
full MPDocBench accuracy and NOT leaderboard parity.

## What the RCA establishes

1. **The repetition root cause is real and suppressible**: the deterministic
   loops are the only divergence source; the official processor's semantics
   (output-only scope, window 1024, prefix-ban) break them at the exact step
   the replay predicts.
2. **Suppression is not sufficient for termination**: all four documents still
   exhaust 8192. The residual failure is (a) short sub-threshold bursts —
   e.g. a 33-token `共享` run, just under the 35-gram size — and (b) dense
   documents whose legitimate content exceeds the budget (handwritten needs
   10,021; a 15-page document projects ≈25–30k). **The binding constraint is
   the output-budget class, exactly as the P2-B decision recorded.**
3. **Side effects on legitimate repeated content are minimal and bounded**:
   content is untouched until the first genuine 35-gram repetition; then at
   most a handful of single-token bans per document. The official scorer
   confirms net quality improvement, including on formula- and table-bearing
   documents (FormulaCDM 90.64→91.35, TableTEDS unchanged).

## P5 consequences (15-page decision)

Per `docs/P2B_READINESS_DECISION.md`'s own gate list:

- Gate 1 (port and review the official n-gram suppression; re-prove the four
  P2-A documents): **DONE and physically evidenced** — the two loop documents
  no longer loop (they remain budget-truncated, which the gate's "explicitly
  account as failures" clause covers: both are TRUNCATED, never relabeled).
- Gates 2–5 (re-specified 39,321-class budget, one real 15-page run, stage
  gating, review): **unmet** — this experiment changed no budget.

**The 15-page Native ladder remains BLOCKED.** What is now admissible to
PREPARE: a reviewed L1 re-spec at the official budget class with suppression
enabled, as one supervised identity, before any 15-page request.

## Provenance and reproduction

- Suppression semantics: `docs/rca/NGRAM35_SOURCE_AUDIT.md` (identity chain:
  official client `infer.py@48ff432`, customized SGLang wheel hash-matched to
  `benchmark-lock.json`, frozen model loop) and its independent review record.
- Evidence: `eval/results/Linux gfx1151 verified/p2b-ngram/` (run receipt, GT
  slice identity receipt, per-document replays, both scorer passes with
  coverage, `p2b-evidence.json`). The raw GT annotation slice itself was
  removed from the published tree by the P0 publication remediation and is
  preserved outside Git — see `docs/validation/P0_PUBLICATION_REMEDIATION.md`;
  its identity stays verifiable via `p2b-gt-slice-receipt.json`
  (slice_sha256 5f2211f8…). Raw model bytes and token ids stay in the
  untracked ledger run (`.b/p2b-runs/…`); nothing in the immutable control
  evidence was rewritten.
- Scorer tree rebuilt 85/85 hash-verified from the P1 closure copies before
  both passes; both official scorer jobs remain green on all PRs.
