# L1 specification (DESIGN ONLY — not executed; requires independent review before any physical run)

**Status: PREPARED. One supervised physical run is admissible only after an
independent review of this spec and of the P2-C evidence. No 30/50/100-page
escalation is authorized or implied.**

**Basis: `P2C_FIFTEEN_PAGE_DECISION.md` (15-page Native stays BLOCKED at
12288/16384; suppression passes with margin), `P2C_COMBINED_RCA.md`,
`P2C_GUARD_COLLISION_SAFETY.md`, `docs/rca/NGRAM35_SOURCE_AUDIT.md`.**

## 1. Experimental identity

| Field | Frozen value |
| --- | --- |
| Experiment name | `L1_FIFTEEN_PAGE_NATIVE` |
| Max pages per request | **15** (native page guard expected_pages = prompt `<PAGE>` count) |
| Output budget | **39 321 new tokens** (official deployment class; single request) |
| Suppression | no_repeat_ngram_size = **35**, trailing window **1024**, generated ids only |
| Model | baidu/Unlimited-OCR-2 @ revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, 18/18 lock hashes |
| Torch stack | torch 2.12.0+rocm7.2 (wheel sha `099e265d…`), triton-rocm 3.7.0, torchvision 0.27.0+rocm7.2; hip 7.2.53211 / rocm 7.1 system userland untouched |
| Reference semantics | SGLang wheel `a3f3e2c9…` (= benchmark-lock) — identity reference only, NOT installed for the run |
| Dataset | MPDocBench local verified copy (D0 receipts); page bytes gated by the frozen per-page SHA index at worker start |
| Decoder / vision | official eager + DASH Triton (12 routers), vision SDPA MATH, BF16, greedy temperature 0 |
| Prompt / crop | official default document prompt; `crop_mode=False` (infer_multi) |
| Production defaults | UNCHANGED — budgets above 8192 and suppression remain opt-in behind pinned-digest policies |

## 2. Document selection (predeclared before any inference)

- One document with **exactly ≥ 15 pages** drawn from the local verified
  MPDocBench copy, chosen for newspaper-class density (the P2-C failure
  class), plus the four frozen P2-A documents are NOT re-run (their evidence
  is immutable).
- Selection frozen by doc id + per-page SHA-256 list committed BEFORE the
  run; the worker refuses any page whose hash misses the frozen index
  (existing `verify_page_gate`).
- Slice identity for the scorer published as a sanitized identity receipt
  only (`P0` publication rules; raw GT never committed).

## 3. Context and memory admission (pre-flight, per document)

- Context arithmetic: 15 pages × 300 upper-bound prompt tokens = 4 500;
  4 500 + 39 321 + 4 096 margin = **47 917 ≤ 131 072** ✓ (measured prompt
  is ~275/page ⇒ ~4 125 actual; upper bound governs admission).
- KV-cache estimate (DynamicCache, ~60 KiB/token, SWA ring disabled):
  (4 125 + 39 321) × 60 KiB ≈ **2.6 GiB**, on top of the vision-prefill
  envelope 8.55 GiB measured at every budget 8 192–16 384 ⇒ expected peak
  **≈ 11–12 GiB** of the 32 GiB UMA (+100 GiB GTT fallback, untouched
  co-tenancy ~1.5 GiB). Stage gate: refuse/stop above **16 GiB allocated**.
- The admission report (existing `admission_report` pattern) is persisted
  with the attempt; ADMISSION_REFUSED is a recorded outcome, never a skip.

## 4. Supervision, timeout, cleanup

- Isolated subprocess per attempt (P1.5 supervision): process-group
  SIGKILL on timeout, bounded 10 s grace reap, `TERMINATION_INCOMPLETE`
  classified as failure — never PASS.
- Timeout: 39 321 tokens at the measured 12.6–13.9 tok/s ≈ 2 900–3 120 s
  decode + ~15 s prefill + ~90 s model load ⇒ **7 200 s timeout** (≥ 2.3×
  margin), same as every prior stage.
- Failures, interrupts and partial attempts are preserved in the ledger
  (never deleted, never relabeled); atomic run identity = digest of
  manifest+config+environment.

## 5. Ledger and accounting

- Full-denominator ledger: all 420 MPDocBench documents accounted; the
  single selection executed, the other 419 NOT_RUN by name.
- Per attempt: finish_reason, output tokens, pages emitted vs expected,
  forced-marker consistency (replay detector over persisted token ids),
  ban-set statistics (first ban step, steps with ban, max/mean ban size),
  repeated-tail cycle scan (long AND short cycles), GPU peak
  allocated/reserved, host RSS, prefill/decode seconds, wall.
- Official scorer on the selection slice, strict (TRUNCATED ⇒ empty) and
  diagnostic (truncation-content) passes with coverage receipts; Overall
  stays null wherever a metric has no sample — never aggregate across
  differing populations.

## 6. Guard-collision risk at this budget class (new, live)

At 39 321 > max_page_tokens (20 000) the audited corner becomes reachable
for the first time: a page whose dense content runs ≥ 20 000 tokens without
a marker makes the guard force `<PAGE>`, and if that `<PAGE>` is in the
current 35-gram ban set the frozen guard yields an all-`-inf` row whose
argmax drops the forced marker (Phase-1 fixtures, 7/7). Requirements:

- The replay forced-marker detector (`forced_token_was_banned`,
  `forced_mismatches`) is a HARD gate of the run report; any occurrence
  fails the run's consistency gate and BLOCKS promotion, with no bypass.
- The P2-C observation lowers the prior: at 16 384 tokens with suppression,
  bans engaged ≤ 30 steps, max ban size 2, and no page ran past ~8 k tokens
  before its marker — the corner needs a >20 k-token page, which the
  15-page budget now admits but the observed dynamics do not produce.
  Still: detect, never assume.

## 7. Explicit differences from the official SGLang deployment

| Dimension | Official deployment | This L1 identity |
| --- | --- | --- |
| Serving | SGLang server, customized wheel `a3f3e2c9…` | frozen HF native loop (same model file), supervised subprocess |
| Long documents | **10-page windows** per request; KV-tail reuse across windows | **one 15-page request**, no windowing, no KV reuse, continuous DynamicCache |
| n-gram history | per ≤10-page window (window resets at boundaries) | continuous over the whole 39 321-token output (stricter: cross-page bans possible where official resets) |
| Page guard | expected_pages per window (≤10) | expected_pages = 15 in one guard instance |
| Budget | per-window budget class | single 39 321 budget |
| Attention | DASH sparse backend in SGLang | same DASH Triton kernels, eager decode |

These differences are the experiment's purpose: they isolate one-request
long-document behavior at the official budget class; they are NOT a claim
of official-parity throughput. No leaderboard claim may be derived from a
single-document run.

## 8. Execution authorization checklist (all required, in order)

1. Independent review of this spec + the P2-C evidence (reviewer record
   committed).
2. New frozen policy `eval/experiments/l1-fifteen-page/policy.json`
   (schema 1; budgets `[39321]`; pages 15; digest pinned in the gate — a
   THIRD authorized digest entry, reviewed like P2-C's).
3. Selection committed (doc id + page hashes) before any GPU minute.
4. Context/memory pre-flight admission report PASS.
5. One supervised physical run; gates above enforced; failures preserved.
6. Strict + diagnostic official scorer passes with coverage receipts.
7. Decision record: promotion of the 15-page Native ladder requires ALL
   gates PASS including page completion and zero forced-marker
   mismatches; otherwise BLOCKED stands and no escalation is automatic.
