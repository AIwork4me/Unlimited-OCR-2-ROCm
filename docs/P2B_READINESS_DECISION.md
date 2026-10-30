# P2-B readiness decision — is the reviewed 15-page Native ladder admissible?

Date: 2026-10-30. Decision input: the P2-A controlled budget experiments
(`docs/validation/P2A_OUTPUT_TOKEN_RCA.md`), the P1.5 ledger re-run, and the
reviewed ladder specification (`docs/LONG_DOCUMENT_READINESS.md` §6: L1 =
15 pages at the **unchanged 8192 budget**, pass requires finish `stop`,
15 `<PAGE>` markers, canonical structure; `finish length` on any run aborts
the ladder).

## Decision

**The 15-page Native ladder (L1 as specified) is NOT admissible now.**
Not because 15 input pages would not fit — but because the output-budget
and degeneration gates fail on measured ≤5-page evidence, and no correct
15-page OCR result exists to audit. No 30/50/100-page stages may start
(each earlier physical gate has not passed).

## The four required distinctions, on measured evidence

### 1. Input-page memory feasibility — feasible by arithmetic, unmeasured at 15

Measured (P2-A/P1.5): GPU peak allocated 8.55 GiB / reserved 8.78 GiB for
4–5 pages, vision-prefill dominated; +8,192 output tokens moved the peak by
0.00 GiB; host RSS ≤ 2.55 GiB; GTT free ≥ 81.6/100 GiB; prompt measured at
~275 tokens/page (1,375 for 5 pages). Extrapolated 15 pages: ~4.1k prompt
tokens and a prefill-activation peak that stays an order of magnitude under
the 32 GiB carve-out. **Likely fits — but this is arithmetic, not
measurement; and per the mission rule a resource-only 15-page diagnostic
would not count as a correct 15-page OCR result, so none was run.**

### 2. Output-budget feasibility — INADMISSIBLE at the specified L1 budget

- Two of the four frozen dense ≤5-page documents already need **8,245 and
  10,021 tokens** to terminate naturally; 8192 truncates them (P1: 4/8
  documents, P2-A: confirmed with per-document natural-stop values).
- A 15-page document at the same density needs roughly 3× the per-page
  output — ≈25–30k tokens — i.e. the official 39,321 class, not 8192 and
  not 16384 (P2-A shows 16384 still truncates the two loop-bound
  newspapers and adds nothing for the budget-bound two).
- L1's own abort condition (`finish length` on any run) would therefore
  trigger with near-certainty: the same corpus produces length-truncation
  at ONE-THIRD the page count. **The ladder as written would abort at L1;
  running it would only burn ~45 min/document to re-measure a known
  failure.**

### 3. Structural completion — page guard ≠ content, stop ≠ completeness

- The native page guard did not even reach page 2 of newspaper_chn at
  16,384 tokens (1/5 markers); guard-forced markers prove page structure,
  not per-page content.
- handwritten_chn_1ae487be terminates with a clean 5/5-page stop and
  TextEdit 98.96, yet HeadingTEDS is only 29.2 — a native stop proves
  termination, never semantic completeness. A 15-page `stop` result must
  still pass full per-page contract audit and scoring before any quality
  claim.

### 4. Actual document quality — no 15-page correct-OCR result exists

No 15-page document has been produced under the reviewed contract. Nothing
can be said about 15-page quality, and no resource-only run may be
substituted for it.

## The degeneration blocker (independent of budget)

Two of four dense documents degenerate into exact repetition loops
(newspaper_chn: `，德不治` from char 1,157, 86.7→93.1 % of output as the
budget grows; newspaper_eng: `Member IREPS - 703-894…` ×552). Budget raises
lengthen the loops; page coverage is frozen. The official deployment
counters this with `no_repeat_ngram_size=35` plus 10-page windowing with
KV-tail reuse; our native parity path deliberately runs with the n-gram
processor disabled. **Any one-request 15-page run without repetition
suppression carries a high probability of the same failure mode.** The
n-gram enablement and the windowed-deployment port are already scoped in
`LONG_DOCUMENT_READINESS.md` as separate, reviewed identity changes — they
remain unported.

## Wall-time blocker (benchmark practicality)

Decode throughput decays with context (14.9 → 10.6 tok/s from 8k → 16k
tokens). A 25–30k-token 15-page document projects ≈ 35–50 min per request
at the official budget. Supervision exists (7200 s budget per attempt,
SIGKILL process-group cleanup, ledger TIMEOUT accounting), but corpus-scale
benchmarking at that speed is impractical without the official windowed
semantics.

## What WOULD make a 15-page ladder admissible (ordered gates)

1. Port and review the official n-gram repetition suppression (35) as its
   own identity change; re-prove the four P2-A documents (the two loop
   documents must terminate or be explicitly accounted as failures).
2. Re-specify L1's budget to the reviewed official 39,321 class (the
   ladder's own L3/L4 already anticipate this), keeping every other L1
   pass criterion (stop, 15 markers, per-page canonical structure, ≥12
   DASH kernels, memory/wall recording).
3. Run ONE real 15-page document end-to-end under the P1.5 supervised
   ledger with strict accounting; audit per-page structure AND official
   scorer results before any 30-page stage.
4. Only then consider 30/50/100 pages — each stage still gated on the
   previous one passing (unchanged rule).

## Parallel-track note (Baidu SGLang ROCm)

Investigation of the original Baidu customized SGLang ROCm implementation
may proceed in parallel as analysis, but the serving stack is not
substituted with any unverified generic backend; the native HF-path
evidence above remains the only measured basis for this decision.

## Status summary

| Distinction | Verdict |
| --- | --- |
| Input-page memory feasibility | Likely feasible (arithmetic; unmeasured; a diagnostic would not count as a result) |
| Output-budget feasibility @8192 (L1 as specified) | **INADMISSIBLE — measured counter-evidence** |
| Structural completion | Guard/stop prove structure/termination, not content |
| Actual 15-page document quality | Unmeasured — no correct 15-page OCR result exists |
| 30/50/100 pages | Not run; every prior gate unpassed |
