# P2-A output-token budget RCA — controlled experiments on the four frozen length-hit documents

Date: 2026-10-30 (Asia/Shanghai). Branch `feat/p2a-token-budget` (PR #20).
Experiment identity: frozen policy
`eval/experiments/p2a-token-budget/policy.json` (SHA256 recorded in every
run/receipt), executed through the P1.5 supervised ledger runner.
**Production default caps are unchanged** (5 pages / 8192 tokens / 16 Mpx —
`run_native` refuses larger budgets unless the frozen policy is presented).
Label: EXPERIMENTAL_BUDGET_ONLY / P2A_RCA.

## 1. First-validation checklist (all verified before any stage ran)

| Gate | Evidence |
| --- | --- |
| Token/context cap source paths | production cap: `inference.py::validate_budget` (8192 unless policy); upstream `modeling_unlimitedocr.py`: `infer(max_length=131072, max_new_tokens=None)`; `_uocr_generate` loop `range(max_new_tokens)`, position bound `pos+1 >= max_length` → `length`; per-page guard cap `max_page_tokens=20000` unchanged |
| Resource admission | pre-flight per attempt: pages×300 + budget + 4,096 ≤ 131,072 — worst case here 5×300+16384+4096 = 21,884 ≪ 131,072; measured prompt 1,375 tokens (5 pages ≈ 275/page, matching the earlier arithmetic); recorded in every attempt's `experiment` metadata |
| External timeout/worker supervision | isolated subprocess per attempt, 7200 s budget, process-group SIGKILL on expiry (P1.5 machinery); ledger seals every outcome |
| GPU/host memory headroom | co-tenant baseline ~1.5 GiB; peak allocated 8.55 GiB / reserved 8.78 GiB at every budget (below); host RSS ≤ 2.55 GiB, system RAM 96 GiB, GTT free ≥ 81.6/100 GiB |
| Exact finish/page metadata | native `finish_reason`, `token_ids` count, `expected_pages` vs `num_pages_out` vs raw `<PAGE>` count recorded and cross-checked per attempt (`completion_metadata` invariant) |

The frozen four documents (images byte-identical — page-index SHA gate in
every worker; model revision 9cf077c; wheel/env unchanged; official prompt;
`crop_mode=False`; eager+DASH decoder; vision MATH; page guard on;
`max_pages` NOT changed) were run at 8192 (baseline = the P1.5 ledger run
`bf55b509…`, same stack), then 12288 and 16384 in fresh per-budget ledger
runs `3031002c…` / `f7ad7a3a…`.

## 2. Cross-budget results (one request per document, greedy, native path)

| Document | Budget | State | finish | tokens | pages out | gen s | peak alloc | peak resv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handwritten_chn_1ae487be | 8192 | TRUNCATED | length | 8,192 | 4/5 | 552 | 8.55 GiB | 8.78 GiB |
| | 12288 | **PASS** | **stop** | **10,021** | 5/5 | 731 | 8.55 GiB | 8.78 GiB |
| | 16384 | **PASS** | stop | 10,021 | 5/5 | 740 | 8.55 GiB | 8.78 GiB |
| paper_chn_3567e55d | 8192 | TRUNCATED | length | 8,192 | 5/5† | 601 | 8.55 GiB | 8.78 GiB |
| | 12288 | **PASS** | **stop** | **8,245** | 5/5 | 625 | 8.55 GiB | 8.78 GiB |
| | 16384 | **PASS** | stop | 8,245 | 5/5 | 640 | 8.55 GiB | 8.78 GiB |
| newspaper_chn_eb87830d | 8192 | TRUNCATED | length | 8,192 | 1/5 | 547 | 8.55 GiB | 8.77 GiB |
| | 12288 | TRUNCATED | length | 12,288 | 1/5 | 976 | 8.55 GiB | 8.77 GiB |
| | 16384 | TRUNCATED | length | 16,384 | 1/5 | 1,549 | 8.55 GiB | 8.77 GiB |
| newspaper_eng_ed12985c | 8192 | TRUNCATED | length | 8,192 | 2/4 | 559 | 8.54 GiB | 8.76 GiB |
| | 12288 | TRUNCATED | length | 12,288 | 2/4 | 1,034 | 8.54 GiB | 8.76 GiB |
| | 16384 | TRUNCATED | length | 16,384 | 2/4 | 1,547 | 8.54 GiB | 8.76 GiB |

† all five page markers emitted but the last block was cut mid-stream at
8192 (raw ends `[859, 921` in P1; the ledger keeps the exact bytes).

**Determinism note:** once a document completes, extra budget changes
nothing — handwritten and paper_chn raw outputs at 16384 are
**byte-identical** to their 12288 outputs (`0656c23d…`, `bce0a994…`).
Run-to-run noise floor (isolated digit flips, §P1.5 report) applies below
that scale.

## 3. Root cause split — budget-bound vs degeneration-bound

- **handwritten, paper_chn are genuinely budget-bound**: natural stops at
  10,021 and 8,245 tokens (+53 over the old cap for paper_chn). A 12288
  budget converts both to `stop` with complete page structure; 16384 adds
  nothing.
- **the two newspapers are degeneration-bound, not budget-bound.** Their
  outputs enter exact repetition loops, and the loop onset is
  deterministic:
  - newspaper_chn: the unit `，德不治` repeats from char 1,157 at every
    budget — 1,879× (86.7 % of output) at 8192, 2,903× (90.9 %) at 12288,
    3,927× (93.1 %) at 16384. Page coverage stays 1/5 at all budgets.
  - newspaper_eng: the unit `Member IREPS - 703-894…` repeats from char
    ~20,249 — 552× (~27.5 % of content) at 16384. Page coverage stays 2/4.
  Raising the budget only lengthens the loop. The official Baidu deployment
  counters exactly this with `no_repeat_ngram_size=35` (official
  long-document settings); our native parity path runs with the n-gram
  processor disabled (`ngram_size=0`), which `LONG_DOCUMENT_READINESS.md`
  already lists as a separate, reviewed identity change. **Budget policy
  cannot fix these two documents; repetition suppression can.**

## 4. Memory and latency measurements (measured, not extrapolated)

- **GPU peak allocated 8.54–8.55 GiB and reserved 8.76–8.78 GiB at every
  budget 8192→16384** — identical across all twelve runs: the peak is
  vision-prefill dominated; the extra KV of +8,192 output tokens (~0.5 GiB
  at ~60 KiB/token) is invisible under it. Headroom to the 32 GiB carve-out
  ≈ 23 GiB; GTT free ≥ 81.6 of 100 GiB throughout.
- Host RSS grows only 2.45 → 2.55 GiB. System RAM ~17–18 % used.
- Prefill 4.1–4.8 s for 4–5 pages. Decode throughput declines with context
  length: 14.9 tok/s (8k) → 12.6 (12k) → 10.6 (16k) — KV-length effect.
- Wall per document: 552–1,561 s (budget-bound docs stay near their natural
  stop; newspapers scale linearly with the budget they burn in loops).

## 5. Official scorer diagnostics (truncation-content view, same 4-doc population)

| Metric | 8192 | 12288 | 16384 |
| --- | --- | --- | --- |
| TextEdit | 53.94 | 54.54 | 54.01 |
| T-TextEdit | 45.30 | 45.26 | 45.24 |
| TableTEDS | 46.97 | 46.97 | 46.97 |
| T-TableTEDS | null | null | null (no merged tables in these four) |
| FormulaCDM | 90.64 | 93.27 | 93.27 |
| ReadOrderEdit | 51.38 | 55.51 | 55.51 |
| FigureF1 | 58.33 | 58.33 | 58.33 |
| HeadingTEDS | 29.07 | 31.30 | 31.30 |
| Overall | null | null | null |

Per-document (diagnostic): completion moves the two budget-bound documents
substantially — handwritten TextEdit 94.51→98.96, ReadOrder 81.1→95.28,
HeadingTEDS 20.2→29.2, FormulaCDM 90.83→93.66 (89→97 rendered formulas,
the eighth formula population growth confirming page 5 content). The
newspapers are flat (newspaper_chn TextEdit 0.89→0.81→…) — their partial
content is dominated by loop text.

**stop ≠ semantic completeness:** handwritten stops with 5/5 pages and
TextEdit 98.96, yet its HeadingTEDS is only 29.2 — a native stop proves
termination, not document quality. No Overall is claimed at any budget.

## 6. P2-A conclusions (input to the P2-B decision)

1. Output budget 12288 is sufficient for the two budget-bound documents
   (natural stops 8,245 / 10,021); 16384 adds nothing for them.
2. No budget ≤16384 rescues the two newspaper documents; their failure
   mode is degeneration, addressable only by the separately-reviewed
   official n-gram suppression (or windowed deployment semantics).
3. Output budget does not move GPU memory in this range; input pages
   dominate. Decode throughput decays ~2.3 %/1k tokens with context length.
4. Strict failure-inclusive accounting at any budget ≤16384 still leaves
   TRUNCATED documents in the denominator — a budget raise alone cannot
   produce a fully-passing dense-document subset.

Raw evidence: ledger runs `.b/mpdoc-runs/…/bf55b509…` (8192 baseline),
`.b/p2a-runs/…/3031002c…` (12288), `.b/p2a-runs/…/f7ad7a3a…` (16384);
scorer runs `.b/scorer-runs/p2a-{8192,12288,16384}-diag` (untracked).
Sanitized receipts: `eval/results/Linux gfx1151 verified/p2a-token-budget/`.
