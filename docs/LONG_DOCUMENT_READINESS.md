# Long-document readiness — analysis before any cap change

Date: 2026-10-29 (Asia/Shanghai). Scope: what must be true before the
production five-image / 8192-token caps may be raised, and a proposed physical
15/30/50/100-page validation ladder with explicit stop conditions. Analysis
only — **no cap is changed by this document**, and independently processed
pages are never treated as equivalent to one full-document request.

Sources analyzed (frozen identities):

- upstream `modeling_unlimitedocr.py` @ `baidu/Unlimited-OCR-2` revision
  `9cf077c4b206e0e2a31879df937b0b0bcba3661c` (`infer`, `_uocr_generate`,
  `_uocr_page_count_guard`, `_uocr_dash_decode_allow`, `_uocr_build_inputs`)
- upstream `config.json` (same revision)
- vendored converter `_upstream/process_unlimitedOCR2.py` (byte-frozen)
- docs/BENCHMARK_PROTOCOL.md official deployment policy (audited client)
- production adapter `inference.py` / `completion.py` caps and contract

## 1. Current caps and where they live

| Cap | Value | Where | Nature |
| --- | --- | --- | --- |
| Images per request | 5 | adapter `validate_inputs` | **our Phase 1 safety cap**, not upstream |
| max_new_tokens | ≤ 8192 (default 1024) | adapter `run_native` | **our Phase 1 safety cap** |
| max_length (context) | 131,072 | upstream `infer` default | checkpoint-backed (`max_position_embeddings` 131,072, YaRN 32,768→131,072, theta 500,000) |
| per-page output cap | 20,000 tokens | upstream `max_page_tokens` | PageCountGuard forced page-flip |
| crops per page | ≤16 (+base) | upstream `max_crops` | vision tiling (single-page gundam mode only) |
| contract stop-vs-length | stop needs one decode iteration beyond returned ids | adapter `completion_metadata` | consistency proof, unchanged |

Upstream `infer` itself accepts an ordered page list of arbitrary length; the
five-image limit is our containment decision, so lifting it is a reviewed
adapter change plus physical proof — never a constant edit.

## 2. Context window and image-token budget (measured from source)

Multi-page (base) mode token cost per page — `nqb = ceil((1024/16)/4) = 16`,
so per page: `<PAGE>` + 273 `<image>` tokens (16×17+1) + a few text tokens
≈ **275 prompt tokens per page** (matches DASH `per_page_image_tokens: 273`).

Single-page gundam mode adds up to 16 tiles @640 (nq=10; a 4×4 tile grid adds
~1,640 tokens) — only relevant to single-page runs, not the long-document path.

| Pages | Prompt tokens (approx) | Output at official 0.3·len budget (MPDocBench 131,072) | Total vs 131,072 |
| --- | --- | --- | --- |
| 15 | ~4.1 K | 39,321 max | fits easily |
| 30 | ~8.3 K | 39,321 max | fits |
| 50 | ~13.8 K | 39,321 max | fits |
| 100 | ~27.5 K | 39,321 max | fits (~66.8 K) |
| ~300 (Marathon max doc) | ~82.5 K | 39,321 | ~121.6 K — near the limit; official Marathon serving used 786,432 context, i.e. 6× the checkpoint's positional table, with one refit path after prompt-length rejection |

Consequence: 15–100 pages fit the native 131,072 window with the official
per-request output budget; documents approaching ~300 pages do not fit under
one native request and require the official **10-page windowed** policy (see
§4), not a context-extension hack.

## 3. EOS / length / page-guard semantics (must not be weakened)

`_uocr_page_count_guard` (per decode step, in place on logits):

- counts generated `<PAGE>` markers (`marker_cum`); if it exceeds the prompt's
  page count → **EOS is forced** (finish_reason `stop`, but this is a forced
  termination, not model-chosen EOS);
- per-page token length (`page_len`) reaching 20,000 → the next token is
  forced to `<PAGE>` (more pages expected) or EOS (last page);
- while fewer markers than expected pages have been emitted → **EOS is
  banned**, so the model cannot stop early;
- plain EOS ends generation with finish `stop`; hitting `max_new_tokens` (or
  the `max_length` positional break) ends with finish `length`.

The adapter's completion contract already proves: returned ids ≤ budget;
`stop` requires one decode iteration beyond the returned ids; `expected_pages`
== input pages; `num_pages_out` == `<PAGE>` count in ids == count in text; and
canonical PAGE/BLOCK structure per page. Raising caps does not change this
contract, but it makes two failure modes *more likely*, so they become explicit
gate items: guard-forced `stop` (semantic truncation risk on over-budget
pages) and `length` truncation (contract FAIL, must be reported, never
retrayed silently).

## 4. Official deployment semantics vs our native path (non-equivalence is the default)

The audited official client (docs/BENCHMARK_PROTOCOL.md):

- enables `SGLANG_UOCR_EANABLE_DOC_LIMIT=1` (upstream spelling) with
  **gundam/base windows of 10 pages**, per-page cap 20,000, **reusing previous
  window KV tails** — i.e. official long documents are processed as chained
  10-page windows, not necessarily one whole-document request;
- whole-document single-window mode is a *separate identity*;
- custom `DeepseekOCRNoRepeatNGramLogitProcessor`: n-gram 35, window 1024
  (multi-page) / 128 (single page) — our native `_uocr_no_repeat_ngram`
  implements the same semantics but the adapter currently invokes `infer`
  with `no_repeat_ngram_size=0` (disabled) in reviewed runs;
- official output budget `int(0.3 × max_model_len)` = 39,321 (MPDocBench) /
  235,929 (Marathon @786,432 context).

Therefore: a native one-request 100-page run is **not automatically
equivalent** to the official windowed pipeline. Both modes must be labeled
distinctly; window boundaries, KV-tail reuse and every forced page-flip /
truncation event must be recorded per document.

## 5. KV cache / SR-SWA memory behavior on gfx1151 (32 GiB carve-out + GTT)

- The checkpoint config carries `_ring_window: 128` / `sliding_window_size:
  128` (SR-SWA ring buffer), but `_uocr_generate` **explicitly disables the
  ring window for inference** (`cfg._ring_window = None; cfg.sliding_window =
  None`) because decode attention is served by DASH. Consequence: the
  `DynamicCache` retains the **full** prompt+output KV; memory grows linearly
  with total tokens — the SR-SWA window does *not* bound memory here.
- Decoder is not MLA (`use_mla: false`, `kv_lora_rank: null`): KV per token
  per layer = 2 × 10 heads × 128 dim = 2,560 BF16 elements = 5 KiB; ×12 layers
  = **~60 KiB per token** (≈ 58.6 GiB at the full 131,072 context — not
  reachable on this host in the 32 GiB carve-out without GTT spill; reachable
  ceilings below).
- DASH decode further restricts *attention* per query to KV blocks holding the
  current page's image tokens (plus fallback-open rows) — this bounds compute,
  not memory.
- Practical native ceilings on this host (weights ~6.7 GiB BF16 + KV + vision
  prefill peaks, single document): KV alone reaches ~24 GiB at ≈ 420 K
  tokens — far beyond 131,072; so **context position table (131,072), not KV
  capacity, is the binding constraint** for ≤100-page runs: 100 pages ≈ 27.5 K
  prompt + ≤39,321 output ≈ 66.8 K tokens ≈ **3.9 GiB KV** — comfortably
  inside the carve-out. Vision prefill peaks (273-token pages are tiny; SAM+CLIP
  encode all pages' 1024² base images) dominate short-run memory and must be
  measured, not assumed.
- APU accounting rule: all figures are shared-system-memory accounting.

## 6. Proposed physical validation ladder (explicit stop conditions)

Preconditions (all must hold before page 1 of the ladder):

1. Linux gfx1151 environment PASS: doctor profile, real BF16 GPU matmul,
   12-layer DASH Triton kernel execution with dense-prefill fallback refused,
   scoped vision MATH SDPA evidence (Task 3/4 gates of this change set).
2. 1/2/3/5-page physical regression PASS unchanged (existing fixtures), as the
   anchor that cap-raising changed nothing in the ≤5-page regime.

Ladder (each step = fresh process, fresh output dir, same fixtures extended by
ordered real pages; single request per run; both finish-reason and structure
contract enforced):

| Step | Pages | max_new_tokens | Pass criteria (all required) | Stop conditions (abort ladder) |
| --- | --- | --- | --- | --- |
| L1 | 15 | 8192 (unchanged cap) | finish `stop`; 15 `<PAGE>` markers; ≥12 DASH kernels; canonical PAGE/BLOCK per page; peak GPU mem recorded; wall time recorded | finish `length` on any run; guard-forced stop with any page structurally malformed; OOM/HCQ/hang; >2× expected wall time |
| L2 | 30 | 8192 | same, 30 pages | same, plus VRAM+GTT headroom < 20% |
| L3 | 50 | 16,384 (first cap raise, reviewed) | same, plus output tokens vs official 0.3·len budget comparison | same |
| L4 | 100 | 39,321 (official budget) | same, plus full contract audit incl. guard event log | any contract violation; prompt+output approaching 131,072 (log position usage) |

Between steps: reconcile raw output page count, per-page token histograms
(page_len distribution vs 20,000 cap), marker positions, KV/memory growth
curve (measured vs 60 KiB/token model), and compare L1–L4 results against the
reviewed Windows ≤5-page evidence only as *regression anchors* (Windows has no
>5-page evidence — no cross-platform long-doc comparison is claimable).

Explicitly out of scope until the ladder passes: windowed 10-page chaining
with KV-tail reuse (official semantics port), >100-page documents (needs the
windowed path by §2 arithmetic), n-gram processor enablement for benchmarks
(separate identity change), SGLang parity.

## 7. What the adapter change itself will look like (proposal, not implemented here)

- Replace the hard `>5` rejection with an explicit `--max-pages` bound whose
  default stays 5 (no silent behavior change; CLI-visible, recorded in every
  manifest).
- Raise the `max_new_tokens ≤ 8192` ceiling to the official 39,321 budget
  **only** behind the same flag family, keeping 1024 default.
- Record per-run: page count, prompt token count, position usage
  (prompt+output vs 131,072), every guard-forced page flip / forced EOS
  (requires exposing guard state — upstream already tracks `marker_cum`,
  `page_len`), finish reason, page structure contract results, memory curve.
- Keep `single` semantics (auto crop) untouched; ladder uses `multi` (base
  mode) only.

This document proposes the PR; implementation follows review of the Linux
physical regression evidence, per the no-automatic-merge constraint.
