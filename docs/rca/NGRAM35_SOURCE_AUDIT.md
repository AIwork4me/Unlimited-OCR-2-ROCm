# No-repeat n-gram (size 35): source audit and native experiment plan

Status: **AUDIT ONLY — no production change, no GPU run, no default modified.**
Branch: `docs/p1-ngram-source-audit`, stacked on the reviewed PR #20 head
(`feat/p2a-token-budget` @ `ec38e30`). This document is the review gate the
P2 opt-in experiment must pass before any GPU execution.

## 1. Audited sources and identity chain

| Tag | Source | Identity (SHA-256) |
|-----|--------|--------------------|
| S1 | Frozen original Baidu model source, `modeling_unlimitedocr.py` (HF snapshot `9cf077c`, 18/18 lock hashes verified at load time; local review copy `upstream-src/modeling_unlimitedocr.py` hash-identical) | `f6df813cef23cc28…` (prefix; full value in the model lock the loader verifies) |
| S2 | Official benchmark client `infer.py` @ `baidu/Unlimited-OCR` commit `48ff432fb68a18ed91f2f70021f29028b1dde354` (frozen copy `eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/upstream/official-infer.py.txt`, which differs only by one trailing newline added when it was saved) | `2308236c646b7400…` (frozen copy), `59f8779715b0ef02…` (upstream bytes) |
| S3 | Customized SGLang wheel `sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` (git-LFS oid at the pinned commit equals the wheel hash in `eval/benchmarks/benchmark-lock.json`; downloaded by parallel range requests, 15,281,032 bytes, hash-verified; statically inspected, **never installed or executed**) | `a3f3e2c94e09de90…`; all 10 pinned files inside verified 10/10, incl. `sglang/srt/sampling/custom_logit_processor.py` = `7d0ad94752cd7148…` |
| S4 | Native generation path: `src/unlimited_ocr_2_rocm/model.py` (loads S1 via `trust_remote_code` from the hash-verified snapshot), `completion.py::infer_document`, `inference.py::run_native` | loader verifies per-file hashes at runtime |

## 2. Exact original algorithm

All three implementations compute the same ban set (S3 `custom_logit_processor.py:343-385`;
S1 `_uocr_no_repeat_ngram` at `modeling_unlimitedocr.py:608-623`; S1
`SlidingWindowNoRepeatNgramProcessor:408-437`, whose docstring states it is aligned
with the SGLang processor):

```
sequence  = generated tokens only            # see §4
start     = max(0, len(sequence) - window)   # trailing window
end       = len(sequence) - ngram_size + 1
prefix    = tuple(sequence[-(ngram_size-1):])
banned    = { sequence[i + ngram_size - 1]
              for i in [start, end)
              if tuple(sequence[i:i+ngram_size-1]) == prefix }
banned   -= whitelist                        # official benchmark passes none
logits[banned] = -inf                        # hard ban before sampling
```

S3 additionally clips banned ids to the logits vocabulary range ("Ignore
multimodal pad sentinels above the vocabulary range", line 379-381) — a no-op
for the native full-vocab logits. Provenance of the algorithm: adapted from
DeepSeek-OCR's vLLM `process/ngram_norepeat.py` (S3 line 265 comment); S1's
functional port's docstring cites "vLLM deepseek_ocr.NoRepeatNGramLogitsProcessor
(on generated ids only)".

## 3. Where ngram_size=35 and window 1024/128 come from

S2 (official client), hardcoded:

- `infer.py:38` `NGRAM_SIZE = 35`
- `infer.py` payload builder: `window = 1024 if len(pages) > 1 else 128`
- sampling: `temperature: 0.0`, `skip_special_tokens: False`,
  `custom_logit_processor: DeepseekOCRNoRepeatNGramLogitProcessor.to_str()`,
  `custom_params: {'ngram_size': 35, 'window_size': window}`

The model's `generation_config.json` contains **no** n-gram settings; 35/1024/128
are benchmark-client constants, not model constants.

## 4. Does it apply to all generated tokens?

Yes, with two scope rules:

- **Prompt excluded.** S3 `custom_logit_processor.py:287-290`: with
  `SGLANG_UOCR_NGRAM_OUTPUT_ONLY` — default **True** (`environ.py:1423`) — the
  window slides over `req.output_ids` only. S1's port uses `gen` (the generated
  list) by construction (`modeling_unlimitedocr.py:1169`).
- **Every decode step.** The ban is recomputed per token before sampling (S1
  loop at `:1168-1176`; S3 invoked from `sampler.py:131-137` "Apply custom logit
  processors" ahead of the argmax/temperature stage).

## 5. Reset per page or per document?

No page-level reset anywhere. The history is continuous for the lifetime of one
**request**:

- Native (S1/S4): one request = one whole document → the window spans the whole
  document's generated tokens across all pages.
- Official SGLang deployment: the doc-limit scheduler splits long documents into
  windows of ≤10 pages (`SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW` /
  `SGLANG_UOCR_DOC_BASE_PAGE_WINDOW`; official README sets 10, wheel default 1 —
  the ≤10-page single-window statement rests on the deployment environment, not
  the wheel defaults). Each window is a **separate request** with its own
  `output_ids`, so the n-gram history **resets at every window boundary**. For
  documents that fit in one window (≤10 pages — including every document in the
  frozen P2 selection at ≤5 pages) the official behavior is exactly
  per-document, identical to native.

## 6. Do SGLang windows affect the scope?

Yes — see §5. Two distinct "windows" must not be confused:

1. `window_size` 1024/128 — the **n-gram trailing-window length** (how much
   generated history the ban consults).
2. The 10-page **doc-limit windows** (KV-tail reuse, `serving_chat.py`
   `_window_request`) — request boundaries that reset the n-gram history and
   carry per-window page-guard state (`KEY_PAGE_BASE`, `KEY_LOOKAHEAD`,
   `_uocr_context_pages`).

Because the frozen P2 documents are ≤5 pages (single official window), a native
per-document implementation reproduces the official scope for them exactly; for
>10-page documents native would have a strictly **larger** suppression scope
than any single official window. That difference is moot for P2 and must be
restated if 15-page experiments are ever unblocked (P5).

## 7. Order relative to other logits processing

- **SGLang official path:** `sampler.py` applies the custom processor (inside
  which the page guard runs **before** the n-gram ban, S3 `:317-341` → `:343-385`),
  then NaN sanitization, then sampling (greedy argmax at the client's
  `temperature: 0.0`). The client sets no repetition/frequency/presence
  penalties, so no other processor competes.
- **Native path (S1 loop `:1166-1176`):** n-gram ban **first**, page-count guard
  second, then argmax.
- **Order interaction — equivalent for guard bans, NOT for one forced corner:**
  guard bans union with n-gram bans in both orders. For forced tokens the two
  implementations diverge: S3's `forced` branch assigns the forced logit `0.0`
  and `continue`s past the n-gram ban (S3 `:334-339`), so the forced token
  always survives. S1's guard instead **restores the post-ban value**
  (`keep = logits[tid].clone()` taken *after* the n-gram ban, `:599,:602`):
  when the per-page-cap branch forces `<PAGE>` while `<PAGE>` is in that
  step's n-gram ban set — reachable precisely for repetition-heavy documents —
  the native row becomes all `-inf` and `torch.argmax` tie-breaks to the lowest
  index, emitting an arbitrary token instead of `<PAGE>`. This corner is a
  real native/SGLang divergence (independent-review finding 8) and gets a
  dedicated post-hoc detector in §10.
- **HF generic `NoRepeatNGramLogitsProcessor` is NOT equivalent**: it scans the
  full `input_ids` (prompt included), has no trailing-window limit, and no
  whitelist. It must not be substituted (BENCHMARK_PROTOCOL already forbids
  this; confirmed by source).

## 8. Interaction with the page guard and EOS

- Official deployment: the **server** injects the page-guard parameters into
  every window's `custom_params` (`serving_chat.py::_apply_page_cap` →
  `doc_split.page_guard_params`: `page_token_id` from
  `sparse_attention_config` (128827), `eos_token_id`, `max_page_tokens`
  (official `SGLANG_UOCR_MAX_PAGE_TOKENS=20000`; wheel default 0 = off — with
  the wheel default the guard never arms), plus `page_stop_only`/lookahead for
  KV-reuse windows). Auto-attaching the processor when a client brings none
  additionally requires `--enable-custom-logit-processor` (immaterial for the
  official benchmark, whose client brings its own). The guard runs inside the
  same processor before the n-gram ban (S3 `:317-341`).
- Guard semantics (S3 `_page_guard:194-262`): pages remaining → EOS banned;
  per-page token cap reached → force `<PAGE>` (or EOS on the last page); marker
  past the last page → force EOS; last page → `<PAGE>` banned. Forced actions
  override any n-gram ban that step.
- **EOS can never be n-gram-banned**: EOS is never part of the generated
  history (generation stops at EOS; S1 breaks before appending at `:1179-1182`,
  S3's `output_ids` likewise), so no seen n-gram ends in EOS — **conditioned on
  `ignore_eos` being unset**, which the official client does (its payload sets
  neither `ignore_eos` nor any penalty). The only way EOS is masked while pages
  remain is the guard's own ban; the only ways EOS is forced are the guard's
  cap/overflow branches.
- Native (S1 `_uocr_page_count_guard:583-605`, docstring citing vLLM
  `_PageCountGuard`): same marker/cap logic at vLLM parity, **except** S3 also
  bans `<PAGE>` on the last owed page (`:260-262`), which the native guard
  lacks — native can therefore emit one marker past the expected count and
  EOS-force on the following step (independent-review finding 9; pre-existing,
  not n-gram-specific). The native guard is on by default
  (`infer(page_count_guard=True)`), and `infer_document` (S4) does not disable
  it — P2 keeps it on.

## 9. Native vs official SGLang — difference table

| Aspect | Official SGLang benchmark | Native (current code) | P2 plan |
|---|---|---|---|
| History scope | `output_ids` only (env default True) | `gen` only | same (no change needed) |
| n-gram size | 35 (client constant) | off (0) | 35, opt-in only |
| Trailing window | 1024 multi-page / 128 single-page | unused; frozen-model default would be **100** if n enabled naively | 1024 (all P2 docs are multi-page) |
| History reset | per doc-limit window (≤10 pages) | per request/document | identical for ≤5-page docs |
| Page guard | server-injected, in-processor, first | model loop, second | unchanged, stays on |
| EOS suppression | never by n-gram; guard only | same | same |
| Whitelist | none passed | none | none |
| Sampling | greedy `temperature=0.0` | greedy (`temperature=0`) | unchanged |
| Vocab clipping | clips ids ≥ vocab (pad sentinels) | n/a (full-vocab logits) | n/a |

## 10. Implementation plan for the P2 opt-in experiment

Reviewable as a small PR after this audit is approved. No production default
changes; production stays `ngram=0`, ≤5 pages, ≤8192 tokens.

1. **Frozen experiment policy** `eval/experiments/p2b-ngram-suppression/policy.json`
   (schema 1): the four frozen documents only, `ngram_size: 35`, `ngram_window: 1024`,
   first-stage bounded budgets (start 8192; escalation only by explicit new
   reviewed policy), plus identity bindings: S1–S4 file hashes (this audit's
   chain), model/wheel hashes already enforced by the loader and lock, policy
   self-hash. Control runs are the existing immutable P2-A evidence
   (`ngram=0`), not re-runs.
2. **Opt-in plumbing** (the only code change): `run_native`/`infer_document`
   accept `no_repeat_ngram_size`/`ngram_window` (default 0/off). Values > 0 are
   **fail-closed**: allowed only when the frozen policy file is presented, its
   SHA-256 matches the digest pinned in the gate, and its identity fields match
   the run — this hardens the one gap found by
   `docs/reviews/POLICY_IDENTITY_AUDIT.md` (schema-only acceptance in
   `validate_budget` for direct callers) instead of repeating it. The
   parameters pass straight into the frozen `model.infer` signature
   (`modeling_unlimitedocr.py:1041-1042`) — zero changes to vendored/frozen
   model code. `page_count_guard` stays at its default True. Every ledger
   attempt records the ngram params + policy sha in run config and receipts,
   with fail-closed tests mirroring `test_p2a_budget_policy.py`.
3. **Order-of-work**: two degenerate newspaper documents first
   (`newspaper_chn_eb87830d`, `newspaper_eng_ed12985c`) under the bounded
   budget; no automatic escalation.
4. **Pre-run checks each invocation**: GPU memory admission, timeout
   supervision (`run_mpdoc_sample.py` supervision path), policy/source identity
   verification (loader hashes + policy sha), current model and wheel hashes.
5. **Measured fields per run** (additions to the existing report metadata):
   repetition onset token index and repeated span text, repeated n-gram
   frequency, output token count, pages covered, finish_reason, canonical
   PAGE/BLOCK validity (`require_complete_pages`), token throughput and wall
   time, GPU allocated/reserved peaks, host RSS, suppression-relevant
   statistics computed **post-hoc from `token_ids`** (the frozen model code
   exposes no processor counters and will not be instrumented): ban-set size
   per step is recomputable deterministically from the emitted ids, so the
   diagnostic script replays the §2 algorithm over `raw token_ids` and reports
   invocation/ban counts without touching generation. Two review-mandated
   properties of the replay: (i) it yields ban sets/counts, never the
   counterfactual un-suppressed token — no logits are persisted; (ii) it is
   paired with a **forced-marker consistency check**: whenever the replayed
   guard state indicates a forced `<PAGE>` step, the emitted token must equal
   the page token id — this detects the §7 native corner (all-`-inf` row) —
   and with a guard-state replay (marker/page-length counters) that also flags
   the marker-overshoot divergence of the native guard. Errors, timeouts and
   retries recorded by the existing ledger states.
6. **Success criteria**: a suppressed loop is not an OCR PASS unless content
   and structure also pass — judged only via the official scorer and the
   canonical page checks, never by repetition metrics alone.

## 11. Open items for the independent reviewer

- Confirm §7's order-interaction argument (guard-forced override in S3; the
  native post-ban restore corner) on the cited source lines.
- Confirm the §5 claim that window reset never differs for ≤10-page documents
  (single window) — in particular that Marathon-scale multi-window behavior is
  out of P2 scope.
- Confirm the policy fail-closed gate design (item 2) blocks every
  non-opt-in path, including direct `run_native` calls.
- Confirm the frozen-model default `ngram_window=100` trap: enabling n-grams
  without an explicit window would silently use 100, not 1024 — the plan makes
  the window an explicit required field for this reason.

## 12. Independent review record

2026-10-30: independent reviewer (fresh context, sources only, no trust in
this document) returned **APPROVE-WITH-NOTES**. All line citations and hashes
reproduced exactly (reviewer findings 1-7); questions (b), (c), (d) of §11
confirmed — (b) with the note that the ≤10-page single-window statement rests
on the deployment environment (README) rather than wheel defaults, (c)
conditioned on `ignore_eos` unset (the official client sets none), (d) the
`ngram_window<=0 → 100` coercion confirmed at S1 `:1092`. Question (a)
confirmed for SGLang and **corrected for native** (finding 8): the native
guard restores the post-ban logit, so the cap-forces-`<PAGE>`-while-banned
corner emits an arbitrary argmax tie-break token; §7 and §10 now carry the
corner and its detector. Finding 9 (S3 last-page `<PAGE>` ban absent from the
native guard — pre-existing marker-overshoot divergence) added to §8.
Finding 10 (exact `SGLANG_UOCR_DOC_*_PAGE_WINDOW` names) fixed in §6. Finding
11 items stay labeled as repo evidence. Finding 12 (gate blocking every
non-opt-in path) and finding 13's replay limitations are now explicit §10
requirements to re-verify at the implementation PR review.

The GPU execution gate for P2 is therefore: implement §10 as its own reviewed
PR (fail-closed pinned-digest gate, replay diagnostics with the forced-marker
consistency check), pass CPU gates and independent review of that PR, and only
then run the two degenerate newspaper documents under the bounded budget.
