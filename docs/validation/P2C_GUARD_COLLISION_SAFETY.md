# P2-C Phase 1 — n-gram safety revalidation and guard-collision fixtures

**Date: 2026-10-30. Scope: CPU-only revalidation of the reviewed P2-B
implementation (PR #23/#24) plus new synthetic guard-collision fixtures. No
model change, no weight change, no production-default change.**

## 1. Identity chain (physically re-verified on this host)

| Identity | Value | Verified |
| --- | --- | --- |
| Policy digest | `0095b0c47da9a693d6bef70305a4385b986ea79aac129d628a500dac0ef8e1f5` | sha256 of `eval/experiments/p2b-ngram-suppression/policy.json` == pinned `inference.EXPECTED_NGRAM_POLICY_SHA256` |
| Policy contents | schema 1, `P2B_NGRAM_SUPPRESSION_RCA`, ngram_sizes `[35]`, window `1024`, output-only scope, frozen 4-document list | gate rejects any other (size, window) pair |
| Model snapshot | `9cf077c4b206e0e2a31879df937b0b0bcba3661c` | 16/16 lock files + 2/2 weight shards SHA-256 verified (18/18) |
| Torch wheel | `torch-2.12.0+rocm7.2-cp310` sha `099e265de73094cb6a2112ce9d7940c59bae0c884e9516f16719e392a5471df6` | matches pinned wheel; `hip 7.2.53211 / rocm 7.2.1` |
| SGLang wheel | `sglang-0.5.20.dev10+g8d62f5fb9c` sha `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` | == `benchmark-lock.json` `sglang.sha256` (reference identity of the official semantics; not installed for native runs) |
| Frozen decoder fns | `_uocr_no_repeat_ngram` sha `c5d54682…`, `_uocr_page_count_guard` sha `51831ce2…` | AST-extracted from the pinned snapshot; tests fail on any drift |
| Special tokens | `eos_token_id=1`, `page_token_id=128827` | frozen `config.json` |

## 2. Native semantics confirmed in the frozen loop

`modeling_unlimitedocr.py` decode loop (pinned revision):

1. logits = lm_head(last hidden);
2. **n-gram ban first** (`_uocr_no_repeat_ngram`): scans the trailing
   `window` (1024) of **generated ids only** — prompt ids are never passed;
   bans every continuation of any n-gram whose (size−1)-prefix equals the
   current trailing prefix; whitelist (empty in P2-B/P2-C) subtracted;
3. **page guard second** (`_uocr_page_count_guard`): marker bookkeeping over
   gen ids; overshoot → force EOS; per-page cap (`max_page_tokens=20000`) →
   force `<PAGE>` (or EOS on the last page); under-count → ban EOS only;
4. greedy argmax at temperature 0; EOS breaks with `finish='stop'` before the
   id enters history — EOS can therefore never appear in the ban set.

The gate (`inference.validate_ngram`) enforces size 35 / window 1024 with the
pinned digest; a non-positive window would be silently coerced to 100 by the
frozen model, which the gate rejects explicitly.

## 3. Guard-collision fixtures (`tests/unit/test_p2c_guard_collision.py`)

Seven fixtures execute the FROZEN functions (extracted by AST, hash-pinned,
executed with torch; skip cleanly where torch or the snapshot is absent):

1. **Collision — forced `<PAGE>` while banned**: a history whose trailing
   34-token prefix repeats the 35-gram that ended page 1 bans `<PAGE>` while
   the guard (page cap reached, markers < expected) forces `<PAGE>`. The
   guard keeps the token's POST-ban logit: the whole row is `-inf` and argmax
   falls to a tie-break index ≠ the forced marker. **This is a real,
   characterized inconsistency of the frozen guard** — the model would emit
   an arbitrary (vocab-index-tie-break) token instead of `<PAGE>`.
2. **No collision — forced `<PAGE>` wins normally** (same guard state, prefix
   not in history): argmax returns the forced marker.
3. **Forced EOS never collides**: 200 randomized histories → EOS (id 1) never
   enters the ban set; the overshoot path keeps a finite EOS logit.
4. **Order net-equivalence**: swapping the two processors at the collision
   yields the identical all-`-inf` row — the native order and the SGLang
   processor order are net-equivalent at temperature 0 at this corner.
5. **Output-only, windowed scope**: a recurring 35-gram beyond the trailing
   1024 tokens does not ban; inside the window it bans; prompt ids never
   enter the scan.
6. **Differential vs the replay analyzer**: 500 randomized cases (ngram
   ∈ {2,3,5,35}, window ∈ {16,128,1024}, varied guard states) — the frozen
   functions and `analyze_ngram_evidence.banned_tokens/guard_decision` agree
   on ban sets and forced/banned decisions.
7. **Reachability arithmetic**: the forced-`<PAGE>` path needs
   `page_len ≥ 20000`; with budgets 8192 / 12288 / 16384 the corner is
   **structurally unreachable** (page_len ≤ budget < 20000). At the official
   39 321 class it becomes live for the first time.

## 4. Verdict

- The reviewed implementation remains identity-clean and the native page
  guard is **SAFE for the P2-C budgets 12288 and 16384**: the only unsafe
  corner (forced marker inside the ban set) cannot engage below the 20 000
  token per-page cap, and the remaining guard paths (EOS ban under-count,
  forced EOS overshoot) cannot collide with the ban set at all.
- The corner is live at the 39 321 class: any L1 spec MUST keep the
  forced-marker consistency detector armed (already implemented in
  `analyze_ngram_evidence.replay` as `forced_token_was_banned` /
  `forced_mismatches`, exercised by P2-B with 0 occurrences at 8192).
- No gate bypass, no guard change, no model change was made or is needed.

Run matrix: fixtures executed with the GPU venv Python (torch 2.12.0+rocm7.2)
— 7/7 PASS; under the CI Python they skip (no torch), with the replay-side
equivalents still covered by the always-on PR #24 tests.
