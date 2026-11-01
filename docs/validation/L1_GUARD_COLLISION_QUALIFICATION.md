# L1 guard-collision qualification (Task B) — CPU results, observer contract, GO decision

**Date: 2026-10-31. Scope: CPU-only qualification of the audited native
guard-collision corner at the 39 321-token budget class, an opt-in
experimental pre-selection observer, and the residual-risk contract for the
single authorized L1 run. No frozen model change, no weight change, no
production-default change.**

## 1. Hazard restatement (from the audited frozen semantics)

`modeling_unlimitedocr.py` (pinned revision `9cf077c4b206e0e2a31879df937b0b0b
cba3661c`) decodes as: n-gram ban (`_uocr_no_repeat_ngram`, 35/1024 over
generated ids only) → page guard (`_uocr_page_count_guard`, expected_pages=15,
max_page_tokens=20000) → greedy argmax. When the guard forces `<PAGE>`
(per-page cap reached) while `<PAGE>` is in the current 35-gram ban set, the
forced token keeps its POST-ban `-inf` logit: the whole row becomes `-inf`
and argmax tie-breaks to an implementation-defined index instead of the
forced marker (P2-C fixture 1). At budgets ≤ 16384 the corner is
structurally unreachable (page_len ≤ budget < 20000); at 39321 it is live
for the first time.

The existing replay detector (`scripts/analyze_ngram_evidence.py`,
`forced_token_was_banned` / `forced_mismatches`) identifies the corner AFTER
execution from persisted token ids. It does not prevent it.

## 2. Opt-in experimental observer (implemented, `collision_observer.py`)

`GuardCollisionObserver` wraps the two frozen module functions BY
CALL-THROUGH at runtime — the same non-algorithmic instrumentation pattern
already reviewed in `model.load_model` (kernel observation). Contract:

1. **Non-invasive**: the frozen snapshot file is never edited (test-pinned);
   safe steps return the frozen result bit-for-bit (`torch.equal`, tested
   over 50 randomized states); the only added per-step work is one
   `isfinite` reduction in the guard path; installation is scoped to one
   generation and fully reverted (identity-checked uninstall).
2. **Pre-selection detection**: after the frozen guard returns, the observer
   counts finite logits. Zero finite entries ⇒ the audited corner exists —
   the observer raises `UnsafeSelectionCollision` BEFORE the loop's argmax
   runs, so an unintended token is never selected, never enters the history,
   and never reaches the output. The exception carries guard state
   (expected_pages, page_len, marker_cum) and the FULL partial token stream
   (delivered to the worker's evidence sink before the raise).
3. **Observation**: forced-marker steps (exactly one finite entry at
   page/eos id) are recorded live with their guard state; a sole finite
   non-marker entry is recorded as an anomaly (impossible via the frozen
   guard alone; replay would flag the divergence).
4. **Abort classification**: the raised exception propagates through
   `run_native`'s failure handling → report `FAIL` with the collision
   traceback → ledger state `FAIL` (never PASS, never silent).
5. **Opt-in only**: installed by the L1 worker when and only when the pinned
   L1 policy opts in; nothing else loads this module.

## 3. CPU qualification results (executed on this host, GPU venv torch 2.12.0+rocm7.2)

`tests/unit/test_l1_collision_observer.py` — 12/12 PASS (AST-extracted,
hash-pinned frozen functions; the P2-C fixture suite
`test_p2c_guard_collision.py` — 7/7 — still passes unchanged):

| # | Required case | Result |
| --- | --- | --- |
| 1 | PAGE forced AND banned (collision) | observer aborts pre-selection; error carries page_len=34, marker_cum=1, step=69, full partial ids |
| 2 | PAGE forced, not banned | no abort; forced_page event; argmax = forced marker |
| 3 | All `-inf` logits row | detection is exactly this condition (finite_count == 0) |
| 4 | Forced EOS (overshoot, mc > expected) | forced_eos event; row keeps finite EOS; no abort |
| 5 | Normal PAGE emission | passthrough bit-identical; no event |
| 6 | Page-count underflow (mc < expected) | EOS ban observed; row not all-`-inf`; no abort |
| 7 | Page-count overshoot (mc > expected) | forced EOS; no abort |
| 8 | Multiple-page histories | scripted decode transitions match `guard_decision` exactly |
| 9 | n-gram window boundaries | stale 35-gram beyond 1024 → no ban → forced PAGE survives; inside window → abort |
| 10 | Token-selection behavior | collision aborts before argmax; safe steps argmax as frozen |
| 11 | Replay detector agreement | 20 randomized decodes (ngram ∈ {2,3,5,35}, window ∈ {16,128,1024}, caps ∈ {8,16,64}): observer forced-PAGE events == replay `forced_step_list` exactly |
| 12 | Event caps / sink hygiene | bounded event lists with dropped counters; broken sink never masks abort; partial ids delivered at abort time |

Known asymmetry (pinned by test 11): a FORCED EOS always terminates the
loop and EOS never enters `token_ids`, so the post-hoc replay cannot see a
terminal forced-EOS step; the observer can. Forced-PAGE steps must agree
exactly and do.

## 4. Residual risks (explicit)

- R1 — The observer checks the row AFTER the frozen guard returns. A hazard
  between the guard's return and the observer's check does not exist (no
  code runs there in the wrapper). The check itself is a single reduction.
- R2 — If the observer's abort path itself failed (e.g., exception swallowed
  by a future frozen-loop change), behavior would degrade to the P2-C
  status quo: post-hoc replay detection, run classified FAIL. The frozen
  loop is hash-pinned and its `try/finally` does not swallow exceptions.
- R3 — A collision abort loses the continuation of the generation (by
  design — emitting from an all-`-inf` row is the failure mode). Partial
  ids are preserved for replay and post-mortem.
- R4 — The observer adds one GPU sync per decode step (~0.5 % of the ~100 ms
  step budget at the observed floor). The timeout margin absorbs this.
- R5 — The `finite_count == 1` classification assumes the n-gram ban set
  cannot cover all but one of ~129 k tokens (window 1024 ⇒ ≤ 1024 banned).
  If a future policy widened the window to the full output, forced-marker
  classification would need revisiting; the L1 policy pins window 1024.

## 5. Decision

**GO for the single authorized L1 run** under the following contract,
recorded in the pinned L1 policy:

1. The observer is installed for the attempt (opt-in, non-invasive,
   call-through, full revert).
2. Any observer abort classifies the attempt FAIL and BLOCKS promotion —
   the collision is a declared failure mode, not a crash to be retried.
3. The post-run replay forced-marker detector remains a HARD gate with no
   bypass; "no mismatch observed" never implies "observer unnecessary".
4. No post-hoc detector claim is ever worded as prevention; prevention
   language is reserved for the pre-selection abort above, which is
   validated by the tests in §3.
