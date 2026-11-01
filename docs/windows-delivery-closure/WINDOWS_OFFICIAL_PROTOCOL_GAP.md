# Windows Official Benchmark Protocol Gap Analysis (Phase 4)

CPU/documentation work only — no full-corpus GPU inference is authorized.
Classifications: PASS / FAIL / BLOCKED / NOT_RUN / NOT_APPLICABLE.

## 4.1 Dataset denominators — PASS (verification only; inference NOT_RUN)

Stages are deliberately not combined. Receipt:
`eval/windows/dataset-readiness.json` (public copy byte-identical, SHA-256
`541fa1fae3a987825958f794d9dac851670412d0178057ee2c90252862e80233`).

| Stage | MPDocBench | MarathonDocBench |
| --- | --- | --- |
| Dataset downloaded | PASS (688,382,455 bytes) | PASS (11,805,490,064 bytes) |
| Files verified (per-page SHA-256, full decode, dimension) | PASS 3135/3135 | PASS 18272/18272 |
| Ground truth verified (frozen GT hash) | PASS (`0a579dca…`, 34,593,225 B) | PASS (`7db0e659…`) |
| Documents verified | PASS 420/420 (scope `785f1888…`) | PASS 100/100 (scope `2515b579…`) |
| GPU inference completed | **NOT_RUN (0/420 docs, 0/3135 pages)** | **NOT_RUN (0/100 docs, 0/18272 pages)** |
| Official scorer completed | **NOT_RUN** | **NOT_RUN** |
| Metric coverage completed | **NOT_RUN (all eight null)** | **NOT_RUN (all eight null)** |
| Official score comparison completed | **NOT_RUN (deltas null)** | **NOT_RUN (deltas null)** |

Denominator integrity is enforced in code, not convention:
`benchmark/manifest.py` hard-codes `DENOMINATORS = {"MPDocBench": (420, 3135),
"MarathonDocBench": (100, 18272)}` and refuses changes; a committed test
reconciles the JSONL manifests to exactly these counts; prediction inventory
must equal the full GT inventory (one `<doc_id>.md` per document — failures
export as explicit empty files, never dropped). Paper denominators
(433/3246) are never interchangeable with the public 420/3135.

## 4.2 Official comparison semantics — FROZEN

- References: Baidu Unlimited-OCR-2 paper, page 10, Tables 1/2 — MPDocBench
  Overall **84.59** (may be displayed 84.6), Marathon Overall **77.06**.
  Per-metric paper cells are recorded in `docs/windows/MPDOCBENCH_FULL_RESULTS.md`
  and `MARATHON_FULL_RESULTS.md` (Edit-family cells are raw distances; the
  scorer reports `100*(1−distance)`).
- Overall is quoted, never recomputed from rounded per-metric cells; future
  deltas use the full-precision values the scorer emits.
- Until a complete, correctly scored campaign exists: Windows Overall = null,
  official delta = null, every unmeasured metric = null. Synthetic Golden
  CER=0 is not a score. Never substitute zero.

## 4.3 Runtime protocol comparison — NOT EQUIVALENT (must never be labeled so)

| Dimension | A. Production Native (Windows, PR #33) | B. Pinned experimental Native L1/L1.1-B (PR #32) | C. Official customized SGLang (Baidu) |
| --- | --- | --- | --- |
| Pages per request | ≤5 (`validate_inputs(limit=5)`) | exactly 15, single document, 15 pinned page hashes | 10-page windows |
| Windowing / KV | none; one request per document, sequential per-document worker subprocesses | one request, no windowing, continuous DynamicCache, no KV reuse | sliding windows with KV-tail reuse across windows |
| Token budget | 8192 (P2A policy may admit 12288/16384) | 39321 (policy digest pinned) | 39321 (MPDocBench) / 235929 (Marathon) with token-table/refit rules |
| n-gram suppression | default 0 (no suppression support on delivery branch) | ngram 35 / window 1024, generated-ids only, continuous history | ngram 35 / window 1024, per-window reset |
| Guard collision | n/a at ngram 0 | opt-in observer, abort = FAIL, blocks promotion | unknown (closed source) |
| Stopping / page guard | native stop; `length` ⇒ TRUNCATED; complete-pages required | native stop + forced-marker replay HARD gate | native protocol semantics |
| Failure classification | PASS/FAIL/TRUNCATED/TIMEOUT/OOM/INTERRUPTED/NOT_RUN | same + watchdog exit 86, observer abort | official campaign semantics |
| Resume | sealed-ledger skip only on exact-identity complete PASS | full-denominator ledger (L1: 1 PASS / 419 NOT_RUN) | n/a |
| Physical evidence | Windows gfx1151 @ `3556e38` (26/26 synthetic) | **Linux gfx1151 only** (run `3562dbb9…`, 1214.5 s, 8.617 GiB peak) | Baidu official runs |

Same model weights do not make A/B/C equivalent. No Marathon-budget (235929)
policy exists anywhere yet.

## 4.4 PR #32 integration implications

Available **after #32 merges** (none of these are on the Windows delivery
branch today):

| Capability | Status in #32 | Effect on Windows benchmark readiness |
| --- | --- | --- |
| N-gram suppression (`validate_ngram`) | PASS, digest-pinned (P2B/P2C/L1/L11B), fail-closed, positive-window guard | Enables ngram35/window1024 under policy only — still not official per-window semantics |
| Budget policy gate | PASS; L1/L11B @39321 pinned at tip `a720c49c` | 39321 admitted for the two pinned single-document experiments only; 235929 still absent |
| Guard-collision observer | PASS, opt-in, abort=fail | Required safety for 39321-class runs |
| Resource watchdog | PASS, opt-in (thread, exit 86) | Available; still not wired into the common runner (Windows production runs unwatched) |
| Longer-document admission | PARTIAL — 15 pages, two pre-declared documents only | No general >5-page path; no 10-page windowing; no KV continuation |

Merging #32 authorizes **nothing new** on Windows: its L1 evidence is Linux
(torch 2.12.0+rocm7.2, triton-rocm 3.7.0 — different from the Windows
rocm7.14.1/triton-windows stack), and every capability is fail-closed behind
policy digests. Windows full-corpus blockers remain: campaign authorization,
5-page/8192 production limits, protocol non-equivalence, and the scorer below.

## 4.5–4.6

See `WINDOWS_SCORER_READINESS.md`, `WINDOWS_LINUX_SCORER_BRIDGE_DESIGN.md`,
`OFFICIAL_METRIC_COVERAGE_GATE.md`, and the machine-readable
`BENCHMARK_READINESS_MATRIX.json`.
