# P1 real-document GPU-to-official-scorer integration — REAL_DATA_INTEGRATION_ONLY

Date: 2026-10-30 (Asia/Shanghai). Phases covered: **P0 integration closure,
Linux ground-truth completion, local official scorer closure, real-document
GPU runner, end-to-end MPDocBench sample evaluation** on the Linux gfx1151
host (HP ZBook Ultra G1a, Ryzen AI MAX+ PRO 395 / Radeon 8060S). Baseline:
PR #11 → PR #12 → PR #13 (`0ec78b0`); all work below is on the branch stacked
on PR #13.

## 1. P0 integration closure — PASS

- GitHub PR status inspected: #7/#8/#9/#11/#12/#13 all OPEN and MERGEABLE;
  #5 (experimental) CONFLICTING and untouched; no automatic merge.
- Clean temporary worktree `integration/p0-closure` at merge commit
  **`4cd5945a8f5de573b272a7e75041069b4f3c74a9`** = PR #13 head + PR #10 head
  (`d8485be`). **Zero file intersection** between the #10 delta
  (`experiments/native_regression/` framework, its 3 test files, Windows
  evidence, docs — 22 files, +7847 lines) and the #11–#13 delta; the merge is
  conflict-free by construction. #10 touches no `src/` runtime code, so no
  targeted GPU regression was required by the phase rules.
- CPU gates on the integration tree: pytest **304 passed / 5 skipped**
  (270 from the #11–#13 stack + 34 from #10's tests), ruff check + format
  clean over `src/ scripts/ tests/ experiments/`, `uv build` (wheel+sdist),
  `pip check`, mypy (17 files, clean), both CLI entry points smoke.
- Integration readiness: the stack is a fast-forward chain — merging #13
  already contains #7–#12; #10 can be merged independently anywhere after
  #9 with no conflicts.

## 2. Linux ground truth — VERIFIED (both pinned files, byte-exact)

| File | Source (pinned) | Bytes | SHA256 match |
| --- | --- | --- | --- |
| MPDocBench GT | `Tongyi-Zhiwen/Qwen-Doc` `MPDocBench/MPDocBench.json` @ `4e5aee32` (the lock's own `revision`), fetched via api.github.com raw (raw.githubusercontent blocked here) | 34,593,225 | `0a579dca…8f40df` **exact** |
| MarathonDocBench GT | HF `baidu/MarathonDocBench` @ `fa05fccc` via hf-mirror | 114,122,727 | `7db0e659…adf51c` **exact** |

Stored outside Git at `../gt/`. Reconciliation run with the real GT (first
annotation-level verification on this host): MPDocBench **420/420
annotation digests + 420/420 ordered-page digests** match the frozen
manifests (409/420 archive page-hash digests + the 11 authorized SlideVQA
decks by page index, as established in D0); MarathonDocBench **100/100 on
all three digests**. Evidence chain closed: GT → manifests (scope-bound) →
page index → D0 per-page receipts → local bytes.

## 3. Local official scorer — PASS, real FormulaCDM rendering

Both scorer source trees rebuilt and hash-verified (85/85 files each):
qwen = upstream `Qwen-Doc` MPDocBench tree; marathon = same tree + the 3
official patch files from the HF dataset's `utils/mpdocbench_patch`
(downloaded individually, SHA-matched to the lock). Torch-free CPython 3.10
venv from `requirements/benchmark-scorer.lock.txt` (73 packages, pip check
PASS). TeX stack assembled entirely under `../scorer-build/tools/` from
jammy 2021 debs (no system changes): texlive-binaries/base/latex-base/
latex-recommended/fonts-recommended/latex-extra/science/xetex/lang-chinese,
fontconfig, teckit, libicu70, libteckit0; xelatex.fmt generated locally;
ImageMagick 7.1.2-34 Q16-HDRI AppImage (PR #9 used 7.1.2-26 Q16-HDRI —
recorded difference); system gs 9.55.0; node v16.13.1 binary byte-identical
to the PR #9 inventory.

Results (`.b/scorer-runs/`, receipts in
`eval/results/Linux gfx1151 verified/scorer/local-scorer-closure.json`):

| Run | Result |
| --- | --- |
| qwen pipeline | all eight metrics; the seven non-formula metrics byte-identical to PR #9; **FormulaCDM 100.0 with real rendering** (CDM distance 0 on the authored x+y=z fixture); Overall 55.21 |
| qwen cases | exact F1=1.0 (both sides truly rendered: 5 color-bbox tokens + valid PNGs); mismatch F1=0.4; error case EXPECTED_RENDER_ERROR; timeout case EXPECTED_TIMEOUT with 0 surviving descendants |
| marathon pipeline | same eight-metric PASS, FormulaCDM 100.0 |
| marathon cases | 4/4 PASS, same classifications |

Integrity gates observed live: `check_sources` refused a `__pycache__` left
by an earlier run (correct refusal, tree re-cleaned); the pre-TeX-fix run
produced FormulaCDM=0.0 with `rendered=false` — identified as a renderer
failure, not a score, and re-run after completing the TeX stack.

## 4. Real-document GPU runner (P1.1/P1.2) — PASS

`scripts/run_mpdoc_sample.py` (4 CPU unit tests) + predeclared selection:
**8 documents / 36 pages**, chosen before any inference by category from
the 171 ≤5-page candidates — Chinese/English text, handwritten Chinese,
tables, formulas, figures, headings, 3–5-page cross-page continuation
(selection + per-document GT category counts frozen in
`predeclared-selection.json`, SHA256 `5cbce32b…`). Before inference every
page is re-hash-checked against the frozen D0 page index. One
`infer_multi` request per document; production limits unchanged (5 pages,
8192 new tokens, 16 Mpx/page). Per document preserved: raw output (kept
locally), finish_reason as reported natively, token counts, DASH kernel
count, prefill/generation timings, GPU allocated/reserved/peak and process
RSS, input page SHA256s, model/wheel/environment evidence.

## 5. End-to-end real MPDocBench sample — COMPLETED, REAL_DATA_INTEGRATION_ONLY

GPU run (venv `linux-gfx1151-rocm`, model cache SHA-verified in PR #11;
total wall 3407 s):

| Document | Pages | Result | finish | tokens |
| --- | --- | --- | --- | --- |
| researchReport_chn_a2f46550 | 5 | INFERENCE_ONLY | stop | 7,603 |
| researchReport_chn_dd0407a2 | 4 | INFERENCE_ONLY | stop | 4,829 |
| examPaper_eng_66b20269 | 4 | INFERENCE_ONLY | stop | 2,894 |
| goverment_eng_aa75e353 | 4 | INFERENCE_ONLY | stop | 3,143 |
| paper_chn_3567e55d | 5 | **FAIL** | **length** | 8,192 |
| handwritten_chn_1ae487be | 5 | **FAIL** | **length** | 8,192 |
| newspaper_chn_eb87830d | 5 | **FAIL** | **length** | 8,192 |
| newspaper_eng_ed12985c | 4 | **FAIL** | **length** | 8,192 |

Four documents (dense Chinese newspapers, a paper, a formula-heavy
handwritten document) exhausted the unchanged 8192-token budget; they are
recorded as FAIL with native `length` — never relabeled `stop` — and stay
in the denominator. 12 DASH kernels per document; GPU peak allocated
8.54–8.55 GiB across all runs (32 GiB carve-out); prefill ≈2.8 s; generation
140–592 s per document.

Official qwen scorer over the real outputs (selftest gate PASS first):
TextEdit **76.89**, T-TextEdit **57.28**, TableTEDS **82.68**,
FormulaCDM **90.94** (102 real formula renders), ReadOrderEdit **75.07**,
FigureF1 **78.23**, HeadingTEDS **53.70**. **T-TableTEDS = null** — the
subset contains no cross-page merged tables (26 single-page tables), so the
metric is not exercised; per the official rule Overall is **null** whenever
any of the eight is invalid. No value was invented for the absent metric.

**Labels: REAL_DATA_INTEGRATION_ONLY / subset, non-official.** These
numbers are NOT full MPDocBench accuracy, NOT leaderboard-comparable, and
NOT parity with Baidu's published scores (different subset, budget and
preprocessing policy).

## 6. Resource and latency measurements (measured, not extrapolated)

- Model load (cold): as recorded in PR #11 evidence; per-document process
  RSS ~2.9 GiB; system memory 17.6 % used at completion.
- GPU peak allocated 8.54–8.55 GiB; peak reserved 9.4 GiB (identical across
  4- and 5-page documents — vision-prefill dominated).
- Prefill 2.8 s for 4–5 pages; decode dominates wall time
  (140–592 s/document at ≤8192 tokens).

## 7. Blockers for Native 15-page (P2) testing — exact list

1. **8192-token budget**: 4/8 documents in a ≤5-page sample already exhaust
   it. 15 pages need the reviewed budget raise (official 39,321) from the
   P2 opt-in policy — not a P1 change.
2. **5-image cap**: `validate_inputs` rejects >5 pages; P2 ladder requires
   the reviewed `--max-pages` opt-in (design already in
   `LONG_DOCUMENT_READINESS.md` §7).
3. Wall-time: 5 pages ≈ 9 min at 8192 tokens; 15 pages at 39,321 tokens
   projects >45 min/document (measured basis: 2.8 s prefill + decode
   throughput observed here) — supervision/timeout policy must be in place
   first.

## 8. Status summary

| Phase | Status |
| --- | --- |
| P0 integration closure | PASS (integration tree `4cd5945`; no runtime change; full CPU gates) |
| Ground truth on Linux | PASS (both files byte-exact; annotation digests verified) |
| Local official scorer | PASS (both profiles, both modes; real CDM rendering) |
| Real-document runner | PASS (limits unchanged; failure accounting proven on real data) |
| Real sample end-to-end | COMPLETED, REAL_DATA_INTEGRATION_ONLY (7/8 metrics valid; T-TableTEDS not exercised; Overall null by rule) |

Raw evidence retained locally under `.b/mpdoc-runs/sample-1/` and
`.b/scorer-runs/` (untracked); sanitized receipts committed under
`eval/results/Linux gfx1151 verified/`. No GT, page images, weights, or
licensed document derivatives enter Git.
