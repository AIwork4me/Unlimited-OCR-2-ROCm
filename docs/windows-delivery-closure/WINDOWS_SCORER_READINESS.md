# Windows Official Scorer Readiness (Phase 4.5)

Classifications: PASS / FAIL / BLOCKED / NOT_RUN / NOT_APPLICABLE.

## The eight-metric contract (from source, `benchmark/scorer.py`)

`METRICS` maps each official name to `(task, metric, aggregate, invert)`:

| Official metric | Task / metric / aggregate | Invert |
| --- | --- | --- |
| TextEdit | text_block / Edit_dist / ALL_page_avg | yes |
| T-TextEdit | merged_text_block / Edit_dist / ALL_page_avg | yes |
| TableTEDS | table / TEDS / all | no |
| T-TableTEDS | merged_table / TEDS / all | no |
| FormulaCDM | display_formula / CDM / all | no |
| ReadOrderEdit | reading_order / Edit_dist / ALL_page_avg | yes |
| FigureF1 | figure / FigureF1 / all | no |
| HeadingTEDS | head / HeadTEDS / all | no |

Values are accepted only as finite numbers in [0, 1] (bools rejected), scaled
to percentages with inversion for the three Edit metrics; anything else is
`None`. **Overall is the arithmetic mean of all eight and is `None` whenever
any metric is `None`** — never an average of survivors, never zero
substitution. Coverage exposes `documents_without_samples` per task; missing
documents are listed, never dropped. Dependencies must be the 75-pin
torch-free lock in a torch-free environment; the official scorer source is
SHA-256-gated file-by-file (no unlocked executables).

## Provenance (frozen)

- Official scorer: `Tongyi-Zhiwen/Qwen-Doc` @ `4e5aee32280febb7b0844237ba087f837e3c8905`
  (85 files, `eval/benchmarks/benchmark-lock.json`); Marathon profile adds
  three hash-pinned patches from `baidu/MarathonDocBench` @ `fa05fccbe82ee6ec628006a1d58bb153954c3ee3`.
- Input format: one Markdown file `<doc_id>.md` per GT document (converter
  implements the official layout-block → Markdown mapping); failed inference
  exports an explicit empty file so the denominator never shrinks.
- GT: hash-pinned JSON per dataset; `cli.py score` refuses GT hash mismatch.

## What is proven where

| Runtime | Status | Evidence |
| --- | --- | --- |
| WSL Ubuntu 22.04 CPU | PASS (PR #9): 8/8 known answers, real CDM F1=1.0, full pipeline | `eval/benchmarks/g05/scorer/` (synthetic; no model quality conclusion) |
| Native Linux gfx1151 CPU | PASS (PR #13): 7 metrics byte-identical to PR #9 + real rendered CDM 100.0 | `eval/results/Linux gfx1151 verified/scorer/local-scorer-closure.json` |
| Hosted Windows CI (PR #33) | PASS **as a 7-metric selftest**: dependency lock installs, source hash gates, seven known-answer metrics + converter/matcher pipeline fixture; FormulaCDM correctly self-reports `BLOCKED / CDM_RENDERER_UNAVAILABLE` with Overall null | `scripts/check_official_scorer.py` (CDM-unavailable is the only tolerated failure); runs 41281751088 / 41281782148 |
| Native Windows eight-metric scorer | **BLOCKED** | below |

Hosted scorer unit checks are **not** proof that a full Windows official
scorer runtime is installed and executable.

## Native Windows blockers (exact)

1. **Tool absence**: `xelatex` (+ `xeCJK`, Source Han Sans SC), ImageMagick
   with working Ghostscript PDF rasterization, `kpsewhich` — probed by
   `cdm_tools`; any miss ⇒ FormulaCDM `CDM_RENDERER_UNAVAILABLE` ⇒ Overall
   null. `latexmlc` (LaTeX-table branch) is absent and unpinned even on Linux.
2. **Code-level refusal**: `worker.py` raises for Marathon scoring on Windows
   (`Official Marathon TEDS timeout requires SIGALRM; score on Linux`).
3. **POSIX assumptions in the official source**: fork-based document budget,
   SIGALRM TEDS guard, POSIX cat/perl redirection (documented; source bytes
   are fetched-and-pinned rather than vendored, so these remain
   documentation-level claims — but the SIGALRM/fork points are corroborated
   by this repo's own code and closures).
4. **No CDM render has ever executed on Windows** in this repo's evidence.

## Architecture options

**Option A — Windows GPU inference + native Windows scorer: NOT_PRACTICAL**
without distorting official semantics. It requires porting/hosting a TeX
stack + Ghostscript + Node/KaTeX on Windows, pinning `latexmlc`, and
neutralizing fork/SIGALRM paths in the *official* scorer — any of which risks
changing what is measured. Not recommended; remains BLOCKED.

**Option B — Windows GPU inference + pinned Linux/WSL CPU scorer:
RECOMMENDED** (smallest solution preserving the official scorer unchanged).
Both Linux closures already prove the scorer stack; predictions are plain
per-document Markdown files, fully portable. See
`WINDOWS_LINUX_SCORER_BRIDGE_DESIGN.md` for the complete contract.
