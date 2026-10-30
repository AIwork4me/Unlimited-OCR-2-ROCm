# P1.5 runner hardening — ledger-integrated supervised execution

Date: 2026-10-30 (Asia/Shanghai). Branch `feat/p15-runner-hardening` (PR #18),
stacked on the PR #16 independent review (PR #17). Addresses review findings
R1 (separated truncation reporting), R2 (ledger integration), R4 (receipt
sanitization), R5 (`.b/` ignore) and R6 (test-environment matrix).

## 1. What changed

`scripts/run_mpdoc_sample.py` now drives the append-only
`benchmark.ledger.Run` interface (new module
`src/unlimited_ocr_2_rocm/benchmark/runner.py`):

- **Immutable run identity** — ledger run id = digest(manifest, config,
  environment); the full frozen 420-document/3,135-page MPDocBench manifest
  is used, so official denominators stay intact and the selection is frozen
  by SHA256 inside the run config (`5cbce32b…`, the unchanged P1 selection).
- **Atomic begin/finish per document** — each attempt is a numbered
  `.pending` directory sealed by `COMMIT.json` hashes; workers cannot touch
  the ledger (only the parent finishes attempts).
- **Explicit accounting** — PASS / FAIL / TRUNCATED / TIMEOUT / OOM /
  NOT_RUN; native `finish_reason=length` maps to TRUNCATED
  (`TOKEN_LIMIT_NATIVE_LENGTH`), never relabeled.
- **Interruption/crash recovery** — abandoned attempts remain visible as
  INTERRUPTED failures; resume appends a new attempt.
- **Validated resume and skip-on-PASS** — `todo()` re-verifies every
  committed artifact hash; PASS documents are skipped.
- **Frozen identities** — GT SHA `0a579dca…`, page-index gate (D0 frozen
  index) re-checked inside every worker before inference, dataset-root
  containment via `manifest.local_path`.
- **No GT leakage** — workers receive page image paths only.
- **Private raw / sanitized exports** — raw bytes and worker logs stay in
  the untracked ledger run under `.b/`; committed receipts drop
  executable paths, hostnames, machine IDs, GPU UUIDs and any local path.

## 2. Failure-injection validation (CPU, real subprocesses)

12 tests (`tests/unit/test_p15_runner_hardening.py`) drive the actual
supervision path — each scenario spawns a genuine worker process that is
hung (SIGKILL after external timeout), crashed (`SIGKILL` mid-run),
OOM-killed (raises a `torch.OutOfMemoryError`-shaped traceback),
length-truncated, structurally malformed (stop without page structure — the
ledger refuses PASS), or interrupted (abandoned `.pending`):

| Injected failure | Recorded state |
| --- | --- |
| native `length` at budget | TRUNCATED `TOKEN_LIMIT_NATIVE_LENGTH`, raw preserved |
| worker raises OOM traceback | OOM `WORKER_OUT_OF_MEMORY` |
| worker hangs past timeout | TIMEOUT `WORKER_TIMEOUT`, process-group killed, log sealed |
| worker SIGKILLs itself | FAIL `WORKER_CRASH_EXIT_<n>`, retry appends attempt 2 → PASS |
| stop with no `<PAGE>` structure | FAIL `EMPTY_MALFORMED_OR_PAGE_MISMATCH` (ledger refusal) |
| abandoned begin() | FAIL `INTERRUPTED`, next run recovers |
| good document twice | second invocation executes nothing (skip-on-PASS) |
| unselected manifest documents | NOT_RUN in the same summary |

Also validated: selection outside the manifest is rejected; GT slice
restricted to selection with original annotation bytes; receipts contain no
`/home/` path, hostname or machine id.

A real misconfiguration became live evidence: the first P2-A worker
invocation rejected an argument (exit 2) and all four documents were
honestly committed as `WORKER_CRASH_EXIT_2` (ledger run `3031002c…`,
attempts `000001`, kept unchanged) before the retry succeeded as `000002`.

## 3. Real 8-document re-run under the ledger (GPU)

Ledger run `bf55b5099f2cd8e334a8793b4c77c7a85273b873ca08a0fb195e81108c10e677`
(full manifest; 8-document selection; limits unchanged: 5 pages, 8192
tokens, 16 Mpx/page; supervision timeout 3600 s per document). Executed as
two invocations to demonstrate resume on real hardware: invocation A ran 2
documents, invocation B ran the remaining 6 and skipped the 2 PASS
documents (no new attempts — resume proven on GPU).

| Document | State | finish | tokens | raw `<PAGE>` |
| --- | --- | --- | --- | --- |
| examPaper_eng_66b20269 | PASS | stop | 2,894 (= P1) | 4/4 |
| researchReport_chn_dd0407a2 | PASS | stop | 4,829 (= P1) | 4/4 |
| goverment_eng_aa75e353 | PASS | stop | 3,143 (= P1) | 4/4 |
| researchReport_chn_a2f46550 | PASS | stop | 7,631 (P1: 7,603) | 5/5 |
| handwritten_chn_1ae487be | TRUNCATED | length | 8,192 | 4/5 (P1: 4/5) |
| newspaper_chn_eb87830d | TRUNCATED | length | 8,192 | 1/5 (P1: 1/5) |
| newspaper_eng_ed12985c | TRUNCATED | length | 8,192 | 2/4 (P1: 2/4) |
| paper_chn_3567e55d | TRUNCATED | length | 8,192 | 5/5 (P1: 5/5) |
| remaining 412 manifest documents | NOT_RUN | — | — | — |

States, token counts and page-marker counts match PR #16 exactly — the
accounting changed (TRUNCATED is now a first-class state, NOT_RUN covers
the unselected manifest), the physics did not. (a2f46550's 28-token delta
vs P1 is the §3 noise floor.)

**Determinism / noise floor.** examPaper reproduced byte-identically
(`3ecbdee5…`). Other documents match P1 in state and token count but differ
in raw bytes by isolated single-character flips (goverment_eng: 30 hunks in
13,088 chars, 99.69 % identical; same finish/token counts). This is
kernel-level floating-point non-associativity under greedy decoding — the
inherent run-to-run noise floor any cross-run comparison on this stack must
quote.

## 4. Report 1 — strict failure-inclusive accounting (mandated)

Official qwen scorer over the **strict** export: PASS documents contribute
their ledger-sealed Markdown; every TRUNCATED document contributes an
**explicit empty file** (no partial content):

| Metric | Strict (4 complete + 4 empty) |
| --- | --- |
| TextEdit | 49.60 |
| T-TextEdit | 25.00 |
| TableTEDS | 71.84 |
| T-TableTEDS | null (0 merged-table samples in this subset) |
| FormulaCDM | 4.96 |
| ReadOrderEdit | 48.06 |
| FigureF1 | 38.10 |
| HeadingTEDS | 38.54 |
| **Overall** | **null by rule** |

The four truncated documents count as total misses — this is the
failure-inclusive denominator required for honest benchmark accounting.

## 5. Report 2 — truncation-content diagnostic scoring (mandated)

Same ledger run, **diagnostic** export (TRUNCATED attempts contribute their
converted partial Markdown — the PR #16 semantics, now explicit and
labeled). Previous scores are NOT changed; PR #16 numbers remain in their
own files:

| Metric | Diagnostic (P1.5) | P1 original (PR #16) |
| --- | --- | --- |
| TextEdit | 76.57 | 76.89 |
| T-TextEdit | 58.98 | 57.28 |
| TableTEDS | 82.68 | 82.68 (identical) |
| T-TableTEDS | null | null |
| FormulaCDM | 90.76 | 90.94 |
| ReadOrderEdit | 73.75 | 75.07 |
| FigureF1 | 71.43 | 78.23 |
| HeadingTEDS | 51.25 | 53.70 |
| **Overall** | **null** | **null** |

Interpretation: TableTEDS is byte-identical (its samples come from
documents that completed); the other deltas are the §3 noise floor
expressed through partial content of truncated documents, plus FigureF1's
small sample population. Nothing here re-labels the truncated documents.

## 6. Test-environment matrix (finding R6)

| Environment | Role | Gates |
| --- | --- | --- |
| system CPython 3.12 + `PYTHONPATH=.` | CPU unit suite | 291 passed / 7 skipped (torch/pyarrow tests skip) |
| `envs/linux-gfx1151-rocm` (CPython 3.10) | production GPU runs | editable install of this repo; no pytest installed |
| `scorer-build/venv` (CPython 3.10, torch-free) | official scorer | 73-package lock, pip check PASS |
| `uvx mypy@1.18.1` | type gate | 18 files clean |

## 7. Status

| Item | Status |
| --- | --- |
| Ledger integration + supervision | PASS (12/12 injection tests; real GPU run A/B resume evidence) |
| Strict failure-inclusive report | DONE (this file §4) |
| Truncation-content diagnostic report | DONE (this file §5) |
| Receipt sanitization + `.b/` ignore | DONE (committed receipts carry no host IDs; history untouched) |
| P1/PR #16 historical evidence | unchanged byte-for-byte |

Raw evidence: ledger runs under `.b/mpdoc-runs/` and scorer runs under
`.b/scorer-runs/{p15-strict,p15-diagnostic}` (untracked). Sanitized
receipts: `eval/results/Linux gfx1151 verified/mpdoc-sample-p15/`.
