# PR #16 independent review — correctness closure

Reviewer session: 2026-10-30 (Asia/Shanghai). Branch under review:
`feat/p1-real-data-integration` @ `7347121` (PR #16, stacked on PR #13 @
`0ec78b0`). Review performed directly on the working tree at that commit;
every claim below was re-derived from the committed artifacts and the raw
local evidence under `.b/` (untracked), not from the PR description.

Verdict up front: **PR #16's evidence is accurate as recorded — no claim in
`docs/validation/LINUX_P1_REAL_DATA_INTEGRATION.md` or in the committed
receipts contradicts the raw artifacts.** Two presentation gaps (truncated
Markdown inside the scored subset; private identifiers inside a committed
receipt) and several runner-reliability gaps are documented below; none
invalidates the recorded numbers. The runner gaps are addressed by the P1.5
hardening PR that follows this review; the historical files are intentionally
left byte-identical.

## 1. Status of every completed P1 gate — verified

| Gate (report §8) | Independently verified evidence | Verdict |
| --- | --- | --- |
| P0 integration closure | merge commit object `4cd5945a…` exists in this clone; PR stack described matches branch topology | PASS as recorded |
| Linux ground truth | `../gt/MPDocBench.json` = `0a579dca…8f40df`, `../gt/MarathonDocBench.json` = `7db0e659…adf51c`, byte counts match §2 of the report | PASS |
| Local official scorer | `eval/results/Linux gfx1151 verified/scorer/local-scorer-closure.json` records both profiles × pipeline/cases; raw per-task outputs present under `.b/scorer-runs/{qwen,marathon}-*` | PASS |
| Real-document runner | `run-receipt.json`: 8 selected, 8 attempted, 4 `stop`/`INFERENCE_ONLY`, 4 `length`/`FAIL` — matches report §5 table exactly | PASS |
| End-to-end sample scoring | `score-selftest.json` PASS gate → `score-scores.json` metrics; coverage recorded | COMPLETED, REAL_DATA_INTEGRATION_ONLY as recorded |
| CPU suite on this commit | re-run now: `PYTHONPATH=. pytest tests/ -q` → **274 passed / 7 skipped** (system CPython 3.12; torch/pyarrow-dependent tests skip; the GPU venv contains no pytest — see finding R6) | consistent with the recorded 270/5 + 34/… split across environments |

The four `length` documents keep native `finish_reason=length` in the receipt
and are counted in `failed_documents`; the runner never relabels them. The
recorded per-document error (`RuntimeError: Empty or native length-truncated
OCR output`) is the *last traceback line* of the enforced failure, and the
raw text for all four is preserved locally under `.b/mpdoc-runs/sample-1/`.

## 2. Scorer known-answer coverage (8/8) vs real subset metric coverage (7/8)

These are two different denominators and the review confirms both numbers:

- **8/8 = metric known-answer coverage.** `score-selftest.json` exercises
  the official scorer on authored known-answer fixtures for **all eight**
  metrics — including **T-TableTEDS = 50.0 (expected 50, PASS)** on the
  merged_table fixture. The scorer implementation is therefore *proven
  capable* of computing every official metric on this host.
- **7/8 = real-data metric coverage.** On the 8 real documents,
  `score-coverage.json` records `merged_table: 0 samples` (all 8 documents
  listed under `documents_without_samples`), so the official scorer returned
  no valid T-TableTEDS population and `metric_values` records `null`. Per the
  documented rule (any invalid metric ⇒ invalid Overall), **Overall = null**
; no value was invented.

Per-task real coverage is additionally uneven and is honestly recorded in
`score-coverage.json`: `figure` spans 7/8 documents, `merged_text_block`
4/8, `display_formula` 3/8, `table` 5/8, while `text_block`, `head` and
`reading_order` span 8/8. The seven reported metric values are the official
scorer's aggregates over exactly these sample populations.

## 3. Semantics of truncated Markdown inside the scored subset

Confirmed from code and raw bytes; this is the review's main *presentation*
finding (the numbers themselves are correctly labelled subset diagnostics):

- `scripts/run_mpdoc_sample.py:113-117` converts `raw.txt` and writes
  `predictions/<doc>.md` **unconditionally whenever raw text exists** —
  including all four `length`-FAIL documents. The scorer then requires the
  prediction inventory to equal the ground-truth-slice inventory (8 files),
  so all four partial documents were scored.
- Truncation states differ per document: `paper_chn_3567e55d` emitted all
  5 `<PAGE>` markers and died **inside a block header**
  (`…<BLOCK>page_number[859, 921` — coordinates cut mid-number);
  `handwritten_chn_1ae487be` emitted only **4 of 5** `<PAGE>` markers.
- The upstream converter does not sanitize partially-emitted structure:
  `predictions/paper_chn_3567e55d.md` ends with the leaked fragment
  `[859, 921` inside the scored Markdown. Text/Table/Heading metrics for
  truncated documents therefore measure **partial content plus spurious
  tail fragments**; e.g. 96 of the 102 formula samples (88 handwritten +
  8 paper) come from truncated documents.

Consequence: the seven reported values are **failure-inclusive subset
diagnostics** (4 complete + 4 partial documents), not comparable to any
full-document benchmark. The report's labels (REAL_DATA_INTEGRATION_ONLY)
already forbid such comparison, but it does not spell out that truncated
partial Markdown is *inside* the scored population. The P1.5 follow-up
provides the two mandated separated reports (strict failure-inclusive
accounting; truncation-content diagnostic scoring) without altering the
recorded numbers.

## 4. The 102 formula samples — verified against renderer/matcher outcomes

- `official/display_formula/CDM/{gt,pred}/bbox/` each contain **102**
  per-index `.jsonl` files; `vis_match/` contains **204** PNGs (base+match
  per sample). All 102 pairs went through real xelatex→ImageMagick
  rendering plus Hungarian/RANSAC matching.
- The per-sample file `display_formula_per_sample_CDM.json` holds 102
  F1 values: min 0.229, max 1.0 (52 exact matches), mean
  0.9094019607843137 — exactly the committed `FormulaCDM 90.94`.
- Cross-check against the scorer's failure semantics
  (`metrics/cdm_metric.py:199-206`): the bare `except` returns F1=0 on any
  render/match crash. **No sample has value 0.0**, and `cal_metric.py`
  additionally retries parallel crashes sequentially before ever recording
  a 0 — so no swallowed renderer failure is hiding inside the mean.

The "102 real renders" statement is therefore accurate.

## 5. `run-receipt.json` privacy audit

Scan of all five committed JSONs under
`eval/results/Linux gfx1151 verified/mpdoc-sample/`:

| Check | Result |
| --- | --- |
| Private absolute paths | **1 occurrence** in `run-receipt.json`: `model.environment.executable` = `/home/amd/Desktop/UnlimitedOCR/envs/linux-gfx1151-rocm/bin/python` (local layout disclosure; no credential) |
| Host identifiers | `host.node` = `amd-HP-ZBook-Ultra` (hostname); `model.environment.linux_hardware.machine_id_sha256` (hashed machine-id — a stable host fingerprint); `render_node`, `product_name`, CPU string (hardware inventory, arguably legitimate environment evidence); GPU uuid/pci fields are `null` on this APU |
| Raw payload leaks | None — zero `<PAGE>/<BLOCK>/<FORK>` marker hits in any committed JSON; predictions exist only as SHA256 digests (`prediction-digests.json`); GT content, page images and weights are not in Git |
| Raw local artifacts | `.b/` is untracked; `.gitignore` does **not** list it (finding R5) |

Per the no-history-rewrite rule the committed file is left unchanged; the
P1.5 runner sanitizes environment/executable/host fields in the receipts it
produces going forward, and this review records the exposure explicitly.

## 6. Historical evidence preserved

All pre-existing files under `eval/results/`, `docs/validation/` and the PR
#16 artifacts remain byte-identical in and above this review; this review
and the follow-up PRs only **add** files. The four `length` documents are
the frozen P2-A set precisely because their original evidence stays
untouched (original 8192 run is reused as the P2-A baseline).

## 7. Prioritized findings

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| R1 | P1 | Truncated partial Markdown is inside the scored subset without a separated diagnostic report (§3 above) | P1.5: two mandated reports, no score changes |
| R2 | P1 | Runner is not ledger-integrated: no resume/skip-on-PASS, no per-document atomic attempts, no TIMEOUT/OOM classification, crash mid-document leaves only an `INTERRUPTED` marker (or nothing on SIGKILL), no dataset-root containment on GT-derived paths | P1.5 hardening PR |
| R3 | P2 | 8192-token budget fails 4/8 dense ≤5-page documents (native `length`, page 5 never reached in one case) — blocks P2 ladder | P2-A controlled budget experiments |
| R4 | P2 | Committed receipt carries private venv path + hostname + hashed machine-id (§5) | sanitize future receipts; history left intact |
| R5 | P3 | `.b/` (raw evidence) is not in `.gitignore`; an accidental `git add .` would commit licensed dataset derivatives | add explicit `.b/` ignore entry in P1.5 |
| R6 | P3 | The GPU venv has no pytest; CPU gates run under system CPython 3.12 while production venv is 3.10 — the test-environment matrix is implicit | record in P1.5 report; no code change |
| R7 | P3 | `score-coverage.json` shows per-metric document coverage varies (3/8–8/8); any future subset comparison must quote coverage alongside values | practice adopted by P1.5 reports |

## 8. Explicit non-findings (checked, no issue)

- Ground-truth leakage into prompts: the runner passes only page-image
  paths to `run_native`; GT JSON is read solely for page lists and the
  scorer slice.
- Selection integrity: `predeclared-selection.json` SHA256 `5cbce32b…`
  matches the report; selection reasons and per-document category counts
  are frozen in the file.
- Page-hash gate: every selected page was re-verified against the frozen
  D0 index before inference (runner raises on first mismatch; the run
  completing proves all 36 pages passed).
- Model/weights: `model.evidence` records revision `9cf077c4…`, the
  verified lock hashes, and BF16-on-GPU parameter checks; no weight or
  default MATH change is part of PR #16.
