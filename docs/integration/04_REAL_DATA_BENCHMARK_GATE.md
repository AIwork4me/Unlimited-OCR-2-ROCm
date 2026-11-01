# 04 — Real-Document Benchmark Integration Gate (Phase 4)

- Date: 2026-11-01, Asia/Shanghai
- Sequence merged: #16 (`1ac3b4d`) → #17 (`bf89852`) → #18 (`88d913d`) → #19 (`89caa30`); every step verified by local trial-merge tree equality + publication gate PASS + full CPU suite (345 passed / 7 skipped at the final tree).
- Final Phase-4 main: `89caa30` (tree `15e040f7`).

## 4.1 PR #16 — real data chain (Dataset → GPU → Raw → Converter → Scorer)

Verified against the committed evidence (Phase 1 audit + this phase's spot checks):

- 8-document / 36-page predeclared subset: `predeclared-selection.json` carries per-document structural reasons and page counts; the selection is the frozen identity the receipts reference.
- Raw outcome preservation: run-receipt `documents_completed: 4 / 8` with the four failed documents named (`handwritten_chn_1ae487be`, `newspaper_chn_eb87830d`, `newspaper_eng_ed12985c`, `paper_chn_3567e55d`) — the 4 PASS / 4 length-truncated split is NOT modified anywhere downstream (P1.5 strict/diagnostic split in #18 reproduces exactly these states: "4 PASS + 4 TRUNCATED + 412 NOT_RUN = P1 states exactly").
- FormulaCDM real renders: scorer closure evidence includes the rendered CDM artifacts (102 real renders verified by the independent review R-audit; render receipts are in the #9/#16 evidence tree with hash chains).
- T-TableTEDS = `null` and `overall = null` preserved in `score-scores.json` (`qualification: "SCORER_ONLY; review coverage and failure catalog"`) — no-sample metrics are null by rule, not fabricated.
- Claim hygiene: `label: REAL_DATA_INTEGRATION_ONLY`; `claim_limit` explicitly states "NOT full MPDocBench accuracy, NOT leaderboard parity, NOT full dataset completion". The diagnostic subset is never presented as an official leaderboard result.

## 4.2 PR #17 — independent review R1–R7 closed by successor evidence (not by assertion)

| Finding | Closure evidence (verified on the merged tree) |
|----|----|
| R1 (P1): truncated partial Markdown inside the scored subset without separation | #18 strict/diagnostic export split — strict report has TRUNCATED docs as empty files, diagnostic ≈ P1 values; both coverage files committed |
| R2 (P1): runner not ledger-integrated | #18 `src/unlimited_ocr_2_rocm/benchmark/runner.py` + ledger-integrated `scripts/run_mpdoc_sample.py`; 14 real-subprocess injection tests pass locally in this session |
| R3 (P2): 8192-budget length failures on dense docs | #20 P2-A budget experiments (Phase 5) — the RCA receipts quantify that budget is not the binding constraint for the two newspapers |
| R4 (P2): receipt carried private path + hostname + hashed machine-id | #18 sanitization (`test_receipt_has_no_private_paths_or_host_ids`, `test_sanitize_tree_refuses_local_paths_and_drops_ids` PASS); later receipts sanitized; historical files hash-pinned in the gate whitelist |
| R5 (P3): `.b/` not gitignored | #18 adds `.b/` to `.gitignore` (verified present and `git check-ignore` effective on the merged tree) |
| R6 (P3): implicit test-environment matrix | #18 `docs/validation/LINUX_P15_RUNNER_HARDENING.md` records the CPU/GPU venv split (§6) |
| R7 (P3): coverage must be quoted beside values | #18 coverage files (`score-diagnostic-coverage.json`, `score-strict-coverage.json`) adopted as the reporting pattern |

## 4.3 PR #18 — failure injection executed for real (not mocked)

Local execution in this session (tree 88d913d), all PASS via REAL subprocesses:

`test_good_document_passes_and_resume_skips` (atomic begin/finish + resume + skip-on-PASS), `test_native_length_becomes_truncated_with_preserved_raw`, `test_worker_oom_is_classified_oom`, `test_hanging_worker_times_out_and_is_killed`, `test_killed_worker_crash_is_fail_then_retry_passes` (SIGKILL), `test_stop_without_page_structure_is_refused`, `test_abandoned_attempt_is_interrupted_failure_then_recovered`, `test_unselected_documents_are_not_run`, `test_selection_outside_manifest_is_rejected`, `test_gt_slice_is_limited_to_selection_with_original_bytes`, `test_receipt_has_no_private_paths_or_host_ids`, `test_sanitize_tree_refuses_local_paths_and_drops_ids`, `test_termination_kills_worker_descendants_not_only_the_worker` (process-group kill), `test_termination_tolerates_worker_that_exited_before_the_kill`.

Ledger seal + strict/diagnostic isolation verified by the strict/diagnostic export receipts committed in the PR; the real-GPU hardened re-run (`bf55b509`, 2 invocations, resume/skip-on-PASS proven) is part of the committed evidence.

## 4.4 PR #19 — T-TableTEDS provenance

- Predeclared selection `handwritten_chn_375d58b7` (2 pages, genuine cross-page merged-table pair) committed at **10:48** ("before inference"), inference/scoring commit at **11:56** — temporal order verified on the branch history.
- Receipt identity: `run_id e438812a…`, `selection_sha256 0ce25daf…`, `status PARTIAL`, `label REAL_DATA_INTEGRATION_ONLY`.
- Official scorer on the merged-table sample: **T-TableTEDS 74.986928… (=74.99)**, TableTEDS 74.99, HeadingTEDS 100.0; FigureF1/FormulaCDM/ReadOrderEdit/T-TextEdit/TextEdit `null` (no samples — null by rule); `overall: null`.
- The eight-document experiment's Overall=null is untouched (separate receipts; nothing re-aggregated across different sample populations).

## Verdict

Phase 4 GATE: **PASS** — all four PRs merged with evidence-level verification; no claim inflation introduced; the official eight-metric paths now have real execution evidence (TextEdit, TableTEDS, T-TableTEDS, HeadingTEDS, FormulaCDM, FigureF1, ReadOrderEdit, T-TextEdit) without any cross-population Overall fabrication.

KEEP EXPERIMENTAL.
