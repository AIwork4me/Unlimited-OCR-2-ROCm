# 06 — L1.1-B Physical Qualification Report (Phase 6)

- Date: 2026-11-01, Asia/Shanghai
- Document: `paper_eng_64b5e1f4` (15 pages, 7 isolated equations, 7 tables, 288 layout dets — unique maximum among the 8 exact-15-page candidates, predeclared)
- ONE attempt, launched 10:45, completed ~11:34 (attempt_wall 2,912.8 s), ledger run id `e9dde491803e1a853b2b829f1d760ff79ff2cd4c24c40a13ffc5d901e0dbdfe5`
- Run tree: `feat/l1-fifteen-page-native @ 39c3813` (= reviewed migration head `3993fc4` + the additive policy-authorization commit; migration history untouched)
- Final policy digest `d95e5bd0…` (draft `64bae56a…` unchanged per C1; delta = exactly the two pre-declared receipt-reference fields); selection receipt `8d80ae7b…` byte-identical to the reviewed packet

## Admission conditions C1–C7 (re-verified immediately before launch)

| Condition | Evidence | Status |
|----|----|----|
| C1 digests unchanged | draft policy `64bae56a…` + receipt `8d80ae7b…` re-hashed at launch; finalization delta recorded in the gate comment | PASS |
| C2 worker verify_page_gate | hard precondition in `worker_command` before any inference (raises on any page-hash mismatch; silent pass = recorded by completion); 15/15 pages verified against the frozen page index | PASS |
| C3 collision observer armed | receipt `collision-observer-events.json`: `aborted: false`, `collision: null`, `anomalies: 0`, guard/ngram/normal steps all 24,051 | PASS |
| C4 watchdogs active | heartbeat receipts: 1,441–1,450 samples, GPU hard limit 16 GiB (`exit 86` armed), watchdog-sampled peaks 8.28/8.49 GiB (heartbeat/metadata snapshots) and native-report `cuda.memory_allocated` peak 8.617 GiB; RSS soft limit 4 GiB, peak 2.696 GiB — all ≤ limits | PASS |
| C5 single attempt, nothing concurrent | exactly one attempt dir (`000001`); GPU baseline 1.5 GiB co-tenancy only (unchanged since every prior gate); no other workload launched | PASS |
| C6 environment pin | prepared `environment.json` capture: torch 2.12.0+rocm7.2, triton-rocm 3.7.0, transformers 4.52.1, python 3.10.20 — exact match | PASS |
| C7 owner authorization | granted by the mission brief §6.2 for EXACTLY ONE controlled run of case B; no other experiment/retry/release authorized | PASS |

## Execution integrity

- Frozen model/weights/prompt/crop unchanged; native BF16 + DASH (Triton block-sparse active during prefill) + vision SDPA MATH; n-gram 35/1024 output-only from the frozen policy; budget 39,321; single request.
- Supervision: driver-managed worker subprocess, 14,400 s timeout (used 2,913 s), process-group termination armed, `terminate_incomplete_is_failure` in force.

## Results by the five mission classes

1. **Execution Integrity: PASS** — attempt committed atomically (`COMMIT.json` hash manifest), finish_reason `stop` (natural, not forced; terminal_empty_marker False), raw sha256 recorded.
2. **Page/Structure Completeness: PASS** — 15/15 raw page markers; `raw_page_markers: 15`; replay confirms page-guard semantics over all 24,051 steps.
3. **Content Correctness (official scorer, qwen profile, pristine source tree, frozen GT slice `cd97423a…`)** — TextEdit **99.69**, T-TextEdit **99.98**, TableTEDS **87.57** (7/7 tables scored), FormulaCDM **99.21** (7/7 equations rendered — real CDM path), HeadingTEDS **99.15**, ReadOrderEdit **100.0**, FigureF1 **100.0**, **T-TableTEDS null** (no cross-page merged tables in this document — null by rule, not fabricated), **overall null by rule** (no cross-population aggregation). Strict and diagnostic exports are identical because the document completed naturally (no truncation to isolate).
4. **Resource/Timeout Safety: PASS** — watchdog-sampled GPU peaks 8.28/8.49 GiB and native-report CUDA peak 8.617 GiB (≤16 GiB hard limit), RSS peak 2.696/4 GiB, wall 2,913/14,400 s, zero watchdog breaches, zero supervision failures.
5. **Official Scorer Coverage: 8/8 metrics executed** — 7 real numeric metrics + T-TableTEDS null-by-rule; coverage files sealed.

## Gates verdict (driver `gates`, replay hard gate included)

`gates-report.json`: `promotion_admitted: true`, `failures: []`, counts `1 PASS + 419 NOT_RUN` (the driver's one-line stdout summary prints the derived form `{"l1_gates": "PASS", "promotion_admitted": true, "failures": []}`); NOT_RUN never PASS; replay: 24,050 steps, first ban step 14,473, **forced mismatches 0**; observer counters agree exactly with the replay.

## Claim limits (unchanged)

This ONE PASS does NOT establish: Native/SGLang parity, 15-page production readiness, official full-dataset benchmark parity, 30/50/100-page support, or any performance claim. `finish_reason=stop` is not, by itself, semantic-completeness proof — the official metrics above are the content evidence, scoped to this one document.

## Case A status

Case A (newspaper-class 15-page) remains BLOCKED (no qualifying candidate exists — objective dataset fact). Any substitute case requires a fresh gate review AFTER this B evidence is sealed, per the frozen policy; this mission does not authorize it.

## Evidence locations

- Sealed sanitized receipts (0700): `<MISSION_ROOT>/l1-migration/l11b-score-out/` (SHA256SUMS.txt, 9 files, private-path scanned CLEAN)
- Raw attempt evidence (untracked, 0700): `l11b-runs/MPDocBench/e9dde491…/` + `l1-exp/.b/l11b-score/`
- Pre-launch seal: `l11b-run/PRE_LAUNCH_SEALED.sha256`; launch receipt: `l1-migration/launch/launch-receipt-l11b.txt`
