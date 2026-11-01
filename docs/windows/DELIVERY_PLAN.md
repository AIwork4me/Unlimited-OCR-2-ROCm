# Windows product delivery plan

**Goal:** Make the existing Windows Native scoped-MATH stack installable and
supervisable from current main, verify actual datasets and the authorized
synthetic regression, and deliver honest full-benchmark readiness/results.

**Architecture:** Reuse the common benchmark ledger, scorer and manifest schemas.
Change only Windows supervision/integration and installation; retain native
whole-document inference and production generation limits. Full campaign fields
remain null until separate run authorization and all admission gates exist.

**Tech stack:** CPython 3.10, PowerShell, Windows Job Objects, existing pinned
AMD ROCm 7.14.1 stack, shared Python ledger and CPU scorer interfaces.

**Spec:** User Windows Radeon Product Delivery Mission, received 2026-11-01;
private exact copy `.migration/windows-product-20261101/MISSION.md`.

## Global constraints

- Base main `52d593e0686176b3842d78fbac3ef76956ecf1ae`; isolated branch
  `delivery/windows-native-benchmark`, retained under `.worktrees/wd`.
- Preserve original dirty checkout, historical worktrees/evidence and environments.
- torch2.12.0+rocm7.14.1, torchvision0.27.0+rocm7.14.1,
  torchaudio2.11.0+rocm7.14.1, triton-windows3.7.1.post27,
  transformers4.52.1, Windows Radeon8060S/gfx1151; no version drift.
- Original official torch wheel415d2dbd…, not candidate6a67d158…; scoped MATH,
  BF16, DASH/SR-SWA and original weights. No AUTO or QuickGELU activation.
- All command cwd is the primary repository root. New project caches/data/venvs
  stay under it. Native app worktree API cannot choose the required destination;
  explicit user directory preference therefore uses Git worktree creation.
- No README/Linux model/scorer/result edits, global PATH/security/driver/SDK
  changes, merge, release or PyPI publication.
- Existing smoke/regression only: at most two complete 13-case synthetic suites
  (26 full OCR requests), one per new clean venv, conditional on earlier gates.
  Any failed correctness/identity/resource gate stops dependent GPU work; no retry.
- No real-document/15+page/39321-token/full-dataset GPU campaign without a frozen
  independent review and explicit run authorization. No fabricated scores.

## Review focus

1. Detached Windows descendants after normal exit, timeout, interruption or crash.
2. Publication scanning inside a `.b` ancestor must inspect the selected subtree.
3. Same-version candidate wheel reuse must not masquerade as a clean original install.
4. Resume must verify exact identities and sealed PASS; failures retain denominators.
5. Null metrics, official comparability, UMA memory and Linux scoring remain explicit.

## Tasks and checks

### Task 1 — Baseline and evidence audit

- [x] Snapshot original42/index; fetch main; create isolated retained worktree.
- [x] Query all requested PRs; distinguish merged main from Draft evidence.
- [x] Run current-main CPU baseline; retain 349 PASS/8 skipped/1 failure.
- [x] Fix reproduced relative-root publication scan with a regression test;
  verify source/fixture whitelist behavior and full CPU suite.
- [x] Rehash/decode all420/3135 and100/18272 frozen dataset pages on Windows;
  preserve complete receipts, missing/extra/duplicate/error information.

### Task 2 — Windows supervision

Files: package benchmark Windows Job helper, existing benchmark/runner.py,
Windows-focused fake-worker tests. Reuse the previously reviewed MIT Job
implementation with provenance, package it normally and retain Linux behavior.

- [x] RED tests: child outlives exited parent; timeout/interruption drains tree;
  unknown/unreadable outcomes never PASS. Tests use real CPU fake workers only.
- [x] Add suspended-before-assignment Job supervision; cleanup is mandatory on
  every exit, and write accounting before deleting only owned scratch files.
- [x] Keep staging under the attempt directory and preserve outcome/failure logs.
  Add triton-windows identity to the existing environment receipt.
- [x] Run covering real-process tests, then common ledger/resume/failure tests.

### Task 3 — Clean installation and known regression

Files: existing Windows setup/reproduction scripts plus a tested small verifier,
an original-wheel lock and Windows-only installation tests/docs. No README edit.

- [x] Add optional verified wheelhouse/offline acquisition; retain online mode.
  Explicit CPU preflight, fresh-venv safety and scoped cache/temp settings.
- [x] Validate all nine original runtime archive hashes and maximum member paths;
  current measured longest `.b/w1`/`.b/w2` installation path is219 (<260).
- [x] Build a complete pinned wheelhouse without modifying old environments.
- [x] Fresh local clone at the exact new commit; fresh `.b/w1` then `.b/w2`.
  Record interpreter/package/archive/source/loaded model/hardware identities.
- [x] Independently review the concrete bounded26-request policy before launch.
  Each owned suite: finite parent/case deadlines, host available >=64GiB before
  load, stop <48GiB, observed allocator reserved >16GiB, no retry after failure.
- [x] Execute only the existing13 cases and independent BF16/Triton/DASH/MATH
  checks; retain raw output/GT privately and all five correctness classifications.

### Task 4 — Product-facing integration and blocked campaign receipts

- [x] Windows PowerShell benchmark entry point delegates to shared ledger/runner;
  no second scorer or hidden long-document cap override.
- [x] Schema adapter exports the required Windows fields with explicit mapping
  to shared schema, full420/100 and3135/18272 denominators and null unrun scores.
- [x] Verify official reference source/revision and protocol comparability;
  audit Windows scorer tools; keep Linux-scoring handoff explicit where needed.
- [x] Write Windows readiness/install/baseline/dataset/protocol/full-result/
  SGLang/handoff docs and Chinese WINDOWS_PRODUCT_DELIVERY_REPORT.md.
- [x] Produce REAL_DOCUMENT_ADMISSION.md with a single <=5-page/two-arm scope
  proposal, fixed resource ceilings and failure handling. Exact executable identity
  and authorization remain BLOCKED on reviewed common policy/scorer integration;
  do not ask for approval of an incomplete identity or launch it.

### Task 5 — Validation and independent handoff

- [x] Ruff, format, mypy, CPU suite, build, pip check, source archive tests,
  tracked/staged publication gates and scope/privacy checks at actual final head.
- [x] Fresh independent branch review; correct material findings with tests.
- [x] Recheck original42/index/historical seals and existing Draft refs.
- [x] Commit small reviewable changes locally; prepare PR description/patches.
  New mission authorizes review-ready preparation, not the prior mission's
  specific remote publication grant for a different pair of branches.

No synthetic regression, dataset readiness, scorer fixture or Linux observation
is promoted to a full Windows official benchmark score.

Final archive receipt is produced separately after the documentation commit;
its exact artifact hashes and extracted-source test output are retained privately.
A failed final packaging check must be corrected before handoff.
