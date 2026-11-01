# MAINLINE INTEGRATION FINAL REPORT — Unlimited-OCR-2-ROCm

- Mission window: 2026-11-01 (Asia/Shanghai), single session
- Final main: **`52d593e0686176b3842d78fbac3ef76956ecf1ae`** ("Merge pull request #23"), tree `831051ad18a462660ea6ae49069aa3f08c99cce6`
- Baseline at freeze: `beb38d19` (Phase 0 doc: `docs/integration/00_BASELINE_AND_PR_GRAPH.md`, local commit f22f9e7)
- Final status: **KEEP EXPERIMENTAL** — no Stable Release, no full-benchmark-parity claim, no production default changed.

## A. PR Merge Ledger

Every merge = GitHub **Merge Commit**; each was preceded by a local trial merge whose tree SHA had to equal the post-merge remote tree, plus publication-gate scan and full CPU suite on the trial tree. Base retargets used the REST API (`PATCH /pulls/N`) with explicit base verification (after the #8 incident below).

| PR | Original head | Original base | Merge commit | Review basis | CI at merge | Notes |
|----|----|----|----|----|----|----|
| #30 (new, security) | 97f8ae2 | main | `69dc21a` | 2-round independent review GO/GO | 8/8 | Publication gate (Phase 1) |
| #7 | d842ce6f | main | `5e11cdd` | Phase-1 audit + local trial | 8/8 | benchmark protocol foundation |
| #8 | afbafac9 | codex/benchmark-protocol | — | — | — | **INCIDENT**: merged into stale base (`532d59d` on codex/benchmark-protocol) because `gh pr edit --base` failed silently; main unaffected; closed as MERGED |
| #31 (recovery of #8) | afbafac9 (unchanged) | main | `722f3f0` | same audit as #8; trial tree 56b587e | 8/8 | carries exactly #8's head |
| #9 | aff12108 | codex/g1-native-completion-contract | `3614422` | 349-file audit + scorer boundary review | 8/8 | tree 2459e83e |
| #10 | d8485be6 | codex/g05-scorer-dataset-closure | `29e28dc` | tarball 489 members re-verified + its own PASS review | 8/8 | tree bf96f2d0 |
| #11 | 3ad8809e | codex/g05-scorer-dataset-closure | `a701b86` | platform-pin-only diff check; profiles PREPARED_ONLY | 8/8 | tree d1056213 |
| #12 | 31b793e5 | feat/linux-gfx1151-native-validation | `7b977ab` | receipts hash-pinned in gate | 8/8 | tree 3c31ee17 |
| #13 | 0ec78b0a | feat/linux-d0-dataset-verification | `2efd2d1` | docs-only | 8/8 | tree 8ee1ae6e |
| #14 | 45d03a0 + pin commit + branch-merge | feat/linux-gfx1151-native-validation → main | `5e3b5ee` | lock pin swap verified byte-identical via Git Data API | 8/8 | tree 17eb55f0 |
| #16 | 73471210 | feat/p0-doc-corrections | `1ac3b4d` | 8-doc identity + null-metrics preserved | 8/8 | tree a1cf3ce6 |
| #17 | 157ff4a0 | feat/p1-real-data-integration | `bf89852` | R1–R7 mapping verified | 8/8 | tree 58e1f12f |
| #18 | 72c19d58 + branch-merge | docs/p16-independent-review → main | `88d913d` | 14 real-injection tests executed locally; R-items closed | 8/8 | tree d28629f2 |
| #19 | 6508a91a | feat/p15-runner-hardening | `89caa30` | T-TableTEDS provenance (predeclare 10:48 < run 11:56) | 8/8 | tree 15e040f7 |
| #20 | ec38e30 + digest-pinning fix | feat/t-tableteds-coverage → main | `52558f0` | fix review = 6/6 negative tests + GPU regression byte-identical | 8/8 | tree afff66cc |
| #23 | a370bf87 | feat/p2a-token-budget → main | `52d593e` | 2-doc audit; source-audit only, ≠ #24 production readiness | 8/8 | tree 831051ad |

**MERGED: #7, #8 (via #31), #9, #10, #11, #12, #13, #14, #16, #17, #18, #19, #20 (with required fix), #23 — all 14 mission-authorized PRs — plus the pre-mainline security PR #30.**
**NOT MERGED (out of scope, by design): #24 (rewritten history preserved on its branch; gate extracted instead), #5, #15, #21, #22, #25, #26, #27 (Windows experiment/RCA chain).**
No squash, no rebase, no amend, no force-push of any existing PR history. Additive commits to #14/#18/#20 branches were pushed (or constructed server-side byte-identically via Git Data API when the git upload path was degraded) and are recorded above.

## B. Safety & Evidence

- **Publication audit**: 2,115 unique paths across the integrated scope independently classified (`01_PUBLICATION_AUDIT.md`); 0 GT payloads, 0 credentials; 5 synthetic annotation fixtures + 22 binary + 22 identity-bearing historical files hash-pinned in the 49-entry whitelist with audited reasons.
- **Gate negative testing**: forged schema-valid policy, byte-edited frozen policy, JSONL/nested GT detection, rename smuggle, staged TOCTOU, whitelisted-tamper, real staged pollution injection in throwaway clones — all executed and blocking (`01_PUBLICATION_GATE_REVIEW.md`, 2 rounds GO).
- **`--scan .` quirk**: reproduced (prefix `.` breaks whitelist anchoring), fixed, regression-tested; not disguised as success.
- **#24 isolation**: branch head `4903976f` (P0-rewritten, leak-free) preserved untouched; NOT merged; its general gate re-implemented fresh on clean baseline; stale local refs never pushed.
- **GitHub Support #4934458**: still OPEN at close (cached-view/PR-ref purge pending on GitHub Support; ticket ≠ purge). Re-checked this session.
- **Evidence immutability**: historical receipts hash-pinned (any edit re-trips the gate); raw run evidence stays under untracked `.b/`/0700 dirs; sanitized receipts public.

## C. Linux Verification (distinctions: RUN / PASS / FAIL / SKIP / BLOCKED / NOT RUN)

| Item | State | Evidence |
|----|----|----|
| Clean installation per #14 portable lock (fresh venv, SHA-verified wheels → lock → editable checkout) | RUN → PASS | fresh_env2.log; pip check PASS; identity = torch 2.12.0+rocm7.2 / triton-rocm 3.7.0 / torchvision 0.27.0+rocm7.2 / py3.10.20 |
| CPU unit suite on final tree | RUN → PASS (351 passed / 7 skipped) | trial merges + final 52d593e |
| Ruff / format / mypy / build / pip check | PASS (CI 8/8 on every merged PR) | GitHub Actions |
| Triton compile+execute | RUN → PASS | p3-evidence triton.json |
| DASH sparse prefill | RUN → PASS | p3-evidence sparse.json |
| Vision SDPA MATH | RUN → PASS | p3-evidence vision_sdpa.json |
| gfx1151 golden regression (6 single + 7 multi) | RUN → PASS, **13/13 byte-identical** to #11-frozen receipts | p3-evidence manifests |
| Real-document runner (8-doc P1 subset) | NOT RUN this session (no re-execution authorized; states preserved from committed evidence) | #16/#18 receipts |
| Fault injection | RUN → PASS (14 real-subprocess tests locally on merged tree) | test_p15_runner_hardening.py |
| Official scorer | RUN → PASS on L1.1-B output (below); selftest 8/8 | l11b-score-out |
| Windows physical regression | NOT RUN (Windows host work out of scope for this session); sealed evidence integrity re-verified (489 members) | #10 tarball + review |

## D. L1.1

- **Admission**: C1–C7 all re-verified at launch (06_L11_B_PHYSICAL_REPORT.md §C1–C7). C7 = the mission brief's explicit single-run authorization. B-policy finalization: draft `64bae56a…` unchanged; final `d95e5bd0…` with exactly the two pre-declared receipt-reference fields; receipt bytes identical `8d80ae7b…`.
- **Real run: YES, exactly one.** Attempt id (ledger): `e9dde491803e1a853b2b829f1d760ff79ff2cd4c24c40a13ffc5d901e0dbdfe5`; run tree `39c3813`; launched 10:45, wall 2,912.8 s.
- **Frozen policy / selection**: digests above; budget 39,321; n-gram 35/1024 output-only; timeout 14,400 s; watchdog 16 GiB hard + 4 GiB RSS soft; single document, single attempt.
- **Actual GPU result**: **PASS on all five classes** — 15/15 pages, natural `stop` at 24,050/39,321 tokens, 0 forced-marker mismatches (observer 24,051 steps agrees with replay), 0 supervision failures; official scorer: TextEdit 99.69, T-TextEdit 99.98, TableTEDS 87.57, FormulaCDM 99.21 (7/7 rendered), HeadingTEDS 99.15, ReadOrderEdit 100.0, FigureF1 100.0, T-TableTEDS null (no merged tables — by rule), overall null (by rule).
- **Independent review**: **CONFIRM**, zero P0/P1/P2 (`06_L11_B_INDEPENDENT_REVIEW.md`); 15/15 page files re-hashed; sealed receipts private-path clean.
- **New RCA needed**: none — no failure to analyze. Case A remains BLOCKED (objective dataset fact: no newspaper/magazine class doc has 15 pages); any substitute requires fresh gate + NEW authorization (not granted).

## E. Remaining Blockers (itemized, not merged into a blur)

1. **#24 GitHub cache/Support** — Support ticket #4934458 OPEN; cached raw view and blobs API purge pending GitHub Support; local/server HEAD history clean.
2. **Windows Exact Parity** — Windows/Linux byte-parity for the full official pipeline remains unproven on this session's evidence; Windows physical re-validation NOT RUN here.
3. **TensorExpr / QuickGELU RCA** — #25/#26 (Windows RCA chain) unmerged; RCA remains open.
4. **SAM/CLIP numerical reference** — #27 unmerged; reference audit open.
5. **15-page Native long-context limits** — one document (paper_eng_64b5e1f4) passed at the 39,321 class; this does NOT generalize: 30/50/100-page support remains unimplemented and unauthorized; official deployment uses 10-page windows + KV-tail reuse (not ported).
6. **Native/SGLang Parity** — E1/E2 fair-parity design exists (migration mission); no parity run executed; no claim made.
7. **n-gram official deployment parity** — suppression proven on native path at experiment level; official SGLang-side window semantics equivalence beyond the source audit (#23) remains un-exercised end-to-end.

## Final tally

**MERGED:** 15 PRs (14 authorized + #30) — see §A. **NOT MERGED:** #24, #5, #15, #21, #22, #25, #26, #27. **PASS:** Phase 1 gate, Phase 3 Linux validation, L1.1-B single run + its gates + independent reviews. **FAIL:** none recorded. **BLOCKED:** case A (objective), #24 cached-view purge (pending GitHub Support). **NOT RUN:** Windows physical re-validation, real-document re-execution, 30/50/100-page stages (unauthorized), Native/SGLang parity.

**Final project status: KEEP EXPERIMENTAL.**
