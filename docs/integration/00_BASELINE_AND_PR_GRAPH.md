# 00 — Baseline & PR Dependency Graph (Phase 0, FROZEN)

- Frozen at: 2026-11-01, Asia/Shanghai
- Remote: `AIwork4me/Unlimited-OCR-2-ROCm` (gh auth: `AIwork4me`)
- `main` HEAD at freeze: **`beb38d1941a0ae85a1c1218209c822c3e2f115e3`** ("Merge pull request #4 …")
- Method: live GitHub API (`gh pr list --state open`) + local `git merge-base` ancestry checks on freshly fetched refs. Past reports were NOT trusted; all identities below were re-derived this session.

## 1. Open PR inventory (22 PRs, #5–#27, no #6)

| PR | Title (short) | State | Base | Head branch | Head SHA | Mergeable | CI |
|----|----|----|----|----|----|----|----|
| #5 | native OCR candidate vision-SDPA (Windows) | DRAFT | main | codex/native-ocr-candidate-vision-sdpa | cfd68422 | CONFLICTING | — |
| #7 | benchmark protocol + full-scope eval infra | READY | main | codex/benchmark-protocol | **d842ce6f** | MERGEABLE | 8/8 PASS |
| #8 | native completion metadata + failure evidence | DRAFT | #7 base | codex/g1-native-completion-contract | afbafac9 | MERGEABLE | 8/8 PASS |
| #9 | official CPU scorer + dataset closure | DRAFT | #8 base | codex/g05-scorer-dataset-closure | aff12108 | MERGEABLE | 8/8 PASS |
| #10 | Windows G1 regression seal | DRAFT | #9 base | codex/g1-physical-windows-regression | d8485be6 | MERGEABLE | 8/8 PASS |
| #11 | Linux gfx1151 primary target | READY | #9 base | feat/linux-gfx1151-native-validation | 3ad8809e | MERGEABLE | 8/8 PASS |
| #12 | D0 dataset verification | READY | #11 | feat/linux-d0-dataset-verification | 31b793e5 | MERGEABLE | 8/8 PASS |
| #13 | P0 doc corrections (KV 7.5 GiB etc.) | READY | #12 | feat/p0-doc-corrections | 0ec78b0a | MERGEABLE | 8/8 PASS |
| #14 | portable Linux dependency lock | DRAFT | #11 | codex/portable-linux-lock | 45d03a0e | MERGEABLE | 8/8 PASS |
| #15 | synthetic SDPA A/B (Windows) | DRAFT | codex/windows-integration-20261030 | codex/synthetic-sdpa-ab | 3b817314 | MERGEABLE | — |
| #16 | P1 real-document integration | READY | #13 | feat/p1-real-data-integration | 73471210 | MERGEABLE | 8/8 PASS |
| #17 | PR #16 independent review doc | READY | #16 | docs/p16-independent-review | 157ff4a0 | MERGEABLE | 8/8 PASS |
| #18 | P1.5 runner hardening | READY | #17 | feat/p15-runner-hardening | 72c19d58 | MERGEABLE | 8/8 PASS |
| #19 | T-TableTEDS coverage | READY | #18 | feat/t-tableteds-coverage | 6508a91a | MERGEABLE | 8/8 PASS |
| #20 | P2A token-budget opt-in | READY | #19 | feat/p2a-token-budget | ec38e305 | MERGEABLE | 8/8 PASS |
| #21 | Windows reliability validation | DRAFT | codex/synthetic-sdpa-ab | codex/windows-reliability-validation | 6a906398 | MERGEABLE | — |
| #22 | cold-start determinism RCA | DRAFT | codex/windows-integration-20261030 | codex/cold-start-determinism-rca | 10067ee6 | MERGEABLE | — |
| #23 | n-gram-35 source audit | READY | #20 | docs/p1-ngram-source-audit | a370bf87 | MERGEABLE | 8/8 PASS |
| #24 | P2B n-gram suppression (rewritten history) | READY | #23 | feat/p2b-ngram-suppression | **4903976f** | MERGEABLE | 8/8 PASS |
| #25 | CLIP TensorExpr BF16 RCA | DRAFT | #22 base | codex/clip-mlp-kernel-rca | ff467134 | MERGEABLE | — |
| #26 | QuickGELU / attention TE mitigation | DRAFT | #25 base | codex/te-mitigation-e2e | 37f46d35 | MERGEABLE | — |
| #27 | SAM/CLIP numerical reference audit | DRAFT | #26 base | codex/sam-clip-numerical-reference | 69f28973 | MERGEABLE | — |

### Reference-SHA comparison (mission §0)

- #7 reference `d842ce6f…` → live head **identical**. No unreviewed change.
- #24 reference `4903976f…` → live head **identical** (this is the post-P0-rewrite history pushed 2026-11-01 under owner authorization; old local refs still carry pre-rewrite history — see §6).
- Closed PRs on repo: only #1, #3, #4 (all MERGED long before freeze). No unexpected closures.

## 2. Verified stacked DAG (git merge-base evidence, not GitHub display)

```
main (beb38d19)
└── #7  codex/benchmark-protocol (d842ce6f)
    └── #8  codex/g1-native-completion-contract (afbafac9)
        └── #9  codex/g05-scorer-dataset-closure (aff12108)
            ├── #10 codex/g1-physical-windows-regression (d8485be6)   [sibling]
            └── #11 feat/linux-gfx1151-native-validation (3ad8809e)
                ├── #12 feat/linux-d0-dataset-verification (31b793e5)
                │   └── #13 feat/p0-doc-corrections (0ec78b0a)
                │       └── #16 feat/p1-real-data-integration (73471210)
                │           └── #17 docs/p16-independent-review (157ff4a0)
                │               └── #18 feat/p15-runner-hardening (72c19d58)
                │                   └── #19 feat/t-tableteds-coverage (6508a91a)
                │                       └── #20 feat/p2a-token-budget (ec38e305)
                │                           └── #23 docs/p1-ngram-source-audit (a370bf87)
                │                               └── #24 feat/p2b-ngram-suppression (4903976f)  [NOT in merge scope]
                └── #14 codex/portable-linux-lock (45d03a0e)              [sibling]
Windows/RCA chains (NOT in merge scope): #5, #15, #21, #22, #25, #26, #27
```

Ancestry checks all PASS (`git merge-base --is-ancestor`): #7∈#8∈#9; #9∈#10 and #9∈#11 with **#10 ∦ #11**; #11∈#12∈#13∈#16∈#17∈#18∈#19∈#20∈#23∈#24; #11∈#14 with **#14 ∦ #12**; main∈#7; none of the chain merged into main yet.

**Base drift: ZERO** on all 15 audited PRs (merge-base(head, base) == base head for every PR).

## 3. Sizes (commits / changed files) and CI

| PR | commits | files | CI |
|----|----|----|----|
| #7 | 3 | 45 | 8/8 SUCCESS |
| #8 | 2 | 8 | 8/8 SUCCESS |
| #9 | 2 | 100 | 8/8 SUCCESS |
| #10 | 6 | 22 | 8/8 SUCCESS |
| #11 | 4 | 85 | 8/8 SUCCESS |
| #12 | 1 | 7 | 8/8 SUCCESS |
| #13 | 1 | 4 | 8/8 SUCCESS |
| #14 | 1 | 2 | 8/8 SUCCESS |
| #16 | 1 | 12 | 8/8 SUCCESS |
| #17 | 1 | 1 | 8/8 SUCCESS |
| #18 | 4 | 13 | 8/8 SUCCESS |
| #19 | 2 | 5 | 8/8 SUCCESS |
| #20 | 3 | 9 | 8/8 SUCCESS |
| #23 | 3 | 2 | 8/8 SUCCESS |
| #24 | 8 | 55 | 8/8 SUCCESS |

## 4. Recommended merge order

All merges are **Merge Commit** (no squash/rebase). After each merge the next PR's base is retargeted to `main` and its diff re-verified for ancestor-duplication.

1. **#7** (base already main)
2. **#8** (retarget → main)
3. **#9** (retarget → main)
4. **#10** (retarget → main; sibling of #11)
5. **#11** (retarget → main)
6. **#14** (retarget → main; child of #11; disjoint from #12/#13)
7. **#12** (retarget → main)
8. **#13** (retarget → main)
9. **#16** (retarget → main)
10. **#17** → 11. **#18** → 12. **#19** → 13. **#20** (after validate_budget hardening per Phase 5) → 14. **#23**

**Out of scope (must NOT merge):** #24 (post-P0-rewrite history preserved, gate code extracted separately in Phase 1), #5/#15/#21/#22/#25/#26/#27 (Windows experiment/RCA chain).

## 5. Conflict analysis

Sibling-pair diffs vs their merge-base were intersected:

- #10 ∩ #11: **0 files**
- #14 ∩ #12: **0 files**
- #14 ∩ #13: **0 files**
- #12 ∩ #13: **0 files**

Sequential-stack pairs cannot conflict by construction (fast-forward ancestry). Local integration prototype `integration-p0` @ `4cd5945` (#13+#10 merge, "zero file intersection", 304 tests green on 2026-10-30) independently corroborates. **No conflicting PRs exist in the merge scope.**

## 6. Local-environment hazards recorded at freeze

1. Local branch `feat/p2b-ngram-suppression` in the main checkout (`<MISSION_ROOT>/Unlimited-OCR-2-ROCm`, worktree `wt-p18`) still points at **pre-rewrite history `33099a5`** which contains leak commit `4d203a1` (blob `4117e16f`). Rule for this mission: **all references to PR #24 content MUST use `origin/feat/p2b-ngram-suppression` = `4903976f` (rewritten, leak-free). Local stale refs are never pushed, never rebased, never deleted.**
2. `integration-p0` @ `4cd5945` (local branch `integration/p0-closure`) is prior work — preserved untouched; used as reference evidence only.
3. GitHub Support ticket **#4934458 remains Open** (cached-view/PR-ref purge of the historical leak is pending on GitHub Support; local rewrite does not equal purge).
4. GPU at freeze: gfx1151, 32 GiB VRAM total, 1.5 GiB baseline in use (ollama/docker co-tenancy), 8% busy — no OCR workload running.

## 7. Worktree plan for this mission

- `integration`  → `<MISSION_ROOT>/integration` (branch `integration/mainline` @ beb38d1) — mainline integration audit work + this doc.
- `publication-audit` / `security-fix` → created in Phase 1 (read-only audit clone / minimal-gate fix branch).
- `l1-validation`, `review` → Phase 6 only (migration repo is at `<MISSION_ROOT>/l1-migration/migration-repo`).

This document is committed to local branch `integration/mainline` only; **not pushed** — any future push passes the Phase 1 publication audit first.
