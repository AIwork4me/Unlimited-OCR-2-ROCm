# PR #32 × Windows Delivery — Compatibility Audit (Phase 1)

States used throughout: PASS / FAIL / PARTIAL / BLOCKED / NOT_RUN / UNVERIFIED.

## 1.1 Frozen remote identity — PASS (re-frozen after head movement)

Re-read live from GitHub on 2026-11-01 (not assumed from prior observation):

| Field | Value at first read | Value at re-read (before publication) |
| --- | --- | --- |
| PR | #32 `https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/32` | unchanged |
| State | OPEN, not draft, `MERGEABLE`, not merged | unchanged |
| Head | `a01db7de0f2cc26142f7b5269fbde46f506a3cff` | **moved to `a720c49c2e9f755633fd7efa95ab523fe072e871`** (+1 commit: "fix(review): pin the L1/L1.1-B policy digests in the budget gate + single-read policies" — touches only `inference.py` and policy tests) |
| Base | `main` @ `52d593e0686176b3842d78fbac3ef76956ecf1ae` — **identical to the Windows branch base** | unchanged |

The head movement was recorded and the affected analysis repeated (mission rule):
the overlap surface is unchanged (still exactly `publication.py` +
`test_publication_gates.py`; the new commit touches disjoint files), and the
merge simulation was re-run at the refreshed pair — both orders still converge
(tree `7eb5c1cd…`) with **460 passed / 10 skipped / 0 failed** on the combined
tree (the extra pass is #32's new policy-admission test).

Open-PR #24 (`feat/p2b-ngram-suppression`) is an ancestor of the PR #32 head
(`git merge-base --is-ancestor` verified): #32 is the aggregate L1/P2B/P2C
integration line; the readiness docs' "integrate PR #24" next-step is satisfied by
#32 semantically. No other open PR touches the overlap surface.

## 1.2 File/behavior overlap — PASS (machine-readable: `CROSS_BRANCH_FILE_MATRIX.json`)

PR #32 changes 69 files; Windows changes 34 (six-commit head) / 34 (seven-commit
head — the follow-up commit touches one already-counted file). Files changed by
**both** branches: exactly two.

| File | PR #32 side | Windows side | Combined behavior |
| --- | --- | --- | --- |
| `src/unlimited_ocr_2_rocm/publication.py` | new case-insensitive `_WINDOWS_PATH_PATTERN`; strict JSON parse with duplicate-key rejection; bounded fail-closed nested annotation search; byte-based (TOCTOU-safe) scanning refactor | one WHITELIST hash update (p15 test) + `scan_tree` selected-root skip-parts fix (`path.relative_to(root).parts`) | all hardening from both sides present and exercised on the merged tree; verified |
| `tests/unit/test_publication_gates.py` | adds fail-closed JSON/JSONL/duplicate-key/binary-like/Windows-path tests | adds selected-root-inside-cache scan test | both suites coexist; all pass merged |

Shared interfaces changed by only one side (no file overlap): PR #32 modifies
`completion.py`, `inference.py`, adds `benchmark/watchdog.py` + `collision_observer.py`;
Windows modifies `benchmark/runner.py`, adds `benchmark/windows_job.py`. Disjoint
adds elsewhere (docs/eval/scripts/tests). No Git conflicts in either merge order
(see 1.6).

## 1.3 Windows Job supervision compatibility — PASS

- Merged-tree suite passes every Windows runner/Job test, including
  `test_supervisor_death_closes_job_and_kills_workers` with PR #32's components present.
- `_supervise_windows_worker` retains the full contract: suspended create → Job
  assign → resume, KILL_ON_JOB_CLOSE, descendant drain in `finally`, outcome
  preservation on timeout/crash/interrupt/nonzero/unreadable, attempt accounting.
- PR #32's `benchmark/watchdog.py` is an in-process daemon `threading.Thread`
  (no `Popen`/`multiprocessing`): nothing in #32 spawns processes outside the Job.
  **Observation (not a defect):** the watchdog is wired into the L1 experiment
  scripts, not the common runner — Windows production runs have no resource
  watchdog; carried into the risk register and the Phase 5 admission design.
- Known gap retained, unchanged: hard supervisor death between process creation
  and Job assignment is outside the guarantee. No unconditional cleanup claim.

## 1.4 Publication safety — PASS after one Windows-side correction

Negative coverage verified on the merged tree: private paths (incl. new lowercase/
mixed-separator Windows forms), user directories, credentials/host identities,
dataset GT (nested/duplicate-key/JSONL), unexpected binaries, malformed JSON(L),
whitelist anchoring at repo root, tracked-tree cleanliness, gate CLI.

**Found (the audit's one P1):** PR #32's stronger detector correctly flags the
synthetic lowercase `c:/users/person` and mixed-case literals that the Windows
branch's `test_private_path_refusal_does_not_echo_path` parametrizes. On the
combined tree this produced 3 publication-gate failures
(`test_scan_repo_root_anchors_whitelist`,
`test_tracked_publication_tree_is_clean`,
`test_gate_cli_scan_dot_anchors_whitelist`) — the Windows branch alone passes
because the base detector misses those forms.

**Resolution (smallest reviewed change, Windows side only, no PR #32 code touched):**
follow-up commit `c432861…` on `delivery/windows-native-benchmark` adds the
established hash-pinned WHITELIST entry for
`tests/unit/test_windows_benchmark_runner.py`
(SHA-256 `dfaf2f9b03a623efd50b203f737c45d0fecf16d86828ad57b510a9017c4c0c1c`,
reason: sanitizer test with synthetic private-path refusal literals). Neither
detector is weakened; the entry stops matching if the file changes
(`*.py text eol=lf` pins the hash cross-platform; blob == working-copy bytes).
No check was deleted to make the merge pass.

## 1.5 Benchmark policy compatibility — PASS

- Production Native path (5 pages / 8192 tokens / default ngram 0) unchanged by
  #32; all Windows preflight/benchmark runner tests pass merged.
- Experimental pinned L1/L1.1-B/P2B/P2C policies ship as opt-in `eval/experiments/*/policy.json`
  with digest-pinned admission; #32's `test_l1_policy_admission`,
  `test_l11_b_policy`, `test_p2b_ngram_policy`, `test_p2c_combined_policy`,
  `test_l1_gates`, `test_p2c_guard_collision` all pass on the merged tree —
  unauthorized budget / missing policy / wrong source digest fail closed.
- Merging #32 grants no long-document or full-benchmark authority on Windows:
  its evidence is Linux gfx1151; nothing auto-enables 15-page/39321-token shapes
  for the Windows production entry.

## Final classification

**COMPATIBLE_WITH_CONDITIONS**

Condition (satisfied): Windows PR head must include follow-up commit `c432861`
(the whitelist entry). No condition requires changing PR #32. The Windows Draft
PR is not hostage to #32's merge: publication is against `main` @ `52d593e`, with
this audit linked.
