# Integration Simulation — PR #32 × Windows Delivery (Phase 1.6)

Disposable local worktrees only (`.worktrees/sim-a`, `.worktrees/sim-b`,
detached, never pushed). Neither source branch was altered. Test interpreter:
`.b/qa` CPU environment, `PYTHONPATH` bound to the simulated tree's `src` so both
pytest and test-spawned subprocesses import the combined code (the main checkout's
editable install points at a different branch and must not leak into results).

## Merge-order matrix

| Combination | Git conflicts | Resulting tree | Combined-tree CPU result |
| --- | --- | --- | --- |
| PR #32 head + Windows `0ba45dd` (order A: Windows applied after #32) | none | `4c090697…` | **3 FAILED** / 446 passed / 10 skipped |
| Windows `0ba45dd` + PR #32 head (order B: #32 after Windows) | none | `4c090697…` (identical) | same 3 failures |
| PR #32 head + Windows `c432861` (fixed head, both orders) | none | `5b26f0fe…` (identical both orders) | **459 passed / 10 skipped / 0 failed** |

Both orders converge to the same tree in every pairing — the integration is
order-independent; eventual merge direction is a maintainer choice, not a
correctness constraint.

## The three failures (original six-commit head)

All one root cause — `tests/unit/test_publication_gates.py`:
`test_scan_repo_root_anchors_whitelist`, `test_tracked_publication_tree_is_clean`,
`test_gate_cli_scan_dot_anchors_whitelist`. Violation tuple:
`('tests/unit/test_windows_benchmark_runner.py', ["private host path 'c:/users/person'"])`.
PR #32's case-insensitive Windows path detector (correctly) flags the Windows
branch's own synthetic refusal literals; the Windows branch alone passes only
because the base detector is case/separator limited. This is precisely the
"clean Git merge is not semantic compatibility" case: zero textual conflicts,
three broken gates.

Also diagnosed and excluded: 10 further failures in a first run
(`test_p15_runner_hardening.py` ×9, `test_supervisor_death…`) were an
environment-binding artifact — test subprocesses resolved
`unlimited_ocr_2_rocm` from the main checkout's editable install (branch
`codex/overnight-closure-20261031`, which predates the runner) instead of the
simulated tree. With `PYTHONPATH` pinned to the simulated tree they pass; they
are not integration findings. Prior-mission runs used the same binding.

## Fixed-head validation on the combined tree `5b26f0fe…`

| Check | Result |
| --- | --- |
| `pytest` (full CPU suite) | 459 passed, 10 skipped, 0 failed (47.5 s) |
| `ruff check src scripts tests` | pass |
| `ruff format --check` | 85 files already formatted |
| `mypy --platform win32 src` | pass (22 files) |
| `mypy --platform linux src` | pass (22 files) |

Combined suite content: Windows suite (runner/Job/preflight/PowerShell/install)
+ #32 suite (L1/L1.1-B/P2B/P2C admission, gates, collision observer, watchdog)
+ fused publication-gate suite, all green together.

## Residual semantics (documented, not blockers)

- Resource watchdog exists in the merged tree but is invoked by L1 experiment
  scripts, not the common runner; Windows production runs remain unwatched.
- #32's `eval/results/Linux gfx1151 verified/**` evidence is Linux-scope; it must
  never be re-labeled as Windows qualification.
- The whitelist fix commit means the Windows PR head is now 7 commits; the
  physical GPU evidence binding is unchanged (src/ byte-identical from
  `3556e38` through `c432861` — the fix touches only `publication.py`
  WHITELIST, not inference code; CPU evidence re-established in Phase 2.6).
