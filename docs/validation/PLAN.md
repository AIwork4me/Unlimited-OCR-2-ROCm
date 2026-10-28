# Current Windows PyTorch 2.12 wheel validation

The attached mission is the approved specification. Execution is autonomous,
native Windows CPython 3.10 x64, on the physical gfx1151 device. No production
environment, historical evidence, issue #2 or PR #3 is changed.

## Execution plan and ledger

1. Fetch latest main and upstream heads; create the requested validation branch.
   Freeze source commits and copy only the immutable standalone reproducer/helpers.
2. Record machine/toolchain, AOTriton pins and overrides. Create separate short-path
   build (`C:/b/be`), baseline (`C:/b/old`), candidate (`C:/b/new`) and CPU-check
   (`C:/b/qa`) environments. Source is `C:/b/pytorch` and `C:/b/therock`;
   artifacts and private original logs remain under `C:/b`, evidence in this tree.
3. Attempt the current official TheRock CLI with SDK ==7.14.1, gfx1151 only,
   FLASH/EFFICIENT enabled, no unrelated projects and MAX_JOBS=8. Capture every
   process exit. A missing prerequisite is BUILD_BLOCKED, never a successful build.
4. Install the exact historical wheel trio in the baseline environment. Run the
   unchanged RCA reproducer in fresh processes, seed 7992 and fixed tolerances;
   include requested fused routes, CPU-profiler MATH controls, downstream probes,
   three fused repetitions and AMD_LOG_LEVEL=4 launch traces.
5. If and only if a candidate wheel exists, inspect/hash it, install by explicit
   path in the candidate environment, verify loaded identities, and run the same
   matrix. If blocked, preserve the actual blocker and mark all candidate fields
   unmeasured. Never repair images or clear HIP errors for acceptance.
6. Add CPU tests for conservative comparison/classification, provenance and
   report parsing. Observe failing tests before implementation. Run Ruff,
   formatting, mypy, CPU pytest, package build and isolated pip checks.
7. Sanitize original logs with an original/published digest ledger; hash all
   evidence. Write decision, provenance, matrix and next-upstream reports.
8. Request a separate read-only review, fix reproducible findings, then commit
   evidence/report, push the dedicated branch, open a new PR against main and
   remotely verify its head, evidence hashes and Windows/Ubuntu CPU CI.

## Review focus

- DEFAULT falling back to MATH must never become FUSED_PASS.
- No numerics after sticky HIP failure means unmeasured, not correct.
- Missing or mixed binary/source provenance prevents acceptance.
- Fixed fixture hashes/tolerances and original script digests must agree.
- A driver/toolchain/source recipe observation does not prove a built artifact.

## Progress

- Branch based on remote main 966265c81a74a73aa0ae6d9a3fa593b342590eb7.
- Upstream heads fetched, new short source checkouts created; no RCA cherry-pick.
- Ruling: use a Git worktree at the user-requested short path C:/uocr; app-managed
  worktree placement would retain a long user-profile path.
- Ruling: the user's complete specification and explicit autonomous execution
  replace further design/permission questions.
- Baseline: all 27 fresh-process conditions completed. 21 fused FAIL, four
  profiled MATH_PASS, copy-only CONTROL_PASS, optional bias FLASH UNSUPPORTED.
  Both launch traces reproduce null-function/zero-block hipErrorInvalidValue.
- First actual TheRock build exited 1 during SDK-devel HTTP timeout, before
  compilation. Retrying downloads and shallow pinned submodule fetches.
- CPU gate tests: 24 initial pass; separate reviewer identified two conservative
  classification issues. Both were reproduced RED and fixed GREEN (26 tests).
- Ruling: the named optional bias FLASH capability probe may be UNSUPPORTED;
  required no-mask FLASH and additive-bias EFFICIENT still must execute.
- SDK 7.14.1 development wheel recovered, ETag/ZIP CRC checked, explicitly
  installed and initialized; build environment pip check passes.
- Final source readiness: 11/37 top-level pins ready; CUTLASS exact fetch failed
  early EOF and TLS EOF on retry (exit 128). No CMake cache or candidate wheel.
  Final decision BUILD_BLOCKED; candidate installation and GPU tests NOT_RUN.
- Final local checks: 86 CPU tests pass, two GPU markers skipped; 27 independent
  physical baseline runs preserved. Ruff/format/mypy/OCR package build and all
  three isolated pip checks pass.
- Separate final review: all 177 evidence Git blobs and 176 export records
  verified; no remaining actionable findings. Publication/hosted CI follows.
