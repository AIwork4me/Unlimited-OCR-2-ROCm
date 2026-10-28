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

## Continuation: unblock and finish PR #4

The user's continuation requests completion using the same frozen sources and
installed SDK. Preserve the original blocked evidence as an earlier attempt.

1. Inspect Windows ROCm dependency guards and setup marker checks; recover exact
   pins via commit-only filtered fetch and sparse checkouts for excluded targets.
   Materialize full CPU/Windows dependencies and their required nested gitlinks.
2. Reuse C:/b/be and SDK 7.14.1. Resume the official frozen builder with
   --no-install-rocm, gfx1151, FLASH/EFFICIENT ON, parallelism 8. Preserve actual
   configure/compiler/linker stdout, stderr and first fault.
3. Add a separately reviewed candidate identity profile without modifying frozen
   baseline producers. Gate source SHA, wheel hash, cp310, HIP/gfx1151 and actual
   loaded binary/image identities; no old audio/vision requirement for torch-only.
   Prove gate behavior with CPU regression tests before GPU execution.
4. If built, inspect/hash the wheel, explicitly install in C:/b/new and verify
   provenance; run identical fixtures/seed/reference/thresholds, actual fused
   routes, three processes, downstream probes and HIP launch logs.
5. Update only PR #4; remotely verify final head, all evidence bytes and both
   hosted CPU jobs. Decision remains evidence-driven, with no network-to-regression
   inference and no production/OCR/upstream source modifications.

- Prior publication bc75cfe: all 177 evidence blobs independently fetched from
  GitHub and SHA-256 verified; Windows/Ubuntu PR CPU jobs passed.
- CUTLASS pin recovered with sparse setup-marker checkout. Frozen setup.py checks
  all declared folders, while CUDA-only CUTLASS kernels are guarded by USE_CUDA.
- All 37 top-level gitlinks and three compiled nested dependencies now pinned;
  root source clean before builder. ONNX/protobuf/oneDNN recovered by batches of
  exact object IDs; no TLS verification disabled and no source revision changed.
- Builder resumed in an MSVC child with --no-install-rocm. Existing local trusted
  Clash/Mihomo proxy is scoped to the build subprocess, with no global settings.
- Candidate identity adapter separately reviewed: 26 candidate tests plus 30
  existing gate tests pass. Review issues (stale pip-check proof and abbreviated
  error-clearing options) reproduced RED, fixed GREEN, and confirmed by reviewer.
  Tensor/reference/seed/mask/backend AST semantics unchanged outside two identity
  nodes. Candidate GPU execution remains conditional on an actual installed wheel.
