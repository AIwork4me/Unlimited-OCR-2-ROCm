# Separate read-only validation review

A separate reviewer audited the scripts and original physical baseline evidence,
then reviewed the completed reports and published-byte candidates. The reviewer
did not modify files or send GitHub messages.

## Findings and fixes

1. **P2:** UNSUPPORTED was returned before HIP checkpoints were inspected. A dirty
   unsupported report could hide a GPU error. A CPU test reproduced the finding;
   classification now checks recorded HIP failures first. RED→GREEN logs retained.
2. **P2:** the optional vision/additive-mask FLASH capability probe could prevent
   an otherwise complete matrix from reaching FIX_VERIFIED. Only that explicitly
   named clean UNSUPPORTED probe is excluded by `acceptance_statuses`; unsupported
   required FLASH/EFFICIENT routes still block acceptance. CPU tests pin both cases.
3. A separate publication check found inherited Git newline normalization would
   change some evidence bytes and a generic `build/` ignore excluded build logs.
   Nothing had been pushed. A path-specific `-text -whitespace` attribute and
   explicit addition of the text-only evidence build directory fixed publication.
   The final review checked actual Git blobs, not only working files.

## Final audit result

**No remaining actionable finding. BUILD_BLOCKED is supported.**

At evidence commit `bc42ebcc6120d056374b8777e323b45330a2cbc4`, the reviewer verified:

- All **177 SHA256SUMS entries**, including **43 build-evidence files**, exist in
  Git and have exactly the published digests.
- All **176 export records** match original staging and published Git bytes/counts.
- The three frozen RCA helpers and AOTriton recipe digest match immutable sources.
- **30 focused CPU tests** pass and both classifier findings are fixed.
- Baseline classification independently recomputes to **21 FAIL, 4 MATH_PASS,
  1 CONTROL_PASS, 1 UNSUPPORTED**; missing fused numerics never become a pass.
- Dispatch, frozen tolerances, fixture digests and loaded DLL/wheel-byte hashes
  agree with the reports. No acceptance error clearing or payload substitution.
- SDK recovery, final CUTLASS exit 128, initial actual builder exit 1, 11/37
  source readiness and absence of a candidate/CMake cache are distinguished.
- The OCR CPU package build is not presented as a PyTorch wheel build.
- Every checked local report link resolves; this review file resolves the one
  link that was pending while the final audit ran.

The full local check suite separately records **86 passed, 2 intentionally skipped
GPU pytest markers**, Ruff/format/mypy success, OCR package build success and
pip-check success in QA, baseline and build environments. The 27 standalone
physical GPU runs were executed separately from these CPU checks.

## Limits

No candidate artifact, fused GPU acceptance or old/new artifact comparison exists.
Several historical cached wheels have no independently advertised upstream digest;
the exact original torch wheel matches the immutable RCA's body hash. Hosted CI
was pending at review time and must be verified on the final pushed PR head.
The final GitHub publication verification occurs after this report commit.
