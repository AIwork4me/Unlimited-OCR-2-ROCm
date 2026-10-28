# Phase 1 final merge-readiness closure

Date: 2026-10-28. PR: [#1](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/1).
Reviewed base: `ccd04ceb4d9aade920752f5db86c836f940107bb`.
Reviewed target: `3cddc3c2907da9dffb76280549ce7d0e398609a6`.
The independent reviewer also inspected the documentation delta delivered
with this record. The final delivery SHA and final-head CI results are
recorded in the PR closure update; this file cannot contain its own commit SHA.

## Findings and fixes

The README setup created two environments while subsequent examples selected
an unrelated `.venv` or relied on global CLI lookup. Both language versions
now use short external environments `C:\uocr\a` and `C:\uocr\b`, a shared
explicit cache `C:\uocr\cache`, and the second environment's explicit Python
interpreter for subsequent commands. The reproduction guide uses the same
paths, including its offline rerun. Examples require unused writable venv
paths and new output directories; an alternate writable short root may be used.

The actual known long AMD wheel entry from setup is 180 characters beneath
`C:\uocr\b`, below the documented 260-character boundary. All ten PowerShell
code blocks in the three edited guides parse successfully. This is a path
calculation and syntax check, not a newly executed GPU installation claim.

A separate reviewer, without this implementation conversation's history,
reviewed the full final diff: runtime, vendored postprocessor, setup/probe/
reproduction scripts, tests, package configuration, workflows and accepted
evidence. Focus areas were native loading/device placement, scoped vision
SDPA, DASH verification, OCR correctness, reproduction integrity, security
and error handling.

**Critical: none. Important: none. Minor: none raised.** No runtime fix or
refactoring was justified. The reviewer independently checked:

- All 26 accepted correctness assessments recompute from raw bytes and truth.
- Raw SHA-256/JSON outputs match; wrapper hashes match runtime source; official
  source/weight inventories match the immutable lock; fixture hashes match.
- Each request records twelve synchronized BF16 `cuda:0` kernel calls.
- The linked two-environment comparison recomputes PASS.

## Fresh closure checks

The implementer reran:

| Check | Result |
|---|---|
| `python -m ruff check src scripts tests` | PASS |
| `python -m ruff format --check src scripts tests` | PASS, 26 files |
| `python -m mypy src` | PASS, 10 maintained source files |
| `python -m pytest -m 'not gpu'` | 56 passed, 2 deselected |
| `python -m build` | PASS, wheel and sdist |
| `python -m pip check` in CPU tooling environment | PASS |
| Offline installed-wheel module/console entry-point smoke | PASS |
| PowerShell documentation block parsing | PASS, 10 blocks |
| Diff against target for runtime, scripts, requirements, tests, workflows, package config and `eval/results` | Empty |

The reviewer separately ran CPU pytest (56 passed, 2 GPU tests skipped),
Ruff lint and mypy. These are fresh CPU checks. The prior physical GPU runs
were inspected, not rerun: runtime code and historical evidence are unchanged.
Hosted Windows/Ubuntu CPU checks must pass on the final delivery commit;
consult the PR checks and closure update for that commit's verified status.

## Remaining risks and scope rulings

The reviewer explicitly declined to extend conclusions to the following.
These limits are retained for merge review:

- Fresh physical GPU execution: unnecessary for this documentation-only delta;
  the original physical evidence and immutable references remain unchanged.
- Other hardware/OS, training/backward and unused kernels: outside Phase 1's
  validated Windows/gfx1151 scope.
- Broad OCR accuracy, complex formulas and production document performance:
  the synthetic acceptance set does not establish these.
- Concurrent embedded use: the supported workflow serializes calls in
  dedicated processes; concurrent embedding is not validated.
- Malicious local rewriting of evidence/dependencies: outside the stated
  local evidence trust boundary; no ordinary workflow defect was demonstrated.
- Full third-party binary security certification: beyond this source review.

The measured default vision SDPA failure still requires the disclosed scoped
GPU MATH adapter. One empty terminal PAGE is still a separately reported
delimiter. Neither decision was changed or hidden during closure. The model
weights, official source pins, decoder algorithms and mandated stack remain
unchanged. No deferred Minor findings.

## Merge recommendation

The independent reviewer recommends **merge after the reviewed documentation
delta is committed and final-head CI passes**. No remaining reproducible
Important/Critical issue was identified. The PR remains open for a human merge
decision. This closure does not authorize automatic merge, Phase 2, PyPI
publication, a GitHub Release or a stable tag.
