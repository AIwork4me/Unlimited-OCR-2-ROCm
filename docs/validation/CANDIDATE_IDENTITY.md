# Candidate-only torch identity gate

The historical three-file reproducer/reference/provenance helpers remain
byte-identical to RCA commit `6b643f51d4078c4bb841f6a48734cf2206090311`.
Its exact historical audio/vision package gate still protects baseline runs.
Candidate validation uses a separately reviewed adapter; it does not install
historical audio/vision packages merely to satisfy that baseline gate.

## What changes

`candidate_identity.adapt_identity_only` first verifies the frozen reproducer's
SHA-256. It then replaces exactly two known AST statements: the historical
expected-version mapping and its baseline identity predicate. The replacement
calls the candidate runtime identity gate. A regression test compares every
other AST node, including all fixture, seed, mask, shape, numerical reference,
tolerance, backend, HIP checkpoint and downstream-operation code.

The original `repro_sdpa.py` file is never edited. Candidate reports retain its
digest and add adapter/profile/gate digests, exact original command and observed
candidate runtime identities. Parsed error-clearing requests are rejected before
the frozen execute function runs, including argparse abbreviations.

## What must be observed

`prepare_candidate_profile.py` reads the actual built Windows cp310 wheel ZIP,
embedded `torch/version.py`, DLLs, AOTriton images/signatures and header. The HIP
runtime bytes are bound to the already verified exact SDK 7.14.1 core wheel.
It rejects an artifact with another source commit, HIP identity, ABI or platform.
Only an explicit local torch installation report is accepted.

Every subsequent fresh acceptance process independently checks:

- Original built wheel SHA-256, recorded installed-wheel SHA-256 and rehashed
  wheel bytes agree; imported torch git version equals frozen source `60dd50f…`.
- CPython cp310 x64/Windows, HIP 7.14.60850 and physical gfx1151 identity agree.
- Every torch DLL/PYD member is independently rehashed from the actual wheel and
  the installed copy. This includes the CPU reference implementation, Python
  extension and C10 libraries, plus native members that happen not to be loaded.
- Every loaded module matching an artifact basename comes from its expected
  candidate path and has the expected hash. Required CPU/Python/C10/HIP/AOTriton
  core modules must be loaded; duplicate or foreign matching modules are refused.
  Genuinely optional modules may remain unloaded, with their installed bytes
  still verified.
- Complete installed image-file inventory/signatures and runtime-header hash
  match the artifact-derived profile.
- A fresh `sys.executable -m pip check` succeeds in that same process environment;
  interpreter, prefix, command, stdout/stderr and exit are recorded.
- No AOTriton source/target override or architecture override is active.

TheRock and AOTriton recipe pins are build declarations, not runtime exports.
Their proof stays in the separate immutable-source/build provenance. Matching
runtime/image identities to artifact bytes does not establish a compatible
contract or correct kernels: actual fused GPU dispatch, valid launches and the
frozen numerical matrix remain mandatory.

## Separate review and tests

A separate read-only reviewer confirmed that the adapter preserves all numerical
and fixture semantics. Two findings were reproduced with CPU tests and fixed:
unbound/stale pip-check evidence and abbreviated error-clearing flags. The final
review confirmed both fixes and **26 candidate gate tests + 30 existing gate tests**
passing, with no remaining blocker to GPU execution after a real wheel passes
artifact/installation checks. That review does not establish build or GPU success.

The 26 candidate tests exercise torch-only acceptance, frozen source/recipe pins,
wheel/source/HIP/ABI/architecture drift, environment/pip interpreter mismatch,
DLL/image/header changes, missing payloads, unchanged AST nodes, frozen producer
digest and all three forms of the prohibited error-clearing option.

Actual-artifact review subsequently found that the initial native inventory
covered only four torch DLLs and omitted loaded CPU/Python/C10 binaries. Seven
new failing regression cases reproduced the gap before the fix. The expanded
gate now has **35 candidate tests + 30 existing gate tests** passing, including
foreign core modules, missing loaded core modules, and omitted or modified
optional payloads. Independent review verified all **15 wheel native binaries
plus the SDK HIP DLL**, seven images and actual loaded paths against the real
candidate installation. The identity gate is cleared for the GPU matrix; this
identity review alone does not establish numerical correctness or valid launches.
