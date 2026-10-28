# Current release/2.12 Windows gfx1151 wheel validation

This is a validation experiment. See [the plan](../../docs/validation/PLAN.md).
Production source and the historical RCA are unchanged.

`repro_sdpa.py`, `reporting.py` and `collect_provenance.py` are copied **byte for
byte** from [RCA commit 6b643f51](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/tree/6b643f51d4078c4bb841f6a48734cf2206090311/experiments/sdpa_rca).
The reproducer requires the mandated 3.10.20/7.14.60850/package trio identity.
It refuses mismatches; it never changes thresholds or fixtures. A future candidate
that changes this identity needs an explicit, separately reviewed identity-only
adapter before this runner can be used; do not loosen numerical acceptance.

Run from a dedicated native Windows CPython 3.10 x64 environment, never production:

```powershell
./experiments/pytorch212_wheel/preflight.ps1 -Python C:/b/be/Scripts/python.exe
python ./experiments/pytorch212_wheel/collect_build_provenance.py `
  --pytorch C:/b/pytorch --therock C:/b/therock --output C:/b/raw/source-provenance.json
./experiments/pytorch212_wheel/run_validation.ps1 `
  -Python C:/b/old/Scripts/python.exe -Output C:/b/baseline
```

The runner writes stdout/stderr, child JSON, UTC, exit codes, exact commands,
fixture and script digests. All 27 conditions use fresh processes. It does not
enable error clearing. CPU profiling names MATH; TorchDispatchMode names fused
operators. AMD_LOG_LEVEL=4 is scoped to two additional minimal launch runs.
The runner's own exit zero means collection completed, **not GPU acceptance**.
Read `summary.json` and each raw report. A report without measured numerics cannot
pass. An UNOBSERVED backend cannot pass. DEFAULT selecting MATH is MATH_PASS.

The exact source recipe, SDK dependencies and raw build attempts are separate
from runtime evidence. `compare_results.decide` requires callers to establish
matrix completeness (both fused routes, three repetitions, matching fixtures and
all controls); a single successful case is insufficient. `acceptance_statuses`
excludes only a clean UNSUPPORTED result for the optional additive-mask FLASH
probe; required fused routes remain mandatory. `valid_provenance` is a
gate over verified facts, not a substitute for collecting those facts.

CPU verification needs no torch download:

```powershell
python -m ruff check src scripts tests experiments/pytorch212_wheel
python -m ruff format --check src scripts tests experiments/pytorch212_wheel
python -m mypy src experiments/pytorch212_wheel/compare_results.py experiments/pytorch212_wheel/collect_build_provenance.py experiments/pytorch212_wheel/validate_wheel.py
python -m pytest
python -m build
python -m pip check
```

Private originals and wheels remain outside Git in `C:/b`. Published evidence
must neutralize user-profile paths, preserve original/published digest pairs,
and checksum the final bytes. Neither SDK distributions nor wheel/DLL payloads
belong in this evidence repository.

The baseline was executed while the collector scripts were uncommitted additions
to main's content. Its recorded checkout commit is therefore main, while the
recorded script digests identify the exact producers. The frozen reproducer and
reference resolve to the immutable RCA commit. A subsequent typing-only change
to the runner is recorded separately; the exact earlier source snapshot is
preserved with its matching execution digest. No acceptance criteria changed.
