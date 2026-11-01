# Admission gate CPU validation

KEEP EXPERIMENTAL. New GPU requests: **0**. Current targeted suite: **37 passed**
on Windows using CPython3.10. Tests launch only tiny CPU workers in temporary
directories. No torch/model download is required. Linux runtime execution is
NOT_RUN locally; Windows/Linux mypy targets and platform-neutral SQLite tests
cover the intended CPU CI split. Native Windows Job cases skip elsewhere.
The complete candidate suite passes **369 passed / 8 skipped**. Lint, format,
both mypy targets, isolated package build and pip check pass.

```powershell
python -m pytest tests/unit/test_experiment_admission.py -q
python -m ruff check src scripts tests experiments/admission
python -m ruff format --check src scripts tests experiments/admission
python -m mypy --platform win32 src
python -m mypy --platform linux src
```

Mandatory refusal scenarios prove worker start count0, no marker and no child:
strict FAIL, PARTIAL, absent/incomplete qualification, changed source/wheel/input/
policy/artifact/receipt, missing fixture, unsupported target and active children.
Additional cases cover missing policy dependency, wrong entry/backend argv/cwd/
environment/shell override, stale execution receipt, invalid timeouts, second
pre-spawn receipt mutation, interrupted dispatch and restart of EVALUATING work.
Thread and separate-controller process races admit exactly one worker. Refused
native Job launches own zero processes; a subprocess invocation of the prospective
CLI refuses missing qualification before fake worker execution; successful and timed-out fake workers
with descendants drain to active0. Nonzero execution persists FAIL, not PASS.

Retained test-first evidence includes initial missing-module RED; an exit-only
mutation control with three false admissions; five command/environment identity
failures; a stale receipt false PASS; a terminal-failure persistence failure; and
a missing-policy-dependency false admission, and a reviewer-found reservation
handoff contention failure. The reserved owner now waits at most five seconds for
a short metadata transaction; deterministic contention coverage proves the owner
still launches once. Long contention remains non-admitted. All corrected behaviors pass under
the final suite. Failed logs and mutation markers remain private, separately
identified; none is relabeled as a passing run.

The new prospective CLI calls Gate immediately around the existing Windows Job
spawn. The immutable historical private controller is not rewritten or claimed
safe. This validation establishes executable refusal within the documented single
ledger/trusted evaluator contract. It does not qualify GPU numerical correctness,
AUTO performance, other hardware or production serving.
