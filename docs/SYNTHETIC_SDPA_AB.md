# Synthetic candidate-wheel MATH/AUTO preparation

Real-document evaluation: **BLOCKED_DATA**. No real-document folder is authorized.
The existing PDF and MPDocBench, MarathonDocBench and SlideVQA payloads are not
inputs to this experiment. Do not transfer them or infer usage permission from
their presence. No new PDFs, images, ground truth, raw OCR or token IDs belong
in Git. Only source, tests, sanitized metrics, hashes and technical evidence
may be published. Original synthetic fixtures already present in the reviewed
source are read locally; they are not republished in this change.

This is experimental infrastructure, **PREPARED_ONLY**, not a completed GPU
benchmark or an AUTO qualification. Default production vision remains MATH;
experimental PR #5 stays draft/unmerged. Current P0 is HOLD. Both driver and
worker require a matching integration-base P0 PASS receipt before GPU work.
No physical run was made to bypass that dependency. Preparation does not need
torch, network access, model loading or a GPU.

## Frozen design

Both arms require candidate SHA256
`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`.
The published original-wheel SHA256 `415d2dbd...` is refused as either arm.
The candidate is a local experimental artifact, not an AMD-supported release.
The preserved Windows CPython 3.10 / torch 2.12.0+rocm7.14.1 /
torchvision 0.27.0+rocm7.14.1 / torchaudio 2.11.0+rocm7.14.1 /
triton-windows 3.7.1.post27 stack and gfx1151 remain mandatory.

Ten original synthetic cases: English, Chinese, mixed text, table, arithmetic,
Unicode-path recovery, 2/3/5-page native documents, and the cross-page table.
Each uses identical image hashes, prompt/preprocessing and token budgets in
both arms. The model lock remains revision
`9cf077c4b206e0e2a31879df937b0b0bcba3661c`. No caps, decoder, DASH or source
model algorithms are changed.

- One diagnostic request per mode/case runs in its own **diagnostic process**.
  Both modules must be observed: MATH/MATH for control; SAM EFFICIENT and
  CLIP FLASH for AUTO. Missing dispatch or silent MATH selection fails.
  Retain actual QKV/mask layouts, numerical checks (BF16 atol/rtol 0.03/0.03),
  finite downstream copy/arithmetic and zero sticky HIP errors. These are
  operator-level dispatch/numerical diagnostics; they do not replace the
  independent lower-level PAL/HIP launch evidence from candidate qualification.
- A separate **measurement process** loads one model and first warms each
  mode/case once. Six paired repeats per case follow in alternating AB/BA
  order, three each. Case order is seeded and frozen before inference.
  Trial order, all repeats and failed/missing attempts remain in the denominator.
  Lightweight operator observation runs symmetrically in timed forwards too;
  an unobserved backend or measured AUTO-to-MATH fallback rejects the trial.
- Each process rechecks wheel bytes, every installed native member/AOTriton
  image, loaded native hashes/paths, pip check, hardware, model hashes and
  dependencies. Frozen historical helper blobs are exported locally from
  immutable commits; every worker verifies their hashes and all current source
  hashes before import. Optimized Python is rejected because historical helpers
  use assertions. No production AUTO option is added: experimental wrappers
  replace only the vision forwards in the private worker process.
- Preserve original correctness gates (CER <=0.02, exact important numbers and
  ordered canonical PAGE/BLOCK structure), plus strict same-candidate synthetic
  raw/prompt-ID/output-ID hash parity across all phases and repeats. Stop must
  be native, token counts must agree, and all 12 BF16 DASH prefill calls must
  run. Exact parity is deliberately predeclared; differences fail rather than
  causing automatic tolerance relaxation.
- AUTO's maximum measured allocated peak must be <=1.05 times MATH's peak for
  every case. Reserved memory is reported separately because the measurement
  process retains a shared allocator pool across counterbalanced modes.

## Timing and memory interpretation

Model load/hash verification time is recorded separately in each local
`load.json`. Native infer timing starts/ends with GPU synchronization around
the original native `infer` call, before output serialization. Wrapper E2E
timing additionally includes native-result validation/conversion and output
writes. Summed synchronized SAM/CLIP forward time supplies vision latency.
Diagnostics include references/profiling and never enter measurement summaries.
Both measured arms retain the same small synchronization/HIP/timing observers;
these are instrumented synchronized measurements, not zero-overhead timing.

Report individual measured repeats plus median, min/max, sample standard
deviation and paired MATH/AUTO latency ratios. No significance or real-document
speedup claim follows from synthetic cases. Allocated/reserved peaks are APU
GPU allocator usage, not dedicated VRAM or total system memory. Process RSS
is separate. Diagnostic host/numerical memory does not enter measured peaks.

## Commands and local artifacts

Run every command from the primary repository root. The experiment source
checkout is `.worktrees/p230`; choose fresh repository-local output directories.
An identity profile is a local path-bearing receipt and must not be committed.
Its wheel/native/image hashes must be the already verified candidate's; only
an explicitly validated environment relocation may change its prefix metadata.

```powershell
& .b/ia-qa/Scripts/python.exe .worktrees/p230/experiments/p2_ab/run.py prepare `
  --identity-profile .b/p230/candidate-profile.json `
  --output .b/p230/prepared-final --repeats 6 --seed 0
```

Preparation creates a private frozen plan, local immutable helper copies and
`public-preparation.json`. Only the explicitly whitelisted public file is
eligible for publication. The plan's absolute paths are local-only.

After P0 is independently closed for integration base
`82951c73b28bf735f5655e6f5c3306cd170a21c2`, and the producer checkout is
committed/clean, the explicit opt-in command is:

```powershell
python .worktrees/p230/experiments/p2_ab/run.py execute `
  --plan .b/p230/prepared-final/plan.json --output .b/p230/physical-new `
  --p0-receipt .b/p230/p0-pass.json --execute-gpu
```

The PASS receipt must have `matrix.P0 == "PASS"` and the matching
`source_commit`. Never edit a historical HOLD receipt to manufacture PASS.
If integration source changes, review/rebuild the source binding and regenerate
the plan. Parent execution requires psutil and the existing physical candidate
interpreter from the profile; no dependency installation or environment mutation
is performed by this harness. A 32 GiB system-available admission floor precedes
each child; the 8 GiB runtime floor and 1800-second default timeout stop it.
Child/observed-descendant identities, exit/error/timeout and cleanup are retained.
The parent creates a kill-on-close Windows Job Object, assigns the child while
suspended, then resumes its primary thread. Descendants inherit job ownership;
cleanup does not rely on polling to discover them. A child exiting with active
descendants is FAIL and the whole job is terminated and awaited. The immediate
grandchild lifecycle has a native Windows CPU test. Physical GPU hard-kill and
recovery qualification remain a separate P3 reliability gate.

Each trial publishes RUNNING identity then an atomic create-only final receipt.
Raw text, token IDs and traces remain private. `public-summary.json` exports
only controlled identifiers, status codes, numeric metrics and hashes. It
omits speedup aggregates if any required trial/identity/quality/dispatch gate
fails. Keep all raw attempts, including failures; never reuse an output folder.

Unsupported/unrun: real documents, official full datasets, >5 pages, >8192
new tokens, other GPUs/OS/dtypes, backward, concurrent inference, physical
OOM/hard-kill qualification and statistical/general workload claims.
Recommendation remains **KEEP EXPERIMENTAL** until physical evidence passes.

## Preparation validation (2026-10-30)

The Windows CPU suite passed **327 tests / 8 explicit skips**; the 40 focused
experiment tests passed, including native Job Object descendant cleanup.
Ruff check/format, production mypy, wheel/sdist build and CPU pip check passed.
An independent reviewer reproduced and rechecked plan integrity, public export
filtering, six-pair minimum, finite deadlines and the immediate-grandchild race.
No remaining blocking finding was reported for preparation.

A real local preparation hashed the preserved candidate wheel and checked its
installed native/AOTriton members without importing torch. Installed metadata
matches the mandatory Windows trio, Triton and Transformers versions. The plan
has 10 cases / 160 trials (20 diagnostic, 20 warmup, 120 measured). The current
HOLD receipt is refused before a worker launches or an execution directory is
created. Current GPU dispatch, quality, latency and memory results are **NOT RUN**;
the numbers above are protocol/test counts. No physical qualification or speedup
is claimed. `eval/p2/synthetic-preparation.json` is the sanitized preparation
manifest; raw plans, profiles, logs and reviewer repros remain under `.b/`.
