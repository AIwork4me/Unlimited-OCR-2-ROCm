# Current PyTorch 2.12 Windows wheel / gfx1151 validation

## Executive summary and final decision

**BUILD_BLOCKED — the current source's fused-SDPA behavior is unmeasured.**
The original official AMD wheel's RED was reproduced on the physical Windows
gfx1151 device. The actual TheRock build attempt failed before compilation while
installing SDK development dependencies. The exact SDK was subsequently recovered
and initialized, but pinned PyTorch submodule preparation failed on CUTLASS GitHub
transfers. No candidate wheel was built or installed. This does **not** establish
REGRESSION_PERSISTS for current source, and the changed AOTriton pin does **not**
establish FIX_VERIFIED.

`SOURCE_BUILT_WHEEL_VERIFIED = false`

`OFFICIAL_PUBLISHED_WHEEL_VERIFIED = false` (no newly corrected published wheel
was validated; the historical official wheel was independently tested and failed).

## Exact machine

Native Windows 11 Home Chinese, **10.0.26200/build 26200**; **AMD Ryzen AI MAX+ 395**,
**AMD Radeon 8060S / gfx1151** confirmed by the actual HIP device properties;
display driver **32.0.22032.6002**, WMI driver date 2025-12-15. Marketed 128 GB
unified memory; WMI reports 132,766,306,304 bytes accessible system RAM.
The preflight snapshot had 101,564,320 KiB free RAM and 422,856,482,816 bytes free
disk. The reported APU GPU memory allocation is not dedicated VRAM.
CPython **3.10.20 x64** was used for baseline, SDK/build and CPU checks.
[Preflight](../../eval/wheel-validation/windows-gfx1151/provenance/preflight.json),
[runtime identity](../../eval/wheel-validation/windows-gfx1151/provenance/baseline-provenance.json).

## Original official wheel

The fresh `C:/b/old` environment used the exact mandated versions:
torch **2.12.0+rocm7.14.1**, torchvision **0.27.0+rocm7.14.1**, torchaudio
**2.11.0+rocm7.14.1**, SDK runtime **7.14.1**, HIP **7.14.60850**.
`torch.version.git_version = 098172edc887399a593f0c74ae5c40b1fe043801`;
`torch.version.rocm` is absent (recorded as null). The imported torch and all five
important loaded DLLs were within the new isolated baseline environment.

`torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl` is **114,472,348 bytes**,
SHA-256 **`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`**,
identical to the immutable RCA's historical artifact. Cached wheel bodies were
inspected and hashed; no cp313 wheel was used. AMD's current index lists the
filenames but does not advertise independent SHA-256 fragments for those files.
That origin-authentication limit is retained explicitly.
[Wheel manifest and loaded-byte checks](../../eval/wheel-validation/windows-gfx1151/provenance/baseline-wheel-manifest.json),
[index observations](../../eval/wheel-validation/windows-gfx1151/provenance/official-wheel-verification.json),
[install resolution](../../eval/wheel-validation/windows-gfx1151/baseline/install-report.json).

The first isolated pip install failed on a missing cached `rocm` wheel during
concurrent installation; TheRock's cache purge is the suspected race, not a
separately controlled experiment. A dedicated-cache retry was terminated
while re-downloading an already cached artifact; the final installation used
explicitly preserved cached wheel files as `--find-links`, with the AMD index
retained for dependencies. Exact commands/failures and final pip report are
preserved. The final baseline `pip check` passed; production Python, venv and
drivers were not modified.

## Upstream snapshot, pins and actual build

PyTorch release/2.12: **`60dd50f143fd9b10ac23cde8cdd34be61232b139`**;
source version **2.12.0**. TheRock recipe: **`6d34c9b6f3e752b8f718b0c81f3070df77ce3283`**,
the initial main head; later main advancement is preserved in provenance.
AOTriton runtime/source **0.14.50tp / `7a6963a7265d2cf0fc48c6a63464386134d32483`**;
declared gfx115x image archive SHA-256
**`e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e`**.
These are source declarations, not candidate binary identities.

The official `external-builds/pytorch/build_prod_wheels.py` CLI and README were
inspected; only PyTorch/gfx1151 was requested, with both fused routes enabled and
bounded parallelism 8. No Triton, vision, audio, Apex or unrelated project was
compiled. MSVC, Windows SDK, Ninja, compiler and exact SDK details, timestamps,
process exits, flags and commands are in [BUILD_PROVENANCE.md](BUILD_PROVENANCE.md).
FLASH/EFFICIENT configuration was never reached. No source/SDK incompatibility
has been demonstrated by a compiler result.

The initial SDK download timeout was recovered. The remaining blocker is an
incomplete pinned source checkout: CUTLASS fetches fail with an RPC early EOF
and then TLS unexpected EOF, final exit **128**. Only 11/37 top-level gitlinks were
verified ready. No CMake cache or candidate wheel exists. Candidate filename,
SHA-256, size, ABI/platform tag, wheel dependency manifest, HIP build version and
installed candidate binaries/images are all **N/A**, not assumed from the recipe.
No candidate install or GPU test was attempted under this incomplete provenance.

## Installed baseline runtime/image contract

The loaded `aotriton_v2.dll` SHA-256 is
`259eeb6f064eaf506a032bdb65dc9b52c6d5b7d9128de5372e5c5b1b7c2b5d22`.
Its packaged header declares **0.12.50 / `05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920`**;
this is metadata, not an exported live DLL commit identity. Header, DLL bytes,
wheel contents, and prior immutable RCA corroborate the original packaging pair.
The installed image signature identifies **aotriton0.13 /
`6e00ef3e335b45dfb49065259533b59c68995bfe`**, with gfx115x ZIP images;
`attn_fwd.zip` SHA-256 is
`adfb2fd4d07046cb40a5cf853303a01eb0cf79bc35094d033cf85955a2906850`.
All important loaded DLL hashes match their original wheel members, including
the HIP runtime. No image or DLL was swapped.

## Physical GPU results, dispatch and accuracy

[Full matrix and numeric results](GPU_TEST_MATRIX.md),
[raw process manifest](../../eval/wheel-validation/windows-gfx1151/baseline/processes.json),
[derived summary](../../eval/wheel-validation/windows-gfx1151/baseline/summary.json).

There were **27 fresh-process runs**: **21 fused FAIL**, **4 profiled MATH_PASS**,
**1 copy-only CONTROL_PASS**, **1 clean UNSUPPORTED** additive-mask FLASH probe.
The key supported fused conditions each had three repetitions. DEFAULT scalar
actually dispatched FLASH, and DEFAULT vision actually dispatched EFFICIENT.
The historical RED is reproduced; BASELINE_DRIFT is not indicated.

Every MATH result was finite and within the frozen BF16 atol/rtol **0.03/0.03**
against the independent CPU FP32 reference on exactly quantized inputs. Vision
max absolute error was **0.0038219690322875977**, with zero failing or nonfinite
elements. Fused numerical results remain **unmeasured** after the first HIP fault;
missing values are never treated as correct. Candidate accuracy and actual
FLASH/EFFICIENT/MATH dispatch remain **NOT_RUN**.

The original reproducer and reference remain byte-identical to the
[immutable RCA](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/tree/6b643f51d4078c4bb841f6a48734cf2206090311/experiments/sdpa_rca).
Seed **7992**, fixture creation, mask/shape semantics, reference and thresholds
were not changed. The producer digests and exact runner snapshot resolve scripts
that were uncommitted additions when the baseline ran; the recorded main checkout
commit is not presented as containing those new files.

## HIP fault and module launches

Before SDPA, the original processes report HIP error **0**; immediately after
fused SDPA they report **1**. Synchronization can return normally while the error
remains pending. Real downstream copies/arithmetic then fail where applicable.
There was no acceptance error clearing. Both AMD_LOG_LEVEL=4 logs show **zero
module-load calls**, then **null function, block (0,0,0), hipErrorInvalidValue**.
[Extracted trace lines with original digests](../../eval/wheel-validation/windows-gfx1151/provenance/baseline-launch-evidence.json).
No successful candidate module load/launch or candidate HIP-clean state exists.

## Controlled-comparison limits and risks

This is a reproduced original RED plus a blocked candidate build, **not** a
completed old/new artifact comparison. No current source-built or newly published
official fixed wheel was installed. No inference is made from a source-pin change,
and no particular upstream commit is credited with repairing the defect.
The unprepared submodules, network failures and current TheRock README's 3.11+
prerequisite prevent a CPython 3.10 source-build compatibility claim.
The additional MATH controls cover this BF16 forward matrix, not training,
backward, other dtypes, GPUs, drivers or operating systems. No SAM/model/PDF run
was needed or performed. Existing production code and its scoped MATH adapter
remain untouched.

## Engineering checks and separate review

CPU-only Ruff, formatting, mypy, pytest, project package build and isolated pip
checks have their own [raw check artifacts](../../eval/wheel-validation/windows-gfx1151/checks/).
Hosted Ubuntu/Windows CPU CI is required on the PR's final head; it does not
constitute physical GPU validation. The original GPU matrix was executed locally
and independently of hosted CI. [Read-only review and dispositions](REVIEW.md).

## Next action and publication

**Smallest next action:** restore successful GitHub source transfers, complete
the exact pinned recursive submodules, check source readiness, then rerun the
official builder using the installed SDK. Inspect/hash the output, install by
explicit local path in a fresh `C:/b/new`, and run the same-input matrix before
changing the final decision. [Conditional upstream recommendation](NEXT_UPSTREAM_PR.md):
ROCm/TheRock for a packaged-wheel regression gate if the candidate passes;
responsible upstream source repository only after a candidate failure is isolated.

[Decision JSON](../../eval/wheel-validation/windows-gfx1151/decision.json),
[comparison JSON](../../eval/wheel-validation/windows-gfx1151/comparison.json),
[evidence SHA256SUMS](../../eval/wheel-validation/windows-gfx1151/SHA256SUMS.txt).
The wheel binaries and SDK payloads remain local; no candidate binary is hosted
on GitHub. The dedicated validation PR targets main and does not modify issue #2,
RCA PR #3, releases, PyPI or any upstream repository.
