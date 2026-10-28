# Build provenance and blocker

## Frozen sources

| Component | Identity |
|---|---|
| Evidence base, latest main fetched at start | `966265c81a74a73aa0ae6d9a3fa593b342590eb7` |
| ROCm/pytorch release/2.12 | `60dd50f143fd9b10ac23cde8cdd34be61232b139` |
| Source version.txt | `2.12.0` |
| ROCm/TheRock main observed/fetched at mission start | `6d34c9b6f3e752b8f718b0c81f3070df77ce3283` |
| TheRock main at later provenance collection | `df8b1b59ccfcc5e861ba410aa3761e38cf2dd22b` |
| AOTriton runtime source pin / 0.14.50tp tag | `7a6963a7265d2cf0fc48c6a63464386134d32483` |
| AOTriton image version | `0.14.50tp` |
| Declared gfx115x image archive SHA-256 | `e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e` |
| aotriton.cmake SHA-256 | `0e29ceff849e1a0129586f81fbb3f9cba8326d64c54730bb8950149b4d1ce71b` |

The source checkouts were new network clones, detached at those immutable SHAs.
TheRock main advanced during preparation; the attempted recipe stayed frozen at
the initially observed SHA. The later observation is retained, not silently
substituted into the first attempt. The PyTorch branch remained at the recorded
head at source collection. [Raw source provenance](../../eval/wheel-validation/windows-gfx1151/provenance/source-provenance.json).

[The immutable Windows CMake recipe](https://github.com/ROCm/pytorch/blob/60dd50f143fd9b10ac23cde8cdd34be61232b139/cmake/External/aotriton.cmake)
selects a source-built AOTriton runtime on Windows with precompiled GPU images.
The image archive above is **declared, not downloaded/installed/validated here**.
No `PYTORCH_AOTRITON_COMMIT`, `PYTORCH_AOTRITON_BASE_URL`,
`AOTRITON_INSTALLED_PREFIX` or `AOTRITON_INSTALL_FROM_SOURCE` override was inherited.
No custom AOTriton binary or repaired image payload was used.

All 37 top-level gitlink pins are preserved in the evidence, including the
blocking CUTLASS pin `da5e086dab31d63815acafdac9a9c5893b1c69e2`.
Only 11 top-level gitlinks were verified at their expected heads in the final
readiness snapshot; recursive readiness was not established. This is not a clean,
buildable checkout. dlfcn-win32 v1.4.2 and xz v5.8.1 remote tag resolutions are
preserved, but neither dependency was compiled. xz's peeled commit is
`a522a226545730551f7e7c2685fab27cf567746c`.

## Toolchain and SDK

CPython **3.10.20 x64**, MSVC tools **14.44.35207** / compiler **19.44.35229**,
VS Build Tools **17.14.41**, Windows SDK **10.0.26100.0**, Ninja **1.13.2**,
CMake **4.4.4**, PowerShell **7.6.5** were observed. The developer command prompt
was probed in a child process. `cl /Bv` prints the version and its expected
missing-input diagnostic; this was not a compilation test.

The exact **7.14.1** family was installed in `C:/b/be`: `rocm`, `rocm-sdk-core`,
`rocm-sdk-devel`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1151`. `rocm_sdk init`
expanded the development contents and exited zero; `pip check` passed.
The SDK clang reports **23.0.0git**, upstream `f30ae3e6b680ad23dcf522e668c6d4c5c90838d0`
with packaged patch identity `1efe4605317a40d2170d33617075bd5675d0d558`.
No alternate SDK was attempted. A **candidate HIP build version does not exist**;
the baseline runtime reports HIP **7.14.60850**.

Development wheel: `rocm_sdk_devel-7.14.1-py3-none-win_amd64.whl`, 655,227,267 bytes,
SHA-256 `cf05e6cdc9dc9a1ba291c76e6777b3d58767a49e34c1223f4475c740401c7532`.
The final download's S3 multipart ETag matches the server's recorded ETag, and
all ZIP-member CRCs passed. This authenticates transport consistency with that
HTTPS object; it is not an independently signed upstream checksum.
Explicit local wheel paths prevented pip from downloading the same SDK again.
[SDK package report](../../eval/wheel-validation/windows-gfx1151/build/sdk-local-install.json),
[SDK verification](../../eval/wheel-validation/windows-gfx1151/build/sdk-archive-verification.json).

The current TheRock README documents Python 3.11+ prerequisites. The mandated
3.10 interpreter successfully ran the current CLI help and SDK initialization;
full source-build compatibility with 3.10 remains unestablished.

## Actual official build attempt

The current CLI `build --help` was inspected. First invocation:

```powershell
$env:MAX_JOBS='8'
$env:CMAKE_BUILD_PARALLEL_LEVEL='8'
C:/b/be/Scripts/python.exe C:/b/therock/external-builds/pytorch/build_prod_wheels.py build `
  --install-rocm --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  --rocm-sdk-version '==7.14.1' --no-pre --rocm-extras device-gfx1151 `
  --pytorch-rocm-arch gfx1151 --pytorch-dir C:/b/pytorch `
  --pytorch-git-ref release/2.12 --no-build-triton --no-build-pytorch-audio `
  --no-build-pytorch-vision --no-build-apex --enable-pytorch-flash-attention `
  --output-dir C:/b/out/wheels
```

The checkout HEAD was the immutable PyTorch SHA; `--pytorch-git-ref` describes
release version naming and does not perform a branch checkout. Only gfx1151 was
requested. FLASH and EFFICIENT would be set ON by this recipe; **no CMake cache
was generated, so neither flag was actually configured or built**.

Start **2026-10-28 12:28:37 UTC**, end **12:31:51 UTC**, process exit **1**.
It failed during pip's SDK development-wheel transfer with ReadTimeoutError,
before compiler configuration. [Raw log](../../eval/wheel-validation/windows-gfx1151/build/build-attempt.txt),
[timestamps/exit](../../eval/wheel-validation/windows-gfx1151/build/build-attempt.json).
The SDK failure was subsequently recovered using bounded byte-range downloads,
verified concatenation and explicit local installs; it is not the remaining blocker.

## Remaining source-preparation blocker

Full recursive fetches stalled; shallow retries exposed incomplete submodule
checkouts. Exact pinned fetches repaired VMA, aiter, composable_kernel and
cudnn_frontend. Fetching CUTLASS then failed with an RPC disconnect, early EOF
and invalid index-pack output. An explicit pinned retry using process-scoped
HTTP/1.1 and OpenSSL retained certificate validation and exited **128** with an
unexpected TLS EOF, **13:03:34–13:03:44 UTC**.

The last retry had a 120-second ceiling but failed before that ceiling. Earlier
stalled preparation/download processes were terminated; their incomplete logs
are labeled as such. No security policy, global Git configuration or registry
setting was changed. No compiler or PyTorch source fix was attempted.
[Retry command and exit](../../eval/wheel-validation/windows-gfx1151/build/submodule-pinned-retry.json),
[raw TLS failure](../../eval/wheel-validation/windows-gfx1151/build/submodule-pinned-retry.txt),
[readiness snapshot](../../eval/wheel-validation/windows-gfx1151/provenance/source-readiness.json).

## Candidate artifact

**None.** No candidate filename, size, ABI/platform tag, wheel hash, dependency
manifest, loaded binaries, image signatures or GPU result exists. `C:/b/out`
contains no wheel and no PyTorch CMake cache exists. The OCR project's separate
CPU package build is not a PyTorch wheel build. No candidate install was attempted.

Smallest next action: complete `git submodule update --init --recursive` at the
recorded immutable pins using a working GitHub transfer path, verify the clean
source tree, then rerun the frozen official builder in an MSVC developer shell.
Use the already installed SDK with `--no-install-rocm` to avoid unnecessary SDK
reinstallation. Preserve the exact SHA and bounded parallelism, inspect the wheel,
install by explicit path in `C:/b/new`, and run the unchanged matrix before any
claim that fused SDPA is fixed.
