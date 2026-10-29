# Frozen Windows source-wheel build provenance

The official build completed with exit **0**. The initial CUTLASS/transport-blocked
attempt is preserved at commit `bc75cfe8b57091ce8b396ef10bb6ec1a9f938dc6` and in
[the initial-attempt snapshot](../../eval/wheel-validation/windows-gfx1151/attempts/initial-blocked/). Its earlier
BUILD_BLOCKED decision does not describe the completed continuation.

## Sources and preparation

| Input | Frozen identity |
| --- | --- |
| ROCm/pytorch release/2.12 | `60dd50f143fd9b10ac23cde8cdd34be61232b139` |
| TheRock recipe | `6d34c9b6f3e752b8f718b0c81f3070df77ce3283` |
| AOTriton runtime 0.14.50tp | `7a6963a7265d2cf0fc48c6a63464386134d32483` |
| AOTriton official image source signature | `c11b5b886a10a00f925f74133d38b3505e67ccd9` |
| xz v5.8.1 | `a522a226545730551f7e7c2685fab27cf567746c` |
| dlfcn-win32 v1.4.2 | `3b52e651f385df00045dd8966407fd9de57fc94b` |
| Windows libuv v1.47.0 | `be6b81a352d17513c95be153afcb3148f1a451cd` |

All 37 root gitlinks and three setup/native nested pins match. Native Git object
fetch batches and authenticated GitHub blob transfers recovered the immutable
objects. A child-scoped existing proxy supplied reliable transport; TLS
verification remained enabled. Local exact-source caches changed transport only.
SDK 7.14.1 core/devel/libraries/device-gfx1151 were reused without repeated SDK
downloads. [Source recovery](../../eval/wheel-validation/windows-gfx1151/continuation/source-ready-before-build.json)
and [complete pin/graph discussion](SOURCE_RECOVERY.md) record the details.

Required inputs extend beyond compiled kernels: setup checks every submodule's
markers; oneDNN configuration traverses test/example CMake files; official HIPIFY
reads MSLK utils; profiler headers require Kineto's ActivityType.h even with
USE_KINETO=OFF; FP16 installation requires PSIMD's psimd.h. Those source trees
were expanded at their existing pins. No handwritten source fix was applied.

Official HIPIFY preprocessing was left uncommitted, preserving PyTorch HEAD.
The final root tracked diff is byte-identical to the captured prebuild HIPIFY
diff. Submodule preprocessing identities are recorded separately; fmt's five
support-script differences are executable-bit metadata only, with unchanged
contents. Optional CUDA/Android/test trees outside this graph retain marker
checkouts. AOTRITON_NOIMAGE_MODE builds the runtime from source and explicitly
skips Triton; official release images are a separate pinned recipe input.

## Actual builder and configuration

```text
C:/b/be/Scripts/python.exe C:/b/therock/external-builds/pytorch/build_prod_wheels.py build --no-install-rocm --index-url https://repo.amd.com/rocm/whl-multi-arch/ --rocm-sdk-version ==7.14.1 --no-pre --rocm-extras device-gfx1151 --pytorch-rocm-arch gfx1151 --pytorch-dir C:/b/pytorch --pytorch-git-ref release/2.12 --no-build-triton --no-build-pytorch-audio --no-build-pytorch-vision --no-build-apex --enable-pytorch-flash-attention --output-dir C:/b/out/wheels
```

The child uses vcvars64, `MAX_JOBS=8`, `CMAKE_BUILD_PARALLEL_LEVEL=8` and the
frozen workflow's locally built libuv. Each Ninja invocation is capped at eight;
the nested AOTriton CXX build overlapped with root compilation, with 15 CXX
compiler processes observed. This is not a system-wide eight-process cap.

Actual CMake configuration: USE_ROCM=ON, USE_CUDA=OFF,
USE_FLASH_ATTENTION=ON, USE_MEM_EFF_ATTENTION=ON, distributed/Gloo ON, tests OFF.
CK SDPA and FBGEMM are automatically disabled by the unmodified platform/arch
guards. All **269 generated HIP compile recipes** use only `gfx1151`;
the attention command defines both fused feature macros.
CPython 3.10.20 x64, CMake 4.0.0, Ninja 1.13.2 and SDK clang-cl 23 were used;
MSVC compiler file version is `19.44.35229.0`, Windows SDK 10.0.26100.0.
The initial preflight CMake version is historical; the official build's pinned
requirements selected 4.0.0. No prohibited AOTriton source/binary override was set.

## Failures and successful completion

Earlier truncated Git/Python/image transfers were preparation failures, not
AOTriton regressions. Rejected image reconstructions never entered a candidate;
only the full archive digest
`e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e`
was accepted, then all seven extracted image files were independently checked.

The first compiler diagnostic was missing `ActivityType.h` from sparse Kineto,
classified INCOMPLETE_SOURCE_MATERIALIZATION and recovered at pinned
`b2103f78d13fde4937af010c0ef8e24313568bc5`. After compilation/linking succeeded,
the first install failure was missing PSIMD `include/psimd.h`, recovered at
`072586a71b55b7f8c584153d223e95687148a900`. No actual compiler/linker source
defect remained. The resumed frozen builder reused the objects, installed,
produced the wheel, installed it in its own environment and reported
`Builds all completed`, exit 0.

[Main compile stdout](../../eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-kineto.stdout.log.gz) ·
[Main compile stderr](../../eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-kineto.stderr.log) ·
[Successful retry stdout](../../eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stdout.log.gz) ·
[Successful retry stderr](../../eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stderr.log.gz) ·
[Final process record](../../eval/wheel-validation/windows-gfx1151/continuation/build-after-psimd-process.json) ·
[Final CMake/source/tool facts](../../eval/wheel-validation/windows-gfx1151/continuation/final-build-provenance.json).
Large logs use deterministic gzip; hashes of original, sanitized-uncompressed
and published compressed bytes are preserved in the export ledger.

## Installed artifact and measurement binding

Wheel SHA-256: `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`; 388,478,127 bytes.
The dist wheel and copied output wheel agree. The exact local output path was
installed with `--no-index` into fresh C:/b/new, with SDK 7.14.1/NumPy 2.2.6
constraints and no historical audio/vision packages. pip check succeeds.

The reviewed profile binds all 15 native torch members, including torch_cpu,
torch_python, C10 and _C.pyd, plus the exact SDK HIP DLL. Each process rehashes
artifact and installed bytes, verifies loaded paths/hashes and complete images,
and runs a fresh pip check. Supplemental inspection verifies 33 loaded stack
modules against provider wheels; HIP runtime/driver APIs both return 71460850.

Run `git_commit` records checkout context. Reviewed diagnostic edits were pending
commit during measurement. Every process/manifest producer hash was subsequently
checked against committed blobs at `3e641d97af952d3fa70a3fbec223d6faed41b047`; the
paired analysis records that binding. The final delivery preserves those exact
producer bytes. This workspace context is distinct from the independently
verified built/imported PyTorch source SHA.

[Built wheel inventory](../../eval/wheel-validation/windows-gfx1151/continuation/built-wheel-inspection.json) ·
[Identity profile](../../eval/wheel-validation/windows-gfx1151/continuation/candidate-identity-profile.json) ·
[Observed profile](../../eval/wheel-validation/windows-gfx1151/continuation/candidate-identity-profile.observed.json) ·
[Explicit installation report](../../eval/wheel-validation/windows-gfx1151/continuation/candidate-torch-install.json) ·
[Loaded native/provider hashes](../../eval/wheel-validation/windows-gfx1151/continuation/installed-native-runtime.json).
