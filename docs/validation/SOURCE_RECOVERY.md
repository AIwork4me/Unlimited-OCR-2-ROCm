# Frozen Windows ROCm source recovery

The continuation recovered the pinned source dependencies without changing
PyTorch `60dd50f143fd9b10ac23cde8cdd34be61232b139` or TheRock
`6d34c9b6f3e752b8f718b0c81f3070df77ce3283`. The original blocked
attempt remains in the baseline publication history.

## Dependency graph and materialization

Frozen `setup.py:530` checks every declared submodule folder for an upstream
marker file, even when that component is excluded from the configured build.
It separately requires FBGEMM's nested asmjit CMakeLists. Therefore a missing
CUTLASS checkout can block setup before the Windows ROCm target graph is used.
The exact CUTLASS pin `da5e086dab31d63815acafdac9a9c5893b1c69e2` was recovered
and its setup markers materialized. Its CUDA kernels are guarded by USE_CUDA
in `aten/src/ATen/CMakeLists.txt:412` and `:873`; the actual cache has
USE_CUDA=OFF and USE_ROCM=ON.

The configured native dependencies include protobuf/ONNX, cpuinfo, pthreadpool,
FP16/FXdiv, pybind11, SLEEF, oneDNN/ideep, Gloo and Windows libuv. Their source
trees were materialized at the gitlinks. oneDNN's CMake traverses tests/examples
directories during configuration even with tests disabled, so its sparse
checkout was expanded. FBGEMM was automatically disabled by the unmodified
platform guards; its setup prerequisites were nevertheless recovered.

Official HIPIFY also reads MSLK's `include/mslk/utils/tuning_cache.cuh`
(`tools/amd_build/build_amd.py:224`). That directory was expanded before
rerunning the frozen TheRock HIPIFY command successfully. The generated source
manifest records the preprocessing; HEAD remains the frozen PyTorch commit.
No hand-written source patch was applied.

The frozen TheRock Windows workflow additionally builds libuv v1.47.0, peeled
commit `be6b81a352d17513c95be153afcb3148f1a451cd`. The same prerequisite was
built locally and supplied through child-process `libuv_ROOT`, preserving Gloo.

AOTriton's runtime is built from `7a6963a7265d2cf0fc48c6a63464386134d32483`
(0.14.50tp). Its xz dependency is `a522a226545730551f7e7c2685fab27cf567746c`
(v5.8.1); dlfcn-win32 is `3b52e651f385df00045dd8966407fd9de57fc94b`
(v1.4.2). The official recipe uses AOTRITON_NOIMAGE_MODE for the runtime and
downloads the release images separately. Frozen AOTriton CMake lines 381–385
explicitly skip Triton in this mode. Its Triton submodule was consequently
excluded from transport, without changing any compiled source or CMake option.

## Transfer and integrity

Large GitHub HTTPS transfers repeatedly ended with TLS EOF/early EOF. Recovery
used smaller exact-object Git fetch batches and authenticated GitHub blob
transfers, with Git object identities verified before use. The existing local
trusted proxy was scoped to child processes. TLS verification stayed enabled.
Local Git URL mappings supplied the verified AOTriton/xz source caches to the
official external-project clone steps; no binary override was supplied.

The official gfx115x AOTriton image archive is 231642586 bytes with SHA-256
`e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e`.
It was reconstructed through checked byte ranges and accepted only after its
full digest matched both the frozen CMake pin and GitHub release digest.
Intermediate corrupt transfers were rejected and never used for a candidate.
The archive entered the official CMake download cache, where the unmodified
recipe performed its verification and extraction.

Python tool transfers were similarly checked against official PyPI digests.
A truncated NumPy transfer inside AOTriton's isolated build environment caused
one configure failure; preinstalling the verified wheel and completing the
declared requirements repaired that environment. This was a transfer/environment
failure, with no compiler or linker defect and no evidence of an AOTriton GPU
regression. SDK 7.14.1 was reused throughout.

## Exact gitlinks

The machine-readable recovery receipts record each expected and observed HEAD,
transfer commands, exit status, and pre-HIPIFY clean-root check. All 37 root
gitlinks and three setup/native nested pins matched before the resumed build.
Sparse marker checkouts intentionally omit components outside this Windows
ROCm graph; they do not establish that every upstream optional target can build.

| Path | Exact commit | Checkout at readiness |
| --- | --- | --- |
| `android/libs/fbjni` | `7e1e1fe3858c63c251c637ae41a20de425dde96f` | sparse markers |
| `third_party/FP16` | `4dfe081cf6bcd15db339cf2680b9281b8451eeb3` | full |
| `third_party/FXdiv` | `b408327ac2a15ec3e43352421954f5b1967701d1` | full |
| `third_party/NNPACK` | `c07e3a0400713d546e0dea2d5466dd22ea389c73` | sparse markers |
| `third_party/NVTX` | `3ebbc93ded7285963bff932c678fa367eb393ba6` | sparse markers |
| `third_party/VulkanMemoryAllocator` | `1d8f600fd424278486eade7ed3e877c99f0846b1` | sparse markers |
| `third_party/XNNPACK` | `51a0103656eff6fc9bfd39a4597923c4b542c883` | full |
| `third_party/aiter` | `9a469a608b2c10b7157df573a38d31e5bf4038b4` | sparse markers |
| `third_party/benchmark` | `299e5928955cc62af9968370293b916f5130916f` | sparse markers |
| `third_party/composable_kernel` | `cd9e41c4eae94f11e7d6e39751e8e13f46653030` | sparse markers |
| `third_party/cpp-httplib` | `4d7c9a788de136071ccf0dd4e96239151e2adadb` | full |
| `third_party/cpuinfo` | `f858c30bcb16f8effd5ff46996f0514539e17abc` | full |
| `third_party/cudnn_frontend` | `a91f0e04dcea10515f0f776fc5a89535e316a9c8` | sparse markers |
| `third_party/cutlass` | `da5e086dab31d63815acafdac9a9c5893b1c69e2` | sparse markers |
| `third_party/fbgemm` | `c246916f9e3804eacc3c95058e51cce02ae00fff` | full |
| `third_party/flash-attention` | `fec3a6a18460c1b40f097208d4c16fe8964a679d` | sparse markers |
| `third_party/flatbuffers` | `a2cd1ea3b6d3fee220106b5fed3f7ce8da9eb757` | full |
| `third_party/fmt` | `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f` | full |
| `third_party/gemmlowp/gemmlowp` | `3fb5c176c17c765a3492cd2f0321b0dab712f350` | full |
| `third_party/gloo` | `3135b0b41b67dde590eef0938a0bf3d6238df5f7` | full |
| `third_party/googletest` | `52eb8108c5bdec04579160ae17225d66034bd723` | sparse markers |
| `third_party/ideep` | `e539e0f9774e2018f0d56fe865da66581f692e3d` | full |
| `third_party/ittapi` | `0c57540822deb5dae43bef6c1cc9b3be4772a033` | full |
| `third_party/kineto` | `b2103f78d13fde4937af010c0ef8e24313568bc5` | sparse markers |
| `third_party/kleidiai` | `d7770c89632329a9914ef1a90289917597639cbe` | sparse markers |
| `third_party/mimalloc` | `048d969a1c5ee2fb89c226298f41ea38445546ef` | full |
| `third_party/mslk` | `3d332d1c0c0ac7765852c97b3979c9ef913e037f` | markers + HIPIFY utils |
| `third_party/nlohmann` | `55f93686c01528224f448c19128836e7df245f72` | full |
| `third_party/onnx` | `e709452ef2bbc1d113faf678c24e6d3467696e83` | full |
| `third_party/pocketfft` | `0fa0ef591e38c2758e3184c6c23e497b9f732ffa` | full |
| `third_party/protobuf` | `d1eca4e4b421cd2997495c4b4e65cea6be4e9b8a` | full |
| `third_party/psimd` | `072586a71b55b7f8c584153d223e95687148a900` | sparse markers |
| `third_party/pthreadpool` | `4fe0e1e183925bf8cfa6aae24237e724a96479b8` | full |
| `third_party/pybind11` | `f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8` | full |
| `third_party/python-peachpy` | `f45429b087dd7d5bc78bb40dc7cf06425c252d67` | sparse markers |
| `third_party/sleef` | `5a1d179df9cf652951b59010a2d2075372d67f68` | full |
| `third_party/tensorpipe` | `2b4cd91092d335a697416b2a3cb398283246849d` | sparse markers |
| `third_party/fbgemm/external/asmjit` | `a3199e8857792cd10b7589ff5d58343d2c9008ea` | full |
| `third_party/fbgemm/external/cpuinfo` | `161a9ec374884f4b3e85725cb22e05f9458fdc93` | full |
| `third_party/ideep/mkl-dnn` | `03c022d3ffdcee958cfacbe720048e725fdf644c` | full |

