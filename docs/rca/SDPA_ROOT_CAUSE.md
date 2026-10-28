# Windows gfx1151 SDPA root-cause report

## 1. Executive summary

The mandated Windows cp310 torch wheel packages an AOTriton **0.12-family runtime with 0.13-family kernel images**. The runtime expects loose AKS2 files, while the device package supplies ZIP archives. On the physical Radeon 8060S, fused SDPA attempts a null-function, zero-block launch and leaves HIP error 1. Changing **only the image payload** in a separate diagnostic environment repairs numerical correctness and kernel launch; restoring the original images restores the failure. Production code, the scoped MATH adapter, official weights and the validated venv remain unchanged.

## 2. Status

**CONFIRMED**, specifically the incompatible packaged AOTriton runtime/image contract and its causal relationship to this failure. No claim that every Windows ROCm SDPA configuration is affected.

## 3. Highest verified level

**L4 — root cause confirmed by a controlled A/B/A.** L0: reproducible symptom; L1: SDPA precedes the sticky error; L2: actual FLASH/EFFICIENT ATen operators; L3: packaged AOTriton runtime/image pair; L4: images-only FAIL→PASS→FAIL, fixed DLLs and driver, independent CPU correctness. The English handoff is ready for maintainers; no upstream acceptance or L5 review is claimed.

## 4. Exact affected environment

Native Windows 11 build 26200, Ryzen AI MAX+ 395, Radeon 8060S/gfx1151, display driver **32.0.22032.6002**, CPython **3.10.20 x64**, torch **2.12.0+rocm7.14.1**, torchvision **0.27.0+rocm7.14.1**, torchaudio **2.11.0+rocm7.14.1**, HIP **7.14.60850**. [Environment](../../eval/rca/gfx1151-rocm7141/environment.json), [baseline provenance](../../eval/rca/gfx1151-rocm7141/provenance-baseline.json).

The secondary torch-only venv uses the same required stack. Its complete package inventory records ordinary dependency differences from the OCR venv (including Pillow); **within A/B/A its entire package inventory is fixed**. No driver, SDK, compiler or wheel upgrade occurred. Standalone torch-only reports record no loaded Python Triton, Transformers, HF or SGLang modules.

## 5. User-visible impact

Unadapted SAM can return from SDPA and then raise hipErrorInvalidValue in window_unpartition's contiguous copy. The original source path is verified by the [SAM failure](../../eval/rca/gfx1151-rocm7141/model-final/sam-default.json). A successful synchronize is insufficient: HIP's last error is still 1. Clearing it hides a failed attention result.

## 6. Minimal standalone reproduction

[Instructions and complete A/B/A commands](../../experiments/sdpa_rca/README.md). No OCR files or weights are needed:

```powershell
python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --output scalar-default.json
python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --backend MATH --output scalar-math.json
```

These are the smallest possible positive Q/K/V dimensions. The no-mask scalar DEFAULT route actually invokes FLASH_ATTENTION. A single-element contiguous operation can be a no-op; the scalar reproducer therefore diagnoses the error through non-clearing HIP peek and exits 1. For a real subsequent-copy exception use the default [54,12,196,64] BF16+bias condition. With bias, the smallest demonstrated fused case is [1,1,1,8]; D=1+bias selects working math, and forced efficient rejects its alignment.

## 7. Expected and actual results

Expected: finite attention matching the CPU reference, no pending HIP error, subsequent copies/arithmetic/casts succeed. Original DEFAULT and supported FLASH/EFFICIENT routes leave error 1. Original MATH and copy-only controls pass. Forced FLASH with an additive bias and cuDNN are recorded **UNSUPPORTED**, not faulty executed kernels. [All child records](../../eval/rca/gfx1151-rocm7141/experiment-manifest.json), [matrix](TEST_MATRIX.md).

## 8. Independent numerical reference

The CPU generator seed is 7992. Q/K/V and bias are quantized to the tested dtype on CPU, hashed, and transferred unchanged. Explicit CPU FP32 matrix multiplication, scaling, masks, softmax and multiplication by V implement the reference. The reference never calls SDPA or computes on GPU. Unit tests use hand-derived averages, mask polarity, causality and scale=0.

Tolerances were committed **before** the first matrix: BF16 atol/rtol 0.03/0.03; FP16 0.005/0.005; FP32 0.0001/0.0001. Counts include all nonfinite elements as failures. Relative denominator floor is 1e-6. Large relative errors near zero do not by themselves fail the combined absolute/relative criterion.

At the original shape, baseline MATH max error is **0.0038219690**, mean **0.0001298194**, zero failing elements of 8,128,512. Compatible-images DEFAULT max error is **0.0038219690**, mean **0.0001888917**, zero failures. The isolated **NON-ACCEPTANCE** clearing diagnostic on original DEFAULT finds max error **1.57019746**, mean **0.09218924**, 6,381,920/8,128,512 failures (**78.51277%**). Its pending error was recorded before clearing and its status remains FAIL. Acceptance processes do not copy/compare corrupted output after detecting a sticky error. Numerical fields absent in those reports mean unmeasured, never correct. [Diagnostic](../../eval/rca/gfx1151-rocm7141/isolation/diagnostic-clearing.json).

## 9. Actual dispatch

| Input | Public request | Observed ATen implementation | Native component evidence |
|---|---|---|---|
| BF16 [54,12,196,64], additive bias | DEFAULT / EFFICIENT | `aten._scaled_dot_product_efficient_attention.default` | torch_hip imports AOTriton v3 attn_fwd |
| BF16 [1,1,1,1], no mask | DEFAULT / FLASH | `aten._scaled_dot_product_flash_attention.default` | AOTriton DLL plus HIP module-launch trace |
| Smaller reference condition | MATH + CPU profiler | `aten::_scaled_dot_product_attention_math` | bmm/softmax decomposition, no fused attention call |
| Original shape + bias | FLASH only | UNSUPPORTED | exact warning retained |
| Tested ROCm wheel | cuDNN only | UNSUPPORTED | wheel has no cuDNN attention |

TorchDispatchMode records actual operators. Composite MATH decomposes before this mode sees it, so those raw reports correctly say UNOBSERVED; the separate [CPU profiler run](../../eval/rca/gfx1151-rocm7141/isolation/profiler-MATH.json) supplies the named MATH operator. An uninstrumented run fails too. Capability flags are recorded separately and are not treated as dispatch proof. [PE imports](../../eval/rca/gfx1151-rocm7141/torch-hip-imports.txt) connect torch_hip to the loaded aotriton_v2.dll. CPU profiling alone cannot identify GPU kernel names. In failing traces **no valid attention device kernel is loaded/launched**; do not invent one.

## 10. First failing operation

SDPA itself leaves the first observed error: before SDPA=0; immediately afterward=1; synchronization returns normally and preserves 1. DEFAULT on the original vision layout selects efficient; no-mask scalar selects flash. Downstream tensor copy is an error-reporting site. The CPU reference and fixture creation occur before the first GPU checkpoint. Copy-only starts clean and passes.

## 11. HIP sequencing

[Full compact traces and extracted launch records](../../eval/rca/gfx1151-rocm7141/hip-launch-evidence.json) bind each trace to its producing run and original SHA-256. A: **0** hipModuleLoadData[Ex] calls, then `hipModuleLaunchKernel` with null function and block (0,0,0), returning hipErrorInvalidValue. B: **1** successful module load, a non-null function and block (128,1,1), then success. Restored A repeats the null launch. Acceptance uses only hipPeekAtLastError; downstream PyTorch can consume the error, but the run retains its earlier FAIL. Trace addresses/timings are observations, not stable kernel identifiers.

## 12. Hypotheses

[H1–H8 predictions, falsification experiments, commands and dispositions](HYPOTHESES.md). The caller, a preceding custom kernel, Python Triton necessity and a particular large BF16/bias shape are excluded as necessary causes. A general driver or HIP ABI defect is not needed to explain the failure; identical runtime/driver work with compatible images. This does not certify all other driver operations.

## 13. Source and binary provenance

[Controlled invariants](../../eval/rca/gfx1151-rocm7141/controlled-ab.json), [wheel bytes](../../eval/rca/gfx1151-rocm7141/wheel-body-verification.json), [source index](../../eval/rca/gfx1151-rocm7141/source-provenance.json).

- Exact cp310 torch wheel SHA-256: `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`. Its actual cached ZIP body was independently hashed and WHEEL tag checked. Embedded aotriton_v2.dll and torch_hip.dll hashes equal the actually loaded files. A cp313 cache candidate is explicitly distinguished and not used as baseline.
- Runtime header: 0.12.50, `05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920`. Loaded DLL SHA-256 `259eeb6f064eaf506a032bdb65dc9b52c6d5b7d9128de5372e5c5b1b7c2b5d22` (15,560,192 bytes). The header is build metadata, not a live DLL Git-SHA export; imports, wheel bytes, old-loader source and A/B behavior corroborate it.
- Original images signature: `6e00ef3e335b45dfb49065259533b59c68995bfe`, aotriton0.13; attn_fwd.zip SHA-256 `adfb2fd4d07046cb40a5cf853303a01eb0cf79bc35094d033cf85955a2906850`. There are ZIP archives and no loose AKS2 files.
- Diagnostic official 0.12.50tp images archive SHA-256 `3177387a15c678b30057f4584d1fc1b8f8db56163890cb5c98f27450209f5a7b`. Signature `fe12b77d…`, aotriton0.12; 1,731 loose AKS2 files plus the signature. Its commit is **not** the runtime commit. Compatibility is established by storage contract and experiments, not by requiring every Git hash to be identical.
- The wheel's [declared PyTorch commit/build recipe](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/cmake/External/aotriton.cmake#L12-L33) requests 0.13.50tp images but pins runtime source to 05f26c2d; its Windows branch builds that runtime from source. We have not reproduced the complete build or proven which build-job overrides were used.

All five observed relevant DLL hashes, package versions, runtime header and WMI driver/system are unchanged in A/B/A. Baseline-before/after inventory confirms the production venv was untouched. Native inspection used existing LLVM PE tools as ordinary shell evidence; no REA MCP provider was connected and no decompilation claim is made.

## 14. Comparison with TheRock #7992

[#7992](https://github.com/ROCm/TheRock/issues/7992) concerns gfx1201, another OS/Python/driver combination. Its [maintainer lead](https://github.com/ROCm/TheRock/issues/7992#issuecomment-5591764777) and [nightly follow-up](https://github.com/ROCm/TheRock/issues/7992#issuecomment-5592002534) guided the investigation, but that broad upgrade is not our causal evidence. [Local reading notes](../../eval/rca/gfx1151-rocm7141/upstream-7992.json) preserve links and exact version identities.

Following the discussion found [TheRock #7315](https://github.com/ROCm/TheRock/issues/7315) and the existing [gfx1151 AOTriton #210](https://github.com/ROCm/aotriton/issues/210). #7315's original missing-loader claim was retracted; we do not repeat it. The [AOTriton maintainer explanation](https://github.com/ROCm/TheRock/issues/7315#issuecomment-5295253365) identifies runtime/image version pairing. [ROCm/pytorch #3568](https://github.com/ROCm/pytorch/pull/3568) and [fix d1abf107](https://github.com/ROCm/pytorch/commit/d1abf10723386209420f7427f24744c5afc4d96f) change the 2.12 build pin to c073da80. We verified that patch, but did not rebuild it or install a new wheel. No duplicate upstream issue was opened.

## 15. Controlled A/B evidence

A = unmodified exact-stack torch-only environment; B = same environment with only the image directory replaced by the official 0.12 image payload; A′ = original images restored. Separate fresh processes for every condition. Original OCR venv is never altered.

Scalar no-mask FLASH: secondary A FAIL → B PASS → A′ FAIL. Three repeated [1,1,1,8] efficient conditions follow the same pattern; three MATH controls pass in every state. These are the same-environment, three-state causal experiments. B also passes every supported entry of the original minimization/dtype/mask/stride/causal/order matrix. [A/B run index](TEST_MATRIX.md).

Scalar run IDs: secondary A `edc25492-5643-4355-a1cd-40669094f8ef`; B `645f5ec1-f514-418e-9b45-1a6e8c638822`; A′ `af8f5c94-0de1-4358-a969-617980e3fe1c`. All use fixture hash `2327964b3cd3acf123400cd57647979abe1aad5fe049f1da631c4dac7fa865d1`.

At the original large shape, the initial FAIL was in the **primary OCR venv** (`3b502856-1295-4640-a134-e9498dabb4a8`), B PASS in the secondary venv (`738b674f-32b1-43f8-8132-c49aeda6db94`), and restored-secondary FAIL (`489afbc7-3dbb-4864-b898-3d8783a25fb2`). The initial secondary A did not run the large shape. Thus large-shape evidence includes a controlled B→A′ reversal plus a separate primary baseline, rather than a complete secondary large-shape A/B/A. The complete index retains every environment, run ID, exact fixture and command. No failed experiment was discarded.

## 16. Causal explanation

The [0.12 loader](https://github.com/ROCm/aotriton/blob/05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920/v3src/packed_kernel_common.cc#L56-L122) opens a per-configuration path with an `.aks2` suffix. The [0.13 loader](https://github.com/ROCm/aotriton/blob/6e00ef3e335b45dfb49065259533b59c68995bfe/v3src/packed_kernel_common.cc#L102-L166) reads ZIP directories instead. The original package does not contain the old loader's loose files.

In the [0.12 device loader](https://github.com/ROCm/aotriton/blob/05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920/v3src/on_device_kernel.cc#L28-L49), failed loading leaves a null function. The [invoke path](https://github.com/ROCm/aotriton/blob/05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920/v3src/triton_kernel.cc#L86-L103) nevertheless calls hipModuleLaunchKernel. Our HIP trace directly records that invalid launch. The [efficient PyTorch bridge](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/aten/src/ATen/native/transformers/cuda/attention.cu#L1612-L1657) stores an AOTriton error without checking it; the [flash bridge](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip#L259-L295) has the same error-handling gap. These source paths explain the delayed exception. The exact packaged image contract defect is causally demonstrated; every machine instruction and the original build pipeline have not been reconstructed.

## 17. Why MATH succeeds

MATH computes through ordinary GPU matmul/softmax instead of the incompatible AOTriton fused image loader. Named profiler/decomposition evidence, clean HIP peeks and the independent CPU reference support this explanation. BF16 input/output remain on GPU; accumulation behavior does not imply CPU fallback.

## 18. Workaround disposition

**Retain the existing scoped GPU MATH adapter on this mandated stack.** Its contexts remain around SAM/CLIP only and restore prior settings; decoder DASH remains native Triton. This PR changes no production runtime implementation. The diagnostic image substitution is not a supported packaging repair or deployment recommendation. Future removal needs a correctly assembled official wheel and the same regression matrix on this physical GPU.

## 19. Permanent fix ownership

Primary: **ROCm/PyTorch Windows wheel build/integration maintainers**, coordinated through TheRock. Build a runtime compatible with the shipped image family and verify the package contents. The known source-pin correction is a concrete candidate for maintainers to verify against the wheel assembly. AOTriton owns safer loader/launch failure handling; PyTorch owns checking returned HIP errors and capability/dispatch diagnostics. We make no source fix in these upstream projects.

## 20. Regression plan and model cross-check

Run the minimal scalar and original vision shape, supported forced backends, dtype/mask/stride/causal variants, copy-only and downstream-operation controls, clean repetitions and actual dispatch. Keep fixture hashes and thresholds fixed. Validate runtime/image provenance before accepting a fixed wheel. Then run the hash-verifying SAM probe and the existing OCR regression.

This investigation reran the unchanged SAM probe: default FAIL, three scoped-MATH block forwards and its numerical check PASS. The official full vision encoders are exercised through two accepted English OCR requests (normal and Unicode/recovery) with the original adapter: both CER=0, exact numbers, each 12 synchronized native DASH BF16 GPU calls. [Model evidence](../../eval/rca/gfx1151-rocm7141/model-final/ocr/suite.json). Source/weight hashes remain pinned at revision 9cf077c4b206e0e2a31879df937b0b0bcba3661c. The 26-request Phase 1 suite was not rerun.

CPU checks, optional torch reference tests and final review are linked in [verification](../../eval/rca/gfx1151-rocm7141/checks/process-manifest.json); hosted CI status belongs to the PR's current head.

## 21. Limits

One actual Windows gfx1151 system, inference forward only; no backward/training, other GPU or Linux claim. No official newer build or rebuilt d1abf107 was tested. The images-only diagnostic proves the assembly defect, not which private CI job introduced it. Header SHA alone is not proof of binary identity, and different source hashes can still share compatible images. Kernel storage, tuning metadata and payload change together as the one image-directory variable; we do not attribute the result to a single changed GPU instruction. Native profiler GPU symbols were unavailable; HIP module/launch traces and actual ATen operators supply the runtime boundary evidence.

## 22. Reproducible artifacts

[Standalone program](../../experiments/sdpa_rca/repro_sdpa.py), [PowerShell instructions](../../experiments/sdpa_rca/README.md), [raw reports/logs](../../eval/rca/gfx1151-rocm7141/), [experiment index](../../eval/rca/gfx1151-rocm7141/experiment-manifest.json), [SHA256SUMS](../../eval/rca/gfx1151-rocm7141/SHA256SUMS.txt), [original→published digests](../../eval/rca/gfx1151-rocm7141/export-manifest.json), [producer source history](../../eval/rca/gfx1151-rocm7141/producer-sources.json), [upstream handoff](UPSTREAM_HANDOFF.md).

Raw child JSON remains authoritative; summaries are derived indexes. Private originals remain local, home paths are neutralized, and published bytes have independent checksums. Producer commit records the working repository HEAD; collector/reporting additions occasionally preceded their own commit, so the source index resolves each **actual script digest** to the later commit containing those exact bytes instead of pretending every producing checkout was clean.

[Independent review](REVIEW.md) verified the numerical reference and artifact hashes, and identified reporting/preservation issues corrected before publication. Historical reports were audited: none of their recorded PASS results contains a HIP error. The collector's locale decoding displays the UTF-8 copyright symbol incorrectly in header text; ASCII identifiers and raw header hashes are unaffected. This display-only minor remains documented.

[GitHub Issue, PR and publication verification](GITHUB_DELIVERY.md).
