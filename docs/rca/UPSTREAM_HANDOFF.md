# Proposed upstream handoff (not posted)

Suggested owner: ROCm/PyTorch Windows wheel packaging maintainers via TheRock, with AOTriton maintainers copied for loader error handling. Existing related reports: [TheRock #7992](https://github.com/ROCm/TheRock/issues/7992), [TheRock #7315](https://github.com/ROCm/TheRock/issues/7315), [AOTriton #210](https://github.com/ROCm/aotriton/issues/210). Please attach this evidence to the appropriate existing investigation rather than opening a duplicate by default.

**Title:** Windows gfx1151: images-only A/B/A confirms AOTriton runtime/image packaging mismatch in torch 2.12.0+rocm7.14.1

**Confidence:** CONFIRMED, L4, on one physical Windows gfx1151 system. The production workaround remains scoped MATH. No official replacement wheel or source rebuild was tested.

Environment: Windows 11 build 26200; Ryzen AI MAX+ 395 / Radeon 8060S; driver 32.0.22032.6002; CPython 3.10.20 x64; torch 2.12.0+rocm7.14.1 (declared commit 098172edc887399a593f0c74ae5c40b1fe043801); HIP 7.14.60850; torchvision 0.27.0+rocm7.14.1; torchaudio 2.11.0+rocm7.14.1. A separate torch-only venv reproduces with no Python Triton, HF, Transformers or model weights.

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'torch[device-gfx1151]==2.12.0+rocm7.14.1' 'torchvision[device-gfx1151]==0.27.0+rocm7.14.1' 'torchaudio==2.11.0+rocm7.14.1'
python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --output scalar-default.json
python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --backend MATH --output scalar-math.json
```

Use a fresh process per call. The smallest failing shape is Q/K/V `[1,1,1,1]`, BF16, no mask. Expected output is exactly V. Actual default dispatch is `aten._scaled_dot_product_flash_attention.default`; SDPA returns but hipPeekAtLastError changes 0→1 and synchronization succeeds without clearing it. The script returns FAIL/exit 1. Original vision-layout BF16 `[54,12,196,64]` with additive `[54,12,196,196]` bias instead dispatches efficient and the next real copy throws hipErrorInvalidValue. MATH passes both. cuDNN is unavailable; FLASH+bias is unsupported and not counted as an executed failure.

The CPU seeded fixture is quantized before reference. Explicit independent CPU FP32 attention uses frozen BF16 atol/rtol .03/.03. At the large shape, MATH max absolute error 0.0038219690, zero mismatches. An explicitly non-acceptance original-image diagnostic records the sticky error, clears it only to inspect corrupted output, and finds max error 1.57019746 and 78.51277% mismatches. Clearing is never an accepted workaround.

**Causal A/B/A:** in a disposable exact-stack environment change only `torch/lib/aotriton.images` to official 0.12.50tp gfx115x images; all five loaded DLL hashes, package versions, runtime header and driver remain fixed. Original images FAIL → compatible images PASS → original restored FAIL. FLASH scalar and repeated EFFICIENT [1,1,1,8] inputs follow that transition; MATH controls always pass. The large shape additionally fails in the original OCR venv, passes in secondary B, and fails after restoring secondary A; its initial secondary A was not run, so it is not presented as a full same-environment three-state experiment. With compatible images, supported dtype/mask/stride/causal/order cases match the same CPU reference. HIP trace changes from zero module loads and a null-function block(0,0,0) launch to a successful module load and valid block(128,1,1) launch.

Runtime header: 0.12.50 / 05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920. Actually loaded aotriton_v2.dll SHA-256 `259eeb6f064eaf506a032bdb65dc9b52c6d5b7d9128de5372e5c5b1b7c2b5d22`; identical to its entry inside the exact cp310 torch wheel (wheel SHA-256 `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`). Original images identify aotriton0.13 / 6e00ef3e; attn_fwd.zip SHA-256 `adfb2fd4d07046cb40a5cf853303a01eb0cf79bc35094d033cf85955a2906850`. Official diagnostic 0.12 archive SHA-256 `3177387a15c678b30057f4584d1fc1b8f8db56163890cb5c98f27450209f5a7b`; its images commit is fe12b77d, not the runtime commit. Source-hash inequality alone is not our proof: old loose-AKS2 vs new ZIP loader contracts plus reversible experiments are.

The exact declared [build recipe](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/cmake/External/aotriton.cmake) combines 0.13.50tp image downloads with the 05f26c2d runtime source pin. [ROCm/pytorch #3568](https://github.com/ROCm/pytorch/pull/3568) / [d1abf107](https://github.com/ROCm/pytorch/commit/d1abf10723386209420f7427f24744c5afc4d96f) correct that pin to c073da80. We have not rebuilt it or proven the original CI override history. The report links exact old/new loader and unchecked error-propagation source lines.

Requested maintainer action: verify/backport correct runtime/image assembly into a public Windows wheel; add packaged-artifact smoke tests for actual fused launch and reference correctness; report loader errors before null launches; check AOTriton return status in the PyTorch bridge. Avoid treating a broad nightly upgrade or image/dll replacement in a production environment as the causal experiment.

Published evidence will accompany this repository's linked Issue and PR. [Complete RCA](SDPA_ROOT_CAUSE.md), [standalone commands](../../experiments/sdpa_rca/README.md), [raw reports and hashes](../../eval/rca/gfx1151-rocm7141/), [matrix](TEST_MATRIX.md). No upstream report was automatically posted.
