# Current PyTorch Windows gfx1151 wheel decision

**FIX_VERIFIED** for the locally source-built Windows cp310 wheel on the physical
Ryzen AI MAX+ 395 / Radeon 8060S, SDK 7.14.1, driver 32.0.22032.6002.
The unchanged BF16 forward SDPA matrix now passes both actual fused backends.

| Result | Immutable baseline | Source-built candidate |
| --- | ---: | ---: |
| Required fused conditions | 21 FAIL | 21 FUSED_PASS |
| Explicit MATH controls | 4 MATH_PASS | 4 MATH_PASS |
| Copy-only control | 1 CONTROL_PASS | 1 CONTROL_PASS |
| Named optional masked-FLASH capability probe | 1 UNSUPPORTED | 1 UNSUPPORTED |

All **27 input fixture hashes** and all arguments except output paths match.
Each of the six required condition groups completed three independent Python
processes. DEFAULT scalar dispatches FLASH; DEFAULT vision dispatches EFFICIENT.
The forced FLASH/EFFICIENT cases also dispatch those actual operators.

Every measured SDPA output passes the original independent CPU FP32 reference
criterion, BF16 `atol=0.03`, `rtol=0.03`, with zero failing/nonfinite elements.
All HIP checkpoints are zero, synchronization succeeds, and downstream copy and
arithmetic operations complete. Both AMD_LOG_LEVEL=4 samples show successful
module loading, `attn_fwd` lookup, nonnull launches with grid `(1,2,1)` and block
`(128,1,1)`, `hipSuccess`, and PAL kernel execution. MATH fallback is not counted
as fused correctness.

## Exact artifact

- PyTorch: `60dd50f143fd9b10ac23cde8cdd34be61232b139`.
- Frozen TheRock recipe: `6d34c9b6f3e752b8f718b0c81f3070df77ce3283`.
- AOTriton runtime: 0.14.50, commit `7a6963a7265d2cf0fc48c6a63464386134d32483`.
- Official 0.14.50tp images identify source `c11b5b886a10a00f925f74133d38b3505e67ccd9`.
  All seven image files match the verified official archive; runtime/image
  compatibility is established by the physical tests, not inferred from IDs.
- Wheel: `torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl`,
  **388,478,127 bytes**, SHA-256
  **`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`**.

The frozen official builder exited **0**, using the existing SDK with
`--no-install-rocm`, only torch/gfx1151, FLASH/EFFICIENT ON and bounded jobs.
The wheel was installed by its exact local path into fresh `C:/b/new`.
All 15 native wheel binaries plus the HIP DLL match the reviewed profile in
every acceptance process. A supplemental inventory verifies all 33 loaded
torch/SDK native modules against their provider wheels.

## Evidence and review

[Decision JSON](../../eval/wheel-validation/windows-gfx1151/decision.json) · [Paired matrix](../../eval/wheel-validation/windows-gfx1151/comparison.json) ·
[Candidate raw reports](../../eval/wheel-validation/windows-gfx1151/candidate/) ·
[Kernel load/launch evidence](../../eval/wheel-validation/windows-gfx1151/continuation/hip-launch-evidence.json) ·
[Artifact inspection](../../eval/wheel-validation/windows-gfx1151/continuation/built-wheel-inspection.json) ·
[Installed native hashes](../../eval/wheel-validation/windows-gfx1151/continuation/installed-native-runtime.json) ·
[SHA256SUMS](../../eval/wheel-validation/windows-gfx1151/SHA256SUMS.txt).

The original three-file producer remains byte-identical to RCA commit
`6b643f51d4078c4bb841f6a48734cf2206090311`. The candidate adapter changes exactly
two identity AST nodes. Independent review checked the real artifact, complete
native identity gate and all 27 raw GPU reports. **121 CPU tests pass**, with two
opt-in GPU pytest markers skipped; the standalone physical matrix ran separately.

This establishes the local source-built artifact on this Windows gfx1151 BF16
forward matrix. A newly published AMD candidate wheel, other dtypes, backward
passes, other hardware and the full OCR model pipeline remain unmeasured.
No single fixing commit was isolated. See [build provenance](BUILD_PROVENANCE.md),
[matrix detail](GPU_TEST_MATRIX.md) and [review](REVIEW.md).
