# Experimental native OCR on the local candidate wheel

This experiment preserves the accepted BF16/native infer/infer_multi path,
official model revision and decoder DASH/Triton. Production CLI commands still
select MATH. Python callers can explicitly request
`load_model(cache_dir, offline=True, vision_sdpa="auto")`; this selection is
scoped separately to SAM and CLIP forward calls and restores caller settings.
AUTO permits FLASH, EFFICIENT and MATH. Selection alone is not evidence of fusion.

Use a fresh CPython 3.10 environment inside the repository. Install the exact
local candidate with `--no-deps` after checking SHA256
`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`.
The required torchvision/torchaudio and native model package versions remain
those in `requirements/windows-cp310.lock.txt`. The two historical
`amd-torch-device-*` overlays are omitted: the candidate is self-contained,
and the gfx115x package would overwrite its validated AOTriton images.
Keep the matching ROCm SDK/device and torchvision device packages.
Never run the ordinary historical setup script against this candidate environment.

Stage 1 reuses the byte-identical regression tools from PR #4 commit
`43933ac86d67aa8089fc01e66eaf9e885b1a4b2a`. Its identity-profile producer verifies
the installed and loaded native libraries against the actual candidate and SDK
wheel members. Its 27-process matrix retains the original fixture hashes,
BF16 numerical tolerances and HIP checks. Preserve the resulting profile and
installation report. This branch does not rebuild or redistribute torch.

After the unchanged `smoke_single.py` and `smoke_multi.py` pass, place their
output directories under one root as `stage2-single` and `stage2-multi`.
The experimental harness requires those exact case sets and successful native
kernel records. Run from the primary repository root, with process-scoped
TEMP/TMP, Triton and pip caches inside the repository:

```powershell
$OcrPython = (Resolve-Path .b/o/Scripts/python.exe).Path
& $OcrPython experiments/native_ocr/validate.py `
  --identity-profile .b/m/identity.json --stage2-root .b/m `
  --cache-dir .b/h --output .b/m/new-auto-single `
  --vision-sdpa auto --suite single --trace
```

Repeat with `--suite multi` and a new output directory. Diagnostic tracing
observes each actual vision SDPA operator, shapes/strides/masks and HIP state.
For the first occurrence of each module/shape/stride/mask-layout signature,
it compares the full output with MATH and three actual batch/head slices with
independent CPU FP32 attention, at the existing BF16 atol/rtol 0.03/0.03.
Subsequent calls retain operator/HIP/downstream checks; they do not repeat the
numerical reference for that layout. Every OCR case still uses the unchanged
CER, exact-number, page-order and table criteria. Reference calculations are
diagnostics; native inference remains on the GPU.

For measurements omit `--suite` and `--trace`, use `--samples 5`, and run
sequentially for `--vision-sdpa math` and `auto`, each with a fresh output
directory. Pass `--auto-diagnostics SINGLE_OUTPUT MULTI_OUTPUT`; the harness
requires successful AUTO quality suites and nonempty, entirely fused traces.
Both use the same wheel/environment/model/input hashes, BF16 and
1024 tokens per page. Each of ten cases has one excluded warmup and five warm
samples. The first measured process loads the model separately; loading and
hash verification timings are retained separately. Each encoder and native
request has synchronization boundaries. Memory manifests distinguish process
RSS, GPU allocated/reserved and available system memory on the shared-memory APU.

Profiling and numerical references are excluded from these timing runs.
`native_inference_seconds` ends immediately after native inference and GPU
synchronization. `synchronized_wall_seconds` additionally includes output
writing, normalization and quality assessment; report these as separate fields.
Report medians and sample maxima, not p95 from five samples. The simple serial
MATH-then-AUTO order does not eliminate thermal/order bias. This is one device,
one driver, synthetic acceptance documents and BF16 forward inference; broader
document coverage, other hardware/dtypes, backward and default enablement need
separate validation. The local artifact is not an official AMD fix.
