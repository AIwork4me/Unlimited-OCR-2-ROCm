# Unlimited-OCR-2-ROCm

Baidu Unlimited OCR 2 on AMD ROCm — native inference, long-document parsing, and reproducible benchmarks

An independent wrapper for the official Baidu V2 native Transformers path.
Phase 1 targets **Windows 11 + Ryzen AI MAX+ 395 + Radeon 8060S (gfx1151)**.
It preserves the trained weights, BF16, native `infer()`, native
`infer_multi()`, DASH routing and the official attention kernels.

**Physical Windows validation: A–E PASS.** English, Chinese, mixed text,
tables, simple arithmetic and repeated native 2/3/5-page inference passed
in two isolated environments on this machine, with normalized CER 0.
Read [the executed validation record](docs/VALIDATION.md) and
[full report](docs/FINAL_REPORT.md) for exact scope and evidence.

The required stack's default vision SDPA is faulty here. A disclosed,
model-local GPU MATH SDPA context covers SAM/CLIP only; decoder DASH/Triton
and native generation remain unchanged. Raw page markers are preserved,
including a possible single empty terminal marker.

[中文说明](README.zh-CN.md) · [Upstream audit](docs/UPSTREAM_AUDIT.md) ·
[Reproduction](docs/REPRODUCIBILITY.md) · [Windows compatibility](docs/WINDOWS_COMPATIBILITY.md)

The exact stack is:

| Component | Version |
|---|---|
| CPython | 3.10.20 x64 |
| torch | 2.12.0+rocm7.14.1 |
| torchvision | 0.27.0+rocm7.14.1 |
| torchaudio | 2.11.0+rocm7.14.1 |
| ROCm release / HIP build | 7.14.1 / 7.14.60850 |
| triton-windows | 3.7.1.post27 |
| Transformers | 4.52.1 |

torch.cuda is also the AMD ROCm API. This APU shares system RAM; its reported
GPU budget is not dedicated physical VRAM. No other operating system or GPU
has been physically validated by this project.

## Setup and validation

Run from the repository root. Select an existing CPython 3.10 interpreter.
Use unused, writable short environment paths: an AMD wheel entry reaches the
Windows extraction limit at 260 total characters. The environments and cache
can live outside a deeply nested checkout; no registry change is needed.
If uv is installed:

```powershell
$Python310 = (uv python find 3.10)
$VenvA = "C:\uocr\a"
$VenvB = "C:\uocr\b"
$CacheDir = "C:\uocr\cache"
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath $VenvA -CacheDir $CacheDir -OutputPath "eval/results/run-a"
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath $VenvB -CacheDir $CacheDir -OutputPath "eval/results/run-b" -BaselinePath "eval/results/run-a/reproduction.json"
```

Otherwise set `$Python310` to that interpreter's full path. Choose another
short writable root if `C:\uocr` is unavailable, and new output directories
if these already contain results. Setup creates a
dedicated venv and installs the exact AMD stack, hash-pinned Triton and locked
dependencies. It never replaces global Python or changes drivers/security.
The distribution has no automatic torch dependency; a plain pip package
installation does not prepare the GPU stack.

To inspect individual steps, use the second environment created above.
These commands select its interpreter explicitly; activation is unnecessary:

```powershell
$OcrPython = "C:\uocr\b\Scripts\python.exe"
$CacheDir = "C:\uocr\cache"
& $OcrPython scripts/doctor_windows.py
& $OcrPython scripts/probe_triton.py
& $OcrPython -m unlimited_ocr_2_rocm download --cache-dir $CacheDir
& $OcrPython scripts/smoke_single.py --cache-dir $CacheDir --output "eval/results/single-new"
& $OcrPython scripts/smoke_multi.py --cache-dir $CacheDir --output "eval/results/multi-new"
```

Run multi-page correctness only after single-page correctness succeeds.
The suite uses the official `infer_multi()` for the whole ordered document.

## Your documents

```powershell
$OcrPython = "C:\uocr\b\Scripts\python.exe"
& $OcrPython -m unlimited_ocr_2_rocm single "C:\documents\中文 page.png" --cache-dir "C:\uocr\cache" --output "C:\OCR output\single"
& $OcrPython -m unlimited_ocr_2_rocm multi "page 1.png" "page 2.png" --cache-dir "C:\uocr\cache" --output "C:\OCR output\multi"
```

Both accept `--cache-dir`, `--offline` and `--max-new-tokens`. Use a new
output directory. Raw text, normalized text, hashes, timing, memory and
successful custom-kernel calls are preserved. User-document commands report
INFERENCE_ONLY; they cannot establish correctness without known ground truth.
Phase 1 limits requests to five images and 16 million pixels per page.

Official model/code revision:
`9cf077c4b206e0e2a31879df937b0b0bcba3661c`.
The complete reviewed code/config/tokenizer and unchanged weight hashes are
in the package's upstream-lock.json. Imports and builds never download models.

The upstream visualization helper evaluates generated coordinate expressions.
This wrapper calls inference with `save_results=False`, writes raw output
itself and inspects markup without evaluating it. See [security notes](docs/SECURITY.md).

## Evidence and scope

Phase 1 PASS requires A: environment, B: native model/kernels, C: OCR
correctness, D: native ordered multi-page correctness and E: a second clean
environment on the same real machine. Unexecuted checks are NOT TESTED.
Synthetic acceptance uses normalized CER <= 0.02, exact important numbers and
exact content-page count/order, with raw outputs retained. One empty terminal
PAGE may be an end delimiter; raw count and this condition are reported separately.

CPU CI runs lint, formatting, types, tests, builds and CLI smoke without torch
or model downloads. It does not imply Linux inference support or GPU testing
on hosted runners. Physical evidence belongs in eval/results and the PR.

This project is pre-alpha. No PyPI publication, stable release or performance
speedup is claimed. The native Transformers implementation has growing caches;
README claims about constant-memory serving must not be applied to it.
See [limitations](docs/LIMITATIONS.md) and [contribution guidance](CONTRIBUTING.md).

Wrapper license: MIT. Official model card: MIT; source includes additional
DeepSeek/Hugging Face Apache-2.0 headers. Model files are downloaded separately.
See [NOTICE](NOTICE) and the immutable upstream audit for attribution.
