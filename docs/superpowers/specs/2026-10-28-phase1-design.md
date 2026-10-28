# Phase 1 design

The user supplied the complete mission and authorized autonomous execution.
Build a small native Windows wrapper around reviewed, immutable official Baidu
Unlimited OCR 2 code and unchanged weights. Correct output and reproducible GPU
evidence take priority over performance.

The only target is Windows 11, Ryzen AI MAX+ 395, Radeon 8060S / gfx1151.
Use Python 3.10 if its wheels work, torch 2.12.0+rocm7.14.1,
torchvision 0.27.0+rocm7.14.1, torchaudio 2.11.0+rocm7.14.1.
Investigate triton-windows 3.7.x and the native Transformers path first.

Gate A: OS, identity, package versions, synchronized BF16 GPU result.
Gate B: unchanged official weights and required native custom kernels.
Gate C: synthetic English, Chinese, mixed and structured OCR correctness.
Gate D: native infer_multi, 2/3/5 pages, raw page markers and ordered content.
Gate E: a second isolated environment on the same physical machine.
All gates are required. Failures and unexecuted checks remain visible.

Implementation choices delegated to the engineer: argparse CLI, pathlib paths,
JSON manifests and UTF-8 raw output, separate CPU and opt-in physical GPU tests.
Use strict exact stack constraints and never put torch in package dependencies.
Keep downloads outside Git. Cache and output paths are configurable.
Safe text inspection precedes trust_remote_code execution. Avoid unsafe upstream
visualization of generated coordinate expressions until reviewed.

Do not replace SR-SWA/DASH, use CPU fallback, install SGLang, change drivers or
global settings, modify V1, claim untested hardware, publish PyPI/releases,
merge the PR or start Phase 2. Public repository, feature branch, clean commits,
push and a review-ready PR are authorized.
