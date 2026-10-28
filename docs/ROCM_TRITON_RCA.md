# ROCm and Triton investigation

## Exact release versus HIP compiler build

The exact mandatory AMD install succeeds under CPython 3.10.20 x64.
torch/version.py identifies torch=2.12.0+rocm7.14.1,
rocm=7.14.1 and hip=7.14.60850. The doctor's original assumption that the
HIP build field equals the package release was wrong. The strict validator
now checks both actual fields and rejects drift; its RED/GREEN test and the
original failure manifest are preserved.

## BF16 cancellation residue

A 32x32 integer-valued BF16 GEMM with FP32 reference initially used zero
tolerance. GPU results had residual 4.76837158203125e-7 at some exact-zero
cancellation positions. A separate int64 CPU GEMM and FP32 CPU GEMM both
returned zero there. This is a small numerical discrepancy, not evidence of
CPU execution. The documented probe acceptance is rtol=0, atol=1e-5; actual
error and device are recorded. Original failure is retained. OCR thresholds
were fixed before inference and were not altered for this discrepancy.

## Triton provenance and execution stages

The AMD install did not bring a Triton distribution. Reviewed candidate:
triton-windows 3.7.1.post27, cp310-cp310-win_amd64. The exact
[post27 release](https://github.com/triton-lang/triton-windows/releases/tag/v3.7.1-windows.post27)
documents RDNA3.5 coverage. Wheel SHA-256:
01ede775b102c91acc89fc6946b946451f966cb64856c650d244d37ea3bbdee5.
PyPI metadata has no default torch dependency; install uses --no-deps and
--require-hashes so it cannot replace torch.

| Stage | Actual local result |
|---|---|
| Triton-0 import/backend | PASS: Triton 3.7.1, HIP, gfx1151 |
| Triton-1 vector add | PASS: freshly compiled and synchronized on real GPU |
| Triton-2 vector reference | PASS: exact match, including odd-size tail |
| Triton-3 official DASH prefill | PASS: unchanged reviewed e2e kernel at two lengths |
| Triton-4 within model | See VALIDATION.md and model inference manifests |

The DASH probe uses B=1, H=10, D=128, T=160 and 173, block size 32,
local window 128, selected remote blocks with additive bias, duplicate and
padding block indices. An independent token-level FP32 reference enforces
causality, local-window precedence and one contribution per selected block.
Fixed tolerances before first launch: rtol=0.03, atol=0.03. Measured max
absolute errors: 0.0086941719 and 0.0084605217; mean errors about 0.0004.

This validates forward inference, not training/backward. The base-stage gather
and block-max kernels are not selected by this checkpoint's native DASH scorer;
do not treat the e2e test as their validation. No source patch, different
attention algorithm, Triton 3.8, Linux wheel or disabled DASH was used.

Native instrumentation checks BF16 GPU query placement, synchronizes each
successful official e2e call, and records its shape. A prefill dense fallback
raises, and fewer than twelve successful decoder-layer calls fails the run.
Official decode remains its original PyTorch bias-aware attention algorithm.
This is observation/guarding, not a kernel rewrite.

## Vision SDPA failure and minimal runtime compatibility

The unchanged model loads 6.69GB of official weights on GPU, but native
single-page inference initially fails at deepencoder.py:890 with
hipErrorInvalidValue. Synchronization before that copy succeeds. A two-call
SAM-block reproduction fails without OCR weights, while twenty plain copies
of the same tensor layout pass. HIP peek instrumentation isolates the earlier
default scaled_dot_product_attention call: it leaves HIP error 1.

A direct BF16 SDPA comparison at [54,12,196,64] also fails numerically
(97.4% of elements outside rtol/atol0.03, max error 5.1182). A fresh process
using GPU MATH SDPA leaves HIP error 0 and matches the independent FP32
softmax(QK/sqrt(d)+bias)V reference (max error about 0.00387). Three repeated
SAM-block forwards pass. Production does not clear or ignore HIP errors.
The diagnostic experiment cleared a recorded error only to inspect the
otherwise corrupted default output; that behavior is not in the adapter.

The narrow adapter scopes the public sdpa_kernel(MATH) context separately
around SAM and CLIP encoder forward calls. It restores prior settings after
each call, preserves BF16 model/input/output, and records GPU tensor shapes.
Decoder DASH, trained routing, page-aware positions, native e2e kernels and
generation remain unchanged. This is an explicitly disclosed runtime
compatibility change; downloaded upstream files and weights remain byte-identical.

[TheRock issue7992](https://github.com/ROCm/TheRock/issues/7992) describes
the same exact-stack pattern on gfx1201. That is corroborating issue history,
not validation of another GPU here. We do not change the mandated versions.
See scripts/probe_vision_sdpa.py and diagnostics for the regression evidence.

## Download observations

The error-clearing reference inspection above is an isolated
**NON-NATIVE / NON-PARITY diagnostic experiment**. It produces no accepted
OCR result and is not used by the production adapter or acceptance workflow.

The first PyPI Triton download took about eleven minutes at roughly 65kB/s.
A range download independently reproduced the exact wheel hash. Ordinary
HTTP weight downloading was replaced with the official hf-xet client in a
separate downloader environment. These are transport observations, not ROCm
compatibility failures. Runtime wheels and model identities remain pinned.
