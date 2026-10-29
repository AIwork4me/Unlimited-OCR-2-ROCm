# Candidate-wheel native OCR validation plan

> Execute inline with the executing-plans skill. The user's four-stage mission is the specification; each later stage is gated by the preceding correctness result.

**Goal:** Establish correctness and measured benefit of the exact local wheel on physical Windows gfx1151, retaining MATH as the default.

**Architecture:** Reuse the accepted native inference/quality suite and immutable PR #4 regression producer. Keep candidate setup and measurement in an experimental harness. Add only an explicit MATH/AUTO selector to the SAM/CLIP wrapper after Stage 2 passes.

**Tech stack:** CPython 3.10.20, torch 2.12.0+rocm7.14.1, torchvision 0.27.0+rocm7.14.1, torchaudio 2.11.0+rocm7.14.1, BF16, native DASH/Triton on Windows gfx1151.

## Constraints and review focus

- Exact wheel SHA256 `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`; no rebuild, redistribution, merge, release or default change.
- All new environments, caches, evidence and tools remain under the primary repository. Preserve migration edits, historical checkouts/environments and evidence.
- Never install the historical gfx115x image overlay over candidate images. Verify installed and loaded native bytes against provider wheels.
- Preserve immutable model/source hashes, BF16, native infer/infer_multi, decoder DASH, thresholds (CER <=0.02, exact numbers, page count/order and table structure).
- Failures must retain tracebacks, raw outputs and measurements. AUTO selecting MATH is not fused evidence. Capture shapes/masks, actual operators and HIP checkpoints separately from timing runs.

## Tasks

- [x] Stage 1: Inspect wheel metadata and file overlap, create fresh `.b/o`, install exact locked packages without dependency resolution replacing torch, verify native identity, pip check, doctor/BF16, Triton and frozen fused regression matrix. Preserve private originals and publish sanitized evidence with checksums.
- [x] Stage 2: Run unchanged single-page acceptance plus repeated Unicode recovery, then ordered native 2/3/5-page repeat and table acceptance. Preserve original Phase 1 evidence; stop experimental work if any case fails.
- [x] Stage 3 (gated): Test default selection, explicit AUTO and invalid selection on CPU before implementing the scoped selector in `compat.py`/`model.py`. Add an experimental observation harness for real SDPA shapes/masks/operators, independent numerical reference and HIP/downstream checkpoints. Execute identical quality acceptance under AUTO.
- [x] Stage 4 (gated): Run sequential paired MATH/AUTO measurements on the same environment and input hashes, with warmup, five warm samples, synchronized vision and end-to-end times, separate cold load, process/GPU/system memory and native DASH counts. Report median and maximum; five samples do not support reliable p95.
- [x] Delivery: run required CPU checks, review complete diff/evidence, document PASS/FAIL/NOT RUN and unsupported conditions, then open a draft experimental PR from current main without merging.

Implementation decisions: expose selection as a keyword argument on `load_model` and `wrap_vision_forward`, with MATH defaults; keep detailed tracing in the experiment so ordinary calls avoid profiling overhead. If an actual incompatibility blocks a stage, investigate and make only an evidence-supported minimal correction; do not weaken gates.

Delivery: experimental draft PR #5, https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/5. All four physical stages and the required local CPU checks passed; independent implementation/evidence review is clean. MATH stays the default. No merge, release or wheel redistribution.
