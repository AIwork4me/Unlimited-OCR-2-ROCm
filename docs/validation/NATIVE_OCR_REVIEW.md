# Candidate native OCR review — 2026-10-29

An independent read-only reviewer checked the implementation and then the final
GPU evidence. No GPU work was delegated or re-executed by the reviewer.

The implementation review confirmed default MATH, scoped SAM/CLIP AUTO and
unchanged decoder behavior. It identified a missing tracked fused gate and a
timing-label ambiguity. Both were corrected with failing-then-passing CPU tests:
the tracked harness requires nonempty fused AUTO diagnostics before benchmarking,
and `native_inference_seconds` ends at final synchronization before output work.
No Important findings remained after those corrections.

The evidence review independently verified:

- All 146 OCR manifests PASS, with normalized CER 0, required numbers/structure
  passing, 12 DASH calls per request and matching raw-output hashes.
- All 100 measured sample timings, medians, maxima and speedup ratios agree with
  the report. Output-token sets match between modes.
- All 1,224 diagnostic attention calls are fused: 408 EFFICIENT and 816 FLASH.
  HIP checkpoints are zero. Nine numerical layout records have maximum absolute
  errors 0.015625 versus MATH and 0.007917642593383789 versus CPU FP32.
- Current measurement-source hashes, preserved Stage 3/4 producer bytes and all
  761 exported checksums agree with the filesystem.

The primary agent separately verified every checksum-listed file against staged
Git blob bytes, scanned exported evidence for unredacted user-profile paths and
excluded binary wheels, DLLs, weights and copied image inputs. The narrow
`.gitattributes` rule preserves raw evidence bytes and whitespace; it does not
modify historical evidence.

The conclusions remain limited to the recorded synthetic fixtures, BF16 forward
inference and one Windows gfx1151 device/driver. Five samples and serial ordering
do not establish a reliable p95 or eliminate thermal/order effects. Retaining
MATH as default is supported.
