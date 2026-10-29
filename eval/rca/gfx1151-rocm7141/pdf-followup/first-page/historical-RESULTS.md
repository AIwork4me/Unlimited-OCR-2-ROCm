# Actual PDF input diagnosis

The input PDF has 15 pages; this diagnostic renders and tests page 1 at 144 DPI.
Directly supplying the PDF to the current Pillow image reader raises UnidentifiedImageError before SDPA. This is separate from the GPU runtime defect.

The existing scoped MATH adapter stays enabled during real page inference. The observer saves actual first SAM and CLIP SDPA inputs without changing official source/weights or returned attention values. Each backend replay is a fresh Python process. Q/K/V GPU strides are restored from the capture metadata. No HIP error is explicitly cleared.

| Actual page path | Requested | Observed | After SDPA HIP | Result | CPU FP32 max abs error |
|---|---|---|---:|---|---:|
| sam_model | DEFAULT | EFFICIENT_ATTENTION | 1 | FAIL | None |
| sam_model | MATH | MATH | 0 | PASS | 0.00390625 |
| vision_model | DEFAULT | FLASH_ATTENTION | 1 | FAIL | None |
| vision_model | MATH | MATH | 0 | PASS | 0.0019503235816955566 |

SAM Q/K/V are BF16 [54,12,196,64], with additive BF16 [54,12,196,196] bias. CLIP Q/K/V are BF16 [6,16,101,64], no mask. MATH uses the previously frozen atol/rtol=.03/.03; both comparisons have zero failing/nonfinite elements.

The five relevant DLL files match the earlier RCA's exact SHA-256 values. Together with that investigation's images-only A/B/A, this establishes the same packaged runtime/image defect for these real first-page SDPA calls. The PDF content does not replace the installed GPU image payload.

The adapted first page completes inference (546 output tokens, stop finish reason, 12 native DASH calls). Output is INFERENCE_ONLY: no independent OCR-accuracy score is claimed. The other 14 pages were not processed; the current Phase 1 multi-image wrapper accepts at most five page images per request.

Raw per-condition JSON/stdout/stderr, CPU tensor captures, PDF hash, page render hash, model source/weight hashes and page output remain in this local diagnostic folder. Nothing here was committed or uploaded.
