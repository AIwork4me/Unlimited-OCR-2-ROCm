# Physical Windows gfx1151 paired matrix

**FIX_VERIFIED** for this local source-built BF16 forward artifact. Both matrices
contain 27 fresh Python processes. The original producer/reference bytes, seed
7992, tensor shapes, masks and thresholds are unchanged. Every case's fixture
hash and arguments except output path match exactly; no error clearing was used.

| Condition | Repetitions | Observed candidate backend | Baseline | Candidate |
| --- | ---: | --- | --- | --- |
| Scalar, no mask; DEFAULT | 3 | FLASH_ATTENTION | FAIL | FUSED_PASS |
| Scalar, no mask; FLASH | 3 | FLASH_ATTENTION | FAIL | FUSED_PASS |
| [1,2,16,64], no mask; FLASH | 3 | FLASH_ATTENTION | FAIL | FUSED_PASS |
| [1,2,16,64], bias; EFFICIENT | 3 | EFFICIENT_ATTENTION | FAIL | FUSED_PASS |
| [54,12,196,64], bias; DEFAULT | 3 | EFFICIENT_ATTENTION | FAIL | FUSED_PASS |
| Vision, bias; EFFICIENT | 3 | EFFICIENT_ATTENTION | FAIL | FUSED_PASS |
| Scalar; MATH | 1 | MATH | MATH_PASS | MATH_PASS |
| Small/no mask; MATH | 1 | MATH | MATH_PASS | MATH_PASS |
| Small/bias; MATH | 1 | MATH | MATH_PASS | MATH_PASS |
| Vision/bias; MATH | 1 | MATH | MATH_PASS | MATH_PASS |
| Small/bias, downstream arithmetic | 1 | EFFICIENT_ATTENTION | FAIL | FUSED_PASS |
| Vision copy-only control | 1 | UNOBSERVED | CONTROL_PASS | CONTROL_PASS |
| Vision/bias; optional FLASH probe | 1 | UNOBSERVED | UNSUPPORTED | UNSUPPORTED |
| FLASH with AMD_LOG_LEVEL=4 | 1 | FLASH_ATTENTION | FAIL | FUSED_PASS |
| EFFICIENT with AMD_LOG_LEVEL=4 | 1 | EFFICIENT_ATTENTION | FAIL | FUSED_PASS |

Totals: baseline **21 FAIL / 4 MATH_PASS / 1 CONTROL_PASS / 1 UNSUPPORTED**;
candidate **21 FUSED_PASS / 4 MATH_PASS / 1 CONTROL_PASS / 1 UNSUPPORTED**.
The named optional additive-mask FLASH probe alone may be UNSUPPORTED. Copy-only
executes no SDPA; its UNOBSERVED backend is intentional. Explicit MATH controls
observe the MATH operator through CPU operator profiling while tensors execute
on the GPU; they are not used as fused proof.

## Numerical comparison

The independent CPU FP32 reference uses the original combined BF16 criterion
`abs(actual-reference) <= 0.03 + 0.03*abs(reference)`. All 25 measured candidate
SDPA results have zero failing and zero nonfinite elements. Baseline fused
numerics remain unmeasured after the original HIP fault.

| Fixture | Baseline MATH max abs error | Candidate fused max abs error |
| --- | ---: | ---: |
| Scalar | 0.0 | 0.0 |
| Small/no mask | 0.003715038299560547 | 0.00450444221496582 |
| Small/bias | 0.006483316421508789 | 0.006483316421508789 |
| Vision/bias | 0.0038219690322875977 | 0.0038219690322875977 |

All candidate MATH controls also pass. The six key condition groups have three
independent processes each; all 27 run IDs are unique. All process-level native,
profile and frozen producer hashes validate. The driver remains 32.0.22032.6002.

| Fixture | SHA-256, identical baseline/candidate |
| --- | --- |
| Scalar | `2327964b3cd3acf123400cd57647979abe1aad5fe049f1da631c4dac7fa865d1` |
| Small/no mask | `e39ed68c0b8e26c1949d4c05c1ff75479a43910b027ac06d1472da1251aae8c0` |
| Small/bias | `a889de7a3cdb612299e6f2e885655f14171566414e9cda06ff3faab686cd5a86` |
| Vision/bias | `73a7c931fe5a66837703701cbc3c7076f188c0537e52f3aa941bae072782808a` |

## Actual dispatch and GPU launches

Dispatch records identify the FLASH or EFFICIENT aten operator for every fused
row, including DEFAULT routes. Required forced routes never become MATH.
Both candidate launch logs show `hipModuleLoadDataEx -> hipSuccess`,
`hipModuleGetFunction(..., attn_fwd) -> hipSuccess`, then a nonnull function
launched with grid `(1,2,1)`, block `(128,1,1)`, `hipSuccess` and PAL
`kernel : attn_fwd`. All before/after-SDPA and downstream HIP checks are zero;
synchronization, copies and arithmetic complete.

The baseline logs instead show no successful module loading and null-function,
zero-block launches returning hipErrorInvalidValue. Neither capability flags,
public SDPA calls nor synchronization alone are accepted as launch proof.

[Paired per-case analysis](../../eval/wheel-validation/windows-gfx1151/comparison.json) ·
[Candidate raw process reports/logs](../../eval/wheel-validation/windows-gfx1151/candidate/) ·
[Baseline immutable reports/logs](../../eval/wheel-validation/windows-gfx1151/baseline/) ·
[Parsed launch evidence](../../eval/wheel-validation/windows-gfx1151/continuation/hip-launch-evidence.json) ·
[FLASH raw log](../../eval/wheel-validation/windows-gfx1151/candidate/flash_launch.stderr.log) ·
[EFFICIENT raw log](../../eval/wheel-validation/windows-gfx1151/candidate/efficient_launch.stderr.log).

FP16/FP32 GPU matrices, backward passes, other hardware and full OCR inference
were not run. Their original thresholds/code remain unchanged.
