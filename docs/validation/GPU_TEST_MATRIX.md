# Physical Windows gfx1151 matrix

The original wheel was tested on the physical Radeon 8060S. Each row represents
fresh CPython 3.10.20 x64 processes, unchanged seed 7992, BF16 inputs and independent
CPU FP32 reference arithmetic. No image/DLL substitutions or acceptance error
clearing were performed. [Raw reports and process manifest](../../eval/wheel-validation/windows-gfx1151/baseline/).

| Condition | Requested | Actually observed | Repetitions | Original result |
|---|---|---|---:|---|
| [1,1,1,1], no mask | DEFAULT | FLASH_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| [1,1,1,1], no mask | FLASH | FLASH_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| [1,2,16,64], no mask | FLASH | FLASH_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| [1,2,16,64], additive bias | EFFICIENT | EFFICIENT_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| [54,12,196,64], [54,12,196,196] bias | DEFAULT | EFFICIENT_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| Same vision condition | EFFICIENT | EFFICIENT_ATTENTION | 3 | FAIL, first HIP error after SDPA |
| Scalar/no mask | MATH | MATH, CPU operator profiler | 1 | MATH_PASS |
| Small/no mask | MATH | MATH, CPU operator profiler | 1 | MATH_PASS |
| Small/additive bias | MATH | MATH, CPU operator profiler | 1 | MATH_PASS |
| Vision/additive bias | MATH | MATH, CPU operator profiler | 1 | MATH_PASS |
| Small/additive bias, downstream arithmetic | EFFICIENT | EFFICIENT_ATTENTION | 1 | FAIL |
| Vision copy-only control | MATH requested; no SDPA executed | UNOBSERVED, expected | 1 | CONTROL_PASS |
| Vision/additive bias capability probe | FLASH | UNOBSERVED | 1 | UNSUPPORTED, exit 2 |
| Small/no mask, AMD_LOG_LEVEL=4 | FLASH | FLASH_ATTENTION | 1 | FAIL, null-function/zero-block launch |
| Small/additive bias, AMD_LOG_LEVEL=4 | EFFICIENT | EFFICIENT_ATTENTION | 1 | FAIL, null-function/zero-block launch |

Total: **27 processes: 21 FAIL, 4 MATH_PASS, 1 CONTROL_PASS, 1 UNSUPPORTED**.
The historical RED is reproduced, so BASELINE_DRIFT is not indicated.

## Numerical observations

BF16 uses the original combined criterion `abs(actual-reference) <=
0.03 + 0.03*abs(reference)`. FP16 (0.005/0.005) and FP32 (0.0001/0.0001)
thresholds remain frozen in the producer; those GPU dtype matrices were not run.

| MATH condition | Max absolute error | Failing/nonfinite elements |
|---|---:|---|
| Scalar | 0 | 0 / 0 |
| Small/no mask | 0.003715038299560547 | 0 / 0 |
| Small/bias | 0.006483316421508789 | 0 / 0 |
| Vision/bias | 0.0038219690322875977 | 0 / 0 |

Fused numerical results are **unmeasured**, because the reproducer detects HIP
error 1 first and does not copy corrupt output to CPU for acceptance. Subsequent
PyTorch operations can consume the HIP error; the earlier failure remains recorded
and decisive. Synchronization alone returned normally in these failing cases and
does not prove a valid attention output. The scalar contiguous copy can be a no-op.

## Dispatch and launch evidence

TorchDispatchMode records `aten._scaled_dot_product_flash_attention.default` or
`aten._scaled_dot_product_efficient_attention.default`. The separate MATH cases
use CPU operator profiling to observe `aten::_scaled_dot_product_attention_math`;
they still execute tensor operations on the GPU. Neither capability flags nor
the mere public SDPA call are treated as dispatch evidence.

Both [FLASH](../../eval/wheel-validation/windows-gfx1151/baseline/flash_launch.stderr.log)
and [EFFICIENT](../../eval/wheel-validation/windows-gfx1151/baseline/efficient_launch.stderr.log)
logs contain zero `hipModuleLoadData[Ex]` calls, followed by a null-function launch
with block dimensions `(0,0,0)` returning `hipErrorInvalidValue`. These observations
do not identify a successfully launched attention kernel. No GPU kernel symbol
is inferred from CPU profiling.

## Candidate

No candidate GPU result exists unless a separate source-wheel build, install and
provenance gate succeeds. The original RED and source pin changes cannot establish
candidate behavior. See the [decision report](PYTORCH_212_CURRENT_WHEEL.md).
