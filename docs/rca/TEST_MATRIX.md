# Executed SDPA test matrix

All 113 standalone child reports below come from fresh native Windows gfx1151 processes. A/B/A changes only the image-directory payload in the secondary environment. Primary baseline remains unchanged. Child stdout/stderr and process exit codes accompany each JSON. The two explicitly non-acceptance clearing diagnostics are retained and excluded from acceptance claims.

Frozen tolerances: BF16 .03/.03; FP16 .005/.005; FP32 .0001/.0001 (atol/rtol). CPU seed 7992, exact quantized fixture hashing. All argument arrays, full run IDs, numerical statistics and provenance joins are in the [machine-readable index](../../eval/rca/gfx1151-rocm7141/experiment-manifest.json).

## Groups

| Group | PASS | FAIL | UNSUPPORTED |
|---|---:|---:|---:|
| baseline | 1 | 2 | 2 |
| hip-A-restored.json | 0 | 1 | 0 |
| hip-A.json | 0 | 1 | 0 |
| hip-B.json | 1 | 0 | 0 |
| isolation | 4 | 26 | 1 |
| torch-only-A | 5 | 5 | 1 |
| torch-only-A-restored-full | 1 | 2 | 2 |
| torch-only-A-restored-small | 5 | 5 | 1 |
| torch-only-B-full | 3 | 0 | 2 |
| torch-only-B-isolation | 30 | 0 | 1 |
| torch-only-B-small | 10 | 0 | 1 |

UNOBSERVED means the tracing mechanism did not expose a named backend; it is never replaced with the requested flag. The separate profiler-MATH case observes `aten::_scaled_dot_product_attention_math`. HIP traces without dispatcher instrumentation remain UNOBSERVED in their raw reports. Unsupported routes execute no attention kernel.

## Individual conditions

`A`: torch-only-A; `B`: torch-only-B-*; `A restored`: torch-only-A-restored-*. baseline/isolation use the untouched original OCR venv. Full names retain provenance context. Max abs is measured against the independent CPU reference; dash means not measured. A process with a pending error does not read its invalid output in acceptance.

| Artifact / condition | Shape | Dtype / mask | Requested → observed | Status | Max abs | Run ID |
|---|---|---|---|---|---:|
| [baseline/cudnn_attention.json](../../eval/rca/gfx1151-rocm7141/baseline/cudnn_attention.json) | 54×12×196×64 | bfloat16 / bias | CUDNN_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `5b50d0c5-cf99-4f2c-89b7-3c4923ff6757` |
| [baseline/default.json](../../eval/rca/gfx1151-rocm7141/baseline/default.json) | 54×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `3b502856-1295-4640-a134-e9498dabb4a8` |
| [baseline/efficient_attention.json](../../eval/rca/gfx1151-rocm7141/baseline/efficient_attention.json) | 54×12×196×64 | bfloat16 / bias | EFFICIENT_ATTENTION → EFFICIENT_ATTENTION | FAIL | — | `dbb8d293-33c5-4b35-965e-034f5382e05a` |
| [baseline/flash_attention.json](../../eval/rca/gfx1151-rocm7141/baseline/flash_attention.json) | 54×12×196×64 | bfloat16 / bias | FLASH_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `39e4b43a-9aeb-47a0-8317-f668c1eb809a` |
| [baseline/math.json](../../eval/rca/gfx1151-rocm7141/baseline/math.json) | 54×12×196×64 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0.00382196903 | `582d15ac-d22b-4b8c-8820-023a63a05cb0` |
| [hip-A-restored.json](../../eval/rca/gfx1151-rocm7141/hip-A-restored.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → UNOBSERVED | FAIL | — | `89b06073-04cc-41d6-9ba8-6988b95b3480` |
| [hip-A.json](../../eval/rca/gfx1151-rocm7141/hip-A.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → UNOBSERVED | FAIL | — | `b95f3025-398f-4e59-8a9b-213d7e4390cd` |
| [hip-B.json](../../eval/rca/gfx1151-rocm7141/hip-B.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → UNOBSERVED | PASS | 0 | `02ea76a8-2637-4900-894f-00f6b4d3a2c4` |
| [isolation/bfloat16-bias.json](../../eval/rca/gfx1151-rocm7141/isolation/bfloat16-bias.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `edb40d43-8030-42fd-8683-9314f6ceffc5` |
| [isolation/bfloat16-bool.json](../../eval/rca/gfx1151-rocm7141/isolation/bfloat16-bool.json) | 1×2×16×64 | bfloat16 / bool | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `9c7ca41b-2a77-4e03-9447-2dd4ff65246f` |
| [isolation/bfloat16-none.json](../../eval/rca/gfx1151-rocm7141/isolation/bfloat16-none.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `05450ec8-6ba9-46b5-8b85-535154688f47` |
| [isolation/causal.json](../../eval/rca/gfx1151-rocm7141/isolation/causal.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `10f13fa1-58ad-428f-8f46-ee40d0c0634d` |
| [isolation/diagnostic-clearing.json](../../eval/rca/gfx1151-rocm7141/isolation/diagnostic-clearing.json) | 54×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL (NON-ACCEPTANCE) | 1.57019746 | `e1809368-a2a1-459c-ad2d-b9bdf3f5176a` |
| [isolation/float16-bias.json](../../eval/rca/gfx1151-rocm7141/isolation/float16-bias.json) | 1×2×16×64 | float16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `de174641-7be1-4226-876b-f11940f360a2` |
| [isolation/float16-bool.json](../../eval/rca/gfx1151-rocm7141/isolation/float16-bool.json) | 1×2×16×64 | float16 / bool | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `b573bddc-91e0-4e72-ac2c-d63b6a3a66d4` |
| [isolation/float16-none.json](../../eval/rca/gfx1151-rocm7141/isolation/float16-none.json) | 1×2×16×64 | float16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `d1030fe7-c49d-42c0-b871-093d2c442beb` |
| [isolation/float32-bias.json](../../eval/rca/gfx1151-rocm7141/isolation/float32-bias.json) | 1×2×16×64 | float32 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `950e8a5c-fe7d-42ef-afd1-3abd73fc012d` |
| [isolation/float32-bool.json](../../eval/rca/gfx1151-rocm7141/isolation/float32-bool.json) | 1×2×16×64 | float32 / bool | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `21f75f93-078c-4477-950e-b1aead81b24c` |
| [isolation/float32-none.json](../../eval/rca/gfx1151-rocm7141/isolation/float32-none.json) | 1×2×16×64 | float32 / none | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `72770705-3c29-4665-9cf4-59a1821dbd67` |
| [isolation/min-1x12x196x64.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x12x196x64.json) | 1×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `0d5aa07d-d6ed-4dd7-82a1-ec7700d04aea` |
| [isolation/min-1x1x16x64.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x16x64.json) | 1×1×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `81efb835-b05e-4d62-9b61-e687fe69c156` |
| [isolation/min-1x1x196x64.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x196x64.json) | 1×1×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `dd215aa6-bf06-4da6-971c-160418d61f09` |
| [isolation/min-1x1x1x1.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x1x1.json) | 1×1×1×1 | bfloat16 / bias | DEFAULT → UNOBSERVED | PASS | 0 | `2e4392e5-d473-4f42-a75a-6e48c7258370` |
| [isolation/min-1x1x1x32.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x1x32.json) | 1×1×1×32 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `e6d6d275-b389-4a62-b056-a7666c64ab8a` |
| [isolation/min-1x1x1x64.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x1x64.json) | 1×1×1×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `95670810-7f1f-4eef-a609-282c45df9091` |
| [isolation/min-1x1x1x8.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x1x8.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `69163531-2f77-461f-8b2b-73959c184e88` |
| [isolation/min-1x1x2x64.json](../../eval/rca/gfx1151-rocm7141/isolation/min-1x1x2x64.json) | 1×1×2×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `eb009887-ab64-4c29-a523-aa105bac8188` |
| [isolation/noinstrumentation.json](../../eval/rca/gfx1151-rocm7141/isolation/noinstrumentation.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → UNOBSERVED | FAIL | — | `83f1aec7-421b-4020-8905-c045fcb3da12` |
| [isolation/nomask-CUDNN_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/isolation/nomask-CUDNN_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | CUDNN_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `f85c0d0d-6e1d-4294-96f7-635350164803` |
| [isolation/nomask-DEFAULT.json](../../eval/rca/gfx1151-rocm7141/isolation/nomask-DEFAULT.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `2206fcc0-bed0-4d23-beff-2ff6e75c4c53` |
| [isolation/nomask-EFFICIENT_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/isolation/nomask-EFFICIENT_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | EFFICIENT_ATTENTION → EFFICIENT_ATTENTION | FAIL | — | `c5e7f2a8-b90e-4f66-a646-40290bbf8349` |
| [isolation/nomask-FLASH_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/isolation/nomask-FLASH_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | FLASH_ATTENTION → FLASH_ATTENTION | FAIL | — | `80f5dc72-339f-4970-bfb4-9a7d41179a2e` |
| [isolation/nomask-MATH.json](../../eval/rca/gfx1151-rocm7141/isolation/nomask-MATH.json) | 1×2×16×64 | bfloat16 / none | MATH → UNOBSERVED | PASS | 0.0037150383 | `4b578b07-a12f-4124-aa42-38455dd8939b` |
| [isolation/noncontiguous.json](../../eval/rca/gfx1151-rocm7141/isolation/noncontiguous.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `e4bcc2b2-bf72-4dec-862a-515da94e9582` |
| [isolation/order-arithmetic.json](../../eval/rca/gfx1151-rocm7141/isolation/order-arithmetic.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `b3091aca-9198-4110-bed9-9b2dfd30c0a5` |
| [isolation/order-cast.json](../../eval/rca/gfx1151-rocm7141/isolation/order-cast.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `d293be34-39a4-4eae-ad59-45ce9001e93a` |
| [isolation/order-copy_only.json](../../eval/rca/gfx1151-rocm7141/isolation/order-copy_only.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → UNOBSERVED | PASS | — | `824de23f-e801-4876-ba7a-af0eb55ce1d7` |
| [isolation/profiler-DEFAULT.json](../../eval/rca/gfx1151-rocm7141/isolation/profiler-DEFAULT.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `84289a52-dbee-4739-8eee-ffb323f0d918` |
| [isolation/profiler-MATH.json](../../eval/rca/gfx1151-rocm7141/isolation/profiler-MATH.json) | 1×2×16×64 | bfloat16 / bias | MATH → MATH | PASS | 0.00648331642 | `969ed70c-153a-48bc-a169-c2e15802319b` |
| [torch-only-A/repeat-default-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-default-1.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `5cc5903e-040b-4582-8c0e-6aea82ebb2a0` |
| [torch-only-A/repeat-default-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-default-2.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `d9ec0cc0-42f5-4935-87d0-4754b25ae0bf` |
| [torch-only-A/repeat-default-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-default-3.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `94fb4278-33c6-48ef-b602-e8df101b26ec` |
| [torch-only-A/repeat-math-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-math-1.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `6a2ca548-f2b2-42a7-9aa4-a356a6d9f625` |
| [torch-only-A/repeat-math-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-math-2.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `a71ce9dd-9374-464c-803b-a59230fab14b` |
| [torch-only-A/repeat-math-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/repeat-math-3.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `b1ef64f6-aebc-4ea7-9806-22eda2399fb9` |
| [torch-only-A/scalar-bias-efficient.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/scalar-bias-efficient.json) | 1×1×1×1 | bfloat16 / bias | EFFICIENT_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `5a9adc1a-e6bd-4dc9-b435-3b8a4ddae5d5` |
| [torch-only-A/scalar-flash.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/scalar-flash.json) | 1×1×1×1 | bfloat16 / none | FLASH_ATTENTION → FLASH_ATTENTION | FAIL | — | `acb6ad36-4157-4a60-bc94-b58a67faf493` |
| [torch-only-A/scalar-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/scalar-math.json) | 1×1×1×1 | bfloat16 / none | MATH → UNOBSERVED | PASS | 0 | `fd7a4940-06ac-4c1f-bf1d-03fe0981b228` |
| [torch-only-A/scalar-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/scalar-none.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `edc25492-5643-4355-a1cd-40669094f8ef` |
| [torch-only-A/small-eight-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-A/small-eight-math.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `fd3da195-28fd-4618-960f-40a6ba4da09b` |
| [torch-only-A-restored-full/cudnn_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-full/cudnn_attention.json) | 54×12×196×64 | bfloat16 / bias | CUDNN_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `c247ed54-713b-4a26-8b0f-487ec64928c5` |
| [torch-only-A-restored-full/default.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-full/default.json) | 54×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `489afbc7-3dbb-4864-b898-3d8783a25fb2` |
| [torch-only-A-restored-full/efficient_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-full/efficient_attention.json) | 54×12×196×64 | bfloat16 / bias | EFFICIENT_ATTENTION → EFFICIENT_ATTENTION | FAIL | — | `b3104930-8d86-4d1e-9f3c-94b1b593873a` |
| [torch-only-A-restored-full/flash_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-full/flash_attention.json) | 54×12×196×64 | bfloat16 / bias | FLASH_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `d1fededd-5852-4f2c-ae4a-df4cf99e69c4` |
| [torch-only-A-restored-full/math.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-full/math.json) | 54×12×196×64 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0.00382196903 | `4406c750-74d8-46ae-ae16-0e9b17794d00` |
| [torch-only-A-restored-small/repeat-default-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-default-1.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `569ba4d4-d0e0-489a-98a4-73560a1c05ae` |
| [torch-only-A-restored-small/repeat-default-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-default-2.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `8d3a1306-9b9d-4c2a-8521-9a4b976c05b6` |
| [torch-only-A-restored-small/repeat-default-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-default-3.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | FAIL | — | `3536d706-ca22-4458-b2a9-b0144ab3428a` |
| [torch-only-A-restored-small/repeat-math-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-math-1.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `82055a0a-a89f-4eda-95b4-33a838f3186c` |
| [torch-only-A-restored-small/repeat-math-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-math-2.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `2222f279-1a51-446f-89fd-c1e1848e9d21` |
| [torch-only-A-restored-small/repeat-math-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/repeat-math-3.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `6eaa11ab-aeb0-4280-a24d-43c47832179e` |
| [torch-only-A-restored-small/scalar-bias-efficient.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/scalar-bias-efficient.json) | 1×1×1×1 | bfloat16 / bias | EFFICIENT_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `2daace1a-960c-44bb-be88-1069e1e9cd13` |
| [torch-only-A-restored-small/scalar-flash.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/scalar-flash.json) | 1×1×1×1 | bfloat16 / none | FLASH_ATTENTION → FLASH_ATTENTION | FAIL | — | `98a81c9d-eed9-4a4c-8889-320083bfa04d` |
| [torch-only-A-restored-small/scalar-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/scalar-math.json) | 1×1×1×1 | bfloat16 / none | MATH → UNOBSERVED | PASS | 0 | `7e46019f-0217-4c19-9a8b-1c52a76a3e98` |
| [torch-only-A-restored-small/scalar-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/scalar-none.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | FAIL | — | `af8f5c94-0de1-4358-a969-617980e3fe1c` |
| [torch-only-A-restored-small/small-eight-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-A-restored-small/small-eight-math.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `e9f03c8a-6c34-408c-aa6c-71eb5f72a38c` |
| [torch-only-B-full/cudnn_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-full/cudnn_attention.json) | 54×12×196×64 | bfloat16 / bias | CUDNN_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `ca9a7b68-4a93-4552-a4fd-0e16d4bdb8ad` |
| [torch-only-B-full/default.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-full/default.json) | 54×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00382196903 | `738b674f-32b1-43f8-8132-c49aeda6db94` |
| [torch-only-B-full/efficient_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-full/efficient_attention.json) | 54×12×196×64 | bfloat16 / bias | EFFICIENT_ATTENTION → EFFICIENT_ATTENTION | PASS | 0.00382196903 | `012c3e6a-fc19-4b91-bab6-1712766e2172` |
| [torch-only-B-full/flash_attention.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-full/flash_attention.json) | 54×12×196×64 | bfloat16 / bias | FLASH_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `f47c3282-d2e0-41b6-bf80-39d80cc71081` |
| [torch-only-B-full/math.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-full/math.json) | 54×12×196×64 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0.00382196903 | `f500e26f-9a5e-4c3f-9070-585302362a30` |
| [torch-only-B-isolation/bfloat16-bias.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/bfloat16-bias.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00648331642 | `90d50cc3-53c6-4be8-a03a-820f1f2ed3ef` |
| [torch-only-B-isolation/bfloat16-bool.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/bfloat16-bool.json) | 1×2×16×64 | bfloat16 / bool | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00433242321 | `fa7fb04f-e4b1-420a-a7cb-871d9d21e8ea` |
| [torch-only-B-isolation/bfloat16-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/bfloat16-none.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | PASS | 0.00450444221 | `da370cdb-6d79-4c81-8f05-b492b65708c8` |
| [torch-only-B-isolation/causal.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/causal.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | PASS | 0.00580596924 | `96989600-a843-41d8-b0b6-e05982c85de1` |
| [torch-only-B-isolation/diagnostic-clearing.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/diagnostic-clearing.json) | 54×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS (NON-ACCEPTANCE) | 0.00382196903 | `128e3eb5-fe1c-4bec-be10-c1e7fab894ad` |
| [torch-only-B-isolation/float16-bias.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float16-bias.json) | 1×2×16×64 | float16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.000514864922 | `737ce7e4-4d9a-4f0d-b4aa-89b3a23f9ff0` |
| [torch-only-B-isolation/float16-bool.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float16-bool.json) | 1×2×16×64 | float16 / bool | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.000548005104 | `1bab7c05-20bc-418b-9da6-aa1d74fe6fae` |
| [torch-only-B-isolation/float16-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float16-none.json) | 1×2×16×64 | float16 / none | DEFAULT → FLASH_ATTENTION | PASS | 0.000553488731 | `79d0232d-6ec6-49dc-81c8-d25e7fda0466` |
| [torch-only-B-isolation/float32-bias.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float32-bias.json) | 1×2×16×64 | float32 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 5.96046448e-07 | `0556078e-6e25-4a6c-bd90-fa2fd7f95641` |
| [torch-only-B-isolation/float32-bool.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float32-bool.json) | 1×2×16×64 | float32 / bool | DEFAULT → EFFICIENT_ATTENTION | PASS | 3.57627869e-07 | `2a93e65b-9de2-40dc-b091-8ba909bc74c0` |
| [torch-only-B-isolation/float32-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/float32-none.json) | 1×2×16×64 | float32 / none | DEFAULT → EFFICIENT_ATTENTION | PASS | 4.76837158e-07 | `ed5b8fff-74d3-4505-a59b-9367088ddda6` |
| [torch-only-B-isolation/min-1x12x196x64.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x12x196x64.json) | 1×12×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00211298466 | `ee6364e9-21c3-44fc-b41c-0649d41e3bad` |
| [torch-only-B-isolation/min-1x1x16x64.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x16x64.json) | 1×1×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00394642353 | `3e806f39-3b0b-48d8-8c54-2e5320c3bdb2` |
| [torch-only-B-isolation/min-1x1x196x64.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x196x64.json) | 1×1×196×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00168466568 | `84a733cd-b345-400c-b9d7-48f6d4805592` |
| [torch-only-B-isolation/min-1x1x1x1.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x1x1.json) | 1×1×1×1 | bfloat16 / bias | DEFAULT → UNOBSERVED | PASS | 0 | `34ef2805-04fe-4cd5-830f-50730a2e1f78` |
| [torch-only-B-isolation/min-1x1x1x32.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x1x32.json) | 1×1×1×32 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `453e822d-cf5b-4a5b-b83a-d3bb74165967` |
| [torch-only-B-isolation/min-1x1x1x64.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x1x64.json) | 1×1×1×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `9769f7ef-9e08-475b-8941-4772a4c182df` |
| [torch-only-B-isolation/min-1x1x1x8.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x1x8.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `cfc8ca7c-65d3-49fb-b218-dbf7af01cb2a` |
| [torch-only-B-isolation/min-1x1x2x64.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/min-1x1x2x64.json) | 1×1×2×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00782632828 | `d2b45f12-e7c1-4392-b395-a04daeb25136` |
| [torch-only-B-isolation/noinstrumentation.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/noinstrumentation.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → UNOBSERVED | PASS | 0.00648331642 | `b0b319c4-6bb8-4ed6-b3e3-a98d45346d05` |
| [torch-only-B-isolation/nomask-CUDNN_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/nomask-CUDNN_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | CUDNN_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `27e9f9f3-c0b0-4f2b-8cd7-6b449bf86e58` |
| [torch-only-B-isolation/nomask-DEFAULT.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/nomask-DEFAULT.json) | 1×2×16×64 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | PASS | 0.00450444221 | `0a03b4b1-311a-4b12-b477-cd286a8a62fb` |
| [torch-only-B-isolation/nomask-EFFICIENT_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/nomask-EFFICIENT_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | EFFICIENT_ATTENTION → EFFICIENT_ATTENTION | PASS | 0.00450444221 | `37addf77-db97-4b4f-a3a9-a5a35aa98012` |
| [torch-only-B-isolation/nomask-FLASH_ATTENTION.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/nomask-FLASH_ATTENTION.json) | 1×2×16×64 | bfloat16 / none | FLASH_ATTENTION → FLASH_ATTENTION | PASS | 0.00450444221 | `568e4666-3402-410b-b062-cd9171e3ba30` |
| [torch-only-B-isolation/nomask-MATH.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/nomask-MATH.json) | 1×2×16×64 | bfloat16 / none | MATH → UNOBSERVED | PASS | 0.0037150383 | `58c42600-83ec-4aac-9bff-444db532156a` |
| [torch-only-B-isolation/noncontiguous.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/noncontiguous.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00648331642 | `48f5e3c1-ae37-4b10-8df6-d7a874c07b6b` |
| [torch-only-B-isolation/order-arithmetic.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/order-arithmetic.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00648331642 | `53a94f6f-1542-46b6-8560-456951c775eb` |
| [torch-only-B-isolation/order-cast.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/order-cast.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00648331642 | `03e1fb92-5dba-4c29-be90-79c9862f45be` |
| [torch-only-B-isolation/order-copy_only.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/order-copy_only.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → UNOBSERVED | PASS | — | `60ccb504-814b-4c35-8659-bbe486921e88` |
| [torch-only-B-isolation/profiler-DEFAULT.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/profiler-DEFAULT.json) | 1×2×16×64 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0.00648331642 | `80b36d17-ee01-4de4-81ff-8355e3086de3` |
| [torch-only-B-isolation/profiler-MATH.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-isolation/profiler-MATH.json) | 1×2×16×64 | bfloat16 / bias | MATH → MATH | PASS | 0.00648331642 | `e8cea347-0342-43d3-b6c5-9718632ca2da` |
| [torch-only-B-small/repeat-default-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-default-1.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `f7604f1d-02b8-4d72-8c6f-878adb5b7d9a` |
| [torch-only-B-small/repeat-default-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-default-2.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `29bd45b3-8e4f-4409-8a7f-98203e896565` |
| [torch-only-B-small/repeat-default-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-default-3.json) | 1×1×1×8 | bfloat16 / bias | DEFAULT → EFFICIENT_ATTENTION | PASS | 0 | `f741fa91-1e7e-44bd-8725-f4317bb0091b` |
| [torch-only-B-small/repeat-math-1.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-math-1.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `f6d74865-1d62-4b45-8227-4c0fa295aead` |
| [torch-only-B-small/repeat-math-2.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-math-2.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `d6bdd5e0-d616-43a9-aadf-490171baba66` |
| [torch-only-B-small/repeat-math-3.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/repeat-math-3.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `86cc6f03-3d8b-47cd-9127-4ee996ae8129` |
| [torch-only-B-small/scalar-bias-efficient.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/scalar-bias-efficient.json) | 1×1×1×1 | bfloat16 / bias | EFFICIENT_ATTENTION → UNOBSERVED | UNSUPPORTED | — | `f21d523a-653c-4a3b-8e23-d1f524c9517a` |
| [torch-only-B-small/scalar-flash.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/scalar-flash.json) | 1×1×1×1 | bfloat16 / none | FLASH_ATTENTION → FLASH_ATTENTION | PASS | 0 | `d5bc25c2-b875-4efa-a690-aeb699ce08fc` |
| [torch-only-B-small/scalar-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/scalar-math.json) | 1×1×1×1 | bfloat16 / none | MATH → UNOBSERVED | PASS | 0 | `2cec052f-0556-4aea-8f08-071e313f6ad8` |
| [torch-only-B-small/scalar-none.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/scalar-none.json) | 1×1×1×1 | bfloat16 / none | DEFAULT → FLASH_ATTENTION | PASS | 0 | `645f5ec1-f514-418e-9b45-1a6e8c638822` |
| [torch-only-B-small/small-eight-math.json](../../eval/rca/gfx1151-rocm7141/torch-only-B-small/small-eight-math.json) | 1×1×1×8 | bfloat16 / bias | MATH → UNOBSERVED | PASS | 0 | `883d4bfd-bc64-4b6d-be63-2eba7156e25e` |

## Model cross-check

[Fresh-process manifest](../../eval/rca/gfx1151-rocm7141/model-final/process-manifest.json): SAM default exits 1; SAM MATH exits 0 after three block forwards and a reference check; two accepted English OCR requests (normal and Unicode/recovery) retain the existing adapter, BF16, unchanged official hashes and 12 native DASH calls per request. The larger Phase 1 suite was not rerun.

## Interpretation

The scalar no-mask FLASH failure has the smallest possible positive dimensions. The scalar bias input uses MATH and passes; efficient D=1 is rejected as unsupported. Thus shape-only minimization that silently changes dispatch can conceal the defect. Copy-only passes; arithmetic/cast/copy following fused SDPA retain or report its earlier error. B passes all supported counterparts of the original isolation matrix. Multiple clean-process repetitions support deterministic reproduction for the fixed seed; no reliability estimate for arbitrary hardware is claimed.

## Post-review harness validation

Later user-requested real-PDF investigations are indexed separately: [first-page actual SAM/CLIP replay and the full 15-page native infer_multi cohorts](FULL_PDF_PIPELINE.md). Their input PDF, model outputs, full traces, failures, numerical thresholds and inventory are in the [PDF evidence supplement](../../eval/rca/gfx1151-rocm7141/pdf-followup/README.md). They do not alter the historical matrix below or its acceptance counts.

The original matrix above contains 113 child reports. After the independent review fixes, three additional fresh-process conditions used the current producer code: [scalar DEFAULT](../../eval/rca/gfx1151-rocm7141/review-regression/scalar-default.json) FAIL with observed FLASH, [scalar MATH with profiler](../../eval/rca/gfx1151-rocm7141/review-regression/scalar-math-profile.json) PASS with observed MATH, and [copy-only](../../eval/rca/gfx1151-rocm7141/review-regression/copy-only.json) PASS. They are indexed separately as `post_review_records` in the experiment manifest; historical records/logs remain untouched.

Final checks: [command manifest](../../eval/rca/gfx1151-rocm7141/checks-final/process-manifest.json), [CPU suite](../../eval/rca/gfx1151-rocm7141/checks-final/pytest-cpu.stdout.log) **68 passed, 5 expected skips**, [physical suite](../../eval/rca/gfx1151-rocm7141/checks-final/pytest-physical.stdout.log) **73 passed, 15 upstream TorchScript warnings**. The physical suite includes all optional CPU-reference tests plus the existing GPU/OCR tests. Ruff, formatting, mypy, build and pip check exit 0. [Independent review and dispositions](REVIEW.md).
