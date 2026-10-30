# Original-wheel CLIP block 0 MLP: minimal mechanism RCA

**KEEP EXPERIMENTAL.** The first changed operation is the activation's scalar
multiply, `aten::mul.Scalar`, with source coefficient **1.702**. TensorExpr
lowering rounds that scalar to BF16 (**1.703125**) before generating float
arithmetic. The fused activation also omits the eager path's intermediate BF16
stores. Both changes are visible in generated code and independently reproduce
the captured numerical results. This is a deterministic execution-plan change
for the tested input, not evidence of random GPU output or kill corruption.

Scope: the original published Windows wheel
`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`, reported
Torch source `098172edc887399a593f0c74ae5c40b1fe043801`, physical gfx1151,
BF16 inference, and one frozen synthetic activation. Torch/vision/audio versions
remain 2.12.0/0.27.0/2.11.0 with `+rocm7.14.1`. No candidate wheel is used for new
GPU experiments. No model weights or production execution settings change.

## Boundary capture

One new original-wheel process makes two requests on the frozen two-page case.
The ordinary cold/warm OCR hashes exactly match the earlier RCA. Private tensor
copies retain shape/stride/dtype and record GPU device and storage offset.
The first page's block 0 snapshots are compared across the two requests:

| Boundary | Shape | Stride | Different elements | Max absolute delta |
| --- | --- | --- | ---: | ---: |
| Before MLP layer normalization | 1×257×1024 | 263168,1024,1 | 0 | 0 |
| MLP input / fc1 input | 1×257×1024 | 263168,1024,1 | 0 | 0 |
| fc1 output / activation input | 1×257×4096 | 1052672,4096,1 | 0 | 0 |
| Activation output / fc2 input | 1×257×4096 | 1052672,4096,1 | 786589 | 0.0078125 |
| fc2 output | 1×257×1024 | 263168,1024,1 | 164605 | 0.0078125 |

All external tensors are `torch.bfloat16`, `cuda:0`, offset 0. The first changed
activation element is flat index 0, coordinate [0,0,0]. Matching upstream hashes
are checked before attributing the change. Full private values and per-element
context remain local; public evidence contains hashes, indices and numerical
deltas. [Boundary evidence](clip-mlp-20261030/boundary.json).

## Actual graphs and dispatch

The original official `quick_gelu` graph is multiply → sigmoid → multiply.
Its first execution uses separate ATen operations with profiling nodes. Warm
execution uses a guarded `prim::TensorExprGroup`; the observed CPU profiler
operation is `fused_mul_sigmoid_mul`. Matching guards take the fused path.
Disabling TensorExpr fusion alone, while graph-executor optimization remains
enabled, restores separate operations and exactly reproduces the cold output.

Fresh processes separately test optimized cold, optimized warm, unoptimized
cold and unoptimized warm, twice each. Additional fresh-process controls change
only TensorExpr fusion, GPU fusion eligibility, or profiling mode. All preserve
the same input bytes and layout. No-opt, TE-off, GPU-fusion-off and profiling-off
cold/warm outputs all equal the original cold activation. Matched conditions
repeat exactly. These are diagnostic controls, not production recommendations.
[Executed graphs and operators](clip-mlp-20261030/replay-matrix.json).

This wheel exposes CPU profiler activity only; Kineto GPU timeline/kernel-symbol
tracing is unavailable. The native kernel identity is supported by the actual
JIT compilation log, its emitted source, the fused profiler event, successful
GPU output and zero HIP errors. It is not represented as a GPU timing trace or
an ISA/disassembly capture. The emitted entry point is
`fused_mul_sigmoid_mul`; the direct reduction emits `fused_mul`.
[Generated kernel evidence](clip-mlp-20261030/generated-kernel.json).

## Numerical mechanism and smallest reductions

The generated warm kernel loads BF16 as float and evaluates
`x * (1 / (1 + exp(-x * 1.703125)))`, then stores BF16. There are no intermediate
BF16 stores in that expression. The scalar constant remains 1.702 in the JIT
graph; its lowering to 1.703125 is exposed by the emitted kernel.

Independent CPU FP32 evaluation, with explicitly disclosed BF16 rounding,
gives these exact comparisons for all 1,052,672 captured inputs:

| Reference coefficient / intermediate BF16 rounding | Differences from cold | Differences from warm |
| --- | ---: | ---: |
| 1.702 / yes | 0 | 786589 |
| 1.702 / no | 789133 | 725114 |
| 1.703125 / yes | 124513 | 780785 |
| 1.703125 / no | 786589 | 0 |

Changing just the coefficient or just intermediate rounding does not reproduce
the complete warm result. Both measured changes are needed for this input.
The CPU reference never replaces BF16 GPU parity execution and introduces no
acceptance tolerance. [Reference factorial](clip-mlp-20261030/reference-factorial.json).

The smallest natural original-operator prefix demonstrated here is **two ops**:
multiply then sigmoid. It reproduces a cold/warm difference in fresh processes:
764078 elements, max absolute delta 0.00390625. The one-op prefix stays exact
under ordinary cold/warm execution and does not naturally form a fusion group
under this tested configuration. No global claim about all minimal graphs is made.

To isolate the first operation further, copy the original graph, retain only
its first `aten::mul.Scalar` node and original 1.702 constant, and remove dead
nodes. Run that identical single-node graph through ordinary unoptimized JIT
and direct TensorExpr codegen on the same BF16 CUDA input. Interpreted fallback
is explicitly refused for direct codegen. The generated kernel uses 1.703125f.
There are **135175** different elements, max absolute delta **0.0625**, first
flat index **7**, coordinate [0,0,7]. Ordinary output exactly matches CPU
`BF16(FP32(x)*1.702)`; TensorExpr output exactly matches the 1.703125 version.
This proves the first multiply's scalar-lowering difference. The forced one-node
test is a mechanism reduction, not a claim that a standalone multiplication
naturally fuses. [Single-op evidence](clip-mlp-20261030/single-multiply.json).

Pinned source explains the observations: multiplication lowering calls
`computeTwoOperand`, whose `promoteInputs` casts the scalar to the tensor's BF16
type; CUDA codegen's `HalfRewriter` widens arithmetic and intermediate casts to
float. The constant has already lost precision before that widening. The
emitted code and independent numerical matches support this mechanism; source
inspection alone is not treated as runtime proof.
[Scalar promotion](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/torch/csrc/jit/tensorexpr/operators/misc.cpp),
[HalfRewriter](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/torch/csrc/jit/tensorexpr/half_support.h).

## Exact optimization API and scope

The prior diagnostic used `torch.jit.optimized_execution(False)` around the
entire worker `run(options)`, including loading and inference. Installed
`torch/jit/_fuser.py` saves, sets and restores the **thread-local graph-executor
optimization flag**. It does not directly toggle GPU fusion eligibility,
TensorExpr enablement, profiling-mode flags, RNG, model weights or CUDA device
initialization. On a fresh executor, that flag selects a fallback execution plan
and bypasses profiling/optimization paths, so operator/kernel dispatch changes
indirectly. Existing optimized-plan caches can be returned before that flag is
consulted; it is not a general cache-flush API.

The new no-opt replay scopes the API only around fixed-tensor calls, after GPU
initialization, input creation and ScriptFunction construction. The TE-off
control instead changes `_jit_set_texpr_fuser_enabled(False)` in its isolated
process while optimization stays enabled. Thus CUDA/model initialization is
not needed to reproduce the activation difference.
[Thread-local flag](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/torch/csrc/jit/python/update_graph_executor_opt.cpp),
[Executor plan selection](https://github.com/pytorch/pytorch/blob/098172edc887399a593f0c74ae5c40b1fe043801/torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp).

## Reproduction, review and gates

The replay loads only the frozen activation and reviewed `deepencoder.py`
definitions; it constructs no OCR model and loads no model weights. From the
primary root, with the retained QA/controller and original-wheel environments:

```powershell
.b/qa/Scripts/python.exe experiments/clip_mlp/run_replay.py --output .b/mlp/local-reproduction --controls normal,te-off --prefix 2 --repeats 1 --execute-gpu
```

The private input is produced once from the authorized frozen fixture by
`experiments/clip_mlp/capture.py`; it is not published as a tensor payload.
`multiply.py` performs the one-node reduction. Supervisors use owned Windows
Jobs, repository-local temporary/compiler caches and fresh output directories.

An independent reviewer reproduced all four optimization/position combinations,
TE-off cold/warm, the two-op prefix and the single multiply: **10 fresh GPU
processes**, exact input and output hashes, zero HIP errors and zero active owned
children. Public evidence holds 38 replay processes and four single-node
processes. Only two new full OCR requests were made, for capture; no large OCR
or performance campaign was run. [Independent review](clip-mlp-20261030/independent-review.json).

P2 remains experimental: PR #15's strict MATH/AUTO failure is unchanged. This
mechanism does not erase the independently observed same-context backend
effects. No new semantic/bbox tolerance is justified or introduced here, and
the original 120 measured requests remain unrun pending a reviewed, predeclared
correctness contract. BF16 operator tolerance is not an OCR acceptance policy.

P3 functional recovery remains distinct from strict cold-start parity. Its
earlier failed/PARTIAL ledger is unchanged. Identifying this executor mechanism
does not retroactively pass the cold golden check or validate a production fix.
End-to-end OCR under a TE-only control, other tensors/dtypes/hardware, gradients,
GPU ISA and rare-failure rates remain unvalidated.

The unchanged prior RCA is [draft PR #22](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/22),
head 10067ee. Its 48 receipts, 298/7 CPU logs and 406 historical hashes were
independently reverified before publication. This follow-up is additive.
No weights, production defaults or historical evidence are edited.

Raw tensors, OCR payloads, first-element values and full traces remain local.
Public graphs have source-location comments removed. Downloaded source copies
are bound to immutable source and exact raw-byte hashes; the documented local
newline conversion is preserved separately. TensorExpr generated-code evidence
is attributed to PyTorch's BSD-licensed implementation; original model code is
loaded unchanged under its existing license. The
[publication seal](clip-mlp-20261030/SEAL.json) binds the sanitized artifacts.
