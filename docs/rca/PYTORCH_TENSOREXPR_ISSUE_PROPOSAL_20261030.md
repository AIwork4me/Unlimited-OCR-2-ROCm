# Local issue proposal: BF16 TorchScript TensorExpr changes scalar and rounding semantics

Draft only. No upstream issue has been opened and no upstream contact is
authorized by this mission. Upstream semantic intent remains an open question.

The published AMD Windows CPython 3.10 wheel reports PyTorch
2.12.0+rocm7.14.1, source 098172edc887399a593f0c74ae5c40b1fe043801.
Its SHA256 is
415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c.
The physically tested device is AMD Radeon 8060S, gfx1151. Other hardware and
operating systems have not been validated. No custom wheel is needed.

## Expected and observed

For identical contiguous BF16 CUDA inputs, the original scripted arithmetic
`x * torch.sigmoid(1.702 * x)` first executes separate ATen operations, then
uses a guarded TensorExpr group. Cold and warm BF16 results differ.
The graph retains the floating constant 1.702, but emitted CUDA-like source
uses 1.703125f, the BF16-rounded coefficient, in float arithmetic. The fused
expression also omits eager BF16 rounding between multiplication, sigmoid
and final multiplication. We expected the scripted path to preserve the
source arithmetic's scalar and intermediate dtype semantics; please clarify
whether these two transformations are intentional under this JIT contract.

## Model-free natural reduction

```python
import torch

unit = torch.jit.CompilationUnit('''
def prefix(x):
    return torch.sigmoid(1.702 * x)
''')
# Use a privately retained frozen BF16 tensor for the exact reported metrics.
# An upstream reviewer can independently generate input and report its hash;
# no claim is made that every input or machine reproduces identical counts.
x = torch.linspace(-8, 8, 4096, device="cuda", dtype=torch.float32).bfloat16()
a = unit.prefix(x)
torch.cuda.synchronize()
graph_a = str(torch.jit.last_executed_optimized_graph())
b = unit.prefix(x)
torch.cuda.synchronize()
graph_b = str(torch.jit.last_executed_optimized_graph())
print(graph_a, graph_b, torch.count_nonzero(a != b).item())
```

The linspace variant above is a portable reviewer starting point, not an
executed evidence claim. The already executed, independently reviewed frozen
input experiment in [PR #25](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/25)
used 1,052,672 BF16 elements and this exact two-operation prefix: 764,078
different elements, max absolute difference 0.00390625. A single multiplication
does not naturally form a fusion group in that tested configuration.
No model, tokenizer, OCR request or weights are loaded by that reproducer.

## Forced single-operation isolation

Copy the original scripted graph, keep only `aten::mul.Scalar` with constant
1.702, and pass that graph to direct TensorExpr code generation with
interpreted fallback refused. Compare to ordinary unfused execution on the
same bytes, BF16 dtype, layout and device. The existing reduction records
135,175 differing elements, max absolute difference 0.0625 and first differing
flat index 7. Ordinary output matches CPU `BF16(FP32(x) * 1.702)` exactly;
direct TE matches `BF16(FP32(x) * 1.703125)` exactly. This is a forced compiler
mechanism reduction, not a claim of natural fusion for one isolated multiply.

## Independent reference and source mechanism

CPU FP32 reference evaluation with explicit BF16 boundary rounding matches
the full cold activation exactly using 1.702. CPU FP32 without intermediate
BF16 rounding matches the full warm activation exactly using 1.703125.
Changing only one factor does not explain the full warm result. The full
factorial, exact input/producer identities, executed graphs and emitted code
are sealed in [the kernel RCA](CLIP_MLP_KERNEL_RCA_20261030.md).

Pinned `tensorexpr/operators/misc.cpp` promotes the scalar to tensor BF16;
`half_support.h` HalfRewriter then widens arithmetic and intermediate casts
to float after scalar precision has already been lost. These source findings
are supported by emitted code and numerical references. GPU Kineto activity
is unavailable in this wheel; profiler evidence consists of CPU dispatch
events plus JIT-emitted kernel source, not an ISA or GPU timeline trace.

Disabling `_jit_set_texpr_fuser_enabled(False)` before constructing/executing
the function in a fresh owned process removes the difference while graph
optimization remains enabled. That internal flag is process-global and does
not invalidate previously cached plans; it is a diagnostic control, not a
thread-safe production workaround. A compiler correction must separately
preserve scalar opmath semantics and required intermediate BF16 rounding.

Requested upstream clarification: scalar promotion rules for this graph,
allowed BF16 fusion roundings, supported scoped controls, and an appropriate
regression-test contract. No upstream behavior outside this exact artifact
and physical device is inferred.

A new, separate two-process prototype experiment preserves coefficient
precision by explicit FP32 opmath. Scalar-only execution differs from original
eager BF16 at 789133 elements, max 0.0078125, cold and warm. Explicit BF16
casts between operations match cold, but warm fusion has the same difference.
Actual emitted kernels retain 1.702000021934509f while omitting intermediate
BF16 roundings; independent CPU references and direct saved-byte inspection
agree. These prototypes are rejected as complete mitigations. This supports
testing both scalar promotion and intermediate-rounding rules in any proposed
compiler correction; it does not prove that every possible correction fails.
