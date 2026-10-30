# Original-wheel cold-start RCA — Windows gfx1151

Decision: **KEEP EXPERIMENTAL**. PR #15 exact parity FAIL and PR #21 strict
restart parity FAIL / historical PARTIAL campaign remain unchanged.

The new original-wheel A1/A2 processes each produce the same cold hash
`e6408c4a652f7a738abdb14299041a346136528098eca7337443e4a00210e49d`
followed by the same warm hash
`c08e7f728ea1e46dc20c2958bf6d3ac857902542935e257a0b64525b7d7b2a8e`.
A3 first-request-only and A4 explicit CUDA init/synchronization before loading
both reproduce the cold hash. Thus these trials show a reproducible invocation
effect, rather than evidence of random cold-start output variation. CUDA
initialization/synchronization alone does not remove it.

The original published wheel SHA256 is
`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`.
The integration source is `82951c73b28bf735f5655e6f5c3306cd170a21c2`;
the immutable official model revision is
`9cf077c4b206e0e2a31879df937b0b0bcba3661c`. Each child checks installed Torch
Python/native members against that wheel and records loaded module identities,
model/source/config/tokenizer/weight hashes, actual worker PID/start time,
executable/import sources, seed/RNG state, flags, cache policy, input/prompt
identities, raw outputs, tokens, HIP status, completion and memory.

Every completed A request observes 24 SAM and 48 CLIP MATH SDPA operations and
12 native DASH prefill calls. The fixed two-page fixtures and generation options
are retained. Seed 0 is set before loading and before each request; initial CPU
and GPU RNG states match. Registered buffer hashes and parameter version
counters do not change across the first request. These observations do not
prove equality of all unregistered Python attributes or native runtime caches.
Each generation creates a fresh official DynamicCache; there is no reused KV
cache. The existing on-disk Triton cache is shared and never deleted; this is a
fresh-process experiment, not an empty-disk-cache experiment.

Two clean processes with stage observers reproduce the same cold/warm hashes.
The first differing observed boundary is CLIP output, with identical
preprocessing, SAM output and CLIP input. Two further deep traces localize it
to CLIP transformer block 0 MLP: fc1 output is identical; fc2 input differs.
The intervening pinned operation is TorchScript `quick_gelu`, implemented as
`x * torch.sigmoid(1.702 * x)`. Block 0 SDPA input/output match before this
difference. This localizes the observed onset outside that SDPA operation;
it does not establish a specific fused-kernel defect.

The first differing output token is index 67 (zero-based), a coordinate token
(`188` versus `189`, the right edge) in page 2's first text box. All prior
generated token IDs match. The cold
pre-guard logits have top-two values 31.875 and 31.75 (margin 0.125); warm values
tie at 31.75 (margin 0). `topk` tie ordering must not be mistaken for argmax's
selected token. The native greedy generation is unchanged. Full token contexts
and traces are local; later logits after a changed prefix are not causal
comparisons of identical contexts.

Two new predeclared scoped TorchScript optimization controls completed. With
`torch.jit.optimized_execution(False)`, all four MATH requests produce the cold
hash, and all captured stage fingerprints match within and across processes.
Under these tested observers, disabling optimization removes the cold/warm
divergence, implicating an optimization-dependent execution path. The broad
intervention does not prove that quick_gelu alone is responsible or identify
a specific fused kernel. Model-held unregistered/native state is not exhaustively
inventoried. No warmup workaround or production runtime change is introduced.
The normal cold trace and optimization-disabled cold trace are not identical:
their first differing boundary is block 1 MLP fc2 input, despite equal final
raw output. Equal OCR output hashes do not prove equality of intermediate
activations. The trace-equality result above is within the disabled condition.

The candidate's first fresh MATH/MATH process independently reproduces the same
cold and warm hashes with normal optimization. This is an observation across
wheel identities, not a substitution for the same-wheel backend matrix.

The first dispatch-observer attempt completed OCR but could not expose the
selected operator; it was rejected. The first JIT-control attempt failed in
the observer before a native return; it was rejected. Both complete failure
records remain local. Later successful controls must not overwrite them.

New sealed metrics are in `cold-start-20261030/`; raw, prompts, tokens and full
failure traces stay under the ignored repository-local experiment workspace.
GPU allocation figures in those receipts describe the APU allocator, not
dedicated VRAM or total system memory. This report makes no speed claim.
