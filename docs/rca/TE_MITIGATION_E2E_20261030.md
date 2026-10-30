# TensorExpr-only end-to-end OCR qualification

**KEEP EXPERIMENTAL.** On the original published wheel, disabling only
TensorExpr before model loading restores exact cold/warm end-to-end OCR parity
for the frozen two-page synthetic case in two independent processes. A narrower
per-model use of the original eager QuickGELU arithmetic reproduces the same
result without changing global JIT settings. This is a new experimental finding;
it does not change PR #15, #21, #22, #25, commit 10067ee or the old reference.

## Identity and prospective controls

All original-wheel OCR arms use SHA256
415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c,
immutable model 9cf077c4b206e0e2a31879df937b0b0bcba3661c and original weights.
Windows gfx1151 / Radeon 8060S and the torch/vision/audio versions remain
2.12.0/0.27.0/2.11.0 with +rocm7.14.1. Frozen integration source is
82951c73b28bf735f5655e6f5c3306cd170a21c2. Workers verify installed wheel
members, loaded native modules, model/source/config/tokenizer/weight hashes,
fixture hashes and physical device. They record input/prompt IDs and generation
options; the evaluator compares output/prompt IDs across matched requests.
No model, production source, default backend or system installation is changed.

The [prospective protocol](../superpowers/plans/2026-10-30-te-mitigation-e2e.md)
declares two requests per fresh P0 process, with seeds reset to zero before
load/request, no inference warmup and one model initialization. Every child is
owned by a Windows Job; descendants are checked after completion. Disk compiler
caches are shared and disclosed; model/module/KV state is fresh. Deterministic
algorithms remain at the original disabled setting; this is recorded, not
silently changed to force parity.

Exploratory P0 v1 is preserved separately. Review found its pre-MLP tensor call
label offset by one; those receipts do not qualify stage parity. Corrected P0
v2 uses fresh identities and aligned invocation labels. Later binding controls
also verify actual arm arguments and child declaration SHA/item, and seal the
Job controller. All qualification requires more than exit zero: physical and
functional checks, actual graphs/dispatch, full layer/call coverage, valid
geometry, raw-file integrity, unchanged parameter versions/buffers and two
distinct fresh process identities. Repeated-input suite aliases are compared
by frozen input/cap equivalence while preserving their ordered case positions.

## Two-page end-to-end result

| Original-wheel option | Cold/warm raw and output-ID parity | Fresh-process parity | Block-0 stage parity |
| --- | --- | --- | --- |
| Default | FAIL | Exact within matched cold or warm contexts | FAIL at activation |
| TE-only disabled before load | PASS | PASS | PASS, every captured stage/call |
| Per-model original eager QuickGELU | PASS | PASS | PASS, every captured stage/call |

Default cold raw SHA256 is
e6408c4a652f7a738abdb14299041a346136528098eca7337443e4a00210e49d;
default warm is
c08e7f728ea1e46dc20c2958bf6d3ac857902542935e257a0b64525b7d7b2a8e.
Every TE-off and eager P0 request equals the cold raw, output-token and
prompt-token result. Content, labels, reading order and non-bbox bytes agree.
The default difference remains page 2's text bbox coordinate 188→189; the new
controls remove that difference with no coordinate tolerance.

All twelve qualified P0 requests finish with stop metadata and functional
checks passing. Two content pages plus the documented empty terminal PAGE
marker and six BLOCK markers are preserved. Actual SAM/CLIP MATH SDPA,
twelve native DASH calls, zero sticky HIP errors, restored runtime state and
all 24 CLIP layers are verified per request. The default first request already
has 47/48 fused QuickGELU invocations: only the first call starts unfused. Its
second request has 48/48. TE-off and eager have zero fused invocations.
Graph-executor optimization remains enabled in every arm.

The first page's pre-MLP and first-linear outputs are byte-identical under
default cold/warm execution; the activation output changes, followed by fc2.
TE-off/eager remove those stage differences on both pages. External tensors
remain BF16 cuda:0 with original shapes and strides. Private tensor values
stay local; public evidence contains hashes/contracts/deltas only. The completed
ten-process kernel RCA is reused as evidence and has not been repeated.

## Three distinct mitigation options

| Option | Scope / cache behavior | Numerical and deployment assessment |
| --- | --- | --- |
| TE-only flag | Process-global `_jit_set_texpr_fuser_enabled(False)` before loading; optimization stays enabled | Works in owned fresh OCR workers; inadmissible as an untracked shared-process toggle |
| Original eager QuickGELU | Verified official Python AST; clone each CLIP MLP forward's globals per model instance; replace only its QuickGELU call | P0 preserves original BF16 arithmetic without global JIT state changes; deployment concurrency/performance remain unqualified |
| Scalar-promotion correction | Separate FP32-opmath / explicit-BF16-boundary prototypes; no custom wheel | Investigative only; preserving the coefficient alone cannot establish the complete eager contract |

TE flag restoration is cleanup, not cache invalidation. Existing optimized plans
can survive restoring flags, and concurrent threads share TE enablement. This
option is admissible only as an explicitly isolated diagnostic process design
here. No shared production context manager is proposed.

The eager scope checks the exact imported deepencoder hash and original MLP
bytecode shape, then uses its unchanged fc1→QuickGELU→fc2 sequence. It applies
to all 24 CLIP layers. The shared original ScriptFunction is left intact but
unused by these instance methods; SAM, other JIT functions, profiling settings,
graph optimization and global TE enablement retain their original configuration.
The observer counters are single-threaded instrumentation. This does not prove
concurrent model serving safe, nor does it eliminate ordinary model concurrency
constraints. The scope must be established once before an exclusively owned
model is served, rather than toggled during concurrent inference.

Eager execution introduces separate pointwise launches and Python dispatch.
Those costs and lost fusion may matter in other workloads/layers/shapes. Exact
hashes do not prove bbox truth, semantic quality or an acceptable latency budget.
The frozen text/number/page/structure contract is checked independently; no
new bbox acceptance tolerance is introduced. Valid coordinates and exact
stability do not establish ground-truth geometric accuracy without annotated
layout reference. Historical bbox failures remain failures.

## Performance and outstanding gates

P0 request timings include tensor copies, CPU profiling and graph serialization.
Suite timings omit tensor copies but retain profiling/serialization. Shared
compiler caches and fixed arm order are disclosed. These are diagnostic
latency/memory observations, not unbiased comparative benchmarks, speedup
claims or release performance qualification. APU allocator peaks measure
shared-system allocations and are not dedicated VRAM or total system memory.

P2's historical strict MATH/AUTO FAIL and P3's earlier FAILED/PARTIAL ledger
remain immutable. Functional recovery and strict cold-start parity are separate
categories; this mission does not rerun kill recovery or claim kill causality.
The new TE control explains a tested cold/warm path without rewriting those
receipts. Production stays experimental and its MATH default is unchanged.

The [matched-context backend protocol](MATCHED_CONTEXT_BACKEND_PROTOCOL_20261030.md)
defines a separate candidate-wheel AA/BB/AB/BA correctness test. The
[local PyTorch issue proposal](PYTORCH_TENSOREXPR_ISSUE_PROPOSAL_20261030.md)
documents the completed scalar/rounding mechanism and unknown upstream intent;
it has not been submitted. No upstream contact, custom wheel installation,
model-weight change, real-document test, old 120-request campaign, merge or
release is authorized or performed by this work.

## Sealed 13-case regression

The newly frozen schedule ran 78 requests: 13 ordered cases in two fresh
processes for each of default, TE-off and eager. All pass the physical,
completion and existing functional checks; maximum normalized CER is zero.
TE-off and eager each pass all 17 repeated-input/fresh-process comparisons
exactly. Default fails the repeated English comparison from first request to
the later Unicode recovery position, while matching across corresponding
fresh-process positions. No weights/source/fixture/option identity drift occurs.

The original P0-only shape checker rejects the first six cases in every suite
process. Its rejection is preserved as `qualification-initial.json`. Independent
source review establishes single-image auto-cropping: six 640px local tiles
produce [6,101,4096], then a 1024px global view produces [1,257,4096], per CLIP
layer. Multipage inputs retain one global call per page. A separate CPU-only
contract derives the tile grid from verified official source and frozen PNG
headers; corrected analysis is separately hashed. This changes invocation/shape
bookkeeping, with no numerical, semantic or bbox tolerance change and no new
inference. The full initial rejection and corrected receipt remain reviewable.

Compared position-for-position with default repeat 0, TE-off changes 20 boxes
over the 13 ordered positions. Signed coordinate deltas are 296 zeros,
15×−1, 8×+1, 5×+2, 3×−2 and 1×−3; minimum valid IoU is
0.9253979567593252. English, mixed, math and the repeated 2/3/5-page cases
have bbox differences. Chinese, table, Unicode recovery and two-page table
are exact. All non-bbox bytes, labels, content-page/reading order and structure
remain exact. No invalid/reversed/out-of-scale box is observed. These are
default-versus-control comparisons, not a replacement golden or acceptance
based on delta≤3. Small coordinate shifts can affect crop boundaries, adjacency
and layout reconstruction; annotated geometric truth is still absent.

Instrumented request ranges are 8.18–34.91 seconds for default,
8.25–33.39 for TE-off and 8.38–33.44 for eager across the mixed case schedule.
Observed peak allocator allocation is 8.589 GiB in each option. Per-case
measurements are retained; these ranges establish no speedup, regression
budget or performance PASS. Clean performance and concurrent serving remain
unqualified.

## Independent scalar-correction experiments

Two fresh original-wheel processes execute two new prototypes on the unchanged
captured BF16 input, without loading the OCR model or weights. A first retained
attempt fails compilation because this runtime does not support scripted
`.bfloat16()`; the new identity uses explicit `.to(torch.bfloat16)`.

| Prototype | Cold differences from original eager | Warm differences | Verdict |
| --- | ---: | ---: | --- |
| FP32 scalar/opmath, final BF16 store | 789133 | 789133 | Reject: already loses intermediate BF16 semantics |
| Explicit BF16 boundaries between operations | 0 | 789133 | Reject: fusion removes the inserted boundaries |

All nonzero maximum absolute differences are 0.0078125. Both fresh processes
agree exactly. Original GPU eager equals the independent staged CPU FP32/BF16
reference. The rejected outputs equal the CPU single-FP32-expression reference.
Actual emitted warm entry points for both prototypes use
1.702000021934509f and a single final BF16 store without intermediate BF16
roundings. Graphs and dispatch agree. The scalar coefficient is preserved at
FP32 precision, but that alone does not preserve the complete eager contract.
This does not rule out a proper upstream compiler correction addressing both
factors. No rejected prototype is applied to full OCR or production.

## New candidate-wheel matched-context result

After independent review cleared the original-wheel gates, the new prospective
candidate-wheel AA/BB/AB/BA test ran in eight fresh processes, two requests
each. Every arm uses wheel
6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e
and the same per-model original eager QuickGELU scope. Its declaration seals
the backend protocol, model identity, source/input hashes and admission receipts.

All 16 OCR results equal the original cold raw/output/prompt result exactly.
Fresh-process repetition, within-backend cold/warm, matched cold backend,
matched warm backend with the same prior mode, and prior-backend interaction
comparisons all pass exact OCR parity. Functional/geometry/completion/HIP/DASH
checks pass. Actual AUTO dispatch is SAM efficient SDPA and CLIP flash SDPA;
no silent MATH fallback occurs. The frozen MATH-only helper's AUTO
`physical_contract=FAIL` receipt is preserved; a new independent evaluator
checks the same physical components and the declared actual AUTO operators.

Internal tensors are not equal between MATH and AUTO. On page 1, the first
retained block-0 pre-MLP boundary has 185409 differing BF16 elements and
maximum absolute delta 0.15625; activation input 257611 / 0.046875,
activation output 204032 / 0.02734375 and final projection 196433 / 0.0078125.
Page 2 also differs. These tensors follow upstream image/attention computation;
they are not isolated SDPA errors and no 0.03 operator tolerance is applied.
Within each backend, all captured stages are exact across cold/warm,
fresh-process and prior-backend controls. Thus attention-mode intervention
changes internal vision numerics in this context, without an observed OCR/bbox
effect on this two-page fixture. Captures do not isolate SAM versus CLIP or
identify its first changed upstream operation. The 13-case AUTO comparison
remains unrun; PR #15's historical strict FAIL is unchanged.

## Evidence, review and remaining unknowns

Sanitized [qualification receipts](te-mitigation-20261030/SHA256SUMS.json),
[executed graphs/dispatch](te-mitigation-20261030/execution.json),
[scalar prototypes](te-mitigation-20261030/scalar-prototypes.json),
[backend matrix](te-mitigation-20261030/backend-matrix.json) and
[backend numerical deltas](te-mitigation-20261030/backend-numerics.json)
retain exact producer/source/artifact hashes without raw OCR, token arrays,
private tensor values, model weights or user-directory paths. Full private
receipts, failed attempts and producer versions remain local and hash-bound.
The original 406 historical file hashes are reverified unchanged.

Independent agent review recomputed P0 and full-suite gates, checked native/raw
receipts and all scalar output bytes, and reviewed source/dispatch/privacy.
Final publication and backend evidence passed a separate review: 20 public
checksums and 1938 private artifact hashes/sizes verified, with no private
payload/path/secret findings. All six local CPU checks pass, including
333 passed / 7 skipped tests. Generated machine receipts are folded in GitHub
for review; their byte seals are preserved. This is local independent evidence review,
not an upstream response
or a claim of external hardware validation.

Proven: the TE-only control restores tested original-wheel end-to-end
cold/warm parity; the narrower original eager path also restores repeatability
on the sealed regression; coefficient correction alone is insufficient; the
new controlled two-page backend test has exact OCR parity despite internal
numeric differences. Unknown: annotated geometric accuracy, clean comparative
performance and serving concurrency, a supported compiler fix preserving both
semantics, upstream semantic intent, other candidate-wheel cases and other
hardware. These limits keep every production gate **EXPERIMENTAL**.
