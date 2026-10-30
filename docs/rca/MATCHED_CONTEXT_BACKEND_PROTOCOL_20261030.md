# New matched-context attention correctness protocol

KEEP EXPERIMENTAL. This is a new prospective experiment, independent of the
strict MATH/AUTO failure in PR #15. It does not authorize the old 120-request
measured performance campaign or change any production backend default.

Admission requires reviewed original-wheel two-page exact cold/warm and
fresh-process parity under the instance-scoped original eager QuickGELU path,
plus successful sealed 13-case functional/physical checks. If those controls
fail, do not execute this comparison; retain this proposal with its unmet gate.

Use the existing source-built candidate wheel, SHA256
6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e,
for every arm. No installation or wheel substitution. Use the same immutable
model revision 9cf077c4b206e0e2a31879df937b0b0bcba3661c, original weights,
reviewed Windows integration source, frozen two-page images, preprocessing,
prompt, generation options and exact BF16 dtype/layout. The per-model eager
QuickGELU scope is held fixed in every arm; TE enablement and graph optimization
remain at their original settings. This scopes out the identified activation
plan transition while preserving attention as the declared independent variable.

Predeclare AA=MATH→MATH, BB=AUTO→AUTO, AB=MATH→AUTO, BA=AUTO→MATH, each in two
independent fresh owned processes: 8 processes, 16 requests total. No inference
warmup, cache deletion or additional measured requests. Seed 0 before model load
and before each request; initialize the model once per process. Shared on-disk
compilation caches are disclosed; model, module, KV and invocation state start
fresh in every process. Retain process/parent creation identities and descendant
cleanup accounting. Freeze exact fixture, ground-truth, source, wheel, model,
producer and protocol hashes in a new predeclared receipt before any child starts.

Require all 24 CLIP layers to use original unfused QuickGELU arithmetic with
actual separate sigmoid/multiply dispatch. Require actual SAM and CLIP MATH
operators in MATH, efficient SAM and flash CLIP operators in AUTO, with no
MATH fallback. Validate DASH, HIP, guard arguments, runtime restoration,
parameter versions and buffer hashes. Record private block-0 MLP/activation
input and output tensors with exact byte/layout identities, actual graphs and
CPU dispatch events. This wheel lacks GPU Kineto timeline activity; do not
claim native GPU symbol timing from CPU profiler events.

Separate comparisons: within-backend repeats; matched first requests in clean
processes; cold-versus-warm effects; MATH-versus-AUTO effects; AB/BA order
interactions. Compare every result against its matching backend/ordinal peers,
then cross backend only after relevant within-backend controls pass. Report
exact raw/output/prompt ID hashes, first token difference, canonical bbox signed
deltas/IoU/validity, labels/page/reading order, exact non-bbox bytes, functional
quality, PAGE/BLOCK counts and stop metadata. No coordinate tolerance is added.

Latency and allocator memory are descriptive under instrumentation. APU
allocation is shared system memory, not dedicated VRAM. A new exact match
does not qualify semantic quality or performance automatically and does not
alter historical verdicts. Any unexplained within-backend or request-order
difference blocks a backend-causal conclusion. Preserve complete failures.
