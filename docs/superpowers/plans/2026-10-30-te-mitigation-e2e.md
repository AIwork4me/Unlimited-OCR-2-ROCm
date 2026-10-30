# Original-wheel TE mitigation qualification protocol

KEEP EXPERIMENTAL. Additive experiments on ff467134; preserve 10067ee,
PR #15/#21/#22/#25 evidence and verdicts. No production/model/weight/default
changes, custom wheel installation, real documents, upstream contact or old
120-request performance campaign. All processes and caches stay in the primary
repository. No completed kernel RCA campaign is repeated.

P0 protocol v1: original published wheel SHA256
415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c;
locked model 9cf077c4b206e0e2a31879df937b0b0bcba3661c; baseline source
82951c73b28bf735f5655e6f5c3306cd170a21c2 and existing frozen fixtures.
Two independent fresh processes each for default and TE-only disabled, two
MATH two-page requests per process. Seed0 before load/request, default graph
optimization, no inference warmup. TE flag set before load and restored in
finally; cache lifetime bounded by process. Actual QuickGELU graphs, all-layer
observations and encoder profiler operators; private first-block tensors;
HIP, DASH, raw/token/prompt hashes, completion, bbox/content/order, measured
instrumented request latency and APU allocator memory. Fresh outputs only.
Success requires exact cold/warm AND independent-process raw/output-ID parity,
functional OCR/structure/stop, zero HIP, 12 native DASH calls, actual MATH and
unfused activation dispatch. Do not replace the original ordered golden.

If P0 succeeds: a separately frozen ordered 13-case schedule with two fresh
processes each for default, TE-off and instance-scoped eager QuickGELU. First
qualify eager QuickGELU on the two-page case with two independent processes.
Use the original official Python arithmetic recovered from its verified source;
clone each CLIP MLP method's globals per instance, replacing only quick_gelu.
No global JIT flag changes for this option. Suite records graphs/profilers but
does not copy numerical tensors into the timed path. Timing is descriptive;
no speedup/release qualification or unsupported SLA threshold is inferred.

P1 option3 is independent: two new fixed-input experiments on scalar-corrected
float opmath with/without explicit BF16 boundary casts, compared to eager BF16.
Record actual graphs/code and CPU reference. Scalar-only correction is rejected
if eager arithmetic is not preserved. No custom wheel or production patch.

P3, only after controlled cold/warm OCR: prospectively freeze candidate-wheel
MATH/AUTO AA/BB/AB/BA, two fresh processes per arm, frozen two-page case and
instance-scoped eager QuickGELU if admissible. Same candidate for all arms;
reject silent MATH fallback. This is a new correctness experiment, not the old
measured campaign. Preserve PR15 FAIL regardless of new result.

Acceptance categories remain separate: exact parity; functional quality;
geometry; operator behavior; latency/memory; cold/warm/process repeatability.
No bbox tolerance is introduced. In-process shared-global TE toggling is not
admissible as a deployment design without cache/thread isolation evidence.

P4: prepare a standalone upstream issue proposal locally, including unknown
semantic intent; do not submit or contact upstream. Independent source/protocol
review before expansion, final evidence review, CPU checks and draft PR.

Protocol v2 prospectively seals the full case definitions, fixture/ground-truth
and producer/source identities before any child starts. The exploratory v1
receipts remain separate: its pre-MLP tensor call label was offset by one.
Corrected v2 labels align all four block-0 stages to the same invocation.
Process completion alone cannot authorize expansion: a separate evaluator must
verify physical contracts, all-layer graph/dispatch observations, exact parity,
functional quality and independent-process repeatability before the suite.
