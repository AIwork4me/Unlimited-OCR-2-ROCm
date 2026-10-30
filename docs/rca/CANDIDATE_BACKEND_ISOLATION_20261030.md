# Candidate-wheel backend isolation matrix

Decision: **KEEP EXPERIMENTAL**. These new controls do not revise PR #15.

All 24 fresh processes / 48 requests completed using the exact retained
candidate wheel SHA256
`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`.
Each case runs AA, BB, AB and BA twice in independent clean processes, with
seed 0 before model loading and each request. A means MATH, B means AUTO.
Normal TorchScript optimization is enabled; the original-wheel optimization
intervention is not used here. Each process loads the model once and makes
exactly two requests. Request-local official KV caches are fresh.

The fixtures are frozen Chinese (previous exact PASS), two pages (previous
delta 1), and mixed text (previous delta 3). Inputs, preprocessing, prompts,
generation, model/source and wheel identity are held within each case.
Both encoders have actual profiler dispatch evidence: MATH uses MATH;
AUTO uses SAM EFFICIENT and CLIP FLASH, rejecting silent MATH fallback.
The 12-layer native DASH/SR-SWA path remains enabled.

The following results agree in both independent repeats. PASS/FAIL means exact
raw bytes plus output IDs and prompt IDs, with no bbox tolerance.

| Case | AA cold/warm | BB cold/warm | Cold MATH/AUTO | Warm MATH/AUTO, prior MATH | Warm MATH/AUTO, prior AUTO | Prior-backend interaction at fixed warm target |
| --- | --- | --- | --- | --- | --- | --- |
| Chinese | PASS | PASS | PASS | PASS | PASS | None observed |
| Two pages | FAIL | PASS | PASS | FAIL | FAIL | None observed |
| Mixed | FAIL | FAIL | FAIL | FAIL | FAIL | None observed |

All matched-context repetitions are exact for both cold and warm positions in
every arm. Thus within-backend repeatability at a fixed invocation context
must be distinguished from cold/warm equality. The two-page cold/warm MATH
difference also exists with the original wheel, and is localized separately.

For two pages, cold requests all produce coordinate 188 at the affected edge;
warm MATH targets produce 189 after either prior backend, while warm AUTO
targets produce 188 after either prior backend. Holding the prior backend fixed
establishes a joint SAM/CLIP backend effect on this warm target output in these
trials. It does not establish that AUTO caused the separate MATH-only cold/warm
difference. Cold MATH/AUTO equality alone would also have missed the warm effect.

For mixed text, matched cold and matched warm MATH/AUTO outputs differ.
Changing only the preceding backend does not change the warm output of a fixed
target backend in either repeat. This supports a target-backend effect under
the tested contexts, alongside a cold/warm effect within each backend.

These arms switch SAM and CLIP together; they do not isolate which encoder or
specific fused kernel accounts for a target-backend discrepancy. Two processes
per arm establish these observations, not universal determinism or an estimate
of rare-failure probability. Disk compiler caches remain shared; process-local
model/module/executor state starts fresh. No timings are accepted as performance
measurements, and no synthetic speedup claim is made.

The new machine-readable matrix retains signed coordinate distributions,
per-box labels/IoU, first differing token indices, exact non-bbox/order checks
and hashes for each comparison. Full raw outputs and token contexts stay local.
The publication gate independently checks actual backend operators, native DASH,
HIP, source/native identity, process completion and cleanup. The inherited
observer's MATH-only physical-contract label is not used to classify AUTO.
