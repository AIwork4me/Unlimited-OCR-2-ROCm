# Scoped QuickGELU qualification summary

KEEP EXPERIMENTAL. CPU lifecycle tests and retained physical observations have
different scopes. Neither changes production MATH or qualifies AUTO performance.
Current engineering checkout: **18 targeted tests passed**, full CPU suite
**350 passed / 8 skipped**. Lint, format, Windows/Linux mypy targets, isolated
package build and pip check pass. These counts exclude untouched uncommitted tests
from the original 356/7 overnight run and an optional local source characterization.

| Dimension | Classification | Exact scope |
| --- | --- | --- |
| Source admission/rollback/isolation | QUALIFIED | CPU synthetic lifecycle contracts; no torch/model download |
| v2 mixed synthetic repeatability | QUALIFIED | Ten completed requests across six distinct fresh process identities, exact retained raw/token/prompt equality |
| Fault/restart | QUALIFIED | Two active-generation interruptions, individually followed by a fresh two-request process; no completion attributed to interrupted work |
| Separate-process overlap | QUALIFIED bounded process lifetimes | Two separately loaded one-request processes overlap; no claim of simultaneous GPU kernels |
| Same-process real-model instances/parallel serving | UNSUPPORTED by reviewed loader | No removal of shared DASH/module guard |
| Same-process physical concurrent inference | NOT_TESTED | CPU instance isolation does not establish it |
| Historical strict SAM-only AUTO parity | FAILED | Different math output/bbox; performance remains blocked |
| Geometry ground truth / clean performance | NOT_TESTED | Stable boxes and diagnostic timings are insufficient |
| Other hardware / Linux | NOT_TESTED | These receipts cover physical Windows gfx1151 only |

Executed v2 adapter SHA256:
`1577fb53da4c9d4187bdee0138457ec1c49c2e77064f5c6b2e8431925141f153`.
Candidate torch wheel SHA256:
`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`.
Model revision: `9cf077c4b206e0e2a31879df937b0b0bcba3661c`.
The three package versions are pinned in the companion engineering review.

Retained declaration `ocr-predeclared-v2.json` and private producer/Job receipts
bind those identities. The private 648-file overnight seal SHA256 is
`cae252e5740b50f37d3ab43b479c83657b72d7312b64f82aa9ec2a15ef9db855`;
the public original scoped summary SHA256 is
`7c4765a27ab8c39a9cc1d971dccad436195585d579490c1e703bb5fe55b3cdf2`.
The exact declarations, raw OCR, token arrays, tensors and environment identities
are private; they are not executable CPU fixtures or an included GPU harness.
These identifiers permit evidence-owner audit, not public re-creation of private
receipts. Reviewers can reproduce all CPU contracts without that private evidence.

This mission rechecks ten completion receipts: stop, HIP, 12 synchronized native
BF16 GPU DASH calls per request, actual SAM/CLIP MATH dispatch, unchanged
parameters/buffers, restored bindings/global flags, and each owned Job active0.
Interrupted attempts have no completion/summary and end with Job active0.
The recorded producer reports 197 overlap samples/minimum host available
69,980,602,368 bytes; individual sample traces were not retained, so these two
aggregates cannot be independently reconstructed. Process lifetimes corroborate
bounded process overlap. APU allocator use is UMA, not dedicated VRAM or total
system memory use.

V1's ten completions and two interruptions remain historical and do not erase
its rollback failures. Only v2 source supports this summary. No new GPU request
is launched for engineering closure. Incomplete original upfront audit coverage
and two unadmitted later mixed-backend completions remain disclosed procedural
failures; this is not retroactive full mission-protocol qualification.
