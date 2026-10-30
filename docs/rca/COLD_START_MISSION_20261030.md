# Windows gfx1151 cold-start determinism and backend isolation

**KEEP EXPERIMENTAL.** No production defaults, model weights, historical
conclusions, PR bases, drivers or SDKs changed.

- [Original-wheel cold-start RCA](COLD_START_RCA_20261030.md): repeatable cold/warm
  difference; first observed onset between CLIP block 0 MLP fc1 output and fc2
  input. Disabling TorchScript optimization removes the difference in two
  diagnostic processes. Specific kernel mechanism remains unknown.
- [Candidate-wheel isolation](CANDIDATE_BACKEND_ISOLATION_20261030.md): 24 clean
  processes, 48 physical requests, AA/BB/AB/BA controls across three frozen cases.
  Matched contexts repeat exactly. Backend effects are case/context dependent;
  no preceding-backend interaction is observed at a fixed warm target.
- [Bbox correctness](BBOX_CORRECTNESS_20261030.md): canonical geometry,
  signed deltas, valid-box IoU, unchanged non-bbox bytes and order. All 104 boxes
  contain measured fixture ink extents under the disclosed inclusive geometry.
  No general bbox tolerance or semantic acceptance policy is inferred.
- [P3 restart controls](P3_RESTART_RCA_20261030.md): termination, cleanup and
  functional recovery pass; cold strict golden parity fails; matched no-kill
  and post-kill outputs agree. The new strict ledger remains PARTIAL.
- [Independent reliability proposal](WINDOWS_RELIABILITY_INDEPENDENCE_20261030.md):
  new baseline branch/PR scope, with no changes to existing PR bases.
- [Source/artifact identities](cold-start-20261030/source-artifact-identities.json),
  [invocations](cold-start-20261030/invocation-index.json),
  [matrix](cold-start-20261030/candidate-backend-matrix.json), and
  [publication seal](cold-start-20261030/SEAL.json).

Known: invocation context explains reproducible differences that cannot be
attributed to AUTO or termination alone; controlled backend effects also exist.
Unknown: the specific optimization/fusion mechanism, which vision encoder
accounts for each joint backend effect, rare-failure probability, and a general
OCR bbox acceptance policy. Registered tensor buffers and parameter version
counters are unchanged; unregistered/native runtime state is not exhaustively
inventoried. Fresh processes share existing disk compiler caches.

No production warmup workaround, tolerance relaxation, synthetic speedup claim,
real-PDF use or transfer, release, merge, or upstream contact is included.
PR #15 exact-parity FAIL and PR #21 strict FAIL/PARTIAL are preserved.

Reports are complete only when `SEAL.json` exists and all member hashes verify.
Raw outputs, token contexts, process paths, full failures, source snapshots and
actual ledgers stay in the ignored repository-local experiment workspace.
