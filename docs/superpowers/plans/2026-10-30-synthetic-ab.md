# Synthetic candidate-wheel A/B infrastructure

Implement the user's explicitly authorized synthetic-only preparation in Python
standard-library experiment modules. No production module or default changes.
Real-document evaluation is BLOCKED_DATA. Physical qualification remains gated
on P0; preparing and CPU-testing the infrastructure does not waive that gate.

Use the ten original synthetic fixture types (five languages/content cases,
Unicode recovery, 2/3/5 pages, cross-page table). No new images or ground truth
are needed. Freeze input/model/config/helper hashes and candidate artifact
6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e.
Both modes use that exact artifact and one model instance per child process.

1. Test and implement protocol.py: deterministic AB/BA counterbalance, separate
   diagnostics/warmups/measurements, six measured pairs per case; validate exact
   identity, complete denominator, strict synthetic raw/prompt/output-ID parity,
   completion/structure, finite timing, memory, no missing trials or fallback.
   Failures block aggregate speedup. Summaries expose whitelisted metrics and
   hashes only. CPU fixtures contain hand-built synthetic receipts, no torch.
2. Implement worker.py with opt-in GPU execution: verify wheel/install/loaded
   native identities, model lock and candidate dependencies; load integrated
   MATH model then apply isolated experiment-only backend wrappers. Reuse exact
   reviewed PR #5 diagnostic code by immutable Git blob/hash, locally exported
   at preparation. Reuse original NativeObserver and input acceptance. Trace
   diagnostic calls separately; require observed SAM EFFICIENT / CLIP FLASH
   for AUTO, observed MATH for controls, finite numeric checks and zero HIP.
   Record actual raw text/token IDs privately. Synchronize vision and inference;
   retain load time separately; record peak GPU allocation/reservation and RSS.
3. Implement run.py preparation, bounded child supervision, and report/export.
   Preparation performs no torch import/model inference. Require an explicit
   P0 PASS receipt and --execute-gpu before spawning a GPU child. Use fresh
   repository-local outputs; publish atomic immutable trial receipts; preserve
   RUNNING/FAIL/TIMEOUT state on interruptions and wait for owned process tree.
4. Run RED→GREEN protocol tests, complete integrated CPU checks, import checks,
   prepare a real local plan without GPU execution, and independent review.
   Documentation states unsupported conditions and separates operator dispatch
   diagnostics from lower-level PAL launch evidence and benchmark measurements.
   Publish source/tests/docs and sanitized preparation summary only; no payloads,
   raw OCR, GT, image/PDF, model or wheel artifacts.

Statistical choices are implementation decisions: six repeats balance AB/BA;
one warmup per mode/case; report median/min/max/sample stdev and paired ratios.
No confidence/significance or real-document generalization claims. Exact
synthetic output parity is declared before running; failures will not trigger
automatic tolerance relaxation. Existing original-wheel golden evidence is
never used as the candidate MATH arm.
