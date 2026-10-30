# Windows gfx1151 cold-start RCA experiment plan

**Goal:** Localize the first observable divergence without changing production behavior or historical acceptance.

**Architecture:** New additive diagnostic harness in the primary checkout; read-only frozen integration source and reviewed observer helpers. Every arm owns a fresh child process, repository-local temporary directory, offline model cache, immutable output directory, and exact producer/source/input identities.

**Spec:** User mission, 2026-10-30. KEEP EXPERIMENTAL. No production edits, weight changes, tolerance relaxation, real-document access, merges, releases or existing PR changes.

- [x] A: Original wheel: A1/A2 two MATH requests, A3 first request only, A4 explicit CUDA initialization/synchronization (no model inference) then first MATH. Seed 0 before load and each request; existing determinism flags recorded, not silently strengthened. No cache deletion. Independent stage-observed repetitions identify preprocessing, vision, decoder-layer and logits divergence; observation is a separate condition.
- [x] B: Exact retained candidate, representative PASS/delta-1/delta-3 fixtures selected from frozen PR15 diagnostics. AA/BB/AB/BA in separate clean processes, twice per case/context; same seed/generation and cache policy. Actual per-encoder dispatch required; AUTO MATH fallback rejects the arm. Separate repeatability, cold/warm, backend and order comparisons; no causal attribution from unmatched contexts.
- [x] C: Canonical original bbox parser, signed distribution, valid-box IoU, labels/content/order equality, clipping/reversal/invalid boxes. Scale verified from pinned model renderer; no acceptance threshold inferred from operator tolerance.
- [x] D: New matched no-kill/post-kill paths only after A; external kill, cleanup, functional recovery, determinism, golden parity and ledger separately classified. Existing FAILED/PARTIAL receipts immutable.
- [x] E: Separate reports and hashes with explicit known/unknown conclusions. Propose independent reliability branch on integration 82951c73b28bf735f5655e6f5c3306cd170a21c2; do not rebase or force-push existing PRs.

Review focus: wrong same-version wheel; mutable source/import path; observation synchronization; differing prefix invalidating later-logit comparisons; malformed bbox syntax silently skipped. Preserve failures and distinguish unrun checks.

Execution is authorized by the mission. Work proceeds inline from the primary root, preserving existing local edits. Raw data stays under ignored `.b/cold-start-rca/`; tracked reports contain sanitized identities and metrics only.

## Adaptive diagnostic predeclaration (before JIT control execution)

Two independent original-wheel deep traces identify the first differing boundary
between CLIP block 0 MLP fc1 output and fc2 input. The intervening pinned source
is `@torch.jit.script quick_gelu(x): x * torch.sigmoid(1.702 * x)`.
Hypothesis: process-local TorchScript optimization state changes BF16 evaluation
across first and subsequent invocations. Test two new clean original-wheel
MATH/MATH processes under scoped `torch.jit.optimized_execution(False)` with
the same deep observers. This is a diagnostic intervention only. Compare
upstream inputs, that boundary, raw IDs and matched-prefix logit margins;
do not infer a specific fused kernel from the broad intervention alone.
No production warmup or runtime default is introduced. B uses normal JIT.
