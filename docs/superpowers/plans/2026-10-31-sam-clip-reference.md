# SAM/CLIP numerical reference implementation plan

> **For agentic workers:** Use executing-plans inline. Preserve all historical evidence.

**Goal:** Audit independent attention references, matched-prefix logits and cache contracts without changing production computation.

**Architecture:** NumPy FP32/FP64 attention equations on retained GPU inputs; analytic CPU tests and independent reviewer cross-check. Only after retained-input analysis, four fresh diagnostic model workers collect missing matched CLIP, logits and KV points. Sanitized scripts/reports form a draft PR; private arrays stay ignored.

**Tech stack:** Existing candidate torch, NumPy, immutable model snapshot, supported Transformers DynamicCache iteration, owned Windows Jobs.

**Spec:** User mission “Independent SAM/CLIP Numerical Reference RCA”, 2026-10-31. Private attachment retained outside Git; this plan records its implementation scope.

## Constraints and review focus

- Preserve versions, hardware identity, eager QuickGELU, production/default/kernel code, fixtures and all old seals.
- No empirical acceptance tolerance, automatic merge, upstream issue, performance campaign or real-document use.
- Distinguish mathematical reference from backend emulator and FP64 projection from higher-precision decoder execution.
- Fail closed on QKV/mask/scale mismatch; distinguish original unmeasured offsets from measured new offsets.
- No raw tokens/OCR/tensors/private paths in Git. Publish only reviewed summaries.

## Tasks

- [x] Audit isolation261, priorRCA379, mixed249, historical406 and declared model/wheel/source/input identities; save new local receipts.
- [x] Add `experiments/numerical_reference/reference.py` and local CPU tests. Watch zero-query mean, additive/boolean masks, scale/causal, fully masked rows, invalid dropout and strided-input tests fail then pass. Cross-check FP64 against scalar summation and independent reviewer.
- [x] Compute FP32/FP64 references from old captured inputs first, retaining equations, hashes, distributions and precision differences. Label unmatched old CLIP inputs correctly.
- [x] Seal four new MM/AA/AM/MA one-request owned workers only after review. Diagnostic teacher forcing feeds the frozen MM prefix for steps0–9 after recording actual unchanged logits/argmax; original model source remains untouched. Capture steps7/10 and supported KV iteration, first CLIP residual/norm/MLP, and exact SDPA inputs/outputs including offsets.
- [x] Analyze matched first CLIP and SAM inputs separately. Audit fixed-hidden-state lm_head FP32/FP64 projection against captured BF16 logits. Preserve token values privately and publish aggregate/hashes only.
- [x] Run independent read-only review and six CPU checks. Rehash historical artifacts, inspect cleanup and sanitize exports. Classify each subsystem without inventing a numeric contract.
- [ ] Create a narrowly scoped draft PR from a new branch with only new experimental code, tests and sanitized docs. Preserve unrelated dirty changes; do not merge or submit upstream.

## Ledger

- Existing original GPU storage offsets were not captured; never infer zero from the CPU copies.
- Existing CLIP MATH/flash captures have different QKV. New matched capture is needed; old own-input errors alone do not isolate a backend.
- Four diagnostic requests are justified by missing full vocabulary step10, common prefixes, firstCLIP pairing and cache contracts. No new performance or broad regression run is admitted.

- Execution ruling: two failed pre-inference instrumentation attempts are preserved under separate identities; neither completed an OCR request. The admitted v3 run completes four diagnostics. Both SAM and CLIP retain INSUFFICIENT EVIDENCE for arithmetic-defect determination; no empirical tolerance was introduced.

- Review: independent report/privacy review passed; 424 capture seals, matched inputs, FP64 references, fixed-hidden projection and 192 cache checks independently verified. CPU baseline 340/7 and nine experimental tests pass. Draft PR delivery is the remaining task.
