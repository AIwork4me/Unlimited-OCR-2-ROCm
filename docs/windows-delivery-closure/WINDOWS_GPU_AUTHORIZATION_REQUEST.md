# Windows GPU Authorization Request — Bounded Real-Document Pilot (Phase 5)

**Status: NOT AUTHORIZED. This document requests a decision; it grants nothing.**

## What is requested

One bounded Windows Native real-document qualification run on the physical
host (Ryzen AI MAX+ PRO 395 / Radeon 8060S gfx1151):

- Document: `handwritten_chn_1ae487be` (5 pages, identities frozen in
  `WINDOWS_REAL_DOCUMENT_SELECTION.json`).
- Two arms, sequential: 8192 tokens ngram 0; then 8192 tokens ngram 35 /
  window 1024 (pinned policy). No 16384 arm. No 39321 arm. No other documents.
- Full contract: `WINDOWS_REAL_DOCUMENT_POLICY.md` revision 2.

## What must be true before execution (prerequisites P1–P6 of the policy)

1. PR #32 (or successor) merged so the ngram arm exists on the execution tree.
2. PR #28 admission contract vendored/integrated by a reviewed commit; the
   dry-run re-executed against the real gate (TOCTOU, concurrency,
   ordered-paths cases included).
3. This authorization, granted explicitly after re-review of revision 2
   (independent review of revision 1 returned REVISION_REQUIRED; all required
   changes applied; re-review pending).
4. Admission receipt minted (scope digest over policy + selection + contract).
5. 1 Hz telemetry sampler + PR #32 ResourceWatchdog wired into the supervised
   Windows runner by a reviewed commit.
6. Fresh synthetic golden rerun at the frozen execution commit on Windows GPU.

## Scope guarantees if authorized

- At most two OCR inference requests for one pre-declared document; everything
  else unchanged (production defaults, no dataset campaign, no scoring claim).
- Predicted outcome (predeclared): both arms TRUNCATED at the 8192 cap.
- All artifacts private; only sanitized identity receipts publishable.
- Failure preserves partial outputs and accounting; no result is upgraded.

## Explicitly NOT requested

Full MPDocBench/Marathon inference, 15-page shapes, 16384/39321-token arms,
any official metric or Overall claim, SGLang execution, or publication of any
document content.

## Decision requested from

The repository owner, after: (a) re-review of policy revision 2, (b) P1/P2/P5
code prerequisites merged, (c) P6 golden rerun receipt sealed. Until all three
plus this authorization exist, the GPU path stays BLOCKED.
