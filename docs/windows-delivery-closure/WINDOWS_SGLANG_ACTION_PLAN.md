# Windows SGLang Action Plan (Phase 6)

Action-oriented handoff derived from the static feasibility assessment
(`docs/windows/WINDOWS_SGLANG_FEASIBILITY.md`, verdict
BLOCKED_WITH_ACTIONABLE_FIX; Windows execution NOT_TESTED). Not a blocker for
the Native delivery (PR #33). No SGLang support is claimed.

## 1. Exact version constraints (frozen identities)

| Artifact | Identity |
| --- | --- |
| Official customized SGLang wheel | `sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl`, 15,281,032 bytes, SHA-256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` (benchmark-lock pinned; local rehash matched) |
| Wheel metadata requires | torch==2.13.0, cuda-python>=13.0, cuda-tile, flash-attn-4, flashinfer_python[cu13], nvidia-cutlass-dsl[cu13], nvidia-mathdx |
| Preserved Windows stack (must not drift) | torch 2.12.0+rocm7.14.1 (original wheel `415d2dbd…`), torchvision 0.27.0, torchaudio 2.11.0, triton-windows 3.7.1.post27 |
| Baidu pinned client/setup | `48ff432fb68a18ed91f2f70021f29028b1dde354` |
| Model | revision `9cf077c4…` (18-file hash pin) |

Hard conflict: the official SGLang wheel's CUDA-13/NVIDIA dependency set cannot
resolve against the preserved ROCm 7.14.1 stack. A universal `py3-none-any`
tag does not certify Windows/HIP native support.

## 2. Native Windows feasibility blockers

- Dependency set unresolvable without `--no-deps` — which hides conflicts and
  proves nothing (forbidden as an "installation").
- flash-attn-4 / flashinfer[cu13] / nvidia-cutlass-dsl / nvidia-mathdx: no
  validated HIP-on-Windows equivalents; DASH custom Triton ops are unvalidated
  on Windows SGLang's code path.
- Native Windows process lifecycle, scheduler/graph mode, long-context window
  continuation and KV-tail reuse behavior: unvalidated.
- Any port would need a frozen, separately reviewable dependency/kernel
  identity — not a flag flip on the working Native installation.

## 3. Linux/WSL alternative viability

- Linux ROCm SGLang work exists upstream and is the natural first target: a
  separate Linux (or WSL2, subject to GPU passthrough validation) environment
  can host the official wheel's dependencies without touching the preserved
  Windows stack. Isolation from the working Native install is mandatory.
- A qualified Linux SGLang route remains **Linux evidence**; it can never be
  promoted to "Windows SGLang PASS" (same rule as every other platform scope
  in this repo).
- WSL viability is untested here; treat as its own environment with its own
  gates, not as Windows.

## 4. Required upstream changes (port workstream)

1. Review existing Linux ROCm SGLang work for reusable kernel/scheduler
   pieces; record what the official customized route actually patches
   (window continuation, KV-tail reuse, ngram35/window1024, budgets).
2. Produce a dependency closure that is honest about each NVIDIA-only
   package: reconcile or replace explicitly, with per-artifact hashes.
3. Resolve DASH custom Triton ops on the target platform; validate or replace.
4. Upstream engagement (SGLang/ROCm) for Windows-HIP support gaps — outside
   this repository's control and timeline.

## 5. Minimum useful future smoke test (per gates, not a claim)

On the isolated experimental environment only, with pinned identities:
1. Install resolves **without** `--no-deps`; `pip check` clean.
2. Server starts; model loads at the pinned revision (hash-verified).
3. One single-page OCR request completes with native stop semantics.
4. One 10-page windowed request demonstrates KV-tail continuation without
   crash/OOM, with memory recorded.
5. ngram35/window1024 parity spot-check against the Native path on identical
   input (bounded, authorized).

Anything less is "dependency installation", which is explicitly not SGLang
support.

## 6. Model protocol parity requirements

Even a perfect install does not equal the official protocol. Parity requires:
10-page windowing with KV-tail reuse; per-window ngram reset semantics
(ngram35/window1024); official budgets (39321 / 235929 with token-table/refit
rules); stopping/page-guard semantics; identical input ordering; scorer-input
format equality. The Native path today has none of these equivalences
(WINDOWS_OFFICIAL_PROTOCOL_GAP.md §4.3) — parity must be demonstrated, not
assumed from shared weights.

## 7. Separate validation gates

- Own environment (no shared venv with the Native stack), own identity freeze,
  own admission (fail-closed, same PR #28 lineage), own evidence directory.
- Physical Windows gfx1151 gates only after the Linux route is qualified —
  and each platform's results stay scoped to that platform.
- Explicit authorization before any GPU execution; no automatic promotion of
  any SGLang result into benchmark claims.

**Bottom line: SGLang on Windows remains a separate port + authorization
workstream. Nothing in the Native delivery depends on it.**
