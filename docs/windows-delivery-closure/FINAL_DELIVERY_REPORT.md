**Windows Radeon Native Product Delivery — KEEP EXPERIMENTAL**

Mission: Windows Radeon Native product delivery, PR #32 integration audit and
official benchmark readiness. Date: 2026-11-01. This report begins with the
mandated title line; every claim below is bound to evidence in this ledger.

## A. GitHub Draft PR

- **URL**: https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/33
- **PR number**: #33 (Draft, OPEN, not Ready-for-Review, not merged)
- **Base**: `main` @ `52d593e0686176b3842d78fbac3ef76956ecf1ae`
- **Head**: `delivery/windows-native-benchmark` @ `a987dca00cf5ecf01f456c90462e188261f219d0`
- **Changed-file inventory**: 36 files (8 under `src/`+`scripts/` code: runner.py, windows_job.py, publication.py, run_windows_benchmark.ps1, setup/reproduce ps1, windows_install_preflight.py, run_mpdoc_sample.py; 14 docs; 5 eval receipts; tests; lock; MANIFEST)
- **Commits** (8): `4abba4f` Job Objects · `c431027` wheel-verified offline install · `3556e38` PowerShell entry + frozen annotations · `5436f80` HF cache containment · `d2caa79` validation docs/evidence · `0ba45dd` single-page user command · `c432861` PR#32-path-detector whitelist fix · `a987dca` product-status matrix + README section
- **Hosted CI**: 8/8 PASS on the head (runs 41281751088, 41281782148) — cpu windows-latest ×2, cpu ubuntu-latest ×2, scorer windows(qwen) ×2, scorer ubuntu(marathon) ×2. Zero fix attempts needed. Hosted scorer checks are a 7-metric selftest, not a full Windows scorer runtime.
- **Independent review**: FINAL_REVIEW_APPROVED (0 P0/P1, 3 P2 notes — one applied, two recorded).
- **Known limitations**: single machine (gfx1151), two environments; observed hashes are not publisher signatures; wheelhouse/model bytes not shipped (hash manifest only); fresh-host acquisition documented but not independently tested; authoritative report is Chinese (`WINDOWS_PRODUCT_DELIVERY_REPORT.md` in the PR tree).
- **Outstanding comments**: none yet (Draft, fresh).

## B. Compatibility with PR #32

- **PR #32 SHA inspected**: `a720c49c2e9f755633fd7efa95ab523fe072e871` (moved from `a01db7de` mid-mission; change recorded and analysis repeated; still OPEN/unmerged).
- **Source-level overlap**: exactly two files (`publication.py`, `test_publication_gates.py`); all other 67+32 changes disjoint.
- **Semantic compatibility**: COMPATIBLE_WITH_CONDITIONS — both merge orders converge to one tree (`7eb5c1cd…`); combined CPU suite 460 passed / 0 failed; lint/format/dual-mypy green. The condition (whitelist commit `c432861`) is included in PR #33.
- **Publication gate preservation**: merged detector is a strict superset of base (18 vs 12 functions); PR #32's fail-closed JSON/JSONL/duplicate-key/Windows-path detection and the Windows selected-root scan fix coexist; the pre-fix combined tree failed 3 gates — found, fixed Windows-side, no #32 code touched, nothing deleted.
- **L1 policy isolation**: L1/L1.1-B/P2B/P2C remain digest-pinned opt-in experiments (Linux evidence only); merging #32 authorizes nothing new on Windows; production defaults unchanged.
- **Remaining integration blockers**: none technical; procedural only — merge order is a maintainer choice (both orders verified).

## C. Windows Native Readiness

- **Installation evidence**: two fresh fully-offline environments (`.b/w1`, `.b/w2`); 13,343 installed wheel members hash-verified per environment; `pip check` clean; 43-artifact offline wheelhouse recipe (9 AMD + 34 transitive) with public hash manifest.
- **26/26 Golden provenance**: bounded synthetic suite at physical commit `3556e38` (device-bound: Radeon 8060S gfx1151, ROCm 7.14.1); 13/13 cross-environment exact token/bbox parity; 6/6 cold-warm pairs; native `stop` on all; CER 0. The Python package is byte-identical from that commit through the PR head apart from one publication whitelist entry (verified: `git diff 3556e38..a987dca -- src/` is 4 lines, non-inference).
- **Supported model/framework/wheel**: model revision `9cf077c4…` (18-file pin); torch 2.12.0+rocm7.14.1 original AMD wheel `415d2dbd…`; torchvision 0.27.0+rocm7.14.1; torchaudio 2.11.0+rocm7.14.1; triton-windows 3.7.1.post27; Transformers 4.52.1; native BF16, scoped MATH vision SDPA, native DASH/Triton/SR-SWA.
- **Existing CLI instructions**: offline install preflight → install → `pip check` → bounded single-page native command (`unlimited_ocr_2_rocm single … --max-new-tokens 1024`) → benchmark entry with deliberate `-ExecuteGpu` opt-in (`docs/windows/WINDOWS_INSTALL_REPRODUCTION.md`, `PRODUCT_STATUS.md`).
- **Windows process reliability**: Job Object supervision (suspended create → assign → resume, KILL_ON_JOB_CLOSE, descendant drain, explicit failure states, never-PASS on unreadable/nonzero); documented residual gap: hard supervisor death between creation and assignment.
- **Unqualified areas**: full-corpus GPU inference (NOT_RUN/NOT_AUTHORIZED), all official metrics (null), native Windows scorer (BLOCKED), native Windows SGLang (BLOCKED/NOT_TESTED), real-document GPU runs (BLOCKED pending re-reviewed policy + prerequisites + authorization), fresh-host acquisition (untested), 15-page/39321 shapes (Linux experiment evidence only).

## D. Official Benchmark Readiness

| Dataset | Docs/pages verified | Docs/pages inferred | Scorer status | Eight metrics | Overall | Official ref | Delta | Reason not measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MPDocBench | 420 / 3135 (PASS) | 0 / 0 | NOT_RUN (native Windows scorer BLOCKED; bridge designed) | all null | null | 84.59 (paper) | null | No GPU campaign authorization; production 5-page/8192 limits; native≠official protocol; scorer blockers |
| MarathonDocBench | 100 / 18272 (PASS) | 0 / 0 | NOT_RUN (same; Marathon scoring refused on Windows/SIGALRM) | all null | null | 77.06 (paper) | null | Same, plus 235929 budget class has no policy |

Nulls are never zeros. Synthetic Golden CER 0 is not a benchmark score.

## E. Next three decisions

1. **Whether to approve the proposed Windows real-document GPU pilot**: policy
   revision 2 exists with an independent REVISION_REQUIRED review fully
   dispositioned; decision requires (a) re-review of rev 2, (b) prerequisites
   P1/P2/P5 (merge #32; vendor PR #28 admission; wire telemetry/watchdog),
   (c) P6 fresh golden rerun at the execution commit, then (d) the explicit
   authorization (`WINDOWS_GPU_AUTHORIZATION_REQUEST.md`). Until then:
   BLOCKED.
2. **Which official scorer deployment route to standardize**: recommended —
   Option B bridge (Windows GPU inference + unchanged pinned Linux scorer
   with hash-bound prediction transfer; `WINDOWS_LINUX_SCORER_BRIDGE_DESIGN.md`).
   Option A (native Windows scorer) is BLOCKED and not recommended. Decision
   needed before any bounded scored set, and transfer remains unauthorized
   until then.
3. **When to align the Windows PR with merged PR #32, if applicable**: PR #33
   targets `main` now and is independently mergeable; both merge orders are
   verified conflict-free and convergent. If #32 merges first, GitHub will
   re-report mergeability — no rebase is required; re-run the combined-tree
   gate suite once more on the actual merge commit as routine hygiene.

**No production release is requested.** Full product acceptance is
NOT_ACHIEVED by design: the benchmark evidence does not exist yet, and the
project stays **KEEP EXPERIMENTAL**.
