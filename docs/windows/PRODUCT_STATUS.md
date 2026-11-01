# Windows Native Product Status — KEEP EXPERIMENTAL

Single-page status of the Windows Radeon Native OCR delivery. This matrix is the
authoritative summary; detailed evidence lives in the linked documents and
`eval/windows/*.json` receipts. No green badge or platform-wide ALL_READY claim
is made anywhere.

## Status matrix

| Item | Windows status |
| --- | --- |
| Original-wheel installation | PASS, two fresh environments |
| Native Transformers BF16 MATH | PASS, bounded scope |
| Golden synthetic OCR (delivered tree `3556e38`) | 26/26 PASS |
| Golden synthetic OCR (frozen execution tree `6d8e481`) | 26/26 PASS (fresh rerun, exact cross-environment parity; allocator/available envelope matches: 10.510/8.589 GiB peaks) |
| Integration of main+PR#33 combined tree | PASS — actual two-parent merge, conflict-free, convergent tree `7eb5c1cd`; combined CPU suite 521 passed / 0 failed; publication gate PASS |
| Fail-closed experiment admission (PR #28 contract) | VENDORED on the execution tree, byte-identical; real-gate dry-run 18 cases / 14 verified refusals |
| 1 Hz resource telemetry + watchdog in the supervised runner | IMPLEMENTED (staleness rule, memory floors, breach classification), CPU suite green |
| Linux scorer bridge (Option B) | IMPLEMENTED, synthetic-fixture tests incl. the real eight-metric positive control; no GT/prediction transfer executed |
| MPDocBench dataset verification | 420 documents / 3135 pages PASS |
| Marathon dataset verification | 100 documents / 18272 pages PASS |
| MPDocBench full GPU inference | NOT_RUN |
| Marathon full GPU inference | NOT_RUN |
| Eight official metrics | null (Windows route BLOCKED; Linux bridge ready but no campaign run) |
| Overall scores | null |
| Difference versus Baidu official | null |
| Native Windows SGLang | BLOCKED / NOT_TESTED |
| Real-document GPU pilot | PREPARED (reconciled policy, prerequisites P1/P2/P5/P6 closed, exact packet returned); NOT AUTHORIZED |
| Production release | KEEP EXPERIMENTAL |

Reference scores from the official paper (not measured here): MPDocBench
Overall 84.59, Marathon Overall 77.06. All local metric fields remain null
until a complete, correctly scored benchmark exists; they are never zero.

## New-user questions

1. **Which Windows version and AMD GPU were validated?** Windows 11 (build
   26200), AMD Ryzen AI MAX+ PRO 395 with Radeon 8060S (gfx1151), driver
   32.0.22032.6002, ROCm 7.14.1 / HIP 7.14.60850
   ([baseline](NATIVE_BASELINE_VALIDATION.md),
   [device ledger](../../eval/windows/golden-validation.json)).
2. **Which packages and original wheel are required?** torch
   `2.12.0+rocm7.14.1` from the original AMD wheel (SHA-256
   `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`),
   torchvision `0.27.0+rocm7.14.1`, torchaudio `2.11.0+rocm7.14.1`,
   triton-windows `3.7.1.post27`, Transformers `4.52.1`. A full offline
   wheelhouse is 43 artifacts — the nine AMD archives alone are not enough
   ([install guide](WINDOWS_INSTALL_REPRODUCTION.md),
   [manifest](../../eval/windows/wheelhouse-SHA256SUMS.json)).
3. **How do I install from a fresh environment?** Preflight, then offline
   install from the wheelhouse with process-local caches, then `pip check`:
   [install guide](WINDOWS_INSTALL_REPRODUCTION.md). The installer refuses
   existing environments and verifies all nine AMD archives before pip runs.
4. **How do I verify the GPU is actually being used?** Run the documented
   bounded single-page native command (install guide §user command); a native
   `stop` finish reason with DASH dispatch metadata is the user-level GPU
   proof. The sealed 26/26 golden receipt records gfx1151 identity, native
   stop metadata and DASH dispatch counts for every request
   ([receipt](../../eval/windows/golden-validation.json)).
5. **How do I run a supported Native OCR command?** The scoped single-page
   command or the PowerShell benchmark entry; production limits are five
   images/pages, 8192 new tokens and the default n-gram behavior
   ([benchmark protocol](WINDOWS_BENCHMARK_PROTOCOL.md)).
6. **What happens if a GPU process hangs or crashes?** Workers start
   suspended, are assigned to a kill-on-close Windows Job before resuming,
   and owned descendants are drained on every exit path. Timeout, crash,
   interruption, nonzero exit and unreadable output all become explicit
   failure states — never PASS. Known gap: hard supervisor death between
   process creation and Job assignment is outside the guarantee
   ([protocol](WINDOWS_BENCHMARK_PROTOCOL.md)).
7. **Which official datasets are fully available?** Both public datasets are
   downloaded and hash-verified: MPDocBench 420 documents / 3135 pages;
   MarathonDocBench 100 documents / 18272 pages
   ([dataset readiness](DATASET_READINESS.md)).
8. **Which results have actually been measured?** Offline install
   reproducibility (two fresh environments, 13,343 installed members
   verified each), and 26/26 bounded synthetic golden OCR requests with
   exact cross-environment token/bbox parity and CER 0 — executed twice:
   at the delivered tree `3556e38` and, after main merged PR #32 and the
   execution tree gained the vendored admission contract, P5 telemetry and
   the scorer bridge, again as a fresh 26/26 rerun at the frozen execution
   commit `6d8e481` (same envelope: 10.510 GiB peak reserved / 8.589 GiB
   peak allocated). Nothing else.
9. **What remains experimental or blocked?** Full-corpus GPU inference
   (NOT_RUN / not authorized), the eight official metrics and Overall
   (null — the Linux bridge is implemented but no campaign ran; a native
   Windows official scorer stays BLOCKED: XeLaTeX, Ghostscript, latexmlc,
   POSIX assumptions), native Windows SGLang (BLOCKED / NOT_TESTED), and
   the real-document GPU pilot, whose exact packet is prepared and awaiting
   explicit owner authorization
   ([readiness](WINDOWS_NATIVE_READINESS.md),
   [SGLang](WINDOWS_SGLANG_FEASIBILITY.md),
   [admission](REAL_DOCUMENT_ADMISSION.md)).
10. **How can I reproduce the validated Golden suite?** Follow the install
    guide, then the benchmark protocol; GPU execution is a deliberate
    opt-in (`-ExecuteGpu`) and is not campaign authorization. The sealed
    result is [golden-validation.json](../../eval/windows/golden-validation.json);
    synthetic golden CER 0 is not a benchmark score.

## Evidence chain

`WINDOWS_PRODUCT_DELIVERY_REPORT.md` (Chinese, authoritative) ·
`eval/windows/benchmark-status.json` (all campaign fields null) ·
`eval/windows/dataset-readiness.json` ·
`eval/windows/wheelhouse-SHA256SUMS.json` ·
[handoff](WINDOWS_HANDOFF.md)
