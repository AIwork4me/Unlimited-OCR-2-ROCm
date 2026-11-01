# Windows benchmark protocol and comparability

Reuse [the frozen common protocol](../BENCHMARK_PROTOCOL.md), `eval/benchmarks/benchmark-lock.json`, common manifest/ledger/runner and `benchmark.scorer.METRICS`. This document adds Windows execution/accounting boundaries; it introduces no scorer or alternative metric formula.

Model revision: `9cf077c4b206e0e2a31879df937b0b0bcba3661c`. Official client/setup revision: [Baidu 48ff432f](https://github.com/baidu/Unlimited-OCR/tree/48ff432fb68a18ed91f2f70021f29028b1dde354). Official Qwen scorer revision: `4e5aee32280febb7b0844237ba087f837e3c8905`; Marathon patches are individually frozen by the common lock. A local run must bind exact source commit, wheels, model/code/tokenizer/weights, ordered inputs, policy and environment identity before execution.

| Property | Current production Native | Official customized SGLang client |
| --- | --- | --- |
| Request scope | One ordered multi-image document; maximum five images | Document windows, default ten pages |
| Generation budget | Maximum 8192 new tokens; no implicit override | Default 39321 MPDocBench / 235929 Marathon; token-table and refit rules in common protocol |
| Cache/window | Native growing caches | KV-tail reuse across official windows |
| Repetition policy | Native default ngram 0 unchanged | Custom ngram 35; multi-page window 1024 |
| Completion | Preserve authoritative native reason, token and page metadata | Preserve every window boundary/cap/rejection/continuation |

These protocols are **not equivalent**. P2-A 16384 is separately experimental and is not a full benchmark route. The ngram35/window1024 policy path is now **merged into main via PR #32** (budget/ngram gates, collision observer, `ResourceWatchdog`); the production default remains ngram 0 and five pages/8192 tokens — the merged gates are opt-in policy, not a new default. Length termination, partial Markdown, missing pages or structural stop cannot become complete PASS. Independent page OCR followed by concatenation is not whole-document inference.

Every document uses common states PASS, FAIL, TRUNCATED, TIMEOUT, OOM, INTERRUPTED, NOT_RUN. The existing ledger owns atomic seals and resume: skip only complete sealed PASS with exact matching identities; preserve all attempts, failures and full denominator. Export strict scored predictions separately from diagnostic partial outputs. Record document_id, expected/processed pages, model identity, input/output hashes, finish_reason, generated_tokens, wall_seconds, GPU peak allocation, process peak RSS, state/error/run identity. APU allocator bytes are shared-memory allocation, not dedicated VRAM or total host usage; record host available memory too.

The PowerShell entry `scripts/run_windows_benchmark.ps1` delegates to the common runner, scopes runtime/cache paths and requires deliberate `-ExecuteGpu` for `run`. That switch is not campaign authorization. Job cleanup/accounting and its residual pre-assignment hard-death gap are described in [readiness](WINDOWS_NATIVE_READINESS.md).

Scoring uses all eight fields: TextEdit, T-TextEdit, TableTEDS, T-TableTEDS, FormulaCDM, ReadOrderEdit, FigureF1, HeadingTEDS. Common scorer converts raw edit distance to `100*(1-distance)` and similarity fields to percentages. Overall stays null if any required metric is unavailable; never average seven fields or replace them with CER. Do not recompute a paper Overall from rounded table cells.

Native Windows full scoring is BLOCKED. Seventy-five dependency requirements match installed torch-free metadata, but positive formula rendering and execution parity are unproven: XeLaTeX, Ghostscript and latexmlc are absent; frozen source uses POSIX cat/perl redirection, fork and SIGALRM assumptions. Existing g05 PASS is WSL/Linux scorer evidence, not native Windows scorer PASS. Keep official source unchanged. A qualified Linux CPU scorer may consume Windows predictions only after separately authorized private transfer and input/output identity verification; no such transfer occurred in this mission.

### Scorer bridge (Option B — implemented on the integration tree)

The approved Option B route is implemented and synthetic-fixture tested on the
integration execution tree (`benchmark/bridge.py`, `scripts/scorer_bridge.py`):
Windows inference -> strict-mode prediction export (failed documents are
explicit empty files that stay in the denominator and failure catalog;
publication-detector sanitization refuses a dirty export before anything is
written) -> hash-bound `predictions-manifest.json` -> Linux-side verification
(per-file hashes, inventory equality against the **local** ground truth — GT
never crosses) -> the unchanged frozen Linux CPU scorer -> the eight official
metrics + Overall comparison report (Overall null while any metric is
missing; explicit `linux-cpu-official` scope labels — a bridge report is
never a Windows-native scorer pass). The real eight-metric positive control
uses the original scorer's verbatim synthetic selftest output; on hosts
without the Linux CDM renderer FormulaCDM stays null and Overall stays null
by design. No GT or prediction transfer has been executed; any real transfer
requires separate explicit authorization with frozen input/output digests.

### Fresh golden regression at the frozen execution commit

Because the execution tree moved past the original physical-evidence commit
(main merged PR #32; the tree also vendored the PR #28 admission contract and
gained P5 telemetry), a fresh bounded synthetic golden rerun was executed on
the physical Windows device at the frozen execution commit under the same
13-case CONTENT/STRUCTURE/PHYSICAL contract, two frozen environments, model
and wheel identities (GOLDEN_PROTOCOL_V3; outcome recorded in
[PRODUCT_STATUS](PRODUCT_STATUS.md) and the integration-closure ledger). The
historical `3556e38` 26/26 receipt remains valid evidence for that commit
only; it never substitutes for the execution tree.

Public campaign receipts carry all user-required hardware/stack/identity/count/metric/reference/comparability/delta/digest fields. For unrun campaigns, actual measured identity and run fields are null; requested target profile is separate. Current counts remain 0/420 and 0/100 documents, 0/3135 and 0/18272 pages; every score and official delta is null.
