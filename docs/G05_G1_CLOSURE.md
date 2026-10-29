# G0.5 closure and G1 native completion contract

Date: 2026-10-29, Asia/Shanghai. Scope: Issue #6 and PR #7 at
`d842ce6f5cf187d17f4675e5ac37c188fc08cf9a`. This is scorer/data readiness and a
reviewed CPU-tested native contract. No full model benchmark was executed.

| Gate | Status | Exact scope and evidence |
| --- | --- | --- |
| Locked official Qwen and Marathon scorers | PASS | Both profiles pass all eight known answers and the complete converter → matcher → evaluator fixture, including a real formula. [Scorer receipt](../eval/benchmarks/g05/scorer/summary.json). |
| FormulaCDM controls | PASS | Real exact F1=1.0 and mismatched F1=0.4; valid GT plus a demonstrated renderer error; externally enforced timeout after observing the real prediction renderer; no active descendants after cleanup. |
| MPDocBench acquisition | PASS | 420 documents / 3,135 pages. All 3,051 original page hashes unchanged; exactly 11 authorized SlideVQA decks / 84 serialized JPEGs added. [Full-scope receipt](../eval/benchmarks/g05/datasets/MPDocBench-full-scope.json). |
| MarathonDocBench acquisition | PASS | 100 documents / 18,272 pages / 11,805,490,064 image bytes acquired and independently reread against frozen LFS hashes and sizes. [Receipt](../eval/benchmarks/g05/datasets/MarathonDocBench.json). |
| Native finish contract and CPU failure boundaries | PASS | Actual native EOS/length dictionary retained, malformed metadata/pages fail closed, interruption/resume/atomic preservation tested; 64 focused tests independently passed with no remaining P1/P2 findings. [Review](../eval/benchmarks/g05/native/review.md). |
| Whole-document native execution above five images | BLOCKED | Five-image / 8,192-new-token / 16M-pixel limits remain. No token-count inference of stop; page-guard EOS does not establish semantic completeness. |
| New physical Windows and Linux validation | NOT RUN | No GPU/model execution in this mission. Linux requires a separate reviewed profile and physical hardware; WSL CPU scoring establishes no Linux GPU support. |
| Full benchmark inference and quality | NOT RUN | 0/420 MPDocBench and 0/100 Marathon documents inferred in this mission; no full benchmark scores, parity or performance claim. |
| Official CDM internal 30-second timer branch | NOT RUN | Natural-return diagnostics did not exercise that branch. The controlled outer timeout above is separately identified. |

Both official scorer profiles produced these predeclared imperfect known answers:

| Metric | Known answer | Full synthetic formula pipeline |
| --- | ---: | ---: |
| TextEdit | 37.5 | 100 |
| T-TextEdit | 37.5 | 100 |
| TableTEDS | 50 | 100 |
| T-TableTEDS | 50 | 100 |
| FormulaCDM | 100 | 100 |
| ReadOrderEdit | 66.66666666666667 | 100 |
| FigureF1 | 25 | 100 |
| HeadingTEDS | 75 | 100 |

These are authored fixture results, not model quality scores. Raw original
aggregates, formula samples, errors, subprocess commands, measured CPU child
maximum RSS, Python packages, binary/font identities and sanitized/public hash
mapping are retained in [the scorer closure](G05_SCORER.md). The official scorer,
its dependency lock and worker remain unchanged. The separate Linux CPU venv
uses repository-local tool overlays and caches. Its runtime does not modify the
preserved native ROCm stack.

[Dataset documentation](G05_DATASETS.md) records frozen revisions, complete page
hash indexes, authorized SlideVQA serialization, Python/Pillow/JPEG identities
and access/licensing restrictions. Restricted payloads, source annotations,
credentials, models and wheels are excluded from Git. Earlier 401, authorization
discovery and original archive receipts remain intact and are superseded only
by separate successful acquisition receipts.

Acquisition-status fields in the unchanged benchmark lock remain the original
P0 observations. Current payload readiness comes from the new G0.5 receipts;
no dataset, scorer or model revision was silently replaced.

[Native contract and next commands](NATIVE_COMPLETION_CONTRACT.md) identifies
the pinned official source lines, EOS exclusion, total-context length exit,
page-guard semantics, exact full-document/base-image mapping and retained caps.
It provides unrun Windows MATH doctor/smoke commands using the preserved short
path, and Linux hardware/provenance commands with explicit loader prerequisites.
Model weights, tokenizer, BF16, DASH/SR-SWA behavior and MATH defaults are unchanged.

The native draft is [PR #8](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/8),
based on PR #7. The scorer/data draft is
[PR #9](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/9), stacked on that
native change because the
CPU worker imports the shared ledger; recorded wrapper producer hashes are thus
present in its ancestry. Neither PR is merged automatically. Historical GPU
evidence and original benchmark/model locks remain unchanged.
