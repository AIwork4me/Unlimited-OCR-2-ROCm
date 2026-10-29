# Benchmark readiness and audit

Subsequent G0.5/G1 work is recorded in [the closure matrix](G05_G1_CLOSURE.md).
The infrastructure-only audit below remains its original dated record.

Date: 2026-10-29 (Asia/Shanghai). This is the first, infrastructure-only task
under [issue #6](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/issues/6).
No full GPU benchmark was executed. Existing evidence was read, not rewritten.

## Audit basis

Base `beb38d1` includes merged RCA #3 and wheel validation #4. Read the current
Phase 1 validation/final/reproduction/review/limitations reports, all current
`docs/validation/` reports, the RCA and whole-PDF diagnostic, and draft #5 at
`cfd684224fd4b306cdc22837b4abcf9c432a1c35` including its native OCR/review reports.
The historical official wheel's default fused failure and local candidate's
physical forward-matrix success are different artifacts. #5 remains unmerged
experimental synthetic evidence. Its 146 requests and the unscored 15-page
PDF diagnostic do not constitute a full-dataset result.

Official client, processor, scorer, GT, Marathon patch and customized SGLang
wheel were inspected at the identities in BENCHMARK_PROTOCOL.md and the lock.
Downloaded GitHub evaluator bytes were checked against their Git blob IDs
before SHA256 locking. Full GT scopes reconcile to 420/3135 and 100/18272.
Only IDs/counts/digests, not GT content or images, enter Git.

## Dataset/scorer status

| Component | Result | Remaining blocker |
| --- | --- | --- |
| Original metric code, both profiles | Source freeze complete | Full-scale execution unrun |
| Seven non-formula metric known answers | PASS on local CPU, original implementations | No GPU inference implied |
| Original converter → matcher → evaluator fixture | PASS, seven perfect metrics; cross-page text/table, heading and page-attributed figure | Broader malformed-content and full-dataset coverage require real runs |
| FormulaCDM / all-eight Overall | **BLOCKED / null** | `xelatex` absent; TeX/font/PDF closure and Windows POSIX-shell render compatibility unvalidated |
| MPDocBench GT, complete scope and image source revisions | Frozen; **D0-verified locally 2026-10-30: 420/420 documents, 3135/3135 pages, all SHA256 + decode PASS** (409 archive docs + 11 authorized SlideVQA decks, 12/12 source shards) | GT JSON files not yet on the Linux host (scorer-stage dependency) |
| Marathon GT, scope and remote page identities | Frozen; **D0-verified locally 2026-10-30: 100/100 documents, 18272/18272 pages, 11,805,490,064 bytes exact** after one authorized repair of transfer-truncated `mp3lan_0165/page_43.png` (receipt chain preserved) | GT JSON files not yet on the Linux host (scorer-stage dependency) |
| Full GPU execution, both datasets | **NOT TESTED** | Adapter limits, all-eight scorer, full data and reviewed quality gates |

The new scorer receipt records imperfect matched-sample known answers, original
raw metric aggregates, exact Python environment and observed renderer identities.
No dependency-unavailable score is invented. CPU CI checks seven available
metrics and the full synthetic processing path, preserving an explicit BLOCKED
all-eight receipt when the renderer is unavailable.

## G1–G5 blockers

Issue #6 numbers five final deliverables but does not name them G1–G5. This
table maps G1–G5 to those five deliverables in the same order; it does not
invent new acceptance thresholds.

| Gate | Deliverable | Status / exact blocker |
| --- | --- | --- |
| G1 | Adaptation matrix | Matrix delivered. **Overall adaptation BLOCKED**: only the linked Windows scopes physically passed; long-document native completion metadata/caps, candidate production separation, Linux W7900 physical profile and SGLang HIP/dependency parity are unfinished. |
| G2 | Frozen benchmark protocol | Rules/source/GT/image identities delivered. **Local dataset payloads D0-verified 2026-10-30** (MPDocBench 3135/3135, Marathon 18272/18272, SlideVQA 12/12 shards — `validation/LINUX_DATASET_READINESS.md`). Still BLOCKED for execution by: pinned CDM/TeX/font/PDF closure and Linux-host GT presence. |
| G3 | Full-scope run inventories and status summaries | Metadata-only scopes and tools delivered. **Full run evidence NOT TESTED**: 0/420 and 0/100 documents inferred in this task; local full page checksums/raw attempts are absent. |
| G4 | Official benchmark results | **BLOCKED**: no full GPU outputs, no measured FormulaCDM, no all-eight score, no reconciled per-document failure/metric coverage or predeclared benchmark quality gate. No BENCHMARK_RESULTS score is fabricated. |
| G5 | Release readiness | **BLOCKED** by G1–G4 plus clean supported-platform installation, external reproduction, long-document stability and security/license review. No release, stable tag, merge or outreach authorized. |

See BENCHMARK_EXECUTION.md for commands and the next physical validation order.
The original historical paths/hashes and retained worktrees remain unchanged.
