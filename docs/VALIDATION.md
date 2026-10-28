# Executed validation

Date: 2026-10-28 (Asia/Shanghai). **Physical A–E: PASS.**
This is one actual Windows machine, two isolated environments in the same
checkout; model/pip/compiler caches may be reused. No second-machine,
Linux or other-GPU validation is claimed.

| Gate | Result | Evidence |
|---|---|---|
| A environment/BF16 | PASS | [strict doctor](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/doctor.json) |
| B model/native kernels | PASS | [kernel reference](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/sparse.json), [official weight load](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/load.json), per-case native kernel records |
| C OCR correctness | PASS | [single-page suite](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/suite.json) |
| D native multi-page correctness | PASS | [multi-page suite](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/suite.json) |
| E clean reproduction | PASS | [baseline](../eval/results/final-baseline/reproduction.json), [fresh linked run](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/reproduction.json), [creation witness](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/environment-creation.json) |

Gate E compares distinct producing environment IDs, a hashed physical machine
identity, 44 artifact hashes, 42 locked package versions and
all required cases. Identities/inventories come from producing reports,
not the interpreter used later to summarize them. The new environment and
result directory contain spaces and Chinese characters. No global settings changed.

| Case | Input pages | Raw PAGE markers | Empty terminal marker | Max page CER | Output tokens | Wall seconds |
|---|---:|---:|---|---:|---:|---:|
| [single/chinese](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/chinese/manifest.json) | 1 | 1 | false | 0.000 | 82 | 6.818 |
| [single/english](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/english/manifest.json) | 1 | 1 | false | 0.000 | 80 | 25.477 |
| [single/math](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/math/manifest.json) | 1 | 1 | false | 0.000 | 72 | 5.618 |
| [single/mixed](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/mixed/manifest.json) | 1 | 1 | false | 0.000 | 84 | 6.413 |
| [single/repeat_unicode_recovery](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/repeat_unicode_recovery/manifest.json) | 1 | 1 | false | 0.000 | 80 | 6.242 |
| [single/table](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/single/table/manifest.json) | 1 | 1 | false | 0.000 | 60 | 4.975 |
| [multi/pages_2_run_1](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_2_run_1/manifest.json) | 2 | 3 | true | 0.000 | 116 | 12.001 |
| [multi/pages_2_run_2](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_2_run_2/manifest.json) | 2 | 3 | true | 0.000 | 116 | 7.518 |
| [multi/pages_3_run_1](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_3_run_1/manifest.json) | 3 | 3 | false | 0.000 | 171 | 13.145 |
| [multi/pages_3_run_2](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_3_run_2/manifest.json) | 3 | 3 | false | 0.000 | 171 | 11.514 |
| [multi/pages_5_run_1](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_5_run_1/manifest.json) | 5 | 5 | false | 0.000 | 286 | 21.246 |
| [multi/pages_5_run_2](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/pages_5_run_2/manifest.json) | 5 | 5 | false | 0.000 | 286 | 21.288 |
| [multi/table_2_pages](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/multi/table_2_pages/manifest.json) | 2 | 3 | true | 0.000 | 94 | 6.643 |

All thirteen cases pass in both accepted environments: 26 requests total.
Each records twelve successful synchronized official DASH e2e prefill calls.
Every page has normalized CER 0 and exact important numbers. Native infer_multi
receives the whole ordered document; it is not repeated independent page OCR.

Acceptance: normalized CER <=0.02 per known page; exact important number tokens;
correct content-page count/order; no missing/duplicated content; valid HTML
table structure for table fixtures. NFKC/whitespace normalization operates on
content from the official V2 block parser. Text/numeric thresholds were set
before inference and were not lowered.

Structural interpretation was corrected after observing native generation:
one empty terminal PAGE can terminate a completed document. Raw count is
retained and raw_marker_count_matches_input remains false in those cases.
Only this one trailing empty segment is allowed; middle/extra blank segments
and incorrect content still fail. The original strict-marker FAIL is preserved.
See ENGINEERING_LOG.md for the source-based rationale and cost to strict consumers.

T01 English, T02 Chinese, T03 mixed, T04 table, T05 two pages, T06 five pages,
T07 repeated single, T08 repeated multi, T09 Unicode/space paths and T10
invalid-input recovery are exercised. Extra checks cover three pages,
simple arithmetic and a two-page table. Complex formula fidelity and broad
cross-document merging quality are not established by these fixtures.

The unadapted vision SDPA fails; the accepted path uses scoped GPU MATH SDPA
only for SAM/CLIP. Official model source/weights and decoder DASH/generation
are unchanged. Original failure, numeric reference and regression are retained.

Timings are synchronized diagnostics, not controlled throughput benchmarks.
Native load segment: 10.882s; repeated English request:
6.242s; five pages:
21.246s. Load timing excludes prior imports,
downloads and hash verification. See manifests for separate process/system
RAM and PyTorch GPU allocation/reservation. They must not be added as if
they were independent physical memory pools.
