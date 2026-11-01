# Windows dataset readiness

**READY_FOR_ADMISSION, not BENCHMARK_PASS.** New native Windows CPU verification on 2026-11-01 used CPython 3.10.20 and Pillow 12.0.0, with no torch/model/GPU import. Every indexed page was read, SHA256 checked, fully decoded and dimension checked. Frozen GT hashes, annotation-record digests, scope digest, explicit GT image order and per-document ordered path/hash digests reconcile.

| Dataset | Documents | Pages verified / decoded | Bytes | Missing / extra / errors | Duplicate content groups |
| --- | ---: | ---: | ---: | --- | ---: |
| MPDocBench public release | 420 | 3135 / 3135 | 688382455 | 0 / 0 / 0 | 3 |
| MarathonDocBench | 100 | 18272 / 18272 | 11805490064 | 0 / 0 / 0 | 19 |

Duplicate content groups are informational: identical page bytes in different legitimate positions. Duplicate ordered paths are zero; no documents/pages were dropped or deduplicated.

Public [dataset-readiness.json](../../eval/windows/dataset-readiness.json) is a byte-identical copy of the allowlisted Windows summary, SHA256 `541fa1fae3a987825958f794d9dac851670412d0178057ee2c90252862e80233`. It binds page indexes, GT, scopes and verifier source hashes. Complete per-page receipts and source images remain private; no GT text/image payload is published here.

| Identity | Frozen value |
| --- | --- |
| MPDocBench scorer/annotation revision | `4e5aee32280febb7b0844237ba087f837e3c8905` |
| MPDocBench GT SHA256 | `0a579dca6cf875a41fdab0975efee6d4e0fea999e0fa6c3ca95ad08b948f40df` |
| Marathon revision | `fa05fccbe82ee6ec628006a1d58bb153954c3ee3` |
| Marathon GT SHA256 | `7db0e659c76661ce87d4e1b3d47e010edbd48adfb446e3e2ee37260c60adf51c` |
| Benchmark lock SHA256 | `39f3dc09a9f9aa917f3ecf5f0deea690a7d1ed91ffb4e8e70803aac9e8dd7e08` |

MPDocBench includes all 84 SlideVQA pages from the pinned authorized acquisition/index. The old lock's 3051 acquired / 84 gated counts describe historical acquisition, not current Windows readiness. Current full index is authoritative for this qualification. The paper's 433 documents / 3246 pages differs from the public 420 / 3135 release: never interchange denominators or claim automatic paper equivalence. See [frozen protocol](../BENCHMARK_PROTOCOL.md).

Dataset PASS does not grant GPU campaign authorization, certify a scorer environment or imply any generated prediction passed.
