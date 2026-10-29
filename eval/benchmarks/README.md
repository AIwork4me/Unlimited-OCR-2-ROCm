# Benchmark metadata and new CPU evidence

`benchmark-lock.json` freezes source/GT identities and explicit unresolved
image-acquisition fields. `manifests/*.jsonl` contains every official document
ID and count, ordered page-path and annotation-record hashes. Marathon also
contains a digest of the ordered remote per-page hashes. These are scope
inventories, not a claim of downloaded/validated image bytes or completed runs.

Generate complete per-page local manifests with `unlimited-ocr2-benchmark
inventory`; keep those, downloaded datasets, environments and raw full runs
under ignored `.b/`. Official dataset documents/annotations/images are never
copied into Git by this tooling.

The new `scorer/` receipts are CPU synthetic tests only. Seven original metrics
and the original cross-page pipeline pass. CDM is BLOCKED and Overall null;
raw logs, fixture/metric outputs and environment identities explain that state.
Read docs/BENCHMARK_PROTOCOL.md, BENCHMARK_READINESS.md and
BENCHMARK_EXECUTION.md before any physical GPU work.
