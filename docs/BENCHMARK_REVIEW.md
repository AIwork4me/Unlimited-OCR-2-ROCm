# Benchmark foundation review

Reviewed base: merged `main` at `beb38d1`; initial implementation `596440d`
and the following working-tree fixes. A fresh read-only reviewer inspected the
code, tests, protocol, source/dependency locks and new synthetic receipts.
No physical GPU/model execution or full benchmark scoring was performed.

One Important finding was reproduced with failing regressions: a verified `.py`
could be superseded by unchecked cached bytecode/native import artifacts.
The source gate now rejects those artifacts, and the worker imports only a
fresh copy of the locked files after rehashing it. Bytecode writes are disabled
for that process and children. The reviewer rechecked the fix and reported
**no remaining Important defect**.

The reviewer independently ran **44 benchmark CPU tests**, including artifact
rejection and real descendant-process timeout cleanup. Both inspected scorer
receipts have valid hash inventories and verified source snapshots, seven
original known-answer metric checks and the original continuation pipeline
PASS. CDM remains BLOCKED and Overall null. The frozen dependency bytes match
their lock. Full implementer CPU verification: **177 passed, five intentional
skips** (two physical opt-ins and three torch-optional reference tests).

New evidence publication uses byte-preserving Git attributes. Earlier local
producer/lock CRLF hashes reconcile to Git's LF source blobs through
`eval/benchmarks/scorer/producer-source-mapping.json`; the original receipts
remain intact. The final source/dependency files use LF bytes so the same
hash gate applies on Windows and Linux. This does not rewrite historical RCA,
wheel, Phase 1 or draft #5 evidence.

Review limitations: no Linux execution, real CDM renderer, full-dataset scoring
or GPU acceptance was established by local review. Human reconciliation of
scorer task applicability and reported sample coverage remains required.
Hosted Windows/Ubuntu results must be read for the delivered PR head; CPU CI
does not validate an AMD device or close G1–G5.
