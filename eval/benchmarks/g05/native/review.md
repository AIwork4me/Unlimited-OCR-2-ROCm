# Independent native completion approval

Narrow re-review of the remaining invalid UTF-8 mapping-key P1, following `.b/g05-review/native-review.md` and `native-review-final.md`. Read `.b/native-completion/cpu-key-fix-receipt.json`, inspected the revised key traversal and top-level/nested full-runner regressions, and independently checked the file hashes below. No source edits, network use, real torch/model imports, or GPU execution. All commands ran from the primary repository root with evidence beneath `.b/g05-review`.

**Outcome: no remaining P1/P2 findings in the reviewed native completion scope.** The original malformed-metadata manifest, ledger fallback-PASS, and repaired-header findings are resolved; the mapping-key follow-up is also resolved.

The revised traversal visits mapping keys before constructing a normal dictionary. An invalid string key becomes a diagnostic entry in `invalid_mapping`, causes metadata rejection, and preserves its identity separately from a valid literal escaped key. The top-level and nested runner regressions assert nonempty parseable FAIL manifests, retained native stop, distinct key identities, and unchanged raw bytes. The narrow change preserves valid JSON dictionaries and does not change inference/generation behavior or resource caps.

Independently reran the original key probe with only a fresh output root (`.b/g05-review/keys-probe-final`): diagnostic invalid=true, no raised exception, raw present, manifest present and 2502 bytes. Original outputs remain untouched. Probe source: `.b/g05-review/native_review_keys_probe_final.py`; receipt: `.b/g05-review/keys-probe-final/receipts.json`.

Independently reran the complete focused native/inference/ledger suite: **64 passed in 2.42s**. This includes original malformed native metadata, both mapping-key cases, canonical-header rejection, real-ledger malformed-block integration, interruption/resume/atomic evidence retention, and generic-ledger >1M-character acceptance. Full-suite results in the implementer receipt remain separately reported evidence (237 passed, six skips); this narrow re-review did not repeat that broader suite.

| Reviewed file | SHA-256 |
| --- | --- |
| `src/unlimited_ocr_2_rocm/completion.py` | `7ea47e656f1dfb5ceb34a1bfa64942838a992cbd12122d87359a5aeaae72f873` |
| `src/unlimited_ocr_2_rocm/inference.py` | `6e8a732c14f12dedf0c0a31d151b80dbef196fd686ff9194b0b2720af5ee1564` |
| `src/unlimited_ocr_2_rocm/benchmark/ledger.py` | `473e0b35bcca808958a890135220d1be4c9fb133ac1f5a4e07cce93355e274f8` |
| `tests/unit/test_native_completion.py` | `1fe1355da7fed1a6191ed98cdc62a8d04d7933c5708b6ad317ce2c035c7c7d75` |
| `docs/NATIVE_COMPLETION_CONTRACT.md` | `cd35c383ec6d569753ad9d21da8e6ea90cd50147243672619a839ec921012394` |

This approves the reviewed implementation and CPU contract boundaries. It does not establish semantic document completeness, any benchmark score, or physical GPU validation. Page-guard-forced EOS limitations remain accurately disclosed, and all existing model/stack/algorithm/crop/DASH/BF16/vision-MATH and native resource limits remain preserved in this scope.
