# Windows handoff to the Linux integrator

Prepared locally on base `52d593e0686176b3842d78fbac3ef76956ecf1ae`; code commits `4abba4f` (Job supervision/publication), `c431027` (original archive/offline install), `3556e3867a086437a43a645fa907689f4dd25260` (PowerShell/common-runner GT binding), `5436f80233477cf53f0e5eda4406c27a9e09c035` (inherited HF cache containment/restoration). No new remote branch/PR was published. Existing Draft #28/#29 remain untouched; README/Linux files/results were not edited.

Integrate common ledger, scorer.METRICS, metric conversion and null aggregation without another scoring system. New Windows entry delegates to scripts/run_mpdoc_sample.py; NativeJob handles suspended creation, ownership, descendants, timeout/interruption and accounting. Frozen GT is checked before prepare. Environment receipts include triton-windows. Three P2 findings were fixed and independently re-reviewed; the hard-death pre-assignment residual remains explicit. Source archive checks include new modules. Selected-root publication scanning is fixed without skipping a requested tree merely because an absolute ancestor is `.b`.

| Public artifact | Meaning |
| --- | --- |
| [dataset-readiness.json](../../eval/windows/dataset-readiness.json) | Windows full byte/hash/decode/order/GT qualification only; SHA256 `541fa1fae3a987825958f794d9dac851670412d0178057ee2c90252862e80233` |
| [benchmark-status.json](../../eval/windows/benchmark-status.json) | Unrun full campaigns; all required receipt fields; actual unknown identities and all scores/deltas null; separate requested target profile |
| [MPDocBench status](MPDOCBENCH_FULL_RESULTS.md) / [Marathon status](MARATHON_FULL_RESULTS.md) | Full denominators, exact paper references and explicit blockers, not completed result reports |
| [Install guide](WINDOWS_INSTALL_REPRODUCTION.md) | Tested offline package path, original archive identity, fresh-venv/cache safeguards |

The document receipt contract maps directly to common ledger metadata: expected/processed pages, model and input/output identities, authoritative completion, token/time/memory, state/error and run ID. Do not interpret a NOT_RUN template as a real document result. Final [baseline proof](NATIVE_BASELINE_VALIDATION.md) records actual two-environment13/13 qualification at inference commit3556. The [allowlisted synthetic receipt](../../eval/windows/golden-validation.json), SHA256 `8d2fda03ad3e979ecf0f33e15b15a7f48ccc1cf1cf94f360b833769f52be4410`, contains26 original entries and separate correctness/parity classifications. A prior environment-policy FAIL with0 infer entries remains sealed and bound into the independently reviewed corrected identity; no failed ledger was reset. Full campaign counts remain0.

Retained private evidence: initial baseline failure, checks-r2, fake-worker RED/GREEN/review logs, source archive check, two installer reports/freezes/pip checks, 43-artifact acquisition manifest/dry-run, per-page dataset qualification. Do not publish host paths, machine identifiers, raw GT/predictions, weights, tensors or model caches. Scan explicit public files and built source archive before handoff.

A qualified Linux CPU official scorer is a possible future scoring endpoint because native Windows renderer closure is blocked. Transfer of private Windows predictions requires explicit authorization and frozen input/output digests; no transfer occurred. Linux GPU validation and Windows validation remain separate hardware/stack evidence.

Next integration dependency is a common reviewed ngram/budget policy with a single bounded real-document admission proposal; no full-campaign execution grant exists. See [protocol](WINDOWS_BENCHMARK_PROTOCOL.md) and [Chinese report](../../WINDOWS_PRODUCT_DELIVERY_REPORT.md).
