# PDF follow-up evidence, 2026-10-28

This supplements the [core SDPA RCA](../../../../docs/rca/SDPA_ROOT_CAUSE.md). Start with the [full-document report and reproduction instructions](../../../../docs/rca/FULL_PDF_PIPELINE.md).

## Fixed input and actual output

- [Original user-supplied PDF](input/Unlimited-OCR-2.pdf): **15 pages, 1,515,878 bytes**, SHA-256 `1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444`. Uploaded at the user's explicit request, unchanged. It is not byte-identical to the separately downloaded official model-repository paper; do not substitute a different PDF and claim the same input hash.
- [Full original OCR output](whole-document/math/raw.txt), [readable output](whole-document/math/readable.md), [generated/prompt token IDs](whole-document/math/token-ids.json).
- [Post-run verified summary](whole-document/summary.json), [full raw MATH manifest](whole-document/math/manifest.json), [full JSONL trace](whole-document/math/trace.jsonl), [stdout](whole-document/math.stdout.log), [stderr](whole-document/math.stderr.log).
- [Unadapted first failure](whole-document/unadapted/first-failure.json), [failed process manifest](whole-document/unadapted/manifest.json), [full failed trace](whole-document/unadapted/trace.jsonl), [stderr](whole-document/unadapted.stderr.log).

The PDF is a third-party research document supplied by the user, not an original work of this wrapper project. Inclusion does not relicense the paper under the wrapper's MIT license. Baidu's copied repository documentation/code notices are retained in [upstream/LICENSE](whole-document/upstream/LICENSE); immutable source URLs and byte/AST checks are recorded in the manifests. No model weights or DLLs are redistributed.

## Research history, not rewritten results

1. `first-page/` preserves the earlier 144-DPI page-1-only check, actual SAM/CLIP input replay results, numerical thresholds, raw failures, output and source/weight hashes. Its Pillow-only PDF failure and five-image wrapper limit did **not** answer the official whole-document question.
2. `whole-document/` records the corrected official 300-DPI reader plus **one native Transformers infer_multi request with all 15 pages**, in separate unadapted and scoped-MATH processes. All 15 images are verified and ordered before inference. The MATH run ends with `stop`, 13,836 tokens and 15 nonempty segments; native DASH remains active. OCR accuracy is unscored.
3. `legacy-appendix/` fills in previously unpublished text helpers, exploratory/install/lint streams, model outputs and final remote-publication receipts from the core RCA. The original 430-file checksum set remains unchanged in its parent directory. A prior publication receipt describes its own older commit, not this new publication.

Files named `historical-*`, `partial-*`, and the first-page `summary.json` retain statements made **at that time** (including local-only publication status and incomplete scope). They are archived evidence and are superseded by the final report. Raw producer statuses such as a model-load `PASS` mean exactly their recorded substep, not OCR accuracy. Native debug `ok=False` strings are preserved without reinterpretation; no full model correctness claim is based on those debug strings.

## Integrity and reproduction limits

- [Export manifest](export-manifest.json) binds each original file digest to the uploaded bytes, text encoding and path sanitization. Home/build-user paths are neutralized; original local evidence is retained.
- [Inventory](inventory.json) accounts for both PDF investigations and core text research, linking already-published original digests or the new copies. Regenerable PNGs and the roughly 98 MB/3.7 MB page-1 tensor captures are omitted from Git; their hashes and sizes are retained. Render images and Q/K/V can be recaptured from the fixed PDF and pinned model.
- [SHA256SUMS](SHA256SUMS.txt) covers this directory except itself. Token IDs and document text are public alongside the explicitly requested PDF; caches, weights, compiled kernels, credentials and user-directory paths are excluded.
- Producer `.py.txt` files are archival text, not installed/importable modules. Their original hashes are in the export manifest. The reproduction guide configures only input/cache paths in fresh local copies and records the newly configured script hash. No old run is overwritten or impersonated.
- The runtime/image cause was proven by the earlier images-only experiment. The whole-PDF comparison corroborates that cause on the real model path; it is not another images-only A/B. Other hardware/OS, rebuilt official wheels, training/backward, and independent OCR accuracy remain unvalidated.
