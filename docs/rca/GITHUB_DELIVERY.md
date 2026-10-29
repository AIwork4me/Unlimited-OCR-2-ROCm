# GitHub delivery

- Repository Issue: https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/issues/2
- Review-ready PR against main: https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/3
- Branch: `investigation/windows-gfx1151-sdpa-root-cause`
- Published evidence commit: `816b09a9018d94d01be7ac6e772c903bc140d22f`
- Final tested code commit: `302ba8a` (subsequent commits add documentation/evidence only)
- Evidence directory: [eval/rca/gfx1151-rocm7141](../../eval/rca/gfx1151-rocm7141/)
- Root cause: **CONFIRMED, L4**, with the limitations in [the report](SDPA_ROOT_CAUSE.md).

[Publication verification](PUBLICATION_VERIFICATION.json) records actual remote Git-ref/blob checks for all 430 SHA256SUMS entries and HTTP GET/byte equality for all 155 then-referenced files at the immutable evidence commit. Original-to-published mappings remain in the evidence directory. Final delivery rechecks the PR's final head after this index is pushed; the PR's checks panel is authoritative for hosted CI.

Local final checks: CPU suite 68 passed/5 expected skips; physical suite 73 passed with 15 upstream TorchScript deprecation warnings. The physical suite includes the optional CPU-reference tests and existing GPU/OCR tests. Ruff, formatting, mypy, build and pip check exit 0. Actual command streams, exit codes, UTC and process IDs are in [the final manifest](../../eval/rca/gfx1151-rocm7141/checks-final/process-manifest.json).

One independent review was completed; its three Important findings were corrected by RED→GREEN tests. See [review dispositions and the one deferred display minor](REVIEW.md). The original 113 child reports remain unchanged; three post-review probe results are indexed separately. Production runtime, source/weight hashes, driver and the original validated environment are preserved. The secondary diagnostic environment has its original images restored.

The English [upstream handoff](UPSTREAM_HANDOFF.md) recommends ROCm/PyTorch wheel integration maintainers via TheRock, coordinated with AOTriton. Related upstream issues and the known source-pin correction were checked. No duplicate upstream issue or automatic upstream post was made; the source correction was not rebuilt and no corrected official wheel is claimed validated. The feature PR remains open and unmerged; no release, stable tag or Phase 2 was started.

## Subsequent PDF evidence delivery

The user subsequently requested publishing all follow-up research and explicitly the original PDF input. The [full-PDF report and exact reproduction instructions](FULL_PDF_PIPELINE.md) and [evidence supplement](../../eval/rca/gfx1151-rocm7141/pdf-followup/README.md) now include the exact **15-page, 1,515,878-byte PDF**, full OCR/token outputs, both whole-document process traces, earlier first-page real-tensor replay records, legacy text gaps, immutable upstream snapshots/notices, source-byte correction, original/published digest mappings and a 555-entry research inventory. Regenerable PNG/tensor binaries, weights, caches and literal user-directory paths are excluded. The original core evidence checksum set is preserved.

Whole-document MATH ends `stop`, 13,836 tokens, 15 nonempty segments; native DASH stays enabled. The unadapted separate process first fails in SAM efficient SDPA before decoder computation. This corroborates L4 and establishes full-document inference completion, not OCR accuracy or a permanently repaired official wheel. A fresh bounded publication review found no findings. CPU-only renderer reproduction regenerates all 15 original PNG hashes. Final post-push remote byte checks and hosted CI for this supplement are recorded in the PR conversation; historical receipts above remain bound to their original commits.

[PDF publication receipt](PDF_PUBLICATION_VERIFICATION.json) binds the actual remote branch and all downloaded bytes to immutable evidence commit **6a3c7318c970015f2f852763f8406274cba1110b**: **565** remote checksum entries (430 original + 135 supplement) and **589** HTTP GETs all match Git blobs, including the exact original PDF. The receipt records successful Windows and Ubuntu CPU jobs in [PR CI run 40661720895](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/actions/runs/40661720895). The receipt's own later commit is verified separately in the PR delivery comment, avoiding a self-referential digest claim. [Archived publication verifier](PDF_PUBLICATION_VERIFIER.py.txt).
