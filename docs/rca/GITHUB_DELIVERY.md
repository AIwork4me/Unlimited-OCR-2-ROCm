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
