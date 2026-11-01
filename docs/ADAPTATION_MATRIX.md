# Adaptation matrix — 2026-10-29

PASS applies only to the explicitly executed scope in each row. This task ran
CPU infrastructure/scorer fixtures only; it did not rerun physical GPU tests.

| Platform/device | Backend | Physically executed evidence | Full MPDocBench / Marathon | Current gate |
| --- | --- | --- | --- | --- |
| Windows 11, Radeon 8060S/gfx1151, driver 32.0.22032.6002 | Native Transformers BF16, scoped GPU MATH | **PASS**, Phase 1 synthetic 1/2/3/5-page suite; separate original-wheel 15-page PDF diagnostic completed, unscored | **NOT TESTED**, both | **BLOCKED** by production five-image/8192-token caps, full finish/window reporting, dataset/CDM readiness |
| Same Windows device/driver, exact local candidate wheel | Native BF16, experimental AUTO vision | **PASS within draft #5 synthetic scope**, including actual fused dispatch and fixed numerical/HIP checks; MATH remains default | **NOT TESTED**, both | **BLOCKED**; AUTO production separation, exact candidate/no-overlays gate and long-document validation pending |
| Linux, Radeon PRO W7900/gfx1100, 48 GiB | Native Transformers | **NOT TESTED** | **NOT TESTED**, both | **BLOCKED** pending physical device access, Linux profile and separately reviewed ROCm/torch/Triton/native identities |
| Linux, Radeon PRO W7900/gfx1100 | Official customized SGLang | **NOT TESTED** | **NOT TESTED**, both | **BLOCKED**; wheel requires torch 2.13/CUDA dependencies, HIP implementation/serving/window parity and device tests absent |

Evidence:

- [Merged Phase 1 validation](VALIDATION.md), [RCA](rca/SDPA_ROOT_CAUSE.md),
  [15-page diagnostic, not scored acceptance](rca/FULL_PDF_PIPELINE.md).
- [Merged PR #4 exact wheel validation](validation/PYTORCH_212_CURRENT_WHEEL.md),
  [complete native identity gate](validation/CANDIDATE_IDENTITY.md).
- [Draft #5 immutable head](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/5),
  [candidate OCR report at cfd6842](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/blob/cfd684224fd4b306cdc22837b4abcf9c432a1c35/docs/validation/NATIVE_OCR_CANDIDATE.md).

The historical official torch wheel and local source-built wheel have different
SHA256 identities despite the same package version. The latter is not an AMD
published release. No Windows wheel or gfx1151 result validates Linux, gfx1100,
gfx1200, other dtypes, backward or serving concurrency. No source wheel is
rebuilt or distributed by this work. Existing historical paths/hashes remain
immutable; repository-local locations are a separate relocation concern.
