# Adaptation matrix — 2026-10-30 (updated: D0 dataset verification PASS on Linux)

PASS applies only to the explicitly executed scope in each row. Scope change
2026-10-29: the **primary Linux validation target is the native Linux
Ryzen AI MAX+ (PRO) 395 / Radeon 8060S / gfx1151 host** (Ubuntu 24.04.4,
OEM kernel 6.17.0-1033-oem — the *verified runtime* on this host is the
`torch 2.12.0+rocm7.2` wheel stack over the untouched system ROCm 7.2.1
userland, **not** ROCm 7.14.1; the 7.14.1 compatibility matrix was only the
reference listing this OEM kernel as supported). **Radeon PRO W7900 / gfx1100
is retained as a future optional independent hardware-compatibility target**:
no gfx1100 result may be derived from gfx1151 evidence, and no gfx1151 result
validates gfx1100. Windows and Linux evidence remain separate records.

| Platform/device | Backend | Physically executed evidence | Full MPDocBench / Marathon | Current gate |
| --- | --- | --- | --- | --- |
| Windows 11, Radeon 8060S/gfx1151, driver 32.0.22032.6002 | Native Transformers BF16, scoped GPU MATH | **PASS**, Phase 1 synthetic 1/2/3/5-page suite; separate original-wheel 15-page PDF diagnostic completed, unscored | **NOT TESTED**, both | **BLOCKED** by production five-image/8192-token caps, full finish/window reporting, dataset/CDM readiness |
| Same Windows device/driver, exact local candidate wheel | Native BF16, experimental AUTO vision | **PASS within draft #5 synthetic scope**, including actual fused dispatch and fixed numerical/HIP checks; MATH remains default | **NOT TESTED**, both | **BLOCKED**; AUTO production separation, exact candidate/no-overlays gate and long-document validation pending |
| **Linux, Ryzen AI MAX+ PRO 395 / Radeon 8060S / gfx1151** (primary Linux target; Ubuntu 24.04.4, kernel 6.17.0-1033-oem) | Native Transformers BF16 (officially published `torch 2.12.0+rocm7.2` + `triton-rocm 3.7.0` + `torchvision 0.27.0+rocm7.2` wheels, SHA-pinned), scoped GPU MATH, DASH Triton | **PASS** — doctor/BF16, Triton compile+execute, DASH sparse kernel, vision SDPA MATH, 1-page suite (5 cases + unicode recovery), 2/3/5-page native suite ×2 + table case; output text identical to Windows evidence | **NOT TESTED**, both (profiles prepared: `eval/benchmarks/profiles/linux-gfx1151-*`) | **BLOCKED** by the same production five-image/8192-token caps; long-document ladder proposed in `docs/LONG_DOCUMENT_READINESS.md`; **datasets D0-verified 2026-10-30** — MPDocBench 420/420 docs + 3135/3135 pages, MarathonDocBench 100/100 docs + 18272/18272 pages, SlideVQA 12/12 shards ([readiness report](validation/LINUX_DATASET_READINESS.md)) |
| Linux, Radeon PRO W7900/gfx1100, 48 GiB (future optional independent target) | Native Transformers | **NOT TESTED** | **NOT TESTED**, both | **BLOCKED** pending physical device access, a separate Linux profile and separately reviewed ROCm/torch/Triton/native identities |
| Linux, Radeon PRO W7900/gfx1100 (future optional independent target) | Official customized SGLang | **NOT TESTED** | **NOT TESTED**, both | **BLOCKED**; wheel requires torch 2.13/CUDA dependencies, HIP implementation/serving/window parity and device tests absent |

Evidence:

- [Merged Phase 1 Windows validation](VALIDATION.md), [RCA](rca/SDPA_ROOT_CAUSE.md),
  [15-page diagnostic, not scored acceptance](rca/FULL_PDF_PIPELINE.md).
- [Merged PR #4 exact wheel validation](validation/PYTORCH_212_CURRENT_WHEEL.md),
  [complete native identity gate](validation/CANDIDATE_IDENTITY.md).
- [Draft #5 immutable head](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/5),
  [candidate OCR report at cfd6842](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/blob/cfd684224fd4b306cdc22837b4abcf9c432a1c35/docs/validation/NATIVE_OCR_CANDIDATE.md).
- **Linux gfx1151 (2026-10-29)**: [host preflight](validation/LINUX_GFX1151_PREFLIGHT.md),
  [environment decision](validation/LINUX_GFX1151_ENVIRONMENT_DECISION.md),
  [physical validation report](validation/LINUX_GFX1151_NATIVE_VALIDATION.md),
  evidence tree `eval/results/Linux gfx1151 verified/` (doctor/BF16, Triton,
  DASH sparse, vision SDPA, single+multi OCR manifests, wheel SHA receipts,
  environment-creation witness).
- **D0 dataset verification (2026-10-30)**: [Linux dataset readiness](
  validation/LINUX_DATASET_READINESS.md), receipts under
  `eval/results/Linux gfx1151 verified/datasets/` (attempt-1 BLOCKED,
  authorized page repair, attempt-2 PASS).

The historical official torch wheel and local source-built wheel have different
SHA256 identities despite the same package version. The latter is not an AMD
published release. The Linux stack is a third, independently published identity
(`torch-2.12.0+rocm7.2` manylinux wheel, SHA-256
`099e265de73094cb6a2112ce9d7940c59bae0c884e9516f16719e392a5471df6`). No
Windows wheel or gfx1151 result validates Linux gfx1100, gfx1200, other
dtypes, backward or serving concurrency, and vice versa. No source wheel is
rebuilt or distributed by this work. Existing historical paths/hashes remain
immutable; repository-local locations are a separate relocation concern.
