# Windows baseline and PR audit

Snapshot: 2026-11-01, fetched main `52d593e0686176b3842d78fbac3ef76956ecf1ae`. Status is the observed snapshot, not a continuing monitor. Merged code and experimental evidence are separate from this mission's physical validation.

| PR | Observed status | Scope |
| --- | --- | --- |
| [#1](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/1) | MERGED | Phase 1: verified native Windows gfx1151 OCR on ROCm 7.14.1 |
| [#3](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/3) | MERGED | Confirm Windows gfx1151 SDPA packaging defect with full-PDF evidence |
| [#4](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/4) | MERGED | test(rocm): verify source-built PyTorch 2.12 fused SDPA on Windows gfx1151 |
| [#5](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/5) | OPEN DRAFT | experiment(ocr): validate native OCR and fused vision on the verified Windows ROCm wheel |
| [#10](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/10) | MERGED | test(g1): seal original-wheel Windows regression and resource strategy |
| [#15](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/15) | OPEN DRAFT | test(sdpa): execute synthetic diagnostics and preserve exact-parity rejection |
| [#16](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/16) | MERGED | P1: real-document GPU→official-scorer integration on Linux gfx1151 (REAL_DATA_INTEGRATION_ONLY) |
| [#17](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/17) | MERGED | docs(review): independent PR #16 correctness-closure review |
| [#18](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/18) | MERGED | feat(p15): runner hardening — ledger-integrated supervised execution |
| [#19](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/19) | MERGED | feat(t-tableteds): predeclared merged-table selection for eight-metric coverage |
| [#20](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/20) | MERGED | feat(p2a): explicit opt-in output-token-budget experiments |
| [#21](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/21) | OPEN DRAFT | test(windows): record external cleanup and cold-restart parity blocker |
| [#22](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/22) | OPEN DRAFT | test(rca): preserve cold-start and same-wheel backend causal controls |
| [#23](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/23) | MERGED | docs(rca): n-gram-35 source audit and opt-in experiment plan (P1) |
| [#24](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/24) | OPEN | feat(p2b): opt-in official n-gram suppression behind a pinned-digest policy |
| [#25](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/25) | OPEN DRAFT | test(rca): isolate BF16 scalar lowering in CLIP TensorExpr kernels |
| [#26](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/26) | OPEN DRAFT | test(rca): qualify scoped QuickGELU and matched attention contexts |
| [#27](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/27) | OPEN DRAFT | test(rca): audit independent SAM and CLIP numerical references |
| [#28](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/28) | OPEN DRAFT | Require identity-bound fail-closed admission before experiment workers |
| [#29](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/29) | OPEN DRAFT | Add a source-locked experimental scoped eager QuickGELU lifecycle |

Use the original-wheel scoped-MATH Windows route established by #1/#10. #3/#4 prove packaging/operator facts, not complete fused-OCR admission. #5/#15/#21/#22/#25–#29 remain experimental or blocked evidence; none promotes AUTO/fused attention or scoped QuickGELU to production.

Reuse merged common ledger, runner hardening, completion, scorer schema and opt-in budget infrastructure from #16–#20. Their Linux observations do not establish Windows physical PASS. #23 is source-audit/planning evidence; #24 is OPEN at this snapshot, so ngram35/window1024 integration is not a merged default. #28 admission is privately reused at its exact reviewed source hash for the bounded existing regression; it is not silently added as a production dependency.

This mission changes no existing PR, README, Linux model/scorer implementation, or Linux result. New source commits and review preparation are local. Full campaign admission remains blocked as recorded in [readiness](WINDOWS_NATIVE_READINESS.md).
