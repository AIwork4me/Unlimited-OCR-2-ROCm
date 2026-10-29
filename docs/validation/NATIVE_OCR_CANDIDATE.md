# Native OCR with the verified local Windows ROCm wheel

Executed 2026-10-29 (Asia/Shanghai) on the physical Ryzen AI MAX+ 395 / Radeon 8060S, Windows gfx1151, driver 32.0.22032.6002. **Stages 1–4: PASS within the tested scope. MATH remains the default.**

## Exact candidate and environment

- Artifact: `torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl`, 388,478,127 bytes. Original compatibility address `C:/b/out/wheels/`; physical repository-local destination `.b/out/wheels/`. SHA256:

  `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`

- Installed into fresh `.b/o`, CPython 3.10.20 x64, using the explicit local artifact and `--no-deps`. PyTorch source identity `60dd50f143fd9b10ac23cde8cdd34be61232b139`; HIP 7.14.60850; ROCm SDK 7.14.1. torchvision 0.27.0+rocm7.14.1, torchaudio 2.11.0+rocm7.14.1, triton-windows 3.7.1.post27, Transformers 4.52.1 and the locked native dependencies are preserved.

- The historical torch gfx1151/gfx115x overlays are excluded. The source-built candidate is self-contained; the gfx115x overlay would overwrite its seven validated AOTriton image files. Matching SDK/device and torchvision-device packages remain installed. All selected provider wheels were inspected for candidate-file overlap.

- `aotriton_v2.dll` SHA256 `941bf05354b8c7239006c519a90d27d8494adbe743e0f6d5d8552095e4b27884`. All 15 torch native members, seven AOTriton image members and the SDK HIP DLL match provider-wheel bytes; required actually loaded native modules match the fresh environment. AOTriton runtime pin `7a6963a7265d2cf0fc48c6a63464386134d32483`; image signature `c11b5b886a10a00f925f74133d38b3505e67ccd9`. No rebuild or image overlay was performed.

[Wheel metadata/dependencies](../../eval/native-ocr/windows-gfx1151/wheel-metadata.json) · [install report](../../eval/native-ocr/windows-gfx1151/candidate-install.json) · [installed payload](../../eval/native-ocr/windows-gfx1151/candidate-payload.json) · [loaded identity](../../eval/native-ocr/windows-gfx1151/identity.observed.json) · [doctor/BF16](../../eval/native-ocr/windows-gfx1151/doctor.json) · [package freeze](../../eval/native-ocr/windows-gfx1151/freeze.log). The wheel is a locally verified source-built artifact, not an official AMD fix.

## Stage 1: PASS

pip check, strict environment doctor, BF16 GPU arithmetic, Triton compile/execute, native sparse-kernel reference and the existing MATH vision regression pass. The byte-identical PR #4 producer runs 27 fresh processes: 21 FUSED_PASS, four MATH_PASS, one CONTROL_PASS and one explicitly optional UNSUPPORTED forced-FLASH additive-mask probe. Numerical thresholds remain BF16 atol/rtol 0.03/0.03, with no failing or nonfinite elements. HIP checkpoints are zero; synchronization and downstream operations succeed.

The initial dependency-install attempt failed because the default Python index lacked the ROCm metapackage; the retry used the previously verified local metapackage. Both logs are retained. This preparation failure is separate from GPU acceptance. The full model environment includes Triton; the frozen producer records its presence in its legacy `prohibited_imports_loaded` field. The producer and numerical checks were not changed.

[27-process summary](../../eval/native-ocr/windows-gfx1151/regression/summary.json) · [processes/producer hashes](../../eval/native-ocr/windows-gfx1151/regression/processes.json) · [FLASH launch](../../eval/native-ocr/windows-gfx1151/regression/flash_launch.stderr.log) · [EFFICIENT launch](../../eval/native-ocr/windows-gfx1151/regression/efficient_launch.stderr.log) · [sparse reference](../../eval/native-ocr/windows-gfx1151/sparse.json).

## Stage 2 and Stage 3 correctness: PASS

Both modes pass all 13 required acceptance requests: five single-page content types, repeated Unicode/space-path recovery, repeated ordered native 2/3/5-page documents and a two-page table. All tested pages have normalized CER 0, exact important numbers, valid required structure and correct content-page count/order. Every request records 12 successful official DASH prefill calls. Raw output, normalized output and page markers remain preserved.

Model/code revision stays `9cf077c4b206e0e2a31879df937b0b0bcba3661c`; all reviewed source/config/tokenizer/weight hashes are verified before trust_remote_code. Weights were copied from the existing verified cache into `.b/h`, with original and destination hashes recorded. BF16, native infer/infer_multi, DASH/SR-SWA and decoder attention dispatch are unchanged.

| Case | MATH / AUTO | Max CER MATH / AUTO | Raw PAGE MATH / AUTO | Empty terminal MATH / AUTO |
|---|---|---:|---:|---|
| chinese | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| english | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| math | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| mixed | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| repeat_unicode_recovery | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| table | PASS / PASS | 0.000 / 0.000 | 1 / 1 | False / False |
| pages_2_run_1 | PASS / PASS | 0.000 / 0.000 | 3 / 3 | True / True |
| pages_2_run_2 | PASS / PASS | 0.000 / 0.000 | 3 / 3 | True / True |
| pages_3_run_1 | PASS / PASS | 0.000 / 0.000 | 3 / 3 | False / False |
| pages_3_run_2 | PASS / PASS | 0.000 / 0.000 | 3 / 3 | False / False |
| pages_5_run_1 | PASS / PASS | 0.000 / 0.000 | 5 / 5 | False / False |
| pages_5_run_2 | PASS / PASS | 0.000 / 0.000 | 5 / 5 | False / False |
| table_2_pages | PASS / PASS | 0.000 / 0.000 | 3 / 3 | True / True |

The pre-existing terminal-marker policy is unchanged: at most one empty trailing PAGE is an end delimiter, separately reported. CER <=0.02, exact-number, page-order and table thresholds were never relaxed. The accepted original Phase 1 evidence remains untouched.

[MATH single](../../eval/native-ocr/windows-gfx1151/stage2-single/suite.json) · [MATH multi](../../eval/native-ocr/windows-gfx1151/stage2-multi/suite.json) · [AUTO single](../../eval/native-ocr/windows-gfx1151/stage3-auto-single/suite/suite.json) · [AUTO multi](../../eval/native-ocr/windows-gfx1151/stage3-auto-multi/suite/suite.json) · [cache copy/hashes](../../eval/native-ocr/windows-gfx1151/cache-copy.json).

## Actual fused vision execution: PASS

Across both diagnostic suites, all 1224 observed vision SDPA calls are fused: 408 EFFICIENT_ATTENTION calls in masked SAM and 816 FLASH_ATTENTION calls in unmasked CLIP. No observed MATH or unobserved route is counted as fused proof.

| Module | Q/K/V shape | Mask | Observed operator |
|---|---|---|---|
| sam_model | [54, 12, 196, 64] | [54, 12, 196, 196] | EFFICIENT_ATTENTION |
| sam_model | [6, 12, 1600, 64] | [6, 12, 1600, 1600] | EFFICIENT_ATTENTION |
| vision_model | [6, 16, 101, 64] | None | FLASH_ATTENTION |
| sam_model | [25, 12, 196, 64] | [25, 12, 196, 196] | EFFICIENT_ATTENTION |
| sam_model | [1, 12, 4096, 64] | [1, 12, 4096, 4096] | EFFICIENT_ATTENTION |
| vision_model | [1, 16, 257, 64] | None | FLASH_ATTENTION |

First occurrences of each layout in each process receive a full-output comparison with MATH and independent CPU FP32 checks on first/middle/last batch/head slices. Maximum absolute differences: 0.015625000 versus MATH and 0.007917643 versus CPU FP32; all pass the unchanged 0.03/0.03 BF16 tolerance. Subsequent calls of the same signature retain operator, HIP and downstream checks. Every HIP checkpoint is zero and downstream copy/arithmetic is finite. This is representative-layout numerical validation, not an independent CPU reference for every layer/input.

[Complete single attention trace](../../eval/native-ocr/windows-gfx1151/stage3-auto-single/attention.json) · [complete multi trace](../../eval/native-ocr/windows-gfx1151/stage3-auto-multi/attention.json). The original Stage 3 producer bytes are preserved separately as `validate-stage3.py.txt`. Independent review subsequently moved the already-applied fused gate into the tracked harness and added a separate native-inference timer; final benchmark producer bytes/hashes are retained.

## Stage 4 controlled A/B

The same candidate environment, physical device/driver, model revision/hash, BF16, input hashes and 1024-token budget per page are used for A=MATH and B=AUTO. AUTO diagnostics above establish actual fused execution. One excluded warmup plus five warm samples per case, sequential MATH then AUTO; no concurrent GPU workload was launched by this mission. Both use identical synchronization and instrumentation. Profilers/numerical-reference tracing are disabled for measurements.

Native OCR times include the complete native infer/infer_multi call through final GPU synchronization, excluding wrapper output writing/normalization/assessment. The separate wrapper wall field retains those costs. Vision time sums synchronized SAM and CLIP forward calls across the document; preprocessing outside those calls remains in native end-to-end time.

### Native end-to-end OCR

| Case | MATH median / max s | AUTO median / max s | Median speedup |
|---|---:|---:|---:|
| chinese | 5.492 / 5.504 | 4.811 / 4.816 | 1.142x |
| english | 5.641 / 5.837 | 4.711 / 4.782 | 1.198x |
| math | 4.781 / 4.819 | 4.298 / 4.379 | 1.112x |
| mixed | 5.430 / 5.460 | 4.900 / 4.924 | 1.108x |
| pages_2 | 6.760 / 6.779 | 6.174 / 6.235 | 1.095x |
| pages_3 | 10.075 / 10.355 | 9.257 / 9.475 | 1.088x |
| pages_5 | 17.695 / 17.814 | 16.518 / 16.616 | 1.071x |
| repeat_unicode_recovery | 5.209 / 5.281 | 4.735 / 4.756 | 1.100x |
| table | 4.220 / 4.286 | 3.718 / 3.788 | 1.135x |
| table_2_pages | 5.636 / 5.675 | 5.139 / 5.145 | 1.097x |

### SAM + CLIP vision encoders

| Case | MATH median / max s | AUTO median / max s | Median speedup |
|---|---:|---:|---:|
| chinese | 0.992 / 1.006 | 0.450 / 0.457 | 2.205x |
| english | 1.003 / 1.325 | 0.453 / 0.553 | 2.214x |
| math | 0.983 / 0.990 | 0.458 / 0.465 | 2.145x |
| mixed | 0.986 / 0.988 | 0.454 / 0.459 | 2.170x |
| pages_2 | 0.923 / 0.924 | 0.380 / 0.389 | 2.429x |
| pages_3 | 1.364 / 1.371 | 0.560 / 0.574 | 2.435x |
| pages_5 | 2.248 / 2.264 | 0.927 / 0.941 | 2.424x |
| repeat_unicode_recovery | 0.983 / 0.989 | 0.452 / 0.455 | 2.177x |
| table | 0.988 / 0.992 | 0.457 / 0.466 | 2.164x |
| table_2_pages | 0.912 / 0.935 | 0.379 / 0.388 | 2.405x |

All 100 measured requests and 20 warmups pass quality; maximum per-page CER is 0 and every request records 12 native DASH calls. Output token counts match between modes for all cases: **True**. Per-case input hashes, token budgets/counts, all samples, SAM/CLIP components and separate memory readings are in the machine-readable result. Maxima describe only these five samples; no reliable p95 or statistical significance is claimed.

### Separate model loading

| Mode | Native load segment s | Load incl. imports/hash checks s |
|---|---:|---:|
| math | 9.415 | 19.739 |
| auto | 9.168 | 19.513 |

Model load is a fresh-process load, with already-populated disk/model/compiler caches; it is not an uncached storage benchmark. The outer loader timer includes model source/weight checks and model dependency imports, but excludes initial candidate-wheel identity verification, interpreter startup and the initial torch import. Each case warmup is separately retained.

### Memory observations

| Measurement (GiB) | MATH | AUTO |
|---|---:|---:|
| Maximum sampled process RSS | 10.969 | 7.001 |
| Maximum process peak RSS | 10.996 | 8.821 |
| Maximum request peak GPU allocated | 8.589 | 6.915 |
| Maximum request peak GPU reserved | 10.510 | 7.656 |
| Minimum sampled available system RAM | 79.579 | 83.907 |

This APU shares system RAM. GPU allocation/reservation, process RSS and system memory are different overlapping observations, not independent physical pools; do not sum them or label the GPU budget dedicated VRAM. System available memory is sampled, not continuously measured peak system consumption.

[Full comparison and samples](../../eval/native-ocr/windows-gfx1151/result.json) · [measurement process commands](../../eval/native-ocr/windows-gfx1151/benchmark-processes.json) · [MATH raw samples](../../eval/native-ocr/windows-gfx1151/benchmark-math/) · [AUTO raw samples](../../eval/native-ocr/windows-gfx1151/benchmark-auto/).

## Regressions, limitations and recommendation

**FAIL:** no OCR or required fused regression in the executed scope. The retained first dependency installation failure is preparation-only. **UNSUPPORTED:** the explicit additive-mask forced-FLASH probe; AUTO successfully uses EFFICIENT for actual SAM masks. **NOT RUN:** other hardware/OS, other dtypes, backward, broad real-document corpus, serving/concurrency and newly published AMD wheels.

Serial MATH-then-AUTO measurements can retain thermal/order effects; five samples do not establish a tail distribution or statistical confidence. The numerical reference covers representative actual layouts; OCR acceptance is synthetic and cannot certify arbitrary document fidelity. No fixing source commit is attributed, and no upstream/official AMD release claim is made.

**Recommendation: retain MATH as the default.** The measured local candidate supports opt-in fused vision evaluation within this scope. Default enablement should wait for a distributable, separately verified wheel plus broader documents, repeated counterbalanced measurements and additional device/driver validation. No default, decoder algorithm, main branch, release or stable tag was changed; no binary wheel is published.

## Verification and evidence integrity

Local CPU checks: Ruff, formatting, mypy, package build and pip check PASS; **66 tests PASS, two opt-in GPU pytest markers skipped**. Standalone physical GPU validation above ran separately. [Independent implementation and evidence review](NATIVE_OCR_REVIEW.md) found no remaining Important finding after the fused-gate and timing improvements.

The branch starts at current main `966265c81a74a73aa0ae6d9a3fa593b342590eb7`. Runtime/measurement source is identified by the committed `0fa486e` content and exact source hashes; some collection began before that commit existed. Original Stage 2/3 producer bytes are retained alongside current measurement bytes, rather than rewriting historical source identities.

[Source hashes](../../eval/native-ocr/windows-gfx1151/measurement-source-hashes.json) · [original/published digest mapping](../../eval/native-ocr/windows-gfx1151/export-manifest.json) · [SHA256SUMS](../../eval/native-ocr/windows-gfx1151/SHA256SUMS.txt). Private originals, fresh environment/cache and every historical environment/log remain under the repository; only user-profile prefixes are redacted in published evidence. No weights or unnecessary binaries are included.
