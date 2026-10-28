# Phase 1 evidence report

## A. Status

**A–E physical acceptance: PASS.** All 13 required case executions pass in
each accepted environment, 26 requests total, with normalized CER 0 and exact
important numbers. Repository delivery is tracked in PR #1; its current checks
and readiness are authoritative for the final delivered commit.

The accepted path includes one disclosed compatibility adapter: GPU MATH SDPA
scoped to SAM/CLIP forward only. It retains BF16 tensors, trained attention,
official model source/weights, decoder DASH/Triton and native generation.

## B. GitHub

- Repository: [AIwork4me/Unlimited-OCR-2-ROCm](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm).
- Exact description verified through GitHub API.
- Branch: feat/phase1-windows-gfx1151-rocm7141
- PR: [#1](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/1).
- Runtime source commit: 5a670e93acb7ac6e4c4ff05fd8f355ac5a6befd5.
- The report/evidence delivery commit is the PR head, also reported in the
  final delivery message; a report cannot embed its own Git commit hash.
- Feature PR remains open; no merge, release, stable tag or PyPI publication.

## C. Actual environment

| Field | Observation |
|---|---|
| Windows | Windows 11 Home Chinese, 10.0.26200/build 26200 |
| CPU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |
| GPU/architecture | AMD Radeon(TM) 8060S Graphics / gfx1151 |
| Driver | 32.0.22032.6002 |
| Installed physical RAM | 137438953472 bytes, 128 GiB |
| OS-managed RAM | 132766306304 bytes, 123.65 GiB |
| PyTorch-visible shared GPU budget | 83954728960 bytes, 78.19 GiB |
| Python | 3.10.20 x64 |
| torch/vision/audio | 2.12.0/0.27.0/2.11.0, each +rocm7.14.1 |
| ROCm/HIP | 7.14.1 / 7.14.60850 |
| Triton | triton-windows 3.7.1.post27, runtime 3.7.1, HIP/gfx1151 |
| Transformers/Hub/tokenizers | 4.52.1 / 0.36.0 / 0.21.4 |

The GPU budget is not dedicated VRAM. GPU allocated/reserved memory and
process/system RAM are reported separately. Actual free RAM at inference
start is stored in every case manifest. No WSL/Docker/VM or alternate device.

## D–E. Inference and correctness

Official weight load, native single-page, Chinese, two-page and five-page:
PASS. Three pages, mixed text, HTML table, simple arithmetic and two-page
table: PASS. Repeated single/multi and Unicode/space paths: PASS.
See VALIDATION.md for the 13-row measured table and direct manifest links.

English ground truth and actual recognized content:

~~~
ROCm OCR Test
Hello from Windows.
Invoice 2026
Total 123.45
~~~

Chinese ground truth and actual recognized content:

~~~
中文识别测试
人工智能读取文档
订单编号 2026
合计 123.45 元
~~~

Five-page order is Atlas/101, Boreal/102, Cedar/103, Delta/104, Ember/105.
All page comparisons have CER 0. Raw V2 BLOCK/FORK/PAGE output, readable
official Markdown, normalized text and hashes remain separate.

One empty terminal PAGE is reported as a delimiter, not hidden as an
unobserved page. Raw marker counts can exceed input pages by one; strict
consumers must inspect raw_marker_count_matches_input. Text/numeric
thresholds stayed unchanged; the documented structural interpretation follows
the native generation guard and official V2 parser.

## F. Runtime and memory

These are synchronized smoke diagnostics, not throughput/speedup claims.

| Measurement | Fresh accepted environment |
|---|---:|
| Native model load segment | 10.882s |
| First English request in process | 25.477s |
| Repeated English request | 6.242s |
| Five-page request | 21.246s |
| Single-page peak GPU allocated/reserved | 8.57/8.91 GiB |
| Single-page process RSS/peak RSS | 9.30/9.32 GiB |
| Available system RAM before English | 73.64 GiB |
| Five-page peak GPU allocated/reserved | 8.59/8.80 GiB |

Every accepted request records twelve successful synchronized official e2e
DASH prefill calls and BF16 CUDA-device query tensors. SAM/CLIP inputs are
recorded on the GPU. All model parameters are BF16 on cuda:0. Dense prefill
fallback raises. Decoder computation remains the official GPU path.

## G. Reproducibility

- [Accepted baseline manifest](../eval/results/final-baseline/reproduction.json).
- [New isolated environment manifest](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/reproduction.json).
- [Run-bound creation witness](../eval/results/Windows%20%E4%B8%AD%E6%96%87%20verified/environment-creation.json).
- Model/code revision: 9cf077c4b206e0e2a31879df937b0b0bcba3661c.
- GitHub revision: 4d0bebab8714e0f3b0c811cd44bf422ddb49daae.
- Complete source/tokenizer/config/weight hashes: package upstream-lock.json.
- Runtime dependencies: requirements/windows-cp310.lock.txt.
- Actual initial artifact origins/hashes: eval/results/checks/package-provenance.json.
- Exact command argument arrays, driver/build, environment controls, outputs,
  errors and package inventory are in the manifests/logs.

Producer IDs differ, machine identity matches, 44 artifact
hashes and 42 locked package versions match, and all required
cases pass. The fresh venv/output paths contain spaces and Chinese characters.
The baseline also exercised an offline cached-model workflow. This is the
same physical machine/checkout, not an independent second machine.

Use the two-run command in REPRODUCIBILITY.md, passing the first manifest as
BaselinePath. A single first run or re-summarization cannot establish Gate E.

## H. Root cause and minimal changes

1. Default vision SDPA leaves HIP error 1, fails numeric reference and causes a
   later layout-copy failure. Minimal SAM/SDPA reproduction is preserved.
   Scoped GPU MATH SDPA passes reference and repeated SAM block tests, restores
   attention flags, and enables real OCR without changing decoder algorithms.
2. HIP compiler build 7.14.60850 differs from ROCm package release 7.14.1;
   doctor checks both exact fields. Initial wrong assumption is preserved.
3. BF16 cancellation residue 4.77e-7 justified a documented 1e-5 diagnostic
   bound. OCR CER 0.02/number thresholds were not relaxed.
4. V2 BLOCK/FORK metadata requires the official GitHub V2 parser, distinct
   from the unsafe HF legacy drawing helper. Converter is vendored unchanged,
   licensed, and bounded against generated heading expansion.
5. Windows long-path extraction failed at 260 characters; shorter paths 251/257
   work. Setup preflights the known boundary; no registry change.
6. Code-review fixes bind provenance to producing runs, preserve raw UTF-8
   bytes and original tracebacks, reject routing overrides and enforce OS/CPU.

See ROCM_TRITON_RCA.md, CODE_REVIEW.md and ENGINEERING_LOG.md. A concrete
upstream proposal is prepared in UPSTREAM_PROPOSAL.md; none was posted.

## I. Engineering and limitations

- 56 CPU tests and 2 actual GPU tests pass together: 58 passed, 15 upstream
  TorchScript deprecation warnings. No mocked inference is GPU evidence.
- Ruff lint/format, maintained-module type checks, wheel/sdist build, installed
  CLI smoke and pip check pass. CPU CI Windows/Ubuntu passed at 3a10345;
  final-head status is visible on PR #1.
- Eight Important review findings fixed with RED/GREEN regressions. No Critical
  or deferred Minor findings. Review scope/revisions are in CODE_REVIEW.md.
- Home paths are redacted in exported logs/manifests; raw synthetic OCR bytes
  are unchanged. Model weights, credentials and unnecessary binaries are not
  committed. Actual local original logs remain outside Git.
- Only this Windows/gfx1151 system is validated. Unused alternate kernels,
  training/backward, other OS/GPUs, complex math, broad benchmark quality and
  production long-document performance are NOT TESTED.

## J. Next steps and rulings

Review PR #1; Phase 1 has no remaining physical gate blocker. Do not start
Phase 2, merge automatically, publish PyPI/releases or tag a stable version.

Rulings made:

- Execute the supplied detailed mission autonomously, using native Windows
  bookkeeping. Cost: durable progress is in the engineering log rather than
  Unix-only skill helper output.
- Build CPU-tested tooling while weights download. Cost: scheduling differed;
  acceptance still waited for real model output.
- Use the measured, scoped vision backend compatibility adapter. Cost: extra
  memory/latency versus an eventually fixed fused backend; no speedup claim.
- Count one empty terminal PAGE separately. Cost: an exact-N raw-delimiter
  consumer must inspect the explicit raw-count flag.

Deferred minors: none. No unsupported platform, independent machine or
unmodified default-SDPA success is implied by this report.
