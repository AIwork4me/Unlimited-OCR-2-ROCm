# Engineering log

## 2026-10-28 — discovery and decisions

- Windows 11 Home Chinese, version 10.0.26200, build 26200.
- CPU AMD RYZEN AI MAX+ 395 w/ Radeon 8060S.
- AMD Radeon(TM) 8060S Graphics driver 32.0.22032.6002.
- OS usable RAM 129654596 KiB; free at discovery 97204120 KiB.
- WMI AdapterRAM is a truncated display value, not dedicated physical VRAM.
- Python launcher absent; base conda Python is 3.13.13. Existing uv-managed
  CPython 3.10 is available. AMD index contains cp310 Windows wheels for the
  exact mandated torch version. Keep preferred Python 3.10.
- Repository did not exist (connector and gh returned 404); created publicly
  with exact description using gh because connector has no create-repo tool.
- Initial commit required repository-local Git identity; use authenticated
  GitHub login and GitHub noreply address. No global Git changes.
- Native Codex worktree created; feature branch is
  feat/phase1-windows-gfx1151-rocm7141.
- HF source downloaded as inert files at immutable model revision. No remote
  code has been executed. Audit runs independently of environment installation.
- Triton candidate 3.7.1.post27 has a cp310-win_amd64 wheel, Python >=3.10,
  no default torch dependency, PyPI project homepage references original
  maintainer repository. Review release and backend before install.
- Ruling: execute the supplied detailed design autonomously without another
  approval checkpoint, as explicitly requested. Use native Windows commands
  instead of the skills' Unix-only bookkeeping helpers; keep this durable log.

## Gate A probes

- Exact mandatory install succeeded under CPython 3.10.20.
- The mandated wheel reports torch.version.rocm=7.14.1 and
  torch.version.hip=7.14.60850. Initial doctor incorrectly assumed both strings
  equal the package release. Inspected torch/version.py and SDK distribution
  identities; fixed strict checks to use both actual fields. Regression RED/GREEN.
- GPU reports physical gfx1151 and PyTorch-visible shared-memory budget
  83954728960 bytes. This is not dedicated VRAM.
- First zero-tolerance BF16 GEMM check failed at cancellation-to-zero entries:
  maximum absolute residue 4.76837158203125e-7. Isolated CPU integer and FP32
  references both gave 0; GPU BF16 gave that small residue. Set documented
  absolute tolerance 1e-5 (rtol 0), far below BF16 spacing at nonzero results.
  This is a numerical-probe correction, not an OCR acceptance threshold change.
  The original failure manifest is preserved. OCR CER remains <=0.02.

## Native kernel and package verification

- Gate A PASS: synchronized real BF16 GPU GEMM, exact packages, native Windows
  and gfx1151 identity. pip check passes.
- Triton-0/1/2 PASS: 4097-element vector-add compiled for HIP/gfx1151;
  exact CPU and PyTorch reference agreement. No AMD Triton distribution was
  present before installation; installed hash-pinned post27 with --no-deps.
- Triton-3 PASS: unchanged official DASH e2e prefill kernel, B=1/H=10/D=128,
  lengths 160/173, local window 128, block32, bias/duplicates/tail exercised.
  Fixed pre-launch tolerance 0.03; max error 0.0086942 / 0.0084605.
- Native source audit completed independently; no model/kernel source edits.
  Use supported mha_eager dispatch. Never invoke unsafe upstream eval-based
  generated-coordinate visualization. Native decode retains official behavior.
- Ruling: build CPU-tested wrapper and scripts while weight downloads run,
  instead of idling. This changes task scheduling only, not inference gates.
- Core CPU tests: 33 passed, 2 physical tests explicitly skipped in CPU run.
  Ruff check/format and mypy pass. Wheel and sdist build; editable CLI
  installation and --version pass. Model execution remains NOT TESTED here.
- Native dependency install preserves torch stack; Transformers4.52.1,
  Hub0.36.0, tokenizers0.21.4 import successfully; pip check passes.
- Separate clean environment setup is running; no Gate E claim yet.
- Large artifact transfer was slow. pip Triton took 11m22s (~65kB/s).
  Independent parallel-range download matched its SHA256. The official Xet
  client in a separate downloader venv cached the smaller model shard;
  larger shard still pending at this entry. No global Python modified.

## First native inference, compatibility RCA and review fixes

- Both official weight shards downloaded and match immutable LFS SHA-256.
  Unmodified native infer loads BF16 parameters on cuda:0, then fails in SAM
  window_unpartition with hipErrorInvalidValue. Full failure is preserved.
- Reduced to two forwards of one random-weight SAM block; plain layout-copy
  loop passes. HIP peek instrumentation finds SDPA leaves error1 before the
  later copy. A default-SDPA output also fails an independent FP32 reference.
- Fresh GPU MATH SDPA leaves HIP error0 and matches the same reference
  (max error 0.00387, predeclared 0.03 tolerance). Three SAM block forwards pass.
  TheRock issue7992 describes the same exact-stack pattern on another GPU;
  that report is context, not our hardware evidence.
- Minimal runtime compatibility: scope public sdpa_kernel(MATH) to each SAM
  and CLIP forward. Restore settings immediately, preserve all official
  source/weights and decoder DASH/Triton. No CPU path or error clearing.
- First adapted English inference finishes with twelve native e2e calls and
  correct text; initial validator incorrectly included V2 block metadata.
  Audited pinned GitHub utils/process_unlimitedOCR2.py in full, vendored it
  unchanged with Baidu MIT license, and use its block parser/convert semantics.
  OCR thresholds unchanged. The original validation FAIL remains preserved.
- English/Chinese/mixed/table plus repeated Unicode/recovery run all PASS,
  CER 0.0 and exact important numbers; twelve native decoder calls per run.
- Independent code review found six Important issues: table numeric cell
  boundaries, Gate E baseline proof, inherited routing overrides, raw newline
  bytes, telemetry masking errors, and Windows11/CPU identity enforcement.
  Each has a regression; no Critical or Minor findings.
- Ruling: preserve native generation's single empty trailing PAGE as an EOF
  terminator, not a content page. The 2-page raw output has correct ordered
  content (CER0 each) and three markers because _uocr_page_count_guard forces
  EOS only after marker_cum>expected_pages. Official _parse_blocks skips an
  empty final segment. Structural policy therefore counts verified content
  pages and separately reports raw markers/terminal_empty_marker; only one
  empty trailing marker is allowed. Middle/extra empty pages still fail.
  This documented structural-interpretation correction follows actual V2
  semantics; the prior strict-marker FAIL is retained. Text/numeric thresholds
  are not relaxed. Cost if wrong: an application requiring exactly N raw
  delimiters must inspect raw_marker_count_matches_input rather than the
  content-page acceptance status.

## Review follow-up and final-run scope

- Follow-up review covered substantive compatibility/postprocessor/repro code
  added after the initial review. It found two additional Important issues.
- Fixed conversion expansion via bounded heading metadata before invoking the
  byte-identical official converter. Raw survives rejection; regression passes.
- Fixed Gate E provenance: identity/inventory come from producing doctor/load/
  case reports, not the summarizer interpreter. All producing IDs must agree.
  A fresh workflow requires a matching run-ID/environment-ID creation witness;
  --fresh cannot override recorded clean_environment=false. Regression passes.
- Earlier exploratory baseline/clean summaries predate these proof corrections
  and are excluded from published acceptance. Their local files remain.
- Final acceptance uses final-baseline and final-clean only. A new isolated
  environment is being created for final-clean after these corrections.

- The final-clean attempt failed during AMD wheel extraction at a260-character
  Tensile file path. The same entry exists at251/257 characters in working
  venvs. Added early setup path-boundary detection and selected a shorter
  isolated path, including spaces/Chinese for direct path-handling validation.
  No long-path registry/security/driver setting changed. The failing install
  and error metadata remain diagnostic evidence; no inference success is
  attributed to that failed environment.
