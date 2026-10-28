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
