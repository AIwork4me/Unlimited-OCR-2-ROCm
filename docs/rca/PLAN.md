# Windows gfx1151 SDPA investigation plan

Goal: publish independently reproducible evidence on the exact mandated Windows baseline; preserve all production code and official weights.

Spec: user-supplied CODEX MISSION, 2026-10-28. Execution is explicitly autonomous; implement inline and request one independent final review.

Constraints: branch investigation/windows-gfx1151-sdpa-root-cause from origin/main 966265c; no runtime/driver changes; fresh child process per GPU condition; no error clearing in acceptance; exact CPU fixtures and predeclared tolerances; no L4 claim without a causal experiment; publish Issue and PR, never merge.

- [ ] Task 1: TDD standalone tooling. Add tests/unit/test_sdpa_rca_reporting.py for nonfinite/mismatch classification, observed backend vs requested flags, redaction, and independent CPU reference semantics. Watch RED, implement reporting.py and repro_sdpa.py, watch GREEN. Fixture: CPU generator seed 7992, quantize before reference; BF16 atol/rtol .03/.03, FP16 .005/.005, FP32 .0001/.0001. Record these before the first acceptance run. Capture operations with TorchDispatchMode; optional CPU profiler as a separate condition.
- [ ] Task 2: Isolate/minimize. compare_backends.py and run_matrix.ps1 launch one fresh process per condition, retain stdout/stderr/exit status and producer hashes. Start 54x12x196x64 additive BF16, force each backend, repeat; reduce dimensions one at a time. Then mask/dtype/stride/causal and operation-order matrix. Separate explicitly non-acceptance error-clearing runs may inspect numerical corruption only.
- [ ] Task 3: Provenance and source. collect_provenance.py records exact distributions, wheel origin hashes, actual loaded DLL paths/hashes, AOTriton images and build metadata. Read full upstream issue/comments and inspect immutable torch/AOTriton/TheRock build references. Run targeted A/B only when one changed variable can distinguish hypotheses.
- [ ] Task 4: Real-model cross-check. Run existing hash-verifying SAM probe both default and math; one accepted OCR case with unchanged production adapter, BF16 and native DASH guards. No full 26-request suite.
- [ ] Task 5: Reports and delivery. Write SDPA_ROOT_CAUSE.md (22 required sections), HYPOTHESES.md (H1-H8), TEST_MATRIX.md and English UPSTREAM_HANDOFF.md. Sanitize producer artifacts, manifest SHA256 and preserve original logs privately. Add README link and CI coverage. Independent review of math/dispatch/claims/authenticity; fix substantive findings with regressions; run CPU checks and build. Commit, push, create linked repository Issue/PR, attach PR, verify remote blobs and checksums.

Review focus: boolean-mask polarity and causal masking; reference on quantized exact inputs; unsupported backend vs actual fault; sticky HIP error never silently cleared; backend observations never inferred from enabled flags; no sanitized artifact loses its producer identity.

Interfaces: repro consumes reporting classify/observed_backend/sanitize; runner consumes one JSON report per fresh process; documentation consumes immutable run IDs and checksums. No production runtime changes are planned.

Ruling: use the existing isolated validated .venv read-only for baseline experiments and .venv-cpu read-only for CPU checks; no package installation or replacement is needed. Standalone imports will explicitly record whether Transformers/HF/Triton were loaded.
