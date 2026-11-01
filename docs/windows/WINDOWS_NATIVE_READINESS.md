# Windows Native readiness — 2026-11-01

Status: **PARTIAL PRODUCT DELIVERY; FULL BENCHMARK BLOCKED**. Stable route is original official AMD torch + Native Transformers BF16, scoped GPU MATH for vision, native DASH/Triton and unchanged weights. No AUTO/fused candidate promotion, CPU fallback or model substitution.

| Gate | Evidence and present limit |
| --- | --- |
| Historical physical Windows inference | A–E PASS on the documented Windows 11 / Ryzen AI MAX+ 395 / Radeon 8060S gfx1151 device, synthetic single/2/3/5-page scope; see [historical validation](../VALIDATION.md). This is not a fresh-commit regression result. |
| Fresh installation | Both new `.b/w1` and `.b/w2` offline package installs and pip checks completed; doctor/GPU checks explicitly skipped. Same source clone `.worktrees/wi` at `3556e3867a086437a43a645fa907689f4dd25260`. |
| New Golden regression | PASS for two fresh environments, 13 cases each /26 original infer entries, independently reread from raw evidence. CONTENT/STRUCTURE, exact cross-env token/bbox and six cold/warm pairs all PASS in this fixed sample. The preserved first policy failed before infer0. See [baseline](NATIVE_BASELINE_VALIDATION.md) and [summary](../../eval/windows/golden-validation.json). Historical failures remain unchanged. |
| CPU code verification | At code commit `5436f80233477cf53f0e5eda4406c27a9e09c035`: 389 passed / 8 skipped; Ruff, format, Windows/Linux-target mypy, build, pip check and source-archive checks PASS. Linux-target mypy is static checking, not Linux execution. Private checks-r2 and checks-r3 logs retained; the latter includes the inherited Hugging Face cache regression. |
| Dataset bytes / order | Windows CPU verification PASS, MPDocBench 420/3135, Marathon 100/18272; [receipt](../../eval/windows/dataset-readiness.json). |
| Native Windows eight-metric scorer | BLOCKED: metadata matches 75 locked requirements in torch-free environment; XeLaTeX, Ghostscript CLI and latexmlc absent; POSIX assumptions remain. |
| Full official GPU benchmark | NOT AUTHORIZED / NOT RUN. Counts and all null metrics in [status receipt](../../eval/windows/benchmark-status.json). |

The initial main CPU baseline (349 PASS, 8 skipped, one publication ancestor failure) and subsequent failures are retained. The publication scanner now inspects its selected root relative to that root; the reviewed source-fixture whitelist change is intentional. No historical result was overwritten.

Windows supervision now uses suspended worker creation, Job assignment before resume, KILL_ON_JOB_CLOSE, process-tree drain/accounting and strict unreadable/nonzero outcome rejection. Three independently reported P2 defects were corrected and re-reviewed. **Residual gap:** hard supervisor death between process creation and Job assignment can leave an unassigned suspended child; arbitrary interruption inside Popen before the reference is returned is also outside the guarantee. CPU fake-process tests do not prove GPU-driver crash recovery.

LongPathsEnabled remains 0. The longest required original AMD archive member under `.b/w1` or `.b/w2` is 219 characters. A retained test failure at 262 characters demonstrates that arbitrarily deep output paths remain unsupported. Use short contained paths and preserve legacy compatibility junctions; no registry/security/driver/SDK/global PATH changes.

See [installation](WINDOWS_INSTALL_REPRODUCTION.md), [protocol](WINDOWS_BENCHMARK_PROTOCOL.md), [handoff](WINDOWS_HANDOFF.md). Dataset readiness and clean package installation do not establish OCR correctness or official benchmark scores.
