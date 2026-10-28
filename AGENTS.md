# Contributor and coding-agent rules

- Correctness before performance; evidence before claims.
- Reproduce, isolate and identify the root cause before a minimal fix.
- Preserve torch 2.12.0+rocm7.14.1, torchvision 0.27.0+rocm7.14.1,
  torchaudio 2.11.0+rocm7.14.1 and Windows gfx1151 identity. No version drift.
- No silent CPU fallback, model substitution, fabricated GPU results or
  disabling DASH/SR-SWA to pass a test.
- Review immutable official model source before trust_remote_code. Verify
  code/config/tokenizer/weight hashes; do not execute mutable remote revisions.
- Record raw outputs, ground truth, thresholds, memory and complete failures.
- Linux, W7900/gfx1100, gfx1200 and all other hardware remain unvalidated
  until physically tested. torch.cuda is the ROCm-compatible API name.
- APU GPU allocation is not dedicated VRAM or total system memory consumption.
- No global driver/SDK/security/PATH/registry changes. Ask explicitly before
  changing system security settings or uninstalling drivers/SDKs.
- No secrets, user-directory paths, weights or unnecessary binary artifacts in Git.
- Respect source/model licenses; retain notices for any copied or patched code.
- No unrelated refactoring; keep PRs small and reviewable.
- CPU CI needs no torch/model download. Real GPU tests are opt-in on the
  physical Windows device; preserve sanitized evidence and mark unrun checks.
- Use PowerShell, pathlib, UTF-8 and subprocess argument lists on Windows.
- Do not modify the prior V1 project. Do not publish PyPI/releases, mark a
  stable tag, merge the feature PR or start Phase 2 during this mission.

Local CPU checks:

```powershell
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest
python -m mypy src
python -m build
python -m pip check
```
