# Source-to-Evidence Matrix — Windows Delivery (Phase 0.2 + 0.3)

Everything below cites retained evidence; nothing was re-executed on the GPU.

## 0.2 Offline reproducibility review — PASS (with disclosed limits)

| Area | Status | Key evidence |
| --- | --- | --- |
| Fresh-install receipts (2) | PASS | `.b/w1/phase1-install/*.json`, `.b/w2/…`, logs `install-w1/w2.log`; every pip stage `--no-index --find-links .b/wh`; both end `pip check` clean |
| Installed-member verification | PASS | `installed-members.json`: 9 wheels, **13,343 members per environment** re-hashed from archive bytes vs installed files (RECORD-only exclusion); zero mismatches; generator `source_commit = 3556e38…` |
| Offline wheelhouse | PASS | `.b/wh` — **43 artifacts = 9 AMD archives + 34 transitive/installer artifacts** (pip/setuptools/wheel/packaging, rocm sdist + rocm_bootstrap from `repo.amd.com`, triton_windows 3.7.1.post27, numpy/transformers 4.52.1/tokenizers/… stack); dry-run resolution proof `wheelhouse-resolution.json` (42 distributions, all `file:///…/.b/wh/…`); public manifest `eval/windows/wheelhouse-SHA256SUMS.json` (43 hashes) |
| Nine AMD archives alone | — | Explicitly **insufficient** for offline install: `rocm`/`rocm-bootstrap` do not resolve from PyPI; docs and installer warn |
| Model identity | PASS | revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, 18 pinned files SHA-256-verified pre-load and re-bound to actually-loaded per-env copies (`golden-original-v2/w{1,2}/remote-source.json`) |
| Golden suite receipts | PASS | `golden-original-v2/` + `eval/windows/golden-validation.json`: **26/26 native GPU PASS** (13 cases × w1/w2), 13/13 cross-env exact token/bbox/prompt parity, 6/6 cold-warm pairs, all `native_finish_reason=stop`, CER=0, `dash_dispatch_count=12`; V1 failure sealed separately |
| Source archive validation | PASS | `archive-check.json` + `package-final-r3/receipt.json` (Phase 0.2 state, six-commit head): 34 changed files Git-byte-exact in sdist; extracted-tree tests **38 passed**; publication scan of full archive exit 0. Re-validated at the final eight-commit head by `package-final-r4/receipt.json` (36 files byte-exact; 38 extracted tests; publication PASS) — see EXECUTION_STATUS.md §Phase 2.6 |
| PowerShell entry-point tests | PASS | `test_windows_benchmark_powershell.py` (4, real `powershell.exe -File`), `test_windows_install_preflight.py` (15); RED→GREEN logs retained |

**Reproducibility boundary (do not overclaim):** hashes are observed receipts, not
publisher signatures; the wheelhouse/model bytes are repo-local (public artifact is
the hash manifest, not the bytes); fresh-host acquisition is documented but was not
independently tested on a second machine; w2 reused w1's pip-built `rocm` meta wheel
(sdist itself hash-pinned `ffea97f6…`).

## 0.3 Source-to-evidence binding — PASS (binding is explicit, not inherited)

| Commit | Classification | Evidence executed at this identity |
| --- | --- | --- |
| `4abba4f` | Executable inference/runtime (runner Job supervision, publication whitelist) | included in CPU suite at later heads |
| `c431027` | Installation/runtime (offline install preflight, lock, setup scripts) | both fresh offline installs (`source_commit` receipts) |
| `3556e38` | Runtime entry (PowerShell entry, frozen annotations, `run_mpdoc_sample.py`) | **all physical GPU Golden evidence** (`physical_git_sha` in `golden-validation.json`; envs installed from dedicated clone `.worktrees/wi` at this SHA) |
| `5436f80` | Runtime behavior (HF cache containment in ps1 + regression test) | **full CPU suite 389 passed / 8 skipped** (`cpu_integration_git_sha`) |
| `d2caa79` | Documentation + `eval/windows/*.json` evidence only | source-freeze packaging receipt |
| `0ba45dd` | Documentation only (2 Markdown files) | final delivery binding `DELIVERY_VERIFICATION.json` |

Decisive fact verified fresh for this mission: `git diff 3556e38..0ba45dd -- src/`
is **empty** — the entire Python package is byte-identical between the physical-GPU
commit and the delivered head. The only non-docs deltas after `3556e38` are
`scripts/run_windows_benchmark.ps1` (+3/−1) and its test. Physical GPU evidence
therefore covers the exact Python inference code in the delivered head; the
PowerShell entry differs by the reviewed cache-containment fix, which is itself
covered by the CPU suite at `5436f80`.

No later commit claims unexecuted physical evidence: `DELIVERY_VERIFICATION.json`
records `status: PASS_AUTHORIZED_SCOPE_ONLY`, `full_product_acceptance: NOT_ACHIEVED`,
all benchmark metrics null.
