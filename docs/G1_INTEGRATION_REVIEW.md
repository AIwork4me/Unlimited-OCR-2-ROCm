# G1 integration and artifact identity review

Date: 2026-10-29. Audited source head:
`aff12108087866a5a9cd5fafc863a18fb42ef182`.
This is a read-only source, publication and environment identity review.
The reviewer did not import torch, execute a model/GPU workload, change an
environment, commit, switch branches or mutate a pull request. Physical runs
and their acceptance decision belong to the separate G1 physical report.

## Pull request stack

| PR | Head | Base | Observed status |
| --- | --- | --- | --- |
| [#5](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/5) | `cfd684224fd4b306cdc22837b4abcf9c432a1c35` | `main`, `966265c81a74a73aa0ae6d9a3fa593b342590eb7` | Open draft; conflicting; preserved unchanged. |
| [#7](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/7) | `d842ce6f5cf187d17f4675e5ac37c188fc08cf9a` | `main`, `beb38d1941a0ae85a1c1218209c822c3e2f115e3` | Open, clean. |
| [#8](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/8) | `afbafac9eb19e1dc11fea5278a8a3ef7b6134457` | PR #7 head | Open draft, clean. |
| [#9](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/9) | `aff12108087866a5a9cd5fafc863a18fb42ef182` | PR #8 head | Open draft, clean. |

Both ancestry edges and exact base/head identities match. GitHub reports all
listed Windows/Ubuntu CPU checks successful; PRs #7–#9 additionally have
successful Windows Qwen and Ubuntu Marathon scorer checks. No submitted GitHub
reviews or review decision were recorded. These are CPU checks, not physical
GPU validation. Local snapshots retain check URLs and timestamps.

PR #8 is source/CPU-review ready for its narrow completion/failure-evidence
change. Its existing independent review reports zero unresolved P1/P2 findings;
this review confirms its stack and inherited producer bindings. The separate
[physical report](G1_PHYSICAL_WINDOWS_REGRESSION.md) now records PASS for all
13 original-wheel MATH before/after pairs at 1/2/3/5 pages, including native
raw/ID equality and disclosed failure controls. Native stop cannot establish semantic
completeness, and the five-page, 8192-new-token and 16-million-pixel limits remain.

PR #9 is source/scorer/dataset-byte-identity ready for its stated closure scope,
provided the PR #7 → #8 → #9 stack is preserved. Its claims do not establish
full-dataset inference quality or Windows/Linux GPU support. Neither draft is
approved for merge by this review; no merge was performed.

## Published evidence and source bindings

- All **336** entries in `eval/benchmarks/g05/SHA256SUMS.json` match the
  published Git blob bytes at the audited head.
- All **309** scorer `TEXT_COPY` artifacts match their public manifest hashes.
  Hash-only retained image artifacts are intentionally absent from Git.
- **42** additional checks pass: scorer/test producers, both final workers'
  imported modules and postprocessor, model/scorer locks, complete dataset
  page counts and unique paths, dataset scope/GT bindings, acquisition and
  reconciliation producers, serialization tests/runtime and receipt links.
- Both final worker selftests bind the six benchmark modules inherited from
  PR #8, including the changed ledger. No stale producer match was assumed.
- Diff against PR #7 is empty for historical `eval/results`, `eval/rca` and
  `eval/wheel-validation`, frozen benchmark/manifests/dependency/model locks,
  and production config/model/compatibility modules.
- The new tracked files contain no page images, dataset shards, model weights,
  wheels, native binaries, host-user paths or detected credentials. Two scan
  matches are explicit synthetic HTTP-test fixtures: a signature query key and
  a synthetic authorization token. Restricted dataset payload was not read.

This verifies publication and binding, not a new independent model-quality run
or a repeated full-payload dataset reread.

## Wheel identity finding and retained baseline

Identical package versions do not identify an identical wheel. ZIP/file hashing
and the installed `direct_url.json` establish:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Preserved historical AMD torch wheel in `.b/wheels/` | 114472348 | `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c` |
| Preserved locally source-built candidate in `.b/out/wheels/` | 388478127 | `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e` |

The existing `.b/o` and `.b/new` installations each match **15/15** candidate
torch DLL/PYD members and only **1/15** original-wheel members. `.b/o` has the
full trio/model runtime, but it is not an original AMD-wheel baseline. The
historical decision explicitly labels this candidate source-built and records
`OFFICIAL_PUBLISHED_WHEEL_VERIFIED=false`. No wheel was rebuilt or installed.

Original full-runtime environments remain under
`.worktrees/phase1-windows-gfx1151/`: `.venv`, `.venv-repro`, `.venv-clean` and
the Unicode-named validation environment each match **15/15** original-wheel
torch native members. Their metadata records torch `2.12.0+rocm7.14.1`,
torchvision `0.27.0+rocm7.14.1`, torchaudio `2.11.0+rocm7.14.1`, transformers
`4.52.1` and triton-windows `3.7.1.post27`. `.b/old` also matches the original
torch native members but lacks transformers/Triton metadata. These are retained
file identities. The subsequent physical preflight and both arms used
`.venv-repro`, verified all 15 loaded original native members and the physical
gfx1151 BF16 device. No retained environment was modified.

Use the same explicitly identified environment and wheel for both sides of a
MATH implementation comparison. Candidate MATH results cannot be described as
original AMD-wheel parity. `C:/uocr` resolves to the primary repository root;
it must not be assumed to identify a retained worktree environment. All command
working directories remain the primary repository root.

Local audit details are in `.b/g1-physical/integration/audit.json`, with separate
PR snapshots, Git publication/binding checks, wheel/member hashes and retained
environment metadata. Raw installation origins stay local.

## Final scoped readiness

The original read-only review above remains distinct from later physical
execution. PR #7 is technically ready for its benchmark infrastructure scope;
PR #8 is technically ready for its completion contract with the measured
original-wheel 1/2/3/5-page regression; PR #9 is technically ready for the
official scorer and complete dataset acquisition scope on that unchanged stack.
Formal GitHub approval and the maintainer's merge decision remain outstanding.
Draft #5 remains unchanged and is outside this readiness finding.

A final API reread at 2026-10-29 14:28 UTC confirmed all four listed heads
unchanged, both exact stack edges and all eight Windows/Ubuntu CPU/scorer checks
successful on each of PRs #7–#9. Its independent receipt is
`integration/final-pr-snapshots.json` in the G1 archive. Local CPU verification
also passes Ruff check/format, mypy, build and pip check, with final pytest **271
passed / 7 skipped**. The skips explicitly cover opt-in GPU/torch checks in the
CPU environment; they do not replace the separate physical evidence.
The sealed archive's earlier 269-test snapshot is retained; the later publisher
tests and final result are bound in `eval/g1/VALIDATION.json`.

The G1 evidence adds diagnostics, tests and reports without changing production
model/wrapper/configuration code. Length remains length; page-guard stop is
explicitly insufficient for OCR completeness. Above-five-page execution, full
420/100 inference, Linux GPU operation and external hard-kill/OOM recovery remain
NOT RUN. Dataset acquisition and hash closure do not change those statuses.
