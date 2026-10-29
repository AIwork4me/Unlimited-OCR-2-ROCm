# Next-stage execution guide

Run PowerShell commands from the primary repository root. This PR provides
inventory, atomic result ingestion, resume/report/export and scorer tools;
**it does not add a long-document GPU inference runner or remove existing caps**.
The physical adaptation steps below belong to the next explicitly reviewed
mission. Preserve all earlier evidence and worktrees.

## CPU tooling and scorer self-test

Use an existing CPython 3.10 executable to create repository-local environments.
Process-scoped caches avoid changing global settings:

```powershell
$env:TEMP = Join-Path $PWD '.b/benchmark-tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $PWD '.b/benchmark-cache'
$env:MPLCONFIGDIR = Join-Path $PWD '.b/benchmark-matplotlib'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
# $Python310 is the explicit path of an already available Python 3.10 interpreter.
& $Python310 -m venv .b/benchmark-cpu
& .b/benchmark-cpu/Scripts/python.exe -m pip install -r requirements/dev.txt
& .b/benchmark-cpu/Scripts/python.exe -m pip install --no-deps -e .
& $Python310 -m venv .b/benchmark-scorer
& .b/benchmark-scorer/Scripts/python.exe -m pip install -r requirements/benchmark-scorer.lock.txt
& .b/benchmark-scorer/Scripts/python.exe -m pip install --no-deps -e .
```

Abort on any nonzero install/check exit. No torch, model or image download is
part of CPU setup. Linux uses the equivalent `bin/python` in these local venvs.
Retain `pip freeze`, `pip check`, Python identity and install artifact hashes.

```powershell
$Cpu = '.b/benchmark-cpu/Scripts/python.exe'
$Scorer = '.b/benchmark-scorer/Scripts/python.exe'
$Lock = 'eval/benchmarks/benchmark-lock.json'
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli fetch-scorer --source .b/qwen-scorer --lock $Lock
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli score --source .b/qwen-scorer --lock $Lock --python $Scorer --output .b/scorer-qwen-run1
```

Use fresh source/output directories. The source downloader fetches only the
locked evaluator/license files, checks bytes before import, and never invokes
upstream upload helpers. `score` captures stdout/stderr and an artifact hash
inventory. Expected readiness here is **exit 2, BLOCKED, Overall null** because
XeLaTeX is missing. Inspect `report.json` and `worker/selftest.json` to distinguish
seven passing metric checks and the passing continuation pipeline from CDM.
An unexpected import/metric/render error is not the expected CDM blocker.

For the separate official Marathon profile:

```powershell
$env:MPDOC_EVAL_NPROC = '1'
$env:MPDOC_DOC_BUDGET = '0'
$env:MPDOC_TEDS_NPROC = '1'
$env:MPDOC_TEDS_TIMEOUT = '300'
$env:MPDOC_CDM_NPROC = '1'
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli fetch-scorer --source .b/marathon-scorer --profile marathon --lock $Lock
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli score --source .b/marathon-scorer --profile marathon --lock $Lock --python $Scorer --output .b/scorer-marathon-run1
```

Serial settings above are fixture settings. Full Marathon scoring must use the
frozen official 32-worker/1800s-match/300s-TEDS profile in the protocol. A change
to resource caps is a separate experiment and must never silently replace it.
The CI integration job deliberately verifies seven metrics plus the pipeline
while retaining a BLOCKED eight-metric receipt; a green CI job is not CDM or GPU
acceptance. All-eight readiness requires a real positive formula render and a
frozen, working TeX/font/PDF toolchain. Do not change system security/PATH or
drivers/SDKs to pass a check.

## Acquire and validate inputs

Acquire frozen GT and images only under `.b/benchmark-data/<dataset>/`.
Marathon uses HF dataset revision
`fa05fccbe82ee6ec628006a1d58bb153954c3ee3`, not `main`; verify all LFS object
hashes from that revision. MPDocBench GT comes from the pinned Qwen commit,
and its ModelScope archive revision/hash and gated SlideVQA revision/shards are
now frozen in the lock. Acquire only those identities. The gated 11-deck subset
and its JPEG serialization/checksum closure still need reconciliation.
Do not execute the upstream downloader against mutable revisions without an
audit. Never add acquired documents or annotations to Git.

```powershell
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli inventory --dataset MPDocBench --ground-truth .b/benchmark-data/MPDocBench/MPDocBench.json --dataset-root .b/benchmark-data/MPDocBench --lock $Lock --output .b/MPDocBench.manifest.json
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli validate --manifest .b/MPDocBench.manifest.json --dataset-root .b/benchmark-data/MPDocBench
```

Repeat with `MarathonDocBench` and its frozen GT. Missing page bytes produce
null hashes and BLOCKED, while scope/duplicate/order/path errors raise. Once
missing bytes are acquired, generate a **new** manifest, preserving the earlier
blocked inventory. Run initialization rehashes all local pages; it cannot
proceed with missing/changed bytes. Do not repair a bad checksum by silently
rewriting an accepted manifest. Compare the scope JSONL and annotation hashes
in the lock before publishing an inventory.

## Record and resume a run

Prepare `.b/protocol.json` and `.b/environment.json` with every field required
by BENCHMARK_PROTOCOL.md, including actual model and wheel hashes, all
source/config/decoder/crop/token/window policies, native GPU/driver/HIP identity,
CLI argument arrays and producing environment evidence. These are supplied
provenance documents; the infrastructure cannot establish GPU truth from a
manually supplied environment JSON.

```powershell
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli init --manifest .b/MPDocBench.manifest.json --dataset-root .b/benchmark-data/MPDocBench --config .b/protocol.json --environment .b/environment.json --output .b/benchmark-runs
# Save the printed dataset-scoped directory as $Run.
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli todo --run $Run
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli begin --run $Run --document $Document
# Save the printed attempt ID as $Attempt. An external reviewed GPU runner
# supplies raw UTF-8 bytes and completion metadata after one whole-doc request.
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli finish --run $Run --document $Document --attempt $Attempt --state PASS --raw .b/request.raw.txt --metadata .b/request.metadata.json
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli report --run $Run --output .b/report-run1
& $Cpu -m unlimited_ocr_2_rocm.benchmark.cli export --run $Run --output .b/predictions-run1
```

Completion metadata must include `finish_reason`, `output_tokens`, `max_tokens`
and measured memory (or explicit measurement failures), plus device/kernel/
nonfinite evidence. An attempted PASS with absent completion data, malformed
pages or token-limit finish is preserved as FAIL/TRUNCATED. Use explicit FAIL,
TIMEOUT or OOM plus exception/partial-output/memory information for failed
requests. If a process dies before completion, the pending attempt remains
visible as INTERRUPTED/FAIL. `todo` retries every latest non-PASS state; retries
append, and all previous failures remain in reports. Single-writer ownership
is required per document; no inference scheduling is provided by this CLI.

Keep outputs UTF-8, use argument lists, avoid active HTML rendering, and never
reuse an artifact path for a second attempt. `Run.finish` uses the unchanged
official converter with the existing heading-expansion guard. A recorded PASS
is completion, not OCR fidelity or quality acceptance. All GPU nonfinite/kernel
checks belong to the producing reviewed runner and must fail its requested state.

Only after the all-eight self-test passes may `score` receive
`--ground-truth <frozen GT> --predictions <complete export>`. It refuses a
different GT hash, missing/extra prediction files or nonmatching scorer source.
Retain original `metric_result.json`, every task's sample output,
`coverage.json`, logs, generated config and full run accounting. Missing task
samples are explicitly listed per document; reconcile them against applicable
GT types and failures before calling any result full-scope. The scorer report
qualifies values as SCORER_ONLY; human-reviewed coverage/error accounting is
still required. Do not publish an Overall or parity/performance claim from an
incomplete, blocked or non-equivalent configuration.

## Next executable physical validation sequence

1. Windows MATH: inspect the preserved original environment and hash-verified
   cache, run existing doctor/BF16, Triton/DASH and single/multi fixture commands
   with fresh repository-local output paths. Preserve the exact mandatory trio
   and gfx1151; run opt-in tests only on this physical host. These commands
   already exist in `scripts/doctor_windows.py`, `probe_sparse.py`,
   `smoke_single.py`, `smoke_multi.py` and `tests/gpu/`.
2. Windows candidate: use the already built artifact, complete existing
   candidate native/image/loaded-library identity gate and no-overlays install
   evidence. Repeat the frozen fused/MATH regression without rebuilding a wheel.
   Keep opt-in AUTO evidence separate and MATH as default.
3. Implement/review the minimal production adapter separately from #5's evidence.
   Expose authoritative finish/page/window fields, errors and atomic ingestion.
   Validate 1/5/15/30/50/100/300-page requests progressively where supported.
   Keep OOM/timeout/unsupported cases; increasing a cap is not acceptance.
4. Physically validate a separately pinned Linux W7900 native profile before
   assigning any Linux PASS. Then audit/port the customized SGLang dependency
   and HIP path in its separate workstream; never replace it with generic SGLang.
5. After data/CDM/protocol/quality gates are reviewed, execute all 420 and all 100
   documents, separately by platform/backend/image mode/precision, with the
   ledger/export/scorer workflow. This full GPU work is **not authorized as part
   of the current infrastructure-only task**.

No merge, release, outreach or upstream issue/PR creation follows automatically.
