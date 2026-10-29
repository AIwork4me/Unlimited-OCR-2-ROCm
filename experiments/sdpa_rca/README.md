# Standalone Windows gfx1151 SDPA RCA

The pinned Windows wheel combines a 0.12-family AOTriton runtime with 0.13-family ZIP kernel images. An images-only A/B/A on the actual gfx1151 machine changes FAIL → PASS → FAIL while PyTorch, HIP, driver and DLL hashes remain fixed. See [the L4 report](../../docs/rca/SDPA_ROOT_CAUSE.md).

## Baseline reproduction

Use native Windows and CPython **3.10.20 x64**. Create a new, short virtual-environment path. `$Py310` below is the absolute path to that interpreter; `$Python` is the resulting venv interpreter. No OCR package, model download, Transformers, Hugging Face, SGLang or Python Triton package is required.

```powershell
$Root = Join-Path $env:USERPROFILE '.codex\v\sdpa-review'
& $Py310 -m venv $Root
$Python = Join-Path $Root 'Scripts\python.exe'
& $Python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'torch[device-gfx1151]==2.12.0+rocm7.14.1' 'torchvision[device-gfx1151]==0.27.0+rocm7.14.1' 'torchaudio==2.11.0+rocm7.14.1' --report install.json

# Smallest positive tensor dimensions: one BF16 element per Q/K/V, no mask.
& $Python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --output scalar-default.json
# Expected original wheel: exit 1, FLASH_ATTENTION, HIP last-error 1.
& $Python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --backend MATH --output scalar-math.json
# Expected: exit 0, output equals V and the independent CPU reference.

# Actual MATH operator evidence (dispatch-mode decomposes this composite).
& $Python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --backend MATH --trace profiler --output math-profile.json

# Original vision shape [54,12,196,64] and additive [54,12,196,196] bias.
.\experiments\sdpa_rca\run_matrix.ps1 -Python $Python -Output '.local\review-baseline'
& $Python experiments/sdpa_rca/collect_provenance.py --install-report install.json --output provenance.json
```

One new interpreter is launched per condition. `compare_backends.py --cases cases.json` accepts a list of `{ "name": "unique-name", "args": ["--shape", "1", "1", "1", "8"] }`. Published [condition files](../../eval/rca/gfx1151-rocm7141/conditions.json) and [repetition files](../../eval/rca/gfx1151-rocm7141/repetition-cases.json) reproduce the expanded matrix. The matrix runner's exit 0 means orchestration completed; each child retains its actual exit code. Child codes: PASS=0, FAIL/INCOMPLETE=1, UNSUPPORTED=2. Never interpret unsupported FLASH+bias or cuDNN as an executed kernel failure.

Frozen tolerances, set before execution: BF16 atol/rtol=0.03/0.03, FP16=0.005/0.005, FP32=0.0001/0.0001. Acceptance is elementwise `abs(actual-reference) <= atol + rtol*abs(reference)` and no NaN/Inf. Relative-error reporting floors its denominator at 1e-6; relative maxima alone are not acceptance criteria. CPU seeded inputs are quantized to the tested dtype before computing explicit FP32 softmax(QKᵀ/sqrt(D)+bias)V. Boolean true means allowed. Causal masks use the upper-left triangle; dropout=0 and GQA disabled. The fixed seed is 7992. CPU fixture bytes, dtypes and shapes are hashed. Noncontiguous cases preserve the same values and last-dimension stride 1.

Acceptance never calls hipGetLastError. A process with a detected error remains FAIL even if a downstream PyTorch API consumes the sticky error. `--clear-error-nonacceptance` explicitly marks an isolated diagnostic, allowing numerical inspection of the invalid result. Its output can never be promoted to GPU acceptance by this report.

## Images-only diagnostic A/B/A

Use a **separate disposable environment**, never the production venv. First run the baseline commands above and save provenance A. Keep every package and DLL unchanged throughout the following steps. The archive is GPU image data, not an alternate Linux runtime.

```powershell
$ImagesUrl = 'https://github.com/ROCm/aotriton/releases/download/0.12.50tp/aotriton-0.12.50tp-images-amd-gfx115x.tar.gz'
$Archive = Join-Path $Root 'images012.tar.gz'
curl.exe --fail -L $ImagesUrl -o $Archive
$Expected = '3177387a15c678b30057f4584d1fc1b8f8db56163890cb5c98f27450209f5a7b'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLower() -ne $Expected) { throw 'Archive digest mismatch' }
$Unpack = Join-Path $Root 'unpacked012'
New-Item -ItemType Directory -Path $Unpack | Out-Null
tar --options hdrcharset=UTF-8 -xf $Archive -C $Unpack
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
$Lib = Join-Path $Root 'Lib\site-packages\torch\lib'
Rename-Item -LiteralPath (Join-Path $Lib 'aotriton.images') -NewName 'aotriton.images-original013'
Copy-Item -LiteralPath (Join-Path $Unpack 'aotriton\lib\aotriton.images') -Destination (Join-Path $Lib 'aotriton.images') -Recurse
# Repeat the exact same commands and fixtures into fresh B output paths.
& $Python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --output scalar-B.json
& $Python experiments/sdpa_rca/collect_provenance.py --output provenance-B.json
# Restore A and repeat, keeping the compatible images only as a diagnostic backup.
Rename-Item -LiteralPath (Join-Path $Lib 'aotriton.images') -NewName 'aotriton.images-compatible012'
Rename-Item -LiteralPath (Join-Path $Lib 'aotriton.images-original013') -NewName 'aotriton.images'
& $Python experiments/sdpa_rca/repro_sdpa.py --shape 1 1 1 1 --mask none --output scalar-A-restored.json
```

The 0.12 archive's signature identifies `fe12b77d…` and `aotriton0.12`; the runtime header identifies `05f26c2d…`. They are **compatible family/build artifacts**, not identical Git commits. Exact hash equality is not the integration contract. The defective original pairing instead combines a loose-AKS2 loader with ZIP storage. This diagnostic does not make edited wheel contents a supported deployment. Keep the existing scoped MATH adapter on the mandated official stack.

For HIP launch evidence set `$env:AMD_LOG_LEVEL='4'` only for one fresh child, then restore the prior value. Use `--trace none`, redirect stdout/stderr to files, and compare hipModuleLoadDataEx/hipModuleLaunchKernel before and after the image change. Source and PE imports are linked in the report. CPU profiler coverage does not identify device kernels by itself.

## Evidence and checks

[Published evidence](../../eval/rca/gfx1151-rocm7141/), [matrix](../../docs/rca/TEST_MATRIX.md), [hypotheses](../../docs/rca/HYPOTHESES.md), [English upstream handoff](../../docs/rca/UPSTREAM_HANDOFF.md).

`export_evidence.py` redacts the home path from selected JSON/log/text files, retains original SHA-256 alongside published SHA-256, and emits SHA256SUMS.txt. Original private stdout/stderr remain local; published logs preserve content except the documented redaction. Git text conversion is disabled for evidence. Do not export a general working directory containing secrets; stage only reviewed evidence.

CPU CI requires no torch. Optional reference/fixture tests run when torch is available; this investigation executes them on CPU using the pinned GPU environment. Run `python -m pytest`, `python -m ruff check src scripts tests experiments/sdpa_rca`, `python -m ruff format --check src scripts tests experiments/sdpa_rca`, `python -m mypy src`, and `python -m build`.
