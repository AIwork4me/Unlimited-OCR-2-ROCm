# Windows installation and reproduction

This guide targets native Windows 11, CPython3.10 x64, Ryzen AI MAX+395 / Radeon8060S gfx1151. Required identities are torch2.12.0+rocm7.14.1, torchvision0.27.0+rocm7.14.1, torchaudio2.11.0+rocm7.14.1, triton-windows3.7.1.post27 and Transformers4.52.1. Original torch archive SHA256 is `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`; candidate `6a67d158…` is a different binary despite the same version. No wheel replacement in old environments.

Run every command from the primary repository root. The mission's fresh local clone `.worktrees/wi` is at `3556e3867a086437a43a645fa907689f4dd25260`, matching reviewed code. These commits are local review preparation, not a remotely published release: an ordinary remote clone cannot yet be assumed to contain them. When the reviewed source is integrated, `$Source` can be the current repository root. No old checkout, cache or environment should be removed.

Both new `.b/w1` and `.b/w2` completed offline installation and pip check; installer GPU doctor was skipped. Subsequent bounded qualification completed13/13 original cases in each new environment, with BF16/Triton/DASH/MATH and independently verified content/structure PASS; see [baseline](NATIVE_BASELINE_VALIDATION.md). The first manually reviewed environment policy failed before infer0 and remains sealed; the corrected sibling identity was separately reviewed before execution. The 43-artifact wheelhouse `.b/wh` passed an offline resolver dry-run; original nine runtime archives were individually hash/tag/path verified (maximum member path219 under w1/w2). Private install/acquisition manifests and logs are retained. This does not establish new GPU correctness.

## Prepared wheelhouse quickstart

```powershell
# Select an existing CPython3.10 x64 interpreter; or set its explicit full path.
$Python310 = (& py -3.10 -c "import sys; print(sys.executable)")
$Source = Join-Path $PWD '.worktrees/wi'
$Wheelhouse = Join-Path $PWD '.b/wh'
$Venv = Join-Path $PWD '.b/w1' # Must be unused; select another short new name on this host.

# CPU-only: no venv creation, installation, model loading or GPU work.
& (Join-Path $Source 'scripts/setup_windows_rocm7141.ps1') -Python $Python310 -VenvPath $Venv -Wheelhouse $Wheelhouse -Offline -PreflightOnly
if ($LASTEXITCODE -ne 0) { throw 'Archive preflight failed' }

# Installs packages into a NEW venv; explicitly defers the GPU doctor.
& (Join-Path $Source 'scripts/setup_windows_rocm7141.ps1') -Python $Python310 -VenvPath $Venv -Wheelhouse $Wheelhouse -Offline -SkipGpuChecks
if ($LASTEXITCODE -ne 0) { throw 'Installation failed; retain evidence' }
$OcrPython = Join-Path $Venv 'Scripts/python.exe'
& $OcrPython -m pip check
```

Existing paths are refused, including same-version candidate environments. Preflight reports the nine verified originals, not the entire dependency closure; a wheelhouse containing only AMD9 is insufficient. Offline applies `--no-index --find-links` to each pip step, including installer tools/native dependencies; editable build uses the installed pinned build tools without isolation downloads. Failures stop installation and remain inspectable. Setup scopes TEMP/TMP, pip, HF and Triton caches below the repository and restores caller environment variables. No global PATH, driver, SDK, registry/security or long-path policy changes.

## Preparing an independent complete wheelhouse

The nine exact artifact names/URLs/hashes are frozen in `requirements/windows-rocm7141.lock.txt`; the full Windows dependency versions are in `requirements/windows-cp310.lock.txt`. Generic package resolution from PyPI alone fails for `rocm` and `rocm-bootstrap`: their recorded original sources are AMD. The following is an acquisition recipe for a **new** wheelhouse; fresh-host acquisition has not been independently tested in this mission. It downloads artifacts without modifying the base Python environment. Network/index availability remains an acquisition gate.

```powershell
$NewHouse = Join-Path $PWD '.b/wh-new'
if (Test-Path $NewHouse) { throw 'Choose a new wheelhouse' }
$Names = @('TEMP','TMP','PIP_CACHE_DIR')
$Prior = @{}
foreach ($n in $Names) { $Prior[$n] = [Environment]::GetEnvironmentVariable($n,'Process') }
try {
    $env:TEMP = Join-Path $PWD '.b/install-acquire/tmp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $PWD '.b/install-acquire/pip'
    New-Item -ItemType Directory -Force $env:TEMP,$env:PIP_CACHE_DIR | Out-Null
    New-Item -ItemType Directory $NewHouse | Out-Null
    & $Python310 -m pip download --no-deps --require-hashes --index-url https://repo.amd.com/rocm/whl-multi-arch/ --dest $NewHouse -r (Join-Path $Source 'requirements/windows-rocm7141.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Original archive download failed' }
    & $Python310 -m pip download --no-deps --dest $NewHouse 'https://repo.amd.com/rocm/whl-multi-arch/rocm_bootstrap-0.1.0-py3-none-any.whl#sha256=33e3d4dad086b1f0e06bc049a618ca874d4fb72f758fff00676e23d2a644b652' 'https://repo.amd.com/rocm/whl-multi-arch/rocm-7.14.1.tar.gz#sha256=ffea97f6a9da9987a9260f9e635995b27a44dd6985d1b2b1663431356aaf669b'
    if ($LASTEXITCODE -ne 0) { throw 'AMD metadata download failed' }
    & $Python310 -m pip download --no-deps --dest $NewHouse --find-links $NewHouse -r (Join-Path $Source 'requirements/windows-cp310.lock.txt') 'pip==26.2.2' 'setuptools==81.0.0' 'wheel==0.48.0' 'packaging==26.3'
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency download failed' }
} finally {
    foreach ($n in $Names) { [Environment]::SetEnvironmentVariable($n,$Prior[$n],'Process') }
}
```

The [observed 43-artifact SHA256 inventory](../../eval/windows/wheelhouse-SHA256SUMS.json) is public and byte-exact to the tested wheelhouse manifest. Verify every name/hash against a newly acquired wheelhouse before installation; a missing or different artifact is a failed identity check. Retain installer reports privately. These observed hashes are not publisher signatures. AMD9 hashes are retained observed receipts rather than publisher signatures. Triton uses its existing hash lock; generic dependency versions alone are not binary hashes. Validate closure with a fresh offline install and pip check, preserving failures. No bundled SDK devel install or manual compiler unpack is required by this setup; compiler/runtime prerequisites still need actual kernel validation.

## Immutable model cache and bounded validation

Model/cache source is [Baidu at the frozen revision](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c). Downloads are separate from package installation. Code/config/tokenizer/weight hashes in upstream-lock.json are checked before trust_remote_code; offline uses local-only snapshot/model resolution. Use `.b/h` as an explicit model cache and process-scoped repository-local TMP/TEMP when invoking model commands directly: dynamic reviewed-module caches use tempfile.

```powershell
$Cache = Join-Path $PWD '.b/h'
# Data-only acquisition/verification; does not run inference.
& $OcrPython -m unlimited_ocr_2_rocm download --cache-dir $Cache
if ($LASTEXITCODE -ne 0) { throw 'Model acquisition failed' }
# Once cached, this checks immutable cached files without network lookup.
& $OcrPython -m unlimited_ocr_2_rocm download --cache-dir $Cache --offline
```

`reproduce_windows.ps1` is a real GPU workflow, not an installation-only command: it executes doctor/BF16, Triton, DASH, MATH and single/multi smoke suites. `-ReuseEnvironment -Offline` reuses a prepared environment/model cache without pip; fresh offline reproduction also accepts a complete `-Wheelhouse`. Use new output paths and an independently reviewed bounded controller. The mission permits at most two existing13-case suites (26 full OCR requests), one per clean environment, conditional on gates; do not launch them by copying this guide. The actual completed baseline proof is [recorded separately](NATIVE_BASELINE_VALIDATION.md); copying these setup commands does not reproduce its independently reviewed controller identity. Full real-data campaigns and larger-token experiments require separate explicit admission/authorization.

## Deliberate one-page user run

After installation and offline model verification, a user can run the public single-page CLI below. Its argument syntax was CPU-checked with `single --help`; this mission's physical proof used the same shared `run_native` inference path through the frozen bounded controller. The standalone CLI process was not separately GPU-run here. Select an unused short runtime/output path, preserving earlier runs.

```powershell
$UserRun = Join-Path $PWD '.b/user1'
if (Test-Path $UserRun) { throw 'Choose an unused short runtime/output path' }
$CacheVariables = @('TEMP','TMP','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE')
$SavedCache = @{}
foreach ($n in $CacheVariables) { $SavedCache[$n] = [Environment]::GetEnvironmentVariable($n,'Process') }
try {
    $env:TEMP = Join-Path $UserRun 'tmp'
    $env:TMP = $env:TEMP
    $env:TRITON_CACHE_DIR = Join-Path $UserRun 'triton'
    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $UserRun 'inductor'
    $env:HF_HUB_CACHE = $Cache
    $env:HUGGINGFACE_HUB_CACHE = $Cache
    New-Item -ItemType Directory -Force $env:TEMP,$env:TRITON_CACHE_DIR,$env:TORCHINDUCTOR_CACHE_DIR | Out-Null
    & $OcrPython -m unlimited_ocr_2_rocm single (Join-Path $Source 'tests/fixtures/english.png') --cache-dir $Cache --offline --max-new-tokens 1024 --output (Join-Path $UserRun 'output')
    if ($LASTEXITCODE -ne 0) { throw 'Native inference failed; retain all output and logs' }
} finally {
    foreach ($n in $CacheVariables) { [Environment]::SetEnvironmentVariable($n,$SavedCache[$n],'Process') }
}
```

This is one whole single-page request, with the unchanged original model and production backend. Use the public `multi` CLI only within the documented five-image/8192-token production limits. Do not substitute page concatenation for official long-document inference or treat a successful smoke output as a benchmark score.
