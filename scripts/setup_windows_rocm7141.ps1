[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvPath = (Join-Path (Get-Location).Path ".b\windows-env"),
    [string]$Wheelhouse = "",
    [switch]$Offline,
    [switch]$PreflightOnly,
    [switch]$SkipGpuChecks,
    [switch]$SkipModelDependencies,
    [string]$RunId = ([guid]::NewGuid().ToString())
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "Installation requires native Windows."
}
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$developmentRoot = [IO.Path]::GetFullPath((Get-Location).Path)
$environmentRoot = [IO.Path]::GetFullPath($VenvPath)
$knownLongEntry = "Lib\site-packages\_rocm_sdk_libraries\bin\hipblaslt\library\gfx1151\TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.dat.zlib"
if ((Join-Path $environmentRoot $knownLongEntry).Length -ge 260) {
    throw "AMD wheel reaches the 260-character path boundary. Use a shorter unused repository-local VenvPath, such as .b\w1. No registry change is required."
}
$cacheNames = @("TEMP", "TMP", "PIP_CACHE_DIR", "TRITON_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "PYTHONIOENCODING")
$priorEnvironment = @{}
foreach ($name in $cacheNames) {
    $priorEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:PYTHONIOENCODING = "utf-8"
    $preflightArgs = @((Join-Path $PSScriptRoot "windows_install_preflight.py"), "--root", $developmentRoot, "--venv", $environmentRoot)
    if ($Wheelhouse) {
        $Wheelhouse = [IO.Path]::GetFullPath($Wheelhouse)
        $preflightArgs += @("--wheelhouse", $Wheelhouse)
    }
    if ($Offline) { $preflightArgs += "--offline" }
    # CPU-only and no directories/venv/pip/GPU calls before all archive checks pass.
    $preflightJson = Invoke-CheckedPython $Python $preflightArgs
    if ($PreflightOnly) {
        $preflightJson | Write-Output
        return
    }
    $cacheRoot = Join-Path $developmentRoot ".b\windows-install-cache"
    $env:TEMP = Join-Path $cacheRoot "tmp"
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
    $env:TRITON_CACHE_DIR = Join-Path $cacheRoot "triton"
    $env:HF_HOME = Join-Path $cacheRoot "huggingface"
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
    foreach ($name in @("TEMP", "PIP_CACHE_DIR", "TRITON_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE")) {
        New-Item -ItemType Directory -Force -Path ([Environment]::GetEnvironmentVariable($name, "Process")) | Out-Null
    }
    Invoke-CheckedPython $Python @("-m", "venv", $environmentRoot)
    $venvPython = Join-Path $environmentRoot "Scripts\python.exe"
    Invoke-CheckedPython $venvPython @("-c", "import sys,struct; assert sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,10) and struct.calcsize('P')==8")
    $evidence = Join-Path $environmentRoot "phase1-install"
    New-Item -ItemType Directory -Path $evidence | Out-Null
    [IO.File]::WriteAllText((Join-Path $evidence "archive-preflight.json"), ($preflightJson -join [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    $packageSources = @()
    if ($Wheelhouse) { $packageSources += @("--find-links", $Wheelhouse) }
    if ($Offline) { $packageSources += "--no-index" }
    Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "pip==26.2.2", "setuptools==81.0.0", "wheel==0.48.0", "packaging==26.3", "--report", (Join-Path $evidence "installer-tools.json")) + $packageSources)
    $amdSources = @($packageSources)
    if (-not $Offline) { $amdSources += @("--index-url", "https://repo.amd.com/rocm/whl-multi-arch/") }
    # Require original archive bytes, not just same-version distribution metadata.
    Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "--no-deps", "--require-hashes", "-r", (Join-Path $repoRoot "requirements\windows-rocm7141.lock.txt"), "--report", (Join-Path $evidence "original-archives.json")) + $amdSources)
    Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "torch[device-gfx1151]==2.12.0+rocm7.14.1", "torchvision[device-gfx1151]==0.27.0+rocm7.14.1", "torchaudio==2.11.0+rocm7.14.1", "-c", (Join-Path $repoRoot "requirements\stack.txt"), "--report", (Join-Path $evidence "rocm-install.json")) + $amdSources)
    Invoke-CheckedPython $venvPython @("-c", "import importlib.metadata as m; ds={d.metadata['Name'].lower() for d in m.distributions()}; assert not ({'triton','pytorch-triton-rocm'} & ds), 'Review installed AMD/generic Triton before proceeding'")
    Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "--no-deps", "--require-hashes", "-r", (Join-Path $repoRoot "requirements\triton.txt"), "--report", (Join-Path $evidence "triton-install.json")) + $packageSources)
    if (-not $SkipModelDependencies) {
        Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "-c", (Join-Path $repoRoot "requirements\stack.txt"), "-r", (Join-Path $repoRoot "requirements\windows-cp310.lock.txt"), "--report", (Join-Path $evidence "native-install.json")) + $packageSources)
    } else {
        Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "psutil==7.1.2", "-c", (Join-Path $repoRoot "requirements\stack.txt")) + $packageSources)
    }
    # Pinned build tools are installed above; disable hidden build-isolation downloads.
    Invoke-CheckedPython $venvPython (@("-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", $repoRoot) + $packageSources)
    Invoke-CheckedPython $venvPython @((Join-Path $PSScriptRoot "record_environment.py"), "--output", (Join-Path $evidence "creation.json"), "--run-id", $RunId, "--created-new")
    Invoke-CheckedPython $venvPython @("--version")
    Invoke-CheckedPython $venvPython @("-m", "pip", "--version")
    Invoke-CheckedPython $venvPython @("-m", "pip", "check")
    $freeze = Invoke-CheckedPython $venvPython @("-m", "pip", "freeze")
    $freeze | Set-Content -Encoding UTF8 (Join-Path $evidence "freeze.txt")
    if (-not $SkipGpuChecks) {
        Invoke-CheckedPython $venvPython @((Join-Path $PSScriptRoot "doctor_windows.py"), "--output", (Join-Path $evidence "doctor.json"))
    }
    Write-Output "Isolated interpreter: $venvPython"
} finally {
    foreach ($name in $cacheNames) {
        [Environment]::SetEnvironmentVariable($name, $priorEnvironment[$name], "Process")
    }
}
