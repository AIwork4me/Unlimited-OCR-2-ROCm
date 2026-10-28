[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvPath = (Join-Path $PSScriptRoot "..\.venv"),
    [switch]$SkipModelDependencies
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
    throw "Phase 1 requires native Windows."
}
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$environmentRoot = [IO.Path]::GetFullPath($VenvPath)
if ($environmentRoot -eq [IO.Path]::GetPathRoot($environmentRoot)) {
    throw "Refusing filesystem root as virtual environment."
}
Invoke-CheckedPython $Python @("-c",
    "import sys,struct; assert sys.platform=='win32' and sys.version_info[:2]==(3,10) and struct.calcsize('P')==8, 'Use 64-bit CPython 3.10'")
if (Test-Path -LiteralPath $environmentRoot) {
    if (-not (Test-Path -LiteralPath (Join-Path $environmentRoot "pyvenv.cfg"))) {
        throw "Existing directory is not a venv; refusing to modify it: $environmentRoot"
    }
} else {
    Invoke-CheckedPython $Python @("-m", "venv", $environmentRoot)
}
$venvPython = Join-Path $environmentRoot "Scripts\python.exe"
Invoke-CheckedPython $venvPython @("-c",
    "import sys,struct; assert sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,10) and struct.calcsize('P')==8")
$evidence = Join-Path $environmentRoot "phase1-install"
New-Item -ItemType Directory -Force -Path $evidence | Out-Null

# Mandatory package names, extras, versions and index are kept exactly.
Invoke-CheckedPython $venvPython @("-m", "pip", "install", "--index-url",
    "https://repo.amd.com/rocm/whl-multi-arch/",
    "torch[device-gfx1151]==2.12.0+rocm7.14.1",
    "torchvision[device-gfx1151]==0.27.0+rocm7.14.1",
    "torchaudio==2.11.0+rocm7.14.1",
    "--report", (Join-Path $evidence "rocm-install.json"))

# The AMD packages above do not install Triton in the observed Windows stack.
# Refuse overlapping distributions; do not uninstall user-selected components.
Invoke-CheckedPython $venvPython @("-c",
    "import importlib.metadata as m; ds={d.metadata['Name'].lower() for d in m.distributions()}; assert not ({'triton','pytorch-triton-rocm'} & ds), 'Review installed AMD/generic Triton before proceeding'")
Invoke-CheckedPython $venvPython @("-m", "pip", "install", "--no-deps", "--require-hashes",
    "-r", (Join-Path $repoRoot "requirements\triton.txt"),
    "--report", (Join-Path $evidence "triton-install.json"))
if (-not $SkipModelDependencies) {
    Invoke-CheckedPython $venvPython @("-m", "pip", "install",
        "-c", (Join-Path $repoRoot "requirements\stack.txt"),
        "-r", (Join-Path $repoRoot "requirements\windows-cp310.lock.txt"),
        "--report", (Join-Path $evidence "native-install.json"))
} else {
    Invoke-CheckedPython $venvPython @("-m", "pip", "install", "psutil==7.1.2",
        "-c", (Join-Path $repoRoot "requirements\stack.txt"))
}
Invoke-CheckedPython $venvPython @("-m", "pip", "install", "--no-deps", "-e", $repoRoot)
Invoke-CheckedPython $venvPython @("--version")
Invoke-CheckedPython $venvPython @("-m", "pip", "--version")
Invoke-CheckedPython $venvPython @("-m", "pip", "check")
& $venvPython -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $evidence "freeze.txt")
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed" }
Invoke-CheckedPython $venvPython @((Join-Path $PSScriptRoot "doctor_windows.py"),
    "--output", (Join-Path $evidence "doctor.json"))
Write-Output "Isolated interpreter: $venvPython"
