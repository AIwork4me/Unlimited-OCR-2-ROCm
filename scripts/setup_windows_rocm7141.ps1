[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvPath = (Join-Path $PSScriptRoot "..\.venv"),
    [switch]$SkipModelDependencies,
    [string]$RunId = ([guid]::NewGuid().ToString())
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    $priorPythonEncoding = $env:PYTHONIOENCODING
    try {
        $env:PYTHONIOENCODING = "utf-8"
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
        }
    } finally {
        $env:PYTHONIOENCODING = $priorPythonEncoding
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
$knownLongEntry = "Lib\site-packages\_rocm_sdk_libraries\bin\hipblaslt\library\gfx1151\TensileLibrary_BB_BB_HA_Bias_Aux_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.dat.zlib"
if ((Join-Path $environmentRoot $knownLongEntry).Length -ge 260) {
    throw "This AMD wheel reaches the Windows 260-character path boundary. Choose a shorter dedicated VenvPath (for example C:\uocr2-env). No registry change is required."
}
Invoke-CheckedPython $Python @("-c",
    "import sys,struct; assert sys.platform=='win32' and sys.version_info[:2]==(3,10) and struct.calcsize('P')==8, 'Use 64-bit CPython 3.10'")
$createdNew = -not (Test-Path -LiteralPath $environmentRoot)
if (-not $createdNew) {
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
Invoke-CheckedPython $venvPython @("-m", "pip", "install", "pip==26.2.2",
    "setuptools==81.0.0", "wheel==0.48.0", "packaging==26.3",
    "--report", (Join-Path $evidence "installer-tools.json"))

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
$creationArgs=@((Join-Path $PSScriptRoot "record_environment.py"), "--output",
    (Join-Path $evidence "creation.json"), "--run-id", $RunId)
if ($createdNew) { $creationArgs += "--created-new" }
Invoke-CheckedPython $venvPython $creationArgs
Invoke-CheckedPython $venvPython @("--version")
Invoke-CheckedPython $venvPython @("-m", "pip", "--version")
Invoke-CheckedPython $venvPython @("-m", "pip", "check")
& $venvPython -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $evidence "freeze.txt")
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed" }
Invoke-CheckedPython $venvPython @((Join-Path $PSScriptRoot "doctor_windows.py"),
    "--output", (Join-Path $evidence "doctor.json"))
Write-Output "Isolated interpreter: $venvPython"
