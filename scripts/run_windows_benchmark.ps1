[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][ValidateSet("prepare", "run", "report", "export", "gt-slice")][string]$Action,
    [string]$RuntimeRoot = (Join-Path (Get-Location).Path ".b\windows-benchmark-runtime"),
    [string[]]$RunnerArguments = @(),
    [switch]$ExecuteGpu
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This entry point requires native Windows."
}
if ($Action -eq "run" -and -not $ExecuteGpu) {
    throw "GPU execution requires -ExecuteGpu and prior scoped run authorization."
}
$root = [IO.Path]::GetFullPath((Get-Location).Path).TrimEnd([IO.Path]::DirectorySeparatorChar)
$runtime = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $runtime.StartsWith(($root + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
    throw "RuntimeRoot must stay inside the development root."
}
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$names = @("PYTHONPATH", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "TEMP", "TMP", "TRITON_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "PIP_CACHE_DIR")
$prior = @{}
foreach ($name in $names) { $prior[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
try {
    $env:PYTHONPATH = Join-Path $sourceRoot "src"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:TEMP = Join-Path $runtime "tmp"
    $env:TMP = $env:TEMP
    $env:TRITON_CACHE_DIR = Join-Path $runtime "triton"
    $env:HF_HOME = Join-Path $runtime "huggingface"
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
    $env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
    $env:PIP_CACHE_DIR = Join-Path $runtime "pip"
    foreach ($path in @($env:TEMP, $env:TRITON_CACHE_DIR, $env:HF_HOME, $env:PIP_CACHE_DIR)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
    # The shared runner owns manifest identity, ledger, statuses and resume.
    # This switch is deliberate execution, not a grant of campaign authorization.
    & $Python (Join-Path $PSScriptRoot "run_mpdoc_sample.py") $Action @RunnerArguments
    if ($LASTEXITCODE -ne 0) { throw "Shared benchmark runner failed (exit $LASTEXITCODE)." }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $prior[$name], "Process") }
}
