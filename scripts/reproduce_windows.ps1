[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvPath = (Join-Path $PSScriptRoot "..\.venv-repro"),
    [string]$CacheDir = (Join-Path $env:USERPROFILE ".cache\unlimited-ocr2-phase1\models"),
    [string]$OutputPath = (Join-Path $PSScriptRoot ("..\eval\results\run-" + (Get-Date -Format "yyyyMMdd-HHmmss"))),
    [switch]$ReuseEnvironment,
    [switch]$Offline,
    [string]$BaselinePath = ""
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outputRoot = [IO.Path]::GetFullPath($OutputPath)
$environmentRoot = [IO.Path]::GetFullPath($VenvPath)
$venvPython = Join-Path $environmentRoot "Scripts\python.exe"
if ((Test-Path $outputRoot) -and (Get-ChildItem -Force $outputRoot)) {
    throw "Use a new output directory to preserve earlier evidence."
}
if ($Offline -and -not $ReuseEnvironment) {
    throw "Offline rerun requires -ReuseEnvironment and an already prepared isolated environment."
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$steps = [Collections.Generic.List[object]]::new()
$status = "FAIL"
$offlineArgs = @()
if ($Offline) { $offlineArgs = @("--offline") }

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    $entry = [ordered]@{name=$Name; command=@($venvPython) + $Arguments; status="FAIL"}
    try {
        & $venvPython @Arguments *> (Join-Path $outputRoot "$Name.log")
        $entry.exit_code = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "$Name failed; inspect $Name.log" }
        $entry.status = "PASS"
    } finally { $steps.Add($entry) }
}

Push-Location $repoRoot
try {
    if ($ReuseEnvironment) {
        if (-not (Test-Path (Join-Path $environmentRoot "pyvenv.cfg"))) {
            throw "Reuse requires an existing dedicated virtual environment."
        }
    } else {
        if (Test-Path $environmentRoot) {
            throw "Clean reproduction requires a new VenvPath; use -ReuseEnvironment for reruns."
        }
        & (Join-Path $PSScriptRoot "setup_windows_rocm7141.ps1") -Python $Python -VenvPath $environmentRoot *> (Join-Path $outputRoot "setup.log")
    }
    Invoke-Step "isolated-python" @("-c", "import sys; assert sys.prefix!=sys.base_prefix")
    Invoke-Step "pip-check" @("-m", "pip", "check")
    Invoke-Step "pip-version" @("-m", "pip", "--version")
    Invoke-Step "freeze" @("-m", "pip", "freeze")
    Invoke-Step "doctor" @((Join-Path $PSScriptRoot "doctor_windows.py"), "--output", (Join-Path $outputRoot "doctor.json"))
    Invoke-Step "triton" @((Join-Path $PSScriptRoot "probe_triton.py"), "--output", (Join-Path $outputRoot "triton.json"))
    Invoke-Step "download" (@("-m", "unlimited_ocr_2_rocm", "download", "--cache-dir", $CacheDir) + $offlineArgs)
    $snapshot=Join-Path $CacheDir "models--baidu--Unlimited-OCR-2\snapshots\9cf077c4b206e0e2a31879df937b0b0bcba3661c"
    Invoke-Step "sparse" @((Join-Path $PSScriptRoot "probe_sparse.py"), "--source", $snapshot, "--output", (Join-Path $outputRoot "sparse.json"))
    Invoke-Step "vision-sdpa" @((Join-Path $PSScriptRoot "probe_vision_sdpa.py"), "--source", $snapshot, "--output", (Join-Path $outputRoot "vision-sdpa.json"))
    Invoke-Step "single" (@((Join-Path $PSScriptRoot "smoke_single.py"), "--fixtures", (Join-Path $repoRoot "tests\fixtures"), "--cache-dir", $CacheDir, "--output", (Join-Path $outputRoot "single")) + $offlineArgs)
    Invoke-Step "multi" (@((Join-Path $PSScriptRoot "smoke_multi.py"), "--fixtures", (Join-Path $repoRoot "tests\fixtures"), "--cache-dir", $CacheDir, "--output", (Join-Path $outputRoot "multi")) + $offlineArgs)
    $status = "PASS"
} catch {
    $failure = $_ | Out-String
    Write-Error -Message $failure -ErrorAction Continue
} finally {
    Pop-Location
    $manifest = [ordered]@{
        status=$status; timestamp_utc=[DateTime]::UtcNow.ToString("o")
        clean_environment=(-not $ReuseEnvironment); offline_model=$Offline.IsPresent
        model_revision="9cf077c4b206e0e2a31879df937b0b0bcba3661c"
        same_physical_machine=$null; steps=@($steps.ToArray())
        gate_A="see doctor.json"; gate_B="see sparse.json and single/load.json"
        gate_C="see single/suite.json"; gate_D="see multi/suite.json"
        gate_E="NOT PROVEN"; phase1_status="BLOCKED"
    }
    if (Get-Variable failure -ErrorAction SilentlyContinue) { $manifest.error=$failure }
    $json=$manifest | ConvertTo-Json -Depth 12
    $json=$json.Replace($env:USERPROFILE.Replace("\","\\"), "<USERPROFILE>").Replace($env:USERPROFILE, "<USERPROFILE>")
    [IO.File]::WriteAllText((Join-Path $outputRoot "reproduction.json"), $json, [Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $venvPython) {
        $summaryArgs=@("-m", "unlimited_ocr_2_rocm.repro", "--root", $outputRoot,
            "--fixtures", (Join-Path $repoRoot "tests\fixtures"))
        if (-not $ReuseEnvironment) {$summaryArgs += "--fresh"}
        if ($BaselinePath) {$summaryArgs += @("--baseline", [IO.Path]::GetFullPath($BaselinePath))}
        & $venvPython @summaryArgs
        if ($LASTEXITCODE -ne 0) { $status="FAIL" }
    }
}
if ($status -ne "PASS") { exit 1 }
