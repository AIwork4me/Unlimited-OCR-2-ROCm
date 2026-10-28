param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Cases
)
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'compare_backends.py'
$arguments = @($runner, '--python', $Python, '--output', $Output)
if ($Cases) { $arguments += @('--cases', $Cases) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Matrix orchestration failed: $LASTEXITCODE" }
Write-Host 'Children recorded. Check individual JSON statuses and exit codes for acceptance.'
