[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$IdentityProfile
)
$ErrorActionPreference = 'Stop'
& $Python -c "import sys,struct;assert sys.platform=='win32' and sys.version_info[:2]==(3,10) and struct.calcsize('P')==8"
if ($LASTEXITCODE -ne 0) { throw 'Use native Windows CPython 3.10 x64' }
$arguments = @('--python', $Python, '--output', $Output)
if ($IdentityProfile) { $arguments += @('--identity-profile', $IdentityProfile) }
& $Python (Join-Path $PSScriptRoot 'validate_wheel.py') @arguments
if ($LASTEXITCODE -ne 0) { throw 'Matrix collector failed; inspect raw evidence' }
& $Python (Join-Path $PSScriptRoot 'compare_results.py') $Output
if ($LASTEXITCODE -ne 0) { throw 'Comparison parser failed' }
# Collector success means records were written. Individual cases may FAIL.
Get-Content (Join-Path $Output 'summary.json') -Encoding utf8
