[CmdletBinding()]
param([string]$Python = 'C:\b\be\Scripts\python.exe', [string]$Output = 'C:\b\raw\preflight.json')
$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem
$system = Get-CimInstance Win32_ComputerSystem
$volume = Get-Volume -DriveLetter C
$probe = 'C:\b\write-probe.tmp'
Set-Content -LiteralPath $probe -Value 'writable' -Encoding utf8
Remove-Item -LiteralPath $probe
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$compiler = Join-Path $vs 'VC\Tools\MSVC'
$sdk = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\Include' -Directory | Select-Object -ExpandProperty Name
$py = & $Python -c "import json,sys,struct;print(json.dumps(dict(version=sys.version,abi='cp310' if sys.version_info[:2]==(3,10) else 'WRONG',bits=struct.calcsize('P')*8,platform=sys.platform)))" | ConvertFrom-Json
$overrides = @{}
foreach ($name in @('PYTORCH_AOTRITON_COMMIT','PYTORCH_AOTRITON_BASE_URL','AOTRITON_INSTALLED_PREFIX','AOTRITON_INSTALL_FROM_SOURCE','CC','CXX','CMAKE_ARGS','PYTORCH_ROCM_ARCH','HSA_OVERRIDE_GFX_VERSION')) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($null -ne $value) { $overrides[$name] = $value }
}
$report = [ordered]@{
    run_id = [guid]::NewGuid().ToString(); utc = [DateTime]::UtcNow.ToString('o')
    os = @{caption=$os.Caption;version=$os.Version;build=$os.BuildNumber}
    cpu = (Get-CimInstance Win32_Processor).Name.Trim()
    gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate)
    ram_bytes = $system.TotalPhysicalMemory; free_ram_kib = $os.FreePhysicalMemory
    disk_bytes = $volume.Size; free_disk_bytes = $volume.SizeRemaining
    python = $py; powershell = $PSVersionTable.PSVersion.ToString()
    git = (& git --version); cmake = (& cmake --version)[0]; ninja = (& ninja --version)
    vs_path = $vs; msvc = @(Get-ChildItem $compiler -Directory | Select-Object -ExpandProperty Name)
    windows_sdk = @($sdk); short_path_writable = $true; inherited_overrides = $overrides
    gpu_architecture = 'PENDING HIP runtime probe; WMI alone does not establish gfx1151'
    rocm_devel = 'PENDING isolated SDK installation; global inference packages are not build prerequisites'
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
if ($py.abi -ne 'cp310' -or $py.bits -ne 64 -or $py.platform -ne 'win32') { throw 'CPython 3.10 x64 required' }
if (-not $vs -or -not $sdk) { throw 'Critical MSVC/Windows SDK prerequisite missing' }
if ([version]$report.ninja -lt [version]'1.13.1') { throw 'Ninja >=1.13.1 required' }
if ($volume.SizeRemaining -lt 100GB) { throw 'Less than bounded-build 100 GiB free disk floor' }
