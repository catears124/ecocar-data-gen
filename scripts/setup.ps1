<#
One-time setup on Windows: creates .venv, installs the harness and the CARLA client that
matches the server, then runs a self-check.

  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-CarlaRoot C:\CARLA_0.9.16] [-PythonVersion 3.12]
#>
param(
    [string]$CarlaRoot = $env:CARLA_ROOT,
    [string]$PythonVersion = "",
    [string]$CarlaVersion = "0.9.16"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# pick an interpreter supported by the CARLA wheels (3.10 - 3.12)
$candidates = if ($PythonVersion) { @($PythonVersion) } else { @("3.12", "3.11", "3.10") }
$py = $null
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher 'py' was not found. Install Python 3.12 from python.org and tick 'py launcher'."
}
$ErrorActionPreference = "Continue"   # a missing version writes to stderr; keep trying the next one
foreach ($v in $candidates) {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $v; break }
}
$ErrorActionPreference = "Stop"
if (-not $py) { throw "Python 3.10-3.12 not found (install from python.org, tick 'py launcher')." }
Write-Host "Using Python $py"

if (-not (Test-Path .venv)) { & py "-$py" -m venv .venv }
$vpy = Join-Path $repo ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip -q
& $vpy -m pip install -e ".[dev]" -q
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# CARLA client: PyPI first, then the wheel shipped with the simulator
# Windows PowerShell 5.1 turns redirected stderr into terminating errors under "Stop"
$ErrorActionPreference = "Continue"
& $vpy -m pip install "carla==$CarlaVersion" -q 2>$null
$carlaFromPyPI = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $carlaFromPyPI) {
    if (-not $CarlaRoot) { $CarlaRoot = @("$env:USERPROFILE\CARLA_$CarlaVersion", "C:\CARLA_$CarlaVersion") | Where-Object { Test-Path $_ } | Select-Object -First 1 }
    $tag = "cp" + $py.Replace(".", "")
    $whl = if ($CarlaRoot) { Get-ChildItem "$CarlaRoot\PythonAPI\carla\dist\carla-*-$tag-*win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($whl) {
        & $vpy -m pip install $whl.FullName -q
    } else {
        Write-Host "`nNOTE: no carla==$CarlaVersion client for Python $py on PyPI or in CARLA_ROOT."
        Write-Host "      The harness is installed; use --backend bicycle. To add CARLA later, pass -CarlaRoot and re-run."
    }
}

& $vpy -m autocross check
Write-Host "`nSetup complete. Activate with:  .venv\Scripts\Activate.ps1"
Write-Host "Dev loop without CARLA:  python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --backend bicycle --view"
