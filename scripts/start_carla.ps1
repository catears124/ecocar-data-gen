<#
Starts the CARLA server with laptop-friendly settings.

  scripts\start_carla.ps1                 # windowed 1280x720, Low quality (fine on an RTX 3070 laptop)
  scripts\start_carla.ps1 -Headless       # no window; pair with --no-render for fastest batch scoring
  scripts\start_carla.ps1 -Quality Epic   # nicer visuals for demos / pitch videos
#>
param(
    [string]$CarlaRoot = $env:CARLA_ROOT,
    [switch]$Headless,
    [ValidateSet("Low", "Epic")][string]$Quality = "Low",
    [int]$Port = 2000,
    [string]$CarlaVersion = "0.9.16"
)
if (-not $CarlaRoot) {
    $CarlaRoot = @("$env:USERPROFILE\CARLA_$CarlaVersion", "C:\CARLA_$CarlaVersion", "D:\CARLA_$CarlaVersion") |
        Where-Object { Test-Path "$_\CarlaUE4.exe" } | Select-Object -First 1
}
$exe = Join-Path "$CarlaRoot" "CarlaUE4.exe"
if (-not (Test-Path $exe)) { throw "CarlaUE4.exe not found. Pass -CarlaRoot or set CARLA_ROOT." }

$carlaArgs = @("-quality-level=$Quality", "-carla-rpc-port=$Port", "-nosound")
if ($Headless) { $carlaArgs += "-RenderOffScreen" } else { $carlaArgs += @("-windowed", "-ResX=1280", "-ResY=720") }
$p = Start-Process -FilePath $exe -ArgumentList $carlaArgs -PassThru
Write-Host "CARLA started (PID $($p.Id)) on port $Port. First launch can take ~30 s; check with: python -m autocross check"
