param(
    [string]$UnityPath = "C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Unity.exe",
    [string]$ProjectPath,
    [string]$Version = "kelly-v1",
    [switch]$NoSmokeTest
)

if (-not $ProjectPath) {
    $ProjectPath = Join-Path -Path $PSScriptRoot -ChildPath "..\digital-kelly\engines\kelly_unity_player"
    $ProjectPath = (Resolve-Path $ProjectPath).Path
}

if (-not (Test-Path $UnityPath)) {
    throw "Unity executable not found at '$UnityPath'. Update -UnityPath to match your installed version."
}

if (-not (Test-Path $ProjectPath)) {
    throw "Unity project path not found at '$ProjectPath'."
}

$env:KELLY_WEBGL_VERSION = $Version
$logDirectory = Join-Path $ProjectPath "Builds"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ("kelly-webgl-{0}.log" -f $Version)

Write-Host "🛠️  Building Kelly WebGL bundle '$Version'..."
Write-Host "   • Unity editor  : $UnityPath"
Write-Host "   • Project path  : $ProjectPath"
Write-Host "   • Log file      : $logPath"

$arguments = @(
    "-quit",
    "-batchmode",
    "-projectPath `"$ProjectPath`"",
    "-executeMethod Kelly.Editor.WebGLBuild.Build",
    "-logFile `"$logPath`""
)

& $UnityPath $arguments
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "Unity exited with code $exitCode. See log at $logPath."
}

$buildOutput = Join-Path $ProjectPath ("Builds\WebGL\{0}" -f $Version)
if (-not (Test-Path $buildOutput)) {
    throw "Expected build output folder not found at '$buildOutput'."
}

$sizeBytes = (Get-ChildItem -Path $buildOutput -Recurse | Measure-Object -Property Length -Sum).Sum
$sizeMB = "{0:N2}" -f ($sizeBytes / 1MB)

Write-Host "✅  Build completed. Output folder:"
Write-Host "   $buildOutput ($sizeMB MB)"
Write-Host "   Log saved to $logPath"

if (-not $NoSmokeTest) {
    Write-Host ""
    Write-Host "📦  Quick smoke test:"
    Write-Host "   npm install -g serve  # once"
    Write-Host ("   serve `"{0}`"" -f $buildOutput)
    Write-Host "   → Visit http://localhost:3000 to confirm the loader appears."
}

