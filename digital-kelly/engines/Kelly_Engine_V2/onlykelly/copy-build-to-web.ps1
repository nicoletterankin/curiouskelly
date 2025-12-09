# Copy Unity WebGL Build to Web Public Folder
# Run this AFTER building in Unity

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   COPY UNITY BUILD TO WEB" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Paths
$BuildSource = ".\Builds\Kelly_Web_Build\Build"
$WebDestination = "..\..\..\..\..\public\unity\kelly-live\Build"

# Verify source exists
if (-not (Test-Path $BuildSource)) {
    Write-Host "`n❌ Build folder not found at: $BuildSource" -ForegroundColor Red
    Write-Host "   Run the Unity build first: Window > Kelly WebGL > Build WebGL" -ForegroundColor Yellow
    exit 1
}

# Get absolute paths
$BuildSourceFull = (Resolve-Path $BuildSource).Path
$WebDestinationFull = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $WebDestination))

Write-Host "`nSource: $BuildSourceFull" -ForegroundColor Gray
Write-Host "Destination: $WebDestinationFull" -ForegroundColor Gray

# Create destination if needed
if (-not (Test-Path $WebDestinationFull)) {
    New-Item -ItemType Directory -Path $WebDestinationFull -Force | Out-Null
    Write-Host "`n✅ Created destination folder" -ForegroundColor Green
}

# List files to copy
Write-Host "`nFiles to copy:" -ForegroundColor Yellow
Get-ChildItem $BuildSourceFull | ForEach-Object {
    $size = "{0:N2} MB" -f ($_.Length / 1MB)
    Write-Host "   $($_.Name) ($size)"
}

# Copy files
Write-Host "`nCopying files..." -ForegroundColor Cyan
Copy-Item -Path "$BuildSourceFull\*" -Destination $WebDestinationFull -Force -Recurse

Write-Host "`n✅ Build copied successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "   1. Start local server: cd public && npx http-server -p 3000 -c-1 --cors"
Write-Host "   2. Open: http://localhost:3000/unity-test.html"
Write-Host "   3. Kelly should now have proper colors!"

Write-Host "`n═══════════════════════════════════════════════════════════" -ForegroundColor Cyan



