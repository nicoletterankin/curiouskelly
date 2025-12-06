# Script to prepare a consolidated build for Cloudflare Pages Smoke Testing
# This ensures that the 'unified-app' (in /lessons) can find assets (in /lesson-player)

$deployDir = "dist-smoke-test"

Write-Host "🧹 Cleaning up previous build..." -ForegroundColor Gray
if (Test-Path $deployDir) { Remove-Item $deployDir -Recurse -Force }
New-Item -ItemType Directory -Path $deployDir | Out-Null

Write-Host "📂 Copying 'lessons' app..." -ForegroundColor Cyan
# Exclude large ps1/py files if needed, but full copy is safer for prototype
Copy-Item -Path "lessons" -Destination "$deployDir\lessons" -Recurse

Write-Host "🎨 Copying 'lesson-player' assets..." -ForegroundColor Cyan
Copy-Item -Path "lesson-player" -Destination "$deployDir\lesson-player" -Recurse

Write-Host "🔗 Creating root entry point..." -ForegroundColor Cyan
# Create a simple redirect at the root so you don't have to type /lessons/unified-app.html
$redirectHtml = '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=lessons/unified-app.html">
    <title>Redirecting to Curious Kelly...</title>
</head>
<body>
    <p>Redirecting to <a href="lessons/unified-app.html">Curious Kelly</a>...</p>
</body>
</html>'
Set-Content -Path "$deployDir\index.html" -Value $redirectHtml

Write-Host ""
Write-Host "✅ Build Complete in folder: $deployDir" -ForegroundColor Green
Write-Host "---------------------------------------------------"
Write-Host "🚀 To Deploy to Cloudflare Pages:" -ForegroundColor Yellow
Write-Host "   npx wrangler pages deploy $deployDir --project-name curiouskelly-smoke-test" -ForegroundColor White
Write-Host ""



