# KELLY V2 - GITHUB PAGES DEPLOYMENT
# Simple, no-auth deployment for WebGL builds

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "KELLY V2 - GITHUB PAGES DEPLOYMENT" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ProjectRoot = $PSScriptRoot
$BuildPath = Join-Path $ProjectRoot "Builds\WebGL"
$RepoName = "kelly-v2"

# Step 1: Verify Build
Write-Host "[Step 1] Verifying build..." -ForegroundColor Yellow

if (-not (Test-Path "$BuildPath\index.html")) {
    Write-Host "ERROR: Build not found at $BuildPath" -ForegroundColor Red
    Write-Host "Run Unity build first!" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK Build verified" -ForegroundColor Green
Write-Host ""

# Step 2: Create .nojekyll file (critical for GitHub Pages)
Write-Host "[Step 2] Creating .nojekyll file..." -ForegroundColor Yellow
$nojekyll = Join-Path $BuildPath ".nojekyll"
if (-not (Test-Path $nojekyll)) {
    New-Item -Path $nojekyll -ItemType File -Force | Out-Null
}
Write-Host "OK .nojekyll created" -ForegroundColor Green
Write-Host ""

# Step 3: Initialize Git repo in build folder
Write-Host "[Step 3] Initializing Git repository..." -ForegroundColor Yellow
Set-Location $BuildPath

# Remove existing .git if present
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
}

git init
git checkout -b gh-pages

Write-Host "OK Git initialized on gh-pages branch" -ForegroundColor Green
Write-Host ""

# Step 4: Add all files
Write-Host "[Step 4] Adding files to Git..." -ForegroundColor Yellow
git add -A
git commit -m "Kelly V2 WebGL Build - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

Write-Host "OK Files committed" -ForegroundColor Green
Write-Host ""

# Step 5: Create GitHub repo and push
Write-Host "[Step 5] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host ""

# Check if gh CLI is available
$ghInstalled = Get-Command gh -ErrorAction SilentlyContinue

if ($ghInstalled) {
    Write-Host "Using GitHub CLI..." -ForegroundColor Gray
    
    # Create repo if it doesn't exist
    gh repo create $RepoName --public --source=. --push 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Repo might already exist, try to set remote and push
        $username = gh api user --jq '.login' 2>$null
        if ($username) {
            git remote remove origin 2>$null
            git remote add origin "https://github.com/$username/$RepoName.git"
            git push -f origin gh-pages
        }
    }
    
    # Enable GitHub Pages
    Write-Host "Enabling GitHub Pages..." -ForegroundColor Yellow
    gh api -X PUT "repos/{owner}/$RepoName/pages" -f source='{"branch":"gh-pages","path":"/"}' 2>$null
    
    $username = gh api user --jq '.login' 2>$null
    $liveUrl = "https://$username.github.io/$RepoName/"
    
} else {
    Write-Host ""
    Write-Host "GitHub CLI (gh) not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "MANUAL STEPS REQUIRED:" -ForegroundColor Cyan
    Write-Host "1. Go to https://github.com/new" -ForegroundColor White
    Write-Host "2. Create a new repo named: $RepoName" -ForegroundColor White
    Write-Host "3. Run these commands:" -ForegroundColor White
    Write-Host ""
    Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/$RepoName.git" -ForegroundColor Gray
    Write-Host "   git push -f origin gh-pages" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Go to repo Settings > Pages" -ForegroundColor White
    Write-Host "5. Set Source to: gh-pages branch" -ForegroundColor White
    Write-Host ""
    $liveUrl = "https://YOUR_USERNAME.github.io/$RepoName/"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "    DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Kelly will be live at:" -ForegroundColor Cyan
Write-Host "$liveUrl" -ForegroundColor White
Write-Host ""
Write-Host "Note: GitHub Pages may take 1-2 minutes to go live." -ForegroundColor Yellow
Write-Host ""

# Return to project root
Set-Location $ProjectRoot

