# Deploy Unity WebGL Build to public/unity/kelly-v1/
# This script builds the Unity WebGL project and deploys it to the public folder

param(
    [string]$UnityPath,
    [string]$Version = "kelly-v1",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Find Unity installation if not provided
if (-not $UnityPath) {
    Write-Host "[*] Searching for Unity installation..."
    
    # Common Unity Hub paths
    $unityHubPaths = @(
        "C:\Program Files\Unity\Hub\Editor",
        "C:\Program Files (x86)\Unity\Hub\Editor"
    )
    
    $foundUnity = $null
    foreach ($hubPath in $unityHubPaths) {
        if (Test-Path $hubPath) {
            # Look for Unity 6.x or 2022.3.x
            $editors = Get-ChildItem -Path $hubPath -Directory | Where-Object {
                $_.Name -match "^(6000|2022\.3)" -or $_.Name -match "^6\."
            } | Sort-Object Name -Descending
            
            if ($editors) {
                $latestEditor = $editors[0]
                $unityExe = Join-Path $latestEditor.FullName "Editor\Unity.exe"
                if (Test-Path $unityExe) {
                    $foundUnity = $unityExe
                    Write-Host "   [OK] Found: $unityExe"
                    break
                }
            }
        }
    }
    
    if (-not $foundUnity) {
        throw "Unity not found. Please specify -UnityPath parameter."
    }
    
    $UnityPath = $foundUnity
}

# Project paths
$projectRoot = Split-Path -Parent $PSScriptRoot
$unityProjectPath = Join-Path $projectRoot "digital-kelly\engines\kelly_unity_player"
$buildOutputPath = Join-Path $unityProjectPath "Builds\WebGL\$Version"
$deployPath = Join-Path $projectRoot "public\unity\$Version"

# Verify Unity project exists
if (-not (Test-Path $unityProjectPath)) {
    throw "Unity project not found at: $unityProjectPath"
}

Write-Host ""
Write-Host "[*] Unity WebGL Build & Deploy"
Write-Host "   Unity: $UnityPath"
Write-Host "   Project: $unityProjectPath"
Write-Host "   Version: $Version"
Write-Host ""

# Step 1: Build Unity WebGL
if (-not $SkipBuild) {
    Write-Host "[*] Step 1: Building Unity WebGL..."
    
    $buildScript = Join-Path $projectRoot "scripts\build_unity_webgl.ps1"
    if (-not (Test-Path $buildScript)) {
        throw "Build script not found: $buildScript"
    }
    
    & $buildScript -UnityPath $UnityPath -Version $Version -NoSmokeTest
    if ($LASTEXITCODE -ne 0) {
        throw "Unity build failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "   [OK] Build completed"
} else {
    Write-Host "[*] Skipping build (using existing build)"
}

# Verify build output exists
if (-not (Test-Path $buildOutputPath)) {
    throw "Build output not found at: $buildOutputPath. Build may have failed."
}

# Step 2: Prepare deployment directory
Write-Host ""
Write-Host "[*] Step 2: Preparing deployment directory..."
if (Test-Path $deployPath) {
    Write-Host "   Cleaning existing deployment..."
    Remove-Item -Path $deployPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $deployPath | Out-Null
Write-Host "   [OK] Deployment directory ready: $deployPath"

# Step 3: Copy build files
Write-Host ""
Write-Host "[*] Step 3: Copying build files..."
$buildFiles = Get-ChildItem -Path $buildOutputPath -Recurse -File
$fileCount = $buildFiles.Count
$copied = 0

foreach ($file in $buildFiles) {
    $relativePath = $file.FullName.Substring($buildOutputPath.Length + 1)
    $destPath = Join-Path $deployPath $relativePath
    $destDir = Split-Path -Parent $destPath
    
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    }
    
    Copy-Item -Path $file.FullName -Destination $destPath -Force
    $copied++
    
    if ($copied % 10 -eq 0) {
        Write-Host "   Copied $copied/$fileCount files..."
    }
}

Write-Host "   [OK] Copied $fileCount files"

# Step 4: Ensure kbridge.js is present
Write-Host ""
Write-Host "[*] Step 4: Verifying messaging bridge..."
$kbridgePath = Join-Path $deployPath "kbridge.js"
if (-not (Test-Path $kbridgePath)) {
    Write-Host "   Copying kbridge.js from public folder..."
    $sourceKbridge = Join-Path $projectRoot "public\unity\$Version\kbridge.js"
    if (Test-Path $sourceKbridge) {
        Copy-Item -Path $sourceKbridge -Destination $kbridgePath -Force
        Write-Host "   [OK] kbridge.js copied"
    } else {
        Write-Host "   [WARN] kbridge.js not found. It should be in the build output."
    }
} else {
    Write-Host "   [OK] kbridge.js already present"
}

# Step 5: Update index.html to include kbridge.js
Write-Host ""
Write-Host "[*] Step 5: Updating index.html..."
$indexHtmlPath = Join-Path $deployPath "index.html"
if (Test-Path $indexHtmlPath) {
    $indexContent = Get-Content -Path $indexHtmlPath -Raw
    
    # Check if kbridge.js is already included
    if ($indexContent -notmatch "kbridge\.js") {
        # Find the closing </body> tag and insert kbridge.js before it
        if ($indexContent -match "(</body>)") {
            $kbridgeScript = "    <script src=`"./kbridge.js`" type=`"module`"></script>`n$($matches[1])"
            $indexContent = $indexContent -replace "</body>", $kbridgeScript
            Set-Content -Path $indexHtmlPath -Value $indexContent -NoNewline
            Write-Host "   [OK] Added kbridge.js to index.html"
        } else {
            Write-Host "   [WARN] Could not find </body> tag in index.html"
        }
    } else {
        Write-Host "   [OK] kbridge.js already referenced in index.html"
    }
} else {
    Write-Host "   [WARN] index.html not found in build output"
}

# Step 6: Calculate deployment size
Write-Host ""
Write-Host "[*] Step 6: Deployment summary..."
$deploySize = (Get-ChildItem -Path $deployPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
$deploySizeMB = "{0:N2}" -f ($deploySize / 1MB)

Write-Host ""
Write-Host "[OK] Deployment complete!"
Write-Host "   Location: $deployPath"
Write-Host "   Size: $deploySizeMB MB"
Write-Host "   Files: $fileCount"
Write-Host ""
Write-Host "[*] Next steps:"
Write-Host "   1. Test locally: serve the public folder and visit /unity/kelly-v1/"
Write-Host "   2. Deploy to production: push changes and deploy to hosting"
Write-Host "   3. Verify: visit curiouskelly.com/lesson-player to see Kelly in action"
Write-Host ""

