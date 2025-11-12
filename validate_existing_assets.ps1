# Automated Asset Validator
# Validates all Reinmaker assets against quality framework
# No API calls required - validates existing assets

param(
    [string]$AssetRoot = ".",
    [string]$OutputDir = "validation_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [switch]$GenerateHTML = $true,
    [switch]$ValidateAll = $true
)

$ErrorActionPreference = "Continue"

# Load reference image functions from generate_assets.ps1
if (Test-Path ".\generate_assets.ps1") {
    . .\generate_assets.ps1
    Write-Host "✅ Loaded generate_assets.ps1 functions" -ForegroundColor Green
} else {
    Write-Host "⚠️ Warning: generate_assets.ps1 not found - reference image loading may fail" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "AUTOMATED ASSET VALIDATOR" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Asset Root: $AssetRoot" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Asset definitions from REINMAKER_COMPLETE_ASSET_LIST.md
$assetDefinitions = @(
    # Category A: Core Gameplay Sprites
    @{
        ID = "A1"
        Name = "Player: Kelly (Runner)"
        Path = "assets\player.png"
        ExpectedSize = @{ Width = 1024; Height = 1280 }
        RequiresCharacter = $true
        Variant = "Reinmaker"
        Priority = "HIGH"
    },
    @{
        ID = "A2"
        Name = "Obstacle: Knowledge Shards"
        Path = "assets\obstacle.png"
        ExpectedSize = @{ Width = 512; Height = 512 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "A3"
        Name = "Ground Stripe"
        Path = "assets\ground_stripe.png"
        ExpectedSize = @{ Width = 60; Height = 6 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    # Category B: Background & Environment
    @{
        ID = "B1"
        Name = "Parallax Skyline"
        Path = "assets\bg.png"
        ExpectedSize = @{ Width = 1024; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "B2"
        Name = "Ground Texture"
        Path = "assets\ground_tex.png"
        ExpectedSize = @{ Width = 512; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    # Category C: UI & Meta
    @{
        ID = "C1"
        Name = "Logo / Title Card (1280x720)"
        Path = "marketing\cover-1280x720.png"
        ExpectedSize = @{ Width = 1280; Height = 720 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "HIGH"
    },
    @{
        ID = "C1b"
        Name = "Logo / Title Card (square-600)"
        Path = "marketing\square-600.png"
        ExpectedSize = @{ Width = 600; Height = 600 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "C2"
        Name = "Favicon"
        Path = "assets\favicon.png"
        ExpectedSize = @{ Width = 256; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    # Category D: Lore Collectibles
    @{
        ID = "D1-Light"
        Name = "Knowledge Stone: Light"
        Path = "assets\stones\stone_light.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Stone"
        Name = "Knowledge Stone: Stone"
        Path = "assets\stones\stone_stone.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Metal"
        Name = "Knowledge Stone: Metal"
        Path = "assets\stones\stone_metal.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Code"
        Name = "Knowledge Stone: Code"
        Path = "assets\stones\stone_code.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Air"
        Name = "Knowledge Stone: Air"
        Path = "assets\stones\stone_air.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Water"
        Name = "Knowledge Stone: Water"
        Path = "assets\stones\stone_water.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D1-Fire"
        Name = "Knowledge Stone: Fire"
        Path = "assets\stones\stone_fire.png"
        ExpectedSize = @{ Width = 64; Height = 64 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Light"
        Name = "Tribe Banner: Light"
        Path = "assets\banners\banner_light.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Stone"
        Name = "Tribe Banner: Stone"
        Path = "assets\banners\banner_stone.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Metal"
        Name = "Tribe Banner: Metal"
        Path = "assets\banners\banner_metal.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Code"
        Name = "Tribe Banner: Code"
        Path = "assets\banners\banner_code.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Air"
        Name = "Tribe Banner: Air"
        Path = "assets\banners\banner_air.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Water"
        Name = "Tribe Banner: Water"
        Path = "assets\banners\banner_water.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    @{
        ID = "D2-Fire"
        Name = "Tribe Banner: Fire"
        Path = "assets\banners\banner_fire.png"
        ExpectedSize = @{ Width = 128; Height = 256 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    # Category E: Narrative Inserts
    @{
        ID = "E1"
        Name = "Opening Splash"
        Path = "marketing\splash_intro.png"
        ExpectedSize = @{ Width = 1280; Height = 720 }
        RequiresCharacter = $true
        Variant = "Reinmaker"
        Priority = "HIGH"
    },
    @{
        ID = "E2"
        Name = "Game Over Panel"
        Path = "assets\gameover_bg.png"
        ExpectedSize = @{ Width = 960; Height = 540 }
        RequiresCharacter = $false
        Variant = "N/A"
        Priority = "MEDIUM"
    },
    # Category F: Marketing & Storefront
    @{
        ID = "F1"
        Name = "Itch.io Banner"
        Path = "marketing\itch-banner-1920x480.png"
        ExpectedSize = @{ Width = 1920; Height = 480 }
        RequiresCharacter = $true
        Variant = "Reinmaker"
        Priority = "HIGH"
    }
)

# Function to get image dimensions using .NET
function Get-ImageDimensions {
    param([string]$ImagePath)
    
    try {
        Add-Type -AssemblyName System.Drawing
        $image = [System.Drawing.Image]::FromFile((Resolve-Path $ImagePath))
        $dimensions = @{
            Width = $image.Width
            Height = $image.Height
            Format = $image.RawFormat.ToString()
        }
        $image.Dispose()
        return $dimensions
    } catch {
        return $null
    }
}

# Function to validate a single asset
function Validate-Asset {
    param(
        [hashtable]$AssetDef,
        [string]$AssetRoot
    )
    
    $result = @{
        ID = $AssetDef.ID
        Name = $AssetDef.Name
        Path = $AssetDef.Path
        FullPath = Join-Path $AssetRoot $AssetDef.Path
        ExpectedSize = $AssetDef.ExpectedSize
        RequiresCharacter = $AssetDef.RequiresCharacter
        Variant = $AssetDef.Variant
        Priority = $AssetDef.Priority
        ValidationDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Status = "UNKNOWN"
        Issues = @()
        Warnings = @()
        TechnicalChecks = @{
            FileExists = $false
            FileSize = 0
            FileSizeKB = 0
            FileModified = $null
            Dimensions = $null
            DimensionsMatch = $false
            AspectRatioMatch = $false
            IsValidImage = $false
        }
        QualityLevel = "UNKNOWN"
        Recommendations = @()
    }
    
    $fullPath = Join-Path $AssetRoot $AssetDef.Path
    
    # Check if file exists
    if (-not (Test-Path $fullPath)) {
        $result.Status = "MISSING"
        $result.QualityLevel = "Unacceptable"
        $result.Issues += "File not found: $($AssetDef.Path)"
        return $result
    }
    
    $result.Status = "EXISTS"
    $result.TechnicalChecks.FileExists = $true
    
    # Get file info
    $fileInfo = Get-Item $fullPath
    $result.TechnicalChecks.FileSize = $fileInfo.Length
    $result.TechnicalChecks.FileSizeKB = [math]::Round($fileInfo.Length / 1024, 2)
    $result.TechnicalChecks.FileModified = $fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    
    # Check file size (warn if suspiciously small or large)
    if ($fileInfo.Length -lt 1024) {
        $result.Warnings += "File size suspiciously small: $($result.TechnicalChecks.FileSizeKB) KB"
    }
    if ($fileInfo.Length -gt 10MB) {
        $result.Warnings += "File size very large: $([math]::Round($fileInfo.Length / 1MB, 2)) MB"
    }
    
    # Get image dimensions
    $dimensions = Get-ImageDimensions $fullPath
    if ($null -eq $dimensions) {
        $result.Issues += "Could not read image dimensions - file may be corrupted or not an image"
        $result.QualityLevel = "Unacceptable"
        return $result
    }
    
    $result.TechnicalChecks.Dimensions = $dimensions
    $result.TechnicalChecks.IsValidImage = $true
    
    # Check if dimensions match expected
    $expectedWidth = $AssetDef.ExpectedSize.Width
    $expectedHeight = $AssetDef.ExpectedSize.Height
    
    if ($dimensions.Width -eq $expectedWidth -and $dimensions.Height -eq $expectedHeight) {
        $result.TechnicalChecks.DimensionsMatch = $true
    } else {
        $result.Warnings += "Dimensions mismatch: Expected $expectedWidth x $expectedHeight, got $($dimensions.Width) x $($dimensions.Height)"
    }
    
    # Check aspect ratio (allow 5% tolerance)
    $expectedRatio = $expectedWidth / $expectedHeight
    $actualRatio = $dimensions.Width / $dimensions.Height
    $ratioDiff = [math]::Abs($expectedRatio - $actualRatio) / $expectedRatio
    
    if ($ratioDiff -le 0.05) {
        $result.TechnicalChecks.AspectRatioMatch = $true
    } else {
        $result.Warnings += "Aspect ratio mismatch: Expected $([math]::Round($expectedRatio, 2)), got $([math]::Round($actualRatio, 2))"
    }
    
    # Determine quality level based on technical checks
    if ($result.Issues.Count -eq 0 -and $result.Warnings.Count -eq 0) {
        $result.QualityLevel = "Good"
    } elseif ($result.Issues.Count -eq 0) {
        $result.QualityLevel = "Acceptable"
    } else {
        $result.QualityLevel = "Unacceptable"
    }
    
    # Add recommendations
    if ($result.Status -eq "MISSING") {
        $result.Recommendations += "Generate asset: $($AssetDef.Name)"
    }
    if (-not $result.TechnicalChecks.DimensionsMatch) {
        $result.Recommendations += "Regenerate with correct dimensions: $expectedWidth x $expectedHeight"
    }
    if ($result.RequiresCharacter -and $result.QualityLevel -ne "Unacceptable") {
        $result.Recommendations += "Validate character consistency against reference images"
    }
    
    return $result
}

# Load reference images if available
Write-Host "Loading reference images..." -ForegroundColor Cyan
$refImages = @()
try {
    if (Get-Command Get-CharacterReferences -ErrorAction SilentlyContinue) {
        $refImages = Get-CharacterReferences
        Write-Host "✅ Loaded $($refImages.Count) character reference image(s)" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Get-CharacterReferences function not available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not load reference images: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Validating $($assetDefinitions.Count) assets..." -ForegroundColor Cyan
Write-Host ""

$validationResults = @()
$summary = @{
    Total = $assetDefinitions.Count
    Exists = 0
    Missing = 0
    Valid = 0
    HasWarnings = 0
    HasIssues = 0
    ByPriority = @{
        HIGH = 0
        MEDIUM = 0
    }
    ByCategory = @{
        A = 0
        B = 0
        C = 0
        D = 0
        E = 0
        F = 0
    }
}

foreach ($asset in $assetDefinitions) {
    Write-Host "Validating: $($asset.Name) ($($asset.ID))..." -ForegroundColor Yellow -NoNewline
    
    $validation = Validate-Asset -AssetDef $asset -AssetRoot $AssetRoot
    $validationResults += $validation
    
    # Update summary
    if ($validation.Status -eq "EXISTS") {
        $summary.Exists++
        if ($validation.Issues.Count -eq 0) {
            $summary.Valid++
        }
        if ($validation.Warnings.Count -gt 0) {
            $summary.HasWarnings++
        }
    } else {
        $summary.Missing++
    }
    
    if ($validation.Issues.Count -gt 0) {
        $summary.HasIssues++
    }
    
    $summary.ByPriority[$asset.Priority]++
    $category = $asset.ID.Substring(0, 1)
    if ($summary.ByCategory.ContainsKey($category)) {
        $summary.ByCategory[$category]++
    }
    
    # Display status
    $statusColor = if ($validation.Status -eq "EXISTS") { "Green" } else { "Red" }
    Write-Host " [$($validation.Status)]" -ForegroundColor $statusColor
    
    if ($validation.Warnings.Count -gt 0) {
        Write-Host "  ⚠️ Warnings: $($validation.Warnings.Count)" -ForegroundColor Yellow
    }
    if ($validation.Issues.Count -gt 0) {
        Write-Host "  ❌ Issues: $($validation.Issues.Count)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total Assets: $($summary.Total)" -ForegroundColor White
Write-Host "  ✅ Exists: $($summary.Exists)" -ForegroundColor Green
Write-Host "  ❌ Missing: $($summary.Missing)" -ForegroundColor Red
Write-Host "  ✓ Valid: $($summary.Valid)" -ForegroundColor Green
Write-Host "  ⚠️ Has Warnings: $($summary.HasWarnings)" -ForegroundColor Yellow
Write-Host "  ❌ Has Issues: $($summary.HasIssues)" -ForegroundColor Red
Write-Host ""

# Export results to JSON
$resultsPath = Join-Path $OutputDir "validation_results.json"
$validationResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $resultsPath -Encoding UTF8
Write-Host "✅ Results exported to: $resultsPath" -ForegroundColor Green

# Export summary
$summaryPath = Join-Path $OutputDir "validation_summary.json"
$summary | ConvertTo-Json -Depth 10 | Out-File -FilePath $summaryPath -Encoding UTF8
Write-Host "✅ Summary exported to: $summaryPath" -ForegroundColor Green

# Generate HTML report if requested
if ($GenerateHTML) {
    Write-Host ""
    Write-Host "Generating HTML report..." -ForegroundColor Cyan
    
    # Create HTML report (will be generated in next step)
    $htmlPath = Join-Path $OutputDir "asset_quality_report.html"
    
    # Generate HTML content
    $reportDate = Get-Date -Format 'yyyy-MM-dd HH:mm'
    $reportDateTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    
    $htmlContent = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asset Quality Report - $reportDate</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }
        .summary-card h3 { color: #333; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
        .summary-card .value { font-size: 32px; font-weight: bold; color: #007bff; }
        .summary-card.exists { border-left-color: #28a745; }
        .summary-card.exists .value { color: #28a745; }
        .summary-card.missing { border-left-color: #dc3545; }
        .summary-card.missing .value { color: #dc3545; }
        .summary-card.warnings { border-left-color: #ffc107; }
        .summary-card.warnings .value { color: #ffc107; }
        .filters { margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
        .filter-btn { padding: 8px 16px; border: 1px solid #ddd; background: white; cursor: pointer; border-radius: 4px; }
        .filter-btn.active { background: #007bff; color: white; border-color: #007bff; }
        .assets-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
        .asset-card { border: 1px solid #ddd; border-radius: 6px; overflow: hidden; background: white; }
        .asset-card.missing { opacity: 0.6; }
        .asset-card.has-issues { border-color: #dc3545; }
        .asset-card.has-warnings { border-color: #ffc107; }
        .asset-header { padding: 15px; background: #f8f9fa; border-bottom: 1px solid #ddd; }
        .asset-header h3 { font-size: 16px; color: #333; margin-bottom: 5px; }
        .asset-id { color: #666; font-size: 12px; }
        .asset-body { padding: 15px; }
        .asset-image { width: 100%; max-height: 300px; object-fit: contain; background: #f8f9fa; margin-bottom: 15px; }
        .asset-info { font-size: 14px; color: #666; }
        .asset-info .row { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .asset-info .label { font-weight: 600; }
        .status-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .status-exists { background: #28a745; color: white; }
        .status-missing { background: #dc3545; color: white; }
        .issues-list { margin-top: 15px; }
        .issues-list .issue { padding: 8px; margin-bottom: 5px; background: #fff3cd; border-left: 3px solid #ffc107; font-size: 13px; }
        .issues-list .issue.error { background: #f8d7da; border-left-color: #dc3545; }
        .recommendations { margin-top: 15px; }
        .recommendations .rec { padding: 8px; margin-bottom: 5px; background: #d1ecf1; border-left: 3px solid #17a2b8; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Asset Quality Report</h1>
        <div class="subtitle">Generated: $reportDateTime</div>
        
        <div class="summary">
            <div class="summary-card exists">
                <h3>Assets Found</h3>
                <div class="value">$($summary.Exists)</div>
            </div>
            <div class="summary-card missing">
                <h3>Missing Assets</h3>
                <div class="value">$($summary.Missing)</div>
            </div>
            <div class="summary-card">
                <h3>Valid Assets</h3>
                <div class="value">$($summary.Valid)</div>
            </div>
            <div class="summary-card warnings">
                <h3>Has Warnings</h3>
                <div class="value">$($summary.HasWarnings)</div>
            </div>
        </div>
        
        <div class="filters">
            <button class="filter-btn active" onclick="filterAssets('all')">All</button>
            <button class="filter-btn" onclick="filterAssets('exists')">Exists</button>
            <button class="filter-btn" onclick="filterAssets('missing')">Missing</button>
            <button class="filter-btn" onclick="filterAssets('has-warnings')">Has Warnings</button>
            <button class="filter-btn" onclick="filterAssets('has-issues')">Has Issues</button>
            <button class="filter-btn" onclick="filterAssets('high-priority')">High Priority</button>
        </div>
        
        <div class="assets-grid" id="assets-grid">
"@

    foreach ($result in $validationResults) {
        $fullPath = $result.FullPath
        if ($result.Status -eq "EXISTS") {
            $imageSrc = $fullPath.Replace('\', '/')
        } else {
            $imageSrc = "data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='400'%20height='300'%3E%3Crect%20fill='%23ddd'%20width='400'%20height='300'/%3E%3Ctext%20x='50%25'%20y='50%25'%20text-anchor='middle'%20dy='.3em'%20fill='%23999'%20font-size='18'%3EMissing%20Asset%3C/text%3E%3C/svg%3E"
        }
        
        $cardClass = "asset-card"
        if ($result.Status -eq "MISSING") { $cardClass += " missing" }
        if ($result.Issues.Count -gt 0) { $cardClass += " has-issues" }
        if ($result.Warnings.Count -gt 0) { $cardClass += " has-warnings" }
        
        $statusClass = if ($result.Status -eq "EXISTS") { "status-exists" } else { "status-missing" }
        $statusLower = $result.Status.ToLower()
        $priorityLower = $result.Priority.ToLower()
        $hasWarnings = ($result.Warnings.Count -gt 0).ToString().ToLower()
        $hasIssues = ($result.Issues.Count -gt 0).ToString().ToLower()
        
        $htmlContent += "            <div class=`"$cardClass`" data-status=`"$statusLower`" data-priority=`"$priorityLower`" data-has-warnings=`"$hasWarnings`" data-has-issues=`"$hasIssues`">`n"
        $htmlContent += "                <div class=`"asset-header`">`n"
        $htmlContent += "                    <h3>$($result.Name)</h3>`n"
        $htmlContent += "                    <div class=`"asset-id`">$($result.ID) | Priority: $($result.Priority)</div>`n"
        $htmlContent += "                </div>`n"
        $htmlContent += "                <div class=`"asset-body`">`n"
        $htmlContent += "                    <img src=`"$imageSrc`" alt=`"$($result.Name)`" class=`"asset-image`">`n"
        $htmlContent += "                    <div class=`"asset-info`">`n"
        $htmlContent += "                        <div class=`"row`">`n"
        $htmlContent += "                            <span class=`"label`">Status:</span>`n"
        $htmlContent += "                            <span class=`"status-badge $statusClass`">$($result.Status)</span>`n"
        $htmlContent += "                        </div>`n"
        $htmlContent += "                        <div class=`"row`">`n"
        $htmlContent += "                            <span class=`"label`">Quality Level:</span>`n"
        $htmlContent += "                            <span>$($result.QualityLevel)</span>`n"
        $htmlContent += "                        </div>`n"
        
        if ($result.Status -eq "EXISTS") {
            if ($result.TechnicalChecks.Dimensions) {
                $dim = $result.TechnicalChecks.Dimensions
                $htmlContent += "                        <div class=`"row`">`n"
                $htmlContent += "                            <span class=`"label`">Dimensions:</span>`n"
                $htmlContent += "                            <span>$($dim.Width) x $($dim.Height)</span>`n"
                $htmlContent += "                        </div>`n"
                $htmlContent += "                        <div class=`"row`">`n"
                $htmlContent += "                            <span class=`"label`">Expected:</span>`n"
                $htmlContent += "                            <span>$($result.ExpectedSize.Width) x $($result.ExpectedSize.Height)</span>`n"
                $htmlContent += "                        </div>`n"
                $htmlContent += "                        <div class=`"row`">`n"
                $htmlContent += "                            <span class=`"label`">File Size:</span>`n"
                $htmlContent += "                            <span>$($result.TechnicalChecks.FileSizeKB) KB</span>`n"
                $htmlContent += "                        </div>`n"
            }
        }
        
        $htmlContent += "                    </div>`n"
        
        if ($result.Warnings.Count -gt 0) {
            $htmlContent += "                    <div class=`"issues-list`">`n"
            foreach ($warning in $result.Warnings) {
                $warningEscaped = $warning.Replace('"', '&quot;')
                $htmlContent += "                        <div class=`"issue`">⚠️ $warningEscaped</div>`n"
            }
            $htmlContent += "                    </div>`n"
        }
        
        if ($result.Issues.Count -gt 0) {
            $htmlContent += "                    <div class=`"issues-list`">`n"
            foreach ($issue in $result.Issues) {
                $issueEscaped = $issue.Replace('"', '&quot;')
                $htmlContent += "                        <div class=`"issue error`">❌ $issueEscaped</div>`n"
            }
            $htmlContent += "                    </div>`n"
        }
        
        if ($result.Recommendations.Count -gt 0) {
            $htmlContent += "                    <div class=`"recommendations`">`n"
            foreach ($rec in $result.Recommendations) {
                $recEscaped = $rec.Replace('"', '&quot;')
                $htmlContent += "                        <div class=`"rec`">💡 $recEscaped</div>`n"
            }
            $htmlContent += "                    </div>`n"
        }
        
        $htmlContent += "                </div>`n"
        $htmlContent += "            </div>`n"
    }
    
    $htmlContent += @"
        </div>
    </div>
    
    <script>
        function filterAssets(filter) {
            const cards = document.querySelectorAll('.asset-card');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            cards.forEach(card => {
                let show = false;
                switch(filter) {
                    case 'all':
                        show = true;
                        break;
                    case 'exists':
                        show = card.dataset.status === 'exists';
                        break;
                    case 'missing':
                        show = card.dataset.status === 'missing';
                        break;
                    case 'has-warnings':
                        show = card.dataset.hasWarnings === 'true';
                        break;
                    case 'has-issues':
                        show = card.dataset.hasIssues === 'true';
                        break;
                    case 'high-priority':
                        show = card.dataset.priority === 'high';
                        break;
                }
                card.style.display = show ? 'block' : 'none';
            });
        }
    </script>`n"
    $htmlContent += "</body>`n"
    $htmlContent += "</html>`n"
    
    $htmlContent | Out-File -FilePath $htmlPath -Encoding UTF8
    Write-Host "✅ HTML report generated: $htmlPath" -ForegroundColor Green
    Write-Host "   Open in browser to view interactive report" -ForegroundColor Gray
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to: $OutputDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Review HTML report: $OutputDir\asset_quality_report.html" -ForegroundColor White
Write-Host "  2. Check JSON results: $OutputDir\validation_results.json" -ForegroundColor White
Write-Host "  3. Address missing assets and issues" -ForegroundColor White
Write-Host "  4. Regenerate assets with issues" -ForegroundColor White
Write-Host ""

