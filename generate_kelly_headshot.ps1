# Generate Perfect Headshot 2 Photo for Kelly
param(
    [string]$OutputPath = "projects\Kelly\Ref\kelly_headshot_perfect_4k.png",
    [int]$Width = 4096,
    [int]$Height = 4096
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Kelly Perfect Headshot 2 Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python helper is available
$pythonHelper = "tools\generate_vertex_image_with_references.py"
$pythonCmd = $null

$pythonCandidates = @("python", "python3", "py")
foreach ($candidate in $pythonCandidates) {
    try {
        $cmd = Get-Command $candidate -ErrorAction Stop
        if ($cmd) {
            $pythonCmd = if ($cmd.Name -eq "py") { "py -3" } else { $candidate }
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python 3.11+" -ForegroundColor Red
    Write-Host ""
    Write-Host "Using manual method instead..." -ForegroundColor Yellow
    $pythonCmd = $null
}

# Create output directory if it doesn't exist
$outputDir = Split-Path -Parent $OutputPath
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "Created directory: $outputDir" -ForegroundColor Green
}

# Perfect Headshot 2 Prompt
$prompt = "Professional headshot portrait photograph of Kelly, a friendly female teacher in her late 20s. Front-facing camera angle, looking directly at viewer. Neutral expression with closed mouth, gentle smile, natural and approachable. Oval face shape, warm brown eyes, long wavy dark brown hair styled professionally. Clean professional appearance, simple background, studio lighting with soft even illumination. High resolution, photorealistic, crisp details, professional photography style. Perfect for 3D head reconstruction - front view, clear facial features, no shadows on face."

Write-Host "Prompt:" -ForegroundColor Yellow
Write-Host $prompt -ForegroundColor Gray
Write-Host ""

# Check if reference images exist
$referenceDir = "projects\Kelly\Ref"
$referenceImages = @()

if (Test-Path $referenceDir) {
    $existingRefs = Get-ChildItem -Path $referenceDir -Filter "*.png" -ErrorAction SilentlyContinue | 
                    Where-Object { $_.Name -like "*headshot*" -or $_.Name -like "*kelly*" } |
                    Select-Object -First 3
    
    if ($existingRefs) {
        Write-Host "Found reference images, will use for consistency:" -ForegroundColor Cyan
        foreach ($ref in $existingRefs) {
            Write-Host "   - $($ref.Name)" -ForegroundColor Gray
            $referenceImages += $ref.FullName
        }
    }
}

# Generate using Vertex AI if Python helper available
if ($pythonCmd -and (Test-Path $pythonHelper)) {
    Write-Host "Generating image using Vertex AI Imagen 3.0..." -ForegroundColor Green
    Write-Host ""
    
    $refArgs = ""
    if ($referenceImages.Count -gt 0) {
        $refArgs = "--reference-images " + ($referenceImages -join " ")
    }
    
    $fullCmd = "$pythonCmd `"$pythonHelper`" --prompt `"$prompt`" --output `"$OutputPath`" --width $Width --height $Height $refArgs"
    
    Write-Host "Running: $fullCmd" -ForegroundColor Gray
    Write-Host ""
    
    Invoke-Expression $fullCmd
    
    if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputPath)) {
        Write-Host ""
        Write-Host "SUCCESS! Generated photo saved to:" -ForegroundColor Green
        Write-Host "   $OutputPath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Yellow
        Write-Host "   1. Open Character Creator 5" -ForegroundColor White
        Write-Host "   2. Go to Headshot 2 tab" -ForegroundColor White
        Write-Host "   3. Click Load Photo and select the file above" -ForegroundColor White
        Write-Host "   4. Set Quality: Ultra High" -ForegroundColor White
        Write-Host "   5. Click Generate" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "Generation failed. Use manual method below..." -ForegroundColor Yellow
    }
} else {
    Write-Host "Python helper not available. Using manual method..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Manual Generation Instructions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 1: Bing Image Creator (FREE - Recommended)" -ForegroundColor Yellow
Write-Host "   1. Go to: https://www.bing.com/images/create" -ForegroundColor Cyan
Write-Host "   2. Sign in with Microsoft account" -ForegroundColor White
Write-Host "   3. Paste this prompt:" -ForegroundColor White
Write-Host "   $prompt" -ForegroundColor Gray
Write-Host "   4. Click Generate" -ForegroundColor White
Write-Host "   5. Download the best result" -ForegroundColor White
Write-Host "   6. Save as: $OutputPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 2: Leonardo.ai (150 free tokens/day)" -ForegroundColor Yellow
Write-Host "   1. Go to: https://leonardo.ai" -ForegroundColor Cyan
Write-Host "   2. Sign up/login" -ForegroundColor White
Write-Host "   3. Go to Image Generation" -ForegroundColor White
Write-Host "   4. Paste prompt above" -ForegroundColor White
Write-Host "   5. Model: Leonardo Diffusion XL" -ForegroundColor White
Write-Host "   6. Aspect Ratio: 1:1" -ForegroundColor White
Write-Host "   7. Resolution: 1024x1024 (or higher)" -ForegroundColor White
Write-Host "   8. Generate and download" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Perfect Headshot 2 Photo Requirements:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Front-facing (looking at camera)" -ForegroundColor Green
Write-Host "Neutral expression (mouth closed)" -ForegroundColor Green
Write-Host "Even lighting (no harsh shadows)" -ForegroundColor Green
Write-Host "High resolution (4K+ recommended)" -ForegroundColor Green
Write-Host "Clear facial features" -ForegroundColor Green
Write-Host "Simple background" -ForegroundColor Green
Write-Host ""
