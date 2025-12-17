# ============================================================================
# 🚀 DAY 351 LAUNCH ASSET GENERATOR
# ============================================================================
# 
# This script generates ALL assets needed for Day 351 (December 17, 2025)
# Topic: "Practicing in Your Mind" (Visualization)
# 
# Usage:
#   .\scripts\generate-day-351-launch.ps1
#   .\scripts\generate-day-351-launch.ps1 -SkipAudio
#   .\scripts\generate-day-351-launch.ps1 -AudioOnly
#   .\scripts\generate-day-351-launch.ps1 -DryRun
#
# ============================================================================

param(
    [switch]$DryRun,
    [switch]$SkipAudio,
    [switch]$AudioOnly,
    [switch]$SkipImages,
    [switch]$SkipVideo,
    [switch]$SkipUpload,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$DAY = 351
$TOPIC = "Practicing in Your Mind"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🚀 DAY $DAY LAUNCH ASSET GENERATOR                             ║" -ForegroundColor Cyan
Write-Host "║  Topic: $TOPIC                                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PHASE 0: Pre-flight Checks
# ============================================================================
Write-Host "━━━ PHASE 0: Pre-flight Checks ━━━" -ForegroundColor Yellow

# Check required env vars
$requiredEnvVars = @(
    "ELEVENLABS_API_KEY",
    "REPLICATE_API_TOKEN",
    "PUBLIC_SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY"
)

$missingVars = @()
foreach ($var in $requiredEnvVars) {
    if (-not [Environment]::GetEnvironmentVariable($var)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "❌ Missing required environment variables:" -ForegroundColor Red
    $missingVars | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Please set these in your .env file or environment." -ForegroundColor Yellow
    if (-not $DryRun) { exit 1 }
} else {
    Write-Host "✅ All required environment variables present" -ForegroundColor Green
}

# Check Node.js
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found" -ForegroundColor Red
    if (-not $DryRun) { exit 1 }
}

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
    if (-not $DryRun) { exit 1 }
}

Write-Host ""

# ============================================================================
# PHASE 1: Verify Kelly Voice
# ============================================================================
if (-not $SkipAudio -and -not $SkipImages -or $AudioOnly) {
    Write-Host "━━━ PHASE 1: Kelly Voice Check ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would run: npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick" -ForegroundColor Magenta
    } else {
        Write-Host "🎤 Running voice check..." -ForegroundColor Cyan
        npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠️ Voice check had issues - review before continuing" -ForegroundColor Yellow
        } else {
            Write-Host "✅ Voice check passed" -ForegroundColor Green
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 2: Generate Kelly Phase Images
# ============================================================================
if (-not $SkipImages -and -not $AudioOnly) {
    Write-Host "━━━ PHASE 2: Kelly Phase Images ━━━" -ForegroundColor Yellow
    
    $imageOutputDir = "public\kelly\phases\$DAY"
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would generate Kelly images to: $imageOutputDir" -ForegroundColor Magenta
        Write-Host "[DRY RUN] Poses: curious, explaining, thoughtful, excited, warm" -ForegroundColor Magenta
    } else {
        Write-Host "🎨 Generating Kelly phase images..." -ForegroundColor Cyan
        
        # Check if phase visual generator exists
        if (Test-Path "scripts\kelly-phase-visuals\phase-visual-generator.ts") {
            npx tsx scripts/kelly-phase-visuals/phase-visual-generator.ts `
                --day $DAY `
                --topic "visualization"
        } else {
            Write-Host "⚠️ Phase visual generator not found - using existing images" -ForegroundColor Yellow
            Write-Host "   Images already at: $imageOutputDir" -ForegroundColor Gray
        }
        
        # Verify images exist
        if (Test-Path $imageOutputDir) {
            $imageCount = (Get-ChildItem $imageOutputDir -Filter "*.png").Count
            Write-Host "✅ $imageCount images in $imageOutputDir" -ForegroundColor Green
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 3: Generate Audio (All Age Variants)
# ============================================================================
if (-not $SkipAudio) {
    Write-Host "━━━ PHASE 3: Audio Generation (ElevenLabs) ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would run: npx tsx scripts/generate-day-audio-elevenlabs.ts --day $DAY --all" -ForegroundColor Magenta
    } else {
        Write-Host "🔊 Generating audio for all age variants..." -ForegroundColor Cyan
        Write-Host "   This may take 15-20 minutes for 36 audio files" -ForegroundColor Gray
        
        npx tsx scripts/generate-day-audio-elevenlabs.ts --day $DAY --all
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Audio generation complete" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Audio generation had errors - check logs" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 4: Generate Lipsync Videos
# ============================================================================
if (-not $SkipVideo -and -not $AudioOnly) {
    Write-Host "━━━ PHASE 4: Lipsync Videos ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would run lipsync pipeline for day $DAY" -ForegroundColor Magenta
    } else {
        Write-Host "🎬 Generating lipsync videos..." -ForegroundColor Cyan
        
        if (Test-Path "scripts\lipsync-pipeline\run-pipeline.ts") {
            npx tsx scripts/lipsync-pipeline/run-pipeline.ts --day $DAY --age adult
        } else {
            Write-Host "⚠️ Lipsync pipeline not found - skipping" -ForegroundColor Yellow
            Write-Host "   You can generate videos later with HeyGen or iClone" -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 5: Generate Infographics
# ============================================================================
if (-not $SkipImages -and -not $AudioOnly) {
    Write-Host "━━━ PHASE 5: Infographics ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would run: npx tsx scripts/generate-day-infographics.ts --day $DAY" -ForegroundColor Magenta
    } else {
        if (Test-Path "scripts\generate-day-infographics.ts") {
            Write-Host "📊 Generating infographics..." -ForegroundColor Cyan
            npx tsx scripts/generate-day-infographics.ts --day $DAY
        } else {
            Write-Host "⚠️ Infographic generator not found - skipping" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 6: Quality Check (Kelly Cop)
# ============================================================================
if (-not $SkipImages -and -not $AudioOnly) {
    Write-Host "━━━ PHASE 6: Quality Check ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would run Kelly Cop face audit" -ForegroundColor Magenta
    } else {
        if (Test-Path "tools\kelly-cop\kelly_face_audit.py") {
            Write-Host "👮 Running Kelly Cop face verification..." -ForegroundColor Cyan
            Push-Location tools\kelly-cop
            python kelly_face_audit.py --html --limit 10
            Pop-Location
            Write-Host "✅ Quality check complete - review HTML report" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Kelly Cop not found - manual review required" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 7: Upload to CDN
# ============================================================================
if (-not $SkipUpload -and -not $AudioOnly) {
    Write-Host "━━━ PHASE 7: CDN Upload ━━━" -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would upload assets to Supabase/R2" -ForegroundColor Magenta
    } else {
        if (Test-Path "scripts\fill-supabase-with-assets.ts") {
            Write-Host "☁️ Uploading to CDN..." -ForegroundColor Cyan
            npx tsx scripts/fill-supabase-with-assets.ts --day $DAY
        } else {
            Write-Host "⚠️ Upload script not found - manual upload required" -ForegroundColor Yellow
        }
    }
    Write-Host ""
}

# ============================================================================
# PHASE 8: Verification
# ============================================================================
Write-Host "━━━ PHASE 8: Verification ━━━" -ForegroundColor Yellow

# Check local files
$localAssets = @(
    "public\lessons\day-$DAY.json",
    "public\data\day-$DAY-complete.js",
    "public\kelly\phases\$DAY\hook.png"
)

Write-Host "📁 Local assets:" -ForegroundColor Cyan
foreach ($asset in $localAssets) {
    if (Test-Path $asset) {
        Write-Host "   ✅ $asset" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $asset" -ForegroundColor Red
    }
}

# Check production (if not dry run)
if (-not $DryRun) {
    Write-Host ""
    Write-Host "🌐 Production verification:" -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri "https://curiouskelly.com/lessons/day-$DAY.json" -Method Head -UseBasicParsing -TimeoutSec 10
        Write-Host "   ✅ Lesson JSON: $($response.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️ Lesson JSON not yet deployed" -ForegroundColor Yellow
    }
}

Write-Host ""

# ============================================================================
# Summary
# ============================================================================
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ DAY $DAY GENERATION COMPLETE                                ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review generated assets in public/kelly/phases/$DAY/" -ForegroundColor White
Write-Host "  2. Listen to audio samples in generated-audio/" -ForegroundColor White
Write-Host "  3. Check Kelly Cop report for face verification" -ForegroundColor White
Write-Host "  4. Push to deploy: git add . && git commit && git push" -ForegroundColor White
Write-Host "  5. Verify at https://curiouskelly.com/learn.html" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "🔍 This was a DRY RUN - no actual generation performed" -ForegroundColor Magenta
}
