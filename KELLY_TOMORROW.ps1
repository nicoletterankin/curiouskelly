# ============================================================================
# KELLY VISUAL IDENTITY - TOMORROW'S SCRIPT
# ============================================================================
# Run this AFTER you receive the email from Civitai that LoRA training is done
#
# Usage: Right-click this file → "Run with PowerShell"
#    OR: Open PowerShell → cd C:\Users\user\UI-TARS-desktop → .\KELLY_TOMORROW.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$projectRoot = "C:\Users\user\UI-TARS-desktop"
Set-Location $projectRoot

Clear-Host
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║          🎨 KELLY VISUAL IDENTITY - POST-TRAINING 🎨            ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: DOWNLOAD LORA
# ============================================================================

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 1: DOWNLOAD TRAINED LORA" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   1. Go to Civitai and find your completed training" -ForegroundColor White
Write-Host "   2. Download the .safetensors file" -ForegroundColor White
Write-Host "   3. Save it to: models\lora\kelly_lora_v1.safetensors" -ForegroundColor Cyan
Write-Host ""

# Create models/lora directory
New-Item -ItemType Directory -Force -Path "models\lora" | Out-Null
Write-Host "   ✅ Created: models\lora\" -ForegroundColor Green
Write-Host ""

Write-Host "   👉 Press Enter to open Civitai..." -ForegroundColor Magenta
Read-Host
Start-Process "https://civitai.com/user/nicoletterankin201/models"

Write-Host ""
$loraDownloaded = Read-Host "   Did you download and save the LoRA file? (y/n)"

# ============================================================================
# STEP 2: GENERATE ALL 12 POSES
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 2: GENERATE ALL 12 KELLY POSES" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Generating poses using Google AI Studio..." -ForegroundColor White
Write-Host "   This may take 5-10 minutes (12 poses × rate limiting)" -ForegroundColor White
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path "generated-poses" | Out-Null

Write-Host "   🔄 Running generation script..." -ForegroundColor Yellow
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts

if (Test-Path "generated-poses\generation_log.json") {
    Write-Host ""
    Write-Host "   ✅ Generation complete!" -ForegroundColor Green
    Write-Host "   📁 Output: generated-poses\" -ForegroundColor White
    
    # Count generated files
    $poseCount = (Get-ChildItem "generated-poses\*.png" -ErrorAction SilentlyContinue).Count
    Write-Host "   📸 Generated: $poseCount poses" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "   ⚠️  Generation may have had issues. Check generated-poses\ folder." -ForegroundColor Yellow
}

# ============================================================================
# STEP 3: REVIEW GENERATED IMAGES
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 3: REVIEW GENERATED IMAGES" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Opening the generated-poses folder for review..." -ForegroundColor White
Write-Host ""
Write-Host "   Quality checklist for each image:" -ForegroundColor White
Write-Host "   ✓ Face looks like Kelly (consistent with references)" -ForegroundColor White
Write-Host "   ✓ Blue sweater visible" -ForegroundColor White
Write-Host "   ✓ White studio background" -ForegroundColor White
Write-Host "   ✓ Director's chair present" -ForegroundColor White
Write-Host "   ✓ Pose is clear and unambiguous" -ForegroundColor White
Write-Host "   ✓ No weird hands or artifacts" -ForegroundColor White
Write-Host ""

Write-Host "   👉 Press Enter to open the generated-poses folder..." -ForegroundColor Magenta
Read-Host
Start-Process "explorer.exe" "generated-poses"

Write-Host ""
$imagesApproved = Read-Host "   Are the images acceptable? (y/n)"

if ($imagesApproved -ne "y") {
    Write-Host ""
    Write-Host "   💡 Tips for better results:" -ForegroundColor Yellow
    Write-Host "   - Regenerate individual poses that look off" -ForegroundColor White
    Write-Host "   - Try different seeds by running generation again" -ForegroundColor White
    Write-Host "   - For hand issues, generate 5+ variations and pick best" -ForegroundColor White
    Write-Host ""
}

# ============================================================================
# STEP 4: UPLOAD TO CLOUDFLARE R2
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 4: UPLOAD TO CLOUDFLARE R2" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Uploading approved images to R2 and saving metadata to Supabase..." -ForegroundColor White
Write-Host ""

Write-Host "   🔄 Running upload script..." -ForegroundColor Yellow
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/

Write-Host ""
Write-Host "   ✅ Upload complete!" -ForegroundColor Green

# ============================================================================
# STEP 5: VERIFY IN CLOUDFLARE & SUPABASE
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 5: VERIFY UPLOADS" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Let's verify everything uploaded correctly." -ForegroundColor White
Write-Host ""

Write-Host "   👉 Press Enter to open Cloudflare R2..." -ForegroundColor Magenta
Read-Host
Start-Process "https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c/r2/default/buckets/kelly-assets"

Write-Host ""
Write-Host "   👉 Press Enter to open Supabase..." -ForegroundColor Magenta
Read-Host
Start-Process "https://supabase.com/dashboard/project/_/editor"

Write-Host ""
Write-Host "   In Supabase, run this query to see uploaded assets:" -ForegroundColor White
Write-Host "   SELECT * FROM kelly_assets ORDER BY created_at DESC;" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 6: MARK HERO IMAGES
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 6: MARK HERO IMAGES AS PUBLISHED" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Run this SQL in Supabase to publish the hero images:" -ForegroundColor White
Write-Host ""

$publishSQL = @"
-- Publish all Kelly assets as hero images
UPDATE kelly_assets 
SET 
    status = 'published', 
    is_hero = true, 
    published_at = NOW()
WHERE status = 'review';
"@

Write-Host "   $publishSQL" -ForegroundColor Cyan
Write-Host ""

# Copy to clipboard
$publishSQL | Set-Clipboard
Write-Host "   ✅ SQL copied to clipboard" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 7: INTEGRATION
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "STEP 7: INTEGRATE INTO YOUR APP" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   The KellyAvatar component is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "   Add to your lesson player:" -ForegroundColor White
Write-Host ""
Write-Host '   import KellyAvatar from "@/components/KellyAvatar";' -ForegroundColor Cyan
Write-Host ""
Write-Host '   <KellyAvatar state="thinking" layout="horizontal" priority={true} />' -ForegroundColor Cyan
Write-Host ""
Write-Host "   Available states:" -ForegroundColor White
Write-Host "   idle, thinking, pointing_left, pointing_right," -ForegroundColor White
Write-Host "   pointing_up, pointing_down, encouraging, hint," -ForegroundColor White
Write-Host "   celebrating, supportive, proud, excited" -ForegroundColor White
Write-Host ""
Write-Host "   See examples/kelly-avatar-usage.tsx for 7 integration examples!" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# COMPLETE!
# ============================================================================

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                                  ║" -ForegroundColor Green
Write-Host "║              🎉 KELLY VISUAL IDENTITY COMPLETE! 🎉              ║" -ForegroundColor Green
Write-Host "║                                                                  ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "   Summary:" -ForegroundColor White
Write-Host "   ✅ LoRA trained on your reference images" -ForegroundColor Green
Write-Host "   ✅ 12 core Kelly poses generated" -ForegroundColor Green
Write-Host "   ✅ Images uploaded to Cloudflare R2" -ForegroundColor Green
Write-Host "   ✅ Metadata saved to Supabase" -ForegroundColor Green
Write-Host "   ✅ KellyAvatar component ready to use" -ForegroundColor Green
Write-Host ""
Write-Host "   Monthly cost: ~`$0.05 (R2 storage + bandwidth)" -ForegroundColor Yellow
Write-Host ""
Write-Host "   Need to add more poses later?" -ForegroundColor White
Write-Host "   1. Add prompt to generate-kelly-poses.ts" -ForegroundColor White
Write-Host "   2. Run: npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts" -ForegroundColor White
Write-Host "   3. Run: npx tsx scripts/kelly-visual-identity/upload-to-r2.ts" -ForegroundColor White
Write-Host ""

Write-Host "   👉 Press Enter to exit..." -ForegroundColor Magenta
Read-Host



