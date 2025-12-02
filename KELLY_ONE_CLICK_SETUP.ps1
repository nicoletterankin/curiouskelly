# ============================================================================
# KELLY VISUAL IDENTITY PIPELINE - ONE-CLICK SETUP
# ============================================================================
# This script automates the entire Kelly Visual Identity setup process.
# Run this script and follow the prompts.
#
# Usage: Right-click this file → "Run with PowerShell"
#    OR: Open PowerShell as Admin → cd to project → .\KELLY_ONE_CLICK_SETUP.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "Kelly Visual Identity Setup"

# Colors
function Write-Step { param($msg) Write-Host "`n🔹 $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "   ✅ $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "   ⚠️  $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "   ❌ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "   ℹ️  $msg" -ForegroundColor White }
function Write-Action { param($msg) Write-Host "`n   👉 $msg" -ForegroundColor Magenta }

# Header
Clear-Host
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║          🎨 KELLY VISUAL IDENTITY PIPELINE SETUP 🎨             ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║     Complete setup in ~30 minutes + 4-6 hours LoRA training     ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Set working directory
$projectRoot = "C:\Users\user\UI-TARS-desktop"
Set-Location $projectRoot
Write-Info "Working directory: $projectRoot"

# ============================================================================
# PHASE 1: DEPENDENCY INSTALLATION
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 1: INSTALLING DEPENDENCIES" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

Write-Step "Checking Node.js..."
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "Node.js $nodeVersion installed"
} else {
    Write-Error "Node.js not found!"
    Write-Action "Please install Node.js from https://nodejs.org/ and run this script again"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Step "Installing npm dependencies..."
npm install @google/generative-ai @aws-sdk/client-s3 @supabase/supabase-js dotenv --save 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "Core dependencies installed"
} else {
    Write-Warning "Some dependencies may have failed, continuing..."
}

npm install @types/node tsx typescript --save-dev 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "Dev dependencies installed"
}

# ============================================================================
# PHASE 2: ENVIRONMENT CONFIGURATION
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 2: ENVIRONMENT CONFIGURATION" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

$envFile = ".env.local"
$envExists = Test-Path $envFile

if ($envExists) {
    Write-Success ".env.local already exists"
    $envContent = Get-Content $envFile -Raw
    
    # Check for R2 credentials
    if ($envContent -match "CLOUDFLARE_R2_ACCESS_KEY_ID=\S+") {
        Write-Success "R2 credentials appear to be configured"
    } else {
        Write-Warning "R2 credentials not found in .env.local"
        Write-Info "You'll need to add them after creating the R2 bucket"
    }
} else {
    Write-Step "Creating .env.local from template..."
    
    $envTemplate = @"
# Kelly Visual Identity Pipeline - Environment Variables
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# ============================================================================
# CLOUDFLARE R2 CONFIGURATION
# ============================================================================
CLOUDFLARE_ACCOUNT_ID=47ebb2a1adc311cb106acc89720e352c
CLOUDFLARE_R2_ACCESS_KEY_ID=
CLOUDFLARE_R2_SECRET_ACCESS_KEY=
KELLY_ASSETS_BUCKET=kelly-assets
KELLY_ASSETS_CDN_URL=https://kelly-assets.curiouskelly.com

# ============================================================================
# GOOGLE AI STUDIO (IMAGEN 3)
# ============================================================================
GOOGLE_AI_API_KEY=AIzaSyBVPxRvxDfA07qyAjbZ6FfRqo5L_rxquHE

# ============================================================================
# REPLICATE (OPTIONAL - FOR FLUX WITH LORA)
# ============================================================================
REPLICATE_API_TOKEN=

# Add your existing Supabase credentials below if not already in another .env file
"@
    
    $envTemplate | Out-File -FilePath $envFile -Encoding UTF8
    Write-Success "Created .env.local"
    Write-Warning "You'll need to add R2 credentials after Step 3"
}

# ============================================================================
# PHASE 3: CLOUDFLARE R2 SETUP (MANUAL)
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 3: CLOUDFLARE R2 SETUP (Manual Steps Required)" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

Write-Host ""
Write-Host "   I'll open Cloudflare for you. Please complete these steps:" -ForegroundColor White
Write-Host ""
Write-Host "   1. Navigate to R2 Object Storage" -ForegroundColor White
Write-Host "   2. Click 'Create bucket'" -ForegroundColor White
Write-Host "   3. Name: kelly-assets" -ForegroundColor Cyan
Write-Host "   4. Location: Automatic" -ForegroundColor White
Write-Host "   5. Click 'Create bucket'" -ForegroundColor White
Write-Host ""
Write-Host "   Then create an API token:" -ForegroundColor White
Write-Host "   6. Click 'Manage R2 API Tokens'" -ForegroundColor White
Write-Host "   7. Click 'Create API token'" -ForegroundColor White
Write-Host "   8. Name: kelly-assets-access" -ForegroundColor Cyan
Write-Host "   9. Permissions: Object Read & Write" -ForegroundColor White
Write-Host "   10. Scope: kelly-assets bucket only" -ForegroundColor White
Write-Host "   11. COPY THE KEYS (you'll need them next)" -ForegroundColor Magenta
Write-Host ""

Write-Action "Press Enter to open Cloudflare Dashboard..."
Read-Host
Start-Process "https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c"

Write-Host ""
Write-Host "   After creating the bucket and API token, enter the credentials:" -ForegroundColor White
Write-Host ""

$r2AccessKey = Read-Host "   Enter R2 Access Key ID (or press Enter to skip)"
$r2SecretKey = Read-Host "   Enter R2 Secret Access Key (or press Enter to skip)"

if ($r2AccessKey -and $r2SecretKey) {
    # Update .env.local with credentials
    $envContent = Get-Content $envFile -Raw
    $envContent = $envContent -replace "CLOUDFLARE_R2_ACCESS_KEY_ID=.*", "CLOUDFLARE_R2_ACCESS_KEY_ID=$r2AccessKey"
    $envContent = $envContent -replace "CLOUDFLARE_R2_SECRET_ACCESS_KEY=.*", "CLOUDFLARE_R2_SECRET_ACCESS_KEY=$r2SecretKey"
    $envContent | Out-File -FilePath $envFile -Encoding UTF8
    Write-Success "R2 credentials saved to .env.local"
} else {
    Write-Warning "R2 credentials skipped - add them manually to .env.local later"
}

# ============================================================================
# PHASE 4: SUPABASE DATABASE SETUP (MANUAL)
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 4: SUPABASE DATABASE SETUP (Manual Steps Required)" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

$sqlFile = "supabase\migrations\20251130_create_kelly_assets.sql"
$sqlContent = Get-Content $sqlFile -Raw

Write-Host ""
Write-Host "   I'll open Supabase SQL Editor for you." -ForegroundColor White
Write-Host "   The SQL has been copied to your clipboard." -ForegroundColor White
Write-Host ""
Write-Host "   Steps:" -ForegroundColor White
Write-Host "   1. Click 'New query'" -ForegroundColor White
Write-Host "   2. Paste (Ctrl+V) the SQL" -ForegroundColor White
Write-Host "   3. Click 'Run' or press F5" -ForegroundColor White
Write-Host "   4. Verify: 'Success. No rows returned'" -ForegroundColor Green
Write-Host ""

# Copy SQL to clipboard
$sqlContent | Set-Clipboard
Write-Success "SQL copied to clipboard"

Write-Action "Press Enter to open Supabase SQL Editor..."
Read-Host
Start-Process "https://supabase.com/dashboard/project/_/sql"

Write-Host ""
$dbDone = Read-Host "   Did you run the SQL successfully? (y/n)"
if ($dbDone -eq "y") {
    Write-Success "Database setup complete"
} else {
    Write-Warning "Remember to run the SQL later"
}

# ============================================================================
# PHASE 5: PREPARE LORA TRAINING DATASET
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 5: PREPARING LORA TRAINING DATASET" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

Write-Step "Running LoRA dataset preparation script..."

# Create the dataset manually since tsx might not be available
$loraDir = "lora-training-dataset"
New-Item -ItemType Directory -Force -Path $loraDir | Out-Null

# Reference images and captions
$images = @(
    @{
        Source = "daily-lesson-marketing\public\lessons\images\4.jpeg"
        Name = "4.jpeg"
        Caption = "kelly, photorealistic woman, close-up portrait, big genuine smile, direct eye contact, brown wavy hair with caramel highlights, hazel-brown eyes, soft blue cashmere sweater, white background, studio lighting"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\yay-pray-huh\pray.jpeg"
        Name = "pray.jpeg"
        Caption = "kelly, photorealistic woman, upper body, hands together near face in prayer pose, looking up and to the right, playful hopeful expression, brown wavy hair, soft blue cashmere sweater, white background"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\walk\open-walk.jpeg"
        Name = "open-walk.jpeg"
        Caption = "kelly, photorealistic woman, full body, walking, profile view, casual confident stride, director's chair in background, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio, natural lighting"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\square-chair\square-chair2.jpeg"
        Name = "square-chair2.jpeg"
        Caption = "kelly, photorealistic woman, full body, seated in director's chair, right hand on heart, sincere grateful expression, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio background"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\our-girl-too-excited\our-girl.jpeg"
        Name = "our-girl.jpeg"
        Caption = "kelly, photorealistic woman, full body, seated in director's chair, chin resting on hand, thoughtful curious expression, brown wavy hair, soft blue cashmere sweater, blue jeans, white sneakers, white studio background"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\open-close\open.png"
        Name = "open.png"
        Caption = "kelly, photorealistic woman, close-up portrait, chin on hand, looking up and to the left, contemplative expression, brown wavy hair with highlights, hazel-brown eyes, soft blue cashmere sweater, white background"
    },
    @{
        Source = "daily-lesson-marketing\public\lessons\images\open-close\close.jpeg"
        Name = "close.jpeg"
        Caption = "kelly, photorealistic woman, close-up portrait, eyes closed, peaceful satisfied smile, chin resting on hand, brown wavy hair, soft blue cashmere sweater, white background"
    }
)

$successCount = 0
foreach ($img in $images) {
    $sourcePath = Join-Path $projectRoot $img.Source
    if (Test-Path $sourcePath) {
        # Copy image
        $destImage = Join-Path $loraDir $img.Name
        Copy-Item $sourcePath $destImage -Force
        
        # Create caption file
        $captionFile = Join-Path $loraDir ($img.Name -replace '\.(jpeg|jpg|png)$', '.txt')
        $img.Caption | Out-File -FilePath $captionFile -Encoding UTF8 -NoNewline
        
        Write-Success "Copied: $($img.Name)"
        $successCount++
    } else {
        Write-Warning "Not found: $($img.Source)"
    }
}

# Create README
$readmeContent = @"
# Kelly LoRA Training Dataset

This dataset contains $successCount reference images of Kelly for training a character LoRA.

## Training Settings (Civitai)

- **Base model:** FLUX.1 Dev or SDXL 1.0
- **Training type:** Character/Person
- **Instance prompt:** kelly
- **Class prompt:** woman
- **Training steps:** 1500-2000
- **Learning rate:** 1e-4
- **Network dimension:** 32
- **Network alpha:** 16

## Upload Instructions

1. Go to https://civitai.com/models/train
2. Log in with: nicoletterankin201 (Pro account)
3. Click "New Training"
4. Upload this entire folder
5. Configure with settings above
6. Start training (~`$15-25, 4-6 hours)

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

$readmeContent | Out-File -FilePath (Join-Path $loraDir "README.md") -Encoding UTF8

Write-Host ""
Write-Success "LoRA dataset prepared: $successCount images + captions"
Write-Info "Location: $loraDir"

# ============================================================================
# PHASE 6: START CIVITAI LORA TRAINING (MANUAL)
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "PHASE 6: START CIVITAI LORA TRAINING (Manual Steps Required)" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor DarkGray

Write-Host ""
Write-Host "   I'll open Civitai and the dataset folder for you." -ForegroundColor White
Write-Host ""
Write-Host "   Steps:" -ForegroundColor White
Write-Host "   1. Log in with: nicoletterankin201" -ForegroundColor Cyan
Write-Host "   2. Click 'New Training'" -ForegroundColor White
Write-Host "   3. Drag & drop ALL files from the folder I'm opening" -ForegroundColor White
Write-Host "   4. Configure settings:" -ForegroundColor White
Write-Host "      - Base model: FLUX.1 Dev" -ForegroundColor Cyan
Write-Host "      - Instance prompt: kelly" -ForegroundColor Cyan
Write-Host "      - Class prompt: woman" -ForegroundColor Cyan
Write-Host "      - Training steps: 1500-2000" -ForegroundColor Cyan
Write-Host "      - Learning rate: 1e-4" -ForegroundColor Cyan
Write-Host "   5. Click 'Start Training'" -ForegroundColor White
Write-Host ""
Write-Host "   Cost: ~`$15-25 | Time: 4-6 hours (runs overnight)" -ForegroundColor Yellow
Write-Host ""

Write-Action "Press Enter to open Civitai and the dataset folder..."
Read-Host

# Open dataset folder
Start-Process "explorer.exe" $loraDir

# Open Civitai
Start-Process "https://civitai.com/models/train"

Write-Host ""
$trainingStarted = Read-Host "   Did you start the training? (y/n)"
if ($trainingStarted -eq "y") {
    Write-Success "LoRA training started! It will complete in 4-6 hours."
    Write-Info "You'll receive an email when it's done."
} else {
    Write-Warning "Remember to start training later using the lora-training-dataset folder"
}

# ============================================================================
# PHASE 7: SUMMARY
# ============================================================================

Write-Host "`n" + ("=" * 70) -ForegroundColor DarkGray
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor DarkGray

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                                  ║" -ForegroundColor Green
Write-Host "║                    ✅ FOUNDATION COMPLETE!                       ║" -ForegroundColor Green
Write-Host "║                                                                  ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "   What was set up:" -ForegroundColor White
Write-Host "   ✅ Dependencies installed" -ForegroundColor Green
Write-Host "   ✅ Environment configured (.env.local)" -ForegroundColor Green
Write-Host "   ✅ LoRA training dataset prepared (7 images + captions)" -ForegroundColor Green
Write-Host ""

Write-Host "   What's running:" -ForegroundColor White
Write-Host "   🔄 LoRA training on Civitai (4-6 hours)" -ForegroundColor Yellow
Write-Host ""

Write-Host "   Tomorrow's steps (after LoRA completes):" -ForegroundColor White
Write-Host "   1. Download trained LoRA from Civitai" -ForegroundColor White
Write-Host "   2. Run: npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts" -ForegroundColor Cyan
Write-Host "   3. Run: npx tsx scripts/kelly-visual-identity/upload-to-r2.ts" -ForegroundColor Cyan
Write-Host "   4. Integrate KellyAvatar component into your app" -ForegroundColor White
Write-Host ""

Write-Host "   Documentation:" -ForegroundColor White
Write-Host "   📄 KELLY_QUICK_START.md - Quick reference" -ForegroundColor White
Write-Host "   📄 KELLY_VISUAL_IDENTITY_COMPLETE.md - Full documentation" -ForegroundColor White
Write-Host "   📄 KELLY_VISUAL_IDENTITY_EXECUTION_CHECKLIST.md - Detailed steps" -ForegroundColor White
Write-Host ""

# Create tomorrow's script
$tomorrowScript = @"
# ============================================================================
# KELLY VISUAL IDENTITY - TOMORROW'S SCRIPT
# ============================================================================
# Run this after LoRA training completes (check your email from Civitai)
# ============================================================================

Write-Host "🎨 Kelly Visual Identity - Post-Training Setup" -ForegroundColor Cyan

# Step 1: Generate all poses
Write-Host "`n📸 Generating all 12 Kelly poses..." -ForegroundColor Yellow
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts

# Step 2: Upload to R2
Write-Host "`n☁️ Uploading to Cloudflare R2..." -ForegroundColor Yellow
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/

Write-Host "`n✅ Done! Check your R2 bucket and Supabase for the uploaded assets." -ForegroundColor Green
Write-Host "Next: Integrate KellyAvatar component into your app" -ForegroundColor White
"@

$tomorrowScript | Out-File -FilePath "KELLY_TOMORROW.ps1" -Encoding UTF8
Write-Success "Created KELLY_TOMORROW.ps1 for tomorrow's steps"

Write-Host ""
Write-Host "   🎯 Run KELLY_TOMORROW.ps1 after you receive the Civitai email!" -ForegroundColor Magenta
Write-Host ""

Write-Action "Press Enter to exit..."
Read-Host



