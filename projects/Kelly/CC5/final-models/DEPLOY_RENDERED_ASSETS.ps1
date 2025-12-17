# DEPLOY RENDERED ASSETS - Copy CC5 renders to production locations
# Run after rendering images from CC5/iClone

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Kelly CC5 Asset Deployment Script  " -ForegroundColor Cyan  
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$sourceDir = "c:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\renders"
$publicDir = "c:\Users\user\UI-TARS-desktop\public"

# Create renders folder if it doesn't exist
if (-not (Test-Path $sourceDir)) {
    New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
    Write-Host "[CREATED] Renders folder: $sourceDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Place your rendered images in this folder, then run again:" -ForegroundColor White
    Write-Host "  $sourceDir" -ForegroundColor Gray
    Write-Host ""
    exit
}

$copied = 0
$missing = 0

function Copy-IfExists {
    param($Source, $Dest, $Description)
    
    if (Test-Path $Source) {
        Copy-Item $Source $Dest -Force
        Write-Host "[OK] $Description" -ForegroundColor Green
        $script:copied++
    } else {
        Write-Host "[--] Missing: $Description" -ForegroundColor DarkGray
        $script:missing++
    }
}

Write-Host "PRIORITY 1: Social Media Profile Pictures" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow

# Master profile should be rendered at 2048x2048, then we'll resize
$profileMaster = "$sourceDir\profile-master-2048.png"
if (Test-Path $profileMaster) {
    Write-Host "[OK] Found profile master - resizing..." -ForegroundColor Green
    # Note: Would need ImageMagick or similar for actual resize
    # For now, copy directly (assumes pre-sized exports)
}

Copy-IfExists "$sourceDir\profile-twitter.png" "$publicDir\images\social\profile-twitter.png" "profile-twitter.png"
Copy-IfExists "$sourceDir\profile-instagram.png" "$publicDir\images\social\profile-instagram.png" "profile-instagram.png"
Copy-IfExists "$sourceDir\profile-youtube.png" "$publicDir\images\social\profile-youtube.png" "profile-youtube.png"
Copy-IfExists "$sourceDir\profile-linkedin.png" "$publicDir\images\social\profile-linkedin.png" "profile-linkedin.png"
Copy-IfExists "$sourceDir\profile-tiktok.png" "$publicDir\images\social\profile-tiktok.png" "profile-tiktok.png"
Copy-IfExists "$sourceDir\profile-facebook.png" "$publicDir\images\social\profile-facebook.png" "profile-facebook.png"

Write-Host ""
Write-Host "PRIORITY 1: Social Media Headers/Covers" -ForegroundColor Yellow
Write-Host "---------------------------------------" -ForegroundColor Yellow

Copy-IfExists "$sourceDir\cover-twitter.png" "$publicDir\images\social\cover-twitter.png" "cover-twitter.png"
Copy-IfExists "$sourceDir\cover-linkedin.png" "$publicDir\images\social\cover-linkedin.png" "cover-linkedin.png"
Copy-IfExists "$sourceDir\cover-facebook.png" "$publicDir\images\social\cover-facebook.png" "cover-facebook.png"
Copy-IfExists "$sourceDir\og-default.png" "$publicDir\images\social\og-default.png" "og-default.png"
Copy-IfExists "$sourceDir\twitter-card-large.png" "$publicDir\images\social\twitter-card-large.png" "twitter-card-large.png"
Copy-IfExists "$sourceDir\twitter-card-summary.png" "$publicDir\images\social\twitter-card-summary.png" "twitter-card-summary.png"

Write-Host ""
Write-Host "PRIORITY 2: Hero Images" -ForegroundColor Yellow
Write-Host "-----------------------" -ForegroundColor Yellow

Copy-IfExists "$sourceDir\kelly-hero-4k.png" "$publicDir\images\kelly-hero-4k.png" "kelly-hero-4k.png"
Copy-IfExists "$sourceDir\kelly-homepage-hero.jpeg" "$publicDir\images\kelly-homepage-hero.jpeg" "kelly-homepage-hero.jpeg"
Copy-IfExists "$sourceDir\kelly-og-image.png" "$publicDir\images\kelly-og-image.png" "kelly-og-image.png"
Copy-IfExists "$sourceDir\kelly-logo.png" "$publicDir\images\kelly-logo.png" "kelly-logo.png"
Copy-IfExists "$sourceDir\kelly-hero.jpeg" "$publicDir\assets\kelly\hero\kelly-hero.jpeg" "kelly-hero.jpeg (assets)"

Write-Host ""
Write-Host "PRIORITY 3: Chair/Teaching Poses" -ForegroundColor Yellow
Write-Host "--------------------------------" -ForegroundColor Yellow

$chairPoses = @("celebrating", "curious", "explaining", "listening", "wisdom")
foreach ($pose in $chairPoses) {
    Copy-IfExists "$sourceDir\kelly-chair-$pose.png" "$publicDir\images\kelly\kelly-chair-$pose.png" "kelly-chair-$pose.png"
    Copy-IfExists "$sourceDir\kelly-chair-$pose.png" "$publicDir\images\kelly\kelly-directors-chair-$pose.png" "kelly-directors-chair-$pose.png"
}

Write-Host ""
Write-Host "PRIORITY 4: Lesson Player Poses" -ForegroundColor Yellow
Write-Host "-------------------------------" -ForegroundColor Yellow

Copy-IfExists "$sourceDir\kelly_welcome.png" "$publicDir\kelly\poses\kelly_welcome.png" "kelly_welcome.png"
Copy-IfExists "$sourceDir\kelly_idle.png" "$publicDir\kelly\poses\kelly_idle.png" "kelly_idle.png"
Copy-IfExists "$sourceDir\kelly_listening.png" "$publicDir\kelly\poses\kelly_listening.png" "kelly_listening.png"
Copy-IfExists "$sourceDir\kelly_choice_left.png" "$publicDir\kelly\poses\kelly_choice_left.png" "kelly_choice_left.png"
Copy-IfExists "$sourceDir\kelly_choice_right.png" "$publicDir\kelly\poses\kelly_choice_right.png" "kelly_choice_right.png"
Copy-IfExists "$sourceDir\kelly_hint.png" "$publicDir\kelly\poses\kelly_hint.png" "kelly_hint.png"
Copy-IfExists "$sourceDir\kelly_hint_flip.png" "$publicDir\kelly\poses\kelly_hint_flip.png" "kelly_hint_flip.png"
Copy-IfExists "$sourceDir\kelly_clasp.png" "$publicDir\kelly\poses\kelly_clasp.png" "kelly_clasp.png"
Copy-IfExists "$sourceDir\bot_right_index.png" "$publicDir\kelly\poses\bot_right_index.png" "bot_right_index.png"
Copy-IfExists "$sourceDir\cam_right_index.png" "$publicDir\kelly\poses\cam_right_index.png" "cam_right_index.png"
Copy-IfExists "$sourceDir\rail_left_thumb.png" "$publicDir\kelly\poses\rail_left_thumb.png" "rail_left_thumb.png"

# Also choice buttons
Copy-IfExists "$sourceDir\choice_left.png" "$publicDir\kelly\choices\choice_left.png" "choice_left.png"
Copy-IfExists "$sourceDir\choice_right.png" "$publicDir\kelly\choices\choice_right.png" "choice_right.png"

Write-Host ""
Write-Host "PRIORITY 5: Expressions" -ForegroundColor Yellow
Write-Host "-----------------------" -ForegroundColor Yellow

$expressions = @("celebrating", "confused", "curious-closeup", "curious-main", "curious-thinking", "explaining", "happy-content", "peaceful", "surprised")
foreach ($expr in $expressions) {
    Copy-IfExists "$sourceDir\$expr.jpeg" "$publicDir\images\expressions\$expr.jpeg" "$expr.jpeg"
}

Write-Host ""
Write-Host "PRIORITY 6: Personas" -ForegroundColor Yellow
Write-Host "--------------------" -ForegroundColor Yellow

$personas = @("scientist", "explorer", "rebel", "architect", "diplomat", "empath", "macgyver", "mystic", "provider", "storyteller", "strategist", "survivor")
foreach ($persona in $personas) {
    Copy-IfExists "$sourceDir\$persona.png" "$publicDir\assets\kelly\personas\$persona.png" "$persona.png"
}

Write-Host ""
Write-Host "PRIORITY 7: Brand/Favicons" -ForegroundColor Yellow
Write-Host "--------------------------" -ForegroundColor Yellow

$faviconSizes = @("16", "32", "48", "64", "96", "128", "192", "256", "512")
foreach ($size in $faviconSizes) {
    Copy-IfExists "$sourceDir\favicon-$size.png" "$publicDir\images\brand\favicon-$size.png" "favicon-$size.png"
}

Copy-IfExists "$sourceDir\favicon.ico" "$publicDir\images\brand\favicon.ico" "favicon.ico"
Copy-IfExists "$sourceDir\apple-touch-icon.png" "$publicDir\images\brand\apple-touch-icon.png" "apple-touch-icon.png"
Copy-IfExists "$sourceDir\android-chrome-192.png" "$publicDir\images\brand\android-chrome-192.png" "android-chrome-192.png"
Copy-IfExists "$sourceDir\android-chrome-512.png" "$publicDir\images\brand\android-chrome-512.png" "android-chrome-512.png"
Copy-IfExists "$sourceDir\kelly-logo-square.png" "$publicDir\images\brand\kelly-logo-square.png" "kelly-logo-square.png"

$markSizes = @("64", "128", "256", "512")
foreach ($size in $markSizes) {
    Copy-IfExists "$sourceDir\kelly-mark-circle-$size.png" "$publicDir\images\brand\kelly-mark-circle-$size.png" "kelly-mark-circle-$size.png"
}

$ringedSizes = @("128", "256", "512")
foreach ($size in $ringedSizes) {
    Copy-IfExists "$sourceDir\kelly-mark-ringed-$size.png" "$publicDir\images\brand\kelly-mark-ringed-$size.png" "kelly-mark-ringed-$size.png"
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "           DEPLOYMENT SUMMARY         " -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Copied:  $copied files" -ForegroundColor Green
Write-Host "  Missing: $missing files" -ForegroundColor $(if ($missing -eq 0) { "Green" } else { "Yellow" })
Write-Host ""

if ($missing -gt 0) {
    Write-Host "To complete deployment, add missing files to:" -ForegroundColor Yellow
    Write-Host "  $sourceDir" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then run this script again." -ForegroundColor Yellow
} else {
    Write-Host "All assets deployed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "1. Review images in browser: http://localhost:4321" -ForegroundColor White
    Write-Host "2. Verify quality on all pages" -ForegroundColor White
    Write-Host "3. Deploy to production" -ForegroundColor White
}

Write-Host ""
