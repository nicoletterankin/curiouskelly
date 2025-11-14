# Quick test - Character Consistency System
$kellyCharacterBase = @"
Kelly Rein, photorealistic digital human, modern timeless "Apple Genius" aesthetic. 
Oval face, clear skin, warm approachable expression with subtle gentle smile. 
Dark brown eyes, direct and engaging gaze. 
Long wavy dark brown hair, parted slightly off-center, falls over shoulders. 
Late 20s to early 30s, athletic build, strong capable presence.
"@

$kellyReinmakerWardrobe = @"
Wearing Reinmaker armor: dark gray ribbed turtleneck base layer, 
form-fitting dark charcoal-gray tactical garment with structured seams and panels, 
metallic dark steel-colored shoulder pauldrons (multi-layered, riveted, curved protective design), 
wide dark gray fabric sash draped diagonally from left shoulder to right hip secured by dark metallic straps, 
wide dark metallic horizontal strap across chest, multiple dark utilitarian belts around waist with rectangular metallic buckle, 
long form-fitting sleeves with fingerless glove-like covering on left hand, textured wrapped detailing on right forearm, 
dark gray tactical pants matching upper garment. 
Color palette: dark grays, charcoal, metallic steel, dark browns. NO bright colors, NO reds, NO yellows, NO light browns, NO Roman/ancient elements.
"@

$mandatoryNegativePrompts = @"
cartoon, stylized, anime, illustration, drawing, sketch, fantasy, medieval, Roman, ancient, historical, 
exaggerated features, unrealistic proportions, memes, internet humor, casual style, 
second person, extra people, multiple faces, bright colors, red, yellow, orange, 
light browns, tan, beige, leather straps, Roman armor, ornate decorations, jewelry, 
low quality, blurry, pixelated, compression artifacts, oversaturated colors, 
unrealistic lighting, watermark, text overlay, logo, CGI, 3D render, game asset, sprite
"@

function Build-KellyPrompt {
    param (
        [string]$SceneDescription,
        [string]$WardrobeVariant = "Reinmaker",
        [string]$Pose = "",
        [string]$Lighting = "professional photography quality lighting, soft key light at 45 degrees with subtle fill, realistic shadows and highlights",
        [string]$AdditionalNegatives = ""
    )
    
    $wardrobe = if ($WardrobeVariant -eq "DailyLesson") { 
        "Wearing modern professional attire: soft approachable clothing (sweater or blouse), clean white studio background, director's chair setting. Same facial features and hair as Reinmaker variant."
    } else { 
        $kellyReinmakerWardrobe 
    }
    
    $prompt = "$SceneDescription, featuring $kellyCharacterBase, $wardrobe"
    
    if ($Pose) {
        $prompt += ", $Pose"
    }
    
    $prompt += ", $Lighting, photorealistic digital human, modern timeless aesthetic, professional photography quality, high detail, realistic skin textures, realistic fabric textures, realistic metallic surfaces."
    
    $negative = $mandatoryNegativePrompts
    if ($AdditionalNegatives) {
        $negative += ", $AdditionalNegatives"
    }
    
    return @{
        Prompt = $prompt
        Negative = $negative
    }
}

# TEST
Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "CHARACTER CONSISTENCY SYSTEM - PROOF OF WORKING" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

$testPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in neutral running pose, facing right" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, dynamic movement, readable silhouette" `
    -Lighting "orthographic camera view, soft forge key light" `
    -AdditionalNegatives "stylized, cel-shaded"

Write-Host ""
Write-Host "Generated Prompt Length: $($testPrompt.Prompt.Length) characters" -ForegroundColor Green
Write-Host "Negative Prompt Length: $($testPrompt.Negative.Length) characters" -ForegroundColor Green
Write-Host ""

Write-Host "VERIFICATION CHECKLIST:" -ForegroundColor Yellow
$checks = @(
    @{Pattern="Kelly Rein"; Name="Character name"},
    @{Pattern="photorealistic digital human"; Name="Photorealistic requirement"},
    @{Pattern="Oval face"; Name="Face shape"},
    @{Pattern="Dark brown eyes"; Name="Eye color"},
    @{Pattern="wavy dark brown hair"; Name="Hair description"},
    @{Pattern="shoulder pauldrons"; Name="Armor element"},
    @{Pattern="dark gray ribbed turtleneck"; Name="Base layer"},
    @{Pattern="NO bright colors"; Name="Color restrictions"}
)

$allPassed = $true
foreach ($check in $checks) {
    if ($testPrompt.Prompt -match $check.Pattern) {
        Write-Host "  [PASS] $($check.Name)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $($check.Name)" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
Write-Host "NEGATIVE PROMPTS CHECK:" -ForegroundColor Yellow
$negChecks = @("cartoon", "stylized", "anime", "memes", "Roman", "fantasy")
foreach ($neg in $negChecks) {
    if ($testPrompt.Negative -match $neg) {
        Write-Host "  [PASS] Blocks $neg" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Missing $neg" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
if ($allPassed) {
    Write-Host "SYSTEM STATUS: WORKING - ALL CHECKS PASSED" -ForegroundColor Green
} else {
    Write-Host "SYSTEM STATUS: ERRORS DETECTED" -ForegroundColor Red
}
Write-Host "="*70 -ForegroundColor Cyan

Write-Host ""
Write-Host "PROMPT PREVIEW (first 700 characters):" -ForegroundColor Yellow
Write-Host "-"*70 -ForegroundColor Gray
Write-Host $testPrompt.Prompt.Substring(0, [Math]::Min(700, $testPrompt.Prompt.Length))
Write-Host "..." -ForegroundColor Gray













