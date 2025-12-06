# 🎨 SIMPLE VISUAL GENERATOR
# Just set your Replicate token and run!

param(
    [Parameter(Mandatory=$true)]
    [string]$ReplicateToken,
    
    [int]$StartDay = 8,
    [int]$EndDay = 365,
    [switch]$DryRun
)

$env:REPLICATE_API_TOKEN = $ReplicateToken

Write-Host ""
Write-Host "🎨 CURIOUS KELLY VISUAL GENERATOR" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No images will be generated" -ForegroundColor Yellow
    Write-Host ""
    
    # Count missing
    $phasesDir = "public\kelly\phases"
    $existingDays = (Get-ChildItem -Path $phasesDir -Directory -ErrorAction SilentlyContinue).Count
    $missingDays = 365 - $existingDays
    $missingImages = $missingDays * 5
    
    Write-Host "📊 STATUS:" -ForegroundColor Green
    Write-Host "   Days with phase visuals: $existingDays"
    Write-Host "   Days missing: $missingDays"
    Write-Host "   Images to generate: $missingImages"
    Write-Host "   Estimated cost: `$$([math]::Round($missingImages * 0.04, 2))"
    Write-Host ""
    return
}

Write-Host "🚀 Starting generation for Days $StartDay to $EndDay..." -ForegroundColor Green
Write-Host ""

# Run the TypeScript generator
npx ts-node scripts/generate-all-phase-visuals.ts --range="$StartDay-$EndDay"

Write-Host ""
Write-Host "✅ Generation complete!" -ForegroundColor Green

