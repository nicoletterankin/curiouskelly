# ═══════════════════════════════════════════════════════════════════════════════
#                         FULL TILT MODE EXECUTOR
#                    Run all 8 workstreams in parallel
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host @"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 FULL TILT MODE EXECUTOR                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"@

$ErrorActionPreference = "Continue"
$projectRoot = "C:\Users\user\UI-TARS-desktop"

# Check Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js not found. Please install Node.js first."
    exit 1
}

# Install tsx if needed
Write-Host "`n📦 Checking dependencies..."
npm list -g tsx 2>$null || npm install -g tsx

# Function to run TypeScript scripts
function Run-Script {
    param([string]$script, [string]$name)
    Write-Host "`n▶️  Running: $name"
    npx tsx "$projectRoot\scripts\$script"
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSTREAM 1: Database Seeding
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "📁 WORKSTREAM 1: Database Seeding"
Start-Job -Name "Seeding" -ScriptBlock {
    Set-Location $using:projectRoot
    npx tsx scripts/bulk-seed-database.ts --days=1-30
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSTREAM 2: Age Variant Generation
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "👶 WORKSTREAM 2: Age Variant Generation"
Start-Job -Name "AgeVariants" -ScriptBlock {
    Set-Location $using:projectRoot
    npx tsx scripts/generate-age-variants.ts --days=1-30
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSTREAM 6: Archetype Hooks
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "🎭 WORKSTREAM 6: Archetype Hooks"
Start-Job -Name "Archetypes" -ScriptBlock {
    Set-Location $using:projectRoot
    npx tsx scripts/generate-archetype-hooks.ts --days=1-10
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSTREAM 8: Quality Validation
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "✅ WORKSTREAM 8: Quality Validation"
Start-Job -Name "Validation" -ScriptBlock {
    Set-Location $using:projectRoot
    npx tsx scripts/quality-validator.ts --days=1-30
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSTREAM 7: Video Pipeline (Foreground - needs GPU)
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "🎬 WORKSTREAM 7: Video Pipeline"
Write-Host "   Run separately on Nicolette's machine with GPU:"
Write-Host "   cd $projectRoot\kelly-sync"
Write-Host "   python scripts/local_video_pipeline.py --day 1 --all-phases"

# ═══════════════════════════════════════════════════════════════════════════════
# Monitor Jobs
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "⏳ Waiting for parallel jobs to complete..."

$jobs = Get-Job | Where-Object { $_.State -eq "Running" }
while ($jobs.Count -gt 0) {
    Start-Sleep -Seconds 5
    foreach ($job in Get-Job) {
        if ($job.State -eq "Completed") {
            Write-Host "  ✅ $($job.Name) completed"
            Receive-Job $job
            Remove-Job $job
        }
        elseif ($job.State -eq "Failed") {
            Write-Host "  ❌ $($job.Name) failed"
            Receive-Job $job
            Remove-Job $job
        }
    }
    $jobs = Get-Job | Where-Object { $_.State -eq "Running" }
}

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════╗
║                         🎉 ALL WORKSTREAMS COMPLETE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Next Steps:                                                                 ║
║  1. Deploy Edge Functions: ./supabase/functions/deploy-all.sh                ║
║  2. View Dashboard: open public/admin/dashboard.html                         ║
║  3. Run Video Pipeline on GPU machine                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"@
