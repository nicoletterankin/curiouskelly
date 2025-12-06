# 🚔 Kelly Cop - Overnight Face Audit
# Run this script and let it process all 2,671 images overnight
# Results will be saved to face_audit_report/

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "overnight_audit_$timestamp.log"

Write-Host "🚔 KELLY COP - OVERNIGHT FACE AUDIT" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting full audit of all Kelly images..."
Write-Host "This will take several hours. Check back in the morning!"
Write-Host ""
Write-Host "Log file: $logFile"
Write-Host "Results will be saved to: face_audit_report/"
Write-Host ""
Write-Host "Started at: $(Get-Date)"
Write-Host ""

# Run the face audit with HTML report
python kelly_face_audit.py --html 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "=================================" -ForegroundColor Green
Write-Host "✅ AUDIT COMPLETE!" -ForegroundColor Green
Write-Host "Finished at: $(Get-Date)"
Write-Host ""
Write-Host "Check results in:"
Write-Host "  - face_audit_report/ (JSON, CSV, HTML reports)"
Write-Host "  - $logFile (full log)"

