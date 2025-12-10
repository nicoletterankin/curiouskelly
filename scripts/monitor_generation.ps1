# Monitor Content Generation Progress
# Run: .\scripts\monitor_generation.ps1

$terminals = @(
    @{Name="ATOMS"; Path="c:\Users\user\.cursor\projects\c-Users-user-AppData-Roaming-Cursor-Workspaces-1764006546110-workspace-json\terminals\17.txt"},
    @{Name="SHARDS"; Path="c:\Users\user\.cursor\projects\c-Users-user-AppData-Roaming-Cursor-Workspaces-1764006546110-workspace-json\terminals\18.txt"},
    @{Name="FULL TEST"; Path="c:\Users\user\.cursor\projects\c-Users-user-AppData-Roaming-Cursor-Workspaces-1764006546110-workspace-json\terminals\16.txt"}
)

while ($true) {
    Clear-Host
    Write-Host "=== CONTENT GENERATION MONITOR ===" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($t in $terminals) {
        if (Test-Path $t.Path) {
            $content = Get-Content $t.Path -Tail 10
            $status = if ($content -match "active_command") { "🟢 RUNNING" } else { "⚪ IDLE" }
            
            Write-Host "=== $($t.Name) $status ===" -ForegroundColor Green
            $content | ForEach-Object { Write-Host "  $_" }
            Write-Host ""
        }
    }
    
    Start-Sleep -Seconds 30
}







