$ErrorActionPreference = 'SilentlyContinue'

Write-Host "Searching for running visual generation processes..." -ForegroundColor Cyan

$patterns = @(
  'generate-lesson-visuals.ts',
  'generate-all-365-visuals.ts',
  'link-gemini-visuals-to-atoms.ts'
)

$procs = Get-CimInstance Win32_Process | Where-Object {
  $cmd = $_.CommandLine
  if (-not $cmd) { return $false }
  foreach ($p in $patterns) {
    if ($cmd -like "*$p*") { return $true }
  }
  return $false
}

if (-not $procs -or $procs.Count -eq 0) {
  Write-Host "No matching processes found." -ForegroundColor Yellow
  exit 0
}

$procs | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize

foreach ($p in $procs) {
  Write-Host "Stopping PID $($p.ProcessId) ..." -ForegroundColor Yellow
  Stop-Process -Id $p.ProcessId -Force
}

Write-Host "Stopped $($procs.Count) process(es)." -ForegroundColor Green
