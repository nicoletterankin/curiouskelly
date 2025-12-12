param(
  [int]$Day = 1,
  [string]$Ages = "adult",
  [string]$Langs = "en",
  [int]$PollSeconds = 30,
  [int]$MaxPolls = 40
)

Write-Host "=========================================="
Write-Host "HEYGEN VIDEO COMPILATION - DAY $Day"
Write-Host "Ages: $Ages | Languages: $Langs"
Write-Host "=========================================="

$ageList = $Ages.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$langList = $Langs.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

Write-Host "\n[1/4] Auditing content..."
& npx tsx scripts/audit-day1-for-heygen.ts --day=$Day
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "\n[2/4] Queuing HeyGen videos (will refuse if talking_photo_ids are not filled)..."
foreach ($age in $ageList) {
  foreach ($lang in $langList) {
    Write-Host "\n>>> Queuing: age=$age, lang=$lang"
    & npx tsx scripts/generate-day-videos-heygen.ts --day=$Day --age=$age --lang=$lang
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
}

Write-Host "\n[3/4] Polling HeyGen processing (every ${PollSeconds}s)..."
for ($i = 1; $i -le $MaxPolls; $i++) {
  Start-Sleep -Seconds $PollSeconds
  Write-Host "Poll attempt $i/$MaxPolls..."
  & npx tsx scripts/poll-heygen-status.ts --day=$Day
}

Write-Host "\n[4/4] Done. (Check Supabase kelly_video_assets for validated/completed rows.)"
Write-Host "=========================================="
Write-Host "✅ DAY $Day COMPILATION SCRIPT FINISHED"
Write-Host "=========================================="
