param([string]$Config="D:\iLearnStudio\config\characters.yml")
$root = "D:\iLearnStudio"
$yaml = Get-Content $Config -Raw
# Naive parse for names (swap to a YAML parser if available)
$names = ($yaml -split "`n") | Where-Object { $_ -match "^\s*-\s*name:\s*(.+)$" } | ForEach-Object { ($_ -replace "^\s*-\s*name:\s*","").Trim() }
foreach($n in $names){
  $base = Join-Path $root "projects\$n"
  foreach($sub in @("Audio","Ref","CC5","iClone","Renders")) {
    $p = Join-Path $base $sub; if(-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
  }
  $todo = Join-Path $base "TODO.md"
  if(-not (Test-Path $todo)){
    Set-Content -Path $todo -Value @"
# $n — Build TODO
1) CC5: Create HD head (Headshot 2 or ActorMIXER), save to projects/$n/CC5
2) Send to iClone. Load DirectorsChair_Template.iProject.
3) Voice: drop WAV into projects/$n/Audio and run AccuLips.
4) (Optional) AccuFACE VIDEO with mouth disabled.
5) Render test to renders/$n/${n}_test_talk_v1.mp4
6) Run contact sheet + frame metrics scripts for $n.
"@
  }
}
Write-Host "Character scaffolds ready."
