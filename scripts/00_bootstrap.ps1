$ErrorActionPreference = "Stop"
$root = "C:\Users\user\Creative-Pipeline"
$dirs = @(
  "$root","$root\config","$root\installers","$root\scripts","$root\tools",
  "$root\docs\receipts","$root\metrics","$root\analytics\Kelly",
  "$root\projects\_Shared\HDRI","$root\projects\_Shared\Fonts","$root\projects\_Shared\Logos",
  "$root\projects\_Shared\iClone",
  "$root\projects\Kelly\Audio","$root\projects\Kelly\Ref","$root\projects\Kelly\CC5","$root\projects\Kelly\iClone","$root\projects\Kelly\Renders",
  "$root\renders\Kelly"
)
$dirs | ForEach-Object { if(-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }

# Git repo + LFS
if(-not (Test-Path "$root\.git")) { git init $root | Out-Null }
Set-Content -Path "$root\.gitignore" -Value @"
# Build / cache / binary
*.pyc
__pycache__/
.vscode/
*.tmp
*.log
installers/
"@
Set-Content -Path "$root\.gitattributes" -Value @"
*.wav filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.iProject filter=lfs diff=lfs merge=lfs -text
*.ccProject filter=lfs diff=lfs merge=lfs -text
"@
git -C $root lfs install

# Seed README
if(-not (Test-Path "$root\README.md")) {
Set-Content -Path "$root\README.md" -Value "# iLearn Studio - CC5/iClone Pipeline`nThis repository contains a complete automation pipeline for Character Creator 5 (CC5) and iClone 8 creative workflows."
}

# Seed characters.yml if missing
if(-not (Test-Path "$root\config\characters.yml")) {
Set-Content -Path "$root\config\characters.yml" -Value @"
characters:
  - name: Kelly
    voice_wav: projects/Kelly/Audio/kelly25_audio.wav
  - name: Ken
  - name: Amina
  - name: Leo
  - name: Priya
  - name: Mateo
  - name: Sofia
  - name: Grace
  - name: Omar
  - name: Hana
  - name: Liam
  - name: Mei
"@
}

# Metrics file
$ms = "$root\metrics\install_status.json"
if(-not (Test-Path $ms)) { Set-Content -Path $ms -Value "{`"created`": `"$((Get-Date).ToString("s"))`"}" }

Write-Host "Bootstrap folders + repo complete."
