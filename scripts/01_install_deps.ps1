$ErrorActionPreference = "Stop"
function MarkStatus($k,$v){
  $f="C:\Users\user\Creative-Pipeline\metrics\install_status.json"
  if(Test-Path $f){$j=Get-Content $f -Raw | ConvertFrom-Json}else{$j=@{}}
  $j | Add-Member -NotePropertyName $k -NotePropertyValue $v -Force
  ($j | ConvertTo-Json -Depth 4) | Set-Content $f
}

# Winget installs
$apps = @(
  @{id="Git.Git"; name="Git"},
  @{id="GitHub.GitLFS"; name="Git LFS"},
  @{id="Python.Python.3.11"; name="Python"},
  @{id="Gyan.FFmpeg"; name="FFmpeg"},
  @{id="7zip.7zip"; name="7-Zip"}
)

foreach($a in $apps){
  try { winget install --id $a.id --silent --accept-package-agreements --accept-source-agreements -e }
  catch { Write-Warning "Failed $($a.name). Try: winget search $($a.name)"; }
}

# Python libs
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install numpy pandas matplotlib librosa soundfile opencv-python

MarkStatus "deps" "PASS"
Write-Host "Dependencies installed."
