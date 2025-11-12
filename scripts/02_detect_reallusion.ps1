$ErrorActionPreference = "SilentlyContinue"
$root="C:\Users\user\Creative-Pipeline"
$versPath="$root\docs\VERSIONS.md"
$status="$root\metrics\install_status.json"

function GetExeVersion($path){
  if(Test-Path $path){
    $v = (Get-Item $path).VersionInfo.ProductVersion
    return $v
  } else { return $null }
}

$paths = @{
  "Character Creator 5" = "C:\Program Files\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe"
  "iClone 8"            = "C:\Program Files\Reallusion\iClone 8\Bin64\iClone.exe"
}

$versions = @()
foreach($k in $paths.Keys){
  $v = GetExeVersion $paths[$k]
  if($v -ne $null) {
    $versions += @{ name=$k; version=$v }
  } else {
    $versions += @{ name=$k; version="(not found)" }
  }
}

# Write VERSIONS.md
$md = "# Installed Versions`r`n"
foreach($rec in $versions){
  $md += "- {0}: {1}`r`n" -f $rec.name, $rec.version
}
Set-Content -Path $versPath -Value $md

# Update status JSON
if(Test-Path $status){ $j=Get-Content $status -Raw | ConvertFrom-Json } else { $j=@{} }
$ok = ($versions | Where-Object { $_.version -ne "(not found)" }).Count -ge 1
if($ok -eq $true) {
  $j | Add-Member -NotePropertyName "reallusion_detect" -NotePropertyValue "PASS" -Force
} else {
  $j | Add-Member -NotePropertyName "reallusion_detect" -NotePropertyValue "WARN" -Force
}
($j | ConvertTo-Json -Depth 4) | Set-Content $status

Write-Host "Detection complete. See $versPath"
