$root="C:\Users\user\UI-TARS-desktop"
$vscode="$root\.vscode"
if(-not (Test-Path $vscode)) { New-Item -ItemType Directory -Path $vscode | Out-Null }
$tasks=@"
{
  "version": "2.0.0",
  "tasks": [
    { "label": "Bootstrap Repo", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\00_bootstrap.ps1"] },
    { "label": "Install Deps", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\01_install_deps.ps1"] },
    { "label": "Detect Reallusion Installs", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\02_detect_reallusion.ps1"] },
    { "label": "Analyze Audio (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\10_audio_analyze.ps1"] },
    { "label": "Contact Sheet (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\20_contact_sheet.ps1"] },
    { "label": "Frame Metrics (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\21_frame_metrics.ps1"] },
    { "label": "Scaffold All Characters", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\30_new_character.ps1","-Config","$root\\config\\characters.yml"] }
  ]
}
"@
Set-Content -Path "$vscode\tasks.json" -Value $tasks
Write-Host "VSCode/Cursor tasks written."
