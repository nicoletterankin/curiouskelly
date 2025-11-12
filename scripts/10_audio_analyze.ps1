py -3.11 "C:\Users\user\UI-TARS-desktop\tools\analyze_audio.py"
if($LASTEXITCODE -eq 0){ Write-Host "Audio analysis OK." } else { throw "Audio analysis failed." }
