py -3.11 "C:\Users\user\UI-TARS-desktop\tools\frame_metrics.py"
if($LASTEXITCODE -ne 0){ throw "Frame metrics failed." } else { Write-Host "Frame metrics OK." }
