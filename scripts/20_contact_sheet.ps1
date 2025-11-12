param(
  [string]$InputVideo = "C:\Users\user\UI-TARS-desktop\renders\Kelly\kelly_test_talk_v1.mp4",
  [string]$OutPng     = "C:\Users\user\UI-TARS-desktop\analytics\Kelly\kelly_contact_sheet_6x5.png",
  [int]$Cols = 6, [int]$Rows = 5, [int]$Width = 640
)
# 30 frames total (Cols x Rows), sample ~1 fps (adjust as needed)
$tile = "$Cols"x"$Rows"
ffmpeg -y -i "$InputVideo" -vf "fps=1,scale=$Width:-1,tile=$tile" -frames:v 1 "$OutPng"
if($LASTEXITCODE -ne 0){ throw "Contact sheet failed." } else { Write-Host "Contact sheet created: $OutPng" }
