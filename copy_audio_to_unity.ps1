# Copy audio files to Unity Resources folder
# Run this from the project root directory

$sourcePath = "curious-kellly\backend\config\audio\water-cycle"
$destPath = "digital-kelly\engines\kelly_unity_player\My project\Assets\Resources\Audio\Lessons\water-cycle"

# Create destination folder if it doesn't exist
New-Item -ItemType Directory -Force -Path $destPath | Out-Null

# Copy all MP3 files
Copy-Item "$sourcePath\*.mp3" -Destination $destPath -Force

Write-Host "✅ Copied audio files to Unity Resources folder!"
Write-Host "   Source: $sourcePath"
Write-Host "   Destination: $destPath"
Write-Host ""
Write-Host "Files copied:"
Get-ChildItem $destPath -Filter "*.mp3" | ForEach-Object {
    Write-Host "   - $($_.Name)"
}










