param(
    [string]$Root = "C:\iLearnStudio",
    [string]$Character = "Kelly"
)

$ErrorActionPreference = "Stop"

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

$charRoot = Join-Path -Path $Root -ChildPath $Character
$refImages = Join-Path $charRoot "00_Reference_Images"
$cc5Source  = Join-Path $charRoot "01_CC5_Source"
$textures   = Join-Path $cc5Source "Textures"
$iclone     = Join-Path $charRoot "02_iClone_Animation"
$audio      = Join-Path $charRoot "03_Audio"
$render     = Join-Path $charRoot "04_Render"
$renderPrev = Join-Path $render "Previews"
$renderFinal= Join-Path $render "Final"

New-DirectoryIfMissing $Root
New-DirectoryIfMissing $charRoot
New-DirectoryIfMissing $refImages
New-DirectoryIfMissing $cc5Source
New-DirectoryIfMissing $textures
New-DirectoryIfMissing $iclone
New-DirectoryIfMissing $audio
New-DirectoryIfMissing $render
New-DirectoryIfMissing $renderPrev
New-DirectoryIfMissing $renderFinal

$readmePath = Join-Path $charRoot "README.txt"
if (-not (Test-Path -LiteralPath $readmePath)) {
@"
This folder follows the iLearn Studio layout.

01_CC5_Source  - Place .ccProject files and Textures here
02_iClone_Animation - Place .iProject and .iAvatar here
03_Audio - Put lesson/test WAV files here
04_Render\Previews and \Final - Output renders
"@ | Set-Content -LiteralPath $readmePath -Encoding UTF8
}

Write-Host "Created/validated structure under: $charRoot"

