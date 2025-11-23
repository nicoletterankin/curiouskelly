# Generate lip-sync data for water-cycle lesson using NVIDIA Audio2Face-3D
# This script processes all water-cycle audio files and generates blendshape data for Unity

param(
    [string]$NvidiaApiKey = $env:NVIDIA_API_KEY,
    [string]$FunctionId = $env:AUDIO2FACE_FUNCTION_ID,
    [switch]$TestOnly = $false
)

$ErrorActionPreference = "Stop"

# Add ffmpeg to PATH if installed via winget
$ffmpegWingetPath = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_*"
$ffmpegInstalled = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "Gyan.FFmpeg_*" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ffmpegInstalled) {
    $ffmpegBinPath = Get-ChildItem -Path (Join-Path $ffmpegInstalled.FullName "ffmpeg-*\bin") -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffmpegBinPath -and $env:PATH -notlike "*$($ffmpegBinPath.FullName)*") {
        $env:PATH = "$($ffmpegBinPath.FullName);$env:PATH"
        Write-Host "✓ Added ffmpeg to PATH" -ForegroundColor Green
    }
}

Write-Host "=== Kelly Audio2Face Lip-Sync Generator ===" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
if (-not $NvidiaApiKey) {
    Write-Host "❌ NVIDIA API Key not found!" -ForegroundColor Red
    Write-Host "Please set it as an environment variable:" -ForegroundColor Yellow
    Write-Host '  $env:NVIDIA_API_KEY = "your_key_here"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Get your API key from: https://build.nvidia.com/" -ForegroundColor Cyan
    exit 1
}

if (-not $FunctionId) {
    Write-Host "❌ Audio2Face Function ID not found!" -ForegroundColor Red
    Write-Host "Please set it as an environment variable:" -ForegroundColor Yellow
    Write-Host '  $env:AUDIO2FACE_FUNCTION_ID = "your_function_id_here"' -ForegroundColor Yellow
    exit 1
}

# Paths
$projectRoot = Split-Path $PSScriptRoot -Parent
$audioSourceDir = Join-Path $projectRoot "curious-kellly\backend\config\audio\water-cycle"
$outputDir = Join-Path $projectRoot "digital-kelly\engines\kelly_unity_player\Assets\Kelly\Audio\water-cycle\a2f_data"
$a2fClientScript = Join-Path $projectRoot "Audio2Face-3D-Samples\scripts\audio2face_3d_api_client\nim_a2f_3d_client.py"
$configFile = Join-Path $projectRoot "Audio2Face-3D-Samples\scripts\audio2face_3d_api_client\config\config_claire.yml"
$helperScript = Join-Path $projectRoot "scripts\a2f_helper.py"

# Verify paths
if (-not (Test-Path $audioSourceDir)) {
    Write-Host "❌ Audio source directory not found: $audioSourceDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $a2fClientScript)) {
    Write-Host "❌ Audio2Face client script not found: $a2fClientScript" -ForegroundColor Red
    Write-Host "Make sure Audio2Face-3D-Samples is set up correctly" -ForegroundColor Yellow
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "✓ Audio source: $audioSourceDir" -ForegroundColor Green
Write-Host "✓ Output directory: $outputDir" -ForegroundColor Green
Write-Host ""

# Get all MP3 files
$audioFiles = Get-ChildItem -Path $audioSourceDir -Filter "*.mp3"

if ($audioFiles.Count -eq 0) {
    Write-Host "❌ No MP3 files found in $audioSourceDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($audioFiles.Count) audio files to process" -ForegroundColor Cyan
Write-Host ""

# Test mode: only process first file
if ($TestOnly) {
    $audioFiles = $audioFiles | Select-Object -First 1
    Write-Host "⚠️ TEST MODE: Processing only 1 file" -ForegroundColor Yellow
    Write-Host ""
}

# Process each audio file
$processed = 0
$failed = 0

foreach ($audioFile in $audioFiles) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($audioFile.Name)
    $outputSubDir = Join-Path $outputDir $baseName
    
    Write-Host "[$($processed + $failed + 1)/$($audioFiles.Count)] Processing: $baseName" -ForegroundColor Cyan
    
    # Create output subdirectory
    New-Item -ItemType Directory -Force -Path $outputSubDir | Out-Null
    
    # Convert MP3 to WAV (Audio2Face requires WAV)
    $wavFile = Join-Path $outputSubDir "$baseName.wav"
    
    Write-Host "  ⏳ Converting MP3 to WAV..." -ForegroundColor Gray
    
    # Use Python helper script for MP3 to WAV conversion
    $convertResult = & python $helperScript "mp3-to-wav" $audioFile.FullName $wavFile 2>&1
    
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $wavFile)) {
        Write-Host "  ✗ MP3 to WAV conversion failed" -ForegroundColor Red
        Write-Host "    $convertResult" -ForegroundColor Red
        Write-Host "    Install pydub: pip install pydub" -ForegroundColor Yellow
        $failed++
        continue
    }
    
    Write-Host "  ✓ WAV file created" -ForegroundColor Green
    
    # Run Audio2Face client
    Write-Host "  ⏳ Generating lip-sync data..." -ForegroundColor Gray
    
    $originalLocation = Get-Location
    Push-Location $outputSubDir
    try {
        # Run Python client with correct arguments
        $pythonCmd = @(
            "python"
            $a2fClientScript
            $wavFile
            $configFile
            "--apikey"
            $NvidiaApiKey
            "--function-id"
            $FunctionId
        )
        
        $result = & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length-1)] 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
    }
    catch {
        Pop-Location
        Write-Host "  ✗ Error running Audio2Face client: $_" -ForegroundColor Red
        $failed++
        continue
    }
    finally {
        Pop-Location
    }
    
    if ($exitCode -eq 0) {
        # Find the timestamped output directory created by the client
        $timestampDirs = Get-ChildItem -Path $outputSubDir -Directory | Where-Object { $_.Name -match "^\d{8}_\d{6}_\d+$" } | Sort-Object LastWriteTime -Descending
        
        if ($timestampDirs.Count -gt 0) {
            $latestDir = $timestampDirs[0].FullName
            $csvFile = Join-Path $latestDir "animation_frames.csv"
            
            if (Test-Path $csvFile) {
                Write-Host "  ✓ Lip-sync data generated" -ForegroundColor Green
                
                # Convert CSV to Unity JSON format
                Write-Host "  ⏳ Converting to Unity format..." -ForegroundColor Gray
                
                $unityJsonFile = Join-Path $outputSubDir "$baseName.a2f.json"
                $jsonResult = & python $helperScript "csv-to-json" $csvFile $unityJsonFile 30 2>&1
                
                if ($LASTEXITCODE -eq 0 -and (Test-Path $unityJsonFile)) {
                    Write-Host "  ✓ Unity format ready: $unityJsonFile" -ForegroundColor Green
                    $processed++
                } else {
                    Write-Host "  ✗ CSV to JSON conversion failed" -ForegroundColor Red
                    Write-Host "    $jsonResult" -ForegroundColor Red
                    $failed++
                }
            } else {
                Write-Host "  ✗ animation_frames.csv not found in output directory" -ForegroundColor Red
                $failed++
            }
        } else {
            Write-Host "  ✗ No output directory found" -ForegroundColor Red
            Write-Host "    Client output: $result" -ForegroundColor Gray
            $failed++
        }
    } else {
        Write-Host "  ✗ Audio2Face processing failed" -ForegroundColor Red
        Write-Host "    $result" -ForegroundColor Red
        $failed++
    }
    
    Write-Host ""
}

# Summary
Write-Host "=== Processing Complete ===" -ForegroundColor Cyan
Write-Host "✓ Processed: $processed" -ForegroundColor Green
Write-Host "✗ Failed: $failed" -ForegroundColor Red
Write-Host ""
Write-Host "Output directory: $outputDir" -ForegroundColor Cyan
Write-Host ""

if ($processed -gt 0) {
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Import the .a2f.json files into Unity Resources folder" -ForegroundColor Yellow
    Write-Host "2. Assign them to the BlendshapeDriver component" -ForegroundColor Yellow
    Write-Host "3. Test lip-sync playback in Unity" -ForegroundColor Yellow
}


