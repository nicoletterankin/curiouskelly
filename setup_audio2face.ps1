# NVIDIA Audio2Face-3D Setup for Kelly
# PowerShell script to set up the environment

Write-Host "🚀 Setting up NVIDIA Audio2Face-3D for Kelly..." -ForegroundColor Green

# Create directory
$audio2faceDir = "nvidia_audio2face"
if (!(Test-Path $audio2faceDir)) {
    New-Item -ItemType Directory -Path $audio2faceDir
    Write-Host "📁 Created directory: $audio2faceDir" -ForegroundColor Yellow
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if Git is available
try {
    $gitVersion = git --version 2>&1
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found. Please install Git" -ForegroundColor Red
    exit 1
}

# Run the Python setup
Write-Host "🔧 Running Python setup script..." -ForegroundColor Yellow
python nvidia_audio2face_setup.py

Write-Host "`n🎉 Setup complete! Next steps:" -ForegroundColor Green
Write-Host "1. Get your NVIDIA API key from: https://api.nvidia.com/" -ForegroundColor Cyan
Write-Host "2. Test with Kelly audio:" -ForegroundColor Cyan
Write-Host "   cd nvidia_audio2face" -ForegroundColor White
Write-Host "   python kelly_audio2face.py ../iLearnStudio/projects/Kelly/Audio/kelly25_audio.wav --api-key YOUR_KEY" -ForegroundColor White
Write-Host "`n📚 See nvidia_audio2face/KELLY_WORKFLOW_GUIDE.md for complete workflow" -ForegroundColor Yellow





















