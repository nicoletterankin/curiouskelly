# Kelly OS - Development Setup (PowerShell)

Write-Host "🔧 Kelly OS - Development Setup" -ForegroundColor Cyan
Write-Host "================================"
Write-Host ""

$missing = $false

# Check Flutter
try {
    $flutter = flutter --version 2>&1 | Select-Object -First 1
    Write-Host "✅ Flutter: $flutter" -ForegroundColor Green
} catch {
    Write-Host "❌ Flutter: Not found" -ForegroundColor Red
    Write-Host "   Install from: https://flutter.dev/docs/get-started/install"
    $missing = $true
}

# Check Dart
try {
    $dart = dart --version 2>&1
    Write-Host "✅ Dart: $dart" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Dart: Not found (usually bundled with Flutter)" -ForegroundColor Yellow
}

# Check Java
try {
    $java = java -version 2>&1 | Select-Object -First 1
    Write-Host "✅ Java: $java" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Java: Not found (needed for Android builds)" -ForegroundColor Yellow
}

# Check Gradle
try {
    $gradle = gradle --version 2>&1 | Select-Object -First 1
    Write-Host "✅ Gradle: $gradle" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Gradle: Not found (will be downloaded by Flutter)" -ForegroundColor Yellow
}

# Check Android SDK
if ($env:ANDROID_HOME) {
    Write-Host "✅ Android SDK: $env:ANDROID_HOME" -ForegroundColor Green
} else {
    Write-Host "⚠️  Android SDK: ANDROID_HOME not set" -ForegroundColor Yellow
    Write-Host "   Set in System Environment Variables"
}

# Check Unity Hub
try {
    $unityhub = unityhub --version 2>&1
    Write-Host "✅ Unity Hub: Found" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Unity Hub: Not found" -ForegroundColor Yellow
    Write-Host "   Install from: https://unity.com/download"
}

Write-Host ""
if ($missing) {
    Write-Host "❌ Setup incomplete. Please install missing dependencies." -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ Setup complete! Ready to build Kelly OS." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Copy .env.example to .env and fill in ELEVENLABS_API_KEY"
    Write-Host "2. cd apps\kelly_app_flutter"
    Write-Host "3. flutter pub get"
    Write-Host "4. flutter run"
}
