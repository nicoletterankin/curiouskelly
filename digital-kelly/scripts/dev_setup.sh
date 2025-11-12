#!/bin/bash

echo "🔧 Kelly OS - Development Setup"
echo "================================"
echo ""

# Check Flutter
if command -v flutter &> /dev/null; then
    FLUTTER_VERSION=$(flutter --version | head -n 1)
    echo "✅ Flutter: $FLUTTER_VERSION"
else
    echo "❌ Flutter: Not found"
    echo "   Install from: https://flutter.dev/docs/get-started/install"
    MISSING=true
fi

# Check Dart
if command -v dart &> /dev/null; then
    DART_VERSION=$(dart --version)
    echo "✅ Dart: $DART_VERSION"
else
    echo "⚠️  Dart: Not found (usually bundled with Flutter)"
fi

# Check Java/Gradle
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    echo "✅ Java: $JAVA_VERSION"
else
    echo "⚠️  Java: Not found (needed for Android builds)"
fi

if command -v gradle &> /dev/null; then
    GRADLE_VERSION=$(gradle --version | head -n 1)
    echo "✅ Gradle: $GRADLE_VERSION"
else
    echo "⚠️  Gradle: Not found (will be downloaded by Flutter)"
fi

# Check Android SDK
if [ -n "$ANDROID_HOME" ]; then
    echo "✅ Android SDK: $ANDROID_HOME"
else
    echo "⚠️  Android SDK: ANDROID_HOME not set"
    echo "   Set in ~/.bashrc or ~/.zshrc"
fi

# Check Unity
if command -v unityhub &> /dev/null; then
    UNITY_VERSION=$(unityhub --version 2>&1 | head -n 1)
    echo "✅ Unity Hub: $UNITY_VERSION"
else
    echo "⚠️  Unity Hub: Not found"
    echo "   Install from: https://unity.com/download"
fi

# Check Unity Editor (optional)
if command -v Unity &> /dev/null; then
    echo "✅ Unity Editor: Found"
else
    echo "⚠️  Unity Editor: Not found (optional for this setup)"
fi

echo ""
if [ "$MISSING" = true ]; then
    echo "❌ Setup incomplete. Please install missing dependencies."
    exit 1
else
    echo "✅ Setup complete! Ready to build Kelly OS."
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env and fill in ELEVENLABS_API_KEY"
    echo "2. cd apps/kelly_app_flutter && flutter pub get"
    echo "3. Place test audio at ~/DigitalKellyTest/audio/kelly_intro.wav"
    echo "4. flutter run"
fi


















