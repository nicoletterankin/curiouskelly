# Unity → Flutter Integration Guide

Complete guide for embedding Unity in Flutter app for Kelly OS.

## Prerequisites

- Unity 2022.3 LTS or 2023.x
- Flutter 3.24+
- Android Studio / Xcode
- Unity Hub

## Export Unity Build

### Android

1. Open Unity project: `engines/kelly_unity_player`
2. File → Build Settings
3. Platform: Android
4. Click "Export"
5. Choose output: `apps/kelly_app_flutter/android/UnityExport`
6. Unity exports `.aar` library

### iOS

1. Open Unity project: `engines/kelly_unity_player`
2. File → Build Settings
3. Platform: iOS
4. Click "Export"
5. Choose output: `apps/kelly_app_flutter/ios/UnityExport`
6. Unity exports `.framework`

## Flutter Integration

### Android Gradle Setup

Edit `apps/kelly_app_flutter/android/app/build.gradle`:

```gradle
dependencies {
    implementation fileTree(dir: "${project(':unityLibrary').projectDir}/libs", include: ['*.jar'])
    implementation project(':unityLibrary')
}
```

Create `android/settings.gradle`:

```gradle
include ':app'
include ':unityLibrary'
project(':unityLibrary').projectDir=new File('..\\..\\unityLibrary')
```

Update `android/gradle.properties`:

```properties
org.gradle.jvmargs=-Xmx4096m
android.enableJetifier=true
android.useAndroidX=true
```

### iOS Setup

1. Copy Unity framework to `Runner/Frameworks`
2. Edit `Runner.xcodeproj`:
   - Add Framework Search Paths
   - Add `-framework UnityFramework` to Other Linker Flags

Edit `ios/Podfile`:

```ruby
platform :ios, '14.0'
use_frameworks!

target 'Runner' do
  # ... existing pods ...
  
  # Unity
  pod 'UnityFramework', :path => 'UnityExport'
end
```

Run `pod install`.

## Unity Script → Flutter Communication

### Flutter Side

```dart
unifiedWidgetController.postMessage(
  'KellyController',
  'LoadAndPlay',
  'path/to/json|path/to/wav',
);
```

### Unity Side (KellyBridge.cs)

```csharp
public void LoadAndPlay(string payload) {
    var parts = payload.Split('|');
    var jsonPath = parts[0];
    var wavPath = parts[1];
    // Load and play...
}
```

## Testing

1. Build Unity project first
2. Copy exports to Flutter project
3. Run `flutter pub get`
4. Run `flutter run`
5. Tap "Play Test" button
6. Check Unity Console logs

## Troubleshooting

### Android: Library not found
- Ensure `android/settings.gradle` includes `:unityLibrary`
- Check `.aar` file exists in expected location

### iOS: Framework error
- Run `pod install` again
- Clean build folder (`flutter clean`)
- Verify Framework Search Paths in Xcode

### Audio sync issues
- Adjust `PlaySynced(startDelay)` in BlendshapeDriver.cs
- Target: ±1 frame at 30fps (±33ms)

## Deployment Targets

- Android: minSdk 24 (Android 7.0)
- iOS: 14.0+
- Xcode: 15.0+
- Gradle: 8.0+

## Unity Player Settings

- Scripting Backend: IL2CPP
- API Compatibility Level: .NET Framework
- Stripping Level: Strip Engine Code
- Target Architectures: ARM64

## File Size Optimization

- Compress audio files (16-bit, 44.1kHz)
- Bake A2F frames (no runtime calculation)
- Asset compression: ASTC/HDRP
- Code stripping enabled

## Security

- Never commit Unity builds to git
- Use `.aar` / `.framework` in CI artifacts
- Runtime asset loading via secure paths
- No hardcoded secrets in Unity scripts

## CI/CD

- Unity builds in GitHub Actions (paid tier)
- Export to Flutter repo as artifacts
- Flutter build pulls Unity artifacts
- Automated testing on both platforms


















