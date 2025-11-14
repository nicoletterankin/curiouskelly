# Kelly OS - Quick Start Guide

Get Kelly speaking in your app in 15 minutes.

## Prerequisites Checklist

Before starting, verify you have:

- [ ] Flutter SDK (3.24+): `flutter doctor`
- [ ] Unity Hub installed
- [ ] Android Studio OR Xcode
- [ ] Git installed
- [ ] ElevenLabs API key (free tier available)

## Step 1: Clone and Setup

```bash
git clone <your-repo-url>
cd digital-kelly
```

**Windows:**
```powershell
.\scripts\dev_setup.ps1
cp .env.example .env
# Edit .env and add your ELEVENLABS_API_KEY
.\scripts\check_env.ps1
```

**macOS/Linux:**
```bash
chmod +x scripts/*.sh
./scripts/dev_setup.sh
cp .env.example .env
# Edit .env and add your ELEVENLABS_API_KEY
./scripts/check_env.sh
```

## Step 2: Flutter Dependencies

```bash
cd apps/kelly_app_flutter
flutter pub get
flutter pub run flutter pub get  # Ensure packages/lesson_models is linked
```

Verify:
```
flutter doctor
flutter analyze
```

## Step 3: Test Audio Setup

Create test directory and audio file:

**Windows:**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\DigitalKellyTest\audio"
# Place kelly_intro.wav in %USERPROFILE%\DigitalKellyTest\audio\
```

**macOS/Linux:**
```bash
mkdir -p ~/DigitalKellyTest/audio
# Place kelly_intro.wav in ~/DigitalKellyTest/audio/
```

**Note:** You need a WAV file. Can be:
- Generated from ElevenLabs API
- Your own recording (16-bit, 44.1kHz recommended)
- Placeholder tone for testing

## Step 4: Unity Scene Setup

1. Open Unity Hub
2. Add project: `digital-kelly/engines/kelly_unity_player`
3. Wait for Unity to compile scripts
4. Open scene: `Assets/Kelly/Scenes/Main.unity`
5. Select KellyController GameObject
6. Verify scripts are attached (no errors in Console)

**Verify:**
- Scene compiles without errors
- KellyBridge script is on KellyController
- BlendshapeDriver script is on KellyController

## Step 5: Run Flutter App

```bash
cd apps/kelly_app_flutter
flutter run
```

**Expected:**
- Black screen with Unity view
- "Play Test" button in top-right
- Press button
- Check console for logs:
  - `📨 Kelly OS: Sent play message to Unity`
  - `📥 KellyBridge: Received load request...`
  - `✅ KellyBridge: Audio playing in sync`

## Troubleshooting

### Unity view is blank
- Ensure you've built Unity project (File → Build Settings)
- Check AndroidManifest.xml has Unity permissions
- Rebuild Flutter: `flutter clean && flutter pub get`

### Audio file not found
- Check path: `~/DigitalKellyTest/audio/kelly_intro.wav` (or `%USERPROFILE%\DigitalKellyTest\audio\kelly_intro.wav` on Windows)
- Verify file exists: `ls ~/DigitalKellyTest/audio/` or `dir %USERPROFILE%\DigitalKellyTest\audio`

### Blendshapes not animating
- Open Unity Console (visible in Editor or via Logcat)
- Check for "Blendshape not found" warnings
- Verify placeholder mesh has "jawOpen" blendshape
- Future: Replace with real Kelly_Head.fbx

### Build errors
- Android: Check `minSdkVersion: 24` in `build.gradle`
- iOS: Check `platform :ios, '14.0'` in Podfile
- Flutter: Run `flutter doctor -v` for details

## Next Steps

- ✅ Kelly speaking - milestone achieved!
- See `docs/NEXT_STEPS.md` for incremental tasks
- Replace placeholder mesh with real FBX
- Add lesson loader UI
- Calibrate sync timing
- Wire blinking and breathing

## Verification Checklist

- [ ] Flutter app boots (black background)
- [ ] Unity view is visible (may be blank until assets loaded)
- [ ] "Play Test" button visible
- [ ] Pressing button logs message in console
- [ ] Unity receives message (check Console)
- [ ] Audio plays (if file exists)
- [ ] Blendshape weights change (Unity Console shows frame updates)

**If all checks pass, you're ready to develop!**



















