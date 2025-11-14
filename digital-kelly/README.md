# Kelly OS - Cross-Platform Prototype

Offline-first, clean architecture for rendering Kelly's 3D avatar with frame-accurate audio-lip-sync in Flutter via Unity embedding.

## Quickstart

```bash
git clone <repo-url>
cd digital-kelly
cp .env.example .env  # Fill ELEVENLABS_API_KEY locally (not committed)
./scripts/dev_setup.sh
# Place your test audio at: ~/DigitalKellyTest/audio/kelly_intro.wav
flutter pub get -C apps/kelly_app_flutter
flutter run -C apps/kelly_app_flutter
```

**Windows:** Use `.\scripts\dev_setup.ps1` instead of `.sh`.

## Project Structure

```
digital-kelly/
├── apps/kelly_app_flutter/      # Flutter shell with Unity embed
├── engines/kelly_unity_player/  # Unity 2022/2023 URP project
├── packages/lesson_models/      # Shared Dart models/schemas
├── assets/                      # Audio, A2F frames, images
└── scripts/                     # Setup and verification
```

## Verification Checklist

- ✅ Flutter app boots (black bg + Unity view)
- ✅ Press "Play Test" → Unity logs receipt of message
- ✅ Blendshape weight changes over time (watch Unity Console)

## Adding Real Assets

### Real Head FBX with Blendshapes

1. Drag `Kelly_Head.fbx` into `Assets/Kelly/Models`
2. Assign to `BlendshapeDriver.headRenderer` in scene
3. Ensure blendshape names match A2F output or add aliases in `BlendshapeDriver.cs`

### Mapping Mismatched Blendshape Names

Edit `Assets/Kelly/Scripts/BlendshapeDriver.cs`:

```csharp
if (key == "blinkleft" && shapeIndex.TryGetValue("eyeblink_left", out idx))
    headRenderer.SetBlendShapeWeight(idx, Mathf.Clamp01(kv.Value) * intensity);
```

Add more aliases in the same pattern as needed.

## Security

- **Never commit** `.env` or large binaries
- Audio files should be placed in documents directory at runtime
- See `assets/audio/README_PLACEHOLDER.txt` for details

## Next Steps

See `docs/EMBED.md` for Unity → Flutter integration details (Gradle/Xcode settings).

Additional tasks:
- Replace placeholder mesh with real Kelly_Head.fbx
- Add path copier in Flutter to move audio/json to documents dir
- Add lesson JSON loader page & bind to Play
- Add delay calibration slider (±60 ms)
- Wire breathing + blink priority

## Test Audio Setup

Create directory: `~/DigitalKellyTest/audio/`
Place `kelly_intro.wav` there before running the app.

For first-run testing:
1. Tap "Copy Demo Assets" button (once available)
2. This copies placeholder assets to writable storage
3. Play with the demo A2F frames

## Kelly Speaking

Kelly just spoke inside your app. That's a milestone—nice work.

## Requirements

- Flutter SDK >= 3.3.0
- Unity 2022.3 LTS or 2023.x
- Android SDK (minSdk 24)
- iOS 14+ target
- Dart SDK (bundled with Flutter)


















