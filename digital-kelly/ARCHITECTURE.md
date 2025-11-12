# Kelly OS Architecture

Offline-first, clean architecture for cross-platform 3D avatar rendering.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter App (Shell)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  UI (Black) │  │  Unity View  │  │  Lesson Loader   │   │
│  │             │  │   (Embed)    │  │                  │   │
│  └─────────────┘  └──────┬───────┘  └──────────────────┘   │
│                          │                                     │
│                          ▼                                     │
│                   KellyBridge.postMessage()                   │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Unity Engine (URP Rendering)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              BlendshapeDriver (Main)                   │
│  │ │  • Load A2F frames                                   │   │
│  │  • Sync with audio (AudioSettings.dspTime)           │   │
│  │  • Apply blendshapes in real-time                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                        │
│  ┌───────────────────┼───────────────────┐                  │
│  │ AutoBlink         │  BreathingLayer   │                  │
│  │ • Every 3-6s      │  • Subtle motion   │                  │
│  │ • Pause on speech │  • Always active   │                  │
│  └────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Flutter → Unity Communication

```dart
// Flutter
unityController.postMessage(
  'KellyController',
  'LoadAndPlay',
  'path/to/json|path/to/wav'
);
```

```
Unity (KellyBridge.cs)
├── Parse payload
├── Load JSON file → A2F frames
├── Load WAV file → AudioClip
└── Trigger BlendshapeDriver.PlaySynced()
```

### 2. Sync Mechanism

```csharp
// BlendshapeDriver.cs
dspStart = AudioSettings.dspTime + startDelay;
audioSource.PlayScheduled(dspStart);

// In Update():
double t = AudioSettings.dspTime - dspStart;  // Elapsed time
int frame = (int)(t / frameTime);               // Current frame
ApplyBlendshapes(data.frames[frame]);          // Apply weights
```

**Accuracy:** ±1 frame at 30fps = ±33ms

## Component Responsibilities

### Flutter Layer

**apps/kelly_app_flutter/**
- UI: Black screen + Unity embed + FAB
- Bridge: Unity communication
- Services: Path resolution, audio (fallback)
- Lessons: JSON loader

### Unity Layer

**engines/kelly_unity_player/**

**Scripts:**
- `BlendshapeDriver.cs`: Core animation + sync
- `KellyBridge.cs`: Flutter integration
- `AutoBlink.cs`: Natural blinking (3-6s intervals)
- `BreathingLayer.cs`: Subtle breathing motion
- `A2FModels.cs`: Data structures

**Scene:**
- Main.unity with KellyController GameObject
- Camera (FOV 38)
- Soft directional light

### Shared Models

**packages/lesson_models/**
- `Lesson`: id, title, script, audioPath, a2fPath
- JSON Schema validation (future)
- Platform-agnostic data structures

## Asset Management

### Local Storage

**Structure:**
```
~/DigitalKellyTest/
├── audio/
│   └── kelly_intro.wav
├── a2f/
│   └── kelly_a2f_cache.json
└── lessons/
    └── *.json
```

**Flutter Bundled:**
- `assets/kelly_front.png` - Placeholder image
- `assets/lessons/sample_lesson.json` - Demo lesson
- Small A2F frames (for UI verification)

**Excluded from Git:**
- Large audio files (*.wav)
- Real FBX models
- Build artifacts

## Sync Precision

### Frame-Accurate Timing

| Component | Timing |
|-----------|--------|
| DSP Time | Hardware audio clock |
| Frame Time | 1/30s = 33.33ms |
| Precision | ±1 frame = ±33ms |
| Latency | < 100ms total |

### Priority System

1. **Speech A2F frames** (highest)
   - Jaw, lip, mouth shapes
2. **Speech-driven blinks** (from A2F)
   - If provided in frame data
3. **AutoBlink** (fallback)
   - Natural blinking when speech doesn't provide
4. **Breathing** (always on, subtle)
   - Micro-expressions, 0.25Hz

## Security

- No secrets in code (use .env)
- Local-only assets (offline-first)
- Runtime path resolution
- No API calls to external services
- Build-time asset embedding only

## Offline-First Guarantee

- All assets stored locally
- No network requests during playback
- Unity embedded in Flutter (no streaming)
- JSON-driven animation (no model inference)
- Audio files on filesystem

## Deployment

### Android
- Unity export: `.aar` library
- Gradle: include in build.gradle
- Min SDK: 24

### iOS
- Unity export: `.framework`
- CocoaPods: include
- Target: iOS 14+

### Web (future)
- WebGL build
- WASM for audio playback
- Same A2F frame format

## Testing Strategy

### Unit Tests
- Lesson model serialization
- Path resolution
- JSON Schema validation

### Integration Tests
- Flutter → Unity message passing
- Audio load + playback
- Blendshape application

### Manual Testing
- Visual sync verification
- Performance profiling (60fps target)
- Memory leak checks

## Extension Points

### Adding New Blendshapes
1. Update A2F output format
2. Add alias in BlendshapeDriver.cs
3. No code changes if names match

### Adding New Lessons
1. Add JSON to assets/lessons/
2. Update pubspec.yaml assets
3. Load via LessonLoader

### Custom Animation Layers
1. Create new MonoBehaviour script
2. Attach to KellyController
3. Run independently or with priority system

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| FPS | 60 | Unity profiler |
| Sync Accuracy | ±1 frame | AudioSettings.dspTime |
| Memory | < 500MB | Built-in profiler |
| Startup Time | < 2s | First frame to visible |


















