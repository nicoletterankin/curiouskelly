# Kelly Avatar 60fps Upgrade Guide

## 🎨 **Overview**

This guide covers the Kelly avatar upgrade to 60fps with gaze tracking and age morphing. The upgrade includes:

- ✅ **60fps animation** (vs 30fps previously)
- ✅ **Gaze tracking** with micro-saccades
- ✅ **Kelly age morphing** (6 variants: 3, 9, 15, 27, 48, 82 years)
- ✅ **Micro-expressions** (blinking, breathing)
- ✅ **Performance monitoring** (ensuring 60fps target)
- ✅ **Flutter integration** (bidirectional communication)

---

## 📦 **New Components**

### 1. BlendshapeDriver60fps.cs
Enhanced animation driver with 60fps support.

**Features:**
- 60fps interpolation for smooth motion
- Age-based micro-expression frequencies
- Real-time gaze tracking
- Automatic blinking (8-12 per minute)
- Subtle breathing animations

**Key Settings:**
```csharp
targetFPS = 60;
enableInterpolation = true;
enableGaze = true;
enableMicroExpressions = true;
```

### 2. AvatarPerformanceMonitor.cs
Real-time performance tracking.

**Monitors:**
- Current/Average/Min/Max FPS
- Frame time (target: 16.67ms for 60fps)
- Memory usage
- Performance status (Excellent/Good/Warning/Poor)

**Targets:**
- 60fps ± 5% (57-63fps acceptable)
- <16.67ms frame time
- Stable memory usage

### 3. KellyAvatarController.cs
Main controller for Kelly avatar.

**Responsibilities:**
- Age morphing logic
- Flutter communication
- Voice parameter adjustment
- Lesson playback coordination

**Key Methods:**
```csharp
SetLearnerAge(int age)          // Update Kelly's age
PlayLesson(string id, int age)  // Start lesson
Speak(string text, int age)     // TTS with lip-sync
SetGazeTarget(float x, float y) // Control eye focus
```

### 4. UnityMessageManager.cs
Bidirectional Unity ↔ Flutter communication.

**Message Types:**
- `setAge` - Update learner age
- `playLesson` - Start lesson playback
- `speak` - TTS request
- `stop` - Stop playback
- `getPerformance` - Request performance stats
- `setGazeTarget` - Update gaze position

---

## 🎯 **Kelly Age Morphing**

### Age Mapping

| Learner Age | Kelly Age | Variant | Voice Pitch | Voice Speed | Blink Rate | Saccade Rate |
|-------------|-----------|---------|-------------|-------------|------------|--------------|
| 2-5         | 3         | Toddler | 1.30x       | 0.90x       | 15/min     | 4/sec        |
| 6-12        | 9         | Kid     | 1.15x       | 1.00x       | 12/min     | 3.5/sec      |
| 13-17       | 15        | Teen    | 1.05x       | 1.10x       | 10/min     | 3/sec        |
| 18-35       | 27        | Adult   | 1.00x       | 1.00x       | 10/min     | 3/sec        |
| 36-60       | 48        | Mentor  | 0.95x       | 0.95x       | 9/min      | 2.5/sec      |
| 61-102      | 82        | Elder   | 0.90x       | 0.85x       | 8/min      | 2/sec        |

### Implementation Notes

- **6 age variants** should be prepped as separate GameObjects
- **Voice pitch/speed** automatically adjusted per age
- **Micro-expressions** adapt to age (kids blink more, elders slower)
- **Smooth transitions** when age changes (future: morph animations)

---

## 👁️ **Gaze Tracking**

### Features

1. **Screen-space targeting**
   - Gaze follows `gazeTarget` transform
   - Convert screen coordinates (x, y) to world position
   - Smooth interpolation for natural motion

2. **Micro-saccades**
   - Small random eye movements (2-4 per second)
   - Adds realism and "alive" feeling
   - Frequency adapts to Kelly's age

3. **Controls**
   ```csharp
   gazeSpeed = 2f;           // Rotation speed
   gazeInfluence = 0.7f;     // Intensity (0-1)
   saccadeFrequency = 3f;    // Movements per second
   ```

---

## ⚡ **60fps Performance**

### Target Metrics

- **FPS:** 60 ± 5% (57-63fps acceptable)
- **Frame Time:** <16.67ms consistently
- **Memory:** <200MB for avatar (excluding lesson assets)
- **CPU:** <30% on mid-range mobile (Snapdragon 700 series)

### Optimization Tips

1. **Blendshape count:** Keep under 50 active shapes
2. **Interpolation:** Smooth but not excessive (lerpSpeed = 15f)
3. **Micro-expressions:** Only when visible (camera culling)
4. **Memory:** Reuse AudioClip references, don't reload
5. **Profiling:** Enable `showDebugInfo = true` during testing

### Performance Monitoring

```csharp
// Get current stats
var stats = avatarPerformanceMonitor.GetCurrentStats();
Debug.Log($"FPS: {stats.averageFPS:F1}, Status: {stats.status}");

// Check if meeting target
bool goodPerf = avatarPerformanceMonitor.IsMeetingTarget();
```

---

## 🔌 **Flutter Integration**

### Setup in Unity

1. Add `UnityMessageManager` to scene (singleton)
2. Attach `KellyAvatarController` to Kelly root GameObject
3. Wire up components in Inspector:
   - `blendshapeDriver` → BlendshapeDriver60fps
   - `performanceMonitor` → AvatarPerformanceMonitor
   - `kellyAgeVariants[]` → 6 age model GameObjects

### Messages from Flutter → Unity

```json
// Set learner age
{
  "type": "setAge",
  "age": 35
}

// Play lesson
{
  "type": "playLesson",
  "lessonId": "leaves-change-color",
  "age": 35
}

// Speak text
{
  "type": "speak",
  "text": "Why do leaves change color?",
  "age": 35
}

// Set gaze target (normalized screen coords)
{
  "type": "setGazeTarget",
  "x": 0.5,
  "y": 0.6
}

// Request performance stats
{
  "type": "getPerformance"
}

// Stop playback
{
  "type": "stop"
}
```

### Messages from Unity → Flutter

```json
// Avatar ready
{
  "type": "ready",
  "data": {
    "kellyAge": 27,
    "learnerAge": 35,
    "fps": 60,
    "platform": "Android"
  },
  "timestamp": "2025-10-30T12:00:00.000Z"
}

// Age updated
{
  "type": "ageUpdated",
  "data": {
    "learnerAge": 35,
    "kellyAge": 27,
    "voicePitch": 1.0,
    "voiceSpeed": 1.0
  },
  "timestamp": "..."
}

// Lesson started
{
  "type": "lessonStarted",
  "data": {
    "lessonId": "leaves-change-color",
    "age": 35,
    "kellyAge": 27
  },
  "timestamp": "..."
}

// Performance stats
{
  "type": "performanceStats",
  "data": {
    "fps": 61.2,
    "avgFps": 60.8,
    "minFps": 58.1,
    "maxFps": 63.4,
    "frameTimeMs": 16.39,
    "memoryMB": 145,
    "meetingTarget": true,
    "status": "Excellent"
  },
  "timestamp": "..."
}

// Playback stopped
{
  "type": "stopped",
  "data": null,
  "timestamp": "..."
}
```

---

## 🧪 **Testing**

### Unity Editor Testing

1. **Open Scene:** `Assets/Kelly/Scenes/KellyAvatar.unity`
2. **Play Mode:** Press Play
3. **On-screen controls:**
   - Top-left: Performance stats
   - Top-right: Blendshape driver status
   - Bottom-left: Controller UI with age test buttons
4. **Test ages:** Click "Test Age 5", "Test Age 35", "Test Age 102"
5. **Monitor FPS:** Should stay 60+ consistently

### Performance Testing

```csharp
// In Unity Editor console
// Watch for warnings if FPS drops below 55
[Performance] Below 60fps target! Current: 54.2fps

// Or enable detailed logging
avatarPerformanceMonitor.logToConsole = true;
```

### Age Morphing Testing

```csharp
// Test all 6 age variants
kellyController.SetLearnerAge(3);   // Toddler
kellyController.SetLearnerAge(9);   // Kid
kellyController.SetLearnerAge(15);  // Teen
kellyController.SetLearnerAge(27);  // Adult
kellyController.SetLearnerAge(48);  // Mentor
kellyController.SetLearnerAge(82);  // Elder

// Verify:
// - Correct model activated
// - Voice parameters updated
// - Micro-expression frequencies adjusted
```

---

## 🚀 **Build & Deploy**

### Unity Build Settings

**For iOS:**
```
Target SDK: iOS 14.0+
Architecture: ARM64
Scripting Backend: IL2CPP
Target Frame Rate: 60
Metal API Validation: Off (production)
```

**For Android:**
```
Target SDK: API 26+ (Android 8.0)
Architecture: ARM64
Scripting Backend: IL2CPP
Target Frame Rate: 60
Rendering: OpenGL ES 3.0 / Vulkan
```

### Build Steps

1. **Open Build Settings:** File → Build Settings
2. **Select platform:** iOS or Android
3. **Add scenes:** Drag `KellyAvatar.unity` to build list
4. **Player Settings:**
   - Set Target Frame Rate: 60
   - Enable Metal (iOS) / Vulkan (Android)
   - Disable Multithreaded Rendering (if issues)
5. **Build:** Click "Build" or "Build and Run"
6. **Output:** Xcode project (iOS) or APK/AAB (Android)

### Flutter Integration

After building Unity:
1. Copy generated files to Flutter project:
   - iOS: `UnityLibrary.xcframework` → `ios/UnityLibrary/`
   - Android: `unityLibrary` → `android/unityLibrary/`
2. Update Flutter dependencies (flutter_unity_widget)
3. Test on device (not simulator for 60fps)

---

## 📊 **Performance Benchmarks**

### Target Devices

| Device | CPU | GPU | Expected FPS | Frame Time | Status |
|--------|-----|-----|--------------|------------|--------|
| iPhone 12 | A14 | Apple GPU | 60+ | <16ms | ✅ Excellent |
| iPhone 13 | A15 | Apple GPU | 60+ | <15ms | ✅ Excellent |
| Pixel 6 | Tensor | Mali-G78 | 58-62 | <17ms | ✅ Good |
| Pixel 7 | Tensor G2 | Mali-G710 | 60+ | <16ms | ✅ Excellent |
| OnePlus 9 | Snapdragon 888 | Adreno 660 | 60+ | <16ms | ✅ Excellent |

### Low-End Targets (Acceptable)

| Device | CPU | GPU | Expected FPS | Frame Time | Status |
|--------|-----|-----|--------------|------------|--------|
| iPhone 11 | A13 | Apple GPU | 55-60 | <18ms | ⚠️ Good |
| Pixel 5 | Snapdragon 765G | Adreno 620 | 50-58 | <20ms | ⚠️ Acceptable |

**Note:** Below 50fps consistently = poor experience, not recommended.

---

## 🐛 **Troubleshooting**

### Issue: FPS below 60

**Possible causes:**
1. Too many active blendshapes (>50)
2. High-poly model (>20k triangles)
3. Expensive shaders
4. Background processes

**Solutions:**
- Reduce blendshape count
- Optimize model (LOD system)
- Use simpler shaders (Mobile/Unlit)
- Close background apps during testing

### Issue: Gaze not working

**Check:**
1. `gazeTarget` assigned in Inspector?
2. `enableGaze = true`?
3. Camera reference valid?

**Debug:**
```csharp
Debug.Log($"Gaze target: {blendshapeDriver.gazeTarget?.position}");
Debug.Log($"Gaze enabled: {blendshapeDriver.enableGaze}");
```

### Issue: Age morphing not switching models

**Check:**
1. All 6 age variants assigned in `kellyAgeVariants[]`?
2. GameObjects named correctly?
3. Only one should be active at a time

**Debug:**
```csharp
foreach (var variant in kellyController.kellyAgeVariants)
{
    Debug.Log($"{variant.name}: {(variant.activeSelf ? "Active" : "Inactive")}");
}
```

### Issue: Flutter communication not working

**Check:**
1. `UnityMessageManager` in scene?
2. Platform-specific code uncommented (#if UNITY_ANDROID)?
3. Flutter side receiving messages?

**Debug:**
```csharp
UnityMessageManager.Instance.SendMessageToFlutter("{\"type\":\"test\"}");
// Should appear in Flutter logs
```

---

## 📚 **API Reference**

### BlendshapeDriver60fps

```csharp
// Set Kelly's age
public void SetKellyAge(int learnerAge)

// Get current Kelly age
public int GetKellyAge()

// Load animation data from JSON
public void LoadRuntimeJson(string json)

// Set audio clip for lip-sync
public void SetAudioClip(AudioClip clip)

// Start synced playback
public void PlaySynced(double startDelay = 0.05)
```

### KellyAvatarController

```csharp
// Set learner age (updates Kelly)
public void SetLearnerAge(int age)

// Play lesson
public void PlayLesson(string lessonId, int age)

// Speak text with TTS
public void Speak(string text, int age)

// Stop all playback
public void StopPlayback()

// Set gaze target (normalized 0-1)
public void SetGazeTarget(float x, float y)
```

### AvatarPerformanceMonitor

```csharp
// Get current performance stats
public PerformanceStats GetCurrentStats()

// Check if meeting 60fps target
public bool IsMeetingTarget()

// Get average FPS
public float GetAverageFPS()

// Get detailed report string
public string GetPerformanceReport()
```

---

## ✅ **Checklist**

### Initial Setup
- [ ] Import all 4 new scripts to `Assets/Kelly/Scripts/`
- [ ] Create `KellyAvatar` scene with Kelly models
- [ ] Assign components in Inspector
- [ ] Test in Unity Editor (Play Mode)
- [ ] Verify 60fps performance
- [ ] Test age morphing (all 6 variants)
- [ ] Test gaze tracking

### Integration
- [ ] Build Unity project (iOS/Android)
- [ ] Copy output to Flutter project
- [ ] Test Flutter ↔ Unity communication
- [ ] Test on real device (60fps target)

### Testing
- [ ] Device matrix testing (iPhone 12+, Pixel 6+)
- [ ] Performance benchmarks (FPS, frame time, memory)
- [ ] Age morphing verification (6 variants)
- [ ] Gaze tracking smoothness
- [ ] Micro-expressions (blinking, breathing)

### Deployment
- [ ] Build production Unity bundle
- [ ] Integrate with Flutter app
- [ ] Submit to App Store / Google Play

---

## 🎉 **Next Steps**

After completing the avatar upgrade:

1. **Week 2-3:** Voice integration (OpenAI Realtime API)
2. **Week 3-4:** Lesson content creation (30 universal topics)
3. **Week 4-5:** Mobile app polish (UI/UX)
4. **Week 5-6:** Testing & QA (device matrix)
5. **Week 6:** Beta launch (TestFlight/Play Internal)

---

**Questions?** Check `60FPS_SETUP_GUIDE.md` for iClone rendering at 60fps.

**Performance issues?** Enable `showDebugInfo = true` and check console logs.

**Need help?** Review `Curious-Kellly_Technical_Blueprint.md` for architecture details.















