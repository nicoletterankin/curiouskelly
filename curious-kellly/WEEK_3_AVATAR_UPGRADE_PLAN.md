# Week 3 - Avatar Upgrade & Audio Sync Plan

**Status**: 🚀 **IN PROGRESS**  
**Start Date**: November 11, 2025  
**Timeline**: 5 days

---

## 🎯 Goals

Upgrade the Unity avatar to 60 FPS with gaze tracking, micro-expressions, and calibrated audio sync.

---

## ✅ What's Already Done

### Voice Integration (Week 2) ✅
- ✅ Flutter Realtime Voice Client complete
- ✅ Backend WebSocket handler operational
- ✅ Safety moderation integrated
- ✅ Session management connected
- ✅ Barge-in/barge-out support
- ✅ Viseme service created

### Current Unity System ✅
- ✅ BlendshapeDriver working (Audio2Face JSON playback)
- ✅ AutoBlink system (natural blinking every 3-6s)
- ✅ BreathingLayer (subtle breathing micro-expression)
- ✅ KellyBridge (Flutter communication)
- ✅ Basic FBX model with blendshapes

---

## 🔨 Week 3 Tasks

### Task 1: Unity 60 FPS Upgrade (Day 1-2)

**Goal**: Upgrade Unity avatar from variable FPS to locked 60 FPS

**What to Do:**
1. ✅ **Profile current frame rate**
   - Add FPS counter to scene
   - Measure on target devices (iPhone 12, Pixel 6)
   - Identify bottlenecks

2. ✅ **Optimize BlendshapeDriver**
   - Use `FixedUpdate()` or `LateUpdate()` for consistent timing
   - Cache blendshape indices
   - Minimize `SetBlendShapeWeight()` calls
   - Only update changed weights

3. ✅ **GPU Optimization**
   - Enable GPU skinning
   - Optimize mesh vertex count
   - Use URP optimized shaders
   - Enable dynamic batching

4. ✅ **Lock frame rate to 60 FPS**
   - Set `Application.targetFrameRate = 60`
   - Ensure VSync is configured properly
   - Test on both iOS and Android

**Files to Modify:**
- `BlendshapeDriver.cs` - Performance optimization
- `Main.unity` - Project settings
- Add: `FPSCounter.cs` - Frame rate monitoring

---

### Task 2: Gaze Tracking System (Day 2-3)

**Goal**: Implement natural gaze tracking with micro-saccades

**What to Do:**
1. ✅ **Create Gaze Controller**
   - Define gaze targets (camera, left, right, up, down, content)
   - Smooth eye rotation using Slerp
   - Add micro-saccades (2-4 per second)
   - Random eye movements when idle

2. ✅ **Screen-Space Gaze**
   - Map screen positions to eye rotations
   - Follow user touch/interaction points
   - Return to neutral when idle

3. ✅ **Integration with Teaching Moments**
   - Read gaze targets from PhaseDNA expression cues
   - Sync gaze with speech timing
   - Look at content when referencing concepts

**Files to Create:**
- `GazeController.cs` - NEW: Gaze tracking system
- `GazeTarget.cs` - NEW: Target definition model
- `MicroSaccade.cs` - NEW: Micro-saccade generator

**Files to Modify:**
- `KellyBridge.cs` - Add gaze message handling
- `BlendshapeDriver.cs` - Integrate gaze with blendshapes

---

### Task 3: Blendshape Mapping for Visemes (Day 3)

**Goal**: Map OpenAI Realtime visemes to Unity blendshapes

**What to Do:**
1. ✅ **Create Viseme Mapper**
   - Map OpenAI viseme IDs to A2F blendshape names
   - Handle missing blendshapes gracefully
   - Smooth transitions between visemes

2. ✅ **Real-time Viseme Updates**
   - Receive visemes from Flutter via KellyBridge
   - Apply blendshapes with proper timing
   - Interpolate for smooth lip-sync

3. ✅ **Testing**
   - Test with known audio samples
   - Verify lip-sync accuracy (<5% error)
   - Measure frame latency

**Viseme Mapping:**
```csharp
// OpenAI Realtime → Audio2Face blendshapes
{
    "sil": "jawOpen:0",           // Silence
    "PP": "mouthPucker",          // P, B, M sounds
    "FF": "mouthFunnel",          // F, V sounds
    "TH": "tongue_out",           // Th sounds
    "DD": "jawOpen:40",           // D, T sounds
    "kk": "jawOpen:20",           // K, G sounds
    "CH": "mouthShrugUpper",      // Ch, J sounds
    "SS": "mouthSmile",           // S, Z sounds
    "nn": "jawOpen:30",           // N sounds
    "RR": "mouthRollUpper",       // R sounds
    "aa": "jawOpen:60",           // Ah sound
    "E": "mouthSmile:60",         // Ee sound
    "I": "mouthSmile:40",         // Ih sound
    "O": "mouthFunnel:50",        // Oh sound
    "U": "mouthPucker:60"         // Oo sound
}
```

**Files to Create:**
- `VisemeMapper.cs` - NEW: Viseme to blendshape mapping
- `RealtimeVisemeDriver.cs` - NEW: Real-time viseme updates

**Files to Modify:**
- `KellyBridge.cs` - Add viseme message handling

---

### Task 4: Expression Cues from PhaseDNA (Day 3-4)

**Goal**: Add micro and macro expressions from teaching moment cues

**What to Do:**
1. ✅ **Parse Expression Cues**
   - Read expressionCues from PhaseDNA
   - Parse type, offset, duration, intensity
   - Schedule expressions based on audio timeline

2. ✅ **Expression Types**
   - Micro-smile: Subtle smile (corners of mouth)
   - Macro-gesture: Eyebrow raise, head nod
   - Gaze-shift: Change gaze target
   - Brow-raise: Raise eyebrows for emphasis
   - Head-nod: Agree

ment nod
   - Breath: Breathing pause

3. ✅ **Blending System**
   - Layer expressions on top of speech blendshapes
   - Blend with proper weights
   - Don't override lip-sync

**Expression Cue Format** (from PhaseDNA):
```json
{
  "id": "cue-001",
  "momentRef": "tm-explain-chlorophyll",
  "type": "micro-smile",
  "offset": 2.5,
  "duration": 1.0,
  "intensity": "subtle",
  "gazeTarget": "content"
}
```

**Files to Create:**
- `ExpressionCueDriver.cs` - NEW: Expression cue system
- `ExpressionBlender.cs` - NEW: Blend expressions with speech

**Files to Modify:**
- `KellyBridge.cs` - Load expression cues from lesson data
- `BlendshapeDriver.cs` - Add expression blending layer

---

### Task 5: Audio Sync Calibration (Day 4-5)

**Goal**: Build calibration system for frame-accurate lip-sync

**What to Do:**
1. ✅ **Create Calibration UI**
   - Slider: -60ms to +60ms
   - Visual indicator: Current offset
   - Test button: Play sample audio
   - Save/Load user preference

2. ✅ **Offset Application**
   - Apply offset to all audio playback
   - Adjust blendshape timing accordingly
   - Persist per-device

3. ✅ **Testing Matrix**
   - Test on 5 devices (2 iOS, 3 Android)
   - Measure lip-sync error
   - Target: <5% error

**Files to Create:**
- `AudioSyncCalibrator.cs` - NEW: Calibration system
- `CalibrationUI.cs` - NEW: Calibration UI (Flutter)

**Files to Modify:**
- `BlendshapeDriver.cs` - Add offset support
- `AudioSource` configuration - Apply offset

---

### Task 6: Device Matrix Testing (Day 5)

**Goal**: Test on target devices and measure performance

**Devices to Test:**
- iPhone 12
- iPhone 13
- iPhone 14
- iPhone 15
- Pixel 6
- Pixel 7
- Pixel 8

**Metrics to Measure:**
- Frame rate (target: 60 FPS)
- Lip-sync error (target: <5%)
- Audio latency (target: <100ms)
- CPU usage (target: <30%)
- GPU usage (target: <50%)
- Memory usage (target: <500MB)

**Test Cases:**
1. Idle avatar (breathing only)
2. Speaking with lip-sync
3. Teaching moment with expressions
4. Gaze tracking + expressions
5. Barge-in scenario

**Files to Create:**
- `PerformanceMonitor.cs` - NEW: Real-time metrics
- `DeviceTestReport.md` - Test results

---

## 📊 Success Criteria

### Performance
- ✅ 60 FPS on iPhone 12 and Pixel 6
- ✅ Lip-sync error <5%
- ✅ Audio latency <100ms
- ✅ CPU usage <30%
- ✅ GPU usage <50%

### Visual Quality
- ✅ Natural gaze with micro-saccades (2-4/s)
- ✅ Smooth blending of expressions
- ✅ No jitter or stuttering
- ✅ Proper eye contact

### Accuracy
- ✅ Frame-accurate lip-sync
- ✅ Expression timing matches audio
- ✅ Gaze follows teaching moment cues

---

## 🔄 Integration Flow

```
Flutter (Lesson Screen)
  ↓ [Teaching Moment Data + Expression Cues]
  ↓
Unity Bridge (KellyBridge.cs)
  ├─→ AudioSource (audio playback)
  ├─→ BlendshapeDriver (lip-sync)
  ├─→ VisemeMapper (real-time visemes)
  ├─→ GazeController (gaze tracking)
  ├─→ ExpressionCueDriver (expressions)
  └─→ AudioSyncCalibrator (offset)
```

---

## 📝 Implementation Order

**Day 1:** FPS optimization + profiling
**Day 2:** Gaze tracking system
**Day 3:** Viseme mapping + expression cues
**Day 4:** Audio sync calibration
**Day 5:** Device testing + bug fixes

---

## 🧪 Testing Checkpoints

### Checkpoint 1 (End of Day 2)
- [ ] 60 FPS on target devices
- [ ] FPS counter working
- [ ] Gaze tracking functional
- [ ] Micro-saccades visible

### Checkpoint 2 (End of Day 3)
- [ ] Visemes mapped correctly
- [ ] Real-time lip-sync working
- [ ] Expression cues trigger
- [ ] Smooth blending

### Checkpoint 3 (End of Day 4)
- [ ] Calibration UI working
- [ ] Offset applied correctly
- [ ] Lip-sync error <5%
- [ ] User preference saved

### Checkpoint 4 (End of Day 5)
- [ ] All devices tested
- [ ] Performance metrics logged
- [ ] All targets met
- [ ] Bug fixes complete

---

## 🚀 Getting Started

### Setup Unity Project

```bash
# 1. Open Unity project
cd digital-kelly/engines/kelly_unity_player/My\ project

# 2. Open in Unity 2022.3 LTS
# File → Open Project

# 3. Open Main.unity scene
# Assets/Kelly/Scenes/Main.unity

# 4. Add FPS counter to scene
# GameObject → Create Empty → Add FPSCounter.cs
```

### Create New Scripts

```bash
cd Assets/Kelly/Scripts

# Create new scripts
touch FPSCounter.cs
touch GazeController.cs
touch GazeTarget.cs
touch MicroSaccade.cs
touch VisemeMapper.cs
touch RealtimeVisemeDriver.cs
touch ExpressionCueDriver.cs
touch ExpressionBlender.cs
touch AudioSyncCalibrator.cs
touch PerformanceMonitor.cs
```

---

## 📖 Resources

**Unity Documentation:**
- [SkinnedMeshRenderer](https://docs.unity3d.com/ScriptReference/SkinnedMeshRenderer.html)
- [Application.targetFrameRate](https://docs.unity3d.com/ScriptReference/Application-targetFrameRate.html)
- [Quaternion.Slerp](https://docs.unity3d.com/ScriptReference/Quaternion.Slerp.html)
- [LateUpdate](https://docs.unity3d.com/ScriptReference/MonoBehaviour.LateUpdate.html)

**Reference Implementation:**
- `BlendshapeDriver.cs` - Current blendshape system
- `AutoBlink.cs` - Natural blinking example
- `BreathingLayer.cs` - Micro-expression example

**PhaseDNA Schema:**
- `curious-kellly/backend/config/lesson-dna-schema.json`
- Expression cues defined per age variant

---

## ✅ Deliverables

1. ✅ Unity project running at 60 FPS on target devices
2. ✅ Gaze tracking with micro-saccades
3. ✅ Viseme-based lip-sync
4. ✅ Expression cues from PhaseDNA
5. ✅ Audio sync calibration system
6. ✅ Device testing report
7. ✅ Performance metrics documented
8. ✅ All scripts commented and documented

---

**Status**: 🚧 Ready to start implementation  
**Next Step**: Create FPSCounter.cs and profile current performance  
**Timeline**: Complete by end of Week 3



