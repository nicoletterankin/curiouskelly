# Device Testing Guide - Week 3

**Purpose**: Validate 60 FPS Unity avatar performance across target devices  
**Duration**: Days 2-5 of Week 3  
**Target Devices**: 7 devices (4 iOS, 3 Android)

---

## 🎯 Testing Objectives

1. ✅ Validate 60 FPS performance on all devices
2. ✅ Measure CPU/GPU/memory usage
3. ✅ Determine per-device audio sync offsets
4. ✅ Validate lip-sync accuracy (<5% error)
5. ✅ Test 5-minute continuous playback
6. ✅ Document device-specific issues

---

## 📱 Target Devices

### iOS Devices:
1. **iPhone 12**
   - iOS 15+
   - A14 Bionic chip
   - Expected offset: -10ms

2. **iPhone 13**
   - iOS 16+
   - A15 Bionic chip
   - Expected offset: -8ms

3. **iPhone 14**
   - iOS 17+
   - A15 Bionic chip
   - Expected offset: -5ms

4. **iPhone 15**
   - iOS 17+
   - A16 Bionic chip
   - Expected offset: -3ms

### Android Devices:
1. **Pixel 6**
   - Android 12+
   - Google Tensor chip
   - Expected offset: +5ms

2. **Pixel 7**
   - Android 13+
   - Google Tensor G2 chip
   - Expected offset: +3ms

3. **Pixel 8**
   - Android 14+
   - Google Tensor G3 chip
   - Expected offset: +2ms

---

## 🔧 Setup Instructions

### 1. Build Unity Project

```bash
# Open Unity 2022.3 LTS
cd digital-kelly/engines/kelly_unity_player/My\ project

# In Unity:
# 1. Open Main.unity scene
# 2. Verify all Week 3 scripts are attached to KellyController
# 3. File → Build Settings
# 4. Select iOS or Android
# 5. Build project
```

### 2. Attach Week 3 Components

In Unity scene, ensure `KellyController` has:
- ✅ `KellyBridge`
- ✅ `OptimizedBlendshapeDriver`
- ✅ `VisemeMapper`
- ✅ `GazeController` (with eye bones assigned)
- ✅ `ExpressionCueDriver`
- ✅ `AudioSyncCalibrator`
- ✅ `AudioSource`

Add to scene root:
- ✅ `FPSCounter` (press F3 to toggle)
- ✅ `PerformanceMonitor` (press F4 to toggle)

### 3. Deploy to Devices

**iOS:**
```bash
# Export to Xcode
# Open .xcodeproj in Xcode
# Select device
# Run
```

**Android:**
```bash
# Export to Android Studio
# Open project in Android Studio
# Select device
# Run
```

---

## 📊 Test Scenarios

### Test 1: Idle Avatar (Baseline)
**Duration**: 1 minute  
**What to Test**: Baseline performance with minimal activity

**Steps:**
1. Launch app
2. Wait for avatar to load
3. Observe for 1 minute (breathing + blinking only)
4. Record metrics

**Expected Results:**
- FPS: 60 FPS (stable)
- CPU: < 20%
- GPU: < 30%
- Memory: < 300MB

---

### Test 2: Speaking with Lip-Sync
**Duration**: 2 minutes  
**What to Test**: Lip-sync performance with Audio2Face

**Steps:**
1. Load test lesson ("Why Do Leaves Change Color?")
2. Play audio with lip-sync
3. Observe for 2 minutes
4. Record metrics

**Expected Results:**
- FPS: 60 FPS (stable)
- CPU: < 30%
- GPU: < 40%
- Memory: < 400MB
- Lip-sync error: < 5%

**How to Measure Lip-Sync Error:**
1. Record video of avatar speaking
2. Count frames where mouth doesn't match audio
3. Error = (mismatched frames / total frames) × 100%

---

### Test 3: Teaching Moment with Expressions
**Duration**: 2 minutes  
**What to Test**: Expression cues + gaze tracking

**Steps:**
1. Load lesson with expression cues
2. Play teaching moment
3. Verify expressions trigger at correct times
4. Verify gaze shifts work
5. Record metrics

**Expected Results:**
- FPS: 58-60 FPS (slight dip acceptable)
- CPU: < 35%
- GPU: < 50%
- Memory: < 450MB
- Expression timing: < 50ms error

---

### Test 4: Real-Time Viseme Updates
**Duration**: 2 minutes  
**What to Test**: Real-time viseme streaming (OpenAI Realtime API simulation)

**Steps:**
1. Connect to backend WebSocket
2. Simulate viseme stream (send test visemes)
3. Observe lip movement
4. Record metrics

**Expected Results:**
- FPS: 60 FPS (stable)
- CPU: < 30%
- GPU: < 45%
- Viseme latency: < 50ms

---

### Test 5: Gaze Tracking + Expressions
**Duration**: 2 minutes  
**What to Test**: Combined gaze + expressions

**Steps:**
1. Enable gaze tracking
2. Enable micro-saccades
3. Trigger various expressions
4. Touch screen to test gaze-to-touch
5. Record metrics

**Expected Results:**
- FPS: 58-60 FPS
- CPU: < 30%
- GPU: < 45%
- Micro-saccade frequency: 2-4/sec
- Gaze smooth: no jitter

---

### Test 6: Barge-In Scenario
**Duration**: 1 minute  
**What to Test**: User interrupts Kelly mid-speech

**Steps:**
1. Start Kelly speaking
2. Trigger barge-in (interrupt)
3. Verify audio stops immediately
4. Verify lip-sync stops
5. Verify avatar returns to idle

**Expected Results:**
- Barge-in latency: < 100ms
- Audio stops immediately
- Blendshapes return to neutral
- No visual artifacts

---

### Test 7: 5-Minute Continuous Playback
**Duration**: 5 minutes  
**What to Test**: Stability and memory leaks

**Steps:**
1. Play lesson continuously for 5 minutes
2. Loop if necessary
3. Monitor metrics every 30 seconds
4. Check for memory growth
5. Check for FPS degradation

**Expected Results:**
- FPS: 60 FPS (stable throughout)
- CPU: < 30% (stable)
- GPU: < 50% (stable)
- Memory: < 500MB (no significant growth)
- No crashes or hangs

---

## 📝 Metrics to Collect

### Per Test:
- [ ] **FPS**: Current, average, min, max
- [ ] **CPU Usage**: Average %
- [ ] **GPU Usage**: Average %
- [ ] **Memory Usage**: MB
- [ ] **Frame Time**: Milliseconds
- [ ] **Temperature**: Device temp (if available)

### Audio Sync:
- [ ] **Calibration Offset**: Test offsets from -60ms to +60ms
- [ ] **Optimal Offset**: Find offset with best lip-sync
- [ ] **Lip-Sync Error**: % mismatched frames
- [ ] **Audio Latency**: Time from trigger to playback

### Gaze Tracking:
- [ ] **Micro-Saccade Frequency**: Saccades per second
- [ ] **Gaze Smoothness**: Visual jitter present? (yes/no)
- [ ] **Touch Response**: Latency from touch to gaze shift

### Expressions:
- [ ] **Trigger Timing**: Offset from expected time
- [ ] **Blend Quality**: Smooth transitions? (yes/no)
- [ ] **Intensity**: Correct strength? (yes/no)

---

## 📋 Testing Checklist (Per Device)

### Pre-Test:
- [ ] Device fully charged
- [ ] Latest iOS/Android version installed
- [ ] App installed and launched successfully
- [ ] FPS counter enabled (F3)
- [ ] Performance monitor enabled (F4)
- [ ] Screen recording started (for lip-sync analysis)

### During Test:
- [ ] Run all 7 test scenarios
- [ ] Record metrics for each test
- [ ] Note any visual glitches
- [ ] Note any audio issues
- [ ] Note any crashes or hangs

### Post-Test:
- [ ] Save screen recordings
- [ ] Export performance metrics (JSON)
- [ ] Document optimal audio offset
- [ ] Note device-specific issues
- [ ] Rate overall experience (1-5 stars)

---

## 📊 Data Collection Template

### Device: [iPhone 12 / iPhone 13 / etc.]

| Test Scenario | FPS (Avg) | CPU % | GPU % | Memory MB | Notes |
|--------------|-----------|-------|-------|-----------|-------|
| 1. Idle Avatar | | | | | |
| 2. Lip-Sync | | | | | |
| 3. Expressions | | | | | |
| 4. Visemes | | | | | |
| 5. Gaze + Expr | | | | | |
| 6. Barge-In | | | | | |
| 7. 5-Min Play | | | | | |

### Audio Sync Testing:

| Offset (ms) | Lip-Sync Error % | Visual Quality | Notes |
|-------------|------------------|----------------|-------|
| -60 | | | |
| -40 | | | |
| -20 | | | |
| 0 | | | |
| +20 | | | |
| +40 | | | |
| +60 | | | |

**Optimal Offset**: _____ ms  
**Lip-Sync Error at Optimal**: _____ %

### Issues Found:
1. 
2. 
3. 

### Overall Rating: ⭐⭐⭐⭐⭐ (1-5)

---

## 🔍 Troubleshooting

### Issue: FPS Below 60
**Possible Causes:**
- Too many blendshapes updating per frame
- GPU skinning not enabled
- VSync disabled
- Background processes consuming CPU

**Solutions:**
1. Reduce `maxBlendshapesPerFrame` in OptimizedBlendshapeDriver
2. Enable GPU skinning in Unity Project Settings
3. Close background apps
4. Check `Application.targetFrameRate = 60` is set

---

### Issue: High CPU Usage (>30%)
**Possible Causes:**
- Too many script updates
- Inefficient blendshape lookups
- GC (garbage collection) spikes

**Solutions:**
1. Profile with Unity Profiler
2. Verify blendshape indices are cached
3. Reduce object allocations
4. Use object pooling

---

### Issue: Poor Lip-Sync
**Possible Causes:**
- Audio offset not calibrated
- Frame timing issues
- Audio buffer underruns

**Solutions:**
1. Calibrate audio offset (-60ms to +60ms)
2. Use AudioSettings.dspTime for sync
3. Increase audio buffer size
4. Check for audio dropouts

---

### Issue: Jittery Gaze
**Possible Causes:**
- Gaze speed too high
- Micro-saccades too frequent
- Eye bone constraints

**Solutions:**
1. Reduce `gazeSpeed` (default: 3f → try 2f)
2. Reduce `saccadeFrequency` (default: 3f → try 2f)
3. Check eye bone setup in FBX model
4. Verify Slerp interpolation is working

---

### Issue: Memory Growth
**Possible Causes:**
- Memory leaks
- Audio clips not released
- Object instances not disposed

**Solutions:**
1. Call `Dispose()` on services
2. Release audio clips after use
3. Clear lists and arrays
4. Check for circular references

---

## 📤 Deliverables

### Required Outputs:
1. ✅ **Completed Testing Checklist** (for all 7 devices)
2. ✅ **Performance Metrics** (JSON exports)
3. ✅ **Audio Calibration Offsets** (per device)
4. ✅ **Screen Recordings** (lip-sync analysis)
5. ✅ **Issue Log** (bugs and glitches)
6. ✅ **Device Test Report** (summary document)

### Report Template:
See `DEVICE_TEST_REPORT_TEMPLATE.md` (to be created)

---

## 🎯 Success Criteria

### Performance Targets:
- ✅ **FPS**: ≥60 FPS on all devices
- ✅ **CPU**: <30% on all devices
- ✅ **GPU**: <50% on all devices
- ✅ **Memory**: <500MB on all devices
- ✅ **Lip-Sync Error**: <5% on all devices
- ✅ **Audio Latency**: <100ms on all devices
- ✅ **5-Min Stability**: No crashes or hangs

### Acceptable Results:
- ✅ **FPS**: 55-60 FPS (minimum 55)
- ✅ **CPU**: <35% (acceptable)
- ✅ **GPU**: <60% (acceptable)
- ✅ **Memory**: <600MB (acceptable)
- ✅ **Lip-Sync Error**: <10% (acceptable)

---

## 📞 Support

### If You Need Help:
1. Check troubleshooting section above
2. Review Unity console for errors
3. Export performance metrics (JSON)
4. Provide device model and OS version
5. Include screen recordings if visual issues

### Common Commands:

**Enable FPS Counter:**
- Press `F3` in app

**Enable Performance Monitor:**
- Press `F4` in app

**Export Metrics:**
```csharp
// In Unity console or Flutter
string metrics = performanceMonitor.ExportToJson();
Debug.Log(metrics);
```

**Test Audio Calibration:**
```csharp
// Via KellyBridge
kellyBridge.SetAudioOffset(-10.0f); // -10ms
kellyBridge.PlayCalibrationTest();
```

---

## ✅ Final Checklist

Before concluding testing:
- [ ] All 7 devices tested
- [ ] All 7 test scenarios completed per device
- [ ] Performance metrics exported
- [ ] Audio offsets documented
- [ ] Screen recordings saved
- [ ] Issues logged
- [ ] Device test report created
- [ ] Optimal settings documented

---

**Status**: Ready to begin testing!  
**Estimated Time**: 4 days (1-2 hours per device)  
**Next Step**: Deploy to first device (iPhone 12) and run Test 1

Good luck! 🚀


