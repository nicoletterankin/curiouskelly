# Avatar 60fps Testing & Validation Plan
**Date**: November 16, 2025
**Status**: TESTING IN PROGRESS
**Target**: Validate 60fps avatar with 558 generated audio files

---

## 🎯 Testing Objectives

### Primary Goals:
1. ✅ Validate Unity project structure and scripts
2. ⏳ Test 60fps performance on target devices
3. ⏳ Validate audio sync with 558 generated MP3 files
4. ⏳ Test lip-sync accuracy across all age variants
5. ⏳ Verify gaze tracking and micro-expressions
6. ⏳ Document performance metrics

---

## 📋 Pre-Testing Validation

### Unity Project Structure ✅

**Scripts Verified:**
- ✅ `BlendshapeDriver60fps.cs` (24,143 bytes)
- ✅ `KellyAvatarController.cs` (15,201 bytes)
- ✅ `GazeController.cs` (8,400 bytes)
- ✅ `VisemeMapper.cs` (8,436 bytes)
- ✅ `ExpressionCueDriver.cs` (11,763 bytes)
- ✅ `OptimizedBlendshapeDriver.cs` (9,761 bytes)
- ✅ `AudioSyncCalibrator.cs` (9,003 bytes)
- ✅ `PerformanceMonitor.cs` (8,677 bytes)
- ✅ `AvatarPerformanceMonitor.cs` (6,670 bytes)
- ✅ `FPSCounter.cs` (4,979 bytes)
- ✅ `AutoBlink.cs` (1,649 bytes)
- ✅ `BreathingLayer.cs` (791 bytes)

**Total Code**: ~109 KB of C# scripts

### Audio Files Available ✅

**Generated Audio:**
- ✅ 558 MP3 files (191.3 MB)
- ✅ 10 lessons complete
- ✅ All age variants (6 buckets)
- ✅ All languages (EN/ES/FR)
- ✅ All sections (welcome, mainContent, wisdomMoment)

---

## 🧪 Test Matrix

### 1. Unity Project Build Test

**Objective**: Ensure Unity project can build for target platforms

**Steps**:
```bash
# Open Unity project
cd digital-kelly/engines/kelly_unity_player/

# Check Unity version
unity -version

# Build for Android
unity -batchmode -projectPath . -buildTarget Android -quit

# Build for iOS
unity -batchmode -projectPath . -buildTarget iOS -quit
```

**Success Criteria**:
- [ ] Project opens without errors
- [ ] No missing script references
- [ ] All prefabs valid
- [ ] Build completes successfully

---

### 2. Performance Testing

**Target Devices**:
- [ ] iPhone 12 (A14 chip)
- [ ] iPhone 13 (A15 chip)
- [ ] iPhone 14 (A16 chip)
- [ ] Pixel 6 (Tensor)
- [ ] Pixel 7 (Tensor G2)
- [ ] Pixel 8 (Tensor G3)

**Metrics to Measure**:
```
Target Performance:
- FPS: 60 ± 5% (57-63 acceptable)
- Frame Time: <16.67ms
- CPU Usage: <30%
- GPU Usage: <50%
- Memory: <500MB
- Lip-sync Error: <5%
- Audio Latency: <100ms
```

**Test Scenarios**:
1. **Idle Avatar** - Breathing only
2. **Speaking** - Lip-sync active
3. **Teaching Moment** - Expressions + gaze
4. **Continuous Playback** - 5 minutes
5. **Stress Test** - All age variants

---

### 3. Audio Integration Test

**Objective**: Test audio playback with all 558 generated files

**Test Cases by Lesson**:

#### The Sun (54 files)
- [ ] Load audio files from `config/audio/the-sun/`
- [ ] Test age 2-5 (9 files: EN/ES/FR × 3 sections)
- [ ] Test age 6-12 (9 files)
- [ ] Test age 13-17 (9 files)
- [ ] Test age 18-35 (9 files)
- [ ] Test age 36-60 (9 files)
- [ ] Test age 61-102 (9 files)
- [ ] Verify audio playback smooth
- [ ] Verify no crackling/distortion

#### Puppies (54 files)
- [ ] Same test pattern as The Sun
- [ ] Verify larger files play correctly (up to 2.1 MB)

#### The Ocean (54 files)
- [ ] Same test pattern
- [ ] Test longest audio files (up to 2.3 MB)

#### The Moon (54 files)
- [ ] Same test pattern

#### Water Cycle (72 files) - Pre-existing
- [ ] Verify compatibility with new system
- [ ] Test all 72 files play correctly

#### Molecular Biology (54 files)
- [ ] Same test pattern
- [ ] Test smaller files (avg 4.2 MB total / 54 = ~78 KB/file)

#### Creative Writing (54 files)
- [ ] Same test pattern

#### Poetry (54 files)
- [ ] Same test pattern

#### Dance Expression (54 files)
- [ ] Same test pattern

#### Negotiation Skills (54 files)
- [ ] Same test pattern

**Audio Test Script**:
```csharp
// Unity test script to load and play all audio files
public class AudioIntegrationTest : MonoBehaviour
{
    void TestLesson(string lessonName)
    {
        var audioPath = $"config/audio/{lessonName}/";
        var ageBuckets = new[] { "2-5", "6-12", "13-17", "18-35", "36-60", "61-102" };
        var languages = new[] { "en", "es", "fr" };
        var sections = new[] { "welcome", "mainContent", "wisdomMoment" };

        foreach (var age in ageBuckets)
        {
            foreach (var lang in languages)
            {
                foreach (var section in sections)
                {
                    var filename = $"{age}-{section}-{lang}.mp3";
                    var fullPath = Path.Combine(audioPath, filename);

                    StartCoroutine(TestAudioFile(fullPath, age, lang, section));
                }
            }
        }
    }

    IEnumerator TestAudioFile(string path, string age, string lang, string section)
    {
        Debug.Log($"Testing: {path}");

        // Load audio
        var clip = LoadAudioClip(path);

        // Verify loaded
        Assert.IsNotNull(clip, $"Failed to load: {path}");

        // Play audio
        audioSource.clip = clip;
        audioSource.Play();

        // Wait for playback
        yield return new WaitWhile(() => audioSource.isPlaying);

        Debug.Log($"✅ Passed: {age}-{section}-{lang}");
    }
}
```

---

### 4. Lip-Sync Accuracy Test

**Objective**: Verify blendshape sync with audio

**Test Method**:
1. Record video of avatar speaking
2. Analyze frame-by-frame
3. Measure lip-sync error:
   ```
   Error = |Audio_Time - Blendshape_Time| / Audio_Duration * 100%
   Target: <5% error
   ```

**Test Files** (Representative Sample):
- [ ] 2-5-mainContent-en.mp3 (puppies) - 868 KB - Child content
- [ ] 6-12-mainContent-en.mp3 (the-sun) - 1.4 MB - Long form
- [ ] 18-35-mainContent-es.mp3 (the-ocean) - 2.3 MB - Spanish, long
- [ ] 61-102-wisdomMoment-fr.mp3 (the-moon) - 489 KB - French, elder

---

### 5. Age Variant Test

**Objective**: Verify all 6 Kelly age variants work correctly

**Test Matrix**:

| Learner Age | Kelly Age | Variant | Voice Pitch | Voice Speed | Test Result |
|-------------|-----------|---------|-------------|-------------|-------------|
| 2-5         | 3         | Toddler | 1.30x       | 0.90x       | [ ]         |
| 6-12        | 9         | Kid     | 1.15x       | 1.00x       | [ ]         |
| 13-17       | 15        | Teen    | 1.05x       | 1.10x       | [ ]         |
| 18-35       | 27        | Adult   | 1.00x       | 1.00x       | [ ]         |
| 36-60       | 48        | Mentor  | 0.95x       | 0.95x       | [ ]         |
| 61-102      | 82        | Elder   | 0.90x       | 0.85x       | [ ]         |

**For Each Variant**:
- [ ] Model loads correctly
- [ ] Blendshapes work
- [ ] Audio pitch adjusted
- [ ] Audio speed adjusted
- [ ] Blink frequency correct
- [ ] Micro-saccade rate correct

---

### 6. Gaze & Expression Test

**Gaze Tracking**:
- [ ] Eyes follow gaze target
- [ ] Smooth Slerp interpolation
- [ ] Micro-saccades (2-4/sec)
- [ ] Max angle clamping (±30°)
- [ ] Screen-space targeting works

**Micro-Expressions**:
- [ ] Auto-blinking (8-12/min)
- [ ] Breathing animations
- [ ] MicroSmile
- [ ] BrowRaise
- [ ] HeadNod
- [ ] Expression blending with speech

---

### 7. Audio Sync Calibration Test

**Objective**: Measure and calibrate per-device audio offset

**Test Devices**:
```
Device          | Expected Offset | Measured Offset | Pass/Fail
----------------|-----------------|-----------------|----------
iPhone 12       | -10ms           | _____ms         | [ ]
iPhone 13       | -8ms            | _____ms         | [ ]
iPhone 14       | -5ms            | _____ms         | [ ]
Pixel 6         | +5ms            | _____ms         | [ ]
Pixel 7         | +3ms            | _____ms         | [ ]
Pixel 8         | +2ms            | _____ms         | [ ]
```

**Calibration Process**:
1. Play test audio with visual marker
2. Record video at 60fps
3. Frame-by-frame analysis
4. Calculate offset
5. Apply to `AudioSyncCalibrator.cs`
6. Re-test and verify <5% error

---

## 📊 Performance Benchmarks

### Expected Results

**iPhone 12+ (A14+)**:
```
FPS:          60-63 fps
Frame Time:   15-16ms
CPU Usage:    20-25%
GPU Usage:    30-40%
Memory:       350-450 MB
Status:       ✅ Excellent
```

**Pixel 6+ (Tensor+)**:
```
FPS:          58-62 fps
Frame Time:   16-17ms
CPU Usage:    25-30%
GPU Usage:    35-45%
Memory:       400-480 MB
Status:       ✅ Good
```

---

## 🔧 Test Scripts

### 1. Unity Build Test Script
```bash
#!/bin/bash
# test_unity_build.sh

UNITY_PATH="/Applications/Unity/Hub/Editor/2022.3.10f1/Unity.app/Contents/MacOS/Unity"
PROJECT_PATH="digital-kelly/engines/kelly_unity_player"

echo "Testing Unity project build..."

$UNITY_PATH -batchmode -quit \
  -projectPath "$PROJECT_PATH" \
  -executeMethod BuildScript.BuildAndroid \
  -logFile build.log

if [ $? -eq 0 ]; then
  echo "✅ Build successful"
else
  echo "❌ Build failed - check build.log"
  cat build.log
fi
```

### 2. Audio File Validation Script
```python
#!/usr/bin/env python3
# validate_audio_files.py

import os
from pathlib import Path
from mutagen.mp3 import MP3

def validate_lesson_audio(lesson_name):
    audio_dir = Path(f'curious-kellly/backend/config/audio/{lesson_name}')

    age_buckets = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102']
    languages = ['en', 'es', 'fr']
    sections = ['welcome', 'mainContent', 'wisdomMoment']

    expected = len(age_buckets) * len(languages) * len(sections)
    found = 0
    errors = []

    for age in age_buckets:
        for lang in languages:
            for section in sections:
                filename = f'{age}-{section}-{lang}.mp3'
                filepath = audio_dir / filename

                if not filepath.exists():
                    errors.append(f'Missing: {filename}')
                    continue

                # Validate MP3
                try:
                    audio = MP3(filepath)
                    duration = audio.info.length
                    bitrate = audio.info.bitrate

                    if bitrate < 96000:  # 96 kbps minimum
                        errors.append(f'Low bitrate ({bitrate}): {filename}')

                    if duration < 1.0:  # Minimum 1 second
                        errors.append(f'Too short ({duration}s): {filename}')

                    found += 1

                except Exception as e:
                    errors.append(f'Invalid MP3 {filename}: {e}')

    print(f'\n{lesson_name}:')
    print(f'  Expected: {expected} files')
    print(f'  Found: {found} files')
    print(f'  Errors: {len(errors)}')

    if errors:
        print('  Issues:')
        for err in errors:
            print(f'    - {err}')

    return len(errors) == 0

# Test all lessons
lessons = [
    'the-sun', 'puppies', 'the-ocean', 'the-moon',
    'water-cycle', 'molecular-biology-dna',
    'creative-writing-dna', 'poetry-dna',
    'dance-expression-dna', 'negotiation-skills-dna'
]

all_pass = True
for lesson in lessons:
    if not validate_lesson_audio(lesson):
        all_pass = False

print('\n' + '='*50)
if all_pass:
    print('✅ All lessons validated successfully!')
else:
    print('❌ Some lessons have errors - see above')
```

### 3. Performance Monitoring Script
```csharp
// PerformanceTestRunner.cs
using UnityEngine;
using System.Collections;

public class PerformanceTestRunner : MonoBehaviour
{
    public AvatarPerformanceMonitor monitor;
    public KellyAvatarController avatar;

    void Start()
    {
        StartCoroutine(RunPerformanceTests());
    }

    IEnumerator RunPerformanceTests()
    {
        yield return new WaitForSeconds(2f); // Warmup

        // Test 1: Idle
        Debug.Log("=== Test 1: Idle Avatar ===");
        yield return TestScenario("Idle", 30f);

        // Test 2: Speaking
        Debug.Log("=== Test 2: Speaking ===");
        avatar.Speak("Why do leaves change color in autumn?", 35);
        yield return TestScenario("Speaking", 30f);

        // Test 3: Teaching moment
        Debug.Log("=== Test 3: Teaching Moment ===");
        avatar.PlayLesson("the-sun", 35);
        yield return TestScenario("Teaching", 60f);

        // Test 4: Continuous playback
        Debug.Log("=== Test 4: Continuous (5 min) ===");
        yield return TestScenario("Continuous", 300f);

        // Print summary
        PrintSummary();
    }

    IEnumerator TestScenario(string name, float duration)
    {
        monitor.ResetStats();
        float elapsed = 0f;

        while (elapsed < duration)
        {
            yield return new WaitForSeconds(1f);
            elapsed += 1f;

            var stats = monitor.GetCurrentStats();
            Debug.Log($"[{name}] {elapsed}s - FPS: {stats.averageFPS:F1}, CPU: {stats.cpuUsage:F1}%");
        }

        var finalStats = monitor.GetCurrentStats();
        Debug.Log($"[{name}] FINAL - Avg FPS: {finalStats.averageFPS:F1}, Min: {finalStats.minFPS:F1}, Max: {finalStats.maxFPS:F1}");
    }

    void PrintSummary()
    {
        Debug.Log("===========================================");
        Debug.Log("PERFORMANCE TEST SUMMARY");
        Debug.Log("===========================================");
        Debug.Log(monitor.GetPerformanceReport());
    }
}
```

---

## ✅ Test Execution Checklist

### Pre-Test Setup
- [ ] Unity project opens without errors
- [ ] All scripts compile successfully
- [ ] No missing prefab references
- [ ] Kelly avatar models present (6 variants)
- [ ] Audio files accessible in build

### Phase 1: Unity Editor Tests
- [ ] Run in editor (Play mode)
- [ ] FPS counter shows 60fps
- [ ] Performance monitor working
- [ ] Avatar loads correctly
- [ ] Test controls functional

### Phase 2: Device Build Tests
- [ ] Build APK/AAB for Android
- [ ] Build for iOS (TestFlight)
- [ ] Install on test devices
- [ ] Launch and verify no crashes

### Phase 3: Performance Tests
- [ ] Run on iPhone 12
- [ ] Run on Pixel 6
- [ ] Measure all metrics
- [ ] Document results

### Phase 4: Audio Integration Tests
- [ ] Test all 558 files
- [ ] Verify playback quality
- [ ] Test age variants
- [ ] Test language switching

### Phase 5: Lip-Sync Tests
- [ ] Record test videos
- [ ] Frame-by-frame analysis
- [ ] Calculate sync error
- [ ] Apply calibration

### Phase 6: Validation
- [ ] All metrics meet targets
- [ ] No critical bugs
- [ ] Documentation complete
- [ ] Ready for production

---

## 📝 Test Results Template

```markdown
## Test Results: [Device Name]

**Device**: iPhone 12 Pro
**OS**: iOS 17.1
**Unity Version**: 2022.3.10f1
**Test Date**: 2025-11-16
**Tester**: Claude

### Performance Metrics

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Average FPS | 60 | 61.2 | ✅ Pass |
| Min FPS | 55+ | 58.1 | ✅ Pass |
| Frame Time | <16.67ms | 16.39ms | ✅ Pass |
| CPU Usage | <30% | 24.5% | ✅ Pass |
| GPU Usage | <50% | 38.2% | ✅ Pass |
| Memory | <500MB | 425MB | ✅ Pass |
| Lip-sync Error | <5% | 2.8% | ✅ Pass |

### Audio Test Results
- ✅ All 558 files loaded successfully
- ✅ No playback issues
- ✅ Quality excellent across all files
- ✅ Age variants work correctly

### Issues Found
None

### Overall: ✅ PASS
```

---

## 🚀 Next Steps After Testing

### If All Tests Pass:
1. Document final performance metrics
2. Create production build
3. Submit to app stores
4. Begin user beta testing

### If Issues Found:
1. Document all bugs
2. Prioritize by severity
3. Create fix plan
4. Re-test after fixes

---

**Status**: Ready for testing
**Estimated Duration**: 4-6 hours
**Required Resources**: iPhone 12+, Pixel 6+, Unity 2022.3+
