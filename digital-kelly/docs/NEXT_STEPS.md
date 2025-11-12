# Next Steps - Kelly OS Development Tasks

Auto-generated GitHub issues for incremental development.

## Issue 1: Replace Placeholder Mesh

**Title:** Replace placeholder mesh with real Kelly_Head.fbx

**Description:**
Currently using a sphere with a test "jawOpen" blendshape. Replace with actual Kelly_Head.fbx model.

**Acceptance Criteria:**
- [ ] Drag Kelly_Head.fbx into Assets/Kelly/Models
- [ ] Assign to BlendshapeDriver.headRenderer in scene
- [ ] Verify jaw, blink, smile blendshapes visible and animating
- [ ] Console shows proper blendshape names being indexed
- [ ] Facial animation plays in sync with audio

**File Changes:**
- `engines/kelly_unity_player/Assets/Kelly/Models/Kelly_Head.fbx` (add)
- `Assets/Kelly/Scenes/Main.unity` (assign mesh)

---

## Issue 2: Flutter Path Copier

**Title:** Add path copier in Flutter to move audio/json to documents dir on first run

**Description:**
Create a "Copy Demo Assets" button that copies placeholder assets from app bundle to writable storage.

**Acceptance Criteria:**
- [ ] Button "Copy Demo Assets" visible on first launch
- [ ] Creates files in documents directory
- [ ] Logs absolute paths to console
- [ ] Subsequent launches skip button (check file exists)
- [ ] Assets load correctly for Unity playback

**File Changes:**
- `apps/kelly_app_flutter/lib/services/asset_copier.dart` (add)
- `apps/kelly_app_flutter/lib/main.dart` (add button logic)
- Update `pubspec.yaml` assets list

---

## Issue 3: Lesson JSON Loader

**Title:** Add lesson JSON loader page & bind to Play

**Description:**
Create a list view of lessons. Tapping a lesson loads its A2F/Audio and plays.

**Acceptance Criteria:**
- [ ] Lesson list page displays all lessons in assets/lessons/
- [ ] Tapping a lesson item loads its A2F and Audio
- [ ] Loading triggers Unity playback via KellyBridge
- [ ] Lesson title and script displayed during playback
- [ ] Back button returns to lesson list

**File Changes:**
- `apps/kelly_app_flutter/lib/pages/lessons_page.dart` (add)
- `apps/kelly_app_flutter/lib/pages/player_page.dart` (add)
- Update navigation in `main.dart`
- Add lesson JSON files to `assets/lessons/`

---

## Issue 4: Delay Calibration Slider

**Title:** Add delay calibration slider (±60 ms)

**Description:**
Allow users to nudge audio-lip-sync timing for their device.

**Acceptance Criteria:**
- [ ] Slider visible in Settings/Controls overlay
- [ ] Range: ±60ms (±1.8 frames at 30fps)
- [ ] Value stored in SharedPreferences / NSUserDefaults
- [ ] Calibration applied to BlendshapeDriver.PlaySynced(delay)
- [ ] Persists across app restarts

**File Changes:**
- `apps/kelly_app_flutter/lib/pages/settings_page.dart` (add)
- `apps/kelly_app_flutter/lib/services/preferences.dart` (add)
- `engines/kelly_unity_player/Assets/Kelly/Scripts/BlendshapeDriver.cs` (add delay param)

---

## Issue 5: Blink/Breathing Priority

**Title:** Wire breathing + blink priority

**Description:**
If A2F provides blink data, AutoBlink should pause. Otherwise, AutoBlink runs.

**Acceptance Criteria:**
- [ ] AutoBlink checks if active A2F frame has blink data
- [ ] If yes: AutoBlink pauses (overriden)
- [ ] If no: AutoBlink runs normally
- [ ] Same logic for BreathingLayer
- [ ] No conflicts visible during speech animation

**File Changes:**
- `engines/kelly_unity_player/Assets/Kelly/Scripts/AutoBlink.cs` (add pause flag)
- `engines/kelly_unity_player/Assets/Kelly/Scripts/BlendshapeDriver.cs` (notify layers)
- `engines/kelly_unity_player/Assets/Kelly/Scripts/BreathingLayer.cs` (add pause logic)

---

## Additional Future Tasks

- Add A/B testing for different audio files
- Implement voice activity detection (VAD)
- Add export functionality for custom lessons
- Optimize blendshape rig for web deployment
- Create automated sync calibration tests
- Support for multiple languages in same lesson
- Real-time waveform visualization
- Cloud sync for user lessons


















