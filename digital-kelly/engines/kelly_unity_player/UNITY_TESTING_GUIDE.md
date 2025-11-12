# 🎮 Kelly Avatar Unity Testing Guide

**Complete Step-by-Step Guide to Test, Verify, and Play with Kelly Avatar in Unity**

---

## 📋 Prerequisites Checklist

Before starting, verify:
- ✅ Unity Hub installed
- ✅ Unity 2022.3 LTS or 2023.x installed
- ✅ Unity project at: `digital-kelly/engines/kelly_unity_player`
- ✅ Kelly model files available (FBX)
- ✅ Water-cycle audio files generated (54 files)

---

## 🚀 PART 1: Open Unity Project (5 minutes)

### Step 1.1: Launch Unity Hub

1. **Open Unity Hub** (from Start Menu or desktop)
2. **Click "Projects"** tab (left sidebar)
3. **Click "Add"** button (top right)

### Step 1.2: Add Project to Hub

1. **Navigate to:** `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player`
2. **Select folder** → Click "Select Folder"
3. Unity Hub will show project in list

### Step 1.3: Open Project

1. **Click on project** in Unity Hub list
2. **Unity will prompt** for version (if needed):
   - Select **Unity 2022.3 LTS** or **2023.x**
   - Click **"Open"**
3. **Wait for Unity to load** (5-10 minutes first time)
   - Bottom-right progress bar shows import progress
   - **Don't close Unity** while importing

### Step 1.4: Verify Project Loaded

✅ **Check Console** (bottom window):
- Click **"Console"** tab
- Should show **no red errors**
- Yellow warnings are OK

✅ **Check Project Window** (bottom-left):
- Should see folders: `Assets`, `Packages`, `ProjectSettings`
- Navigate to `Assets/Kelly/Scripts/` - should see scripts

---

## 🎨 PART 2: Set Up Scene with Kelly Avatar (10 minutes)

### Step 2.1: Open or Create Scene

**Option A: Use Existing Scene**
1. In **Project** window, navigate: `Assets/Scenes/`
2. **Double-click** `SampleScene.unity` (or `Main.unity` if exists)
3. Scene opens in Scene view

**Option B: Create New Scene**
1. **File → New Scene**
2. Choose **"Basic (Built-in)"** or **"URP"** template
3. **File → Save As**
4. Name: `KellyTestScene.unity`
5. Save in `Assets/Scenes/`

### Step 2.2: Add Kelly Model to Scene

1. **In Project window**, navigate: `Assets/Kelly/Models/`
2. **Find Kelly FBX file** (e.g., `iclone_unity_kelly_export1.fbx` or `kelly_character_cc.fbx`)
3. **Drag FBX** from Project window → **Hierarchy** window (left side)
4. **Kelly appears** in Scene view

### Step 2.3: Position Kelly in Scene

1. **Select Kelly** in Hierarchy (click on GameObject name)
2. **Inspector** window (right side) shows Transform component
3. **Set Transform values:**
   - **Position:** X: 0, Y: 0, Z: 0
   - **Rotation:** X: 0, Y: 0, Z: 0
   - **Scale:** X: 1, Y: 1, Z: 1
4. **Adjust if needed:**
   - If Kelly appears below ground, increase **Position Y** to 1 or 2
   - Use **Scene view** to verify position

### Step 2.4: Set Up Camera

1. **Select Main Camera** in Hierarchy
2. **Inspector → Transform:**
   - **Position:** X: 0, Y: 1.6, Z: -3
   - **Rotation:** X: 0, Y: 0, Z: 0
3. **Inspector → Camera:**
   - **Field of View:** 38
   - **Clear Flags:** Solid Color
   - **Background:** Black (R:0, G:0, B:0)

### Step 2.5: Add Lighting

1. **Select Directional Light** in Hierarchy (if exists)
2. **Inspector → Light:**
   - **Type:** Directional
   - **Intensity:** 1.0
   - **Shadows:** Soft Shadows
3. **Transform → Rotation:** X: 45, Y: -30, Z: 0

---

## 🔧 PART 3: Configure Kelly Components (15 minutes)

### Step 3.1: Find Kelly's Head Renderer

**Important:** We need to find the SkinnedMeshRenderer component on Kelly's head/body.

1. **Select Kelly** GameObject in Hierarchy
2. **Expand Kelly** in Hierarchy (click ▶ arrow)
3. **Look for child objects** like:
   - `Body`
   - `Head`
   - `Mesh`
   - Or similar names
4. **Click on child object** that has mesh
5. **Inspector** should show **SkinnedMeshRenderer** component
6. **Note this GameObject** - we'll need it next

### Step 3.2: Add BlendshapeDriver60fps Component

1. **Select Kelly** root GameObject in Hierarchy
2. **Inspector** window → **Click "Add Component"** button
3. **Search:** "BlendshapeDriver60fps"
4. **Click** to add component
5. **Configure component:**
   - **Head Renderer:** Drag the SkinnedMeshRenderer GameObject from Step 3.1
   - **Audio Source:** (leave empty for now - will add next)
   - **Target FPS:** 60
   - **Enable Interpolation:** ✅ Checked
   - **Enable Gaze:** ✅ Checked
   - **Enable Micro Expressions:** ✅ Checked
   - **Intensity:** 100
   - **Show Debug Info:** ✅ Checked (for testing)

### Step 3.3: Add Audio Source Component

1. **Still on Kelly GameObject** → Inspector
2. **Add Component** → Search "Audio Source"
3. **Configure Audio Source:**
   - **Play On Awake:** ❌ Unchecked
   - **Volume:** 1.0
   - **Spatial Blend:** 0 (2D sound)
   - **Loop:** ❌ Unchecked
4. **Drag Audio Source** to BlendshapeDriver60fps → **Audio Source** field

### Step 3.4: Add KellyAvatarController Component

1. **Still on Kelly GameObject** → Inspector
2. **Add Component** → Search "KellyAvatarController"
3. **Configure component:**
   - **Blendshape Driver:** Drag BlendshapeDriver60fps component (auto-assigns)
   - **Performance Monitor:** (leave empty - will add next)
   - **Current Learner Age:** 35
   - **Current Kelly Age:** 27 (auto-calculates)

### Step 3.5: Add AvatarPerformanceMonitor Component

1. **Still on Kelly GameObject** → Inspector
2. **Add Component** → Search "AvatarPerformanceMonitor"
3. **Configure component:**
   - **Enable Monitoring:** ✅ Checked
   - **Log To Console:** ✅ Checked
   - **Show On Screen:** ✅ Checked
   - **Update Interval:** 0.5
4. **Drag Performance Monitor** to KellyAvatarController → **Performance Monitor** field

### Step 3.6: Add KellyTalkTest Component (Optional - for quick testing)

1. **Still on Kelly GameObject** → Inspector
2. **Add Component** → Search "KellyTalkTest"
3. **Configure component:**
   - **Auto Play On Start:** ✅ Checked (for testing)
   - **Test Audio Clip:** (leave empty - we'll add audio next)

---

## 🎵 PART 4: Set Up Audio Files (10 minutes)

### Step 4.1: Copy Audio Files to Unity

1. **Open Windows File Explorer**
2. **Navigate to:** `C:\Users\user\UI-TARS-desktop\curious-kellly\backend\config\audio\water-cycle\`
3. **Select all MP3 files** (Ctrl+A)
4. **Copy** (Ctrl+C)

5. **In Unity Project window:**
   - Navigate to: `Assets/Resources/`
   - **Create folder:** `Audio` (if doesn't exist)
   - **Inside Audio**, create folder: `Lessons`
   - **Inside Lessons**, create folder: `water-cycle`
   - **Paste** MP3 files here (Ctrl+V)

6. **Wait for Unity to import** (watch bottom-right progress bar)
   - Unity automatically imports audio files
   - Should take 1-2 minutes

### Step 4.2: Verify Audio Import

1. **Select one MP3 file** in Project window
2. **Inspector** should show **AudioClip** import settings:
   - **Load Type:** Streaming (recommended for large files)
   - **Compression Format:** PCM (highest quality)
3. **If settings look wrong**, select all audio files → Inspector → **Apply** button

### Step 4.3: Create Audio Folder Structure (Alternative Method)

**If Resources folder doesn't exist:**

1. **Project window** → Right-click `Assets`
2. **Create → Folder** → Name: `Resources`
3. **Inside Resources**, create: `Audio/Lessons/water-cycle/`
4. **Copy MP3 files** into `water-cycle` folder

**Why Resources folder?**
- Unity's `Resources.Load()` can access files here
- Required for `LessonAudioPlayer` script

---

## ▶️ PART 5: Test Kelly Avatar (5 minutes)

### Step 5.1: Quick Test Setup

1. **Select Kelly** GameObject in Hierarchy
2. **Inspector → KellyTalkTest component:**
   - **Test Audio Clip:** Drag any audio file from Project (e.g., `18-35-welcome-en.mp3`)
   - **Auto Play On Start:** ✅ Checked

### Step 5.2: Enter Play Mode

1. **Click Play button** ▶️ (top center of Unity Editor)
2. **Unity enters Play Mode** (Play button turns blue ⏸️)
3. **Watch Console** (bottom window):
   - Should see: `[KellyController] Avatar initialized and ready`
   - Should see: `[Kelly60fps] Indexed X blendshapes`
   - Should see: `[Performance] FPS: 60.x`

4. **Watch Scene view:**
   - Kelly should be visible
   - Audio should play automatically (if Auto Play enabled)
   - Top-left corner should show FPS overlay

### Step 5.3: Test Controls (Runtime)

**While in Play Mode:**

- **SPACE** - Play test audio clip
- **T** - Trigger test speech message
- **P** - Pause/Resume audio playback
- **S** - Stop audio playback
- **1-6** - Change learner age (1=2-5, 2=6-12, 3=13-17, 4=18-35, 5=36-60, 6=61-102)

### Step 5.4: Expected Results

✅ **Console Logs:**
```
[KellyController] Avatar initialized and ready
[Kelly60fps] Indexed 45 blendshapes
[Kelly60fps] Started playback at 60fps
[Performance] FPS: 60.2 (min: 58.7, max: 62.1), Frame Time: 16.39ms, Status: Excellent
```

✅ **Visual:**
- Kelly visible in Scene view
- FPS overlay shows ~60fps
- Audio plays (if audio clip assigned)
- No red errors in Console

✅ **Performance:**
- FPS: 60 ± 5% (57-63fps acceptable)
- Frame Time: <16.67ms
- Status: "Excellent" or "Good"

### Step 5.5: Exit Play Mode

1. **Click Play button again** (⏸️ becomes ▶️)
2. **Unity exits Play Mode**
3. Changes made in Play Mode are discarded

---

## 🎯 PART 6: Test Water-Cycle Lesson Audio (10 minutes)

### Step 6.1: Add LessonAudioPlayer Component

1. **Select Kelly** GameObject
2. **Inspector → Add Component** → Search "LessonAudioPlayer"
3. **If script doesn't exist**, skip to Step 6.4 (manual test)

### Step 6.2: Configure LessonAudioPlayer

1. **Inspector → LessonAudioPlayer:**
   - **Lesson Id:** `water-cycle`
   - **Age Group:** `18-35`
   - **Language:** `en`
   - **Auto Play On Load:** ✅ Checked

### Step 6.3: Test Lesson Playback

1. **Press Play** ▶️
2. **Audio should play** automatically
3. **Console should show:**
   ```
   [LessonAudioPlayer] Loading lesson: water-cycle
   [LessonAudioPlayer] Playing: 18-35-welcome-en.mp3
   ```

### Step 6.4: Manual Audio Test (Alternative)

**If LessonAudioPlayer doesn't exist:**

1. **Select Kelly** GameObject
2. **Inspector → Audio Source component:**
   - **Audio Clip:** Drag `18-35-welcome-en.mp3` from Project
3. **Press Play** ▶️
4. **In Play Mode:**
   - **Select Kelly** in Hierarchy
   - **Inspector → Audio Source**
   - **Click Play button** on Audio Source component
   - **Audio plays** and Kelly's mouth should move (if blendshapes configured)

### Step 6.5: Test All Languages

**Test EN, ES, FR audio:**

1. **Change Audio Clip** in Audio Source:
   - `18-35-welcome-en.mp3` (English)
   - `18-35-welcome-es.mp3` (Spanish)
   - `18-35-welcome-fr.mp3` (French)
2. **Press Play** after each change
3. **Verify** audio plays correctly for each language

### Step 6.6: Test All Age Variants

**Test different age groups:**

1. **Select Kelly** GameObject
2. **Inspector → KellyAvatarController:**
   - **Current Learner Age:** Change to different values:
     - `3` → Kelly becomes age 3 (toddler)
     - `12` → Kelly becomes age 9 (kid)
     - `17` → Kelly becomes age 15 (teen)
     - `35` → Kelly becomes age 27 (adult) ← Default
     - `50` → Kelly becomes age 48 (mentor)
     - `80` → Kelly becomes age 82 (elder)
3. **Change Audio Clip** to match age:
   - `2-5-welcome-en.mp3` for age 3
   - `6-12-welcome-en.mp3` for age 9
   - etc.
4. **Press Play** and verify age-appropriate audio

---

## 📊 PART 7: Verify Performance (5 minutes)

### Step 7.1: Check FPS Overlay

**In Play Mode:**

1. **Look at top-left corner** of Game view
2. **Should see FPS overlay:**
   ```
   FPS: 60.2
   Frame Time: 16.39ms
   Status: Excellent
   ```

### Step 7.2: Check Performance Monitor

1. **Select Kelly** GameObject
2. **Inspector → AvatarPerformanceMonitor:**
   - **Current FPS:** Should show ~60
   - **Average FPS:** Should show ~60
   - **Min FPS:** Should be >55
   - **Max FPS:** Should be <65
   - **Status:** Should be "Excellent" or "Good"

### Step 7.3: Check Console Logs

**Console should show:**
```
[Performance] FPS: 60.2 (min: 58.7, max: 62.1)
[Performance] Frame Time: 16.39ms
[Performance] Status: Excellent
```

### Step 7.4: Verify No Errors

1. **Check Console** for red errors
2. **Should see zero red errors**
3. **Yellow warnings** are OK (usually just missing references)

---

## 🐛 Troubleshooting

### Problem: "BlendshapeDriver60fps script not found"

**Solution:**
1. Check script exists: `Assets/Kelly/Scripts/BlendshapeDriver60fps.cs`
2. If missing, verify Unity project is correct: `digital-kelly/engines/kelly_unity_player`
3. Restart Unity if needed

### Problem: "Head Renderer is null"

**Solution:**
1. Expand Kelly GameObject in Hierarchy
2. Find child object with SkinnedMeshRenderer
3. Drag that GameObject to BlendshapeDriver60fps → Head Renderer field

### Problem: "Audio doesn't play"

**Solution:**
1. Check Audio Source volume > 0
2. Verify audio clip is assigned
3. Check system audio is working
4. Look for errors in Console

### Problem: "FPS shows 30 instead of 60"

**Solution:**
1. **Edit → Project Settings → Quality**
   - **VSync Count:** Don't Sync
2. **Edit → Project Settings → Time**
   - **Fixed Timestep:** 0.01666667 (60fps)
3. **Or add to script:**
   ```csharp
   Application.targetFrameRate = 60;
   QualitySettings.vSyncCount = 0;
   ```

### Problem: "Kelly's mouth doesn't move"

**Solution:**
1. Verify Head Renderer is assigned
2. Check Kelly model has blendshapes
3. Ensure A2F JSON data is loaded (if using pre-rendered animation)
4. Check Console for blendshape errors

### Problem: "Audio files not found"

**Solution:**
1. Ensure files are in `Assets/Resources/Audio/Lessons/water-cycle/`
2. Verify files are imported as AudioClip assets
3. Check file naming matches: `{ageGroup}-{section}-{language}.mp3`

---

## ✅ Success Checklist

Before considering testing complete, verify:

- ✅ Unity project opens without errors
- ✅ Kelly model visible in scene
- ✅ All required components added
- ✅ Head Renderer assigned to BlendshapeDriver60fps
- ✅ Audio Source configured and working
- ✅ Water-cycle audio files imported
- ✅ Play Mode shows 60fps consistently
- ✅ Audio plays correctly
- ✅ Console shows expected logs
- ✅ No red errors in Console
- ✅ Performance status is "Excellent" or "Good"

---

## 🎉 Next Steps

**Once basic testing works:**

1. **Add Viseme Updates**
   - Connect OpenAI Realtime API viseme streaming
   - Test lip-sync accuracy with real-time voice

2. **Test Age Morphing**
   - Verify Kelly's appearance changes with age
   - Test all 6 age variants (3, 9, 15, 27, 48, 82)

3. **Test Gaze Tracking**
   - Move gaze target in Play Mode
   - Verify Kelly's eyes follow target

4. **Integrate with Flutter**
   - Build Unity for iOS/Android
   - Test bidirectional communication
   - Test full lesson playback flow

---

## 📁 Quick Reference

**Unity Project:** `digital-kelly/engines/kelly_unity_player`  
**Scene:** `Assets/Scenes/SampleScene.unity`  
**Scripts:** `Assets/Kelly/Scripts/`  
**Audio:** `Assets/Resources/Audio/Lessons/water-cycle/`  
**Kelly Model:** `Assets/Kelly/Models/`  

**Key Scripts:**
- `BlendshapeDriver60fps.cs` - Animation driver
- `KellyAvatarController.cs` - Main controller
- `AvatarPerformanceMonitor.cs` - Performance tracking

---

**Ready to test!** Follow steps in order, and Kelly should be talking at 60fps! 🎮🎉







