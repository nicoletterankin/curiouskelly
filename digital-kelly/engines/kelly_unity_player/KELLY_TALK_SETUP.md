# Getting Kelly Talking in Unity - Quick Setup Guide

## ✅ Step 1: Fix Compilation Error (COMPLETED)

The duplicate `[Serializable]` attribute error has been fixed by removing the duplicate `A2FData` class definition from `BlendshapeDriver60fps.cs`. The class is now only defined in `A2FModels.cs`.

**Status:** ✅ Fixed - Unity should compile without errors now.

---

## 🎯 Step 2: Add Kelly Avatar to Scene (5 minutes)

### 2.1. Open Unity Scene
1. Open Unity project: `digital-kelly/engines/kelly_unity_player`
2. Open scene: `Assets/Scenes/SampleScene.unity` (or create a new scene)

### 2.2. Add Kelly Model to Scene
1. In **Project** window, navigate to: `Assets/Kelly/Models/`
2. Find the Kelly model file (e.g., `iclone_unity_kelly_export1.fbx` or `kelly_character_cc.fbx`)
3. **Drag** the model into the **Hierarchy** window
4. The Kelly avatar should appear in the Scene view

### 2.3. Position Kelly
1. Select Kelly in **Hierarchy**
2. In **Inspector**, set **Transform**:
   - **Position:** (0, 0, 0)
   - **Rotation:** (0, 0, 0)
   - **Scale:** (1, 1, 1)
3. Adjust **Position Y** if Kelly appears below ground level

---

## 🔧 Step 3: Configure Kelly Components (10 minutes)

### 3.1. Add Required Scripts
1. Select **Kelly** GameObject in Hierarchy
2. In **Inspector**, click **Add Component**
3. Add these components (in order):
   - ✅ **BlendshapeDriver60fps**
   - ✅ **KellyAvatarController**
   - ✅ **AvatarPerformanceMonitor**
   - ✅ **LessonAudioPlayer** (optional, for lesson audio)
   - ✅ **Audio Source** (Unity component)
   - ✅ **KellyTalkTest** (our test script)

### 3.2. Configure BlendshapeDriver60fps
1. Select Kelly in Hierarchy
2. In **Inspector**, find **BlendshapeDriver60fps** component:
   - **Head Renderer:** Drag the SkinnedMeshRenderer from Kelly's head/body mesh
   - **Audio Source:** Drag the AudioSource component (auto-assigned if same GameObject)
   - **Target FPS:** 60
   - **Enable Interpolation:** ✅ Checked
   - **Enable Gaze:** ✅ Checked (optional)
   - **Enable Micro Expressions:** ✅ Checked (optional)

**How to find Head Renderer:**
- Expand Kelly GameObject in Hierarchy
- Find child object with SkinnedMeshRenderer (usually named "Body" or "Head")
- Drag that GameObject to the **Head Renderer** field

### 3.3. Configure KellyAvatarController
1. In **Inspector**, find **KellyAvatarController** component:
   - **Blendshape Driver:** Drag the BlendshapeDriver60fps component
   - **Performance Monitor:** Drag the AvatarPerformanceMonitor component
   - **Current Learner Age:** 35 (default adult)
   - **Current Kelly Age:** 27 (will auto-calculate)

### 3.4. Configure Audio Source
1. In **Inspector**, find **Audio Source** component:
   - **Play On Awake:** ❌ Unchecked
   - **Volume:** 1.0
   - **Spatial Blend:** 0 (2D sound)
   - **Loop:** ❌ Unchecked

---

## 🎵 Step 4: Set Up Audio (5 minutes)

### Option A: Use Test Audio Clip (Quick Test)

1. Select **Kelly** in Hierarchy
2. In **Inspector**, find **KellyTalkTest** component:
   - **Test Audio Clip:** Drag any AudioClip from Project (or create one)
   - **Auto Play On Start:** ✅ Checked (for testing)

### Option B: Use Lesson Audio (Full Setup)

1. Copy audio files to Unity:
   ```
   Create folder: Assets/Resources/Audio/Lessons/water-cycle/
   Copy MP3 files from: curious-kellly/backend/config/audio/water-cycle/
   ```

2. Configure **LessonAudioPlayer**:
   - **Lesson Id:** water-cycle
   - **Age Group:** 18-35
   - **Auto Play On Load:** ✅ (for testing)

3. Audio files should be named:
   - `18-35-welcome.mp3`
   - `18-35-mainContent.mp3`
   - `18-35-wisdomMoment.mp3`

---

## ▶️ Step 5: Test Kelly Talking (2 minutes)

### Quick Test with KellyTalkTest Script

1. **Press Play** ▶️ in Unity
2. Press **SPACE** to play test audio
3. Kelly should:
   - ✅ Play audio
   - ✅ Move mouth/blendshapes (if A2F data is loaded)
   - ✅ Show debug info in top-left corner

### Test Controls (Runtime)
- **SPACE** - Play test audio
- **T** - Trigger test speech message
- **P** - Pause/Resume audio
- **S** - Stop audio

### Expected Results
- ✅ Audio plays clearly
- ✅ Kelly's mouth moves (if blendshapes configured)
- ✅ Console shows debug logs:
  ```
  [KellyTalkTest] Playing audio: [clip name]
  [Kelly60fps] Started playback at 60fps
  ```

---

## 🐛 Troubleshooting

### Issue: "No audio plays"
**Solutions:**
- Check AudioSource volume > 0
- Verify audio clip is assigned
- Check system audio is working
- Look for errors in Console

### Issue: "Kelly's mouth doesn't move"
**Solutions:**
- Verify Head Renderer is assigned in BlendshapeDriver60fps
- Check that Kelly model has blendshapes
- Ensure A2F JSON data is loaded (if using pre-rendered animation)
- Check Console for blendshape errors

### Issue: "Audio clips are null"
**Solutions:**
- Ensure audio files are in `Assets/Resources/Audio/Lessons/` folder
- Check file naming matches expected format: `{ageGroup}-{section}`
- Verify files are imported as AudioClip assets

### Issue: "Compilation errors"
**Solutions:**
- Check Console for specific errors
- Ensure all scripts are in `Assets/Kelly/Scripts/` folder
- Verify Unity has finished importing assets

---

## 📋 Quick Checklist

Before testing, verify:
- ✅ Kelly model is in scene
- ✅ All required components are added
- ✅ Head Renderer is assigned to BlendshapeDriver60fps
- ✅ Audio Source is configured
- ✅ Test audio clip is assigned (or lesson audio files are in Resources)
- ✅ Unity compiles without errors

---

## 🚀 Next Steps

Once basic audio playback works:

1. **Add A2F Animation Data**
   - Load A2F JSON file into BlendshapeDriver60fps's `a2fJsonAsset` field
   - This enables lip-sync animation

2. **Connect to Flutter**
   - Ensure UnityMessageManager is set up
   - Test message passing from Flutter app

3. **Test All Age Variants**
   - Change learner age in KellyAvatarController
   - Verify Kelly's appearance and voice change

4. **Add Viseme Updates**
   - Connect realtime viseme updates from OpenAI Realtime API
   - Test lip-sync accuracy

---

## 📁 Files Created/Modified

- ✅ `BlendshapeDriver60fps.cs` - Removed duplicate A2FData class
- ✅ `KellyTalkTest.cs` - New test script for quick audio testing
- ✅ `KELLY_TALK_SETUP.md` - This guide

---

**Ready to test!** Press Play in Unity and hit SPACE to hear Kelly talk! 🎉










