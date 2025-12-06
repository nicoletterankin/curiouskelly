# Unity Testing Handoff: Kelly Lip-Sync Integration

**Date:** November 17, 2025  
**Status:** Audio playback working ✓ | Lip-sync in progress ⏳  
**Current Task:** Generate and test lip-sync for water-cycle lesson

---

## ✅ What's Working Now

1. **Unity Scene Setup**
   - Kelly character model (`kelly_character`) in scene at (0,0,0)
   - Audio playing correctly through AudioSource component
   - Test audio: `6-12-welcome-en.mp3` (water-cycle lesson)

2. **Audio Files Ready**
   - Location: `curious-kellly/backend/config/audio/water-cycle/`
   - 72 files total: 6 age groups × 3 phases × 4 languages
   - All imported into Unity Resources folder

3. **Scripts in Place**
   - `BlendshapeDriver.cs` - Drives facial blendshapes from A2F data
   - `SimpleKellyAudioTest.cs` - Basic audio playback test
   - Kelly has blendshapes ready for lip-sync

---

## ⚠️ Known Issues

1. **Wrong Voice ID**
   - Water-cycle audio uses incorrect ElevenLabs voice
   - Should use: `wAdymQH5YucAkXwmrdL0` (Kelly's trained voice)
   - Need to regenerate all water-cycle audio files

2. **No Lip-Sync Data Yet**
   - Audio plays but Kelly's mouth doesn't move
   - Need Audio2Face (A2F) blendshape data files
   - Two methods available: iClone AccuLips OR NVIDIA Audio2Face-3D

---

## 🎯 Current Tasks

### **Task 1: iClone Lip-Sync Test (USER ACTION REQUIRED)**

**Steps for User:**

1. **Locate Kelly FBX in Unity:**
   - Project panel: `Assets > Kelly > Models`
   - Right-click `kelly_character` → Show in Explorer
   - Note the file path

2. **Launch iClone 8**

3. **Import Kelly:**
   - `File > Import FBX`
   - Select the `kelly_character.fbx` file
   - Import with default settings

4. **Import Audio:**
   - Right-click timeline audio track
   - Import Audio
   - Select: `C:\Users\user\UI-TARS-desktop\curious-kellly\backend\config\audio\water-cycle\6-12-welcome-en.mp3`

5. **Apply AccuLips:**
   - Select Kelly character in scene
   - Menu: `Animation > Facial Animation > AccuLips`
   - Settings:
     - Audio: Select imported track
     - Language: English
     - Quality: High
   - Click Apply (wait 1-3 min)

6. **Test Playback:**
   - Press Spacebar to play
   - Verify lips sync with audio

7. **Export Animation:**
   - Select Kelly
   - `File > Export FBX`
   - Check ✓ Export Animation
   - Save as: `kelly_water_cycle_welcome.fbx`
   - Save location: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\Assets\Kelly\Animations\`

**Expected Result:**
- FBX file with baked facial animation
- Import back to Unity and test

---

### **Task 2: Audio2Face-3D Pipeline Setup (AUTOMATED)**

**Prerequisites:**

1. **NVIDIA API Key**
   - Get from: https://build.nvidia.com/
   - Sign up for NVIDIA AI Foundation account
   - Generate API key from dashboard

2. **Audio2Face Function ID**
   - Available after API key setup
   - Check NVIDIA dashboard for Function ID

3. **Install ffmpeg** (for audio conversion):
   - Download: https://ffmpeg.org/download.html
   - Add to Windows PATH

**Setup Environment Variables:**

```powershell
# In PowerShell (run once)
$env:NVIDIA_API_KEY = "your_nvidia_api_key_here"
$env:AUDIO2FACE_FUNCTION_ID = "your_function_id_here"

# Make permanent (optional):
[System.Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", "your_key", "User")
[System.Environment]::SetEnvironmentVariable("AUDIO2FACE_FUNCTION_ID", "your_id", "User")
```

**Install Python Dependencies:**

```powershell
cd C:\Users\user\UI-TARS-desktop

# Install Audio2Face client requirements
pip install -r Audio2Face-3D-Samples/scripts/audio2face_3d_api_client/requirements

# Install NVIDIA ACE wheel
pip install Audio2Face-3D-Samples/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

**Run Test (Single File):**

```powershell
cd C:\Users\user\UI-TARS-desktop
scripts\generate_lipsync_for_watercycle.ps1 -TestOnly
```

**Run Full Batch (All 72 Files):**

```powershell
cd C:\Users\user\UI-TARS-desktop
scripts\generate_lipsync_for_watercycle.ps1
```

**Expected Output:**
- `lessons/audio/water-cycle/[filename]/[filename].a2f.json` for each audio file
- Blendshape animation data ready for Unity

---

## 🔄 Unity Integration Workflow

Once you have A2F data files:

1. **Copy A2F JSON files to Unity:**
   ```
   Source: lessons/audio/water-cycle/[filename]/[filename].a2f.json
   Destination: digital-kelly/engines/kelly_unity_player/Assets/Resources/Audio/Lessons/water-cycle/
   ```

2. **Assign to BlendshapeDriver:**
   - Select `kelly_character` in Hierarchy
   - Add component: `BlendshapeDriver`
   - Assign:
     - Head Renderer: Kelly's SkinnedMeshRenderer
     - Audio Source: Kelly's AudioSource
     - A2F Json Asset: The imported JSON file

3. **Test Playback:**
   - Click Play in Unity
   - Kelly should now lip-sync perfectly!

---

## 📋 Testing Checklist

### Scene Validation
- [ ] Kelly model visible in scene
- [ ] Kelly positioned at (0, 0, 0)
- [ ] Camera positioned to see Kelly's face
- [ ] Lighting adequate to see facial features

### Audio Validation
- [ ] Audio files imported to Unity Resources
- [ ] AudioSource component on Kelly
- [ ] Audio plays on scene start
- [ ] Volume audible (not too loud/quiet)

### Lip-Sync Validation
- [ ] A2F JSON data generated
- [ ] BlendshapeDriver component attached
- [ ] Blendshapes mapped correctly
- [ ] Lips move in sync with audio
- [ ] Mouth shapes match phonemes
- [ ] No lag or timing issues

### Performance Testing
- [ ] Scene runs at 60 FPS
- [ ] No dropped frames during playback
- [ ] Memory usage stable
- [ ] No console errors

---

## 🐛 Troubleshooting

### Kelly Not Visible
- Check Transform position is (0,0,0)
- Adjust camera to face Kelly
- Check Model import settings (scale)

### No Audio Playing
- Verify AudioSource has clip assigned
- Check "Play On Awake" is enabled
- Verify audio file imported correctly
- Check Unity audio settings (not muted)

### Lips Not Moving
- Verify A2F JSON file assigned
- Check BlendshapeDriver has headRenderer
- Verify blendshape count > 0 (check Console logs)
- Check audio and A2F data have same duration

### Audio2Face API Errors
- Verify API key is correct
- Check Function ID matches
- Ensure WAV format (not MP3)
- Check audio is 16-bit PCM, mono

---

## 📝 Next Steps After Testing

1. **Fix Voice ID Issue:**
   - Regenerate water-cycle audio with Kelly's voice
   - Voice ID: `wAdymQH5YucAkXwmrdL0`
   - Use ElevenLabs API or existing generation scripts

2. **Generate A2F for All Lessons:**
   - Batch process all 7 complete lessons
   - Store A2F data alongside audio files
   - Create manifest system for audio + A2F pairs

3. **WebGL Build with Lip-Sync:**
   - Build Unity project for WebGL
   - Deploy to `public/unity/kelly-v1/`
   - Test in lesson-player iframe integration

4. **Integrate with Lesson Player:**
   - Wire Unity iframe to lesson player
   - Pass lesson ID and phase to Unity
   - Unity loads correct audio + A2F data
   - Synchronized playback with lesson UI

---

## 📞 Support & References

**Key Files:**
- Unity Scene: `digital-kelly/engines/kelly_unity_player/Assets/Scenes/KellyTest.unity`
- Lip-Sync Script: `digital-kelly/engines/kelly_unity_player/Assets/Kelly/Scripts/BlendshapeDriver.cs`
- Audio Files: `curious-kellly/backend/config/audio/water-cycle/`
- A2F Pipeline: `scripts/generate_lipsync_for_watercycle.ps1`

**Documentation:**
- `docs/guides/KELLY_TALKING_TODAY.md` - iClone workflow
- `kelly_audio2face/KELLY_WORKFLOW_GUIDE.md` - Audio2Face-3D setup
- `digital-kelly/engines/kelly_unity_player/WEBGL_EMBED_GUIDE.md` - Unity WebGL deployment

**Voice Generation:**
- Kelly Voice ID: `wAdymQH5YucAkXwmrdL0`
- Generation Script: `synthetic_tts/generate_kelly_lipsync.py`
- ElevenLabs API: https://api.elevenlabs.io/docs

---

**Status Update:** Waiting for user to complete iClone AccuLips test, then will proceed with Audio2Face-3D automation setup.





