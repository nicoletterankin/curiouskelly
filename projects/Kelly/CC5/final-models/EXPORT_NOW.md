# EXPORT KELLY FROM ICLONE — DO THIS EXACTLY

**One shot. Perfect export. No second run.**

---

## BEFORE YOU EXPORT

### Verify These Are Done:
- [ ] Audio is loaded in timeline (39 seconds)
- [ ] Viseme track shows lip-sync data
- [ ] Face Key panel shows "CC4 Extended"
- [ ] Kelly is selected in Scene panel

### Set Timeline Range:
1. Look at the timeline at the bottom
2. Note where the audio ENDS (around frame 975 at 25fps for 39 seconds)
3. Set your timeline END marker to match

---

## EXPORT STEPS

### Step 1: Open Export Dialog
```
File → Export → FBX...
```

### Step 2: Export Settings (CRITICAL)

**FBX Options Panel — Set EXACTLY:**

| Setting | Value | Why |
|---------|-------|-----|
| **Target Tool Preset** | Unity 3D | Correct bone orientation |
| **FBX Format** | Binary | Smaller file, faster load |

**Export Range:**
| Setting | Value | Why |
|---------|-------|-----|
| **Frame Range** | Range | NOT "All" — we want the animated portion |
| **Start Frame** | 0 | Beginning |
| **End Frame** | 1799 | Where audio ends |

**Include:**
| Setting | Value | Why |
|---------|-------|-----|
| ✅ **Mesh** | ON | Need the model |
| ✅ **Motion** | ON | CRITICAL — includes animation |
| ✅ **Embed Texture** | ON | All textures inside FBX |
| ❌ **Delete Hidden Faces** | OFF | Keep all geometry |

**Mesh:**
| Setting | Value | Why |
|---------|-------|-----|
| ✅ **Merge Material Subdivisions** | ON | Cleaner materials |
| ❌ **Merge Mesh as One** | OFF | Keep blendshapes separate |

**Bone:**
| Setting | Value | Why |
|---------|-------|-----|
| ✅ **Human IK** | ON | Unity humanoid rig |
| ❌ **Remove Unimportant Bones** | OFF | Keep all bones |

**Motion:**
| Setting | Value | Why |
|---------|-------|-----|
| ✅ **Blend Shape** | ON | CRITICAL — lip sync + expressions |
| ✅ **Body Motion** | ON | Any body animation |
| ✅ **Facial (Head & Eye)** | ON | Head movement |
| ❌ **Smooth Rotation** | OFF | Can cause issues |

### Step 3: File Name and Location
```
File Name: kelly_intro_full.fbx
Location: c:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\
```

### Step 4: Click Export

Wait for it to complete. May take 1-2 minutes.

---

## VERIFY EXPORT SUCCESS

Check the output folder for:
- [ ] `kelly_intro_full.fbx` (should be 50-200 MB)
- [ ] If textures are separate: `.fbm` folder with textures

---

## ALSO EXPORT: Audio File

Copy your audio to the same location:
```
From: c:\Users\user\UI-TARS-desktop\projects\Kelly\ElevenLabs_2025-12-16T21_35_29_kelly2 voice_ivc_sp98_s50_sb94_se0_b_m2.mp3

To: c:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\kelly_intro_audio.mp3
```

Or just rename it to `kelly_intro_audio.mp3` for clarity.

---

## WHAT YOU'LL HAVE AFTER EXPORT

| File | Contains |
|------|----------|
| `kelly_intro_full.fbx` | Kelly model + skeleton + ALL blendshapes + baked lip-sync animation + any expressions |
| `kelly_intro_audio.mp3` | The voice audio |

---

## NEXT: UNITY IMPORT

Once exported, the FBX goes to Unity:
1. Drag into Unity Assets folder
2. Use CCIC Auto Setup (High Quality URP)
3. Add KellyAnimationPlayer component
4. Set audio source
5. Build WebGL

---

## IF SOMETHING GOES WRONG

| Problem | Fix |
|---------|-----|
| No lip movement in Unity | Re-export with "Blend Shape" ON |
| Textures missing | Re-export with "Embed Texture" ON |
| Animation missing | Re-export with "Motion" ON and "Range" not "Current Frame" |
| Wrong scale | Target was wrong — use "Unity 3D" preset |

---

**This is it. Export once. Export perfect.**
