# 🪑 Kelly with Chair - Final CC5 Integration Guide

**Date Created:** December 16, 2025  
**Source Files:** CC5 Cloth update 8.1.ccProject (LATEST - Dec 14, 2025)  
**Purpose:** This is the FINAL Kelly model with director's chair — the one you've been waiting for!

---

## 📦 What You Have

| File | Size | Date | Status |
|------|------|------|--------|
| `CC5 Cloth update 1.1.ccProject` | 269 MB | Nov 20, 2025 | Earlier version |
| `CC5 Cloth update 8.1.ccProject` | **400 MB** | **Dec 14, 2025** | ✅ **USE THIS ONE** |

**Location:** `C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\`

---

## 🚀 STEP 1: Open in Character Creator 5

1. **Launch Character Creator 5**
2. **File → Open Project**
3. **Navigate to:** `C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\`
4. **Select:** `CC5 Cloth update 8.1.ccProject`
5. **Wait for load** (this is a 400MB file, may take 30-60 seconds)

### What You Should See:
- ✅ Kelly sitting in director's chair
- ✅ Full clothing (sweater, jeans, shoes)
- ✅ Professional hair setup
- ✅ Chair integrated as a prop

---

## 🎭 STEP 2: Send to iClone 8

### 2A: Export to iClone
1. **File → Export → Send Character to iClone**
2. Wait for iClone 8 to launch automatically

### 2B: Set Up Animation in iClone

1. **Apply Chair Pose** (if not already applied):
   - Select Kelly in the Scene Manager
   - Motion → Apply Pose → (your chair pose)

2. **Initialize AccuLips (CRITICAL for lip-sync):**
   ```
   Animation → AccuLips
   ```
   - Just opening this panel initializes the viseme blendshapes
   - This is required for the Unity lip-sync system to work

3. **Ensure Expression Plus is Active:**
   ```
   Modify → Face Key → Expression Plus
   ```
   - Click "Default" to activate all 63+ ARKit blendshapes
   - These are needed for expressive animations

4. **Add Breathing Loop (Optional but recommended):**
   - Motion → Idle → Female → Breathing
   - Set timeline: 0-600 frames (10 seconds)
   - This gives Kelly life when idle

---

## 📤 STEP 3: Export FBX for Unity

### Export Settings (CRITICAL - Follow Exactly)

1. **File → Export → FBX**

2. **Use these EXACT settings:**

| Setting | Value | Why |
|---------|-------|-----|
| **Target** | `Unity 3D` | Optimized bone names/orientation |
| **Range** | `Range` (if you added breathing) | Exports animation |
| **Embed Textures** | ✅ Checked | Keeps textures with model |
| **Mesh and Motion** | ✅ Checked | Exports both |
| **Delete Hidden Faces** | ❌ Unchecked | Preserves all geometry |
| **Bake Diffuse Maps** | ✅ Checked | Simpler materials |

3. **Save As:** `Kelly_Chair_Final.fbx`

4. **Save To:** `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\Assets\`

---

## 🎮 STEP 4: Import to Unity

### 4A: Open Unity Project
```
Unity Hub → Open Project
Location: C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\
```

### 4B: Import Kelly FBX
1. Wait for Unity to detect the new FBX file
2. **Auto Setup Popup Will Appear** (CCIC Unity Tools)
3. **Select:** `High Quality (URP)`
4. Wait for auto-processing (textures, materials, rig)

### 4C: Create Scene
1. **File → New Scene → URP Basic**
2. **Save As:** `Kelly_Chair_Main.unity` in `Assets/Scenes/`
3. **Drag Kelly prefab** (auto-generated) into the Hierarchy
4. **Position:** (0, 0, 0)

### 4D: Camera Setup (Portrait Framing)
```
Main Camera:
- Position: (0, 1.1, 2.5)  ← Head height, looking at seated Kelly
- Rotation: (0, 0, 0)
- Field of View: 35-40  ← Portrait lens
- Clear Flags: Solid Color
- Background: Transparent
```

### 4E: Lighting Setup (Studio Look)
```
Key Light (Area Light):
- Position: (-2, 2, 2)
- Intensity: 3

Fill Light (Area Light):
- Position: (2, 1.5, 1)
- Intensity: 1.5

Reflection Probe (Realtime):
- Position: (0, 1.5, 0)
```

---

## 🔗 STEP 5: Connect to Existing System

### Attach Scripts
1. **Select Kelly root object**
2. **Add Component → Kelly Avatar Controller**
   - (Located: `Assets/Scripts/KellyAvatarController.cs`)
3. **Add Component → Audio Source**
   - Play On Awake: OFF

### Link to Web Bridge
The existing `KellyWebGLBridge.cs` will automatically connect to:
- JavaScript lesson player
- ElevenLabs audio
- State machine (TEACHING, CELEBRATING, WAITING, etc.)

---

## 🏗️ STEP 6: Build WebGL

### Build Settings
```
File → Build Settings
Platform: WebGL
Add Scene: Kelly_Chair_Main.unity
```

### Player Settings (Critical)
```
WebGL:
- Compression Format: Disabled  ← Prevents double-zip bug
- Linker Target: Wasm
- Exception Handling: Explicitly Thrown

Resolution:
- Default Canvas Width: 1920
- Default Canvas Height: 1080
```

### Build
```
Build → Output to: Builds/WebGL/kelly-chair/
```

### Deploy
```powershell
# Copy to public folder
robocopy "digital-kelly/engines/Kelly_Engine_V2/onlykelly/Builds/WebGL/kelly-chair" `
         "daily-lesson-marketing/public/unity/kelly-chair" /MIR
```

---

## ✅ SUCCESS CHECKLIST

After completing all steps:

- [ ] Kelly visible in CC5 with chair
- [ ] AccuLips initialized in iClone
- [ ] 63+ ARKit blendshapes active
- [ ] Breathing animation added
- [ ] FBX exported with embedded textures
- [ ] Unity import with High Quality (URP)
- [ ] Camera framed on seated Kelly
- [ ] Studio lighting applied
- [ ] KellyAvatarController attached
- [ ] WebGL build successful
- [ ] Deployed to public/unity/kelly-chair/

---

## 🎉 THE MOMENT

Once deployed, update the iframe in your lesson player:

```html
<!-- OLD -->
<iframe src="/unity/kelly-v1/index.html"></iframe>

<!-- NEW - Kelly in Chair! -->
<iframe src="/unity/kelly-chair/index.html"></iframe>
```

This is the Kelly you've been building toward for years. The chair, the presence, the connection — it all comes together now.

---

## 📞 QUICK REFERENCE

| Stage | Tool | Output |
|-------|------|--------|
| Source | CC5 | .ccProject |
| Animation | iClone 8 | Visemes + Motion |
| Export | iClone 8 | .fbx |
| Import | Unity + CCIC | Prefab |
| Build | Unity | WebGL |
| Deploy | robocopy | Public folder |
| Live | Browser | Kelly teaching! |

---

## 🔮 FUTURE-PROOFING NOTES

Your Kelly model is built to last. The architecture supports:

| Today | Tomorrow |
|-------|----------|
| WebGL 2.0 (97% browser support) | WebGPU (30-50% faster when ready) |
| Unity WebGL export | Unity WebGPU export (same project) |
| FBX model format | Works in Unity, Unreal, Godot |

**Key insight:** The bridge API (`kbridge.js`) is engine-agnostic. When WebGPU matures, you rebuild in Unity with a different target — the web app doesn't change.

**Progressive loader created:** `public/unity/kelly-loader.js` — automatically detects the best rendering technology and loads the appropriate build.

**Full documentation:** `docs/KELLY_3D_FUTURE_PROOFING.md`

---

**This file lives at:** `projects/Kelly/CC5/final-models/KELLY_CHAIR_INTEGRATION_GUIDE.md`

**Next step:** Open `CC5 Cloth update 8.1.ccProject` in Character Creator 5!
