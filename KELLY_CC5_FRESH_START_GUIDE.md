# 🎯 Complete Kelly Avatar Setup: CC5 → Unity (Fresh Start)

**Goal:** Create Kelly avatar in Character Creator 5 with proper lip-sync blendshapes, then import to Unity for TTS integration.

**Status:** ✅ ElevenLabs TTS working → Now need proper avatar setup

---

## 🔍 What Went Wrong (Current Issues)

1. ❌ **"Ball in throat" moving with audio** → Wrong blendshape assigned to audio
2. ❌ **Lips don't move** → Blendshapes not properly configured
3. ✅ **TTS working** → ElevenLabs integration successful

**Solution:** Start fresh with proper CC5 → Unity workflow

---

## 📋 Pre-Flight Checklist

Before starting, verify you have:

- [ ] **Character Creator 5** installed and launched
- [ ] **Unity project** open (`digital-kelly/engines/kelly_unity_player`)
- [ ] **Backend running** (`http://localhost:3000`) - Already done ✅
- [ ] **Kelly headshot photo** (if using Headshot 2) - Optional but recommended

---

## 🎨 PHASE 1: Create Kelly in Character Creator 5

### **Step 1.1: Launch CC5 & Create New Project**

1. **Open Character Creator 5**
   - Windows Start Menu → Search "Character Creator 5"
   - Click to launch
   - Wait for full load (30-60 seconds)

2. **Create New Project**
   - Top menu: **File → New Project**
   - Name: `Kelly_Unity_Production`
   - Save location: `C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\`
   - Click **OK**

---

### **Step 1.2: Create Base Character**

**Option A: Start from Base Character (Fastest)**

1. **In Content Panel (left side)**, expand **"Actor"** folder
2. **Click** **"CC3+ Character"** folder
3. **Select** **"CC3_Base_Plus"** character
4. **Click** **"Apply"** button at bottom
5. Character appears in viewport

**Option B: Use Headshot 2 (Most Realistic)**

1. **Click** **"Headshot 2"** tab (top toolbar)
2. **Click** **"Load Photo"** button
3. **Navigate** to your Kelly headshot: `projects/Kelly/Ref/` (if you have one)
4. **Settings:**
   - Resolution: **Ultra High**
   - Mesh Density: **Maximum**
   - Detail Level: **Maximum**
5. **Click** **"Generate"** (wait ~10 minutes)
6. **Click** **"Apply to Character"**
7. **Click** **"Accept"**

---

### **Step 1.3: Configure Character for Unity**

1. **Click** **"Modify"** tab (top toolbar)

2. **Set SubD Levels (CRITICAL for Unity):**
   - Find **"SubD Levels"** section in Modify panel
   - **Viewport:** Set to **4** (maximum)
   - **Render:** Set to **4** (maximum)
   - **Click** **"Subdivide"**
   - Wait 5-10 minutes for processing

3. **Verify Facial Profile:**
   - **Modify → Facial Profile**
   - Ensure **"Facial Profile"** is **enabled** ✓
   - This ensures all 53 blendshapes are available

---

### **Step 1.4: Add Hair (Optional)**

1. **Modify → Hair** tab
2. **Browse** Hair HD library
3. **Choose** long wavy dark brown hair
4. **Apply** to character
5. **(Optional)** Apply hair physics preset if you have `Kelly_Hair_Physics.json`

---

### **Step 1.5: Verify Blendshapes**

**CRITICAL STEP - This ensures lip-sync will work:**

1. **Modify → Facial Profile** tab
2. **Click** **"Expression"** dropdown
3. **Test** a few blendshapes:
   - Click **"jawOpen"** - Jaw should open
   - Click **"lipCloser"** - Lips should close
   - Click **"mouthSmile"** - Should smile
4. **If blendshapes work**, you're good! ✅
5. **If blendshapes DON'T work**, go back to Step 1.2 and ensure "Facial Profile" is enabled

---

### **Step 1.6: Save Project**

1. **Ctrl + S** or **File → Save**
2. **Verify** file saved: `Kelly_Unity_Production.ccProject`

---

## 📤 PHASE 2: Export from CC5 to Unity

### **Step 2.1: Select Character**

1. **In CC5 viewport**, **LEFT-CLICK** directly on Kelly character
2. **Verify** Kelly is selected (selection highlights visible)

---

### **Step 2.2: Export as FBX**

1. **Top menu:** **File → Export → Export Character as FBX**
2. **Export dialog** appears

---

### **Step 2.3: Configure Export Settings (CRITICAL)**

**In Export Dialog:**

1. **Target Tool Preset:**
   - **Dropdown:** Select **"Unity 3D"** ✓

2. **Embed Textures:**
   - **Checkbox:** ✓ **CHECK** "Embed Textures"

3. **Convert Skinned Expressions to Morphs:**
   - **CRITICAL:** ✓ **CHECK** "Convert Skinned Expressions to Morphs"
   - This exports all 53 blendshapes as Unity blendshapes!

4. **Max Texture Size:**
   - **Dropdown:** Select **"4096"** or **"8192"** (higher quality)

5. **Include Animations:**
   - **UNCHECK** (we don't need animations, just the mesh)

6. **Export Path:**
   - **Click** **"..."** button
   - **Navigate** to: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\Assets\Kelly\Models\`
   - **Create folder** if it doesn't exist
   - **Filename:** `Kelly_CC5_Export.fbx`
   - **Click** **"Save"**

7. **Click** **"Export"** button
8. **Wait** 2-5 minutes for export

---

## 🎮 PHASE 3: Import into Unity

### **Step 3.1: Open Unity Project**

1. **Open Unity Hub**
2. **Click** **"Open"** → Navigate to: `digital-kelly\engines\kelly_unity_player`
3. **Click** **"Open"** and wait for Unity to load

---

### **Step 3.2: Import FBX File**

1. **In Unity Project window**, navigate to: `Assets/Kelly/Models/`
2. **Right-click** `Kelly_CC5_Export.fbx`
3. **Click** **"Reimport"** (if already imported, or **"Import"** if new)

---

### **Step 3.3: Configure FBX Import Settings**

**CRITICAL - This ensures blendshapes work:**

1. **Select** `Kelly_CC5_Export.fbx` in Project window
2. **Inspector** shows import settings

3. **Model Tab:**
   - **Scale Factor:** `1`
   - **Mesh Compression:** `Off`
   - **Read/Write Enabled:** ✓ **CHECK**
   - **Optimize Mesh:** ✓ **CHECK**
   - **Import Blendshapes:** ✓ **CHECK** (CRITICAL!)
   - **Import Visibility:** ✓ **CHECK**
   - **Import Cameras:** ✗ **UNCHECK**
   - **Import Lights:** ✗ **UNCHECK**

4. **Rig Tab:**
   - **Animation Type:** `None` (we're not using animations)
   - **Optimize Game Objects:** ✓ **CHECK**

5. **Materials Tab:**
   - **Import Materials:** ✓ **CHECK**
   - **Location:** `Use External Materials (Legacy)`
   - **Naming:** `From Model's Material`
   - **Search:** `Local Material Folder`

6. **Click** **"Apply"** button (bottom of Inspector)

---

### **Step 3.4: Verify Blendshapes Imported**

1. **In Project window**, select `Kelly_CC5_Export.fbx`
2. **Inspector → Model tab**
3. **Scroll down** to **"Blend Shapes"** section
4. **You should see** list of blendshapes:
   - `jawOpen`
   - `lipCloser`
   - `lipPucker`
   - `mouthSmile`
   - etc. (should be 50+ blendshapes)
5. **If you see blendshapes**, ✅ **SUCCESS!**
6. **If NO blendshapes**, go back to Step 2.3 and ensure "Convert Skinned Expressions to Morphs" was checked

---

## 🎯 PHASE 4: Set Up Kelly in Unity Scene

### **Step 4.1: Create Kelly GameObject**

1. **In Hierarchy**, **right-click** empty space
2. **Create Empty** → Name it `kelly_character_new`

---

### **Step 4.2: Add Character Model**

1. **Drag** `Kelly_CC5_Export.fbx` from Project window
2. **Drop** onto `kelly_character_new` in Hierarchy
3. **Rename** child object to `Kelly_Mesh`

---

### **Step 4.3: Set Up SkinnedMeshRenderer**

1. **Select** `Kelly_Mesh` in Hierarchy
2. **In Inspector**, find **"Skinned Mesh Renderer"** component
3. **Verify** it shows:
   - **Mesh:** `Kelly_CC5_Export` ✓
   - **Root Bone:** Set correctly
   - **Materials:** Assigned

---

### **Step 4.4: Add Required Scripts**

1. **Select** `kelly_character_new` GameObject
2. **Inspector → Add Component:**

   **a) BlendshapeDriver60fps:**
   - **Add Component** → Search "Blendshape Driver 60fps"
   - **Configure:**
     - **Head Renderer:** Drag `Kelly_Mesh` here
     - **Audio Source:** (we'll add this next)
     - **Intensity:** `100`
     - **Target FPS:** `60`
     - **Enable Interpolation:** ✓ **CHECK**

   **b) Audio Source:**
   - **Add Component** → Search "Audio Source"
   - **Configure:**
     - **Play On Awake:** ✗ **UNCHECK**
     - **Spatial Blend:** `0` (2D sound)

   **c) KellyTalkTest:**
   - **Add Component** → Search "Kelly Talk Test"
   - **Configure:**
     - **Auto Play On Start:** ✗ **UNCHECK**

   **d) KellyTTSClient:**
   - **Add Component** → Search "Kelly TTS Client"
   - **Configure:**
     - **Text To Speak:** `"Hello! I'm Kelly, and I'm here to help you learn."`
     - **Learner Age:** `35`

---

### **Step 4.5: Connect Components**

1. **Select** `kelly_character_new`
2. **In Inspector**, find **BlendshapeDriver60fps** component:
   - **Head Renderer:** Drag `Kelly_Mesh` from Hierarchy
   - **Audio Source:** Drag `kelly_character_new` (self) from Hierarchy

3. **Find KellyTalkTest** component:
   - **Avatar Controller:** Leave empty (auto-finds)
   - **Blendshape Driver:** Leave empty (auto-finds)
   - **Audio Source:** Should auto-find

---

### **Step 4.6: Position Character**

1. **Select** `kelly_character_new` in Hierarchy
2. **Transform** settings:
   - **Position:** `X: 0, Y: 0, Z: 0`
   - **Rotation:** `X: 0, Y: 0, Z: 0`
   - **Scale:** `X: 1, Y: 1, Z: 1`

---

### **Step 4.7: Set Up Camera**

1. **Select** `Main Camera` in Hierarchy
2. **Position:** `X: 0, Y: 1.6, Z: -2` (face-height, 2m back)
3. **Rotation:** `X: 0, Y: 0, Z: 0` (looking straight)
4. **Frame** Kelly's head and shoulders in Game view

---

## 🧪 PHASE 5: Test Lip-Sync

### **Step 5.1: Test in Unity**

1. **Press Play** ▶️ in Unity
2. **In Game view**, click **"Speak Now (ElevenLabs)"** button
3. **Watch Console** for:
   - `[KellyTTSClient] Requesting TTS...`
   - `[KellyTTSClient] Audio received! Length: X.XXs`
4. **Watch Kelly's face:**
   - ✅ **Mouth should open/close** with audio
   - ✅ **Lips should move** naturally
   - ❌ **NO "ball in throat"** - that was the old bug!

---

### **Step 5.2: Troubleshooting**

**If lips don't move:**

1. **Check Console** for errors
2. **Verify** BlendshapeDriver60fps has:
   - **Head Renderer** assigned correctly
   - **Audio Source** assigned correctly
3. **Check** if blendshapes imported:
   - Select `Kelly_Mesh` → Inspector → Skinned Mesh Renderer
   - Expand **"Blend Shapes"** dropdown
   - Should see list of blendshapes

**If "ball in throat" still appears:**

1. **Old avatar** might still be in scene
2. **Delete** old `kelly_character` GameObject
3. **Use** only `kelly_character_new`

---

## ✅ Success Checklist

- [ ] Kelly created in CC5 with Facial Profile enabled
- [ ] FBX exported with "Convert Skinned Expressions to Morphs" ✓
- [ ] Unity import shows 50+ blendshapes in Blend Shapes list
- [ ] BlendshapeDriver60fps has Head Renderer assigned
- [ ] Audio Source assigned and connected
- [ ] Press Play → Click "Speak Now" → Kelly's lips move! ✅

---

## 🎉 Next Steps

Once lip-sync works:

1. **Fine-tune** blendshape intensity in BlendshapeDriver60fps
2. **Add** more realistic expressions (eye blinks, micro-expressions)
3. **Generate** Audio2Face JSON files for even better sync
4. **Integrate** with lesson system

---

**Ready to start?** Follow Phase 1, Step 1.1 → Begin with CC5!











